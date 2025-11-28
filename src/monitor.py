#  Copyright (C) 2025 lukerm of www.zl-labs.tech
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv; load_dotenv(dotenv_path=Path(".env"), override=False)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("monitor")


class ErrorFolderMonitor:
    """Monitor artefacts/error/ directory for error folders."""

    def __init__(
        self,
        s3_bucket: str,
        s3_prefix: str,
        aws_region: str = "eu-west-1",
        aws_profile: Optional[str] = None,
        error_dir: Path = Path("artefacts/error"),
        sns_topic_arn: Optional[str] = None,
        no_delete: bool = False,
    ):
        """
        Initialize the error folder monitor.

        Args:
            s3_bucket: S3 bucket name for uploading error files
            s3_prefix: S3 prefix (folder path) for uploads
            aws_region: AWS region for S3 (optional, uses default if not set)
            aws_profile: AWS profile name (optional, uses default if not set)
            error_dir: Path to the error directory to monitor
            sns_topic_arn: SNS topic ARN for notifications (optional)
        """
        self.error_dir = error_dir
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip('/') if s3_prefix else ""
        self.sns_topic_arn = sns_topic_arn
        self.s3_client = None
        self.sns_client = None
        self.no_delete = no_delete

        # Build session for boto3 clients
        session_kwargs = {}
        if aws_region:
            session_kwargs['region_name'] = aws_region
        if aws_profile:
            session_kwargs['profile_name'] = aws_profile
            logger.info(f"Using AWS profile: {aws_profile}")

        # Create boto3 session if profile is specified, otherwise use default
        if session_kwargs:
            session = boto3.Session(**session_kwargs)
        else:
            session = boto3.Session()

        if s3_bucket:
            try:
                self.s3_client = session.client('s3')
                logger.info(f"S3 upload enabled: s3://{s3_bucket}/{self.s3_prefix}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")

        if sns_topic_arn:
            try:
                self.sns_client = session.client('sns')
                logger.info(f"SNS notifications enabled: {sns_topic_arn}")
            except Exception as e:
                logger.error(f"Failed to initialize SNS client: {e}")

        self.ensure_error_dir_exists()
        logger.info(f"Initialized monitor for directory: {self.error_dir}")

    def ensure_error_dir_exists(self) -> None:
        if not self.error_dir.exists():
            self.error_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created error directory: {self.error_dir}")

    def upload_folder_to_s3(self, folder: Path) -> tuple[bool, Optional[str], int, str|None]:
        """
        Upload error folder contents to S3.

        Structure: PREFIX/YYYY-MM-DD/run_id/filename

        Args:
            folder: Path to the error folder

        Returns:
            Tuple of (success, s3_path, files_uploaded, final_health_description)
        """
        if not self.s3_client or not self.s3_bucket:
            logger.warning("S3 not configured - skipping upload")
            return False, None, 0

        try:
            # Get run_id from folder name
            run_id = folder.name

            # Get date from folder creation time
            created_time = datetime.fromtimestamp(folder.stat().st_ctime)
            date_str = created_time.strftime("%Y-%m-%d")

            # Build S3 path: PREFIX/YYYY-MM-DD/run_id/
            if self.s3_prefix:
                s3_path = f"{self.s3_prefix}/{date_str}/{run_id}"
            else:
                s3_path = f"{date_str}/{run_id}"

            logger.info(f"Uploading to s3://{self.s3_bucket}/{s3_path}/")

            # Collect files to upload
            trace_file = folder / "execution_trace.json"
            image_files = list(folder.glob("*.png"))
            files_to_upload = [trace_file] + image_files

            files_uploaded = 0

            for file in sorted(files_to_upload):
                s3_key = f"{s3_path}/{file.name}"
                try:
                    self.s3_client.upload_file(
                        str(file),
                        self.s3_bucket,
                        s3_key
                    )
                    logger.info(f"  Uploaded: {file.name} -> s3://{self.s3_bucket}/{s3_key}")
                    files_uploaded += 1
                except ClientError as e:
                    logger.error(f"  Failed to upload {file.name}: {e}")

            final_health_description = None
            with open(folder / "execution_trace.json", "r") as j:
                execution_trace = json.load(j)
                final_health_description = execution_trace.get("final_health_description")

            total_files = len(files_to_upload)
            logger.info(f"  Upload complete: {files_uploaded}/{total_files} files uploaded successfully")

            return files_uploaded > 0, s3_path, files_uploaded, final_health_description

        except Exception as e:
            logger.error(f"Failed to upload {folder.name} to S3: {e}", exc_info=True)
            return False, None, 0, None

    def publish_sns_notification(self, folder: Path, s3_path: str, files_uploaded: int, final_health_description: str|None) -> bool:
        """
        Publish a notification to SNS about the uploaded error.

        Args:
            folder: Path to the error folder
            s3_path: S3 path where files were uploaded
            files_uploaded: Number of files successfully uploaded
            final_health_description: Final health description
        Returns:
            True if notification sent successfully, False otherwise
        """
        if not self.sns_client or not self.sns_topic_arn:
            logger.warning("SNS not configured - skipping notification")
            return False

        try:
            run_id = folder.name
            created_time = datetime.fromtimestamp(folder.stat().st_ctime)
            date_str = created_time.strftime("%Y-%m-%d")

            # Build notification message
            subject = f"QA Error Detected: {run_id[:20]}..."

            message = f"""An error was detected in the QA monitoring process.

Error Details:
- Run ID: {run_id}
- Date: {date_str}
- Created: {created_time.strftime("%Y-%m-%d %H:%M:%S")}
- Files Uploaded: {files_uploaded}
- Final Health Description: {final_health_description or 'unknown'}

S3 Location:
s3://{self.s3_bucket}/{s3_path}/

Please review the S3 bucket for detailed error information.
"""

            # Publish to SNS
            response = self.sns_client.publish(
                TopicArn=self.sns_topic_arn,
                Subject=subject,
                Message=message,
                MessageAttributes={
                    'run_id': {
                        'DataType': 'String',
                        'StringValue': run_id
                    },
                    'date': {
                        'DataType': 'String',
                        'StringValue': date_str
                    },
                    's3_path': {
                        'DataType': 'String',
                        'StringValue': f"s3://{self.s3_bucket}/{s3_path}/"
                    }
                }
            )

            message_id = response.get('MessageId')
            logger.info(f"  SNS notification sent successfully (MessageId: {message_id})")
            return True

        except ClientError as e:
            logger.error(f"Failed to publish SNS notification for {folder.name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error publishing SNS notification: {e}", exc_info=True)
            return False

    def scan_for_folders(self) -> List[Path]:
        """
        Scan the error directory for all folders.

        Returns:
            List of folder paths found in the error directory
        """
        folders = []
        try:
            # Get all directories in error folder
            folders = [
                item for item in self.error_dir.iterdir()
                if item.is_dir()
            ]
            # Sort by creation time (oldest first)
            folders.sort(key=lambda p: p.stat().st_ctime)

        except Exception as e:
            logger.error(f"Error scanning directory {self.error_dir}: {e}")

        return folders

    def delete_folder(self, folder: Path, no_delete: bool = False) -> bool:
        """
        Delete a folder and all its contents.

        Args:
            folder: Path to the folder to delete
            no_delete: If True, do not delete the folder (still returns True)
        Returns:
            True if deletion succeeded, False otherwise
        """
        if no_delete:
            logger.info(f"  Skipping deletion of folder: {folder}")
            return True

        try:
            shutil.rmtree(folder)
            logger.info(f"  Deleted local folder: {folder}")
            return True
        except Exception as e:
            logger.error(f"  Failed to delete folder {folder}: {e}")
            return False

    def process_folders(self, folders: List[Path]) -> int:
        """
        Process folders by:
         - uploading to S3
         - sending SNS notifications
         - deleting local folder after successful processing

        Args:
            folders: List of folder paths to process

        Returns:
            Number of folders processed successfully
        """
        if not folders:
            logger.info("No error folders found.")
            return 0

        logger.info(f"Found {len(folders)} error folder(s)")
        processed_successfully = 0

        for folder in folders:
            logger.info("=" * 80)
            logger.info(f"Processing ERROR FOLDER: {folder.name}")
            logger.info(f"Full path: {folder}")
            logger.info(f"Created: {datetime.fromtimestamp(folder.stat().st_ctime)}")
            logger.info("-" * 80)

            # Upload to S3
            success, s3_path, files_uploaded, final_health_description = self.upload_folder_to_s3(folder)

            # Send SNS notification if upload was successful
            if success and s3_path:
                self.publish_sns_notification(folder, s3_path, files_uploaded, final_health_description)

                # (Maybe) delete local folder after successful processing
                if self.delete_folder(folder, no_delete=self.no_delete):
                    processed_successfully += 1
                else:
                    logger.warning(f"  Folder processed but failed to delete - may be reprocessed next run")
            elif not success:
                logger.warning(f"  Skipping SNS notification due to upload failure")
                logger.warning(f"  Folder not deleted - will retry on next run")

            logger.info("=" * 80)

        logger.info(f"Successfully processed {processed_successfully}/{len(folders)} folder(s)")
        return processed_successfully

    def run(self) -> int:
        """
        Run the monitor once - scan and process all folders, then exit.

        Returns:
            Number of error folders found and processed
        """
        logger.info("Scanning error folder for issues...")

        try:
            folders = self.scan_for_folders()
            self.process_folders(folders)
            logger.info(f"Completed processing {len(folders)} error folder(s)")
            return len(folders)

        except Exception as e:
            logger.error(f"Monitor encountered an error: {e}", exc_info=True)
            raise


def main():
    """Entry point for the monitor script."""
    # Get configuration from environment
    no_delete = os.getenv("NO_DELETE", "false").lower() == "true"
    error_dir = Path(os.getenv("ERROR_DIR", "artefacts/error"))
    s3_bucket = os.getenv("S3_BUCKET")
    s3_prefix = os.getenv("S3_PREFIX", "qa-monitoring")
    sns_topic_arn = os.getenv("SNS_TOPIC_ARN")

    aws_region = os.getenv("AWS_REGION", "eu-west-1")
    aws_profile = os.getenv("AWS_PROFILE")

    # Validate S3 configuration
    if not s3_bucket:
        logger.error("S3_BUCKET not configured - S3 upload required")
        logger.info("Set S3_BUCKET environment variable to enable S3 uploads")
        return

    # Validate SNS configuration
    if not sns_topic_arn:
        logger.warning("SNS_TOPIC_ARN not configured - SNS notifications disabled")
        logger.info("Set SNS_TOPIC_ARN environment variable to enable notifications")

    monitor = ErrorFolderMonitor(
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
        aws_region=aws_region,
        aws_profile=aws_profile,
        error_dir=error_dir,
        sns_topic_arn=sns_topic_arn,
        no_delete=no_delete,
    )
    count = monitor.run()

    if count > 0:
        logger.warning(f"Found {count} error folder(s) - check S3 and email for details")
    else:
        logger.info("No error folders found - all clear!")


if __name__ == "__main__":
    main()

