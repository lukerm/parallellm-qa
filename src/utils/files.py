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
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from selenium import webdriver

logger = logging.getLogger(__name__)


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_artefacts_dir(subfolder: str, ts: Optional[str] = None) -> Path:
    ts = ts or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("artefacts") / ts / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_html(driver: webdriver.Chrome, out_dir: Path, name: str) -> Path:
    html = driver.page_source
    fp = out_dir / f"{name}.html"
    fp.write_text(html, encoding="utf-8")
    return fp


def save_screenshot(driver: webdriver.Chrome, out_dir: Path, name: str) -> Path:
    fp = out_dir / f"{name}.png"
    driver.save_screenshot(str(fp))
    return fp


def copy_trace_to_error_folder(artefacts_dir: Path) -> Optional[Path]:
    """
    Copy execution trace and any PNG files to the error folder in a subdirectory named by run_id.

    Args:
        artefacts_dir: Path to the artefacts directory containing the execution trace

    Returns:
        Path to the copied error trace file, or None if the trace file doesn't exist
    """
    trace_file = artefacts_dir / "execution_trace.json"
    if not trace_file.exists():
        logger.warning(f"Execution trace not found at {trace_file}")
        return None

    # Use the full run_id from the trace file for the folder name
    with trace_file.open("r", encoding="utf-8") as f:
        trace_data = json.load(f)
        run_id = trace_data.get("run_id", str(uuid.uuid4()))

    # Create error subdirectory with the full run_id
    error_run_dir = Path("artefacts/error") / run_id
    error_run_dir.mkdir(parents=True, exist_ok=True)

    # Copy trace file with original filename
    error_trace_file = error_run_dir / "execution_trace.json"
    shutil.copy2(trace_file, error_trace_file)
    logger.info(f"Copied execution trace to error folder: {error_trace_file}")

    # Copy all PNG files from the artefacts directory
    png_files = list(artefacts_dir.glob("*.png"))
    for png_file in png_files:
        dest_file = error_run_dir / png_file.name
        shutil.copy2(png_file, dest_file)
        logger.info(f"Copied screenshot to error folder: {dest_file}")

    return error_trace_file

