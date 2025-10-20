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
    Copy execution trace to the error folder with a unique filename based on run_id.

    Args:
        artefacts_dir: Path to the artefacts directory containing the execution trace

    Returns:
        Path to the copied error trace file, or None if the trace file doesn't exist
    """
    error_dir = Path("artefacts/error")
    error_dir.mkdir(parents=True, exist_ok=True)

    trace_file = artefacts_dir / "execution_trace.json"
    if not trace_file.exists():
        logger.warning(f"Execution trace not found at {trace_file}")
        return None

    # Use the run_id from the trace file for unique filename
    with trace_file.open("r", encoding="utf-8") as f:
        trace_data = json.load(f)
        run_id = trace_data.get("run_id", str(uuid.uuid4()))[:8]

    error_trace_file = error_dir / f"{run_id}_execution_trace.json"
    shutil.copy2(trace_file, error_trace_file)
    logger.info(f"Copied execution trace to error folder: {error_trace_file}")

    return error_trace_file

