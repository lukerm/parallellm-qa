from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from selenium import webdriver


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

