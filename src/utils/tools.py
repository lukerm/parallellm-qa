import logging
import re
import time
from pathlib import Path
from typing import Dict, List

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from langchain_core.tools import tool

from .files import save_html, save_screenshot


logger = logging.getLogger(__name__)


def build_common_tools(driver: webdriver.Chrome, creds: Dict[str, str] = None):
    """Build common tools for browser automation that can be used across different tasks.

    Args:
        driver: Selenium Chrome driver
        creds: Optional credentials dictionary for placeholder substitution in type_text
    """
    creds = creds or {}

    @tool("get_page_html")
    def get_page_html() -> str:
        """Return the current page HTML with all <script>...</script> tags removed."""
        html = driver.page_source
        cleaned = re.sub(r"<script\b[^>]*>[\s\S]*?<\/script>", "", html, flags=re.IGNORECASE)
        try:
            logger.info(f"[tool:get_page_html] sanitized_html_length={len(cleaned)}")
        except Exception:
            logger.info("[tool:get_page_html] returned sanitized html")
        return cleaned

    @tool("type_text")
    def type_text(selector: str, by: str, text: str) -> str:
        """Type text into an element identified by a selector. 'by' is one of css,id,name,xpath.

        Placeholder policy: Do not include raw secrets. Use placeholders like <PASSWORD>, <EMAIL>.
        They will be substituted with secure values from creds at runtime.
        """
        by_map = {
            "css": By.CSS_SELECTOR,
            "id": By.ID,
            "name": By.NAME,
            "xpath": By.XPATH,
        }
        by_key = by_map.get(by)
        if by_key is None:
            return f"Unsupported selector strategy: {by}"

        # Replace known placeholders with runtime secrets (values used directly)
        placeholder_map = {
            "<PASSWORD>": creds.get("password", ""),
            "<EMAIL>": creds.get("email", ""),
        }
        real_text = text
        used_placeholders: List[str] = []
        for placeholder, value in placeholder_map.items():
            if placeholder in real_text:
                real_text = real_text.replace(placeholder, value)
                used_placeholders.append(placeholder)

        try:
            el = driver.find_element(by_key, selector)
            el.clear()
            el.send_keys(real_text)
            # Redacted logging
            if used_placeholders:
                logger.info(f"[tool:type_text] selector={selector} by={by} substituted={used_placeholders}")
            else:
                logger.info(f"[tool:type_text] selector={selector} by={by} text_len={len(real_text)}")
            return "OK"
        except Exception as e:
            logger.error(f"[tool:type_text] Error: {e}")
            return f"Error: {str(e)}"

    @tool("click")
    def click(selector: str, by: str) -> str:
        """Click an element identified by a selector. 'by' is one of css,id,name,xpath."""
        by_map = {
            "css": By.CSS_SELECTOR,
            "id": By.ID,
            "name": By.NAME,
            "xpath": By.XPATH,
        }
        by_key = by_map.get(by)
        if by_key is None:
            return f"Unsupported selector strategy: {by}"
        try:
            el = driver.find_element(by_key, selector)
            el.click()
            logger.info(f"[tool:click] selector={selector} by={by}")
            return "OK"
        except Exception as e:
            logger.error(f"[tool:click] Error: {e}")
            return f"Error: {str(e)}"

    @tool("sleep")
    def sleep(seconds: float) -> str:
        """Sleep for a number of seconds to allow the page to update."""
        time.sleep(float(seconds))
        logger.info(f"[tool:sleep] seconds={seconds}")
        return "OK"

    return [get_page_html, type_text, click, sleep]


def build_login_tools(driver: webdriver.Chrome, creds: Dict[str, str], is_logged_in_func, graph_artefacts_dir: List[str]):
    """Build tools for login automation.

    Args:
        driver: Selenium Chrome driver
        creds: Credentials dictionary
        is_logged_in_func: Function to check if user is logged in
        graph_artefacts_dir: List containing artefacts directory path
    """
    # Get common tools with credentials support
    common_tools = build_common_tools(driver, creds)

    # Add login-specific tools
    @tool("check_is_logged_in")
    def check_is_logged_in() -> bool:
        """Return True if the user appears to be logged in on the current page."""
        result = is_logged_in_func(driver)
        logger.info(f"[tool:check_is_logged_in] result={result}")
        return result

    @tool("navigate")
    def navigate(url: str) -> str:
        """Navigate the browser to a URL."""
        driver.get(url)
        logger.info(f"[tool:navigate] url={url}")
        return driver.current_url

    @tool("post_login_capture")
    def post_login_capture() -> str:
        """Save HTML and a screenshot after login to the artefacts directory."""
        out_dir = Path(graph_artefacts_dir[0])
        save_html(driver, out_dir, "post_login")
        save_screenshot(driver, out_dir, "post_login")
        logger.info(f"[tool:post_login_capture] saved to {out_dir}")
        return str(out_dir)

    return common_tools + [check_is_logged_in, navigate, post_login_capture]


def build_chat_tools(driver: webdriver.Chrome, graph_artefacts_dir: List[str]):
    """Build tools for chat interface automation.

    Args:
        driver: Selenium Chrome driver
        graph_artefacts_dir: List containing artefacts directory path
    """
    # Get common tools (no credentials needed for chats)
    common_tools = build_common_tools(driver, creds={})

    # Add chat-specific tools
    @tool("check_submit_button_present")
    def check_submit_button_present() -> bool:
        """Check if the submit button is present. This indicates responses are complete."""
        try:
            possible_selectors = [
                "button[type='submit']", "button.submit", "input[type='submit']"
            ]
            for selector in possible_selectors:
                try:
                    driver.find_element(By.CSS_SELECTOR, selector)
                    logger.info(f"[tool:check_submit_button_present] Found: {selector}")
                    return True
                except NoSuchElementException:
                    continue
            logger.info("[tool:check_submit_button_present] Not found")
            return False
        except Exception as e:
            logger.error(f"[tool:check_submit_button_present] Error: {e}")
            return False

    @tool("save_chat_capture")
    def save_chat_capture(name: str) -> str:
        """Save HTML and screenshot to artefacts directory."""
        out_dir = Path(graph_artefacts_dir[0])
        save_html(driver, out_dir, name)
        save_screenshot(driver, out_dir, name)
        logger.info(f"[tool:save_chat_capture] saved {name} to {out_dir}")
        return str(out_dir)

    @tool("report_completion")
    def report_completion(health: str, health_description: str) -> str:
        """Report task completion with health status.

        Args:
            health: Either 'OK' if system is healthy, or 'ERROR' if issues detected
            health_description: Brief description of the health status or any issues found
        """
        logger.info(f"[tool:report_completion] health={health}, description={health_description}")
        return f"Completion reported: health={health}, description={health_description}"

    return common_tools + [check_submit_button_present, save_chat_capture, report_completion]

