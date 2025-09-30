import os
import re
import time
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
import logging
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode


BASE_URL = "https://chat.parallellm.com"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("login_agent")


@contextmanager
def get_driver():
    options = Options()
    # options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        yield driver
    finally:
        driver.quit()


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_artifacts_dir() -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("artifacts") / "login" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_html(driver: webdriver.Chrome, out_dir: Path, name: str) -> Path:
    html = driver.page_source
    fp = out_dir / f"{name}.html"
    fp.write_text(html, encoding="utf-8")
    return fp


def _save_screenshot(driver: webdriver.Chrome, out_dir: Path, name: str) -> Path:
    fp = out_dir / f"{name}.png"
    driver.save_screenshot(str(fp))
    return fp


def _is_logged_in(driver: webdriver.Chrome) -> bool:
    url = (driver.current_url or "").lower()
    if "login" in url or "signin" in url:
        return False
    try:
        _ = driver.find_element(By.CSS_SELECTOR, "input[type='password']")
        return False
    except NoSuchElementException:
        return True


def build_tools(driver: webdriver.Chrome, creds: Dict[str, str]):
    @tool("get_page_html")
    def get_page_html() -> str:
        """Return the current page HTML with all <script>...</script> tags removed."""
        html = driver.page_source
        # Remove all <script> tags and their contents (case-insensitive, across newlines)
        cleaned = re.sub(r"<script\b[^>]*>[\s\S]*?<\/script>", "", html, flags=re.IGNORECASE)
        try:
            logger.info(f"[tool:get_page_html] sanitized_html_length={len(cleaned)}")
        except Exception:
            logger.info("[tool:get_page_html] returned sanitized html")
        return cleaned

    @tool("type_text")
    def type_text(selector: str, by: str, text: str) -> str:  # by in {css, id, name, xpath}
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

        el = driver.find_element(by_key, selector)
        el.clear()
        el.send_keys(real_text)
        # Redacted logging
        if used_placeholders:
            logger.info(f"[tool:type_text] selector={selector} by={by} substituted={used_placeholders}")
        else:
            logger.info(f"[tool:type_text] selector={selector} by={by} text_len={len(real_text)}")
        return "OK"

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
        el = driver.find_element(by_key, selector)
        el.click()
        logger.info(f"[tool:click] selector={selector} by={by}")
        return "OK"

    @tool("check_is_logged_in")
    def check_is_logged_in() -> bool:
        """Return True if the user appears to be logged in on the current page."""
        result = _is_logged_in(driver)
        logger.info(f"[tool:check_is_logged_in] result={result}")
        return result

    @tool("sleep")
    def sleep(seconds: float) -> str:
        """Sleep for a number of seconds to allow the page to update."""
        time.sleep(float(seconds))
        logger.info(f"[tool:sleep] seconds={seconds}")
        return "OK"

    @tool("navigate")
    def navigate(url: str) -> str:
        """Navigate the browser to a URL."""
        driver.get(url)
        logger.info(f"[tool:navigate] url={url}")
        return driver.current_url

    @tool("post_login_capture")
    def post_login_capture() -> str:
        """Save HTML and a screenshot after login to the artifacts directory."""
        out_dir = Path(graph_artifacts_dir[0])
        _save_html(driver, out_dir, "post_login")
        _save_screenshot(driver, out_dir, "post_login")
        logger.info(f"[tool:post_login_capture] saved to {out_dir}")
        return str(out_dir)

    # Expose tools list
    return [get_page_html, type_text, click, check_is_logged_in, sleep, navigate, post_login_capture]


# A simple global holder for artifacts dir used inside tool closure
graph_artifacts_dir: List[str] = [""]


def build_graph(driver: webdriver.Chrome, initial_html_cleaned: str, goal: str, creds: Dict[str, str], artifacts_dir: Path):
    tools = build_tools(driver, creds)
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0).bind_tools(tools)

    class State(MessagesState):
        goal: str
        creds: Dict[str, str]
        status: str
        artifacts_dir: str

    def agent_node(state: State) -> Dict[str, Any]:
        logger.debug("[agent] invoking model with messages")
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    def check_node(state: State) -> Dict[str, Any]:
        if _is_logged_in(driver):
            return {"status": "logged_in"}
        return {"status": "continue"}

    def should_continue(state: State):
        last_message = state["messages"][-1]
        # If there are tool calls, route to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        # Otherwise check if we're done
        return "check"

    post_tools = ToolNode(tools)

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", post_tools)
    graph.add_node("check", check_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "check": "check"
        }
    )
    graph.add_edge("tools", "check")
    # Conditional: if logged in -> END (post handled outside), else loop back to agent
    def route_after_check(state: State):
        return "end" if state.get("status") == "logged_in" else "loop"

    graph.add_conditional_edges(
        "check",
        route_after_check,
        {
            "end": END,
            "loop": "agent",
        },
    )

    app = graph.compile()
    initial_messages: List[AnyMessage] = [
        SystemMessage(content=(
            "You are an automation agent controlling a headless browser via tools. "
            "A Selenium driver will be used externally to control the browser."
            "It is your task to interpret the HTML to infer next steps. "
            "If you do not see any HTML, then you will need to use the get_page_html tool to fetch the HTML. "
            "Goal: log in to the target website. Use navigate to go to the login page, "
            "use get_page_html to understand the form, then type_text and click to submit. "
            "Use check_is_logged_in to check progress. Keep iterating until logged in. "
            "Policy: Never include raw secrets in tool arguments. Use these placeholders: <EMAIL> for email fields, <PASSWORD> for password fields. "
            "Placeholders will be substituted with secure values at execution time. "
            "Only use tools; do not fabricate steps."
        )),
        HumanMessage(content=(
            f"Instructions: {goal}"
            f"Initial HTML (cleaned): {initial_html_cleaned}"
        )),
    ]

    state: State = {
        "messages": initial_messages,
        "goal": goal,
        "creds": creds,
        "status": "start",
        "artifacts_dir": str(artifacts_dir),
    }

    return app, state


def do_login(profile: Optional[str] = None) -> Tuple[bool, Path]:
    load_dotenv(dotenv_path=Path(".env"), override=False)

    login_profile = profile or os.getenv("LOGIN_PROFILE", "default")
    logger.info(f"Selected login profile: {login_profile}")
    secrets_path = Path("config/secret/logins.yaml.env")
    state_path = Path("config/state.yaml")
    secrets = _read_yaml(secrets_path)
    run_state = _read_yaml(state_path)
    logger.info(f"Loaded credentials from: {secrets_path}")
    logger.info(f"Loaded run state from: {state_path}")
    creds: Dict[str, str] = secrets.get(login_profile, {})
    if not creds:
        raise RuntimeError(f"No credentials found for profile '{login_profile}' in {secrets_path}")

    artifacts_dir = _ensure_artifacts_dir()
    graph_artifacts_dir[0] = str(artifacts_dir)
    logger.info(f"Artifacts directory: {artifacts_dir}")

    with get_driver() as driver:
        driver.set_window_size(1280, 1200)
        logger.info(f"Navigating to base URL: {BASE_URL}")
        driver.get(BASE_URL)

        initial_html = driver.page_source
        initial_html_cleaned = re.sub(r"<script\b[^>]*>[\s\S]*?<\/script>", "", initial_html, flags=re.IGNORECASE)

        goal_text = str(run_state.get("instructions", "Log in successfully and reach the main app."))
        logger.info(f"Instructions: {goal_text}")
        app, state = build_graph(driver, initial_html_cleaned, goal_text, creds, artifacts_dir)
        logger.info("Graph compiled. Beginning execution loop...")

        # Prime the agent with a suggested plan and initial actions
        # It can choose to call navigate, get_page_html, type_text, click, etc.
        stream = app.stream(state)
        for _ in stream:
            pass

        success = _is_logged_in(driver)
        logger.info(f"Login success status after graph run: {success}")

        # Post-login capture
        if success:
            logger.info("Saving post-login HTML and screenshot...")
            _save_html(driver, artifacts_dir, "post_login")
            _save_screenshot(driver, artifacts_dir, "post_login")

        return success, artifacts_dir


if __name__ == "__main__":
    logger.info("Invoking do_login()...")
    ok, out = do_login()
    logger.info(f"login_success={ok} artifacts_dir={out}")
