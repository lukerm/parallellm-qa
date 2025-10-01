import os
import re
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from . utils.selenium import get_driver


BASE_URL = "https://chat.parallellm.com"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("login_agent")


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _ensure_artefacts_dir(subfolder: List[str], ts: Optional[str] = None) -> Path:
    ts = ts or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("artefacts") / ts / subfolder
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
        """Save HTML and a screenshot after login to the artefacts directory."""
        out_dir = Path(graph_artefacts_dir[0])
        _save_html(driver, out_dir, "post_login")
        _save_screenshot(driver, out_dir, "post_login")
        logger.info(f"[tool:post_login_capture] saved to {out_dir}")
        return str(out_dir)

    # Expose tools list
    return [get_page_html, type_text, click, check_is_logged_in, sleep, navigate, post_login_capture]


# A simple global holder for artefacts dir used inside tool closure
graph_artefacts_dir: List[str] = [""]


def build_graph(driver: webdriver.Chrome, initial_html_cleaned: str, goal: str, creds: Dict[str, str], artefacts_dir: Path):
    tools = build_tools(driver, creds)
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0).bind_tools(tools)

    class State(MessagesState):
        goal: str
        creds: Dict[str, str]
        status: str
        artefacts_dir: str

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

    # Conditional: if logged in -> END (post handled outside), else loop back to agent
    def route_after_check(state: State):
        return "end" if state.get("status") == "logged_in" else "loop"


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
        "artefacts_dir": str(artefacts_dir),
    }

    return app, state


def message_to_dict(msg: BaseMessage) -> dict:
    """Convert a LangChain message to a JSON-serializable dict."""
    result = {
        "type": msg.__class__.__name__,
        "content": msg.content,
    }

    # Add tool calls if present
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = [
            {
                "name": tc.get("name"),
                "args": tc.get("args"),
                "id": tc.get("id"),
            }
            for tc in msg.tool_calls
        ]

    # Add tool call ID if this is a tool response
    if hasattr(msg, "tool_call_id"):
        result["tool_call_id"] = msg.tool_call_id

    return result


def run_and_save_execution_trace(stream, artefacts_dir: Path) -> Path:
    """Save the full execution trace to JSON."""
    trace = {
        "timestamp": datetime.utcnow().isoformat(),
        "steps": []
    }

    final_state = None
    for step in stream:
        final_state = step

        # Convert each step to a serializable format
        step_data = {}
        for node_name, node_state in step.items():
            step_data[node_name] = {
                "status": node_state.get("status"),
                "goal": node_state.get("goal"),
                "messages": [message_to_dict(msg) for msg in node_state.get("messages", [])],
                "artefacts_dir": node_state.get("artefacts_dir"),
            }

        trace["steps"].append(step_data)
        logger.debug(f"Captured step {len(trace['steps'])}: {list(step.keys())}")

    # Add final state summary
    if final_state:
        # Get the final status from the last state in the stream
        for node_state in final_state.values():
            if "status" in node_state:
                trace["final_status"] = node_state.get("status")
                break
        trace["total_steps"] = len(trace["steps"])

    # Save to JSON
    trace_file = artefacts_dir / "execution_trace.json"
    with trace_file.open("w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    logger.info(f"Execution trace saved to: {trace_file}")
    return trace_file


def run_login(driver: webdriver.Chrome, profile: Optional[str] = None, run_ts: str = None) -> Tuple[bool, Path]:
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

    artefacts_dir = _ensure_artefacts_dir(subfolder="run_login", ts=run_ts)
    graph_artefacts_dir[0] = str(artefacts_dir)
    logger.info(f"artefacts directory: {artefacts_dir}")

    driver.set_window_size(1280, 1200)
    logger.info(f"Navigating to base URL: {BASE_URL}")
    driver.get(BASE_URL)

    initial_html = driver.page_source
    initial_html_cleaned = re.sub(r"<script\b[^>]*>[\s\S]*?<\/script>", "", initial_html, flags=re.IGNORECASE)

    goal_text = str(run_state.get("run_login", {}).get("instructions", "Log in successfully and reach the main app."))
    logger.info(f"Instructions: {goal_text}")
    app, state = build_graph(driver, initial_html_cleaned, goal_text, creds, artefacts_dir)
    logger.info("Graph compiled. Beginning execution loop...")

    # Prime the agent with a suggested plan and initial actions
    # It can choose to call navigate, get_page_html, type_text, click, etc.
    _ = run_and_save_execution_trace(app.stream(state), artefacts_dir)

    success = _is_logged_in(driver)
    logger.info(f"Login success status after graph run: {success}")

    # Post-login capture
    if success:
        logger.info("Saving post-login HTML and screenshot...")
        _save_html(driver, artefacts_dir, "post_login")
        _save_screenshot(driver, artefacts_dir, "post_login")

    return success, artefacts_dir


if __name__ == "__main__":
    run_ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    with get_driver() as driver:
        logger.info("Invoking run_login()...")
        ok, out = run_login(driver, run_ts=run_ts)
    logger.info(f"login_success={ok} artefacts_dir={out}")
