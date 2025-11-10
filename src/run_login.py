import argparse
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from . utils.files import read_yaml, ensure_artefacts_dir, save_html, save_screenshot, copy_trace_to_error_folder
from . utils.selenium import get_driver
from . utils.tools import build_login_tools


BASE_URL = "https://chat.parallellm.com"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("login_runner")


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
    """Build tools for login automation. Delegates to utils.tools.build_login_tools."""
    return build_login_tools(driver, creds, _is_logged_in, graph_artefacts_dir)


# A simple global holder for artefacts dir used inside tool closure
graph_artefacts_dir: List[str] = [""]


def build_graph(driver: webdriver.Chrome, initial_html_cleaned: str, goal: str, creds: Dict[str, str], artefacts_dir: Path):
    tools = build_tools(driver, creds)
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0).bind_tools(tools)

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
        "run_id": str(uuid.uuid4()),
        "run_type": artefacts_dir.name,
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
    secrets = read_yaml(secrets_path)
    run_state = read_yaml(state_path)
    logger.info(f"Loaded credentials from: {secrets_path}")
    logger.info(f"Loaded run state from: {state_path}")
    creds: Dict[str, str] = secrets.get(login_profile, {})
    if not creds:
        raise RuntimeError(f"No credentials found for profile '{login_profile}' in {secrets_path}")

    artefacts_dir = ensure_artefacts_dir(subfolder="run_login", ts=run_ts)
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
    _ = run_and_save_execution_trace(
        app.stream(state, config={"recursion_limit": 25}),
        artefacts_dir
    )

    success = _is_logged_in(driver)
    logger.info(f"Login success status after graph run: {success}")

    # Post-login capture
    if success:
        logger.info("Saving post-login HTML and screenshot...")
        save_html(driver, artefacts_dir, "post_login")
        save_screenshot(driver, artefacts_dir, "post_login")

    return success, artefacts_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QA test for the login interface")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()
    logger.info(f"Running in headless mode: {args.headless}")

    run_ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    with get_driver(headless=args.headless) as driver:
        logger.info("Invoking run_login()...")
        ok, out = run_login(driver, run_ts=run_ts)
        if not ok:
            copy_trace_to_error_folder(out)

    logger.info(f"login_success={ok} artefacts_dir={out}")
