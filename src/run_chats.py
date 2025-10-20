import logging
import os
import re
import time
import json
import random
import uuid
from datetime import datetime
from typing import Optional, Tuple, Any, Dict, List
from pathlib import Path

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

from .run_login import run_login
from .utils.files import read_yaml, ensure_artefacts_dir, save_html, save_screenshot
from .utils.selenium import get_driver
from .utils.tools import build_chat_tools


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("chats_runner")


# A simple global holder for artefacts dir used inside tool closure
graph_artefacts_dir: List[str] = [""]


def contains_html(content: str) -> bool:
    """Check if content contains substantial HTML (indicating page source dumps)."""
    if not isinstance(content, str):
        return False
    # Look for common HTML patterns that indicate page source
    html_indicators = ['<!DOCTYPE', '<html', '<head>', '<body>', '<div', '<script', '<span']
    return any(indicator in content for indicator in html_indicators)


def truncate_html_tool_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Truncate HTML content in older ToolMessages to reduce token usage while maintaining message structure.
    Returns both the filtered list (for immediate use) and RemoveMessage instances (for state cleanup).

    Strategy:
    - Keep all messages intact (to maintain tool_call/response pairing)
    - For ToolMessages with HTML content, keep the LAST one at full length
    - Truncate all EARLIER ToolMessages with HTML to first 100 and last 100 characters
    - This preserves conversation structure and latest page state while significantly reducing tokens
    """
    # First, identify all ToolMessages with HTML
    tool_messages_with_html = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and contains_html(str(msg.content)):
            tool_messages_with_html.append(msg)

    # Get the ID of the last HTML ToolMessage (to keep it full)
    last_html_msg_id = tool_messages_with_html[-1].id if tool_messages_with_html else None

    # Now process all messages
    messages_to_return = []
    truncated_count = 0

    for msg in messages:
        if isinstance(msg, ToolMessage) and contains_html(str(msg.content)):
            # Keep the last HTML message at full length
            if msg.id == last_html_msg_id:
                messages_to_return.append(msg)
            else:
                # Truncate older HTML messages
                content = str(msg.content)
                if len(content) > 300:  # Only truncate if significantly longer than 200 chars
                    truncated_content = content[:100] + f"\n... [truncated {len(content) - 200} characters] ...\n" + content[-100:]
                    # Create a new ToolMessage with truncated content
                    truncated_msg = ToolMessage(
                        content=truncated_content,
                        tool_call_id=msg.tool_call_id,
                        id=msg.id
                    )
                    messages_to_return.append(truncated_msg)
                    truncated_count += 1
                else:
                    messages_to_return.append(msg)
        else:
            messages_to_return.append(msg)

    if truncated_count > 0:
        logger.info(f"Truncated HTML content in {truncated_count} older ToolMessage(s), kept latest at full length")

    return messages_to_return


def build_tools_chat(driver: webdriver.Chrome):
    """Build tools for interacting with the chat interface."""
    return build_chat_tools(driver, graph_artefacts_dir)


def message_to_dict(msg: BaseMessage) -> dict:
    """Convert a LangChain message to a JSON-serializable dict."""
    result = {"type": msg.__class__.__name__, "content": msg.content}
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")}
            for tc in msg.tool_calls
        ]
    if hasattr(msg, "tool_call_id"):
        result["tool_call_id"] = msg.tool_call_id
    return result


def run_and_save_execution_trace(stream, artefacts_dir: Path) -> Dict[str, Any]:
    """Save execution trace to JSON and return final state info."""
    trace = {
        "run_id": str(uuid.uuid4()),
        "run_type": artefacts_dir.name,
        "timestamp": datetime.utcnow().isoformat(),
        "steps": []
    }
    final_state = None

    for step in stream:
        final_state = step
        step_data = {}
        for node_name, node_state in step.items():
            step_data[node_name] = {
                "status": node_state.get("status"),
                "health": node_state.get("health"),
                "health_description": node_state.get("health_description"),
                "goal": node_state.get("goal"),
                "messages": [message_to_dict(msg) for msg in node_state.get("messages", [])],
                "artefacts_dir": node_state.get("artefacts_dir"),
            }
        trace["steps"].append(step_data)
        logger.debug(f"Captured step {len(trace['steps'])}: {list(step.keys())}")

    if final_state:
        for node_state in final_state.values():
            if "status" in node_state:
                trace["final_status"] = node_state.get("status")
            if "health" in node_state:
                trace["final_health"] = node_state.get("health")
            if "health_description" in node_state:
                trace["final_health_description"] = node_state.get("health_description")
        trace["total_steps"] = len(trace["steps"])

    trace_file = artefacts_dir / "execution_trace.json"
    with trace_file.open("w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
    logger.info(f"Execution trace saved to: {trace_file}")

    if final_state:
        for node_state in final_state.values():
            return {
                "health": node_state.get("health", "UNKNOWN"),
                "health_description": node_state.get("health_description", "No description"),
            }
    return {"health": "UNKNOWN", "health_description": "No final state"}


def build_graph_chat(driver: webdriver.Chrome, initial_html_cleaned: str, num_turns: int, artefacts_dir: Path, goal: str):
    """Build the LangGraph for chat interaction."""
    tools = build_tools_chat(driver)
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0).bind_tools(tools)

    class State(MessagesState):
        num_turns: int
        turns_completed: int
        status: str
        health: str
        health_description: str
        goal: str
        artefacts_dir: str

    def agent_node(state: State) -> Dict[str, Any]:
        time.sleep(2)

        logger.debug("[agent] invoking model with messages")
        # Filter messages to reduce tokens for this invocation AND get RemoveMessages for state cleanup
        messages = truncate_html_tool_messages(state["messages"])

        # print summary of filtered messages
        logger.info("Filtered messages summary:")
        for msg in messages:
            msg_type = type(msg).__name__
            content = getattr(msg, "content", "")
            logger.info(f"  {msg_type}: content length={len(content) if content is not None else 0}")

        # Invoke the model with the potentially truncated messages (helps avoid token limit on this call)
        response = model.invoke(messages)
        return {"messages": [response]}

    def tools_node(state: State) -> Dict[str, Any]:
        """Execute tools and extract health/status info from report_completion calls."""
        # Execute tools using ToolNode
        tool_node = ToolNode(tools)
        result = tool_node.invoke(state)

        # Check if report_completion was called and extract health info
        updates = {"messages": result["messages"]}
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls"):
            for tool_call in last_message.tool_calls:
                if tool_call.get("name") == "report_completion":
                    args = tool_call.get("args", {})
                    updates["health"] = args.get("health", "UNKNOWN")
                    updates["health_description"] = args.get("health_description", "")
                    updates["status"] = "completed"
                    logger.info(f"Extracted completion: health={updates['health']}")

        return updates

    def check_node(state: State) -> Dict[str, Any]:
        """Check if we've completed the required number of chat turns."""
        if state.get("status") == "completed":
            # Preserve health fields when completing
            return {
                "status": "completed",
                "health": state.get("health"),
                "health_description": state.get("health_description"),
            }
        return {"status": "continue"}

    def should_continue(state: State):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "check"

    def route_after_check(state: State):
        return "end" if state.get("status") == "completed" else "loop"

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("check", check_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "check": "check"})
    graph.add_edge("tools", "check")
    graph.add_conditional_edges("check", route_after_check, {"end": END, "loop": "agent"})

    app = graph.compile()

    initial_messages: List[AnyMessage] = [
        SystemMessage(content=(
            "You are an automation agent controlling a browser to test a multi-LLM chat interface. "
            "Your task is to conduct a SMALL chat conversation to verify it's working correctly. "
            f"You must complete exactly {num_turns} turn(s) of conversation. "
            "Use get_page_html to understand the page structure, after carrying out any actions, as the page may update compared to the original HTML supplied to you."
            "Each turn: 1) type a message, 2) click submit, 3) wait for ALL responses to complete.\n\n"
            "IMPORTANT: The submit button may not appear until you have started typing your message. You will have to re-fetch the HTML after entering text, in order to see the appropriate button."
            "IMPORTANT: After submitting, it takes a few seconds for all LLM responses to return, the length depending on the complexity of the request of the request."
            "Whilst generating responses, the text area will become disabled somehow, and there is an extra generating spinner."
            "Use your best judgment to determine when all responses are complete or whether one or more are still generating."
            "Keep conversation SMALL and simple - brief pleasantries like:\n"
            "- Turn 1: 'Hi' or 'Hello'\n"
            "- Turn 2: 'What's your name?' or 'How are you?'\n"
            "- Turn 3: 'What can you do?' or 'Tell me a joke'\n\n"
            "After completing all turns:\n"
            "1. Inspect final HTML using get_page_html to examine responses\n"
            "2. Check if responses look reasonable (not empty, not error messages)\n"
            "3. Use the report_completion tool to report health status:\n"
            "   - health: 'OK' if all responses appear normal\n"
            "   - health: 'ERROR' if you detect issues (empty responses, error messages, missing responses, etc.)\n"
            "   - health_description: Brief explanation of what you found\n\n"
            "Use save_chat_capture to save screenshots after each turn for debugging.\n"
            "Only use tools; be systematic and thorough."
        )),
        HumanMessage(content=(
            f"Instructions: {goal}\n"
            f"Number of turns to complete: {num_turns}\n"
            f"Initial HTML (cleaned): {initial_html_cleaned}"
        )),
    ]

    state: State = {
        "messages": initial_messages,
        "num_turns": num_turns,
        "turns_completed": 0,
        "status": "start",
        "health": "UNKNOWN",
        "health_description": "",
        "goal": goal,
        "artefacts_dir": str(artefacts_dir),
    }

    return app, state


def run_chats(driver: webdriver.Chrome, profile: Optional[str] = None, run_ts: str = None) -> Tuple[bool, Path]:
    """
    Run chat tests on a multi-LLM chat interface.

    Args:
        driver: Selenium Chrome driver that is already logged in
        profile: Optional profile name (unused for chats, kept for API consistency)
        run_ts: Optional timestamp for artefacts directory

    Returns:
        Tuple of (success: bool, artefacts_dir: Path)
        success is True if health status is 'OK', False otherwise
    """
    load_dotenv(dotenv_path=Path(".env"), override=False)

    # Load configuration
    state_path = Path("config/state.yaml")
    run_state = read_yaml(state_path)
    logger.info(f"Loaded run state from: {state_path}")

    goal_text = str(run_state.get("run_chats", {}).get("instructions", "Have a small conversation with the chat interface."))
    logger.info(f"Instructions: {goal_text}")

    # Setup artefacts directory
    artefacts_dir = ensure_artefacts_dir(subfolder="run_chats", ts=run_ts)
    graph_artefacts_dir[0] = str(artefacts_dir)
    logger.info(f"Artefacts directory: {artefacts_dir}")

    # Driver is already logged in and at the chat interface
    # Capture initial state
    logger.info("Capturing initial page state...")
    save_html(driver, artefacts_dir, "initial")
    save_screenshot(driver, artefacts_dir, "initial")

    initial_html = driver.page_source
    initial_html_cleaned = re.sub(r"<script\b[^>]*>[\s\S]*?<\/script>", "", initial_html, flags=re.IGNORECASE)

    # Determine number of turns (1 or 2 random)
    num_turns = random.randint(1, 2)
    logger.info(f"Will conduct {num_turns} turn(s) of conversation")

    # Build and execute the graph
    logger.info("Building chat interaction graph...")
    app, state = build_graph_chat(driver, initial_html_cleaned, num_turns, artefacts_dir, goal_text)
    logger.info("Graph compiled. Beginning execution loop...")

    # Execute the agent
    final_state_info = run_and_save_execution_trace(
        app.stream(state, config={"recursion_limit": 100}),
        artefacts_dir
    )

    # Extract health status
    health = final_state_info.get("health", "UNKNOWN")
    health_description = final_state_info.get("health_description", "No description")
    logger.info(f"Chat health status: {health}")
    logger.info(f"Health description: {health_description}")

    # Final capture
    logger.info("Saving final HTML and screenshot...")
    save_html(driver, artefacts_dir, "final")
    save_screenshot(driver, artefacts_dir, "final")

    # Success is True if health is 'OK'
    success = (health == "OK")

    return success, artefacts_dir


if __name__ == "__main__":
    run_ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    with get_driver() as driver:
        logger.info("Invoking run_login()...")
        ok, out = run_login(driver, run_ts=run_ts)

        time.sleep(10)

        logger.info("Invoking run_chats()...")
        ok, out = run_chats(driver, run_ts=run_ts)

        # Final digest
        logger.info(f"login_success={ok} artefacts_dir={out}")
        logger.info(f"chats_success={ok} artefacts_dir={out}")


