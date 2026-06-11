"""llm_step_node: calls the LLM, validates structured response, extracts code.

The first user message may contain key frame images (sighted mode).
Subsequent messages are text-only.
"""

import asyncio
import logging
import re
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from spatial_agent.llm.react_response_schema import ReactResponseValidator
from spatial_agent.llm.response_schema import LLMResponseValidator
from spatial_agent.state import AgentState

_logger = logging.getLogger(__name__)


async def llm_step_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Call the LLM and validate the structured response.

    On validation failure: increments ``failure_count``, appends an error
    message, and sets ``current_llm_response = None`` so ``execute_node``
    skips execution.
    """
    cfg = config["configurable"]
    llm_client = cfg["llm_client"]
    logger = cfg.get("logger")

    # Build messages for the LLM (text-only, no images)
    messages = _state_messages_to_openai(state["messages"])

    try:
        agent_config = cfg["agent_config"]
        raw_text, reasoning = await llm_client.generate(
            messages, role_params=agent_config.main_params,
            session_id=state["session_id"],
            usage_session_id=state["session_id"],
        )
    except Exception as exc:
        # LLM call failed entirely (timeout, connection error, API error)
        error_msg = f"LLM call failed: {type(exc).__name__}: {exc}"
        _logger.warning("[llm_step_node] %s (step %d)", error_msg, state["step_count"])

        if logger:
            logger.log_step(state["session_id"], {
                "event_type": "llm_call",
                "step_index": state["step_count"],
                "raw_response": "",
                "error": error_msg,
            })

        # Connection errors are transient infra issues — do NOT increment
        # failure_count (which would cause premature force-termination).
        from openai import APIConnectionError, APITimeoutError
        is_transient = isinstance(exc, (APIConnectionError, APITimeoutError,
                                        ConnectionError, OSError, asyncio.TimeoutError))

        return {
            "messages": [AIMessage(content=f"[LLM Error] {error_msg}")],
            "current_llm_response": None,
            "failure_count": state["failure_count"] if is_transient else state["failure_count"] + 1,
            "last_error_type": "llm_connection_error" if is_transient else "llm_call_failed",
        }

    # Validate structured response (code agent: markdown+code; react: markdown+JSON tool call)
    executor_type = getattr(agent_config, "executor_type", "code")
    try:
        if executor_type == "react":
            parsed = ReactResponseValidator.validate(raw_text, step=state["step_count"])
        else:
            parsed = LLMResponseValidator.validate(raw_text)
    except ValueError as exc:
        error_msg = str(exc)
        # Minimal placeholder so the assistant turn is represented in context.
        # Don't echo the raw malformed output — it's noise and bloats context.
        ai_msg = AIMessage(content=f"[Format Error] {error_msg}")
        feedback_msg = HumanMessage(
            content=(
                "Re-read the required format in the system prompt and try again."
            )
        )

        if logger:
            logger.log_step(state["session_id"], {
                "event_type": "llm_call",
                "step_index": state["step_count"],
                "raw_response": raw_text[:2000],
                "error": error_msg,
            })

        return {
            "messages": [ai_msg, feedback_msg],
            "current_llm_response": None,
            "failure_count": state["failure_count"] + 1,
            "last_error_type": "validation_failed",
        }

    # Success: extract structured response
    response_dict = {
        "purpose": parsed.purpose,
        "reasoning": parsed.reasoning,
        "next_goal": parsed.next_goal,
        "code": parsed.code,
    }

    # Build AI message with structured content. In ReAct mode, echo the
    # translated code as a python fence too — the original tool-call JSON is
    # preserved in parsed.raw_content and in the trace log, but downstream
    # reasoning (reflection, planning rehydration) reads code fences.
    body_label = "Tool Call (translated)" if executor_type == "react" else "Code"
    ai_content = (
        f"**Purpose**: {parsed.purpose}\n"
        f"**Reasoning**: {parsed.reasoning}\n"
        f"**Next Goal**: {parsed.next_goal}\n"
        f"**{body_label}**:\n```python\n{parsed.code}\n```"
    )
    ai_msg = AIMessage(content=ai_content)

    if logger:
        logger.log_step(state["session_id"], {
            "event_type": "llm_call",
            "step_index": state["step_count"],
            "raw_response": raw_text[:5000],
            "parsed_response": response_dict,
            "reasoning_content": reasoning,
        })

    return {
        "messages": [ai_msg],
        "current_llm_response": response_dict,
        "failure_count": 0,  # reset on success
        "last_error_type": None,
    }


def _state_messages_to_openai(messages) -> list:
    """Convert LangGraph messages to OpenAI-format dicts."""
    result = []
    for msg in messages:
        if hasattr(msg, "type"):
            role = {"system": "system", "human": "user", "ai": "assistant"}.get(
                msg.type, "user"
            )
        else:
            role = "user"

        # Preserve multimodal content (list of content parts) as-is;
        # only the first user message may contain image_url parts.
        if isinstance(msg.content, (str, list)):
            content = msg.content
        else:
            content = str(msg.content)
        result.append({"role": role, "content": content})
    return result
