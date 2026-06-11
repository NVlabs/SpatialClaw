"""Router: conditional edge logic and force_terminate node."""

import logging
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from spatial_agent.state import AgentState

logger = logging.getLogger(__name__)

# ── CoT fallback prompt (reused from cot_baseline.py) ──────────────────────

COT_FALLBACK_SYSTEM_PROMPT = """\
You are an expert visual spatial reasoning assistant.

You will be given one or more images (video frames) and a question about spatial relationships, motion, geometry, or scene understanding.

**Instructions:**
1. Carefully examine ALL provided images.
2. Reason step-by-step about the spatial relationships, motion, distances, orientations, or other relevant aspects.
3. After your reasoning, place your final answer inside \\boxed{}.
   - Multiple choice: \\boxed{B}
   - Numerical: \\boxed{3.5}
   - Word or phrase: \\boxed{left of the table}

Important:
- Consider temporal ordering of frames (earlier frames come first).
- Pay attention to camera motion, object motion, and spatial layout.
- If frames show a video sequence, reason about how objects and the scene change over time.
- Base your answer ONLY on what you can observe in the images."""


def should_continue(state: AgentState) -> str:
    """Determine the next node: continue, force terminate, or end.

    Returns:
        One of 'llm_step_node', 'force_terminate', or END.
    """
    if state.get("final_answer") is not None:
        return END

    if state["step_count"] >= state["max_steps"]:
        return "force_terminate"

    if state["failure_count"] >= state["max_failures"]:
        return "force_terminate"

    if state["max_tool_calls"] > 0 and state["total_tool_calls"] >= state["max_tool_calls"]:
        return "force_terminate"

    return "llm_step_node"


async def force_terminate(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Forcefully terminate the agent when escape conditions are hit.

    Falls back to a CoT VLM call with key frames for a best-guess answer.
    If the VLM call fails, falls back to regex extraction from conversation.
    """
    cfg = config.get("configurable", {})
    session_logger = cfg.get("logger")
    km = cfg.get("kernel_manager")
    llm_client = cfg.get("llm_client")
    key_frames = cfg.get("key_frames", [])
    agent_config = cfg.get("agent_config")

    # Determine reason
    reason = "unknown"
    if state["step_count"] >= state["max_steps"]:
        reason = "max_steps"
    elif state["failure_count"] >= state["max_failures"]:
        reason = "max_failures"
    elif state["max_tool_calls"] > 0 and state["total_tool_calls"] >= state["max_tool_calls"]:
        reason = "max_tool_calls"

    # ── Use last submitted answer if available (e.g. rejected by reflection) ──
    partial_answer = ""
    fallback_method = "none"

    last_submitted = state.get("last_submitted_answer")
    if last_submitted and last_submitted.get("raw_value"):
        partial_answer = str(last_submitted["raw_value"])
        fallback_method = "last_submitted"
        logger.info("Using last submitted answer: %s", partial_answer[:200])

    # ── CoT fallback: only if no prior answer exists ────────────────────
    if not partial_answer and llm_client and key_frames:
        cot_question = (
            state["instruction"]
            + "\n\nThink step-by-step, then place your final answer inside \\boxed{}."
        )
        try:
            vlm_response = await llm_client.generate_vision_query(
                images=key_frames,
                question=cot_question,
                system_prompt=COT_FALLBACK_SYSTEM_PROMPT,
                role_params=agent_config.general_params if agent_config else None,
                session_id=state["session_id"],
                usage_session_id=state["session_id"],
            )
            parsed = _parse_boxed_answer(vlm_response)
            partial_answer = parsed if parsed else vlm_response.strip()
            fallback_method = "cot_vlm"
            logger.info("CoT fallback produced answer: %s", partial_answer[:200])
        except Exception as exc:
            logger.warning("CoT fallback VLM call failed: %s — falling back to regex", exc)
            partial_answer = _extract_partial_answer(state)
            fallback_method = "regex"

    if not partial_answer:
        partial_answer = _extract_partial_answer(state)
        fallback_method = fallback_method or "regex"

    # Checklist stats for post-hoc analysis
    checklist = state.get("checklist", [])
    high_pending = sum(
        1 for item in checklist
        if item.get("priority") == "HIGH" and item.get("status") == "PENDING"
    )

    # Build summary
    summary = (
        f"Agent terminated: {reason}. "
        f"Steps: {state['step_count']}/{state['max_steps']}, "
        f"Failures: {state['failure_count']}/{state['max_failures']}, "
        f"Tool calls: {state['total_tool_calls']}/{state['max_tool_calls']}. "
        f"Fallback: {fallback_method}."
    )
    if checklist:
        summary += (
            f" Checklist: {high_pending} HIGH pending out of {len(checklist)} total."
        )
    if partial_answer:
        summary += f" Best guess: {partial_answer[:200]}"

    final_answer = {
        "text": partial_answer or "",
        "mode": "narrative",
        "raw_value": partial_answer or "",
    }

    if session_logger:
        session_logger.log_step(state["session_id"], {
            "event_type": "termination",
            "reason": reason,
            "fallback_method": fallback_method,
            "summary": summary,
            "final_answer": final_answer,
            "checklist_high_pending": high_pending,
            "checklist_total": len(checklist),
        })

    # Do NOT shut down the shared kernel — it's reused across samples.
    # Just clear the sentinel in case of partial state.
    if km:
        try:
            await km.clear_sentinel()
        except Exception:
            pass

    return {
        "messages": [HumanMessage(content=f"[System] {summary}")],
        "final_answer": final_answer,
        "termination_reason": reason,
    }


def _parse_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{X} in VLM response. Returns '' if not found."""
    # Match \boxed{...} — typically a single letter for MC, but allow any content
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_partial_answer(state: AgentState) -> str:
    """Try to extract a partial answer from conversation or variables."""

    # Look for any ReturnAnswer-like variables
    for name, info in state.get("variable_registry", {}).items():
        if "answer" in name.lower() or "result" in name.lower():
            return f"{name} = {info}"

    # Scan messages for MC letter patterns or answer content
    messages = state.get("messages", [])
    for msg in reversed(messages[-10:]):
        content = msg.content if hasattr(msg, "content") else str(msg)
        # Look for explicit answer mentions like "answer is B" or "The answer: A"
        m = re.search(r"(?:answer|choice)\s*(?:is|:)\s*([A-Z])\b", content, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Look for ReturnAnswer calls in code
        m = re.search(r'ReturnAnswer\(["\']([A-Z])["\']\)', content)
        if m:
            return m.group(1).upper()

    return ""
