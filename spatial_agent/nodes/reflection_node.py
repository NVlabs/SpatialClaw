"""reflection_node: optional self-reflection after each execution step.

Reviews the conversation so far and flags concerns (logical errors, geometric
mistakes, goal drift, dead loops) before the router decides the next step.
Also manages the verification checklist — verifying, flagging, or adding items.
"""

import copy
import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from spatial_agent.kernel.feedback_collector import FeedbackCollector
from spatial_agent.llm.reflection_prompt import build_reflection_user_message
from spatial_agent.llm.response_schema import LLMResponseValidator
from spatial_agent.state import AgentState, ChecklistItem

_logger = logging.getLogger(__name__)


async def reflection_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Review the agent's latest step and optionally inject a warning.

    When ``final_answer`` is set (ReturnAnswer was called), reflection always
    runs and acts as the gatekeeper — rejecting the answer with a reason and
    suggestion if it finds a concern, or accepting it by returning ``{}``.

    Skips silently (returns ``{}``) when:
    - ``enable_reflection`` is False
    - ``current_llm_response`` is None (validation/execution failed)
    - No ``final_answer`` and ``step_count <= 1`` (too early to reflect)

    On internal error: if an answer is pending, rejects it so the agent
    retries rather than accepting unchecked.  Otherwise returns ``{}``.
    """
    cfg = config["configurable"]
    agent_config = cfg["agent_config"]

    # --- guard clauses ---
    if not agent_config.enable_reflection:
        return {}
    if state.get("current_llm_response") is None:
        return {}

    has_final_answer = state.get("final_answer") is not None
    step = state.get("step_count", 0)
    reflect_every = getattr(agent_config, "reflect_every_n_steps", 1)

    # Always reflect when ReturnAnswer was called (gates answer acceptance).
    # Otherwise only reflect every N steps, and skip step <= 1 (too early).
    if not has_final_answer:
        if step <= 1:
            return {}
        if step % reflect_every != 0:
            return {}

    try:
        return await _run_reflection(state, cfg, agent_config, reflect_every)
    except Exception as exc:
        _logger.warning("[reflection_node] Skipping due to error: %s", exc)
        # If an answer was submitted but reflection failed, reject it so the
        # agent retries — otherwise the answer passes through unchecked.
        if has_final_answer:
            budget_remaining = state["max_steps"] - state["step_count"]
            total_attempts = state.get("total_answer_attempts", 0)
            min_budget = agent_config.min_budget_for_answer_reject
            max_total = agent_config.max_total_answer_attempts
            budget_ok = min_budget < 0 or budget_remaining > min_budget
            attempts_ok = max_total < 0 or total_attempts < max_total
            if budget_ok and attempts_ok:
                _logger.warning("[reflection_node] Rejecting answer due to reflection error")
                return {
                    "final_answer": None,
                    "termination_reason": None,
                    "messages": [HumanMessage(
                        content="[ANSWER REJECTED] Self-reflection encountered an error. "
                        "Please verify your answer and resubmit."
                    )],
                }
        return {}


async def _run_reflection(state, cfg, agent_config, reflect_every: int = 1) -> Dict[str, Any]:
    """Core reflection logic, separated for clean error handling."""
    from spatial_agent.nodes.llm_step_node import _state_messages_to_openai

    llm_client = cfg["llm_client"]
    logger = cfg.get("logger")

    # Build checklist text for the prompt
    checklist = state.get("checklist", [])
    checklist_text = FeedbackCollector.format_checklist(checklist)

    # Build messages: keep original system prompt + conversation (KV cache prefix),
    # append reflection instruction as final user message
    openai_messages = _state_messages_to_openai(state["messages"])
    reflection_prompt = build_reflection_user_message(checklist_text, reflect_every)
    openai_messages.append({"role": "user", "content": reflection_prompt})

    # Call LLM with reflection params
    raw_text, _reasoning = await llm_client.generate(
        openai_messages, role_params=agent_config.reflection_params,
        session_id=state["session_id"],
        usage_session_id=state["session_id"],
    )

    # Strip thinking tokens
    clean = LLMResponseValidator._strip_thinking(raw_text)

    # Parse JSON response with fallbacks
    parsed = _parse_reflection_response(clean)

    # Apply checklist operations if present
    updated_checklist = None
    checklist_ops = parsed.get("checklist_ops", [])
    if checklist_ops:
        updated_checklist = _apply_checklist_ops(
            checklist, checklist_ops, state.get("step_count", 0)
        )

    # Log
    if logger:
        logger.log_step(state["session_id"], {
            "event_type": "reflection",
            "step_index": state["step_count"],
            "status": parsed.get("status", "unknown"),
            "explanation": parsed.get("explanation", ""),
            "checklist_ops": checklist_ops if checklist_ops else [],
            "raw_response": raw_text[:2000],
        })

    result: Dict[str, Any] = {}

    has_final_answer = state.get("final_answer") is not None
    is_concern = parsed.get("status") == "concern" and parsed.get("explanation")

    # If concern, inject warning into messages
    if is_concern:
        explanation = parsed["explanation"]
        suggestion = parsed.get("suggestion", "")
        _logger.info("[reflection_node] Step %d concern: %s", state["step_count"], explanation)

        # If ReturnAnswer was submitted, reject it on concern
        # so the agent can fix the issue and re-submit.
        # But don't reject if budget is too low — force_terminate is worse
        # than a slightly wrong answer.
        if has_final_answer:
            budget_remaining = state["max_steps"] - state["step_count"]
            total_attempts = state.get("total_answer_attempts", 0)
            min_budget = agent_config.min_budget_for_answer_reject
            max_total = agent_config.max_total_answer_attempts
            budget_ok = min_budget < 0 or budget_remaining > min_budget
            attempts_ok = max_total < 0 or total_attempts < max_total
            if budget_ok and attempts_ok:
                _logger.info("[reflection_node] Rejecting ReturnAnswer due to concern")
                result["final_answer"] = None
                result["termination_reason"] = None

                # Build rejection message with reason, suggestion, and protocol
                rejected_text = state["final_answer"].get("raw_value", "")
                parts = [
                    f"[ANSWER REJECTED] Your answer \"{rejected_text}\" was rejected.",
                    f"Reason: {explanation}",
                ]
                if suggestion:
                    parts.append(f"Suggestion: {suggestion}")
                parts.append(
                    "\nBefore resubmitting, you MUST do ONE of the following:"
                    "\n- Address the concern: run the suggested verification or fix the flagged issue, then resubmit (same or different answer)."
                    "\n- Refute the concern: if you believe your answer is correct despite the rejection, explain concretely why the concern does not apply, then resubmit."
                    "\n- Acknowledge and move on: if you cannot address or refute the concern (tool failed, data too noisy), state this explicitly and submit your best answer."
                    "\nDo NOT resubmit without doing one of the above — explain your reasoning first."
                )
                result["messages"] = [HumanMessage(content="\n".join(parts))]

                if logger:
                    logger.log_step(state["session_id"], {
                        "event_type": "answer_rejected",
                        "step_index": state["step_count"],
                        "rejected_answer": state["final_answer"],
                        "reason": explanation,
                        "suggestion": suggestion,
                    })
            else:
                # Budget too low — accept the answer despite concern
                _logger.info(
                    "[reflection_node] Concern found but accepting answer "
                    "(budget=%d, attempts=%d)", budget_remaining, total_attempts,
                )
        else:
            # No final answer — just a mid-step warning
            warning = f"[Self-Reflection] Warning: {explanation}"
            if suggestion:
                warning += f"\nSuggestion: {suggestion}"
            result["messages"] = [HumanMessage(content=warning)]

    # Update checklist if ops were applied
    if updated_checklist is not None:
        result["checklist"] = updated_checklist
        # Notify the main LLM about checklist changes so it doesn't have to
        # wait until the next feedback_node to see updated statuses.
        if "messages" not in result:
            summary = _summarize_checklist_ops(checklist_ops)
            if summary:
                result["messages"] = [HumanMessage(content=summary)]

    return result


def _summarize_checklist_ops(ops: List[Dict[str, Any]]) -> str:
    """Build a short message summarizing checklist changes for the main LLM."""
    parts = []
    for op_dict in ops:
        if not isinstance(op_dict, dict):
            continue
        op = op_dict.get("op", "")
        if op == "verify":
            note = op_dict.get("note", "")
            parts.append(f"  VERIFIED {op_dict.get('item_id', '?')}: {note}" if note
                         else f"  VERIFIED {op_dict.get('item_id', '?')}")
        elif op == "flag":
            note = op_dict.get("note", "")
            parts.append(f"  FLAGGED {op_dict.get('item_id', '?')}: {note}" if note
                         else f"  FLAGGED {op_dict.get('item_id', '?')}")
        elif op == "add":
            parts.append(f"  ADDED [{op_dict.get('priority', '?')}] {op_dict.get('description', '?')}")
    if not parts:
        return ""
    return "[Checklist Update]\n" + "\n".join(parts)


MAX_CHECKLIST_ITEMS = 10


def _apply_checklist_ops(
    checklist: List[ChecklistItem],
    ops: List[Dict[str, Any]],
    step_count: int,
) -> List[ChecklistItem]:
    """Apply checklist operations from the reflection response.

    Deep-copies the list and applies verify/flag/add operations.
    Silently skips malformed ops or unknown item_ids.

    Caps total items at ``MAX_CHECKLIST_ITEMS``. Reflection ``add`` ops
    with ``priority: "HIGH"`` are downgraded to ``"MEDIUM"`` — only the
    planning node (which has full task context) may create HIGH items.
    """
    updated = copy.deepcopy(checklist)
    id_map = {item["item_id"]: item for item in updated}

    # Track next ID for new items
    max_id = 0
    for item in updated:
        m = re.match(r"chk_(\d+)", item["item_id"])
        if m:
            max_id = max(max_id, int(m.group(1)))

    for op_dict in ops:
        if not isinstance(op_dict, dict):
            continue
        op = op_dict.get("op", "")

        if op == "verify":
            item_id = op_dict.get("item_id", "")
            if item_id in id_map:
                id_map[item_id]["status"] = "VERIFIED"
                id_map[item_id]["resolved_at_step"] = step_count
                id_map[item_id]["resolution_note"] = op_dict.get("note", "")

        elif op == "flag":
            item_id = op_dict.get("item_id", "")
            if item_id in id_map:
                id_map[item_id]["status"] = "FLAGGED"
                id_map[item_id]["resolved_at_step"] = step_count
                id_map[item_id]["resolution_note"] = op_dict.get("note", "")

        elif op == "add":
            # Cap total items to prevent unbounded growth
            if len(updated) >= MAX_CHECKLIST_ITEMS:
                continue
            desc = op_dict.get("description", "")
            priority = op_dict.get("priority", "MEDIUM").upper()
            if desc and priority in ("HIGH", "MEDIUM", "LOW"):
                # Reflection cannot add HIGH items — downgrade to MEDIUM.
                # Only the plan node (with full task context) creates HIGH items.
                if priority == "HIGH":
                    priority = "MEDIUM"
                max_id += 1
                new_item = ChecklistItem(
                    item_id=f"chk_{max_id:03d}",
                    description=desc,
                    priority=priority,  # type: ignore[arg-type]
                    status="PENDING",
                    added_at_step=step_count,
                    resolved_at_step=None,
                    resolution_note=None,
                )
                updated.append(new_item)
                id_map[new_item["item_id"]] = new_item

    return updated


def _parse_reflection_response(text: str) -> dict:
    """Parse the reflection JSON response with multiple fallbacks.

    Tries: direct parse → code block extraction → regex → heuristic.
    """
    # 1. Direct JSON parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Extract from code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Regex for JSON object anywhere in text (allow nested for checklist_ops)
    # Try to find the outermost { ... } that contains "status"
    for pattern in [
        r"\{[^{}]*\"status\"\s*:\s*\"[^\"]+\"[^{}]*\"checklist_ops\"\s*:\s*\[.*?\][^{}]*\}",
        r"\{[^{}]*\"status\"\s*:\s*\"[^\"]+\"[^{}]*\}",
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

    # 4. Heuristic: check for "concern" keyword
    if "concern" in text.lower():
        return {"status": "concern", "explanation": text.strip()[:500]}

    return {"status": "ok"}
