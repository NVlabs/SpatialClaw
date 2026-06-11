"""plan_node: generates a concrete execution plan before coding begins.

Runs as an **isolated LLM session** with its own system prompt.  The planner
receives frame metadata (count, indices) but never sees actual images.
This prevents the planner from answering the question visually instead of
planning an investigation.

The agent never sees the planning prompt or reasoning — only the final plan
text is injected into the agent's message history.

Non-fatal: if planning fails, the node returns {} and the agent proceeds
directly to coding (same behavior as enable_planning=False).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from spatial_agent.llm.planning_prompt import (
    build_planning_system_prompt,
    build_planning_user_message,
)
from spatial_agent.llm.response_schema import LLMResponseValidator
from spatial_agent.state import AgentState, ChecklistItem

_logger = logging.getLogger(__name__)

TRANSITION_PROMPT_CODE = (
    "Good plan. Now execute it step by step. Each response MUST use the required markdown format "
    "(**Purpose**, **Reasoning**, **Next Goal**, **Code** with a ```python fenced block). "
    "Start with step 1 of your plan."
)

TRANSITION_PROMPT_REACT = (
    "Good plan. Now execute it step by step. Each response MUST use the required markdown format "
    "(**Purpose**, **Reasoning**, **Next Goal**, **Tool Call** with a ```json fenced block containing "
    "a single `{\"tool\": ..., \"args\": {...}}` object). "
    "Start with step 1 of your plan."
)

TRANSITION_PROMPT = TRANSITION_PROMPT_CODE  # backward-compat alias


async def plan_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate a concrete execution plan in an isolated LLM session.

    The planner receives frame metadata but never sees images.  The agent
    only receives the final plan text — it never sees the planning prompt
    or any reasoning.

    Does NOT count against max_steps. Non-fatal: returns {} on failure.
    """
    cfg = config["configurable"]
    agent_config = cfg["agent_config"]

    if not getattr(agent_config, "enable_planning", True):
        return {}

    # Skip planning in single-turn mode — the agent must solve in one step,
    # a multi-step plan would be misleading.
    if getattr(agent_config, "max_steps", 30) <= 1:
        return {}

    llm_client = cfg["llm_client"]
    logger = cfg.get("logger")
    key_frame_indices = cfg.get("key_frame_indices", [])
    key_frame_list_indices = cfg.get("key_frame_list_indices", [])
    key_frame_video_idx = cfg.get("key_frame_video_idx") or []
    metadata_obj = cfg.get("metadata_obj")
    input_images = cfg.get("input_images")
    input_images_list = cfg.get("input_images_list")
    if input_images_list:
        num_total_images = sum(len(ii) for ii in input_images_list)
    else:
        num_total_images = len(input_images) if input_images else 0
    num_ref_images = len(cfg.get("ref_images", []) or [])

    # Build isolated planning session messages.
    # Planner is blind: key_frames=None so it gets text-only input.
    planning_system = build_planning_system_prompt(
        agent_config, metadata_obj, num_ref_images=num_ref_images,
    )
    planning_user = build_planning_user_message(
        state["instruction"],
        key_frames=None,
        key_frame_indices=key_frame_indices,
        key_frame_list_indices=key_frame_list_indices,
        key_frame_video_idx=key_frame_video_idx,
        num_total_images=num_total_images,
    )

    messages_for_llm = [
        {"role": "system", "content": planning_system},
        planning_user,
    ]

    try:
        # Use a fixed session_id so all planning calls route to the same
        # endpoint, enabling vLLM prefix caching across concurrent plans.
        raw_text, reasoning = await llm_client.generate(
            messages_for_llm, role_params=agent_config.planning_params,
            session_id="__planning__",
            usage_session_id=state["session_id"],
        )
    except Exception as exc:
        _logger.warning("[plan_node] LLM call failed: %s — skipping planning", exc)
        if logger:
            logger.log_step(state["session_id"], {
                "event_type": "plan",
                "error": str(exc),
            })
        return {}

    # Strip thinking tokens
    plan_text = LLMResponseValidator._strip_thinking(raw_text)

    # Inject only the clean plan into the agent's message history.
    # The agent sees: "Here is your execution plan:" + plan + transition.
    # It does NOT see the planning system prompt or key frames.
    plan_injection = HumanMessage(
        content=f"Here is your execution plan:\n\n{plan_text}"
    )
    executor_type = getattr(agent_config, "executor_type", "code")
    transition_text = (
        TRANSITION_PROMPT_REACT if executor_type == "react" else TRANSITION_PROMPT_CODE
    )
    transition_msg = HumanMessage(content=transition_text)

    # Extract checklist items from plan (only when reflection is enabled)
    checklist: List[ChecklistItem] = []
    checklist_repair = False
    checklist_generated = False
    if getattr(agent_config, "enable_reflection", False):
        checklist, raw_section = _extract_checklist_from_plan(plan_text)
        # If a CHECKLIST section exists but parsing failed, try a repair call
        if not checklist and raw_section is not None:
            _logger.info("[plan_node] Checklist JSON parse failed — attempting repair call")
            checklist = await _repair_checklist(
                llm_client, agent_config, raw_section,
                usage_session_id=state["session_id"],
            )
            checklist_repair = bool(checklist)
        # If still no checklist (model skipped section entirely), generate one
        if not checklist:
            _logger.info("[plan_node] No checklist found — generating from scratch")
            checklist = await _generate_checklist(
                llm_client, agent_config, plan_text,
                usage_session_id=state["session_id"],
            )
            checklist_generated = bool(checklist)

    if logger:
        logger.log_step(state["session_id"], {
            "event_type": "plan",
            "plan_text": plan_text[:5000],
            "reasoning_content": reasoning,
            "checklist_items": len(checklist),
            "checklist_repair": checklist_repair,
            "checklist_generated": checklist_generated,
            "checklist": [dict(item) for item in checklist],
        })

    result: Dict[str, Any] = {
        "messages": [
            plan_injection,
            transition_msg,
        ],
        "plan": plan_text,
    }
    if checklist:
        result["checklist"] = checklist
    return result


def _extract_checklist_from_plan(
    plan_text: str,
) -> tuple[List[ChecklistItem], str | None]:
    """Parse ``### CHECKLIST`` section from plan text into ChecklistItems.

    Returns:
        (items, raw_section) — ``raw_section`` is the text of the CHECKLIST
        section if one was found (even if parsing failed), or ``None`` if
        no CHECKLIST heading exists at all.  This lets callers distinguish
        "no section" (nothing to retry) from "bad JSON" (worth retrying).
    """
    # Find the CHECKLIST section
    match = re.search(
        r"###?\s*(?:Verification\s+)?CHECKLIST\s*\n(.*?)(?:\n###?\s|\Z)",
        plan_text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return [], None

    section = match.group(1)
    items = _parse_checklist_json(section)
    return items, section


def _parse_checklist_json(text: str) -> List[ChecklistItem]:
    """Try to parse a JSON array of checklist items from *text*.

    Looks for a fenced code block first, then a bare JSON array.
    Returns an empty list on any parse failure.
    """
    raw_json = None
    # 1. Fenced code block
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        raw_json = m.group(1)
    else:
        # 2. Bare JSON array
        m = re.search(r"(\[.*\])", text, re.DOTALL)
        if m:
            raw_json = m.group(1)

    if not raw_json:
        return []

    try:
        parsed = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []

    valid_priorities = {"HIGH", "MEDIUM", "LOW"}
    items: List[ChecklistItem] = []
    for idx, entry in enumerate(parsed, start=1):
        if not isinstance(entry, dict):
            continue
        priority = str(entry.get("priority", "")).upper()
        description = str(entry.get("description", "")).strip()
        if priority not in valid_priorities or not description:
            continue
        items.append(ChecklistItem(
            item_id=f"chk_{idx:03d}",
            description=description,
            priority=priority,  # type: ignore[arg-type]
            status="PENDING",
            added_at_step=0,
            resolved_at_step=None,
            resolution_note=None,
        ))
    return items


_REPAIR_PROMPT = """\
The following text was supposed to contain a JSON array of verification \
checklist items, but it could not be parsed. Rewrite it as a valid JSON \
array. Output ONLY the JSON array — no explanation, no markdown fences.

Each element must be an object with exactly two keys:
  "priority": one of "HIGH", "MEDIUM", or "LOW"
  "description": a short verification task description

Broken text:
{raw_section}"""


async def _repair_checklist(
    llm_client,
    agent_config,
    raw_section: str,
    usage_session_id: Optional[str] = None,
) -> List[ChecklistItem]:
    """One-shot repair call: ask the LLM to fix malformed checklist JSON.

    Uses a fresh, minimal context (no plan text, no images) so the failed
    response is not carried forward. Returns an empty list if repair fails.
    """
    messages = [
        {"role": "user", "content": _REPAIR_PROMPT.format(raw_section=raw_section[:2000])},
    ]
    try:
        raw_text, _ = await llm_client.generate(
            messages, role_params=agent_config.planning_params,
            usage_session_id=usage_session_id,
        )
        clean = LLMResponseValidator._strip_thinking(raw_text)
        items = _parse_checklist_json(clean)
        if items:
            _logger.info("[plan_node] Checklist repair succeeded: %d items", len(items))
        else:
            _logger.warning("[plan_node] Checklist repair produced no valid items")
        return items
    except Exception as exc:
        _logger.warning("[plan_node] Checklist repair call failed: %s", exc)
        return []


_GENERATE_CHECKLIST_PROMPT = """\
The following is an execution plan for a spatial reasoning task. \
Generate a verification checklist for it. Output ONLY a JSON array — \
no explanation, no markdown fences.

Each element must be an object with exactly two keys:
  "priority": one of "HIGH", "MEDIUM", or "LOW"
  "description": a short verification task description

Focus on:
- HIGH: Things that, if wrong, make the answer wrong (wrong object identity, \
wrong mask, wrong coordinate frame)
- MEDIUM: Things that could affect accuracy (reconstruction quality, VLM \
coordinate precision)
- LOW: Nice-to-verify sanity checks

Generate 3-5 items.

Plan:
{plan_text}"""


async def _generate_checklist(
    llm_client,
    agent_config,
    plan_text: str,
    usage_session_id: Optional[str] = None,
) -> List[ChecklistItem]:
    """Generate a checklist from scratch when the model skipped it entirely.

    Uses a fresh, minimal context with the plan text. Returns an empty list
    if generation fails.
    """
    messages = [
        {"role": "user", "content": _GENERATE_CHECKLIST_PROMPT.format(
            plan_text=plan_text[:4000],
        )},
    ]
    try:
        raw_text, _ = await llm_client.generate(
            messages, role_params=agent_config.planning_params,
            usage_session_id=usage_session_id,
        )
        clean = LLMResponseValidator._strip_thinking(raw_text)
        items = _parse_checklist_json(clean)
        if items:
            _logger.info("[plan_node] Checklist generation succeeded: %d items", len(items))
        else:
            _logger.warning("[plan_node] Checklist generation produced no valid items")
        return items
    except Exception as exc:
        _logger.warning("[plan_node] Checklist generation call failed: %s", exc)
        return []
