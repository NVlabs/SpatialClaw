"""Model-opt-in direct-answer anchor, evidence gate, and reserved final node."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from spatial_agent.llm.answer_policy import (
    ADJUDICATION_SYSTEM_PROMPT,
    DIRECT_ANSWER_SYSTEM_PROMPT,
    allowed_values,
    build_adjudication_question,
    build_direct_question,
    normalize_schema_answer,
)
from spatial_agent.llm.client import LLMGenerationTimeoutError
from spatial_agent.state import AgentState


logger = logging.getLogger(__name__)


def _warning_fail_open(message: str, *args: Any) -> None:
    """Keep logging-handler I/O failures out of controller decisions."""
    try:
        logger.warning(message, *args)
    except Exception:
        pass


def _log_step_fail_open(
    session_logger: Any,
    session_id: str,
    record: Dict[str, Any],
) -> None:
    """Emit diagnostic telemetry without letting logger failures affect a sample."""
    if session_logger is None:
        return
    try:
        session_logger.log_step(session_id, record)
    except Exception as exc:
        _warning_fail_open(
            "Diagnostic telemetry write failed for %s (%s): %s",
            session_id,
            record.get("event_type", "unknown"),
            exc,
        )


async def _generate_direct_candidate(
    state: AgentState,
    cfg: Dict[str, Any],
    *,
    event_type: str,
) -> tuple[
    Optional[str],
    Optional[str],
    Optional[LLMGenerationTimeoutError],
]:
    llm_client = cfg.get("llm_client")
    key_frames = cfg.get("key_frames") or []
    agent_config = cfg.get("agent_config")
    session_logger = cfg.get("logger")
    if llm_client is None or not key_frames or agent_config is None:
        return None, None, None

    question = build_direct_question(
        state["instruction"], state.get("answer_schema")
    )
    raw: Optional[str] = None
    error: Optional[str] = None
    candidate: Optional[str] = None
    generation_timeout: Optional[LLMGenerationTimeoutError] = None
    try:
        raw = await llm_client.generate_vision_query(
            images=key_frames,
            question=question,
            system_prompt=DIRECT_ANSWER_SYSTEM_PROMPT,
            role_params=agent_config.direct_answer_params,
            session_id=state["session_id"],
            usage_session_id=state["session_id"],
        )
        candidate = normalize_schema_answer(raw, state.get("answer_schema"))
    except LLMGenerationTimeoutError as exc:
        error = str(exc)
        generation_timeout = exc
        _warning_fail_open(
            "Direct answer call failed for %s: %s", state["session_id"], exc
        )
    except Exception as exc:  # deterministic fallback is handled by callers
        error = str(exc)
        _warning_fail_open(
            "Direct answer call failed for %s: %s", state["session_id"], exc
        )

    event = {
        "event_type": event_type,
        "candidate": candidate,
        "raw_response": (raw or "")[:3000],
        "error": error,
        "terminal_generation_timeout": generation_timeout is not None,
    }
    if generation_timeout is not None:
        event.update({
            "error_type": type(generation_timeout).__name__,
            "generation_timeout_request_kind": generation_timeout.request_kind,
            "generation_timeout_attempts": generation_timeout.attempts,
        })
    _log_step_fail_open(session_logger, state["session_id"], event)
    return candidate, raw[:3000] if raw is not None else None, generation_timeout


async def direct_answer_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Generate a controller-only direct candidate from the initial key frames."""
    cfg = config["configurable"]
    agent_config = cfg["agent_config"]
    if not getattr(agent_config, "direct_answer_anchor", False):
        return {}

    candidate, raw, generation_timeout = await _generate_direct_candidate(
        state, cfg, event_type="direct_answer_anchor"
    )
    updates: Dict[str, Any] = {
        "direct_candidate": candidate,
        "direct_candidate_raw": raw,
    }
    if generation_timeout is not None:
        updates["llm_generation_timed_out"] = True
        updates["llm_generation_timeout_attempts"] = generation_timeout.attempts
    if candidate is not None:
        updates["messages"] = [HumanMessage(content=(
            f"[Controller prior] Initial direct visual candidate: {candidate}. "
            "Revise it only if a targeted observation provides concrete contrary evidence."
        ))]
    return updates


async def _adjudicate_with_timeout(
    state: AgentState,
    cfg: Dict[str, Any],
    direct_candidate: Optional[str],
    proposed_answer: Optional[str],
) -> tuple[
    Optional[str],
    str,
    Optional[LLMGenerationTimeoutError],
]:
    """Run one constrained visual adjudication; fail closed to the anchor."""
    schema = state.get("answer_schema")
    direct_candidate = normalize_schema_answer(direct_candidate, schema)
    proposed_answer = normalize_schema_answer(proposed_answer, schema)

    if direct_candidate is not None and proposed_answer == direct_candidate:
        return direct_candidate, "agreement", None
    if direct_candidate is None:
        return proposed_answer, "proposal_without_anchor", None
    if state.get("evidence_count", 0) <= 0 and (
        proposed_answer is None
        or not getattr(cfg.get("agent_config"), "adjudicate_when_no_evidence", False)
    ):
        return direct_candidate, "anchor_no_evidence", None

    llm_client = cfg.get("llm_client")
    key_frames = cfg.get("key_frames") or []
    evidence_images = list(state.get("evidence_images") or [])[-2:]
    agent_config = cfg.get("agent_config")
    session_logger = cfg.get("logger")
    if llm_client is None or not key_frames or agent_config is None:
        return direct_candidate, "anchor_adjudication_unavailable", None

    question = build_adjudication_question(
        state["instruction"],
        schema,
        direct_candidate,
        proposed_answer,
        state.get("evidence_summaries") or [],
        restrict_to_candidates=getattr(
            agent_config, "restrict_adjudicator_to_candidates", False
        ),
    )
    raw = ""
    error = None
    selected = None
    rejected_answer = None
    generation_timeout: Optional[LLMGenerationTimeoutError] = None
    try:
        raw = await llm_client.generate_vision_query(
            images=key_frames + evidence_images,
            question=question,
            system_prompt=ADJUDICATION_SYSTEM_PROMPT,
            role_params=agent_config.direct_answer_params,
            session_id=state["session_id"],
            usage_session_id=state["session_id"],
        )
        selected = normalize_schema_answer(raw, schema)
        if getattr(agent_config, "restrict_adjudicator_to_candidates", False):
            candidates = {
                candidate
                for candidate in (direct_candidate, proposed_answer)
                if candidate is not None
            }
            if selected is not None and selected not in candidates:
                rejected_answer = selected
                selected = None
    except LLMGenerationTimeoutError as exc:
        error = str(exc)
        generation_timeout = exc
        _warning_fail_open(
            "Answer adjudication failed for %s: %s", state["session_id"], exc
        )
    except Exception as exc:
        error = str(exc)
        _warning_fail_open(
            "Answer adjudication failed for %s: %s", state["session_id"], exc
        )

    event = {
        "event_type": "answer_adjudication",
        "direct_candidate": direct_candidate,
        "proposed_answer": proposed_answer,
        "selected_answer": selected,
        "rejected_answer": rejected_answer,
        "candidate_restriction_applied": bool(getattr(
            agent_config, "restrict_adjudicator_to_candidates", False
        )),
        "evidence_count": state.get("evidence_count", 0),
        "raw_response": raw[:3000],
        "error": error,
    }
    if generation_timeout is not None:
        event.update({
            "terminal_generation_timeout": True,
            "error_type": type(generation_timeout).__name__,
            "generation_timeout_request_kind": generation_timeout.request_kind,
            "generation_timeout_attempts": generation_timeout.attempts,
        })
    _log_step_fail_open(session_logger, state["session_id"], event)
    if selected is not None:
        return selected, "adjudicated", generation_timeout
    if rejected_answer is not None:
        return direct_candidate, "anchor_adjudication_out_of_set", generation_timeout
    return direct_candidate, "anchor_adjudication_failed", generation_timeout


async def _adjudicate(
    state: AgentState,
    cfg: Dict[str, Any],
    direct_candidate: Optional[str],
    proposed_answer: Optional[str],
) -> tuple[Optional[str], str]:
    """Compatibility wrapper around adjudication with timeout diagnostics."""
    selected, method, _generation_timeout = await _adjudicate_with_timeout(
        state,
        cfg,
        direct_candidate,
        proposed_answer,
    )
    return selected, method


async def answer_gate_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Protect the direct candidate from unsupported normal-loop overrides."""
    cfg = config["configurable"]
    agent_config = cfg["agent_config"]
    final_answer = state.get("final_answer")
    if final_answer is None or not getattr(agent_config, "evidence_gated_override", False):
        return {}

    schema = state.get("answer_schema")
    proposal = normalize_schema_answer(final_answer.get("raw_value"), schema)
    direct = normalize_schema_answer(state.get("direct_candidate"), schema)
    selected, method, generation_timeout = await _adjudicate_with_timeout(
        state, cfg, direct, proposal
    )
    timeout_updates: Dict[str, Any] = {}
    if generation_timeout is not None:
        timeout_updates = {
            "llm_generation_timed_out": True,
            "llm_generation_timeout_attempts": generation_timeout.attempts,
        }
    if selected is None:
        # No valid anchor and an invalid proposal: clear it so the normal loop
        # can recover rather than resurrecting it during forced termination.
        return {
            "final_answer": None,
            "last_submitted_answer": None,
            "proposed_answer": proposal,
            "termination_reason": None,
            "messages": [HumanMessage(content=(
                "[ANSWER REJECTED] The submitted answer does not satisfy the "
                "declared answer schema. Submit a valid answer."
            ))],
        }

    selected_answer = {
        "text": selected,
        "raw_value": selected,
        "mode": "controller",
    }
    session_logger = cfg.get("logger")
    _log_step_fail_open(session_logger, state["session_id"], {
        "event_type": "answer_gate",
        "proposed_answer": proposal,
        "direct_candidate": direct,
        "selected_answer": selected,
        "method": method,
        "evidence_count": state.get("evidence_count", 0),
    })
    return {
        "messages": [HumanMessage(content=(
            f"[Controller answer gate] Agent proposal: {proposal or '<invalid>'}; "
            f"selected answer: {selected} via {method}."
        ))],
        "final_answer": selected_answer,
        "last_submitted_answer": selected_answer,
        "proposed_answer": proposal,
        "termination_reason": "completed",
        **timeout_updates,
    }


async def final_answer_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Use the reserved controller turn to produce a schema-valid final answer."""
    cfg = config["configurable"]
    agent_config = cfg.get("agent_config")
    schema = state.get("answer_schema")
    direct = normalize_schema_answer(state.get("direct_candidate"), schema)
    proposal = normalize_schema_answer(state.get("proposed_answer"), schema)
    timeout_updates: Dict[str, Any] = {}

    terminal_timeout = bool(
        state.get("finalize_on_llm_timeout", False)
        and state.get("llm_generation_timed_out", False)
    )
    if terminal_timeout:
        # The explicit generation deadline exhausted this sample's request
        # budget. Do not launch another model call: choose the strongest
        # schema-valid value already present, then use a deterministic schema
        # fallback (multiple-choice schemas always supply declared labels).
        selected = direct or proposal
        method = "anchor_generation_timeout" if direct else "proposal_generation_timeout"
        if selected is None:
            last_submitted = state.get("last_submitted_answer") or {}
            selected = normalize_schema_answer(last_submitted.get("raw_value"), schema)
            if selected is not None:
                method = "submitted_generation_timeout"
        if selected is None:
            from spatial_agent.nodes.router import _extract_partial_answer

            partial = _extract_partial_answer(
                state,
                safe_multimodal_fallback=True,
            )
            selected = normalize_schema_answer(partial, schema)
            if selected is not None:
                method = "partial_generation_timeout"
        if selected is None:
            declared = allowed_values(schema)
            selected = declared[0] if declared else "unknown"
            method = "schema_default_generation_timeout"
    else:
        if direct is None and getattr(agent_config, "direct_answer_anchor", False):
            direct, _raw, retry_timeout = await _generate_direct_candidate(
                state, cfg, event_type="direct_answer_retry"
            )
            if retry_timeout is not None:
                timeout_updates = {
                    "llm_generation_timed_out": True,
                    "llm_generation_timeout_attempts": retry_timeout.attempts,
                }
            if (
                retry_timeout is not None
                and state.get("finalize_on_llm_timeout", False)
            ):
                terminal_timeout = True
                declared = allowed_values(schema)
                selected = proposal or (declared[0] if declared else "unknown")
                method = (
                    "proposal_generation_timeout"
                    if proposal is not None
                    else "schema_default_generation_timeout"
                )
            else:
                selected, method, adjudication_timeout = (
                    await _adjudicate_with_timeout(state, cfg, direct, proposal)
                )
                if adjudication_timeout is not None:
                    timeout_updates = {
                        "llm_generation_timed_out": True,
                        "llm_generation_timeout_attempts": adjudication_timeout.attempts,
                    }
        else:
            selected, method, adjudication_timeout = await _adjudicate_with_timeout(
                state, cfg, direct, proposal
            )
            if adjudication_timeout is not None:
                timeout_updates = {
                    "llm_generation_timed_out": True,
                    "llm_generation_timeout_attempts": adjudication_timeout.attempts,
                }
    if selected is None:
        # Preserve the historical force-termination recovery as a last resort.
        from spatial_agent.nodes.router import force_terminate

        forced = await force_terminate(state, config)
        return {**forced, **timeout_updates}

    final_answer = {
        "text": selected,
        "raw_value": selected,
        "mode": "controller",
    }
    from spatial_agent.nodes.router import controller_trigger_reason

    trigger = (
        "llm_generation_timeout"
        if terminal_timeout
        else controller_trigger_reason(state)
    )
    session_logger = cfg.get("logger")
    _log_step_fail_open(session_logger, state["session_id"], {
        "event_type": "controller_final",
        "method": method,
        "trigger": trigger,
        "final_answer": final_answer,
        "steps": state.get("step_count", 0),
        "evidence_count": state.get("evidence_count", 0),
    })
    return {
        "messages": [HumanMessage(content=(
            f"[System] Controller finalized the answer via {method}: {selected}"
        ))],
        "final_answer": final_answer,
        "last_submitted_answer": final_answer,
        "termination_reason": trigger,
        **timeout_updates,
    }
