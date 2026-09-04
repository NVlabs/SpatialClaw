"""Pure helpers for schema-aware direct answers and controller arbitration."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional


DIRECT_ANSWER_SYSTEM_PROMPT = """\
You are a careful visual reasoning assistant. Inspect the supplied images and
answer the question directly. Do not propose tools or code. Base the answer on
the visible evidence. Put only the final answer inside \\boxed{}.
"""


ADJUDICATION_SYSTEM_PROMPT = """\
You are the final answer controller for a visual reasoning task. Reconsider the
original images and question together with the compact observations supplied by
the controller. Select the best final answer. Do not propose tools or code. Put
only the final answer inside \\boxed{}.
"""


def allowed_values(answer_schema: Optional[Dict[str, Any]]) -> list[str]:
    """Return normalized declared values for a multiple-choice schema."""
    if not answer_schema or answer_schema.get("kind") != "multiple_choice":
        return []
    values = answer_schema.get("allowed_values") or []
    return [str(value).strip().upper() for value in values if str(value).strip()]


def normalize_schema_answer(
    value: Any,
    answer_schema: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Extract and validate an answer without using ground truth.

    Multiple-choice values are accepted only when they match a declared label.
    Open answers prefer an explicit boxed/ReturnAnswer value and otherwise keep
    the non-empty text verbatim.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    candidates: list[str] = []
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text, flags=re.DOTALL)
    candidates.extend(reversed([item.strip() for item in boxed]))

    returned = re.findall(
        r"\bReturnAnswer\(\s*(?:[rubfRUBF]*)(?:['\"]([^'\"]+)['\"]|([A-Za-z0-9_.+-]+))\s*\)",
        text,
    )
    candidates.extend(
        (quoted or bare).strip() for quoted, bare in reversed(returned)
    )

    explicit = re.findall(
        r"(?:final\s+answer|answer|choice)\s*(?:is|:|=)\s*([A-Za-z0-9_.+-]+)",
        text,
        flags=re.IGNORECASE,
    )
    candidates.extend(reversed([item.strip() for item in explicit]))
    candidates.append(text)

    allowed = allowed_values(answer_schema)
    if allowed:
        allowed_set = set(allowed)
        for candidate in candidates:
            normalized = candidate.strip().strip("`*.$ ").upper()
            if normalized in allowed_set:
                return normalized
        return None

    for candidate in candidates:
        candidate = candidate.strip()
        if candidate:
            return candidate
    return None


def schema_instruction(answer_schema: Optional[Dict[str, Any]]) -> str:
    allowed = allowed_values(answer_schema)
    if allowed:
        return (
            "Your final answer must be exactly one of: "
            + ", ".join(allowed)
            + ". Return it inside \\boxed{}."
        )
    return "Return a concise final answer inside \\boxed{}."


def build_direct_question(
    instruction: str,
    answer_schema: Optional[Dict[str, Any]],
) -> str:
    return f"{instruction}\n\n{schema_instruction(answer_schema)}"


def build_adjudication_question(
    instruction: str,
    answer_schema: Optional[Dict[str, Any]],
    direct_candidate: Optional[str],
    proposed_answer: Optional[str],
    evidence_summaries: Iterable[str],
    *,
    restrict_to_candidates: bool = False,
) -> str:
    evidence = [str(item).strip()[:1200] for item in evidence_summaries if str(item).strip()]
    evidence_text = "\n".join(f"- {item}" for item in evidence[-2:]) or "- No reliable new observation."
    candidate_instruction = ""
    if restrict_to_candidates:
        candidates = list(dict.fromkeys(
            str(value).strip()
            for value in (direct_candidate, proposed_answer)
            if value is not None and str(value).strip()
        ))
        if candidates:
            candidate_instruction = (
                " You must select exactly one of the controller candidates: "
                + ", ".join(candidates)
                + ". Do not introduce a third answer."
            )
    return (
        f"Original task:\n{instruction}\n\n"
        f"Direct visual candidate: {direct_candidate or '<none>'}\n"
        f"Agent proposal: {proposed_answer or '<none>'}\n\n"
        f"Controller-observed evidence:\n{evidence_text}\n\n"
        f"Re-evaluate the images and choose the best answer. "
        f"{schema_instruction(answer_schema)}{candidate_instruction}"
    )
