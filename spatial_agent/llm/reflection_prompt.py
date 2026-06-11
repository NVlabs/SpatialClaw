"""Reflection prompt appended as a user message to preserve KV cache prefix.

The original system prompt and full conversation history remain unchanged so
vLLM automatic prefix caching can reuse the KV cache from the main agent call.
Only this user message is appended at the end.

Prompt sections can be excluded or overridden via the
``prompt_section_ablations`` config field.  Section names use the ``reflection_``
prefix (e.g., ``reflection_what_to_check``).
"""

_REFLECTION_BASE_SINGLE = """\
[Self-Reflection Check] Review the most recent reasoning step and execution \
result above. Decide whether the agent is on the right track or making a mistake."""

_REFLECTION_BASE_MULTI = """\
[Self-Reflection Check] Review the last {n_steps} reasoning steps and their \
execution results above (since the previous reflection). Decide whether the \
agent is on the right track or making a mistake."""

_REFLECTION_WHAT_TO_CHECK = """

## What to check
- **Logical errors**: Is the reasoning consistent with the execution output?
- **Geometric mistakes**: Wrong coordinate systems, incorrect distance/angle formulas, \
confused reference frames, mismatched per-frame data.
- **Tool misuse**: Calling tools with wrong arguments, ignoring tool output, \
misinterpreting returned values.
- **Goal drift**: Is the agent still working toward answering the original question?
- **Dead loops**: Repeating the same action that already failed.
- **Unsupported spatial claims**: The agent made a quantitative or spatial conclusion \
(distance, direction, motion, size) without sufficient evidence. Is the claim backed \
by computation, or is it a visual guess? If tools were planned, they should have been used.
- **Unverified tool inputs**: The agent used tool outputs (3D positions, distances) without \
verifying that inputs were correct (e.g., mask on the right object, correct coordinates). \
Wrong inputs produce confidently wrong outputs.
- **Tool/question shape mismatch**: The chosen tool's evidence shape does not match what the question \
asks. Examples to flag: `vlm.locate` called with a non-coordinate question (scene description, action, \
identity); `vlm.ask_with_thinking` called for an answer that a specialized tool already produced \
clearly (a coordinate via `vlm.locate`, a mask via SAM3, a metric distance via Reconstruct/Geometry, \
on-image text via OCR-style tools).
- **Stuck on "Cannot determine"**: A `vlm.ask_with_thinking` call returned "Cannot determine from \
the images." and the agent re-issued the same call with the same frame selection and same phrasing, \
or proceeded as if it had received an answer.
- **Priming in `vlm.ask_with_thinking` queries**: The `question` argument embeds the agent's prior \
conclusion or expected answer (e.g., "the car is moving left, right?"). Phrasing should let the \
answer be formed from the images, not lead the answer.
- **Manual coordinate guessing**: If the agent estimates coordinates from `show()` output \
(e.g., "roughly at 60% width, 40% height"), this is WRONG. Flag it — the agent should use \
`vlm.locate` to get precise 0-1000 normalized center points or bounding boxes instead.

## What NOT to flag
- Minor style issues or suboptimal but correct approaches.
- The agent using `vlm.locate` for coordinate grounding (its intended use).
- Multiple `vlm.ask_with_thinking` calls with different frames or phrasings — these are valid \
when the agent is cross-checking or recovering from an ambiguous answer.
- Early exploration steps where the agent is gathering information via `show()`."""

_OUTPUT_FORMAT_NO_CHECKLIST = """
## Output format
Respond with ONLY a JSON object (no markdown fences, no extra text):

If everything looks fine:
{"status": "ok"}

If you spot a concern:
{"status": "concern", "explanation": "What is wrong (1 sentence).", "suggestion": "What to do next (1 sentence)."}
"""

_CHECKLIST_PREAMBLE = """
## Verification Checklist

The agent is tracking these verification items:

"""

_CHECKLIST_OPS_AND_FORMAT = """

You may update checklist items using `checklist_ops`. Supported operations:
- **verify**: Mark an item as VERIFIED (you confirmed it's correct based on evidence in the conversation).
- **flag**: Mark an item as FLAGGED (you found a problem).
- **add**: Add a new verification item you noticed is needed.

IMPORTANT: Verify items proactively based on evidence already in the conversation. Do NOT leave items PENDING when evidence is available.

## Output format
Respond with ONLY a JSON object (no markdown fences, no extra text):

If everything looks fine (no checklist changes):
{"status": "ok"}

If everything looks fine but you can verify/flag checklist items:
{"status": "ok", "checklist_ops": [{"op": "verify", "item_id": "chk_001", "note": "Confirmed via show()"}]}

If you spot a concern:
{"status": "concern", "explanation": "What is wrong (1 sentence).", "suggestion": "What to do next (1 sentence).", "checklist_ops": [{"op": "flag", "item_id": "chk_002", "note": "Mask on wrong object"}]}

To add a new checklist item:
{"status": "ok", "checklist_ops": [{"op": "add", "description": "New thing to verify", "priority": "MEDIUM"}]}
"""


# Valid section names for the reflection prompt.
REFLECTION_PROMPT_SECTIONS = {
    "reflection_base",
    "reflection_what_to_check",
    "reflection_checklist",
    "reflection_output_format",
}

def _get_reflection_base(steps_since_last: int) -> str:
    """Return the appropriate opening paragraph based on how many steps to review."""
    if steps_since_last <= 1:
        return _REFLECTION_BASE_SINGLE
    return _REFLECTION_BASE_MULTI.format(n_steps=steps_since_last)


def build_reflection_user_message(
    checklist_text: str = "",
    steps_since_last: int = 1,
) -> str:
    """Build the reflection user message, optionally including checklist state.

    Args:
        checklist_text: Formatted checklist text (from FeedbackCollector.format_checklist).
            If empty, the prompt uses the simpler format without checklist ops.
        steps_since_last: Number of agent steps since the last reflection.
            When > 1 the prompt instructs the reviewer to check all recent steps.

    Uses string concatenation instead of .format() to avoid KeyError when
    checklist descriptions contain curly braces like ``{x,y}``.
    """
    from spatial_agent.config import get_config
    from spatial_agent.llm.prompt_common import resolve_section, warn_unknown_sections

    config = get_config()
    ablations = config.prompt_section_ablations
    warn_unknown_sections(ablations, REFLECTION_PROMPT_SECTIONS, "reflection prompt")
    r = resolve_section

    base = r("reflection_base", _get_reflection_base(steps_since_last), ablations)
    what_to_check = r("reflection_what_to_check", _REFLECTION_WHAT_TO_CHECK, ablations)

    if checklist_text:
        checklist_section = r(
            "reflection_checklist",
            _CHECKLIST_PREAMBLE + checklist_text + _CHECKLIST_OPS_AND_FORMAT,
            ablations,
        )
        output_format = ""  # included in checklist section
    else:
        checklist_section = ""
        output_format = r("reflection_output_format", _OUTPUT_FORMAT_NO_CHECKLIST, ablations)

    return base + what_to_check + checklist_section + output_format
