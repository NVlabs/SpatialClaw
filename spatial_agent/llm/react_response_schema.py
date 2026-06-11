"""ReAct response schema and validation.

The ReAct baseline uses the same markdown envelope as the code agent
(``**Purpose**``, ``**Reasoning**``, ``**Next Goal**``) but replaces the
``**Code**`` block with a ``**Tool Call**`` block containing a single JSON
object::

    **Purpose**: Segment the red car
    **Reasoning**: Need mask to measure distance
    **Next Goal**: Compute 3D centroid
    **Tool Call**:
    ```json
    {"tool": "tools.SAM3.segment_image_by_text",
     "args": {"image": "InputImages[0]", "prompt": "\\"red car\\""}}
    ```

The validator parses this envelope, delegates to the translator to emit a
code string, and returns the same :class:`LLMResponse` shape as the code
agent so ``llm_step_node`` stays agnostic to the executor variant.
"""

import json
import re
from typing import Optional

from spatial_agent.llm.react_translator import translate
from spatial_agent.llm.response_schema import LLMResponse, LLMResponseValidator


REQUIRED_KEYS = {"purpose", "reasoning", "next_goal", "tool_call"}


class ReactResponseValidator:
    """Extract and validate a ReAct-style structured response.

    Parses the markdown-with-JSON format above, then converts the parsed
    tool call into a Python code string via
    :func:`spatial_agent.llm.react_translator.translate`.
    """

    @staticmethod
    def _parse_markdown_schema(text: str) -> Optional[dict]:
        """Return a dict with purpose / reasoning / next_goal / tool_call, or ``None``."""
        result: dict = {}
        header_map = {
            "purpose": r"\*\*Purpose\*\*",
            "reasoning": r"\*\*Reasoning\*\*",
            "next_goal": r"\*\*Next[ _]Goal\*\*",
            "tool_call": r"\*\*Tool[ _]Call\*\*",
        }
        section_end = (
            r"(?=\*\*(?:Purpose|Reasoning|Next[ _]Goal|Tool[ _]Call)\*\*|\Z)"
        )
        for key, header_re in header_map.items():
            pattern = header_re + r"[:\s]*(.*?)" + section_end
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m:
                value = m.group(1).strip()
                if key == "tool_call":
                    # Extract JSON from a fenced block — prefer ```json but
                    # accept bare ``` with JSON inside.
                    cm = re.search(
                        r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL
                    )
                    if cm:
                        value = cm.group(1).strip()
                    else:
                        # Truncated fence (LLM ran out of tokens): take
                        # everything after the opening ```.
                        cm = re.search(r"```(?:json)?\s*(.*)", value, re.DOTALL)
                        if cm:
                            value = cm.group(1).strip()
                result[key] = value
        if REQUIRED_KEYS.issubset(set(result.keys())):
            return result
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def validate(raw_text: str, step: int) -> LLMResponse:
        """Parse ``raw_text`` as a ReAct response, return an ``LLMResponse``.

        The ``code`` field of the returned dataclass holds the synthesized
        Python code that ``execute_node`` will run. The tool-call JSON is
        preserved verbatim in ``raw_content`` for logging.

        Raises:
            ValueError on any parse/schema/translation failure. The message
            is surfaced to the LLM as feedback.
        """
        # Reuse the thinking-preamble stripper from the code-agent validator.
        clean = LLMResponseValidator._strip_thinking(raw_text)
        search_text = clean if clean else raw_text

        data = ReactResponseValidator._parse_markdown_schema(search_text)
        if data is None:
            raise ValueError(
                "Could not parse response. Use the markdown format with "
                "**Purpose**, **Reasoning**, **Next Goal**, and **Tool Call** "
                "sections, where Tool Call is a fenced ```json block."
            )

        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"Missing required sections: {missing}. "
                f"Every response must include: {REQUIRED_KEYS}"
            )

        for key in ("purpose", "reasoning", "next_goal"):
            val = data[key]
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f'Section "{key}" must be a non-empty string.')

        tool_call_raw = data["tool_call"]
        if not tool_call_raw.strip():
            raise ValueError('Section "Tool Call" must contain a JSON object.')

        try:
            tool_call = json.loads(tool_call_raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Tool Call JSON is invalid: {exc.msg} at line {exc.lineno} "
                f"col {exc.colno}. Emit exactly one JSON object of the form "
                f'{{"tool": "...", "args": {{...}}}}.'
            )

        if isinstance(tool_call, list):
            raise ValueError(
                "Tool Call must be a single JSON object, not a list. "
                "ReAct mode allows only one tool call per step."
            )

        # translate() raises ValueError with a specific diagnostic on
        # unknown tools or disallowed arg expressions.
        code = translate(tool_call, step=step)

        return LLMResponse(
            purpose=data["purpose"].strip(),
            reasoning=data["reasoning"].strip(),
            next_goal=data["next_goal"].strip(),
            code=code,
            raw_content=raw_text,
        )
