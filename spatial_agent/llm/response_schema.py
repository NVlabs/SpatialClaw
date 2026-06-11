"""LLM response schema and validation.

Every LLM response must contain four sections:
  - **Purpose**: what this code step accomplishes
  - **Reasoning**: brief chain of thought
  - **Next Goal**: what to do after this step
  - **Code**: Python code in a fenced block

Format is Markdown with bold headers. Code appears verbatim in fenced blocks.
"""

import re
from dataclasses import dataclass
from typing import Optional


REQUIRED_KEYS = {"purpose", "reasoning", "next_goal", "code"}


@dataclass
class LLMResponse:
    """Validated and parsed LLM response."""

    purpose: str
    reasoning: str
    next_goal: str
    code: str
    raw_content: str  # original LLM output, kept for logging


class LLMResponseValidator:
    """Extracts and validates structured responses from raw LLM text.

    Parses Markdown bold-header format: **Purpose**, **Reasoning**,
    **Next Goal**, **Code** with code in fenced blocks (verbatim).
    """

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip thinking preambles from Qwen3.5 / other thinking models.

        Handles:
        - ``<think>...</think>`` full blocks
        - Orphan ``</think>`` (when ``<think>`` was before our window): strip
          everything up to and including ``</think>``
        """
        # Strip complete <think>...</think> blocks
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Strip orphan </think> (thinking ran from before our view): drop everything
        # up to and including the first </think>
        if "</think>" in stripped:
            stripped = stripped[stripped.index("</think>") + len("</think>"):].strip()
        return stripped if stripped else text

    @staticmethod
    def _parse_markdown_schema(text: str) -> Optional[dict]:
        """Parse **Purpose**: / **Reasoning**: / **Next Goal**: / **Code**: markdown format.

        Code is extracted from fenced code blocks (```python ... ```), so it
        appears verbatim — no escaping issues.
        """
        result: dict = {}
        header_map = {
            "purpose": r"\*\*Purpose\*\*",
            "reasoning": r"\*\*Reasoning\*\*",
            "next_goal": r"\*\*Next[ _]Goal\*\*",
            "code": r"\*\*Code\*\*",
        }
        # A section runs until the next ** header or end of string
        section_end = r"(?=\*\*(?:Purpose|Reasoning|Next[ _]Goal|Code)\*\*|\Z)"
        for key, header_re in header_map.items():
            pattern = header_re + r"[:\s]*(.*?)" + section_end
            m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if m:
                value = m.group(1).strip()
                if key == "code":
                    # Extract from fenced code block if present
                    cm = re.search(r"```(?:python)?\s*(.*?)\s*```", value, re.DOTALL)
                    if cm:
                        value = cm.group(1).strip()
                    else:
                        # Handle truncated code block (model ran out of tokens
                        # before closing ```).  Extract everything after the
                        # opening ``` marker.
                        cm = re.search(r"```(?:python)?\s*(.*)", value, re.DOTALL)
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
    def validate(raw_text: str) -> LLMResponse:
        """Parse *raw_text*, validate required keys, return ``LLMResponse``.

        Raises ``ValueError`` with a descriptive message on any failure so the
        caller can feed the error back into the LLM for self-correction.
        """
        # Strip thinking preamble first
        clean = LLMResponseValidator._strip_thinking(raw_text)
        search_text = clean if clean else raw_text

        data = LLMResponseValidator._parse_markdown_schema(search_text)

        if data is None:
            raise ValueError(
                "Could not parse response. Use the markdown format with "
                "**Purpose**, **Reasoning**, **Next Goal**, and **Code** sections."
            )

        # Check required keys
        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(
                f"Missing required sections: {missing}. "
                f"Every response must include: {REQUIRED_KEYS}"
            )

        # Check value types
        for key in REQUIRED_KEYS:
            val = data[key]
            if not isinstance(val, str):
                raise ValueError(
                    f'Section "{key}" must be a string, got {type(val).__name__}.'
                )
            if not val.strip():
                raise ValueError(f'Section "{key}" must be non-empty.')

        return LLMResponse(
            purpose=data["purpose"].strip(),
            reasoning=data["reasoning"].strip(),
            next_goal=data["next_goal"].strip(),
            code=data["code"],
            raw_content=raw_text,
        )
