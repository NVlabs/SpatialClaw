"""FeedbackCollector: format execution results into text feedback for the LLM."""

import re
from typing import Any, Dict, List, Optional


def _compact_error(error: str, max_len: int = 600) -> str:
    """Compact a Jupyter traceback into a concise error message.

    Strips ANSI codes, extracts the exception type + message and the
    offending source line from the traceback.  Keeps the result under
    ``max_len`` chars.  Code-level context (snippets, pattern search) is
    handled separately by ``_extract_error_snippet`` in the condensed
    AIMessage — this function only summarizes the error itself.
    """
    # Strip ANSI escape sequences
    clean = re.sub(r"\x1b\[[0-9;]*m", "", error)

    # Find the final exception line and the first offending source line (user code)
    lines = [l.strip() for l in clean.splitlines() if l.strip()]
    exc_line = ""
    source_line = ""
    for line in reversed(lines):
        if not exc_line and re.match(r"^[A-Z]\w*(Error|Exception|Warning)", line):
            exc_line = line
        # Jupyter marks the offending line with --->  or ----> prefix
        if not source_line and re.match(r"^-*>\s*\d+\s", line):
            source_line = re.sub(r"^-*>\s*", "", line).strip()
    if not exc_line:
        # Fallback: just truncate the raw error
        return clean[:max_len]

    result = exc_line
    if source_line:
        result = f"at: {source_line}\n{result}"

    if len(result) > max_len:
        result = result[:max_len - 3] + "..."
    return result


def _extract_error_line_number(error: str) -> Optional[int]:
    """Extract the 1-based line number from a traceback.

    Recognises:
      * Jupyter's ``----> N source_code`` marker.
      * Python ``SyntaxError`` style ``(<unknown>, line 26)`` (used by the
        SecuritySandbox AST check, which has no Jupyter traceback).
    Returns ``None`` if no recognised pattern is found.
    """
    clean = re.sub(r"\x1b\[[0-9;]*m", "", error)
    for line in clean.splitlines():
        m = re.match(r"^-*>\s*(\d+)\s", line.strip())
        if m:
            return int(m.group(1))
    m = re.search(r"\bline\s+(\d+)\b", clean)
    if m:
        return int(m.group(1))
    return None


def _search_code_for_pattern(code_lines: List[str], error: str) -> Optional[int]:
    """Try to find the offending line by searching the code for the error pattern.

    Handles security violations (``Forbidden builtin call: 'eval()'``,
    ``Forbidden module: 'os'``) by searching for the forbidden token.
    Returns a 1-based line number or ``None``.
    """
    clean = re.sub(r"\x1b\[[0-9;]*m", "", error)
    # Security: "Forbidden builtin call: 'eval()'"
    m = re.search(r"Forbidden builtin call: '(\w+)\(\)'", clean)
    if m:
        token = m.group(1) + "("
        for i, line in enumerate(code_lines):
            if token in line:
                return i + 1
        return None
    # Security: "Forbidden module: 'os'"
    m = re.search(r"Forbidden module: '(\w+)'", clean)
    if m:
        mod = m.group(1)
        for i, line in enumerate(code_lines):
            if f"import {mod}" in line or f"{mod}." in line:
                return i + 1
        return None
    return None


def _extract_error_snippet(
    code: str, error: str, context_lines: int = 2, max_len: int = 400,
) -> str:
    """Return a short code snippet around the error line.

    If the error line can be determined from the Jupyter traceback, returns
    ``context_lines`` lines above and below it with a ``# <-- ERROR`` marker.
    When no traceback is available (security violations, timeouts), tries to
    find the offending line by pattern-matching the error against the code.
    Falls back to the first 3 + last 3 lines of the code.
    """
    lines = code.splitlines()
    if not lines:
        return ""

    error_line = _extract_error_line_number(error)
    if error_line is None:
        error_line = _search_code_for_pattern(lines, error)

    # Truncate individual lines so the # <-- ERROR marker is never cut off
    max_line = 120

    if error_line is not None and 1 <= error_line <= len(lines):
        idx = error_line - 1  # 0-based
        start = max(0, idx - context_lines)
        end = min(len(lines), idx + context_lines + 1)
        parts = []
        for i in range(start, end):
            text = lines[i]
            marker = "  # <-- ERROR" if i == idx else ""
            if len(text) > max_line:
                text = text[:max_line] + "..."
            parts.append(f"{i + 1}: {text}{marker}")
        snippet = "\n".join(parts)
    else:
        # Unknown error line — show head + tail
        def _fmt(i: int) -> str:
            text = lines[i]
            if len(text) > max_line:
                text = text[:max_line] + "..."
            return f"{i + 1}: {text}"

        if len(lines) <= 7:
            snippet = "\n".join(_fmt(i) for i in range(len(lines)))
        else:
            head = [_fmt(i) for i in range(3)]
            tail = [_fmt(i) for i in range(len(lines) - 3, len(lines))]
            snippet = "\n".join(head) + "\n...\n" + "\n".join(tail)

    if len(snippet) > max_len:
        snippet = snippet[:max_len - 3] + "..."
    return snippet


def _truncate_code_at_error(
    code: str, error: str, max_len: int = 2000,
) -> Optional[str]:
    """Keep code verbatim up to the error line, cut the rest.

    Returns the code lines *before* the error line plus commented error
    marker lines.  If the error line cannot be determined, returns
    ``None`` so the caller can fall back to the snippet-based condensation.

    When the kept code exceeds *max_len*, lines are trimmed from the
    **top** (keeping the error-adjacent lines that are most relevant)
    and a ``# ... (earlier lines omitted)`` header is prepended.
    """
    lines = code.splitlines()
    if not lines:
        return None

    error_line = _extract_error_line_number(error)
    if error_line is None:
        error_line = _search_code_for_pattern(lines, error)
    if error_line is None:
        return None

    # Keep lines before the error line (0-indexed: 0 .. error_line-2),
    # then include the error line itself with a marker.
    kept = lines[: error_line]  # includes the error line
    compact = _compact_error(error)

    # Build marker: the error line is already in `kept`, so just append
    # the exception info as comments.
    # _compact_error returns "at: N source\nExceptionType: msg" or just
    # "ExceptionType: msg".  Extract just the exception line(s).
    compact_lines = compact.splitlines()
    # Skip the "at: ..." line (redundant — the error line is in `kept`)
    exc_lines = [l for l in compact_lines if not l.startswith("at:")]
    if not exc_lines:
        exc_lines = compact_lines  # fallback: keep everything
    marker = "\n".join(f"# {l}" for l in exc_lines)

    # Mark the error line in kept code
    if kept:
        last = kept[-1]
        if len(last) > 100:
            last = last[:100] + "..."
        kept[-1] = f"{last}  # <-- ERROR"

    result = "\n".join(kept) + "\n" + marker if kept else marker
    if len(result) > max_len:
        # Trim from the top, keeping lines closest to the error
        trimmed = kept[:]
        header = "# ... (earlier lines omitted)\n"
        while trimmed and len("\n".join(trimmed) + "\n" + marker) + len(header) > max_len:
            trimmed.pop(0)
        result = header + "\n".join(trimmed) + "\n" + marker if trimmed else header + marker
    return result


class FeedbackCollector:
    """Formats execution results into structured text feedback."""

    @staticmethod
    def build_feedback(
        step_result: Dict[str, Any],
        var_summaries: Dict[str, str],
        large_var_warnings: List[str],
        final_answer: Optional[Dict] = None,
        condensed: bool = False,
        has_survivors: bool = False,
    ) -> str:
        """Build the feedback text that becomes the next HumanMessage.

        When *condensed* is True AND the step errored:
        - **has_survivors=True** (partial error): variables created before
          the error line were kept.  Shows survivor summaries and images.
        - **has_survivors=False** (full rollback): all new variables were
          deleted.  Strips variables, images, and tool counts.
        In both cases the error itself is in the condensed AIMessage.
        """
        parts = []
        step_idx = step_result.get("step_index", "?")
        parts.append(f"=== Step {step_idx} Execution Feedback ===")

        # --- Execution status ---
        error = step_result.get("error")
        exec_time = step_result.get("execution_time_sec", 0)
        full_rollback = error and condensed and not has_survivors
        partial_error = error and condensed and has_survivors
        if full_rollback:
            parts.append(
                "\n[ERROR] (see code above)\n"
                "All variables from this step have been rolled back."
            )
        elif partial_error:
            error_line = _extract_error_line_number(error) if error else None
            if error_line:
                parts.append(
                    f"\n[PARTIAL ERROR] Code errored at line {error_line}. "
                    f"Variables created before the error were kept."
                )
            else:
                parts.append(
                    "\n[PARTIAL ERROR] Code errored. "
                    "Variables created before the error were kept."
                )
        elif error:
            parts.append(f"\n[ERROR] {_compact_error(error)}")
        else:
            parts.append(f"\n[SUCCESS] Code executed in {exec_time:.2f}s.")

        # --- Stdout (kept on error — contains VLM answers & debug prints) ---
        stdout = step_result.get("stdout", "").strip()
        if stdout:
            max_stdout = 4000
            if len(stdout) > max_stdout:
                tail_size = max_stdout * 3 // 4
                head_size = max_stdout - tail_size
                stdout = (
                    stdout[:head_size]
                    + f"\n... ({len(stdout)} chars, middle truncated) ...\n"
                    + stdout[-tail_size:]
                )
            parts.append(f"\n[Output]\n{stdout}")

        # --- Stderr (warnings) ---
        stderr = step_result.get("stderr", "").strip()
        if stderr and not error:
            stderr = re.sub(r"\x1b\[[0-9;]*m", "", stderr)
            stderr = "\n".join(
                l for l in stderr.splitlines()
                if not re.search(
                    r"ServeReplica|frame loading|removed session|"
                    r"Started.*router|CALL \w+ OK|live sessions|"
                    r"Failed to get queue length|get_draining",
                    l,
                )
            ).strip()
            if stderr:
                if len(stderr) > 500:
                    stderr = stderr[:500] + "\n... (truncated)"
                parts.append(f"\n[Warnings]\n{stderr}")

        # --- VLM queries (kept on error — shows what VLM perceived) ---
        vlm_queries = step_result.get("vlm_queries", [])
        if vlm_queries:
            parts.append(f"\n[VLM Queries] ({len(vlm_queries)} queries)")
            for q in vlm_queries:
                qtype = q.get("query_type") or ""
                tag = f" [{qtype}]" if qtype else ""
                parts.append(f"  Q{tag}: {q.get('question', '?')}")
                parts.append(f"  A{tag}: {q.get('answer', '(no answer)')}")
                parts.append("")

        # --- Variable changes ---
        if var_summaries and not full_rollback:
            if partial_error:
                parts.append("\n[Variables] Kept from before error:")
            else:
                parts.append("\n[Variables] New/changed:")
            for summary in var_summaries.values():
                parts.append(f"  {summary}")

        # --- Large variable warnings (skip when full rollback) ---
        if not full_rollback:
            for warning in large_var_warnings:
                parts.append(f"\n[WARNING] {warning}")

        # --- Final answer confirmation ---
        if final_answer:
            parts.append(
                f"\n[ANSWER SUBMITTED] {final_answer.get('text', '?')}"
            )

        # --- Show images (skip when full rollback, keep on partial error) ---
        if not full_rollback:
            show_images = step_result.get("show_images", [])
            if show_images:
                total_imgs = sum(len(e.get("images", [])) for e in show_images)
                labels = [e.get("label", "") for e in show_images if e.get("label")]
                parts.append(
                    f"\n[Inline Images] {total_imgs} image(s) attached below."
                )
                if labels:
                    parts.append(f"  Labels: {', '.join(labels)}")
            show_trunc = step_result.get("show_truncation_warning", "")
            if show_trunc:
                parts.append(f"\n[WARNING] {show_trunc}")

        # --- Tool call count (skip when full rollback) ---
        if not full_rollback:
            tc = step_result.get("tool_call_count", 0)
            if tc > 0:
                parts.append(f"\n[Tool Calls] {tc} tool/VLM call(s) this step.")

        return "\n".join(parts)

    @staticmethod
    def format_checklist(checklist: list) -> str:
        """Render the verification checklist as readable text for the LLM.

        Returns empty string if checklist is empty.
        """
        if not checklist:
            return ""

        lines = ["\n[Verification Checklist]"]
        high_pending = 0
        for item in checklist:
            status = item.get("status", "PENDING")
            priority = item.get("priority", "?")
            desc = item.get("description", "")
            prefix = f"  {status:<8} [{priority}] {desc}"
            lines.append(prefix)
            note = item.get("resolution_note")
            if note:
                lines.append(f"           Note: {note}")
            if priority == "HIGH" and status == "PENDING":
                high_pending += 1

        if high_pending > 0:
            lines.append(
                f"\n  WARNING: {high_pending} HIGH-priority item(s) still PENDING. "
                "You MUST verify these before calling ReturnAnswer."
            )
        return "\n".join(lines)

    @staticmethod
    def format_checklist_compact(checklist: list) -> str:
        """One-liner checklist summary for normal (non-answer) feedback steps.

        Returns empty string if checklist is empty.
        """
        if not checklist:
            return ""

        total = len(checklist)
        verified = sum(1 for item in checklist if item.get("status") == "VERIFIED")
        flagged = sum(1 for item in checklist if item.get("status") == "FLAGGED")
        high_pending = sum(
            1 for item in checklist
            if item.get("priority") == "HIGH" and item.get("status") == "PENDING"
        )

        parts = [f"{verified}/{total} verified"]
        if flagged:
            parts.append(f"{flagged} flagged")
        if high_pending:
            parts.append(f"{high_pending} HIGH pending")

        return f"[Checklist: {', '.join(parts)}]"
