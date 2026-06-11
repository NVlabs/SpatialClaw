"""Generate per-session interactive HTML reports."""

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image


def _img_to_base64(img: Image.Image, max_size: int = 400) -> str:
    """Encode PIL image as inline base64 for HTML."""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _img_path_to_base64(path: str, max_size: int = 400) -> str:
    """Load image from path and encode."""
    try:
        img = Image.open(path).convert("RGB")
        return _img_to_base64(img, max_size)
    except Exception:
        return ""


def generate_session_report(
    session_dir: str,
    session_id: str,
    instruction: str,
    input_images: List[Image.Image],
    ground_truth: Optional[str] = None,
    final_answer: Optional[Dict] = None,
    termination_reason: Optional[str] = None,
    result_score: Optional[float] = None,
) -> str:
    """Generate an interactive HTML report for one session.

    Reads trace.jsonl from the session directory and renders all steps.
    Returns the path to the generated HTML file.
    """
    # Read trace events
    trace_path = os.path.join(session_dir, "trace.jsonl")
    events = []
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # Group events by step
    steps: Dict[int, List[Dict]] = {}
    for ev in events:
        si = ev.get("step_index", -1)
        if si not in steps:
            steps[si] = []
        steps[si].append(ev)

    # Extract plan text and checklist from events
    plan_text = None
    plan_checklist = []
    for ev in events:
        if ev.get("event_type") == "plan" and ev.get("plan_text"):
            plan_text = ev["plan_text"]
            plan_checklist = ev.get("checklist", [])
            break

    # Read system prompt from the init trace event
    system_prompt = None
    for ev in events:
        if ev.get("event_type") == "init" and ev.get("system_prompt"):
            system_prompt = ev["system_prompt"]
            break

    # Build HTML
    html_parts = [_html_header(session_id)]

    # System prompt section (collapsed)
    if system_prompt:
        # Render as markdown using marked.js (client-side)
        import json as _json
        escaped_json = _json.dumps(system_prompt)  # JS-safe string
        prompt_id = "system-prompt-md"
        html_parts.append(
            f'<div class="card"><details><summary><strong>System Prompt</strong></summary>'
            f'<div id="{prompt_id}" class="markdown-body"></div>'
            f'<script>document.getElementById("{prompt_id}").innerHTML = marked.parse({escaped_json});</script>'
            f'</details></div>'
        )

    # Input section
    html_parts.append(_input_section(instruction, input_images, ground_truth))

    # Plan section
    if plan_text:
        html_parts.append(_plan_section(plan_text, plan_checklist))

    # Step cards
    for step_idx in sorted(steps.keys()):
        if step_idx < 0:
            continue
        step_events = steps[step_idx]
        html_parts.append(_step_card(step_idx, step_events, session_dir))

    # Footer / verdict
    html_parts.append(_verdict_section(final_answer, ground_truth, termination_reason, result_score))
    html_parts.append(_html_footer())

    # Write
    out_path = os.path.join(session_dir, "session_report.html")
    with open(out_path, "w") as f:
        f.write("\n".join(html_parts))

    return out_path


# ---------------------------------------------------------------------------
# HTML sections
# ---------------------------------------------------------------------------

def _html_header(session_id: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Session {session_id}</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
.markdown-body {{ font-size: 14px; line-height: 1.6; }}
.markdown-body h1, .markdown-body h2, .markdown-body h3 {{ border-bottom: 1px solid #e1e4e8; padding-bottom: 4px; margin-top: 16px; margin-bottom: 8px; }}
.markdown-body code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 13px; }}
.markdown-body pre {{ background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 4px; padding: 12px; overflow-x: auto; }}
.markdown-body pre code {{ background: none; padding: 0; }}
.markdown-body table {{ border-collapse: collapse; width: 100%; }}
.markdown-body table th, .markdown-body table td {{ border: 1px solid #e1e4e8; padding: 6px 12px; }}
.markdown-body ul, .markdown-body ol {{ padding-left: 24px; }}
.markdown-body blockquote {{ border-left: 4px solid #dfe2e5; padding: 0 12px; color: #6a737d; margin: 8px 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; background: #f6f8fa; margin: 0; padding: 20px; color: #24292e; }}
.container {{ max-width: 1000px; margin: 0 auto; }}
.card {{ background: white; border: 1px solid #e1e4e8; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
.card h3 {{ margin-top: 0; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
.badge-success {{ background: #dcffe4; color: #22863a; }}
.badge-error {{ background: #ffeef0; color: #cb2431; }}
.badge-info {{ background: #dbedff; color: #0366d6; }}
.code-block {{ background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 4px; padding: 12px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 13px; overflow-x: auto; white-space: pre-wrap; }}
.output-block {{ background: #1b1f23; color: #e1e4e8; border-radius: 4px; padding: 12px; font-family: monospace; font-size: 13px; overflow-x: auto; white-space: pre-wrap; }}
.vlm-query {{ background: #fff8e1; border-left: 4px solid #f9a825; padding: 8px 12px; margin: 8px 0; border-radius: 0 4px 4px 0; }}
.img-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }}
.img-row img {{ max-height: 150px; border-radius: 4px; border: 1px solid #e1e4e8; cursor: pointer; }}
details {{ margin: 8px 0; }}
summary {{ cursor: pointer; color: #0366d6; }}
.condense-block {{ background: #fff3e0; border: 1px solid #ffe0b2; border-radius: 4px; padding: 12px; margin: 8px 0; }}
.condense-block summary {{ color: #e65100; }}
.condense-block .code-block {{ background: #fff8e1; border-color: #ffe0b2; }}
.verdict {{ border: 2px solid; border-radius: 8px; padding: 16px; margin-top: 20px; text-align: center; }}
.verdict-correct {{ border-color: #22863a; background: #dcffe4; }}
.verdict-incorrect {{ border-color: #cb2431; background: #ffeef0; }}
.verdict-unknown {{ border-color: #959da5; background: #f1f1f1; }}
.verdict-partial {{ border-color: #b08800; background: #fff8e1; }}
.var-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.var-table td, .var-table th {{ padding: 4px 8px; border-bottom: 1px solid #e1e4e8; text-align: left; }}
.checklist-item {{ padding: 4px 0; font-size: 14px; }}
.checklist-high {{ color: #cb2431; font-weight: 600; }}
.checklist-medium {{ color: #b08800; }}
.checklist-low {{ color: #586069; }}
.reflection-ok {{ background: #dcffe4; border-left: 4px solid #22863a; padding: 8px 12px; margin: 8px 0; border-radius: 0 4px 4px 0; }}
.reflection-concern {{ background: #ffeef0; border-left: 4px solid #cb2431; padding: 8px 12px; margin: 8px 0; border-radius: 0 4px 4px 0; }}
.checklist-ops {{ background: #fff3cd; border-left: 4px solid #f9a825; padding: 8px 12px; margin: 8px 0; border-radius: 0 4px 4px 0; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
<h1>Spatial Agent Report &mdash; {session_id}</h1>
"""


def _plan_section(plan_text: str, checklist: list = None) -> str:
    """Render the agent's execution plan as a card.

    Uses marked.js (already included in the page header) for full markdown rendering.
    """
    import json as _json

    escaped_json = _json.dumps(plan_text)  # JS-safe string
    plan_id = "plan-md"

    # Render checklist items if present
    checklist_html = ""
    if checklist:
        checklist_html = _render_checklist_section(checklist)

    return f'''<div class="card">
<h3>Execution Plan</h3>
<div id="{plan_id}" class="markdown-body" style="background: #f0f7ff; border: 1px solid #c8e1ff; border-radius: 4px; padding: 12px;"></div>
<script>document.getElementById("{plan_id}").innerHTML = marked.parse({escaped_json});</script>
{checklist_html}
</div>'''


_PRIORITY_EMOJI = {"HIGH": "\u2757", "MEDIUM": "\u26a0\ufe0f", "LOW": "\u2139\ufe0f"}
_STATUS_EMOJI = {"PENDING": "\u23f3", "VERIFIED": "\u2705", "FLAGGED": "\u274c"}


def _render_checklist_section(checklist: list) -> str:
    """Render a checklist as an HTML block with emoji status markers."""
    if not checklist:
        return ""
    rows = []
    for item in checklist:
        status = item.get("status", "PENDING")
        priority = item.get("priority", "?")
        desc = item.get("description", "")
        s_emoji = _STATUS_EMOJI.get(status, "")
        p_emoji = _PRIORITY_EMOJI.get(priority, "")
        css_class = f"checklist-{priority.lower()}" if priority in _PRIORITY_EMOJI else ""
        note = item.get("resolution_note", "")
        note_html = f' &mdash; <em>{_escape(note)}</em>' if note else ""
        rows.append(
            f'<div class="checklist-item {css_class}">'
            f'{s_emoji} [{priority}] {_escape(desc)}{note_html}'
            f'</div>'
        )
    return (
        '<div style="margin-top: 12px; padding: 10px; background: #fffbea; '
        'border: 1px solid #f0e68c; border-radius: 4px;">'
        '<strong>Verification Checklist</strong>'
        + "\n".join(rows)
        + '</div>'
    )


def _render_reflection_event(ev: Dict) -> str:
    """Render a reflection trace event as HTML."""
    status = ev.get("status", "unknown")
    explanation = ev.get("explanation", "")
    checklist_ops = ev.get("checklist_ops", [])

    if status == "concern":
        css = "reflection-concern"
        icon = "\u274c"
    else:
        css = "reflection-ok"
        icon = "\u2705"

    parts = [f'<div class="{css}">']
    parts.append(f'<strong>{icon} Self-Reflection:</strong> {status.upper()}')
    if explanation:
        parts.append(f'<p style="margin:4px 0 0 0;">{_escape(explanation)}</p>')
    parts.append('</div>')

    if checklist_ops:
        op_parts = ['<div class="checklist-ops"><strong>Checklist Updates:</strong>']
        for op_dict in checklist_ops:
            if not isinstance(op_dict, dict):
                continue
            op = op_dict.get("op", "")
            if op == "verify":
                op_parts.append(
                    f'<div>\u2705 VERIFIED <code>{_escape(op_dict.get("item_id", "?"))}</code>'
                    f' &mdash; {_escape(op_dict.get("note", ""))}</div>'
                )
            elif op == "flag":
                op_parts.append(
                    f'<div>\u274c FLAGGED <code>{_escape(op_dict.get("item_id", "?"))}</code>'
                    f' &mdash; {_escape(op_dict.get("note", ""))}</div>'
                )
            elif op == "add":
                prio = op_dict.get("priority", "?")
                p_emoji = _PRIORITY_EMOJI.get(prio, "")
                op_parts.append(
                    f'<div>\u2795 ADDED [{prio}] {p_emoji} {_escape(op_dict.get("description", "?"))}</div>'
                )
        op_parts.append('</div>')
        parts.extend(op_parts)

    return "\n".join(parts)


def _input_section(instruction: str, images: List[Image.Image], gt: Optional[str]) -> str:
    parts = ['<div class="card"><h3>Input</h3>']
    parts.append(f'<p><strong>Instruction:</strong> {_escape(instruction)}</p>')
    if images:
        n = len(images)
        max_show = 32
        if n <= max_show:
            indices = list(range(n))
        else:
            indices = [round(i * (n - 1) / (max_show - 1)) for i in range(max_show)]
        parts.append(f'<p><em>{n} frames total, showing {len(indices)} uniformly sampled</em></p>')
        parts.append('<div class="img-row">')
        for i in indices:
            b64 = _img_to_base64(images[i], 200)
            parts.append(f'<img src="data:image/png;base64,{b64}" title="Frame {i}">')
        parts.append('</div>')
    if gt:
        parts.append(f'<p><strong>Ground Truth:</strong> <code>{_escape(gt)}</code></p>')
    parts.append('</div>')
    return "\n".join(parts)


def _step_card(step_idx: int, events: List[Dict], session_dir: str) -> str:
    parts = [f'<div class="card"><h3>Step {step_idx}</h3>']

    # Check if this step was condensed (affects how we render execution)
    condense_ev = None
    for ev in events:
        if ev.get("event_type") == "condense":
            condense_ev = ev
            break

    for ev in events:
        etype = ev.get("event_type", "")

        if etype == "llm_call":
            parsed = ev.get("parsed_response", {})
            if parsed:
                parts.append(f'<span class="badge badge-info">{_escape(parsed.get("purpose", ""))}</span>')
                # Hide reasoning/next_goal when condensed — they won't be in LLM context
                if condense_ev:
                    parts.append(f'<details><summary><em>Reasoning / Next Goal</em> (condensed out of LLM context)</summary>')
                    parts.append(f'<p><em>Reasoning:</em> {_escape(parsed.get("reasoning", ""))}</p>')
                    parts.append(f'<p><em>Next Goal:</em> {_escape(parsed.get("next_goal", ""))}</p>')
                    parts.append('</details>')
                else:
                    parts.append(f'<p><em>Reasoning:</em> {_escape(parsed.get("reasoning", ""))}</p>')
                    parts.append(f'<p><em>Next Goal:</em> {_escape(parsed.get("next_goal", ""))}</p>')
            if ev.get("error"):
                parts.append(f'<span class="badge badge-error">Parse Error</span>')
                parts.append(f'<div class="output-block">{_escape(ev["error"])}</div>')

        elif etype == "execution":
            code = ev.get("code", "")
            error = ev.get("error")

            if error:
                parts.append(f'<span class="badge badge-error">Error</span>')
                if condense_ev:
                    # Show condensed view (what LLM saw) as primary
                    condensed = condense_ev.get("condensed_content", "")
                    parts.append(
                        '<div class="condense-block">'
                        '<details open><summary><strong>LLM Context (condensed)</strong></summary>'
                        f'<div class="code-block">{_escape(condensed)}</div>'
                        '</details></div>'
                    )
                    # Full code + error collapsed for reference
                    if code:
                        parts.append('<details><summary>Full Code (reference)</summary>')
                        parts.append(f'<div class="code-block">{_escape(code)}</div>')
                        parts.append('</details>')
                    parts.append('<details><summary>Full Error</summary>')
                    parts.append(f'<div class="output-block">{_escape(error)}</div>')
                    parts.append('</details>')
                else:
                    # No condensation — original layout
                    if code:
                        parts.append('<details open><summary>Code</summary>')
                        parts.append(f'<div class="code-block">{_escape(code)}</div>')
                        parts.append('</details>')
                    parts.append(f'<div class="output-block">{_escape(error)}</div>')
            else:
                if code:
                    parts.append('<details open><summary>Code</summary>')
                    parts.append(f'<div class="code-block">{_escape(code)}</div>')
                    parts.append('</details>')
                parts.append(f'<span class="badge badge-success">Success</span>')
                t = ev.get("execution_time_sec", 0)
                parts.append(f' <small>({t:.2f}s)</small>')

            stdout = ev.get("stdout", "")
            if stdout:
                parts.append('<details><summary>Output</summary>')
                parts.append(f'<div class="output-block">{_escape(stdout[:3000])}</div>')
                parts.append('</details>')

        elif etype == "condense":
            pass  # handled above in the execution block

        elif etype == "reflection":
            parts.append(_render_reflection_event(ev))

        elif etype == "answer_rejected":
            rejected = ev.get("rejected_answer", {})
            reason = ev.get("reason", "")
            parts.append(
                f'<div class="reflection-concern">'
                f'<strong>\U0001f6ab Answer Rejected by Reflection</strong> &mdash; '
                f'<code>{_escape(str(rejected.get("text", "")))}</code><br>'
                f'Reason: {_escape(reason)}'
                f'</div>'
            )

        elif etype == "answer_blocked":
            pending_ids = ev.get("pending_high_items", [])
            parts.append(
                f'<div class="reflection-concern">'
                f'<strong>\u26d4 Answer Blocked</strong> &mdash; '
                f'{len(pending_ids)} HIGH item(s) still pending: '
                f'{", ".join(f"<code>{_escape(i)}</code>" for i in pending_ids)}'
                f'</div>'
            )

        elif etype == "feedback":
            fb_text = ev.get("feedback_text", "")
            if fb_text:
                parts.append('<details><summary>Feedback to LLM</summary>')
                parts.append(f'<div class="output-block">{_escape(fb_text)}</div>')
                parts.append('</details>')

    # Look for VLM query images
    vlm_dir = os.path.join(session_dir, "vlm_queries")
    vlm_events = [e for e in events if e.get("event_type") == "execution"]
    for ve in vlm_events:
        vc = ve.get("vlm_queries_count", 0)
        if vc > 0:
            loc = ve.get("vlm_locate_count", 0)
            thi = ve.get("vlm_thinking_count", 0)
            breakdown = []
            if loc:
                breakdown.append(f"{loc} locate")
            if thi:
                breakdown.append(f"{thi} thinking")
            other = vc - loc - thi
            if other:
                breakdown.append(f"{other} other")
            label = f"{vc} ({', '.join(breakdown)})" if breakdown else str(vc)
            parts.append(f'<p><strong>VLM Queries:</strong> {label}</p>')

            queries = ve.get("vlm_queries") or []
            for q in queries:
                qtype = q.get("query_type", "")
                qid = q.get("query_id", "")
                question = q.get("question", "")
                answer = q.get("answer", "")
                num_images = q.get("num_images", 0)
                badge = f' <code>{_escape(qtype)}</code>' if qtype else ''
                parts.append('<details>')
                parts.append(
                    f'<summary><strong>{_escape(qid)}</strong>{badge} '
                    f'({num_images} image{"s" if num_images != 1 else ""})</summary>'
                )
                parts.append(f'<p><em>Q:</em> {_escape(str(question))}</p>')
                parts.append(f'<p><em>A:</em> {_escape(str(answer))}</p>')
                # Render any saved images for this query
                if qtype in ("locate", "thinking") and qid:
                    qdir = os.path.join(vlm_dir, qtype)
                    if os.path.isdir(qdir):
                        img_paths = sorted(
                            os.path.join(qdir, f)
                            for f in os.listdir(qdir)
                            if f.startswith(qid + "_img_")
                        )
                        if img_paths:
                            parts.append('<div class="img-row">')
                            for ip in img_paths:
                                b64 = _img_path_to_base64(ip, 320)
                                if b64:
                                    parts.append(f'<img src="{b64}"/>')
                            parts.append('</div>')
                parts.append('</details>')

    # Show images (from show() calls)
    show_dir = os.path.join(session_dir, "show_images")
    if os.path.isdir(show_dir):
        # Find show images for this step by scanning trace for SHOW markers
        for ve in vlm_events:
            stdout = ve.get("stdout", "")
            import re as _re
            for m in _re.finditer(r'\[SHOW:(\{.*?\})\]', stdout):
                try:
                    meta = json.loads(m.group(1))
                    show_id = meta.get("show_id", "")
                    label = meta.get("label", "")
                    # Find labeled versions first, fall back to originals
                    img_paths = sorted(
                        p for p in (
                            os.path.join(show_dir, f)
                            for f in os.listdir(show_dir)
                            if f.startswith(show_id + "_labeled_")
                        )
                        if os.path.exists(p)
                    )
                    if not img_paths:
                        img_paths = sorted(
                            p for p in (
                                os.path.join(show_dir, f)
                                for f in os.listdir(show_dir)
                                if f.startswith(show_id + "_img_")
                            )
                            if os.path.exists(p)
                        )
                    if img_paths:
                        display_label = f"Visualization of {label}" if label else "show()"
                        parts.append(f'<p><strong>show():</strong> {_escape(display_label)}</p>')
                        parts.append('<div class="img-row">')
                        for ip in img_paths:
                            b64 = _img_path_to_base64(ip, 400)
                            if b64:
                                parts.append(
                                    f'<img src="data:image/png;base64,{b64}" '
                                    f'title="{_escape(display_label)}" '
                                    f'style="max-height:250px;">'
                                )
                        parts.append('</div>')
                except (json.JSONDecodeError, KeyError):
                    pass

    parts.append('</div>')
    return "\n".join(parts)


def _verdict_section(
    final_answer: Optional[Dict],
    ground_truth: Optional[str],
    termination_reason: Optional[str],
    result_score: Optional[float] = None,
) -> str:
    agent_ans = final_answer.get("text", "") if final_answer else ""

    # Determine correctness: prefer pre-computed result score
    css = "verdict-unknown"
    status = "Unknown"
    score_html = ""
    if result_score is not None:
        if result_score >= 1.0:
            css = "verdict-correct"
            status = "Correct"
        elif result_score > 0.0:
            css = "verdict-partial"
            status = f"Partial ({result_score:.3f})"
        else:
            css = "verdict-incorrect"
            status = "Incorrect"
        if 0.0 < result_score < 1.0:
            score_html = f'<p><strong>Score:</strong> {result_score:.4f}</p>'
    elif ground_truth and agent_ans:
        if agent_ans.strip().upper() == str(ground_truth).strip().upper():
            css = "verdict-correct"
            status = "Correct"
        else:
            css = "verdict-incorrect"
            status = "Incorrect"

    parts = [f'<div class="verdict {css}">']
    parts.append(f'<h2>{status}</h2>')
    parts.append(f'<p><strong>Agent Answer:</strong> {_escape(agent_ans)}</p>')
    if ground_truth:
        parts.append(f'<p><strong>Ground Truth:</strong> {_escape(str(ground_truth))}</p>')
    if score_html:
        parts.append(score_html)
    if termination_reason:
        parts.append(f'<p><em>Termination:</em> {termination_reason}</p>')
    parts.append('</div>')
    return "\n".join(parts)


def _html_footer() -> str:
    return """
</div>
</body>
</html>"""


def _escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
