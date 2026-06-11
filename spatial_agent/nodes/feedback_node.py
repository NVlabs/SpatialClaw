"""feedback_node: builds feedback text, detects ReturnAnswer, manages kernel state."""

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from PIL import Image, ImageDraw, ImageFont

from spatial_agent.kernel.variable_tracker import VariableTracker
from spatial_agent.kernel.feedback_collector import (
    FeedbackCollector, _compact_error, _extract_error_snippet,
    _extract_error_line_number, _search_code_for_pattern,
    _truncate_code_at_error,
)
from spatial_agent.state import AgentState


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    """Word-wrap *text* so each line fits within *max_width* pixels."""
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _add_label(image: Image.Image, label: str, padding: int = 32) -> Image.Image:
    """Add white padding on top of the image with a word-wrapped, centered label."""
    w, h = image.size
    # Use a temporary draw to measure text
    tmp = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(tmp)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()

    margin = 8  # horizontal margin on each side
    lines = _wrap_text(draw, label, font, max_width=w - 2 * margin)
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 4  # height + spacing
    total_text_h = line_height * len(lines)
    actual_padding = max(padding, total_text_h + 8)

    new_img = Image.new("RGB", (w, h + actual_padding), (255, 255, 255))
    new_img.paste(image, (0, actual_padding))
    draw = ImageDraw.Draw(new_img)

    y = (actual_padding - total_text_h) // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (w - text_w) // 2
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += line_height
    return new_img


def _build_multimodal_message(
    feedback_text: str,
    show_images: List[Dict[str, Any]],
    add_labels: bool = True,
) -> HumanMessage:
    """Build a multimodal HumanMessage with text + inline images."""
    from spatial_agent.llm.client import image_to_base64_url

    content_parts = [{"type": "text", "text": feedback_text}]
    for entry in show_images:
        label = entry.get("label", "")
        base_label = f"Visualization of {label}" if label else "Visualization"
        images = entry.get("images", [])
        for idx, img in enumerate(images):
            if add_labels:
                if len(images) > 1:
                    display_label = f"{base_label} ({idx + 1}/{len(images)})"
                else:
                    display_label = base_label
                img = _add_label(img, display_label)
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": image_to_base64_url(img)},
            })
    return HumanMessage(content=content_parts)


def _apply_session_budget(
    show_images: List[Dict[str, Any]], remaining: int,
) -> List[Dict[str, Any]]:
    """Truncate show_images to fit within *remaining* image budget."""
    if remaining <= 0:
        return []
    total = sum(len(e.get("images", [])) for e in show_images)
    if total <= remaining:
        return show_images
    kept: List[Dict[str, Any]] = []
    budget = remaining
    for entry in show_images:
        if budget <= 0:
            break
        imgs = entry.get("images", [])
        if len(imgs) <= budget:
            kept.append(entry)
            budget -= len(imgs)
        else:
            trimmed = dict(entry)
            trimmed["images"] = imgs[:budget]
            if "_paths" in trimmed:
                trimmed["_paths"] = trimmed["_paths"][:budget]
            kept.append(trimmed)
            budget = 0
    return kept


def _condense_error_ai_message(
    llm_response: Dict[str, Any],
    step_result: Dict[str, Any],
    executor_type: str = "code",
) -> str:
    """Build a condensed AIMessage for an errored step.

    Preserves the LLM's markdown structure (Purpose / Reasoning / Next Goal /
    Code) so the conversation history stays format-consistent.  Replaces
    Reasoning and Next Goal with short placeholders.

    For the Code section: keeps the code verbatim up to the error line and
    cuts everything after, so the LLM can see what variables were created.
    Falls back to a short snippet if the error line cannot be determined.
    """
    purpose = llm_response.get("purpose", "unknown")
    error = step_result.get("error", "")
    code = step_result.get("code", "")

    # Try to keep code up to error line (best for preserving context)
    truncated = _truncate_code_at_error(code, error)
    if truncated is not None:
        code_section = truncated
    else:
        # Fallback: snippet around error (current behavior for timeouts etc.)
        compact = _compact_error(error)
        snippet = _extract_error_snippet(code, error)
        code_section = f"# [ERROR] {compact}\n{snippet}" if snippet else f"# [ERROR] {compact}"

    body_label = "Tool Call (translated)" if executor_type == "react" else "Code"
    return (
        f"**Purpose**: {purpose}\n"
        f"**Reasoning**: [errored — condensed]\n"
        f"**Next Goal**: [errored — condensed]\n"
        f"**{body_label}**:\n```python\n{code_section}\n```"
    )


def _build_condense_messages(
    state: AgentState, step_result: Dict[str, Any], executor_type: str = "code",
) -> List:
    """Return [RemoveMessage, AIMessage] to replace a verbose errored AIMessage.

    Returns an empty list when condensation is not applicable (success, no
    LLM response to condense, condense_errors disabled, or msg.id missing).
    """
    if not step_result.get("error") or not state.get("current_llm_response"):
        return []
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.id:
            condensed = _condense_error_ai_message(
                state["current_llm_response"],
                step_result,
                executor_type=executor_type,
            )
            return [RemoveMessage(id=msg.id), AIMessage(content=condensed)]
    return []


def _maybe_condense(state, step_result, agent_config, logger) -> List:
    """Build condense messages and log if applicable. Shared by both return paths."""
    if not agent_config.condense_errors:
        return []
    executor_type = getattr(agent_config, "executor_type", "code")
    condense = _build_condense_messages(state, step_result, executor_type=executor_type)
    if condense and logger:
        logger.log_step(state["session_id"], {
            "event_type": "condense",
            "step_index": state["step_count"],
            "condensed_content": condense[-1].content,
        })
    return condense


async def feedback_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Collect execution feedback and append it as a HumanMessage.

    Also checks for ReturnAnswer sentinel and manages kernel restarts.
    """
    cfg = config["configurable"]
    km = cfg["kernel_manager"]
    agent_config = cfg["agent_config"]
    logger = cfg.get("logger")

    step_result = state.get("current_step_result")
    if step_result is None:
        return {"step_count": state["step_count"] + 1}

    # 1. Get current variables from kernel
    current_vars = await km.get_variables()

    # 2. Diff variables
    prev_vars = state.get("variable_registry", {})
    new_vars, changed_vars = VariableTracker.diff(prev_vars, current_vars)

    # 3. Determine rollback strategy for errored steps.
    #    - If we can find the error line AND there are new variables,
    #      those variables were assigned before the error → keep them
    #      (partial rollback / "has_survivors").
    #    - Otherwise (timeout, security, no new vars) → full rollback.
    is_error = bool(step_result.get("error"))
    is_condensed = is_error and agent_config.condense_errors
    new_var_names = set(current_vars) - set(prev_vars) if is_condensed else set()
    has_survivors = False
    if new_var_names:
        error_str = step_result.get("error", "")
        error_line = _extract_error_line_number(error_str)
        if error_line is None:
            code_lines = step_result.get("code", "").splitlines()
            error_line = _search_code_for_pattern(code_lines, error_str)
        has_survivors = error_line is not None

    full_rollback = is_condensed and not has_survivors

    # Full rollback: delete all new variables from kernel
    if full_rollback and new_var_names:
        delete_lines = [
            f"try:\n    del {name}\nexcept NameError:\n    pass"
            for name in sorted(new_var_names)
        ]
        try:
            await km.execute("\n".join(delete_lines), timeout=5)
        except Exception:
            pass  # best-effort cleanup
    # has_survivors: keep all variables — they were assigned before the error

    # 4. Check for large variables (skip on full rollback)
    large_var_warnings = []
    if not full_rollback:
        large_var_warnings = VariableTracker.check_large_variables(
            current_vars, max_size_mb=agent_config.max_variable_size_mb
        )

    # 5. Update step_result with variable info
    var_summaries = {}
    if full_rollback:
        pass  # all rolled back — no summaries
    elif has_survivors:
        # Show only new variables (survivors from before the error)
        for name, info in new_vars.items():
            var_summaries[name] = VariableTracker.format_summary(name, info)
    else:
        # Normal success path
        for name, info in {**new_vars, **changed_vars}.items():
            var_summaries[name] = VariableTracker.format_summary(name, info)
    step_result["new_variables"] = var_summaries

    # 6. Check for ReturnAnswer sentinel
    checklist = state.get("checklist", [])
    final_answer = await km.check_sentinel("_return_answer_result")
    if final_answer is not None:
        # Validate that the answer is usable
        if "text" in final_answer:
            # Clear sentinel for clean state
            await km.clear_sentinel("_return_answer_result")
            total_attempts = state.get("total_answer_attempts", 0)

            if logger:
                logger.log_step(state["session_id"], {
                    "event_type": "answer_submitted",
                    "step_index": state["step_count"],
                    "answer": final_answer,
                })

            # Build final feedback message
            feedback_text = FeedbackCollector.build_feedback(
                step_result, var_summaries, large_var_warnings,
                final_answer=final_answer,
                condensed=is_condensed,
                has_survivors=has_survivors,
            )
            checklist_text = FeedbackCollector.format_checklist(checklist)
            if checklist_text:
                feedback_text += checklist_text

            show_images = [] if full_rollback else step_result.get("show_images", [])
            images_this_step = 0
            max_session = agent_config.max_show_images_per_session
            prev_total = state.get("total_show_images", 0)
            if show_images and max_session >= 0:
                remaining = max_session - prev_total
                show_images = _apply_session_budget(show_images, remaining)
            images_this_step = sum(len(e.get("images", [])) for e in show_images)

            if show_images:
                message = _build_multimodal_message(
                    feedback_text, show_images,
                    add_labels=agent_config.show_image_labels,
                )
            else:
                message = HumanMessage(content=feedback_text)

            condense = _maybe_condense(state, step_result, agent_config, logger)
            result = {
                "messages": condense + [message],
                "step_count": state["step_count"] + 1,
                "total_show_images": prev_total + images_this_step,
                "variable_registry": prev_vars if full_rollback else current_vars,
                "final_answer": final_answer,
                "last_submitted_answer": final_answer,
                "termination_reason": "completed",
                "current_step_result": step_result,
                "answer_block_count": 0,
                "total_answer_attempts": total_attempts + 1,
            }
            if full_rollback:
                result["total_tool_calls"] = (
                    state["total_tool_calls"]
                    - step_result.get("tool_call_count", 0)
                )
                result["total_show_images"] = prev_total
            return result

    # 6. Build condense messages first (needed to decide feedback format)
    condense = _maybe_condense(state, step_result, agent_config, logger)

    # 6a. Build feedback text
    feedback_text = FeedbackCollector.build_feedback(
        step_result, var_summaries, large_var_warnings,
        condensed=bool(condense),
        has_survivors=has_survivors,
    )
    # Append checklist: full on step 1 (first time LLM sees it), compact otherwise
    if checklist:
        if state["step_count"] == 1:
            feedback_text += FeedbackCollector.format_checklist(checklist)
        else:
            compact = FeedbackCollector.format_checklist_compact(checklist)
            if compact:
                feedback_text += "\n" + compact

    # 6b. Warn the LLM on its last step before force-termination.
    steps_remaining = state["max_steps"] - (state["step_count"] + 1)
    if steps_remaining <= 1:
        feedback_text += (
            "\n\n[LAST STEP] This is your final step. The session will be "
            "force-terminated after this. You MUST call ReturnAnswer(...) now "
            "to submit your best answer."
        )

    # 6c. Enforce session-level show() budget and append budget info
    # Skip show images on full rollback only; keep on partial error
    # (images generated before the error are valid).
    show_images = [] if full_rollback else step_result.get("show_images", [])
    images_this_step = 0
    max_session = agent_config.max_show_images_per_session
    prev_total = state.get("total_show_images", 0)

    if show_images:
        if max_session >= 0:
            remaining = max_session - prev_total
            if remaining <= 0:
                show_images = []
                feedback_text += (
                    f"\n\n[show() budget] EXHAUSTED — 0/{max_session} remaining. "
                    f"Further show() calls will not produce inline images."
                )
            else:
                show_images = _apply_session_budget(show_images, remaining)

        images_this_step = sum(len(e.get("images", [])) for e in show_images)

        if images_this_step > 0 and max_session >= 0:
            budget_left = max_session - prev_total - images_this_step
            feedback_text += (
                f"\n[show() budget] {budget_left}/{max_session} images remaining."
            )

    if logger:
        logger.log_step(state["session_id"], {
            "event_type": "feedback",
            "step_index": state["step_count"],
            "feedback_text": feedback_text[:3000],
            "new_variables": list(new_vars.keys()),
            "changed_variables": list(changed_vars.keys()),
            "large_var_warnings": large_var_warnings,
        })

    # Build message (multimodal if show images present)
    if show_images:
        message = _build_multimodal_message(
            feedback_text, show_images,
            add_labels=agent_config.show_image_labels,
        )
    else:
        message = HumanMessage(content=feedback_text)

    result = {
        "messages": condense + [message],
        "step_count": state["step_count"] + 1,
        "total_show_images": prev_total + images_this_step,
        "variable_registry": prev_vars if full_rollback else current_vars,
        "current_step_result": step_result,
        "answer_block_count": 0,
    }
    if full_rollback:
        # Undo tool call and show image increments from execute_node
        result["total_tool_calls"] = (
            state["total_tool_calls"] - step_result.get("tool_call_count", 0)
        )
        result["total_show_images"] = prev_total
    return result
