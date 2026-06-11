"""execute_node: security-checks the code and runs it in the Jupyter kernel."""

import ast
import json
import re
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from PIL import Image

from spatial_agent.kernel.safety import SecuritySandbox
from spatial_agent.state import AgentState


# ---------------------------------------------------------------------------
# Helpers for sighted feedback (show() image collection)
# ---------------------------------------------------------------------------

_SHOW_RE = re.compile(r"^\[SHOW:(\{.*\})\]$", re.MULTILINE)


def _parse_show_markers(stdout: str) -> List[Dict[str, Any]]:
    """Parse [SHOW:{...}] markers from stdout and load images from disk."""
    results = []
    for m in _SHOW_RE.finditer(stdout):
        try:
            meta = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        images = []
        for p in meta.get("paths", []):
            try:
                images.append(Image.open(p).copy())
            except Exception:
                pass
        if images:
            results.append({
                "show_id": meta.get("show_id", ""),
                "label": meta.get("label", ""),
                "images": images,
                "_paths": meta.get("paths", []),
            })
    return results


def _strip_show_markers(stdout: str) -> str:
    """Remove [SHOW:...] lines from stdout so the agent doesn't see raw JSON."""
    return _SHOW_RE.sub("", stdout).strip()


def _extract_show_labels(code: str) -> List[str]:
    """AST-parse code to extract the source text of show() call arguments."""
    labels = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return labels
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match top-level show(...) or feedback.show(...)
        func = node.func
        is_show = (
            (isinstance(func, ast.Name) and func.id == "show")
            or (
                isinstance(func, ast.Attribute)
                and func.attr == "show"
                and isinstance(func.value, ast.Name)
                and func.value.id == "feedback"
            )
        )
        if not is_show or not node.args:
            continue
        # Extract source segment for the first positional argument
        segment = ast.get_source_segment(code, node.args[0])
        if segment:
            labels.append(segment)
        else:
            labels.append("")
    return labels


async def execute_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute the LLM-generated code in the Jupyter kernel.

    If ``current_llm_response`` is None (parse failure), skip execution.
    """
    cfg = config["configurable"]
    km = cfg["kernel_manager"]
    agent_config = cfg["agent_config"]
    logger = cfg.get("logger")
    feedback_module = cfg.get("feedback_module")
    vlm_module = cfg.get("vlm_module")

    llm_response = state.get("current_llm_response")
    if llm_response is None:
        # Parse failure or LLM call error in llm_step_node -- nothing to execute
        error_type = state.get("last_error_type", "unknown")
        if error_type == "llm_call_failed":
            error_msg = "Skipped: LLM call failed (check logs for details)."
        elif error_type == "validation_failed":
            error_msg = "Skipped: LLM response was not in the required format."
        else:
            error_msg = "Skipped: No valid LLM response."
        return {
            "current_step_result": {
                "step_index": state["step_count"],
                "code": "",
                "stdout": "",
                "stderr": "",
                "error": error_msg,
                "new_variables": {},
                "vlm_queries": [],
                "show_images": [],
                "tool_call_count": 0,
                "execution_time_sec": 0.0,
            }
        }

    code = llm_response["code"]

    # 1. Security check
    safety_error = SecuritySandbox.check(code)
    if safety_error:
        step_result = {
            "step_index": state["step_count"],
            "code": code,
            "stdout": "",
            "stderr": "",
            "error": f"Security violation: {safety_error}",
            "new_variables": {},
            "vlm_queries": [],
            "show_images": [],
            "tool_call_count": 0,
            "execution_time_sec": 0.0,
        }
        if logger:
            logger.log_step(state["session_id"], {
                "event_type": "execution",
                "step_index": state["step_count"],
                "code": code,
                "error": safety_error,
            })
        return {"current_step_result": step_result}

    # 2. Clear feedback/vlm query buffers before execution
    if feedback_module and hasattr(feedback_module, "get_and_clear_queries"):
        feedback_module.get_and_clear_queries()
    if vlm_module and hasattr(vlm_module, "get_and_clear_queries"):
        vlm_module.get_and_clear_queries()

    # 3. Execute in the Jupyter kernel
    result = await km.execute(code, timeout=agent_config.timeout_sec)

    # 4. Count tool calls (regex on code)
    tool_calls = len(re.findall(r"tools\.\w+\.\w+\(", code))
    vlm_calls = len(re.findall(r"vlm\.\w+\(", code))
    total_new_calls = tool_calls + vlm_calls

    # 5. Collect VLM queries from the vlm + feedback modules
    vlm_queries = []
    if vlm_module and hasattr(vlm_module, "get_and_clear_queries"):
        for q in vlm_module.get_and_clear_queries():
            vlm_queries.append({
                "query_id": q["query_id"],
                "query_type": q.get("query_type", ""),
                "image_source": q.get("source", ""),
                "question": q["question"],
                "answer": q["answer"],
                "num_images": q.get("num_images", 0),
            })
    if feedback_module and hasattr(feedback_module, "get_and_clear_queries"):
        for q in feedback_module.get_and_clear_queries():
            vlm_queries.append({
                "query_id": q["query_id"],
                "query_type": "log",
                "image_source": q.get("source", ""),
                "question": q["question"],
                "answer": q["answer"],
                "num_images": q.get("num_images", 0),
            })

    # 6. Collect show images (sighted feedback)
    show_images = []
    show_truncation_warning = ""
    if agent_config.enable_sighted_feedback and result.stdout:
        show_images = _parse_show_markers(result.stdout)
        # Patch labels from AST extraction when show() was called without explicit label
        ast_labels = _extract_show_labels(code)
        for i, entry in enumerate(show_images):
            if not entry["label"] and i < len(ast_labels) and ast_labels[i]:
                entry["label"] = ast_labels[i]
        # Enforce max images per step (negative = disabled)
        max_imgs = agent_config.max_show_images_per_step
        if max_imgs >= 0:
            total_img_count = sum(len(e["images"]) for e in show_images)
            if total_img_count > max_imgs:
                kept = []
                budget = max_imgs
                for entry in show_images:
                    if budget <= 0:
                        break
                    entry_imgs = entry["images"]
                    if len(entry_imgs) <= budget:
                        kept.append(entry)
                        budget -= len(entry_imgs)
                    else:
                        trimmed = dict(entry)
                        trimmed["images"] = entry_imgs[:budget]
                        if "_paths" in trimmed:
                            trimmed["_paths"] = trimmed["_paths"][:budget]
                        kept.append(trimmed)
                        budget = 0
                show_images = kept
                show_truncation_warning = (
                    f"{total_img_count} images requested but only "
                    f"{max_imgs} can be shown per step. "
                    f"Use fewer show() calls or fewer images per call."
                )
        # Save labeled versions to disk for HTML report
        if agent_config.show_image_labels:
            from spatial_agent.nodes.feedback_node import _add_label
            for entry in show_images:
                label = entry.get("label", "")
                base_label = f"Visualization of {label}" if label else "Visualization"
                paths = entry.get("_paths", [])
                images = entry.get("images", [])
                for j, img in enumerate(images):
                    if j < len(paths) and paths[j]:
                        if len(images) > 1:
                            display_label = f"{base_label} ({j + 1}/{len(images)})"
                        else:
                            display_label = base_label
                        labeled_path = paths[j].replace("_img_", "_labeled_")
                        try:
                            _add_label(img, display_label).save(labeled_path)
                        except Exception:
                            pass
        # Strip markers from stdout
        result_stdout = _strip_show_markers(result.stdout)
    else:
        result_stdout = result.stdout
    # Clear show buffer
    if feedback_module and hasattr(feedback_module, "get_and_clear_show_images"):
        feedback_module.get_and_clear_show_images()

    # 7. Handle timeout (kernel needs reset)
    if result.error and "timed out" in result.error.lower():
        try:
            await km.reset_namespace()
            # Re-inject per-sample objects after reset
            if km._injection_code:
                await km.execute(km._injection_code, timeout=30)
        except Exception:
            # Reset failed — full restart
            try:
                await km.shutdown()
                await km.start()
                if km._init_code:
                    await km.execute(km._init_code, timeout=120)
                if km._injection_code:
                    await km.execute(km._injection_code, timeout=30)
            except Exception:
                pass  # Kernel is dead; will be caught by next init_node

    step_result = {
        "step_index": state["step_count"],
        "code": code,
        "stdout": result_stdout,
        "stderr": result.stderr,
        "error": result.error,
        "new_variables": {},  # populated by feedback_node
        "vlm_queries": vlm_queries,
        "show_images": show_images,
        "show_truncation_warning": show_truncation_warning,
        "tool_call_count": total_new_calls,
        "execution_time_sec": result.execution_time_sec,
    }

    if logger:
        _ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        locate_count = sum(1 for q in vlm_queries if q.get("query_type") == "locate")
        thinking_count = sum(1 for q in vlm_queries if q.get("query_type") == "thinking")
        logger.log_step(state["session_id"], {
            "event_type": "execution",
            "step_index": state["step_count"],
            "code": code,
            "stdout": result.stdout[:2000],
            "stderr": _ansi_re.sub("", result.stderr[:1000]),
            "error": _ansi_re.sub("", result.error) if result.error else None,
            "tool_call_count": total_new_calls,
            "vlm_queries_count": len(vlm_queries),
            "vlm_locate_count": locate_count,
            "vlm_thinking_count": thinking_count,
            "vlm_queries": vlm_queries,
            "execution_time_sec": result.execution_time_sec,
        })

    return {
        "current_step_result": step_result,
        "total_tool_calls": state["total_tool_calls"] + total_new_calls,
    }
