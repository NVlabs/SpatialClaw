"""System prompt builder for the ReAct baseline.

Mirrors ``build_system_prompt`` but restricts the LLM to emitting **one
structured tool call per step** instead of arbitrary Python code. The tool
catalog (``tools.Reconstruct``, ``tools.SAM3``, etc.) is reused verbatim —
only the response format and the surrounding rules differ.
"""

import re
from typing import Any, Dict, List, Optional

from spatial_agent.llm.prompt_common import (
    build_input_description,
    coordinate_system_section,
    evidence_hierarchy_section,
    resolve_section,
    return_answer_section,
    robust_computation_section,
    vlm_api_section,
    warn_unknown_sections,
)


REACT_PROMPT_SECTIONS = {
    "header",
    "response_format",
    "vlm_api",
    "available_tools",
    "coordinate_system",
    "robust_computation",
    "evidence_hierarchy",
    "return_answer",
    "tool_call_rules",
    "workflow",
    "session_input",
    "ref_images",
}


def build_react_system_prompt(
    metadata: Dict[str, Any],
    tool_docs: str = "",
    key_frame_indices: Optional[List[int]] = None,
    key_frame_list_indices: Optional[List[int]] = None,
    key_frame_video_idx: Optional[List[int]] = None,
    num_ref_images: int = 0,
) -> str:
    """Build the full system prompt for the ReAct baseline agent."""
    from spatial_agent.config import get_config

    config = get_config()
    ablations = config.prompt_section_ablations
    from spatial_agent.tools import ToolsModule
    all_sections = REACT_PROMPT_SECTIONS | ToolsModule.get_all_tool_section_names()
    warn_unknown_sections(ablations, all_sections, "ReAct system prompt")

    num_images = metadata.get("num_images", 0)
    is_video = metadata.get("is_video", False)
    fps = metadata.get("fps")
    total_frames = metadata.get("total_frames")
    duration = metadata.get("duration_sec")
    reconstruct_max_frames = metadata.get("reconstruct_max_frames", 32)
    num_videos = metadata.get("num_videos", 1)
    videos_meta = metadata.get("videos")
    is_multi_video = num_videos > 1 and bool(videos_meta)
    max_steps = config.max_steps or 20

    if is_multi_video:
        _num_videos_suffix = f", `num_videos={num_videos}`"
        _video_lines = []
        for i, v in enumerate(videos_meta):
            vname = v.get("name") or f"video_{i+1}"
            vframes = v.get("num_frames", 0)
            vfps = v.get("fps")
            bits = [f"`InputImages_{i+1}` ({vname})", f"{vframes} frames"]
            if vfps:
                bits.append(f"{vfps} FPS")
            _video_lines.append("  - " + ": ".join([bits[0], ", ".join(bits[1:])]))
        _multi_video_section = (
            f"  - `Metadata.num_videos` — {num_videos} (multi-video input)\n"
            f"This sample has {num_videos} videos, each exposed as its own "
            f"`InputImages_<N>` (1-indexed). There is **no concatenated "
            "`InputImages`** — pass the per-video variable to tools (e.g. "
            "`{\"image\": \"InputImages_1[0]\"}`). For SAM3 video tracking, "
            "pass the matching `video_index` arg "
            "(e.g. `{\"prompts\": [\"person\"], \"video_index\": 2}`).\n"
            + "\n".join(_video_lines) + "\n"
        )
    else:
        _num_videos_suffix = ""
        _multi_video_section = ""

    input_desc = build_input_description(metadata)
    sighted = key_frame_indices is not None and len(key_frame_indices) > 0

    def _kf_var_name(pos: int) -> str:
        if is_multi_video and key_frame_video_idx and pos < len(key_frame_video_idx):
            v = key_frame_video_idx[pos]
            if v:
                return f"InputImages_{v}"
        return "InputImages"

    _kf_mapping_text = ""
    if sighted and key_frame_list_indices:
        _kf_lines = []
        n_kf = len(key_frame_indices)
        if n_kf <= 16:
            for i, (li, ai) in enumerate(
                zip(key_frame_list_indices, key_frame_indices)
            ):
                _kf_lines.append(
                    f"  Key frame #{i+1} → {_kf_var_name(i)}[{li}] (video frame {ai})"
                )
        else:
            for i in range(5):
                li, ai = key_frame_list_indices[i], key_frame_indices[i]
                _kf_lines.append(
                    f"  Key frame #{i+1} → {_kf_var_name(i)}[{li}] (video frame {ai})"
                )
            _kf_lines.append(f"  ... ({n_kf - 10} more) ...")
            for i in range(n_kf - 5, n_kf):
                li, ai = key_frame_list_indices[i], key_frame_indices[i]
                _kf_lines.append(
                    f"  Key frame #{i+1} → {_kf_var_name(i)}[{li}] (video frame {ai})"
                )
        _kf_mapping_text = "\n".join(_kf_lines)

    if sighted:
        num_kf = len(key_frame_indices)
        kf_source_label = (
            f"InputImages_1..{num_videos}" if is_multi_video else "InputImages"
        )
        vision_desc = (
            f"Your first message contains {num_kf} key frames — a visual "
            f"overview subset of {kf_source_label}."
        )
    else:
        no_visual_label = (
            f"InputImages_1..{num_videos}" if is_multi_video else "InputImages"
        )
        vision_desc = (
            f"You have no direct visual access to {no_visual_label}. Use `show` to "
            "view frames, `vlm.locate` to ground coordinates, "
            "`vlm.ask_with_thinking` for delegated visual reasoning, and "
            "SAM3 for masks."
        )

    if is_multi_video:
        input_images_desc = (
            f"- `InputImages_1`..`InputImages_{num_videos}` — one list per video "
            f"({num_images} total frames across all). Pass them by index/slice "
            f"in tool-call args (e.g. `\"InputImages_1[0]\"`, `\"InputImages_2[:32]\"`). "
            "There is **no concatenated `InputImages`**."
        )
    else:
        input_images_desc = (
            f"- `InputImages` — ALL {num_images} sampled frames, accessible by "
            f"index or slice in tool-call args (e.g. `InputImages[0]`, "
            f"`InputImages[:32]`)."
        )

    from spatial_agent.tools.sam3_tool import SAM3_VIDEO_METHODS_PROMPT

    aggregated_tool_docs = ToolsModule.get_all_prompt_descriptions_static(
        reconstruct_max_frames=reconstruct_max_frames,
        verify_visual="",
        sam3_video_methods=SAM3_VIDEO_METHODS_PROMPT.format(
            sam3_max_video_frames=config.sam3_max_video_frames,
        )
        if is_video
        else "",
        sam3_max_video_frames=config.sam3_max_video_frames,
    )
    if tool_docs:
        aggregated_tool_docs += "\n" + tool_docs

    # ---------- sections ----------

    _header = (
        "# Spatial Reasoning Agent — ReAct Tool-Call Mode\n\n"
        "You solve visual-spatial questions by invoking pre-defined tools "
        "**one at a time**. You do NOT write free-form Python code — every "
        "step is a single structured tool call whose result is captured in a "
        "persistent Jupyter kernel and named `result_<step>` for later use."
    )

    _response_format = (
        "## Response Format (MANDATORY)\n\n"
        "Every response MUST use this exact markdown format with a single "
        "JSON tool call:\n\n"
        "**Purpose**: What this step accomplishes\n\n"
        "**Reasoning**: Your chain-of-thought\n\n"
        "**Next Goal**: What you plan to call next\n\n"
        "**Tool Call**:\n"
        "```json\n"
        "{\"tool\": \"<dotted.name>\", \"args\": {\"<kwarg>\": <value>}}\n"
        "```\n\n"
        "Rules:\n"
        "- Exactly ONE JSON object per response. Multiple calls in one step "
        "are rejected.\n"
        "- `tool` is a dotted name: `tools.<Module>.<method>`, "
        "`vlm.locate`, `vlm.ask_with_thinking`, `show`, or `ReturnAnswer`. "
        "There is NO `feedback.show` — the tool is just `show`.\n"
        "- `args` is a JSON object mapping kwarg name → value. The value "
        "may be any JSON type:\n"
        "  - A **JSON string** that parses as a restricted Python expression "
        "(variable ref, attribute, subscript, slice, tuple/list/dict of "
        "the same) is injected as that expression. "
        "Example: `\"image\": \"InputImages[0]\"` → `image=InputImages[0]`.\n"
        "  - A **JSON string** of prose or any other natural-language text "
        "is treated as a Python string literal. "
        "Example: `\"question\": \"What is this?\"` → `question='What is this?'`. "
        "You do NOT need to double-escape quotes.\n"
        "  - A **JSON array** is rendered as a Python list, with each "
        "element interpreted by the same rule. "
        "Example: `\"visual_input\": [\"InputImages[0]\", \"InputImages[15]\"]` "
        "→ `visual_input=[InputImages[0], InputImages[15]]`.\n"
        "  - A **JSON object** is rendered as a Python dict under the same "
        "rules applied recursively to each value.\n"
        "- Allowed expression constructs: literals, variable refs "
        "(`InputImages`, `result_3`), attribute access (`result_5.depth`), "
        "subscripts/slices (`InputImages[:32]`, `result_2.frame_indices[0]`), "
        "tuples, lists, dicts.\n"
        "- A **single method call** per leaf is allowed when its receiver "
        "chain is rooted at a kernel-bound base (`result_<N>`, "
        "`InputImages`, `Metadata`, etc.). "
        "Example: `\"position\": \"result_2.get_centroid_3d(result_1, frame=result_1.frame_indices[0], object=0)\"` "
        "→ `position=result_2.get_centroid_3d(result_1, frame=result_1.frame_indices[0], object=0)`. "
        "Nested calls inside method args are rejected.\n"
        "- NOT allowed in expressions: operators (`+`, `*`, etc.), "
        "comparisons, comprehensions, lambdas, nested/free calls on "
        "non-kernel names (`np.argmax(...)`, `print(...)`). "
        "If you need a computed value, compute it with a prior tool call "
        "and reference the `result_<step>` variable."
    )

    _available_tools = (
        "## Available Tools\n\n"
        "Each tool is invoked as `{\"tool\": \"<name>\", \"args\": {...}}`. "
        "The result of each step is bound to `result_<step>` in the kernel "
        "and can be referenced in later arg expressions.\n\n"
        f"{aggregated_tool_docs}"
    )

    _tool_call_rules = (
        "## Tool Call Rules\n\n"
        "- **One tool per step.** Sequence dependent operations across "
        "multiple steps, storing intermediates as `result_<step>` and "
        "referencing them in subsequent calls.\n"
        "- **No free-form code.** If you attempt to embed expressions like "
        "`np.argmax(x)` inside an arg, the call is rejected with a parse "
        "error. Use the provided `tools.*` methods to obtain values.\n"
        "- **Inspect images with `show`.** Example: "
        "`{\"tool\": \"show\", \"args\": {\"image\": \"InputImages[13]\"}}`. "
        "The kwarg name is flexible (`image`, `images`, `visual_input`). "
        "There is NO `feedback.show`.\n"
        "- **Ground coordinates with `vlm.locate`.** Example: "
        "`{\"tool\": \"vlm.locate\", \"args\": {\"visual_input\": "
        "\"InputImages[13]\", \"question\": \"Give the (x,y) center "
        "coordinates in 0-1000 normalized scale for the red circle. Reply "
        "with ONLY the numbers.\"}}`.\n"
        "- **Delegate visual reasoning with `vlm.ask_with_thinking`** "
        "when the question needs an interpretation across one or more "
        "frames that other tools do not produce, or when a tool's output "
        "did not actually answer the question. Example: "
        "`{\"tool\": \"vlm.ask_with_thinking\", \"args\": {\"visual_input\": "
        "[\"InputImages[0]\", \"InputImages[15]\", \"InputImages[30]\"], "
        "\"question\": \"Across these frames, does the person walk toward "
        "or away from the camera?\"}}`. Up to 64 images per call.\n"
        "- **Submit the final answer** by calling `ReturnAnswer` with a "
        "string/int/float. Example: "
        "`{\"tool\": \"ReturnAnswer\", \"args\": {\"answer\": \"B\"}}` — "
        "the JSON string `\"B\"` is auto-wrapped as a Python literal.\n"
        "- **Variable naming:** previous results are `result_0`, `result_1`, "
        "..., indexed by the step that produced them. `InputImages`, "
        "`Metadata`, `tools`, `feedback`, `vlm`, and `ReturnAnswer` are "
        "always available."
    )

    single_turn = max_steps <= 1
    if single_turn:
        _workflow_body = (
            "## Workflow — SINGLE-TURN MODE\n"
            "You have **exactly ONE tool call** — there is no next step. "
            "Your one call MUST be `ReturnAnswer(...)` with your final "
            "answer, after whatever single evidence call you choose. "
            "If you do not call `ReturnAnswer`, the system force-terminates."
        )
    else:
        _workflow_body = (
            "## Workflow\n"
            "A plan has been prepared for you. Work the plan one tool call "
            "at a time. After each step, feedback tells you what changed "
            "and whether the call succeeded. Use `vlm.locate` for "
            "coordinate grounding, then SAM3 for masks, then Reconstruct "
            "for 3D, then geometry/mask utilities. Use "
            "`vlm.ask_with_thinking` when the question needs visual "
            "interpretation across frames not produced by the other "
            "tools. Cross-validate before calling `ReturnAnswer`."
        )

    if is_multi_video:
        _input_images_indexing = (
            f"  - Each `InputImages_<N>[i]` → `FrameImage` (PIL-compatible, "
            f"`.frame_index` attribute).\n"
            f"  - `InputImages_<N>[start:end]` → `InputImages` subset.\n"
            f"  - `InputImages_<N>.frame_indices` — absolute frame indices for "
            f"video N.\n"
        )
    else:
        _input_images_indexing = (
            f"  - `InputImages[i]` → `FrameImage` (PIL-compatible, "
            f"`.frame_index` attribute).\n"
            f"  - `InputImages[start:end]` → `InputImages` subset.\n"
            f"  - `InputImages.frame_indices` — list of absolute video frame "
            f"indices.\n"
        )

    _session_input = (
        "## This Session's Input\n\n"
        f"{vision_desc}\n\n"
        f"{input_desc}\n\n"
        f"{input_images_desc}\n"
        + _input_images_indexing
        + (f"- **Key frame → variable mapping:**\n{_kf_mapping_text}\n"
           if _kf_mapping_text else "")
        + f"- `Metadata` — `is_video={is_video}`, `fps={fps}`, "
        + f"`total_frames={total_frames}`, `num_images={num_images}`"
        + (f", `duration_sec={duration:.1f}`" if duration else "")
        + _num_videos_suffix
        + "\n"
        + (_multi_video_section if _multi_video_section else "")
    )

    # Reference images — appended only when the sample carries them, so
    # samples without ref images produce byte-identical prompt output.
    if num_ref_images > 0:
        _ref_images = (
            "## Reference Images\n\n"
            f"The question contains {num_ref_images} inline "
            "`[reference image #N]` tag(s) (1-indexed). Each corresponds to "
            "an actual image shown in your first message, and is also "
            "accessible as `RefImages[N-1]` — a list of PIL Images in the "
            "kernel namespace you can reference in tool-call args.\n\n"
            "Use them the same way as `InputImages[i]` in your tool calls "
            "(e.g. pass `RefImages[0]` to `vlm.locate` for grounding or "
            "to `vlm.ask_with_thinking` for visual reasoning)."
        )
    else:
        _ref_images = ""

    # ---------- resolve (ablations) ----------

    r = resolve_section

    resolved_workflow = r("workflow", _workflow_body, ablations)
    if resolved_workflow and not single_turn:
        resolved_workflow += f"\n\n## Budget: {max_steps} steps."

    sections = [
        r("header", _header, ablations),
        r("response_format", _response_format, ablations),
        r("vlm_api", vlm_api_section(), ablations),
        r("available_tools", _available_tools, ablations),
        r("coordinate_system", coordinate_system_section(), ablations),
        r("robust_computation", robust_computation_section(), ablations),
        r("evidence_hierarchy", evidence_hierarchy_section(), ablations),
        r("return_answer", return_answer_section(), ablations),
        r("tool_call_rules", _tool_call_rules, ablations),
        resolved_workflow,
        r("session_input", _session_input, ablations),
        r("ref_images", _ref_images, ablations),
    ]

    prompt = "\n\n".join(s for s in sections if s)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    return prompt.strip()
