"""System prompt builder for the main agent.

Supports two modes:
- **Sighted** (key_frame_indices provided): agent sees key frames in context
- **Blind** (key_frame_indices=None): agent relies on `show()` and the
  `vlm` module (vlm.locate / vlm.ask_with_thinking)

Tool descriptions are NOT hardcoded here.  Each tool class defines its own
``TOOL_PROMPT_DESCRIPTION`` (see ``tools/base.py``).  This module aggregates
them at runtime via ``ToolsModule.get_all_prompt_descriptions_static()``.

Prompt sections can be excluded or overridden via the
``prompt_section_ablations`` config field.  See ``resolve_section()`` in
``prompt_common.py``.
"""

import re
from typing import Any, Dict, List, Optional

from spatial_agent.llm.prompt_common import (
    build_input_description,
    code_rules_section,
    coordinate_system_section,
    evidence_hierarchy_section,
    resolve_section,
    return_answer_section,
    robust_computation_section,
    show_api_section,
    vlm_api_section,
    warn_unknown_sections,
)

# Valid section names for the main system prompt (used for typo detection).
MAIN_PROMPT_SECTIONS = {
    "header",
    "response_format",
    "show_api",
    "vlm_api",
    "available_tools",
    "coordinate_system",
    "robust_computation",
    "evidence_hierarchy",
    "return_answer",
    "code_rules",
    "workflow",
    "session_input",
    "ref_images",
}


def build_system_prompt(
    metadata: Dict[str, Any],
    tool_docs: str = "",
    key_frame_indices: Optional[List[int]] = None,
    key_frame_list_indices: Optional[List[int]] = None,
    key_frame_video_idx: Optional[List[int]] = None,
    num_ref_images: int = 0,
) -> str:
    """Build the full system prompt for the main agent."""
    from spatial_agent.config import get_config

    config = get_config()
    ablations = config.prompt_section_ablations
    from spatial_agent.tools import ToolsModule
    all_sections = MAIN_PROMPT_SECTIONS | ToolsModule.get_all_tool_section_names()
    warn_unknown_sections(ablations, all_sections, "main system prompt")

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

    # Build multi-video section (outside the main f-string to avoid nested brace issues)
    if is_multi_video:
        _num_videos_suffix = f', `num_videos={num_videos}`'
        _video_lines = []
        for i, v in enumerate(videos_meta):
            vname = v.get("name") or f"video_{i+1}"
            vframes = v.get("num_frames", 0)
            vfps = v.get("fps")
            bits = [f"`InputImages_{i+1}` ({vname})", f"{vframes} frames"]
            if vfps:
                bits.append(f"{vfps} FPS")
            _video_lines.append("  - " + ": ".join([bits[0], ", ".join(bits[1:])]))
        _video_block = "\n".join(_video_lines)
        _multi_video_section = (
            f"  - `Metadata.num_videos` — {num_videos} (this is a **multi-video** input)\n"
            "\n"
            "### Multi-Video Access\n"
            f"This sample has {num_videos} separate videos, each exposed as its own "
            f"`InputImages_<N>` (1-indexed). There is **no concatenated `InputImages`** "
            "— always use the per-video variables.\n\n"
            f"{_video_block}\n\n"
            "Each `InputImages_<N>` is a regular `InputImages` (PIL-compatible "
            "`FrameImage` list with `.frame_indices`). Pass them to tools just "
            "like a single-video `InputImages`:\n"
            "```python\n"
            "seg = tools.SAM3.segment_image_by_text(InputImages_1[0], \"red car\")\n"
            "recon1 = tools.Reconstruct.Reconstruct(InputImages_2[:32])\n"
            "```\n"
            "For SAM3 video tracking, pass the matching `video_index` (1-indexed):\n"
            "```python\n"
            "# Track 'person' through video 2's frames\n"
            "seg = tools.SAM3.segment_video_by_text([\"person\"], video_index=2)\n"
            "```\n"
            "Per-video metadata is in `Metadata.videos[i-1]` "
            "(keys: `name`, `fps`, `num_frames`, `duration_sec`). "
            "Analyze each video separately when needed, then compare across videos "
            "to answer cross-video questions."
        )
    else:
        _num_videos_suffix = ""
        _multi_video_section = ""

    input_desc = build_input_description(metadata)
    sighted = key_frame_indices is not None and len(key_frame_indices) > 0

    # Build key frame mapping text. In multi-video mode each line points at the
    # owning video's per-video variable (`InputImages_<N>[i]`).
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
            for i, (li, ai) in enumerate(zip(key_frame_list_indices, key_frame_indices)):
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

    # Vision intro (sighted mode — agent can see images via show())
    if sighted:
        num_kf = len(key_frame_indices)
        kf_source_label = (
            f"InputImages_1..{num_videos}" if is_multi_video else "InputImages"
        )
        vision_desc = (
            f"Your first message contains {num_kf} **key frames** — a visual overview subset of "
            f"{kf_source_label}. "
            f"Use `show(image)` to inspect additional frames or intermediate results inline."
        )
    else:
        vision_desc = (
            "You can see images by calling `show(image)`. Images appear inline in the next feedback. "
            "Use `show()` as your primary visual inspection method."
        )
    if is_multi_video:
        input_images_desc = (
            f"- `InputImages_1`..`InputImages_{num_videos}` — one per video "
            f"({num_images} total frames across all). Each is a `FrameImage` list. "
            f"There is **no concatenated `InputImages`** in multi-video mode."
        )
    else:
        input_images_desc = (
            f"- `InputImages` — ALL {num_images} sampled frames (the full set, not just key frames). "
            f"Use `show()` to see any of them."
        )

    # Aggregate tool descriptions from each tool class
    _vv_var = "InputImages_1" if is_multi_video else "InputImages"
    verify_visual = (
        "use `show()` to visually inspect the mask overlay yourself:\n"
        "```\n"
        "vis = seg.visualize(frame_idx)\n"
        f"show([{_vv_var}[img_idx], vis])  # compare original vs mask\n"
        "```\n"
        "Check whether the mask covers the correct object. "
    )

    from spatial_agent.tools.sam3_tool import SAM3_VIDEO_METHODS_PROMPT

    aggregated_tool_docs = ToolsModule.get_all_prompt_descriptions_static(
        reconstruct_max_frames=reconstruct_max_frames,
        verify_visual=verify_visual,
        sam3_video_methods=SAM3_VIDEO_METHODS_PROMPT.format(
            sam3_max_video_frames=config.sam3_max_video_frames,
        ) if is_video else "",
        sam3_max_video_frames=config.sam3_max_video_frames,
    )
    if tool_docs:
        aggregated_tool_docs += "\n" + tool_docs

    # -- Build each named section (default content) --

    _input_var_intro = (
        f"`InputImages_1`..`InputImages_{num_videos}` (one `FrameImage` list per video) "
        f"and `Metadata`"
        if is_multi_video
        else "`InputImages` (a list of `FrameImage` objects) and `Metadata`"
    )
    _header = (
        "# Spatial Reasoning Agent\n\n"
        "You solve visual-spatial questions by writing Python code in a Jupyter kernel.\n"
        f"Your input is {_input_var_intro}. "
        "Full details in \"This Session's Input\" below."
    )

    _response_format = (
        "## Response Format (MANDATORY)\n\n"
        "Every response MUST use this exact markdown format:\n\n"
        "**Purpose**: What this code step accomplishes\n\n"
        "**Reasoning**: Your chain-of-thought reasoning\n\n"
        "**Next Goal**: What you plan to do next\n\n"
        "**Code**:\n"
        "```python\n"
        "# Python code to execute — written verbatim, no escaping needed\n"
        "print(\"hello\")\n"
        "```\n\n"
        "All four sections are required. Code must be inside a fenced ```python block."
    )

    _available_tools = (
        "## Available Tools\n\n"
        "Access via `tools.X.method(...)`.\n\n"
        f"{aggregated_tool_docs}"
    )

    single_turn = max_steps <= 1
    if single_turn:
        _workflow_body = (
            "## Workflow — SINGLE-TURN MODE\n"
            "You have **exactly ONE code execution step**. There is NO second chance.\n"
            "Your code block MUST:\n"
            "1. Gather all needed evidence (visual queries, tool calls, computation)\n"
            "2. Call `ReturnAnswer(...)` at the end to submit your final answer\n\n"
            "If you do not call `ReturnAnswer`, the system will force-terminate and guess.\n"
            "Do NOT leave analysis for a \"next step\" — there is none."
        )
    else:
        _workflow_body = (
            "## Workflow\n"
            "An execution plan has been prepared for you. **Follow the plan faithfully**, writing code for each step.\n"
            "- If the plan specifies tool calls, execute them. Do not skip planned steps.\n"
            "- **Cross-validate**: Before answering, ensure your conclusion is supported by at least two independent lines of evidence. If visual perception and geometric computation disagree, diagnose why before concluding.\n"
            "- When quantitative methods are available, prefer them over qualitative visual judgments — but verify that the pipeline's inputs (masks, coordinates) are correct."
        )

    if is_multi_video:
        _input_images_indexing = (
            f"  - Each `InputImages_<N>[i]` → `FrameImage` (PIL-compatible, "
            f"`.frame_index` attribute).\n"
            f"  - `i` is a list index inside that video (0 to len-1), NOT an absolute video frame number.\n"
            f"  - `InputImages_<N>[start:end]` → `InputImages` subset (preserves frame indices).\n"
            f"  - `InputImages_<N>.frame_indices` — absolute frame indices for video N (its own, not concatenated).\n"
            f"  - **Key frames in your first message are a SUBSET of these per-video lists.**\n"
        )
        _metadata_extras = (
            f"  - `Metadata.video_source` — path to original video container (may be `None`)\n"
            f"  - `Metadata.videos` — list of {num_videos} dicts, one per video "
            f"(`Metadata.videos[i-1]` describes `InputImages_<i>`).\n"
        )
    else:
        _input_images_indexing = (
            f"  - `InputImages[i]` → `FrameImage` (PIL-compatible, has `.frame_index` attribute)\n"
            f"  - **`i` is a list index (0 to {num_images}-1), NOT a video frame number.** Use `img.frame_index` to get the absolute video frame number.\n"
            f"  - `InputImages[start:end]` → `InputImages` subset (preserves frame indices)\n"
            f"  - `InputImages.frame_indices` — list of absolute video frame indices (e.g., [0, 15, 30, ...])\n"
            f"  - Example: if `InputImages.frame_indices == [0, 15, 30]`, then `InputImages[1]` is the frame at video position 15.\n"
            f"  - **Key frames in your first message are a SUBSET of InputImages.** Not all InputImages entries are shown as key frames.\n"
        )
        _metadata_extras = (
            f"  - `Metadata.video_source` — path to original video file (may be `None`)\n"
        )

    _session_input = (
        "## This Session's Input\n\n"
        f"{vision_desc}\n\n"
        f"{input_desc}\n\n"
        f"{input_images_desc}\n"
        + _input_images_indexing
        + (f"- **Key frame → variable mapping:**\n{_kf_mapping_text}\n" if _kf_mapping_text else "")
        + f"- `Metadata` — `is_video={is_video}`, `fps={fps}`, `total_frames={total_frames}`, `num_images={num_images}`"
        + (f", `duration_sec={duration:.1f}`" if duration else "")
        + _num_videos_suffix + "\n"
        + _metadata_extras
        + (_multi_video_section if _multi_video_section else "")
    )

    # Reference images section — included only when ref images are present,
    # so a sample with none produces byte-identical prompt output.
    if num_ref_images > 0:
        _ref_images = (
            "## Reference Images\n\n"
            f"The question contains {num_ref_images} inline "
            "`[reference image #N]` tag(s) (1-indexed). Each corresponds to "
            "an actual image shown in your first message, and is also "
            "accessible in Python as `RefImages[N-1]` — a list of PIL Images "
            "in the kernel namespace.\n\n"
            "Use them the same way as `InputImages[i]`:\n"
            "- `show(RefImages[0])` to inspect inline.\n"
            "- `vlm.locate(RefImages[0], \"Give the (x, y) center \"\n"
            "  \"coordinates in 0-1000 normalized scale for ...\")` for grounding.\n"
            "- Pass to any tool that accepts a PIL image."
        )
    else:
        _ref_images = ""

    # -- Resolve sections (apply ablations) --

    r = resolve_section  # shorthand

    # Workflow: body is overridable, budget line always appended
    resolved_workflow = r("workflow", _workflow_body, ablations)
    if resolved_workflow and not single_turn:
        resolved_workflow += f"\n\n## Budget: {max_steps} steps."

    sections = [
        r("header", _header, ablations),
        r("response_format", _response_format, ablations),
        r("show_api", show_api_section(config.max_show_images_per_step, config.max_show_images_per_session, num_videos=num_videos), ablations),
        r("vlm_api", vlm_api_section(), ablations),
        r("available_tools", _available_tools, ablations),
        r("coordinate_system", coordinate_system_section(), ablations),
        r("robust_computation", robust_computation_section(), ablations),
        r("evidence_hierarchy", evidence_hierarchy_section(), ablations),
        r("return_answer", return_answer_section(), ablations),
        r("code_rules", code_rules_section(num_videos=num_videos), ablations),
        resolved_workflow,
        r("session_input", _session_input, ablations),
        r("ref_images", _ref_images, ablations),
    ]

    # Join non-empty sections and collapse excessive blank lines
    prompt = "\n\n".join(s for s in sections if s)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    return prompt.strip()
