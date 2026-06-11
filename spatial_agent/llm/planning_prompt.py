"""Planning system prompt and message builder for the isolated planning session.

The planner receives frame metadata (count, indices) but never sees actual
images.  This prevents the planner from answering the question visually and
forces it to plan a tool-based investigation.
This is entirely separate from the agent's system prompt (``system_prompt.py``).

Prompt sections can be excluded or overridden via the
``prompt_section_ablations`` config field.  Section names use the ``planning_``
prefix (e.g., ``planning_tool_decision``).  The embedded
``evidence_hierarchy`` and ``robust_computation`` sub-sections inside
``planning_tool_decision`` are resolved with their unprefixed names so that
excluding those shared names affects both the main and planning prompts.
"""

import re
from typing import List, Optional

from spatial_agent.llm.prompt_common import (
    build_input_description,
    coordinate_system_section,
    evidence_hierarchy_section,
    resolve_section,
    robust_computation_section,
    show_api_section,
    vlm_api_section,
    warn_unknown_sections,
)

# Valid section names for the planning prompt.
PLANNING_PROMPT_SECTIONS = {
    "planning_header",
    "planning_available_tools",
    "planning_show_api",
    "planning_vlm_api",
    "planning_coordinate_system",
    "planning_tool_decision",
    "planning_input",
    "planning_task",
    "planning_rules",
    "planning_checklist",
    "planning_ref_images",
    # Shared sub-section names (resolved within planning_tool_decision)
    "evidence_hierarchy",
    "robust_computation",
}


def build_planning_system_prompt(config, metadata_obj, num_ref_images: int = 0) -> str:
    """Build a dedicated system prompt for the planning LLM session."""
    from spatial_agent.tools import ToolsModule

    ablations = config.prompt_section_ablations
    all_sections = PLANNING_PROMPT_SECTIONS | ToolsModule.get_all_tool_section_names()
    warn_unknown_sections(ablations, all_sections, "planning prompt")
    r = resolve_section  # shorthand

    _vv_num_videos = getattr(metadata_obj, "num_videos", 1)
    _vv_var = "InputImages_1" if _vv_num_videos > 1 else "InputImages"
    verify_visual = (
        "use `show()` to visually inspect the mask overlay yourself:\n"
        "```\n"
        "vis = seg.visualize(frame_idx)\n"
        f"show([{_vv_var}[img_idx], vis])  # compare original vs mask\n"
        "```\n"
        "Check whether the mask covers the correct object. "
    )

    from spatial_agent.tools.sam3_tool import SAM3_VIDEO_METHODS_PROMPT

    is_video = getattr(metadata_obj, "is_video", False)
    tool_docs = ToolsModule.get_all_prompt_descriptions_static(
        reconstruct_max_frames=config.reconstruct_max_frames,
        verify_visual=verify_visual,
        sam3_video_methods=SAM3_VIDEO_METHODS_PROMPT.format(
            sam3_max_video_frames=config.sam3_max_video_frames,
        ) if is_video else "",
        sam3_max_video_frames=config.sam3_max_video_frames,
    )

    has_gpu_tools = any(
        t in config.tools_to_use for t in ["Reconstruct", "SAM3"]
    )

    # Build input description from metadata object
    metadata_dict = metadata_obj.to_dict() if hasattr(metadata_obj, "to_dict") else {
        "is_video": getattr(metadata_obj, "is_video", False),
        "num_images": getattr(metadata_obj, "num_images", 0),
        "fps": getattr(metadata_obj, "fps", None),
        "total_frames": getattr(metadata_obj, "total_frames", None),
        "duration_sec": getattr(metadata_obj, "duration_sec", None),
    }
    input_desc = build_input_description(metadata_dict)

    # Add multi-video guidance for the planner
    num_videos = metadata_dict.get("num_videos", 1)
    videos_meta = metadata_dict.get("videos")
    is_multi_video = num_videos > 1 and bool(videos_meta)
    multi_video_note = ""
    if is_multi_video:
        names = [
            (v.get("name") or f"video_{i+1}")
            for i, v in enumerate(videos_meta)
        ]
        per_video_lines = []
        for i, v in enumerate(videos_meta):
            vname = v.get("name") or f"video_{i+1}"
            per_video_lines.append(
                f"  - `InputImages_{i+1}` ({vname}, {v.get('num_frames', 0)} frames)"
            )
        multi_video_note = (
            f"\n**Multi-Video Input:** This sample has {num_videos} videos, each "
            f"exposed as its own `InputImages_<N>` (1-indexed). There is **no "
            f"concatenated `InputImages`** in multi-video mode.\n"
            + "\n".join(per_video_lines) + "\n"
            f"Per-video metadata is in `Metadata.videos[i-1]`. "
            f"Plan to analyze each video separately, then compare across videos.\n"
        )

    # -- Build default content for each section --

    _header = (
        "# Spatial Reasoning Planner\n\n"
        "You are a planning-only assistant. You will be given a spatial reasoning question "
        "and metadata about the input frames. You do NOT see the images. Your job is to "
        "create a concrete execution plan that a separate agent (which CAN see images) will follow."
    )

    _available_tools = (
        "## Available Tools\n\n"
        f"{tool_docs}"
    )

    # Tool decision section with embedded sub-sections
    if has_gpu_tools:
        # Resolve shared sub-sections individually so excluding "evidence_hierarchy"
        # or "robust_computation" also removes them from the planning prompt.
        _evidence = r("evidence_hierarchy", evidence_hierarchy_section(), ablations)
        _robust = r("robust_computation", robust_computation_section(), ablations)
        _sub_sections = ""
        if _evidence:
            _sub_sections += "\n" + _evidence
        if _robust:
            _sub_sections += "\n" + _robust

        _tool_decision = (
            "## Planning Strategy — Tool Selection\n\n"
            "For each step, choose the tool whose evidence shape matches what the question needs AND "
            "that is expected to produce a reliable result on this input:\n"
            "- `vlm.locate`             — pixel coordinates of an object describable in words.\n"
            "- `tools.SAM3`             — per-frame masks given a point/box or text prompt.\n"
            "- `tools.Reconstruct`      — 3D geometry, depth, camera pose for a frame range.\n"
            "- `tools.Geometry`/`Mask`  — numeric ops on coordinates / mask statistics.\n"
            "- `show()`                 — the executing agent looks at frames and decides.\n"
            "- `vlm.ask_with_thinking`  — a separate visual reasoner returns a text answer about the provided frames.\n\n"
            "`vlm.ask_with_thinking` is appropriate when:\n"
            "  (a) the question needs a kind of visual judgment that none of the specialized tools above produces, or\n"
            "  (b) a specialized tool was the right shape but its output on this input does not actually answer the "
            "question (failed, ambiguous, or self-inconsistent).\n"
            "You can plan multiple `vlm.ask_with_thinking` calls — different frame selections or rephrasings — when "
            "one call is not sufficient.\n\n"
            "### Annotation overlays\n"
            "If the question references annotation overlays (colored points, circles, dots, markers), plan a `show()` "
            "step on the specific frame mentioned to identify what each annotation marks before any computation.\n\n"
            "### Coordinate grounding\n"
            "For SAM3 prompts that need pixel coordinates, plan a `vlm.locate` step asking for 0-1000 normalized "
            "center points or bounding boxes, then convert with `tools.Geometry.normalized_to_pixel()`.\n\n"
            "### Quantitative vs qualitative\n"
            "- Metric quantities (distances, angles, speeds, sizes in real units), 3D spatial relationships, and "
            "quantitative comparisons — plan tool-based geometric computations.\n"
            "- Object identity, scene description, action recognition, qualitative judgments visible in a frame — "
            "plan `show()` so the agent can reason from what is observed; when the judgment is uncertain or requires "
            "integrating evidence across frames, plan `vlm.ask_with_thinking`.\n\n"
            "### Verification\n"
            "- Plan to verify tool inputs (masks, coordinates) before relying on their outputs.\n"
            "- When two evidence sources may disagree, name the diagnostic step.\n"
            "- Multiple consistent `vlm.ask_with_thinking` answers across different frame selections or phrasings "
            "are stronger evidence than a single call.\n\n"
            "**NEVER pre-conclude the answer.** Your job is to plan an investigation, not to answer the question. "
            "Do NOT write \"the answer is likely X\" or \"I expect the answer is Y\" anywhere in your plan. "
            "If you catch yourself forming a hypothesis from the key frames, that hypothesis is EXACTLY what tools "
            "must verify — it is not evidence."
            + _sub_sections
        )
        example_plan = ""
    else:
        _tool_decision = ""
        _ex_var = "InputImages_1" if _vv_num_videos > 1 else "InputImages"
        example_plan = (
            "**Example plan** (for a \"what is to the left of the table?\" question):\n"
            f"1. `show([{_ex_var}[0], {_ex_var}[5]])` → visually inspect the scene layout yourself\n"
            "2. Identify objects and spatial relationships from what you see\n"
            "3. If the visual layout is ambiguous from a single look, plan a `vlm.ask_with_thinking` "
            "call with the relevant frames to get a text reading of the layout.\n"
            "4. Synthesize observations → `ReturnAnswer(\"A\")`\n"
            "This is just an example. Your plan may look very different depending on the question."
        )

    if is_multi_video:
        _indexing_note = (
            "**IMPORTANT:** Each `InputImages_<N>` is a list indexed 0 to len-1 "
            "(NOT by absolute video frame numbers). Absolute video frame indices "
            "are in `InputImages_<N>.frame_indices`. In your plan, use list "
            "indices like `InputImages_1[0]`, `InputImages_2[-1]`, "
            "`InputImages_3[len(InputImages_3)//2]` — NOT absolute frame numbers."
        )
    else:
        _indexing_note = (
            "**IMPORTANT:** `InputImages` is a list indexed 0 to N-1 (NOT by video frame numbers). "
            "The absolute video frame indices are in `InputImages.frame_indices`. In your plan, use list "
            "indices like `InputImages[0]`, `InputImages[-1]`, `InputImages[len(InputImages)//2]` — "
            "NOT absolute frame numbers."
        )

    _input = (
        "## This Session's Input\n\n"
        f"{input_desc}\n"
        "You will be told the frame count and frame indices, but you cannot see the actual images. "
        "Do NOT guess or infer answers from the question text — plan an investigation using tools.\n"
        f"{multi_video_note}\n"
        f"{_indexing_note}"
    )

    _task = (
        "## Your Task\n\n"
        "Create a plan that covers:\n\n"
        "1. **Task Analysis** — What type of spatial question is this? What is the target variable? "
        "What objects are involved? What answer format?\n"
        "   - **Resolve the implicit coordinate system.** The question almost certainly assumes a "
        "frame of reference without stating it. Decide: is it asking about pixel appearance, "
        "camera-relative 3D, or world-space 3D? If ambiguous (e.g., \"moving left\"), state your "
        "interpretation and why. (See the Coordinate Systems section.)\n\n"
        "2. **Information Needs** — What spatial information do you need to answer this reliably? "
        "Refer to the tool selection heuristic above. If the question requires information that is "
        "hard to judge visually, plan to use tools.\n\n"
        "3. **Computation Plan** — Ordered list of concrete steps. For each step: the tool call or "
        "VLM query, what data it produces, how it feeds into the next step. **Every spatial/geometric "
        "claim must be backed by tool-computed numbers, not VLM impressions.**\n\n"
        + (f"{example_plan}\n\n" if example_plan else "")
        + "4. **Verification Checklist** — See the CHECKLIST section below. You MUST include this "
        "section with a JSON array of verification items. Generate this IMMEDIATELY after your "
        "computation plan, before writing anything else.\n\n"
        "5. **Verification** — Cross-validate before answering:\n"
        "   - Every spatial conclusion must be supported by at least two independent pieces of evidence.\n"
        "   - Visually verify tool inputs and outputs before using them for the final answer.\n"
        "   - Name concrete verification steps (not conditional \"if needed\" phrases).\n"
        "   - If two evidence sources disagree, include a diagnostic step.\n\n"
        "6. **Fallbacks** — What could go wrong? Alternative approach?"
    )

    _rules = (
        "## CRITICAL RULES\n"
        "- Respond with your plan in **plain text only**.\n"
        "- Do NOT write JSON objects (except for the CHECKLIST section).\n"
        "- Do NOT write executable Python code.\n"
        "- Do NOT produce an answer or pre-conclude what the answer might be.\n"
        "- Do NOT write phrases like \"the answer is likely X\", \"therefore the answer is\", "
        "\"I expect the answer is\". You are BLIND — you cannot answer the question.\n"
        "- Your ONLY job is to PLAN. A separate agent will execute the plan later.\n"
        "- Use pseudocode sketches (e.g., `tools.Reconstruct.Reconstruct(...)`) to describe steps, "
        "but do NOT write full code."
    )

    # Checklist section uses plain string to avoid curly-brace conflicts with JSON.
    _checklist = (
        "## Verification Checklist\n\n"
        "After your plan, add a `### CHECKLIST` section with a JSON array of verification items. "
        "Each item has a `priority` (\"HIGH\", \"MEDIUM\", or \"LOW\") and a `description`:\n\n"
        "```\n"
        "### CHECKLIST\n"
        "```json\n"
        "[\n"
        '  {"priority": "HIGH", "description": "Verify the identified objects match what the question asks about"},\n'
        '  {"priority": "HIGH", "description": "Confirm inputs to each computation step are correct '
        '(visually verify intermediate results)"},\n'
        '  {"priority": "MEDIUM", "description": "Cross-check the conclusion using an independent method '
        'or evidence source"},\n'
        '  {"priority": "LOW", "description": "Sanity-check numerical results (order of magnitude, sign, range)"}\n'
        "]\n"
        "```\n\n"
        "**Priority guide:**\n"
        "- **HIGH**: If wrong, the answer is almost certainly wrong (e.g., wrong mask, wrong coordinate frame, "
        "wrong object identity).\n"
        "- **MEDIUM**: Could affect accuracy but has fallback (e.g., VLM identification needs cross-check).\n"
        "- **LOW**: Nice to verify but unlikely to cause errors (e.g., reconstruction quality check).\n\n"
        "HIGH-priority items MUST be verified before submitting the final answer."
    )

    # Reference images — included only when the sample has them so planner
    # prompts for other samples stay byte-identical.
    if num_ref_images > 0:
        _ref_images = (
            "## Reference Images\n\n"
            f"The question contains {num_ref_images} `[reference image #N]` "
            "tag(s) referring to static images the executing agent can access "
            "as `RefImages[N-1]` in the kernel. You are blind to their "
            "content (same as video frames) — plan steps that use "
            "`show(RefImages[N-1])`, `vlm.locate(RefImages[N-1], ...)`, or "
            "`vlm.ask_with_thinking(RefImages[N-1], ...)` during execution."
        )
    else:
        _ref_images = ""

    # -- Resolve sections (apply ablations) --

    sections = [
        r("planning_header", _header, ablations),
        r("planning_available_tools", _available_tools, ablations),
        r("planning_show_api", show_api_section(config.max_show_images_per_step, config.max_show_images_per_session, num_videos=_vv_num_videos), ablations),
        r("planning_vlm_api", vlm_api_section(), ablations),
        r("planning_coordinate_system", coordinate_system_section(), ablations),
        r("planning_tool_decision", _tool_decision, ablations),
        r("planning_input", _input, ablations),
        r("planning_ref_images", _ref_images, ablations),
        r("planning_task", _task, ablations),
        r("planning_rules", _rules, ablations),
        r("planning_checklist", _checklist, ablations),
    ]

    # Join non-empty sections and collapse excessive blank lines
    prompt = "\n\n".join(s for s in sections if s)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    return prompt.strip()


def build_planning_user_message(
    instruction: str,
    key_frames: Optional[List] = None,
    key_frame_indices: Optional[List[int]] = None,
    key_frame_list_indices: Optional[List[int]] = None,
    key_frame_video_idx: Optional[List[int]] = None,
    num_total_images: int = 0,
) -> dict:
    """Build an OpenAI-format user message with key frames for the planner."""

    def _kf_var(pos: int) -> str:
        if key_frame_video_idx and pos < len(key_frame_video_idx) and key_frame_video_idx[pos]:
            return f"InputImages_{key_frame_video_idx[pos]}"
        return "InputImages"

    def _build_mapping_text(
        kf_indices: List[int],
        kf_list_indices: List[int],
    ) -> str:
        """Build compact key frame mapping text."""
        n = len(kf_indices)
        lines = []
        if n <= 16:
            for i, (li, ai) in enumerate(zip(kf_list_indices, kf_indices)):
                lines.append(f"  #{i+1} → {_kf_var(i)}[{li}] (video frame {ai})")
        else:
            for i in range(5):
                li, ai = kf_list_indices[i], kf_indices[i]
                lines.append(f"  #{i+1} → {_kf_var(i)}[{li}] (video frame {ai})")
            lines.append(f"  ... ({n - 10} more) ...")
            for i in range(n - 5, n):
                li, ai = kf_list_indices[i], kf_indices[i]
                lines.append(f"  #{i+1} → {_kf_var(i)}[{li}] (video frame {ai})")
        return "\n".join(lines)

    if key_frames:
        from spatial_agent.llm.client import image_to_base64_url

        n_kf = len(key_frames)
        header = (
            f"Here are {n_kf} key frames — a visual overview subset of "
            f"InputImages ({num_total_images} total frames)."
        )
        if key_frame_list_indices and key_frame_indices:
            mapping = _build_mapping_text(key_frame_indices, key_frame_list_indices)
            header += (
                f"\nKey frame mapping (context position → InputImages index → video frame):\n"
                f"{mapping}"
            )

        content_parts = []
        content_parts.append({
            "type": "text",
            "text": header,
        })
        for img in key_frames:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": image_to_base64_url(img)},
            })
        content_parts.append({
            "type": "text",
            "text": f"Question: {instruction}",
        })
        return {"role": "user", "content": content_parts}
    else:
        text = f"Question: {instruction}"
        if key_frame_indices:
            text = (
                f"The input has {len(key_frame_indices)} frames "
                f"(frame indices: {key_frame_indices}).\n\n"
                f"Question: {instruction}"
            )
        return {"role": "user", "content": text}
