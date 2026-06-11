"""Shared prompt sections used by both system_prompt.py and planning_prompt.py.

Centralizes VLM API docs, ReturnAnswer docs, coordinate system, and input
description builders so they stay in sync across both prompts.
"""

import logging
from typing import Any, Dict, List, Optional, Set

_logger = logging.getLogger(__name__)


def resolve_section(name: str, default_content: str, ablations: dict) -> str:
    """Exclude, override, or return default content for a named prompt section.

    Args:
        name: Section identifier (e.g., "evidence_hierarchy", "planning_checklist").
        default_content: The content to use when no ablation applies.
        ablations: The ``prompt_section_ablations`` dict from config.
            Expected keys: ``exclude`` (list of names), ``override`` (dict of
            name → file path).

    Returns:
        The section text, or empty string if excluded.
    """
    if not ablations:
        return default_content

    if name in ablations.get("exclude", []):
        _logger.info("[prompt ablation] EXCLUDED: %s", name)
        return ""

    override_path = ablations.get("override", {}).get(name)
    if override_path:
        _logger.info("[prompt ablation] OVERRIDDEN: %s -> %s", name, override_path)
        with open(override_path, "r") as f:
            return f.read().rstrip()

    return default_content


def warn_unknown_sections(ablations: dict, valid_sections: Set[str], prompt_name: str) -> None:
    """Log warnings for unrecognized section names in ablation config (typo protection)."""
    if not ablations:
        return
    for name in ablations.get("exclude", []):
        if name not in valid_sections:
            _logger.warning(
                "[prompt ablation] Unknown section '%s' in exclude list for %s. "
                "Valid sections: %s",
                name, prompt_name, sorted(valid_sections),
            )
    for name in ablations.get("override", {}):
        if name not in valid_sections:
            _logger.warning(
                "[prompt ablation] Unknown section '%s' in override dict for %s. "
                "Valid sections: %s",
                name, prompt_name, sorted(valid_sections),
            )


def build_input_description(metadata: Dict[str, Any]) -> str:
    """Build a human-readable input description from metadata."""
    is_video = metadata.get("is_video", False)
    num_images = metadata.get("num_images", 0)
    fps = metadata.get("fps")
    total_frames = metadata.get("total_frames")
    duration = metadata.get("duration_sec")
    num_videos = metadata.get("num_videos", 1)
    videos = metadata.get("videos")

    if is_video:
        if num_videos > 1 and videos:
            # Multi-video: each video is exposed as InputImages_<N>
            desc = f"MULTI-VIDEO INPUT: {num_videos} videos, {num_images} total frames."
            parts = []
            for i, v in enumerate(videos):
                vname = v.get("name") or f"video_{i+1}"
                vfps = v.get("fps")
                vframes = v.get("num_frames", 0)
                vdur = v.get("duration_sec")
                bits = [f"InputImages_{i+1} ({vname})", f"{vframes} frames"]
                if vfps:
                    bits.append(f"{vfps} FPS")
                if vdur:
                    bits.append(f"{vdur:.1f}s")
                parts.append(": ".join([bits[0], ", ".join(bits[1:])]))
            desc += "\n  " + "\n  ".join(parts)
        else:
            if num_images == total_frames:
                desc = f"VIDEO: {num_images} frames at {fps} FPS."
            else:
                desc = (
                    f"VIDEO: {num_images} sampled frames "
                    f"from {total_frames} total at {fps} FPS."
                )
            if duration:
                desc += f" Duration: {duration:.1f}s."
    else:
        desc = f"{num_images} static images."

    return desc


def vlm_api_section(**_kwargs) -> str:
    """Return the VLM API documentation section."""
    return """## Visual Access — `show`, `vlm.locate`, `vlm.ask_with_thinking`

You have three ways to obtain visual information:

- `show(visual_input)` — display image(s) inline in the next feedback so you can see them yourself.
- `vlm.locate(visual_input, question)` → `str` — ask a grounding VLM for pixel coordinates of an object you describe. Up to 8 images per call.
- `vlm.ask_with_thinking(visual_input, question)` → `str` — ask a separate visual reasoner that deliberates over the provided frames and returns a text answer. Up to 64 images per call.

### Choosing a tool

Each tool produces a particular shape of evidence:

- `vlm.locate`             — pixel coordinates of an object you can describe in words.
- `tools.SAM3`             — per-frame masks given a point/box or text prompt.
- `tools.Reconstruct`      — 3D geometry, depth, camera pose for a frame range.
- `tools.Geometry`         — numeric ops on coordinates and arrays.
- `tools.Mask`             — mask statistics (centroid, area, IoU).
- `show()`                 — you look at images yourself and decide.
- `vlm.ask_with_thinking`  — a visual reasoner returns a text answer about the provided frames.

Use a tool when its evidence shape matches what the question needs AND you expect it to produce a reliable result on this specific input. "Reliable" depends on the input, not just the question type — for example, `tools.Reconstruct` gives strong 3D geometry when the frames have enough camera motion and structure, and weaker output otherwise; `vlm.locate` needs an object describable in words.

`vlm.ask_with_thinking` covers two cases:
  (a) the question needs a kind of evidence none of the specialized tools produces;
  (b) a specialized tool was the right shape on paper but its output on this input does not actually answer the question (failed, ambiguous, or self-inconsistent).

`show()` is for judgments you want to make yourself by looking.

### Working with `vlm.locate`

Ask for coordinates in 0-1000 normalized scale.
- Center point: `vlm.locate(image, "Give the (x, y) center coordinates in 0-1000 normalized scale for <object>. Reply with ONLY the numbers.")` → use with `tools.SAM3.segment_image_by_points()`.
- Bounding box: `vlm.locate(image, "Give the bounding box (x1, y1, x2, y2) in 0-1000 normalized scale for <object>. Reply with ONLY the numbers.")` → use with `tools.SAM3.segment_image_by_box()`.
Convert to pixels:
```
px, py = tools.Geometry.normalized_to_pixel((vlm_x, vlm_y), W, H)
```
Estimating coordinates yourself by eye is unreliable — call `vlm.locate` for grounding.

`vlm.locate` may return the literal string `Not visible` (optionally followed by a short note) when the requested object/annotation is absent or ambiguous in the image. Always check for this before parsing the answer as coordinates; if you see it, try a different frame, refine the description, or first locate the target frame with `vlm.ask_with_thinking`.

### Working with `vlm.ask_with_thinking`

The session is independent: it sees only the images and the question you pass in. Do not write your prior conclusions, prior vlm answers, or expected answers into the question — phrase the question on its own terms so the answer is formed from the images.

Pick the frames the question is about (up to 64). Frames unrelated to the question add noise; include only the ones you want considered.

You can re-call the tool freely:
- if the answer is "Cannot determine from the images.", try a different frame selection or a different phrasing,
- to confirm an answer, call again with different frames or call `show()` and look yourself.

Multiple consistent answers across calls are stronger evidence than a single call. Inconsistent answers across calls mean the question is under-constrained — refine the framing or pick more informative frames.

### Pitfalls

- **NEVER** overwrite the `vlm` or `feedback` variables: `vlm = vlm.locate(...)` or `feedback = feedback.show(...)` destroys the module. Always assign to a different name: `coords = vlm.locate(...)`, `answer = vlm.ask_with_thinking(...)`."""


def return_answer_section() -> str:
    """Return the ReturnAnswer documentation section."""
    return """## ReturnAnswer — Submit Your Final Answer

```python
ReturnAnswer("B")                              # multiple-choice letter
ReturnAnswer(3)                                # integer
ReturnAnswer(3.14)                             # float
ReturnAnswer("The dog is left of the cat")     # free-form text
```

Accepts `str`, `int`, or `float`. Call once to submit your final answer and terminate."""


def coordinate_system_section() -> str:
    """Return the coordinate system documentation section."""
    return """## Coordinate Systems — Resolving the Implicit Frame of Reference

Every spatial question implicitly assumes a coordinate system, but almost never says which one. Your first job on any spatial question is to **resolve this ambiguity** — the wrong coordinate system gives the wrong answer even with perfect computation.

### The Core Problem

There are four coordinate spaces, and they can disagree:

- **Pixel space** (2D image plane): Where things appear in the image. Pixel position is a projection — it conflates camera motion with object motion. An object sliding rightward in the image could mean the object moved right, or the camera panned left, or both. **Pixel-space observations about motion or spatial relationships are fundamentally ambiguous** and should never be used as evidence for real-world spatial claims.

- **Camera space** (3D relative to a camera at a specific frame): "Left," "right," "in front," "behind" as seen from a particular viewpoint. This frame changes every frame as the camera moves — "to the left" at frame 0 may be "behind" at frame 50. Any camera-relative claim requires specifying **which frame's camera**.

- **World space** (3D global): Fixed coordinates that don't move with the camera. Object positions in world space are consistent across time — this is where real distances, speeds, and trajectories live. But "world left" is defined by the reconstruction (anchored to the first camera), which is arbitrary — it has no inherent meaning like "north" or "east."


- **Object perspective** (relative to an object's facing direction): "To the car's left," "in front of the person." This is NOT camera-relative — an object facing the camera has its left/right **mirrored** vs. the camera's. First determine which way the object faces, then define left/right/front/behind from that heading.

### Technical Convention

- Camera poses from Reconstruct are 4x4 **c2w** (camera-to-world) SE(3) matrices (OpenCV convention).
  Columns: `pose[:3, 0]`=right, `pose[:3, 1]`=down, `pose[:3, 2]`=forward, `pose[:3, 3]`=position.
- After reconstruction, the **world frame** is gravity-aligned relative to the first camera:
  - **+Y** = up (opposite gravity)
  - **-Z** = first camera's forward direction (into the scene)
  - **+X** = right from the first camera's perspective

### Relative Direction Computation (LEFT/RIGHT/FRONT/BEHIND)

**ALWAYS use this quantitative method for direction questions. NEVER guess direction from 2D image appearance.**

Given a target object's 3D position and the camera pose at the reference time:
```python
# Get 3D position using absolute frame index
fi = seg.frame_indices[0]                         # use absolute frame index
target_3d = seg.get_centroid_3d(recon, frame=fi, object=0)  # (3,) or None

# Get camera pose at the SAME frame
pose = recon.extrinsics[fi]   # (4, 4) c2w at reference time
cam_pos = pose[:3, 3]         # camera position in world
cam_fwd = pose[:3, 2]         # camera forward (into scene) in world
cam_right = pose[:3, 0]       # camera right in world

vec_to_target = target_3d - cam_pos              # world-frame vector to target
dot_fwd   = np.dot(vec_to_target, cam_fwd)       # positive = FRONT, negative = BEHIND
dot_right = np.dot(vec_to_target, cam_right)     # positive = RIGHT, negative = LEFT
```"""


def evidence_hierarchy_section(**_kwargs) -> str:
    """Return guidance on when to trust computation vs. VLM perception."""
    return """## Cross-Validation Principle

No single evidence source is reliable alone. Every spatial conclusion must be supported
by at least two independent lines of evidence before you answer.

### Evidence Sources and Their Limitations
- **Visual perception** (`show()`): Good at object identity, appearance, scene semantics,
  qualitative layout. Unreliable for metric quantities, precise spatial relationships, and
  distinguishing real motion from apparent motion.
- **Geometric computation** (Reconstruct, SAM3, code): Good at metric distances, angles, 3D
  positions, quantitative comparisons. Only as reliable as its inputs — wrong segmentation or
  noisy reconstruction propagates to wrong answers.
- **Visualizations** (BEV, depth maps, plots): Good for sanity-checking and debugging.
  But visualizations are lossy representations — a BEV is a 2D projection of 3D data,
  and its appearance depends on reference frame, scale, and rendering choices. A visualization
  that "looks wrong" is not proof that the underlying data is wrong.
- **Logical reasoning** (code, math): Good at combining evidence and checking consistency.
  Only as reliable as its premises.

### When Sources Disagree — Diagnostic Protocol
Do NOT pick a side based on intuition. Every disagreement has a **root cause** — your job
is to find it by tracing each evidence chain back to its inputs:

1. **Identify the specific claim in conflict.** e.g., "Dot product says RIGHT, but BEV plot
   shows the object on the left side."
2. **Audit the computation chain.** For each step, print and verify:
   - Are the segmentation masks non-empty and on the correct objects? (`show()` the overlay)
   - Are 3D coordinates non-NaN and physically plausible? (`print()` the values)
   - Is the correct frame index used? (camera pose at frame X, centroid from frame X)
   - Is the coordinate system correct? (camera-relative vs world-relative)
3. **Audit the visual evidence.** For each visual observation, ask:
   - Could the 2D appearance be misleading? (projection, foreshortening, reference frame)
   - Am I interpreting the visualization's axes and labels correctly?
   - Is this a qualitative impression or a precise measurement?
4. **Find the concrete error.** The disagreement must come from a specific, identifiable
   mistake — wrong mask, wrong frame, wrong coordinate system, misleading projection, etc.
   If you cannot identify a concrete error in either chain, you do not have enough information
   to override either conclusion. Gather more evidence instead.
5. **Never override evidence with intuition.** "It looks wrong" is not a diagnosis.
   You must point to the specific broken step before changing your answer."""


def show_api_section(
    max_show_images_per_step: int = -1,
    max_show_images_per_session: int = 250,
    num_videos: int = 1,
) -> str:
    """Return the show() API documentation section.

    In multi-video mode the inline examples use ``InputImages_1`` so the
    agent doesn't try to dereference a non-existent ``InputImages``.
    """
    budget_lines = []
    if max_show_images_per_session >= 0:
        budget_lines.append(
            f"- **show() budget: {max_show_images_per_session} total images** across "
            f"the entire session. Remaining budget is reported as `[show() budget]` "
            f"in feedback after each step that uses show()."
        )
    if max_show_images_per_step >= 0:
        budget_lines.append(
            f"- Max **{max_show_images_per_step} images per step** "
            f"(across all show() calls combined). Excess images are dropped."
        )
    if not budget_lines:
        budget_lines.append(
            "- Pass only the frames the next reasoning step needs to see; "
            "unrelated frames make the next step harder to read."
        )
    budget_text = "\n".join(budget_lines)

    var = "InputImages_1" if num_videos > 1 else "InputImages"

    return f"""## Visual Inspection — `show()` (PRIMARY)

`show(visual_input)` — display image(s) inline in the next feedback message so **you can see them yourself**.

| Argument | Type | Description |
|----------|------|-------------|
| `visual_input` | image, list of images, or `VisualFeedback` | What to display |

`show()` is your **primary way to see visual content**. Use it liberally:
- **Visual grounding**: `show({var}[idx])` to see annotated frames and identify objects yourself.
- **Intermediate results**: `show(recon.render_bev(masks=seg))` to inspect BEV, segmentation overlays, depth maps.
- **Matplotlib plots**: `plt.show()` is **auto-captured** — any figure is rendered and shown inline.
{budget_text}
- **NEVER pass large lists** like `show({var}[:30])`. Select only the most informative frames: `show({var}[0], {var}[15], {var}[31])`."""


def robust_computation_section() -> str:
    """Return guidance on robust statistics and physical-unit reasoning."""
    return """## Robust Computation Principles

- **Use `np.median()` over `np.mean()`** for all aggregations. Filter point clouds by `recon.confidence > threshold` before computing centroids.
- **Never trust a single frame.** Compare across multiple frames — consistent values are reliable, one-off values are noise.
- **Reason in metric units, not pixels.** Always convert to meters/degrees/m·s⁻¹ before judging significance (e.g., a 2px shift is meaningless without depth context).
- **Sanity-check magnitudes** (e.g., pedestrian ~1-2 m/s, car ~10-30 m/s, angular noise < 2°, real rotation > 5°). If results violate common sense, suspect bad inputs.
- **Print values before concluding.** When margins are small relative to measurement noise, acknowledge low confidence."""


def code_rules_section(num_videos: int = 1) -> str:
    """Return the code rules section (system prompt only).

    In multi-video mode the reserved names list is expanded to cover the
    per-video ``InputImages_<N>`` variables that the kernel actually
    binds; in single-video mode it stays unchanged so existing benchmarks
    produce byte-identical output.
    """
    if num_videos > 1:
        if num_videos <= 4:
            input_names = ", ".join(
                f"`InputImages_{i}`" for i in range(1, num_videos + 1)
            )
        else:
            input_names = (
                f"`InputImages_1`, `InputImages_2`, ..., `InputImages_{num_videos}`"
            )
        reserved = (
            f"`feedback`, `tools`, {input_names}, `Metadata`, `ReturnAnswer`, "
            f"`show`, `RefImages`"
        )
    else:
        reserved = (
            "`feedback`, `tools`, `InputImages`, `Metadata`, `ReturnAnswer`, "
            "`show`, `RefImages`"
        )
    return f"""## Code Rules

- **Pre-imported** (do NOT re-import): `numpy as np`, `math`, `collections`, `itertools`, `functools`, `matplotlib`, `matplotlib.pyplot as plt`, `scipy` (with `ndimage`, `spatial`, `signal`, `optimize`)
- **FORBIDDEN**: os, subprocess, sys, torch, open(), file I/O, exec(), eval()
- Use `print()` for debug output. Variables with `_` prefix are private.
- One logical step per response. Keep code concise.
- **NEVER** reassign built-in names: {reserved}."""
