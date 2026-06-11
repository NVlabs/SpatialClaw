"""SAM3 GPU tool: video segmentation and object existence checking.

Wraps SAM3Model on the GPU server and converts outputs to
spatial_agent per-frame numpy types.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from spatial_agent.kernel_types.per_frame_types import PerFrameMask
from spatial_agent.tools.base import GPUTool, ensure_image_list


def _normalize_points_per_object(points_per_object):
    """Normalize various LLM point formats to [[[x, y], ...], ...].

    Handles common LLM mistakes:
    - [x, y] (flat scalars) → [[[x, y]]]
    - [[x, y]] (missing object nesting) → [[[x, y]]]
    - [[[x], [y]]] (split coords) → [[[x, y]]]
    """
    if not isinstance(points_per_object, (list, tuple)) or len(points_per_object) == 0:
        return points_per_object

    # Case 1: flat [x, y] — two scalars at top level
    if (
        len(points_per_object) == 2
        and isinstance(points_per_object[0], (int, float))
        and isinstance(points_per_object[1], (int, float))
    ):
        result = [[[points_per_object[0], points_per_object[1]]]]
        print(f"[SAM3] Auto-wrapped flat [x, y] into [[[x, y]]]: {result}")
        return result

    # Case 2: [[x, y]] or [[x, y], [x2, y2]] — missing object nesting
    # Each element is [x, y] (a 2-element list of scalars)
    if all(
        isinstance(pt, (list, tuple))
        and len(pt) == 2
        and isinstance(pt[0], (int, float))
        and isinstance(pt[1], (int, float))
        for pt in points_per_object
    ):
        result = [points_per_object]
        print(f"[SAM3] Auto-wrapped [[x, y], ...] into [[[x, y], ...]]: {result}")
        return result

    # Case 3: [[[x], [y]]] — split coords (each "point" is a 1-element list)
    for obj_idx, obj_pts in enumerate(points_per_object):
        if not isinstance(obj_pts, (list, tuple)):
            continue
        if all(
            isinstance(pt, (list, tuple))
            and len(pt) == 1
            and isinstance(pt[0], (int, float))
            for pt in obj_pts
        ) and len(obj_pts) >= 2 and len(obj_pts) % 2 == 0:
            # Reshape: [[x], [y], [x2], [y2]] → [[x, y], [x2, y2]]
            flat = [pt[0] for pt in obj_pts]
            paired = [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]
            points_per_object = list(points_per_object)
            points_per_object[obj_idx] = paired
            print(f"[SAM3] Auto-reshaped split coords for object {obj_idx}: {paired}")

    return points_per_object


def _normalize_boxes(boxes):
    """Normalize box input: auto-wrap flat [x1,y1,x2,y2] into [[x1,y1,x2,y2]]."""
    if not isinstance(boxes, (list, tuple)) or len(boxes) == 0:
        return boxes

    # Flat [x1, y1, x2, y2] — four scalars at top level
    if (
        len(boxes) == 4
        and all(isinstance(v, (int, float)) for v in boxes)
    ):
        result = [list(boxes)]
        print(f"[SAM3] Auto-wrapped flat [x1,y1,x2,y2] into [[x1,y1,x2,y2]]: {result}")
        return result

    return boxes


SAM3_VIDEO_METHODS_PROMPT = """
**Video methods (track objects across frames):**

- **Max {sam3_max_video_frames} video frames.** Use `start_frame` / `end_frame` to select a temporal window. Raises error if range exceeds the limit.
- `start_frame` / `end_frame`: absolute video frame indices (inclusive/exclusive) selecting which portion of the video to process.
- `prompt_frame_idx`: local index within the extracted range (0 = first frame of the window). E.g., with `start_frame=100, end_frame=200`, `prompt_frame_idx=0` detects on video frame 100.
- `video_index`: which video to track in (1-indexed). Single-video samples use the default `video_index=1`. Multi-video samples must pass the index that matches the `InputImages_<N>` you intend to track in (e.g. `video_index=2` corresponds to `InputImages_2`).
- Output `PerFrameMask` uses **absolute** video frame indices.

4. `tools.SAM3.segment_video_by_text(prompts, labels=None, prompt_frame_idx=0, start_frame=None, end_frame=None, video_index=1)` → `PerFrameMask` (**SLOW**, ~15s)
   - Track objects across video using text descriptions.
   - `prompts`: one per object, e.g. `["red car", "walking person"]`.

5. `tools.SAM3.segment_video_by_points(points_per_object, point_labels_per_object, labels, prompt_frame_idx=0, start_frame=None, end_frame=None, video_index=1)` → `PerFrameMask` (**SLOW**, ~15s)
   - Track objects across video using point clicks on the prompt frame.
   - `points_per_object`: list of point lists, one per object, e.g. `[[[100, 200]], [[300, 150]]]` in **pixel** coordinates.
   - `point_labels_per_object`: parallel labels, e.g. `[[1], [1]]`.
   - `labels`: human-readable labels, e.g. `["red circle object", "green circle object"]`.
   - **Only use when tracking the same object across multiple frames.**

6. `tools.SAM3.segment_video_by_box(boxes, labels, prompt_frame_idx=0, start_frame=None, end_frame=None, video_index=1)` → `PerFrameMask` (**SLOW**, ~15s)
   - Track objects across video using bounding boxes on the prompt frame.
   - `boxes`: list of `[x1, y1, x2, y2]` boxes (**pixel** xyxy), one per object. Each box is tracked independently.
   - `labels`: human-readable labels for each box. Must match length of `boxes`.
"""


class SAM3Tool(GPUTool):
    """Client wrapper for the SAM3 GPU server.

    ``SAM3Model.segment_video`` on the server requires a video file path
    (str) for ``video_source``, not PIL images. ``_video_source`` stores the
    path and is set by ``ToolsModule`` at construction time.

    Usage::

        seg = tools.SAM3.segment_video(["car", "person"])
        exists = tools.SAM3.is_object_exist(InputImages[:5], "dog")
    """

    TOOL_ABLATION_PREFIX = "tool_sam3"

    TOOL_PROMPT_SECTIONS = OrderedDict([
        ("api", """
### tools.SAM3 - Segmentation (GPU)

**SAM3 supports text, point, and box prompts.** Text prompts find all matching instances without coordinate extraction. Point/box prompts target specific objects with precise coordinates.

**Workflow A: Text segmentation (simpler, finds all instances)**
```
seg = tools.SAM3.segment_image_by_text(image, "car")         # all detected cars
```

**Workflow B: VLM grounding → coordinate-based (targets specific instance)**
1. Ask the grounding VLM for object location in 0-1000 normalized scale (choose one):
   - **Center point**: `vlm.locate(image, "Give the (x, y) center coordinates in 0-1000 normalized scale for <object>. Reply with ONLY the numbers.")`
   - **Bounding box**: `vlm.locate(image, "Give the bounding box (x1, y1, x2, y2) in 0-1000 normalized scale for <object>. Reply with ONLY the numbers.")`
2. Read the coordinate numbers from the VLM response and use them in your code.
3. Convert 0-1000 coords to pixels: `tools.Geometry.normalized_to_pixel((x1, y1, x2, y2), W, H)`
4. Pass pixel coords to SAM3 point/box methods below.

**When to use text vs point/box:**
- Text: object is visually distinctive, or you want ALL instances ("segment every person")
- Point/box: you need a SPECIFIC instance among similar objects, or text returned empty masks
- Fallback: if text → empty mask, fall back to VLM grounding + point/box

**Image methods (always available):**

1. `tools.SAM3.segment_image_by_points(image, points, point_labels, label)` → `PerFrameMask` (**FAST**, ~0.5s)
   - Segment using pixel coordinate clicks.
   - `image`: a single PIL Image (e.g. `InputImages[0]`).
   - `points`: list of [x, y] **pixel** coordinates, e.g. `[[320, 240]]` for a single foreground click.
   - `point_labels`: `[1]` for foreground, `[0]` for background. Must match length of `points`.
   - Best for selecting specific objects or instances.

2. `tools.SAM3.segment_image_by_box(image, box, label)` → `PerFrameMask` (**FAST**, ~0.5s)
   - Segment using a bounding box in **pixel** coordinates.
   - `box`: `[x1, y1, x2, y2]` in pixels (xyxy format).
   - Best when the VLM provides bounding box coordinates.

3. `tools.SAM3.segment_image_by_text(image, prompt, label=None)` → `PerFrameMask` (**FAST**, ~0.5s)
   - Segment ALL instances matching the text description.
   - Returns multi-object mask: `seg.num_objects` gives count, `fi = seg.frame_indices[0]; seg.get_mask(frame=fi, object=i)` for each.
   - Best when: finding all instances of an object class, or object is semantically distinct.
{sam3_video_methods}
- `tools.SAM3.is_object_exist(images, object_name)` → dict (**FAST per image**)
   - Checks whether an object appears in each image and how many instances.
   - Returns `{{"exists": [bool, ...], "counts": [int, ...], "summary": str}}`

**VLM → pixel conversion** (required for point/box methods only):
Read the coordinate values from the VLM response, then convert 0-1000 normalized coords to pixels:
```
W, H = image.width, image.height
# Read the coordinate values from the VLM answer, then convert:
px, py = tools.Geometry.normalized_to_pixel((vlm_x, vlm_y), W, H)                    # center point
px1, py1, px2, py2 = tools.Geometry.normalized_to_pixel((x1, y1, x2, y2), W, H)      # bounding box
```

**PerFrameMask attributes** (exact access patterns — use **ABSOLUTE** frame indices):
```
seg.frame_indices   # list[int] — absolute frame indices (e.g. [15] for single image, [0,1,...,31] for video)
seg.labels          # list[str] — object label per mask channel
seg.num_frames      # int
seg.num_objects     # int

# CRITICAL: Use the ABSOLUTE frame index from seg.frame_indices, NOT 0:
fi = seg.frame_indices[0]                  # get the actual frame index
seg.get_mask(frame=fi, object=0)           # (H, W) bool — single object mask
seg.get_mask(frame=fi, object='car')       # same, by label
seg[fi]                                    # (N_obj, H, W) bool — all objects at frame
seg.get_centroid_3d(recon, frame=fi, object=0)  # (3,) mean 3D position, or None
seg.get_masked_points(recon, frame=fi, object=0)  # (K, 3) world points under mask
seg.visualize(fi)                          # VisualFeedback for visual inspection
# WRONG: seg.visualize(0) when frame_indices=[15] → KeyError!
```
"""),
        ("verify", """
**Verify segmentation (two steps)**:
1. **Programmatic check FIRST** — check mask area before calling VLM. An empty mask is an obvious failure:
```
for fi in seg.frame_indices:
    for i, label in enumerate(seg.labels):
        area = seg.get_mask(frame=fi, object=i).sum()
        print(f"  frame {{fi}}, {{label}}: {{area}} pixels")
        if area == 0:
            print(f"  WARNING: {{label}} mask is EMPTY — segmentation failed, retry with different prompt")
```
2. **Visual verification on non-empty masks** — {verify_visual}
If the mask is wrong, retry with a different prompt before proceeding.

**Tip**: Keep a reference to the original image so you can compare:
```
img = InputImages[5]  # keep a reference
seg = tools.SAM3.segment_image_by_text(img, "car")
fi = seg.frame_indices[0]
show([img, seg.visualize(fi)])  # compare side by side
```
"""),
        ("notes", """
**Resolution**: SAM3 masks and Reconstruct point maps are both at **input image resolution** — no resizing needed:
```
fi = seg.frame_indices[0]
centroid_3d = seg.get_centroid_3d(recon, frame=fi, object=0)
```

**Note**: `PerFrameMask` is returned by SAM3 methods — do NOT import it. Just use the returned object directly.
"""),
    ])

    def __init__(self, video_source: Optional[str] = None, input_images=None,
                 gpu_tool_max_retries: int = 3, sam3_max_video_frames: int = 200,
                 total_video_frames: int = 0, num_videos: int = 1,
                 input_images_list: Optional[List] = None,
                 video_sources_list: Optional[List[str]] = None):
        super().__init__(deployment_name="spatial_SAM3",
                         gpu_tool_max_retries=gpu_tool_max_retries)
        self._video_source = video_source
        self._input_images = input_images  # InputImages instance for frame lookup
        self._max_video_frames = sam3_max_video_frames
        self._total_video_frames = total_video_frames
        # Multi-video state. ``video_sources_list`` and ``input_images_list``
        # are 0-indexed lists with one entry per backing video; agent code
        # selects a video via the 1-indexed ``video_index`` kwarg on
        # segment_video_*. Single-video samples leave both as ``None``.
        self._num_videos = num_videos
        self._input_images_list = input_images_list
        self._video_sources_list = video_sources_list

    def _resolve_video(self, video_index: int):
        """Pick (video_source, input_images) for a given 1-indexed ``video_index``.

        - Single-video sample: returns ``(self._video_source, self._input_images)``
          regardless of ``video_index`` (default 1).
        - Multi-video sample: returns the video at ``video_index-1``. Raises a
          clear error if the index is out of range or the video has no source
          file (e.g. MMSI-Video frames-only multi-video samples).
        """
        if self._num_videos <= 1 or self._video_sources_list is None:
            return self._video_source, self._input_images
        n = self._num_videos
        if not isinstance(video_index, int) or video_index < 1 or video_index > n:
            raise ValueError(
                f"video_index={video_index!r} is out of range for a sample with "
                f"{n} videos. Pass an integer in [1, {n}] (1-indexed; "
                f"video_index=1 → InputImages_1)."
            )
        idx = video_index - 1
        vs = self._video_sources_list[idx] if idx < len(self._video_sources_list) else None
        ii = (
            self._input_images_list[idx]
            if self._input_images_list and idx < len(self._input_images_list)
            else None
        )
        if vs is None:
            raise RuntimeError(
                f"SAM3.segment_video_* requires a backing video file, but "
                f"video_index={video_index} has no source video (this sample "
                f"only carries pre-extracted frames). Use segment_image_* on "
                f"InputImages_{video_index}[i] frames instead."
            )
        return vs, ii

    def segment_image(
        self,
        image: Image.Image,
        prompts: List[str],
        frame_index: Optional[int] = None,
    ) -> PerFrameMask:
        """DISABLED: Use segment_image_by_text, segment_image_by_points, or segment_image_by_box instead."""
        raise RuntimeError(
            "SAM3.segment_image() is disabled. Use one of:\n"
            "  For text-based: tools.SAM3.segment_image_by_text(image, 'red car')\n"
            "  For point-based: tools.SAM3.segment_image_by_points(image, [[px, py]], [1], 'label')\n"
            "  For box-based: tools.SAM3.segment_image_by_box(image, [x1, y1, x2, y2], 'label')"
        )

    def segment_video(
        self,
        prompts: List[str],
        prompt_frame_idx: int = 0,
        frame_indices: Optional[List[int]] = None,
    ) -> PerFrameMask:
        """DISABLED: Use segment_video_by_text, segment_video_by_points, or segment_video_by_box instead."""
        raise RuntimeError(
            "SAM3.segment_video() is disabled. Use one of:\n"
            "  For text-based: tools.SAM3.segment_video_by_text(['red car', 'person'])\n"
            "  For point-based: tools.SAM3.segment_video_by_points([[[px, py]]], [[1]], ['label'])\n"
            "  For box-based: tools.SAM3.segment_video_by_box([[x1, y1, x2, y2]], ['label'])"
        )

    def is_object_exist(
        self,
        images: List[Image.Image],
        object_name: str,
    ) -> Dict[str, Any]:
        """Check if an object exists in each image.

        Args:
            images: List of PIL images.
            object_name: Text description of the object.

        Returns:
            Dict with keys: ``exists`` (list of bool), ``counts`` (list of int),
            ``summary`` (str).
        """
        images = ensure_image_list(images)
        if len(images) == 0:
            raise ValueError("SAM3.is_object_exist requires at least one image.")
        if not isinstance(object_name, str) or not object_name.strip():
            raise ValueError(
                f"`object_name` must be a non-empty string, got {object_name!r}."
            )

        exists_list = []
        counts_list = []

        # Unwrap FrameImage → PIL for serialization
        from spatial_agent.kernel_types.frame_image import FrameImage

        for img in images:
            plain_img = img.image if isinstance(img, FrameImage) else img
            try:
                raw = self._call_remote(
                    "detect",
                    image_source=plain_img,
                    prompt=object_name,
                )
                result = raw.result if hasattr(raw, "result") else raw
                if result is not None and hasattr(result, "masks"):
                    masks = self._to_numpy(result.masks)
                    n_instances = masks.shape[0] if masks.ndim >= 3 else 0
                    exists_list.append(n_instances > 0)
                    counts_list.append(n_instances)
                else:
                    exists_list.append(False)
                    counts_list.append(0)
            except Exception:
                exists_list.append(False)
                counts_list.append(0)

        n_found = sum(exists_list)
        total = len(images)
        avg_count = np.mean(counts_list) if n_found > 0 else 0
        summary = (
            f"Object '{object_name}' found in {n_found}/{total} images, "
            f"avg {avg_count:.1f} instances per image where found."
        )

        return {
            "exists": exists_list,
            "counts": counts_list,
            "summary": summary,
        }

    def segment_image_by_points(
        self,
        image: Image.Image,
        points: List[List[float]],
        point_labels: List[int],
        label: str = "object",
    ) -> PerFrameMask:
        """Segment an object in a single image using point prompts.

        Args:
            image: A single PIL image or FrameImage.
            points: Pixel coordinates [[x, y], ...].
            point_labels: 1=foreground, 0=background per point.
            label: Human-readable label for the segmented object.

        Returns:
            ``PerFrameMask`` with shape ``(1, 1, H, W)`` bool mask.
        """
        from spatial_agent.kernel_types.frame_image import FrameImage

        if not isinstance(image, (Image.Image, FrameImage)):
            raise TypeError(
                f"`image` must be a single PIL Image or FrameImage, got {type(image).__name__}."
            )
        if not isinstance(points, (list, tuple)) or len(points) == 0:
            raise ValueError("`points` must be a non-empty list of [x, y] coordinates.")
        # Auto-wrap flat [x, y] into [[x, y]] — common LLM mistake
        if (
            len(points) == 2
            and not isinstance(points[0], (list, tuple))
            and isinstance(points[0], (int, float))
        ):
            points = [points]
            print(f"[SAM3] Auto-wrapped flat [x, y] into [[x, y]]: {points}")
        if len(points) != len(point_labels):
            raise ValueError(
                f"`points` ({len(points)}) and `point_labels` ({len(point_labels)}) must have the same length. "
                f"Expected points=[[x1, y1], [x2, y2], ...] (list of coordinate pairs), "
                f"not [x, y] (flat pair). Got points={points!r}"
            )

        frame_index = getattr(image, "frame_index", 0)
        plain_image = image.image if isinstance(image, FrameImage) else image
        W, H = plain_image.size

        # Convert pixel coords to normalized 0-1
        norm_pts = [[x / W, y / H] for x, y in points]

        raw = self._call_remote(
            "detect",
            image_source=plain_image,
            points=norm_pts,
            point_labels=point_labels,
            obj_id=0,
        )
        result = raw.result if hasattr(raw, "result") else raw

        mask = None
        if result is not None and hasattr(result, "masks") and result.masks is not None:
            masks_np = self._to_numpy(result.masks)  # (N, H, W)
            if masks_np.ndim == 3 and masks_np.shape[0] > 0:
                mask = masks_np[0]

        if mask is None or int(mask.sum()) == 0:
            raise RuntimeError(
                f"SAM3.segment_image_by_points produced no mask for points={points} "
                f"on this image (frame_index={frame_index}). The detector either "
                f"found no object near these points, or its mask logits collapsed "
                f"to all background. Try one of:\n"
                f"  - a different frame from the same video,\n"
                f"  - point(s) more centered on the target object,\n"
                f"  - segment_image_by_text(image, '<descriptive label>') and pick "
                f"the mask with the largest IoU around the target region."
            )

        masks_stacked = mask[np.newaxis, np.newaxis]  # (1, 1, H, W)

        return PerFrameMask(
            masks=masks_stacked.astype(bool),
            labels=[label],
            object_ids=[0],
            frame_indices=[frame_index],
            frames={frame_index: plain_image},
        )

    def segment_image_by_box(
        self,
        image: Image.Image,
        box: List[float],
        label: str = "object",
    ) -> PerFrameMask:
        """Segment an object in a single image using a bounding box prompt.

        Args:
            image: A single PIL image or FrameImage.
            box: Pixel coordinates [x1, y1, x2, y2] (xyxy format).
            label: Human-readable label for the segmented object.

        Returns:
            ``PerFrameMask`` with shape ``(1, 1, H, W)`` bool mask.
        """
        from spatial_agent.kernel_types.frame_image import FrameImage

        if not isinstance(image, (Image.Image, FrameImage)):
            raise TypeError(
                f"`image` must be a single PIL Image or FrameImage, got {type(image).__name__}."
            )
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("`box` must be [x1, y1, x2, y2] in pixel coordinates.")

        frame_index = getattr(image, "frame_index", 0)
        plain_image = image.image if isinstance(image, FrameImage) else image
        W, H = plain_image.size

        x1, y1, x2, y2 = box
        # Convert pixel xyxy to normalized xywh
        norm_xywh = [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]

        raw = self._call_remote(
            "detect",
            image_source=plain_image,
            bounding_boxes=[norm_xywh],
            bounding_box_labels=[1],
        )
        result = raw.result if hasattr(raw, "result") else raw

        mask = None
        if result is not None and hasattr(result, "masks") and result.masks is not None:
            masks_np = self._to_numpy(result.masks)  # (N, H, W)
            if masks_np.ndim == 3 and masks_np.shape[0] > 0:
                mask = masks_np[0]

        if mask is None or int(mask.sum()) == 0:
            raise RuntimeError(
                f"SAM3.segment_image_by_box produced no mask for box={box} on this "
                f"image (frame_index={frame_index}). The detector either found no "
                f"object aligned with this box, or its mask logits collapsed to all "
                f"background. Try one of:\n"
                f"  - a different frame from the same video,\n"
                f"  - a slightly larger or repositioned box,\n"
                f"  - segment_image_by_text(image, '<descriptive label>') and pick "
                f"the mask with the largest IoU against your box."
            )

        masks_stacked = mask[np.newaxis, np.newaxis]  # (1, 1, H, W)

        return PerFrameMask(
            masks=masks_stacked.astype(bool),
            labels=[label],
            object_ids=[0],
            frame_indices=[frame_index],
            frames={frame_index: plain_image},
        )

    def segment_image_by_text(
        self,
        image: Image.Image,
        prompt: str,
        label: str = None,
        confidence_threshold: float = 0.3,
    ) -> PerFrameMask:
        """Segment ALL instances matching a text description in a single image.

        Args:
            image: A single PIL image or FrameImage.
            prompt: Text description of the object(s) to segment, e.g. "red car".
            label: Human-readable label. Defaults to ``prompt``.
            confidence_threshold: Minimum detection score to keep (default 0.3).

        Returns:
            ``PerFrameMask`` with shape ``(1, N_obj, H, W)`` bool masks for all
            detected instances above the confidence threshold.
        """
        from spatial_agent.kernel_types.frame_image import FrameImage

        if not isinstance(image, (Image.Image, FrameImage)):
            raise TypeError(
                f"`image` must be a single PIL Image or FrameImage, got {type(image).__name__}."
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"`prompt` must be a non-empty string, got {prompt!r}.")

        if label is None:
            label = prompt

        frame_index = getattr(image, "frame_index", 0)
        plain_image = image.image if isinstance(image, FrameImage) else image
        W, H = plain_image.size

        raw = self._call_remote(
            "detect",
            image_source=plain_image,
            prompt=prompt,
        )
        result = raw.result if hasattr(raw, "result") else raw

        masks_np = None
        if result is not None and hasattr(result, "masks") and result.masks is not None:
            masks_np = self._to_numpy(result.masks)  # (N, H, W)
            scores = self._to_numpy(result.scores) if hasattr(result, "scores") and result.scores is not None else None

            if masks_np.ndim == 3 and masks_np.shape[0] > 0:
                # Filter by confidence threshold
                if scores is not None and len(scores) == masks_np.shape[0]:
                    keep = scores >= confidence_threshold
                    masks_np = masks_np[keep]
            else:
                masks_np = None

        # All retained masks must have at least one positive pixel — otherwise
        # SAM3 saw "something" but produced empty mask logits.  Raise an
        # actionable error so the caller can switch strategy.
        if masks_np is None or masks_np.shape[0] == 0 or int(masks_np.reshape(masks_np.shape[0], -1).sum(axis=1).max()) == 0:
            raise RuntimeError(
                f"SAM3.segment_image_by_text(prompt={prompt!r}) produced no usable "
                f"mask on this image (frame_index={frame_index}, confidence_threshold="
                f"{confidence_threshold}). SAM3 either detected nothing matching the "
                f"prompt, or its mask logits collapsed to all background. Try one of:\n"
                f"  - a different frame from the same video (this frame may be in a "
                f"transient state),\n"
                f"  - a more specific prompt (e.g. an attribute + noun),\n"
                f"  - lowering confidence_threshold,\n"
                f"  - segment_image_by_box(image, [x1, y1, x2, y2]) with a known box "
                f"location."
            )

        n_obj = masks_np.shape[0]
        masks_stacked = masks_np[np.newaxis]  # (1, N_obj, H, W)
        labels = [label] if n_obj == 1 else [f"{label}_{i}" for i in range(n_obj)]
        object_ids = list(range(n_obj))

        return PerFrameMask(
            masks=masks_stacked.astype(bool),
            labels=labels,
            object_ids=object_ids,
            frame_indices=[frame_index],
            frames={frame_index: plain_image},
        )

    def segment_video_by_text(
        self,
        prompts: List[str],
        labels: List[str] = None,
        prompt_frame_idx: int = 0,
        frame_indices: Optional[List[int]] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        video_index: int = 1,
    ) -> PerFrameMask:
        """Track objects across video using text descriptions.

        Args:
            prompts: Text descriptions, one per object, e.g. ["red car", "walking person"].
            labels: Human-readable labels. Defaults to ``prompts``.
            prompt_frame_idx: Frame index to detect objects on (relative to the
                extracted range when start_frame/end_frame are used).
            frame_indices: Absolute frame indices to return masks for.
            start_frame: First video frame to process (inclusive). None = 0.
            end_frame: Last video frame to process (exclusive). None = end of video.
            video_index: 1-indexed video to track in (multi-video samples only;
                e.g. ``video_index=2`` → ``InputImages_2``). Default ``1``.

        Returns:
            ``PerFrameMask`` with shape ``(T, N_obj, H, W)`` bool masks.
        """
        video_source, ii = self._resolve_video(video_index)
        if video_source is None:
            raise RuntimeError(
                "SAM3.segment_video_by_text requires a video file (Metadata.is_video must be True). "
                "For image inputs, use segment_image_by_text instead:\n"
                "  seg = tools.SAM3.segment_image_by_text(InputImages[frame_idx], 'object_name')"
            )
        if not isinstance(prompts, (list, tuple)) or len(prompts) == 0:
            raise ValueError("`prompts` must be a non-empty list of strings.")
        for i, p in enumerate(prompts):
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"`prompts[{i}]` must be a non-empty string, got {p!r}.")

        if labels is None:
            labels = list(prompts)
        if len(labels) != len(prompts):
            raise ValueError(
                f"`labels` ({len(labels)}) and `prompts` ({len(prompts)}) must have the same length."
            )

        self._validate_frame_range(start_frame, end_frame)

        raw = self._call_remote(
            "segment_video",
            video_source=video_source,
            prompts=prompts,
            prompt_frame_idx=prompt_frame_idx,
            frame_indices=frame_indices,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        if hasattr(raw, "err") and raw.err:
            raise RuntimeError(f"SAM3 segmentation failed: {raw.err['msg']}")

        result = raw.result if hasattr(raw, "result") else raw
        return self._build_per_frame_mask(result, labels, frame_indices, input_images=ii)

    def segment_video_by_points(
        self,
        points_per_object: List[List[List[float]]],
        point_labels_per_object: List[List[int]],
        labels: List[str],
        prompt_frame_idx: int = 0,
        frame_indices: Optional[List[int]] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        video_index: int = 1,
    ) -> PerFrameMask:
        """Track objects across video using point prompts on the prompt frame.

        Args:
            points_per_object: Per-object pixel coordinates, e.g. [[[100, 200]], [[300, 150]]].
            point_labels_per_object: Per-object point labels (1=fg, 0=bg).
            labels: Human-readable label per object.
            prompt_frame_idx: Frame index to apply prompts on (relative to the
                extracted range when start_frame/end_frame are used).
            frame_indices: Absolute frame indices to return masks for.
            start_frame: First video frame to process (inclusive). None = 0.
            end_frame: Last video frame to process (exclusive). None = end of video.
            video_index: 1-indexed video to track in (multi-video samples only).

        Returns:
            ``PerFrameMask`` with shape ``(T, N_obj, H, W)`` bool masks.
        """
        video_source, ii = self._resolve_video(video_index)
        if video_source is None:
            raise RuntimeError(
                "SAM3.segment_video_by_points requires a video file (Metadata.is_video must be True). "
                "For image inputs, use segment_image_by_points instead:\n"
                "  seg = tools.SAM3.segment_image_by_points(InputImages[frame_idx], points, point_labels, 'label')"
            )
        points_per_object = _normalize_points_per_object(points_per_object)
        if len(points_per_object) != len(point_labels_per_object):
            raise ValueError(
                "`points_per_object` and `point_labels_per_object` must have the same length."
            )
        if len(points_per_object) != len(labels):
            raise ValueError(
                "`points_per_object` and `labels` must have the same length."
            )

        # Get image dimensions for normalization (use the resolved video's frames)
        if ii is not None and len(ii) > 0:
            W, H = ii[0].width, ii[0].height
        else:
            raise RuntimeError("Cannot determine image dimensions for coordinate normalization.")

        # Convert pixel coords to normalized 0-1
        norm_pts = [
            [[x / W, y / H] for x, y in pts]
            for pts in points_per_object
        ]
        obj_ids = list(range(len(labels)))

        self._validate_frame_range(start_frame, end_frame)

        raw = self._call_remote(
            "segment_video",
            video_source=video_source,
            points_per_object=norm_pts,
            point_labels_per_object=point_labels_per_object,
            obj_ids=obj_ids,
            prompt_frame_idx=prompt_frame_idx,
            frame_indices=frame_indices,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        if hasattr(raw, "err") and raw.err:
            raise RuntimeError(f"SAM3 segmentation failed: {raw.err['msg']}")

        result = raw.result if hasattr(raw, "result") else raw
        return self._build_per_frame_mask(result, labels, frame_indices, input_images=ii)

    def segment_video_by_box(
        self,
        boxes: List[List[float]],
        labels: List[str],
        prompt_frame_idx: int = 0,
        frame_indices: Optional[List[int]] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        video_index: int = 1,
    ) -> PerFrameMask:
        """Track objects across video using bounding box prompts.

        Args:
            boxes: Per-object pixel bounding boxes [x1, y1, x2, y2] (xyxy).
            labels: Human-readable label per object.
            prompt_frame_idx: Frame index to apply prompts on (relative to the
                extracted range when start_frame/end_frame are used).
            frame_indices: Absolute frame indices to return masks for.
            start_frame: First video frame to process (inclusive). None = 0.
            end_frame: Last video frame to process (exclusive). None = end of video.
            video_index: 1-indexed video to track in (multi-video samples only).

        Returns:
            ``PerFrameMask`` with shape ``(T, N_obj, H, W)`` bool masks.
        """
        video_source, ii = self._resolve_video(video_index)
        if video_source is None:
            raise RuntimeError(
                "SAM3.segment_video_by_box requires a video file (Metadata.is_video must be True). "
                "For image inputs, use segment_image_by_box instead:\n"
                "  seg = tools.SAM3.segment_image_by_box(InputImages[frame_idx], [x1, y1, x2, y2], 'label')"
            )
        boxes = _normalize_boxes(boxes)
        if len(boxes) != len(labels):
            raise ValueError("`boxes` and `labels` must have the same length.")

        # Get image dimensions for normalization (use the resolved video's frames)
        if ii is not None and len(ii) > 0:
            W, H = ii[0].width, ii[0].height
        else:
            raise RuntimeError("Cannot determine image dimensions for coordinate normalization.")

        # Convert pixel xyxy to normalized xywh
        norm_boxes = []
        for x1, y1, x2, y2 in boxes:
            norm_boxes.append([x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H])
        box_labels = [1] * len(boxes)

        self._validate_frame_range(start_frame, end_frame)

        raw = self._call_remote(
            "segment_video",
            video_source=video_source,
            boxes_per_object=norm_boxes,
            box_labels_per_object=box_labels,
            prompt_frame_idx=prompt_frame_idx,
            frame_indices=frame_indices,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        if hasattr(raw, "err") and raw.err:
            raise RuntimeError(f"SAM3 segmentation failed: {raw.err['msg']}")

        result = raw.result if hasattr(raw, "result") else raw
        return self._build_per_frame_mask(result, labels, frame_indices, input_images=ii)

    def _validate_frame_range(
        self,
        start_frame: Optional[int],
        end_frame: Optional[int],
    ) -> None:
        """Validate that the requested frame range fits within the max video frames limit.

        When start_frame/end_frame are omitted, counts actual video frames
        and raises a guiding error if the video exceeds the limit.
        """
        if start_frame is not None and end_frame is not None:
            n_frames = end_frame - start_frame
            if n_frames <= 0:
                raise ValueError(
                    f"end_frame ({end_frame}) must be greater than start_frame ({start_frame})."
                )
            if n_frames > self._max_video_frames:
                raise ValueError(
                    f"Requested frame range ({start_frame}–{end_frame}) is {n_frames} frames, "
                    f"which exceeds the maximum of {self._max_video_frames}. "
                    f"Use a narrower start_frame/end_frame window."
                )
        else:
            # When start/end are (partially) omitted, use known frame count
            total = self._total_video_frames
            if total > 0:
                s = start_frame or 0
                e = end_frame if end_frame is not None else total
                n_frames = e - s
                if n_frames > self._max_video_frames:
                    raise ValueError(
                        f"Video has {n_frames} frames (indices {s}–{e}), "
                        f"which exceeds the maximum of {self._max_video_frames}. "
                        f"You MUST specify start_frame and end_frame to select a "
                        f"window of at most {self._max_video_frames} frames. "
                        f"Example: start_frame=0, end_frame={self._max_video_frames}"
                    )

    def _build_per_frame_mask(
        self, result, labels: List[str], frame_indices: Optional[List[int]],
        input_images=None,
    ) -> PerFrameMask:
        """Shared post-processing for segment_video_by_*.

        ``input_images`` selects which video's frames to use for mask resize
        and the frames_dict (visualization). Multi-video callers pass the
        ``InputImages_<N>`` matching the ``video_index`` they tracked in;
        single-video callers can omit it (we fall back to ``self._input_images``).
        """
        ii = input_images if input_images is not None else self._input_images

        masks_np = self._to_numpy(result.masks)  # (T, N_detected, H_vid, W_vid)
        result_fi = (
            list(result.frame_indices)
            if hasattr(result, "frame_indices")
            else frame_indices
        )

        # Resize masks to the resolved video's resolution
        if ii is not None and len(ii) > 0:
            target_h, target_w = ii[0].height, ii[0].width
            H_mask, W_mask = masks_np.shape[2], masks_np.shape[3]
            if (H_mask, W_mask) != (target_h, target_w):
                from scipy.ndimage import zoom
                sy, sx = target_h / H_mask, target_w / W_mask
                masks_np = zoom(masks_np.astype(np.float32), (1, 1, sy, sx), order=0) > 0.5

        # Pad/truncate to match labels
        n_returned = masks_np.shape[1]
        n_labels = len(labels)
        if n_returned < n_labels:
            T, _, H, W = masks_np.shape
            padding = np.zeros((T, n_labels - n_returned, H, W), dtype=masks_np.dtype)
            masks_np = np.concatenate([masks_np, padding], axis=1)
        elif n_returned > n_labels:
            masks_np = masks_np[:, :n_labels]

        object_ids = list(range(n_labels))

        # Build frames dict from the resolved video's InputImages for visualization
        frames_dict = {}
        if ii is not None and result_fi is not None:
            fi_set = set(result_fi)
            input_fi = getattr(ii, "frame_indices", None)
            if input_fi is not None:
                for local_idx, abs_idx in enumerate(input_fi):
                    if abs_idx in fi_set and local_idx < len(ii):
                        frames_dict[abs_idx] = ii[local_idx]

        return PerFrameMask(
            masks=masks_np.astype(bool),
            labels=list(labels),
            object_ids=object_ids,
            frame_indices=result_fi,
            frames=frames_dict,
        )

    @staticmethod
    def _to_numpy(arr) -> np.ndarray:
        """Ensure input is a numpy array."""
        return np.asarray(arr)

    def __repr__(self) -> str:
        return "SAM3Tool(methods: segment_image_by_text, segment_image_by_points, segment_image_by_box, segment_video_by_text, segment_video_by_points, segment_video_by_box, is_object_exist)"
