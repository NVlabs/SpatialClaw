"""Mask utility module.  All operations on numpy arrays."""

from typing import List, Optional, Tuple

import numpy as np

from spatial_agent.tools.base import CPUTool


def _check_mask_2d(mask: np.ndarray, name: str = "mask") -> None:
    """Validate that mask is a 2D boolean-like numpy array."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(
            f"`{name}` must be a numpy array, got {type(mask).__name__}. "
            f"If you have a PerFrameMask, index it: seg.masks[frame_idx, obj_idx]"
        )
    if mask.ndim != 2:
        raise ValueError(
            f"`{name}` must be 2D (H, W), got shape {mask.shape}. "
            f"If shape is (N, H, W), index the first dim: mask[0] or mask[i]"
        )


class MaskUtils(CPUTool):
    """Mask analysis utilities.

    All methods are static and operate on ``np.ndarray`` boolean masks.
    """

    TOOL_PROMPT_DESCRIPTION = """
### tools.Mask — Mask Utilities (CPU)

All methods are **static** — call directly on `tools.Mask`.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `centroid` | `(mask)` | `(cx, cy)` or `(nan, nan)` | Center of mass of a 2D boolean mask |
| `centroids` | `(masks)` | `list[(cx, cy)]` | Batch centroids for `(N, H, W)` array |
| `area` | `(mask)` | `int` | Number of True pixels |
| `bounding_box` | `(mask)` | `(x1, y1, x2, y2)` or `None` | Tight bounding box |
| `iou` | `(mask_a, mask_b)` | `float` | Intersection over union |
| `intersection` | `(mask_a, mask_b)` | `np.ndarray` | Element-wise AND |
| `mask_to_bbox` | `(mask)` | `np.array([x1,y1,x2,y2])` or `None` | Bbox as array |

**Input**: All `mask` args must be **2D numpy arrays** `(H, W)`.
Use `seg.get_mask(frame, object)` to get a 2D mask from a `PerFrameMask`.

```python
# Correct usage (absolute frame index):
fi = seg.frame_indices[0]                        # absolute frame index
mask = seg.get_mask(frame=fi, object=0)          # (H, W) bool
c = tools.Mask.centroid(mask)                    # (cx, cy) tuple — (nan, nan) if empty
a = tools.Mask.area(mask)                        # int
bb = tools.Mask.bounding_box(mask)               # (x1, y1, x2, y2) tuple

# WRONG — seg.masks is 4D, not 2D:
# c = tools.Mask.centroid(seg.masks)  # ERROR!
```
"""

    @staticmethod
    def centroid(mask: np.ndarray) -> Tuple[float, float]:
        """Compute ``(cx, cy)`` centroid of a 2D boolean mask.

        Returns ``(nan, nan)`` if the mask is empty (safe for arithmetic and formatting).
        """
        _check_mask_2d(mask, "mask")
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return (float('nan'), float('nan'))
        return (float(np.median(xs)), float(np.median(ys)))

    @staticmethod
    def centroids(masks: np.ndarray) -> List[Optional[Tuple[float, float]]]:
        """Compute centroids for a batch of masks ``(N, H, W)``."""
        if not isinstance(masks, np.ndarray):
            raise TypeError(
                f"`masks` must be a numpy array, got {type(masks).__name__}."
            )
        if masks.ndim != 3:
            raise ValueError(
                f"`masks` must be 3D (N, H, W), got shape {masks.shape}. "
                f"If shape is (N, N_obj, H, W), select one object: masks[:, obj_idx]"
            )
        return [MaskUtils.centroid(masks[i]) for i in range(masks.shape[0])]

    @staticmethod
    def area(mask: np.ndarray) -> int:
        """Number of True pixels in the mask."""
        _check_mask_2d(mask, "mask")
        return int(np.sum(mask))

    @staticmethod
    def bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return ``(x1, y1, x2, y2)`` bounding box, or None if empty."""
        _check_mask_2d(mask, "mask")
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        if len(xs) > 100:
            # Use percentiles to ignore outlier pixels on large masks
            x1, x2 = np.percentile(xs, [1, 99])
            y1, y2 = np.percentile(ys, [1, 99])
            return (int(np.floor(x1)), int(np.floor(y1)),
                    int(np.ceil(x2)), int(np.ceil(y2)))
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    @staticmethod
    def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Intersection over Union between two boolean masks."""
        _check_mask_2d(mask_a, "mask_a")
        _check_mask_2d(mask_b, "mask_b")
        if mask_a.shape != mask_b.shape:
            raise ValueError(
                f"mask_a {mask_a.shape} and mask_b {mask_b.shape} must have the same shape."
            )
        intersection = np.sum(mask_a & mask_b)
        union = np.sum(mask_a | mask_b)
        if union == 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def intersection(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
        """Element-wise AND of two masks."""
        _check_mask_2d(mask_a, "mask_a")
        _check_mask_2d(mask_b, "mask_b")
        if mask_a.shape != mask_b.shape:
            raise ValueError(
                f"mask_a {mask_a.shape} and mask_b {mask_b.shape} must have the same shape."
            )
        return mask_a & mask_b

    @staticmethod
    def mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
        """Convert mask to xyxy bounding box as ``np.ndarray([x1,y1,x2,y2])``."""
        bb = MaskUtils.bounding_box(mask)
        if bb is None:
            return None
        return np.array(bb, dtype=np.float64)

    def __repr__(self) -> str:
        return "MaskUtils(static methods: centroid, centroids, area, bounding_box, iou, intersection, mask_to_bbox)"
