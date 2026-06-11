"""Drawing utilities for image annotation."""

from typing import List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

from spatial_agent.tools.base import CPUTool

Color = Union[Tuple[int, int, int], str]


def _resolve_color(color: Color) -> Tuple[int, int, int]:
    """Convert a color name or RGB tuple to an (R, G, B) uint8 tuple.

    Named colors are resolved via matplotlib (CSS4 + base colors).
    """
    if isinstance(color, str):
        try:
            rgba = mcolors.to_rgba(color)
        except ValueError:
            raise ValueError(
                f"Unknown color name '{color}'. "
                f"Use any matplotlib/CSS4 color name (e.g. 'red', 'dodgerblue', 'salmon') "
                f"or an (R, G, B) tuple."
            )
        return (int(round(rgba[0] * 255)), int(round(rgba[1] * 255)), int(round(rgba[2] * 255)))
    if isinstance(color, (tuple, list)) and len(color) == 3:
        return (int(color[0]), int(color[1]), int(color[2]))
    raise ValueError(
        f"Color must be an (R, G, B) tuple or a name string, got {color!r}"
    )


def _to_numpy(image) -> np.ndarray:
    """Convert image to (H, W, 3) uint8 numpy array.

    Accepts numpy array, PIL Image, or FrameImage (InputImages[i]).
    """
    # FrameImage delegates .convert() to its inner PIL Image via __getattr__
    if hasattr(image, "convert") and callable(image.convert):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"`image` must be (H, W, 3), got shape {image.shape}."
            )
        return image
    raise TypeError(
        f"`image` must be a numpy array, PIL Image, or InputImages[i], "
        f"got {type(image).__name__}."
    )


def _is_single_coord(seq) -> bool:
    """True if seq looks like a single coordinate tuple (all elements are numbers)."""
    if isinstance(seq, np.ndarray):
        return seq.ndim == 1
    return len(seq) > 0 and isinstance(seq[0], (int, float, np.integer, np.floating))


def _broadcast_colors(
    colors: Optional[Union[Color, List[Color]]],
    n: int,
    default: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    """Broadcast colors to length n."""
    if colors is None:
        return [default] * n
    # Single color (string or tuple)
    if isinstance(colors, str):
        return [_resolve_color(colors)] * n
    if isinstance(colors, (tuple, list)):
        # Check if it's a single RGB tuple (3 ints) vs. a list of colors
        if len(colors) == 3 and isinstance(colors[0], (int, float, np.integer, np.floating)):
            return [_resolve_color(colors)] * n
        # List of colors
        resolved = [_resolve_color(c) for c in colors]
        if len(resolved) != n:
            raise ValueError(
                f"Got {len(resolved)} colors but {n} primitives. "
                f"Provide one color for all, or one per primitive."
            )
        return resolved
    return [_resolve_color(colors)] * n


class DrawUtils(CPUTool):
    """Image drawing utilities for annotation.

    All methods are static. Accept numpy (H, W, 3) or PIL Image input.
    Always return a PIL Image. The original is never modified.
    """

    TOOL_PROMPT_DESCRIPTION = """
### tools.Draw — Image Drawing Utilities (CPU)

All methods are **static** — call directly on `tools.Draw`.
All methods accept a **numpy array (H, W, 3)** or **PIL Image** and return a **PIL Image** (same size). The original is never modified.

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `draw_bbox` | `(image, bboxes, colors=None, thickness=None)` | `PIL.Image` | Draw rectangle outlines |
| `draw_line` | `(image, lines, colors=None, thickness=None)` | `PIL.Image` | Draw line segments |
| `draw_point` | `(image, points, colors=None, radius=None)` | `PIL.Image` | Draw filled circles |

**Input**: `image` — numpy `(H, W, 3)` uint8 array, PIL Image, or `InputImages[i]` directly.
**Colors**: RGB tuple `(255, 0, 0)` or any matplotlib/CSS4 color name string (e.g. `"red"`, `"dodgerblue"`, `"salmon"`, `"darkgreen"`).
**Single or batch**: Each method accepts a single primitive or a list of them.

```python
# Single bbox
annotated = tools.Draw.draw_bbox(InputImages[0], (x1, y1, x2, y2), colors="red")

# Multiple bboxes with per-element colors
annotated = tools.Draw.draw_bbox(InputImages[0], [(10,10,100,100), (200,50,400,300)], colors=["red", "green"])

# Points and lines (chain PIL Images)
annotated = tools.Draw.draw_point(annotated, [(50,50), (300,175)], colors="blue")
annotated = tools.Draw.draw_line(annotated, (0, 0, 100, 100), colors="yellow")
```
"""

    @staticmethod
    def draw_bbox(
        image: Union[np.ndarray, Image.Image],
        bboxes: Union[Sequence, List[Sequence]],
        colors: Optional[Union[Color, List[Color]]] = None,
        thickness: Optional[int] = None,
    ) -> Image.Image:
        """Draw bounding box rectangles on an image.

        Args:
            image: ``(H, W, 3)`` uint8 array or PIL Image.
            bboxes: Single ``(x1, y1, x2, y2)`` or list of them.
            colors: Single color or per-bbox list. Default: red.
            thickness: Line thickness in pixels. Default: adaptive to image size.

        Returns:
            Annotated PIL Image (same size).
        """
        image = _to_numpy(image)
        H, W = image.shape[:2]
        scale = min(H, W)
        if thickness is None:
            thickness = max(2, round(scale * 0.004))

        # Normalize to list of bboxes
        if _is_single_coord(bboxes):
            bbox_list = [bboxes]
        else:
            bbox_list = list(bboxes)

        color_list = _broadcast_colors(colors, len(bbox_list), default=(255, 0, 0))

        out = image.copy()
        for bbox, color in zip(bbox_list, color_list):
            if len(bbox) != 4:
                raise ValueError(
                    f"Each bbox must have 4 values (x1, y1, x2, y2), got {len(bbox)}."
                )
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        return Image.fromarray(out)

    @staticmethod
    def draw_line(
        image: Union[np.ndarray, Image.Image],
        lines: Union[Sequence, List[Sequence]],
        colors: Optional[Union[Color, List[Color]]] = None,
        thickness: Optional[int] = None,
    ) -> Image.Image:
        """Draw line segments on an image.

        Args:
            image: ``(H, W, 3)`` uint8 array or PIL Image.
            lines: Single ``(x1, y1, x2, y2)`` or list of them.
            colors: Single color or per-line list. Default: green.
            thickness: Line thickness in pixels. Default: adaptive to image size.

        Returns:
            Annotated PIL Image (same size).
        """
        image = _to_numpy(image)
        H, W = image.shape[:2]
        scale = min(H, W)
        if thickness is None:
            thickness = max(2, round(scale * 0.004))

        if _is_single_coord(lines):
            line_list = [lines]
        else:
            line_list = list(lines)

        color_list = _broadcast_colors(colors, len(line_list), default=(0, 255, 0))

        out = image.copy()
        for line, color in zip(line_list, color_list):
            if len(line) != 4:
                raise ValueError(
                    f"Each line must have 4 values (x1, y1, x2, y2), got {len(line)}."
                )
            x1, y1, x2, y2 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            cv2.line(out, (x1, y1), (x2, y2), color, thickness)
        return Image.fromarray(out)

    @staticmethod
    def draw_point(
        image: Union[np.ndarray, Image.Image],
        points: Union[Sequence, List[Sequence]],
        colors: Optional[Union[Color, List[Color]]] = None,
        radius: Optional[int] = None,
    ) -> Image.Image:
        """Draw filled circles (points) on an image.

        Args:
            image: ``(H, W, 3)`` uint8 array or PIL Image.
            points: Single ``(x, y)`` or list of them.
            colors: Single color or per-point list. Default: blue.
            radius: Circle radius in pixels. Default: adaptive to image size.

        Returns:
            Annotated PIL Image (same size).
        """
        image = _to_numpy(image)
        H, W = image.shape[:2]
        scale = min(H, W)
        if radius is None:
            radius = max(3, round(scale * 0.008))

        if _is_single_coord(points):
            point_list = [points]
        else:
            point_list = list(points)

        color_list = _broadcast_colors(colors, len(point_list), default=(0, 0, 255))

        out = image.copy()
        for pt, color in zip(point_list, color_list):
            if len(pt) != 2:
                raise ValueError(
                    f"Each point must have 2 values (x, y), got {len(pt)}."
                )
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(out, (x, y), radius, color, -1)
        return Image.fromarray(out)

    def __repr__(self) -> str:
        return "DrawUtils(static methods: draw_bbox, draw_line, draw_point)"
