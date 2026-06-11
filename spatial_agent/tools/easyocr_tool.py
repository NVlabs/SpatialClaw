"""EasyOCR GPU tool: text extraction from images."""

from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spatial_agent.kernel_types.visual_feedback import VisualFeedback
from spatial_agent.tools.base import GPUTool


class EasyOCRTool(GPUTool):
    """Client wrapper for EasyOCR on the GPU server.

    Usage::

        result = tools.EasyOCR.ocr(image)

    """

    TOOL_PROMPT_DESCRIPTION = """
### tools.EasyOCR — Text Extraction (GPU)

`tools.EasyOCR.ocr(image)` → `dict`

Extracts text from a **single PIL image**.

**Returns** a dict with:
- `texts`: `list[str]` — detected text strings
- `boxes`: `list[[x1, y1, x2, y2]]` — bounding boxes
- `confidences`: `list[float]` — confidence scores
- `visualization`: `VisualFeedback` — annotated image with boxes

```python
result = tools.EasyOCR.ocr(InputImages[0])
print(result["texts"])  # ["EXIT", "Room 301"]
vis = result["visualization"]  # VisualFeedback for visual inspection
```
"""

    def ocr(self, image: Image.Image) -> Dict[str, Any]:
        """Extract text from an image.

        Args:
            image: PIL image or FrameImage.

        Returns:
            Dict with keys: ``texts``, ``boxes``, ``confidences``, ``visualization``.
        """
        # Input validation
        from spatial_agent.kernel_types.frame_image import FrameImage

        if isinstance(image, FrameImage):
            image = image.image
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"`image` must be a PIL Image, got {type(image).__name__}. "
                f"If you have a list, pass a single image: images[0]"
            )

        raw = self._call_remote("ocr", image_source=image)

        if hasattr(raw, "err") and raw.err:
            raise RuntimeError(f"EasyOCR failed: {raw.err['msg']}")

        result = raw.result if hasattr(raw, "result") else raw

        # Parse results
        texts = []
        boxes = []
        confidences = []

        if hasattr(result, "texts"):
            texts = list(result.texts)
            boxes = [list(b) for b in result.boxes] if hasattr(result, "boxes") else []
            confidences = list(result.confidences) if hasattr(result, "confidences") else []
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    box, text = item[0], item[1]
                    conf = item[2] if len(item) > 2 else 1.0
                    texts.append(str(text))
                    boxes.append(box)
                    confidences.append(float(conf))

        # Create visualization
        vis_img = image.copy()
        draw = ImageDraw.Draw(vis_img)
        for i, (box, text) in enumerate(zip(boxes, texts)):
            if isinstance(box, (list, np.ndarray)) and len(box) >= 4:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1 - 12), text, fill="red")

        visualization = VisualFeedback(
            image=vis_img,
            source="EasyOCR.ocr",
            description=f"OCR results: {len(texts)} text regions detected: {texts[:5]}",
        )

        return {
            "texts": texts,
            "boxes": boxes,
            "confidences": confidences,
            "visualization": visualization,
        }

    def __repr__(self) -> str:
        return "EasyOCRTool(method: ocr)"
