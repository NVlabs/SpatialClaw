"""FeedbackModule: provides ``feedback.show()`` and ``feedback.log_visual()``.

VLM querying lives in ``vlm_module.VLMModule`` (injected as ``vlm``).

Usage in the Jupyter kernel::

    feedback.show(seg.visualize(fi))
    feedback.log_visual(some_visual)
"""

import os
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from spatial_agent.kernel_types.visual_feedback import VisualFeedback


class FeedbackModule:
    """Injected into the kernel as ``feedback``.

    Provides ``show()`` (display image(s) inline in the next feedback) and
    ``log_visual()`` (save an image for logging without VLM call). VLM
    queries are handled by the separate ``vlm`` module.
    """

    def __init__(
        self,
        session_dir: str,
        enable_sighted_feedback: bool = True,
        session_id: str = "",
    ):
        self._session_dir = session_dir
        self._enable_sighted_feedback = enable_sighted_feedback
        self._session_id = session_id
        self._queries: List[Dict[str, Any]] = []
        self._query_counter = 0
        self._show_items: List[Dict[str, Any]] = []
        self._show_counter = 0

    # ------------------------------------------------------------------
    # Public API (called from kernel code)
    # ------------------------------------------------------------------

    def log_visual(
        self,
        visual_input: Union[VisualFeedback, Image.Image],
        source: str = "",
    ) -> None:
        """Save a visual to the vlm_queries directory for logging (no VLM call).

        Args:
            visual_input: VisualFeedback or PIL.Image to log.
            source: Human-readable label for the logged image.
        """
        self._query_counter += 1
        query_id = f"vlm_q_{self._query_counter:04d}"

        images = self._resolve_images(visual_input)
        self._save_query_images(query_id, images)

        description = ""
        if isinstance(visual_input, VisualFeedback):
            description = visual_input.description or ""
            source = source or visual_input.source or ""

        self._queries.append(
            {
                "query_id": query_id,
                "question": f"[auto-logged] {source}",
                "answer": description,
                "num_images": len(images),
                "source": source,
            }
        )

    def show(
        self,
        *args,
        label: str = "",
    ) -> None:
        """Show image(s) inline in the next step's feedback.

        When sighted feedback is enabled, images are saved to disk and a
        structured marker is printed to stdout.  ``execute_node`` reads the
        marker and ``feedback_node`` embeds the images into the HumanMessage.

        Accepts variadic args for convenience:
            ``show(img)`` — single image
            ``show(img1, img2, img3)`` — multiple images
            ``show([img1, img2])`` — list of images
            ``show(img1, img2, label="comparison")`` — with keyword label

        Args:
            *args: VisualFeedback, PIL.Image, FrameImage, or list of these.
            label: Optional label.  If empty, the label is auto-extracted
                from the call-site source code by ``execute_node``.
        """
        if not args:
            print("[show] No images provided.")
            return

        # Resolve variadic args into a single visual_input
        if len(args) == 1:
            visual_input = args[0]
        else:
            visual_input = list(args)

        # Type-check early; _to_pil() handles actual conversion (incl. uint8 ndarray)
        from spatial_agent.kernel_types.frame_image import FrameImage
        import numpy as np

        _ACCEPTED = (Image.Image, VisualFeedback, FrameImage)

        def _is_showable(obj):
            return isinstance(obj, _ACCEPTED) or (
                isinstance(obj, np.ndarray) and obj.dtype == np.uint8
            )

        def _is_sequence(obj):
            """True for lists and iterables, but NOT for single images/arrays/strings."""
            if isinstance(obj, (str, np.ndarray, *_ACCEPTED)):
                return False
            return isinstance(obj, list) or hasattr(obj, "__iter__")

        if _is_sequence(visual_input):
            for item in visual_input:
                if not _is_showable(item):
                    raise TypeError(
                        f"show() got {type(item).__name__}; expected PIL.Image, "
                        f"VisualFeedback, FrameImage, or uint8 numpy array."
                    )
        elif not _is_showable(visual_input):
            raise TypeError(
                f"show() got {type(visual_input).__name__}; expected PIL.Image, "
                f"VisualFeedback, FrameImage, or uint8 numpy array."
            )

        self._show_counter += 1
        show_id = f"show_{self._show_counter:04d}"

        images = self._resolve_images(visual_input)

        # Save images to disk
        img_dir = os.path.join(self._session_dir, "show_images")
        os.makedirs(img_dir, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            path = os.path.join(img_dir, f"{show_id}_img_{i}.png")
            img.save(path)
            paths.append(path)

        import json as _json

        marker = _json.dumps({
            "show_id": show_id,
            "label": label,
            "num_images": len(images),
            "paths": paths,
        })
        print(f"[SHOW:{marker}]")

        self._show_items.append({
            "show_id": show_id,
            "label": label,
            "paths": paths,
        })

    def get_and_clear_show_images(self) -> List[Dict[str, Any]]:
        """Return all show items from the current step and reset the list."""
        items = self._show_items.copy()
        self._show_items.clear()
        return items

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_images(
        self, visual_input: Union[VisualFeedback, Image.Image, List]
    ) -> List[Image.Image]:
        import numpy as np
        # Handle InputImages objects and other iterables (but not strings/arrays)
        if isinstance(visual_input, list) or (
            hasattr(visual_input, "__iter__")
            and not isinstance(visual_input, (str, np.ndarray, Image.Image, VisualFeedback))
        ):
            return [self._to_pil(v) for v in visual_input]
        return [self._to_pil(visual_input)]

    @staticmethod
    def _to_pil(obj: Any) -> Image.Image:
        from spatial_agent.kernel_types.frame_image import FrameImage
        import numpy as np

        if isinstance(obj, FrameImage):
            return obj.image
        if isinstance(obj, VisualFeedback):
            return obj.image
        if isinstance(obj, Image.Image):
            return obj
        if isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
            return Image.fromarray(obj)
        raise TypeError(
            f"Cannot convert {type(obj).__name__} to PIL.Image. "
            f"Accepted: FrameImage, VisualFeedback, PIL.Image, uint8 ndarray."
        )

    def _save_query_images(self, query_id: str, images: List[Image.Image]) -> None:
        img_dir = os.path.join(self._session_dir, "vlm_queries")
        os.makedirs(img_dir, exist_ok=True)
        for i, img in enumerate(images):
            path = os.path.join(img_dir, f"{query_id}_img_{i}.png")
            img.save(path)

    # ------------------------------------------------------------------
    # Called by feedback_node to retrieve this step's queries
    # ------------------------------------------------------------------

    def get_and_clear_queries(self) -> List[Dict[str, Any]]:
        """Return logged visuals from log_visual() in the current step and reset."""
        queries = self._queries.copy()
        self._queries.clear()
        return queries
