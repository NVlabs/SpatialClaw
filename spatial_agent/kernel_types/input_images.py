"""InputImages constant injected into the Jupyter kernel."""

from typing import List, Optional

from PIL import Image

from spatial_agent.kernel_types.frame_image import FrameImage


class InputImages(list):
    """List of input images (video frames or static images).

    Subclasses ``list`` so it passes ``isinstance(obj, list)`` checks
    in downstream tools (e.g. Reconstruct) and can be used directly
    as ``tools.Reconstruct.Reconstruct(InputImages, ...)``.

    Each element is a ``FrameImage`` — a PIL-compatible wrapper that
    carries a ``.frame_index`` attribute.  This means ``InputImages[i]``
    returns a ``FrameImage`` and downstream tools can auto-extract frame
    indices without the caller passing them explicitly.

    Extra attribute:
        frame_indices: absolute video frame indices for each image.
    """

    def __init__(
        self,
        images: List,
        frame_indices: Optional[List[int]] = None,
        max_edge: Optional[int] = None,
        backend: str = "pi3",
    ):
        if not frame_indices:
            frame_indices = list(range(len(images)))
        wrapped = [
            img if isinstance(img, FrameImage)
            else FrameImage(img, idx, max_edge=max_edge, backend=backend)
            for img, idx in zip(images, frame_indices)
        ]
        super().__init__(wrapped)

    @property
    def frame_indices(self) -> List[int]:
        return [fi.frame_index for fi in self]

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            new = InputImages.__new__(InputImages)
            list.__init__(new, result)
            return new
        return result  # FrameImage for int key

    def __repr__(self) -> str:
        fi = self.frame_indices
        if len(fi) > 6:
            fi_str = f"[{fi[0]}, {fi[1]}, ..., {fi[-2]}, {fi[-1]}]"
        else:
            fi_str = str(fi)
        return f"InputImages({len(self)} images, frame_indices={fi_str})"
