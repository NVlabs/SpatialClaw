"""Pi3-aligned image resize pipeline.

The same Pi3 training grid (k, m) drives every resize stage so that
the final Pi3 input lands exactly on (k*14, m*14) with zero aspect drift.

Stages (all using the same (k, m) computed from the *original* aspect):
  1. ``resize_for_input_images``: center-crop original to aspect k:m,
     then LANCZOS-resize so long edge ≈ ``max_long_edge`` while keeping
     dimensions on the (k, m) grid (= (k*s, m*s) for integer s).
  2. ``resize_for_pi3``: pure-ratio LANCZOS-resize InputImages to
     (k*14, m*14) - Pi3's training distribution.

(k, m) is chosen by :func:`pi3_training_grid`, which mirrors the
official ``load_images_as_tensor`` helper in
``Pi3/pi3/utils/basic.py`` (PIXEL_LIMIT = 255_000 ≈ 518² area).
"""

import math
from typing import Tuple

from PIL import Image

PI3_PIXEL_LIMIT = 255_000  # Pi3 training distribution (matches official helper)
PI3_PATCH_SIZE = 14


def pi3_training_grid(
    width: int, height: int, pixel_limit: int = PI3_PIXEL_LIMIT
) -> Tuple[int, int]:
    """Pick (k, m) integers approximating the input aspect under area cap.

    Pi3 expects inputs of shape (..., 3, m*14, k*14). Returns (k, m) such
    that k*m*196 <= pixel_limit and aspect k/m best matches width/height.
    """
    if width <= 0 or height <= 0:
        return 1, 1
    scale = math.sqrt(pixel_limit / (width * height))
    w_t, h_t = width * scale, height * scale
    k = max(1, round(w_t / PI3_PATCH_SIZE))
    m = max(1, round(h_t / PI3_PATCH_SIZE))
    while (k * PI3_PATCH_SIZE) * (m * PI3_PATCH_SIZE) > pixel_limit:
        if k == 1 and m == 1:
            break
        if k / max(m, 1) > w_t / max(h_t, 1e-9):
            k = max(1, k - 1)
        else:
            m = max(1, m - 1)
    return k, m


def crop_to_aspect(img: Image.Image, target_aspect: float) -> Image.Image:
    """Center-crop ``img`` so that its W/H equals ``target_aspect``."""
    w, h = img.size
    if h <= 0 or target_aspect <= 0:
        return img
    orig_aspect = w / h
    if abs(orig_aspect - target_aspect) < 1e-6:
        return img
    if orig_aspect > target_aspect:
        new_w = max(1, round(h * target_aspect))
        offset = (w - new_w) // 2
        return img.crop((offset, 0, offset + new_w, h))
    new_h = max(1, round(w / target_aspect))
    offset = (h - new_h) // 2
    return img.crop((0, offset, w, offset + new_h))


def resize_for_input_images(
    img: Image.Image,
    max_long_edge: int,
    pixel_limit: int = PI3_PIXEL_LIMIT,
) -> Image.Image:
    """Crop + LANCZOS-resize so the result is on Pi3's grid (k*s, m*s).

    Long edge ends up <= ``max_long_edge`` (typically within one (k,m) step).
    All InputImages and key frames go through this so Pi3's downstream
    resize is a pure 1/s scale.
    """
    w, h = img.size
    k, m = pi3_training_grid(w, h, pixel_limit)
    target_aspect = k / m
    cropped = crop_to_aspect(img, target_aspect)
    s = max(1, max_long_edge // max(k, m))
    target_w, target_h = k * s, m * s
    if cropped.size == (target_w, target_h):
        return cropped
    return cropped.resize((target_w, target_h), Image.LANCZOS)


def resize_for_pi3(
    img: Image.Image,
    pixel_limit: int = PI3_PIXEL_LIMIT,
) -> Image.Image:
    """LANCZOS-resize directly to (k*14, m*14) for Pi3 inference.

    Assumes ``img`` is already on a (k*s, m*s) grid (via
    :func:`resize_for_input_images`); falls back to a crop if not.
    """
    w, h = img.size
    k, m = pi3_training_grid(w, h, pixel_limit)
    target_w, target_h = k * PI3_PATCH_SIZE, m * PI3_PATCH_SIZE
    if (w, h) == (target_w, target_h):
        return img
    target_aspect = k / m
    if h > 0 and abs(w / h - target_aspect) > 1e-3:
        img = crop_to_aspect(img, target_aspect)
    return img.resize((target_w, target_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# DA3 (Depth Anything 3) grid
# ---------------------------------------------------------------------------

DA3_PATCH_SIZE = 14
DA3_LONG_EDGE = 768

# Curated set of model input shapes (W, H). All multiples of 14.
# Aspects: 1:1, 4:3 (small/large), 3:2 (small/large), 9:5, 16:9, 2:3 portrait.
DA3_TARGET_SHAPES: Tuple[Tuple[int, int], ...] = (
    (504, 504),
    (504, 378),
    (504, 336),
    (504, 280),
    (336, 504),
    (896, 504),
    (756, 504),
    (672, 504),
)


def da3_pick_target_shape(width: int, height: int) -> Tuple[int, int]:
    """Pick (W, H) from ``DA3_TARGET_SHAPES`` whose aspect best matches the input.

    Tie-break: among shapes with the same aspect, pick the largest by area
    (max-quality default — e.g. 4:3 → 672×504 over 504×378).
    """
    if width <= 0 or height <= 0:
        return DA3_TARGET_SHAPES[0]
    aspect = width / height
    return min(
        DA3_TARGET_SHAPES,
        key=lambda s: (abs(s[0] / s[1] - aspect), -s[0] * s[1]),
    )


def da3_training_grid(width: int, height: int) -> Tuple[int, int]:
    """Reduce DA3's chosen target shape to its smallest (k, m) form.

    For each unique aspect, returns the GCD-reduced ``(W//g, H//g)``
    (e.g. 504×504 → (1, 1); 672×504 → (4, 3); 896×504 → (16, 9)).
    Used by :func:`resize_for_input_images_da3` to lay images out on a
    (k*s, m*s) grid keeping integer scale.
    """
    target_w, target_h = da3_pick_target_shape(width, height)
    g = math.gcd(target_w, target_h)
    return target_w // g, target_h // g


def resize_for_input_images_da3(
    img: Image.Image,
    max_long_edge: int = DA3_LONG_EDGE,
) -> Image.Image:
    """Crop + LANCZOS-resize so the result is on DA3's grid (k*s, m*s).

    Pipeline parallels :func:`resize_for_input_images` but constrains the
    aspect ratio to DA3's target set. Long edge ends up ≤ ``max_long_edge``.
    """
    w, h = img.size
    target_w, target_h = da3_pick_target_shape(w, h)
    target_aspect = target_w / target_h
    cropped = crop_to_aspect(img, target_aspect)
    g = math.gcd(target_w, target_h)
    k, m = target_w // g, target_h // g
    s = max(1, max_long_edge // max(k, m))
    out_w, out_h = k * s, m * s
    if cropped.size == (out_w, out_h):
        return cropped
    return cropped.resize((out_w, out_h), Image.LANCZOS)


def resize_for_da3(img: Image.Image) -> Image.Image:
    """LANCZOS-resize directly to the closest DA3 target shape (model input).

    Picks the shape via :func:`da3_pick_target_shape`, center-crops to that
    aspect if needed, then resizes to (target_w, target_h).
    """
    w, h = img.size
    target_w, target_h = da3_pick_target_shape(w, h)
    target_aspect = target_w / target_h
    if h > 0 and abs(w / h - target_aspect) > 1e-3:
        img = crop_to_aspect(img, target_aspect)
    if img.size == (target_w, target_h):
        return img
    return img.resize((target_w, target_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Backend dispatcher
# ---------------------------------------------------------------------------

def resize_for_input_images_for_backend(
    img: Image.Image,
    max_long_edge: int,
    backend: str = "pi3",
) -> Image.Image:
    """Dispatch InputImages preprocessing to the active reconstruct backend.

    ``"da3"`` snaps to DA3's 6-aspect grid (504/672/756/896 × 504 family).
    ``"pi3"`` and ``"mapanything"`` fall through to Pi3's flexible grid
    (MapAnything does its own ``fixed_mapping`` resize internally, so the
    upstream grid choice doesn't matter for it).
    """
    if backend == "da3":
        return resize_for_input_images_da3(img, max_long_edge)
    return resize_for_input_images(img, max_long_edge)
