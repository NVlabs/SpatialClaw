"""Shared point parsing and mask scoring for the pointing-suite benchmarks.

Conventions. Each arm sees the image after ITS harness
resize (no-tool arm: Pi3 aspect grid, ``resize_for_input_images``; agent arm: DA3 target aspects,
``resize_for_input_images_da3`` — selected by ``SPATIAL_AGENT_SHOWN_GRID``, see ``crosspointbench.shown_frame``), always
LANCZOS to a ≤ 768-px long edge. A predicted coordinate pair is read as:

    both values in [0, 1] (floats)      -> normalized coordinates of the SHOWN image          (the official instruction)
    otherwise (integers / > 1)          -> per-arm integer convention from SPATIAL_AGENT_POINT_COORD_FORMAT:
        relative_1000   0-1000 grid of the shown image   (no-tool arm; the model answers this way even when told 0-1)
        absolute_shown  pixels of the shown image        (agent arm; tool outputs live in that frame)
        absolute_original  pixels of the original image  (official assumption for Qwen in RoboRefer's scorer; reference)

Every point is then mapped to the ORIGINAL image frame (inverse of the crop + resize) before the mask lookup, so the
benchmark's own mask and rule apply unchanged. Points outside the original image count as misses (RefSpatial /
Where2Place rule); RoboSpatial-Home's official scorer clamps instead (see its loader).

Parsers (union of the official ones, most specific first): ``[(x, y), ...]`` tuples (RoboPoint / RoboRefer regex),
JSON ``{"x":..,"y":..}`` / ``point_2d`` / ``point`` lists (Qwen / Gemini style, incl. fenced blocks), Molmo
``<point x="..%" y="..%">`` XML (0-100 %), and ``\\boxed{...}`` wrappers (the content is parsed recursively).
"""

import json
import os
import re
from typing import List, Optional, Tuple

from spatial_agent.evals.crosspointbench import COORD_FORMATS, COORD_FORMAT_ENV, shown_grid, to_original  # noqa: F401 (shown_grid re-exported)

Point = Tuple[float, float]

_TUPLE_RE = re.compile(r"\(\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\)")
_BRACKET_PAIR_RE = re.compile(r"\[\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\]")
_XML_RE = re.compile(r'x\d*="(-?\d+\.?\d*)"\s+y\d*="(-?\d+\.?\d*)"')


def coord_format() -> str:
    fmt = os.environ.get(COORD_FORMAT_ENV, "absolute_original")
    if fmt not in COORD_FORMATS:
        raise ValueError(f"{COORD_FORMAT_ENV}={fmt!r} not in {COORD_FORMATS}")
    return fmt


def _json_points(text: str) -> List[Point]:
    pts: List[Point] = []
    blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        blocks = re.findall(r"(\[\s*\{.*?\}\s*\]|\{[^{}]*\})", text, re.DOTALL)
    for b in blocks:
        try:
            data = json.loads(re.sub(r"//.*?(?=\n|$)", "", b.strip(), flags=re.MULTILINE))
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if not isinstance(it, dict):
                continue
            for key in ("point_2d", "point"):
                v = it.get(key)
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        pts.append((float(v[0]), float(v[1])))
                    except (TypeError, ValueError):
                        pass
            if it.get("x") is not None and it.get("y") is not None:
                try:
                    pts.append((float(it["x"]), float(it["y"])))
                except (TypeError, ValueError):
                    pass
    return pts


def parse_points(text: str) -> Tuple[List[Point], str]:
    """Return (points, kind) with kind in {'tuple', 'json', 'xml', 'none'}; XML points are 0-100 % (converted to 0-1)."""
    if not text:
        return [], "none"
    s = str(text)
    boxed = re.findall(r"\\boxed\{(.*)\}", s, re.DOTALL)
    if boxed:
        pts, kind = parse_points(boxed[-1])
        if pts:
            return pts, kind
    tuples = _TUPLE_RE.findall(s)
    if tuples:
        return [(float(a), float(b)) for a, b in tuples], "tuple"
    js = _json_points(s)
    if js:
        return js, "json"
    xml = _XML_RE.findall(s)
    if xml:
        return [(float(a) / 100.0, float(b) / 100.0) for a, b in xml], "xml"
    pairs = _BRACKET_PAIR_RE.findall(s)
    if pairs:
        return [(float(a), float(b)) for a, b in pairs], "tuple"
    return [], "none"


def points_to_original(points: List[Point], fmt: str, width: int, height: int, grid: Optional[str] = None) -> List[Point]:
    """Map parsed points to pixels of the original image under the declared conventions (shown grid ``grid``; default env)."""
    from spatial_agent.evals.crosspointbench import shown_frame
    x0, y0, cw, ch, sw, sh = shown_frame(width, height, grid=grid)
    out: List[Point] = []
    for x, y in points:
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            # normalized of the shown image -> shown pixel -> original pixel
            px, py = x * sw, y * sh
            out.append((x0 + px * cw / sw, y0 + py * ch / sh))
        else:
            out.append(to_original(x, y, fmt, width, height, grid=grid))
    return out


def in_mask_fraction(points: List[Point], mask, threshold: int = 127) -> float:
    """RoboPoint / RoboRefer rule: fraction of points on a mask pixel; out-of-image points count 0; no points -> 0."""
    if not points:
        return 0.0
    h, w = mask.shape[:2]
    hits = 0
    for x, y in points:
        xi, yi = int(x), int(y)
        if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > threshold:
            hits += 1
    return hits / len(points)


def any_in_mask_clamped(points: List[Point], mask, width: int, height: int, limit: int = 2) -> bool:
    """RoboSpatial-Home rule: first ``limit`` points, nearest-pixel rounding with clamping to the image, any hit (> 0)."""
    h, w = mask.shape[:2]
    for x, y in points[:limit]:
        px = min(max(int(round(x)), 0), w - 1)
        py = min(max(int(round(y)), 0), h - 1)
        if int(mask[py, px]) > 0:
            return True
    return False


def extract_yes_no(text: str) -> Optional[str]:
    """'yes' / 'no' from a direct answer, a \\boxed{} wrapper or the last explicit answer; None if neither."""
    if not text:
        return None
    s = str(text).strip()
    boxed = re.findall(r"\\boxed\{(.*?)\}", s, re.DOTALL)
    if boxed:
        s = boxed[-1].strip()
    s = re.sub(r"[\*_`\"']+", " ", s).strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    ms = re.findall(r"(?:answer|final answer)\s*(?:is|:)?\s*\(?\s*(yes|no)\b", s)
    if ms:
        return ms[-1]
    tail = re.findall(r"\b(yes|no)\b[\.\!]?\s*$", s)
    if tail:
        return tail[-1]
    return None
