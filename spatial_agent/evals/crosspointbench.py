"""CrossPoint-Bench (Wang et al., arXiv 2512.04686) data loader and scorer.

1,000 items on ScanNet / ScanNet++ indoor scenes (images pre-resized to 896×644 or 924×616):

    Fine-grained Grounding     161  single image; point output scored by in-mask hit
    Visibility Reasoning       220  two views; 2-way MCQ (A = Yes / B = No) embedded in the question
    Correspondence-Judgement   156  two views; 3-way MCQ over coloured candidate dots (A/B/C)
    Correspondence-Pointing    463  two views; (x, y) in image 2 of the point marked red in image 1, in-mask hit

jsonl fields: idx, type, level (object | part), images (1–2 paths relative to ``image/``), question,
answer (option letter, or a base64 PNG mask at the resolution of the shipped target image).

Official protocol (github.com/WangYipu2002/CrossPoint, eval/): the question is used verbatim; pointing
questions get the suffix " Output the point coordinates in JSON format."; MCQ letters are regex-extracted
and exact-matched; a point is correct iff the pixel (int x, int y) lies inside the mask (value > 128) of the
TARGET image (the last image of the item) after conversion to absolute pixels (``--coord_format``
absolute | relative_1000 | relative_1); unparsable = wrong; Overall = mean over all 1,000 items, plus
per type and per level.

Coordinate frame: each arm sees the image after ITS harness resize — the
no-tool arm ``resize_for_input_images`` (centre-crop to the Pi3 aspect grid, LANCZOS to a ≤ 768-px long edge,
e.g. 640×480 → 738×558), the agent arm ``resize_for_input_images_for_backend(..., config.reconstruct_backend)``
(default ``da3``: centre-crop to DA3's target aspects, e.g. 640×480 → 768×576, 480×640 → crop 427×640 → 512×768).
Which grid the arm showed is given by ``SPATIAL_AGENT_SHOWN_GRID`` (``pi3`` | ``da3``; the entrypoints set it from
their own pipeline). The predicted point is interpreted in that SHOWN frame under the per-arm convention
given by ``SPATIAL_AGENT_POINT_COORD_FORMAT``:
    relative_1000      x, y are on a 0–1000 grid of the shown image   (no-tool arm)
    absolute_shown     x, y are pixels of the shown image             (agent arm; tool outputs live in that frame)
    absolute_original  x, y are pixels of the original shipped image  (official default, for reference)
and mapped back to the original frame (inverse of the arm's crop + resize) before the mask lookup.

Reference: https://huggingface.co/datasets/WangYipu2002/CrossPoint-Bench
"""

import base64
import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.cvbench2024 import _extract_letter
from spatial_agent.evals.scoring import get_prediction, mean_or_zero, write_json, write_results_summary

JSONL = "CrossPoint-Bench.jsonl"
IMAGE_DIR = "image"
POINT_TYPES = ("Fine-grained Grounding", "Correspondence-Pointing")
MCQ_TYPES = ("Visibility Reasoning", "Correspondence-Judgement")
TYPES = ["Fine-grained Grounding", "Visibility Reasoning", "Correspondence-Judgement", "Correspondence-Pointing"]
LEVELS = ["object", "part"]
OFFICIAL_POINT_SUFFIX = " Output the point coordinates in JSON format."
COORD_FORMAT_ENV = "SPATIAL_AGENT_POINT_COORD_FORMAT"
COORD_FORMATS = ("relative_1000", "absolute_shown", "absolute_original")
MASK_THRESHOLD = 128


SHOWN_GRID_ENV = "SPATIAL_AGENT_SHOWN_GRID"
SHOWN_GRIDS = ("pi3", "da3")


def shown_grid() -> str:
    """Which InputImages grid the arm that produced the answer showed.

    ``pi3``: ``resize_for_input_images`` (the no-tool arm, ``cot_baseline.py``); ``da3``: ``resize_for_input_images_da3``
    (the agent arm whenever ``config.reconstruct_backend == "da3"``, the default). Read from ``SPATIAL_AGENT_SHOWN_GRID``;
    ``run.py`` / ``cot_baseline.py`` set it from their own pipeline, campaign drivers set it per arm. Default ``pi3``.
    """
    g = os.environ.get(SHOWN_GRID_ENV, "pi3").strip().lower()
    if g not in SHOWN_GRIDS:
        raise ValueError(f"{SHOWN_GRID_ENV}={g!r} not in {SHOWN_GRIDS}")
    return g


def shown_frame(width: int, height: int, max_long_edge: int = 768, grid: Optional[str] = None) -> Tuple[int, int, int, int, int, int]:
    """(crop_x0, crop_y0, crop_w, crop_h, shown_w, shown_h) produced by the arm's InputImages resize.

    ``grid='pi3'`` mirrors ``resize_for_input_images`` (Pi3 aspect grid), ``grid='da3'`` mirrors
    ``resize_for_input_images_da3`` (DA3 target shapes, GCD-reduced to (k, m)); both reproduce ``crop_to_aspect``'s
    centre crop and the ``k*s, m*s`` output size exactly (asserted offline against the real resize for every
    image size). ``grid=None`` reads ``SPATIAL_AGENT_SHOWN_GRID``.
    """
    import math
    from spatial_agent.gpu_models.image_resize import da3_pick_target_shape, pi3_training_grid
    g = (grid or shown_grid()).strip().lower()
    if g == "da3":
        tw, th = da3_pick_target_shape(width, height)
        gcd = math.gcd(tw, th)
        k, m = tw // gcd, th // gcd
    elif g == "pi3":
        k, m = pi3_training_grid(width, height)
    else:
        raise ValueError(f"unknown shown grid {g!r}")
    target_aspect = k / m
    cw, ch, x0, y0 = width, height, 0, 0
    if abs(width / height - target_aspect) >= 1e-6:
        if width / height > target_aspect:
            cw = max(1, round(height * target_aspect)); x0 = (width - cw) // 2
        else:
            ch = max(1, round(width / target_aspect)); y0 = (height - ch) // 2
    s = max(1, max_long_edge // max(k, m))
    return x0, y0, cw, ch, k * s, m * s


def to_original(x: float, y: float, fmt: str, width: int, height: int, grid: Optional[str] = None) -> Tuple[float, float]:
    """Map a predicted point under convention ``fmt`` (shown on grid ``grid``, default: env) to original-image pixels."""
    if fmt == "absolute_original":
        return x, y
    x0, y0, cw, ch, sw, sh = shown_frame(width, height, grid=grid)
    if fmt == "relative_1000":
        x, y = x / 1000.0 * sw, y / 1000.0 * sh
    elif fmt != "absolute_shown":
        raise ValueError(f"unknown coordinate format {fmt!r}")
    return x0 + x * cw / sw, y0 + y * ch / sh


# ── answer extraction (port of eval/extract_answers.py, same order of preference) ─────────────

def _coords_from_data(data: Any) -> Optional[Tuple[float, float]]:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if isinstance(data, dict):
        for key in ("point_2d", "point"):
            v = data.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                return float(v[0]), float(v[1])
        if data.get("x") is not None and data.get("y") is not None:
            try:
                return float(data["x"]), float(data["y"])
            except (TypeError, ValueError):
                pass
        for nested in ("position", "coordinates"):
            d = data.get(nested)
            if isinstance(d, dict) and d.get("x") is not None and d.get("y") is not None:
                return float(d["x"]), float(d["y"])
        v = data.get("bbox_2d")
        if isinstance(v, (list, tuple)) and len(v) >= 4:  # degenerate: use the box centre
            return (float(v[0]) + float(v[2])) / 2, (float(v[1]) + float(v[3])) / 2
    return None


def extract_point(text: str) -> Optional[Tuple[float, float]]:
    """Point from the model text: fenced / bare JSON first (official order), then text patterns."""
    if not text:
        return None
    s = str(text)
    boxed = re.findall(r"\\boxed\{(.*)\}", s, re.DOTALL)
    if boxed:
        p = extract_point(boxed[-1])
        if p:
            return p
    blocks = re.findall(r"```json\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
    if not blocks:
        for pat in (r"\{[^{}]*\"point_2d\"[^{}]*\}", r"\{[^{}]*\"point\"[^{}]*\}",
                    r"\{[^{}]*\"(?:x|position|coordinates|bbox_2d)\"[^{}]*\}", r"\[\s*\{[^\[\]]*\}\s*\]"):
            blocks.extend(re.findall(pat, s, re.DOTALL))
    for b in blocks:
        b = re.sub(r"//.*?(?=\n|$)", "", b.strip(), flags=re.MULTILINE)
        b = re.sub(r"/\*.*?\*/", "", b, flags=re.DOTALL)
        try:
            p = _coords_from_data(json.loads(b))
        except (json.JSONDecodeError, ValueError, TypeError):
            p = None
        if p:
            return p
    # bare key-value text, e.g. the entrypoint's \boxed{}-stripped '"x": 335, "y": 360' (braces consumed by the wrapper)
    m = re.search(r"[\"']?\bx[\"']?\s*[:=]\s*([0-9.]+)\s*,?\s*[\"']?\by[\"']?\s*[:=]\s*([0-9.]+)", s, re.IGNORECASE | re.DOTALL)
    if m:
        try:
            return float(m.group(1).rstrip(".")), float(m.group(2).rstrip("."))
        except ValueError:
            pass
    patterns = [
        (r'<point\s+x="([0-9.]+)"\s+y="([0-9.]+)"[^>]*>', False),
        (r"(?:coordinates?|position|location).*?(?:are|is)\s+([0-9.]+),\s*([0-9.]+)", False),
        (r"[\"']?[xX][\"']?\s*[:=]\s*([0-9.]+).*?[\"']?[yY][\"']?\s*[:=]\s*([0-9.]+)", False),
        (r"[\"']?[yY][\"']?\s*[:=]\s*([0-9.]+).*?[\"']?[xX][\"']?\s*[:=]\s*([0-9.]+)", True),
        (r"\[([0-9.]+),\s*([0-9.]+)\]", False),
        (r"\(([0-9.]+),\s*([0-9.]+)\)", False),
        (r"([0-9.]+),\s*([0-9.]+)", False),
    ]
    for pat, swap in patterns:
        ms = re.findall(pat, s, re.IGNORECASE | re.DOTALL)
        if ms:
            try:
                a, b = float(ms[-1][0].rstrip(".")), float(ms[-1][1].rstrip("."))
            except (ValueError, IndexError):
                continue
            return (b, a) if swap else (a, b)
    return None


@dataclass
class CrossPointSample(BaseBenchmarkSample):
    """One CrossPoint-Bench item. ``question`` is verbatim (options are inline), so no ``choices``."""

    idx: int = 0
    task: str = ""                 # one of TYPES
    level: str = ""                # object | part
    is_point: bool = False
    options: Tuple[str, ...] = ()  # valid letters for MCQ items
    mask_b64: str = ""             # pointing items: base64 PNG mask of the target image
    target_size: Tuple[int, int] = (0, 0)   # (w, h) of the target (last) image


class CrossPointBench(BaseBenchmark):
    """CrossPoint-Bench loader and scorer (official in-mask / letter rules; arm-specific coordinate frame)."""

    data_specific_prompt = (
        "For a multiple-choice question, answer with the option letter only. For a pointing question, output "
        "the point as JSON {\"x\": <x>, \"y\": <y>} in pixel coordinates of the image it refers to "
        "(x to the right, y down)."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self.coord_format = os.environ.get(COORD_FORMAT_ENV, "absolute_original")
        if self.coord_format not in COORD_FORMATS:
            raise ValueError(f"{COORD_FORMAT_ENV}={self.coord_format!r} not in {COORD_FORMATS}")
        self.shown_grid = shown_grid()   # captured at construction, like the coordinate format (one loader per arm)
        self._mask_cache: Dict[str, Any] = {}
        super().__init__(data_path, question_type)

    def _wants(self, task: str, level: str) -> bool:
        if not self.question_type_filter:
            return True
        wanted = {str(q).strip().lower() for q in self.question_type_filter}
        return bool({task.lower(), level.lower()} & wanted)

    def read_data(self) -> None:
        from PIL import Image
        self.data_path = os.path.abspath(self.data_path)
        path = os.path.join(self.data_path, JSONL)
        if not os.path.exists(path):
            raise FileNotFoundError(f"CrossPoint-Bench jsonl not found: {path} (download HF WangYipu2002/CrossPoint-Bench)")
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            task, level = str(r["type"]), str(r.get("level", ""))
            if task not in TYPES:
                raise ValueError(f"CrossPoint-Bench: unknown type {task!r} (idx {r.get('idx')})")
            if not self._wants(task, level):
                continue
            images = [os.path.join(self.data_path, IMAGE_DIR, p) for p in r["images"]]
            for p in images:
                if not os.path.exists(p):
                    raise FileNotFoundError(f"CrossPoint-Bench image missing: {p}")
            is_point = task in POINT_TYPES
            question = str(r["question"]).strip()
            if is_point:
                question = question + OFFICIAL_POINT_SUFFIX
            with Image.open(images[-1]) as im:
                tw, th = im.size
            options: Tuple[str, ...] = ()
            if not is_point:
                options = ("A", "B", "C") if task == "Correspondence-Judgement" else ("A", "B")
            self.data.append(CrossPointSample(
                sample_id=str(r["idx"]), question=question, question_type=task, images=images,
                answer=("" if is_point else str(r["answer"]).strip().upper()),
                idx=int(r["idx"]), task=task, level=level, is_point=is_point, options=options,
                mask_b64=(str(r["answer"]) if is_point else ""), target_size=(tw, th),
            ))
        print(f"[CrossPointBench] Loaded {len(self.data)} items from {path} (coordinate format: {self.coord_format}, shown grid: {self.shown_grid})")

    # ── scoring ──────────────────────────────────────────────────────────

    def _mask(self, sample: CrossPointSample):
        import numpy as np
        from PIL import Image
        key = str(sample.sample_id)
        if key not in self._mask_cache:
            arr = np.array(Image.open(io.BytesIO(base64.b64decode(sample.mask_b64))))
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            self._mask_cache[key] = arr
        return self._mask_cache[key]

    def point_hit(self, sample: CrossPointSample, x: float, y: float) -> bool:
        """Official rule on the ORIGINAL frame: int pixel inside the image and mask value > 128."""
        mask = self._mask(sample)
        h, w = mask.shape[:2]
        xi, yi = int(x), int(y)
        if not (0 <= xi < w and 0 <= yi < h):
            return False
        return bool(mask[yi, xi] > MASK_THRESHOLD)

    def extract_answer(self, prediction: str) -> str:
        return _extract_letter(prediction)

    def score_item(self, sample: CrossPointSample, prediction: str) -> Dict[str, Any]:
        if sample.is_point:
            p = extract_point(prediction)
            if p is None:
                return {"correct": 0.0, "parsed": False}
            w, h = sample.target_size
            ox, oy = to_original(p[0], p[1], self.coord_format, w, h, grid=self.shown_grid)
            return {"correct": 1.0 if self.point_hit(sample, ox, oy) else 0.0, "parsed": True,
                    "pred_raw": [p[0], p[1]], "pred_original": [ox, oy]}
        letter = _extract_letter(prediction)
        if letter and letter not in sample.options:
            letter = ""
        return {"correct": 1.0 if (letter and letter == sample.answer) else 0.0, "parsed": bool(letter), "pred": letter}

    def evaluate_single(self, sample: BaseBenchmarkSample, prediction: str) -> Optional[float]:
        return self.score_item(sample, prediction)["correct"]

    def evaluate(self, predictions: Dict[Any, str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        detailed = []
        missing = 0
        for s in self.data:
            raw = get_prediction(predictions, s.sample_id, default=None)
            if raw is None:
                missing += 1
                raw = ""
            sc = self.score_item(s, raw)
            detailed.append({"sample_id": s.sample_id, "type": s.task, "level": s.level, "is_point": s.is_point,
                             "n_images": len(s.images), "answer": (s.answer if not s.is_point else "<mask>"), **sc})

        def acc(items):
            return mean_or_zero(d["correct"] for d in items)

        results: Dict[str, Any] = {
            "coord_format": self.coord_format, "n_items": len(detailed), "missing_predictions": missing,
            "unparsed_predictions": sum(1 for d in detailed if not d["parsed"]),
            "overall_accuracy": acc(detailed),
            "by_type": {t: acc([d for d in detailed if d["type"] == t]) for t in TYPES},
            "by_type_level": {f"{t}/{l}": acc([d for d in detailed if d["type"] == t and d["level"] == l])
                              for t in TYPES for l in LEVELS},
            "pointing_accuracy": acc([d for d in detailed if d["is_point"]]),
            "mcq_accuracy": acc([d for d in detailed if not d["is_point"]]),
            "unparsed_by_type": {t: sum(1 for d in detailed if d["type"] == t and not d["parsed"]) for t in TYPES},
            "counts_by_type": {t: sum(1 for d in detailed if d["type"] == t) for t in TYPES},
            "detailed_results": detailed,
        }
        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), detailed)
        return results
