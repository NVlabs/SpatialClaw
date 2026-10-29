"""Where2Place (RoboPoint, Yuan et al., CoRL 2024; arXiv 2406.10721) loader and scorer.

100 single images (640×480) with one free-space instruction each ("Identify several spots within the vacant space
... list of tuples ... between 0 and 1 ...") and a binary mask; ``category`` seen 70 / unseen 30 (the paper's
Where2Place(h) subset, reported here only as a breakdown). Data: HF wentao-yuan/where2place (Apache-2.0):
``point_questions.jsonl``, ``images/XX.jpg``, ``masks/XX.jpg`` (8-bit JPEG masks, thresholded at 127).

Official scorer (robopoint/eval/summarize_vqa.py): per image the fraction of parsed points on the mask (mask/255 →
pixel value), out-of-image points count 0, unparsable = 0; mean over the 100 images.
Coordinate conventions: ``pointing_common``.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.pointing_common import coord_format, in_mask_fraction, parse_points, points_to_original, shown_grid
from spatial_agent.evals.scoring import get_prediction, mean_or_zero, write_json, write_results_summary


@dataclass
class Where2PlaceSample(BaseBenchmarkSample):
    category: str = ""
    mask_path: str = ""
    size: tuple = (0, 0)


class Where2Place(BaseBenchmark):
    """Where2Place loader and scorer."""

    data_specific_prompt = ""

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self.coord_format = coord_format()
        self.shown_grid = shown_grid()   # the InputImages grid the arm showed, captured at construction
        self._mask_cache: Dict[str, Any] = {}
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        qpath = os.path.join(self.data_path, "point_questions.jsonl")
        if not os.path.exists(qpath):
            raise FileNotFoundError(f"Where2Place question file not found: {qpath}")
        for line in open(qpath):
            if not line.strip():
                continue
            r = json.loads(line)
            cat = str(r.get("category") or "")
            if self.question_type_filter and cat not in {q.lower() for q in self.question_type_filter}:
                continue
            img = os.path.join(self.data_path, "images", r["image"])
            msk = os.path.join(self.data_path, "masks", f"{int(r['question_id']):02d}.jpg")
            for p in (img, msk):
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Where2Place file missing: {p}")
            with Image.open(img) as im:
                size = im.size
            self.data.append(Where2PlaceSample(sample_id=str(r["question_id"]), question=str(r["text"]).strip(),
                                               question_type=cat, images=[img], answer="<mask>", category=cat,
                                               mask_path=msk, size=size))
        print(f"[Where2Place] Loaded {len(self.data)} items from {self.data_path} (coordinate format: {self.coord_format}, shown grid: {self.shown_grid})")

    def _mask(self, s: Where2PlaceSample):
        if s.mask_path not in self._mask_cache:
            self._mask_cache[s.mask_path] = np.array(Image.open(s.mask_path).convert("L"), dtype=np.uint8)
        return self._mask_cache[s.mask_path]

    def score_item(self, s: Where2PlaceSample, prediction: str) -> Dict[str, Any]:
        pts, kind = parse_points(prediction)
        if not pts:
            return {"correct": 0.0, "parsed": False, "n_points": 0}
        orig = points_to_original(pts, self.coord_format, *s.size, grid=self.shown_grid)
        frac = in_mask_fraction(orig, self._mask(s), threshold=127)
        return {"correct": frac, "parsed": True, "n_points": len(pts), "parse_kind": kind}

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
            detailed.append({"sample_id": s.sample_id, "category": s.category, **self.score_item(s, raw)})

        def acc(items):
            return mean_or_zero(d["correct"] for d in items)

        results = {
            "coord_format": self.coord_format, "n_items": len(detailed), "missing_predictions": missing,
            "unparsed_predictions": sum(1 for d in detailed if not d["parsed"]),
            "overall_accuracy": acc(detailed),
            "by_category": {c: acc([d for d in detailed if d["category"] == c]) for c in ("seen", "unseen")},
            "mean_points_per_item": mean_or_zero(d["n_points"] for d in detailed),
            "detailed_results": detailed,
        }
        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), detailed)
        return results
