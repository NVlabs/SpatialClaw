"""RefSpatial-Bench (RoboRefer, Zhou et al., NeurIPS 2025; arXiv 2506.04308) loader and scorer.

277 single-image spatial-referring items with one human-annotated binary mask each:
    Location 100 (point on the referred object), Placement 100 (point in free space under spatial constraints),
    Unseen 77 (relation compositions held out from RoboRefer's training); ``step`` 1-5 = number of anchor relations.
Data under ``data/RefSpatial-Bench/<Split>/{question.json,image/,mask/}`` (HF BAAI/RefSpatial-Bench, Apache-2.0).

Prompt: the dataset's ``prompt`` + ``suffix`` verbatim (the suffix asks for ``[(x1, y1)]`` with coordinates in [0, 1]).
Official metric (Evaluation/summarize_acc.py): points parsed from the text, rescaled to pixels, item score = fraction of
points inside the mask (mask > 0; out-of-image points count 0; no points -> 0); reported per split and averaged.
Coordinate conventions and the shown-frame mapping: see ``pointing_common``.
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

SPLITS = ["Location", "Placement", "Unseen"]


@dataclass
class RefSpatialSample(BaseBenchmarkSample):
    split: str = ""
    step: int = 0
    target: str = ""
    mask_path: str = ""
    size: tuple = (0, 0)


class RefSpatialBench(BaseBenchmark):
    """RefSpatial-Bench loader and scorer (official fraction-in-mask rule, per-arm coordinate convention)."""

    data_specific_prompt = ""   # the dataset's own suffix carries the answer format

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self.coord_format = coord_format()
        self.shown_grid = shown_grid()   # the InputImages grid the arm showed, captured at construction
        self._mask_cache: Dict[str, Any] = {}
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        for split in SPLITS:
            qpath = os.path.join(self.data_path, split, "question.json")
            if not os.path.exists(qpath):
                raise FileNotFoundError(f"RefSpatial-Bench question file not found: {qpath}")
            if self.question_type_filter and split.lower() not in {q.lower() for q in self.question_type_filter}:
                continue
            for r in json.load(open(qpath)):
                img = os.path.join(self.data_path, split, r["rgb_path"])
                msk = os.path.join(self.data_path, split, r["mask_path"])
                for p in (img, msk):
                    if not os.path.exists(p):
                        raise FileNotFoundError(f"RefSpatial-Bench file missing: {p}")
                with Image.open(img) as im:
                    size = im.size
                self.data.append(RefSpatialSample(
                    sample_id=f"{split.lower()}_{r['id']}", question=(str(r["prompt"]).strip() + " " + str(r["suffix"]).strip()),
                    question_type=split.lower(), images=[img], answer="<mask>", split=split.lower(),
                    step=int(r.get("step") or 0), target=str(r.get("object") or ""), mask_path=msk, size=size))
        print(f"[RefSpatialBench] Loaded {len(self.data)} items from {self.data_path} (coordinate format: {self.coord_format}, shown grid: {self.shown_grid})")

    def _mask(self, s: RefSpatialSample):
        if s.mask_path not in self._mask_cache:
            arr = np.array(Image.open(s.mask_path))
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            self._mask_cache[s.mask_path] = arr
        return self._mask_cache[s.mask_path]

    def score_item(self, s: RefSpatialSample, prediction: str) -> Dict[str, Any]:
        pts, kind = parse_points(prediction)
        if not pts:
            return {"correct": 0.0, "parsed": False, "n_points": 0}
        orig = points_to_original(pts, self.coord_format, *s.size, grid=self.shown_grid)
        frac = in_mask_fraction(orig, self._mask(s), threshold=0)
        return {"correct": frac, "parsed": True, "n_points": len(pts), "parse_kind": kind, "first_point_original": list(orig[0])}

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
            detailed.append({"sample_id": s.sample_id, "split": s.split, "step": s.step, **sc})

        def acc(items):
            return mean_or_zero(d["correct"] for d in items)

        results = {
            "coord_format": self.coord_format, "n_items": len(detailed), "missing_predictions": missing,
            "unparsed_predictions": sum(1 for d in detailed if not d["parsed"]),
            "overall_accuracy": acc(detailed),
            "by_split": {sp: acc([d for d in detailed if d["split"] == sp]) for sp in ("location", "placement", "unseen")},
            "split_average": mean_or_zero(acc([d for d in detailed if d["split"] == sp]) for sp in ("location", "placement", "unseen")),
            "by_step": {str(k): acc([d for d in detailed if d["step"] == k]) for k in sorted({d["step"] for d in detailed})},
            "mean_points_per_item": mean_or_zero(d["n_points"] for d in detailed),
            "detailed_results": detailed,
        }
        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), detailed)
        return results
