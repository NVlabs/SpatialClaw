"""RoboSpatial-Home (Song et al., CVPR 2025 oral; arXiv 2411.16537) loader and scorer.

350 single-image items from real homes (1440×1920 / 1920×1440 iPhone photos; the shipped LiDAR depth is NOT given
to either arm): configuration 123 and compatibility 105 are yes/no questions ("... Answer yes or no."); context 122
ask for several points in a vacant region ("... list of tuples ... between 0 and 1 ..."), scored against a binary mask.
Data: HF chanhee-luke/RoboSpatial-Home (Apache-2.0), parquet per split with embedded images, dumped once to
``<data_path>/images/<split>_<i>.jpg`` and ``masks/<split>_<i>.png``.

Official scorer (RoboSpatial-Eval evaluation.py): yes/no correct iff the lower-cased answer STARTS with the gold word
(we take the \\boxed{} / final answer first, since both arms reason before answering — declared parser deviation);
context: the first 2 parsed points, each mapped with nearest-pixel rounding and CLAMPED to the image, correct iff any
lands on a mask pixel > 0. Overall = mean over all items; per category; always-yes prior 72.3 % on the binary half,
so balanced accuracy is reported too. Coordinate conventions: ``pointing_common``.
"""

import glob
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.pointing_common import any_in_mask_clamped, coord_format, extract_yes_no, parse_points, points_to_original, shown_grid
from spatial_agent.evals.scoring import get_prediction, mean_or_zero, write_json, write_results_summary

SPLITS = ["configuration", "compatibility", "context"]


@dataclass
class RoboSpatialHomeSample(BaseBenchmarkSample):
    split: str = ""
    is_point: bool = False
    mask_path: str = ""
    size: tuple = (0, 0)


class RoboSpatialHome(BaseBenchmark):
    """RoboSpatial-Home loader and scorer."""

    data_specific_prompt = ""   # each question carries its own answer-format instruction

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self.coord_format = coord_format()
        self.shown_grid = shown_grid()   # the InputImages grid the arm showed, captured at construction
        self._mask_cache: Dict[str, Any] = {}
        super().__init__(data_path, question_type)

    def _dump(self, split: str) -> List[Dict[str, Any]]:
        import pyarrow.parquet as pq
        files = sorted(glob.glob(os.path.join(self.data_path, "data", f"{split}-*.parquet")))
        if not files:
            raise FileNotFoundError(f"RoboSpatial-Home parquet for split {split!r} not found under {self.data_path}/data")
        img_dir, msk_dir = os.path.join(self.data_path, "images"), os.path.join(self.data_path, "masks")
        os.makedirs(img_dir, exist_ok=True); os.makedirs(msk_dir, exist_ok=True)
        rows = []
        i = 0
        for f in files:
            for r in pq.read_table(f).to_pylist():
                img = os.path.join(img_dir, f"{split}_{i}.jpg")
                if not os.path.exists(img):
                    Image.open(io.BytesIO(r["img"]["bytes"])).convert("RGB").save(img + ".tmp.jpg", quality=95); os.replace(img + ".tmp.jpg", img)
                msk = ""
                m = r.get("mask")
                if isinstance(m, dict) and m.get("bytes"):
                    msk = os.path.join(msk_dir, f"{split}_{i}.png")
                    if not os.path.exists(msk):
                        Image.open(io.BytesIO(m["bytes"])).convert("L").save(msk + ".tmp.png"); os.replace(msk + ".tmp.png", msk)
                rows.append({"i": i, "question": str(r["question"]).strip(), "answer": str(r["answer"]).strip(), "img": img, "mask": msk})
                i += 1
        return rows

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        for split in SPLITS:
            if self.question_type_filter and split not in {q.lower() for q in self.question_type_filter}:
                continue
            for r in self._dump(split):
                is_point = split == "context"
                with Image.open(r["img"]) as im:
                    size = im.size
                self.data.append(RoboSpatialHomeSample(
                    sample_id=f"{split}_{r['i']}", question=r["question"], question_type=split, images=[r["img"]],
                    answer=("<mask>" if is_point else r["answer"].lower()), split=split, is_point=is_point,
                    mask_path=r["mask"], size=size))
        print(f"[RoboSpatialHome] Loaded {len(self.data)} items from {self.data_path} (coordinate format: {self.coord_format}, shown grid: {self.shown_grid})")

    def _mask(self, s: RoboSpatialHomeSample):
        if s.mask_path not in self._mask_cache:
            self._mask_cache[s.mask_path] = np.array(Image.open(s.mask_path).convert("L"), dtype=np.uint8)
        return self._mask_cache[s.mask_path]

    def score_item(self, s: RoboSpatialHomeSample, prediction: str) -> Dict[str, Any]:
        if s.is_point:
            pts, kind = parse_points(prediction)
            if not pts:
                return {"correct": 0.0, "parsed": False, "n_points": 0}
            orig = points_to_original(pts, self.coord_format, *s.size, grid=self.shown_grid)
            hit = any_in_mask_clamped(orig, self._mask(s), *s.size, limit=2)
            return {"correct": 1.0 if hit else 0.0, "parsed": True, "n_points": len(pts), "parse_kind": kind}
        yn = extract_yes_no(prediction)
        return {"correct": 1.0 if (yn and yn == s.answer) else 0.0, "parsed": yn is not None, "pred": yn or ""}

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
            detailed.append({"sample_id": s.sample_id, "split": s.split, "is_point": s.is_point, "answer": s.answer, **sc})

        def acc(items):
            return mean_or_zero(d["correct"] for d in items)

        binary = [d for d in detailed if not d["is_point"]]
        yes_items = [d for d in binary if d["answer"] == "yes"]; no_items = [d for d in binary if d["answer"] == "no"]
        results = {
            "coord_format": self.coord_format, "n_items": len(detailed), "missing_predictions": missing,
            "unparsed_predictions": sum(1 for d in detailed if not d["parsed"]),
            "overall_accuracy": acc(detailed),
            "by_split": {sp: acc([d for d in detailed if d["split"] == sp]) for sp in SPLITS},
            "binary_accuracy": acc(binary),
            "binary_balanced_accuracy": (acc(yes_items) + acc(no_items)) / 2 if yes_items and no_items else None,
            "binary_pred_yes_rate": mean_or_zero(1.0 if d.get("pred") == "yes" else 0.0 for d in binary),
            "context_accuracy": acc([d for d in detailed if d["is_point"]]),
            "detailed_results": detailed,
        }
        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), detailed)
        return results
