"""SpatialTree-Bench data loader.

Data structure:
    data/SpatialTree-Bench/annotations_plain.parquet  (6300 samples)
    data/SpatialTree-Bench/videos/*.mp4
    data/SpatialTree-Bench/images/*.jpg

Hierarchical spatial reasoning benchmark with 4 levels (L1-L4), covering
geometry, attributes, simulation, and agentic competence.

Question types: multiple-choice (4248), open (1852), judge (200).

Metrics depend on metricfunc in extra_info:
    - meanrelativeacc → Mean Relative Accuracy (MRA)
    - accuracy / exact_match → exact match accuracy
    - Others → mapped to closest metric

Reference: https://huggingface.co/datasets/LongfeiLi/SpatialTree-Bench
           arXiv:2512.20617
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


import numpy as np
import pandas as pd

from spatial_agent.evals.base import BaseBenchmark, LazyVideoSample, VideoFrameBenchmarkMixin
from spatial_agent.config import get_config
from spatial_agent.evals.scoring import (
    get_prediction,
    mean_relative_accuracy,
    write_results_summary,
)


# ── Metrics ──────────────────────────────────────────────────────────────

def _mean_relative_accuracy(
    pred: float, target: float,
    start: float = 0.5, end: float = 0.95, interval: float = 0.05,
) -> float:
    return mean_relative_accuracy(
        pred,
        target,
        start=start,
        end=end,
        interval=interval,
    )


def _to_float(s) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _fuzzy_matching_num(pred: str) -> Optional[str]:
    pred = pred.strip().lower()
    number_words = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "zero": "0",
    }
    for word, digit in number_words.items():
        if re.search(r"\b" + word + r"\b", pred):
            return digit
    m = re.search(r"(\d+(?:\.\d+)?)", pred)
    if m:
        return m.group(1)
    return None


def _fuzzy_matching_mc(pred: str) -> str:
    m = re.search(r"^[A-Za-z]\.?$", pred.split(" ")[0].strip())
    if m:
        return m.group(0).rstrip(".").upper()
    return pred.strip()


# Metric functions mapped by metricfunc string
METRIC_FUNCS = {
    "meanrelativeacc": "mra",
    "meanrelativeacc2": "mra",
    "accuracy": "accuracy",
    "exact_match": "accuracy",
    "angulardifference": "mra",  # Treat angular as MRA
    "l2distance": "mra",
    "successrate": "accuracy",
}

# Metrics that require external tools (LLM judge, graph isomorphism, etc.)
# These cannot be scored locally — results are marked as None
_UNSCOREABLE_METRICS = {
    "gpteval", "cogmapeval", "affmask", "manipulateeval",
    "agenticnaveval", "gravityeval",
}


# ── Sample ───────────────────────────────────────────────────────────────

@dataclass
class SpatialTreeSample(LazyVideoSample):
    """SpatialTree-Bench sample."""

    choices: List[str] = field(default_factory=list)
    hint: str = ""
    level: str = ""          # L1, L2, L3, L4
    category: str = ""       # Geometry, Attribute, etc.
    subcategory: str = ""    # Distance, Size, etc.
    metricfunc: str = ""     # meanrelativeacc, accuracy, etc.


class SpatialTreeBench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """SpatialTree-Bench loader (6300 spatial reasoning samples)."""

    data_specific_prompt = (
        "Answer the spatial reasoning question. "
        "For multiple-choice, answer with the option letter. "
        "For open-ended numeric questions, answer with a single number. "
        "For yes/no questions, answer Yes or No."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._config = get_config()
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        parquet_path = os.path.join(self.data_path, "annotations_plain.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"SpatialTree data not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        for _, row in df.iterrows():
            qtype = row.get("question_type", "open")
            if self.question_type_filter and qtype not in self.question_type_filter:
                continue

            # Parse extra_info
            extra = {}
            if row.get("extra_info"):
                try:
                    extra = json.loads(row["extra_info"]) if isinstance(row["extra_info"], str) else row["extra_info"]
                except (json.JSONDecodeError, TypeError):
                    pass

            # Resolve media paths
            video_path = ""
            image_paths = []
            videos = row.get("video", [])
            images_col = row.get("image", [])
            if hasattr(videos, "tolist"):
                videos = videos.tolist()
            if hasattr(images_col, "tolist"):
                images_col = images_col.tolist()

            if videos and len(videos) > 0 and videos[0]:
                video_path = os.path.join(self.data_path, videos[0])
            if images_col and len(images_col) > 0:
                image_paths = [
                    os.path.join(self.data_path, img)
                    for img in images_col if img
                ]

            # Parse options
            options = row.get("option")
            choices = []
            if options is not None:
                if hasattr(options, "tolist"):
                    choices = options.tolist()
                elif isinstance(options, list):
                    choices = list(options)
            # Filter out None entries
            choices = [c for c in choices if c is not None]

            sample = SpatialTreeSample(
                sample_id=row["session_id"],
                question=row["question"],
                question_type=qtype,
                images=image_paths,
                answer=str(row.get("answer", "")),
                video=video_path,
                choices=choices,
                hint=str(row.get("hint", "") or ""),
                level=extra.get("spatree0", ""),
                category=extra.get("spatree1", ""),
                subcategory=extra.get("spatree2", ""),
                metricfunc=extra.get("metricfunc", ""),
                _bench_ref=self,
            )
            self.data.append(sample)

    def _score_sample(self, sample: "SpatialTreeSample", pred_raw: str) -> Optional[float]:
        """Score a single sample. Returns None for unscoreable metrics."""
        # Unscoreable metrics (need LLM judge, graph isomorphism, mask decode, etc.)
        if sample.metricfunc in _UNSCOREABLE_METRICS:
            return None

        metric_type = METRIC_FUNCS.get(sample.metricfunc, "accuracy")

        if sample.question_type == "multiple-choice":
            # MC: extract letter and compare
            pred = self._extract_mc_answer(pred_raw, sample.choices)
            gt = sample.answer.strip().upper()
            return 1.0 if pred == gt else 0.0

        elif metric_type == "mra":
            # Numeric: MRA
            pred_num_str = _fuzzy_matching_num(pred_raw) if pred_raw else None
            pred_f = _to_float(pred_num_str)
            gt_f = _to_float(sample.answer)
            if pred_f is not None and gt_f is not None and gt_f != 0:
                return _mean_relative_accuracy(pred_f, gt_f)
            return 0.0

        elif sample.question_type == "judge":
            # Yes/No: find last occurrence of yes/no
            pred_lower = pred_raw.strip().lower()
            gt_lower = sample.answer.strip().lower()
            last_yes = pred_lower.rfind("yes")
            last_no = pred_lower.rfind("no")
            if last_yes > last_no:
                pred_answer = "yes"
            elif last_no > last_yes:
                pred_answer = "no"
            else:
                pred_answer = pred_lower
            return 1.0 if pred_answer == gt_lower else 0.0

        else:
            # Fallback: try numeric MRA, then string match
            pred_num_str = _fuzzy_matching_num(pred_raw) if pred_raw else None
            pred_f = _to_float(pred_num_str)
            gt_f = _to_float(sample.answer)
            if pred_f is not None and gt_f is not None and gt_f != 0:
                return _mean_relative_accuracy(pred_f, gt_f)
            # String match
            return 1.0 if pred_raw.strip().lower() == sample.answer.strip().lower() else 0.0

    def _extract_mc_answer(self, prediction: str, choices: List[str]) -> str:
        if not prediction:
            return ""
        prediction = prediction.strip()

        # Boxed format
        m = re.search(r"\\boxed{\s*([A-Za-z])\s*}", prediction)
        if m:
            return m.group(1).upper()

        # Letter patterns
        for pat in [r"\b([A-K])\.", r"\(([A-K])\)", r"\b([A-K]):"]:
            m = re.search(pat, prediction, re.I)
            if m:
                return m.group(1).upper()

        # First token
        first = _fuzzy_matching_mc(prediction)
        if len(first) == 1 and first.isalpha():
            return first.upper()

        return first

    def evaluate_single(self, sample, prediction: str) -> Optional[float]:
        """Score a single sample. Returns None for unscoreable metrics."""
        return self._score_sample(sample, prediction.strip() if prediction else "")

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        # Per-level, per-category scores
        per_level: Dict[str, List[float]] = {}
        per_category: Dict[str, List[float]] = {}
        per_subcategory: Dict[str, List[float]] = {}
        per_qtype: Dict[str, List[float]] = {}
        detailed = []

        per_metricfunc: Dict[str, List[float]] = {}
        unscored_count = 0

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            score = self._score_sample(sample, pred_raw)

            detailed.append({
                "id": sid, "level": sample.level,
                "category": sample.category,
                "subcategory": sample.subcategory,
                "question_type": sample.question_type,
                "metricfunc": sample.metricfunc,
                "ground_truth": sample.answer,
                "prediction": pred_raw,
                "score": score,  # None for unscoreable
            })

            if score is None:
                unscored_count += 1
                continue

            per_level.setdefault(sample.level, []).append(score)
            cat_key = f"{sample.level}/{sample.category}"
            per_category.setdefault(cat_key, []).append(score)
            sub_key = f"{sample.level}/{sample.category}/{sample.subcategory}"
            per_subcategory.setdefault(sub_key, []).append(score)
            per_qtype.setdefault(sample.question_type, []).append(score)
            per_metricfunc.setdefault(sample.metricfunc, []).append(score)

        all_scores = [d["score"] for d in detailed if d["score"] is not None]
        overall = float(np.mean(all_scores)) if all_scores else 0.0

        results = {
            "total_samples": len(detailed),
            "scored_samples": len(all_scores),
            "unscored_samples": unscored_count,
            "correct_samples": sum(1 for s in all_scores if s >= 1.0),
            "overall_accuracy": overall,
            "overall_score_pct": overall * 100,
            "per_level": {
                k: {"score": float(np.mean(v)) * 100, "count": len(v)}
                for k, v in sorted(per_level.items())
            },
            "per_category": {
                k: {"score": float(np.mean(v)) * 100, "count": len(v)}
                for k, v in sorted(per_category.items())
            },
            "per_subcategory": {
                k: {"score": float(np.mean(v)) * 100, "count": len(v)}
                for k, v in sorted(per_subcategory.items())
            },
            "per_question_type": {
                k: {"score": float(np.mean(v)) * 100, "count": len(v)}
                for k, v in sorted(per_qtype.items())
            },
            "per_metricfunc": {
                k: {"score": float(np.mean(v)) * 100, "count": len(v)}
                for k, v in sorted(per_metricfunc.items())
            },
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*70}")
        print("SpatialTree-Bench Evaluation Results")
        print(f"{'='*70}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Scored: {results['scored_samples']}, Unscored: {results['unscored_samples']}")
        print(f"Overall score (scored only): {results['overall_score_pct']:.2f}")
        print(f"\n--- Per Level ---")
        for k, v in results.get("per_level", {}).items():
            print(f"  {k:10s} {v['score']:6.2f}  (n={v['count']})")
        print(f"\n--- Per Category ---")
        for k, v in results.get("per_category", {}).items():
            print(f"  {k:40s} {v['score']:6.2f}  (n={v['count']})")
        print(f"\n--- Per Metric Function ---")
        for k, v in results.get("per_metricfunc", {}).items():
            print(f"  {k:20s} {v['score']:6.2f}  (n={v['count']})")
        print(f"\n--- Per Question Type ---")
        for k, v in results.get("per_question_type", {}).items():
            print(f"  {k:20s} {v['score']:6.2f}  (n={v['count']})")
        print(f"\nNote: {results['unscored_samples']} samples with metrics requiring")
        print(f"external tools (gpteval, cogmapeval, affmask, manipulateeval,")
        print(f"agenticnaveval, gravityeval) are excluded from scoring.")
        print(f"{'='*70}\n")
