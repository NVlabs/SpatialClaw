"""OSI-Bench benchmark data loader.

Data structure:
    data/OSI-Bench/data.parquet          (8766 samples)
    data/OSI-Bench/*.mp4                 (1000 videos)

9 categories in 2 scoring types:
    MCQ (scored by accuracy):
        Trajectory Description, Relative Distance, Relative Direction
    Numerical (scored by Mean Relative Accuracy):
        Absolute Speed, Absolute Displacement, Absolute Distance,
        Object 3D Localization, Depth-Aware Counting, Trajectory Length

Evaluation:
    - MCQ: case-insensitive exact match of extracted letter
    - Numerical: MRA with official zero-handling for motion-style quantities
    - Overall = weighted mean over all samples (×100)

Reference: CVPR 2026
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


import numpy as np
import pandas as pd

from spatial_agent.evals.base import BaseBenchmark, LazyVideoSample, VideoFrameBenchmarkMixin
from spatial_agent.evals.scoring import (
    get_prediction,
    mean_or_zero,
    mean_relative_accuracy,
    write_results_summary,
)
from spatial_agent.config import get_config


MCQ_CATEGORIES = [
    "trajectory_description",
    "relative_distance",
    "relative_direction_categorical_ordinal",
]

NUMERICAL_CATEGORIES = [
    "absolute_speed",
    "absolute_displacement",
    "absolute_distance",
    "object_3d_localization",
    "depth_aware_counting",
    "trajectory_length",
]

ALL_CATEGORIES = MCQ_CATEGORIES + NUMERICAL_CATEGORIES

ZERO_AWARE_THRESHOLDS = {
    "absolute_speed": 0.30,
    "absolute_displacement": 0.30,
    "trajectory_length": 2.0,
}


def _mean_relative_accuracy(
    pred: float, target: float,
    start: float = 0.5, end: float = 0.95, interval: float = 0.05,
) -> float:
    """MRA: fraction of threshold levels passed."""
    return mean_relative_accuracy(
        pred,
        target,
        start=start,
        end=end,
        interval=interval,
    )


def _mean_relative_accuracy_consider_zero(
    pred: float,
    target: float,
    threshold: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    """OSI-Bench zero-aware MRA variant for specific numerical categories."""
    return mean_relative_accuracy(
        pred,
        target,
        start=start,
        end=end,
        interval=interval,
        zero_policy="threshold",
        zero_threshold=threshold,
    )


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _fuzzy_matching(pred: str) -> str:
    """First space-delimited token, strip trailing period."""
    return pred.split(" ")[0].rstrip(".").strip()


def _parse_options(raw) -> Dict[str, str]:
    """Parse array of 4 option strings into {"A": ..., "B": ..., "C": ..., "D": ...}."""
    if raw is None:
        return {}
    # Handle numpy arrays
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list) or len(raw) == 0:
        return {}
    result = {}
    for i, opt in enumerate(raw):
        if opt is None or (isinstance(opt, float) and np.isnan(opt)):
            continue
        opt_str = str(opt).strip()
        # Try "A. text" format
        m = re.match(r"([A-Da-d])[.)]\s*(.*)", opt_str)
        if m:
            result[m.group(1).upper()] = m.group(2).strip()
        else:
            letter = chr(65 + i)
            result[letter] = opt_str
    return result


# ── sample & benchmark ────────────────────────────────────────────────────

@dataclass
class OSIBenchSample(LazyVideoSample):
    """OSI-Bench sample with video and optional MC options."""

    choices: Dict[str, str] = field(default_factory=dict)
    category: str = ""


class OSIBench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """OSI-Bench loader (8766 outdoor spatial intelligence questions).

    Evaluation uses accuracy for MCQ categories and MRA for numerical categories.
    Overall matches the official weighted mean over all samples.
    """

    data_specific_prompt = (
        "These are frames of a video.\n"
        "Answer the question. For multiple-choice, answer with the option's "
        "letter from the given choices directly.\n"
        "For numerical questions, answer with a single number."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._config = get_config()
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        parquet_path = os.path.join(self.data_path, "data.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"OSI-Bench data not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            category = row["category"]
            # Filter by category (question_type_filter maps to category)
            if self.question_type_filter and category not in self.question_type_filter:
                continue

            qtype = row["question_type"]  # "mcq" or "numerical"
            video_path = os.path.join(self.data_path, row["file_name"])

            choices = _parse_options(row.get("options"))

            duration = row.get("video_length")
            if duration is not None and isinstance(duration, float) and np.isnan(duration):
                duration = 0.0
            elif duration is None:
                duration = 0.0

            sample = OSIBenchSample(
                sample_id=row["index"],
                question=row["question"],
                question_type=qtype,
                images=[],
                answer=str(row["answer"]).strip(),
                video=video_path,
                choices=choices,
                category=category,
                duration_sec=float(duration),
                _bench_ref=self,
            )
            self.data.append(sample)

    # ── answer extraction ─────────────────────────────────────────────────

    def extract_answer(self, prediction: str) -> str:
        if not prediction:
            return ""
        prediction = str(prediction).strip()

        # Try boxed format
        m = re.search(r"\\boxed{\s*([A-Da-d])\s*}", prediction)
        if m:
            return m.group(1).upper()

        return _fuzzy_matching(prediction)

    def _extract_mc_answer(self, prediction: str, choices: Optional[Dict[str, str]] = None) -> str:
        if not prediction:
            return ""
        prediction = str(prediction).strip()

        # Boxed
        m = re.search(r"\\boxed{\s*([A-Da-d])\s*}", prediction)
        if m:
            return m.group(1).upper()

        # Common patterns
        for pat in [r"\b([A-D])\.", r"\(([A-D])\)", r"\b([A-D]):"]:
            m = re.search(pat, prediction, re.IGNORECASE)
            if m:
                return m.group(1).upper()

        # First token
        first = _fuzzy_matching(prediction)
        if len(first) == 1 and first.upper() in "ABCD":
            return first.upper()

        # Match against choice texts
        if choices:
            pred_lower = prediction.lower().strip()
            for letter, text in choices.items():
                if pred_lower == text.lower().strip():
                    return letter.upper()

        return first

    def _extract_numerical_answer(self, prediction: str) -> Optional[float]:
        if not prediction:
            return None
        prediction = str(prediction).strip()

        # Try boxed format with number
        m = re.search(r"\\boxed{\s*([+-]?\d+\.?\d*)\s*}", prediction)
        if m:
            return _to_float(m.group(1))

        # Extract first number from text
        m = re.search(r"[+-]?\d+\.?\d*", prediction)
        if m:
            return _to_float(m.group(0))

        return _to_float(_fuzzy_matching(prediction))

    # ── evaluation ────────────────────────────────────────────────────────

    def evaluate_single(self, sample, prediction: str) -> float:
        """Score a single sample: accuracy for MCQ, MRA for numerical."""
        cat = sample.category
        pred_raw = prediction.strip() if prediction else ""
        if cat in MCQ_CATEGORIES:
            pred = self._extract_mc_answer(pred_raw, choices=sample.choices)
            gt = sample.answer.strip().upper()
            return 1.0 if pred.upper() == gt else 0.0
        elif cat in NUMERICAL_CATEGORIES:
            pred_f = self._extract_numerical_answer(pred_raw)
            gt_f = _to_float(sample.answer)
            if pred_f is not None and gt_f is not None:
                if cat in ZERO_AWARE_THRESHOLDS:
                    return _mean_relative_accuracy_consider_zero(
                        pred_f, gt_f, ZERO_AWARE_THRESHOLDS[cat]
                    )
                return _mean_relative_accuracy(pred_f, gt_f)
            return 0.0
        return 0.0

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        per_cat: Dict[str, List[float]] = {}
        detailed = []

        for sample in self.data:
            sid = sample.sample_id
            cat = sample.category
            pred_raw = get_prediction(predictions, sid)

            if cat not in per_cat:
                per_cat[cat] = []

            if cat in MCQ_CATEGORIES:
                pred = self._extract_mc_answer(pred_raw, choices=sample.choices)
                gt = sample.answer.strip().upper()
                score = 1.0 if pred.upper() == gt else 0.0
                extracted = pred
            elif cat in NUMERICAL_CATEGORIES:
                pred_f = self._extract_numerical_answer(pred_raw)
                gt_f = _to_float(sample.answer)
                if pred_f is not None and gt_f is not None:
                    if cat in ZERO_AWARE_THRESHOLDS:
                        score = _mean_relative_accuracy_consider_zero(
                            pred_f, gt_f, ZERO_AWARE_THRESHOLDS[cat]
                        )
                    else:
                        score = _mean_relative_accuracy(pred_f, gt_f)
                else:
                    score = 0.0
                extracted = str(pred_f) if pred_f is not None else ""
            else:
                score = 0.0
                extracted = pred_raw

            per_cat[cat].append(score)
            detailed.append({
                "id": sid,
                "category": cat,
                "question_type": sample.question_type,
                "ground_truth": sample.answer,
                "prediction": pred_raw,
                "extracted": extracted,
                "score": score,
            })

        # Per-category scores
        per_cat_scores: Dict[str, float] = {}
        for cat, scores in per_cat.items():
            per_cat_scores[cat] = float(np.mean(scores)) if scores else 0.0

        overall = mean_or_zero(d["score"] for d in detailed)

        results: Dict[str, Any] = {
            "total_samples": len(detailed),
            "correct_samples": sum(1 for d in detailed if d["score"] >= 1.0),
            "overall_accuracy": overall,
            "overall_accuracy_pct": overall * 100,
            "overall_aggregation": "weighted_mean_over_samples",
            "per_category_scores": {
                cat: {"score": s * 100, "count": len(per_cat.get(cat, []))}
                for cat, s in per_cat_scores.items()
            },
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*70}")
        print("OSI-Bench Evaluation Results")
        print(f"{'='*70}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Overall score: {results['overall_accuracy_pct']:.2f}")
        print(f"{'='*70}")

        # Display name mapping
        display_names = {
            "trajectory_description": "Trajectory Description",
            "relative_distance": "Relative Distance",
            "relative_direction_categorical_ordinal": "Relative Direction",
            "absolute_speed": "Absolute Speed",
            "absolute_displacement": "Absolute Displacement",
            "absolute_distance": "Absolute Distance",
            "object_3d_localization": "Object 3D Localization",
            "depth_aware_counting": "Depth-Aware Counting",
            "trajectory_length": "Trajectory Length",
        }

        # MCQ categories
        print("  MCQ (Accuracy):")
        for cat in MCQ_CATEGORIES:
            if cat in results.get("per_category_scores", {}):
                info = results["per_category_scores"][cat]
                label = display_names.get(cat, cat)
                print(f"    {label:30s} {info['score']:6.2f}  (n={info['count']})")

        # Numerical categories
        print("  Numerical (MRA):")
        for cat in NUMERICAL_CATEGORIES:
            if cat in results.get("per_category_scores", {}):
                info = results["per_category_scores"][cat]
                label = display_names.get(cat, cat)
                print(f"    {label:30s} {info['score']:6.2f}  (n={info['count']})")

        print(f"{'='*70}\n")
