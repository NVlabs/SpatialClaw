"""SPBench data loader.

Data structure:
    data/SPBench/SPBench-SI.parquet   (1009 single-image samples)
    data/SPBench/SPBench-MV.parquet   (319 multi-view samples)
    data/SPBench/{scene_name}/*.jpg   (ScanNet images)

3D spatial perception benchmark from ScanNet scenes. Evaluates distance,
size, direction, and counting from single or multi-view images.

Question types (same taxonomy as VSI-Bench):
    MCA: object_rel_direction, object_rel_distance
    NA:  object_abs_distance, object_size_estimation, object_counting

Metrics:
    MCA: exact match accuracy
    NA:  Mean Relative Accuracy (MRA) with thresholds 0.5-0.95

Reference: https://github.com/ZJU-REAL/SpatialLadder
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.scoring import get_prediction, write_results_summary


# Question type classification (from VSI-Bench / SpatialLadder)
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_direction",
    "object_rel_distance",
]

NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]


# ── Metrics ──────────────────────────────────────────────────────────────

def _abs_dist_norm(pred: float, target: float) -> float:
    return abs(pred - target) / target


def _mean_relative_accuracy(
    pred: float, target: float,
    start: float = 0.5, end: float = 0.95, interval: float = 0.05,
) -> float:
    num_pts = int((end - start) / interval + 2)
    thresholds = np.linspace(start, end, num_pts)
    accuracy = _abs_dist_norm(pred, target) <= 1 - thresholds
    return float(accuracy.mean())


def _to_float(s) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _fuzzy_matching_mc(pred: str) -> str:
    m = re.search(r"^[A-Da-d]\.?$", pred.split(" ")[0].strip())
    if m:
        return m.group(0).rstrip(".").upper()
    return pred.strip()


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


def _parse_options(raw) -> Dict[str, str]:
    """Parse options from parquet (list of 'A. text' strings or None)."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        # Try parsing as JSON or eval-style list
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        return {}
    result = {}
    for opt in raw:
        m = re.match(r"([A-Da-d])[.)]\s*(.*)", str(opt))
        if m:
            result[m.group(1).upper()] = m.group(2).strip()
        else:
            letter = chr(65 + len(result))
            result[letter] = str(opt).strip()
    return result


# ── Sample & Benchmark ───────────────────────────────────────────────────

@dataclass
class SPBenchSample(BaseBenchmarkSample):
    """SPBench sample."""

    scene_name: str = ""
    choices: Dict[str, str] = field(default_factory=dict)
    subset: str = ""  # "SI" or "MV"


class SPBench(BaseBenchmark):
    """SPBench loader (1328 spatial perception samples from ScanNet).

    Combines SI (single-image) and MV (multi-view) subsets.
    Uses VSI-Bench metrics: accuracy for MCA, MRA for NA.
    """

    data_specific_prompt = (
        "Answer the spatial reasoning question about the 3D scene. "
        "For multiple-choice, answer with the option's letter directly. "
        "For numerical questions, answer with a single number."
    )

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)

        for subset, filename in [("SI", "SPBench-SI.parquet"), ("MV", "SPBench-MV.parquet")]:
            parquet_path = os.path.join(self.data_path, filename)
            if not os.path.exists(parquet_path):
                continue

            df = pd.read_parquet(parquet_path)
            for _, row in df.iterrows():
                qtype = row.get("question_type", "")
                if self.question_type_filter and qtype not in self.question_type_filter:
                    continue

                scene = row.get("scene_name", "")
                image_names = row.get("images", [])
                if hasattr(image_names, "tolist"):
                    image_names = image_names.tolist()

                image_paths = [
                    os.path.join(self.data_path, scene, img)
                    for img in image_names
                ]

                choices = _parse_options(row.get("options"))

                sample = SPBenchSample(
                    sample_id=f"{subset}_{row['id']}",
                    question=row.get("question", ""),
                    question_type=qtype,
                    images=image_paths,
                    answer=str(row.get("ground_truth", "")).strip(),
                    scene_name=scene,
                    choices=choices,
                    subset=subset,
                )
                self.data.append(sample)

    def extract_answer(self, prediction: str) -> str:
        if not prediction:
            return ""
        prediction = str(prediction).strip()
        m = re.search(r"\\boxed{\s*([A-Da-d])\s*}", prediction)
        if m:
            return m.group(1).upper()
        return _fuzzy_matching_mc(prediction)

    def _extract_mc_answer(self, prediction: str, choices: Optional[Dict[str, str]] = None) -> str:
        if not prediction:
            return ""
        prediction = str(prediction).strip()

        m = re.search(r"\\boxed{\s*([A-Da-d])\s*}", prediction)
        if m:
            return m.group(1).upper()

        for pat in [r"\b([A-D])\.", r"\(([A-D])\)", r"\b([A-D]):"]:
            m = re.search(pat, prediction, re.I)
            if m:
                return m.group(1).upper()

        first = _fuzzy_matching_mc(prediction)
        if len(first) == 1 and first.upper() in "ABCD":
            return first.upper()

        if choices:
            pred_lower = prediction.lower().strip()
            for letter, text in choices.items():
                if pred_lower == text.lower().strip():
                    return letter.upper()

        return first

    def evaluate_single(self, sample, prediction: str) -> float:
        """Score a single sample: accuracy for MCA, MRA for NA."""
        qtype = sample.question_type
        pred_raw = prediction.strip() if prediction else ""
        if qtype in MCA_QUESTION_TYPES:
            pred = self._extract_mc_answer(pred_raw, sample.choices)
            gt = sample.answer.strip().upper()
            return 1.0 if pred.lower() == gt.lower() else 0.0
        elif qtype in NA_QUESTION_TYPES:
            pred_str = _fuzzy_matching_num(pred_raw) if pred_raw else None
            pred_f = _to_float(pred_str)
            gt_f = _to_float(sample.answer)
            if pred_f is not None and gt_f is not None and gt_f != 0:
                return _mean_relative_accuracy(pred_f, gt_f)
            return 0.0
        return 0.0

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        per_qtype: Dict[str, List[float]] = {}
        per_subset: Dict[str, List[float]] = {}
        detailed = []

        for sample in self.data:
            sid = sample.sample_id
            qtype = sample.question_type
            pred_raw = get_prediction(predictions, sid)

            if qtype not in per_qtype:
                per_qtype[qtype] = []

            if qtype in MCA_QUESTION_TYPES:
                pred = self._extract_mc_answer(pred_raw, sample.choices)
                gt = sample.answer.strip().upper()
                score = 1.0 if pred.lower() == gt.lower() else 0.0
            elif qtype in NA_QUESTION_TYPES:
                pred_str = _fuzzy_matching_num(pred_raw) if pred_raw else None
                pred_f = _to_float(pred_str)
                gt_f = _to_float(sample.answer)
                if pred_f is not None and gt_f is not None and gt_f != 0:
                    score = _mean_relative_accuracy(pred_f, gt_f)
                else:
                    score = 0.0
            else:
                score = 0.0

            per_qtype[qtype].append(score)
            per_subset.setdefault(sample.subset, []).append(score)

            detailed.append({
                "id": sid, "subset": sample.subset,
                "question_type": qtype,
                "ground_truth": sample.answer,
                "prediction": pred_raw, "score": score,
            })

        # Aggregate per-task scores
        per_task_scores = {
            k: float(np.mean(v)) for k, v in per_qtype.items() if v
        }

        overall = float(np.mean(list(per_task_scores.values()))) if per_task_scores else 0.0

        results = {
            "total_samples": len(detailed),
            "correct_samples": sum(1 for d in detailed if d["score"] >= 1.0),
            "overall_accuracy": overall,
            "overall_score_pct": overall * 100,
            "per_task_scores": {
                k: {"score": v * 100, "count": len(per_qtype.get(k, []))}
                for k, v in per_task_scores.items()
            },
            "per_subset": {
                k: {"score": float(np.mean(v)) * 100, "count": len(v)}
                for k, v in per_subset.items()
            },
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*70}")
        print("SPBench Evaluation Results")
        print(f"{'='*70}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Overall score: {results['overall_score_pct']:.2f}")
        print(f"\n--- Per Subset ---")
        for k, v in results.get("per_subset", {}).items():
            print(f"  {k:10s} {v['score']:6.2f}  (n={v['count']})")
        print(f"\n--- Per Task ---")
        display_order = [
            ("object_counting", "Object Counting (MRA)"),
            ("object_abs_distance", "Abs Distance (MRA)"),
            ("object_size_estimation", "Object Size (MRA)"),
            ("object_rel_distance", "Rel Distance (Acc)"),
            ("object_rel_direction", "Rel Direction (Acc)"),
        ]
        for key, label in display_order:
            if key in results.get("per_task_scores", {}):
                info = results["per_task_scores"][key]
                print(f"  {label:30s} {info['score']:6.2f}  (n={info['count']})")
        # Print any remaining
        shown = {k for k, _ in display_order}
        for key, info in results.get("per_task_scores", {}).items():
            if key not in shown:
                print(f"  {key:30s} {info['score']:6.2f}  (n={info['count']})")
        print(f"{'='*70}\n")
