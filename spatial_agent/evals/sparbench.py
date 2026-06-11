"""SPAR-Bench data loader.

SPAR-Bench: Spatial Perception And Reasoning Benchmark.
7,207 manually verified QA pairs across 20 spatial tasks.

Data structure:
    data/SPAR-Bench/data/test-00000-of-00004.parquet
    data/SPAR-Bench/data/test-00001-of-00004.parquet
    ...

Metrics:
    - Accuracy: for multiple-choice (select) questions
    - MRA (Mean Relative Accuracy): for numerical (fill) questions
    - VCI: for view_change_infer task

Reference: https://github.com/LogosRoboticsGroup/SPAR
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample, save_embedded_image
from spatial_agent.evals.scoring import get_prediction, write_results_summary

# ---------------------------------------------------------------------------
# Task taxonomy
# ---------------------------------------------------------------------------

MCA_TASKS = [
    "obj_spatial_relation_oo",
    "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv",
    "spatial_imagination_oc",
    "spatial_imagination_oo",
    "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv",
    "position_matching",
    "camera_motion_infer",
    "distance_infer_center_oo",
    "distance_infer_center_oo_mv",
]

NA_TASKS = [
    "depth_prediction_oc",
    "depth_prediction_oo",
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oc",
    "distance_prediction_oo",
    "distance_prediction_oc_mv",
    "distance_prediction_oo_mv",
]

VCI_TASKS = ["view_change_infer"]

# Multi-view NA tasks: take the LAST extracted number instead of first
MV_NA_TASKS = [
    "depth_prediction_oc_mv",
    "depth_prediction_oo_mv",
    "distance_prediction_oc_mv",
    "distance_prediction_oo_mv",
]

COGNITIVE_LEVELS = {
    "Low": [
        "depth_prediction_oc", "depth_prediction_oo",
        "depth_prediction_oc_mv", "depth_prediction_oo_mv",
        "distance_prediction_oc", "distance_prediction_oo",
        "distance_prediction_oc_mv", "distance_prediction_oo_mv",
    ],
    "Middle": [
        "view_change_infer", "position_matching", "camera_motion_infer",
    ],
    "High": [
        "obj_spatial_relation_oo", "obj_spatial_relation_oc_mv",
        "obj_spatial_relation_oo_mv",
        "spatial_imagination_oc", "spatial_imagination_oo",
        "spatial_imagination_oc_mv", "spatial_imagination_oo_mv",
        "distance_infer_center_oo", "distance_infer_center_oo_mv",
    ],
}


# ---------------------------------------------------------------------------
# Metric helpers (following SPAR eval_scripts/utils.py)
# ---------------------------------------------------------------------------

def _abs_dist_norm(pred: float, target: float) -> float:
    if target == 0.0:
        return abs(pred - target)
    return abs((pred - target) / target)


def _mean_relative_accuracy(
    pred: float, target: float,
    start: float = 0.5, end: float = 0.95, interval: float = 0.05,
) -> float:
    """MRA: average binary accuracy across confidence thresholds."""
    num_pts = int((end - start) / interval + 2)
    thresholds = np.linspace(start, end, num_pts)
    rel_err = _abs_dist_norm(pred, target)
    accuracy = (rel_err <= (1.0 - thresholds)).astype(float)
    return float(accuracy.mean())


def _exact_match(pred: str, target: str) -> float:
    """Lenient exact match for MCA questions."""
    pred = pred.strip().lower()
    target = target.strip().lower()
    if not pred:
        return 0.0
    if pred == target:
        return 1.0
    if pred in target:
        return 1.0
    if pred[0] == target:
        return 1.0
    return 0.0


def _extract_number(pred: str, task: str) -> Optional[float]:
    """Extract a numeric answer from prediction text."""
    numbers = re.findall(r"(?<!\^)\d+\.\d+|(?<!\^)\d+", pred)
    if not numbers:
        return None
    extracted = [float(n) if "." in n else int(n) for n in numbers]
    if task in MV_NA_TASKS:
        return float(extracted[-1])  # last number for multi-view
    return float(extracted[0])  # first number for single-view


VCI_AXES = ["move_right", "move_up", "move_forward", "turn_right", "turn_up"]
VCI_OPPOSITES = {
    "move_right": "move_left",
    "move_up": "move_down",
    "move_forward": "move_backward",
    "turn_right": "turn_left",
    "turn_up": "turn_down",
}


def _parse_vci(text: str) -> Dict[str, float]:
    """Parse view_change_infer structured answer: 'move_right:X,move_down:Y,...'"""
    result = {}
    for pair in text.replace(" ", "").split(","):
        if ":" not in pair:
            continue
        key, val = pair.split(":", 1)
        try:
            result[key] = float(val)
        except ValueError:
            continue
    # Collapse opposites into canonical axes
    collapsed = {}
    for axis, opposite in VCI_OPPOSITES.items():
        pos = result.get(axis, 0.0)
        neg = result.get(opposite, 0.0)
        collapsed[axis] = pos - neg
    return collapsed


def _vci_metric(pred_text: str, target_text: str) -> float:
    """VCI metric: MRA averaged across 5 axes."""
    pred = _parse_vci(pred_text)
    target = _parse_vci(target_text)
    scores = []
    for axis in VCI_AXES:
        p = pred.get(axis, 0.0)
        t = target.get(axis, 0.0)
        scores.append(_mean_relative_accuracy(p, t))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Sample and Benchmark
# ---------------------------------------------------------------------------

@dataclass
class SPARBenchSample(BaseBenchmarkSample):
    """A single SPAR-Bench sample."""

    img_type: str = ""       # single_view / multi_view
    format_type: str = ""    # select / fill
    task: str = ""
    source: str = ""


class SPARBench(BaseBenchmark):
    """SPAR-Bench loader.

    Reads HuggingFace-format parquet files with embedded images.
    Images are extracted to a temporary directory on first load.
    """

    data_specific_prompt = (
        "Answer the spatial reasoning question. "
        "For multiple-choice questions, answer with a single letter (A, B, C, or D). "
        "For numerical questions, answer with a single number."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None, **kwargs):
        self._image_dir: Optional[str] = None
        super().__init__(data_path, question_type, **kwargs)

    def read_data(self) -> None:
        import pandas as pd

        parquet_dir = os.path.join(self.data_path, "data")
        if not os.path.isdir(parquet_dir):
            print(f"[Warning] SPAR-Bench data dir not found at {parquet_dir}")
            return

        # Find all parquet files
        parquet_files = sorted([
            os.path.join(parquet_dir, f)
            for f in os.listdir(parquet_dir)
            if f.endswith(".parquet")
        ])
        if not parquet_files:
            print(f"[Warning] No parquet files found in {parquet_dir}")
            return

        # Create image cache directory alongside the data
        self._image_dir = os.path.join(self.data_path, ".image_cache")
        os.makedirs(self._image_dir, exist_ok=True)

        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
        print(f"[SPAR-Bench] Loaded {len(df)} samples from {len(parquet_files)} parquet files")

        for idx, row in df.iterrows():
            task = row.get("task", "")
            if self.question_type_filter and task not in self.question_type_filter:
                continue

            # Extract embedded images to disk
            image_paths = []
            images_col = row.get("image", [])
            if images_col is not None:
                for img_idx, img_data in enumerate(images_col):
                    img_filename = f"{row['id']}_{img_idx}.jpg"
                    img_path = os.path.join(self._image_dir, img_filename)
                    save_embedded_image(img_path, img_data, convert_rgb=True)
                    image_paths.append(img_path)

            sample = SPARBenchSample(
                sample_id=row["id"],
                question=row.get("question", ""),
                question_type=task,
                images=image_paths,
                answer=str(row.get("answer", "")),
                img_type=row.get("img_type", ""),
                format_type=row.get("format_type", ""),
                task=task,
                source=row.get("source", ""),
            )
            self.data.append(sample)

        print(f"[SPAR-Bench] {len(self.data)} samples after filtering")

    def extract_answer(self, prediction: str) -> str:
        """Extract answer — return raw text, evaluation handles type-specific parsing."""
        return prediction.strip() if prediction else ""

    def evaluate_single(self, sample, prediction: str) -> float:
        """Score a single sample: accuracy for MCA, MRA for NA, VCI metric."""
        task = sample.task
        gt = sample.answer.strip()
        pred_raw = prediction.strip() if prediction else ""
        if task in MCA_TASKS:
            return _exact_match(self._extract_mca(pred_raw), gt)
        elif task in NA_TASKS:
            pred_num = _extract_number(pred_raw, task)
            try:
                gt_num = float(gt)
            except ValueError:
                gt_num = 0.0
            if pred_num is not None:
                return _mean_relative_accuracy(pred_num, gt_num)
            return 0.0
        elif task in VCI_TASKS:
            return _vci_metric(pred_raw, gt)
        else:
            return _exact_match(self._extract_mca(pred_raw), gt)

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        per_task_scores: Dict[str, List[float]] = {}
        detailed = []

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            task = sample.task
            gt = sample.answer.strip()

            # Compute score based on task type
            if task in MCA_TASKS:
                pred_extracted = self._extract_mca(pred_raw)
                score = _exact_match(pred_extracted, gt)
                metric_type = "accuracy"
            elif task in NA_TASKS:
                pred_num = _extract_number(pred_raw, task)
                try:
                    gt_num = float(gt)
                except ValueError:
                    gt_num = 0.0
                if pred_num is not None:
                    score = _mean_relative_accuracy(pred_num, gt_num)
                else:
                    score = 0.0
                pred_extracted = str(pred_num) if pred_num is not None else ""
                metric_type = "mra"
            elif task in VCI_TASKS:
                score = _vci_metric(pred_raw, gt)
                pred_extracted = pred_raw.strip()
                metric_type = "vci"
            else:
                # Unknown task, try MCA
                pred_extracted = self._extract_mca(pred_raw)
                score = _exact_match(pred_extracted, gt)
                metric_type = "accuracy"

            per_task_scores.setdefault(task, []).append(score)
            detailed.append({
                "id": sid, "task": task, "metric": metric_type,
                "ground_truth": gt, "prediction": pred_raw,
                "extracted": pred_extracted, "score": score,
            })

        # Per-task averages
        per_task_avg = {
            task: float(np.mean(scores))
            for task, scores in per_task_scores.items()
        }

        # Overall = mean of per-task averages (not per-sample)
        overall = float(np.mean(list(per_task_avg.values()))) if per_task_avg else 0.0

        # Cognitive level averages
        level_scores = {}
        for level, tasks in COGNITIVE_LEVELS.items():
            level_task_avgs = [per_task_avg[t] for t in tasks if t in per_task_avg]
            level_scores[level] = float(np.mean(level_task_avgs)) if level_task_avgs else 0.0

        results = {
            "total_samples": len(detailed),
            "overall_score": overall * 100.0,
            "level_scores": {k: v * 100.0 for k, v in level_scores.items()},
            "per_task": {
                task: {
                    "score": avg * 100.0,
                    "count": len(per_task_scores[task]),
                    "metric": "accuracy" if task in MCA_TASKS
                             else "mra" if task in NA_TASKS
                             else "vci" if task in VCI_TASKS
                             else "accuracy",
                }
                for task, avg in per_task_avg.items()
            },
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*70}")
        print(f"SPAR-Bench Results")
        print(f"{'='*70}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Overall score: {results['overall_score']:.2f}")
        print()

        # Cognitive levels
        print("Cognitive Level Scores:")
        for level in ["Low", "Middle", "High"]:
            score = results.get("level_scores", {}).get(level, 0.0)
            print(f"  {level:8s}: {score:.2f}")
        print()

        # Per-task breakdown
        print(f"{'Task':<35s} {'Metric':>8s} {'Score':>8s} {'Count':>6s}")
        print(f"{'-'*35} {'-'*8} {'-'*8} {'-'*6}")
        for task, info in sorted(results.get("per_task", {}).items()):
            print(f"{task:<35s} {info['metric']:>8s} {info['score']:>7.2f}% {info['count']:>5d}")
        print(f"{'='*70}\n")

    def _extract_mca(self, prediction: str) -> str:
        """Extract multiple-choice answer letter."""
        if not prediction:
            return ""
        pred = prediction.strip()
        # Try \boxed{X}
        m = re.search(r"\\boxed\{([A-Da-d])\}", pred)
        if m:
            return m.group(1).upper()
        # Try (A), A., A:, A)
        m = re.search(r"\(?([A-Da-d])\)?[\.\:\)]", pred)
        if m:
            return m.group(1).upper()
        # Single letter
        clean = re.sub(r"[^A-Da-d]", "", pred)
        if len(clean) == 1:
            return clean.upper()
        # First letter if starts with A-D
        if pred and pred[0].upper() in "ABCD":
            return pred[0].upper()
        return pred
