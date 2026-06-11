"""VSTIBench (Video Spatial-Temporal Intelligence Bench) data loader.

Data structure:
    data/vstibench/test.json           (6,042 samples)
    data/vstibench/ScanNet/videos/val/*.mp4

7 MCA (multiple-choice) question types:
    obj_obj_relative_pos_nf (near/far)
    obj_obj_relative_pos_ud (up/down)
    obj_obj_relative_pos_lr (left/right)
    camera_obj_rel_dist_v1
    camera_obj_rel_dist_v2
    camera_obj_rel_dist_v3
    camera_movement_direction

2 NA (numerical answer) question types:
    camera_obj_abs_dist
    camera_displacement

Evaluation follows the VLM-3R / thinking-in-space protocol:
    - MCA: case-insensitive exact match of first token (fuzzy_matching)
    - NA: Mean Relative Accuracy (MRA) with the official threshold formula
    - Overall = mean of per-question-type scores × 100

Reference: https://github.com/VITA-Group/VLM-3R
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


import numpy as np

from spatial_agent.evals.base import BaseBenchmark, LazyVideoSample, VideoFrameBenchmarkMixin
from spatial_agent.evals.scoring import (
    get_prediction,
    write_json,
    write_results_summary,
)
from spatial_agent.config import get_config


MCA_QUESTION_TYPES = [
    "obj_obj_relative_pos_nf",
    "obj_obj_relative_pos_ud",
    "obj_obj_relative_pos_lr",
    "camera_obj_rel_dist_v1",
    "camera_obj_rel_dist_v2",
    "camera_obj_rel_dist_v3",
    "camera_movement_direction",
]

NA_QUESTION_TYPES = [
    "camera_obj_abs_dist",
    "camera_displacement",
    "camera_obj_dist_change",
]


# ── metrics (matching VLM-3R implementation) ──────────────────────────────

def _fuzzy_matching(pred: str) -> str:
    """Extract first token, strip trailing period. Matches VLM-3R."""
    return pred.split(" ")[0].rstrip(".").strip()


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _abs_dist_norm(pred: float, target: float) -> float:
    return abs(pred - target) / target


def _mean_relative_accuracy(
    pred: float,
    target: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    """MRA as defined in VSTIBench: fraction of threshold levels passed."""
    num_pts = int((end - start) / interval + 2)
    thresholds = np.linspace(start, end, num_pts)
    accuracy = _abs_dist_norm(pred, target) <= 1 - thresholds
    return float(accuracy.mean())


# ── data types ────────────────────────────────────────────────────────────

@dataclass
class VSTIBenchSample(LazyVideoSample):
    """VSTIBench sample with video and optional MC options."""

    choices: Optional[List[str]] = None
    mc_answer: Optional[str] = None
    ground_truth_text: str = ""


# ── benchmark ─────────────────────────────────────────────────────────────

class VSTIBench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """VSTIBench loader (6,042 video-based spatial-temporal reasoning questions).

    Evaluation follows the VLM-3R protocol with MCA accuracy and NA MRA.
    """

    data_specific_prompt = (
        "## VSTIBench frame-reference convention\n"
        "Questions reference points in the **original video** using 'frame X of N' "
        "(e.g. 'frame 23 of 32', 'frame 6 of 31'). Here N is the question's "
        "uniform-sampling protocol count and X is 1-indexed.\n"
        "\n"
        "**'frame X of N' is NOT `InputImages[X]`.** The original video typically has "
        "hundreds-to-thousands of frames, while `InputImages` is a much smaller, "
        "independent sampling. Never index `InputImages` with the literal number from "
        "the question. Convert first:\n"
        "```python\n"
        "# Parse X and N from the question text (X is 1-indexed).\n"
        "fraction = (X - 1) / max(N - 1, 1)\n"
        "target_video_frame = int(round(fraction * (Metadata.total_frames - 1)))\n"
        "# Pick the closest available frame in InputImages:\n"
        "i = min(range(len(InputImages)),\n"
        "        key=lambda j: abs(InputImages.frame_indices[j] - target_video_frame))\n"
        "frame = InputImages[i]\n"
        "```\n"
        "For ranges ('between frame A and frame B of N'), convert each endpoint to a "
        "video frame index, then take the `InputImages` slice whose `frame_indices` "
        "fall inside that range.\n"
        "\n"
        "## Answer format\n"
        "- Multiple-choice questions: answer with the option letter (A, B, C, or D).\n"
        "- Open-ended numerical questions: answer with a single number."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._config = get_config()
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        test_path = os.path.join(self.data_path, "test.json")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"VSTIBench test.json not found at {test_path}")

        with open(test_path, "r") as f:
            items = json.load(f)

        for item in items:
            qtype = item.get("question_type", "")
            if self.question_type_filter and qtype not in self.question_type_filter:
                continue

            sample_id = item["id"]
            video_path = os.path.join(self.data_path, item["video_path"])
            raw_options = item.get("options")  # list of "A. ...", or None for NA
            # Strip letter prefixes (e.g. "A. foo" → "foo") so run.py can format uniformly
            if raw_options is not None:
                options = [re.sub(r"^[A-Z]\.\s*", "", o) for o in raw_options]
            else:
                options = None
            mc_answer = item.get("mc_answer")  # "A"/"B"/"C"/"D" or None
            ground_truth = str(item.get("ground_truth", ""))

            # For MCA: answer is the letter; for NA: answer is the number string
            is_mca = options is not None
            if is_mca:
                answer = mc_answer or ""
            else:
                answer = ground_truth
            question = item["question"]

            self.data.append(
                VSTIBenchSample(
                    sample_id=sample_id,
                    question=question,
                    question_type=qtype,
                    images=[],
                    answer=answer,
                    video=video_path,
                    choices=options,
                    mc_answer=mc_answer,
                    ground_truth_text=ground_truth,
                    _bench_ref=self,
                )
            )

    def extract_answer(self, prediction: str) -> str:
        """Extract answer using VLM-3R fuzzy_matching (first token)."""
        if not prediction:
            return ""
        return _fuzzy_matching(prediction)

    def evaluate_single(self, sample, prediction: str) -> float:
        """Score a single sample: accuracy for MCA, MRA for NA."""
        is_mca = sample.choices is not None
        extracted = self.extract_answer(prediction)
        if is_mca:
            gt = sample.mc_answer or sample.answer
            return 1.0 if extracted.upper() == gt.upper() else 0.0
        else:
            pred_f = _to_float(extracted)
            gt_f = _to_float(sample.answer)
            if pred_f is not None and gt_f is not None and gt_f != 0:
                return _mean_relative_accuracy(pred_f, gt_f)
            return 0.0

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        # Per-question-type accumulation
        per_type_scores: Dict[str, List[float]] = {}
        details = []

        for sample in self.data:
            sample_id = sample.sample_id
            raw_pred = get_prediction(predictions, sample_id)
            qtype = sample.question_type
            is_mca = sample.choices is not None

            if qtype not in per_type_scores:
                per_type_scores[qtype] = []

            extracted = self.extract_answer(raw_pred)

            if is_mca:
                gt = sample.mc_answer or sample.answer
                score = 1.0 if extracted.upper() == gt.upper() else 0.0
            else:
                # NA: MRA
                pred_f = _to_float(extracted)
                gt_f = _to_float(sample.answer)
                if pred_f is not None and gt_f is not None and gt_f != 0:
                    score = _mean_relative_accuracy(pred_f, gt_f)
                else:
                    score = 0.0

            per_type_scores[qtype].append(score)
            details.append(
                {
                    "sample_id": sample_id,
                    "question_type": qtype,
                    "is_mca": is_mca,
                    "prediction": raw_pred,
                    "extracted": extracted,
                    "ground_truth": sample.answer,
                    "score": score,
                }
            )

        # Per-type mean
        per_type_mean = {
            qt: float(np.mean(scores)) if scores else 0.0
            for qt, scores in sorted(per_type_scores.items())
        }

        # Overall = mean of per-type means × 100
        overall = float(np.mean(list(per_type_mean.values()))) * 100 if per_type_mean else 0.0

        # Counts
        total = sum(len(s) for s in per_type_scores.values())
        per_type_counts = {
            qt: {"total": len(scores), "mean_score": per_type_mean[qt]}
            for qt, scores in sorted(per_type_scores.items())
        }

        results = {
            "overall_score": overall,
            "total_samples": total,
            "per_question_type": per_type_mean,
            "per_question_type_counts": per_type_counts,
            "overall_aggregation": "mean_of_question_type_scores",
            "detailed_results": details,
        }

        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), details)

        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*65}")
        print("VSTIBench Results")
        print(f"{'='*65}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Overall score: {results['overall_score']:.2f}")
        print()

        per_type = results.get("per_question_type", {})
        per_counts = results.get("per_question_type_counts", {})

        # Display MCA types first, then NA
        mca_types = [qt for qt in MCA_QUESTION_TYPES if qt in per_type]
        na_types = [qt for qt in NA_QUESTION_TYPES if qt in per_type]
        # Include any types not in predefined lists
        known = set(MCA_QUESTION_TYPES + NA_QUESTION_TYPES)
        other_types = [qt for qt in sorted(per_type) if qt not in known]

        print(f"  {'Question Type':<30} {'Metric':<6} {'Score':>8}  {'N':>5}")
        print(f"  {'-'*55}")

        for qt in mca_types:
            n = per_counts.get(qt, {}).get("total", 0)
            print(f"  {qt:<30} {'Acc':<6} {per_type[qt]:>8.4f}  {n:>5}")

        if mca_types and na_types:
            print(f"  {'-'*55}")

        for qt in na_types:
            n = per_counts.get(qt, {}).get("total", 0)
            print(f"  {qt:<30} {'MRA':<6} {per_type[qt]:>8.4f}  {n:>5}")

        for qt in other_types:
            n = per_counts.get(qt, {}).get("total", 0)
            metric = "MRA" if qt in NA_QUESTION_TYPES else "Acc"
            print(f"  {qt:<30} {metric:<6} {per_type[qt]:>8.4f}  {n:>5}")

        print(f"{'='*65}\n")
