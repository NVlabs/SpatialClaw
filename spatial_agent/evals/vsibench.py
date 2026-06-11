"""VSI-Bench benchmark data loader.

Data structure:
    data/VSI-Bench/test.jsonl                    (5130 samples, original)
    data/VSI-Bench/test_debiased.parquet         (2362 samples, unbiased variant)
    data/VSI-Bench/{scannet,scannetpp,arkitscenes}/*.mp4

10 question types in 2 categories:
    MCA (multiple-choice, scored by accuracy):
        object_rel_direction_easy, object_rel_direction_medium,
        object_rel_direction_hard, object_rel_distance,
        route_planning, obj_appearance_order
    NA (numerical answer, scored by Mean Relative Accuracy):
        object_abs_distance, object_counting,
        object_size_estimation, room_size_estimation

Evaluation follows the official VSI-Bench protocol:
    - MCA: case-insensitive exact match of first token
    - NA: MRA with thresholds linspace(0.5, 0.95, 11)
    - object_rel_direction_{easy,medium,hard} averaged into one score
    - Overall = mean of 8 per-task scores (×100)

Variants:
    - "original": full test set from test.jsonl (5130 samples)
    - "unbiased": debiased subset from test_debiased.parquet (2362 samples)

Reference: https://github.com/vision-x-nyu/thinking-in-space
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


MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]

NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]


# ── metrics (matching official implementation) ───────────────────────────

def _mean_relative_accuracy(
    pred: float, target: float,
    start: float = 0.5, end: float = 0.95, interval: float = 0.05,
) -> float:
    """MRA as defined in VSI-Bench: fraction of threshold levels passed."""
    return mean_relative_accuracy(
        pred,
        target,
        start=start,
        end=end,
        interval=interval,
    )


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _fuzzy_matching(pred: str) -> str:
    """Official extraction: first space-delimited token, strip trailing period."""
    return pred.split(" ")[0].rstrip(".").strip()


def _parse_options(raw: Optional[list]) -> Dict[str, str]:
    """Parse ["A. front-left", "B. back-right"] into {"A": "front-left", ...}."""
    if not raw:
        return {}
    result = {}
    for opt in raw:
        m = re.match(r"([A-Da-d])[.)]\s*(.*)", str(opt))
        if m:
            result[m.group(1).upper()] = m.group(2).strip()
        else:
            # Fallback: assign sequential letters
            letter = chr(65 + len(result))
            result[letter] = str(opt).strip()
    return result


# ── sample & benchmark ───────────────────────────────────────────────────

@dataclass
class VSIBenchSample(LazyVideoSample):
    """VSI-Bench sample with video and optional MC options."""

    dataset_source: str = ""  # scannet / scannetpp / arkitscenes
    scene_name: str = ""
    choices: Dict[str, str] = field(default_factory=dict)


class VSIBench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """VSI-Bench loader (5130 video-based spatial reasoning questions).

    Evaluation follows the official VSI-Bench protocol with MCA accuracy
    and NA Mean Relative Accuracy (MRA).
    """

    data_specific_prompt = (
        "These are frames of a video.\n"
        "Answer the question. For multiple-choice, answer with the option's "
        "letter from the given choices directly.\n"
        "For numerical questions, answer with a single number."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None, variant: str = "original"):
        self._variant = variant
        self._config = get_config()
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)

        if self._variant == "unbiased":
            self._read_parquet("test_debiased.parquet")
        else:
            self._read_jsonl("test.jsonl")

    def _read_jsonl(self, filename: str) -> None:
        jsonl_path = os.path.join(self.data_path, filename)
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"VSI-Bench data not found: {jsonl_path}")

        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self._add_sample(item)

    def _read_parquet(self, filename: str) -> None:
        parquet_path = os.path.join(self.data_path, filename)
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"VSI-Bench data not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            item = row.to_dict()
            # Parquet stores options as None for NA types; normalize to list/None
            if item.get("options") is not None and not isinstance(item["options"], list):
                item["options"] = list(item["options"])
            self._add_sample(item)

    def _add_sample(self, item: dict) -> None:
        qtype = item["question_type"]
        if self.question_type_filter and qtype not in self.question_type_filter:
            return

        dataset_src = item["dataset"]
        scene = item["scene_name"]
        video_path = os.path.join(self.data_path, dataset_src, f"{scene}.mp4")

        sample = VSIBenchSample(
            sample_id=item["id"],
            question=item["question"],
            question_type=qtype,
            images=[],
            answer=str(item["ground_truth"]).strip(),
            video=video_path,
            dataset_source=dataset_src,
            scene_name=scene,
            choices=_parse_options(item.get("options")),
            _bench_ref=self,
        )
        self.data.append(sample)

    # ── answer extraction ────────────────────────────────────────────────

    def extract_answer(self, prediction: str) -> str:
        """Extract answer from prediction.

        For MCA: extract letter (A/B/C/D).
        For NA: extract first token (number).
        Uses boxed format first, then fuzzy matching as fallback.
        """
        if not prediction:
            return ""
        prediction = str(prediction).strip()

        # Try boxed format first
        m = re.search(r"\\boxed{\s*([A-Da-d])\s*}", prediction)
        if m:
            return m.group(1).upper()

        # Fuzzy matching (official): first token, strip trailing period
        return _fuzzy_matching(prediction)

    def _extract_mc_answer(self, prediction: str, choices: Optional[Dict[str, str]] = None) -> str:
        """Extract MC letter specifically.

        Tries letter extraction first, then falls back to matching the
        prediction text against the choice texts (case-insensitive).
        """
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

        # Fuzzy: first token
        first = _fuzzy_matching(prediction)
        if len(first) == 1 and first.upper() in "ABCD":
            return first.upper()

        # Match prediction text against choice texts
        if choices:
            pred_lower = prediction.lower().strip()
            for letter, text in choices.items():
                if pred_lower == text.lower().strip():
                    return letter.upper()

        return first

    # ── evaluation ───────────────────────────────────────────────────────

    def evaluate_single(self, sample, prediction: str) -> float:
        """Score a single sample: accuracy for MCA, MRA for NA."""
        qtype = sample.question_type
        pred_raw = prediction.strip() if prediction else ""
        if qtype in MCA_QUESTION_TYPES:
            pred = self._extract_mc_answer(pred_raw, choices=sample.choices)
            gt = sample.answer.strip().upper()
            return 1.0 if pred.lower() == gt.lower() else 0.0
        elif qtype in NA_QUESTION_TYPES:
            pred = _fuzzy_matching(pred_raw) if pred_raw else ""
            pred_f = _to_float(pred)
            gt_f = _to_float(sample.answer)
            if pred_f is not None and gt_f is not None and gt_f != 0:
                return _mean_relative_accuracy(pred_f, gt_f)
            return 0.0
        return 0.0

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate following official VSI-Bench protocol."""

        # Per question-type accumulators
        per_qtype: Dict[str, List[float]] = {}
        detailed = []

        for sample in self.data:
            sid = sample.sample_id
            qtype = sample.question_type
            pred_raw = get_prediction(predictions, sid)

            if qtype not in per_qtype:
                per_qtype[qtype] = []

            if qtype in MCA_QUESTION_TYPES:
                pred = self._extract_mc_answer(pred_raw, choices=sample.choices)
                gt = sample.answer.strip().upper()
                score = 1.0 if pred.lower() == gt.lower() else 0.0
            elif qtype in NA_QUESTION_TYPES:
                pred = _fuzzy_matching(pred_raw) if pred_raw else ""
                pred_f = _to_float(pred)
                gt_f = _to_float(sample.answer)
                if pred_f is not None and gt_f is not None and gt_f != 0:
                    score = _mean_relative_accuracy(pred_f, gt_f)
                else:
                    score = 0.0
            else:
                pred = pred_raw
                score = 0.0

            per_qtype[qtype].append(score)
            detailed.append({
                "id": sid,
                "question_type": qtype,
                "ground_truth": sample.answer,
                "prediction": pred_raw,
                "extracted": pred if qtype in NA_QUESTION_TYPES else (
                    self._extract_mc_answer(pred_raw, choices=sample.choices) if qtype in MCA_QUESTION_TYPES else pred_raw
                ),
                "score": score,
            })

        # Aggregate per-task-type
        per_task_scores: Dict[str, float] = {}
        for qtype, scores in per_qtype.items():
            per_task_scores[qtype] = float(np.mean(scores)) if scores else 0.0

        # Merge direction easy/medium/hard into one score
        dir_keys = [
            "object_rel_direction_easy",
            "object_rel_direction_medium",
            "object_rel_direction_hard",
        ]
        dir_scores = [per_task_scores.pop(k) for k in dir_keys if k in per_task_scores]
        if dir_scores:
            per_task_scores["object_rel_direction"] = float(np.mean(dir_scores))

        # Overall = mean of all task scores
        overall = float(np.mean(list(per_task_scores.values()))) if per_task_scores else 0.0

        results: Dict[str, Any] = {
            "total_samples": len(detailed),
            "correct_samples": sum(1 for d in detailed if d["score"] >= 1.0),
            "overall_accuracy": overall,
            "overall_accuracy_pct": overall * 100,
            "per_task_scores": {
                k: {"score": v * 100, "count": len(per_qtype.get(k, per_qtype.get(k + "_easy", [])))}
                for k, v in per_task_scores.items()
            },
            "detailed_results": detailed,
        }

        # Fix count for merged direction
        if "object_rel_direction" in results["per_task_scores"]:
            results["per_task_scores"]["object_rel_direction"]["count"] = sum(
                len(per_qtype.get(k, [])) for k in dir_keys
            )

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*70}")
        print("VSI-Bench Evaluation Results")
        print(f"{'='*70}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Overall score: {results['overall_accuracy_pct']:.2f}")
        print(f"{'='*70}")

        # Canonical display order
        display_order = [
            ("object_counting", "Object Counting (MRA)"),
            ("object_abs_distance", "Abs Distance (MRA)"),
            ("object_size_estimation", "Object Size (MRA)"),
            ("room_size_estimation", "Room Size (MRA)"),
            ("object_rel_distance", "Rel Distance (Acc)"),
            ("object_rel_direction", "Rel Direction (Acc)"),
            ("route_planning", "Route Planning (Acc)"),
            ("obj_appearance_order", "Appearance Order (Acc)"),
        ]
        for key, label in display_order:
            if key in results.get("per_task_scores", {}):
                info = results["per_task_scores"][key]
                print(f"  {label:30s} {info['score']:6.2f}  (n={info['count']})")
        print(f"{'='*70}\n")
