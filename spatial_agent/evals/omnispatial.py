"""OmniSpatial benchmark data loader.

Data structure:
    data/OmniSpatial/OmniSpatial-test/data.json     (1,533 samples)
    data/OmniSpatial/OmniSpatial-test/{task_type}/{img_id}.png

4 task types (10 sub-task types, 50 fine-grained tasks):
    Dynamic_Reasoning:    Manipulation, Motion_Analysis
    Spatial_Interaction:  Traffic_Analysis, Localization, Geospatial_Strategy
    Complex_Logic:        Pattern_Recognition, Geometric_Reasoning
    Perspective_Taking:   Egocentric, Allocentric, Hypothetical

All questions are 4-choice MC (A/B/C/D). Ground truth is an integer index
into the options list.

Evaluation follows the OmniSpatial repo protocol:
    - Answer extraction: regex for "Answer: X" pattern (last match), fallback "A"
    - Metric: exact match accuracy (per sub-task, per task, overall)
    - Overall = total correct / total samples × 100

Reference: https://github.com/qizekun/OmniSpatial
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.scoring import (
    get_prediction,
    write_json,
    write_results_summary,
)


TASK_TYPES = [
    "Dynamic_Reasoning",
    "Spatial_Interaction",
    "Complex_Logic",
    "Perspective_Taking",
]

SUB_TASK_HIERARCHY = {
    "Dynamic_Reasoning": ["Manipulation", "Motion_Analysis"],
    "Spatial_Interaction": ["Traffic_Analysis", "Localization", "Geospatial_Strategy"],
    "Complex_Logic": ["Pattern_Recognition", "Geometric_Reasoning"],
    "Perspective_Taking": ["Egocentric", "Allocentric", "Hypothetical"],
}

# Regex matching OmniSpatial repo: last "Answer: X" pattern
_ANSWER_PATTERN = re.compile(r"Answer\s*:\s*([A-D])\b", re.IGNORECASE)


@dataclass
class OmniSpatialSample(BaseBenchmarkSample):
    """OmniSpatial sample with MC options and task hierarchy."""

    choices: List[str] = field(default_factory=list)
    task_type: str = ""
    sub_task_type: str = ""
    raw_id: str = ""


class OmniSpatialBench(BaseBenchmark):
    """OmniSpatial benchmark loader (1,533 image-based spatial reasoning questions)."""

    data_specific_prompt = (
        "Answer with the option letter (A, B, C, or D) corresponding to the correct choice."
    )

    def __init__(
        self,
        data_path: str,
        question_type: Optional[List[str]] = None,
        split: str = "test",
    ):
        self.split = split
        super().__init__(data_path=data_path, question_type=question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)

        # Map split to directory name
        split_dirs = {
            "test": "OmniSpatial-test",
            "train": "OmniSpatial-train",
            "full": "OmniSpatial-full",
        }
        split_dir = split_dirs.get(self.split, f"OmniSpatial-{self.split}")
        data_dir = os.path.join(self.data_path, split_dir)
        json_path = os.path.join(data_dir, "data.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"OmniSpatial data.json not found at {json_path}"
            )

        with open(json_path, "r") as f:
            items = json.load(f)

        letters = "ABCD"

        for item in items:
            task_type = item["task_type"]
            sub_task_type = item["sub_task_type"]

            # question_type_filter matches on task_type or sub_task_type
            if self.question_type_filter:
                if (
                    task_type not in self.question_type_filter
                    and sub_task_type not in self.question_type_filter
                ):
                    continue

            raw_id = item["id"]
            # OmniSpatial ids are not globally unique across task/sub-task
            # groups. Runtime session logs, resume bookkeeping, and evaluator
            # prediction maps all require unique sample ids.
            sample_id = f"{task_type}_{sub_task_type}_{raw_id}"
            options = item["options"]
            answer_idx = item["answer"]  # integer 0-3
            gt_letter = letters[answer_idx]

            # Image path: {split_dir}/{task_type}/{img_id}.png
            img_id = raw_id.split("_")[0]
            img_path = os.path.join(data_dir, task_type, f"{img_id}.png")

            question = item["question"]

            self.data.append(
                OmniSpatialSample(
                    sample_id=sample_id,
                    question=question,
                    question_type=sub_task_type,
                    images=[img_path],
                    answer=gt_letter,
                    choices=options,
                    task_type=task_type,
                    sub_task_type=sub_task_type,
                    raw_id=raw_id,
                )
            )

    def extract_answer(self, prediction: str) -> str:
        """Extract answer letter using OmniSpatial regex (last 'Answer: X' match).

        Falls back to base class heuristics, then to 'A' if nothing matches.
        """
        if not prediction:
            return "A"

        # Strategy 1: OmniSpatial regex — last "Answer: X" match
        matches = _ANSWER_PATTERN.findall(prediction)
        if matches:
            return matches[-1].upper()

        # Strategy 2: base class heuristics (boxed, A., single letter)
        base_result = super().extract_answer(prediction)
        if len(base_result) == 1 and base_result in "ABCD":
            return base_result

        # Strategy 3: first character if it's A-D
        first_char = prediction.strip().upper()[:1]
        if first_char in "ABCD":
            return first_char

        # Fallback: "A" (matches OmniSpatial repo behavior)
        return "A"

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        # Nested accumulation: task_type → sub_task_type → list of bools
        task_scores: Dict[str, Dict[str, List[bool]]] = {}
        all_scores: List[bool] = []
        details = []

        for sample in self.data:
            sample_id = sample.sample_id
            raw_pred = get_prediction(predictions, sample_id)
            extracted = self.extract_answer(raw_pred)
            is_correct = extracted.upper() == sample.answer.upper()

            tt = sample.task_type
            st = sample.sub_task_type

            if tt not in task_scores:
                task_scores[tt] = {}
            if st not in task_scores[tt]:
                task_scores[tt][st] = []

            task_scores[tt][st].append(is_correct)
            all_scores.append(is_correct)

            details.append(
                {
                    "sample_id": sample_id,
                    "task_type": tt,
                    "sub_task_type": st,
                    "prediction": raw_pred,
                    "extracted": extracted,
                    "ground_truth": sample.answer,
                    "correct": is_correct,
                }
            )

        total = len(all_scores)
        correct = sum(all_scores)
        overall = correct / total if total > 0 else 0.0

        # Per task_type accuracy
        per_task = {}
        for tt in TASK_TYPES:
            if tt not in task_scores:
                continue
            tt_scores = [s for sub_scores in task_scores[tt].values() for s in sub_scores]
            per_task[tt] = {
                "accuracy": sum(tt_scores) / len(tt_scores) if tt_scores else 0.0,
                "correct": sum(tt_scores),
                "total": len(tt_scores),
            }

        # Per sub_task_type accuracy
        per_sub_task = {}
        for tt, sub_dict in task_scores.items():
            for st, scores in sub_dict.items():
                per_sub_task[st] = {
                    "accuracy": sum(scores) / len(scores) if scores else 0.0,
                    "correct": sum(scores),
                    "total": len(scores),
                    "task_type": tt,
                }

        results = {
            "overall_accuracy": overall,
            "correct_samples": correct,
            "total_samples": total,
            "per_task_type": per_task,
            "per_sub_task_type": per_sub_task,
            "overall_aggregation": "weighted_mean_over_samples",
            "detailed_results": details,
        }

        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), details)

        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*70}")
        print(f"OmniSpatial Results ({self.split} split)")
        print(f"{'='*70}")
        print(f"Total: {results['total_samples']}")
        print(f"Correct: {results['correct_samples']}")
        print(f"Overall Accuracy: {results['overall_accuracy'] * 100:.2f}%")

        per_task = results.get("per_task_type", {})
        per_sub = results.get("per_sub_task_type", {})

        print(f"\n  {'Category':<30} {'Acc':>8}  {'Correct':>8} / {'Total':>5}")
        print(f"  {'-'*60}")

        for tt in TASK_TYPES:
            if tt not in per_task:
                continue
            t = per_task[tt]
            print(f"  {tt:<30} {t['accuracy'] * 100:>7.2f}%  {t['correct']:>8} / {t['total']:>5}")

            # Sub-tasks under this task type
            sub_tasks = SUB_TASK_HIERARCHY.get(tt, [])
            for st in sub_tasks:
                if st not in per_sub:
                    continue
                s = per_sub[st]
                print(f"    {st:<28} {s['accuracy'] * 100:>7.2f}%  {s['correct']:>8} / {s['total']:>5}")

        print(f"{'='*70}\n")
