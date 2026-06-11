"""BLINK Benchmark data loader.

BLINK: a benchmark for core visual perception abilities.
3,807 multiple-choice questions across 14 subtasks.

Data structure (HuggingFace-format parquets, one dir per subtask):
    data/BLINK/Art_Style/val-00000-of-00001.parquet
    data/BLINK/Counting/val-00000-of-00001.parquet
    ...

Metric: exact-match accuracy, macro-averaged across 14 subtasks.

Reference: https://github.com/zeyofu/BLINK_Benchmark
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample, save_embedded_image
from spatial_agent.evals.scoring import get_prediction, write_results_summary

SUBTASKS = [
    "Art_Style",
    "Counting",
    "Forensic_Detection",
    "Functional_Correspondence",
    "IQ_Test",
    "Jigsaw",
    "Multi-view_Reasoning",
    "Object_Localization",
    "Relative_Depth",
    "Relative_Reflectance",
    "Semantic_Correspondence",
    "Spatial_Relation",
    "Visual_Correspondence",
    "Visual_Similarity",
]


@dataclass
class BLINKSample(BaseBenchmarkSample):
    """A single BLINK benchmark sample."""

    subtask: str = ""
    choices: List[str] = field(default_factory=list)
    prompt: str = ""         # pre-formatted prompt with question + choices
    explanation: str = ""


class BLINKBench(BaseBenchmark):
    """BLINK Benchmark loader.

    Reads per-subtask HuggingFace parquet files with embedded images.
    Uses the 'val' split by default (test answers are withheld).
    """

    data_specific_prompt = (
        "Select the best answer from the given options. "
        "Answer with a single letter in parentheses, e.g. (A)."
    )

    def __init__(
        self,
        data_path: str,
        question_type: Optional[List[str]] = None,
        split: str = "val",
        **kwargs,
    ):
        self._split = split
        self._image_dir: Optional[str] = None
        super().__init__(data_path, question_type, **kwargs)

    def read_data(self) -> None:
        import pandas as pd

        if not os.path.isdir(self.data_path):
            print(f"[Warning] BLINK data dir not found at {self.data_path}")
            return

        self._image_dir = os.path.join(self.data_path, ".image_cache")
        os.makedirs(self._image_dir, exist_ok=True)

        # Determine which subtasks to load
        subtasks_to_load = SUBTASKS
        if self.question_type_filter:
            subtasks_to_load = [
                s for s in SUBTASKS if s in self.question_type_filter
            ]

        total_loaded = 0
        for subtask in subtasks_to_load:
            subtask_dir = os.path.join(self.data_path, subtask)
            if not os.path.isdir(subtask_dir):
                print(f"[Warning] Subtask dir not found: {subtask_dir}")
                continue

            # Find parquet files for the requested split
            parquet_files = sorted([
                os.path.join(subtask_dir, f)
                for f in os.listdir(subtask_dir)
                if f.endswith(".parquet") and f.startswith(self._split)
            ])
            if not parquet_files:
                continue

            df = pd.concat(
                [pd.read_parquet(f) for f in parquet_files], ignore_index=True
            )

            for _, row in df.iterrows():
                # Extract embedded images to disk
                image_paths = []
                for img_col in ["image_1", "image_2", "image_3", "image_4"]:
                    img_data = row.get(img_col)
                    if img_data is None:
                        continue
                    img_filename = f"{row['idx']}_{img_col}.jpg"
                    # Sanitize filename
                    img_filename = img_filename.replace("/", "_")
                    img_path = os.path.join(self._image_dir, img_filename)
                    save_embedded_image(img_path, img_data, convert_rgb=True)
                    image_paths.append(img_path)

                choices_raw = row.get("choices", [])
                if hasattr(choices_raw, "tolist"):
                    choices_raw = choices_raw.tolist()
                choices = list(choices_raw) if choices_raw is not None else []

                sample = BLINKSample(
                    sample_id=row["idx"],
                    question=row.get("question", ""),
                    question_type=subtask,
                    images=image_paths,
                    answer=row.get("answer", ""),
                    subtask=subtask,
                    choices=choices,
                    prompt=row.get("prompt", ""),
                    explanation=row.get("explanation", ""),
                )
                self.data.append(sample)

            total_loaded += len(df)

        print(f"[BLINK] Loaded {total_loaded} {self._split} samples "
              f"across {len(subtasks_to_load)} subtasks")

    def extract_answer(self, prediction: str) -> str:
        """Extract answer letter in (X) format from prediction text.

        Follows the BLINK evaluation logic with added support for
        \\boxed{X} format common in thinking models.
        """
        if not prediction:
            return "(Z)"

        prediction = prediction.strip()
        valid_answers = {"(A)", "(B)", "(C)", "(D)", "(E)"}

        # Direct match: single letter
        if prediction.upper() in ("A", "B", "C", "D", "E"):
            return f"({prediction.upper()})"

        # Direct match: already in (X) format
        if prediction.upper() in valid_answers:
            return prediction.upper()

        # \boxed{X} — take the LAST one (final answer after reasoning)
        boxed = re.findall(r"\\boxed\{([A-Ea-e])\}", prediction)
        if boxed:
            return f"({boxed[-1].upper()})"

        # Token intersection on full text
        tokens = set(prediction.split())
        matches = tokens & valid_answers
        if len(matches) == 1:
            return matches.pop()

        # Token intersection on last paragraph
        last_para = prediction.split("\n\n")[-1]
        tokens_last = set(last_para.split())
        matches_last = tokens_last & valid_answers
        if len(matches_last) == 1:
            return matches_last.pop()

        # Regex: find all (A)-(E) patterns, take the last one
        found = re.findall(r"\(([A-E])\)", prediction)
        if found:
            return f"({found[-1]})"

        # Regex: standalone letters after "answer is" or similar
        m = re.search(r"(?:answer|choice|option)\s+(?:is\s+)?([A-E])\b", prediction, re.I)
        if m:
            return f"({m.group(1).upper()})"

        # Single uppercase letter at the very start
        if prediction[0].upper() in "ABCDE" and (
            len(prediction) == 1 or not prediction[1].isalpha()
        ):
            return f"({prediction[0].upper()})"

        return "(Z)"

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        per_task: Dict[str, Dict[str, int]] = {}
        detailed = []

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            pred = self.extract_answer(pred_raw)
            gt = sample.answer.strip()

            # Normalize ground truth to (X) format if not already
            if gt and gt[0] != "(":
                gt = f"({gt.upper()})"
            gt = gt.upper()

            is_correct = pred == gt
            task = sample.subtask

            if task not in per_task:
                per_task[task] = {"correct": 0, "total": 0}
            per_task[task]["total"] += 1
            if is_correct:
                per_task[task]["correct"] += 1

            detailed.append({
                "idx": sid, "subtask": task, "ground_truth": gt,
                "prediction": pred_raw, "extracted": pred,
                "correct": is_correct,
            })

        # Per-task accuracy
        per_task_acc = {}
        for task, counts in per_task.items():
            per_task_acc[task] = counts["correct"] / max(counts["total"], 1)

        # Overall = macro-average across subtasks (BLINK convention)
        overall = (
            sum(per_task_acc.values()) / len(per_task_acc) if per_task_acc else 0.0
        )

        total_correct = sum(c["correct"] for c in per_task.values())
        total_samples = sum(c["total"] for c in per_task.values())

        results = {
            "total_samples": total_samples,
            "correct_samples": total_correct,
            "overall_accuracy": overall,
            "micro_accuracy": total_correct / max(total_samples, 1),
            "per_subtask": {
                task: {
                    "accuracy": per_task_acc[task],
                    "correct": per_task[task]["correct"],
                    "total": per_task[task]["total"],
                }
                for task in sorted(per_task.keys())
            },
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*70}")
        print(f"BLINK Benchmark Results")
        print(f"{'='*70}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Correct: {results['correct_samples']}")
        print(f"Overall (macro-avg): {results['overall_accuracy']*100:.2f}%")
        print(f"Overall (micro-avg): {results['micro_accuracy']*100:.2f}%")
        print()

        print(f"{'Subtask':<30s} {'Acc':>8s} {'Correct':>8s} {'Total':>6s}")
        print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*6}")
        for task, info in sorted(results.get("per_subtask", {}).items()):
            print(
                f"{task:<30s} {info['accuracy']*100:>7.2f}% "
                f"{info['correct']:>7d} {info['total']:>5d}"
            )
        print(f"{'='*70}\n")
