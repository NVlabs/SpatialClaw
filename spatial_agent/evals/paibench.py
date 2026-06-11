"""PAI-Bench (Physical AI Bench — Understanding) benchmark data loader.

Video-based multiple-choice benchmark with 1,214 test samples across
2 categories (Embodied Reasoning, Common Sense) and 16 subcategories.

Data structure:
    data/PAI-Bench/data/test-00000-of-00001.parquet  (1214 rows)
    data/PAI-Bench/videos/  (927 MP4 files in 7 subdirectories)

Columns: question, index2ans (dict A/B/C/D → text|None), answer (letter),
         video_path (relative), category, subcategory
"""

import json
import os
import re
import string
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


import pandas as pd

from spatial_agent.config import get_config
from spatial_agent.evals.base import BaseBenchmark, LazyVideoSample, VideoFrameBenchmarkMixin
from spatial_agent.evals.scoring import get_prediction, write_results_summary


@dataclass
class PAIBenchSample(LazyVideoSample):
    """PAI-Bench sample with video and category metadata."""

    choices: Dict[str, str] = field(default_factory=dict)
    category: str = ""
    subcategory: str = ""


class PAIBench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """PAI-Bench loader (1214 video-based multiple-choice spatial questions).

    Evaluation: standard MC accuracy with per-category and per-subcategory breakdowns.
    """

    data_specific_prompt = (
        "Answer with a single letter (A, B, C, or D) corresponding to the correct choice."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._config = get_config()
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        parquet_path = os.path.join(self.data_path, "data", "test-00000-of-00001.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"PAI-Bench parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        for idx, row in df.iterrows():
            category = str(row["category"]).strip()
            subcategory = str(row["subcategory"]).strip()

            # Filter by question_type (matches subcategory or category)
            if self.question_type_filter:
                if (subcategory not in self.question_type_filter
                        and category not in self.question_type_filter):
                    continue

            # Build choices dict, filtering out None values
            index2ans = row["index2ans"]
            choices = {}
            for letter in ["A", "B", "C", "D"]:
                val = index2ans.get(letter)
                if val is not None:
                    choices[letter] = str(val)

            question_text = str(row["question"]).strip()

            # Resolve video path
            video_path = os.path.join(self.data_path, str(row["video_path"]))

            sample = PAIBenchSample(
                sample_id=int(idx),
                question=question_text,
                question_type=subcategory,
                images=[],
                answer=str(row["answer"]).strip().upper(),
                video=video_path,
                choices=choices,
                category=category,
                subcategory=subcategory,
                _bench_ref=self,
            )
            self.data.append(sample)

    # ── answer extraction ────────────────────────────────────────────────

    def extract_answer(self, prediction: str) -> str:
        """Extract answer letter from prediction (same chain as MMSI-Bench)."""
        if not prediction:
            return ""

        prediction = str(prediction).strip()

        # 1. Double backticks
        m = re.search(r"``([^`]*)``", prediction)
        if m:
            letter = re.search(r"\b([A-D])\b", m.group(1))
            if letter:
                return letter.group(1).upper()

        # 2. Single backticks
        m = re.search(r"`([^`]*)`", prediction)
        if m:
            letter = re.search(r"\b([A-D])\b", m.group(1))
            if letter:
                return letter.group(1).upper()

        # 3. \boxed{X}
        m = re.search(r"\\boxed{\s*([A-D])\s*}", prediction, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # 4. Common patterns: X., (X), X:
        for pat in [r"\b([A-D])\.", r"\(([A-D])\)", r"\b([A-D]):"]:
            m = re.search(pat, prediction, re.IGNORECASE)
            if m:
                return m.group(1).upper()

        # 5. Isolated A-D letter
        m = re.search(r"\b([A-D])\b(?!\s[a-zA-Z])", prediction)
        if m:
            return m.group(1).upper()

        # 6. Single letter after stripping punctuation
        cleaned = prediction.translate(
            str.maketrans("", "", string.punctuation)
        ).replace(" ", "")
        if len(cleaned) == 1 and cleaned.upper() in "ABCD":
            return cleaned.upper()

        return ""

    # ── evaluation ───────────────────────────────────────────────────────

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate predictions: overall, per-category, per-subcategory accuracy."""

        per_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        per_subcategory: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        detailed = []
        total = correct = invalid = 0

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            pred = self.extract_answer(pred_raw)
            gt = sample.answer
            total += 1

            if not pred:
                invalid += 1

            is_correct = pred == gt
            if is_correct:
                correct += 1

            per_category[sample.category]["total"] += 1
            per_subcategory[sample.subcategory]["total"] += 1
            if is_correct:
                per_category[sample.category]["correct"] += 1
                per_subcategory[sample.subcategory]["correct"] += 1

            detailed.append({
                "id": sid,
                "category": sample.category,
                "subcategory": sample.subcategory,
                "ground_truth": gt,
                "prediction": pred_raw,
                "extracted": pred,
                "correct": is_correct,
            })

        # Build results
        cat_results = {}
        for cat, counts in sorted(per_category.items()):
            cat_results[cat] = {
                "total_samples": counts["total"],
                "correct_samples": counts["correct"],
                "accuracy": counts["correct"] / max(counts["total"], 1),
            }

        subcat_results = {}
        for subcat, counts in sorted(per_subcategory.items()):
            subcat_results[subcat] = {
                "total_samples": counts["total"],
                "correct_samples": counts["correct"],
                "accuracy": counts["correct"] / max(counts["total"], 1),
            }

        results: Dict[str, Any] = {
            "total_samples": total,
            "correct_samples": correct,
            "invalid_samples": invalid,
            "overall_accuracy": correct / max(total, 1),
            "per_category": cat_results,
            "per_subcategory": subcat_results,
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*64}")
        print("PAI-Bench Evaluation Results")
        print(f"{'='*64}")
        print(f"Total samples   : {results['total_samples']:6d}")
        print(f"Correct samples : {results['correct_samples']:6d}")
        print(f"Invalid samples : {results['invalid_samples']:6d}")
        print(f"Overall accuracy: {results['overall_accuracy']:6.2%}")
        print(f"{'='*64}")

        # Per-category
        print("Accuracy by Category:")
        print(f"{'-'*64}")
        for cat, info in results.get("per_category", {}).items():
            print(
                f"  {cat:30s} {info['accuracy']:6.2%} "
                f"({info['correct_samples']:4d}/{info['total_samples']:4d})"
            )

        # Per-subcategory
        print(f"{'-'*64}")
        print("Accuracy by Subcategory:")
        print(f"{'-'*64}")
        for subcat, info in results.get("per_subcategory", {}).items():
            print(
                f"  {subcat:30s} {info['accuracy']:6.2%} "
                f"({info['correct_samples']:4d}/{info['total_samples']:4d})"
            )
        print(f"{'='*64}\n")
