"""MMSI-Bench benchmark data loader.

Data structure:
    data/MMSI-Bench/MMSI_Bench.parquet  (1000 samples)
    Images are stored as raw bytes in the parquet and dumped to
    data/MMSI-Bench/images/{id}_{n}.jpg on first load.

11 question types across 4 subsets:
    Positional Relationship: Cam.-Cam., Cam.-Obj., Cam.-Reg., Obj.-Obj., Obj.-Reg., Reg.-Reg.
    Motion: Cam., Obj.
    Attribute: Appr., Meas.
    MSR (multi-step reasoning)

Reference: https://github.com/InternRobotics/MMSI-Bench
"""

import os
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample, write_bytes_if_missing
from spatial_agent.evals.scoring import get_prediction, write_results_summary


SUBSETS = ["Positional Relationship", "Motion", "Attribute", "MSR"]

QUESTION_TYPES = [
    "Positional Relationship (Cam.\u2013Cam.)",
    "Positional Relationship (Cam.\u2013Obj.)",
    "Positional Relationship (Cam.\u2013Reg.)",
    "Positional Relationship (Obj.\u2013Obj.)",
    "Positional Relationship (Obj.\u2013Reg.)",
    "Positional Relationship (Reg.\u2013Reg.)",
    "Motion (Cam.)",
    "Motion (Obj.)",
    "Attribute (Appr.)",
    "Attribute (Meas.)",
    "MSR",
]


def _get_subset(question_type: str) -> str:
    if question_type.startswith("Positional"):
        return "Positional Relationship"
    if question_type.startswith("Motion"):
        return "Motion"
    if question_type.startswith("Attribute"):
        return "Attribute"
    return "MSR"


@dataclass
class MMSIBenchSample(BaseBenchmarkSample):
    """MMSI-Bench sample."""

    difficulty: str = ""
    thought: str = ""
    subset: str = ""


class MMSIBench(BaseBenchmark):
    """MMSI-Bench loader (1000 multi-image spatial reasoning questions).

    Evaluation follows the official MMSI-Bench protocol:
    - Answer extraction: backticks > boxed > common patterns > single letter
    - Accuracy reported per subset and per question type
    """

    data_specific_prompt = (
        "Answer with the option's letter from the given choices directly."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._image_dir: str = ""
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        parquet_path = os.path.join(self.data_path, "MMSI_Bench.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"MMSI-Bench parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        # Dump images to disk (once)
        self._image_dir = os.path.join(self.data_path, "images")
        if not os.path.isdir(self._image_dir):
            os.makedirs(self._image_dir, exist_ok=True)
            for _, row in df.iterrows():
                imgs = row.get("images")
                if imgs is None:
                    continue
                for n, img_bytes in enumerate(imgs):
                    path = os.path.join(self._image_dir, f"{row['id']}_{n}.jpg")
                    write_bytes_if_missing(path, img_bytes)

        for _, row in df.iterrows():
            qtype = row["question_type"]

            # Filter by question_type
            if self.question_type_filter and qtype not in self.question_type_filter:
                continue

            # Build image paths
            imgs = row.get("images")
            n_imgs = len(imgs) if imgs is not None else 0
            image_paths = [
                os.path.join(self._image_dir, f"{row['id']}_{n}.jpg")
                for n in range(n_imgs)
            ]

            sample = MMSIBenchSample(
                sample_id=row["id"],
                question=row["question"],
                question_type=qtype,
                images=image_paths,
                answer=str(row["answer"]).strip().upper(),
                difficulty=row.get("difficulty", ""),
                thought=row.get("thought", ""),
                subset=_get_subset(qtype),
            )
            self.data.append(sample)

    # ── answer extraction ────────────────────────────────────────────────

    def extract_answer(self, prediction: str) -> str:
        """Extract answer letter from prediction.

        Matches the official MMSI-Bench extraction logic:
        1. Double backticks ``X``
        2. Single backticks `X`
        3. \\boxed{X} (with option-text fallback)
        4. Common patterns: X., (X), X:
        5. Single standalone A-D letter
        """
        if not prediction:
            return ""

        prediction = str(prediction).strip()

        # 1. Double backticks
        m = re.search(r"``([^`]*)``", prediction)
        if m:
            inner = m.group(1)
            letter = re.search(r"\b([A-D])\b", inner)
            if letter:
                return letter.group(1).upper()

        # 2. Single backticks
        m = re.search(r"`([^`]*)`", prediction)
        if m:
            inner = m.group(1)
            letter = re.search(r"\b([A-D])\b", inner)
            if letter:
                return letter.group(1).upper()

        # 3. \boxed{X}
        m = re.search(r"\\boxed{\s*([A-D])\s*}", prediction, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # 4. Common patterns
        for pat in [r"\b([A-D])\.", r"\(([A-D])\)", r"\b([A-D]):"]:
            m = re.search(pat, prediction, re.IGNORECASE)
            if m:
                return m.group(1).upper()

        # 5. Word-boundary isolated A-D (official: not followed by space+word)
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
        """Evaluate predictions following official MMSI-Bench protocol."""

        # Per-subset and per-question-type accumulators
        per_subset: Dict[str, Dict[str, Any]] = {}
        for s in SUBSETS:
            per_subset[s] = {"correct": 0, "total": 0, "qtypes": {}}

        detailed = []
        total = correct = 0

        for sample in self.data:
            sid = sample.sample_id
            subset = sample.subset
            qtype = sample.question_type

            pred_raw = get_prediction(predictions, sid)
            pred = self.extract_answer(pred_raw)
            gt = sample.answer
            is_correct = pred == gt

            total += 1
            per_subset[subset]["total"] += 1
            if qtype not in per_subset[subset]["qtypes"]:
                per_subset[subset]["qtypes"][qtype] = {"correct": 0, "total": 0}
            per_subset[subset]["qtypes"][qtype]["total"] += 1

            if is_correct:
                correct += 1
                per_subset[subset]["correct"] += 1
                per_subset[subset]["qtypes"][qtype]["correct"] += 1

            detailed.append(
                {
                    "id": sid,
                    "subset": subset,
                    "question_type": qtype,
                    "ground_truth": gt,
                    "prediction": pred_raw,
                    "extracted": pred,
                    "correct": is_correct,
                }
            )

        results: Dict[str, Any] = {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": correct / max(total, 1),
            "subset_accuracy": {},
            "detailed_results": detailed,
        }

        for s in SUBSETS:
            if per_subset[s]["total"] == 0:
                continue
            sub = per_subset[s]
            sub_result: Dict[str, Any] = {
                "total_samples": sub["total"],
                "correct_samples": sub["correct"],
                "accuracy": sub["correct"] / max(sub["total"], 1),
                "question_type_accuracy": {},
            }
            for qt, counts in sub["qtypes"].items():
                sub_result["question_type_accuracy"][qt] = {
                    "total_samples": counts["total"],
                    "correct_samples": counts["correct"],
                    "accuracy": counts["correct"] / max(counts["total"], 1),
                }
            results["subset_accuracy"][s] = sub_result

        # Save
        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*64}")
        print("MMSI-Bench Evaluation Results")
        print(f"{'='*64}")
        print(f"Total samples   : {results['total_samples']:6d}")
        print(f"Correct samples : {results['correct_samples']:6d}")
        print(f"Overall accuracy: {results['overall_accuracy']:6.2%}")
        print(f"{'='*64}")
        print("Accuracy by Subset / Question Type:")
        print(f"{'='*64}")
        for subset, sub in results.get("subset_accuracy", {}).items():
            print(
                f"- {subset}: {sub['accuracy']:7.2%} "
                f"({sub['correct_samples']:3d}/{sub['total_samples']:3d})"
            )
            for qt, s in sub["question_type_accuracy"].items():
                print(
                    f"    {qt:42s} {s['accuracy']:6.2%} "
                    f"({s['correct_samples']:3d}/{s['total_samples']:3d})"
                )
        print(f"{'='*64}\n")
