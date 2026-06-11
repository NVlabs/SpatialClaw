"""Omni3D-Bench data loader.

Data structure:
    data/Omni3D-Bench/annotations.json
    data/Omni3D-Bench/images/ARKitScenes/Training/{scene_id}/{image}.jpg

501 spatial reasoning questions over 201 indoor images from ARKitScenes.
Answer types: float (ratios/measurements), int (counts), str (yes/no/object names).
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.scoring import get_prediction, write_results_summary


@dataclass
class Omni3DBenchSample(BaseBenchmarkSample):
    """Omni3D-Bench sample."""

    answer_type: str = "str"  # "float", "int", or "str"
    answer_raw: Any = None  # original typed answer


class Omni3DBench(BaseBenchmark):
    """Omni3D-Bench loader."""

    data_specific_prompt = (
        "This is a spatial reasoning question about an indoor scene. "
        "Answer with a single value: a number (float or integer) or a short text answer. "
        "Do NOT include units or explanation — just the value."
    )

    # Float evaluation: Mean Relative Accuracy (MRA) across thresholds
    # Following VADAR: accuracy at each threshold, then averaged
    FLOAT_THRESHOLDS = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

    def read_data(self) -> None:
        json_path = os.path.join(self.data_path, "annotations.json")
        if not os.path.exists(json_path):
            print(f"[Warning] Omni3D-Bench annotations not found at {json_path}")
            return

        with open(json_path, "r") as f:
            data = json.load(f)

        questions = data.get("questions", [])

        for item in questions:
            answer_type = item.get("answer_type", "str")
            if self.question_type_filter and answer_type not in self.question_type_filter:
                continue

            image_filename = item.get("image_filename", "")
            image_path = os.path.join(self.data_path, "images", image_filename)

            answer_raw = item.get("answer")
            answer_str = str(answer_raw)

            sample = Omni3DBenchSample(
                sample_id=item.get("question_index", 0),
                question=item.get("question", ""),
                question_type=answer_type,
                images=[image_path],
                answer=answer_str,
                answer_type=answer_type,
                answer_raw=answer_raw,
            )
            self.data.append(sample)

    def extract_answer(self, prediction: str) -> str:
        """Extract answer from prediction text.

        Tries to pull out a numeric value or short text answer.
        """
        if not prediction:
            return ""

        import re

        text = prediction.strip()

        # Try \boxed{X} first
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        if m:
            text = m.group(1).strip()

        # Try to extract a number (possibly negative, with decimal)
        # Look for the last number in the text (often the final answer)
        numbers = re.findall(r"-?\d+\.?\d*", text)
        if numbers:
            # If the original text is mostly a number, return it
            clean = text.strip().strip(".")
            try:
                float(clean)
                return clean
            except ValueError:
                pass
            # Otherwise return the last number found
            return numbers[-1]

        # For text answers, clean up
        text = text.strip().strip("\"'").lower()
        return text

    def _float_relative_error(self, pred_str: str, gt_raw: Any) -> Optional[float]:
        """Compute relative error for float prediction. Returns None if unparseable."""
        try:
            pred_val = float(pred_str)
            gt_val = float(gt_raw)
            if gt_val == 0:
                return abs(pred_val)
            return abs(pred_val - gt_val) / abs(gt_val)
        except (ValueError, TypeError):
            return None

    def _compare_int(self, pred_str: str, gt_raw: Any) -> bool:
        """Exact match for int answers."""
        try:
            return int(round(float(pred_str))) == int(gt_raw)
        except (ValueError, TypeError):
            return False

    def _compare_str(self, pred_str: str, gt_raw: Any) -> bool:
        """Case-insensitive exact match for str answers."""
        return pred_str.strip().lower() == str(gt_raw).strip().lower()

    def evaluate_single(self, sample, prediction: str) -> float:
        """Score a single sample: rel error for float, exact match for int/str."""
        pred = self.extract_answer(prediction)
        at = sample.answer_type
        if at == "float":
            rel_err = self._float_relative_error(pred, sample.answer_raw)
            return 1.0 if (rel_err is not None and rel_err < 0.1) else 0.0
        elif at == "int":
            return 1.0 if self._compare_int(pred, sample.answer_raw) else 0.0
        else:
            return 1.0 if self._compare_str(pred, sample.answer_raw) else 0.0

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        # Collect per-sample extracted predictions and metadata
        float_errors: List[Optional[float]] = []  # relative errors for float samples
        int_correct = 0
        int_total = 0
        str_correct = 0
        str_total = 0
        detailed = []

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            pred = self.extract_answer(pred_raw)
            at = sample.answer_type

            if at == "float":
                rel_err = self._float_relative_error(pred, sample.answer_raw)
                float_errors.append(rel_err)
                # For detailed log, mark correct at the 0.1 threshold as reference
                is_correct = rel_err is not None and rel_err < 0.1
            elif at == "int":
                is_correct = self._compare_int(pred, sample.answer_raw)
                int_total += 1
                if is_correct:
                    int_correct += 1
            else:  # str
                is_correct = self._compare_str(pred, sample.answer_raw)
                str_total += 1
                if is_correct:
                    str_correct += 1

            detailed.append({
                "id": sid,
                "answer_type": at,
                "ground_truth": sample.answer,
                "prediction": pred_raw,
                "extracted": pred,
                "correct": is_correct,
            })

        # Float: Mean Relative Accuracy (MRA) across thresholds (VADAR protocol)
        float_total = len(float_errors)
        float_mra = 0.0
        float_per_threshold = {}
        if float_total > 0:
            for thr in self.FLOAT_THRESHOLDS:
                correct_at_thr = sum(
                    1 for e in float_errors if e is not None and e < thr
                )
                float_per_threshold[f"{thr:.2f}"] = {
                    "correct": correct_at_thr,
                    "total": float_total,
                    "accuracy": correct_at_thr / float_total,
                }
            float_mra = sum(
                v["accuracy"] for v in float_per_threshold.values()
            ) / len(self.FLOAT_THRESHOLDS)

        int_acc = int_correct / max(int_total, 1)
        str_acc = str_correct / max(str_total, 1)

        # Overall: weighted average across types (each type contributes proportionally)
        total = float_total + int_total + str_total
        overall = (
            (float_mra * float_total + int_acc * int_total + str_acc * str_total)
            / max(total, 1)
        )

        results = {
            "total_samples": total,
            "overall_accuracy": overall,
            "per_type": {
                "float": {
                    "total": float_total,
                    "mra": float_mra,
                    "per_threshold": float_per_threshold,
                },
                "int": {
                    "correct": int_correct,
                    "total": int_total,
                    "accuracy": int_acc,
                },
                "str": {
                    "correct": str_correct,
                    "total": str_total,
                    "accuracy": str_acc,
                },
            },
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        pt = results["per_type"]
        print(f"\n{'='*60}")
        print(f"Benchmark: Omni3D-Bench")
        print(f"Total: {results['total_samples']}")
        print(f"Overall accuracy: {results['overall_accuracy']:.4f}")
        print(f"\nfloat ({pt['float']['total']} samples):")
        print(f"  MRA (mean over thresholds): {pt['float']['mra']:.4f}")
        for thr, stats in pt["float"].get("per_threshold", {}).items():
            print(f"    @{thr}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.4f})")
        print(f"int ({pt['int']['total']} samples):")
        print(f"  Exact match: {pt['int']['correct']}/{pt['int']['total']} ({pt['int']['accuracy']:.4f})")
        print(f"str ({pt['str']['total']} samples):")
        print(f"  Exact match: {pt['str']['correct']}/{pt['str']['total']} ({pt['str']['accuracy']:.4f})")
        print(f"{'='*60}\n")
