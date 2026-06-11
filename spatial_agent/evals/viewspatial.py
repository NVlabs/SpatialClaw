"""ViewSpatial-Bench data loader.

Multi-perspective spatial localization benchmark from "ViewSpatial-Bench:
Evaluating Multi-perspective Spatial Localization in Vision-Language Models"
(Li et al., 2025; arXiv:2505.21500). 5,712 multiple-choice samples across
five task types over ScanNet v2 val and MS-COCO val2017 imagery.

Data layout under ``data/ViewSpatial-Bench``:
    ViewSpatial-Bench.json
    scannetv2_val/scene{ID}/original_images/{frame}.jpg
    val2017/{COCO_ID}.jpg

Each JSON entry has:
    question_type, image_path (list[str]), question, answer (e.g. "A. right"),
    choices (newline-separated "A. ... B. ... C. ... D. ...").

Image paths in the JSON are prefixed with ``ViewSpatial-Bench/`` (the HF
repo name); we strip that prefix when resolving against ``data_path``.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.scoring import get_prediction, write_results_summary


QUESTION_TYPES = [
    "Camera perspective - Relative Direction",
    "Camera perspective - Object View Orientation",
    "Person perspective - Relative Direction",
    "Person perspective - Object View Orientation",
    "Person perspective - Scene Simulation Relative Direction",
]


@dataclass
class ViewSpatialSample(BaseBenchmarkSample):
    """ViewSpatial-Bench sample."""

    choices_text: str = ""


class ViewSpatialBench(BaseBenchmark):
    """ViewSpatial-Bench loader."""

    data_specific_prompt = (
        "This question is from ViewSpatial-Bench. The task tests spatial "
        "localization either from the camera's own perspective or from "
        "another person's perspective in the scene. Multiple images may be "
        "different views of the same scene.\n\n"
        "Answer with a single option letter: A, B, C, or D."
    )

    def __init__(
        self,
        data_path: str,
        question_type: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(data_path, question_type, **kwargs)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def read_data(self) -> None:
        json_path = os.path.join(self.data_path, "ViewSpatial-Bench.json")
        if not os.path.exists(json_path):
            print(f"[Warning] ViewSpatial-Bench.json not found at {json_path}")
            return

        with open(json_path, "r") as f:
            entries = json.load(f)

        kept = 0
        skipped_missing = 0
        skipped_filtered = 0
        per_type: Dict[str, int] = {}

        for idx, entry in enumerate(entries):
            qtype = str(entry.get("question_type", "unknown"))
            if self.question_type_filter and qtype not in self.question_type_filter:
                skipped_filtered += 1
                continue

            rel_paths = entry.get("image_path") or []
            abs_paths = [
                self._resolve_image_path(p) for p in rel_paths
            ]
            if not abs_paths or not all(os.path.exists(p) for p in abs_paths):
                skipped_missing += 1
                continue

            choices_text = str(entry.get("choices", ""))
            question_text = self._format_question(
                str(entry.get("question", "")), choices_text
            )

            sample = ViewSpatialSample(
                sample_id=idx,
                question=question_text,
                question_type=qtype,
                images=abs_paths,
                answer=str(entry.get("answer", "")).strip(),
                choices_text=choices_text,
            )
            self.data.append(sample)
            kept += 1
            per_type[qtype] = per_type.get(qtype, 0) + 1

        type_str = ", ".join(f"{k}: {v}" for k, v in sorted(per_type.items()))
        print(
            f"[ViewSpatial] Loaded {kept} samples "
            f"(skipped: {skipped_missing} missing-images, "
            f"{skipped_filtered} filtered)"
        )
        if type_str:
            print(f"[ViewSpatial]   by type: {type_str}")

    def _resolve_image_path(self, p: str) -> str:
        """Strip the ``ViewSpatial-Bench/`` prefix that the HF JSON uses."""
        p = p.lstrip("./")
        if p.startswith("ViewSpatial-Bench/"):
            p = p[len("ViewSpatial-Bench/"):]
        return os.path.join(self.data_path, p)

    @staticmethod
    def _format_question(question: str, choices: str) -> str:
        question = question.strip()
        choices = choices.strip()
        if not choices:
            return question
        return f"{question}\n\n{choices}"

    # ------------------------------------------------------------------
    # Answer extraction
    # ------------------------------------------------------------------

    def extract_answer(self, prediction: str) -> str:
        """Extract a single A/B/C/D letter from the prediction."""
        if not prediction:
            return ""

        text = str(prediction).strip()

        # 1. \boxed{X}
        m = re.search(r"\\boxed\{\s*\(?\s*([A-Da-d])\s*[\.\)]?\s*\}", text)
        if m:
            return m.group(1).upper()

        # 2. <answer>X</answer>
        m = re.search(r"<answer>\s*\(?\s*([A-Da-d])\s*[\.\)]?\s*</answer>",
                      text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).upper()

        # 3. Explicit answer markers — last occurrence wins
        marker = re.compile(
            r"(?:final\s+answer|correct\s+answer|answer\s+is|the\s+answer|"
            r"best\s+answer|best\s+option|correct\s+option|correct\s+choice|"
            r"answer)\b\s*[:\-=]?\s*\*{0,2}\(?\s*([A-Da-d])\b",
            re.IGNORECASE,
        )
        ms = list(marker.finditer(text))
        if ms:
            return ms[-1].group(1).upper()

        # 4. Common option patterns — last match wins
        for pat in (r"\(([A-Da-d])\)", r"\b([A-Da-d])\.", r"\b([A-Da-d]):"):
            ms = list(re.finditer(pat, text))
            if ms:
                return ms[-1].group(1).upper()

        # 5. Whole prediction is a single letter
        cleaned = re.sub(r"[^A-Da-d]", "", text)
        if len(cleaned) == 1:
            return cleaned.upper()

        # 6. Last standalone capital A-D in the tail
        tail = text[-400:]
        ms = list(re.finditer(r"\b([A-D])\b", tail))
        if ms:
            return ms[-1].group(1)

        return ""

    @staticmethod
    def _gt_letter(answer: str) -> str:
        """Pull the option letter out of the GT string ('A. right' -> 'A')."""
        if not answer:
            return ""
        m = re.match(r"\s*\(?([A-Da-d])\b", answer)
        return m.group(1).upper() if m else ""

    def evaluate_single(
        self, sample: BaseBenchmarkSample, prediction: str
    ) -> Optional[float]:
        pred = self.extract_answer(prediction)
        gt = self._gt_letter(sample.answer)
        return 1.0 if (pred and pred == gt) else 0.0

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        per_type: Dict[str, Dict[str, int]] = {}
        detailed = []
        total = correct = 0

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            pred = self.extract_answer(pred_raw)
            gt = self._gt_letter(sample.answer)
            is_correct = bool(pred) and pred == gt

            total += 1
            if is_correct:
                correct += 1

            qtype = sample.question_type
            per_type.setdefault(qtype, {"correct": 0, "total": 0})
            per_type[qtype]["total"] += 1
            if is_correct:
                per_type[qtype]["correct"] += 1

            detailed.append({
                "id": sid,
                "question_type": qtype,
                "ground_truth": sample.answer,
                "gt_letter": gt,
                "prediction": pred_raw,
                "extracted": pred,
                "correct": is_correct,
            })

        results: Dict[str, Any] = {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": correct / max(total, 1),
            "per_question_type": {
                k: {
                    "correct": v["correct"],
                    "total": v["total"],
                    "accuracy": v["correct"] / max(v["total"], 1),
                }
                for k, v in sorted(per_type.items())
            },
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*72}")
        print("ViewSpatial-Bench Results")
        print(f"{'='*72}")
        print(
            f"Overall accuracy: {results['overall_accuracy']*100:6.2f}% "
            f"({results['correct_samples']}/{results['total_samples']})"
        )
        print(f"{'-'*72}")
        print("Per question type:")
        for qt, stats in results.get("per_question_type", {}).items():
            print(
                f"  {qt:60s} {stats['accuracy']*100:6.2f}% "
                f"({stats['correct']}/{stats['total']})"
            )
        print(f"{'='*72}\n")
