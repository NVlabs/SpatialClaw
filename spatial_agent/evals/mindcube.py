"""MindCube benchmark data loader.

Data structure:
    data/MindCube/data/raw/MindCube.jsonl          (21154 samples)
    data/MindCube/data/raw/MindCube_tinybench.jsonl (subset)
    data/MindCube/data/other_all_image/{among,around,rotation,translation}/

Each sample has 2-4 images and a multiple-choice question (A/B/C/D).
Settings derived from sample ID: among, around, rotation, translation, other.
Per MindCube evaluation protocol, translation samples are excluded from overall accuracy.

Reference: https://github.com/mll-lab-nu/MindCube
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.scoring import get_prediction, write_results_summary


@dataclass
class MindCubeBenchSample(BaseBenchmarkSample):
    """MindCube sample with category and type metadata."""

    category: List[str] = field(default_factory=list)
    sample_type: str = ""


class MindCubeBench(BaseBenchmark):
    """MindCube benchmark loader.

    Evaluation follows the official MindCube protocol:
    - Settings: among, around, rotation, translation, other
    - Translation samples are excluded from overall accuracy
    - Answer extraction supports multiple formats (boxed, tag, text patterns)
    """

    VALID_SETTINGS = ["among", "around", "rotation", "translation"]

    data_specific_prompt = (
        "Based on these images, answer the question based on this rule: You only need to provide "
        "*ONE* correct answer selecting from the options listed below. For example, if you think "
        "the correct answer is 'A. above' from 'A. above B. under C. front D. behind.', your "
        "response should only be 'A. above'."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)

        # Find the JSONL file
        jsonl_path = self._find_jsonl()
        if jsonl_path is None:
            raise FileNotFoundError(
                f"MindCube data file not found under: {self.data_path}\n"
                f"Expected MindCube.jsonl or MindCube_tinybench.jsonl in data/raw/"
            )

        # Find image base directory (parent of other_all_image/)
        self.image_base_dir = self._find_image_base_dir()

        # Load samples
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                setting = self._get_setting(item["id"])

                # Filter by question_type (= setting)
                if self.question_type_filter:
                    if setting not in self.question_type_filter:
                        continue

                # Resolve image paths
                image_paths = []
                for img_rel in item.get("images", []):
                    if os.path.isabs(img_rel):
                        image_paths.append(img_rel)
                    else:
                        image_paths.append(os.path.join(self.image_base_dir, img_rel))

                sample = MindCubeBenchSample(
                    sample_id=item["id"],
                    question=item["question"],
                    question_type=setting,
                    images=image_paths,
                    answer=str(item.get("gt_answer", "")).strip().upper(),
                    category=item.get("category", []),
                    sample_type=item.get("type", ""),
                )
                self.data.append(sample)

    def _find_jsonl(self) -> Optional[str]:
        """Search for the JSONL data file."""
        candidates = ["MindCube.jsonl", "MindCube_tinybench.jsonl"]
        search_dirs = [
            self.data_path,
            os.path.join(self.data_path, "raw"),
            os.path.join(self.data_path, "data", "raw"),
        ]
        for d in search_dirs:
            for fname in candidates:
                path = os.path.join(d, fname)
                if os.path.exists(path):
                    return path
        return None

    def _find_image_base_dir(self) -> str:
        """Find the directory that contains other_all_image/."""
        candidates = [
            self.data_path,
            os.path.join(self.data_path, "data"),
        ]
        for d in candidates:
            if os.path.isdir(os.path.join(d, "other_all_image")):
                return d
        # Fallback
        return self.data_path

    @staticmethod
    def _get_setting(sample_id: str) -> str:
        """Extract setting type from sample ID (matches official MindCube logic)."""
        sid = str(sample_id).lower()
        if "around" in sid:
            return "around"
        elif "rotation" in sid:
            return "rotation"
        elif "translation" in sid:
            return "translation"
        elif "among" in sid:
            return "among"
        else:
            return "other"

    def extract_answer(self, prediction: str) -> str:
        """Extract answer letter from prediction (matches official MindCube extractor).

        Priority order:
        1. LaTeX boxed: \\boxed{A}
        2. Simple format: last occurrence of A. B. C. etc.
        3. Tag format: <Answer>A</Answer>
        4. Text patterns: "My answer is A", "The answer is B"
        5. Single letter: last standalone A-E
        """
        if not prediction:
            return ""

        prediction = str(prediction).strip()

        # 1. LaTeX boxed
        m = re.search(r"\\boxed{\s*([A-E])\s*}", prediction, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        # 2. Simple format: last "X." where X is A-E
        matches = list(re.finditer(r"\b([A-E])\.", prediction))
        if matches:
            return matches[-1].group(1).upper()

        # 3. Tag format
        for tag in ["Answer", "answer"]:
            tag_match = re.search(f"<{tag}>(.*?)</{tag}>", prediction, re.DOTALL)
            if tag_match:
                section = tag_match.group(1)
                for pat in [r"\b([A-E])\.", r"\b([A-E])\b"]:
                    inner = list(re.finditer(pat, section))
                    if inner:
                        return inner[-1].group(1).upper()

        # 4. Text patterns
        text_patterns = [
            r"[Mm]y answer is ([A-E])",
            r"[Tt]he answer is ([A-E])",
            r"(?:Answer|answer):\s*([A-E])",
            r"\b([A-E])\s*[:.]\s*[A-Z]",
        ]
        for pat in text_patterns:
            matches = list(re.finditer(pat, prediction))
            if matches:
                return matches[-1].group(1).upper()

        # 5. Single standalone letter (last occurrence)
        matches = list(re.finditer(r"\b([A-E])\b", prediction))
        if matches:
            return matches[-1].group(1).upper()

        return ""

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate predictions following official MindCube protocol.

        Translation samples are excluded from overall accuracy.
        """
        # Per-setting accumulators
        per_setting: Dict[str, Dict[str, int]] = {}
        detailed = []

        for sample in self.data:
            sid = sample.sample_id
            setting = sample.question_type  # = self._get_setting(sid)

            # Skip translation from overall (per official MindCube protocol)
            if setting == "translation":
                continue

            pred_raw = get_prediction(predictions, sid)
            pred = self.extract_answer(pred_raw)
            gt = sample.answer
            is_correct = pred == gt

            if setting not in per_setting:
                per_setting[setting] = {"correct": 0, "total": 0}
            per_setting[setting]["total"] += 1
            if is_correct:
                per_setting[setting]["correct"] += 1

            detailed.append(
                {
                    "id": sid,
                    "question_type": setting,
                    "ground_truth": gt,
                    "prediction": pred_raw,
                    "extracted": pred,
                    "correct": is_correct,
                }
            )

        # Aggregate
        total = sum(s["total"] for s in per_setting.values())
        correct = sum(s["correct"] for s in per_setting.values())

        results: Dict[str, Any] = {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": correct / max(total, 1),
            "question_type_accuracy": {},
            "detailed_results": detailed,
        }

        for setting, counts in per_setting.items():
            results["question_type_accuracy"][setting] = {
                "total_samples": counts["total"],
                "correct_samples": counts["correct"],
                "accuracy": counts["correct"] / max(counts["total"], 1),
            }

        # Save
        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*60}")
        print("MindCube Evaluation Results")
        print(f"{'='*60}")
        print(f"Total samples   : {results['total_samples']:6d}")
        print(f"Correct samples : {results['correct_samples']:6d}")
        print(f"Overall accuracy: {results['overall_accuracy']:6.2%}")
        print(f"{'='*60}")
        print("Accuracy by Setting (translation excluded from overall):")
        print(f"{'='*60}")
        for setting, s in results.get("question_type_accuracy", {}).items():
            print(
                f"  {setting:20s} {s['accuracy']:6.2%} "
                f"({s['correct_samples']:5d}/{s['total_samples']:5d})"
            )
        print(f"{'='*60}\n")
