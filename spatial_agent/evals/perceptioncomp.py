"""PerceptionComp benchmark data loader.

PerceptionComp: A Video Benchmark for Complex Perception-Centric Reasoning
Paper: https://arxiv.org/abs/2603.26653

Data structure:
    data/PerceptionComp/questions.json   (1114 five-choice questions)
    data/PerceptionComp/data/            (273 videos as {video_id}.mp4)

Each question has:
    - key: unique question ID (str)
    - video_id: maps to data/{video_id}.mp4
    - question: multi-step perception question
    - answer_choice_0..4: five answer choices (A-E)
    - answer_id: index of correct answer (0-4)
    - answer: text of correct answer
    - category: one of 7 categories
    - difficulty: 1, 2, or 3
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from spatial_agent.evals.base import BaseBenchmark, LazyVideoSample, VideoFrameBenchmarkMixin
from spatial_agent.config import get_config
from spatial_agent.evals.scoring import get_prediction, write_results_summary


ANSWER_INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

CATEGORIES = [
    "outdoor tour", "shopping", "sport",
    "variety show", "home tour", "game", "movie",
]


@dataclass
class PerceptionCompSample(LazyVideoSample):
    """PerceptionComp sample with video and five choices."""

    choices: Dict[str, str] = field(default_factory=dict)
    category: str = ""
    difficulty: int = 0


class PerceptionCompBench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """PerceptionComp benchmark loader with lazy frame extraction.

    1114 five-choice video QA questions across 273 videos.
    7 categories, 3 difficulty levels.
    """

    data_specific_prompt = (
        "Answer the single-choice question based on the video frames. "
        "Answer with a single letter (A, B, C, D, or E) corresponding to the correct choice."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._config = get_config()
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        json_path = os.path.join(self.data_path, "questions.json")
        if not os.path.exists(json_path):
            print(f"[Warning] PerceptionComp questions.json not found at {json_path}")
            return

        with open(json_path, "r") as f:
            items = json.load(f)

        video_dir = os.path.join(self.data_path, "data")

        for item in items:
            category = item.get("category", "")
            if self.question_type_filter and category not in self.question_type_filter:
                continue

            video_id = item["video_id"]
            video_path = os.path.join(video_dir, f"{video_id}.mp4")

            # Build choices dict: A-E
            choices = {}
            for i in range(5):
                key = f"answer_choice_{i}"
                if key in item:
                    choices[ANSWER_INDEX_TO_LETTER[i]] = str(item[key])

            # Ground-truth answer letter
            answer_id = item.get("answer_id")
            if isinstance(answer_id, int) and answer_id in ANSWER_INDEX_TO_LETTER:
                answer_letter = ANSWER_INDEX_TO_LETTER[answer_id]
            else:
                answer_letter = str(answer_id)

            sample = PerceptionCompSample(
                sample_id=item["key"],
                question=item.get("question", ""),
                question_type=category,
                images=[],
                answer=answer_letter,
                video=video_path,
                choices=choices,
                category=category,
                difficulty=item.get("difficulty", 0),
                _bench_ref=self,
            )
            self.data.append(sample)

    def extract_answer(self, prediction: str) -> str:
        """Extract answer letter from prediction.

        Supports formats from the reference repo (api_evaluate.py):
        1. <answer>X</answer> tags
        2. Answer: X pattern
        3. \\boxed{X}
        4. Last standalone A-E letter
        """
        if not prediction:
            return ""

        # Strategy 1: <answer> tags (reference repo primary format)
        m = re.search(r"<answer>\s*([A-Ea-e])\s*</answer>", prediction)
        if m:
            return m.group(1).upper()

        # Strategy 2: \boxed{X}
        m = re.search(r"\\boxed\{([A-Ea-e])\}", prediction)
        if m:
            return m.group(1).upper()

        # Strategy 3: "Answer: X" or "answer is X"
        m = re.search(r"[Aa]nswer[\s:]*([A-Ea-e])\b", prediction)
        if m:
            return m.group(1).upper()

        # Strategy 4: last standalone A-E letter
        matches = re.findall(r"\b([A-Ea-e])\b", prediction)
        if matches:
            return matches[-1].upper()

        # Strategy 5: single letter in cleaned text
        clean = re.sub(r"[^A-Za-z]", "", prediction)
        if len(clean) == 1 and clean.upper() in "ABCDE":
            return clean.upper()

        return prediction.strip()

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        correct = 0
        total = 0
        per_category: Dict[str, Dict[str, int]] = {}
        per_difficulty: Dict[int, Dict[str, int]] = {}
        detailed = []

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            pred = self.extract_answer(pred_raw)
            gt = sample.answer.strip().upper()
            is_correct = pred == gt
            if is_correct:
                correct += 1
            total += 1

            # Per-category
            cat = sample.category
            if cat not in per_category:
                per_category[cat] = {"correct": 0, "total": 0}
            per_category[cat]["total"] += 1
            if is_correct:
                per_category[cat]["correct"] += 1

            # Per-difficulty
            diff = sample.difficulty
            if diff not in per_difficulty:
                per_difficulty[diff] = {"correct": 0, "total": 0}
            per_difficulty[diff]["total"] += 1
            if is_correct:
                per_difficulty[diff]["correct"] += 1

            detailed.append({
                "id": sid,
                "category": cat,
                "difficulty": diff,
                "ground_truth": gt,
                "prediction": pred_raw,
                "extracted": pred,
                "correct": is_correct,
            })

        results = {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": correct / max(total, 1),
            "per_category": {
                k: {**v, "accuracy": v["correct"] / max(v["total"], 1)}
                for k, v in sorted(per_category.items())
            },
            "per_difficulty": {
                k: {**v, "accuracy": v["correct"] / max(v["total"], 1)}
                for k, v in sorted(per_difficulty.items())
            },
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*70}")
        print(f"PerceptionComp Results")
        print(f"{'='*70}")
        print(f"Overall: {results['correct_samples']}/{results['total_samples']}"
              f" ({results['overall_accuracy']:.4f})")

        print(f"\n{'─'*70}")
        print(f"Per Category:")
        for cat, vals in results.get("per_category", {}).items():
            print(f"  {cat:20s}: {vals['correct']:4d}/{vals['total']:4d}"
                  f"  ({vals['accuracy']:.4f})")

        print(f"\n{'─'*70}")
        print(f"Per Difficulty:")
        for diff, vals in results.get("per_difficulty", {}).items():
            print(f"  Level {diff}: {vals['correct']:4d}/{vals['total']:4d}"
                  f"  ({vals['accuracy']:.4f})")
        print(f"{'='*70}\n")
