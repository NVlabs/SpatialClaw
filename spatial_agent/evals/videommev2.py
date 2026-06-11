"""Video-MME-v2 benchmark data loader.

Data structure:
    data/Video-MME-v2/
    ├── test.parquet              # 3200 rows (800 videos × 4 questions)
    └── videos/                   # 800 .mp4 files (001.mp4 .. 800.mp4)

Evaluation protocol:
    Questions are grouped by video (4 per video). Each group has a `group_type`:
    - "relevance": exponential scoring based on number of correct answers in group
    - "logic": sequential scoring — counts consecutive correct answers from Q1,
      weighted by a group_structure that encodes dependency between questions

    Scores per group are in [0, 100]. Final metrics average across groups.
    Breakdowns: per-level (1/2/3), per-second_head, per-third_head.

Reference: https://github.com/MME-Benchmarks/Video-MME-v2
"""

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from spatial_agent.config import get_config
from spatial_agent.evals.base import BaseBenchmark, LazyVideoSample, VideoFrameBenchmarkMixin
from spatial_agent.evals.scoring import get_prediction, write_results_summary


@dataclass
class VideoMMEv2Sample(LazyVideoSample):
    """Video-MME-v2 sample with video, choices, and group metadata."""

    choices: Dict[str, str] = field(default_factory=dict)
    group_type: str = ""           # "relevance" or "logic"
    group_structure: str = ""      # e.g. "[1,2,3,4]", "[1,[2,3],4]"
    level: Optional[str] = None    # "1", "2", "3", or None
    second_head: Optional[str] = None
    third_head: Optional[str] = None


def _extract_answer_letter(s: str) -> str:
    """Extract answer letter A-H from model response (Video-MME-v2 protocol)."""
    s = s.strip()
    answer_prefixes = [
        "Final Answer:",
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, "")

    # \\boxed{X}
    m = re.search(r"\\boxed\{([A-H])\}", s)
    if m:
        return m.group(1)

    if len(s.split()) > 10 and not re.search("[A-H]", s):
        return ""
    matches = re.search(r"[A-H]", s)
    return matches[0] if matches else ""


def _cal_relevance(scores: List[int]):
    """Exponential scoring for relevance groups."""
    score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    correct_count = sum(scores)
    return score_map.get(correct_count, 0.0), correct_count * 25.0


def _cal_logic(scores: List[int], group_structure: str):
    """Sequential scoring for logic groups (structure-dependent)."""
    group_structure_list = ast.literal_eval(group_structure)
    last_correct_idx = -1
    for idx, val in enumerate(scores):
        if val:
            last_correct_idx = idx
        else:
            break

    if group_structure_list == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    elif group_structure_list == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
        if last_correct_idx == 0 and scores[2]:
            last_correct_idx += 1
    elif group_structure_list == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
        if last_correct_idx == -1 and scores[1]:
            last_correct_idx += 1
    else:
        raise ValueError(f"Unknown group_structure_list: {group_structure_list}")

    return score_map.get(last_correct_idx + 1, 0.0)


class VideoMMEv2Bench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """Video-MME-v2 benchmark loader with grouped scoring."""

    data_specific_prompt = (
        "Select the best answer to the following multiple-choice question based on the video. "
        "Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._config = get_config()
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        parquet_path = os.path.join(self.data_path, "test.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        video_dir = os.path.join(self.data_path, "videos")

        for _, row in df.iterrows():
            # Filter by question_type (matches on level, second_head, third_head, or group_type)
            if self.question_type_filter:
                level = str(row.get("level", ""))
                second_head = str(row.get("second_head", ""))
                third_head = str(row.get("third_head", ""))
                group_type = str(row.get("group_type", ""))
                if not any(
                    f in self.question_type_filter
                    for f in [level, second_head, third_head, group_type]
                ):
                    continue

            video_id = str(row["video_id"])
            video_path = os.path.join(video_dir, f"{video_id}.mp4")

            # Parse options: "A. Malaysian.\nB. British.\n..." -> {A: "Malaysian.", ...}
            options_str = str(row["options"])
            choices = {}
            for line in options_str.split("\n"):
                line = line.strip()
                m = re.match(r"^([A-H])\.\s*(.*)", line)
                if m:
                    choices[m.group(1)] = m.group(2)

            sample = VideoMMEv2Sample(
                sample_id=str(row["question_id"]),
                question=str(row["question"]),
                question_type=str(row.get("group_type", "multiple-choice")),
                images=[],
                answer=str(row["answer"]).strip().upper(),
                video=video_path,
                choices=choices,
                group_type=str(row.get("group_type", "")),
                group_structure=str(row.get("group_structure", "")),
                level=str(row["level"]) if row.get("level") is not None else None,
                second_head=str(row["second_head"]) if row.get("second_head") is not None else None,
                third_head=str(row["third_head"]) if row.get("third_head") is not None else None,
                _bench_ref=self,
            )
            self.data.append(sample)

    def extract_answer(self, prediction: str) -> str:
        """Extract answer letter A-H using Video-MME-v2 protocol."""
        if not prediction:
            return ""
        return _extract_answer_letter(prediction)

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        # Score each sample individually
        scored = []
        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            extracted = self.extract_answer(pred_raw)
            gt = sample.answer.strip().upper()
            score = int(extracted == gt) if extracted else -1
            scored.append({
                "sample_id": sid,
                "ground_truth": gt,
                "prediction": pred_raw,
                "extracted": extracted,
                "score": score,
                "group_type": sample.group_type,
                "group_structure": sample.group_structure,
                "level": sample.level,
                "second_head": sample.second_head,
                "third_head": sample.third_head,
            })

        # Simple accuracy (valid extractions only)
        valid = [s for s in scored if s["score"] >= 0]
        simple_acc = sum(s["score"] for s in valid) / max(len(valid), 1)
        n_failed = sum(1 for s in scored if s["score"] == -1)

        # Group scoring: 4 consecutive samples form a group
        groups = [scored[i:i + 4] for i in range(0, len(scored), 4)]
        final_rating = {
            "level_1": [], "level_2": [], "level_3": [],
            "relevance_score": [], "relevance_linear_score": [],
            "logic_score": [], "total": [],
        }
        second_head_rating: Dict[str, List[float]] = {}
        third_head_rating: Dict[str, List[float]] = {}

        for group in groups:
            last = group[-1]
            group_type = last["group_type"]
            group_structure = last["group_structure"]
            level = last["level"]
            second_head = last["second_head"]
            third_head = last["third_head"]
            scores = [max(s["score"], 0) for s in group]  # treat -1 as 0

            if group_type == "relevance":
                exp_score, linear_score = _cal_relevance(scores)
                final_rating["relevance_score"].append(exp_score)
                final_rating["relevance_linear_score"].append(linear_score)
            elif group_type == "logic":
                exp_score = _cal_logic(scores, group_structure)
                final_rating["logic_score"].append(exp_score)
            else:
                raise ValueError(f"Unknown group_type: {group_type}")

            if level is not None and level != "None":
                final_rating[f"level_{int(level)}"].append(exp_score)
            final_rating["total"].append(exp_score)

            if second_head not in second_head_rating:
                second_head_rating[second_head] = []
            second_head_rating[second_head].append(exp_score)

            if third_head not in third_head_rating:
                third_head_rating[third_head] = []
            third_head_rating[third_head].append(exp_score)

        # Average each metric
        avg_rating = {k: (sum(v) / len(v) if v else 0.0) for k, v in final_rating.items()}
        avg_second = {k: (sum(v) / len(v) if v else 0.0) for k, v in second_head_rating.items()}
        avg_third = {k: (sum(v) / len(v) if v else 0.0) for k, v in third_head_rating.items()}

        results = {
            "total_samples": len(scored),
            "total_groups": len(groups),
            "correct_samples": sum(1 for s in scored if s["score"] == 1),
            "failed_extractions": n_failed,
            "simple_accuracy": simple_acc,
            "overall_accuracy": avg_rating.get("total", 0.0) / 100.0,
            "final_rating": avg_rating,
            "second_head_rating": avg_second,
            "third_head_rating": avg_third,
            "detailed_results": scored,
        }

        if output_dir:
            write_results_summary(output_dir, results)
            # Also save the full rating JSON
            rating_path = os.path.join(output_dir, "videommev2_rating.json")
            with open(rating_path, "w") as f:
                json.dump(
                    {"final_rating": avg_rating, "second_head_rating": avg_second, "third_head_rating": avg_third},
                    f, indent=2, default=str,
                )

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        fr = results["final_rating"]
        print(f"\n{'='*70}")
        print(f"Benchmark: Video-MME-v2")
        print(f"Total: {results['total_samples']}  Groups: {results['total_groups']}  "
              f"Simple Acc: {results['simple_accuracy']:.4f}  "
              f"Failed extractions: {results['failed_extractions']}")
        print(f"{'='*70}")

        # Main metrics
        print(f"\n{'Metric':<30} {'Score':>8}")
        print("-" * 40)
        for k in ["total", "level_1", "level_2", "level_3",
                   "relevance_score", "relevance_linear_score", "logic_score"]:
            print(f"{k:<30} {fr.get(k, 0.0):>8.2f}")

        # Second head breakdown
        sh = results.get("second_head_rating", {})
        non_none = {k: v for k, v in sh.items() if k is not None and str(k) != "None"}
        if non_none:
            print(f"\n{'Second Head':<40} {'Score':>8}")
            print("-" * 50)
            for k, v in sorted(non_none.items()):
                print(f"{str(k):<40} {v:>8.2f}")

        # Third head breakdown
        th = results.get("third_head_rating", {})
        non_none = {k: v for k, v in th.items() if k is not None and str(k) != "None"}
        if non_none:
            print(f"\n{'Third Head':<40} {'Score':>8}")
            print("-" * 50)
            for k, v in sorted(non_none.items()):
                print(f"{str(k):<40} {v:>8.2f}")

        print(f"{'='*70}\n")
