"""Video-MME benchmark data loader.

Data structure:
    data/Video-MME/
    ├── videomme/test-00000-of-00001.parquet   # 2700 rows
    ├── data/                                   # 900 .mp4 files
    └── subtitle/                               # 744 .srt files

Reference: https://video-mme.github.io/
"""

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
class VideoMMESample(LazyVideoSample):
    """Video-MME sample with video, choices, and metadata."""

    choices: Dict[str, str] = field(default_factory=dict)
    duration: str = ""          # short / medium / long
    domain: str = ""            # Knowledge, Life Record, etc.
    task_type: str = ""         # Counting Problem, Object Reasoning, etc.
    subtitle_path: Optional[str] = None


class VideoMMEBench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """Video-MME benchmark loader with lazy frame extraction."""

    data_specific_prompt = (
        "Answer with a single letter (A, B, C, or D) corresponding to the correct choice."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._config = get_config()
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)

        # Find the parquet file
        parquet_dir = os.path.join(self.data_path, "videomme")
        parquet_files = [
            f for f in os.listdir(parquet_dir)
            if f.endswith(".parquet")
        ] if os.path.isdir(parquet_dir) else []
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in {parquet_dir}"
            )
        parquet_path = os.path.join(parquet_dir, parquet_files[0])
        df = pd.read_parquet(parquet_path)

        video_dir = os.path.join(self.data_path, "data")
        subtitle_dir = os.path.join(self.data_path, "subtitle")

        for _, row in df.iterrows():
            # Filter by question_type_filter (matches on task_type or duration)
            if self.question_type_filter:
                task_type = str(row.get("task_type", ""))
                duration = str(row.get("duration", ""))
                if task_type not in self.question_type_filter and duration not in self.question_type_filter:
                    continue

            video_id = str(row["videoID"])
            video_path = os.path.join(video_dir, f"{video_id}.mp4")

            # Parse options: ["A. Apples.", "B. Candles.", ...] -> {A: "Apples.", B: "Candles.", ...}
            options = row["options"]
            choices = {}
            for opt in options:
                opt_str = str(opt)
                m = re.match(r"^([A-D])\.\s*(.*)", opt_str)
                if m:
                    choices[m.group(1)] = m.group(2)
                else:
                    # Fallback: assign by position
                    letter = chr(65 + len(choices))
                    choices[letter] = opt_str

            # Check subtitle
            srt_path = os.path.join(subtitle_dir, f"{video_id}.srt")
            subtitle_path = srt_path if os.path.exists(srt_path) else None

            sample = VideoMMESample(
                sample_id=str(row["question_id"]),
                question=str(row["question"]),
                question_type=str(row.get("task_type", "multiple-choice")),
                images=[],
                answer=str(row["answer"]).strip().upper(),
                video=video_path,
                choices=choices,
                duration=str(row.get("duration", "")),
                domain=str(row.get("domain", "")),
                task_type=str(row.get("task_type", "")),
                subtitle_path=subtitle_path,
                _bench_ref=self,
            )
            self.data.append(sample)

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        correct = 0
        total = 0
        per_duration: Dict[str, Dict[str, int]] = {}
        per_domain: Dict[str, Dict[str, int]] = {}
        per_task_type: Dict[str, Dict[str, int]] = {}
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

            # Per-duration
            dur = sample.duration
            if dur not in per_duration:
                per_duration[dur] = {"correct": 0, "total": 0}
            per_duration[dur]["total"] += 1
            if is_correct:
                per_duration[dur]["correct"] += 1

            # Per-domain
            dom = sample.domain
            if dom not in per_domain:
                per_domain[dom] = {"correct": 0, "total": 0}
            per_domain[dom]["total"] += 1
            if is_correct:
                per_domain[dom]["correct"] += 1

            # Per-task-type
            tt = sample.task_type
            if tt not in per_task_type:
                per_task_type[tt] = {"correct": 0, "total": 0}
            per_task_type[tt]["total"] += 1
            if is_correct:
                per_task_type[tt]["correct"] += 1

            detailed.append({
                "id": sid,
                "duration": dur,
                "domain": dom,
                "task_type": tt,
                "ground_truth": gt,
                "prediction": pred_raw,
                "extracted": pred,
                "correct": is_correct,
            })

        def _acc(d):
            return {
                k: {
                    "total": v["total"],
                    "correct": v["correct"],
                    "accuracy": v["correct"] / max(v["total"], 1),
                }
                for k, v in sorted(d.items())
            }

        results = {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": correct / max(total, 1),
            "per_duration": _acc(per_duration),
            "per_domain": _acc(per_domain),
            "per_task_type": _acc(per_task_type),
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*70}")
        print(f"Benchmark: Video-MME")
        print(f"Total: {results['total_samples']}  Correct: {results['correct_samples']}  "
              f"Accuracy: {results['overall_accuracy']:.4f}")
        print(f"{'='*70}")

        # Duration breakdown
        if "per_duration" in results:
            print(f"\n{'Duration':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
            print("-" * 40)
            for dur in ["short", "medium", "long"]:
                if dur in results["per_duration"]:
                    d = results["per_duration"][dur]
                    print(f"{dur:<12} {d['correct']:>8} {d['total']:>8} {d['accuracy']:>10.4f}")

        # Domain breakdown
        if "per_domain" in results:
            print(f"\n{'Domain':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
            print("-" * 55)
            for dom, d in results["per_domain"].items():
                print(f"{dom:<25} {d['correct']:>8} {d['total']:>8} {d['accuracy']:>10.4f}")

        # Task type breakdown
        if "per_task_type" in results:
            print(f"\n{'Task Type':<35} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
            print("-" * 65)
            for tt, d in results["per_task_type"].items():
                print(f"{tt:<35} {d['correct']:>8} {d['total']:>8} {d['accuracy']:>10.4f}")

        print(f"{'='*70}\n")
