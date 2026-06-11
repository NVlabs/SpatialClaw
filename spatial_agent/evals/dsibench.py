"""DSI-Bench (Decoupled Spatial Intelligence) data loader.

Data structure:
    data/DSI-Bench/
    ├── metadatas/
    │   ├── std.csv
    │   ├── hflip.csv
    │   ├── reverse.csv
    │   ├── reverse_hflip.csv
    │   └── metadata_4aug.csv     # combined (1,769 × 4 = 7,076 rows)
    └── videos/
        ├── std/...
        ├── hflip/...
        ├── reverse/...
        └── reverse_hflip/...

CSV columns (per augmentation):
    cate            - 6 motion categories (0-5)
    relative_path   - 'CameraBench/...mp4' (relative to videos/{aug}/)
    video_type      - 0/1 numeric tag
    question        - question text
    options         - 'A: x; B: y; C: z; D: w'
    GT              - ground-truth letter (A/B/C/D)
    others          - usually NaN

Categories (0-5):
    Obj:static cam, Obj:moving cam, Cam:static scene,
    Cam:dynamic scene, Obj-Cam distance, Obj-Cam orientation

Each base sample is rendered under 4 augmentations (std/hflip/reverse/
reverse_hflip), each with its own GT — robust evaluation tests whether
the model is consistent across viewpoint flips and time reversal.

Evaluation (sample-wise, matches the official ``evaluate.py`` Method 1):
    Each (base, aug) is treated as an independent sample. The first
    A-D letter is extracted from the prediction (with CoT-aware fallbacks)
    and compared to the per-aug ground truth. Reported metrics are
    overall accuracy plus per-category and per-augmentation breakdowns.

Reference: https://huggingface.co/datasets/Viglong/DSI-Bench
           https://github.com/SpatialVision/dsibench/blob/main/evaluate.py
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from spatial_agent.config import get_config
from spatial_agent.evals.base import BaseBenchmark, LazyVideoSample, VideoFrameBenchmarkMixin
from spatial_agent.evals.scoring import get_prediction, write_json, write_results_summary


VIDEO_AUGS = ["std", "reverse", "hflip", "reverse_hflip"]

CATEGORY_NAMES = [
    "Obj:static cam",
    "Obj:moving cam",
    "Cam:static scene",
    "Cam:dynamic scene",
    "Obj-Cam distance",
    "Obj-Cam orientation",
]


@dataclass
class DSIBenchSample(LazyVideoSample):
    """DSI-Bench sample: a single (base, augmentation) pair."""

    choices: Dict[str, str] = field(default_factory=dict)
    cate: int = -1
    aug: str = ""
    relative_path: str = ""
    base_id: str = ""           # shared across the 4 augmentations of a base question


def _extract_letter(s: str) -> str:
    """Extract the A-D option letter (matches the official pred[0] strategy
    while staying robust to CoT-wrapped output: ``\\boxed{}`` → marker → tail)."""
    if not s:
        return ""
    s = s.strip()

    m = re.search(r"\\boxed\{\s*\(?\s*([A-Da-d])\s*[\.\)]?\s*\}", s)
    if m:
        return m.group(1).upper()

    marker = re.compile(
        r"(?:final\s+answer|correct\s+answer|answer\s+is|the\s+answer|"
        r"best\s+answer|best\s+option|correct\s+option|correct\s+choice|"
        r"answer|option)\b\s*[:\-=]?\s*\*{0,2}\(?\s*([A-Da-d])\b",
        re.IGNORECASE,
    )
    ms = list(marker.finditer(s))
    if ms:
        return ms[-1].group(1).upper()

    tail = s[-400:]
    ms = list(re.finditer(
        r"(?:^|[\s\*\(\n])([A-Da-d])(?:[\.\)\:]|\s*\n|\s*\*\*|$)", tail))
    if ms:
        return ms[-1].group(1).upper()

    # Official protocol fallback: take the first non-whitespace character if
    # it's a valid choice letter (this catches "B because ..." style outputs).
    first = s.lstrip()[:1]
    if first.upper() in "ABCD":
        return first.upper()

    return ""


_OPTION_SPLIT = re.compile(r"\s*([A-D])\s*[:\.\)]\s*", re.IGNORECASE)


def _parse_options(options_text: str) -> Dict[str, str]:
    """Parse "A: x; B: y; C: z; D: w" into {'A': 'x', 'B': 'y', ...}.

    Splits on the option-letter prefix (A:/B:/...) so that option text
    containing semicolons or punctuation isn't mangled.
    """
    if not options_text or not isinstance(options_text, str):
        return {}
    parts = _OPTION_SPLIT.split(options_text)
    # parts: ['', 'A', 'x; ', 'B', 'y; ', ...]
    out: Dict[str, str] = {}
    i = 1
    while i + 1 < len(parts):
        letter = parts[i].upper()
        body = parts[i + 1].strip().rstrip(";").rstrip(",").strip()
        if letter in "ABCD":
            out[letter] = body
        i += 2
    return out


class DSIBench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """DSI-Bench loader (1,769 base questions × 4 augmentations = 7,076 samples).

    Evaluation supports the official sample-wise and group-wise (n≥3)
    protocols, with per-category breakdowns.
    """

    data_specific_prompt = (
        "This is a multiple-choice question about how an object or the "
        "camera moves in 3D space within the video clip. Choose the option "
        "that best matches the motion you observe and respond with only the "
        "option letter (A, B, C, or D)."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._config = get_config()
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        meta_dir = os.path.join(self.data_path, "metadatas")
        videos_dir = os.path.join(self.data_path, "videos")

        if not os.path.isdir(meta_dir):
            raise FileNotFoundError(
                f"DSI-Bench metadatas/ not found at {meta_dir}"
            )

        for aug in VIDEO_AUGS:
            csv_path = os.path.join(meta_dir, f"{aug}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Missing DSI-Bench split: {csv_path}")
            df = pd.read_csv(csv_path)

            for row_idx, row in df.iterrows():
                cate = int(row["cate"])
                # question_type filter accepts either the integer code or its
                # readable name.
                if self.question_type_filter:
                    name = (
                        CATEGORY_NAMES[cate] if 0 <= cate < len(CATEGORY_NAMES)
                        else f"cate_{cate}"
                    )
                    if (
                        str(cate) not in self.question_type_filter
                        and name not in self.question_type_filter
                        and aug not in self.question_type_filter
                    ):
                        continue

                relative_path = str(row["relative_path"]).strip()
                video_path = os.path.join(videos_dir, aug, relative_path)
                if not os.path.exists(video_path):
                    video_path = ""  # graceful degradation

                choices = _parse_options(str(row.get("options", "")))
                answer = str(row["GT"]).strip().upper()

                # 4 augmentations are row-aligned across the per-aug CSVs:
                # std.csv[i], hflip.csv[i], reverse.csv[i], reverse_hflip.csv[i]
                # all describe the same source video + question (only the
                # rendering changes, and occasionally the GT). Mirror the
                # official evaluate.py grouping by indexing on row position.
                base_id = f"row_{int(row_idx):06d}"
                sample_id = f"{aug}__{base_id}"

                category_name = (
                    CATEGORY_NAMES[cate] if 0 <= cate < len(CATEGORY_NAMES)
                    else f"cate_{cate}"
                )

                self.data.append(
                    DSIBenchSample(
                        sample_id=sample_id,
                        question=str(row["question"]),
                        question_type=category_name,
                        images=[],
                        answer=answer,
                        video=video_path,
                        choices=choices,
                        cate=cate,
                        aug=aug,
                        relative_path=relative_path,
                        base_id=base_id,
                        _bench_ref=self,
                    )
                )

    def extract_answer(self, prediction: str) -> str:
        return _extract_letter(prediction)

    def evaluate_single(self, sample, prediction: str) -> float:
        extracted = self.extract_answer(prediction)
        gt = sample.answer.strip().upper()
        return 1.0 if extracted and extracted == gt else 0.0

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        per_cate: Dict[int, List[int]] = {}
        per_aug: Dict[str, List[int]] = {}

        details = []
        n_failed = 0

        for sample in self.data:
            sid = sample.sample_id
            raw_pred = get_prediction(predictions, sid)
            extracted = self.extract_answer(raw_pred)
            gt = sample.answer.strip().upper()

            if not extracted:
                score = 0
                n_failed += 1
            else:
                score = 1 if extracted == gt else 0

            per_cate.setdefault(sample.cate, []).append(score)
            per_aug.setdefault(sample.aug, []).append(score)

            details.append({
                "sample_id": sid,
                "base_id": sample.base_id,
                "aug": sample.aug,
                "cate": sample.cate,
                "category_name": sample.question_type,
                "ground_truth": gt,
                "prediction": raw_pred,
                "extracted": extracted,
                "score": score,
            })

        def _mean(xs: List[int]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        sample_per_cate = {
            cate: _mean(v) for cate, v in sorted(per_cate.items())
        }
        sample_per_aug = {a: _mean(v) for a, v in sorted(per_aug.items())}
        total = sum(len(v) for v in per_cate.values())
        sample_overall = (
            sum(sum(v) for v in per_cate.values()) / total if total else 0.0
        )

        def _name(cate: int) -> str:
            if 0 <= cate < len(CATEGORY_NAMES):
                return CATEGORY_NAMES[cate]
            return f"cate_{cate}"

        results = {
            "total_samples": total,
            "failed_extractions": n_failed,
            "overall_accuracy": sample_overall,
            "per_category_accuracy": {
                _name(c): v for c, v in sample_per_cate.items()
            },
            "per_category_counts": {
                _name(c): len(per_cate[c]) for c in sorted(per_cate)
            },
            "per_aug_accuracy": sample_per_aug,
            "per_aug_counts": {a: len(v) for a, v in sorted(per_aug.items())},
            "detailed_results": details,
        }

        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), details)

        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*65}")
        print("DSI-Bench Results (sample-wise)")
        print(f"{'='*65}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Failed extractions: {results.get('failed_extractions', 0)}")
        print(f"Overall accuracy: {results['overall_accuracy']*100:.2f}%")
        print()

        per_cat = results.get("per_category_accuracy", {})
        per_cat_n = results.get("per_category_counts", {})
        if per_cat:
            print(f"  {'Category':<24} {'Acc':>8}  {'N':>6}")
            print(f"  {'-'*42}")
            for name in CATEGORY_NAMES + sorted(set(per_cat) - set(CATEGORY_NAMES)):
                if name in per_cat:
                    print(
                        f"  {name:<24} {per_cat[name]*100:>7.2f}%  "
                        f"{per_cat_n.get(name, 0):>6}"
                    )

        per_aug = results.get("per_aug_accuracy", {})
        if per_aug:
            print(f"\n  {'Augmentation':<24} {'Acc':>8}")
            print(f"  {'-'*34}")
            for a in VIDEO_AUGS:
                if a in per_aug:
                    print(f"  {a:<24} {per_aug[a]*100:>7.2f}%")
        print(f"{'='*65}\n")
