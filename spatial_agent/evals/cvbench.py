"""CVBench (Cross-Video Benchmark) data loader.

CVBench is a 1,000 sample multi-video reasoning benchmark with 15 task types.
Each sample provides 2-4 short YouTube clips and a multiple-choice question
(either A/B/C/D MC or Yes/No). The agent must reason across clips.

Dataset layout (after `unzip CVBench.zip` into ``data/CVBench``):
    data/CVBench/data/test-00000-of-00001.parquet   (1000 rows annotation)
    data/CVBench/videos/<task_dir>/<youtube_id>.mp4 (1315 videos)

Parquet columns:
    id           int             unique 0..999
    task_type    str             one of TASK_CATEGORIES (15 classes)
    video_1..4   str (nullable)  relative mp4 paths under videos/
    question     str
    options      list[str]       1-4 options, MC ("A. ...") or yes/no ("Yes.")
    answer       str             "A"/"B"/"C"/"D" or "Yes"/"No"

Evaluation: per-task-category accuracy + overall mean accuracy.
Reference: https://github.com/Hokhim2/CVBench (Video-R1/src/eval_bench.py)
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from spatial_agent.config import get_config
from spatial_agent.evals.base import (
    BaseBenchmark,
    BaseBenchmarkSample,
    VideoFrameBenchmarkMixin,
)
from spatial_agent.evals.scoring import get_prediction, write_results_summary


TASK_CATEGORIES = [
    "Cross-video Anomaly Detection",
    "Cross-video Scene Recognition",
    "Multi-video Key-Action Recognition",
    "Cross-video Event Retrieval",
    "Cross-video Object Recognition",
    "Multi-video Attribute Recognition",
    "Joint-video Counting",
    "Cross-video Entity Matching",
    "Multi-view Scene Understanding",
    "Multi-video Temporal Reasoning",
    "Joint-video Spatial Navigating",
    "Video Difference Caption",
    "Cross-video Counterfactual Reasoning",
    "Joint-video Summarization",
    "Cross-video Procedural Transfer",
]


def _is_yesno(options: List[str]) -> bool:
    return all(
        str(o).strip().rstrip(".").strip().lower() in ("yes", "no") for o in options
    )


def _parse_mc_options(options: List[str]) -> Dict[str, str]:
    """Parse ['A. foo.', 'B. bar.'] into {'A': 'foo.', 'B': 'bar.'}."""
    out: Dict[str, str] = {}
    for opt in options:
        m = re.match(r"\s*([A-Da-d])[.)]\s*(.*)", str(opt))
        if m:
            out[m.group(1).upper()] = m.group(2).strip()
        else:
            letter = chr(ord("A") + len(out))
            out[letter] = str(opt).strip()
    return out


@dataclass
class CVBenchSample(BaseBenchmarkSample):
    """CVBench sample with 2-4 video paths and MC/yes-no options.

    On ``ensure_frames_loaded()`` the per-video frame paths populate
    ``image_groups``/``frame_indices_groups`` (and per-video fps/length/name
    metadata) so the workflow injects ``InputImages_1, InputImages_2, ...``
    rather than a single concatenated ``InputImages``.
    """

    video_paths: List[str] = field(default_factory=list)
    options: List[str] = field(default_factory=list)
    choices: Dict[str, str] = field(default_factory=dict)
    letter_choices: Dict[str, str] = field(default_factory=dict)
    is_yesno: bool = False
    _bench_ref: Any = field(default=None, repr=False)
    _frames_loaded: bool = field(default=False, repr=False)

    def ensure_frames_loaded(self) -> None:
        """Extract frames for every backing video on first access."""
        if self._frames_loaded:
            return
        bench = self._bench_ref
        if bench is None or not self.video_paths:
            self._frames_loaded = True
            return
        groups: List[List[str]] = []
        indices_groups: List[List[int]] = []
        fps_list: List[float] = []
        totals: List[int] = []
        durations: List[float] = []
        names: List[str] = []
        flat: List[str] = []
        for vp in self.video_paths:
            frames, indices, fps, total = bench._extract_frames(vp)
            groups.append(frames)
            indices_groups.append(indices)
            fps_list.append(fps)
            totals.append(total)
            durations.append(total / fps if fps else 0.0)
            names.append(os.path.basename(vp))
            flat.extend(frames)
        self.image_groups = groups
        self.frame_indices_groups = indices_groups
        self.fps_per_video = fps_list
        self.total_frames_per_video = totals
        self.duration_per_video = durations
        self.video_names = names
        # The original .mp4 paths so SAM3.segment_video_*(video_index=N) can
        # decode the right backing video on the GPU server.
        self.video_sources_per_video = list(self.video_paths)
        self.images = flat
        self._frames_loaded = True


class CVBench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """CVBench loader (1000 multi-video MC questions across 15 task types)."""

    data_specific_prompt = (
        "This sample contains 2-4 separate videos, each exposed in the kernel "
        "as its own `InputImages_<N>` (1-indexed). To compare across videos "
        "pass frames from multiple videos to `vlm.ask_with_thinking(...)`. "
        "For multiple-choice questions answer with the option letter "
        "(A/B/C/D); for yes/no questions answer with 'Yes' or 'No'."
    )

    def __init__(
        self,
        data_path: str,
        question_type: Optional[List[str]] = None,
    ):
        self._config = get_config()
        super().__init__(data_path, question_type)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        parquet_path = os.path.join(
            self.data_path, "data", "test-00000-of-00001.parquet"
        )
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"CVBench parquet not found: {parquet_path}. Download from "
                "huggingface.co/datasets/Dongyh35/CVBench (parquet + "
                "CVBench.zip) and unzip videos under data/CVBench/videos/."
            )

        videos_root = self._resolve_videos_root()
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            self._add_sample(row.to_dict(), videos_root)

    def _resolve_videos_root(self) -> str:
        """Locate the directory containing the per-task ``<id>/`` folders.

        After unzipping CVBench.zip, the videos may live directly under
        ``data_path`` or inside a ``videos/``/``CVBench/`` subdirectory.
        """
        for candidate in ("videos", "CVBench", "."):
            root = os.path.normpath(os.path.join(self.data_path, candidate))
            if os.path.isdir(root):
                # Sanity: at least one numeric subdir like '0/' or '102/'
                try:
                    entries = os.listdir(root)
                except OSError:
                    continue
                if any(e.isdigit() for e in entries):
                    return root
        # Fall back to data_path; missing videos surface as warnings later.
        return self.data_path

    def _add_sample(self, item: Dict[str, Any], videos_root: str) -> None:
        qtype = item["task_type"]
        if self.question_type_filter and qtype not in self.question_type_filter:
            return

        video_paths: List[str] = []
        for key in ("video_1", "video_2", "video_3", "video_4"):
            v = item.get(key)
            if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
                continue
            video_paths.append(os.path.join(videos_root, str(v)))

        raw_options = item.get("options")
        if raw_options is None:
            options: List[str] = []
        else:
            options = [str(o) for o in list(raw_options)]
        is_yesno = bool(options) and _is_yesno(options)
        # ``choices`` (letter→text map) is used by evaluate_single's text-match
        # fallback when the LLM answers with the option text instead of its
        # letter. ``letter_choices`` (a *separate* dict held privately on the
        # sample) is what we hand to the extractor; ``sample.choices`` is left
        # empty so run.py's worker does not duplicate the options into the
        # instruction (we inline them into ``question`` below).
        letter_choices = {} if is_yesno else _parse_mc_options(options)

        # Inline the options into the question so both MC and yes/no samples
        # see them verbatim. ``sample.choices`` is intentionally empty so the
        # worker won't append them a second time.
        question_text = str(item["question"]).strip()
        if options:
            question_text = question_text + "\n" + "\n".join(options)

        sample = CVBenchSample(
            sample_id=int(item["id"]),
            question=question_text,
            question_type=qtype,
            images=[],
            answer=str(item["answer"]).strip(),
            video_paths=video_paths,
            options=options,
            choices={},  # empty so run.py worker doesn't duplicate options
            letter_choices=letter_choices,
            is_yesno=is_yesno,
            _bench_ref=self,
        )
        self.data.append(sample)

    # ── answer extraction ────────────────────────────────────────────────

    @staticmethod
    def _extract_tagged(text: str) -> str:
        """Return content of the last ``<answer>...</answer>`` block, if any."""
        matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        return matches[-1].strip() if matches else ""

    def _extract_yesno(self, prediction: str) -> str:
        """Return 'Yes', 'No', or '' if the prediction names neither."""
        if not prediction:
            return ""
        text = self._extract_tagged(prediction) or prediction
        m = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
        return ""

    def _extract_mc_letter(
        self, prediction: str, choices: Optional[Dict[str, str]] = None
    ) -> str:
        """Extract A/B/C/D letter from prediction. Falls back to text match."""
        if not prediction:
            return ""
        text = self._extract_tagged(prediction) or prediction
        text = text.strip()

        m = re.search(r"\\boxed\{\s*([A-Da-d])\s*\}", text)
        if m:
            return m.group(1).upper()
        for pat in (r"\b([A-D])\.", r"\(([A-D])\)", r"\b([A-D]):", r"^\s*([A-D])\b"):
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).upper()
        # Single-letter prediction
        clean = re.sub(r"[^A-Da-d]", "", text)
        if len(clean) == 1:
            return clean.upper()
        # Match against choice text
        if choices:
            low = text.lower().strip().rstrip(".")
            for letter, txt in choices.items():
                if low == txt.lower().strip().rstrip("."):
                    return letter.upper()
        return ""

    def extract_answer(self, prediction: str) -> str:
        """Default extraction (used when sample type is unknown)."""
        if not prediction:
            return ""
        # Prefer letter; if absent, try yes/no.
        letter = self._extract_mc_letter(prediction)
        if letter:
            return letter
        yn = self._extract_yesno(prediction)
        if yn:
            return yn
        return prediction.strip()

    # ── evaluation ───────────────────────────────────────────────────────

    def evaluate_single(
        self, sample: BaseBenchmarkSample, prediction: str
    ) -> Optional[float]:
        if not isinstance(sample, CVBenchSample):
            return super().evaluate_single(sample, prediction)
        gt = sample.answer.strip()
        if sample.is_yesno:
            pred = self._extract_yesno(prediction)
            return 1.0 if pred and pred.lower() == gt.lower() else 0.0
        pred = self._extract_mc_letter(prediction, sample.letter_choices)
        return 1.0 if pred and pred.upper() == gt.upper() else 0.0

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        per_type: Dict[str, Dict[str, int]] = {}
        detailed: List[Dict[str, Any]] = []
        correct = 0
        total = 0

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            score = self.evaluate_single(sample, pred_raw)
            score = 0.0 if score is None else float(score)
            is_correct = score >= 1.0

            qt = sample.question_type
            stats = per_type.setdefault(qt, {"correct": 0, "total": 0})
            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1
            total += 1
            if is_correct:
                correct += 1

            if sample.is_yesno:
                extracted = self._extract_yesno(pred_raw)
            else:
                extracted = self._extract_mc_letter(pred_raw, sample.letter_choices)
            detailed.append({
                "id": sid,
                "task_type": qt,
                "ground_truth": sample.answer,
                "prediction": pred_raw,
                "extracted": extracted,
                "is_yesno": sample.is_yesno,
                "score": score,
                "correct": is_correct,
            })

        per_type_out: Dict[str, Dict[str, Any]] = {}
        for qt in TASK_CATEGORIES:
            stats = per_type.get(qt, {"correct": 0, "total": 0})
            acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
            per_type_out[qt] = {**stats, "accuracy": acc}
        # Include any unexpected task types observed in the data
        for qt, stats in per_type.items():
            if qt not in per_type_out:
                acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
                per_type_out[qt] = {**stats, "accuracy": acc}

        results: Dict[str, Any] = {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": correct / total if total else 0.0,
            "overall_accuracy_pct": (correct / total * 100) if total else 0.0,
            "per_task_type": per_type_out,
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)
        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'=' * 70}")
        print("CVBench Evaluation Results")
        print(f"{'=' * 70}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Correct: {results['correct_samples']}")
        print(f"Overall accuracy: {results['overall_accuracy_pct']:.2f}%")
        print(f"{'=' * 70}")
        for qt in TASK_CATEGORIES:
            info = results["per_task_type"].get(qt)
            if not info:
                continue
            acc_pct = info["accuracy"] * 100
            print(
                f"  {qt:42s} {acc_pct:6.2f}%  "
                f"({info['correct']}/{info['total']})"
            )
        print(f"{'=' * 70}\n")
