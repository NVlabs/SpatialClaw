"""ReVSI (Zhang, Chen et al., ICML 2026; arXiv 2604.24300) — data loader and official scorer, 32-frame configuration.

Data layout (``data/ReVSI/``, HF ``3dlg-hcvc/ReVSI``, Apache-2.0):
    32_frame/test-00000-of-00001.parquet        6,158 rows (test split; configs 16/32/64/all_frame are nested)
    <videos>/32_frame/<scene_id>.mp4            the extracted ``video.zip`` (``<videos>`` searched: video/, videos/, revsi_videos/, .)
    metadata/{tiny_set_question_ids.txt, sampled_video_frame_idx.json, 3d_annotation.json, obj_visibility.json}
Fields: id, dataset (ARKitScenes / ScanNetv2 / ScanNetPPv2 / 3RScan / MultiScan), scene_id, question_type (13 sub-types), question,
ground_truth (number as text, or a letter), options (MCQ: ["A. ...", ...]; None for numeric), num_frames ("32"), queried_object_ids.

Official protocol (lmms-eval ``tasks/revsi``; VLMEvalKit ``vlmeval/dataset/revsi.py``; both mirrored here):
    prompt   = question [+ "Options:\\n" + options] + post prompt — numeric: "Answer the question using a single integer or decimal
               number."; MCQ: "Answer with the option's letter from the given choices directly." (the official pre-prompt "These are
               frames of a video." is implied by the harness's own video description)
    numeric  = Mean Relative Accuracy: mean over the 10 thresholds 0.50, 0.55, ..., 0.95 of 1[|pred - gt| / gt <= 1 - thr]
               (the official code's linspace(start, end, int((end - start) / interval + 2)) evaluates to 10 points)
    MCQ      = exact (case-insensitive) letter match
    category = unweighted mean of its sub-type means (rel_direction 4, rel_distance 2, counting 2, room_size 2); Overall = unweighted
               mean of the 7 category scores (so a sub-type-balanced cohort has item mean == official Overall up to rounding).
Answer extraction is the harness's generic one (``\\boxed{}`` / ReturnAnswer content first, then letter patterns or the first
number in the text, units ignored); unparsable -> 0. Nothing here depends on which arm produced the answer.

Frames: the ground truth is only valid for the dataset's own 32 frames, so the sample's ``video`` is the 32-frame mp4 (it contains
exactly those frames) and the harness extracts all of them (native fps, no cap: ``video_max_fps`` None); the no-tool arm is run
with ``--max_frames 32`` and the agent's ``num_key_frames`` is 32, so both arms see the same 32 frames.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from spatial_agent.evals.base import BaseBenchmark, LazyVideoSample, VideoFrameBenchmarkMixin
from spatial_agent.evals.scoring import get_prediction, mean_relative_accuracy, write_results_summary

NQ_TYPES = [
    "object_counting_single", "object_counting_multiple", "object_abs_distance", "object_size_estimation",
    "room_size_estimation_single", "room_size_estimation_multiple",
]
MCQ_TYPES = [
    "object_rel_direction_forward_easy", "object_rel_direction_backward_easy", "object_rel_direction_forward_hard",
    "object_rel_direction_backward_hard", "object_rel_distance_closest", "object_rel_distance_farthest", "route_planning",
]
CATEGORY_OF = {
    "object_abs_distance": "object_abs_distance",
    "object_size_estimation": "object_size_estimation",
    "object_counting_single": "object_counting", "object_counting_multiple": "object_counting",
    "room_size_estimation_single": "room_size_estimation", "room_size_estimation_multiple": "room_size_estimation",
    "object_rel_distance_closest": "object_rel_distance", "object_rel_distance_farthest": "object_rel_distance",
    "object_rel_direction_forward_easy": "object_rel_direction", "object_rel_direction_backward_easy": "object_rel_direction",
    "object_rel_direction_forward_hard": "object_rel_direction", "object_rel_direction_backward_hard": "object_rel_direction",
    "route_planning": "route_planning",
}
CATEGORIES = ["object_abs_distance", "object_counting", "object_rel_direction", "object_rel_distance",
              "object_size_estimation", "room_size_estimation", "route_planning"]
NQ_POST_PROMPT = "Answer the question using a single integer or decimal number."
MCQ_POST_PROMPT = "Answer with the option's letter from the given choices directly."
VIDEO_ROOTS = ("video", "videos", "revsi_videos", "")

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _parse_options(raw) -> Dict[str, str]:
    """["A. wall picture", "B. radiator"] -> {"A": "wall picture", "B": "radiator"}."""
    if raw is None:
        return {}
    out: Dict[str, str] = {}
    for opt in list(raw):
        m = re.match(r"\s*([A-Ha-h])[.)]\s*(.*)", str(opt))
        if m:
            out[m.group(1).upper()] = m.group(2).strip()
        else:
            out[chr(65 + len(out))] = str(opt).strip()
    return out


def _boxed_inner(text: str) -> Optional[str]:
    m = _BOXED_RE.findall(text)
    return m[-1].strip() if m else None


def extract_letter(prediction: str, choices: Optional[Dict[str, str]] = None) -> str:
    """Generic MCQ letter extraction: boxed letter, 'X.' / '(X)' / 'X:' patterns, a lone first token, or the option text."""
    if not prediction:
        return ""
    s = str(prediction).strip()
    inner = _boxed_inner(s)
    if inner is not None:
        s_in = inner.strip()
        if len(s_in) == 1 and s_in.upper() in "ABCDEFGH":
            return s_in.upper()
        s = s_in or s
    for pat in (r"^\s*\(?([A-H])\)?[.:)]?\s*$", r"\b([A-H])\.", r"\(([A-H])\)", r"\b([A-H]):", r"\b[Oo]ption\s+([A-H])\b", r"\b[Aa]nswer\s*(?:is|:)?\s*\(?([A-H])\)?\b"):
        m = re.search(pat, s)
        if m:
            return m.group(1).upper()
    first = s.split()[0].rstrip(".").strip() if s.split() else ""
    if len(first) == 1 and first.upper() in "ABCDEFGH":
        return first.upper()
    if choices:
        low = s.lower().strip().rstrip(".")
        for letter, text in choices.items():
            if text and (low == text.lower().strip() or low.endswith(text.lower().strip())):
                return letter
    return first.upper() if first else ""


def extract_number(prediction: str) -> Optional[float]:
    """Generic numeric extraction: the boxed content if present, then the first number in the text (commas and units ignored)."""
    if not prediction:
        return None
    s = str(prediction).strip()
    inner = _boxed_inner(s)
    if inner is not None:
        s = inner
    s = s.replace(",", "")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


@dataclass
class ReVSISample(LazyVideoSample):
    dataset_source: str = ""
    scene_id: str = ""
    category: str = ""
    subtype: str = ""
    is_numeric: bool = False
    option_map: Dict[str, str] = field(default_factory=dict)   # letter -> option text (NOT named `choices`: both entrypoints
    n_options: int = 0                                           # would append a `choices` attribute to the prompt a second time)
    orientation: str = ""                                        # portrait / landscape of the source video (metadata/scene_video_sizes.json)


class ReVSI(VideoFrameBenchmarkMixin, BaseBenchmark):
    """ReVSI loader and official scorer (32-frame configuration by default)."""

    data_specific_prompt = ""   # the official per-item answer-format sentence is part of each question text

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None, config_name: str = "32_frame"):
        from spatial_agent.config import get_config
        self._config_name = config_name
        self._config = get_config()   # VideoFrameBenchmarkMixin reads video_max_fps / video_frame_resize_short_edge from it
        super().__init__(data_path, question_type)

    # ── loading ──────────────────────────────────────────────────────────

    def _extract_frames(self, video_path: str):
        """The mp4s store exactly the sampled frames (32 for this config) at a fractional fps (≈ 0.07–1.75). The harness
        extracts them in passthrough mode; anything other than the config's frame count is a defect, not a
        rounding issue, and must stop the run rather than feed a different frame set to an arm."""
        frames, frame_indices, fps, total = super()._extract_frames(video_path)
        expected = int(self._config_name.split("_")[0]) if self._config_name.split("_")[0].isdigit() else None
        if frames and expected is not None and len(frames) != expected:
            raise RuntimeError(
                f"ReVSI {os.path.basename(video_path)}: extracted {len(frames)} frames, expected {expected} "
                f"(probe total {total}, fps {fps}); clear its .frame_cache entry and check the ffmpeg extraction")
        if frames:
            total = len(frames)
        return frames, frame_indices, fps, total

    def _video_root(self) -> str:
        for root in VIDEO_ROOTS:
            d = os.path.join(self.data_path, root, self._config_name) if root else os.path.join(self.data_path, self._config_name)
            if os.path.isdir(d) and glob.glob(os.path.join(d, "*.mp4")):
                return d
        raise FileNotFoundError(
            f"ReVSI videos not found: expected <data>/{{video|videos|revsi_videos|.}}/{self._config_name}/<scene_id>.mp4 under "
            f"{self.data_path} (download video.zip from HF 3dlg-hcvc/ReVSI and extract it)")

    def _wants(self, subtype: str) -> bool:
        if not self.question_type_filter:
            return True
        wanted = {str(q).strip().lower() for q in self.question_type_filter}
        return bool({subtype.lower(), CATEGORY_OF.get(subtype, "").lower()} & wanted)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        files = sorted(glob.glob(os.path.join(self.data_path, self._config_name, "test-*.parquet")))
        if not files:
            raise FileNotFoundError(f"ReVSI parquet not found under {self.data_path}/{self._config_name}/ (HF 3dlg-hcvc/ReVSI)")
        video_root = self._video_root()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        sizes_path = os.path.join(self.data_path, "metadata", "scene_video_sizes.json")   # optional: {scene_id: [width, height]} of the source videos
        scene_sizes: Dict[str, Any] = {}
        if os.path.exists(sizes_path):
            try:
                import json as _json
                scene_sizes = _json.load(open(sizes_path))
            except (OSError, ValueError):
                scene_sizes = {}
        missing_videos = 0
        for row in df.to_dict("records"):
            subtype = str(row["question_type"])
            if subtype not in CATEGORY_OF:
                raise ValueError(f"ReVSI: unknown question_type {subtype!r} (id {row.get('id')})")
            if not self._wants(subtype):
                continue
            video = os.path.join(video_root, f"{row['scene_id']}.mp4")
            if not os.path.exists(video):
                missing_videos += 1
            is_numeric = subtype in NQ_TYPES
            opts = row.get("options")
            opts = list(opts) if opts is not None and not (isinstance(opts, float)) else None
            choices = _parse_options(opts) if (opts and not is_numeric) else {}
            question = str(row["question"]).strip()
            if is_numeric:
                text = "\n".join([question, NQ_POST_PROMPT])
            else:
                text = "\n".join([question, "Options:\n" + "\n".join(str(o) for o in (opts or [])), MCQ_POST_PROMPT])
            wh = scene_sizes.get(str(row["scene_id"]))
            orientation = ("portrait" if wh[1] > wh[0] else "landscape") if isinstance(wh, (list, tuple)) and len(wh) == 2 else ""
            self.data.append(ReVSISample(
                sample_id=str(row["id"]), question=text, question_type=subtype, images=[],
                answer=str(row["ground_truth"]).strip(), video=video, dataset_source=str(row["dataset"]),
                scene_id=str(row["scene_id"]), category=CATEGORY_OF[subtype], subtype=subtype, is_numeric=is_numeric,
                option_map=choices, n_options=len(choices), orientation=orientation, _bench_ref=self,
            ))
        if missing_videos:
            raise FileNotFoundError(f"ReVSI: {missing_videos} of {len(self.data)} items have no video under {video_root}")
        print(f"[ReVSI] Loaded {len(self.data)} items ({self._config_name}) from {self.data_path}; videos at {video_root}")

    # ── scoring ──────────────────────────────────────────────────────────

    def extract_answer(self, prediction: str) -> str:
        if not prediction:
            return ""
        inner = _boxed_inner(str(prediction))
        return (inner if inner is not None else str(prediction)).strip()

    def score_item(self, s: ReVSISample, prediction: str) -> Dict[str, Any]:
        pred_raw = str(prediction or "").strip()
        if s.is_numeric:
            val = extract_number(pred_raw)
            try:
                gt = float(s.answer)
            except ValueError:
                return {"correct": 0.0, "parsed": False, "pred": pred_raw[:80], "gold": s.answer, "kind": "numeric"}
            if val is None or gt == 0:
                return {"correct": 0.0, "parsed": val is not None, "pred": pred_raw[:80], "gold": s.answer, "kind": "numeric"}
            return {"correct": float(mean_relative_accuracy(val, gt)), "parsed": True, "pred": repr(val), "gold": s.answer,
                    "kind": "numeric", "rel_err": abs(val - gt) / gt}
        letter = extract_letter(pred_raw, s.option_map)
        return {"correct": 1.0 if letter and letter.lower() == s.answer.strip().lower() else 0.0,
                "parsed": bool(letter), "pred": letter, "gold": s.answer, "kind": "mcq"}

    def evaluate_single(self, sample: BaseBenchmark, prediction: str) -> Optional[float]:
        return self.score_item(sample, prediction)["correct"]

    def evaluate(self, predictions: Dict[Any, str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        detailed, missing = [], 0
        for s in self.data:
            pred = get_prediction(predictions, s.sample_id)
            if pred is None or pred == "":
                missing += 1
            r = self.score_item(s, pred or "")
            detailed.append({"sample_id": s.sample_id, "category": s.category, "subtype": s.subtype, "dataset": s.dataset_source,
                             "scene_id": s.scene_id, "is_numeric": s.is_numeric, "n_options": s.n_options,
                             "orientation": s.orientation or "unknown", **r})
        by_sub: Dict[str, List[float]] = {}
        for d in detailed:
            by_sub.setdefault(d["subtype"], []).append(float(d["correct"]))
        sub_means = {k: sum(v) / len(v) for k, v in by_sub.items() if v}
        cat_means: Dict[str, float] = {}
        for cat in CATEGORIES:
            subs = [m for st, m in sub_means.items() if CATEGORY_OF[st] == cat]
            if subs:
                cat_means[cat] = sum(subs) / len(subs)
        overall_official = sum(cat_means.values()) / len(cat_means) if cat_means else 0.0
        n = len(detailed)
        item_mean = sum(float(d["correct"]) for d in detailed) / n if n else 0.0
        num = [d for d in detailed if d["is_numeric"]]
        mcq = [d for d in detailed if not d["is_numeric"]]
        by_ds: Dict[str, List[float]] = {}
        for d in detailed:
            by_ds.setdefault(d["dataset"], []).append(float(d["correct"]))
        results = {
            "overall_official": overall_official,                      # unweighted mean of the 7 category scores (lmms-eval overall_acc)
            "overall_item_mean": item_mean,                            # plain mean over scored items
            "numeric_mra_mean": (sum(d["correct"] for d in num) / len(num)) if num else None,
            "mcq_accuracy": (sum(d["correct"] for d in mcq) / len(mcq)) if mcq else None,
            "n_numeric": len(num), "n_mcq": len(mcq), "n_items": n, "missing_predictions": missing,
            "unparsed_predictions": sum(1 for d in detailed if not d["parsed"]),
            "per_category": cat_means, "per_subtype": sub_means,
            "per_dataset": {k: sum(v) / len(v) for k, v in by_ds.items()},
            "config_name": self._config_name,
            "detailed_results": detailed,
        }
        if output_dir:
            write_results_summary(output_dir, {k: v for k, v in results.items() if k != "detailed_results"})
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"ReVSI ({results.get('config_name')}): official Overall {100 * results['overall_official']:.1f} | item mean "
              f"{100 * results['overall_item_mean']:.1f} | numeric MRA {100 * (results['numeric_mra_mean'] or 0):.1f} (n={results['n_numeric']}) "
              f"| MCQ acc {100 * (results['mcq_accuracy'] or 0):.1f} (n={results['n_mcq']}) | unparsed {results['unparsed_predictions']} "
              f"| missing {results['missing_predictions']}")
        for cat, v in results["per_category"].items():
            print(f"  {cat:28s} {100 * v:5.1f}")
