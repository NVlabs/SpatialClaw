"""STI-Bench (Spatial-Temporal Intelligence Benchmark, ICCV 2025) data loader.

STI-Bench (Li et al., arXiv 2503.23765) asks 2,064 five-way multiple-choice
questions about precise spatial-temporal quantities in 369 real-world videos
from Omni6DPose (desktop), ScanNet (indoor) and Waymo (outdoor). Eight tasks:

    Static Understanding   Dimensional Measurement, Spatial Relation,
                           3D Video Grounding
    Dynamic Understanding  Displacement & Path Length, Speed & Acceleration,
                           Ego-Centric Orientation, Trajectory Description,
                           Pose Estimation

Data structure (HF: MINT-SJTU/STI-Bench, config ``default`` / split ``test``):
    data/STI-Bench/data/test-00000-of-00001.parquet   2,064 rows (canonical file)
    data/STI-Bench/video/<Video>                        flat dir of mp4 files
    data/STI-Bench/video/.frame_cache/<Video>/          ffmpeg frames (lazy, once per video)

Parquet columns used:
    Video        mp4 basename                              -> video path
    Source       ScanNet | Waymo | Omni6DPose              -> source (1:1 with scene)
    scene        indoor | outdoor | desktop
    Task         one of the 8 task names                   -> question_type
    QType        always "Single Choice"
    Question     question text (often ends with a unit hint)
    Prompt       auxiliary object description; kept as a field, NOT appended
                 (none of the official scripts / VLMEvalKit use it)
    time_start   window start in seconds of absolute video time
    time_end     window end in seconds (== time_start for point questions)
    Candidates   struct {A..E: str | null}                 -> choices (verbatim)
    Answer       option letter A-E
    Answer Detail unrounded ground-truth value (metadata only)
    ID           per-video question index (NOT globally unique; row index is)

Text protocol (official openai_test.py, no frame clipping): the question shown
to the model is ``"From {time_start} seconds to {time_end} seconds. " + Question``
followed by the options one per line. The WHOLE video is provided and the
question's time window is only stated in text, exactly as in the paper; the
loader therefore uses the inherited whole-video frame extraction and never
clips frames to [time_start, time_end]. Frame time labels stay in absolute
video time because many questions cite absolute timestamps ("between 5s and
15s", "at t=23s").

Frame setting: spatial_agent/config/dataset/stibench.json =
{"benchmark": "stibench", "video_max_fps": 1} -- the same 1 fps cap as the
shipped videomme.json (the paper used 30 uniformly sampled frames per video).

Metric (official): accuracy = exact match of the predicted option letter with
``Answer``, micro-averaged over all 2,064 questions (paper Table 2 "Avg.").
Also reported: per Task (8), per Source / scene (3), Static vs Dynamic groups,
point vs interval time windows. Unparsable predictions count as wrong.
Random baseline 20%; constant-"A" baseline 23.8%.

Malformed row: parquet row 817 (Waymo, Spatial Relation) has a single option
A="Front Back Right Left Up" with B-E null and Answer="B". The row is kept
(N=2,064 matches the paper) and flagged in ``malformed_sample_ids``; its
letter set is the available keys plus the answer letter so official
exact-match scoring still applies.

Reference: https://huggingface.co/datasets/MINT-SJTU/STI-Bench
           https://github.com/MINT-SJTU/STI-Bench
           https://arxiv.org/abs/2503.23765
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from spatial_agent.config import get_config
from spatial_agent.evals.base import (
    BaseBenchmark,
    BaseBenchmarkSample,
    LazyVideoSample,
    VideoFrameBenchmarkMixin,
)
from spatial_agent.evals.scoring import get_prediction, write_json, write_results_summary

_ALL_LETTERS = "ABCDE"

# Fixed reporting order (paper Table 2 column order).
TASKS = [
    "Dimensional Measurement",
    "Spatial Relation",
    "3D Video Grounding",
    "Displacement & Path Length",
    "Speed & Acceleration",
    "Ego-Centric Orientation",
    "Trajectory Description",
    "Pose Estimation",
]
STATIC_TASKS = ["Dimensional Measurement", "Spatial Relation", "3D Video Grounding"]
DYNAMIC_TASKS = [t for t in TASKS if t not in STATIC_TASKS]
GROUPS = ["Static Understanding", "Dynamic Understanding"]
SOURCES = ["Waymo", "ScanNet", "Omni6DPose"]          # paper Table 3 order (outdoor/indoor/desktop)
SCENES = ["outdoor", "indoor", "desktop"]
WINDOW_TYPES = ["point", "interval"]

PARQUET_RELPATH = os.path.join("data", "test-00000-of-00001.parquet")
VIDEO_SUBDIR = "video"


def task_group_for(task: str) -> str:
    """Map a task name to the paper's Static / Dynamic grouping."""
    if task in STATIC_TASKS:
        return GROUPS[0]
    if task in DYNAMIC_TASKS:
        return GROUPS[1]
    return "other"


def format_seconds(value: float) -> str:
    """Render a time in seconds the way the official prompt does (float repr),
    rounded to 2 decimals to hide float artefacts (0.30000000000000004 -> 0.3)."""
    return repr(round(float(value), 2))


def build_question(question: str, time_start: float, time_end: float) -> str:
    """Official text protocol: prefix the question with its time window."""
    return (
        f"From {format_seconds(time_start)} seconds to "
        f"{format_seconds(time_end)} seconds. {str(question).strip()}"
    )


def _normalize_option_text(text: str) -> str:
    """Whitespace-insensitive, case-insensitive key for option-text matching."""
    return re.sub(r"\s+", "", str(text)).strip().lower().rstrip(".")


@dataclass
class STIBenchSample(LazyVideoSample):
    """STI-Bench sample: one question on one whole video (frames lazily extracted).

    Inherited (all defaulted): video, frame_indices, fps, total_video_frames,
    duration_sec, _bench_ref, _frames_loaded.
    """

    choices: Dict[str, str] = field(default_factory=dict)   # {"A": "2.05m", ...} verbatim
    task: str = ""              # == question_type (8 tasks)
    group: str = ""             # Static Understanding | Dynamic Understanding
    source: str = ""            # ScanNet | Waymo | Omni6DPose
    scene: str = ""             # indoor | outdoor | desktop
    qtype: str = ""             # "Single Choice"
    time_start: float = 0.0     # seconds of absolute video time
    time_end: float = 0.0
    prompt: str = ""            # auxiliary object description (not shown; official parity)
    answer_detail: str = ""     # unrounded ground-truth value (metadata)
    video_name: str = ""        # mp4 basename
    question_id: int = -1       # per-video ID from the parquet (not unique globally)
    raw_question: str = ""      # Question column without the time-window prefix
    valid_letters: str = ""     # letters the sample can be scored on, e.g. "ABCDE"
    answer_in_choices: bool = True

    @property
    def window_type(self) -> str:
        return "point" if self.time_start == self.time_end else "interval"


class STIBench(VideoFrameBenchmarkMixin, BaseBenchmark):
    """STI-Bench loader and scorer (official letter exact-match, micro accuracy)."""

    data_specific_prompt = "Answer with the single letter of the correct option."

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._config = get_config()
        self.malformed_sample_ids: List[str] = []
        self.missing_video_sample_ids: List[str] = []
        super().__init__(data_path, question_type)

    # ── loading ──────────────────────────────────────────────────────────
    def _matches_filter(self, task: str, source: str, scene: str, group: str) -> bool:
        if not self.question_type_filter:
            return True
        wanted = {str(q).strip().lower() for q in self.question_type_filter}
        candidates = {
            task.lower(), source.lower(), scene.lower(), group.lower(),
            group.lower().replace(" understanding", ""),
        }
        return bool(wanted & candidates)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        parquet_path = os.path.join(self.data_path, PARQUET_RELPATH)
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"STI-Bench parquet not found at {parquet_path}. Download "
                f"MINT-SJTU/STI-Bench (data/test-*.parquet + video.zip, unzipped "
                f"into {os.path.join(self.data_path, VIDEO_SUBDIR)}/)."
            )
        video_dir = os.path.join(self.data_path, VIDEO_SUBDIR)
        if not os.path.isdir(video_dir):
            print(f"[STIBench][Warning] video dir not found: {video_dir}")

        df = pd.read_parquet(parquet_path)

        for row_idx in range(len(df)):                     # deterministic parquet row order
            row = df.iloc[row_idx]
            task = str(row["Task"]).strip()
            source = str(row["Source"]).strip()
            scene = str(row.get("scene", "") or "").strip()
            group = task_group_for(task)
            if not self._matches_filter(task, source, scene, group):
                continue

            sample_id = str(row_idx)
            video_name = str(row["Video"]).strip()
            video_path = os.path.join(video_dir, video_name)
            if not os.path.exists(video_path):
                print(f"[STIBench][Warning] row {row_idx}: missing video {video_path}")
                self.missing_video_sample_ids.append(sample_id)
                video_path = ""                            # graceful degradation: images stay []

            cands = row["Candidates"]
            cands = dict(cands) if cands is not None else {}
            choices: Dict[str, str] = {
                k: str(cands[k]) for k in _ALL_LETTERS
                if k in cands and cands[k] is not None
            }
            answer = str(row["Answer"]).strip().upper()
            valid_letters = "".join(choices.keys())
            answer_in_choices = answer in choices
            if not answer_in_choices:
                print(
                    f"[STIBench][Note] row {row_idx}: answer {answer!r} is not among the "
                    f"available options {list(choices)}; keeping the sample (malformed row)"
                )
                self.malformed_sample_ids.append(sample_id)
                if answer in _ALL_LETTERS:
                    valid_letters = "".join(
                        ch for ch in _ALL_LETTERS if ch in choices or ch == answer
                    )

            time_start = float(row["time_start"])
            time_end = float(row["time_end"])
            raw_question = str(row["Question"]).strip()

            self.data.append(
                STIBenchSample(
                    sample_id=sample_id,
                    question=build_question(raw_question, time_start, time_end),
                    question_type=task,
                    images=[],                             # filled lazily by ensure_frames_loaded()
                    answer=answer,
                    video=video_path,
                    choices=choices,
                    task=task,
                    group=group,
                    source=source,
                    scene=scene,
                    qtype=str(row.get("QType", "") or ""),
                    time_start=time_start,
                    time_end=time_end,
                    prompt=str(row.get("Prompt", "") or ""),
                    answer_detail=str(row.get("Answer Detail", "") or ""),
                    video_name=video_name,
                    question_id=int(row["ID"]) if "ID" in row and pd.notna(row["ID"]) else -1,
                    raw_question=raw_question,
                    valid_letters=valid_letters,
                    answer_in_choices=answer_in_choices,
                    _bench_ref=self,
                )
            )

        print(f"[STIBench] Loaded {len(self.data)} samples")
        if self.malformed_sample_ids:
            print(f"[STIBench] {len(self.malformed_sample_ids)} malformed row(s) kept: "
                  f"{self.malformed_sample_ids}")
        if self.missing_video_sample_ids:
            print(f"[STIBench][Warning] {len(self.missing_video_sample_ids)} sample(s) "
                  f"without a video file")

    # ── answer extraction ────────────────────────────────────────────────
    # Explicit conclusion markers; the letter follows. Last occurrence wins.
    # Marker words are case-insensitive ((?i:...)); the letter itself is not:
    # a capital letter counts anywhere, a lowercase one only as the closing
    # token ("answer is c."), so "the answer is a meter" does not yield "A".
    _MARKER_RE = re.compile(
        r"(?i:final\s+answer|correct\s+(?:answer|option|choice|letter)|best\s+(?:answer|option|choice)"
        r"|(?:my|the)\s+answer|answer|ans|option|choice|i\s+(?:choose|select|pick)|i(?:'d|\s+would)\s+go\s+with)"
        r"\b\s*\**\s*(?i:is|would\s+be|should\s+be|must\s+be)?\s*\**\s*[:=\-]?\s*\**\s*[\(\[]?\s*['\"]?"
        r"(?:([A-E])(?![A-Za-z0-9_])|([a-e])(?=['\"]?[\.\)\]]?\**\s*$))"
    )
    # "X is correct" / "(X) is the right answer" (postfix form).
    _POSTFIX_RE = re.compile(
        r"(?<![A-Za-z0-9_])([A-E])\)?\**\s+(?i:is|was|would\s+be|seems|looks)\s+(?i:the\s+)?"
        r"(?i:correct|right|best|answer)\b"
    )
    # Labelled letter: "(X)" anywhere, or "X." / "X:" / "X)" as a closing token.
    _LABELLED_RE = re.compile(r"(?<![A-Za-z0-9_])\(([A-E])\)|(?<![A-Za-z0-9_])([A-E])[\.:)](?=\s|$|\*)")
    # Standalone capital letter (not glued to other alphanumerics).
    _STANDALONE_RE = re.compile(r"(?<![A-Za-z0-9_])([A-E])(?![A-Za-z0-9_])")
    # Text that OPENS with a lone letter token followed by a separator / newline /
    # end: "D\n\nBecause ...", "D (2.78 m)", "D, since ...", "**D** ...".
    _LEADING_RE = re.compile(r"\s*\**\s*[\(\[]?\s*([A-E])\s*[\)\]]?\**(?=$|[\s\.,:;\-\u2013\u2014\(\)/])")
    # "A" directly followed by a lowercase word is the English article ("A truck"),
    # "E.g." is an abbreviation -- neither is an option letter.
    _ARTICLE_TAIL_RE = re.compile(r"\s+[a-z]")

    @classmethod
    def _is_incidental_letter(cls, letter: str, tail: str) -> bool:
        if letter == "A" and cls._ARTICLE_TAIL_RE.match(tail):
            return True
        if letter == "E" and tail.startswith(".g."):
            return True
        return False

    @classmethod
    def _standalone_letters(cls, text: str) -> List[str]:
        """Standalone capital letters A-E in order, skipping article-'A' / 'E.g.'."""
        out: List[str] = []
        for m in cls._STANDALONE_RE.finditer(text):
            if not cls._is_incidental_letter(m.group(1), text[m.end():]):
                out.append(m.group(1))
        return out
    # A short snippet that is just one letter, optionally wrapped: C, (C), C., **C**, c
    _LONE_RE = re.compile(r"\s*\**\s*[\(\[]?\s*([A-Ea-e])\s*[\.\):\]]?\**\s*\.?\s*")

    @classmethod
    def _letter_in_snippet(cls, text: str) -> str:
        """Letter from a short snippet (boxed / tagged content); "" if none."""
        m = cls._LONE_RE.fullmatch(text)
        if m:
            return m.group(1).upper()
        # e.g. "C. 2.78m" or "Option C" inside the box
        m = re.match(r"\s*\**\s*\(?([A-E])[\.\):]\s", text)
        if m:
            return m.group(1).upper()
        ms = list(cls._MARKER_RE.finditer(text))
        if ms:
            return (ms[-1].group(1) or ms[-1].group(2)).upper()
        letters = cls._standalone_letters(text)
        if letters:
            return letters[-1].upper()
        return ""

    def _extract_letter(self, prediction: Any, valid_letters: str = _ALL_LETTERS) -> str:
        """Return an upper-case option letter in ``valid_letters`` or "".

        Handles the raw agent final answer (bare letter, "(C)", "C.", a sentence)
        and the CoT's boxed / boxed-stripped text. Stages, last occurrence wins:
        <answer>X</answer> -> \\boxed{X} -> conclusion markers ("answer is X",
        "Answer: X", "Option X", "X is correct") -> labelled "(X)" / "X." ->
        leading lone letter ("X\n\nBecause ...") -> last standalone capital
        letter (an "A" followed by a lowercase word is the English article and
        "E.g." is an abbreviation; both are ignored) -> the whole text is one
        letter. A letter that is not in ``valid_letters`` yields "" (never a
        default letter).
        """
        if prediction is None:
            return ""
        text = str(prediction).strip()
        if not text:
            return ""
        valid = "".join(ch for ch in _ALL_LETTERS if ch in str(valid_letters).upper()) or _ALL_LETTERS

        def _check(letter: str) -> str:
            return letter if letter in valid else ""

        # 1. <answer>X</answer> (last block)
        tagged = re.findall(r"<answer>\s*(.*?)\s*</?answer>", text, re.DOTALL | re.IGNORECASE)
        for content in reversed(tagged):
            letter = self._letter_in_snippet(content)
            if letter:
                return _check(letter)

        # 2. \boxed{X} / \boxed{(X)} / \boxed{\text{X}} / \boxed{C. 2.78m} (last)
        boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
        for content in reversed(boxed):
            inner = re.sub(r"\\(?:text|mathrm|textbf)\{([^{}]*)\}", r"\1", content)
            letter = self._letter_in_snippet(inner)
            if letter:
                return _check(letter)

        # 3. Conclusion markers (prefix or postfix form), last by position
        found = [
            (m.start(1) if m.group(1) else m.start(2), (m.group(1) or m.group(2)).upper())
            for m in self._MARKER_RE.finditer(text)
        ]
        found += [(m.start(1), m.group(1).upper()) for m in self._POSTFIX_RE.finditer(text)]
        if found:
            found.sort(key=lambda t: t[0])
            return _check(found[-1][1])

        # 4. Labelled letter "(X)" / "X." / "X:" / "X)"
        labelled = [(m.group(1) or m.group(2)).upper() for m in self._LABELLED_RE.finditer(text)]
        labelled = [ch for ch in labelled if ch in valid]
        if labelled:
            return labelled[-1]

        # 4b. The text opens with a lone letter token ("D\n\nBecause ...", "D (2.78 m) ...");
        #     it wins over incidental capitals later in the explanation.
        m = self._LEADING_RE.match(text)
        if m and not self._is_incidental_letter(m.group(1), text[m.end():]):
            return _check(m.group(1))

        # 5. Standalone capital letter (last), ignoring article-"A" / "E.g."
        standalone = [ch for ch in self._standalone_letters(text) if ch in valid]
        if standalone:
            return standalone[-1]

        # 6. The whole text is a single (possibly lowercase) letter
        clean = re.sub(r"[^A-Za-z]", "", text)
        if len(clean) == 1:
            return _check(clean.upper())

        return ""

    def extract_answer(self, prediction: str) -> str:
        """Extract an option letter (A-E) from free text; "" if none."""
        return self._extract_letter(prediction, _ALL_LETTERS)

    def _match_option_text(self, prediction: Any, choices: Dict[str, str]) -> str:
        """Fallback: the prediction equals exactly one option's text (e.g. "2.78m",
        a pose JSON). Whitespace/case-insensitive; also tries boxed content."""
        if prediction is None or not choices:
            return ""
        text = str(prediction).strip()
        boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
        candidates = [text] + [b.strip() for b in boxed]
        for cand in candidates:
            key = _normalize_option_text(cand)
            if not key:
                continue
            hits = [letter for letter, opt in choices.items() if _normalize_option_text(opt) == key]
            if len(hits) == 1:
                return hits[0]
        return ""

    # ── scoring ──────────────────────────────────────────────────────────
    def evaluate_single(self, sample: BaseBenchmarkSample, prediction: str) -> Optional[float]:
        valid = getattr(sample, "valid_letters", "") or _ALL_LETTERS
        pred = self._extract_letter(prediction, valid)
        if not pred:
            pred = self._match_option_text(prediction, getattr(sample, "choices", {}) or {})
        gt = str(sample.answer).strip().upper()
        return 1.0 if pred and pred == gt else 0.0

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Micro accuracy over self.data plus per-task / source / scene / group tables."""
        per_task: Dict[str, Dict[str, int]] = {}
        per_source: Dict[str, Dict[str, int]] = {}
        per_scene: Dict[str, Dict[str, int]] = {}
        per_group: Dict[str, Dict[str, int]] = {}
        per_window: Dict[str, Dict[str, int]] = {}
        detailed: List[Dict[str, Any]] = []
        unparsable = 0
        chance_sum = 0.0

        def _bump(table: Dict[str, Dict[str, int]], key: str, ok: bool) -> None:
            st = table.setdefault(key, {"correct": 0, "total": 0})
            st["total"] += 1
            st["correct"] += int(ok)

        for sample in self.data:
            pred_raw = get_prediction(predictions, sample.sample_id)
            valid = getattr(sample, "valid_letters", "") or _ALL_LETTERS
            choices = getattr(sample, "choices", {}) or {}
            extracted = self._extract_letter(pred_raw, valid)
            if not extracted:
                extracted = self._match_option_text(pred_raw, choices)
            gt = str(sample.answer).strip().upper()
            ok = bool(extracted) and extracted == gt
            score = 1.0 if ok else 0.0
            if not extracted:
                unparsable += 1
            if getattr(sample, "answer_in_choices", True):
                chance_sum += 1.0 / max(len(choices), 1)   # row 817: answer not shown -> chance 0

            task = sample.question_type
            group = getattr(sample, "group", "") or task_group_for(task)
            window = getattr(sample, "window_type", "")
            _bump(per_task, task, ok)
            _bump(per_source, getattr(sample, "source", ""), ok)
            _bump(per_scene, getattr(sample, "scene", ""), ok)
            _bump(per_group, group, ok)
            if window:
                _bump(per_window, window, ok)

            detailed.append({
                "sample_id": sample.sample_id,
                "question_type": task,
                "group": group,
                "source": getattr(sample, "source", ""),
                "scene": getattr(sample, "scene", ""),
                "video": getattr(sample, "video_name", ""),
                "time_start": getattr(sample, "time_start", None),
                "time_end": getattr(sample, "time_end", None),
                "window_type": window,
                "valid_letters": valid,
                "ground_truth": gt,
                "prediction": pred_raw,
                "extracted": extracted,
                "score": score,
                "correct": ok,
            })

        total = len(detailed)
        correct = sum(int(d["correct"]) for d in detailed)
        evaluated_ids = {d["sample_id"] for d in detailed}
        malformed_here = [sid for sid in self.malformed_sample_ids if sid in evaluated_ids]

        def _table(counts: Dict[str, Dict[str, int]], order: List[str]) -> Dict[str, Dict[str, Any]]:
            keys = [k for k in order if k in counts] + sorted(set(counts) - set(order))
            return {
                k: {
                    "total": counts[k]["total"],
                    "correct": counts[k]["correct"],
                    "accuracy": counts[k]["correct"] / max(counts[k]["total"], 1),
                }
                for k in keys
            }

        results: Dict[str, Any] = {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": correct / max(total, 1),
            "overall_aggregation": "micro",
            "unparsable_samples": unparsable,
            "chance_accuracy": chance_sum / max(total, 1),
            "malformed_samples": malformed_here,          # only rows in this evaluation (--limit aware)
            "per_task": _table(per_task, TASKS),
            "per_group": _table(per_group, GROUPS),
            "per_source": _table(per_source, SOURCES),
            "per_scene": _table(per_scene, SCENES),
            "per_window_type": _table(per_window, WINDOW_TYPES),
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), detailed)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'=' * 72}")
        print("STI-Bench Results (official letter exact-match, micro accuracy)")
        print(f"{'=' * 72}")
        print(f"Total: {results.get('total_samples', 0)}  Correct: {results.get('correct_samples', 0)}  "
              f"Accuracy: {results.get('overall_accuracy', 0.0) * 100:.2f}%  "
              f"(chance {results.get('chance_accuracy', 0.0) * 100:.1f}%, "
              f"unparsable {results.get('unparsable_samples', 0)})")
        for title, key in (
            ("Task", "per_task"),
            ("Group", "per_group"),
            ("Source", "per_source"),
            ("Scene", "per_scene"),
            ("Time window", "per_window_type"),
        ):
            table = results.get(key) or {}
            if not table:
                continue
            print(f"\n  {title:<30} {'Acc':>8}  {'Correct':>7}  {'Total':>6}")
            print(f"  {'-' * 56}")
            for name, st in table.items():
                print(f"  {name:<30} {st['accuracy'] * 100:>7.2f}%  {st['correct']:>7}  {st['total']:>6}")
        if results.get("malformed_samples"):
            print(f"\n  malformed rows kept (answer not among options): {results['malformed_samples']}")
        print(f"{'=' * 72}\n")
