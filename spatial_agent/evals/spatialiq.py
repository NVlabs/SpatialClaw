"""Spatial-IQ data loader (NVIDIA Research + Yale, 2026; free-response text track).

Spatial-IQ ("Spatial-IQ: Deconstructing Spatial Intelligence via Hierarchical
Capability Tests", arXiv 2607.22864) renders procedurally generated structures of
stacked identical objects (cubes, factory boxes, soup cans, spam tins) and asks a
hierarchy of counting questions about ONE rendered view: the headline task
(``task_main``: total number of objects incl. hidden supporting ones) and
atomic sub-tasks (top-layer count, columns, layers, visible objects, direct
supports, support columns, visible-above-hidden, clusters).

Data structure (HF repo patrickqrim/spatial-iq, CC-BY-4.0; 3,000 leaf view dirs
= the paper's evaluation set, identical to inference_results/subset_eval.txt of
the official code x 4 camera variants):
    data/Spatial-IQ/<data_YYYYMMDD_HHMMSS>/level_1/sample_<NN>/view_<VV>/<variant>/
        view.png    512x512 RGBA PNG (the only input image; passed unmodified)
        gt.json     object_type, sample_idx, view_idx, offset, fov, dist, task1 (word),
                    task_main, task2..task11 (int), total_blocks, num_hidden_blocks,
                    num_columns, num_layers, visible_blocks (list), mcq (dict)
    <variant> in offset3_fov1_dist1 | offset12.5_fov1_dist1 | offset3_fov3_dist0.3 |
              offset12.5_fov3_dist0.3 (750 view dirs each); 554 structures, 750
              structure-views (1-4 distinct view_<VV> per structure).

Items: one sample per (view dir, task) for the NINE integer tasks, in the fixed
order task_main, task2, ..., task9 within a view; view dirs sorted
lexicographically by their relative path (deterministic across machines).
    -> 27,000 samples (3,000 per task)
Excluded: task1 (categorical word "blocks", not scored by the official text
evaluator) and task10/task11 (two-view questions; the second view is not part
of the HF release).
    sample_id      "<relative view dir>#<task>", e.g.
                   "data_20260403_170859/level_1/sample_01/view_03/offset12.5_fov1_dist1#task_main"
    question_type  task key (task_main, task2, ... task9); readable names in TASK_NAMES
    question       the OFFICIAL free-response prompt of inference/prompt_text_Type1.json
                   (NVIDIA/Spatial-IQ) with {CRITICAL_INSTRUCTION} / {DEFINITIONS}
                   expanded exactly like the official clients (str.format_map)
    answer         str(gt[task]), a non-negative integer
    images         [absolute path to view.png]

Scoring (semantics of the official evaluation/evaluation_text.py): per-item exact
integer match. Official parser chain: strip <think>...</think> / GLM box tokens ->
exact integer -> the single integer in the text -> the LAST integer in the text.
Our shared parser (both arms: raw agent text and \\boxed{}-stripped CoT text) adds,
in front of that chain, the content of the LAST ``\\boxed{...}`` (LaTeX ``\\text{}``
wrappers, spacing macros and markdown removed) and, behind it, a number-word
fallback (zero ... ninety-nine). Decimal tokens are treated as integers only when
their fractional part is zero ("7.0" -> 7; "7.5" is not an integer and is skipped;
the official regex would read "7.5" as the two integers 7 and 5).
DECLARED DIFFERENCE: the official scorer flags unparsable predictions for manual
review and EXCLUDES them from % correct; here an unparsable prediction scores 0.0,
stays in the denominator and is counted in ``unparsed_predictions``.

Reported: overall exact-match accuracy over loaded items; per task (headline =
task_main), per camera variant (+ offset / fov-dist marginals), per object_type,
per total_blocks bin (<=8, 9-15, 16-25, >25); MAE (over parsed predictions) and
RA@10% (|pred-gt| <= 0.1*gt, gt=0 -> exact; unparsed = miss) per task and overall;
``summation_consistency``: among views whose task_main AND task5 predictions are
both present, the fraction with pred_main - pred_task5 == gt_main - gt_task5
(i.e. the implied hidden-object count is right), with its n.

Reference: https://github.com/NVIDIA/Spatial-IQ (BSD-3-Clause)
           https://huggingface.co/datasets/patrickqrim/spatial-iq
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.scoring import (
    get_prediction,
    write_json,
    write_results_summary,
)

IMAGE_FILE = "view.png"
GT_FILE = "gt.json"

# Evaluated tasks, in the fixed within-view order (headline first).
TASKS = ["task_main", "task2", "task3", "task4", "task5", "task6", "task7", "task8", "task9"]
TASK_NAMES = {
    "task_main": "object_counting_total",
    "task2": "top_layer_count",
    "task3": "column_count",
    "task4": "layer_count",
    "task5": "visible_object_count",
    "task6": "direct_support_count",
    "task7": "support_column_count",
    "task8": "visible_above_hidden_count",
    "task9": "cluster_count",
}
# gt.json key -> key of the official prompt table ("main" for task_main)
_PROMPT_KEY = {t: ("main" if t == "task_main" else t) for t in TASKS}

VARIANTS = [
    "offset3_fov1_dist1",
    "offset12.5_fov1_dist1",
    "offset3_fov3_dist0.3",
    "offset12.5_fov3_dist0.3",
]
OBJECT_TYPES = ["cube", "factory_box", "soup_can", "spam"]
TOTAL_BLOCKS_BINS: List[Tuple[str, int, int]] = [
    ("<=8", 0, 8),
    ("9-15", 9, 15),
    ("16-25", 16, 25),
    (">25", 26, 10**9),
]

_DATA_DIR_RE = re.compile(r"^data_\d{8}_\d{6}$")
_LEVEL_RE = re.compile(r"^level_\d+$")
_SAMPLE_RE = re.compile(r"^sample_\d+$")
_VIEW_RE = re.compile(r"^view_\d+$")

# ── official prompt templates ────────────────────────────────────────────────
# Prompt templates from NVIDIA/Spatial-IQ, BSD-3-Clause
# (inference/prompt_text_Type1.json; copied verbatim, incl. the leading/trailing
# newlines of the two variable blocks). The official clients expand the
# {CRITICAL_INSTRUCTION} / {DEFINITIONS} placeholders with str.format_map.
CRITICAL_INSTRUCTION = (
    "CRITICAL INSTRUCTION: You are undergoing a precision capability test. The input image shows a 3D structure built from objects of the same type, positioned at the center of the image. The objects are primarily boxes or cubes, but may also include stackable objects such as cylindrical objects, cans, and mugs. Answer the following questions using the absolute minimum number of words possible. Output only the exact value, count, or object name. Do not include full sentences, explanations, pleasantries, or context.\n"
)
DEFINITIONS = (
    "\nDEFINITIONS:\n1. A COLUMN is a vertical stack of one or more objects in which each object lies either on the ground or solely on top of the object immediately beneath it.\n2. A LAYER is a horizontal group of one or more objects at the same vertical height in the structure. The objects within the same layer could be laterally detached or attached. A single object also counts as a layer.\n3. A VISIBLE OBJECT is an object with at least one face fully or partially visible in the image.\n4. A CLUSTER is a group of one or more objects that are connected through direct contact, possibly across multiple layers. Two objects are considered connected if they are in direct physical contact (i.e., directly adjacent horizontally or vertically with no gap between them); diagonal placement that results in no contact does not count. This applies to both box-shaped and cylindrical objects. A cluster includes all objects that are connected directly or indirectly through such contacts.\n5. A SUPPORTING OBJECT is any object that is beneath a given object in the same column. A DIRECTLY SUPPORTING OBJECT is the supporting object in immediate contact beneath a given object. If an object is directly supported by the ground, then it has no supporting objects.\n6. A HIDDEN OBJECT is an object with no visible faces in the image. Any HIDDEN OBJECT must be a SUPPORTING OBJECT of at least one visible object. Otherwise, such an object does not exist in a valid structure.\n"
)
TASK_PROMPT_TEMPLATES = {
    "main": "{CRITICAL_INSTRUCTION}{DEFINITIONS}QUESTION: How many TOTAL objects are present in the object structure? Your answer should be a single integer ONLY.",
    "task2": "{CRITICAL_INSTRUCTION}{DEFINITIONS}QUESTION: How many objects are in the top-most layer (i.e., the highest layer) of the whole structure? Your answer should be a single integer ONLY.",
    "task3": "{CRITICAL_INSTRUCTION}{DEFINITIONS}QUESTION: How many columns are in the object structure? Your answer should be a single integer ONLY.",
    "task4": "{CRITICAL_INSTRUCTION}{DEFINITIONS}QUESTION: How many layers are in the object structure? Your answer should be a single integer ONLY.",
    "task5": "{CRITICAL_INSTRUCTION}{DEFINITIONS}QUESTION: How many VISIBLE objects are there in the object structure? Your answer should be a single integer ONLY.",
    "task6": "{CRITICAL_INSTRUCTION}{DEFINITIONS}QUESTION: How many VISIBLE OBJECTS are directly supporting (i.e., directly BELOW) the top-most object(s) in the whole structure? Note: the answer may be 0 if all columns have a height of 1 (i.e., no object has a supporting object). Your answer should be a single integer ONLY.",
    "task7": "{CRITICAL_INSTRUCTION}{DEFINITIONS}QUESTION: How many VISIBLE OBJECTS are in the column(s) that contain the top-most object(s), excluding the top-most object(s) themselves (i.e., every object below the top-most object in those columns)? Note: the answer may be 0 if all columns have a height of 1. Your answer should be a single integer ONLY.",
    "task8": "{CRITICAL_INSTRUCTION}{DEFINITIONS}QUESTION: First, reason about whether any HIDDEN OBJECTS must exist to support the visible structure. For each inferred HIDDEN OBJECT, identify the single VISIBLE OBJECT in immediate contact directly above it. How many such VISIBLE OBJECTS are there (i.e., VISIBLE OBJECTS that are in direct contact directly above a HIDDEN OBJECT)? Do NOT count other objects higher up in the same column. Note: the answer may be 0 if there are no HIDDEN OBJECTS in the scene. Your answer should be a single integer ONLY.",
    "task9": "{CRITICAL_INSTRUCTION}{DEFINITIONS}QUESTION: How many distinct clusters are there in the object structure? Recall that two objects are connected only if they are in direct physical contact horizontally or vertically (not diagonally). Your answer should be a single integer ONLY.",
}
_PROMPT_VARIABLES = {"CRITICAL_INSTRUCTION": CRITICAL_INSTRUCTION, "DEFINITIONS": DEFINITIONS}
# Fully expanded question text per evaluated task (what the model sees verbatim).
QUESTION_TEXT: Dict[str, str] = {
    task: TASK_PROMPT_TEMPLATES[_PROMPT_KEY[task]].format_map(_PROMPT_VARIABLES) for task in TASKS
}


# ── integer parsing (module-level so tests can import it) ────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_MODEL_TOKEN_RE = re.compile(r"<\|begin_of_box\|>|<\|end_of_box\|>|</think>|<think>")
_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")
_LATEX_WRAP_RE = re.compile(
    r"\\(?:text|textbf|textrm|textit|textnormal|mathrm|mathbf|mathit|mbox|"
    r"operatorname|boldsymbol)\s*\{([^{}]*)\}"
)
_LATEX_SPACE_RE = re.compile(r"\\[,;:!> ]|\\(?:quad|qquad|thinspace|enspace|left|right)\b|~")
_MARKDOWN_RE = re.compile(r"[\*`]+|(?<![A-Za-z0-9])_+|_+(?![A-Za-z0-9])")
_EXACT_INT_RE = re.compile(r"[-+]?\d+")
# A number token: optional sign, digits, optional fraction. Not glued to a
# preceding digit nor to "<digit>." (so "7.5" is one token and "1.2.3" never
# yields 3), while "...7" / "No.7" still parse as 7 like the official regex.
_NUM_TOKEN_RE = re.compile(r"(?<!\d)(?<!\d\.)([-+]?)(\d+)(?:\.(\d+))?")

_UNIT_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS_WORDS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_ONES = "one|two|three|four|five|six|seven|eight|nine"
_NUMWORD_RE = re.compile(
    r"\b(?:(" + "|".join(_TENS_WORDS) + r")(?:[-\s]+(" + _ONES + r"))?|("
    + "|".join(_UNIT_WORDS) + r"))\b",
    re.IGNORECASE,
)


def _boxed_spans(text: str) -> List[Tuple[int, int, int]]:
    """(match_start, content_start, content_end) of every balanced ``\\boxed{...}``."""
    spans: List[Tuple[int, int, int]] = []
    pos = 0
    for m in _BOXED_OPEN_RE.finditer(text):
        if m.start() < pos:
            continue
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        if depth:
            break  # unbalanced: ignore this and anything after it
        spans.append((m.start(), m.end(), i - 1))
        pos = i
    return spans


def boxed_contents(text: str) -> List[str]:
    """Contents of every ``\\boxed{...}`` (nested braces allowed), in order."""
    if not text:
        return []
    text = str(text)
    return [text[a:b] for _, a, b in _boxed_spans(text)]


def strip_boxed(text: str) -> str:
    """Replace every ``\\boxed{...}`` (nested braces allowed) with its content."""
    if not text:
        return ""
    text = str(text)
    out: List[str] = []
    pos = 0
    for start, a, b in _boxed_spans(text):
        out.append(text[pos:start])
        out.append(text[a:b])
        pos = b + 1
    out.append(text[pos:])
    return "".join(out).strip()


def clean_latex(text: str) -> str:
    """Unwrap ``\\text{..}``/``\\mathrm{..}``, drop LaTeX spacing macros and stray braces."""
    if not text:
        return ""
    t = str(text)
    for _ in range(3):  # nested wrappers
        t2 = _LATEX_WRAP_RE.sub(r" \1 ", t)
        if t2 == t:
            break
        t = t2
    t = _LATEX_SPACE_RE.sub(" ", t)
    t = re.sub(r"[{}]", " ", t)
    return t.strip()


def strip_model_tokens(text: str) -> str:
    """Official pre-processing: drop <think>...</think> blocks and GLM box tokens."""
    if not text:
        return ""
    t = _THINK_RE.sub("", str(text))
    t = _MODEL_TOKEN_RE.sub("", t)
    return t.strip()


def answer_candidates(text: str) -> List[str]:
    """Texts to parse, most authoritative first.

    [0] the LAST ``\\boxed{}`` content (if any), then the whole text with the
    boxes unwrapped -- both LaTeX-cleaned and with markdown emphasis removed.
    """
    if not text:
        return []
    text = strip_model_tokens(text)
    cands: List[str] = []
    boxed = boxed_contents(text)
    if boxed:
        cands.append(clean_latex(boxed[-1]))
    cands.append(clean_latex(strip_boxed(text)))
    out: List[str] = []
    for c in cands:
        c = _MARKDOWN_RE.sub(" ", c).strip()
        if c and c not in out:
            out.append(c)
    return out


def find_integers(text: str) -> List[int]:
    """All integer tokens in ``text`` in order. "7.0" counts as 7; "7.5" is skipped."""
    ints: List[int] = []
    for sign, whole, frac in _NUM_TOKEN_RE.findall(text or ""):
        if frac and frac.strip("0"):
            continue  # genuine decimal: not an integer answer
        ints.append(int(sign + whole))
    return ints


def find_number_words(text: str) -> List[int]:
    """All spelled-out numbers (zero .. ninety-nine) in ``text`` in order."""
    out: List[int] = []
    for tens, ones, unit in _NUMWORD_RE.findall(text or ""):
        if tens:
            out.append(_TENS_WORDS[tens.lower()] + (_UNIT_WORDS[ones.lower()] if ones else 0))
        else:
            out.append(_UNIT_WORDS[unit.lower()])
    return out


def _parse_one(text: str) -> Optional[int]:
    """Official chain on one candidate: exact int -> single int -> last int -> number word."""
    t = text.strip().strip("\"'").strip()
    if not t:
        return None
    if _EXACT_INT_RE.fullmatch(t):
        return int(t)
    ints = find_integers(t)
    if ints:
        return ints[-1]  # one integer -> that one; several -> the last one
    words = find_number_words(t)
    if words:
        return words[-1]
    return None


def extract_integer(text: str) -> Optional[int]:
    """Integer answer in a model output, or None if nothing parses (never a default)."""
    if text is None:
        return None
    for cand in answer_candidates(str(text)):
        value = _parse_one(cand)
        if value is not None:
            return value
    return None


# ── sample ──────────────────────────────────────────────────────────────────


@dataclass
class SpatialIQSample(BaseBenchmarkSample):
    """One (view, task) item. No ``choices``: the question is open-ended."""

    task: str = ""                 # task key == question_type (task_main, task2, ...)
    task_name: str = ""            # readable name (TASK_NAMES)
    view_id: str = ""              # relative view dir path (data_*/level_1/sample_*/view_*/<variant>)
    structure_id: str = ""         # data_*/level_1/sample_* (sample_NN names repeat across data dirs)
    view_idx: int = 0
    variant: str = ""              # offset<O>_fov<F>_dist<D>
    offset: float = 0.0
    fov: float = 0.0
    dist: float = 0.0
    object_type: str = ""          # cube | factory_box | soup_can | spam
    total_blocks: int = 0
    num_hidden_blocks: int = 0
    num_columns: int = 0
    num_layers: int = 0
    num_visible_blocks: int = 0    # len(gt.visible_blocks) == task5


# ── benchmark ───────────────────────────────────────────────────────────────


class SpatialIQBench(BaseBenchmark):
    """Spatial-IQ (free-response text track) loader and exact-match scorer."""

    data_specific_prompt = "Give the final answer as a single integer."

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        super().__init__(data_path, question_type)

    # ── loading ─────────────────────────────────────────────────────────────

    @staticmethod
    def _subdirs(path: str, pattern: "re.Pattern[str]") -> List[str]:
        """Sorted child directory names of ``path`` matching ``pattern``."""
        try:
            names = sorted(os.listdir(path))
        except OSError:
            return []
        return [n for n in names if pattern.match(n) and os.path.isdir(os.path.join(path, n))]

    @classmethod
    def discover_view_dirs(cls, root: str) -> List[str]:
        """Relative paths of all leaf view dirs (contain view.png + gt.json), sorted."""
        rels: List[str] = []
        for dd in cls._subdirs(root, _DATA_DIR_RE):
            for lvl in cls._subdirs(os.path.join(root, dd), _LEVEL_RE):
                for smp in cls._subdirs(os.path.join(root, dd, lvl), _SAMPLE_RE):
                    for view in cls._subdirs(os.path.join(root, dd, lvl, smp), _VIEW_RE):
                        view_path = os.path.join(root, dd, lvl, smp, view)
                        for var in sorted(os.listdir(view_path)):
                            if var.startswith("."):
                                continue
                            try:
                                files = set(os.listdir(os.path.join(view_path, var)))
                            except OSError:  # a file, not a directory
                                continue
                            if IMAGE_FILE in files and GT_FILE in files:
                                rels.append("/".join((dd, lvl, smp, view, var)))
        rels.sort()  # lexicographic by relative path (stable cohort order)
        return rels

    def _wants(self, sample: SpatialIQSample) -> bool:
        """question_type filter: task key, readable name, 'main', object_type or variant."""
        if not self.question_type_filter:
            return True
        wanted = {str(q).strip().lower() for q in self.question_type_filter}
        keys = {sample.task.lower(), sample.task_name.lower(), sample.object_type.lower(),
                sample.variant.lower()}
        if sample.task == "task_main":
            keys.add("main")
        return bool(keys & wanted)

    @staticmethod
    def _gt_int(gt: Dict[str, Any], key: str, where: str) -> int:
        value = gt.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Spatial-IQ gt.json {where}: field {key!r} must be a non-negative int, got {value!r}")
        return value

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(
                f"Spatial-IQ data dir not found: {self.data_path}. "
                "Download patrickqrim/spatial-iq (data_*/level_1/sample_*/view_*/<variant>/{view.png,gt.json}) there."
            )
        view_dirs = self.discover_view_dirs(self.data_path)
        if not view_dirs:
            raise FileNotFoundError(f"No Spatial-IQ view dirs (view.png + gt.json) found under {self.data_path}")

        for rel in view_dirs:
            view_path = os.path.join(self.data_path, rel)
            with open(os.path.join(view_path, GT_FILE), "r", encoding="utf-8") as f:
                gt = json.load(f)
            parts = rel.split("/")
            structure_id = "/".join(parts[:3])
            variant = parts[4]
            image_path = os.path.join(view_path, IMAGE_FILE)
            visible = gt.get("visible_blocks") or []

            for task in TASKS:
                answer = self._gt_int(gt, task, rel)
                sample = SpatialIQSample(
                    sample_id=f"{rel.replace('/', '__')}#{task}",  # flat id: run.py uses str(sample_id) as the session dir name
                    question=QUESTION_TEXT[task],
                    question_type=task,
                    images=[image_path],
                    answer=str(answer),
                    task=task,
                    task_name=TASK_NAMES[task],
                    view_id=rel,
                    structure_id=structure_id,
                    view_idx=int(gt.get("view_idx", 0)),
                    variant=variant,
                    offset=float(gt.get("offset", 0.0)),
                    fov=float(gt.get("fov", 0.0)),
                    dist=float(gt.get("dist", 0.0)),
                    object_type=str(gt.get("object_type", "")),
                    total_blocks=self._gt_int(gt, "total_blocks", rel),
                    num_hidden_blocks=self._gt_int(gt, "num_hidden_blocks", rel),
                    num_columns=self._gt_int(gt, "num_columns", rel),
                    num_layers=self._gt_int(gt, "num_layers", rel),
                    num_visible_blocks=len(visible),
                )
                if self._wants(sample):
                    self.data.append(sample)

        print(f"[SpatialIQBench] Loaded {len(self.data)} samples from {len(view_dirs)} view dirs")

    # ── answer extraction ───────────────────────────────────────────────────

    def extract_answer(self, prediction: str) -> str:
        """Canonical integer string ("7") or "" if unparsable (never a default value)."""
        value = extract_integer(prediction)
        return "" if value is None else str(value)

    # ── scoring ─────────────────────────────────────────────────────────────

    @staticmethod
    def total_blocks_bin(total_blocks: int) -> str:
        for name, lo, hi in TOTAL_BLOCKS_BINS:
            if lo <= total_blocks <= hi:
                return name
        return "other"

    @staticmethod
    def _score(gt: int, pred: Optional[int]) -> Tuple[bool, bool, Optional[int]]:
        """(exact, within 10% relative tolerance, |pred-gt|) for one item; unparsed -> misses."""
        if pred is None:
            return False, False, None
        err = abs(pred - gt)
        exact = err == 0
        ra10 = exact if gt == 0 else err <= 0.1 * gt
        return exact, ra10, err

    def evaluate_single(self, sample: BaseBenchmarkSample, prediction: str) -> Optional[float]:
        """1.0 on exact integer match, else 0.0 (unparsable -> 0.0)."""
        exact, _, _ = self._score(int(sample.answer), extract_integer(prediction))
        return 1.0 if exact else 0.0

    @staticmethod
    def _new_stats() -> Dict[str, Any]:
        return {"correct": 0, "total": 0, "parsed": 0, "ra10_ok": 0, "abs_err_sum": 0}

    @staticmethod
    def _bump(stats: Dict[str, Any], ok: bool, ra10: bool, err: Optional[int]) -> None:
        stats["total"] += 1
        stats["correct"] += int(ok)
        stats["ra10_ok"] += int(ra10)
        if err is not None:
            stats["parsed"] += 1
            stats["abs_err_sum"] += err

    @staticmethod
    def _finalize(stats: Dict[str, Any]) -> Dict[str, Any]:
        total = stats["total"]
        parsed = stats["parsed"]
        return {
            "accuracy": stats["correct"] / max(total, 1),
            "correct": stats["correct"],
            "total": total,
            "parsed": parsed,
            "unparsed": total - parsed,
            "mae": (stats["abs_err_sum"] / parsed) if parsed else None,
            "ra10": stats["ra10_ok"] / max(total, 1),
        }

    def _table(self, stats: Dict[str, Dict[str, Any]], order: List[str]) -> Dict[str, Any]:
        keys = [k for k in order if k in stats] + sorted(set(stats) - set(order))
        return {k: self._finalize(stats[k]) for k in keys}

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        overall = self._new_stats()
        per_task: Dict[str, Dict[str, Any]] = {}
        per_variant: Dict[str, Dict[str, Any]] = {}
        per_offset: Dict[str, Dict[str, Any]] = {}
        per_fov_dist: Dict[str, Dict[str, Any]] = {}
        per_object_type: Dict[str, Dict[str, Any]] = {}
        per_bin: Dict[str, Dict[str, Any]] = {}
        detailed: List[Dict[str, Any]] = []
        unparsed = 0
        missing = 0
        # view_id -> {task: (gt, pred_or_None, present)} for the hierarchy diagnostic
        by_view: Dict[str, Dict[str, Tuple[int, Optional[int], bool]]] = {}

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid, default=None)
            present = pred_raw is not None
            if not present:
                missing += 1
                pred_raw = ""
            gt = int(sample.answer)
            pred = extract_integer(pred_raw)
            ok, ra10, err = self._score(gt, pred)
            if pred is None:
                unparsed += 1

            variant = sample.variant
            offset_key = f"offset{sample.offset:g}"
            fov_dist_key = f"fov{sample.fov:g}_dist{sample.dist:g}"
            tb_bin = self.total_blocks_bin(sample.total_blocks)

            self._bump(overall, ok, ra10, err)
            for table, key in (
                (per_task, sample.task),
                (per_variant, variant),
                (per_offset, offset_key),
                (per_fov_dist, fov_dist_key),
                (per_object_type, sample.object_type),
                (per_bin, tb_bin),
            ):
                self._bump(table.setdefault(key, self._new_stats()), ok, ra10, err)

            if sample.task in ("task_main", "task5"):
                by_view.setdefault(sample.view_id, {})[sample.task] = (gt, pred, present)

            detailed.append({
                "sample_id": sid,
                "question_type": sample.question_type,
                "task_name": sample.task_name,
                "view_id": sample.view_id,
                "structure_id": sample.structure_id,
                "variant": variant,
                "object_type": sample.object_type,
                "total_blocks": sample.total_blocks,
                "total_blocks_bin": tb_bin,
                "ground_truth": sample.answer,
                "prediction": pred_raw,
                "extracted": "" if pred is None else str(pred),
                "parsed": pred is not None,
                "correct": ok,
                "abs_error": err,
                "within_10pct": ra10,
            })

        # Hierarchy diagnostic: implied hidden count (main - visible) must match GT.
        n_pairs = n_consistent = n_pairs_unparsed = 0
        for tasks in by_view.values():
            if "task_main" not in tasks or "task5" not in tasks:
                continue
            (g_main, p_main, pres_main), (g_vis, p_vis, pres_vis) = tasks["task_main"], tasks["task5"]
            if not (pres_main and pres_vis):
                continue
            n_pairs += 1
            if p_main is None or p_vis is None:
                n_pairs_unparsed += 1
                continue
            if p_main - p_vis == g_main - g_vis:
                n_consistent += 1

        per_task_out = self._table(per_task, TASKS)
        for task, entry in per_task_out.items():
            entry["name"] = TASK_NAMES.get(task, task)
        overall_out = self._finalize(overall)
        headline = per_task_out.get("task_main", {}).get("accuracy")

        results: Dict[str, Any] = {
            "total_samples": overall_out["total"],
            "correct_samples": overall_out["correct"],
            "overall_accuracy": overall_out["accuracy"],
            "overall_aggregation": (
                "overall_accuracy = exact-match micro accuracy over all loaded (view, task) items, "
                "unparsable/missing predictions scored 0 and kept in the denominator; the paper's "
                "headline is the task_main accuracy (headline_task_main_accuracy); per_task rows "
                "reproduce the official per-task % correct except for the unparsable handling"
            ),
            "scoring": (
                "exact integer match (official evaluation_text.py chain + \\boxed{} / number-word "
                "handling); official scorer excludes unparsable predictions from % correct, here they count as wrong"
            ),
            "headline_task_main_accuracy": headline,
            "overall_mae": overall_out["mae"],
            "overall_ra10": overall_out["ra10"],
            "unparsed_predictions": unparsed,
            "missing_predictions": missing,
            "per_task": per_task_out,
            "per_variant": self._table(per_variant, VARIANTS),
            "per_offset": self._table(per_offset, ["offset3", "offset12.5"]),
            "per_fov_dist": self._table(per_fov_dist, ["fov1_dist1", "fov3_dist0.3"]),
            "per_object_type": self._table(per_object_type, OBJECT_TYPES),
            "per_total_blocks_bin": self._table(per_bin, [b[0] for b in TOTAL_BLOCKS_BINS]),
            "summation_consistency": {
                "fraction": (n_consistent / n_pairs) if n_pairs else None,
                "n": n_pairs,
                "n_consistent": n_consistent,
                "n_pairs_with_unparsed": n_pairs_unparsed,
                "definition": (
                    "views whose task_main and task5 predictions are both present; consistent iff both parse "
                    "and pred_main - pred_task5 == gt_main - gt_task5 (= num_hidden_blocks)"
                ),
            },
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), detailed)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        def _pct(v: Optional[float]) -> str:
            return "   n/a " if v is None else f"{v * 100:6.2f}%"

        def _mae(v: Optional[float]) -> str:
            return "  n/a" if v is None else f"{v:5.2f}"

        print(f"\n{'=' * 78}")
        print("Spatial-IQ (free-response text track) Results")
        print(f"{'=' * 78}")
        print(f"Total samples: {results['total_samples']}   Correct: {results['correct_samples']}   "
              f"Unparsed: {results.get('unparsed_predictions', 0)}   Missing: {results.get('missing_predictions', 0)}")
        print(f"Overall exact-match accuracy: {_pct(results['overall_accuracy'])}   "
              f"MAE: {_mae(results.get('overall_mae'))}   RA@10%: {_pct(results.get('overall_ra10'))}")
        print(f"Headline task_main accuracy:  {_pct(results.get('headline_task_main_accuracy'))}")
        sc = results.get("summation_consistency", {})
        print(f"Summation consistency (main - visible == hidden): {_pct(sc.get('fraction'))}  "
              f"(n={sc.get('n', 0)}, unparsed pairs={sc.get('n_pairs_with_unparsed', 0)})")
        print()
        for title, key in (
            ("Task", "per_task"),
            ("Camera variant", "per_variant"),
            ("Offset", "per_offset"),
            ("FOV / distance", "per_fov_dist"),
            ("Object type", "per_object_type"),
            ("Total blocks bin", "per_total_blocks_bin"),
        ):
            table = results.get(key, {})
            if not table:
                continue
            print(f"{title:<38s} {'Acc':>8s} {'Correct':>8s} {'Total':>6s} {'MAE':>6s} {'RA@10%':>8s}")
            print(f"{'-' * 38} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 8}")
            for name, info in table.items():
                label = f"{name} ({info['name']})" if key == "per_task" and info.get("name") else name
                print(
                    f"{label:<38s} {info['accuracy'] * 100:>7.2f}% {info['correct']:>8d} {info['total']:>6d} "
                    f"{_mae(info.get('mae')):>6s} {_pct(info.get('ra10')):>8s}"
                )
            print()
        print(f"{'=' * 78}\n")
