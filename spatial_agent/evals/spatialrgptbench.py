"""SpatialRGPT-Bench data loader (NeurIPS 2024).

SpatialRGPT-Bench (Cheng et al., "SpatialRGPT: Grounded Spatial Reasoning in
Vision-Language Models", arXiv 2406.01584) is a region-referring spatial VQA
benchmark: 1,406 single-image, open-ended questions about one or two marked
regions of an image.

    qualitative  (657): 12 *_predicate categories (yes/no) + 6 *_choice
                        categories (pick Region [0] or Region [1]); grouped
                        by the official scorer into 6 buckets
                        below/above, left/right, big/small, tall/short,
                        wide/thin, behind/front (substring of the category)
    quantitative (749): width_data, height_data (1 region, a size),
                        distance_data, horizontal_distance_data,
                        vertical_distance_data (2 regions, a distance),
                        direction (2 regions, a clock hour 1-12)

Data structure (HF repo a8cheng/SpatialRGPT-Bench; official eval reads the JSON):
    data/SpatialRGPT-Bench/SpatialRGPT-Bench_v1.json    1,406 items (valid JSON)
    data/SpatialRGPT-Bench/images/<id>.png              one raw image per item

JSON fields used:
    id             str    unique, "qualitative_<10>" | "quantitative_<10>" -> sample_id
    image_info     dict   {height, width, dataset, landmark, file_path}
    qa_info        dict   {type, category, class}  (class names are NOT exposed
                          to the model, matching the official region-aware setup)
    conversations  list   [human, gpt]; conversations[1]['value'] = free-form GT
    text_q         str    question with "Region [k]" tokens (k = bbox order)
    bbox           list   one pixel-space [x1, y1, x2, y2] per region
    file_path      str    "images/<id>.png"

Region convention (benchmark-agnostic, coordinates-in-text): the question is
text_q verbatim, followed by one sentence stating that region boxes are
normalized 0-1000 [x1, y1, x2, y2] image coordinates, one legend line per
region, and a one-sentence answer-format hint derived from the item's own
question kind (yes/no, region, number+unit, clock position). The image is
passed unmodified (no boxes drawn).

Metric -- DETERMINISTIC PROXY of the official GPT-4-turbo judge
(scripts/srgpt/eval/evaluate_spatial_with_gpt4.py). This is NOT the official
scorer; results are flagged ``scoring = "deterministic_proxy (non-official)"``.
    All parsers look at the LAST ``\\boxed{}`` content first (LaTeX ``\\text{}``
    wrappers and spacing removed), then at the whole text with boxes unwrapped.
    numeric   : last number+unit in the prediction (bare number -> GT unit),
                both converted to meters; success iff 0.75*gt <= pred <= 1.25*gt;
                abs_rel = |pred-gt|/gt (mean over parsed predictions only)
    direction : integer clock hour 1-12 (0 -> 12); success iff the circular
                difference <= 1; error reported as mean(diff)*30 degrees
    predicate : GT polarity from its first token (Yes/Correct/Indeed/That is True
                vs No/Incorrect/Actually/In fact/That is False); prediction
                polarity: leading yes/no -> "answer ... yes/no" -> first yes/no
                -> (not) true/false/correct/incorrect; success iff equal
    choice    : GT region = the single "Region [k]" in the GT; prediction region
                = the only region named, else the one after a conclusion marker
                ("the answer is Region [1]", "so Region [1]"), else the one NOT
                after "than", else the most frequent, else the first (or a bare
                0/1 / first/second); success iff equal
    unparsable -> 0.0 (counted in the denominator, like the official scorer)
Reported: per-bucket qualitative accuracy + qualitative overall (n=657),
per-category quantitative success rate (+ mean abs_rel / mean clock error in
degrees) + quantitative overall (n=749), and ``overall_accuracy`` =
correct/total over all loaded items (the paper defines no single overall).

Reference: https://github.com/AnjieCheng/SpatialRGPT
           https://huggingface.co/datasets/a8cheng/SpatialRGPT-Bench
"""

import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.scoring import (
    get_prediction,
    mean_or_zero,
    write_json,
    write_results_summary,
)

ANNOTATION_FILE = "SpatialRGPT-Bench_v1.json"
IMAGE_DIR = "images"

NUMERIC_CATEGORIES = [
    "distance_data",
    "horizontal_distance_data",
    "vertical_distance_data",
    "width_data",
    "height_data",
]
DIRECTION_CATEGORY = "direction"
QUANTITATIVE_CATEGORIES = NUMERIC_CATEGORIES + [DIRECTION_CATEGORY]

# Official qualitative buckets (evaluate_spatial_with_gpt4.py lines 350-363):
# a category is assigned to the first bucket whose substring it contains.
QUALITATIVE_BUCKETS: List[Tuple[str, Tuple[str, ...]]] = [
    ("below/above", ("below", "above")),
    ("left/right", ("left", "right")),
    ("big/small", ("big", "small")),
    ("tall/short", ("tall", "short")),
    ("wide/thin", ("wide", "thin")),
    ("behind/front", ("behind", "front")),
]
BUCKET_ORDER = [name for name, _ in QUALITATIVE_BUCKETS]

KIND_NUMERIC = "numeric"
KIND_DIRECTION = "direction"
KIND_PREDICATE = "predicate"
KIND_CHOICE = "choice"

ANSWER_FORMAT_HINT = {
    KIND_PREDICATE: "Answer yes or no.",
    KIND_CHOICE: "Answer with the region, e.g. Region [0] or Region [1].",
    KIND_NUMERIC: "Answer with a single number and its unit.",
    KIND_DIRECTION: "Answer with a clock position from 1 to 12.",
}
REGION_LEGEND_INTRO = (
    "Region boxes are given as normalized 0-1000 [x1, y1, x2, y2] image coordinates."
)

# Official tolerance (evaluate_spatial_with_gpt4.py lines 279-281).
NUMERIC_LOWER, NUMERIC_UPPER = 0.75, 1.25
CLOCK_TOLERANCE_HOURS = 1


# ── parsing helpers (module-level so tests can import them) ─────────────────

_NUM_RE = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"

# (regex alternative, meters per unit, canonical label). Longest forms first so
# "mm"/"cm" win over "m" and "inches" over "in".
_UNIT_SPECS: List[Tuple[str, float, str]] = [
    (r"millimet(?:er|re)s?|mm", 0.001, "mm"),
    (r"centimet(?:er|re)s?|cm", 0.01, "cm"),
    (r"met(?:er|re)s?|m", 1.0, "m"),
    (r"inches|inch", 0.0254, "in"),
    # bare "in" is a preposition too: reject when followed by a common noun phrase
    (
        r"in(?!\s+(?:the|a|an|this|that|these|those|total|front|which|order|"
        r"terms|all|both|each|between|my|our|your|its|their)\b)",
        0.0254,
        "in",
    ),
    (r"\"|″|''", 0.0254, "in"),
    (r"feet|foot|ft", 0.3048, "ft"),
    (r"'|′", 0.3048, "ft"),
]
_UNIT_ALT = "|".join(f"(?:{pat})" for pat, _, _ in _UNIT_SPECS)
_LENGTH_RE = re.compile(
    rf"(?<![\w.])(?P<num>{_NUM_RE})\s*(?P<unit>{_UNIT_ALT})(?![A-Za-z])",
    re.IGNORECASE,
)
_UNIT_LOOKUP: List[Tuple[re.Pattern, float, str]] = [
    (re.compile(rf"^(?:{pat})$", re.IGNORECASE), factor, label)
    for pat, factor, label in _UNIT_SPECS
]
_GT_UNIT_FACTOR = {"inches": 0.0254, "feet": 0.3048, "meters": 1.0, "centimeters": 0.01}
_GT_UNIT_LABEL = {"inches": "in", "feet": "ft", "meters": "m", "centimeters": "cm"}

_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")
# LaTeX wrappers/spacing that CoT answers put around units: \boxed{0.76\,\text{m}}
_LATEX_WRAP_RE = re.compile(
    r"\\(?:text|textbf|textrm|textit|textnormal|mathrm|mathbf|mathit|mbox|"
    r"operatorname|boldsymbol)\s*\{([^{}]*)\}"
)
_LATEX_SPACE_RE = re.compile(r"\\[,;:!> ]|\\(?:quad|qquad|thinspace|enspace|left|right)\b|~")
# symbol units (30", 5') and the quote characters that make a number "quoted", not measured
_QUOTE_MARKS = {'"', "″", "''", "'", "′"}
_QUOTE_CHARS = set("\"'“”‘’″′`´")

_REGION_TOKEN_RE = re.compile(r"region\s*\[\s*\d+\s*\]", re.IGNORECASE)
_REGION_IDX_RE = re.compile(r"region\s*\[?\s*(\d+)\s*\]?", re.IGNORECASE)
# "the answer is Region [1]" / "..., so Region [1] is taller" -> conclusion marker
_REGION_MARKER_RE = re.compile(
    r"\b(?:answer|choice|option|conclusion|so|therefore|thus|hence|is)\b\s*(?:is\s*)?"
    r"[:\-–]?\s*(?:the\s+)?region\s*\[?\s*(\d+)",
    re.IGNORECASE,
)
# "Region [1] is taller than Region [0]" -> the region after 'than' is NOT the answer
_REGION_THAN_RE = re.compile(
    r"\b(?:than|compared\s+(?:to|with)|relative\s+to|versus|vs\.?)\s+(?:the\s+)?region\s*\[?\s*(\d+)",
    re.IGNORECASE,
)
_BARE_IDX_RE = re.compile(r"^\W*\[?\s*(\d+)\s*\]?\W*$")
_BARE_NUM_RE = re.compile(rf"(?<![\w.]){_NUM_RE}(?![\w.]\w)")
_BARE_INT_RE = re.compile(r"(?<![\d.])(\d{1,2})(?!\.?\d)")

_CLOCK_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_OCLOCK_RE = re.compile(
    r"\b(\d{1,2}|" + "|".join(_CLOCK_WORDS) + r")\s*(?::\s*\d{2})?\s*o\s*['’`´]?\s*clock",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"(?<![\d.])(\d{1,2})\s*:\s*\d{2}(?![\d.])")

# yes/no wins over true/false/correct/incorrect ("The correct answer is no." -> no);
# a leading token or an "answer ... yes/no" marker wins over a yes/no anywhere.
_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_YESNO_LEAD_RE = re.compile(r"^[\W_]*(yes|no)\b", re.IGNORECASE)
_YESNO_MARKER_RE = re.compile(r"\banswer\b[^.\n]{0,20}?\b(yes|no)\b", re.IGNORECASE)
_TRUTH_RE = re.compile(r"\b(?:(not)\s+)?(true|false|correct|incorrect)\b", re.IGNORECASE)
_AFFIRMATIVE = {"yes", "true", "correct"}
_GT_AFFIRMATIVE_FIRST = {"yes", "correct", "indeed"}
_GT_NEGATIVE_FIRST = {"no", "incorrect", "actually"}



def _maybe_literal(value: Any) -> Any:
    """Parse a Python-repr string (single-quoted dict/list) if needed; pass through otherwise."""
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value

def _unit_factor(unit_text: str) -> Tuple[float, str]:
    for pat, factor, label in _UNIT_LOOKUP:
        if pat.match(unit_text):
            return factor, label
    raise ValueError(f"unknown unit {unit_text!r}")


def _to_float(num_text: str) -> float:
    return float(num_text.replace(",", ""))


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


def answer_candidates(text: str) -> List[str]:
    """Texts to parse, most authoritative first.

    [0] the LAST ``\\boxed{}`` content (if any), then the whole text with the
    boxes unwrapped -- both LaTeX-cleaned. Parsers try them in order so an
    explicit boxed answer wins over stray numbers/regions in the reasoning.
    """
    if not text:
        return []
    text = str(text)
    cands: List[str] = []
    boxed = boxed_contents(text)
    if boxed:
        cands.append(clean_latex(boxed[-1]))
    cands.append(clean_latex(strip_boxed(text)))
    out: List[str] = []
    for c in cands:
        if c and c not in out:
            out.append(c)
    return out


def kind_for_category(category: str) -> str:
    """Map a qa_info.category to one of the four answer kinds."""
    if category == DIRECTION_CATEGORY:
        return KIND_DIRECTION
    if category in NUMERIC_CATEGORIES or category.endswith("_data"):
        return KIND_NUMERIC
    if category.endswith("_choice"):
        return KIND_CHOICE
    if category.endswith("_predicate"):
        return KIND_PREDICATE
    raise ValueError(f"cannot infer answer kind for category {category!r}")


def bucket_for_category(category: str) -> str:
    """Official 6-way qualitative bucket, or "" for quantitative categories."""
    if category in QUANTITATIVE_CATEGORIES:
        return ""
    for name, subs in QUALITATIVE_BUCKETS:
        if any(s in category for s in subs):
            return name
    return ""


def parse_length_meters(
    text: str, default_unit_factor: Optional[float] = None
) -> Optional[Tuple[float, str]]:
    """Parse the LAST number+unit in ``text`` -> (meters, canonical form).

    A bare number (no recognised unit anywhere) is accepted only when
    ``default_unit_factor`` is given (the GT's unit). ``Region [k]`` tokens
    are ignored so their indices are never mistaken for the answer.
    """
    for cand in answer_candidates(text):
        parsed = _parse_length_explicit(cand)
        if parsed is None and default_unit_factor is not None:
            parsed = _parse_length_bare(cand, default_unit_factor)
        if parsed is not None:
            return parsed
    return None


def _parse_length_explicit(text: str) -> Optional[Tuple[float, str]]:
    """LAST number with an explicit unit in ``text`` -> (meters, canonical); None if none."""
    matches = list(_LENGTH_RE.finditer(text))
    while matches:
        last = matches[-1]
        unit_text = last.group("unit")
        q = last.start("num") - 1
        # a quoted bare number ("30", '5') is not an inch/foot mark
        if (
            unit_text in _QUOTE_MARKS
            and q >= 0
            and text[q] in _QUOTE_CHARS
            and not (len(matches) >= 2 and matches[-2].end() > q)  # 2'6": the ' belongs to 2'
        ):
            matches.pop()
            continue
        value = _to_float(last.group("num"))
        factor, label = _unit_factor(unit_text)
        meters = value * factor
        canonical = f"{value:g} {label}"
        # "5 ft 3 in" / "5' 3\"" -> combine feet and inches
        if label == "in" and len(matches) >= 2:
            prev = matches[-2]
            between = text[prev.end():last.start()]
            _, prev_label = _unit_factor(prev.group("unit"))
            if prev_label == "ft" and re.fullmatch(r"[\s,]*(?:and\s+)?", between):
                feet = _to_float(prev.group("num"))
                meters = feet * 0.3048 + value * 0.0254
                canonical = f"{feet:g} ft {value:g} in"
        return meters, canonical
    return None


def _parse_length_bare(text: str, default_unit_factor: float) -> Optional[Tuple[float, str]]:
    """LAST bare number (Region [k] indices ignored) in the given default unit."""
    cleaned = _REGION_TOKEN_RE.sub(" ", text)
    nums = _BARE_NUM_RE.findall(cleaned)
    if not nums:
        return None
    value = _to_float(nums[-1])
    return value * default_unit_factor, f"{value:g}"


def parse_gt_length_meters(gt: str) -> Optional[Tuple[float, str]]:
    """Parse the single '<number> <unit>' of a GT sentence -> (meters, unit word)."""
    m = re.search(rf"({_NUM_RE})\s*(inches|feet|meters|centimeters)\b", gt or "", re.IGNORECASE)
    if not m:
        return None
    unit = m.group(2).lower()
    return _to_float(m.group(1)) * _GT_UNIT_FACTOR[unit], unit


def parse_clock_hour(text: str) -> Optional[int]:
    """Parse a clock position 1-12 (0 -> 12); None if nothing parses."""
    for cand in answer_candidates(text):
        hour = _parse_clock_hour_one(cand)
        if hour is not None:
            return hour
    return None


def _parse_clock_hour_one(text: str) -> Optional[int]:
    hour: Optional[int] = None
    m_all = list(_OCLOCK_RE.finditer(text))
    if m_all:
        tok = m_all[-1].group(1).lower()
        hour = _CLOCK_WORDS.get(tok, None) if not tok.isdigit() else int(tok)
    else:
        t_all = list(_TIME_RE.finditer(text))
        if t_all:
            hour = int(t_all[-1].group(1))
        else:
            cleaned = _REGION_TOKEN_RE.sub(" ", text)
            ints = [int(x) for x in _BARE_INT_RE.findall(cleaned)]
            ints = [x for x in ints if 0 <= x <= 12]
            if ints:
                hour = ints[-1]
            else:  # a lone clock word: "three" / "Answer: three"
                w = re.fullmatch(
                    r"[\W_]*(?:(?:final\s+)?answer\W*)?(" + "|".join(_CLOCK_WORDS) + r")[\W_]*",
                    cleaned, re.IGNORECASE,
                )
                if w:
                    hour = _CLOCK_WORDS[w.group(1).lower()]
    if hour is None:
        return None
    if hour == 0:
        hour = 12
    if not 1 <= hour <= 12:
        return None
    return hour


def clock_diff_hours(a: int, b: int) -> int:
    d = abs(a - b) % 12
    return min(d, 12 - d)


def parse_prediction_polarity(text: str) -> Optional[bool]:
    """Polarity of a model answer (None if it has no yes/no-like token).

    Priority: leading yes/no -> "answer ... yes/no" marker -> first yes/no
    anywhere -> (not) true/false/correct/incorrect. The boxed content, if
    any, is tried before the full text.
    """
    for cand in answer_candidates(text):
        pol = _parse_polarity_one(cand)
        if pol is not None:
            return pol
    return None


def _parse_polarity_one(text: str) -> Optional[bool]:
    m = _YESNO_LEAD_RE.match(text) or _YESNO_MARKER_RE.search(text) or _YESNO_RE.search(text)
    if m:
        return m.group(1).lower() == "yes"
    m = _TRUTH_RE.search(text)
    if not m:
        return None
    affirmative = m.group(2).lower() in _AFFIRMATIVE
    if m.group(1):  # "not true" / "not correct"
        affirmative = not affirmative
    return affirmative


def parse_gt_polarity(gt: str) -> Optional[bool]:
    """Polarity of a GT sentence from its first token(s)."""
    if not gt:
        return None
    low = gt.strip().lower()
    if low.startswith("that is true"):
        return True
    if low.startswith("that is false"):
        return False
    if low.startswith("in fact"):
        return False
    first = re.split(r"[\s,.!;:]+", low, maxsplit=1)[0]
    if first in _GT_AFFIRMATIVE_FIRST:
        return True
    if first in _GT_NEGATIVE_FIRST:
        return False
    return None


def parse_region_index(text: str) -> Optional[int]:
    """Region index named by a model answer (None if none).

    One distinct 'Region [k]' -> k. Several -> the region after a conclusion
    marker ("the answer is Region [1]", "..., so Region [1] is taller"), else
    the region NOT following 'than' ("Region [1] is taller than Region [0]"),
    else the most frequently named, else the first. A bare 'k' / '[k]' or
    'first'/'second' is accepted when no region token is present. The boxed
    content, if any, is tried before the full text.
    """
    for cand in answer_candidates(text):
        idx = _parse_region_index_one(cand)
        if idx is not None:
            return idx
    return None


def _parse_region_index_one(text: str) -> Optional[int]:
    found = [int(m.group(1)) for m in _REGION_IDX_RE.finditer(text)]
    if found:
        distinct = sorted(set(found))
        if len(distinct) == 1:
            return distinct[0]
        markers = list(_REGION_MARKER_RE.finditer(text))
        if markers:
            return int(markers[-1].group(1))
        if len(distinct) == 2:
            than = list(_REGION_THAN_RE.finditer(text))
            if than:
                return next(k for k in distinct if k != int(than[-1].group(1)))
        ranked = sorted(distinct, key=lambda k: -found.count(k))
        if found.count(ranked[0]) > found.count(ranked[1]):
            return ranked[0]
        return found[0]
    m = _BARE_IDX_RE.match(text)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(first|second)\b", text, re.IGNORECASE)
    if m:
        return 0 if m.group(1).lower() == "first" else 1
    return None


def parse_gt_region_index(gt: str) -> Optional[int]:
    found = re.findall(r"Region\s*\[(\d+)\]", gt or "")
    return int(found[0]) if len(found) == 1 else None


# ── sample ──────────────────────────────────────────────────────────────────


@dataclass
class SpatialRGPTBenchSample(BaseBenchmarkSample):
    """One SpatialRGPT-Bench item (no ``choices``: the question is open-ended)."""

    qa_type: str = ""            # "qualitative" | "quantitative" (qa_info.type)
    category: str = ""           # fine category (qa_info.category) == question_type
    bucket: str = ""             # official 6-way qualitative bucket, "" for quantitative
    expected_kind: str = ""      # numeric | direction | predicate | choice
    text_q: str = ""             # verbatim question (what the official judge sees)
    bboxes: List[List[int]] = field(default_factory=list)       # pixel [x1,y1,x2,y2] per region
    bboxes_norm: List[List[int]] = field(default_factory=list)  # normalized 0-1000 per region
    image_width: int = 0
    image_height: int = 0
    source_dataset: str = ""     # image_info.dataset (hypersim, SUNRGBD, ...)
    landmark: str = ""           # image_info.landmark (indoor | outdoor)
    object_classes: List[str] = field(default_factory=list)  # qa_info.class (metadata only, never shown)


# ── benchmark ───────────────────────────────────────────────────────────────


class SpatialRGPTBench(BaseBenchmark):
    """SpatialRGPT-Bench loader with a deterministic proxy of the official judge."""

    data_specific_prompt = "Give your final answer in the requested format."

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._image_dir = os.path.join(data_path, IMAGE_DIR)
        super().__init__(data_path, question_type)

    # ── loading ─────────────────────────────────────────────────────────────

    def _matches_filter(self, sample: SpatialRGPTBenchSample) -> bool:
        if not self.question_type_filter:
            return True
        wanted = {str(q).strip().lower() for q in self.question_type_filter}
        keys = {
            sample.category.lower(),
            sample.qa_type.lower(),
            sample.expected_kind.lower(),
            sample.bucket.lower(),
        }
        return bool(wanted & keys)

    @staticmethod
    def _normalize_box(box: List[int], width: int, height: int) -> List[int]:
        x1, y1, x2, y2 = box
        return [
            int(round(x1 / width * 1000)),
            int(round(y1 / height * 1000)),
            int(round(x2 / width * 1000)),
            int(round(y2 / height * 1000)),
        ]

    @staticmethod
    def _build_question(text_q: str, bboxes_norm: List[List[int]], kind: str) -> str:
        lines = [text_q.strip(), REGION_LEGEND_INTRO]
        for k, box in enumerate(bboxes_norm):
            lines.append(f"Region [{k}]: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
        lines.append(ANSWER_FORMAT_HINT[kind])
        return "\n".join(lines)

    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        self._image_dir = os.path.join(self.data_path, IMAGE_DIR)
        ann_path = os.path.join(self.data_path, ANNOTATION_FILE)
        if not os.path.exists(ann_path):
            raise FileNotFoundError(
                f"SpatialRGPT-Bench annotations not found: {ann_path}. "
                "Download via spatial_agent/scripts/download_datasets.sh"
            )
        with open(ann_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        for item in items:  # file order is the official order (deterministic)
            # The shipped JSON stores these as real dicts/lists; the HF parquet copy
            # stores them as Python-repr strings, so accept both.
            image_info = _maybe_literal(item["image_info"])
            qa_info = _maybe_literal(item["qa_info"])
            bboxes = _maybe_literal(item["bbox"])
            conversations = _maybe_literal(item["conversations"])

            category = str(qa_info["category"])
            kind = kind_for_category(category)
            width, height = int(image_info["width"]), int(image_info["height"])
            bboxes = [[int(v) for v in box] for box in bboxes]
            bboxes_norm = [self._normalize_box(box, width, height) for box in bboxes]
            text_q = str(item["text_q"]).strip()

            rel_path = item.get("file_path") or f"{IMAGE_DIR}/{item['id']}.png"
            image_path = os.path.join(self.data_path, rel_path)

            sample = SpatialRGPTBenchSample(
                sample_id=str(item["id"]),
                question=self._build_question(text_q, bboxes_norm, kind),
                question_type=category,
                images=[image_path],
                answer=str(conversations[1]["value"]).strip(),
                qa_type=str(qa_info["type"]),
                category=category,
                bucket=bucket_for_category(category),
                expected_kind=kind,
                text_q=text_q,
                bboxes=bboxes,
                bboxes_norm=bboxes_norm,
                image_width=width,
                image_height=height,
                source_dataset=str(image_info.get("dataset", "")),
                landmark=str(image_info.get("landmark", "")),
                object_classes=[str(c) for c in (qa_info.get("class") or [])],
            )
            if not self._matches_filter(sample):
                continue
            self.data.append(sample)

        print(f"[SpatialRGPTBench] Loaded {len(self.data)} samples")

    # ── answer extraction ───────────────────────────────────────────────────

    def extract_answer(self, prediction: str, kind: Optional[str] = None) -> str:
        """Canonical answer text or "" if unparsable (never a default value).

        ``kind`` selects the parser (numeric | direction | predicate | choice);
        without it the parsers are tried in a fixed order. Handles the raw
        agent text and the ``\\boxed{}``-stripped CoT text alike. Bare numbers
        for numeric questions need the GT unit and are resolved in
        evaluate_single; here they are returned as the bare number string.
        """
        if not prediction:
            return ""
        cands = answer_candidates(prediction)
        if not cands:
            return ""
        if kind == KIND_NUMERIC:
            parsed = parse_length_meters(prediction, default_unit_factor=1.0)
            return parsed[1] if parsed else ""
        if kind == KIND_DIRECTION:
            hour = parse_clock_hour(prediction)
            return f"{hour} o'clock" if hour is not None else ""
        if kind == KIND_PREDICATE:
            pol = parse_prediction_polarity(prediction)
            return "" if pol is None else ("yes" if pol else "no")
        if kind == KIND_CHOICE:
            idx = parse_region_index(prediction)
            return f"Region [{idx}]" if idx is not None else ""
        # kind unknown: decide on the most authoritative candidate (the boxed content
        # if present): explicit region -> yes/no -> o'clock -> number+unit -> bare number
        text = cands[0]
        if _REGION_IDX_RE.search(text):
            idx = parse_region_index(text)
            return f"Region [{idx}]" if idx is not None else ""
        pol = parse_prediction_polarity(text)
        if pol is not None:
            return "yes" if pol else "no"
        if _OCLOCK_RE.search(text) or _TIME_RE.search(text):
            hour = parse_clock_hour(text)
            return f"{hour} o'clock" if hour is not None else ""
        parsed = parse_length_meters(text, default_unit_factor=1.0)
        return parsed[1] if parsed else ""

    # ── scoring ─────────────────────────────────────────────────────────────

    def score_detail(self, sample: SpatialRGPTBenchSample, prediction: str) -> Dict[str, Any]:
        """Per-item proxy score with the parsed values (used by evaluate/evaluate_single)."""
        kind = sample.expected_kind or kind_for_category(sample.question_type)
        gt = sample.answer
        detail: Dict[str, Any] = {"kind": kind, "extracted": "", "parsed": False, "score": 0.0}
        text = prediction or ""

        if kind == KIND_NUMERIC:
            gt_parsed = parse_gt_length_meters(gt)
            if gt_parsed is None:
                detail["gt_parse_error"] = True
                return detail
            gt_m, gt_unit = gt_parsed
            detail["gt_m"] = gt_m
            pred_parsed = parse_length_meters(text, default_unit_factor=_GT_UNIT_FACTOR[gt_unit])
            if pred_parsed is None:
                return detail
            pred_m, canonical = pred_parsed
            if canonical and not re.search(r"[A-Za-z\"'″′]", canonical):
                canonical = f"{canonical} {_GT_UNIT_LABEL[gt_unit]} (unit assumed from GT)"
            detail.update(extracted=canonical, parsed=True, pred_m=pred_m)
            detail["abs_rel"] = abs(pred_m - gt_m) / gt_m if gt_m > 0 else None
            ok = NUMERIC_LOWER * gt_m <= pred_m <= NUMERIC_UPPER * gt_m
            detail["score"] = 1.0 if ok else 0.0
            return detail

        if kind == KIND_DIRECTION:
            gt_hour = parse_clock_hour(gt)
            if gt_hour is None:
                detail["gt_parse_error"] = True
                return detail
            detail["gt_hour"] = gt_hour
            pred_hour = parse_clock_hour(text)
            if pred_hour is None:
                return detail
            err = clock_diff_hours(pred_hour, gt_hour)
            detail.update(
                extracted=f"{pred_hour} o'clock", parsed=True, pred_hour=pred_hour,
                clock_err_hours=err, clock_err_deg=err * 30,
            )
            detail["score"] = 1.0 if err <= CLOCK_TOLERANCE_HOURS else 0.0
            return detail

        if kind == KIND_PREDICATE:
            gt_pol = parse_gt_polarity(gt)
            if gt_pol is None:
                detail["gt_parse_error"] = True
                return detail
            detail["gt_polarity"] = "yes" if gt_pol else "no"
            pred_pol = parse_prediction_polarity(text)
            if pred_pol is None:
                return detail
            detail.update(extracted="yes" if pred_pol else "no", parsed=True)
            detail["score"] = 1.0 if pred_pol == gt_pol else 0.0
            return detail

        if kind == KIND_CHOICE:
            gt_idx = parse_gt_region_index(gt)
            if gt_idx is None:
                detail["gt_parse_error"] = True
                return detail
            detail["gt_region"] = gt_idx
            pred_idx = parse_region_index(text)
            if pred_idx is None:
                return detail
            detail.update(extracted=f"Region [{pred_idx}]", parsed=True, pred_region=pred_idx)
            detail["score"] = 1.0 if pred_idx == gt_idx else 0.0
            return detail

        raise ValueError(f"unknown answer kind {kind!r}")

    def evaluate_single(self, sample: BaseBenchmarkSample, prediction: str) -> Optional[float]:
        """1.0 / 0.0 under the deterministic proxy; unparsable predictions score 0.0."""
        return float(self.score_detail(sample, prediction)["score"])  # type: ignore[arg-type]

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        per_category: Dict[str, Dict[str, Any]] = {}
        per_bucket: Dict[str, Dict[str, int]] = {}
        detailed: List[Dict[str, Any]] = []
        qual_correct = qual_total = quan_correct = quan_total = 0
        unparsed = 0

        for sample in self.data:
            pred_raw = get_prediction(predictions, sample.sample_id)
            d = self.score_detail(sample, pred_raw)  # type: ignore[arg-type]
            score = float(d["score"])
            ok = score >= 1.0
            if not d["parsed"]:
                unparsed += 1

            cat = per_category.setdefault(
                sample.category,
                {"type": sample.qa_type, "kind": sample.expected_kind, "bucket": sample.bucket,
                 "correct": 0, "total": 0, "parsed": 0, "abs_rel": [], "clock_err_hours": []},
            )
            cat["total"] += 1
            cat["correct"] += int(ok)
            cat["parsed"] += int(bool(d["parsed"]))
            if d.get("abs_rel") is not None:
                cat["abs_rel"].append(d["abs_rel"])
            if d.get("clock_err_hours") is not None:
                cat["clock_err_hours"].append(d["clock_err_hours"])

            if sample.qa_type == "qualitative":
                qual_total += 1
                qual_correct += int(ok)
                b = per_bucket.setdefault(sample.bucket or "other", {"correct": 0, "total": 0})
                b["total"] += 1
                b["correct"] += int(ok)
            else:
                quan_total += 1
                quan_correct += int(ok)

            row = {
                "sample_id": sample.sample_id,
                "qa_type": sample.qa_type,
                "question_type": sample.question_type,
                "category": sample.category,
                "bucket": sample.bucket,
                "expected_kind": sample.expected_kind,
                "question": sample.text_q,
                "ground_truth": sample.answer,
                "prediction": pred_raw,
                "extracted": d["extracted"],
                "parsed": bool(d["parsed"]),
                "score": score,
                "correct": ok,
            }
            for key in ("gt_m", "pred_m", "abs_rel", "gt_hour", "pred_hour",
                        "clock_err_hours", "clock_err_deg", "gt_polarity",
                        "gt_region", "pred_region", "gt_parse_error"):
                if key in d:
                    row[key] = d[key]
            detailed.append(row)

        # per-category table (official quantitative reporting + fine qualitative categories)
        per_category_out: Dict[str, Dict[str, Any]] = {}
        for cat_name in sorted(per_category):
            c = per_category[cat_name]
            entry: Dict[str, Any] = {
                "type": c["type"], "kind": c["kind"], "bucket": c["bucket"],
                "correct": c["correct"], "total": c["total"],
                "accuracy": c["correct"] / max(c["total"], 1),
                "parsed": c["parsed"], "parse_failures": c["total"] - c["parsed"],
            }
            if c["kind"] == KIND_NUMERIC:
                entry["mean_abs_rel"] = mean_or_zero(c["abs_rel"]) if c["abs_rel"] else None
            if c["kind"] == KIND_DIRECTION:
                entry["mean_clock_error_deg"] = (
                    mean_or_zero(c["clock_err_hours"]) * 30 if c["clock_err_hours"] else None
                )
            per_category_out[cat_name] = entry

        qualitative_buckets = {
            name: {**per_bucket[name], "accuracy": per_bucket[name]["correct"] / max(per_bucket[name]["total"], 1)}
            for name in (BUCKET_ORDER + sorted(set(per_bucket) - set(BUCKET_ORDER)))
            if name in per_bucket
        }
        quantitative_categories = {
            name: per_category_out[name] for name in QUANTITATIVE_CATEGORIES if name in per_category_out
        }

        total = len(self.data)
        correct = qual_correct + quan_correct
        results: Dict[str, Any] = {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": correct / max(total, 1),
            "scoring": "deterministic_proxy (non-official; the official metric uses a GPT-4-turbo judge)",
            "overall_aggregation": (
                "overall_accuracy = correct/total over all loaded items (harness-level number). "
                "The paper reports qualitative (n=657) and quantitative (n=749) separately and "
                "defines no single overall score."
            ),
            "qualitative": {
                "correct": qual_correct,
                "total": qual_total,
                "accuracy": qual_correct / max(qual_total, 1) if qual_total else None,
                "per_bucket": qualitative_buckets,
            },
            "quantitative": {
                "correct": quan_correct,
                "total": quan_total,
                "accuracy": quan_correct / max(quan_total, 1) if quan_total else None,
                "per_category": quantitative_categories,
            },
            "per_category": per_category_out,
            "unparsed_predictions": unparsed,
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), detailed)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'=' * 72}")
        print("SpatialRGPT-Bench Results  [deterministic proxy scorer, NOT the official GPT-4 judge]")
        print(f"{'=' * 72}")
        print(f"Total: {results['total_samples']}  Correct: {results['correct_samples']}  "
              f"Micro accuracy: {results['overall_accuracy'] * 100:.2f}%  "
              f"Unparsed: {results.get('unparsed_predictions', 0)}")
        qual = results.get("qualitative", {})
        if qual.get("total"):
            print(f"\nQualitative  {qual['accuracy'] * 100:6.2f}%  ({qual['correct']}/{qual['total']})")
            for name, s in qual.get("per_bucket", {}).items():
                print(f"  {name:<28s} {s['accuracy'] * 100:6.2f}%  ({s['correct']}/{s['total']})")
        quan = results.get("quantitative", {})
        if quan.get("total"):
            print(f"\nQuantitative {quan['accuracy'] * 100:6.2f}%  ({quan['correct']}/{quan['total']})")
            for name, s in quan.get("per_category", {}).items():
                extra = ""
                if s.get("mean_abs_rel") is not None:
                    extra = f"  abs_rel={s['mean_abs_rel']:.3f}"
                if s.get("mean_clock_error_deg") is not None:
                    extra = f"  err={s['mean_clock_error_deg']:.1f} deg"
                print(f"  {name:<28s} {s['accuracy'] * 100:6.2f}%  ({s['correct']}/{s['total']}){extra}")
        print(f"{'=' * 72}\n")
