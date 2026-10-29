"""CountQA data loader (Google / Waymo, 2025; open-ended counting in the wild).

CountQA ("CountQA: How Well Do MLLMs Count in the Wild?", Tamarapalli et al.,
arXiv 2508.06585) is an open-ended object-counting benchmark: 1,528 QA pairs
over 1,001 author-captured real-world photos chosen for high object density,
clutter and occlusion, with in-situ (not post-hoc) ground-truth counts.

Data structure (HF repo Jayant-Sravan/CountQA, CC-BY-4.0; single ``test`` split):
    data/CountQA/data/test-000NN-of-00013.parquet   13 shards x 77 rows = 1,001 rows (one per image)
    data/CountQA/images/<global_row_idx>.<jpg|png>  raw embedded bytes, dumped once on first load

Parquet columns:
    image        struct {bytes, path}  ``path`` is always "image.jpeg" but 16 payloads are PNG
                                        (13 of them RGBA): the extension is taken from the BYTES
    questions    list[str]             1..28 questions per image (836 images have exactly one)
    answers      list[str]             aligned with ``questions``; canonical non-negative integers
    objects      list[str]             PER-IMAGE list of countable object nouns -- NOT aligned
                                        with ``questions`` (some entries carry a leading space)
    categories   list[str]             PER-IMAGE scene categories, 1-4 per image -- NOT aligned
                                        with ``questions``; 10 distinct labels (CATEGORIES)
    is_focused   bool                  clear subject (True) vs visually busy / cluttered (False)
    full_config  str                   JSON {QA, objects, categories, focused}; a redundant copy

Items: one per (row, question index): shards in sorted filename order, rows in
parquet order, questions in list order (deterministic across machines).
    sample_id      "<global_row_idx>_<q_idx>", global_row_idx = 0..1000 across shards
    question       questions[q] verbatim (no reformatting)
    answer         str(int(answers[q]))
    question_type  categories[0] -- the image's PRIMARY (first-listed) scene category --
                   or "uncategorised" when the list is empty
    images         [absolute path of the dumped original image, unmodified]
    metadata       categories (all), objects (whitespace-stripped), is_focused, full_config (parsed)

Metric: exact integer match (EM; the official primary). Secondary: RA@5% and
RA@10% (|pred - gt| <= 0.05 gt / 0.10 gt; gt = 0 requires an exact 0 -- note
that for gt <= 19 RA@5% coincides with EM and for gt <= 9 so does RA@10%),
MAE over parsed predictions. Tables: per is_focused (focused / cluttered), per
GT-count bin (0, 1-5, 6-10, 11-20, 21-50, 51+), per primary category (a
partition of the items) and per category multi-label (an item is counted
under EVERY category of its image, so totals exceed the item count).

Parser (identical for the raw agent text and the ``\\boxed{}``-stripped CoT
text): the LAST ``\\boxed{...}`` content first (LaTeX ``\\text{}`` wrappers,
spacing macros, braces and markdown emphasis removed), then the integer chain
exact integer -> the number right after the LAST explicit answer marker
("Answer: 7 (3 left, 4 right)" -> 7) -> single integer -> LAST integer in the
text, then number words (zero .. nine hundred ninety-nine). Integer-valued
decimals ("7.0") count as integers, genuine decimals ("7.5") do not; ordinals
("2nd shelf") and percentages ("50%") are never counts; "1,200" is 1200.
Unparsable -> "" scored 0.0 and kept in the denominator
(``unparsed_predictions``); never a default number.

Reference: https://arxiv.org/abs/2508.06585
           https://huggingface.co/datasets/Jayant-Sravan/CountQA
"""

import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.scoring import (
    get_prediction,
    write_json,
    write_results_summary,
)

PARQUET_SUBDIR = "data"
IMAGE_SUBDIR = "images"
_PARQUET_NAME_RE = re.compile(r"^test-\d{5}-of-\d{5}\.parquet$")
_IMAGE_FILE_RE = re.compile(r"^(\d+)\.([A-Za-z0-9]+)$")

UNCATEGORISED = "uncategorised"
# The 10 scene categories of the release (alphabetical; used for table ordering).
CATEGORIES = [
    "Clothing & Wearables",
    "Consumables & Groceries",
    "Electronics & Components",
    "Furniture & Home Decor",
    "Kitchen & Dining",
    "Office & Media",
    "Others (Abstract & Unclassified)",
    "Outdoor, Vehicles & Structural",
    "Storage & Organization",
    "Tools & Hardware",
]
COUNT_BINS: List[Tuple[str, int, int]] = [
    ("0", 0, 0),
    ("1-5", 1, 5),
    ("6-10", 6, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51+", 51, 10**9),
]
FOCUS_LABELS = {True: "focused", False: "cluttered"}
FOCUS_ORDER = ["focused", "cluttered"]
RA_TOLERANCES = {"ra5": 0.05, "ra10": 0.10}

_FORMAT_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "BMP": "bmp", "TIFF": "tiff"}


# ── integer parsing (module-level so tests can import it) ────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_BOXED_OPEN_RE = re.compile(r"\\boxed\s*\{")
_LATEX_WRAP_RE = re.compile(
    r"\\(?:text|textbf|textrm|textit|textnormal|mathrm|mathbf|mathit|mbox|"
    r"operatorname|boldsymbol)\s*\{([^{}]*)\}"
)
_LATEX_SPACE_RE = re.compile(r"\\[,;:!> ]|\\(?:quad|qquad|thinspace|enspace|left|right)\b|~")
_MARKDOWN_RE = re.compile(r"[\*`]+|(?<![A-Za-z0-9])_+|_+(?![A-Za-z0-9])")
_EXACT_INT_RE = re.compile(r"[-+]?\d+")
# A number token: optional sign, digits, optional fraction; not glued to a
# preceding digit/dot (so "7.5" is one token, not "7" and "5", and the "-" in
# "6-7" is a separator, not a sign). Ordinals ("the 2nd shelf") and
# percentages ("50%") are not counts and never match.
_NUM_TOKEN_RE = re.compile(r"(?<![\d.])([-+]?)(\d+)(?:\.(\d+))?(?!\d|\.\d|\s*%|(?:st|nd|rd|th)\b)")
# "1,200" -> "1200": a comma between a digit and exactly three digits.
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
# An explicit answer marker ("Answer: 7", "The final answer is 7", "Total: 12",
# "Final count: 7", "Count = 7"). The number right after the LAST marker beats
# stray numbers in the surrounding explanation ("Answer: 7 (3 left, 4 right)").
_ANSWER_MARKER_RE = re.compile(
    r"(?:\b(?:final\s+)?answer\b|\bfinal\s+count\b|\btotal(?:\s+count)?\b|\bcount(?=\s*[:=]))"
    r"[\"']?(?:\s+(?:is|would\s+be|should\s+be))?"
    r"\s*[:=\-–—]?\s*"
    r"(?:(?:approximately|approx\.?|about|roughly|around|likely|probably|exactly|nearly|~|≈|\$)\s*)*",
    re.IGNORECASE,
)

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
_TENS = "|".join(_TENS_WORDS)
_UNITS = "|".join(_UNIT_WORDS)
# zero .. nine hundred ninety-nine:
#   "<ones|a> hundred [and] [<tens>[-<ones>] | <unit>]" | "<tens>[-<ones>]" | "<unit>"
_NUMWORD_RE = re.compile(
    rf"\b(?:(?P<h>{_ONES}|a)\s+hundred(?:(?:\s+and)?[-\s]+(?:(?P<ht>{_TENS})(?:[-\s]+(?P<ho>{_ONES}))?|(?P<hu>{_UNITS})))?"
    rf"|(?P<t>{_TENS})(?:[-\s]+(?P<o>{_ONES}))?"
    rf"|(?P<u>{_UNITS}))\b",
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
    t = t.replace("\\%", "%")  # escaped percent sign: keep "7\\%" recognisable as a percentage
    t = re.sub(r"[{}]", " ", t)
    return t.strip()


def strip_think_blocks(text: str) -> str:
    """Drop ``<think>...</think>`` reasoning blocks when an answer remains outside them."""
    if not text:
        return ""
    t = str(text)
    stripped = _THINK_RE.sub("", t).strip()
    return stripped if stripped else t.strip()


def answer_candidates(text: str) -> List[str]:
    """Texts to parse, most authoritative first.

    [0] the LAST ``\\boxed{}`` content (if any), then the whole text with the
    boxes unwrapped -- both LaTeX-cleaned and with markdown emphasis removed.
    """
    if not text:
        return []
    text = strip_think_blocks(text)
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


def _numword_value(m: "re.Match[str]") -> int:
    g = m.groupdict()
    if g["h"]:
        value = 100 if g["h"].lower() == "a" else 100 * _UNIT_WORDS[g["h"].lower()]
        if g["ht"]:
            value += _TENS_WORDS[g["ht"].lower()] + (_UNIT_WORDS[g["ho"].lower()] if g["ho"] else 0)
        elif g["hu"]:
            value += _UNIT_WORDS[g["hu"].lower()]
        return value
    if g["t"]:
        return _TENS_WORDS[g["t"].lower()] + (_UNIT_WORDS[g["o"].lower()] if g["o"] else 0)
    return _UNIT_WORDS[g["u"].lower()]


def find_number_words(text: str) -> List[int]:
    """All spelled-out numbers (zero .. nine hundred ninety-nine) in ``text`` in order."""
    return [_numword_value(m) for m in _NUMWORD_RE.finditer(text or "")]


def marker_answer(text: str) -> Optional[int]:
    """The number (digits or words) right after the LAST explicit answer marker, else None.

    "Answer: 7 (3 on the left, 4 on the right)" -> 7; "Answer: seven. I saw 3 rows." -> 7;
    "To answer the question, I count 6 then 7." -> None (no number follows the marker);
    a genuine decimal after the marker ("Answer: 7.5") -> None.
    """
    for m in reversed(list(_ANSWER_MARKER_RE.finditer(text or ""))):
        n = _NUM_TOKEN_RE.match(text, m.end())
        if n:
            sign, whole, frac = n.groups()
            return None if (frac and frac.strip("0")) else int(sign + whole)
        w = _NUMWORD_RE.match(text, m.end())
        if w:
            return _numword_value(w)
    return None


def _parse_one(text: str) -> Optional[int]:
    """Integer chain on one candidate: exact int -> int after an answer marker -> single int -> last int -> number word."""
    t = _THOUSANDS_RE.sub("", text).strip().strip("\"'").strip()
    if not t:
        return None
    if _EXACT_INT_RE.fullmatch(t):
        return int(t)
    marked = marker_answer(t)
    if marked is not None:
        return marked
    ints = find_integers(t)
    if ints:
        return ints[-1]  # one integer -> that one; several -> the last one
    words = find_number_words(t)
    if words:
        return words[-1]
    return None


def extract_integer(text: Any) -> Optional[int]:
    """Integer answer in a model output, or None if nothing parses (never a default)."""
    if text is None:
        return None
    for cand in answer_candidates(str(text)):
        value = _parse_one(cand)
        if value is not None:
            return value
    return None


def count_bin(value: int) -> str:
    """Name of the GT-count bin containing ``value``."""
    for name, lo, hi in COUNT_BINS:
        if lo <= value <= hi:
            return name
    return "other"  # negative values only; never produced by the dataset


# ── sample ──────────────────────────────────────────────────────────────────


@dataclass
class CountQASample(BaseBenchmarkSample):
    """One (image, question) item. No ``choices``: the question is open-ended."""

    row_idx: int = 0                     # global row index across the sorted shards (== image id)
    q_idx: int = 0                       # index of the question within its image
    num_questions: int = 0               # questions asked about this image
    primary_category: str = ""           # categories[0] == question_type
    categories: List[str] = field(default_factory=list)   # all scene categories of the image
    objects: List[str] = field(default_factory=list)      # per-image countable nouns (stripped)
    is_focused: bool = False
    focus: str = ""                      # "focused" | "cluttered"
    count_bin: str = ""                  # GT-count bin name (COUNT_BINS)
    image_format: str = ""               # "jpg" | "png" (sniffed from the bytes)
    full_config: Optional[Dict[str, Any]] = None          # parsed full_config JSON (shared per image)


# ── benchmark ───────────────────────────────────────────────────────────────


class CountQABench(BaseBenchmark):
    """CountQA loader and exact-match counting scorer."""

    data_specific_prompt = "Give the final answer as a single integer."

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._image_dir = os.path.join(os.path.abspath(data_path), IMAGE_SUBDIR)
        self.num_rows = 0                      # images (rows) seen across all shards
        self.shard_files: List[str] = []       # parquet shard paths in load order
        self.rows_per_shard: List[int] = []
        super().__init__(data_path, question_type)

    # ── loading ─────────────────────────────────────────────────────────────

    def _wants(self, sample: CountQASample) -> bool:
        """question_type filter: any category of the image, focused/cluttered, or a count bin."""
        if not self.question_type_filter:
            return True
        wanted = {str(q).strip().lower() for q in self.question_type_filter}
        keys = {c.lower() for c in sample.categories}
        keys.update({sample.primary_category.lower(), sample.focus.lower(), sample.count_bin.lower()})
        return bool(keys & wanted)

    def _existing_images(self) -> Dict[int, str]:
        """{row_idx: absolute path} of non-empty dumped images (any extension)."""
        found: Dict[int, str] = {}
        try:
            names = os.listdir(self._image_dir)
        except OSError:
            return found
        for name in sorted(names):  # deterministic if several files exist for one row
            m = _IMAGE_FILE_RE.match(name)
            if not m:
                continue  # *.tmp files of concurrent loaders, hidden files, ...
            path = os.path.join(self._image_dir, name)
            try:
                if os.path.getsize(path) > 0:
                    found[int(m.group(1))] = path
            except OSError:
                continue
        return found

    @staticmethod
    def _image_ext(payload: bytes) -> str:
        """File extension matching the ACTUAL encoding of ``payload`` (the parquet path hint lies)."""
        if payload[:3] == b"\xff\xd8\xff":
            return "jpg"
        if payload[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as im:
            fmt = (im.format or "").upper()
        return _FORMAT_EXT.get(fmt, fmt.lower() or "bin")

    def _write_image(self, row_idx: int, payload: bytes) -> str:
        """Atomically persist the ORIGINAL bytes as <row_idx>.<ext> (tmp + rename, pid-unique)."""
        ext = self._image_ext(payload)
        final = os.path.join(self._image_dir, f"{row_idx}.{ext}")
        tmp = os.path.join(self._image_dir, f"{row_idx}.{os.getpid()}.tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(payload)
            os.replace(tmp, final)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return final

    def _dump_missing_images(self, pf, first_row_idx: int, existing: Dict[int, str]) -> int:
        """Persist embedded images of one shard that are not on disk yet. Returns #written.

        Only the heavy ``image`` column of row groups containing a missing image
        is read, so repeat loads never touch the image bytes.
        """
        n_rows = pf.metadata.num_rows
        missing = [i for i in range(first_row_idx, first_row_idx + n_rows) if i not in existing]
        if not missing:
            return 0
        os.makedirs(self._image_dir, exist_ok=True)
        missing_set = set(missing)
        written = 0
        offset = first_row_idx
        for rg in range(pf.num_row_groups):
            rg_rows = pf.metadata.row_group(rg).num_rows
            idxs = list(range(offset, offset + rg_rows))
            offset += rg_rows
            if not any(i in missing_set for i in idxs):
                continue
            images = pf.read_row_group(rg, columns=["image"]).column("image").to_pylist()
            for row_idx, img in zip(idxs, images):
                if row_idx not in missing_set or img is None:
                    continue
                payload = img.get("bytes") if isinstance(img, dict) else img
                if not payload:
                    raise ValueError(f"CountQA row {row_idx}: embedded image payload is empty")
                existing[row_idx] = self._write_image(row_idx, bytes(payload))
                written += 1
        return written

    @staticmethod
    def _canonical_answer(raw: Any, where: str) -> int:
        text = str(raw).strip() if raw is not None else ""
        if not re.fullmatch(r"\d+", text):
            raise ValueError(f"CountQA {where}: answer {raw!r} is not a non-negative integer")
        return int(text)

    def read_data(self) -> None:
        import pyarrow.parquet as pq

        self.data_path = os.path.abspath(self.data_path)
        self._image_dir = os.path.join(self.data_path, IMAGE_SUBDIR)
        parquet_dir = os.path.join(self.data_path, PARQUET_SUBDIR)
        if not os.path.isdir(parquet_dir):
            raise FileNotFoundError(
                f"CountQA parquet dir not found: {parquet_dir}. "
                "Download Jayant-Sravan/CountQA (data/test-*.parquet) into data/CountQA/."
            )
        shard_names = sorted(n for n in os.listdir(parquet_dir) if _PARQUET_NAME_RE.match(n))
        if not shard_names:
            raise FileNotFoundError(f"No CountQA shards (test-*.parquet) found under {parquet_dir}")
        self.shard_files = [os.path.join(parquet_dir, n) for n in shard_names]
        self.rows_per_shard = []

        existing = self._existing_images()
        n_written = 0
        row_idx = 0
        for path in self.shard_files:
            pf = pq.ParquetFile(path)
            meta_cols = [c for c in pf.schema_arrow.names if c != "image"]
            rows = pf.read(columns=meta_cols).to_pylist()
            self.rows_per_shard.append(len(rows))

            n_written += self._dump_missing_images(pf, row_idx, existing)

            for row in rows:
                where = f"row {row_idx} ({os.path.basename(path)})"
                questions = [str(q) for q in (row.get("questions") or [])]
                answers = list(row.get("answers") or [])
                if len(questions) != len(answers):
                    raise ValueError(
                        f"CountQA {where}: {len(questions)} questions but {len(answers)} answers"
                    )
                categories = [str(c).strip() for c in (row.get("categories") or []) if str(c).strip()]
                objects = [str(o).strip() for o in (row.get("objects") or []) if str(o).strip()]
                is_focused = bool(row.get("is_focused"))
                full_config: Optional[Dict[str, Any]] = None
                raw_cfg = row.get("full_config")
                if isinstance(raw_cfg, str) and raw_cfg.strip():
                    try:
                        parsed = json.loads(raw_cfg)
                        full_config = parsed if isinstance(parsed, dict) else {"value": parsed}
                    except json.JSONDecodeError:
                        full_config = {"raw": raw_cfg}
                image_path = existing.get(row_idx)
                if image_path is None:
                    raise FileNotFoundError(f"CountQA {where}: image was not dumped to {self._image_dir}")
                primary = categories[0] if categories else UNCATEGORISED

                for q_idx, question in enumerate(questions):
                    gt = self._canonical_answer(answers[q_idx], f"{where} q{q_idx}")
                    sample = CountQASample(
                        sample_id=f"{row_idx}_{q_idx}",
                        question=question,
                        question_type=primary,
                        images=[image_path],
                        answer=str(gt),
                        row_idx=row_idx,
                        q_idx=q_idx,
                        num_questions=len(questions),
                        primary_category=primary,
                        categories=categories,
                        objects=objects,
                        is_focused=is_focused,
                        focus=FOCUS_LABELS[is_focused],
                        count_bin=count_bin(gt),
                        image_format=os.path.splitext(image_path)[1].lstrip("."),
                        full_config=full_config,
                    )
                    if self._wants(sample):
                        self.data.append(sample)
                row_idx += 1
        self.num_rows = row_idx

        if n_written:
            print(f"[CountQABench] Extracted {n_written} embedded images to {self._image_dir}")
        print(
            f"[CountQABench] Loaded {len(self.data)} samples from {self.num_rows} images "
            f"({len(self.shard_files)} shards)"
        )

    # ── answer extraction ───────────────────────────────────────────────────

    def extract_answer(self, prediction: str) -> str:
        """Canonical integer string ("7") or "" if unparsable (never a default value)."""
        value = extract_integer(prediction)
        return "" if value is None else str(value)

    # ── scoring ─────────────────────────────────────────────────────────────

    @staticmethod
    def _score(gt: int, pred: Optional[int]) -> Dict[str, Any]:
        """exact / RA@5% / RA@10% / |pred-gt| for one item; unparsed -> all misses, error None."""
        if pred is None:
            return {"exact": False, "ra5": False, "ra10": False, "abs_err": None}
        err = abs(pred - gt)
        exact = err == 0
        out: Dict[str, Any] = {"exact": exact, "abs_err": err}
        for key, tol in RA_TOLERANCES.items():
            out[key] = exact if gt == 0 else err <= tol * gt
        return out

    def evaluate_single(self, sample: BaseBenchmarkSample, prediction: str) -> Optional[float]:
        """1.0 on exact integer match, else 0.0 (unparsable -> 0.0)."""
        return 1.0 if self._score(int(sample.answer), extract_integer(prediction))["exact"] else 0.0

    @staticmethod
    def _new_stats() -> Dict[str, Any]:
        return {"correct": 0, "total": 0, "parsed": 0, "ra5_ok": 0, "ra10_ok": 0, "abs_err_sum": 0}

    @staticmethod
    def _bump(stats: Dict[str, Any], d: Dict[str, Any]) -> None:
        stats["total"] += 1
        stats["correct"] += int(d["exact"])
        stats["ra5_ok"] += int(d["ra5"])
        stats["ra10_ok"] += int(d["ra10"])
        if d["abs_err"] is not None:
            stats["parsed"] += 1
            stats["abs_err_sum"] += d["abs_err"]

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
            "ra5": stats["ra5_ok"] / max(total, 1),
            "ra10": stats["ra10_ok"] / max(total, 1),
        }

    def _table(self, stats: Dict[str, Dict[str, Any]], order: List[str]) -> Dict[str, Any]:
        keys = [k for k in order if k in stats] + sorted(set(stats) - set(order))
        return {k: self._finalize(stats[k]) for k in keys}

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        overall = self._new_stats()
        per_focus: Dict[str, Dict[str, Any]] = {}
        per_bin: Dict[str, Dict[str, Any]] = {}
        per_category: Dict[str, Dict[str, Any]] = {}
        per_category_ml: Dict[str, Dict[str, Any]] = {}
        detailed: List[Dict[str, Any]] = []
        unparsed = 0
        missing = 0
        images: set = set()

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid, default=None)
            present = pred_raw is not None
            if not present:
                missing += 1
                pred_raw = ""
            gt = int(sample.answer)
            pred = extract_integer(pred_raw)
            d = self._score(gt, pred)
            if pred is None:
                unparsed += 1
            images.add(sample.row_idx)

            self._bump(overall, d)
            self._bump(per_focus.setdefault(sample.focus, self._new_stats()), d)
            self._bump(per_bin.setdefault(sample.count_bin, self._new_stats()), d)
            self._bump(per_category.setdefault(sample.primary_category, self._new_stats()), d)
            for cat in (sample.categories or [UNCATEGORISED]):
                self._bump(per_category_ml.setdefault(cat, self._new_stats()), d)

            detailed.append({
                "sample_id": sid,
                "row_idx": sample.row_idx,
                "q_idx": sample.q_idx,
                "question_type": sample.question_type,
                "categories": list(sample.categories),
                "is_focused": sample.is_focused,
                "focus": sample.focus,
                "count_bin": sample.count_bin,
                "question": sample.question,
                "ground_truth": sample.answer,
                "prediction": pred_raw,
                "extracted": "" if pred is None else str(pred),
                "parsed": pred is not None,
                "correct": bool(d["exact"]),
                "abs_error": d["abs_err"],
                "within_5pct": bool(d["ra5"]),
                "within_10pct": bool(d["ra10"]),
            })

        overall_out = self._finalize(overall)
        results: Dict[str, Any] = {
            "total_samples": overall_out["total"],
            "correct_samples": overall_out["correct"],
            "overall_accuracy": overall_out["accuracy"],
            "overall_aggregation": (
                "overall_accuracy = exact-match micro accuracy over all loaded (image, question) items; "
                "unparsable/missing predictions score 0 and stay in the denominator. Items are QA pairs, "
                "so images with several questions weigh more (n_images reports the distinct images)."
            ),
            "scoring": (
                "exact integer match (official primary); RA@5%/RA@10% = |pred-gt| <= 0.05/0.10 * gt "
                "(gt = 0 requires exact); MAE over parsed predictions only; parser = last \\boxed{} -> "
                "exact int -> int after the last explicit answer marker -> single int -> last int -> number words"
            ),
            "n_images": len(images),
            "overall_ra5": overall_out["ra5"],
            "overall_ra10": overall_out["ra10"],
            "overall_mae": overall_out["mae"],
            "parsed_predictions": overall_out["parsed"],
            "unparsed_predictions": unparsed,
            "missing_predictions": missing,
            "per_focus": self._table(per_focus, FOCUS_ORDER),
            "per_count_bin": self._table(per_bin, [b[0] for b in COUNT_BINS]),
            "per_category": self._table(per_category, CATEGORIES),
            "per_category_multilabel": self._table(per_category_ml, CATEGORIES),
            "per_category_note": (
                "per_category uses the image's primary (first-listed) category and partitions the items; "
                "per_category_multilabel counts an item under every category of its image (totals exceed "
                "total_samples)"
            ),
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
        print("CountQA Results")
        print(f"{'=' * 78}")
        print(f"Total samples: {results['total_samples']}   Images: {results.get('n_images', '?')}   "
              f"Correct: {results['correct_samples']}   Unparsed: {results.get('unparsed_predictions', 0)}   "
              f"Missing: {results.get('missing_predictions', 0)}")
        print(f"Exact match: {_pct(results['overall_accuracy'])}   RA@5%: {_pct(results.get('overall_ra5'))}   "
              f"RA@10%: {_pct(results.get('overall_ra10'))}   MAE: {_mae(results.get('overall_mae'))}")
        print()
        for title, key in (
            ("Focus", "per_focus"),
            ("GT count bin", "per_count_bin"),
            ("Primary category", "per_category"),
            ("Category (multi-label)", "per_category_multilabel"),
        ):
            table = results.get(key, {})
            if not table:
                continue
            print(f"{title:<34s} {'EM':>8s} {'Correct':>8s} {'Total':>6s} {'MAE':>6s} {'RA@5%':>8s} {'RA@10%':>8s}")
            print(f"{'-' * 34} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 8}")
            for name, info in table.items():
                print(
                    f"{name:<34s} {info['accuracy'] * 100:>7.2f}% {info['correct']:>8d} {info['total']:>6d} "
                    f"{_mae(info.get('mae')):>6s} {_pct(info.get('ra5')):>8s} {_pct(info.get('ra10')):>8s}"
                )
            print()
        print(f"{'=' * 78}\n")
