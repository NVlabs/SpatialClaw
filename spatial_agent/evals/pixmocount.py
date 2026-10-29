"""PixMo-Count data loader (Ai2 Molmo, CVPR 2025).

PixMo-Count (Deitke et al., "Molmo and PixMo: Open Weights and Open Data for
State-of-the-Art Vision-Language Models", arXiv 2409.17146) is a counting
benchmark of web (Flickr) images, each paired with an object label and a
human-verified count in 2..10. The official annotations (HF
allenai/pixmo-count, ODC-BY; columns image_url, image_sha256, count, points,
label; val 540 / test 540 / train 36,916) ship Flickr URLs only. The images
come from the lmms-eval re-host (HF kcz358/pixmo-count; columns id, image,
question, answer; test 534 / validation 535 -- 11 Flickr images were lost).

Data structure (data dir data/PixMo-Count):
    data/PixMo-Count/data/{test,validation}-00000-of-00001.parquet    re-host, images embedded
    data/PixMo-Count/official/data/{test,validation,train}-*.parquet  official annotations
    data/PixMo-Count/images/<split>_<id>.<jpg|png>                    dumped once on first load
                                                                      (original bytes, no re-encoding)

Re-host columns used (verified on all 1,069 rows, 2026-10-07):
    id        str     sha256 hex == official ``image_sha256`` (join key; 1,069/1,069 match,
                      count and label agree on every row; unique across both splits)
    image     struct  {bytes, path=None}; JPEG (1,057) or PNG (12); the bytes do NOT hash to
                      ``id`` (the re-host re-downloaded/re-encoded the pictures)
    question  str     "How many <label> are in this image?\\n<image>"      (exact template)
    answer    str     "There are <N> <label> in this image."              (exact template)

Items = every row of the re-host ``test`` split, then ``validation``, in parquet
row order (shards sorted lexicographically) -> 1,069 samples.
    sample_id      "<split>:<id>"
    question       re-host question with the lmms-eval "<image>" placeholder removed,
                   i.e. exactly the official lmms-eval pixmo_count ``doc_to_text``
                   (``doc["question"].replace("<image>", "").strip()``)
    answer         the integer of the answer sentence as a canonical string ("5")
    question_type  "count_<label>" with the official label ("count_people", "count_wild birds")
The image is passed unmodified at full resolution (median long side 2,362 px,
max 11,648 x 8,736; two images exceed PIL's default MAX_IMAGE_PIXELS -> warning only).

Metric: exact integer match (Molmo Table 1; lmms-eval pixmo_count ``acc``).
lmms-eval appends "\\nAnswer with a single number." and takes the FIRST integer
(regex ``\\d+``) of the output. We use the shared integer parser for
both arms (raw agent text and \\boxed{}-stripped CoT text alike): <think> blocks
and GLM box tokens dropped, the LAST ``\\boxed{...}`` content first (LaTeX
``\\text{}`` wrappers / spacing / markdown removed), then exact integer ->
the integer right after the LAST explicit answer marker ("Answer: 7 (not 6)"
-> 7) -> single integer -> LAST integer ("7.0" counts as 7; "7.5", "8th" and
"100%" are skipped; "1,000" is 1000), then spelled-out numbers (zero ..
ninety-nine, last one). Unparsable -> "" and scored 0 (counted in the
denominator; reported as ``unparsed_predictions``).
Secondary: MAE and mean signed error over parsed predictions, RA@10%
(|pred - gt| <= 0.1 * gt; for counts 2..9 this equals exact match, at gt = 10
it additionally accepts 9 and 11). Tables per split, per gt count (2..10) and
per label; ``official_join`` reports the annotation match rate.

Reference: https://huggingface.co/datasets/allenai/pixmo-count
           https://huggingface.co/datasets/kcz358/pixmo-count
           https://github.com/EvolvingLMMs-Lab/lmms-eval (lmms_eval/tasks/pixmo_count)
"""

import glob
import io
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.scoring import (
    get_prediction,
    mean_or_zero,
    write_json,
    write_results_summary,
)

SPLITS = ["test", "validation"]                       # load order: re-host test, then validation
REHOST_SUBDIR = "data"                                # kcz358/pixmo-count shards (images embedded)
OFFICIAL_SUBDIR = os.path.join("official", "data")    # allenai/pixmo-count annotation shards
IMAGE_DIR = "images"
MIN_COUNT, MAX_COUNT = 2, 10                          # human-verified eval counts (official README)
COUNT_ORDER = [str(c) for c in range(MIN_COUNT, MAX_COUNT + 1)]
RA_TOLERANCE = 0.10                                   # RA@10%
IMAGE_TOKEN = "<image>"                               # lmms-eval placeholder in the re-host question

# Re-host templates (exact on all 1,069 rows, verified 2026-10-07).
_QUESTION_RE = re.compile(r"^How many (?P<label>.+?) are in this image\?\s*(?:<image>\s*)*$", re.DOTALL)
_ANSWER_RE = re.compile(r"^There are (?P<count>\d+) (?P<label>.+?) in this image\.?\s*$", re.DOTALL)


# ── integer parsing (shared parser; module-level so tests can import it) ──

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
# yields 3), while "No.7" / "...7" still parse as 7 like the official regex;
# the "-" in "6-7" is a separator, not a sign. Ordinals ("the 8th") and
# percentages ("100%") are not counts and never match.
_NUM_TOKEN_RE = re.compile(
    r"(?<!\d)(?<!\d\.)([-+]?)(\d+)(?:\.(\d+))?(?!\d|\.\d|\s*%|(?:st|nd|rd|th)\b)"
)
# "1,000" -> "1000": a comma between a digit and exactly three digits.
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
# An explicit answer marker ("Answer: 7", "The final answer is 7", "Total: 7",
# "Count = 7"). The number right after the LAST marker beats stray numbers in
# the surrounding explanation ("Answer: 7 (not 6)" -> 7, not 6).
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
    """Drop <think>...</think> blocks and GLM-style box tokens."""
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


def _numword_value(m: "re.Match[str]") -> int:
    tens, ones, unit = m.groups()
    if tens:
        return _TENS_WORDS[tens.lower()] + (_UNIT_WORDS[ones.lower()] if ones else 0)
    return _UNIT_WORDS[unit.lower()]


def find_number_words(text: str) -> List[int]:
    """All spelled-out numbers (zero .. ninety-nine) in ``text`` in order."""
    return [_numword_value(m) for m in _NUMWORD_RE.finditer(text or "")]


def marker_answer(text: str) -> Optional[int]:
    """The number (digits or words) right after the LAST explicit answer marker, else None.

    "Answer: 7 (not 6)" -> 7; "The final answer is seven. I saw 3 rows." -> 7;
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
    """Chain on one candidate: exact int -> int after an answer marker -> single int -> last int -> number word."""
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


def extract_integer(text: str) -> Optional[int]:
    """Integer answer in a model output, or None if nothing parses (never a default)."""
    if text is None:
        return None
    for cand in answer_candidates(str(text)):
        value = _parse_one(cand)
        if value is not None:
            return value
    return None


# ── image payload helpers ───────────────────────────────────────────────────

_MAGIC = [
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF8", "gif"),
    (b"BM", "bmp"),
]


def sniff_image_ext(data: bytes) -> str:
    """File extension for an encoded image payload, from its magic bytes (PIL fallback)."""
    head = bytes(data[:16])
    for magic, ext in _MAGIC:
        if head.startswith(magic):
            return ext
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        fmt = (im.format or "bin").lower()
    return {"jpeg": "jpg"}.get(fmt, fmt)


# ── sample ──────────────────────────────────────────────────────────────────


@dataclass
class PixMoCountSample(BaseBenchmarkSample):
    """One PixMo-Count item (no ``choices``: the question is open-ended)."""

    split: str = ""                 # test | validation (re-host split)
    image_sha256: str = ""          # re-host id == official image_sha256
    label: str = ""                 # object label, plural noun as in the data ("people", "wild birds")
    label_source: str = ""          # official | rehost_question
    count: int = 0                  # integer ground truth (== int(answer))
    answer_text: str = ""           # raw re-host answer sentence
    raw_question: str = ""          # raw re-host question incl. the "<image>" placeholder
    image_url: str = ""             # official Flickr URL (metadata only)
    row_index: int = 0              # row index within the split
    official_matched: bool = False  # joined to allenai/pixmo-count by sha256


# ── benchmark ───────────────────────────────────────────────────────────────


class PixMoCountBench(BaseBenchmark):
    """PixMo-Count loader and exact-integer-match scorer."""

    data_specific_prompt = "Give the final answer as a single integer."

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        data_path = os.path.abspath(data_path)
        self._image_dir = os.path.join(data_path, IMAGE_DIR)
        self.join_stats: Dict[str, Any] = {}
        self.n_images_written = 0
        super().__init__(data_path, question_type)

    # ── loading ─────────────────────────────────────────────────────────────

    @staticmethod
    def question_text(raw_question: str) -> str:
        """Official lmms-eval pixmo_count doc_to_text: drop the "<image>" placeholder."""
        return str(raw_question).replace(IMAGE_TOKEN, "").strip()

    @staticmethod
    def question_label(raw_question: str) -> str:
        """Object label named by the re-host question template ("" if off-template)."""
        m = _QUESTION_RE.match(str(raw_question).strip())
        return m.group("label").strip() if m else ""

    @staticmethod
    def parse_answer_sentence(answer: str) -> Tuple[Optional[int], str]:
        """(count, label) of a re-host answer sentence; (None, "") if it does not parse."""
        m = _ANSWER_RE.match(str(answer).strip())
        if m:
            return int(m.group("count")), m.group("label").strip()
        ints = find_integers(str(answer))
        return (ints[0] if len(ints) == 1 else None), ""

    def _wants(self, sample: PixMoCountSample) -> bool:
        """question_type filter: "count_<label>", bare label or split, case-insensitively."""
        if not self.question_type_filter:
            return True
        wanted = {str(q).strip().lower() for q in self.question_type_filter}
        keys = {sample.question_type.lower(), sample.label.lower(), sample.split.lower()}
        return bool(keys & wanted)

    @staticmethod
    def _shards(directory: str, split: str) -> List[str]:
        return sorted(glob.glob(os.path.join(directory, f"{split}-*.parquet")))

    def _load_official(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, int]]]:
        """sha256 -> {split, count, label, image_url} and the per-split count histogram.

        Returns ({}, {}) when the official annotation shards are absent.
        """
        import pyarrow.parquet as pq

        table: Dict[str, Dict[str, Any]] = {}
        hist: Dict[str, Dict[str, int]] = {}
        official_dir = os.path.join(self.data_path, OFFICIAL_SUBDIR)
        for split in SPLITS:
            for shard in self._shards(official_dir, split):
                rows = pq.ParquetFile(shard).read(
                    columns=["image_url", "image_sha256", "count", "label"]
                ).to_pylist()
                for r in rows:
                    sha = str(r["image_sha256"])
                    table.setdefault(sha, {
                        "split": split,
                        "count": int(r["count"]),
                        "label": str(r["label"] or "").strip(),
                        "image_url": str(r["image_url"] or ""),
                    })
                    h = hist.setdefault(split, {})
                    key = str(int(r["count"]))
                    h[key] = h.get(key, 0) + 1
        return table, hist

    @staticmethod
    def _image_stem(split: str, rid: str) -> str:
        return f"{split}_{rid}"

    @staticmethod
    def _image_ok(path: str) -> bool:
        """True if the dumped image exists and is not a truncated 0-byte file."""
        try:
            return os.path.getsize(path) > 0
        except OSError:
            return False

    def _existing_images(self) -> Dict[str, str]:
        """stem -> absolute path of every non-empty dumped image (one listdir)."""
        out: Dict[str, str] = {}
        try:
            names = os.listdir(self._image_dir)
        except OSError:
            return out
        for name in names:
            stem, ext = os.path.splitext(name)
            if not ext or ".tmp" in stem:  # half-written temp files: <stem>.<pid>.tmp.<ext>
                continue
            path = os.path.join(self._image_dir, name)
            if self._image_ok(path):
                out[stem] = path
        return out

    def _write_image(self, stem: str, payload: Any) -> str:
        """Atomically persist one embedded image with its original encoding (tmp + rename)."""
        if isinstance(payload, dict):
            payload = payload.get("bytes")
        if payload is None:
            raise ValueError(f"PixMo-Count row {stem}: embedded image has no bytes")
        data = bytes(payload)
        ext = sniff_image_ext(data)
        os.makedirs(self._image_dir, exist_ok=True)
        final = os.path.join(self._image_dir, f"{stem}.{ext}")
        # pid-unique temp name: concurrent first-time loaders (e.g. a job array)
        # must not remove / rename each other's half-written files
        tmp = os.path.join(self._image_dir, f"{stem}.{os.getpid()}.tmp.{ext}")
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, final)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return final

    def _dump_missing_images(self, pf, split: str, existing: Dict[str, str]) -> Dict[str, str]:
        """Persist embedded images that are not on disk yet; returns stem -> path of those written.

        Only the heavy ``image`` column of row groups containing a missing
        image is read, so repeat loads cost about a second.
        """
        written: Dict[str, str] = {}
        for rg in range(pf.num_row_groups):
            ids = [str(i) for i in pf.read_row_group(rg, columns=["id"]).column("id").to_pylist()]
            stems = [self._image_stem(split, rid) for rid in ids]
            if all(s in existing for s in stems):
                continue
            images = pf.read_row_group(rg, columns=["image"]).column("image").to_pylist()
            for stem, img in zip(stems, images):
                if stem in existing:
                    continue
                written[stem] = self._write_image(stem, img)
        return written

    def read_data(self) -> None:
        import pyarrow.parquet as pq

        self.data_path = os.path.abspath(self.data_path)
        self._image_dir = os.path.join(self.data_path, IMAGE_DIR)
        rehost_dir = os.path.join(self.data_path, REHOST_SUBDIR)
        if not os.path.isdir(rehost_dir):
            raise FileNotFoundError(
                f"PixMo-Count re-host dir not found: {rehost_dir}. Download kcz358/pixmo-count "
                f"(data/{{test,validation}}-*.parquet) to {self.data_path} and allenai/pixmo-count "
                f"to {os.path.join(self.data_path, 'official')}."
            )

        official, official_hist = self._load_official()
        stats: Dict[str, Any] = {
            "official_available": bool(official),
            "official_rows": len(official),
            "official_count_hist": official_hist,
            "rehost_rows": 0,
            "matched": 0,
            "unmatched": 0,
            "count_mismatch": 0,
            "label_mismatch": 0,
            "split_mismatch": 0,
            "mismatch_examples": [],
        }

        existing = self._existing_images()
        n_written = 0
        for split in SPLITS:
            shards = self._shards(rehost_dir, split)
            if not shards:
                raise FileNotFoundError(f"PixMo-Count re-host parquet not found: {rehost_dir}/{split}-*.parquet")
            row_index = 0
            for shard in shards:
                pf = pq.ParquetFile(shard)
                meta_cols = [c for c in pf.schema_arrow.names if c != "image"]
                rows = pf.read(columns=meta_cols).to_pylist()

                written = self._dump_missing_images(pf, split, existing)
                existing.update(written)
                n_written += len(written)

                for row in rows:
                    rid = str(row["id"])
                    stem = self._image_stem(split, rid)
                    image_path = existing.get(stem)
                    if image_path is None:
                        raise FileNotFoundError(f"PixMo-Count image missing after dump: {stem}")

                    raw_question = str(row.get("question") or "")
                    answer_text = str(row.get("answer") or "")
                    count, answer_label = self.parse_answer_sentence(answer_text)
                    if count is None:
                        raise ValueError(f"PixMo-Count {split} row {row_index}: cannot parse answer {answer_text!r}")
                    q_label = self.question_label(raw_question) or answer_label

                    off = official.get(rid)
                    stats["rehost_rows"] += 1
                    if off is not None:
                        stats["matched"] += 1
                        label, label_source = off["label"] or q_label, "official"
                        if off["count"] != count:
                            stats["count_mismatch"] += 1
                            stats["mismatch_examples"].append({"id": rid, "field": "count", "rehost": count, "official": off["count"]})
                        if off["label"] and off["label"] != q_label:
                            stats["label_mismatch"] += 1
                            stats["mismatch_examples"].append({"id": rid, "field": "label", "rehost": q_label, "official": off["label"]})
                        if off["split"] != split:
                            stats["split_mismatch"] += 1
                    else:
                        stats["unmatched"] += 1
                        label, label_source = q_label, ("rehost_question" if q_label else "")

                    sample = PixMoCountSample(
                        sample_id=f"{split}:{rid}",
                        question=self.question_text(raw_question),
                        question_type=f"count_{label}" if label else "count",
                        images=[image_path],
                        answer=str(count),
                        split=split,
                        image_sha256=rid,
                        label=label,
                        label_source=label_source,
                        count=count,
                        answer_text=answer_text,
                        raw_question=raw_question,
                        image_url=off["image_url"] if off else "",
                        row_index=row_index,
                        official_matched=off is not None,
                    )
                    row_index += 1
                    if self._wants(sample):
                        self.data.append(sample)

        stats["match_rate"] = stats["matched"] / max(stats["rehost_rows"], 1)
        stats["mismatch_examples"] = stats["mismatch_examples"][:10]
        self.join_stats = stats
        self.n_images_written = n_written

        if n_written:
            print(f"[PixMoCountBench] Extracted {n_written} embedded images to {self._image_dir}")
        if not official:
            print("[PixMoCountBench] WARNING: official annotations not found; labels taken from the re-host question")
        elif stats["count_mismatch"] or stats["label_mismatch"] or stats["split_mismatch"]:
            print(f"[PixMoCountBench] WARNING: official join mismatches: {stats}")
        print(
            f"[PixMoCountBench] Loaded {len(self.data)} samples "
            f"(official join {stats['matched']}/{stats['rehost_rows']})"
        )

    # ── answer extraction ───────────────────────────────────────────────────

    def extract_answer(self, prediction: str) -> str:
        """Canonical integer string ("7") or "" if unparsable (never a default value)."""
        value = extract_integer(prediction)
        return "" if value is None else str(value)

    # ── scoring ─────────────────────────────────────────────────────────────

    @staticmethod
    def _score(gt: int, pred: Optional[int]) -> Tuple[bool, bool, Optional[int]]:
        """(exact, within RA_TOLERANCE, pred - gt) for one item; unparsed -> misses."""
        if pred is None:
            return False, False, None
        err = pred - gt
        exact = err == 0
        ra = exact if gt == 0 else abs(err) <= RA_TOLERANCE * gt
        return exact, ra, err

    def evaluate_single(self, sample: BaseBenchmarkSample, prediction: str) -> Optional[float]:
        """1.0 on exact integer match, else 0.0 (unparsable -> 0.0)."""
        exact, _, _ = self._score(int(sample.answer), extract_integer(prediction))
        return 1.0 if exact else 0.0

    @staticmethod
    def _new_stats() -> Dict[str, Any]:
        return {"correct": 0, "total": 0, "parsed": 0, "ra10": 0, "errors": []}

    @staticmethod
    def _bump(stats: Dict[str, Any], exact: bool, ra: bool, err: Optional[int]) -> None:
        stats["total"] += 1
        stats["correct"] += int(exact)
        stats["ra10"] += int(ra)
        if err is not None:
            stats["parsed"] += 1
            stats["errors"].append(err)

    @staticmethod
    def _finalize(stats: Dict[str, Any]) -> Dict[str, Any]:
        errors = stats["errors"]
        return {
            "accuracy": stats["correct"] / max(stats["total"], 1),
            "correct": stats["correct"],
            "total": stats["total"],
            "ra10": stats["ra10"] / max(stats["total"], 1),
            "mae": mean_or_zero(abs(e) for e in errors) if errors else None,
            "mean_signed_error": mean_or_zero(errors) if errors else None,
            "parsed": stats["parsed"],
            "unparsed": stats["total"] - stats["parsed"],
        }

    def _table(self, stats: Dict[str, Dict[str, Any]], order: List[str]) -> Dict[str, Any]:
        keys = [k for k in order if k in stats] + sorted(set(stats) - set(order))
        return {k: self._finalize(stats[k]) for k in keys}

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        overall = self._new_stats()
        per_split: Dict[str, Dict[str, Any]] = {}
        per_count: Dict[str, Dict[str, Any]] = {}
        per_label: Dict[str, Dict[str, Any]] = {}
        detailed: List[Dict[str, Any]] = []
        matched = 0

        for sample in self.data:
            pred_raw = get_prediction(predictions, sample.sample_id)
            pred_int = extract_integer(pred_raw)
            gt = int(sample.answer)
            exact, ra, err = self._score(gt, pred_int)

            self._bump(overall, exact, ra, err)
            split = getattr(sample, "split", "")
            label = getattr(sample, "label", "")
            self._bump(per_split.setdefault(split, self._new_stats()), exact, ra, err)
            self._bump(per_count.setdefault(str(gt), self._new_stats()), exact, ra, err)
            self._bump(per_label.setdefault(label or "(unknown)", self._new_stats()), exact, ra, err)
            matched += int(bool(getattr(sample, "official_matched", False)))

            detailed.append({
                "sample_id": sample.sample_id,
                "split": split,
                "question_type": sample.question_type,
                "label": label,
                "count": gt,
                "ground_truth": sample.answer,
                "prediction": pred_raw,
                "extracted": "" if pred_int is None else str(pred_int),
                "parsed": pred_int is not None,
                "abs_error": None if err is None else abs(err),
                "correct": exact,
                "official_matched": bool(getattr(sample, "official_matched", False)),
            })

        summary = self._finalize(overall)
        label_order = sorted(per_label, key=lambda k: (-per_label[k]["total"], k))
        results: Dict[str, Any] = {
            "total_samples": summary["total"],
            "correct_samples": summary["correct"],
            "overall_accuracy": summary["accuracy"],
            "scoring": (
                "exact integer match (official Molmo / lmms-eval pixmo_count protocol); parser = last "
                "\\boxed{} -> exact integer / integer after the last answer marker / single / last integer "
                "-> number words; unparsable -> 0 and counted"
            ),
            "overall_aggregation": (
                "overall_accuracy = micro exact-match accuracy over loaded items (test + validation "
                "pooled); ra10 = |pred - gt| <= 0.1 * gt over all items (unparsed = miss; equals exact "
                "match for counts 2..9, accepts 9 and 11 at gt = 10); mae / mean_signed_error over "
                "parsed predictions only"
            ),
            "unparsed_predictions": summary["unparsed"],
            "parsed_predictions": summary["parsed"],
            "ra10": summary["ra10"],
            "mae": summary["mae"],
            "mean_signed_error": summary["mean_signed_error"],
            "official_join": {
                "matched": matched,
                "total": summary["total"],
                "match_rate": matched / max(summary["total"], 1),
            },
            "per_split": self._table(per_split, SPLITS),
            "per_count": self._table(per_count, COUNT_ORDER),
            "per_label": self._table(per_label, label_order),
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

        def _num(v: Optional[float]) -> str:
            return "  n/a" if v is None else f"{v:5.2f}"

        print(f"\n{'=' * 72}")
        print("PixMo-Count Results  [exact integer match; test + validation of the lmms-eval re-host]")
        print(f"{'=' * 72}")
        print(f"Total: {results['total_samples']}  Correct: {results['correct_samples']}  "
              f"Accuracy: {_pct(results['overall_accuracy'])}  RA@10%: {_pct(results.get('ra10'))}")
        print(f"MAE (parsed): {_num(results.get('mae'))}  Bias (pred-gt): {_num(results.get('mean_signed_error'))}  "
              f"Unparsed: {results.get('unparsed_predictions', 0)}  "
              f"Official join: {results.get('official_join', {}).get('matched', '?')}/{results['total_samples']}")
        for title, key, limit in (("Split", "per_split", None), ("GT count", "per_count", None), ("Label", "per_label", 5)):
            table = results.get(key, {})
            if not table:
                continue
            print(f"\n{title:<24s} {'Acc':>8s} {'RA@10':>8s} {'MAE':>6s} {'Correct':>8s} {'Total':>6s} {'Unparsed':>9s}")
            print(f"{'-' * 24} {'-' * 8} {'-' * 8} {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 9}")
            hidden = 0
            for name, s in table.items():
                if limit is not None and s["total"] < limit:
                    hidden += 1
                    continue
                print(f"{str(name)[:24]:<24s} {_pct(s['accuracy']):>8s} {_pct(s['ra10']):>8s} {_num(s['mae']):>6s} "
                      f"{s['correct']:>8d} {s['total']:>6d} {s['unparsed']:>9d}")
            if hidden:
                print(f"(+{hidden} labels with fewer than {limit} items)")
        print(f"{'=' * 72}\n")
