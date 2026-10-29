"""MultihopSpatial benchmark data loader.

MultihopSpatial (ECCV 2026): multi-hop compositional spatial reasoning on a
single image. Each item is a 4-way multiple-choice question whose options are
inlined in the question text as ``<choice>(a) ..., (b) ..., (c) ..., (d) ...</choice>``
followed by a request for the bounding box of the answer object.
4,500 items = 750 per (hop in {1hop, 2hop, 3hop}) x (view in {ego, exo}).

Data structure (HuggingFace repo etri-vilab/MultihopSpatial, shipped as-is):
    data/MultihopSpatial/data/multihop_test_4500.json      (4,500 rows)
    data/MultihopSpatial/data/images/<image_path>          (jpg / jpeg)
    data/MultihopSpatial/parquet/test-00000-of-00001.parquet (fallback; embedded images)

JSON row fields:
    id (int, 1..4500), image_path, image_resolution ("WxH"), view, hop,
    question, question_tag (with <ATT>/<POS>/<REL> tags), answer ("(x) text"),
    bbox ([x, y, w, h] in pixels of the original image)

Metrics (official lmms-eval / VLMEvalKit protocol, denominator = all items):
    - MCQ accuracy (overall_accuracy; primary metric used by the harness)
    - Acc@50IoU  = MCQ correct AND IoU(pred box, gt box) >= 0.5
    - Avg. IoU   = mean IoU over MCQ-correct items with a valid parsed box
  Predicted boxes are xyxy; if any coordinate > 1 the box is treated as
  0-1000 and divided by 1000, otherwise 0-1. Boxes are valid only if every
  coordinate lies in [0, 1] and x2 > x1, y2 > y1.
  GT boxes are normalized by the image size actually on disk. This equals
  ``image_resolution`` for 4,494 rows; six tail rows (4437, 4444, 4445,
  4452, 4455, 4464) carry a stale "640x480" while their GT boxes are in the
  real file's pixel space, so the file size is used and a warning logged
  (the official scorer trusts the metadata there and mis-scores those rows).

Reference: https://github.com/youngwanLEE/multihopspatial
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample, save_embedded_image
from spatial_agent.evals.scoring import get_prediction, write_results_summary

HOPS = ["1hop", "2hop", "3hop"]
VIEWS = ["ego", "exo"]
LETTERS = "abcd"

_TEST_JSON = "multihop_test_4500.json"
_TEST_PARQUET = "test-00000-of-00001.parquet"
_NUM = r"-?\d*\.?\d+(?:[eE][-+]?\d+)?"
_BOX4 = (
    r"\[\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*("
    + _NUM + r")\s*,\s*(" + _NUM + r")\s*\]"
)


@dataclass
class MultihopSpatialSample(BaseBenchmarkSample):
    """A single MultihopSpatial sample (one image, inline 4-way MCQ + box)."""

    hop: str = ""                      # "1hop" | "2hop" | "3hop"
    view: str = ""                     # "ego" | "exo"
    question_tag: str = ""             # question with <ATT>/<POS>/<REL> tags
    answer_text: str = ""              # GT option text (after the "(x) " prefix)
    # Parsed inline options letter -> text. Deliberately NOT named `choices`
    # so run.py / cot_baseline do not append them a second time.
    options: Dict[str, str] = field(default_factory=dict)
    gt_bbox: List[float] = field(default_factory=list)   # [x, y, w, h] pixels
    image_width: int = 0               # size used for scoring (file on disk)
    image_height: int = 0
    image_resolution: str = ""         # raw metadata "WxH" (may be stale)
    image_path: str = ""               # relative path inside data/images


class MultihopSpatialBench(BaseBenchmark):
    """MultihopSpatial loader + scorer (MCQ accuracy, Acc@50IoU, avg IoU)."""

    data_specific_prompt = (
        "Answer with the option letter, then give the bounding box of the "
        "object your answer refers to as [x1, y1, x2, y2] in normalized "
        "0-1000 coordinates."
    )

    def __init__(
        self,
        data_path: str,
        question_type: Optional[List[str]] = None,
        split: str = "test",
    ):
        self._split = split
        self._image_dir: Optional[str] = None
        self._size_cache: Dict[str, Optional[Tuple[int, int]]] = {}
        self.load_warnings: List[str] = []
        super().__init__(data_path, question_type)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _locate_json(self) -> Optional[str]:
        name = _TEST_JSON if self._split == "test" else f"multihop_{self._split}.json"
        for cand in (
            os.path.join(self.data_path, "data", name),
            os.path.join(self.data_path, name),
        ):
            if os.path.isfile(cand):
                return cand
        if self._split != "test":
            # Allow e.g. split="train" resolving to multihop_train_6791.json
            for base in (os.path.join(self.data_path, "data"), self.data_path):
                if os.path.isdir(base):
                    hits = sorted(
                        f for f in os.listdir(base)
                        if f.startswith(f"multihop_{self._split}_") and f.endswith(".json")
                    )
                    if hits:
                        return os.path.join(base, hits[0])
        return None

    def _locate_parquet(self) -> Optional[str]:
        name = _TEST_PARQUET if self._split == "test" else f"{self._split}-00000-of-00001.parquet"
        for cand in (
            os.path.join(self.data_path, "parquet", name),
            os.path.join(self.data_path, "data", name),
            os.path.join(self.data_path, name),
        ):
            if os.path.isfile(cand):
                return cand
        return None

    def _matches_filter(self, hop: str, view: str) -> bool:
        if not self.question_type_filter:
            return True
        wanted = {str(q).lower() for q in self.question_type_filter}
        return bool(
            {f"{hop}-{view}", hop, view, f"{hop}_{view}", f"{hop}/{view}"} & wanted
        )

    def read_data(self) -> None:
        if not os.path.isdir(self.data_path):
            print(f"[Warning] MultihopSpatial data dir not found at {self.data_path}")
            return

        json_path = self._locate_json()
        if json_path is not None:
            self._image_dir = os.path.join(os.path.dirname(json_path), "images")
            with open(json_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            image_getter = self._image_from_json
        else:
            parquet_path = self._locate_parquet()
            if parquet_path is None:
                print(
                    f"[Warning] MultihopSpatial: neither {_TEST_JSON} nor "
                    f"{_TEST_PARQUET} found under {self.data_path}"
                )
                return
            import pandas as pd

            self._image_dir = os.path.join(self.data_path, ".image_cache")
            os.makedirs(self._image_dir, exist_ok=True)
            df = pd.read_parquet(parquet_path)
            rows = df.to_dict("records")
            image_getter = self._image_from_parquet

        # Deterministic order: by integer id (JSON is already sorted; be safe).
        rows = sorted(rows, key=lambda r: int(r["id"]))

        n_loaded = 0
        for row in rows:
            hop = str(row.get("hop", "")).strip()
            view = str(row.get("view", "")).strip()
            if not self._matches_filter(hop, view):
                continue

            sid = str(row["id"])
            question = str(row.get("question", ""))
            answer_raw = str(row.get("answer", ""))
            gt_letter, gt_text = self._split_answer(answer_raw)
            options = self.parse_options(question)

            if not gt_letter:
                self.load_warnings.append(f"{sid}: unparsable GT answer {answer_raw!r}")
            if len(options) != 4:
                self.load_warnings.append(
                    f"{sid}: parsed {len(options)} options (expected 4)"
                )
            elif gt_letter and gt_letter not in options:
                self.load_warnings.append(
                    f"{sid}: GT letter {gt_letter!r} not among options {sorted(options)}"
                )

            image_path, rel_path = image_getter(row)
            if not image_path or not os.path.isfile(image_path):
                self.load_warnings.append(f"{sid}: image missing {image_path!r}")

            # Scoring size = the file on disk (header read only, cached per
            # path). The metadata "WxH" is stale on six tail rows while the
            # GT boxes are in the real file's pixel space.
            meta_res = str(row.get("image_resolution") or "")
            width, height = self._parse_resolution(meta_res)
            actual = self._image_size(image_path) if image_path else None
            if actual is None:
                if image_path and os.path.isfile(image_path):
                    self.load_warnings.append(f"{sid}: cannot read image size of {image_path!r}")
            else:
                if width > 0 and height > 0 and (width, height) != actual:
                    self.load_warnings.append(
                        f"{sid}: image_resolution {width}x{height} != file "
                        f"{actual[0]}x{actual[1]}; using file size"
                    )
                width, height = actual

            bbox = row.get("bbox", [])
            if hasattr(bbox, "tolist"):
                bbox = bbox.tolist()
            bbox = [float(v) for v in (bbox or [])]
            if len(bbox) != 4:
                self.load_warnings.append(f"{sid}: GT bbox has {len(bbox)} values")

            sample = MultihopSpatialSample(
                sample_id=sid,
                question=question,
                question_type=f"{hop}-{view}",
                images=[image_path] if image_path else [],
                answer=gt_letter,
                hop=hop,
                view=view,
                question_tag=str(row.get("question_tag", "")),
                answer_text=gt_text,
                options=options,
                gt_bbox=bbox,
                image_width=width,
                image_height=height,
                image_resolution=meta_res,
                image_path=rel_path,
            )
            self.data.append(sample)
            n_loaded += 1

        if self.load_warnings:
            print(
                f"[MultihopSpatial] {len(self.load_warnings)} data irregularities "
                f"(first 5): {self.load_warnings[:5]}"
            )
        print(f"[MultihopSpatial] Loaded {n_loaded} samples")

    def _image_from_json(self, row: Dict[str, Any]) -> Tuple[str, str]:
        rel = str(row.get("image_path", "")).strip()
        if not rel:
            return "", ""
        return os.path.abspath(os.path.join(self._image_dir, rel)), rel

    def _image_from_parquet(self, row: Dict[str, Any]) -> Tuple[str, str]:
        img = row.get("image")
        if img is None:
            return "", ""
        rel = ""
        if isinstance(img, dict):
            rel = str(img.get("path") or "")
        rel = os.path.basename(rel) if rel else f"{int(row['id'])}.jpg"
        out = os.path.abspath(os.path.join(self._image_dir, rel))
        try:
            save_embedded_image(out, img)
        except Exception as exc:
            self.load_warnings.append(f"{row.get('id')}: cannot save embedded image ({exc})")
            return "", rel
        return out, rel

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _image_size(self, image_path: str) -> Optional[Tuple[int, int]]:
        """(W, H) of the image file (PIL header read, cached), or None."""
        if not image_path or not os.path.isfile(image_path):
            return None
        if image_path in self._size_cache:
            return self._size_cache[image_path]
        size: Optional[Tuple[int, int]] = None
        try:
            from PIL import Image

            with Image.open(image_path) as im:
                size = (int(im.size[0]), int(im.size[1]))
        except Exception:  # pragma: no cover - defensive
            size = None
        self._size_cache[image_path] = size
        return size

    @staticmethod
    def _parse_resolution(res: Any) -> Tuple[int, int]:
        if not res:
            return 0, 0
        m = re.match(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*$", str(res))
        if not m:
            return 0, 0
        return int(m.group(1)), int(m.group(2))

    @staticmethod
    def _split_answer(answer: str) -> Tuple[str, str]:
        """'(c) frame of the reed picture' -> ('c', 'frame of the reed picture')."""
        if not answer:
            return "", ""
        m = re.match(r"\s*\(?\s*([A-Da-d])\s*[\)\.:]?\s*(.*)$", answer.strip(), re.S)
        if not m:
            return "", answer.strip()
        return m.group(1).lower(), m.group(2).strip()

    @staticmethod
    def parse_options(question: str) -> Dict[str, str]:
        """Parse inline ``<choice>(a) ..., (b) ...</choice>`` options.

        Robust to a missing space after the letter ('(c)umbrella'), missing
        commas between options, and a malformed closing tag ('/choice>').
        """
        if not question:
            return {}
        # Closing tag may be malformed: '</choice>' or '/choice>' (id 3057).
        m = re.search(r"<choice>(.*?)(?:<?/?choice>|$)", question, re.S | re.I)
        inner = m.group(1) if m else question
        # One row (id 4445) has no <choice> tags: options are followed directly
        # by the fixed grounding request. Trim that sentence so the last option
        # does not absorb it.
        inner = re.split(r"\.?\s*And provide the bounding box", inner, maxsplit=1, flags=re.I)[0]
        marks = list(re.finditer(r"\(([A-Da-d])\)", inner))
        options: Dict[str, str] = {}
        for i, mk in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(inner)
            text = inner[mk.end():end].strip()
            text = re.sub(r"<?/?choice>?\s*$", "", text, flags=re.I)  # stray tag fragment
            text = text.rstrip(",;.").strip()
            letter = mk.group(1).lower()
            if letter not in options:
                options[letter] = text
        return options

    @staticmethod
    def _norm_text(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def _letter_from_option_text(
        cls, text: str, options: Optional[Dict[str, str]]
    ) -> str:
        """Map free text to an option letter via the option text (unique match)."""
        if not options or not text:
            return ""
        norm = cls._norm_text(text)
        if not norm:
            return ""
        # Exact equality first
        for letter, opt in options.items():
            if cls._norm_text(opt) == norm:
                return letter
        # Unique containment (whole-phrase); prefer the longest option text
        hits = []
        for letter, opt in options.items():
            o = cls._norm_text(opt)
            if o and re.search(r"(?<![a-z0-9])" + re.escape(o) + r"(?![a-z0-9])", norm):
                hits.append((len(o), letter))
        if not hits:
            return ""
        hits.sort(reverse=True)
        if len(hits) == 1 or hits[0][0] > hits[1][0]:
            # Only accept when a single option matches, or one is strictly
            # longer (e.g. 'red van' vs 'van').
            if len(hits) == 1:
                return hits[0][1]
            # several distinct options mentioned -> ambiguous unless nested
            longest = options[hits[0][1]]
            nested = all(
                cls._norm_text(options[l]) in cls._norm_text(longest) for _, l in hits[1:]
            )
            return hits[0][1] if nested else ""
        return ""

    @staticmethod
    def _letter_rules(text: str) -> str:
        """Ordered letter-extraction rules on already-unwrapped text."""
        if not text:
            return ""
        t = text.strip()

        # <answer>(c)</answer>
        m = re.search(
            r"<answer>\s*\(?\s*([A-Da-d])\s*[\.\)]?\s*</answer>", t, re.I | re.S
        )
        if m:
            return m.group(1).lower()

        # Leading answer: "(c) ...", "C. ...", "c) ...", "C: ...", "C\n..."
        m = re.match(r"\W*\(\s*([A-Da-d])\s*\)", t)
        if m:
            return m.group(1).lower()
        m = re.match(r"\W*([A-Da-d])\s*[\.\):]", t)
        if m:
            return m.group(1).lower()
        m = re.match(r"\W*([A-D])(?:\s*$|\s*\n)", t)
        if m:
            return m.group(1).lower()

        # Explicit markers - last occurrence wins. Case-insensitivity is
        # scoped to the marker words: a lowercase letter counts only when it
        # is parenthesized or followed by punctuation/quote/end, so the
        # article in "the answer is a bottle" is not taken as option (a).
        # Quotes / backticks / bold markers around the marker or the letter
        # are tolerated ('"answer": "c"', '**Answer:** (c)', 'Answer: `C`').
        marker = re.compile(
            r"(?i:final\s+answer|correct\s+answer|answer\s+is|the\s+answer|my\s+answer|"
            r"best\s+answer|best\s+option|correct\s+option|correct\s+choice|"
            r"answer|option|choice)\b[\"'`*]*\s*(?i:is|:|-|=)?\s*[\"'`*]*\s*"
            r"(?:\(\s*([A-Da-d])\s*\)|([A-D])(?![A-Za-z])|"
            r"([a-d])(?=[ \t]*[\.\):,;\"'`*]|[ \t]*(?:\r?\n|$)))"
        )
        ms = list(marker.finditer(t))
        if ms:
            g = ms[-1]
            return (g.group(1) or g.group(2) or g.group(3)).lower()

        # Parenthesized letter anywhere - last occurrence wins
        ms = list(re.finditer(r"\(\s*([A-Da-d])\s*\)", t))
        if ms:
            return ms[-1].group(1).lower()

        # "C." / "C)" / "C:" not glued to a word or digit (avoids 'etc.',
        # '2d:') - last wins
        ms = list(re.finditer(r"(?<![A-Za-z0-9])([A-Da-d])[\.\):](?![A-Za-z0-9])", t))
        if ms:
            return ms[-1].group(1).lower()

        # Whole text is a single letter
        m = re.match(r"^\W*([A-Da-d])\W*$", t)
        if m:
            return m.group(1).lower()

        return ""

    def extract_answer(
        self, prediction: str, options: Optional[Dict[str, str]] = None
    ) -> str:
        """Return the predicted option letter in lowercase ('a'-'d') or ''.

        Handles the agent's raw final answer (bare letter, '(c)', 'C.', a
        sentence, '\\boxed{C}', JSON with a bbox) and the CoT's boxed-stripped
        text. If ``options`` are given and the model wrote only the option
        text, the letter is recovered from the text.
        """
        if not prediction:
            return ""
        text = str(prediction).strip()
        if not text:
            return ""

        # 1. \boxed{...} - prefer the LAST boxed content (final answer), then
        #    earlier ones (e.g. "\boxed{C} ... \boxed{[x1, y1, x2, y2]}").
        boxed_re = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        boxed = re.findall(boxed_re, text, re.S)
        if boxed:
            for inner in reversed(boxed):
                inner = inner.strip()
                letter = self._letter_rules(inner)
                if letter:
                    return letter
                letter = self._letter_from_option_text(inner, options)
                if letter:
                    return letter
            # fall through with the wrappers removed but their content kept
            text = re.sub(boxed_re, r" \1 ", text, flags=re.S).strip()
            if not text:
                return ""

        # 2. Strip bbox payloads (the boxes and their keys such as "bbox_2d":)
        #    so numbers / JSON keys never confuse the letter rules. Only the
        #    box parts are removed, so '{"answer": "c", "bbox_2d": [...]}'
        #    keeps its letter.
        stripped = re.sub(_BOX4, " ", text)
        stripped = re.sub(
            r"[\"'`]?\bbbox(?:_2d)?\b[\"'`]?\s*[:=]?", " ", stripped, flags=re.I
        )

        letter = self._letter_rules(stripped)
        if letter:
            return letter

        # 3. Option text only
        letter = self._letter_from_option_text(stripped, options)
        if letter:
            return letter

        # 4. Weak fallback: a single standalone uppercase letter in the tail
        #    that is not used as an article ("A cat") and not part of a token
        #    such as "N/A". Several distinct letters ("B or C") -> ambiguous.
        tail = stripped[-300:]
        standalone = re.findall(r"(?<![A-Za-z/])([A-D])(?![A-Za-z])", tail)
        if len(set(standalone)) == 1:
            ms = re.findall(r"(?<![A-Za-z/])([A-D])(?![A-Za-z])(?!\s+[a-z])", tail)
            if ms:
                return ms[-1].lower()

        return ""

    @staticmethod
    def parse_bbox(prediction: str) -> Optional[List[float]]:
        """Extract a predicted xyxy box normalized to [0, 1], or None.

        Search order (official): ``"bbox_2d": [..]`` -> ``Bounding Box: [..]``
        -> first bare ``[n, n, n, n]``. If any coordinate > 1 the box is
        interpreted as 0-1000 and divided by 1000. Invalid boxes (outside
        [0, 1], zero/negative extent) return None.
        """
        if not prediction:
            return None
        text = str(prediction)
        patterns = [
            r"[\"']?bbox_2d[\"']?\s*:\s*" + _BOX4,
            r"bounding\s*box\s*(?:coordinates?)?\s*[:=]?\s*(?:\{[^\[\]{}]*)?" + _BOX4,
            _BOX4,
        ]
        nums: Optional[List[float]] = None
        for pat in patterns:
            m = re.search(pat, text, re.I | re.S)
            if m:
                try:
                    nums = [float(m.group(i)) for i in range(1, 5)]
                except (TypeError, ValueError):
                    nums = None
                if nums is not None:
                    break
        if nums is None:
            return None
        if any(v > 1.0 for v in nums):
            nums = [v / 1000.0 for v in nums]
        x1, y1, x2, y2 = nums
        if any(v < 0.0 or v > 1.0 for v in nums):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def gt_bbox_norm(sample: MultihopSpatialSample) -> Optional[List[float]]:
        """GT [x, y, w, h] pixels -> normalized xyxy (not clipped, as official)."""
        if len(sample.gt_bbox) != 4 or sample.image_width <= 0 or sample.image_height <= 0:
            return None
        x, y, w, h = sample.gt_bbox
        W, H = float(sample.image_width), float(sample.image_height)
        return [x / W, y / H, (x + w) / W, (y + h) / H]

    @staticmethod
    def iou_xyxy(a: List[float], b: List[float]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_sample(
        self, sample: MultihopSpatialSample, prediction: str
    ) -> Dict[str, Any]:
        """Per-item scoring: MCQ letter, box, IoU, Acc@50IoU flag."""
        options = getattr(sample, "options", None)
        pred_letter = self.extract_answer(prediction, options)
        gt_letter = (sample.answer or "").strip().lower()
        mcq_correct = bool(pred_letter) and pred_letter == gt_letter

        pred_box = self.parse_bbox(prediction)
        gt_box = self.gt_bbox_norm(sample)
        iou: Optional[float] = None
        if pred_box is not None and gt_box is not None:
            iou = round(self.iou_xyxy(pred_box, gt_box), 4)
        acc50 = bool(mcq_correct and iou is not None and iou >= 0.5)
        return {
            "pred_letter": pred_letter,
            "gt_letter": gt_letter,
            "mcq_correct": mcq_correct,
            "pred_bbox": pred_box,
            "gt_bbox_norm": gt_box,
            "iou": iou,
            "acc50iou": acc50,
        }

    def evaluate_single(
        self, sample: BaseBenchmarkSample, prediction: str
    ) -> Optional[float]:
        """Primary metric: MCQ letter match (1.0 / 0.0)."""
        options = getattr(sample, "options", None)
        pred = self.extract_answer(prediction, options)
        gt = (sample.answer or "").strip().lower()
        return 1.0 if (pred and pred == gt) else 0.0

    @staticmethod
    def _new_bucket() -> Dict[str, Any]:
        return {"total": 0, "correct": 0, "acc50_correct": 0, "iou_sum": 0.0, "iou_count": 0}

    @staticmethod
    def _finalize_bucket(b: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "total": b["total"],
            "correct": b["correct"],
            "accuracy": b["correct"] / max(b["total"], 1),
            "acc50_correct": b["acc50_correct"],
            "acc_at_50iou": b["acc50_correct"] / max(b["total"], 1),
            "avg_iou": (b["iou_sum"] / b["iou_count"]) if b["iou_count"] else 0.0,
            "avg_iou_count": b["iou_count"],
        }

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        overall = self._new_bucket()
        per_hop_view: Dict[str, Dict[str, Any]] = {}
        per_hop: Dict[str, Dict[str, Any]] = {}
        per_view: Dict[str, Dict[str, Any]] = {}
        detailed: List[Dict[str, Any]] = []
        n_unparsed_letter = 0
        n_parsed_box = 0

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            s = self.score_sample(sample, pred_raw)

            if not s["pred_letter"]:
                n_unparsed_letter += 1
            if s["pred_bbox"] is not None:
                n_parsed_box += 1

            buckets = [
                overall,
                per_hop_view.setdefault(sample.question_type, self._new_bucket()),
                per_hop.setdefault(sample.hop, self._new_bucket()),
                per_view.setdefault(sample.view, self._new_bucket()),
            ]
            for b in buckets:
                b["total"] += 1
                if s["mcq_correct"]:
                    b["correct"] += 1
                    if s["iou"] is not None:
                        b["iou_sum"] += s["iou"]
                        b["iou_count"] += 1
                if s["acc50iou"]:
                    b["acc50_correct"] += 1

            detailed.append({
                "id": sid,
                "hop": sample.hop,
                "view": sample.view,
                "question_type": sample.question_type,
                "ground_truth": sample.answer,
                "ground_truth_text": sample.answer_text,
                "gt_bbox_xywh_px": sample.gt_bbox,
                "prediction": pred_raw,
                "extracted": s["pred_letter"],
                "pred_bbox_norm": s["pred_bbox"],
                "iou": s["iou"],
                "correct": s["mcq_correct"],
                "acc50iou": s["acc50iou"],
            })

        def _sorted(d: Dict[str, Dict[str, Any]], order: List[str]) -> Dict[str, Any]:
            keys = [k for k in order if k in d] + sorted(k for k in d if k not in order)
            return {k: self._finalize_bucket(d[k]) for k in keys}

        hv_order = [f"{h}-{v}" for h in HOPS for v in VIEWS]
        ov = self._finalize_bucket(overall)
        results: Dict[str, Any] = {
            "total_samples": ov["total"],
            "correct_samples": ov["correct"],
            "overall_accuracy": ov["accuracy"],            # MCQ accuracy (primary)
            "overall_aggregation": "micro MCQ accuracy over all items (unparsable = wrong)",
            "mcq_accuracy": ov["accuracy"],
            "acc_at_50iou": ov["acc_at_50iou"],
            "acc_at_50iou_correct": ov["acc50_correct"],
            "avg_iou": ov["avg_iou"],
            "avg_iou_count": ov["avg_iou_count"],
            "parsed_bbox_samples": n_parsed_box,
            "unparsed_answer_samples": n_unparsed_letter,
            "per_hop_view": _sorted(per_hop_view, hv_order),
            "per_hop": _sorted(per_hop, HOPS),
            "per_view": _sorted(per_view, VIEWS),
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*78}")
        print("MultihopSpatial Results")
        print(f"{'='*78}")
        print(
            f"MCQ accuracy : {results['overall_accuracy']*100:6.2f}% "
            f"({results['correct_samples']}/{results['total_samples']})"
        )
        print(
            f"Acc@50IoU    : {results.get('acc_at_50iou', 0.0)*100:6.2f}% "
            f"({results.get('acc_at_50iou_correct', 0)}/{results['total_samples']})"
        )
        print(
            f"Avg IoU      : {results.get('avg_iou', 0.0)*100:6.2f}  "
            f"(over {results.get('avg_iou_count', 0)} MCQ-correct items with a valid box)"
        )
        print(
            f"Parsed boxes : {results.get('parsed_bbox_samples', 0)}   "
            f"Unparsed answers: {results.get('unparsed_answer_samples', 0)}"
        )
        print(f"{'-'*78}")
        print(f"{'Group':<14s} {'Acc':>8s} {'Acc@50':>8s} {'AvgIoU':>8s} {'Correct':>8s} {'Total':>6s}")
        print(f"{'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for section in ("per_hop_view", "per_hop", "per_view"):
            for name, st in results.get(section, {}).items():
                print(
                    f"{name:<14s} {st['accuracy']*100:>7.2f}% {st['acc_at_50iou']*100:>7.2f}% "
                    f"{st['avg_iou']*100:>8.2f} {st['correct']:>8d} {st['total']:>6d}"
                )
        print(f"{'='*78}\n")
