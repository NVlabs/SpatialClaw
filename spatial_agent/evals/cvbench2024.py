"""CV-Bench (Cambrian-1, 2024) data loader.

CV-Bench is the Cambrian Vision-Centric Benchmark from Cambrian-1
(Tong et al., 2024, arXiv 2406.16860): 2,638 single-image multiple-choice
questions that repurpose classic vision datasets into VQA form.

    2D (1,438): Count (788)  + Relation (650)   sources ADE20K (633), COCO (805)
    3D (1,200): Depth (600)  + Distance (600)   source  Omni3D (Hypersim / SUN RGB-D / nuScenes)

NOTE: this is DISTINCT from the cross-video "CVBench" (Dongyh35/CVBench)
registered under the key ``cvbench`` in spatial_agent/evals/cvbench.py.

Data structure (HF repo nyu-visionx/CV-Bench, images embedded in parquet):
    data/CV-Bench-2024/test_2d.parquet      1438 rows, global idx 0..1437
    data/CV-Bench-2024/test_3d.parquet      1200 rows, global idx 1438..2637
    data/CV-Bench-2024/images/<idx>.png     dumped once on first load

Parquet columns used:
    idx        int     global unique index 0..2637 -> sample_id
    type       str     "2D" | "3D"
    task       str     Count | Relation | Depth | Distance -> question_type
    image      struct  {bytes, path}
    prompt     str     question + inline "(A) ..\\n(B) .." options (used verbatim)
    choices    list    option texts in letter order (kept for a text-match fallback)
    answer     str     "(A)".."(F)" -> stripped to the bare letter
    source     str     ADE20K | COCO | Omni3D (grouping key of the official metric)
    source_dataset, filename, bbox, target_class  (metadata only)

Metric: exact option-letter match. Reported: micro accuracy over all loaded
samples, per task / source / source_dataset / type, and the official aggregate
    acc_2d      = mean(acc_ADE20K, acc_COCO)
    acc_3d      = acc_Omni3D
    cvbench_acc = mean(acc_2d, acc_3d)
(dataset README "Accuracy" section; Cambrian-1 Appendix C.3).

Reference: https://huggingface.co/datasets/nyu-visionx/CV-Bench
           https://github.com/cambrian-mllm/cambrian
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from spatial_agent.evals.base import (
    BaseBenchmark,
    BaseBenchmarkSample,
    save_embedded_image,
)
from spatial_agent.evals.scoring import (
    get_prediction,
    mean_or_zero,
    write_json,
    write_results_summary,
)

TASKS = ["Count", "Relation", "Depth", "Distance"]
SOURCES = ["ADE20K", "COCO", "Omni3D"]
TYPES = ["2D", "3D"]
PARQUET_FILES = ["test_2d.parquet", "test_3d.parquet"]  # load order: 2D then 3D

_LETTERS = "ABCDEF"
_LETTER_CLASS = "[A-F]"
_LETTER_CLASS_I = "[A-Fa-f]"

# Full-width / typographic punctuation and invisible characters -> ASCII, so
# "（C）", "Answer：C", "C – table" or "B\u200b" parse like their ASCII forms.
_PUNCT_NORMALISE = str.maketrans({
    "（": "(", "）": ")", "：": ":", "．": ".", "，": ",", "－": "-",
    "–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "", "\u00a0": " ",
})

# Marker words that explicitly announce the answer (STRONG) vs. a mere mention
# of an option/choice (WEAK). Priority: strong marker > leading delimited
# letter ("C. table") > weak marker, so that "Answer: C. Explanation: option B
# is farther." and "C. The bookcase (option B) is farther." both yield C.
_STRONG_MARKERS = (
    r"final\s+answer|correct\s+answer|best\s+answer|the\s+answer|answer\s+is|"
    r"best\s+option|correct\s+option|best\s+choice|correct\s+choice|answer"
)
_WEAK_MARKERS = r"option|choice"


def _marker_pattern(words: str) -> "re.Pattern[str]":
    """<marker>[ is| would be| should be][: | - | =] followed by a letter.

    Accepted letter forms after a marker:
      (x)      parenthesised, any case
      X        upper-case, not glued to a word or number
      x<end>   lower-case only as the final token (so "answer is a chair"
               never yields A while "answer is a" / "answer: b." do)
    """
    return re.compile(
        r"(?i:" + words + r")\b\s*(?i:is|would\s+be|should\s+be)?\s*[:\-=]?\s*"
        r"(?:\(\s*(" + _LETTER_CLASS_I + r")\s*\)"
        r"|(" + _LETTER_CLASS + r")(?![A-Za-z0-9])"
        r"|(" + _LETTER_CLASS_I + r")(?=[\s\.\!\)\,]*$))"
    )


_STRONG_RE = _marker_pattern(_STRONG_MARKERS)
_WEAK_RE = _marker_pattern(_WEAK_MARKERS)


def _norm_text(s: str) -> str:
    """Normalise free text for option-text comparison."""
    s = str(s).translate(_PUNCT_NORMALISE).strip().lower()
    s = re.sub(r"[\*`\"']", "", s)          # markdown / quotes
    s = s.strip().strip("()[]").strip()
    s = s.rstrip(".").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _extract_letter(text: str) -> str:
    """Return an upper-case option letter A-F or "" (never a default letter).

    Handles both the agent's raw final-answer text and the CoT's
    ``\\boxed{}``-stripped text: "C", "(C)", "C.", "C:", "C - table",
    "\\boxed{C}", "\\boxed{(C)}", "The answer is C", "**C**", "c", "(c)",
    "c.", "Answer: \"C\"", "（C）" ...

    Lower-case letters are accepted only in unambiguous positions (bare
    letter, "(c)", "c." / "c:" at the start of the text, inside \\boxed{}, or
    as the final token after an explicit answer marker) so that the article
    "a" in a sentence is never mistaken for option A.
    """
    if not text:
        return ""
    s = str(text).translate(_PUNCT_NORMALISE).strip()
    if not s:
        return ""

    # 1. \boxed{C} / \boxed{(C)} / \boxed{C.} / \boxed{\text{C}} -- take the LAST one
    boxed = re.findall(
        r"\\boxed\{\s*(?:\\text\{)?\s*[\(\"']?\s*(" + _LETTER_CLASS_I + r")\s*[\.\)\"']?\s*\}?\s*\}",
        s,
    )
    if boxed:
        return boxed[-1].upper()

    # Strip markdown emphasis and quotes so "**C**" / "**Answer:** C" /
    # 'Answer: "C"' parse cleanly
    s = re.sub(r"[\*_`\"']+", " ", s).strip()

    # 2. Whole text is a bare letter: "C", "(C)", "C.", "C)", "C!", "c"
    m = re.fullmatch(r"\(?\s*(" + _LETTER_CLASS_I + r")\s*[\)\.\:\!\,]?", s)
    if m:
        return m.group(1).upper()

    # 3. Explicit answer markers ("the answer is X", "Answer: X", "correct
    #    option (x)"), take the LAST occurrence
    ms = list(_STRONG_RE.finditer(s))
    if ms:
        return next(g for g in ms[-1].groups() if g).upper()

    # 4. Letter with delimiter at the very start: "C. table", "(c) table",
    #    "C: ...", "C - table", "C) because ..."
    m = re.match(r"\(?\s*(" + _LETTER_CLASS_I + r")\s*[\)\.\:\-](?![A-Za-z0-9])", s)
    if m:
        return m.group(1).upper()

    # 5. Weak markers ("option X", "choice (x)"), take the LAST occurrence
    ms = list(_WEAK_RE.finditer(s))
    if ms:
        return next(g for g in ms[-1].groups() if g).upper()

    # 6. Parenthesised upper-case letter anywhere, take the LAST: "... is (C)."
    ms = re.findall(r"\((" + _LETTER_CLASS + r")\)", s)
    if ms:
        return ms[-1].upper()

    # 7. Upper-case letter followed by '.' or ':' anywhere, take the LAST: "... C."
    ms = re.findall(r"(?<![A-Za-z0-9])(" + _LETTER_CLASS + r")[\.\:](?![A-Za-z0-9])", s)
    if ms:
        return ms[-1].upper()

    # 8. Upper-case letter as the final token: "I think it is C"
    m = re.search(r"(?<![A-Za-z0-9])(" + _LETTER_CLASS + r")[\.\)\!\,]?\s*$", s)
    if m:
        return m.group(1).upper()

    # 9. Only one letter character left after stripping everything else
    clean = re.sub(r"[^A-Za-z]", "", s)
    if len(clean) == 1 and clean.upper() in _LETTERS:
        return clean.upper()

    return ""


@dataclass
class CVBench2024Sample(BaseBenchmarkSample):
    """A single CV-Bench (2024) sample.

    ``question`` holds the dataset ``prompt`` verbatim (options already
    inline), so no ``choices`` attribute is defined: run.py / cot_baseline
    would otherwise append the options a second time. The letter -> text map
    lives in ``options`` and is only used for the answer-text fallback.
    """

    task: str = ""                       # Count | Relation | Depth | Distance
    data_type: str = ""                  # 2D | 3D
    source: str = ""                     # ADE20K | COCO | Omni3D
    source_dataset: str = ""             # e.g. Omni3D_Hypersim
    filename: str = ""                   # original img/... name from the dataset
    options: Dict[str, str] = field(default_factory=dict)   # {"A": "table", ...}
    bbox: Optional[List[List[float]]] = None                 # Omni3D rows only
    target_class: str = ""               # COCO rows only


class CVBench2024(BaseBenchmark):
    """CV-Bench (Cambrian-1, 2024) loader and scorer."""

    data_specific_prompt = (
        "Answer with the letter of the correct option only."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        self._image_dir = os.path.join(os.path.abspath(data_path), "images")
        super().__init__(data_path, question_type)

    # ── loading ──────────────────────────────────────────────────────────

    def _image_path(self, idx: int) -> str:
        return os.path.join(self._image_dir, f"{int(idx)}.png")

    def _wants(self, task: str, data_type: str, source: str) -> bool:
        """question_type filter: match task, type (2D/3D) or source, case-insensitively."""
        if not self.question_type_filter:
            return True
        wanted = {str(q).strip().lower() for q in self.question_type_filter}
        return bool({task.lower(), data_type.lower(), source.lower()} & wanted)

    @staticmethod
    def _image_ok(path: str) -> bool:
        """True if the cached image exists and is not a truncated 0-byte file."""
        try:
            return os.path.getsize(path) > 0
        except OSError:
            return False

    def _write_image(self, idx: int, img: Any) -> None:
        """Atomically persist one embedded image as <idx>.png (tmp + rename)."""
        final = self._image_path(idx)
        # pid-unique temp name: concurrent first-time loaders (e.g. a job
        # array) must not remove / rename each other's half-written files
        tmp = os.path.join(self._image_dir, f"{int(idx)}.{os.getpid()}.tmp.png")
        if os.path.exists(tmp):
            os.remove(tmp)
        try:
            save_embedded_image(tmp, img, convert_rgb=True)
            os.replace(tmp, final)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _dump_missing_images(self, pf, rows: List[Dict[str, Any]]) -> int:
        """Persist embedded images that are not on disk yet. Returns #written.

        Only the heavy ``image`` column of row groups containing a missing
        image is read, so repeat loads cost ~1 s.
        """
        missing = {int(r["idx"]) for r in rows if not self._image_ok(self._image_path(r["idx"]))}
        if not missing:
            return 0
        os.makedirs(self._image_dir, exist_ok=True)
        written = 0
        for rg in range(pf.num_row_groups):
            idxs = [int(i) for i in pf.read_row_group(rg, columns=["idx"]).column("idx").to_pylist()]
            if not any(i in missing for i in idxs):
                continue
            images = pf.read_row_group(rg, columns=["image"]).column("image").to_pylist()
            for idx, img in zip(idxs, images):
                if idx not in missing or img is None:
                    continue
                self._write_image(idx, img)
                written += 1
        return written

    def read_data(self) -> None:
        import pyarrow.parquet as pq

        self.data_path = os.path.abspath(self.data_path)
        if not os.path.isdir(self.data_path):
            raise FileNotFoundError(
                f"CV-Bench-2024 data dir not found: {self.data_path}. "
                "Download nyu-visionx/CV-Bench (test_2d.parquet, test_3d.parquet) there."
            )

        n_written = 0
        for fname in PARQUET_FILES:
            path = os.path.join(self.data_path, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"CV-Bench-2024 parquet not found: {path}")

            pf = pq.ParquetFile(path)
            meta_cols = [c for c in pf.schema_arrow.names if c != "image"]
            rows = pf.read(columns=meta_cols).to_pylist()

            # Persist images once (only reads the heavy column when needed)
            n_written += self._dump_missing_images(pf, rows)

            for row in rows:
                idx = int(row["idx"])
                task = str(row.get("task") or "").strip()
                data_type = str(row.get("type") or "").strip()
                source = str(row.get("source") or "").strip()
                if not self._wants(task, data_type, source):
                    continue

                choice_texts = [str(c) for c in (row.get("choices") or [])]
                options = {
                    _LETTERS[i]: txt for i, txt in enumerate(choice_texts) if i < len(_LETTERS)
                }

                answer = str(row.get("answer") or "").strip().strip("()").strip().upper()

                bbox = row.get("bbox")
                if bbox is not None:
                    bbox = [[float(v) for v in box] for box in bbox]

                prompt = str(row.get("prompt") or "").strip()
                if not prompt:  # defensive: rebuild from question + choices
                    prompt = str(row.get("question") or "").strip() + "".join(
                        f"\n({_LETTERS[i]}) {txt}" for i, txt in enumerate(choice_texts)
                    )

                self.data.append(
                    CVBench2024Sample(
                        sample_id=str(idx),
                        question=prompt,
                        question_type=task,
                        images=[self._image_path(idx)],
                        answer=answer,
                        task=task,
                        data_type=data_type,
                        source=source,
                        source_dataset=str(row.get("source_dataset") or ""),
                        filename=str(row.get("filename") or ""),
                        options=options,
                        bbox=bbox,
                        target_class=str(row.get("target_class") or ""),
                    )
                )

        if n_written:
            print(f"[CVBench2024] Extracted {n_written} embedded images to {self._image_dir}")
        print(f"[CVBench2024] Loaded {len(self.data)} samples")

    # ── answer extraction ────────────────────────────────────────────────

    def extract_answer(self, prediction: str) -> str:
        """Return the predicted option letter (A-F) or "" if unparsable."""
        return _extract_letter(prediction)

    def _extract_with_fallback(self, sample: BaseBenchmarkSample, prediction: str) -> str:
        """Letter extraction, then an exact option-text match (unambiguous only)."""
        letter = _extract_letter(prediction)
        if letter:
            return letter
        options = getattr(sample, "options", None) or {}
        if not options or not prediction:
            return ""

        text = str(prediction)
        candidates = [_norm_text(text)]
        # \boxed{table} / \boxed{6} (raw agent text; cot_baseline pre-strips it)
        boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
        if boxed:
            candidates.append(_norm_text(boxed[-1]))
        # Text after the last explicit answer marker, e.g. "The answer is table."
        ms = list(re.finditer(r"(?:answer|option|choice)\s*(?:is|:|=)?\s*", text, re.IGNORECASE))
        if ms:
            candidates.append(_norm_text(text[ms[-1].end():]))

        for cand in candidates:
            if not cand:
                continue
            hits = [ltr for ltr, txt in options.items() if _norm_text(txt) == cand]
            if len(hits) == 1:
                return hits[0]
        return ""

    # ── scoring ──────────────────────────────────────────────────────────

    def evaluate_single(
        self, sample: BaseBenchmarkSample, prediction: str
    ) -> Optional[float]:
        pred = self._extract_with_fallback(sample, prediction)
        gt = str(sample.answer).strip().strip("()").upper()
        return 1.0 if (pred and pred == gt) else 0.0

    @staticmethod
    def _acc(stats: Dict[str, int]) -> float:
        return stats["correct"] / max(stats["total"], 1)

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        per_task: Dict[str, Dict[str, int]] = {}
        per_source: Dict[str, Dict[str, int]] = {}
        per_source_dataset: Dict[str, Dict[str, int]] = {}
        per_type: Dict[str, Dict[str, int]] = {}
        detailed: List[Dict[str, Any]] = []
        total = 0
        correct = 0
        unparsed = 0

        def _bump(table: Dict[str, Dict[str, int]], key: str, ok: bool) -> None:
            st = table.setdefault(key, {"correct": 0, "total": 0})
            st["total"] += 1
            st["correct"] += int(ok)

        for sample in self.data:
            sid = sample.sample_id
            pred_raw = get_prediction(predictions, sid)
            extracted = self._extract_with_fallback(sample, pred_raw)
            gt = str(sample.answer).strip().strip("()").upper()
            ok = bool(extracted) and extracted == gt

            total += 1
            correct += int(ok)
            if not extracted:
                unparsed += 1

            task = getattr(sample, "task", sample.question_type)
            source = getattr(sample, "source", "")
            _bump(per_task, task, ok)
            _bump(per_source, source, ok)
            _bump(per_source_dataset, getattr(sample, "source_dataset", ""), ok)
            _bump(per_type, getattr(sample, "data_type", ""), ok)

            detailed.append({
                "sample_id": sid,
                "task": task,
                "type": getattr(sample, "data_type", ""),
                "source": source,
                "source_dataset": getattr(sample, "source_dataset", ""),
                "ground_truth": gt,
                "prediction": pred_raw,
                "extracted": extracted,
                "correct": ok,
            })

        def _table(stats: Dict[str, Dict[str, int]], order: List[str]) -> Dict[str, Any]:
            keys = [k for k in order if k in stats] + sorted(set(stats) - set(order))
            return {
                k: {
                    "accuracy": self._acc(stats[k]),
                    "correct": stats[k]["correct"],
                    "total": stats[k]["total"],
                }
                for k in keys
            }

        # Official aggregate (dataset README / Cambrian-1 App. C.3), over the
        # groups present in self.data. Absent groups are reported as None and
        # excluded from the means.
        acc_ade = self._acc(per_source["ADE20K"]) if "ADE20K" in per_source else None
        acc_coco = self._acc(per_source["COCO"]) if "COCO" in per_source else None
        acc_omni = self._acc(per_source["Omni3D"]) if "Omni3D" in per_source else None
        parts_2d = [a for a in (acc_ade, acc_coco) if a is not None]
        acc_2d = mean_or_zero(parts_2d) if parts_2d else None
        acc_3d = acc_omni
        parts_all = [a for a in (acc_2d, acc_3d) if a is not None]
        cvbench_acc = mean_or_zero(parts_all) if parts_all else None

        results: Dict[str, Any] = {
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": correct / max(total, 1),
            "overall_aggregation": (
                "overall_accuracy = micro accuracy over loaded samples; "
                "cvbench_acc = mean(acc_2d, acc_3d) with acc_2d = mean(acc_ADE20K, acc_COCO) "
                "and acc_3d = acc_Omni3D (official README formula), over groups present"
            ),
            "unparsed_predictions": unparsed,
            "cvbench_acc": cvbench_acc,
            "acc_2d": acc_2d,
            "acc_3d": acc_3d,
            "official_groups": {
                "ADE20K": {"accuracy": acc_ade, "n": per_source.get("ADE20K", {}).get("total", 0)},
                "COCO": {"accuracy": acc_coco, "n": per_source.get("COCO", {}).get("total", 0)},
                "Omni3D": {"accuracy": acc_omni, "n": per_source.get("Omni3D", {}).get("total", 0)},
            },
            "per_task": _table(per_task, TASKS),
            "per_source": _table(per_source, SOURCES),
            "per_source_dataset": _table(per_source_dataset, []),
            "per_type": _table(per_type, TYPES),
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), detailed)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        def _fmt(v: Optional[float]) -> str:
            return "   n/a " if v is None else f"{v * 100:6.2f}%"

        print(f"\n{'=' * 70}")
        print("CV-Bench (Cambrian-1, 2024) Results")
        print(f"{'=' * 70}")
        print(f"Total samples: {results['total_samples']}")
        print(f"Correct:       {results['correct_samples']}")
        print(f"Unparsed:      {results.get('unparsed_predictions', 0)}")
        print(f"Micro accuracy:          {_fmt(results['overall_accuracy'])}")
        print(f"CV-Bench acc (official): {_fmt(results.get('cvbench_acc'))}")
        print(f"  acc_2d = mean(ADE20K, COCO): {_fmt(results.get('acc_2d'))}")
        print(f"  acc_3d = Omni3D:             {_fmt(results.get('acc_3d'))}")
        print()
        for title, key in (
            ("Task", "per_task"),
            ("Source", "per_source"),
            ("Source dataset", "per_source_dataset"),
            ("Type", "per_type"),
        ):
            table = results.get(key, {})
            if not table:
                continue
            print(f"{title:<28s} {'Acc':>8s} {'Correct':>8s} {'Total':>6s}")
            print(f"{'-' * 28} {'-' * 8} {'-' * 8} {'-' * 6}")
            for name, info in table.items():
                print(
                    f"{name:<28s} {info['accuracy'] * 100:>7.2f}% "
                    f"{info['correct']:>7d} {info['total']:>5d}"
                )
            print()
        print(f"{'=' * 70}\n")
