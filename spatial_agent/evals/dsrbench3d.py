"""3DSRBench (Ma et al., ICCV 2025; arXiv 2412.07825) data loader and scorer.

3DSRBench-real: 2,100 human-annotated 3D spatial-reasoning VQAs on MS-COCO images
plus 525 test-time-augmented variants = 2,625 base questions, 12 sub-types in four
types (official grouping, compute_3dsrbench_results_circular.py):

    height       height_higher (350)
    location     location_above (350), location_closer_to_camera (350), location_next_to (175)
    orientation  orientation_in_front_of (175), orientation_on_the_left (175), orientation_viewpoint (175)
    multi_object multi_object_closer_to, _facing, _parallel, _same_direction,
                 _viewpoint_towards_object (175 each)

Every base question has (for 2,532 of the 2,625) a horizontally mirrored twin
``<index>-flip`` with the same question and options and the answer adjusted where
mirroring changes it (left/right, viewpoint). The public parquet does NOT ship the
mirrored images: this loader mirrors the COCO image on first use and caches it under
``<data_path>/flipped/<split>/<file>``. That is an input-side transform only.

Data (HF ccvl/3DSRBench, CC-BY-4.0), expected under ``data/3DSRBench``:
    3dsrbench_v1-00000-of-00001.parquet   columns index, question, A, B, C, D, answer,
                                          category, image_source, image_url
    coco_images/{train2017,val2017}/*.jpg  images named by the url basename

Answer format: a single option letter; 2 options for 2,275 base items (A/B), 4 for the
350 viewpoint items (A-D). Letters are extracted with the CV-Bench-2024 parser
(shared ``_extract_letter`` + exact option-text fallback); unparsable = wrong.

Metrics written by ``evaluate``:
    vanilla accuracy per row (all rows / base rows / flip rows), per type and sub-type;
    FlipEval accuracy per base question (correct iff the base row and its flip twin are
    both correct), overall / per type / per sub-type as the mean over base questions —
    the official aggregation without the circular option permutations (CircularEval
    is NOT run here; results are therefore not comparable to the leaderboard numbers).

Reference: https://huggingface.co/datasets/ccvl/3DSRBench
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.cvbench2024 import _extract_letter, _norm_text
from spatial_agent.evals.scoring import (
    get_prediction,
    mean_or_zero,
    write_json,
    write_results_summary,
)

PARQUET = "3dsrbench_v1-00000-of-00001.parquet"
IMAGE_DIR = "coco_images"
FLIP_DIR = "flipped"
LETTERS = "ABCD"

TYPE_OF = {
    "height_higher": "height",
    "location_above": "location",
    "location_closer_to_camera": "location",
    "location_next_to": "location",
    "orientation_in_front_of": "orientation",
    "orientation_on_the_left": "orientation",
    "orientation_viewpoint": "orientation",
    "multi_object_closer_to": "multi_object",
    "multi_object_facing": "multi_object",
    "multi_object_parallel": "multi_object",
    "multi_object_same_direction": "multi_object",
    "multi_object_viewpoint_towards_object": "multi_object",
}
TYPES = ["height", "location", "orientation", "multi_object"]
SUBTYPES = list(TYPE_OF)


def _is_none(v: Any) -> bool:
    return v is None or str(v).strip().lower() in ("", "none", "nan")


@dataclass
class DSRBench3DSample(BaseBenchmarkSample):
    """One 3DSRBench row (a base question or its ``-flip`` twin)."""

    choices: Dict[str, str] = field(default_factory=dict)   # {"A": "...", "B": "..."} (non-empty options only)
    base_id: str = ""                                        # index without the "-flip" suffix
    is_flip: bool = False
    qtype: str = ""                                          # height | location | orientation | multi_object
    subtype: str = ""                                        # official category, e.g. height_higher
    n_options: int = 2
    image_url: str = ""


class DSRBench3D(BaseBenchmark):
    """3DSRBench loader and scorer (vanilla + FlipEval, no CircularEval)."""

    data_specific_prompt = "Answer with the letter of the correct option only."

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        super().__init__(data_path, question_type)

    # ── loading ──────────────────────────────────────────────────────────

    def _local_image(self, url: str) -> str:
        parts = urlparse(url).path.strip("/").split("/")
        split, fname = (parts[-2], parts[-1]) if len(parts) >= 2 else ("train2017", parts[-1])
        cand = os.path.join(self.data_path, IMAGE_DIR, split, fname)
        if os.path.exists(cand):
            return cand
        for s in ("train2017", "val2017"):
            alt = os.path.join(self.data_path, IMAGE_DIR, s, fname)
            if os.path.exists(alt):
                return alt
        raise FileNotFoundError(f"3DSRBench image not found locally: {url} (looked under {self.data_path}/{IMAGE_DIR})")

    def _flipped_image(self, src: str) -> str:
        """Horizontally mirrored copy of ``src`` (cached; written atomically)."""
        rel = os.path.relpath(src, os.path.join(self.data_path, IMAGE_DIR))
        dst = os.path.join(self.data_path, FLIP_DIR, rel)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return dst
        from PIL import Image, ImageOps
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = f"{dst}.{os.getpid()}.tmp"
        try:
            with Image.open(src) as im:
                ImageOps.mirror(im.convert("RGB")).save(tmp, format="JPEG", quality=95)
            os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return dst

    def _wants(self, subtype: str, qtype: str) -> bool:
        if not self.question_type_filter:
            return True
        wanted = {str(q).strip().lower() for q in self.question_type_filter}
        return bool({subtype.lower(), qtype.lower()} & wanted)

    def read_data(self) -> None:
        import pyarrow.parquet as pq

        self.data_path = os.path.abspath(self.data_path)
        path = os.path.join(self.data_path, PARQUET)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"3DSRBench parquet not found: {path}. Download HF ccvl/3DSRBench "
                f"(parquet + coco_images.zip) into {self.data_path}."
            )
        rows = pq.read_table(path).to_pylist()
        n_flipped = 0
        for row in rows:
            index = str(row["index"]).strip()
            subtype = str(row.get("category") or "").strip()
            qtype = TYPE_OF.get(subtype, "")
            if not qtype:
                raise ValueError(f"3DSRBench: unknown category {subtype!r} for {index}")
            if not self._wants(subtype, qtype):
                continue
            is_flip = index.endswith("-flip")
            base_id = index[:-5] if is_flip else index
            choices = {L: str(row[L]).strip() for L in LETTERS if not _is_none(row.get(L))}
            answer = str(row.get("answer") or "").strip().upper()
            if answer not in choices:
                raise ValueError(f"3DSRBench: answer {answer!r} not among options {list(choices)} for {index}")
            src = self._local_image(str(row["image_url"]))
            if is_flip:
                img = self._flipped_image(src)
                n_flipped += 1
            else:
                img = src
            self.data.append(
                DSRBench3DSample(
                    sample_id=index,
                    question=str(row["question"]).strip(),
                    question_type=subtype,
                    images=[img],
                    answer=answer,
                    choices=choices,
                    base_id=base_id,
                    is_flip=is_flip,
                    qtype=qtype,
                    subtype=subtype,
                    n_options=len(choices),
                    image_url=str(row["image_url"]),
                )
            )
        print(f"[3DSRBench] Loaded {len(self.data)} rows ({n_flipped} mirrored twins) from {path}")

    # ── answer extraction / scoring ──────────────────────────────────────

    def extract_answer(self, prediction: str) -> str:
        return _extract_letter(prediction)

    def _extract(self, sample: BaseBenchmarkSample, prediction: str) -> str:
        letter = _extract_letter(prediction)
        choices = getattr(sample, "choices", None) or {}
        if letter:
            return letter if letter in choices or not choices else ""
        if not prediction or not choices:
            return ""
        import re
        text = str(prediction)
        cands = [_norm_text(text)]
        boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
        if boxed:
            cands.append(_norm_text(boxed[-1]))
        ms = list(re.finditer(r"(?:answer|option|choice)\s*(?:is|:|=)?\s*", text, re.IGNORECASE))
        if ms:
            cands.append(_norm_text(text[ms[-1].end():]))
        for cand in cands:
            if not cand:
                continue
            hits = [L for L, t in choices.items() if _norm_text(t) == cand]
            if len(hits) == 1:
                return hits[0]
        return ""

    def evaluate_single(self, sample: BaseBenchmarkSample, prediction: str) -> Optional[float]:
        pred = self._extract(sample, prediction)
        return 1.0 if (pred and pred == str(sample.answer).strip().upper()) else 0.0

    def evaluate(self, predictions: Dict[Any, str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        detailed = []
        per_row: Dict[str, float] = {}
        unparsed = 0
        missing = 0
        for s in self.data:
            raw = get_prediction(predictions, s.sample_id, default=None)
            if raw is None:
                missing += 1
                raw = ""
            pred = self._extract(s, raw)
            if not pred and raw:
                unparsed += 1
            correct = 1.0 if pred == s.answer else 0.0
            per_row[str(s.sample_id)] = correct
            detailed.append({
                "sample_id": s.sample_id, "base_id": s.base_id, "is_flip": s.is_flip,
                "qtype": s.qtype, "subtype": s.subtype, "n_options": s.n_options,
                "answer": s.answer, "pred": pred, "correct": correct,
            })

        def acc(items):
            return mean_or_zero(d["correct"] for d in items)

        base_rows = [d for d in detailed if not d["is_flip"]]
        flip_rows = [d for d in detailed if d["is_flip"]]
        results: Dict[str, Any] = {
            "n_rows": len(detailed), "n_base_rows": len(base_rows), "n_flip_rows": len(flip_rows),
            "unparsed_predictions": unparsed, "missing_predictions": missing,
            "vanilla_accuracy": acc(detailed),
            "vanilla_accuracy_base_rows": acc(base_rows),
            "vanilla_accuracy_flip_rows": acc(flip_rows),
            "vanilla_by_type": {t: acc([d for d in detailed if d["qtype"] == t]) for t in TYPES},
            "vanilla_by_subtype": {t: acc([d for d in detailed if d["subtype"] == t]) for t in SUBTYPES},
            "vanilla_by_n_options": {n: acc([d for d in detailed if d["n_options"] == n]) for n in (2, 4)},
        }
        # FlipEval: a base question counts as correct iff the base row and its twin are both correct
        # (official product rule; base questions without a twin in the loaded set count on their own).
        by_base: Dict[str, Dict[str, Any]] = {}
        for d in detailed:
            b = by_base.setdefault(d["base_id"], {"correct": 1.0, "qtype": d["qtype"], "subtype": d["subtype"], "n": 0})
            b["correct"] *= d["correct"]
            b["n"] += 1
        pairs = {k: v for k, v in by_base.items() if v["n"] == 2}
        results.update({
            "flipeval_n_base_questions": len(by_base),
            "flipeval_n_complete_pairs": len(pairs),
            "flipeval_accuracy": mean_or_zero(v["correct"] for v in by_base.values()),
            "flipeval_accuracy_complete_pairs": mean_or_zero(v["correct"] for v in pairs.values()),
            "flipeval_by_type": {t: mean_or_zero(v["correct"] for v in by_base.values() if v["qtype"] == t) for t in TYPES},
            "flipeval_by_subtype": {t: mean_or_zero(v["correct"] for v in by_base.values() if v["subtype"] == t) for t in SUBTYPES},
            "counts_by_subtype": {t: sum(1 for d in detailed if d["subtype"] == t) for t in SUBTYPES},
            "detailed_results": detailed,
        })
        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), detailed)
        return results
