"""SpinBench data loader.

SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs
(Zhang et al., arXiv 2509.25390). Image-only multiple-choice diagnostic
benchmark with 2,739 questions over 53 task subtypes grouped into 7 task
groups (identity matching, object-relation grounding, dynamic rotation,
dynamic translation, canonical view selection, perspective taking, mental
rotation). Every question has 1-5 interleaved images and 2, 3 or 4 letter
options; the letter set is fixed per item by the problem text.

Data structure (HF: YuyouZhang/SpinBench; unzip images.zip next to the jsonl):
    data/SpinBench/test.jsonl          (2,739 rows = official full benchmark)
    data/SpinBench/test_small.jsonl    (200-row quick-test subset, duplicates of
                                        test.jsonl rows -- NOT loaded)
    data/SpinBench/images/<file>.{png,jpg}

Row fields: problem (text with one ``<image>`` placeholder per image, in
order), answer (single capital letter), images (relative paths, in order),
metadata (task_type always present; dataset_type absent for the 120
infinigen_mental_rotation rows).

Metric: micro accuracy over all questions (official). Per-question
correctness = extracted letter == answer; unparsable answers count as wrong.
Breakdowns: per task group (7), per task_type (53), per dataset_type, per
option count.

Reference: https://github.com/ZhangYuyou-10/SpinBench
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from spatial_agent.evals.base import BaseBenchmark, BaseBenchmarkSample
from spatial_agent.evals.scoring import get_prediction, write_json, write_results_summary

_ALL_LETTERS = "ABCD"

# Fixed reporting order for the seven task groups (paper v2 Table 4).
TASK_GROUPS = [
    "identity_matching",
    "object_relation_grounding",
    "dynamic_rotation",
    "dynamic_translation",
    "canonical_view_selection",
    "perspective_taking",
    "mental_rotation",
]

# Ordered (substring, group) rules applied to metadata.task_type.
_GROUP_RULES = [
    ("_identity", "identity_matching"),
    ("spatial_relation_grounding", "object_relation_grounding"),
    ("spatial_relationship_front_behind", "object_relation_grounding"),
    ("rotation_classification", "dynamic_rotation"),
    ("spatial_relationship_dynamic", "dynamic_translation"),
    ("canonical_view_selection", "canonical_view_selection"),
    ("infinigen_rotation_selection", "perspective_taking"),
    ("spatial_relation_transformation", "perspective_taking"),
    ("mental_rotation", "mental_rotation"),
]

# The trailing "Only answer with ... (A, B, or C)." instruction line.
_INSTRUCTION_LINE_RE = re.compile(r"Only answer[^\n]*", re.IGNORECASE)
# A parenthesised letter list: (A, B) / (A, B, C, D) / (A, B, or C) / (A or B).
_LETTER_GROUP_RE = re.compile(r"\(\s*([A-D](?:\s*,?\s*(?:or\s+)?[A-D])*)\s*\)")
# Option labels at the start of a line: "A. <image>" / "A: <image>" / "A. mug".
_LINE_LABEL_RE = re.compile(r"(?m)^\s*([A-D])\s*[.:]")
# Option labels anywhere: " A. mug, B.mustard bottle".
_INLINE_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])([A-D])[.:](?=\s|<image>|[A-Za-z])")


def task_group_for(task_type: str) -> str:
    """Map an official task_type to one of the seven task groups."""
    for needle, group in _GROUP_RULES:
        if needle in task_type:
            return group
    return "other"


def parse_valid_letters(problem: str, n_images: int) -> str:
    """Return the contiguous letter set (e.g. "ABC") allowed for an item.

    Preference order: the letter list inside the "Only answer ..." instruction
    line; else option labels ("A. <image>" at line start or inline " A. mug,
    B.mustard"); else one letter per image (image-first identity tasks).
    Falls back to "ABCD".
    """
    letters: List[str] = []
    instr = _INSTRUCTION_LINE_RE.findall(problem)
    if instr:
        groups = _LETTER_GROUP_RE.findall(instr[-1])
        if groups:
            letters = re.findall(r"[A-D]", groups[-1])
    if not letters:
        letters = _LINE_LABEL_RE.findall(problem) + _INLINE_LABEL_RE.findall(problem)
    if letters:
        n = max(_ALL_LETTERS.index(ch) for ch in set(letters)) + 1
    elif 2 <= n_images <= 4:
        n = n_images
    else:
        n = 4
    return _ALL_LETTERS[:n]


def render_question(problem: str, n_images: int) -> str:
    """Replace the k-th ``<image>`` placeholder with ``[Image k]`` (1-based)."""
    parts = problem.split("<image>")
    if len(parts) - 1 != n_images:
        raise ValueError(
            f"placeholder/image mismatch: {len(parts) - 1} <image> tags vs {n_images} images"
        )
    out = parts[0]
    for k, part in enumerate(parts[1:], start=1):
        out += f"[Image {k}]" + part
    return out


@dataclass
class SpinBenchSample(BaseBenchmarkSample):
    """SpinBench sample. Options are inline in ``question``; ``choices`` stays empty."""

    choices: List[str] = field(default_factory=list)
    task_type: str = ""          # official subtype (53 values) == question_type
    group: str = ""              # one of TASK_GROUPS
    dataset_type: str = ""       # cars / faces / infinigen / abo
    valid_letters: str = ""      # e.g. "AB", "ABC", "ABCD"
    n_options: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SpinBench(BaseBenchmark):
    """SpinBench loader and scorer (micro accuracy, official letter protocol)."""

    data_specific_prompt = (
        "Answer with the single letter of the correct option."
    )

    def __init__(self, data_path: str, question_type: Optional[List[str]] = None):
        super().__init__(data_path, question_type)

    # ── loading ──────────────────────────────────────────────────────────
    def read_data(self) -> None:
        self.data_path = os.path.abspath(self.data_path)
        jsonl_path = os.path.join(self.data_path, "test.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"SpinBench test.jsonl not found at {jsonl_path}. Download "
                f"test.jsonl and images.zip from HF YuyouZhang/SpinBench and "
                f"unzip images.zip into {self.data_path}/"
            )

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for row_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                meta = dict(row.get("metadata") or {})
                task_type = str(meta.get("task_type", "")).strip()
                group = task_group_for(task_type)
                dataset_type = str(meta.get("dataset_type") or "infinigen").strip()

                if self.question_type_filter and not (
                    task_type in self.question_type_filter
                    or group in self.question_type_filter
                    or dataset_type in self.question_type_filter
                ):
                    continue

                problem = str(row["problem"])
                rel_images = list(row.get("images") or [])
                images = [
                    p if os.path.isabs(p) else os.path.join(self.data_path, p)
                    for p in rel_images
                ]
                question = render_question(problem, len(images))

                answer = str(row["answer"]).strip().upper()
                valid_letters = parse_valid_letters(problem, len(images))
                if answer not in valid_letters:
                    print(
                        f"[SpinBench][Warning] row {row_idx}: answer {answer!r} "
                        f"outside parsed letter set {valid_letters!r}; extending set"
                    )
                    valid_letters = _ALL_LETTERS[: _ALL_LETTERS.index(answer) + 1]

                self.data.append(
                    SpinBenchSample(
                        sample_id=str(row_idx),
                        question=question,
                        question_type=task_type,
                        images=images,
                        answer=answer,
                        choices=[],
                        task_type=task_type,
                        group=group,
                        dataset_type=dataset_type,
                        valid_letters=valid_letters,
                        n_options=len(valid_letters),
                        metadata=meta,
                    )
                )

        print(f"[SpinBench] Loaded {len(self.data)} samples")

    # ── answer extraction ────────────────────────────────────────────────
    # Text between two *different* letters that is made only of connectors
    # ("A or B", "A/B", "A, B", "either A or B", "(A) or (B)") marks an
    # undecided answer: it yields "" rather than crediting the last letter.
    # At least one real connector is required, so "Not A. B." (whitespace
    # only between the labels) still resolves to the last letter.
    _CONNECTOR_RE = re.compile(
        r"^[\s,;/&|()]*(?:(?:or|and|nor|vs\.?|versus|either)\b[\s,;/&|()]*)*$", re.IGNORECASE
    )
    _CONNECTOR_TOKEN_RE = re.compile(r"[,;/&|]|\b(?:or|and|nor|vs\.?|versus|either)\b", re.IGNORECASE)
    # Strong conclusion markers (letter follows): "final answer: X",
    # "the answer is X", "Answer: X", "correct option is X", "I choose X",
    # "I'd go with X". Uppercase letter anywhere; lowercase only as the
    # closing token ("answer is c.").
    _STRONG_PREFIX_RE = re.compile(
        r"(?i:final\s+answer|correct\s+(?:answer|option|choice|letter)"
        r"|best\s+(?:answer|option|choice)|(?:my|the)\s+answer|answer"
        r"|(?:i(?:'d|\s+would)?\s+)?(?:choose|select|pick|go\s+with))\b"
        r"\s*(?i:is|would\s+be|should\s+be|must\s+be)?\s*[:\-=]?\s*\**\s*\(?\s*"
        r"(?:([A-D])(?![A-Za-z0-9])|([a-d])(?=[\.\)\]]?\**\s*$))"
    )
    # Strong postfix marker (letter precedes): "X is correct", "(X) is the
    # right answer", "**X** would be the best choice".
    _STRONG_POSTFIX_RE = re.compile(
        r"(?<![A-Za-z0-9])([A-D])\)?\**\s+(?i:is|would\s+be|seems|looks|appears)\s+"
        r"(?i:the\s+)?(?i:correct|right|best|answer)\b"
    )
    # Weak markers: CoT text says "option A shows ..." while explaining, so
    # these only count when no strong marker is present.
    _WEAK_PREFIX_RE = re.compile(
        r"(?i:option|choice)\b\s*(?i:is|would\s+be)?\s*[:\-=]?\s*\**\s*\(?\s*([A-D])(?![A-Za-z0-9])"
    )
    # Labelled letter: "(X)" anywhere, or "X." / "X:" / "X)" as a closing token.
    _LABELLED_RE = re.compile(r"(?<![A-Za-z0-9])\(([A-D])\)|(?<![A-Za-z0-9])([A-D])[\.:)](?=\s|$)")
    # Standalone capital letter (not glued to other alphanumerics).
    _STANDALONE_RE = re.compile(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])")

    @classmethod
    def _letter_in(cls, text: str) -> str:
        """Letter from a short snippet (tag/boxed content): C, (C), C., **C**, c.

        A lone letter may be lowercase; inside longer text only a standalone
        capital letter counts (so "a mug" does not yield "A"). "A or B" is
        undecided and yields "".
        """
        m = re.fullmatch(r"\s*\**\(?\s*([A-Da-d])\s*[\.\):]?\**\s*\.?\s*", text)
        if m:
            return m.group(1).upper()
        picked = cls._last_committed(text, list(cls._STANDALONE_RE.finditer(text)), _ALL_LETTERS)
        return picked or ""

    @classmethod
    def _last_committed(cls, text: str, matches: List["re.Match"], valid: str) -> Optional[str]:
        """Pick the last match whose letter is in ``valid``.

        Returns "" when the last two matches carry different letters joined
        only by connectors (undecided "A or B"), None when no match has a
        letter in ``valid`` (caller falls through to the next stage).
        """
        if not matches:
            return None
        if len(matches) >= 2:
            prev, last = matches[-2], matches[-1]
            l_prev = next(g for g in prev.groups() if g)
            l_last = next(g for g in last.groups() if g)
            between = text[prev.end():last.start()]
            if (
                l_prev != l_last
                and cls._CONNECTOR_RE.match(between)
                and cls._CONNECTOR_TOKEN_RE.search(between)
            ):
                return ""
        for m in reversed(matches):
            letter = next(g for g in m.groups() if g).upper()
            if letter in valid:
                return letter
        return None

    def _extract_letter(self, prediction: str, valid_letters: str = _ALL_LETTERS) -> str:
        """Return an upper-case option letter or "".

        Explicit-format stages (``<answer>X</answer>``, ``\\boxed{X}``,
        strong markers "final answer / answer is / Answer: / correct option /
        I choose / X is correct", then weak markers "option X / choice X") are
        matched over A-D and checked against ``valid_letters``: an explicit
        letter outside the item's set is an invalid answer and yields "". The
        looser stages ("(X)", "X.", a standalone capital letter, a lone
        lowercase letter) are searched over A-D so that an undecided "C or D"
        is recognised even on a 2-option item, then restricted to
        ``valid_letters``. Last occurrence wins throughout.
        """
        if not prediction:
            return ""
        text = str(prediction).strip()
        if not text:
            return ""
        valid = "".join(ch for ch in _ALL_LETTERS if ch in valid_letters.upper()) or _ALL_LETTERS

        def _check(letter: str) -> str:
            return letter if letter in valid else ""

        # 1. <answer>X</answer> (last block; tolerate the official "<answer>X<answer>" typo)
        tagged = re.findall(r"<answer>\s*(.*?)\s*</?answer>", text, re.DOTALL | re.IGNORECASE)
        for content in reversed(tagged):
            letter = self._letter_in(content)
            if letter:
                return _check(letter)

        # 2. \boxed{X} / \boxed{(X)} / \boxed{\text{X}} (last)
        boxed = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text)
        for content in reversed(boxed):
            inner = re.sub(r"\\(?:text|mathrm|textbf)\{([^{}]*)\}", r"\1", content)
            letter = self._letter_in(inner)
            if letter:
                return _check(letter)

        # 3a. Strong conclusion markers (prefix or postfix form); last by position.
        strong: List[Any] = []
        for m in self._STRONG_PREFIX_RE.finditer(text):
            strong.append((m.start(1) if m.group(1) else m.start(2), (m.group(1) or m.group(2)).upper()))
        for m in self._STRONG_POSTFIX_RE.finditer(text):
            strong.append((m.start(1), m.group(1)))
        if strong:
            strong.sort(key=lambda t: t[0])
            return _check(strong[-1][1])

        # 3b. Weak markers "option X" / "choice X" (last)
        weak = list(self._WEAK_PREFIX_RE.finditer(text))
        if weak:
            return _check(weak[-1].group(1))

        # 4. Labelled letter: "(X)" anywhere, or "X." / "X:" / "X)" as a closing token
        picked = self._last_committed(text, list(self._LABELLED_RE.finditer(text)), valid)
        if picked is not None:
            return picked

        # 5. Standalone capital letter
        picked = self._last_committed(text, list(self._STANDALONE_RE.finditer(text)), valid)
        if picked is not None:
            return picked

        # 6. The whole text is a single (possibly lowercase) letter
        clean = re.sub(r"[^A-Za-z]", "", text)
        if len(clean) == 1 and clean.upper() in valid:
            return clean.upper()

        return ""

    def extract_answer(self, prediction: str) -> str:
        """Extract an option letter (A-D) from free text; "" if none."""
        return self._extract_letter(prediction, _ALL_LETTERS)

    # ── scoring ──────────────────────────────────────────────────────────
    def evaluate_single(self, sample: BaseBenchmarkSample, prediction: str) -> Optional[float]:
        valid = getattr(sample, "valid_letters", "") or _ALL_LETTERS
        pred = self._extract_letter(prediction, valid)
        gt = str(sample.answer).strip().upper()
        return 1.0 if pred and pred == gt else 0.0

    def evaluate(
        self, predictions: Dict[Any, str], output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Micro accuracy over self.data plus group/subtype/source/option-count tables."""
        per_group: Dict[str, Dict[str, int]] = {}
        per_task: Dict[str, Dict[str, int]] = {}
        per_source: Dict[str, Dict[str, int]] = {}
        per_nopt: Dict[str, Dict[str, int]] = {}
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
            extracted = self._extract_letter(pred_raw, valid)
            score = self.evaluate_single(sample, pred_raw)
            score = 0.0 if score is None else float(score)
            ok = score >= 1.0
            if not extracted:
                unparsable += 1
            n_opt = getattr(sample, "n_options", 0) or len(valid)
            chance_sum += 1.0 / max(n_opt, 1)

            _bump(per_group, getattr(sample, "group", "other"), ok)
            _bump(per_task, sample.question_type, ok)
            _bump(per_source, getattr(sample, "dataset_type", ""), ok)
            _bump(per_nopt, str(n_opt), ok)

            detailed.append({
                "sample_id": sample.sample_id,
                "question_type": sample.question_type,
                "group": getattr(sample, "group", ""),
                "dataset_type": getattr(sample, "dataset_type", ""),
                "n_options": n_opt,
                "valid_letters": valid,
                "ground_truth": sample.answer,
                "prediction": pred_raw,
                "extracted": extracted,
                "score": score,
                "correct": ok,
            })

        total = len(detailed)
        correct = sum(int(d["correct"]) for d in detailed)

        def _table(counts: Dict[str, Dict[str, int]], order: List[str]) -> Dict[str, Dict[str, Any]]:
            keys = [k for k in order if k in counts] + sorted(set(counts) - set(order))
            return {
                k: {
                    "accuracy": counts[k]["correct"] / max(counts[k]["total"], 1),
                    "correct": counts[k]["correct"],
                    "total": counts[k]["total"],
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
            "per_group": _table(per_group, TASK_GROUPS),
            "per_task_type": _table(per_task, []),
            "per_dataset_type": _table(per_source, ["infinigen", "abo", "faces", "cars"]),
            "per_n_options": _table(per_nopt, ["2", "3", "4"]),
            "detailed_results": detailed,
        }

        if output_dir:
            write_results_summary(output_dir, results)
            write_json(os.path.join(output_dir, "results_details.json"), detailed)

        self.pretty_print_results(results)
        return results

    def pretty_print_results(self, results: Dict[str, Any]) -> None:
        print(f"\n{'='*72}")
        print("SpinBench Results")
        print(f"{'='*72}")
        print(
            f"Total: {results['total_samples']}  Correct: {results['correct_samples']}  "
            f"Accuracy (micro): {results['overall_accuracy']*100:.2f}%  "
            f"(chance {results.get('chance_accuracy', 0)*100:.1f}%, "
            f"unparsable {results.get('unparsable_samples', 0)})"
        )
        for title, key in (
            ("Task group", "per_group"),
            ("Data source", "per_dataset_type"),
            ("Option count", "per_n_options"),
            ("Task type", "per_task_type"),
        ):
            table = results.get(key, {})
            if not table:
                continue
            print(f"\n{title:<56s} {'Acc':>8s} {'Correct':>8s} {'Total':>6s}")
            print(f"{'-'*56} {'-'*8} {'-'*8} {'-'*6}")
            for name, s in table.items():
                print(
                    f"{name:<56s} {s['accuracy']*100:>7.2f}% "
                    f"{s['correct']:>8d} {s['total']:>6d}"
                )
        print(f"{'='*72}\n")
