"""FastAPI app for browsing spatial agent work_dir results."""

import datetime
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from jinja2 import BaseLoader, Environment

from . import templates
from .category_maps import get_category_map, is_numerical_category, clear_cache as clear_category_cache, get_spatialtree_meta

# ---------------------------------------------------------------------------
# Jinja2 setup — load templates from Python strings
# ---------------------------------------------------------------------------

class StringLoader(BaseLoader):
    """Load Jinja2 templates from the templates module."""
    TEMPLATES = {
        "base.html": templates.BASE_LAYOUT,
        "dashboard.html": templates.DASHBOARD_PAGE,
        "benchmark.html": templates.BENCHMARK_PAGE,
        "experiment.html": templates.EXPERIMENT_PAGE,
        "sample.html": templates.SAMPLE_DETAIL_PAGE,
        "compare.html": templates.COMPARE_PAGE,
        "compare_sample.html": templates.COMPARE_SAMPLE_PAGE,
        "models.html": templates.MODELS_PAGE,
        "model_detail.html": templates.MODEL_DETAIL_PAGE,
    }

    def get_source(self, environment, template):
        source = self.TEMPLATES.get(template)
        if source is None:
            raise Exception(f"Template {template!r} not found")
        return source, template, lambda: True


jinja_env = Environment(loader=StringLoader(), autoescape=True)
METHOD_DISPLAY_NAMES = {"cot": "CoT", "spatial": "Agent", "other": "Other"}
jinja_env.globals["method_label"] = lambda m: METHOD_DISPLAY_NAMES.get(m, m)

from urllib.parse import quote as _url_quote
jinja_env.globals["url_encode"] = lambda s: _url_quote(s, safe="")


def _fmt_int(n) -> str:
    if n is None:
        return "—"
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


jinja_env.filters["fmt_int"] = _fmt_int


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMeta:
    dir_name: str
    dir_path: str
    method: str
    benchmark: str
    model: str
    tools: list[str]
    num_sessions: int
    num_predictions: int
    accuracy: Optional[float]
    correct_samples: Optional[int]
    total_samples: Optional[int]
    config: dict
    created_date: str  # YYYY-MM-DD from config.json mtime


def extract_answer(text: str) -> str:
    """Extract a structured answer from raw prediction text.

    Mirrors evals/base.py logic: handles \\boxed{X}, (A)/(A), single letter,
    and parenthesized ground truths like '(D)'.
    """
    if not text:
        return ""
    text = text.strip()

    # Strategy 1: \boxed{X}
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()

    # Strategy 2: "Final Answer:" or "answer is" followed by a letter (near end of text)
    # Search from the end to prefer the final answer in CoT responses
    m = re.search(r"(?:final\s+answer|the\s+answer\s+is)[:\s]*\(?([A-Za-z])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Strategy 3: (A), A., A: — but only if the text is short (not CoT)
    # For long texts, this would match random letters in the reasoning
    if len(text) <= 20:
        m = re.search(r"\(?([A-Za-z])\)?[\.\:\)]", text)
        if m:
            return m.group(1).upper()

    # Strategy 4: single letter (after stripping non-alpha)
    clean = re.sub(r"[^A-Za-z]", "", text)
    if len(clean) == 1:
        return clean.upper()

    # Strategy 5: for short texts, return as-is uppercased
    if len(text) <= 10:
        return text.upper()

    return text


def _normalize_text(text: str) -> str:
    """Normalize whitespace for sequence/text comparison."""
    return re.sub(r"\s+", " ", text.strip()).rstrip(".")


# Benchmarks where answers are sequences/free-text (not MC letters).
# For these, use normalized exact-match instead of extract_answer.
_EXACT_MATCH_BENCHMARKS = {"dsr-bench"}

# Benchmarks using tolerance-based numeric evaluation (not MC or exact-match).
# SpatialTree metrics that can be scored without external tools (LLM judge, etc.)
_SPATIALTREE_SCOREABLE_METRICS = {"multichoiceeval", "meanrelativeacc", "judge"}


def _spatialtree_is_correct(content: str, ground_truth: str) -> bool:
    """SpatialTree correctness check using GT-format heuristics.

    Infers metric type from ground_truth format:
    - Single letter A-D → MC letter extraction
    - "yes"/"no" → judge
    - Numeric → MRA threshold (>=0.5)
    - JSON/base64 → always False (unscoreable without external tools)
    - Otherwise → normalized string match
    """
    if not content or not content.strip():
        return False
    gt = ground_truth.strip()

    # Judge: yes/no
    if gt.lower() in ("yes", "no"):
        pred_lower = content.strip().lower()
        last_yes = pred_lower.rfind("yes")
        last_no = pred_lower.rfind("no")
        if last_yes > last_no:
            return gt.lower() == "yes"
        elif last_no > last_yes:
            return gt.lower() == "no"
        return False

    # MC: single letter A-D (or parenthesized)
    gt_clean = re.sub(r"[^A-Za-z]", "", gt)
    if len(gt_clean) == 1 and gt_clean.isalpha():
        pred = extract_answer(content)
        return pred == gt_clean.upper()

    # Skip JSON (manipulateeval, agenticnaveval, gravityeval, cogmapeval)
    if gt.startswith("{") or gt.startswith("["):
        return False

    # Skip base64 (affmask)
    if len(gt) > 50 and not any(c in gt for c in " .,;:!?"):
        return False

    # Numeric: try float
    gt_f = _extract_number(gt)
    if gt_f is not None:
        pred_f = _extract_number(content)
        if pred_f is not None and gt_f != 0:
            return _mean_relative_accuracy(pred_f, gt_f) >= 0.5
        return False

    # Fallback: normalized string match
    return _normalize_text(content) == _normalize_text(gt)


def _spatialtree_is_scoreable(ground_truth: str) -> bool:
    """Check if a SpatialTree sample can be scored without external tools."""
    gt = ground_truth.strip()
    # Judge
    if gt.lower() in ("yes", "no"):
        return True
    # MC letter
    gt_clean = re.sub(r"[^A-Za-z]", "", gt)
    if len(gt_clean) == 1 and gt_clean.isalpha():
        return True
    # Skip JSON answers (manipulateeval, agenticnaveval, gravityeval, cogmapeval)
    if gt.startswith("{") or gt.startswith("["):
        return False
    # Skip base64 answers (affmask) — long alphanumeric strings
    if len(gt) > 50 and not any(c in gt for c in " .,;:!?"):
        return False
    # Numeric (only for simple number strings)
    if _extract_number(gt) is not None:
        return True
    # Everything else (gpteval text, etc.)
    return False


def _is_correct(content: str, ground_truth: str, benchmark: str = "") -> bool:
    """Check if prediction matches ground truth.

    Uses extract_answer for MC benchmarks, normalized string match for
    sequence/text benchmarks, tolerance-based for STRIDE-QA.
    """
    if not content or not content.strip():
        return False
    if benchmark == "spatialtree":
        return _spatialtree_is_correct(content, ground_truth)
    if benchmark in _EXACT_MATCH_BENCHMARKS:
        return _normalize_text(content) == _normalize_text(ground_truth)
    pred = extract_answer(content)
    gt = extract_answer(ground_truth)
    return bool(pred) and pred == gt


def _extract_display(content: str, ground_truth: str, benchmark: str = "") -> tuple[str, str]:
    """Extract display-friendly prediction and ground truth.

    Returns (extracted_pred, extracted_gt).
    """
    if benchmark in _EXACT_MATCH_BENCHMARKS or benchmark in _TOLERANCE_BENCHMARKS or benchmark == "spatialtree":
        # Show raw text for non-MC benchmarks
        pred = content.strip() if content else ""
        gt = ground_truth.strip() if ground_truth else ""
        return pred, gt
    return extract_answer(content), extract_answer(ground_truth)


def _extract_number(text: str) -> Optional[float]:
    """Extract a numerical answer from prediction text."""
    if not text:
        return None
    text = text.strip()
    # Try direct float parse
    try:
        return float(text)
    except ValueError:
        pass
    # Look for \boxed{number}
    m = re.search(r"\\boxed\{([+-]?\d+\.?\d*)\}", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Look for "answer is NUMBER" or "Final Answer: NUMBER"
    m = re.search(r"(?:final\s+answer|the\s+answer\s+is)[:\s]*([+-]?\d+\.?\d*)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # First number in text
    m = re.search(r"[+-]?\d+\.?\d*", text)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            pass
    return None


def _mean_relative_accuracy(
    pred: float, target: float,
    start: float = 0.5, end: float = 0.95, interval: float = 0.05,
    thresholds: Optional[list[float]] = None,
) -> float:
    """MRA: fraction of threshold levels passed (matches official implementation)."""
    if target == 0:
        return 0.0
    if thresholds is None:
        num_pts = int((end - start) / interval + 2)
        # Use manual linspace to avoid numpy dependency
        thresholds = [start + i * (end - start) / (num_pts - 1) for i in range(num_pts)]
    rel_err = abs(pred - target) / abs(target)
    passed = sum(1 for t in thresholds if rel_err <= 1 - t)
    return passed / len(thresholds)



def _detect_method(dir_name: str, config: dict) -> str:
    """Infer method from dir name or config."""
    dn = dir_name.lower()
    if dn.startswith("cot_") or "cot" in config.get("work_dir", "").lower().split("/")[-1].split("_"):
        return "cot"
    # Check if it looks like a spatial agent run (has tools or kernel-related config)
    if config.get("max_steps") or config.get("tools_to_use"):
        return "spatial"
    return "other"


def _extract_model(config: dict) -> str:
    """Extract model name from config."""
    return config.get("llm_model", "") or "unknown"


def _extract_benchmark(config: dict) -> str:
    """Extract benchmark from config."""
    return config.get("benchmark", "unknown")


def _count_sessions(dir_path: str) -> int:
    """Count session-* directories."""
    count = 0
    try:
        for entry in os.scandir(dir_path):
            if entry.is_dir() and entry.name.startswith("session-"):
                count += 1
    except OSError:
        pass
    return count


def _count_predictions(dir_path: str) -> int:
    """Count lines in predictions.jsonl."""
    pred_file = os.path.join(dir_path, "predictions.jsonl")
    if not os.path.isfile(pred_file):
        return 0
    count = 0
    try:
        with open(pred_file) as f:
            for line in f:
                if line.strip():
                    count += 1
    except OSError:
        pass
    return count


def _load_results_summary(dir_path: str) -> Optional[dict]:
    """Load results_summary.json if it exists."""
    path = os.path.join(dir_path, "results_summary.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_config(dir_path: str) -> Optional[dict]:
    """Load config.json."""
    path = os.path.join(dir_path, "config.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _pred_result(p: dict) -> Optional[float]:
    """Get pre-computed result score from prediction entry, or None if absent."""
    r = p.get("result")
    if r is None:
        return None
    try:
        return float(r)
    except (ValueError, TypeError):
        return 1.0 if r else 0.0


def _pred_is_correct(p: dict, benchmark: str, category: str = "") -> bool:
    """Check if a single prediction is correct, using pre-computed result if available.

    For MC/binary: result >= 1.0 means correct.
    Falls back to _is_correct when no result field.
    """
    r = _pred_result(p)
    if r is not None:
        return r >= 1.0
    return _is_correct(p.get("content", ""), p.get("ground_truth", ""), benchmark)


def _compute_blended_accuracy(
    preds: list[dict], benchmark: str, data_root: str
) -> tuple[Optional[float], Optional[int], int]:
    """Compute blended accuracy: MC accuracy for MC samples, MRA for numerical.

    Returns (accuracy, correct_count, total).
    correct_count is only for MC samples (None-ish for mixed).
    """
    if not preds:
        return None, None, 0

    cat_map = get_category_map(benchmark, data_root)

    mc_correct = 0
    mc_total = 0
    num_scores = []
    skipped = 0
    for p in preds:
        sid = str(p.get("sample_id", ""))
        category = cat_map.get(sid, "")
        content = p.get("content", "")
        gt = p.get("ground_truth", "")

        # SpatialTree: skip samples with unscoreable metrics (gpteval, cogmapeval, etc.)
        if benchmark == "spatialtree" and not _spatialtree_is_scoreable(gt):
            skipped += 1
            continue

        if is_numerical_category(benchmark, category):
            # Use pre-computed score if available (already MRA)
            pre = _pred_result(p)
            if pre is not None:
                num_scores.append(pre)
            else:
                pred_f = _extract_number(content)
                gt_f = _extract_number(gt)
                if pred_f is not None and gt_f is not None and gt_f != 0:
                    num_scores.append(_mean_relative_accuracy(pred_f, gt_f))
                else:
                    num_scores.append(0.0)
        else:
            mc_total += 1
            if _pred_is_correct(p, benchmark, category):
                mc_correct += 1

    total = len(preds) - skipped
    if not num_scores:
        # Pure MC benchmark
        return (mc_correct / total if total > 0 else None), mc_correct, total

    # Blended: weighted average of MC accuracy and mean MRA
    mc_acc = mc_correct / mc_total if mc_total > 0 else 0
    num_acc = sum(num_scores) / len(num_scores) if num_scores else 0
    blended = (mc_acc * mc_total + num_acc * len(num_scores)) / total if total > 0 else None
    correct_equiv = mc_correct  # for display, only MC correct is meaningful
    return blended, correct_equiv, total


def scan_work_dir(work_dir: str, data_root: str = "data") -> list[ExperimentMeta]:
    """Scan work_dir for experiment directories."""
    experiments = []
    if not os.path.isdir(work_dir):
        return experiments

    for entry in sorted(os.scandir(work_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        config = _load_config(entry.path)
        if config is None:
            continue

        results = _load_results_summary(entry.path)
        accuracy = results.get("overall_accuracy") if results else None
        correct = results.get("correct_samples") if results else None
        total = results.get("total_samples") if results else None

        # If no results_summary, compute from predictions
        benchmark = _extract_benchmark(config)
        if accuracy is None:
            preds = _load_predictions_cached(entry.path)
            if preds:
                accuracy, correct, total = _compute_blended_accuracy(
                    preds, benchmark, data_root
                )

        # Get creation date from config.json mtime
        config_path = os.path.join(entry.path, "config.json")
        try:
            mtime = os.path.getmtime(config_path)
            created_date = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        except OSError:
            created_date = ""

        exp = ExperimentMeta(
            dir_name=entry.name,
            dir_path=entry.path,
            method=_detect_method(entry.name, config),
            benchmark=_extract_benchmark(config),
            model=_extract_model(config),
            tools=config.get("tools_to_use", []) or [],
            num_sessions=_count_sessions(entry.path),
            num_predictions=_count_predictions(entry.path),
            accuracy=accuracy,
            correct_samples=correct,
            total_samples=total,
            config=config,
            created_date=created_date,
        )
        experiments.append(exp)
    return experiments


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

def _load_experiment_token_usage(dir_path: str) -> dict:
    """Aggregate per-session token usage from predictions.jsonl.

    Each prediction row carries a ``usage`` dict (written by entrypoints/run.py)
    with the same shape as the ``session_usage`` trace event. Sourcing from
    predictions.jsonl reuses the existing cache and avoids scanning every
    session's trace.jsonl, which is large because it contains the full
    per-step LLM/tool log.
    """
    sessions = 0
    num_calls = 0
    prompt = 0
    completion = 0
    reasoning = 0
    max_prompt = 0
    max_completion = 0
    for pred in _load_predictions_cached(dir_path):
        usage = pred.get("usage")
        if not usage:
            continue
        sessions += 1
        num_calls += usage.get("num_calls", 0) or 0
        prompt += usage.get("total_prompt_tokens", 0) or 0
        completion += usage.get("total_completion_tokens", 0) or 0
        reasoning += usage.get("total_reasoning_tokens", 0) or 0
        mp = usage.get("max_prompt_tokens", 0) or 0
        mc = usage.get("max_completion_tokens", 0) or 0
        if mp > max_prompt:
            max_prompt = mp
        if mc > max_completion:
            max_completion = mc
    if sessions == 0:
        return {}
    total = prompt + completion
    return {
        "sessions": sessions,
        "num_calls": num_calls,
        "total_prompt_tokens": prompt,
        "total_completion_tokens": completion,
        "total_reasoning_tokens": reasoning,
        "total_tokens": total,
        "max_prompt_tokens": max_prompt,
        "max_completion_tokens": max_completion,
        "avg_prompt_per_session": prompt / sessions,
        "avg_completion_per_session": completion / sessions,
        "avg_total_per_session": total / sessions,
        "avg_calls_per_session": num_calls / sessions,
    }


@lru_cache(maxsize=256)
def _load_predictions_cached(dir_path: str) -> list[dict]:
    """Load and cache predictions.jsonl."""
    pred_file = os.path.join(dir_path, "predictions.jsonl")
    if not os.path.isfile(pred_file):
        return []
    predictions = []
    try:
        with open(pred_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        predictions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return predictions


def _extract_breakdowns(results: Optional[dict]) -> list[dict]:
    """Extract all breakdown sections from results_summary.json.

    Returns a list of {title, rows} where each row is {name, correct, total, accuracy}
    or {name, score} depending on the data shape. Handles the many different formats
    used by different benchmarks.
    """
    if not results:
        return []

    SKIP_KEYS = {"total_samples", "correct_samples", "overall_accuracy",
                 "overall_accuracy_pct", "overall_score"}
    breakdowns = []

    for key, value in results.items():
        if key in SKIP_KEYS or not isinstance(value, dict):
            continue

        # Distinguish flat stats dicts ({total, correct, accuracy}) from
        # per-category dicts ({cat_name: {correct, total, accuracy}}).
        first_val = next(iter(value.values()), None) if value else None

        if first_val is None:
            continue

        # Case 1: nested dict — per-category breakdown
        if isinstance(first_val, dict):
            rows = []
            for cat_name, cat_data in value.items():
                # Normalize field names
                correct = cat_data.get("correct", cat_data.get("correct_samples"))
                total = cat_data.get("total", cat_data.get("total_samples"))
                accuracy = cat_data.get("accuracy")
                if accuracy is None and correct is not None and total:
                    accuracy = correct / total
                if accuracy is not None:
                    rows.append({
                        "name": cat_name,
                        "correct": correct,
                        "total": total,
                        "accuracy": accuracy,
                    })
            if rows:
                title = key.replace("_", " ").title()
                breakdowns.append({"title": title, "rows": rows})

        # Case 2: flat stats dict like {total: N, correct: N, accuracy: F}.
        elif isinstance(first_val, (int, float)) and ("accuracy" in value or "correct" in value):
            correct = value.get("correct", value.get("correct_samples"))
            total = value.get("total", value.get("total_samples"))
            accuracy = value.get("accuracy")
            if accuracy is None and correct is not None and total:
                accuracy = correct / total
            if accuracy is not None:
                title = key.replace("_", " ").title()
                breakdowns.append({"title": title, "rows": [{
                    "name": key,
                    "correct": correct,
                    "total": total,
                    "accuracy": accuracy,
                }]})

        # Case 3: scalar value — score table (e.g., sparbench level_scores)
        elif isinstance(first_val, (int, float)):
            rows = []
            for cat_name, score in value.items():
                if isinstance(score, (int, float)):
                    rows.append({
                        "name": cat_name,
                        "correct": None,
                        "total": None,
                        "accuracy": score / 100 if score > 1 else score,
                    })
            if rows:
                title = key.replace("_", " ").title()
                breakdowns.append({"title": title, "rows": rows})

    return breakdowns


def _compute_category_breakdowns(
    predictions: list[dict], benchmark: str, data_root: str
) -> list[dict]:
    """Compute per-category accuracy/MRA from predictions using category_maps.

    Returns list of breakdown dicts compatible with _extract_breakdowns output.
    Uses MRA for numerical categories, accuracy for MC categories.
    """
    cat_map = get_category_map(benchmark, data_root)
    if not cat_map:
        return []

    # Accumulate per-category stats
    mc_stats: dict[str, dict] = {}  # category -> {correct, total}
    num_stats: dict[str, list] = {}  # category -> [mra_scores]
    unscored_stats: dict[str, int] = {}  # category -> count (for unscoreable)
    for p in predictions:
        sid = str(p.get("sample_id", ""))
        category = cat_map.get(sid)
        if category is None:
            continue

        gt = p.get("ground_truth", "")

        # SpatialTree: skip unscoreable samples but track count
        if benchmark == "spatialtree" and not _spatialtree_is_scoreable(gt):
            unscored_stats[category] = unscored_stats.get(category, 0) + 1
            continue

        if is_numerical_category(benchmark, category):
            if category not in num_stats:
                num_stats[category] = []
            pre = _pred_result(p)
            if pre is not None:
                num_stats[category].append(pre)
            else:
                pred_f = _extract_number(p.get("content", ""))
                gt_f = _extract_number(gt)
                if pred_f is not None and gt_f is not None and gt_f != 0:
                    num_stats[category].append(_mean_relative_accuracy(pred_f, gt_f))
                else:
                    num_stats[category].append(0.0)
        else:
            if category not in mc_stats:
                mc_stats[category] = {"correct": 0, "total": 0}
            mc_stats[category]["total"] += 1
            if _pred_is_correct(p, benchmark, category):
                mc_stats[category]["correct"] += 1

    if not mc_stats and not num_stats:
        return []

    rows = []
    for cat_name, s in sorted(mc_stats.items()):
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        rows.append({
            "name": cat_name,
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": acc,
            "metric": "accuracy",
        })
    for cat_name, scores in sorted(num_stats.items()):
        avg_mra = sum(scores) / len(scores) if scores else 0
        rows.append({
            "name": f"{cat_name} (MRA)",
            "correct": None,
            "total": len(scores),
            "accuracy": avg_mra,
            "metric": "mra",
        })
    for cat_name, count in sorted(unscored_stats.items()):
        rows.append({
            "name": f"{cat_name} (unscored)",
            "correct": None,
            "total": count,
            "accuracy": None,
            "metric": "unscored",
        })

    return [{"title": "Per Category", "rows": rows}]


def _list_session_ids(dir_path: str) -> list[str]:
    """List all session-* directory IDs, sorted numerically."""
    ids = []
    try:
        for entry in os.scandir(dir_path):
            if entry.is_dir() and entry.name.startswith("session-"):
                sid = entry.name[len("session-"):]
                ids.append(sid)
    except OSError:
        pass
    # Sort numerically if possible
    def sort_key(s):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)
    return sorted(ids, key=sort_key)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def create_app(work_dir: str, data_root: str = "data") -> FastAPI:
    app = FastAPI(title="Spatial Agent Results Viewer")

    # State
    state = {"experiments": scan_work_dir(work_dir, data_root), "work_dir": work_dir, "data_root": data_root}

    def _get_exp(dir_name: str) -> ExperimentMeta:
        for exp in state["experiments"]:
            if exp.dir_name == dir_name:
                return exp
        raise HTTPException(status_code=404, detail=f"Experiment {dir_name!r} not found")

    # --- HTML pages ---

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        experiments = state["experiments"]
        # Build benchmark summary
        benchmarks = {}
        for exp in experiments:
            bname = exp.benchmark
            if bname not in benchmarks:
                benchmarks[bname] = {"count": 0, "best_acc": None, "best_exp": None}
            benchmarks[bname]["count"] += 1
            if exp.accuracy is not None:
                if benchmarks[bname]["best_acc"] is None or exp.accuracy > benchmarks[bname]["best_acc"]:
                    benchmarks[bname]["best_acc"] = exp.accuracy
                    benchmarks[bname]["best_exp"] = exp.dir_name

        # Build model summary
        models_map: dict[str, dict] = {}
        for exp in experiments:
            mname = exp.model or "Unknown"
            if mname not in models_map:
                models_map[mname] = {
                    "name": mname, "count": 0, "benchmarks": set(),
                    "best_acc": None, "acc_sum": 0.0, "acc_count": 0,
                }
            m = models_map[mname]
            m["count"] += 1
            m["benchmarks"].add(exp.benchmark)
            if exp.accuracy is not None:
                m["acc_sum"] += exp.accuracy
                m["acc_count"] += 1
                if m["best_acc"] is None or exp.accuracy > m["best_acc"]:
                    m["best_acc"] = exp.accuracy
        model_list = []
        for m in models_map.values():
            m["benchmarks"] = sorted(m["benchmarks"])
            m["avg_acc"] = m["acc_sum"] / m["acc_count"] if m["acc_count"] > 0 else None
            model_list.append(m)
        model_list.sort(key=lambda x: x["best_acc"] if x["best_acc"] is not None else -1, reverse=True)

        template = jinja_env.get_template("dashboard.html")
        return template.render(experiments=experiments, benchmarks=benchmarks, models=model_list)

    @app.get("/models", response_class=HTMLResponse)
    async def models_page():
        experiments = state["experiments"]
        models: dict[str, dict] = {}
        for exp in experiments:
            mname = exp.model or "Unknown"
            if mname not in models:
                models[mname] = {
                    "name": mname,
                    "count": 0,
                    "benchmarks": set(),
                    "best_acc": None,
                    "acc_sum": 0.0,
                    "acc_count": 0,
                }
            m = models[mname]
            m["count"] += 1
            m["benchmarks"].add(exp.benchmark)
            if exp.accuracy is not None:
                m["acc_sum"] += exp.accuracy
                m["acc_count"] += 1
                if m["best_acc"] is None or exp.accuracy > m["best_acc"]:
                    m["best_acc"] = exp.accuracy

        model_list = []
        for m in models.values():
            m["benchmarks"] = sorted(m["benchmarks"])
            m["avg_acc"] = m["acc_sum"] / m["acc_count"] if m["acc_count"] > 0 else None
            model_list.append(m)
        model_list.sort(key=lambda x: x["best_acc"] if x["best_acc"] is not None else -1, reverse=True)

        template = jinja_env.get_template("models.html")
        return template.render(models=model_list, total_experiments=len(experiments))

    @app.get("/model/{model_name:path}", response_class=HTMLResponse)
    async def model_detail(model_name: str):
        experiments = [e for e in state["experiments"] if (e.model or "Unknown") == model_name]
        if not experiments:
            raise HTTPException(status_code=404, detail=f"No experiments for model {model_name!r}")

        by_benchmark: dict[str, list] = {}
        for exp in experiments:
            by_benchmark.setdefault(exp.benchmark, []).append(exp)

        chart_data = []
        for bname, exps in sorted(by_benchmark.items()):
            best = max((e.accuracy for e in exps if e.accuracy is not None), default=None)
            if best is not None:
                chart_data.append({"benchmark": bname, "accuracy": best})

        benchmark_groups = [
            {"name": bname, "experiments": exps}
            for bname, exps in sorted(by_benchmark.items())
        ]

        token_usage_by_exp = {
            e.dir_name: _load_experiment_token_usage(e.dir_path) for e in experiments
        }

        template = jinja_env.get_template("model_detail.html")
        return template.render(
            model_name=model_name,
            experiments=experiments,
            benchmark_groups=benchmark_groups,
            chart_data=chart_data,
            token_usage_by_exp=token_usage_by_exp,
        )

    @app.get("/benchmark/{benchmark_name}", response_class=HTMLResponse)
    async def benchmark_view(benchmark_name: str):
        experiments = [e for e in state["experiments"] if e.benchmark == benchmark_name]
        if not experiments:
            raise HTTPException(status_code=404, detail=f"No experiments for benchmark {benchmark_name!r}")

        # Chart data
        chart_data = []
        for exp in experiments:
            if exp.accuracy is not None:
                chart_data.append({"name": exp.dir_name, "accuracy": exp.accuracy})
        chart_data.sort(key=lambda x: x["accuracy"], reverse=True)

        # Per-category data for experiments
        category_data = []
        for exp in experiments:
            results = _load_results_summary(exp.dir_path)
            breakdowns = _extract_breakdowns(results)
            if not breakdowns:
                preds = _load_predictions_cached(exp.dir_path)
                if preds:
                    breakdowns = _compute_category_breakdowns(
                        preds, exp.benchmark, state["data_root"]
                    )
            if breakdowns:
                bd = breakdowns[0]
                categories = {
                    row["name"]: {"accuracy": row["accuracy"]}
                    for row in bd["rows"]
                }
                category_data.append({
                    "name": exp.dir_name,
                    "categories": categories,
                })

        token_usage_by_exp = {
            e.dir_name: _load_experiment_token_usage(e.dir_path) for e in experiments
        }

        template = jinja_env.get_template("benchmark.html")
        return template.render(
            benchmark_name=benchmark_name,
            experiments=experiments,
            chart_data=chart_data,
            category_data=category_data if len(category_data) >= 1 else None,
            token_usage_by_exp=token_usage_by_exp,
        )

    @app.get("/experiment/{dir_name}", response_class=HTMLResponse)
    async def experiment_view(dir_name: str):
        exp = _get_exp(dir_name)
        predictions = _load_predictions_cached(exp.dir_path)
        session_ids = set(_list_session_ids(exp.dir_path))

        # Get category map for numerical detection
        cat_map = get_category_map(exp.benchmark, state["data_root"])

        # Annotate predictions with session availability and extracted answers
        annotated = []
        for p in predictions:
            sid = str(p.get("sample_id", ""))
            content = p.get("content", "")
            gt = p.get("ground_truth", "")
            category = cat_map.get(sid, "")
            is_numerical = is_numerical_category(exp.benchmark, category)

            # SpatialTree: mark unscoreable samples
            is_unscoreable = (exp.benchmark == "spatialtree" and not _spatialtree_is_scoreable(gt))

            if is_numerical:
                # Use pre-computed MRA if available
                pre = _pred_result(p)
                if pre is not None:
                    mra = pre
                else:
                    pred_f = _extract_number(content)
                    gt_f = _extract_number(gt)
                    if pred_f is not None and gt_f is not None and gt_f != 0:
                        mra = _mean_relative_accuracy(pred_f, gt_f)
                    else:
                        mra = 0.0
                pred_f = _extract_number(content)
                gt_f = _extract_number(gt)
                extracted_pred = f"{pred_f:.4g}" if pred_f is not None else ""
                extracted_gt = f"{gt_f:.4g}" if gt_f is not None else gt
            else:
                mra = None
                extracted_pred, extracted_gt = _extract_display(content, gt, exp.benchmark)

            # CoT: raw content differs from extracted (i.e., content has reasoning)
            has_cot = bool(content) and content.strip() != extracted_pred
            annotated.append({
                **p,
                "has_session": sid in session_ids,
                "extracted_pred": extracted_pred,
                "extracted_gt": extracted_gt,
                "is_correct": (not is_numerical and not is_unscoreable) and _pred_is_correct(p, exp.benchmark, category),
                "is_empty": not extracted_pred,
                "has_cot": has_cot,
                "is_numerical": is_numerical,
                "is_unscoreable": is_unscoreable,
                "mra_score": mra,
                "category": category,
            })

        # Results breakdown — from results_summary.json or computed from predictions
        results = _load_results_summary(exp.dir_path)
        breakdowns = _extract_breakdowns(results)
        if not breakdowns:
            breakdowns = _compute_category_breakdowns(
                predictions, exp.benchmark, state["data_root"]
            )

        token_usage = _load_experiment_token_usage(exp.dir_path)

        template = jinja_env.get_template("experiment.html")
        return template.render(
            exp=exp,
            predictions=annotated,
            config_json=json.dumps(exp.config, indent=2),
            breakdowns=breakdowns,
            token_usage=token_usage,
        )

    @app.get("/experiment/{dir_name}/sample/{sample_id}", response_class=HTMLResponse)
    async def sample_detail(dir_name: str, sample_id: str):
        exp = _get_exp(dir_name)
        predictions = _load_predictions_cached(exp.dir_path)

        # Find this prediction and annotate with extracted answer
        cat_map = get_category_map(exp.benchmark, state["data_root"])
        prediction = None
        for p in predictions:
            if str(p.get("sample_id", "")) == sample_id:
                content = p.get("content", "")
                gt = p.get("ground_truth", "")
                category = cat_map.get(sample_id, "")
                is_numerical = is_numerical_category(exp.benchmark, category)

                if is_numerical:
                    pred_f = _extract_number(content)
                    gt_f = _extract_number(gt)
                    extracted_pred = f"{pred_f:.4g}" if pred_f is not None else ""
                    extracted_gt = f"{gt_f:.4g}" if gt_f is not None else gt
                    pre = _pred_result(p)
                    if pre is not None:
                        mra = pre
                    elif pred_f is not None and gt_f is not None and gt_f != 0:
                        mra = _mean_relative_accuracy(pred_f, gt_f)
                    else:
                        mra = 0.0
                else:
                    extracted_pred, extracted_gt = _extract_display(content, gt, exp.benchmark)
                    mra = None

                prediction = {
                    **p,
                    "extracted_pred": extracted_pred,
                    "extracted_gt": extracted_gt,
                    "is_correct": (not is_numerical) and _pred_is_correct(p, exp.benchmark, category),
                    "is_empty": not extracted_pred,
                    "is_numerical": is_numerical,
                    "mra_score": mra,
                    "category": category,
                }
                break

        # Prev/next navigation
        pred_ids = [str(p.get("sample_id", "")) for p in predictions]
        prev_id = None
        next_id = None
        if sample_id in pred_ids:
            idx = pred_ids.index(sample_id)
            if idx > 0:
                prev_id = pred_ids[idx - 1]
            if idx < len(pred_ids) - 1:
                next_id = pred_ids[idx + 1]

        # Check for session report
        session_dir = os.path.join(exp.dir_path, f"session-{sample_id}")
        has_report = os.path.isfile(os.path.join(session_dir, "session_report.html"))
        token_usage = (prediction or {}).get("usage")

        template = jinja_env.get_template("sample.html")
        return template.render(
            exp=exp,
            sample_id=sample_id,
            prediction=prediction,
            prev_id=prev_id,
            next_id=next_id,
            has_report=has_report,
            token_usage=token_usage,
        )

    @app.get("/compare", response_class=HTMLResponse)
    async def compare_view(exp: list[str] = Query(default=[])):
        if len(exp) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 experiments to compare")

        experiments = []
        all_preds = {}  # dir_name -> {sample_id: prediction}
        for dir_name in exp:
            e = _get_exp(dir_name)
            experiments.append(e)
            preds = _load_predictions_cached(e.dir_path)
            all_preds[dir_name] = {str(p["sample_id"]): p for p in preds}

        # Find common sample IDs
        common_ids_set = None
        for dir_name, preds_dict in all_preds.items():
            ids = set(preds_dict.keys())
            common_ids_set = ids if common_ids_set is None else common_ids_set & ids
        common_ids_set = common_ids_set or set()
        common_ids = sorted(common_ids_set, key=lambda s: (int(s) if s.isdigit() else float('inf'), s))

        # Helper: compute blended accuracy on a set of sample IDs
        def _compute_acc(dir_name: str, sample_ids: set, benchmark: str) -> dict:
            preds_dict = all_preds[dir_name]
            subset_preds = [
                preds_dict[sid] for sid in sample_ids if sid in preds_dict
            ]
            acc, correct, total = _compute_blended_accuracy(
                subset_preds, benchmark, state["data_root"]
            )
            return {"correct": correct, "total": total, "accuracy": acc}

        # Helper: compute per-category breakdown on a set of sample IDs
        # Delegates to _compute_category_breakdowns for consistent scoring
        def _compute_cat_acc(dir_name: str, sample_ids: set, benchmark: str) -> list:
            preds_dict = all_preds[dir_name]
            subset = [preds_dict[sid] for sid in sample_ids if sid in preds_dict]
            breakdowns = _compute_category_breakdowns(subset, benchmark, state["data_root"])
            return breakdowns[0]["rows"] if breakdowns else []

        # Build per-experiment performance: full + common-only
        benchmark = experiments[0].benchmark  # assume same benchmark for comparison
        perf_data = []
        for e in experiments:
            full_ids = set(all_preds[e.dir_name].keys())
            full_acc = _compute_acc(e.dir_name, full_ids, e.benchmark)
            common_acc = _compute_acc(e.dir_name, common_ids_set, e.benchmark)
            full_cats = _compute_cat_acc(e.dir_name, full_ids, e.benchmark)
            common_cats = _compute_cat_acc(e.dir_name, common_ids_set, e.benchmark)
            perf_data.append({
                "dir_name": e.dir_name,
                "method": e.method,
                "model": e.model,
                "full": full_acc,
                "common": common_acc,
                "full_categories": full_cats,
                "common_categories": common_cats,
            })

        # Precompute session dirs for report detection
        session_sets = {}
        for e in experiments:
            session_sets[e.dir_name] = set(_list_session_ids(e.dir_path))

        # Build comparison rows
        comparison_rows = []
        both_correct = 0
        both_wrong = 0
        disagree = 0
        cmp_benchmark = experiments[0].benchmark
        cmp_cat_map = get_category_map(cmp_benchmark, state["data_root"])
        for sid in common_ids:
            gt_raw = all_preds[exp[0]][sid].get("ground_truth", "")
            _, gt = _extract_display("", gt_raw, cmp_benchmark)
            category = cmp_cat_map.get(sid, "")
            is_numerical = is_numerical_category(cmp_benchmark, category)
            predictions_list = []
            correctness = []
            for dir_name in exp:
                p = all_preds[dir_name][sid]
                pred_raw = p.get("content", "")

                if is_numerical:
                    pred_f = _extract_number(pred_raw)
                    gt_f = _extract_number(gt_raw)
                    display = f"{pred_f:.4g}" if pred_f is not None else ""
                    pre = _pred_result(p)
                    if pre is not None:
                        mra = pre
                    elif pred_f is not None and gt_f is not None and gt_f != 0:
                        mra = _mean_relative_accuracy(pred_f, gt_f)
                    else:
                        mra = 0.0
                    is_corr = mra >= 0.5
                else:
                    display, _ = _extract_display(pred_raw, gt_raw, cmp_benchmark)
                    mra = None
                    is_corr = _pred_is_correct(p, cmp_benchmark, category)

                predictions_list.append({
                    "display": display,
                    "is_correct": is_corr,
                    "is_numerical": is_numerical,
                    "mra_score": mra,
                })
                correctness.append(is_corr)

            has_any_report = any(
                sid in session_sets.get(e.dir_name, set()) for e in experiments
            )
            agreement = all(c == correctness[0] for c in correctness)
            comparison_rows.append({
                "sample_id": sid,
                "ground_truth": gt,
                "predictions": predictions_list,
                "agreement": agreement,
                "has_any_report": has_any_report,
            })

            if all(correctness):
                both_correct += 1
            elif not any(correctness):
                both_wrong += 1
            else:
                disagree += 1

        stats = {
            "common": len(common_ids),
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "disagree": disagree,
        } if len(exp) == 2 else None

        # Collect all category names for the breakdown table
        all_cat_names = []
        seen = set()
        for pd in perf_data:
            for row in pd["full_categories"] + pd["common_categories"]:
                if row["name"] not in seen:
                    all_cat_names.append(row["name"])
                    seen.add(row["name"])

        # Build query string for experiment list
        from urllib.parse import urlencode
        exp_query = urlencode([("exp", e) for e in exp])

        template = jinja_env.get_template("compare.html")
        return template.render(
            experiments=experiments,
            perf_data=perf_data,
            all_cat_names=all_cat_names,
            comparison_rows=comparison_rows,
            stats=stats,
            exp_query=exp_query,
        )

    @app.get("/compare/sample/{sample_id}", response_class=HTMLResponse)
    async def compare_sample_view(sample_id: str, exp: list[str] = Query(default=[])):
        if len(exp) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 experiments")

        from urllib.parse import urlencode
        exp_query = urlencode([("exp", e) for e in exp])

        experiments = []
        all_preds = {}
        for dir_name in exp:
            e = _get_exp(dir_name)
            experiments.append(e)
            preds = _load_predictions_cached(e.dir_path)
            all_preds[dir_name] = {str(p["sample_id"]): p for p in preds}

        # Common IDs for prev/next navigation
        common_ids_set = None
        for dir_name, preds_dict in all_preds.items():
            ids = set(preds_dict.keys())
            common_ids_set = ids if common_ids_set is None else common_ids_set & ids
        common_ids = sorted(
            common_ids_set or set(),
            key=lambda s: (int(s) if s.isdigit() else float('inf'), s),
        )

        # Prev/next
        prev_id = None
        next_id = None
        if sample_id in common_ids:
            idx = common_ids.index(sample_id)
            if idx > 0:
                prev_id = common_ids[idx - 1]
            if idx < len(common_ids) - 1:
                next_id = common_ids[idx + 1]

        # Build per-experiment prediction info
        prediction = all_preds[exp[0]].get(sample_id, {})
        cmp_benchmark = experiments[0].benchmark
        cmp_cat_map = get_category_map(cmp_benchmark, state["data_root"])
        category = cmp_cat_map.get(sample_id, "")
        is_numerical = is_numerical_category(cmp_benchmark, category)
        exp_predictions = []
        for e in experiments:
            p = all_preds[e.dir_name].get(sample_id, {})
            content = p.get("content", "")
            gt = p.get("ground_truth", "")

            if is_numerical:
                pred_f = _extract_number(content)
                gt_f = _extract_number(gt)
                extracted_pred = f"{pred_f:.4g}" if pred_f is not None else ""
                extracted_gt = f"{gt_f:.4g}" if gt_f is not None else gt
                pre = _pred_result(p)
                if pre is not None:
                    mra = pre
                elif pred_f is not None and gt_f is not None and gt_f != 0:
                    mra = _mean_relative_accuracy(pred_f, gt_f)
                else:
                    mra = 0.0
            else:
                extracted_pred, extracted_gt = _extract_display(content, gt, cmp_benchmark)
                mra = None

            session_dir = os.path.join(e.dir_path, f"session-{sample_id}")
            has_report = os.path.isfile(os.path.join(session_dir, "session_report.html"))
            exp_predictions.append({
                "dir_name": e.dir_name,
                "method": e.method,
                "content": content,
                "extracted_pred": extracted_pred,
                "extracted_gt": extracted_gt,
                "is_correct": (not is_numerical) and _pred_is_correct(p, cmp_benchmark, category),
                "is_numerical": is_numerical,
                "mra_score": mra,
                "is_empty": not extracted_pred,
                "has_report": has_report,
            })

        template = jinja_env.get_template("compare_sample.html")
        return template.render(
            sample_id=sample_id,
            experiments=experiments,
            exp_predictions=exp_predictions,
            prediction=prediction,
            prev_id=prev_id,
            next_id=next_id,
            exp_query=exp_query,
        )

    # --- Static file serving ---

    @app.get("/static/report/{dir_name}/{sample_id}")
    async def serve_report(dir_name: str, sample_id: str):
        exp = _get_exp(dir_name)
        path = os.path.join(exp.dir_path, f"session-{sample_id}", "session_report.html")
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Report not found")
        return FileResponse(path, media_type="text/html")

    @app.get("/static/image/{dir_name}/{sample_id}/{subdir}/{filename}")
    async def serve_image(dir_name: str, sample_id: str, subdir: str, filename: str):
        exp = _get_exp(dir_name)
        # Restrict subdir to known directories
        if subdir not in ("vlm_queries", "show_images", "input_images"):
            raise HTTPException(status_code=403, detail="Forbidden subdirectory")
        # Sanitize filename
        if ".." in filename or "/" in filename:
            raise HTTPException(status_code=403, detail="Invalid filename")
        path = os.path.join(exp.dir_path, f"session-{sample_id}", subdir, filename)
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Image not found")
        # Determine media type
        ext = Path(filename).suffix.lower()
        media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
        return FileResponse(path, media_type=media_types.get(ext, "application/octet-stream"))

    # --- JSON API endpoints ---

    @app.get("/api/experiments")
    async def api_experiments():
        return JSONResponse([
            {
                "dir_name": e.dir_name,
                "method": e.method,
                "benchmark": e.benchmark,
                "model": e.model,
                "tools": e.tools,
                "num_sessions": e.num_sessions,
                "num_predictions": e.num_predictions,
                "accuracy": e.accuracy,
                "correct_samples": e.correct_samples,
                "total_samples": e.total_samples,
                "created_date": e.created_date,
            }
            for e in state["experiments"]
        ])

    @app.get("/api/experiment/{dir_name}/predictions")
    async def api_predictions(dir_name: str):
        exp = _get_exp(dir_name)
        return JSONResponse(_load_predictions_cached(exp.dir_path))

    @app.get("/api/experiment/{dir_name}/results")
    async def api_results(dir_name: str):
        exp = _get_exp(dir_name)
        results = _load_results_summary(exp.dir_path)
        if results:
            return JSONResponse(results)
        # Compute from predictions
        preds = _load_predictions_cached(exp.dir_path)
        if not preds:
            return JSONResponse({"error": "No predictions found"}, status_code=404)
        accuracy, correct, total = _compute_blended_accuracy(
            preds, exp.benchmark, state["data_root"]
        )
        return JSONResponse({
            "total_samples": total,
            "correct_samples": correct,
            "overall_accuracy": accuracy,
        })

    @app.get("/api/experiment/{dir_name}/prediction/{sample_id}")
    async def api_prediction(dir_name: str, sample_id: str):
        exp = _get_exp(dir_name)
        preds = _load_predictions_cached(exp.dir_path)
        for p in preds:
            if str(p.get("sample_id", "")) == sample_id:
                return JSONResponse(p)
        raise HTTPException(status_code=404, detail=f"Sample {sample_id!r} not found")

    @app.get("/api/experiment/{dir_name}/samples")
    async def api_samples(dir_name: str):
        exp = _get_exp(dir_name)
        return JSONResponse(_list_session_ids(exp.dir_path))

    @app.get("/api/refresh")
    async def api_refresh():
        # Clear caches
        _load_predictions_cached.cache_clear()
        clear_category_cache()
        state["experiments"] = scan_work_dir(work_dir, data_root)
        return JSONResponse({"status": "ok", "count": len(state["experiments"])})

    return app
