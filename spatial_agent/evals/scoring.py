"""Shared scoring helpers for benchmark evaluators."""

import json
import os
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


def get_prediction(
    predictions: Mapping[Any, str], sample_id: Any, default: str = ""
) -> str:
    """Fetch a prediction by sample id, accepting both native and string keys."""
    if sample_id in predictions:
        return predictions[sample_id]
    sample_id_str = str(sample_id)
    if sample_id_str in predictions:
        return predictions[sample_id_str]
    return default


def mean_or_zero(values: Iterable[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty iterable."""
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def mean_relative_accuracy(
    pred: float,
    target: float,
    *,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
    zero_policy: str = "exact",
    zero_threshold: Optional[float] = None,
) -> float:
    """Compute MRA using the official threshold construction."""
    if target == 0:
        if zero_policy == "threshold":
            if zero_threshold is None:
                raise ValueError("zero_threshold is required for threshold zero_policy")
            if pred < zero_threshold:
                return 1.0
            target = zero_threshold
        else:
            return 1.0 if pred == 0 else 0.0

    num_pts = int((end - start) / interval + 2)
    thresholds = np.linspace(start, end, num_pts)
    relative_error = abs(pred - target) / abs(target)
    return float(np.mean(relative_error <= (1.0 - thresholds)))


def write_json(path: str, payload: Any) -> None:
    """Write JSON with stable formatting."""
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_results_summary(
    output_dir: str,
    results: Mapping[str, Any],
    *,
    detail_keys: Sequence[str] = ("detailed_results",),
) -> None:
    """Persist a benchmark summary while omitting large detail blobs."""
    os.makedirs(output_dir, exist_ok=True)
    detail_keys = set(detail_keys)
    summary = {k: v for k, v in results.items() if k not in detail_keys}
    write_json(os.path.join(output_dir, "results_summary.json"), summary)
