"""Ordered sample panels for paired runs on original benchmark datasets."""

import json
from pathlib import Path


def require_empty_panel_work_dir(work_dir):
    """Protect existing artifacts before initializing a fresh panel run."""
    if any(Path(work_dir).iterdir()):
        raise FileExistsError(
            f"Panel output directory {work_dir!r} is not empty. "
            "Use --resume for the same run or choose a new --work_dir."
        )


def load_sample_panel(path, config, *, subsample, shuffle):
    """Read the current benchmark's IDs before constructing its evaluator."""
    if (
        config.sample_ids is not None
        or config.limit is not None
        or config.question_type is not None
        or subsample is not None
        or shuffle
    ):
        raise ValueError(
            "--sample_panel cannot be combined with sample_ids, limit, "
            "subsample, shuffle, or question_type"
        )
    with open(path, encoding="utf-8") as handle:
        panels = json.load(handle)
    if not isinstance(panels, dict) or config.benchmark not in panels:
        raise ValueError(f"Missing benchmark {config.benchmark!r} in panel {path}")
    ids = panels[config.benchmark]
    if not isinstance(ids, list) or not ids or not all(
        isinstance(sid, str) and sid for sid in ids
    ):
        raise ValueError(
            f"Panel for {config.benchmark!r} must be a non-empty list of string IDs"
        )
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate requested sample ID in panel for {config.benchmark!r}")
    return ids


def select_panel_samples(samples, ids):
    """Return the original sample objects in panel order, requiring unique matches."""
    requested = set(ids)
    found = {}
    for sample in samples:
        sid = str(sample.sample_id)
        if sid in requested:
            if sid in found:
                raise ValueError(f"Multiple source samples match panel ID {sid!r}")
            found[sid] = sample
    missing = [sid for sid in ids if sid not in found]
    if missing:
        raise ValueError(f"Panel sample IDs missing from the source dataset: {missing}")
    return [found[sid] for sid in ids]
