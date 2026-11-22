"""Aggregate the fixed 20 × 200 Cosmos v4 panel from native per-sample results."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


# Compact JSON of the published panel, retaining benchmark and sample order.
PANEL_SHA256 = "59e271d418541910d957d311b7cce672e2466fc3288a2957dcbdb7d2e9fb4b2d"
# Sorted IDs whose SpatialTree metricfunc requires an external evaluator.
SPATIALTREE_UNSCORED_SHA256 = "dac94939b3b2ad3e3d27023c36bd1e10b026678d0048a23698fb11a060c16654"
PROFILES = {
    "cosmos3-nano": "cosmos3-nano-spatialclaw-v4",
    "cosmos3-super": "cosmos3-super-spatialclaw-v4-transfer",
}
BOOTSTRAPS = 20_000
CONFIG = Path(__file__).resolve().parents[1] / "config"


def id_digest(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def panel_rows(path, ids):
    expected = set(ids)
    seen = set()
    with path.open() as rows:
        for line_number, line in enumerate(rows, 1):
            row = json.loads(line)
            sid = row["sample_id"]
            if sid not in expected or sid in seen:
                raise ValueError(f"{path}:{line_number}: unexpected or duplicate sample {sid}")
            seen.add(sid)
            yield sid, row
    if seen != expected:
        raise ValueError(f"{path}: missing {len(expected - seen)} panel rows")


def visual_signature(row):
    if row.get("fingerprint_valid") is not True:
        raise ValueError(f"{row['sample_id']}: invalid input fingerprint")
    signature = {}
    for group, positions in [
        ("key_frames", ("context_position", "video_index", "local_index", "frame_index")),
        ("reference_images", ("reference_index",)),
    ]:
        frames = row.get(group)
        if not isinstance(frames, list) or (group == "key_frames" and not frames):
            raise ValueError(f"{row['sample_id']}: missing {group}")
        signature[group] = []
        for frame in frames:
            size = frame["size"]
            if (
                not re.fullmatch(r"[0-9a-f]{64}", frame["rgb_sha256"])
                or not isinstance(size, list) or len(size) != 2
                or any(type(n) is not int or n <= 0 for n in size)
                or not isinstance(frame["mode"], str)
                or any(type(frame[key]) is not int for key in positions)
            ):
                raise ValueError(f"{row['sample_id']}: invalid frame fingerprint in {group}")
            fields = ("rgb_sha256", "size", "mode", *positions)
            signature[group].append(tuple(frame[key] for key in fields))
    return signature


def read_arm(directory, benchmark, arm, ids):
    path = directory / "run_manifest.json"
    manifest = json.loads(path.read_text())
    model = manifest.get("model")
    if model not in PROFILES or manifest.get("model_profile") != PROFILES[model]:
        raise ValueError(f"{path}: expected a Cosmos v4 model and its matching profile")
    model_config = CONFIG / "model" / f"{model}.json"
    dataset_config = CONFIG / "dataset" / f"{benchmark}.json"
    expected = {
        "benchmark": benchmark,
        "entrypoint": {"cot": "cot_baseline", "agent": "spatial_agent"}[arm],
        "seed": 0,
        "concurrency": 2,
        "sample_ids": ids,
        "sample_ids_sha256": id_digest(ids),
        "applied_samples": len(ids),
        "model_config_sha256": hashlib.sha256(model_config.read_bytes()).hexdigest(),
        "dataset_config_sha256": hashlib.sha256(dataset_config.read_bytes()).hexdigest(),
    }
    if arm == "cot":
        expected.update(system_prompt="cot", frame_preparation="spatial-agent")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{path}: {key} differs from the panel protocol")
    path = directory / "predictions.jsonl"
    scores = {}
    for sid, row in panel_rows(path, ids):
        if row.get("terminal_failure") is not None:
            raise ValueError(f"{path}: terminal failure for {sid}")
        # Native entrypoints omit result when the local evaluator returns None.
        score = row.get("result")
        if score is not None and (type(score) not in (int, float) or not 0 <= score <= 1):
            raise ValueError(f"{path}: invalid native score for {sid}")
        scores[sid] = score
    unscored = sorted(sid for sid in ids if scores[sid] is None)
    if benchmark == "spatialtree":
        if id_digest(unscored) != SPATIALTREE_UNSCORED_SHA256:
            raise ValueError(f"{path}: unscored IDs differ from SpatialTree's external-metric subset")
    elif unscored:
        raise ValueError(f"{path}: missing numeric scores for {unscored}")
    signatures = {
        sid: visual_signature(row)
        for sid, row in panel_rows(directory / "input_manifest.jsonl", ids)
    }
    return model, [scores[sid] for sid in ids if scores[sid] is not None], signatures


def summarize(panel_path, results):
    panel = json.loads(panel_path.read_text())
    panel_bytes = json.dumps(panel, ensure_ascii=False, separators=(",", ":")).encode()
    if hashlib.sha256(panel_bytes).hexdigest() != PANEL_SHA256:
        raise ValueError(f"{panel_path}: expected the published 20 × 200 panel in its original order")

    rng = np.random.default_rng(20261028)
    overall_draws = np.zeros(BOOTSTRAPS)
    model = None
    routes = []
    for benchmark, ids in panel.items():
        arms = {}
        visuals = {}
        for arm in ("cot", "agent"):
            arm_model, scores, visuals[arm] = read_arm(results / benchmark / arm, benchmark, arm, ids)
            if model is not None and model != arm_model:
                raise ValueError(f"{benchmark}/{arm}: cannot combine different backbones")
            model = arm_model
            arms[arm] = np.asarray(scores, dtype=float)
        for sid in ids:
            if visuals["cot"][sid] != visuals["agent"][sid]:
                raise ValueError(f"{benchmark}/{sid}: initial visual inputs differ between arms")
        differences = arms["agent"] - arms["cot"]
        draws = np.empty(BOOTSTRAPS)
        for start in range(0, BOOTSTRAPS, 1000):
            indices = rng.integers(len(differences), size=(1000, len(differences)))
            draws[start:start + len(indices)] = 100 * differences[indices].mean(axis=1)
        overall_draws += draws / len(panel)
        routes.append({
            "benchmark": benchmark,
            "assigned_pairs": len(ids),
            "scoreable_pairs": len(differences),
            "cot_percent": float(100 * arms["cot"].mean()),
            "agent_percent": float(100 * arms["agent"].mean()),
            "delta_pp": float(100 * differences.mean()),
            "wins": int((differences > 1e-12).sum()),
            "losses": int((differences < -1e-12).sum()),
            "ties": int((np.abs(differences) <= 1e-12).sum()),
            "case_bootstrap95_pp": np.quantile(draws, [0.025, 0.975]).tolist(),
        })
    return {
        "model": model,
        "outcomes": 2 * sum(r["assigned_pairs"] for r in routes),
        "scoreable_pairs": sum(r["scoreable_pairs"] for r in routes),
        "wins": sum(r["wins"] for r in routes),
        "losses": sum(r["losses"] for r in routes),
        "ties": sum(r["ties"] for r in routes),
        "summary": {k: float(np.mean([r[k] for r in routes]))
                    for k in ("cot_percent", "agent_percent", "delta_pp")},
        "overall_case_bootstrap95_pp": np.quantile(overall_draws, [0.025, 0.975]).tolist(),
        "routes": routes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True, help="ordered benchmark → sample ID lists")
    parser.add_argument("--results", type=Path, required=True, help="directory containing <benchmark>/{cot,agent}")
    args = parser.parse_args()
    try:
        print(json.dumps(summarize(args.panel, args.results), indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
