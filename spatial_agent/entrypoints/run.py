"""Main entry point for benchmark evaluation.

Usage::

    python -m spatial_agent.entrypoints.run \\
        --dataset spatial_agent/config/dataset/erqa.json \\
        --model spatial_agent/config/model/qwen3.5-122b-a10b.json \\
        --concurrency 4
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from typing import Any, Dict

from tqdm.asyncio import tqdm


def _sha256_file(path: str | None) -> str | None:
    """Hash a configuration artifact when it exists."""
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256(package_dir: str) -> str:
    """Fingerprint tracked and untracked source used by an experiment."""
    digest = hashlib.sha256()
    excluded = {"work_dir", "logs", "__pycache__"}
    for root, dirs, files in os.walk(package_dir):
        dirs[:] = sorted(name for name in dirs if name not in excluded)
        for name in sorted(files):
            if not name.endswith((".py", ".json", ".sh", ".jinja")):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, package_dir)
            digest.update(rel.encode("utf-8") + b"\0")
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _sealed_git_head_from_environment() -> str | None:
    """Return an optional controller-sealed commit for provenance metadata."""
    value = os.environ.get("SPATIAL_AGENT_SEALED_GIT_HEAD")
    if value is None:
        return None
    value = value.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40,64}", value):
        raise ValueError("SPATIAL_AGENT_SEALED_GIT_HEAD is malformed")
    return value


def _effective_evaluation_config_sha256(config) -> str:
    """Fingerprint result-affecting runtime config independent of output path."""
    payload = config.to_dict()
    payload.pop("work_dir", None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _run_manifest_identity(manifest: Dict[str, Any]) -> str:
    """Digest immutable run fields while excluding invocation timestamps."""
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"created_at", "resume_requested", "run_identity_sha256"}
    }
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _establish_run_manifest(
    work_dir: str,
    manifest: Dict[str, Any],
    predictions_path: str,
    *,
    enforce_identity: bool,
) -> None:
    """Verify a resume before atomically publishing run provenance."""
    manifest_path = os.path.join(work_dir, "run_manifest.json")
    expected = dict(manifest)
    expected["run_identity_sha256"] = _run_manifest_identity(expected)
    predictions_exist = (
        os.path.isfile(predictions_path)
        and os.path.getsize(predictions_path) > 0
    )

    observed = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                observed = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            if enforce_identity:
                raise RuntimeError(
                    f"Existing run manifest is unreadable: {exc}"
                ) from exc
    elif enforce_identity and predictions_exist:
        raise RuntimeError(
            "Predictions exist without run_manifest.json; refusing an "
            "unverifiable resume"
        )

    if enforce_identity and observed is not None:
        if not isinstance(observed, dict):
            raise RuntimeError("Existing run manifest root is not an object")
        if _run_manifest_identity(observed) != expected["run_identity_sha256"]:
            raise RuntimeError(
                "Run manifest does not match existing evaluator outputs; "
                "refusing a mixed-provenance resume"
            )
        # Preserve the originally sealed manifest and creation timestamp.
        return

    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=".run_manifest.", suffix=".tmp", dir=work_dir,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(expected, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
        temporary = None
        directory_fd = os.open(
            work_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial Agent Benchmark Evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model config JSON (config/model/<model>.json)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset config JSON (config/dataset/<benchmark>.json). "
                             "The benchmark name is inferred from the JSON's \"benchmark\" field.")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--question_type", nargs="+", default=None)
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--llm_base_url", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--sample_ids", nargs="+", default=None)
    parser.add_argument("--executor_type", type=str, default=None,
                        choices=["code", "react", "single_pass"],
                        help="Executor variant: code (default, free-form Python), "
                             "react (one tool-call per step), "
                             "single_pass (code with max_steps=1, no planning/reflection)")
    parser.add_argument("--enable_reflection", action="store_true", default=None,
                        help="Enable self-reflection node after each execution step")
    parser.add_argument(
        "--planning", dest="enable_planning",
        action=argparse.BooleanOptionalAction, default=None,
        help="Enable/disable the isolated planning call.",
    )
    parser.add_argument(
        "--direct-answer-anchor", dest="direct_answer_anchor",
        action=argparse.BooleanOptionalAction, default=None,
        help="Enable/disable the initial direct visual candidate.",
    )
    parser.add_argument(
        "--evidence-gated-override", dest="evidence_gated_override",
        action=argparse.BooleanOptionalAction, default=None,
        help="Enable/disable evidence-gated answer arbitration.",
    )
    parser.add_argument(
        "--restrict-adjudicator-to-candidates",
        dest="restrict_adjudicator_to_candidates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Require answer adjudication to choose the direct candidate or "
            "agent proposal; fail closed to the direct candidate otherwise."
        ),
    )
    parser.add_argument(
        "--require-novel-visual-override-evidence",
        dest="require_novel_visual_override_evidence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Count shown images as override evidence only when their pixels "
            "differ from initial key frames and previously accepted visuals."
        ),
    )
    parser.add_argument(
        "--reserve-final-answer-turn", dest="reserve_final_answer_turn",
        action=argparse.BooleanOptionalAction, default=None,
        help="Enable/disable the controller-reserved final selection.",
    )
    parser.add_argument(
        "--normalize-answer-symbols", dest="normalize_answer_symbols",
        action=argparse.BooleanOptionalAction, default=None,
        help="Enable/disable schema-aware bare answer-label normalization.",
    )
    parser.add_argument(
        "--record-input-fingerprints", dest="record_input_fingerprints",
        action=argparse.BooleanOptionalAction, default=None,
        help="Record exact RGB hashes for overlap-matched comparisons.",
    )
    parser.add_argument("--no-progress-limit", type=int, default=None)
    parser.add_argument("--soft-step-limit", type=int, default=None)
    parser.add_argument(
        "--generate-report", dest="generate_report",
        action=argparse.BooleanOptionalAction, default=None,
        help="Generate per-sample HTML/PDF reports (disable for throughput runs).",
    )
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle samples before applying --limit (for random sampling)")
    parser.add_argument("--subsample", type=int, default=None, metavar="N",
                        help="Deterministically subsample N random samples (seed=42). "
                             "Shortcut for --shuffle --limit N.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Set the sampling seed for every LLM role (for paired repeat runs).",
    )
    return parser.parse_args()


def _derive_answer_schema(
    sample: Any,
    *,
    robust_embedded_choice_schema: bool = False,
) -> Dict[str, Any]:
    """Derive the answer contract from visible choices, never ground truth."""
    choices = getattr(sample, "choices", None)
    if isinstance(choices, dict) and choices:
        labels = [str(label).strip().upper() for label in choices if str(label).strip()]
        if labels:
            return {"kind": "multiple_choice", "allowed_values": labels}
    if isinstance(choices, list) and choices:
        labels = [chr(65 + index) for index in range(len(choices))]
        return {"kind": "multiple_choice", "allowed_values": labels}

    # Some loaders embed choices in the visible question rather than exposing
    # ``sample.choices``. Require at least two line-leading option labels to
    # avoid treating incidental mentions of a letter as a choice schema.
    question = str(getattr(sample, "question", "") or "")
    # Preserve the historical consuming parser unless a model profile opts in
    # to robust embedded-choice recovery.
    trailing_text = (
        r"(?=\s+\S)" if robust_embedded_choice_schema else r"\s+\S"
    )
    line_pattern = re.compile(
        r"(?m)^\s*(?:\(([A-Z])\)|([A-Z])\s*[.):])" + trailing_text
    )
    embedded = [
        match.group(1) or match.group(2)
        for match in line_pattern.finditer(question)
    ]
    if len(set(embedded)) < 2:
        # Also support compact visible blocks such as
        # ``Choices: A. first B. second C. third`` (used by several loaders).
        match = re.search(r"\bchoices?\s*:\s*(.*)", question, re.IGNORECASE | re.DOTALL)
        segment = match.group(1) if match else ""
        compact_pattern = re.compile(
            r"(?:^|\s)(?:\(([A-Z])\)|([A-Z])\s*[.):])" + trailing_text
        )
        embedded = [
            option.group(1) or option.group(2)
            for option in compact_pattern.finditer(segment)
        ]
    if robust_embedded_choice_schema:
        # Weakly formatted choice blocks may use label-like tokens as option
        # text (for example, ``B. E.``). Recover only the ordered contiguous
        # A/B/C/... markers in both multiline and compact schemas while
        # ignoring those distractors.
        ordered = []
        for label in embedded:
            expected = chr(ord("A") + len(ordered))
            if label == expected:
                ordered.append(label)
        embedded = ordered
    labels = list(dict.fromkeys(embedded))
    if len(labels) >= 2:
        return {"kind": "multiple_choice", "allowed_values": labels}
    return {"kind": "open", "allowed_values": []}


def _benchmark_evidence(benchmark, answer_text):
    """Extract result-neutral benchmark evidence after authoritative scoring."""
    try:
        extracted = benchmark.extract_answer(answer_text)
        if not isinstance(extracted, str):
            raise TypeError("benchmark extraction did not return a string")
        return {
            "evidence_schema_version": 1,
            "benchmark_extraction_valid": True,
            "benchmark_extracted": extracted,
        }
    except Exception as exc:
        return {
            "evidence_schema_version": 1,
            "benchmark_extraction_valid": False,
            "benchmark_evidence_error": {
                "stage": "benchmark_extraction",
                "exception_type": type(exc).__name__,
            },
        }


def _input_evidence(run_result):
    fingerprint = (
        run_result.get("input_fingerprint")
        if isinstance(run_result, dict) else None
    )
    if (
        isinstance(fingerprint, dict)
        and fingerprint.get("evidence_schema_version") == 1
        and isinstance(fingerprint.get("fingerprint_valid"), bool)
    ):
        return dict(fingerprint)
    return {
        "evidence_schema_version": 1,
        "fingerprint_valid": False,
        "evidence_error": {
            "stage": "workflow_input_fingerprint",
            "exception_type": (
                "MalformedEvidence" if isinstance(fingerprint, dict)
                else "MissingEvidence"
            ),
        },
    }


def _prediction_from_run_result(run_result, config):
    """Resolve a scored prediction and result-neutral timeout metadata.

    The historical/default path continues to score the controller's final
    answer.  A model profile that opts into ``finalize_on_llm_timeout`` instead
    records the exhausted generation budget as an infrastructure failure, so
    a deterministic controller fallback cannot be mistaken for an ordinary
    benchmark prediction. The fallback text itself remains unchanged; strict
    campaigns can reject the row using the metadata without altering scoring.
    """
    timeout_attempts = int(
        run_result.get("llm_generation_timeout_attempts", 0) or 0
    )
    if (
        getattr(config, "finalize_on_llm_timeout", False)
        and timeout_attempts > 0
    ):
        return run_result.get("final_answer", {}).get("text", ""), {
            "reason": "llm_generation_timeout",
            "attempts": timeout_attempts,
            "request_timeout_sec": getattr(
                config, "llm_request_timeout_sec", None
            ),
        }
    return run_result.get("final_answer", {}).get("text", ""), None


async def worker(workflow, benchmark, sample, predictions, pred_file, semaphore, lock):
    """Process a single sample."""
    async with semaphore:
        record_evidence = bool(
            getattr(workflow.config, "record_input_fingerprints", False)
        )
        sample_started_at = time.monotonic() if record_evidence else None
        sid = sample.sample_id
        run_result = None
        terminal_failure = None
        try:
            if hasattr(sample, "ensure_frames_loaded"):
                await asyncio.to_thread(sample.ensure_frames_loaded)

            instruction = f"{sample.question}\n\n{benchmark.data_specific_prompt}"

            if hasattr(sample, "choices") and sample.choices:
                if isinstance(sample.choices, dict):
                    for letter, text in sample.choices.items():
                        instruction += f"\n{letter}. {text}"
                elif isinstance(sample.choices, list):
                    for i, text in enumerate(sample.choices):
                        instruction += f"\n{chr(65+i)}. {text}"

            run_result = await workflow.arun(
                instruction=instruction,
                images=sample.images,
                answer=sample.answer,
                session_id=str(sid),
                frame_indices=getattr(sample, "frame_indices", None),
                video_source=getattr(sample, "video", None),
                fps=getattr(sample, "fps", None),
                total_video_frames=getattr(sample, "total_video_frames", None),
                duration_sec=getattr(sample, "duration_sec", None),
                image_groups=getattr(sample, "image_groups", None),
                frame_indices_groups=getattr(sample, "frame_indices_groups", None),
                fps_per_video=getattr(sample, "fps_per_video", None),
                total_frames_per_video=getattr(sample, "total_frames_per_video", None),
                duration_per_video=getattr(sample, "duration_per_video", None),
                video_names=getattr(sample, "video_names", None),
                video_sources_per_video=getattr(sample, "video_sources_per_video", None),
                ref_images=getattr(sample, "ref_images", None),
                answer_schema=_derive_answer_schema(
                    sample,
                    robust_embedded_choice_schema=getattr(
                        workflow.config, "robust_embedded_choice_schema", False
                    ),
                ),
                defer_report=True,
            )
            answer_text, terminal_failure = _prediction_from_run_result(
                run_result, workflow.config
            )
        except Exception as exc:
            import traceback
            print(f"[Error] Sample {sid}: {exc}")
            traceback.print_exc()
            answer_text = ""
            terminal_failure = {
                "reason": "worker_exception",
                "exception_type": type(exc).__name__,
            }

        gt = getattr(sample, "answer", None)
        async with lock:
            predictions[sid] = answer_text
            entry = {"sample_id": str(sid), "content": answer_text}
            if record_evidence:
                entry["wall_time_sec"] = time.monotonic() - sample_started_at
            if gt is not None:
                entry["ground_truth"] = str(gt)
            score = benchmark.evaluate_single(sample, answer_text)
            if score is not None:
                entry["result"] = score
            if record_evidence:
                entry.update(_benchmark_evidence(benchmark, answer_text))
            if run_result and "usage" in run_result:
                entry["usage"] = run_result["usage"]
            if terminal_failure is not None:
                entry["terminal_failure"] = terminal_failure
            if record_evidence and run_result:
                for metadata_key in (
                    "termination_reason", "step_count", "total_tool_calls",
                ):
                    if metadata_key in run_result:
                        entry[metadata_key] = run_result[metadata_key]
            with open(pred_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            if record_evidence:
                input_entry = {
                    "sample_id": str(sid),
                    **_input_evidence(run_result),
                }
                manifest_file = os.path.join(
                    os.path.dirname(pred_file), "input_manifest.jsonl"
                )
                try:
                    with open(manifest_file, "a") as f:
                        f.write(json.dumps(input_entry, default=str) + "\n")
                except Exception as exc:
                    print(
                        f"[Warning] Input evidence write failed for sample {sid}: "
                        f"{type(exc).__name__}"
                    )

        report_ctx = run_result.get("_report_context") if run_result else None
        if report_ctx and workflow.config.generate_report:
            try:
                await asyncio.to_thread(
                    workflow.generate_report,
                    session_dir=report_ctx["session_dir"],
                    session_id=str(sid),
                    instruction=report_ctx["instruction"],
                    images=report_ctx["images"],
                    ground_truth=gt,
                    final_state=report_ctx["final_state"],
                    result_score=score,
                )
            except Exception as exc:
                print(f"[Warning] Failed to generate report for sample {sid}: {exc}")


async def main():
    args = parse_args()

    from spatial_agent.config import SpatialAgentConfig, set_config
    config = SpatialAgentConfig()

    # Apply in order so each layer wins over the previous:
    # dataset JSON → model JSON → CLI args.
    config.update_from_dataset_json(args.dataset)
    if args.model:
        config.update_from_model_json(args.model)
    config.update_from_args(args)
    if args.seed is not None:
        for role_attr in (
            "main_params",
            "vlm_params",
            "vlm_grounding_params",
            "planning_params",
            "general_params",
            "direct_answer_params",
            "reflection_params",
        ):
            getattr(config, role_attr).seed = args.seed
    config.validate()

    # Default work_dir lives next to the spatial_agent package, not cwd.
    # A named model profile gets an isolated directory so the new Cosmos
    # default cannot resume predictions produced by its legacy configuration.
    if not config.work_dir:
        _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_short = config.llm_model.split("/")[-1][:30] if config.llm_model else "unknown"
        model_variant = config.model_profile or model_short
        dir_name = f"spatial_{config.benchmark}_{model_variant}"
        config.work_dir = os.path.join(_pkg_dir, "work_dir", dir_name)
    os.makedirs(config.work_dir, exist_ok=True)
    preflight_predictions = os.path.join(config.work_dir, "predictions.jsonl")
    preflight_identity_enforced = bool(
        config.record_input_fingerprints
        or os.environ.get("SPATIAL_AGENT_SEALED_GIT_HEAD")
    )
    if not (
        args.resume
        and preflight_identity_enforced
        and os.path.isfile(preflight_predictions)
        and os.path.getsize(preflight_predictions) > 0
    ):
        # Keep the historical early snapshot for empty/new work directories,
        # but never overwrite evidence beside an unverified sealed resume.
        with open(os.path.join(config.work_dir, "config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)

    set_config(config)

    from spatial_agent.evals.factory import BenchmarkFactory
    benchmark = BenchmarkFactory.create_benchmark(
        config.benchmark, question_type=config.question_type
    )
    if benchmark is None:
        print("No benchmark selected.")
        return

    if args.subsample is not None:
        args.shuffle = True
        config.limit = args.subsample

    if config.sample_ids:
        id_set = set(config.sample_ids)
        benchmark.data = [s for s in benchmark.data if str(s.sample_id) in id_set]
    else:
        if args.shuffle:
            import random
            random.seed(42)
            random.shuffle(benchmark.data)
        if config.limit:
            benchmark.data = benchmark.data[: config.limit]

    sealed_git_head = _sealed_git_head_from_environment()
    provenance_enabled = bool(
        config.record_input_fingerprints or sealed_git_head is not None
    )
    run_manifest = None
    if provenance_enabled:
        sample_ids = [str(sample.sample_id) for sample in benchmark]
        cohort_bytes = ("\n".join(sample_ids) + "\n").encode("utf-8")
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        from spatial_agent.llm.client import resolve_request_timeouts
        run_manifest = {
            "schema_version": 1,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "entrypoint": "spatial_agent",
            "benchmark": config.benchmark,
            "model": config.llm_model,
            "model_profile": config.model_profile,
            "seed": args.seed,
            "requested_subsample": args.subsample,
            "selection_seed": 42 if args.subsample is not None else None,
            "applied_samples": len(sample_ids),
            "concurrency": config.concurrency,
            "resume_requested": bool(args.resume),
            "sample_ids_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
            "sample_ids": sample_ids,
            "model_config": args.model,
            "model_config_sha256": _sha256_file(args.model),
            "dataset_config": args.dataset,
            "dataset_config_sha256": _sha256_file(args.dataset),
            "policy_config": None,
            "policy_config_sha256": None,
            "effective_request_timeouts": resolve_request_timeouts(config),
            "effective_evaluation_config_sha256": (
                _effective_evaluation_config_sha256(config)
            ),
            "sealed_git_head": sealed_git_head,
            "source_tree_sha256": _source_tree_sha256(package_dir),
        }

    pred_file = os.path.join(config.work_dir, "predictions.jsonl")
    input_manifest_file = os.path.join(config.work_dir, "input_manifest.jsonl")
    if not args.resume:
        # Clear stale result-bearing artifacts before sealing a fresh manifest.
        if os.path.exists(pred_file):
            os.remove(pred_file)
        if os.path.exists(input_manifest_file):
            os.remove(input_manifest_file)
        for entry in os.listdir(config.work_dir):
            if entry.startswith("session-"):
                shutil.rmtree(
                    os.path.join(config.work_dir, entry), ignore_errors=True,
                )
    if run_manifest is not None:
        _establish_run_manifest(
            config.work_dir,
            run_manifest,
            pred_file,
            enforce_identity=True,
        )
    with open(os.path.join(config.work_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    print(f"Benchmark: {benchmark.__class__.__name__} ({len(benchmark)} samples)")

    completed_ids = set()
    if args.resume and os.path.exists(pred_file):
        with open(pred_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    completed_ids.add(str(entry["sample_id"]))
                except Exception:
                    pass
        print(f"Resuming: {len(completed_ids)} samples already completed.")

    from spatial_agent.workflow import SpatialAgentWorkflow
    workflow = SpatialAgentWorkflow(config)

    semaphore = asyncio.Semaphore(config.concurrency)
    lock = asyncio.Lock()
    predictions: Dict = {}

    if args.resume:
        for sid in completed_ids:
            predictions[sid] = ""  # placeholder

    tasks = []
    for sample in benchmark:
        if str(sample.sample_id) in completed_ids:
            continue
        tasks.append(
            worker(workflow, benchmark, sample, predictions, pred_file, semaphore, lock)
        )

    if tasks:
        await tqdm.gather(*tasks, desc=f"Evaluating {benchmark.__class__.__name__}")
    else:
        print("All samples already completed.")

    all_preds = {}
    if os.path.exists(pred_file):
        with open(pred_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    all_preds[entry["sample_id"]] = entry["content"]
                except Exception:
                    pass

    results = benchmark.evaluate(all_preds, output_dir=config.work_dir)

    workflow.shutdown()

    print(f"\nResults saved to: {config.work_dir}")


if __name__ == "__main__":
    asyncio.run(main())
