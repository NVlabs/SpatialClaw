"""CoT baseline: single VLM call per sample, no agent loop or tools.

Usage::

    python -m spatial_agent.entrypoints.cot_baseline \
        --dataset spatial_agent/config/dataset/erqa.json \
        --model spatial_agent/config/model/qwen3.5-122b-a10b.json \
        --concurrency 32
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm.asyncio import tqdm


# ── prompt ──────────────────────────────────────────────────────────────────

COT_SYSTEM_PROMPT = """\
You are an expert visual spatial reasoning assistant.

You will be given one or more images (video frames) and a multiple-choice question about spatial relationships, motion, geometry, or scene understanding.

**Instructions:**
1. Carefully examine ALL provided images.
2. Reason step-by-step about the spatial relationships, motion, distances, orientations, or other relevant aspects.
3. After your reasoning, output your final answer as a single uppercase letter inside \\boxed{}, e.g. \\boxed{A}.

Important:
- Consider temporal ordering of frames (earlier frames come first).
- Pay attention to camera motion, object motion, and spatial layout.
- If frames show a video sequence, reason about how objects and the scene change over time.
"""

DIRECT_SYSTEM_PROMPT = ""


# ── helpers ─────────────────────────────────────────────────────────────────

def _select_key_frames(
    image_paths: List[str], max_frames: int = 8
) -> List[str]:
    """Uniformly sample up to *max_frames* from *image_paths*."""
    n = len(image_paths)
    if n <= max_frames:
        return image_paths
    step = n / max_frames
    indices = [int(i * step) for i in range(max_frames)]
    return [image_paths[i] for i in indices]


def _load_images(
    paths: List[str], max_long_edge: Optional[int] = 768
) -> List[Image.Image]:
    """Load PIL images from disk, resizing onto the Pi3 grid if requested."""
    from spatial_agent.gpu_models.image_resize import resize_for_input_images

    imgs: List[Image.Image] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        if max_long_edge:
            img = resize_for_input_images(img, max_long_edge)
        imgs.append(img)
    return imgs


def _endpoint_inclusive_indices(num_images: int, budget: int) -> List[int]:
    """Match ``SpatialAgentWorkflow`` key-frame selection exactly."""
    if num_images <= 0 or budget <= 0:
        return []
    count = min(num_images, budget)
    if count >= num_images:
        return list(range(num_images))
    if count == 1:
        return [0]
    return [round(i * (num_images - 1) / (count - 1)) for i in range(count)]


def _load_spatial_agent_image(
    image: Any,
    max_long_edge: Optional[int],
    backend: str,
) -> Image.Image:
    """Load and resize one image with the workflow's backend-aware path."""
    if isinstance(image, (str, os.PathLike)):
        loaded = Image.open(image).convert("RGB")
    else:
        loaded = image
    if max_long_edge:
        from spatial_agent.gpu_models.image_resize import (
            resize_for_input_images_for_backend,
        )

        loaded = resize_for_input_images_for_backend(
            loaded, max_long_edge, backend
        )
    return loaded


def _rgb_fingerprint(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def _prepare_spatial_agent_visuals(
    sample: Any,
    *,
    num_key_frames: int,
    max_long_edge: Optional[int],
    backend: str,
    record_fingerprints: bool,
) -> tuple[List[Image.Image], Optional[Dict[str, Any]]]:
    """Prepare the exact visual sequence supplied by ``SpatialAgentWorkflow``.

    This is an opt-in CoT control path. The legacy CoT frame sampler remains
    unchanged so prior baselines for other backbones keep their behavior.
    """
    images = getattr(sample, "images", None) or []
    image_groups = getattr(sample, "image_groups", None) or []
    frame_indices = getattr(sample, "frame_indices", None)
    frame_indices_groups = getattr(sample, "frame_indices_groups", None) or []
    is_multi_video = bool(image_groups) and len(image_groups) > 1

    key_frames: List[Image.Image] = []
    key_frame_indices: List[int] = []
    key_frame_list_indices: List[int] = []
    key_frame_video_indices: List[int] = []
    if num_key_frames > 0:
        if is_multi_video:
            for video_index, group in enumerate(image_groups):
                selected = _endpoint_inclusive_indices(
                    len(group), num_key_frames
                )
                source_indices = (
                    frame_indices_groups[video_index]
                    if video_index < len(frame_indices_groups)
                    else None
                )
                for local_index in selected:
                    key_frames.append(_load_spatial_agent_image(
                        group[local_index], max_long_edge, backend
                    ))
                    absolute_index = (
                        source_indices[local_index]
                        if source_indices and local_index < len(source_indices)
                        else local_index
                    )
                    key_frame_indices.append(absolute_index)
                    key_frame_list_indices.append(local_index)
                    key_frame_video_indices.append(video_index + 1)
        else:
            for local_index in _endpoint_inclusive_indices(
                len(images), num_key_frames
            ):
                key_frames.append(_load_spatial_agent_image(
                    images[local_index], max_long_edge, backend
                ))
                absolute_index = (
                    frame_indices[local_index]
                    if frame_indices and local_index < len(frame_indices)
                    else local_index
                )
                key_frame_indices.append(absolute_index)
                key_frame_list_indices.append(local_index)
                key_frame_video_indices.append(0)

    reference_images = [
        _load_spatial_agent_image(image, max_long_edge, backend)
        for image in (getattr(sample, "ref_images", None) or [])
    ]

    fingerprint: Optional[Dict[str, Any]] = {
        "evidence_schema_version": 1,
        "fingerprint_valid": False,
    }
    if record_fingerprints:
        try:
            fingerprint = {
                "evidence_schema_version": 1,
                "fingerprint_valid": True,
                "video_source": getattr(sample, "video", None),
                "video_sources": (
                    getattr(sample, "video_sources_per_video", None) or []
                ),
                "key_frames": [
                    {
                        "context_position": position,
                        "video_index": key_frame_video_indices[position],
                        "local_index": key_frame_list_indices[position],
                        "frame_index": key_frame_indices[position],
                        "mode": image.mode,
                        "size": list(image.size),
                        "rgb_sha256": _rgb_fingerprint(image),
                    }
                    for position, image in enumerate(key_frames)
                ],
                "reference_images": [
                    {
                        "reference_index": position,
                        "mode": image.mode,
                        "size": list(image.size),
                        "rgb_sha256": _rgb_fingerprint(image),
                    }
                    for position, image in enumerate(reference_images)
                ],
            }
        except Exception as exc:
            fingerprint = {
                "evidence_schema_version": 1,
                "fingerprint_valid": False,
                "evidence_error": {
                    "stage": "input_fingerprint",
                    "exception_type": type(exc).__name__,
                },
            }

    return key_frames + reference_images, fingerprint


def _extract_boxed(text: str) -> str:
    """Strip \\boxed{...} wrapper from LLM output, returning the inner content."""
    import re
    m = re.search(r"\\boxed\{(.+)\}", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _benchmark_evidence(benchmark, answer_text: str) -> Dict[str, Any]:
    """Extract comparison-only evidence after the normal scoring call."""
    try:
        value = benchmark.extract_answer(answer_text)
        if not isinstance(value, str):
            raise TypeError("benchmark extraction did not return a string")
        return {
            "evidence_schema_version": 1,
            "benchmark_extraction_valid": True,
            "benchmark_extracted": value,
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


def _invalid_input_evidence(stage: str, exception_type: str) -> Dict[str, Any]:
    return {
        "evidence_schema_version": 1,
        "fingerprint_valid": False,
        "evidence_error": {
            "stage": stage,
            "exception_type": exception_type,
        },
    }


def _build_question(sample, benchmark, prompt_style: str = "cot") -> str:
    """Assemble the user-facing question text (question + choices)."""
    parts = [sample.question]

    if prompt_style == "cot":
        parts.append("")
        parts.append(benchmark.data_specific_prompt)

    # Append choices
    if hasattr(sample, "choices") and sample.choices:
        parts.append("")
        if isinstance(sample.choices, dict):
            for letter, text in sample.choices.items():
                parts.append(f"{letter}. {text}")
        elif isinstance(sample.choices, list):
            for i, text in enumerate(sample.choices):
                parts.append(f"{chr(65 + i)}. {text}")

    parts.append("")
    if prompt_style == "cot":
        parts.append("Think step-by-step, then provide your final answer inside \\boxed{}.")
    else:
        parts.append("Answer with a single letter.")
    return "\n".join(parts)


# ── worker ──────────────────────────────────────────────────────────────────

async def worker(
    llm_client,
    benchmark,
    sample,
    predictions: Dict,
    pred_file: str,
    input_manifest_file: str,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
    max_frames: int,
    max_long_edge: Optional[int],
    frame_preparation: str,
    num_key_frames: int,
    reconstruct_backend: str,
    record_input_fingerprints: bool,
    finalize_on_llm_timeout: bool = False,
    generation_timeout_sec: Optional[float] = None,
    role_params=None,
    system_prompt: str = COT_SYSTEM_PROMPT,
    prompt_style: str = "cot",
):
    async with semaphore:
        sample_started_at = (
            time.monotonic() if record_input_fingerprints else None
        )
        sid = sample.sample_id
        input_fingerprint: Optional[Dict[str, Any]] = _invalid_input_evidence(
            "input_preparation", "MissingEvidence",
        )
        terminal_failure: Optional[Dict[str, Any]] = None
        try:
            # Ensure video frames are extracted (VLM4D lazy loading)
            if hasattr(sample, "ensure_frames_loaded"):
                await asyncio.get_event_loop().run_in_executor(
                    None, sample.ensure_frames_loaded
                )

            # Select and load key frames (run in a thread to avoid blocking the
            # event loop). The opt-in path exactly mirrors SpatialClaw's
            # endpoint-inclusive selection and backend-aware preprocessing.
            if frame_preparation == "spatial-agent":
                images, input_fingerprint = await asyncio.to_thread(
                    _prepare_spatial_agent_visuals,
                    sample,
                    num_key_frames=num_key_frames,
                    max_long_edge=max_long_edge,
                    backend=reconstruct_backend,
                    record_fingerprints=record_input_fingerprints,
                )
            else:
                key_paths = _select_key_frames(sample.images, max_frames)
                images = await asyncio.to_thread(
                    _load_images, key_paths, max_long_edge
                )
                if record_input_fingerprints:
                    try:
                        input_fingerprint = {
                            "evidence_schema_version": 1,
                            "fingerprint_valid": True,
                            "frame_preparation": "legacy",
                            "video_source": str(
                                getattr(sample, "video", "") or ""
                            ),
                            "video_sources": [
                                str(value)
                                for value in (
                                    getattr(
                                        sample,
                                        "video_sources_per_video",
                                        None,
                                    ) or []
                                )
                            ],
                            "key_frames": [
                                {
                                    "context_position": position,
                                    "source": str(key_paths[position]),
                                    "mode": image.mode,
                                    "size": list(image.size),
                                    "rgb_sha256": _rgb_fingerprint(image),
                                }
                                for position, image in enumerate(images)
                            ],
                            "reference_images": [],
                        }
                    except Exception as exc:
                        input_fingerprint = _invalid_input_evidence(
                            "input_fingerprint", type(exc).__name__,
                        )
                else:
                    input_fingerprint = {
                        "evidence_schema_version": 1,
                        "fingerprint_valid": False,
                    }
            if not images:
                raise RuntimeError(f"No images loaded for sample {sid}")

            question = _build_question(sample, benchmark, prompt_style=prompt_style)

            # Single VLM call — retry indefinitely on server unavailability
            from openai import APIConnectionError, APITimeoutError
            from spatial_agent.llm.client import LLMGenerationTimeoutError
            _server_errors = (
                asyncio.TimeoutError, ConnectionError, OSError,
                APIConnectionError, APITimeoutError,
            )
            while True:
                try:
                    answer_text = await llm_client.generate_vision_query(
                        images=images,
                        question=question,
                        system_prompt=system_prompt,
                        role_params=role_params,
                        session_id=str(sid),
                        usage_session_id=str(sid),
                    )
                    break
                except LLMGenerationTimeoutError as exc:
                    if not finalize_on_llm_timeout:
                        print(
                            f"[Wait] Sample {sid}: server unavailable "
                            f"({type(exc).__name__}), retrying in 30s..."
                        )
                        # Preserve the historical retry/re-discovery behavior
                        # unless the selected model profile explicitly opts in.
                        llm_client._last_discovery = 0
                        await asyncio.sleep(30)
                        continue
                    answer_text = ""
                    terminal_failure = {
                        "reason": "llm_generation_timeout",
                        "request_kind": exc.request_kind,
                        "attempts": exc.attempts,
                        "request_timeout_sec": generation_timeout_sec,
                    }
                    print(
                        f"[Terminal] Sample {sid}: {exc}; recording failed row."
                    )
                    break
                except _server_errors as exc:
                    print(
                        f"[Wait] Sample {sid}: server unavailable ({type(exc).__name__}), "
                        f"retrying in 30s..."
                    )
                    # Force re-discovery by resetting TTL
                    llm_client._last_discovery = 0
                    await asyncio.sleep(30)
        except Exception as exc:
            import traceback
            print(f"[Error] Sample {sid}: {exc}")
            traceback.print_exc()
            answer_text = ""
            terminal_failure = {
                "reason": "worker_exception",
                "exception_type": type(exc).__name__,
            }

        async with lock:
            extracted = _extract_boxed(answer_text)
            predictions[sid] = extracted
            gt = getattr(sample, "answer", None)
            entry = {"sample_id": str(sid), "content": answer_text, "extracted": extracted}
            if record_input_fingerprints:
                entry["wall_time_sec"] = time.monotonic() - sample_started_at
            if gt is not None:
                entry["ground_truth"] = str(gt)
            result = benchmark.evaluate_single(sample, extracted)
            if result is not None:
                entry["result"] = result
            if record_input_fingerprints:
                entry.update(_benchmark_evidence(benchmark, extracted))
            entry["usage"] = llm_client.pop_session_usage(str(sid))
            if terminal_failure is not None:
                entry["terminal_failure"] = terminal_failure
            with open(pred_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
            if record_input_fingerprints:
                input_entry = {
                    "sample_id": str(sid),
                    **(input_fingerprint or _invalid_input_evidence(
                        "input_preparation", "MissingEvidence",
                    )),
                }
                try:
                    with open(input_manifest_file, "a") as f:
                        f.write(json.dumps(input_entry, default=str) + "\n")
                except Exception as exc:
                    print(
                        f"[Warning] Input evidence write failed for sample {sid}: "
                        f"{type(exc).__name__}"
                    )


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CoT Baseline Evaluation")
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
    parser.add_argument("--max_frames", type=int, default=32,
                        help="Max frames to send to the VLM per sample")
    parser.add_argument(
        "--frame-preparation",
        choices=["legacy", "spatial-agent"],
        default=None,
        help=(
            "Visual preparation path. 'spatial-agent' exactly matches "
            "SpatialClaw key-frame selection and backend-aware resizing; "
            "'legacy' preserves historical CoT behavior. When omitted, the "
            "model config's cot_frame_preparation value is used."
        ),
    )
    parser.add_argument(
        "--record-input-fingerprints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record exact RGB hashes for the visual inputs supplied to CoT.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Set the generation seed for paired repeat runs.",
    )
    parser.add_argument("--sample_ids", nargs="+", default=None)
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle samples before applying --limit (for random sampling)")
    parser.add_argument("--subsample", type=int, default=None, metavar="N",
                        help="Deterministically subsample N random samples (seed=42). "
                             "Shortcut for --shuffle --limit N.")
    parser.add_argument("--system_prompt", type=str, default="cot",
                        choices=["cot", "direct"],
                        help="System prompt style: 'cot' (chain-of-thought) or 'direct' (single letter answer)")
    return parser.parse_args()


async def main():
    args = parse_args()

    # ── config ──────────────────────────────────────────────────────────
    from spatial_agent.config import SpatialAgentConfig, set_config

    config = SpatialAgentConfig()

    # Loading order: dataset JSON -> model JSON -> CLI args
    config.update_from_dataset_json(args.dataset)
    if args.model:
        config.update_from_model_json(args.model)
    if args.frame_preparation is None:
        args.frame_preparation = config.cot_frame_preparation
    else:
        config.cot_frame_preparation = args.frame_preparation
    config.update_from_args(args)
    if args.seed is not None:
        config.general_params.seed = args.seed

    # Override defaults for CoT baseline (no tools, no kernel)
    config.tools_to_use = []
    config.validate()

    # Default concurrency is higher for CoT (no GPU tools needed)
    if config.concurrency == 1 and args.concurrency is None:
        config.concurrency = 32

    # Work dir. Named model profiles use isolated output roots so a Cosmos
    # configuration update cannot resume historical baseline predictions.
    if not config.work_dir:
        _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_short = (
            config.llm_model.split("/")[-1][:30] if config.llm_model else "unknown"
        )
        model_variant = config.model_profile or model_short
        config.work_dir = os.path.join(
            _pkg_dir, "work_dir", f"cot_{config.benchmark}_{model_variant}"
        )
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

    # ── benchmark ───────────────────────────────────────────────────────
    from spatial_agent.evals.factory import BenchmarkFactory

    benchmark = BenchmarkFactory.create_benchmark(
        config.benchmark, question_type=config.question_type
    )
    if benchmark is None:
        print("No benchmark selected.")
        return

    # --subsample is a shortcut for --shuffle --limit N
    if args.subsample is not None:
        args.shuffle = True
        config.limit = args.subsample

    if config.sample_ids:
        id_set = set(config.sample_ids)
        benchmark.data = [
            s for s in benchmark.data
            if s.sample_id in id_set or str(s.sample_id) in id_set
        ]
    else:
        if args.shuffle:
            import random
            random.seed(42)
            random.shuffle(benchmark.data)
        if config.limit:
            benchmark.data = benchmark.data[: config.limit]

    from spatial_agent.entrypoints.run import (
        _effective_evaluation_config_sha256,
        _establish_run_manifest,
        _sealed_git_head_from_environment,
        _sha256_file,
        _source_tree_sha256,
    )
    from spatial_agent.llm.client import resolve_request_timeouts

    effective_request_timeouts = resolve_request_timeouts(config)

    record_input_fingerprints = bool(
        getattr(config, "record_input_fingerprints", False)
    )
    sealed_git_head = _sealed_git_head_from_environment()
    provenance_enabled = bool(
        record_input_fingerprints or sealed_git_head is not None
    )
    run_manifest = None
    if provenance_enabled:
        sample_ids = [str(sample.sample_id) for sample in benchmark]
        cohort_bytes = ("\n".join(sample_ids) + "\n").encode("utf-8")
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        run_manifest = {
            "schema_version": 1,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "entrypoint": "cot_baseline",
            "benchmark": config.benchmark,
            "model": config.llm_model,
            "model_profile": config.model_profile,
            "seed": args.seed,
            "requested_subsample": args.subsample,
            "selection_seed": 42 if args.subsample is not None else None,
            "applied_samples": len(sample_ids),
            "concurrency": config.concurrency,
            "max_frames": args.max_frames,
            "system_prompt": args.system_prompt,
            "resume_requested": bool(args.resume),
            "sample_ids_sha256": hashlib.sha256(cohort_bytes).hexdigest(),
            "sample_ids": sample_ids,
            "frame_preparation": args.frame_preparation,
            "num_key_frames": (
                config.num_key_frames
                if args.frame_preparation == "spatial-agent"
                else args.max_frames
            ),
            "image_max_long_edge": config.image_max_long_edge,
            "reconstruct_backend": (
                config.reconstruct_backend
                if args.frame_preparation == "spatial-agent"
                else None
            ),
            "record_input_fingerprints": record_input_fingerprints,
            "model_config": args.model,
            "model_config_sha256": _sha256_file(args.model),
            "dataset_config": args.dataset,
            "dataset_config_sha256": _sha256_file(args.dataset),
            "policy_config": None,
            "policy_config_sha256": None,
            "effective_request_timeouts": effective_request_timeouts,
            "effective_evaluation_config_sha256": (
                _effective_evaluation_config_sha256(config)
            ),
            "sealed_git_head": sealed_git_head,
            "source_tree_sha256": _source_tree_sha256(package_dir),
        }

    pred_file = os.path.join(config.work_dir, "predictions.jsonl")
    input_manifest_file = os.path.join(config.work_dir, "input_manifest.jsonl")
    if not args.resume:
        if os.path.exists(pred_file):
            os.remove(pred_file)
        if os.path.exists(input_manifest_file):
            os.remove(input_manifest_file)
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
    print(f"Model: {config.llm_model}")
    print(f"Max frames per sample: {args.max_frames}")
    print(f"Frame preparation: {args.frame_preparation}")
    print(f"General params: {config.general_params.to_dict()}")
    print(f"Concurrency: {config.concurrency}")
    print(f"Work dir: {config.work_dir}")

    # ── resume / fresh ──────────────────────────────────────────────────
    completed_ids: set = set()
    if args.resume and os.path.exists(pred_file):
        with open(pred_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    completed_ids.add(str(entry["sample_id"]))
                except Exception:
                    pass
        print(f"Resuming: {len(completed_ids)} samples already completed.")

    # ── LLM client ──────────────────────────────────────────────────────
    from spatial_agent.llm.client import LLMClient

    llm_client = LLMClient(config)

    # ── system prompt ────────────────────────────────────────────────────
    prompt_map = {"cot": COT_SYSTEM_PROMPT, "direct": DIRECT_SYSTEM_PROMPT}
    active_prompt = prompt_map[args.system_prompt]
    print(f"System prompt: {args.system_prompt}")

    # ── run ──────────────────────────────────────────────────────────────
    semaphore = asyncio.Semaphore(config.concurrency)
    lock = asyncio.Lock()
    predictions: Dict[Any, str] = {}

    if args.resume:
        for sid in completed_ids:
            predictions[sid] = ""

    tasks = []
    for sample in benchmark:
        if str(sample.sample_id) in completed_ids:
            continue
        tasks.append(
            worker(
                llm_client,
                benchmark,
                sample,
                predictions,
                pred_file,
                input_manifest_file,
                semaphore,
                lock,
                max_frames=args.max_frames,
                max_long_edge=config.image_max_long_edge,
                frame_preparation=args.frame_preparation,
                num_key_frames=config.num_key_frames,
                reconstruct_backend=config.reconstruct_backend,
                record_input_fingerprints=record_input_fingerprints,
                finalize_on_llm_timeout=bool(
                    getattr(config, "finalize_on_llm_timeout", False)
                ),
                generation_timeout_sec=effective_request_timeouts[
                    "vlm_request_timeout_sec"
                ],
                role_params=config.general_params,
                system_prompt=active_prompt,
                prompt_style=args.system_prompt,
            )
        )

    if tasks:
        print(f"\nProcessing {len(tasks)} samples...")
        await tqdm.gather(*tasks, desc=f"CoT {benchmark.__class__.__name__}")
    else:
        print("All samples already completed.")

    # ── evaluate ────────────────────────────────────────────────────────
    all_preds: Dict[str, str] = {}
    if os.path.exists(pred_file):
        with open(pred_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    all_preds[entry["sample_id"]] = entry.get("extracted", _extract_boxed(entry["content"]))
                except Exception:
                    pass

    results = benchmark.evaluate(all_preds, output_dir=config.work_dir)

    print(f"\nResults saved to: {config.work_dir}")


if __name__ == "__main__":
    asyncio.run(main())
