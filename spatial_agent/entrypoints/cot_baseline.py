"""CoT baseline: single VLM call per sample, no agent loop or tools.

Usage::

    python -m spatial_agent.entrypoints.cot_baseline \
        --dataset spatial_agent/config/dataset/erqa.json \
        --model spatial_agent/config/model/qwen3.5-122b-a10b.json \
        --concurrency 32
"""

import argparse
import asyncio
import json
import os
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


def _extract_boxed(text: str) -> str:
    """Strip \\boxed{...} wrapper from LLM output, returning the inner content."""
    import re
    m = re.search(r"\\boxed\{(.+)\}", text, re.DOTALL)
    return m.group(1).strip() if m else text


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
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
    max_frames: int,
    max_long_edge: Optional[int],
    role_params=None,
    system_prompt: str = COT_SYSTEM_PROMPT,
    prompt_style: str = "cot",
):
    async with semaphore:
        sid = sample.sample_id
        try:
            # Ensure video frames are extracted (VLM4D lazy loading)
            if hasattr(sample, "ensure_frames_loaded"):
                await asyncio.get_event_loop().run_in_executor(
                    None, sample.ensure_frames_loaded
                )

            # Select and load key frames (run in executor to avoid blocking event loop)
            key_paths = _select_key_frames(sample.images, max_frames)
            images = await asyncio.get_event_loop().run_in_executor(
                None, _load_images, key_paths, max_long_edge
            )
            if not images:
                raise RuntimeError(f"No images loaded for sample {sid}")

            question = _build_question(sample, benchmark, prompt_style=prompt_style)

            # Single VLM call — retry indefinitely on server unavailability
            from openai import APIConnectionError, APITimeoutError
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

        async with lock:
            extracted = _extract_boxed(answer_text)
            predictions[sid] = extracted
            gt = getattr(sample, "answer", None)
            entry = {"sample_id": str(sid), "content": answer_text, "extracted": extracted}
            if gt is not None:
                entry["ground_truth"] = str(gt)
            result = benchmark.evaluate_single(sample, extracted)
            if result is not None:
                entry["result"] = result
            entry["usage"] = llm_client.pop_session_usage(str(sid))
            with open(pred_file, "a") as f:
                f.write(json.dumps(entry) + "\n")


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
    config.update_from_args(args)

    # Override defaults for CoT baseline (no tools, no kernel)
    config.tools_to_use = []

    # Default concurrency is higher for CoT (no GPU tools needed)
    if config.concurrency == 1 and args.concurrency is None:
        config.concurrency = 32

    # Work dir
    if not config.work_dir:
        _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_short = (
            config.llm_model.split("/")[-1][:30] if config.llm_model else "unknown"
        )
        config.work_dir = os.path.join(
            _pkg_dir, "work_dir", f"cot_{config.benchmark}_{model_short}"
        )
    os.makedirs(config.work_dir, exist_ok=True)

    # Save config snapshot
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

    print(f"Benchmark: {benchmark.__class__.__name__} ({len(benchmark)} samples)")
    print(f"Model: {config.llm_model}")
    print(f"Max frames per sample: {args.max_frames}")
    print(f"General params: {config.general_params.to_dict()}")
    print(f"Concurrency: {config.concurrency}")
    print(f"Work dir: {config.work_dir}")

    # ── resume / fresh ──────────────────────────────────────────────────
    pred_file = os.path.join(config.work_dir, "predictions.jsonl")
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
    elif not args.resume:
        if os.path.exists(pred_file):
            os.remove(pred_file)

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
                semaphore,
                lock,
                max_frames=args.max_frames,
                max_long_edge=config.image_max_long_edge,
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
