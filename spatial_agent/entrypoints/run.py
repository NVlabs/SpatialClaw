"""Main entry point for benchmark evaluation.

Usage::

    python -m spatial_agent.entrypoints.run \\
        --dataset spatial_agent/config/dataset/erqa.json \\
        --model spatial_agent/config/model/qwen3.5-122b-a10b.json \\
        --concurrency 4
"""

import argparse
import asyncio
import json
import os
import shutil
from typing import Dict

from tqdm.asyncio import tqdm


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
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle samples before applying --limit (for random sampling)")
    parser.add_argument("--subsample", type=int, default=None, metavar="N",
                        help="Deterministically subsample N random samples (seed=42). "
                             "Shortcut for --shuffle --limit N.")
    return parser.parse_args()


async def worker(workflow, benchmark, sample, predictions, pred_file, semaphore, lock):
    """Process a single sample."""
    async with semaphore:
        sid = sample.sample_id
        run_result = None
        try:
            if hasattr(sample, "ensure_frames_loaded"):
                sample.ensure_frames_loaded()

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
                defer_report=True,
            )
            answer_text = run_result.get("final_answer", {}).get("text", "")
        except Exception as exc:
            import traceback
            print(f"[Error] Sample {sid}: {exc}")
            traceback.print_exc()
            answer_text = ""

        async with lock:
            predictions[sid] = answer_text
            gt = getattr(sample, "answer", None)
            entry = {"sample_id": str(sid), "content": answer_text}
            if gt is not None:
                entry["ground_truth"] = str(gt)
            score = benchmark.evaluate_single(sample, answer_text)
            if score is not None:
                entry["result"] = score
            if run_result and "usage" in run_result:
                entry["usage"] = run_result["usage"]
            with open(pred_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

            report_ctx = run_result.get("_report_context") if run_result else None
            if report_ctx and workflow.config.generate_report:
                try:
                    workflow.generate_report(
                        session_dir=report_ctx["session_dir"],
                        session_id=str(sid),
                        instruction=report_ctx["instruction"],
                        images=report_ctx["images"],
                        ground_truth=gt,
                        final_state=report_ctx["final_state"],
                        result_score=score,
                    )
                except Exception:
                    pass


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

    # Default work_dir lives next to the spatial_agent package, not cwd.
    if not config.work_dir:
        _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_short = config.llm_model.split("/")[-1][:30] if config.llm_model else "unknown"
        dir_name = f"spatial_{config.benchmark}_{model_short}"
        config.work_dir = os.path.join(_pkg_dir, "work_dir", dir_name)
    os.makedirs(config.work_dir, exist_ok=True)

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

    print(f"Benchmark: {benchmark.__class__.__name__} ({len(benchmark)} samples)")

    pred_file = os.path.join(config.work_dir, "predictions.jsonl")
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
    elif not args.resume:
        # Fresh run: clear stale predictions and session logs before re-running.
        if os.path.exists(pred_file):
            os.remove(pred_file)
        for entry in os.listdir(config.work_dir):
            if entry.startswith("session-"):
                shutil.rmtree(os.path.join(config.work_dir, entry), ignore_errors=True)

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
