"""
Background experiment chain manager.

Each chain runs as an independent subprocess that manages a continuous loop
of sequential 4-hour SLURM jobs with automatic resume for one benchmark.

Can be invoked directly:
    python -m spatial_agent.launch_managers.agent_manager.experiment_chain --config '{"experiment_id": ..., ...}'
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from spatial_agent.launch_managers.slurm_utils import (
    cancel_job,
    cancel_jobs,
    check_slurm_available,
    filter_alive_jobs,
    get_job_status,
    wait_for_job_visible,
)
from spatial_agent.launch_managers.agent_manager import dispatcher as agent_dispatcher


# Stop a chain after this many consecutive jobs that ended without
# producing any new predictions (e.g. dataloader crash early in startup).
# 3 strikes tolerates transient SLURM issues (NODE_FAIL, preemption) on
# an otherwise healthy chain without killing a long-running experiment.
_MAX_NO_PROGRESS_JOBS = 3
from spatial_agent.launch_managers.agent_manager.state import ExperimentState, ExperimentStateManager


class ExperimentChain:
    """Manages continuous chain of sequential 4-hour SLURM jobs for one experiment."""

    def __init__(
        self,
        experiment_id: str,
        benchmark: str,
        model_name: str,
        experiment_name: str,
        account: str,
        partition: str,
        gpus: int,
        time_limit: str,
        concurrency: int,
        subsample: int,
        project_root: str,
        experiment_type: str = "agent",
        max_frames: int = 0,
        system_prompt: str = "cot",
        scheduled_for: str = "",
    ):
        self.experiment_id = experiment_id
        self.benchmark = benchmark
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.account = account
        self.partition = partition
        self.gpus = gpus
        self.time_limit = time_limit
        self.concurrency = concurrency
        self.subsample = subsample
        self.project_root = Path(project_root)
        self.experiment_type = experiment_type
        self.max_frames = max_frames
        self.system_prompt = system_prompt
        self.scheduled_for = scheduled_for

        self.total_seconds = self._parse_time(time_limit)

        self.running = True
        self._stop_event = threading.Event()
        self.job_counter = 1
        self.current_job_id: Optional[str] = None
        self.submitted_job_ids: List[str] = []

        # Paths
        self.model_config = f"spatial_agent/config/model/{model_name}.json"
        self.dataset_config = f"spatial_agent/config/dataset/{benchmark}.json"

        if experiment_type == "cot":
            self.log_dir = self.project_root / "spatial_agent" / "logs" / "slurm_cot"
        else:
            self.agent_script = self.project_root / "spatial_agent" / "scripts" / "agent" / "slurm" / "run.sh"
            self.log_dir = self.project_root / "spatial_agent" / "logs" / "slurm_agent"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Derive work_dir to match run.py + run.sh logic
        self.work_dir = self._derive_work_dir()

        self.state_manager = ExperimentStateManager(self.project_root)

        self._cached_total: Optional[int] = None

    def _parse_time(self, time_str: str) -> int:
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = map(int, parts)
        elif len(parts) == 2:
            h = 0
            m, s = map(int, parts)
        else:
            raise ValueError(f"Invalid time format: {time_str}")
        return h * 3600 + m * 60 + s

    def _ts(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log(self, msg: str):
        print(f"[{self._ts()}] {msg}", flush=True)

    def _derive_work_dir(self) -> str:
        """Derive work_dir matching run.py / cot_baseline.py logic."""
        model_config_path = self.project_root / self.model_config
        model_short = "unknown"
        if model_config_path.exists():
            try:
                with open(model_config_path) as f:
                    cfg = json.load(f)
                model_short = cfg["llm_model"].split("/")[-1][:30]
            except Exception:
                pass

        prefix = "cot" if self.experiment_type == "cot" else "spatial"
        pkg_dir = self.project_root / "spatial_agent"
        work_dir = pkg_dir / "work_dir" / f"{prefix}_{self.benchmark}_{model_short}"
        if self.experiment_name:
            work_dir = Path(f"{work_dir}_{self.experiment_name}")
        return str(work_dir)

    def _count_predictions(self) -> int:
        """Count completed samples from predictions.jsonl."""
        pred_file = os.path.join(self.work_dir, "predictions.jsonl")
        if not os.path.exists(pred_file):
            return 0
        try:
            with open(pred_file) as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    def _get_total_samples(self) -> int:
        """Get total number of samples. Caches after first successful load."""
        if self._cached_total is not None:
            return self._cached_total

        # 1. Check if the CLI already computed total_samples in the state file
        for exp in self.state_manager.list_experiments():
            if exp.experiment_id == self.experiment_id and exp.total_samples > 0:
                self._cached_total = exp.total_samples
                self._log(f"Total samples (from state): {exp.total_samples}")
                return exp.total_samples

        # 2. Try computing from benchmark factory (may fail in background process)
        try:
            from spatial_agent.evals.factory import BenchmarkFactory
            benchmark = BenchmarkFactory.create_benchmark(self.benchmark)
            if benchmark is not None:
                total = len(benchmark)
                if self.subsample > 0:
                    total = min(total, self.subsample)
                self._cached_total = total
                self.state_manager.update_experiment_total(
                    self.experiment_id, total
                )
                self._log(f"Total samples: {total}")
                return total
        except Exception as e:
            self._log(f"Warning: could not determine total samples: {e}")

        # 3. If subsample is set, use it as best guess
        if self.subsample > 0:
            self._cached_total = self.subsample
            return self.subsample

        return -1

    def _is_experiment_complete(self) -> bool:
        """Check if all samples have been processed."""
        total = self._get_total_samples()
        if total <= 0:
            return False
        completed = self._count_predictions()
        return completed >= total

    def _generate_sbatch(self, job_number: int) -> str:
        if self.experiment_type == "cot":
            return self._generate_cot_sbatch(job_number)
        return self._generate_agent_sbatch(job_number)

    def _generate_agent_sbatch(self, job_number: int) -> str:
        job_name = f"spatial-{self.benchmark}"
        gpu_line = f"#SBATCH --gpus-per-node={self.gpus}" if self.gpus > 0 else ""
        exclusive = "" if ("interactive" in self.partition or self.gpus == 0) else "#SBATCH --exclusive"
        mem_gb = min(max(self.concurrency * 4, 16), 48)
        cpus = min(self.concurrency + 2, 8)
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
{gpu_line}
#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={self.time_limit}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
{exclusive}

echo "=========================================="
echo "Spatial Agent Evaluation - Chain {self.experiment_id[:8]} - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Benchmark: {self.benchmark}"
echo "Concurrency: {self.concurrency}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

cd {self.project_root}

CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent

bash {self.agent_script} "{self.benchmark}" {self.concurrency} \\
    "{self.model_config}" "{self.dataset_config}" "{self.experiment_name}" {self.subsample}

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
"""

    def _generate_cot_sbatch(self, job_number: int) -> str:
        job_name = f"cot-{self.benchmark}"
        exclusive = ""
        gpu_line = f"#SBATCH --gpus-per-node={self.gpus}" if self.gpus > 0 else ""
        mem_gb = 8
        cpus = 4

        # Build the python command
        cmd_parts = [
            "python -m spatial_agent.entrypoints.cot_baseline",
            f"--dataset {self.dataset_config}",
            f"--model {self.model_config}",
            f"--concurrency {self.concurrency}",
            "--resume",
        ]
        if self.max_frames > 0:
            cmd_parts.append(f"--max_frames {self.max_frames}")
        if self.system_prompt:
            cmd_parts.append(f"--system_prompt {self.system_prompt}")
        if self.subsample > 0:
            cmd_parts.append(f"--subsample {self.subsample}")
        if self.work_dir:
            cmd_parts.append(f"--work_dir {self.work_dir}")

        cmd = " \\\n    ".join(cmd_parts)

        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
{gpu_line}
#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={self.time_limit}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
{exclusive}

echo "=========================================="
echo "CoT Baseline Evaluation - Chain {self.experiment_id[:8]} - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Benchmark: {self.benchmark}"
echo "Concurrency: {self.concurrency}"
echo "Max Frames: {self.max_frames}"
echo "System Prompt: {self.system_prompt}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

cd {self.project_root}

# Disable core dumps
ulimit -c 0
export PYTHONUNBUFFERED=1

CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent

{cmd}

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
"""

    def _wait_for_job_to_start(self, job_id: str, timeout: int = 7200) -> bool:
        """Wait for a SLURM job to start running."""
        self._log(f"Waiting for job {job_id} to start...")
        # Force-refresh the squeue cache so a just-submitted job becomes
        # visible immediately (covers the brief sbatch→squeue lag).
        wait_for_job_visible(job_id)
        start = time.time()
        last_status = None
        last_update_time = start

        while self.running:
            status = get_job_status(job_id)
            now = time.time()

            if status != last_status:
                elapsed_m = int((now - start) / 60)
                self._log(f"Job {job_id} status: {status} (waited {elapsed_m}m)")
                last_status = status
                last_update_time = now

            if status in ("PENDING", "CONFIGURING") and now - last_update_time >= 300:
                elapsed_m = int((now - start) / 60)
                self._log(f"Still waiting for job {job_id}... ({elapsed_m}m)")
                last_update_time = now

            if status == "RUNNING":
                return True
            elif status in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "NOT_FOUND"):
                self._log(f"Job {job_id} failed: {status}")
                return False

            if now - start > timeout:
                self._log(f"Timeout waiting for job {job_id} to start")
                return False

            if self._stop_event.wait(timeout=30):
                return False

        return False

    def _wait_for_job_completion(self, job_id: str) -> bool:
        """Wait for a SLURM job to complete. Returns True if completed."""
        self._log(f"Waiting for job {job_id} to complete...")
        start = time.time()
        last_update_time = start

        while self.running:
            status = get_job_status(job_id)
            now = time.time()

            # Periodic progress updates every 10 minutes
            if now - last_update_time >= 600:
                elapsed_m = int((now - start) / 60)
                completed = self._count_predictions()
                total = self._get_total_samples()
                if total > 0:
                    self._log(f"Job {job_id} running ({elapsed_m}m) — progress: {completed}/{total}")
                else:
                    self._log(f"Job {job_id} running ({elapsed_m}m) — completed: {completed}")
                last_update_time = now

            if status == "RUNNING":
                if self._stop_event.wait(timeout=60):
                    return False
                continue
            elif status in ("COMPLETED", "NOT_FOUND"):
                # NOT_FOUND means job left the queue — check sacct
                if status == "NOT_FOUND":
                    cmd = ["sacct", "-j", str(job_id), "-n", "-o", "State"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        sacct_status = result.stdout.strip().split()[0].upper()
                        if "COMPLETED" not in sacct_status:
                            self._log(f"Job {job_id} ended: {sacct_status}")
                            return False
                elapsed_m = int((now - start) / 60)
                self._log(f"Job {job_id} completed (ran {elapsed_m}m)")
                return True
            elif status in ("FAILED", "TIMEOUT", "NODE_FAIL", "CANCELLED"):
                elapsed_m = int((now - start) / 60)
                self._log(f"Job {job_id} ended: {status} (after {elapsed_m}m)")
                return False
            else:
                if self._stop_event.wait(timeout=60):
                    return False

        return False

    def _build_agent_script_args(self) -> List[str]:
        """Args matching the `bash run.sh ...` invocation in _generate_agent_sbatch."""
        return [
            self.benchmark,
            str(self.concurrency),
            self.model_config,
            self.dataset_config,
            self.experiment_name,
            str(self.subsample),
        ]

    def submit_job(self) -> bool:
        """Submit a SLURM job."""
        self._log(f"Submitting job #{self.job_counter}...")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "cot" if self.experiment_type == "cot" else "spatial"
        job_name = f"{prefix}-{self.benchmark}"
        script_path = self.log_dir / f"{job_name}_{timestamp}.sbatch"

        script_content = self._generate_sbatch(self.job_counter)
        with open(script_path, "w") as f:
            f.write(script_content)
        script_path.chmod(0o755)

        result = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True,
        )

        if result.returncode != 0:
            self._log(f"Failed to submit job: {result.stderr}")
            return False

        job_id = None
        for word in result.stdout.split():
            if word.isdigit():
                job_id = word
                break

        self.current_job_id = job_id
        self.submitted_job_ids.append(job_id)
        self._update_state_jobs()

        self._log(f"Job #{self.job_counter} submitted: SLURM ID {job_id}")
        self._log(f"Log: {self.log_dir}/{job_name}_{job_id}.out")

        return True

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self._log("Received shutdown signal. Stopping chain...")
        self.running = False
        self._stop_event.set()

        running_jobs = filter_alive_jobs(self.submitted_job_ids)
        if running_jobs:
            self._log(f"Cancelling {len(running_jobs)} job(s)...")
            cancel_jobs(running_jobs)

    def _update_state_jobs(self):
        try:
            self.state_manager.update_experiment_jobs(
                self.experiment_id, list(self.submitted_job_ids),
            )
        except Exception as e:
            self._log(f"Warning: failed to update state: {e}")

    def _update_status(self, status: str):
        try:
            self.state_manager.update_experiment_status(self.experiment_id, status)
        except Exception as e:
            self._log(f"Warning: failed to update status: {e}")

    def _wait_until_scheduled(self) -> bool:
        """Block until self.scheduled_for (ISO) is reached. Returns False if stopped."""
        if not self.scheduled_for:
            return True
        try:
            target = datetime.datetime.fromisoformat(self.scheduled_for).timestamp()
        except ValueError:
            self._log(f"Warning: unparseable scheduled_for={self.scheduled_for!r}, starting now.")
            return True

        remaining = target - time.time()
        if remaining <= 0:
            return True

        self._log(f"Scheduled for {self.scheduled_for} (in {int(remaining // 60)}m {int(remaining % 60)}s).")
        while self.running:
            remaining = target - time.time()
            if remaining <= 0:
                break
            if remaining > 60:
                self._log(f"Waiting for scheduled start: {int(remaining // 60)}m remaining.")
            if self._stop_event.wait(timeout=min(60.0, remaining)):
                return False
        if not self.running:
            return False
        self._log("Scheduled time reached — starting chain.")
        return True

    def run(self):
        """Run the continuous job submission loop."""
        if not check_slurm_available():
            self._log("ERROR: SLURM not available")
            sys.exit(1)

        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        type_label = "CoT baseline" if self.experiment_type == "cot" else "Agent"
        self._log(f"Starting {type_label} experiment chain: {self.benchmark} ({self.experiment_name})")
        self._log(f"Experiment ID: {self.experiment_id}")
        self._log(f"Model: {self.model_name}")
        self._log(f"Work dir: {self.work_dir}")
        self._log(f"Concurrency: {self.concurrency}, Subsample: {self.subsample or 'all'}")
        self._log(f"Job duration: {self.time_limit} (sequential, no overlap)")
        if self.experiment_type == "cot":
            self._log(f"Max frames: {self.max_frames}, System prompt: {self.system_prompt}")

        if not self._wait_until_scheduled():
            self._log("Chain manager stopped before scheduled start.")
            return

        consecutive_no_progress = 0
        terminal_status: Optional[str] = None

        while self.running:
            # Check if experiment is already complete
            if self._is_experiment_complete():
                self._update_status("completed")
                terminal_status = "completed"
                completed = self._count_predictions()
                total = self._get_total_samples()
                self._log(f"All samples completed! ({completed}/{total})")
                break

            predictions_before = self._count_predictions()

            # Try overlay first (agent type only). On success the agent ran
            # to completion as an `srun --overlap` step inside an existing
            # vLLM/gpu_server SLURM job — no new sbatch, no wait phases.
            # On failure (no slot within backoff window) fall back to the
            # existing sbatch path.
            overlay_dispatched = False
            job_completed = False
            if self.experiment_type == "agent":
                overlay_args = self._build_agent_script_args()
                if agent_dispatcher.try_dispatch_overlay(
                    self.project_root,
                    str(self.agent_script),
                    overlay_args,
                    self.concurrency,
                    log_fn=self._log,
                    stop_event=self._stop_event,
                ):
                    overlay_dispatched = True
                    job_completed = True

            # If overlay returned False because stop_event was set during the
            # call, don't bother sbatching — we're shutting down.
            if not overlay_dispatched and not self.running:
                break

            if not overlay_dispatched:
                # Submit next job. Submission failures count toward the same
                # no-progress budget as jobs that produce zero predictions —
                # otherwise a bad account/partition would spin forever.
                if not self.submit_job():
                    consecutive_no_progress += 1
                    self._log(
                        f"Failed to submit "
                        f"({consecutive_no_progress}/{_MAX_NO_PROGRESS_JOBS} consecutive)."
                    )
                    if consecutive_no_progress >= _MAX_NO_PROGRESS_JOBS:
                        self._log(
                            f"Aborting chain: {_MAX_NO_PROGRESS_JOBS} consecutive "
                            f"submission failures. Check your account/partition/script."
                        )
                        self._update_status("failed")
                        terminal_status = "failed"
                        break
                    if self._stop_event.wait(timeout=60):
                        break
                    continue

                # Wait for job to start. Same policy — a job that never runs
                # can't make progress, so it counts against the budget.
                if not self._wait_for_job_to_start(self.current_job_id):
                    consecutive_no_progress += 1
                    self._log(
                        f"Job failed to start "
                        f"({consecutive_no_progress}/{_MAX_NO_PROGRESS_JOBS} consecutive)."
                    )
                    try:
                        cancel_job(self.current_job_id)
                    except Exception:
                        pass
                    self._prune_submitted_jobs()
                    if consecutive_no_progress >= _MAX_NO_PROGRESS_JOBS:
                        self._log(
                            f"Aborting chain: {_MAX_NO_PROGRESS_JOBS} consecutive jobs "
                            f"failed to start. Check the latest job log."
                        )
                        self._update_status("failed")
                        terminal_status = "failed"
                        break
                    if self._stop_event.wait(timeout=60):
                        break
                    continue

                # Wait for job to complete (sequential, no overlap)
                job_completed = self._wait_for_job_completion(self.current_job_id)

                if not job_completed:
                    self._log("Job ended abnormally, retrying in 60s...")

            # Detect a stuck experiment: job ended (success or fail) but
            # produced no new predictions. Most often a dataloader / startup
            # error that fails fast and would loop forever otherwise.
            predictions_after = self._count_predictions()
            progress = predictions_after - predictions_before
            if progress <= 0:
                consecutive_no_progress += 1
                self._log(
                    f"Job made no progress "
                    f"({consecutive_no_progress}/{_MAX_NO_PROGRESS_JOBS} consecutive)."
                )
                if consecutive_no_progress >= _MAX_NO_PROGRESS_JOBS:
                    self._log(
                        f"Aborting chain: {_MAX_NO_PROGRESS_JOBS} consecutive jobs "
                        f"produced no new predictions. Check the latest job log "
                        f"for the underlying error."
                    )
                    self._update_status("failed")
                    terminal_status = "failed"
                    break
            else:
                consecutive_no_progress = 0

            # Prune the submitted-job list so we don't keep thousands of
            # finished IDs (used to be cancelled redundantly on stop and
            # to bloat dashboard squeue queries).
            self._prune_submitted_jobs()

            if not job_completed:
                if self._stop_event.wait(timeout=60):
                    break

            self.job_counter += 1

            if self.running:
                completed = predictions_after
                total = self._get_total_samples()
                if total > 0:
                    self._log(f"Progress: {completed}/{total} — submitting next job...")
                else:
                    self._log(f"Completed: {completed} — submitting next job...")

        # Only remove from state on abnormal/in-progress exit. Keep
        # completed/failed entries so they're visible in the dashboard.
        if terminal_status is None:
            try:
                self.state_manager.remove_experiment(self.experiment_id)
            except Exception:
                pass

        self._log("Chain manager stopped.")

    def _prune_submitted_jobs(self) -> None:
        """Drop finished job IDs from the tracked list to keep it small.

        Uses one batched squeue call instead of N serial is_job_alive() calls.
        """
        alive = filter_alive_jobs(self.submitted_job_ids)
        if len(alive) != len(self.submitted_job_ids):
            self.submitted_job_ids = alive
            self._update_state_jobs()


def start_experiment_background(
    experiment_id: str,
    benchmark: str,
    model_name: str,
    experiment_name: str,
    account: str,
    partition: str,
    gpus: int,
    time_limit: str,
    concurrency: int,
    subsample: int,
    project_root: Path,
    experiment_type: str = "agent",
    max_frames: int = 0,
    system_prompt: str = "cot",
    defer_minutes: int = 0,
) -> Tuple[ExperimentState, Path]:
    """Start an experiment chain as an independent background process."""
    if defer_minutes > 0:
        scheduled_for = (
            datetime.datetime.now() + datetime.timedelta(minutes=defer_minutes)
        ).isoformat()
    else:
        scheduled_for = ""

    config = {
        "experiment_id": experiment_id,
        "benchmark": benchmark,
        "model_name": model_name,
        "experiment_name": experiment_name,
        "account": account,
        "partition": partition,
        "gpus": gpus,
        "time_limit": time_limit,
        "concurrency": concurrency,
        "subsample": subsample,
        "project_root": str(project_root),
        "experiment_type": experiment_type,
        "max_frames": max_frames,
        "system_prompt": system_prompt,
        "scheduled_for": scheduled_for,
    }

    # Derive work_dir (same logic as ExperimentChain._derive_work_dir)
    model_config_path = project_root / f"spatial_agent/config/model/{model_name}.json"
    model_short = "unknown"
    if model_config_path.exists():
        try:
            with open(model_config_path) as f:
                cfg = json.load(f)
            model_short = cfg["llm_model"].split("/")[-1][:30]
        except Exception:
            pass
    prefix = "cot" if experiment_type == "cot" else "spatial"
    pkg_dir = project_root / "spatial_agent"
    work_dir = str(pkg_dir / "work_dir" / f"{prefix}_{benchmark}_{model_short}")
    if experiment_name:
        work_dir = f"{work_dir}_{experiment_name}"

    # Compute total_samples eagerly (CLI has access to benchmark data)
    total_samples = -1
    try:
        from spatial_agent.evals.factory import BenchmarkFactory
        bench = BenchmarkFactory.create_benchmark(benchmark)
        if bench is not None:
            total_samples = len(bench)
            if subsample > 0:
                total_samples = min(total_samples, subsample)
    except Exception:
        # Fall back to subsample as best guess
        if subsample > 0:
            total_samples = subsample

    # Log file for the chain process
    log_subdir = "slurm_cot" if experiment_type == "cot" else "slurm_agent"
    log_dir = project_root / "spatial_agent" / "logs" / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    chain_log = log_dir / f"chain_{benchmark}_{experiment_name}_{experiment_id[:8]}.log"

    log_f = open(chain_log, "w")
    # stdin closed so the chain (and any srun it spawns) can't steal
    # keystrokes from the interactive agent-manager CLI.
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "spatial_agent.launch_managers.agent_manager.experiment_chain",
            "--config", json.dumps(config),
        ],
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=log_f,
        start_new_session=True,
        cwd=str(project_root),
    )

    started_at = datetime.datetime.now().isoformat()
    exp_state = ExperimentState(
        experiment_id=experiment_id,
        benchmark=benchmark,
        model_name=model_name,
        experiment_name=experiment_name,
        pid=proc.pid,
        slurm_job_ids=[],
        started_at=started_at,
        account=account,
        partition=partition,
        gpus=gpus,
        concurrency=concurrency,
        subsample=subsample,
        work_dir=work_dir,
        total_samples=total_samples,
        status="running",
        experiment_type=experiment_type,
        scheduled_for=scheduled_for,
    )

    state_manager = ExperimentStateManager(project_root)
    state_manager.add_experiment(exp_state)

    return exp_state, chain_log


def main():
    parser = argparse.ArgumentParser(description="Experiment chain subprocess")
    parser.add_argument("--config", type=str, required=True, help="JSON config string")
    args = parser.parse_args()

    config = json.loads(args.config)
    chain = ExperimentChain(**config)
    chain.run()


if __name__ == "__main__":
    main()
