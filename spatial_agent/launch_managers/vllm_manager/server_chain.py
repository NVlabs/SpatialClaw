"""
Background server chain manager.

Each chain runs as an independent subprocess that manages a continuous loop
of 4-hour SLURM jobs with overlap for zero-downtime restarts.

Can be invoked directly:
    python -m spatial_agent.launch_managers.vllm_manager.server_chain --config '{"chain_id": ..., ...}'
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
from typing import List, Optional

from spatial_agent.launch_managers.slurm_utils import (
    batch_query_jobs,
    cancel_job,
    cancel_jobs,
    check_slurm_available,
    filter_alive_jobs,
    get_job_status,
    is_job_alive,
    wait_for_job_visible,
)

_ALIVE_STATUSES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}
from spatial_agent.launch_managers.vllm_manager.state import ChainState, ChainStateManager, FileLock


# Stop a chain after this many consecutive jobs that died within
# _FAST_FAIL_SECONDS of starting (e.g. model load crash).
_MAX_FAST_FAILURES = 3
_FAST_FAIL_SECONDS = 300


class ServerChain:
    """Manages continuous chain of 4-hour SLURM jobs with overlap."""

    def __init__(
        self,
        chain_id: str,
        served_name: str,
        model_name: str,
        model_path: str,
        account: str,
        partition: str,
        max_model_len: int,
        max_num_seqs: int,
        tp_size: int,
        kv_cache_dtype: str = "auto",
        quantization: str = "none",
        gpus: int = 8,
        time_limit: str = "4:00:00",
        restart_before_minutes: int = 20,
        project_root: str = ".",
    ):
        self.chain_id = chain_id
        self.served_name = served_name
        self.model_name = model_name
        self.model_path = model_path
        self.account = account
        self.partition = partition
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.tp_size = tp_size
        self.kv_cache_dtype = kv_cache_dtype
        self.quantization = quantization
        self.gpus = gpus
        self.time_limit = time_limit
        self.restart_before_minutes = restart_before_minutes
        self.project_root = Path(project_root)

        self.total_seconds = self._parse_time(time_limit)
        self.wait_seconds = self.total_seconds - (restart_before_minutes * 60)

        self.running = True
        self._stop_event = threading.Event()
        self.job_counter = 1
        self.current_job_id: Optional[str] = None
        self.job_submit_time: Optional[float] = None
        self.submitted_job_ids: List[str] = []

        self.log_dir = self.project_root / "spatial_agent" / "logs" / "slurm_vllm"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.vllm_script = Path(__file__).parent / "run_vllm.sh"
        self.state_manager = ChainStateManager(self.project_root)

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

    def _format_time(self, seconds: int) -> str:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _log(self, msg: str, flush: bool = True):
        print(f"[{self._ts()}] {msg}", flush=flush)

    def _generate_sbatch(self, job_number: int) -> str:
        job_name = f"vllm-{self.served_name}"
        exclusive_line = "#SBATCH --exclusive\n" if self.gpus >= 8 else ""
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={self.gpus}
#SBATCH --mem-per-gpu=240G
#SBATCH --time={self.time_limit}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
{exclusive_line}
echo "=========================================="
echo "SLURM vLLM Server - Chain {self.chain_id[:8]} - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

cd {self.project_root}

bash {self.vllm_script} "{self.model_path}" "{self.served_name}" {self.max_model_len} {self.max_num_seqs} {self.gpus} {self.tp_size} {self.kv_cache_dtype} {self.quantization}

echo "End time: $(date)"
"""

    def _wait_for_job_to_start(self, job_id: str) -> bool:
        self._log(f"Waiting for job {job_id} to start...")
        # Force-refresh the squeue cache so a just-submitted job becomes
        # visible immediately (covers the brief sbatch→squeue lag).
        wait_for_job_visible(job_id)
        start = time.time()
        last_status = None

        while self.running:
            status = get_job_status(job_id)
            now = time.time()

            if status != last_status:
                elapsed_m = int((now - start) / 60)
                self._log(f"Job {job_id} status: {status} (waited {elapsed_m}m)")
                last_status = status

            if status == "RUNNING":
                return True
            elif status in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "NOT_FOUND"):
                self._log(f"Job {job_id} failed: {status}")
                return False

            if self._stop_event.wait(timeout=30):
                return False

        return False

    def submit_job(self) -> bool:
        self._log(f"Submitting job #{self.job_counter}...")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"vllm-{self.served_name}"
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

        if self._wait_for_job_to_start(job_id):
            self.job_submit_time = time.time()
            return True
        else:
            self._log(f"Cancelling failed job {job_id}...")
            cancel_job(job_id)
            return False

    def wait_with_progress(self) -> bool:
        wait_h = self.wait_seconds // 3600
        wait_m = (self.wait_seconds % 3600) // 60
        self._log(f"Waiting {wait_h}h {wait_m}m before overlap submission...")

        start = time.time()
        last_check = 0

        while self.running:
            if self._stop_event.wait(timeout=10):
                return False

            elapsed = int(time.time() - start)
            if elapsed >= self.wait_seconds:
                break

            if elapsed - last_check >= 60:
                last_check = elapsed

                if self.current_job_id and not is_job_alive(self.current_job_id):
                    elapsed_m = int((time.time() - self.job_submit_time) / 60) if self.job_submit_time else 0
                    self._log(f"Job {self.current_job_id} died after {elapsed_m}m!")
                    return False

                if elapsed % 600 < 60 and elapsed > 0:
                    remaining = self.wait_seconds - elapsed
                    self._log(f"Next submission in {self._format_time(remaining)}")
                    self._cleanup_serve_registry()
                    self.submitted_job_ids = filter_alive_jobs(self.submitted_job_ids)
                    self._update_state_jobs()

        return True

    def handle_shutdown(self, signum, frame):
        self._log("Received shutdown signal. Stopping chain...")
        self.running = False
        self._stop_event.set()

        running_jobs = filter_alive_jobs(self.submitted_job_ids)
        if running_jobs:
            self._log(f"Cancelling {len(running_jobs)} job(s)...")
            cancel_jobs(running_jobs)

    def _update_state_jobs(self):
        try:
            self.state_manager.update_chain_jobs(
                self.chain_id, list(self.submitted_job_ids),
            )
        except Exception as e:
            self._log(f"Warning: failed to update state: {e}")

    def _cleanup_serve_registry(self):
        serve_file = self.project_root / "spatial_agent" / "logs" / "serve.json"
        lock_file = str(serve_file) + ".lock"
        if not serve_file.exists():
            return

        try:
            # Phase 1: Read under lock to get a consistent snapshot
            with FileLock(lock_file):
                if not serve_file.exists():
                    return
                with open(serve_file, "r") as f:
                    registry = json.load(f)

            # Phase 2: Check liveness WITHOUT lock (squeue can be slow).
            # Collect every slurm_id first, do ONE batched squeue, then
            # decide what to remove — avoids one squeue per registry entry.
            slurm_ids = []
            for servers in registry.values():
                for info in servers.values():
                    sid = info.get("slurm_job_id")
                    if sid:
                        slurm_ids.append(sid)
            job_status = batch_query_jobs(slurm_ids) if slurm_ids else {}

            # Defensive: if we expected results but got none, controller may
            # have transiently failed — skip this cleanup round rather than
            # nuking every entry.
            if slurm_ids and not job_status:
                return

            to_remove = set()  # (model_name, server_id) pairs
            for model_name, servers in registry.items():
                for server_id, info in servers.items():
                    slurm_id = info.get("slurm_job_id")
                    should_remove = False

                    if slurm_id:
                        ji = job_status.get(str(slurm_id))
                        if ji is None or ji["status"].upper() not in _ALIVE_STATUSES:
                            should_remove = True
                    else:
                        create_str = info.get("create_time")
                        if create_str:
                            try:
                                ct = datetime.datetime.strptime(create_str, "%Y/%m/%d %H:%M:%S")
                                age_h = (datetime.datetime.now() - ct).total_seconds() / 3600
                                if age_h > 8:
                                    should_remove = True
                            except Exception:
                                should_remove = True

                    if should_remove:
                        to_remove.add((model_name, server_id))

            if not to_remove:
                return

            # Phase 3: Re-read under lock and apply only confirmed removals
            with FileLock(lock_file):
                if not serve_file.exists():
                    return
                with open(serve_file, "r") as f:
                    registry = json.load(f)

                removed = 0
                for model_name, server_id in to_remove:
                    if model_name in registry and server_id in registry[model_name]:
                        del registry[model_name][server_id]
                        removed += 1
                        if not registry[model_name]:
                            del registry[model_name]

                if removed > 0:
                    with open(serve_file, "w") as f:
                        json.dump(registry, f, indent=2, ensure_ascii=False)
                    self._log(f"Cleaned serve.json: removed {removed} dead entries")

        except Exception as e:
            self._log(f"Warning: serve.json cleanup failed: {e}")

    def run(self):
        if not check_slurm_available():
            self._log("ERROR: SLURM not available")
            sys.exit(1)

        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        self._log(f"Starting server chain for {self.model_name} ({self.served_name})")
        self._log(f"Chain ID: {self.chain_id}")
        self._log(f"Job duration: {self.time_limit}, overlap: {self.restart_before_minutes}m")

        self._cleanup_serve_registry()

        consecutive_fast_failures = 0

        while self.running:
            self._cleanup_serve_registry()

            submit_t = time.time()
            if not self.submit_job():
                # submit_job returns False on sbatch error or fail-to-start —
                # treat both as fast failures so a broken config doesn't
                # spawn jobs forever.
                consecutive_fast_failures += 1
                self._log(
                    f"Failed to submit/start "
                    f"({consecutive_fast_failures}/{_MAX_FAST_FAILURES} consecutive)."
                )
                if consecutive_fast_failures >= _MAX_FAST_FAILURES:
                    self._log(
                        f"Aborting chain: {_MAX_FAST_FAILURES} consecutive jobs "
                        f"failed to start. Check the latest job log."
                    )
                    break
                if self._stop_event.wait(timeout=60):
                    break
                continue

            self.job_counter += 1

            if self.running:
                if self.wait_with_progress():
                    consecutive_fast_failures = 0
                    self._log(f"Overlap window: submitting next job ({self.restart_before_minutes}m overlap)")
                else:
                    # Job died early. If it ran longer than the fast-fail
                    # threshold we treat it as a normal restart cycle.
                    elapsed = time.time() - submit_t
                    if elapsed < _FAST_FAIL_SECONDS:
                        consecutive_fast_failures += 1
                        self._log(
                            f"Job died after {int(elapsed)}s "
                            f"({consecutive_fast_failures}/{_MAX_FAST_FAILURES} consecutive fast failures)."
                        )
                        if consecutive_fast_failures >= _MAX_FAST_FAILURES:
                            self._log(
                                f"Aborting chain: {_MAX_FAST_FAILURES} consecutive jobs "
                                f"died within {_FAST_FAIL_SECONDS}s. Check the latest job log."
                            )
                            break
                    else:
                        consecutive_fast_failures = 0
                    self._log("Job failed early, submitting replacement...")

        # Deregister on exit
        try:
            self.state_manager.remove_chain(self.chain_id)
        except Exception:
            pass

        self._log("Chain manager stopped.")


def start_chain_background(
    chain_id: str,
    served_name: str,
    model_name: str,
    model_path: str,
    account: str,
    partition: str,
    max_model_len: int,
    max_num_seqs: int,
    tp_size: int,
    kv_cache_dtype: str = "auto",
    quantization: str = "none",
    gpus: int = 8,
    time_limit: str = "4:00:00",
    restart_before_minutes: int = 20,
    project_root: Path = Path("."),
) -> ChainState:
    """Start a server chain as an independent background process."""
    config = {
        "chain_id": chain_id,
        "served_name": served_name,
        "model_name": model_name,
        "model_path": model_path,
        "account": account,
        "partition": partition,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "tp_size": tp_size,
        "kv_cache_dtype": kv_cache_dtype,
        "quantization": quantization,
        "gpus": gpus,
        "time_limit": time_limit,
        "restart_before_minutes": restart_before_minutes,
        "project_root": str(project_root),
    }

    # Log file for the chain manager process itself
    log_dir = project_root / "spatial_agent" / "logs" / "slurm_vllm"
    log_dir.mkdir(parents=True, exist_ok=True)
    chain_log = log_dir / f"chain_{served_name}_{chain_id[:8]}.log"

    log_f = open(chain_log, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "spatial_agent.launch_managers.vllm_manager.server_chain",
            "--config", json.dumps(config),
        ],
        stdout=log_f,
        stderr=log_f,
        start_new_session=True,  # Survives CLI exit
        cwd=str(project_root),
    )

    started_at = datetime.datetime.now().isoformat()
    chain_state = ChainState(
        chain_id=chain_id,
        served_name=served_name,
        model_name=model_name,
        model_path=model_path,
        pid=proc.pid,
        slurm_job_ids=[],
        started_at=started_at,
        account=account,
        partition=partition,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        tp_size=tp_size,
        gpus=gpus,
    )

    # Register in persistent state
    state_manager = ChainStateManager(project_root)
    state_manager.add_chain(chain_state)

    return chain_state, chain_log


def main():
    parser = argparse.ArgumentParser(description="Server chain subprocess")
    parser.add_argument("--config", type=str, required=True, help="JSON config string")
    args = parser.parse_args()

    config = json.loads(args.config)
    chain = ServerChain(**config)
    chain.run()


if __name__ == "__main__":
    main()
