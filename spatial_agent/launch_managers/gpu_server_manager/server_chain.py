"""
Background GPU server chain manager.

Each chain runs as an independent subprocess that manages a continuous loop
of 4-hour SLURM jobs with overlap for zero-downtime restarts.

Can be invoked directly:
    python -m spatial_agent.launch_managers.gpu_server_manager.server_chain --config '{"chain_id": ..., ...}'
"""

import argparse
import datetime
import fcntl
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
    cancel_job,
    cancel_jobs,
    check_slurm_available,
    filter_alive_jobs,
    get_job_status,
    is_job_alive,
    wait_for_job_visible,
)
from spatial_agent.launch_managers.gpu_server_manager.state import (
    GPUServerState,
    GPUServerStateManager,
)


# Stop a chain after this many consecutive jobs that died within
# _FAST_FAIL_SECONDS of starting (e.g. server crashes immediately).
_MAX_FAST_FAILURES = 3
_FAST_FAIL_SECONDS = 300


class ServerChain:
    """Manages continuous chain of 4-hour SLURM jobs for one GPU server."""

    def __init__(
        self,
        chain_id: str,
        account: str,
        partition: str,
        gpus: int = 1,
        reconstruct_backend: str = "pi3",
        time_limit: str = "4:00:00",
        restart_before_minutes: int = 20,
        project_root: str = ".",
    ):
        self.chain_id = chain_id
        self.account = account
        self.partition = partition
        self.gpus = gpus
        self.reconstruct_backend = reconstruct_backend
        self.time_limit = time_limit
        self.restart_before_minutes = restart_before_minutes
        self.project_root = Path(project_root)

        self.total_seconds = self._parse_time(time_limit)
        self.wait_seconds = self.total_seconds - (restart_before_minutes * 60)

        self.running = True
        self._stop_event = threading.Event()  # For fast shutdown
        self.job_counter = 1
        self.current_job_id: Optional[str] = None
        self.job_submit_time: Optional[float] = None
        self.submitted_job_ids: List[str] = []

        self.log_dir = self.project_root / "spatial_agent" / "logs" / "slurm_gpu_server"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.gpu_server_script = Path(__file__).parent / "run_gpu_server.sh"
        self.state_manager = GPUServerStateManager(self.project_root)

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
        job_name = f"gpu-{self.chain_id[:8]}"
        exclusive_line = "#SBATCH --exclusive\n" if self.gpus >= 8 else ""
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={self.gpus}
#SBATCH --mem-per-gpu=228G
#SBATCH --time={self.time_limit}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
{exclusive_line}
echo "=========================================="
echo "GPU Server - Chain {self.chain_id[:8]} - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

cd {self.project_root}

bash {self.gpu_server_script} {self.gpus} {self.reconstruct_backend}

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
                return False  # Signalled to stop

        return False

    def submit_job(self) -> bool:
        self._log(f"Submitting job #{self.job_counter}...")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"gpu-{self.chain_id[:8]}"
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
            # Sleep in short bursts so SIGTERM is handled promptly
            if self._stop_event.wait(timeout=10):
                return False  # Signalled to stop

            elapsed = int(time.time() - start)
            if elapsed >= self.wait_seconds:
                break

            # Periodic checks every ~60s
            if elapsed - last_check >= 60:
                last_check = elapsed

                if self.current_job_id and not is_job_alive(self.current_job_id):
                    elapsed_m = int((time.time() - self.job_submit_time) / 60) if self.job_submit_time else 0
                    self._log(f"Job {self.current_job_id} died after {elapsed_m}m!")
                    return False

                if elapsed % 600 < 60 and elapsed > 0:
                    remaining = self.wait_seconds - elapsed
                    self._log(f"Next submission in {self._format_time(remaining)}")
                    self._cleanup_gpu_registry()
                    self.submitted_job_ids = filter_alive_jobs(self.submitted_job_ids)
                    self._update_state_jobs()

        return True

    def handle_shutdown(self, signum, frame):
        self._log("Received shutdown signal. Stopping chain...")
        self.running = False
        self._stop_event.set()  # Wake up any sleeping waits immediately

        running_jobs = filter_alive_jobs(self.submitted_job_ids)
        if running_jobs:
            self._log(f"Cancelling {len(running_jobs)} job(s)...")
            cancel_jobs(running_jobs)

    def _update_state_jobs(self):
        try:
            self.state_manager.update_server_jobs(
                self.chain_id, list(self.submitted_job_ids),
            )
        except Exception as e:
            self._log(f"Warning: failed to update state: {e}")

    def _cleanup_gpu_registry(self):
        """Remove dead entries from gpu_server.json."""
        reg_file = self.project_root / "spatial_agent" / "logs" / "gpu_server.json"
        lock_file = self.project_root / "spatial_agent" / "logs" / "gpu_server.json.lock"
        if not reg_file.exists():
            return

        try:
            # Batch-query all SLURM job IDs BEFORE taking the lock
            # (squeue can be slow; don't hold the lock during I/O).
            with open(reg_file, "r") as f:
                preview = json.load(f)

            slurm_ids = [
                info.get("slurm_job_id")
                for info in preview.values()
                if info.get("slurm_job_id")
            ]
            if not slurm_ids:
                return

            from spatial_agent.launch_managers.slurm_utils import batch_query_jobs
            job_status = batch_query_jobs(slurm_ids)

            # If squeue returned nothing at all, it likely failed transiently
            # (SLURM controller overloaded, network blip).  Do NOT treat every
            # job as dead — just skip cleanup this round.
            if not job_status and slurm_ids:
                return

            alive_statuses = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}

            # Take the file lock and re-read (another process may have
            # modified the file between our preview read and now).
            lock_f = open(lock_file, "a+")
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

                try:
                    with open(reg_file, "r") as f:
                        registry = json.load(f)
                except (json.JSONDecodeError, OSError):
                    return

                removed = 0
                for uid in list(registry.keys()):
                    info = registry[uid]
                    slurm_id = info.get("slurm_job_id")
                    should_remove = False

                    if slurm_id:
                        if slurm_id not in set(slurm_ids):
                            # Entry was added between preview and lock — we
                            # didn't query this job, so leave it alone.
                            continue
                        ji = job_status.get(slurm_id)
                        if ji:
                            if ji["status"].upper() not in alive_statuses:
                                should_remove = True
                        else:
                            # Job not in squeue at all — likely finished.
                            # But only if we got SOME results back (guarded above).
                            should_remove = True
                    else:
                        pid = info.get("pid")
                        if pid:
                            try:
                                os.kill(pid, 0)
                            except (ProcessLookupError, PermissionError, OSError):
                                should_remove = True

                    if should_remove:
                        del registry[uid]
                        removed += 1

                if removed > 0:
                    with open(reg_file, "w") as f:
                        json.dump(registry, f, indent=2)
                    self._log(f"Cleaned gpu_server.json: removed {removed} dead entries")
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                lock_f.close()

        except Exception as e:
            self._log(f"Warning: registry cleanup failed: {e}")

    def run(self):
        if not check_slurm_available():
            self._log("ERROR: SLURM not available")
            sys.exit(1)

        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        self._log(f"Starting GPU server chain ({self.gpus} GPU(s), backend={self.reconstruct_backend})")
        self._log(f"Chain ID: {self.chain_id}")
        self._log(f"Job duration: {self.time_limit}, overlap: {self.restart_before_minutes}m")

        self._cleanup_gpu_registry()

        consecutive_fast_failures = 0

        while self.running:
            self._cleanup_gpu_registry()

            submit_t = time.time()
            if not self.submit_job():
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
            self.state_manager.remove_server(self.chain_id)
        except Exception:
            pass

        self._log("Chain manager stopped.")


def start_chain_background(
    chain_id: str,
    account: str,
    partition: str,
    gpus: int = 1,
    reconstruct_backend: str = "pi3",
    time_limit: str = "4:00:00",
    restart_before_minutes: int = 20,
    project_root: Path = Path("."),
):
    """Start a GPU server chain as an independent background process.

    Returns (GPUServerState, chain_log_path).
    """
    config = {
        "chain_id": chain_id,
        "account": account,
        "partition": partition,
        "gpus": gpus,
        "reconstruct_backend": reconstruct_backend,
        "time_limit": time_limit,
        "restart_before_minutes": restart_before_minutes,
        "project_root": str(project_root),
    }

    log_dir = project_root / "spatial_agent" / "logs" / "slurm_gpu_server"
    log_dir.mkdir(parents=True, exist_ok=True)
    chain_log = log_dir / f"chain_{chain_id[:8]}.log"

    log_f = open(chain_log, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m",
            "spatial_agent.launch_managers.gpu_server_manager.server_chain",
            "--config", json.dumps(config),
        ],
        stdout=log_f,
        stderr=log_f,
        start_new_session=True,
        cwd=str(project_root),
    )

    started_at = datetime.datetime.now().isoformat()
    server_state = GPUServerState(
        chain_id=chain_id,
        pid=proc.pid,
        slurm_job_ids=[],
        started_at=started_at,
        account=account,
        partition=partition,
        gpus=gpus,
        reconstruct_backend=reconstruct_backend,
    )

    state_manager = GPUServerStateManager(project_root)
    state_manager.add_server(server_state)

    return server_state, chain_log


def main():
    parser = argparse.ArgumentParser(description="GPU server chain subprocess")
    parser.add_argument("--config", type=str, required=True, help="JSON config string")
    args = parser.parse_args()

    config = json.loads(args.config)
    chain = ServerChain(**config)
    chain.run()


if __name__ == "__main__":
    main()
