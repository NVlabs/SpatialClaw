#!/usr/bin/env python3
"""
SLURM vLLM Server with Automatic 4-Hour Restarts

This script runs OUTSIDE SLURM and manages a continuous chain of 4-hour SLURM jobs
with 20-minute overlaps to ensure zero downtime.

Usage:
    python scripts/slurm_vllm_server.py
"""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from termcolor import colored


class SlurmVLLMManager:
    def __init__(
        self,
        job_name: str = "vllm-qwen3-vl",
        account: str = "nvr_taiwan_rvos",
        partition: str = "batch_singlenode,batch_block1,batch_block3,batch_block4",
        gpus: int = 8,
        time_limit: str = "4:00:00",
        restart_before_minutes: int = 20,
        vllm_script: str = "scripts/vllm/run.sh",
        output_dir: str = None,
        model: str = "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
        served_name: str = "qwen3-vl-235b-a22b-thinking",
        max_model_len: int = 49152,
        max_num_seqs: int = 64,
        tp_size: int = None,
        conda_env: str = "spatialclaw-cuda",
        venv_path: str = "",
    ):
        self.job_name = job_name
        self.account = account
        self.partition = partition
        self.gpus = gpus
        self.time_limit = time_limit
        self.restart_before_minutes = restart_before_minutes
        self.model = model
        self.served_name = served_name
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.tp_size = tp_size if tp_size is not None else gpus
        self.conda_env = conda_env
        self.venv_path = venv_path

        # Get the project root directory (go up 3 levels: slurm -> vllm -> scripts -> project)
        self.project_root = Path(__file__).parent.parent.parent.absolute()
        self.vllm_script_path = self.project_root / vllm_script

        if not self.vllm_script_path.exists():
            raise FileNotFoundError(f"vLLM script not found: {self.vllm_script_path}")

        # Set output directory
        if output_dir is None:
            self.output_dir = self.project_root / "logs" / "slurm_vllm"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate timings
        self.total_seconds = self._parse_time_to_seconds(time_limit)
        self.wait_seconds = self.total_seconds - (restart_before_minutes * 60)

        self.running = True
        self.job_counter = 1
        self.current_job_id = None
        self.job_submit_time = None
        self.submitted_job_ids = []  # Track all submitted jobs for cleanup

        print(colored("=" * 60, "cyan"))
        print(colored("SLURM vLLM Server Chain Manager", "cyan", attrs=["bold"]))
        print(colored("=" * 60, "cyan"))
        print(f"Job name:          {self.job_name}")
        print(f"Account:           {self.account}")
        print(f"Partition:         {self.partition}")
        print(f"GPUs:              {self.gpus}")
        print(f"Time limit:        {self.time_limit} ({self.total_seconds}s)")
        print(f"Restart before:    {self.restart_before_minutes} minutes")
        print(f"Wait time:         {self.wait_seconds // 60} minutes")
        print(f"vLLM script:       {self.vllm_script_path}")
        print(f"Output directory:  {self.output_dir}")
        print(colored("=" * 60, "cyan"))
        print()

    def _parse_time_to_seconds(self, time_str: str) -> int:
        """Parse time string (HH:MM:SS) to seconds."""
        parts = time_str.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = map(int, parts)
        else:
            raise ValueError(f"Invalid time format: {time_str}")
        return hours * 3600 + minutes * 60 + seconds

    def _check_slurm_available(self) -> bool:
        """Check if SLURM is available."""
        result = subprocess.run("which sbatch", shell=True, capture_output=True)
        return result.returncode == 0

    def _get_job_status(self, job_id: str) -> str:
        """Get the status of a SLURM job."""
        if not job_id:
            return "UNKNOWN"

        cmd = ["squeue", "-j", str(job_id), "-h", "-o", "%T"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return "NOT_FOUND"

        return result.stdout.strip().upper()

    def _is_job_running(self, job_id: str) -> bool:
        """Check if a SLURM job is still running."""
        status = self._get_job_status(job_id)
        return status in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]

    def _wait_for_job_to_start(self, job_id: str, timeout: int = 7200) -> bool:
        """
        Wait for a SLURM job to start running.
        Default timeout is 2 hours to handle long queue times.
        Returns True if job started, False if failed.
        """
        print(colored(
            f"[{self._get_timestamp()}] Waiting for job {job_id} to start running...",
            "cyan"
        ), flush=True)

        start_time = time.time()
        last_status = None
        last_update_time = start_time

        while self.running:
            status = self._get_job_status(job_id)
            current_time = time.time()

            # Print status changes
            if status != last_status:
                elapsed = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._get_timestamp()}] Job {job_id} status: {status} (waited {elapsed}m)",
                    "cyan"
                ), flush=True)
                last_status = status
                last_update_time = current_time

            # Print periodic updates every 5 minutes while pending
            if status in ["PENDING", "CONFIGURING"] and current_time - last_update_time >= 300:
                elapsed = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._get_timestamp()}] Still waiting for job {job_id}... ({elapsed}m elapsed)",
                    "cyan"
                ), flush=True)
                last_update_time = current_time

            if status == "RUNNING":
                elapsed = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._get_timestamp()}] ✓ Job {job_id} is now running! (waited {elapsed}m)",
                    "green"
                ), flush=True)
                return True
            elif status in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"]:
                print(colored(
                    f"[{self._get_timestamp()}] ✗ Job {job_id} failed with status: {status}",
                    "red"
                ), flush=True)
                return False
            elif status == "NOT_FOUND":
                # Job disappeared - probably cancelled or system issue
                print(colored(
                    f"[{self._get_timestamp()}] ✗ Job {job_id} not found in queue",
                    "red"
                ), flush=True)
                return False
            elif status in ["PENDING", "CONFIGURING"]:
                # Job is queued, keep waiting (no timeout for pending jobs)
                time.sleep(5)
            else:
                # Unknown status, wait a bit
                time.sleep(5)

        # self.running became False (Ctrl+C), exit gracefully
        return False

    def _cleanup_serve_registry(self):
        """Remove dead server entries from serve.json by checking SLURM job status."""
        serve_file = self.project_root / "logs" / "serve.json"

        if not serve_file.exists():
            return

        try:
            # Read current registry
            with open(serve_file, "r") as f:
                registry = json.load(f)

            # Track what we remove
            removed_count = 0
            total_count = 0

            # Check each model's servers
            for model_name, servers in list(registry.items()):
                for server_id, server_info in list(servers.items()):
                    total_count += 1
                    slurm_job_id = server_info.get("slurm_job_id")
                    pid = server_info.get("pid")

                    # Check if server should be removed
                    should_remove = False

                    if slurm_job_id:
                        # Check SLURM job status
                        if not self._is_job_running(slurm_job_id):
                            should_remove = True
                            reason = f"SLURM job {slurm_job_id} not running"
                    else:
                        # No SLURM job ID (old entry or non-SLURM), check age
                        # Remove entries older than 8 hours (2 job cycles)
                        create_time_str = server_info.get("create_time")
                        if create_time_str:
                            try:
                                create_time = datetime.datetime.strptime(create_time_str, "%Y/%m/%d %H:%M:%S")
                                age_hours = (datetime.datetime.now() - create_time).total_seconds() / 3600
                                if age_hours > 8:
                                    should_remove = True
                                    reason = f"old entry (>{age_hours:.1f}h)"
                            except Exception:
                                # Can't parse time, remove if no job ID
                                should_remove = True
                                reason = "no SLURM job ID and invalid timestamp"

                    if should_remove:
                        print(colored(
                            f"[{self._get_timestamp()}] Removing dead server: {model_name}/{server_id[:8]}... ({reason})",
                            "yellow"
                        ), flush=True)
                        del servers[server_id]
                        removed_count += 1

                # Remove model entry if no servers left
                if not servers:
                    del registry[model_name]

            # Write back if anything changed
            if removed_count > 0:
                with open(serve_file, "w") as f:
                    json.dump(registry, f, indent=2, ensure_ascii=False)

                print(colored(
                    f"[{self._get_timestamp()}] Cleaned up serve.json: removed {removed_count}/{total_count} dead servers",
                    "green"
                ), flush=True)

        except Exception as e:
            print(colored(
                f"[{self._get_timestamp()}] Warning: Failed to clean serve.json: {e}",
                "yellow"
            ), flush=True)

    def _format_time(self, seconds: int) -> str:
        """Format seconds as HH:MM:SS."""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _get_timestamp(self) -> str:
        """Get current timestamp for logging."""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _generate_slurm_script(self, job_number: int) -> str:
        """Generate SLURM batch script."""
        script = f"""#!/bin/bash
#SBATCH --job-name={self.job_name}
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={self.gpus}
#SBATCH --time={self.time_limit}
#SBATCH --output={self.output_dir}/{self.job_name}_%j.out
#SBATCH --error={self.output_dir}/{self.job_name}_%j.err
#SBATCH --exclusive

echo "=========================================="
echo "SLURM vLLM Server - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Navigate to project root
cd {self.project_root}

# Run the vLLM server (pass configuration as arguments)
bash {self.vllm_script_path} "{self.model}" "{self.served_name}" {self.max_model_len} {self.max_num_seqs} {self.gpus} {self.tp_size} "{self.conda_env}" "{self.venv_path}"

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
"""
        return script

    def submit_job(self) -> bool:
        """Submit a SLURM job."""
        print(colored(f"[{self._get_timestamp()}] Submitting job #{self.job_counter}...", "cyan"), flush=True)

        # Create SLURM batch script
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = self.output_dir / f"{self.job_name}_{timestamp}.sbatch"

        script_content = self._generate_slurm_script(self.job_counter)
        with open(script_path, "w") as f:
            f.write(script_content)

        script_path.chmod(0o755)

        # Submit the job
        cmd = ["sbatch", str(script_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # Extract job ID from output: "Submitted batch job 12345"
            job_id = None
            for word in result.stdout.split():
                if word.isdigit():
                    job_id = word
                    break

            # Track the current job
            self.current_job_id = job_id
            self.submitted_job_ids.append(job_id)

            print(colored(
                f"[{self._get_timestamp()}] ✓ Job #{self.job_counter} submitted: SLURM job ID {job_id}",
                "green"
            ), flush=True)
            print(colored(
                f"[{self._get_timestamp()}] Log file: {self.output_dir}/{self.job_name}_{job_id}.out",
                "cyan"
            ), flush=True)

            # Wait for job to start running before starting timer
            if self._wait_for_job_to_start(job_id):
                self.job_submit_time = time.time()
                return True
            else:
                # Job failed to start - cancel it before returning
                print(colored(
                    f"[{self._get_timestamp()}] Cancelling failed job {job_id}...",
                    "yellow"
                ), flush=True)
                try:
                    subprocess.run(["scancel", str(job_id)], capture_output=True)
                    print(colored(
                        f"[{self._get_timestamp()}] ✓ Cancelled job {job_id}",
                        "green"
                    ), flush=True)
                except Exception as e:
                    print(colored(
                        f"[{self._get_timestamp()}] Warning: Failed to cancel job {job_id}: {e}",
                        "yellow"
                    ), flush=True)
                return False
        else:
            print(colored(
                f"[{self._get_timestamp()}] ✗ Failed to submit job #{self.job_counter}",
                "red"
            ), flush=True)
            print(result.stderr, flush=True)
            return False

    def wait_with_progress(self) -> bool:
        """
        Wait for the specified time, showing progress.
        Returns True if wait completed normally, False if job died.
        """
        wait_hours = self.wait_seconds // 3600
        wait_minutes = (self.wait_seconds % 3600) // 60

        print(colored(
            f"[{self._get_timestamp()}] Waiting {wait_hours}h {wait_minutes}m before submitting next job...",
            "cyan"
        ), flush=True)

        elapsed = 0
        sleep_interval = 60  # Check every minute

        while elapsed < self.wait_seconds and self.running:
            time.sleep(min(sleep_interval, self.wait_seconds - elapsed))
            elapsed += sleep_interval

            # Check if current job is still running (every minute)
            if self.current_job_id and not self._is_job_running(self.current_job_id):
                # Job terminated unexpectedly
                elapsed_minutes = int((time.time() - self.job_submit_time) / 60)
                print(colored(
                    f"[{self._get_timestamp()}] ⚠ Job {self.current_job_id} terminated unexpectedly after {elapsed_minutes} minutes!",
                    "red"
                ), flush=True)
                print(colored(
                    f"[{self._get_timestamp()}] Submitting replacement job immediately...",
                    "yellow"
                ), flush=True)
                return False

            # Print progress every 10 minutes
            if elapsed % 600 == 0 and elapsed > 0 and elapsed < self.wait_seconds:
                remaining = self.wait_seconds - elapsed
                print(colored(
                    f"[{self._get_timestamp()}] Time until next job submission: {self._format_time(remaining)}",
                    "cyan"
                ), flush=True)
                # Also print job status
                if self.current_job_id:
                    print(colored(
                        f"[{self._get_timestamp()}] Current job {self.current_job_id} is still running ✓",
                        "green"
                    ), flush=True)
                # Clean up serve.json every 10 minutes
                self._cleanup_serve_registry()
                # Clean up finished jobs from tracking list
                self.submitted_job_ids = [job_id for job_id in self.submitted_job_ids if self._is_job_running(job_id)]

        return True

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(colored(
            f"\n[{self._get_timestamp()}] Received shutdown signal. Stopping chain...",
            "yellow"
        ), flush=True)

        self.running = False

        # Cancel only currently running jobs
        if self.submitted_job_ids:
            # Filter to only running jobs
            running_jobs = [job_id for job_id in self.submitted_job_ids if self._is_job_running(job_id)]
            
            if running_jobs:
                print(colored(
                    f"[{self._get_timestamp()}] Cancelling {len(running_jobs)} running job(s)...",
                    "yellow"
                ), flush=True)

                for job_id in running_jobs:
                    try:
                        result = subprocess.run(
                            ["scancel", str(job_id)],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode == 0:
                            print(colored(
                                f"[{self._get_timestamp()}] ✓ Cancelled job {job_id}",
                                "green"
                            ), flush=True)
                        else:
                            # Job might have just finished
                            if "Invalid job id" not in result.stderr:
                                print(colored(
                                    f"[{self._get_timestamp()}] Note: Job {job_id} - {result.stderr.strip()}",
                                    "yellow"
                                ), flush=True)
                    except Exception as e:
                        print(colored(
                            f"[{self._get_timestamp()}] Warning: Failed to cancel job {job_id}: {e}",
                            "yellow"
                        ), flush=True)

                print(colored(
                    f"[{self._get_timestamp()}] All running jobs cancelled.",
                    "green"
                ), flush=True)
            else:
                print(colored(
                    f"[{self._get_timestamp()}] No running jobs to cancel.",
                    "green"
                ), flush=True)

    def run(self):
        """Run the continuous job submission loop."""
        if not self._check_slurm_available():
            print(colored("ERROR: SLURM not available. Cannot submit jobs.", "red"))
            sys.exit(1)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        print(colored(f"[{self._get_timestamp()}] Starting continuous vLLM server chain...", "green", attrs=["bold"]), flush=True)
        print(colored(f"[{self._get_timestamp()}] Press Ctrl+C to stop the chain", "cyan"), flush=True)
        print()

        # Initial cleanup of stale server registry
        print(colored(f"[{self._get_timestamp()}] Cleaning up stale server registry...", "cyan"), flush=True)
        self._cleanup_serve_registry()
        print()

        while self.running:
            # Clean up stale server registry before submitting
            self._cleanup_serve_registry()

            # Submit the next job
            if not self.submit_job():
                print(colored(
                    f"[{self._get_timestamp()}] Failed to submit job, retrying in 60 seconds...",
                    "yellow"
                ), flush=True)
                time.sleep(60)
                continue

            self.job_counter += 1

            # Wait before submitting the next job
            if self.running:
                wait_completed = self.wait_with_progress()

                if not wait_completed:
                    # Job died early, submit replacement immediately
                    print(colored(
                        f"[{self._get_timestamp()}] Skipping wait period due to job failure",
                        "yellow"
                    ), flush=True)
                    print()
                    continue

            if self.running:
                print(colored(
                    f"[{self._get_timestamp()}] Time to submit next job ({self.restart_before_minutes}-minute overlap begins)",
                    "green"
                ), flush=True)
                print()

        print(colored(f"[{self._get_timestamp()}] Chain manager stopped.", "green", attrs=["bold"]), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Manage vLLM server with automatic SLURM restarts every 4 hours"
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="vllm-qwen3-vl",
        help="SLURM job name"
    )
    parser.add_argument(
        "--account",
        type=str,
        default="nvr_taiwan_rvos",
        help="SLURM account"
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="batch_singlenode,batch_block1,batch_block3,batch_block4",
        help="SLURM partition(s)"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=8,
        help="Number of GPUs to request"
    )
    parser.add_argument(
        "--time",
        type=str,
        default="4:00:00",
        help="SLURM time limit (HH:MM:SS)"
    )
    parser.add_argument(
        "--restart-before",
        type=int,
        default=20,
        help="Minutes before timeout to start next server"
    )
    parser.add_argument(
        "--vllm-script",
        type=str,
        default="scripts/vllm/run.sh",
        help="Path to vLLM server script"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for SLURM logs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
        help="Model name/path to serve"
    )
    parser.add_argument(
        "--served-name",
        type=str,
        default="qwen3-vl-235b-a22b-thinking",
        help="Served model name for API"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=49152,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=64,
        help="Maximum number of sequences"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallel size (defaults to --gpus if not set)"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="spatialclaw-cuda",
        help="Conda environment to use"
    )
    parser.add_argument(
        "--venv-path",
        type=str,
        default="",
        help="Path to .venv virtual environment (overrides --conda-env when set)"
    )

    args = parser.parse_args()

    # Create manager and run
    manager = SlurmVLLMManager(
        job_name=args.job_name,
        account=args.account,
        partition=args.partition,
        gpus=args.gpus,
        time_limit=args.time,
        restart_before_minutes=args.restart_before,
        vllm_script=args.vllm_script,
        output_dir=args.output_dir,
        model=args.model,
        served_name=args.served_name,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        tp_size=args.tp_size,
        conda_env=args.conda_env,
        venv_path=args.venv_path,
    )

    manager.run()


if __name__ == "__main__":
    main()
