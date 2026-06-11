#!/usr/bin/env python3
"""
SLURM Spatial Agent Runner with Automatic 4-Hour Restarts

This script runs OUTSIDE SLURM and manages a continuous chain of 4-hour SLURM jobs
for spatial agent benchmark evaluations with automatic resume.

Unlike the vLLM server, there is NO overlap - each job completes before the next starts.

Usage:
    python spatial_agent/scripts/agent/slurm/manager.py --benchmark erqa
"""

import argparse
import datetime
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from termcolor import colored


class SlurmAgentManager:
    def __init__(
        self,
        benchmark: str,
        job_name: str = "spatial-agent",
        account: str = "nvr_taiwan_rvos",
        partition: str = "batch_singlenode,batch_block1,batch_block3,batch_block4",
        gpus: int = 8,
        time_limit: str = "4:00:00",
        concurrency: int = 24,
        agent_script: str = "spatial_agent/scripts/agent/slurm/run.sh",
        output_dir: str = None,
        model_config: str = "",
        dataset_config: str = "",
        experiment_name: str = "",
        subsample: int = 0,
    ):
        self.benchmark = benchmark
        self.job_name = job_name
        self.account = account
        self.partition = partition
        self.gpus = gpus
        self.time_limit = time_limit
        self.concurrency = concurrency
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.experiment_name = experiment_name
        self.subsample = subsample

        # Get the project root directory (go up 4 levels: slurm -> agent -> scripts -> spatial_agent -> project)
        self.project_root = Path(__file__).parent.parent.parent.parent.parent.absolute()
        self.agent_script_path = self.project_root / agent_script

        if not self.agent_script_path.exists():
            raise FileNotFoundError(f"Agent script not found: {self.agent_script_path}")

        # Set output directory
        if output_dir is None:
            self.output_dir = self.project_root / "spatial_agent" / "logs" / "slurm_agent"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate timings (no overlap - wait for completion)
        self.total_seconds = self._parse_time_to_seconds(time_limit)

        self.running = True
        self.job_counter = 1
        self.current_job_id = None
        self.submitted_job_ids = []  # Track all submitted jobs for cleanup

        print(colored("=" * 60, "cyan"))
        print(colored("SLURM Spatial Agent Chain Manager", "cyan", attrs=["bold"]))
        print(colored("=" * 60, "cyan"))
        print(f"Benchmark:         {self.benchmark}")
        print(f"Job name:          {self.job_name}")
        print(f"Account:           {self.account}")
        print(f"Partition:         {self.partition}")
        print(f"GPUs:              {self.gpus}")
        print(f"Time limit:        {self.time_limit} ({self.total_seconds}s)")
        print(f"Concurrency:       {self.concurrency}")
        print(f"Model config:      {self.model_config or '<none>'}")
        print(f"Dataset config:    {self.dataset_config or '<none>'}")
        print(f"Experiment name:   {self.experiment_name or '<default>'}")
        print(f"Subsample:         {self.subsample or '<all>'}")
        print(f"Agent script:      {self.agent_script_path}")
        print(f"Output directory:  {self.output_dir}")
        print(colored("=" * 60, "cyan"))
        print(colored("Note: No overlap - each job completes before next starts", "yellow"))
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
        """Check if a SLURM job is still running or pending."""
        status = self._get_job_status(job_id)
        return status in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]

    def _wait_for_job_to_start(self, job_id: str, timeout: int = 7200) -> bool:
        """
        Wait for a SLURM job to start running.
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
                print(colored(
                    f"[{self._get_timestamp()}] ✗ Job {job_id} not found in queue",
                    "red"
                ), flush=True)
                return False
            elif status in ["PENDING", "CONFIGURING"]:
                time.sleep(5)
            else:
                time.sleep(5)

        return False

    def _wait_for_job_completion(self, job_id: str) -> bool:
        """
        Wait for a SLURM job to complete.
        Returns True if completed successfully, False if failed.
        """
        print(colored(
            f"[{self._get_timestamp()}] Waiting for job {job_id} to complete...",
            "cyan"
        ), flush=True)

        start_time = time.time()
        last_update_time = start_time
        check_interval = 60  # Check every minute

        while self.running:
            status = self._get_job_status(job_id)
            current_time = time.time()

            # Print periodic updates every 10 minutes
            if current_time - last_update_time >= 600:
                elapsed_minutes = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._get_timestamp()}] Job {job_id} still running... ({elapsed_minutes}m elapsed)",
                    "cyan"
                ), flush=True)
                last_update_time = current_time

            if status == "RUNNING":
                # Still running, keep waiting
                time.sleep(check_interval)
                continue
            elif status == "COMPLETED":
                elapsed_minutes = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._get_timestamp()}] ✓ Job {job_id} completed successfully! (ran for {elapsed_minutes}m)",
                    "green"
                ), flush=True)
                return True
            elif status in ["FAILED", "TIMEOUT", "NODE_FAIL", "CANCELLED"]:
                elapsed_minutes = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._get_timestamp()}] ✗ Job {job_id} ended with status: {status} (after {elapsed_minutes}m)",
                    "red"
                ), flush=True)
                return False
            elif status == "NOT_FOUND":
                # Job finished and is no longer in queue - check sacct for final status
                print(colored(
                    f"[{self._get_timestamp()}] Job {job_id} not in queue, checking completion status...",
                    "yellow"
                ), flush=True)

                # Check sacct for job completion status
                cmd = ["sacct", "-j", str(job_id), "-n", "-o", "State"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    sacct_status = result.stdout.strip().split()[0].upper()
                    if "COMPLETED" in sacct_status:
                        elapsed_minutes = int((current_time - start_time) / 60)
                        print(colored(
                            f"[{self._get_timestamp()}] ✓ Job {job_id} completed (confirmed via sacct, ran for {elapsed_minutes}m)",
                            "green"
                        ), flush=True)
                        return True
                    else:
                        print(colored(
                            f"[{self._get_timestamp()}] ✗ Job {job_id} ended with status: {sacct_status}",
                            "red"
                        ), flush=True)
                        return False
                else:
                    # Can't determine status, assume completed
                    print(colored(
                        f"[{self._get_timestamp()}] Warning: Cannot determine job status, assuming completed",
                        "yellow"
                    ), flush=True)
                    return True
            else:
                # Unknown status, wait a bit
                time.sleep(check_interval)

        # Interrupted by Ctrl+C
        return False

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
{"" if "interactive" in self.partition else "#SBATCH --exclusive"}

echo "=========================================="
echo "SLURM Spatial Agent Evaluation - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Benchmark: {self.benchmark}"
echo "Concurrency: {self.concurrency}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Navigate to project root
cd {self.project_root}

# Activate conda environment
CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent

# Run the spatial agent evaluation
bash {self.agent_script_path} "{self.benchmark}" {self.concurrency} \\
    "{self.model_config}" "{self.dataset_config}" "{self.experiment_name}" {self.subsample}

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
            # Extract job ID
            job_id = None
            for word in result.stdout.split():
                if word.isdigit():
                    job_id = word
                    break

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

            return True
        else:
            print(colored(
                f"[{self._get_timestamp()}] ✗ Failed to submit job #{self.job_counter}",
                "red"
            ), flush=True)
            print(result.stderr, flush=True)
            return False

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(colored(
            f"\n[{self._get_timestamp()}] Received shutdown signal. Stopping chain...",
            "yellow"
        ), flush=True)

        self.running = False

        # Cancel all submitted jobs
        if self.submitted_job_ids:
            print(colored(
                f"[{self._get_timestamp()}] Cancelling {len(self.submitted_job_ids)} submitted job(s)...",
                "yellow"
            ), flush=True)

            for job_id in self.submitted_job_ids:
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
                f"[{self._get_timestamp()}] All jobs cancelled.",
                "green"
            ), flush=True)

    def run(self):
        """Run the continuous job submission loop."""
        if not self._check_slurm_available():
            print(colored("ERROR: SLURM not available. Cannot submit jobs.", "red"))
            sys.exit(1)

        # Set up signal handlers
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        print(colored(f"[{self._get_timestamp()}] Starting continuous spatial agent evaluation chain...", "green", attrs=["bold"]), flush=True)
        print(colored(f"[{self._get_timestamp()}] Benchmark: {self.benchmark}", "green"), flush=True)
        print(colored(f"[{self._get_timestamp()}] Press Ctrl+C to stop the chain", "cyan"), flush=True)
        print()

        while self.running:
            # Submit the next job
            if not self.submit_job():
                print(colored(
                    f"[{self._get_timestamp()}] Failed to submit job, retrying in 60 seconds...",
                    "yellow"
                ), flush=True)
                time.sleep(60)
                continue

            # Wait for job to start
            if not self._wait_for_job_to_start(self.current_job_id):
                print(colored(
                    f"[{self._get_timestamp()}] Job failed to start, will retry...",
                    "yellow"
                ), flush=True)
                # Cancel the failed job
                try:
                    subprocess.run(["scancel", str(self.current_job_id)], capture_output=True)
                except Exception:
                    pass
                time.sleep(60)
                continue

            # Wait for job to complete (no overlap)
            job_completed = self._wait_for_job_completion(self.current_job_id)

            if not job_completed:
                print(colored(
                    f"[{self._get_timestamp()}] Job ended abnormally, will retry in 60 seconds...",
                    "yellow"
                ), flush=True)
                time.sleep(60)

            # Increment counter for next job
            self.job_counter += 1

            if self.running:
                print(colored(
                    f"[{self._get_timestamp()}] Job completed. Submitting next job to resume evaluation...",
                    "green"
                ), flush=True)
                print()

        print(colored(f"[{self._get_timestamp()}] Chain manager stopped.", "green", attrs=["bold"]), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Manage spatial agent evaluation with automatic SLURM restarts every 4 hours"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Benchmark to evaluate (e.g. erqa, mmsivideo)"
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="spatial-agent",
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
        "--concurrency",
        type=int,
        default=24,
        help="Number of parallel agent workers"
    )
    parser.add_argument(
        "--agent-script",
        type=str,
        default="spatial_agent/scripts/agent/slurm/run.sh",
        help="Path to agent execution script (relative to project root)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for SLURM logs"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="",
        help="Path to model config JSON (e.g., spatial_agent/config/model/qwen3.5-122b-a10b.json)"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="",
        help="Path to dataset config JSON (e.g., spatial_agent/config/dataset/mmsi.json)"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="",
        help="Experiment name appended to work_dir folder name"
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help="Deterministically subsample N random samples (seed=42). 0 = use all."
    )

    args = parser.parse_args()

    # Create manager and run
    manager = SlurmAgentManager(
        benchmark=args.benchmark,
        job_name=args.job_name,
        account=args.account,
        partition=args.partition,
        gpus=args.gpus,
        time_limit=args.time,
        concurrency=args.concurrency,
        agent_script=args.agent_script,
        output_dir=args.output_dir,
        model_config=args.model_config,
        dataset_config=args.dataset_config,
        experiment_name=args.experiment_name,
        subsample=args.subsample,
    )

    manager.run()


if __name__ == "__main__":
    main()
