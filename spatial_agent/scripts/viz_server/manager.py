#!/usr/bin/env python3
"""
SLURM Visualization Server Manager with Automatic 24-Hour Restarts

This script runs OUTSIDE SLURM and manages a continuous chain of 24-hour SLURM
jobs for the results visualization web server with automatic restart.

No GPUs are requested — this is a lightweight web server.

Usage:
    python spatial_agent/scripts/viz_server/manager.py --port 8501
"""

import argparse
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from termcolor import colored


class SlurmVizManager:
    def __init__(
        self,
        port: int = 8501,
        job_name: str = "viz-server",
        account: str = "nvr_lpr_nvgptvision",
        partition: str = "cpu_long",
        time_limit: str = "23:59:00",
        output_dir: str = None,
    ):
        self.port = port
        self.job_name = job_name
        self.account = account
        self.partition = partition
        self.time_limit = time_limit

        # Project root: go up 4 levels (viz_server -> scripts -> spatial_agent -> project)
        self.project_root = Path(__file__).parent.parent.parent.parent.absolute()

        if output_dir is None:
            self.output_dir = self.project_root / "spatial_agent" / "logs" / "slurm_viz"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.total_seconds = self._parse_time_to_seconds(time_limit)
        self.running = True
        self.job_counter = 1
        self.current_job_id = None
        self.submitted_job_ids = []

        print(colored("=" * 60, "cyan"))
        print(colored("SLURM Visualization Server Chain Manager", "cyan", attrs=["bold"]))
        print(colored("=" * 60, "cyan"))
        print(f"Port:              {self.port}")
        print(f"Job name:          {self.job_name}")
        print(f"Account:           {self.account}")
        print(f"Partition:         {self.partition}")
        print(f"Time limit:        {self.time_limit} ({self.total_seconds}s)")
        print(f"Output directory:  {self.output_dir}")
        print(colored("=" * 60, "cyan"))
        print()

    def _parse_time_to_seconds(self, time_str: str) -> int:
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
        result = subprocess.run("which sbatch", shell=True, capture_output=True)
        return result.returncode == 0

    def _get_job_status(self, job_id: str) -> str:
        if not job_id:
            return "UNKNOWN"
        cmd = ["squeue", "-j", str(job_id), "-h", "-o", "%T"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return "NOT_FOUND"
        return result.stdout.strip().upper()

    def _is_job_running(self, job_id: str) -> bool:
        status = self._get_job_status(job_id)
        return status in ["RUNNING", "PENDING", "CONFIGURING", "COMPLETING"]

    def _wait_for_job_to_start(self, job_id: str, timeout: int = 7200) -> bool:
        print(colored(
            f"[{self._ts()}] Waiting for job {job_id} to start running...", "cyan"
        ), flush=True)

        start_time = time.time()
        last_status = None
        last_update_time = start_time

        while self.running:
            status = self._get_job_status(job_id)
            current_time = time.time()

            if status != last_status:
                elapsed = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._ts()}] Job {job_id} status: {status} (waited {elapsed}m)", "cyan"
                ), flush=True)
                last_status = status
                last_update_time = current_time

            if status in ["PENDING", "CONFIGURING"] and current_time - last_update_time >= 300:
                elapsed = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._ts()}] Still waiting for job {job_id}... ({elapsed}m elapsed)", "cyan"
                ), flush=True)
                last_update_time = current_time

            if status == "RUNNING":
                elapsed = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._ts()}] Job {job_id} is now running! (waited {elapsed}m)", "green"
                ), flush=True)
                return True
            elif status in ["FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"]:
                print(colored(
                    f"[{self._ts()}] Job {job_id} failed with status: {status}", "red"
                ), flush=True)
                return False
            elif status == "NOT_FOUND":
                print(colored(
                    f"[{self._ts()}] Job {job_id} not found in queue", "red"
                ), flush=True)
                return False
            else:
                time.sleep(5)

        return False

    def _wait_for_job_completion(self, job_id: str) -> bool:
        print(colored(
            f"[{self._ts()}] Waiting for job {job_id} to complete...", "cyan"
        ), flush=True)

        start_time = time.time()
        last_update_time = start_time
        check_interval = 60

        while self.running:
            status = self._get_job_status(job_id)
            current_time = time.time()

            if current_time - last_update_time >= 600:
                elapsed_minutes = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._ts()}] Job {job_id} still running... ({elapsed_minutes}m elapsed)",
                    "cyan"
                ), flush=True)
                last_update_time = current_time

            if status == "RUNNING":
                time.sleep(check_interval)
                continue
            elif status == "COMPLETED":
                elapsed_minutes = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._ts()}] Job {job_id} completed! (ran for {elapsed_minutes}m)", "green"
                ), flush=True)
                return True
            elif status in ["FAILED", "TIMEOUT", "NODE_FAIL", "CANCELLED"]:
                elapsed_minutes = int((current_time - start_time) / 60)
                print(colored(
                    f"[{self._ts()}] Job {job_id} ended with status: {status} (after {elapsed_minutes}m)",
                    "red"
                ), flush=True)
                return False
            elif status == "NOT_FOUND":
                print(colored(
                    f"[{self._ts()}] Job {job_id} not in queue, checking sacct...", "yellow"
                ), flush=True)
                cmd = ["sacct", "-j", str(job_id), "-n", "-o", "State"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    sacct_status = result.stdout.strip().split()[0].upper()
                    if "COMPLETED" in sacct_status:
                        elapsed_minutes = int((current_time - start_time) / 60)
                        print(colored(
                            f"[{self._ts()}] Job {job_id} completed (via sacct, ran for {elapsed_minutes}m)",
                            "green"
                        ), flush=True)
                        return True
                    else:
                        print(colored(
                            f"[{self._ts()}] Job {job_id} ended with: {sacct_status}", "red"
                        ), flush=True)
                        return False
                else:
                    print(colored(
                        f"[{self._ts()}] Cannot determine job status, assuming completed", "yellow"
                    ), flush=True)
                    return True
            else:
                time.sleep(check_interval)

        return False

    def _ts(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _resolve_ip(self, hostname: str) -> str:
        """Resolve hostname to IP address."""
        try:
            return socket.gethostbyname(hostname)
        except socket.gaierror:
            return hostname

    def _write_serve_json(self, node: str, job_id: str):
        """Write server info to logs/web_serve.json."""
        serve_json_path = self.project_root / "spatial_agent" / "logs" / "web_serve.json"
        serve_json_path.parent.mkdir(parents=True, exist_ok=True)

        ip = self._resolve_ip(node)
        data = {
            "node": node,
            "ip": ip,
            "port": self.port,
            "url": f"http://{ip}:{self.port}",
            "job_id": job_id,
            "create_time": self._ts(),
        }

        with open(serve_json_path, "w") as f:
            json.dump(data, f, indent=2)

        print(colored(
            f"[{self._ts()}] Wrote server info to {serve_json_path}", "cyan"
        ), flush=True)

    def _clear_serve_json(self):
        """Remove web_serve.json on shutdown."""
        serve_json_path = self.project_root / "spatial_agent" / "logs" / "web_serve.json"
        if serve_json_path.exists():
            serve_json_path.unlink()
            print(colored(
                f"[{self._ts()}] Removed {serve_json_path}", "cyan"
            ), flush=True)

    def _generate_slurm_script(self, job_number: int) -> str:
        script = f"""#!/bin/bash
#SBATCH --job-name={self.job_name}
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time={self.time_limit}
#SBATCH --output={self.output_dir}/{self.job_name}_%j.out
#SBATCH --error={self.output_dir}/{self.job_name}_%j.err

echo "=========================================="
echo "SLURM Visualization Server - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Port: {self.port}"
echo "Start time: $(date)"
echo "=========================================="
echo ""
echo "Access the server at:"
echo "  http://$SLURM_NODELIST:{self.port}"
echo ""

# Navigate to project root
cd {self.project_root}

# Activate conda environment
CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent

# Run the visualization server
python -m spatial_agent.visualization_server --port {self.port}

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
"""
        return script

    def submit_job(self) -> bool:
        print(colored(f"[{self._ts()}] Submitting job #{self.job_counter}...", "cyan"), flush=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = self.output_dir / f"{self.job_name}_{timestamp}.sbatch"

        script_content = self._generate_slurm_script(self.job_counter)
        with open(script_path, "w") as f:
            f.write(script_content)
        script_path.chmod(0o755)

        cmd = ["sbatch", str(script_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            job_id = None
            for word in result.stdout.split():
                if word.isdigit():
                    job_id = word
                    break

            self.current_job_id = job_id
            self.submitted_job_ids.append(job_id)

            print(colored(
                f"[{self._ts()}] Job #{self.job_counter} submitted: SLURM job ID {job_id}", "green"
            ), flush=True)
            print(colored(
                f"[{self._ts()}] Log: {self.output_dir}/{self.job_name}_{job_id}.out", "cyan"
            ), flush=True)
            return True
        else:
            print(colored(
                f"[{self._ts()}] Failed to submit job #{self.job_counter}", "red"
            ), flush=True)
            print(result.stderr, flush=True)
            return False

    def handle_shutdown(self, signum, frame):
        print(colored(
            f"\n[{self._ts()}] Received shutdown signal. Stopping chain...", "yellow"
        ), flush=True)

        self.running = False

        if self.submitted_job_ids:
            print(colored(
                f"[{self._ts()}] Cancelling {len(self.submitted_job_ids)} submitted job(s)...",
                "yellow"
            ), flush=True)
            for job_id in self.submitted_job_ids:
                try:
                    result = subprocess.run(
                        ["scancel", str(job_id)], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        print(colored(f"[{self._ts()}] Cancelled job {job_id}", "green"), flush=True)
                    elif "Invalid job id" not in result.stderr:
                        print(colored(
                            f"[{self._ts()}] Note: Job {job_id} - {result.stderr.strip()}", "yellow"
                        ), flush=True)
                except Exception as e:
                    print(colored(
                        f"[{self._ts()}] Warning: Failed to cancel job {job_id}: {e}", "yellow"
                    ), flush=True)
            print(colored(f"[{self._ts()}] All jobs cancelled.", "green"), flush=True)

        self._clear_serve_json()

    def run(self):
        if not self._check_slurm_available():
            print(colored("ERROR: SLURM not available. Cannot submit jobs.", "red"))
            sys.exit(1)

        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        print(colored(f"[{self._ts()}] Starting continuous visualization server chain...", "green", attrs=["bold"]), flush=True)
        print(colored(f"[{self._ts()}] Press Ctrl+C to stop the chain", "cyan"), flush=True)
        print()

        while self.running:
            if not self.submit_job():
                print(colored(
                    f"[{self._ts()}] Failed to submit job, retrying in 60 seconds...", "yellow"
                ), flush=True)
                time.sleep(60)
                continue

            if not self._wait_for_job_to_start(self.current_job_id):
                print(colored(
                    f"[{self._ts()}] Job failed to start, will retry...", "yellow"
                ), flush=True)
                try:
                    subprocess.run(["scancel", str(self.current_job_id)], capture_output=True)
                except Exception:
                    pass
                time.sleep(60)
                continue

            # Print access info and write serve JSON once running
            cmd = ["squeue", "-j", str(self.current_job_id), "-h", "-o", "%N"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                node = result.stdout.strip()
                ip = self._resolve_ip(node)
                print(colored(
                    f"[{self._ts()}] Server accessible at: http://{ip}:{self.port}",
                    "green", attrs=["bold"]
                ), flush=True)
                self._write_serve_json(node, self.current_job_id)

            job_completed = self._wait_for_job_completion(self.current_job_id)

            if not job_completed:
                print(colored(
                    f"[{self._ts()}] Job ended abnormally, will retry in 60 seconds...", "yellow"
                ), flush=True)
                time.sleep(60)

            self.job_counter += 1

            if self.running:
                print(colored(
                    f"[{self._ts()}] Job ended. Submitting next job to restart server...", "green"
                ), flush=True)
                print()

        print(colored(f"[{self._ts()}] Chain manager stopped.", "green", attrs=["bold"]), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Manage visualization server with automatic SLURM restarts"
    )
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--job-name", type=str, default="viz-server")
    parser.add_argument("--account", type=str, default="nvr_lpr_nvgptvision")
    parser.add_argument("--partition", type=str, default="cpu_long")
    parser.add_argument("--time", type=str, default="23:59:00")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    manager = SlurmVizManager(
        port=args.port,
        job_name=args.job_name,
        account=args.account,
        partition=args.partition,
        time_limit=args.time,
        output_dir=args.output_dir,
    )

    manager.run()


if __name__ == "__main__":
    main()
