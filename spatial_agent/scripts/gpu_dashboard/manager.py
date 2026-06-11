#!/usr/bin/env python3
"""SLURM chain manager for the GPU dashboard.

Runs OUTSIDE SLURM and:
  1. Starts a `SamplerThread` in-process that polls `ssh <node> nvidia-smi`
     every N seconds and writes GPU + agent-count rows to a SQLite DB. This
     sampler lives in the login-node process so it survives every 24 h web
     restart uninterrupted.
  2. Submits a chain of 24 h SLURM jobs that each run
     `python -m spatial_agent.gpu_dashboard --port ... --db ...` and writes
     the current server endpoint to `logs/gpu_dashboard_serve.json`.

Structurally mirrors scripts/viz_server/manager.py.
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

# Ensure the project root is on sys.path when run as a plain script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from spatial_agent.gpu_dashboard.sampler import SamplerThread


class SlurmGpuDashboardManager:
    def __init__(
        self,
        port: int = 8502,
        job_name: str = "gpu-dashboard",
        account: str = "llmservice_fm_vision",
        partition: str = "cpu_long",
        time_limit: str = "23:59:00",
        output_dir: str = None,
        db_path: str = None,
        sample_interval: int = 5,
        history_sec: int = 3600,
        node_timeout: int = 15,
    ):
        self.port = port
        self.job_name = job_name
        self.account = account
        self.partition = partition
        self.time_limit = time_limit
        self.sample_interval = sample_interval
        self.history_sec = history_sec
        self.node_timeout = node_timeout

        self.project_root = _PROJECT_ROOT.absolute()

        if output_dir is None:
            self.output_dir = self.project_root / "spatial_agent" / "logs" / "slurm_gpu_dashboard"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if db_path is None:
            self.db_path = self.project_root / "spatial_agent" / "logs" / "gpu_dashboard.db"
        else:
            self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.total_seconds = self._parse_time_to_seconds(time_limit)
        self.running = True
        self.job_counter = 1
        self.current_job_id = None
        self.submitted_job_ids = []
        self.sampler: SamplerThread | None = None

        print(colored("=" * 60, "cyan"))
        print(colored("SLURM GPU Dashboard Chain Manager", "cyan", attrs=["bold"]))
        print(colored("=" * 60, "cyan"))
        print(f"Port:              {self.port}")
        print(f"Job name:          {self.job_name}")
        print(f"Account:           {self.account}")
        print(f"Partition:         {self.partition}")
        print(f"Time limit:        {self.time_limit} ({self.total_seconds}s)")
        print(f"DB path:           {self.db_path}")
        print(f"Sample interval:   {self.sample_interval}s")
        print(f"History retention: {self.history_sec}s")
        print(f"Output directory:  {self.output_dir}")
        print(colored("=" * 60, "cyan"))
        print()

    @staticmethod
    def _parse_time_to_seconds(time_str: str) -> int:
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = map(int, parts)
        elif len(parts) == 2:
            h = 0
            m, s = map(int, parts)
        else:
            raise ValueError(f"Invalid time format: {time_str}")
        return h * 3600 + m * 60 + s

    @staticmethod
    def _ts() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _check_slurm_available() -> bool:
        return subprocess.run("which sbatch", shell=True, capture_output=True).returncode == 0

    def _get_job_status(self, job_id: str) -> str:
        if not job_id:
            return "UNKNOWN"
        cmd = ["squeue", "-j", str(job_id), "-h", "-o", "%T"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return "NOT_FOUND"
        return res.stdout.strip().upper()

    def _wait_for_job_to_start(self, job_id: str) -> bool:
        print(colored(f"[{self._ts()}] Waiting for job {job_id} to start...", "cyan"), flush=True)
        start = time.time()
        last_status = None
        last_update = start
        while self.running:
            status = self._get_job_status(job_id)
            now = time.time()
            if status != last_status:
                mins = int((now - start) / 60)
                print(colored(f"[{self._ts()}] Job {job_id} status: {status} ({mins}m)", "cyan"), flush=True)
                last_status = status
                last_update = now
            if status in ("PENDING", "CONFIGURING") and now - last_update >= 300:
                mins = int((now - start) / 60)
                print(colored(f"[{self._ts()}] Still waiting for job {job_id}... ({mins}m)", "cyan"), flush=True)
                last_update = now
            if status == "RUNNING":
                print(colored(f"[{self._ts()}] Job {job_id} is now running", "green"), flush=True)
                return True
            if status in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"):
                print(colored(f"[{self._ts()}] Job {job_id} failed: {status}", "red"), flush=True)
                return False
            if status == "NOT_FOUND":
                print(colored(f"[{self._ts()}] Job {job_id} not found in queue", "red"), flush=True)
                return False
            time.sleep(5)
        return False

    def _wait_for_job_completion(self, job_id: str) -> bool:
        print(colored(f"[{self._ts()}] Waiting for job {job_id} to complete...", "cyan"), flush=True)
        start = time.time()
        last_log = start
        while self.running:
            status = self._get_job_status(job_id)
            now = time.time()
            if now - last_log >= 600:
                mins = int((now - start) / 60)
                print(colored(f"[{self._ts()}] Job {job_id} still running ({mins}m)", "cyan"), flush=True)
                last_log = now
            if status == "RUNNING":
                time.sleep(60)
                continue
            if status == "COMPLETED":
                print(colored(f"[{self._ts()}] Job {job_id} completed", "green"), flush=True)
                return True
            if status in ("FAILED", "TIMEOUT", "NODE_FAIL", "CANCELLED"):
                print(colored(f"[{self._ts()}] Job {job_id} ended: {status}", "red"), flush=True)
                return False
            if status == "NOT_FOUND":
                res = subprocess.run(
                    ["sacct", "-j", str(job_id), "-n", "-o", "State"],
                    capture_output=True, text=True,
                )
                if res.returncode == 0 and res.stdout.strip():
                    st = res.stdout.strip().split()[0].upper()
                    ok = "COMPLETED" in st
                    print(colored(f"[{self._ts()}] Job {job_id} via sacct: {st}", "green" if ok else "red"), flush=True)
                    return ok
                print(colored(f"[{self._ts()}] Job {job_id} status unknown, assuming done", "yellow"), flush=True)
                return True
            time.sleep(60)
        return False

    @staticmethod
    def _resolve_ip(hostname: str) -> str:
        try:
            return socket.gethostbyname(hostname)
        except socket.gaierror:
            return hostname

    def _write_serve_json(self, node: str, job_id: str) -> None:
        path = self.project_root / "spatial_agent" / "logs" / "gpu_dashboard_serve.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        ip = self._resolve_ip(node)
        data = {
            "node": node, "ip": ip, "port": self.port,
            "url": f"http://{ip}:{self.port}",
            "job_id": job_id, "db_path": str(self.db_path),
            "create_time": self._ts(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(colored(f"[{self._ts()}] Wrote {path}", "cyan"), flush=True)

    def _clear_serve_json(self) -> None:
        path = self.project_root / "spatial_agent" / "logs" / "gpu_dashboard_serve.json"
        if path.exists():
            path.unlink()

    def _print_access_banner(self, node: str, ip: str) -> None:
        urls = [f"http://{ip}:{self.port}", f"http://{node}:{self.port}"]
        width = max(len(u) for u in urls) + 6
        bar = "=" * max(width, 56)
        print()
        print(colored(bar, "green", attrs=["bold"]), flush=True)
        print(colored("  GPU DASHBOARD IS LIVE", "green", attrs=["bold"]), flush=True)
        print(colored(bar, "green", attrs=["bold"]), flush=True)
        for u in urls:
            print(colored(f"   {u}", "green", attrs=["bold"]), flush=True)
        print(colored(f"   (node {node}, job {self.current_job_id})", "cyan"), flush=True)
        print(colored(bar, "green", attrs=["bold"]), flush=True)
        print(flush=True)

    def _generate_slurm_script(self, job_number: int) -> str:
        return f"""#!/bin/bash
#SBATCH --job-name={self.job_name}
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time={self.time_limit}
#SBATCH --output={self.output_dir}/{self.job_name}_%j.out
#SBATCH --error={self.output_dir}/{self.job_name}_%j.err

echo "=========================================="
echo "SLURM GPU Dashboard - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Port:         {self.port}"
echo "DB:           {self.db_path}"
echo "Start:        $(date)"
echo "=========================================="

cd {self.project_root}

CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent

python -m spatial_agent.gpu_dashboard --port {self.port} --db {self.db_path}

echo "End: $(date)"
"""

    def submit_job(self) -> bool:
        print(colored(f"[{self._ts()}] Submitting job #{self.job_counter}...", "cyan"), flush=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_path = self.output_dir / f"{self.job_name}_{stamp}.sbatch"
        script_path.write_text(self._generate_slurm_script(self.job_counter))
        script_path.chmod(0o755)

        res = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        if res.returncode != 0:
            print(colored(f"[{self._ts()}] sbatch failed: {res.stderr.strip()}", "red"), flush=True)
            return False
        job_id = next((w for w in res.stdout.split() if w.isdigit()), None)
        self.current_job_id = job_id
        self.submitted_job_ids.append(job_id)
        print(colored(f"[{self._ts()}] Job #{self.job_counter} submitted: {job_id}", "green"), flush=True)
        return True

    def _start_sampler(self) -> None:
        if self.sampler and self.sampler.is_alive():
            return
        self.sampler = SamplerThread(
            project_root=self.project_root,
            db_path=str(self.db_path),
            interval_sec=self.sample_interval,
            history_sec=self.history_sec,
            node_timeout=self.node_timeout,
        )
        self.sampler.start()
        print(colored(f"[{self._ts()}] Sampler thread started", "green"), flush=True)

    def _stop_sampler(self) -> None:
        if not self.sampler:
            return
        self.sampler.stop()
        self.sampler.join(timeout=15)
        print(colored(f"[{self._ts()}] Sampler stopped", "cyan"), flush=True)

    def handle_shutdown(self, signum, frame):
        print(colored(f"\n[{self._ts()}] Shutdown signal received", "yellow"), flush=True)
        self.running = False
        if self.submitted_job_ids:
            print(colored(f"[{self._ts()}] Cancelling {len(self.submitted_job_ids)} job(s)", "yellow"), flush=True)
            for jid in self.submitted_job_ids:
                try:
                    subprocess.run(["scancel", str(jid)], capture_output=True)
                except Exception:
                    pass
        self._stop_sampler()
        self._clear_serve_json()

    def run(self) -> None:
        if not self._check_slurm_available():
            print(colored("ERROR: SLURM not available.", "red"))
            sys.exit(1)

        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        print(colored(f"[{self._ts()}] Starting dashboard chain...", "green", attrs=["bold"]), flush=True)
        print(colored(f"[{self._ts()}] Press Ctrl+C to stop", "cyan"), flush=True)
        print()

        # Sampler starts IMMEDIATELY and keeps running across SLURM job rotations.
        self._start_sampler()

        while self.running:
            if not self.submit_job():
                print(colored(f"[{self._ts()}] Retry sbatch in 60s", "yellow"), flush=True)
                time.sleep(60)
                continue

            if not self._wait_for_job_to_start(self.current_job_id):
                print(colored(f"[{self._ts()}] Job never started, retrying...", "yellow"), flush=True)
                try:
                    subprocess.run(["scancel", str(self.current_job_id)], capture_output=True)
                except Exception:
                    pass
                time.sleep(60)
                continue

            # Publish endpoint
            res = subprocess.run(
                ["squeue", "-j", str(self.current_job_id), "-h", "-o", "%N"],
                capture_output=True, text=True,
            )
            if res.returncode == 0 and res.stdout.strip():
                node = res.stdout.strip()
                ip = self._resolve_ip(node)
                self._write_serve_json(node, self.current_job_id)
                self._print_access_banner(node, ip)

            self._wait_for_job_completion(self.current_job_id)
            self.job_counter += 1
            if self.running:
                print(colored(f"[{self._ts()}] Restarting web server...", "green"), flush=True)
                print()

        print(colored(f"[{self._ts()}] Chain manager stopped.", "green", attrs=["bold"]), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="SLURM chain manager for the GPU dashboard")
    p.add_argument("--port", type=int, default=8502)
    p.add_argument("--job-name", type=str, default="gpu-dashboard")
    p.add_argument("--account", type=str, default="llmservice_fm_vision")
    p.add_argument("--partition", type=str, default="cpu_long")
    p.add_argument("--time", type=str, default="23:59:00")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--db", type=str, default=None)
    p.add_argument("--sample-interval", type=int, default=5)
    p.add_argument("--history-sec", type=int, default=3600)
    p.add_argument("--node-timeout", type=int, default=15)
    args = p.parse_args()

    SlurmGpuDashboardManager(
        port=args.port,
        job_name=args.job_name,
        account=args.account,
        partition=args.partition,
        time_limit=args.time,
        output_dir=args.output_dir,
        db_path=args.db,
        sample_interval=args.sample_interval,
        history_sec=args.history_sec,
        node_timeout=args.node_timeout,
    ).run()


if __name__ == "__main__":
    main()
