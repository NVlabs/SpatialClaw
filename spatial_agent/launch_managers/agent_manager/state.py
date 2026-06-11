"""Persistent state for active experiment chain processes."""

import fcntl
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ExperimentState:
    experiment_id: str
    benchmark: str
    model_name: str          # e.g. "qwen3.5-122b-a10b"
    experiment_name: str     # user-given name
    pid: int                 # chain process PID
    slurm_job_ids: List[str] = field(default_factory=list)
    started_at: str = ""
    account: str = ""
    partition: str = ""
    gpus: int = 8
    concurrency: int = 8
    subsample: int = 0
    work_dir: str = ""       # absolute path to work_dir for this experiment
    total_samples: int = -1  # -1 = unknown yet
    status: str = "running"  # "running" | "completed" | "failed"
    experiment_type: str = "agent"  # "agent" | "cot"
    scheduled_for: str = ""  # ISO timestamp; "" = start immediately


class FileLock:
    """fcntl-based file lock."""

    def __init__(self, lock_path: str):
        self.lock_path = lock_path
        self._f = None

    def __enter__(self):
        self._f = open(self.lock_path, "a+")
        self._f.seek(0)
        fcntl.flock(self._f.fileno(), fcntl.LOCK_EX)
        return self._f

    def __exit__(self, *exc):
        if self._f:
            fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
            self._f.close()
            self._f = None


class ExperimentStateManager:
    """Read/write agent_manager_state.json with file locking."""

    def __init__(self, project_root: Path):
        self.state_dir = project_root / "spatial_agent" / "logs"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "agent_manager_state.json"
        self.lock_file = str(self.state_file) + ".lock"

    def _read(self) -> List[dict]:
        if not self.state_file.exists():
            return []
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, experiments: List[dict]) -> None:
        with open(self.state_file, "w") as f:
            json.dump(experiments, f, indent=2)

    def add_experiment(self, exp: ExperimentState) -> None:
        with FileLock(self.lock_file):
            experiments = self._read()
            experiments.append(asdict(exp))
            self._write(experiments)

    def remove_experiment(self, experiment_id: str) -> None:
        with FileLock(self.lock_file):
            experiments = self._read()
            experiments = [e for e in experiments if e.get("experiment_id") != experiment_id]
            self._write(experiments)

    def update_experiment_jobs(self, experiment_id: str, slurm_job_ids: List[str]) -> None:
        with FileLock(self.lock_file):
            experiments = self._read()
            for e in experiments:
                if e.get("experiment_id") == experiment_id:
                    e["slurm_job_ids"] = slurm_job_ids
                    break
            self._write(experiments)

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        with FileLock(self.lock_file):
            experiments = self._read()
            for e in experiments:
                if e.get("experiment_id") == experiment_id:
                    e["status"] = status
                    break
            self._write(experiments)

    def remove_experiments(self, experiment_ids: List[str]) -> None:
        if not experiment_ids:
            return
        ids = set(experiment_ids)
        with FileLock(self.lock_file):
            experiments = self._read()
            experiments = [e for e in experiments if e.get("experiment_id") not in ids]
            self._write(experiments)

    def update_experiment_total(self, experiment_id: str, total_samples: int) -> None:
        with FileLock(self.lock_file):
            experiments = self._read()
            for e in experiments:
                if e.get("experiment_id") == experiment_id:
                    e["total_samples"] = total_samples
                    break
            self._write(experiments)

    def list_experiments(self) -> List[ExperimentState]:
        with FileLock(self.lock_file):
            raw = self._read()
        known = {f.name for f in ExperimentState.__dataclass_fields__.values()}
        result = []
        for e in raw:
            try:
                result.append(ExperimentState(**{k: v for k, v in e.items() if k in known}))
            except TypeError:
                continue
        return result

    def is_experiment_alive(self, exp: ExperimentState) -> bool:
        """Check if the chain process PID is still running."""
        try:
            os.kill(exp.pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def cleanup_dead_experiments(self) -> List[ExperimentState]:
        """Remove experiments whose chain PID is dead and status is 'running'."""
        with FileLock(self.lock_file):
            experiments = self._read()
            alive = []
            dead = []
            for e in experiments:
                pid = e.get("pid", 0)
                status = e.get("status", "running")
                # Keep completed/failed experiments for history
                if status != "running":
                    alive.append(e)
                    continue
                try:
                    os.kill(pid, 0)
                    alive.append(e)
                except (ProcessLookupError, PermissionError, OSError):
                    dead.append(e)
            self._write(alive)
        result = []
        for e in dead:
            try:
                result.append(ExperimentState(**e))
            except TypeError:
                continue
        return result
