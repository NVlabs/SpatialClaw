"""Persistent state for active GPU server chain processes."""

import fcntl
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class GPUServerState:
    chain_id: str
    pid: int
    slurm_job_ids: List[str] = field(default_factory=list)
    started_at: str = ""
    account: str = ""
    partition: str = ""
    gpus: int = 1
    reconstruct_backend: str = "pi3"


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


class GPUServerStateManager:
    """Read/write gpu_server_manager_state.json with file locking."""

    def __init__(self, project_root: Path):
        self.state_dir = project_root / "spatial_agent" / "logs"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "gpu_server_manager_state.json"
        self.lock_file = str(self.state_file) + ".lock"

    def _read(self) -> List[dict]:
        if not self.state_file.exists():
            return []
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, servers: List[dict]) -> None:
        with open(self.state_file, "w") as f:
            json.dump(servers, f, indent=2)

    def add_server(self, server: GPUServerState) -> None:
        with FileLock(self.lock_file):
            servers = self._read()
            servers.append(asdict(server))
            self._write(servers)

    def remove_server(self, chain_id: str) -> None:
        with FileLock(self.lock_file):
            servers = self._read()
            servers = [s for s in servers if s.get("chain_id") != chain_id]
            self._write(servers)

    def update_server_jobs(self, chain_id: str, slurm_job_ids: List[str]) -> None:
        with FileLock(self.lock_file):
            servers = self._read()
            for s in servers:
                if s.get("chain_id") == chain_id:
                    s["slurm_job_ids"] = slurm_job_ids
                    break
            self._write(servers)

    def list_servers(self) -> List[GPUServerState]:
        with FileLock(self.lock_file):
            raw = self._read()
        result = []
        for s in raw:
            try:
                result.append(GPUServerState(**s))
            except TypeError:
                continue
        return result

    def is_server_alive(self, server: GPUServerState) -> bool:
        """Check if the manager process PID is still running."""
        try:
            os.kill(server.pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def cleanup_dead_servers(self) -> List[GPUServerState]:
        """Remove servers whose manager PID is dead. Returns removed servers."""
        with FileLock(self.lock_file):
            servers = self._read()
            alive = []
            dead = []
            for s in servers:
                pid = s.get("pid", 0)
                try:
                    os.kill(pid, 0)
                    alive.append(s)
                except (ProcessLookupError, PermissionError, OSError):
                    dead.append(s)
            self._write(alive)
        result = []
        for s in dead:
            try:
                result.append(GPUServerState(**s))
            except TypeError:
                continue
        return result
