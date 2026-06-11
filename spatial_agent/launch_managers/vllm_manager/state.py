"""Persistent state for active server chain processes."""

import fcntl
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ChainState:
    chain_id: str
    served_name: str
    model_name: str
    model_path: str
    pid: int
    slurm_job_ids: List[str] = field(default_factory=list)
    started_at: str = ""
    account: str = ""
    partition: str = ""
    max_model_len: int = 0
    max_num_seqs: int = 0
    tp_size: int = 0
    gpus: int = 8


class FileLock:
    """fcntl-based file lock (same pattern as entrypoints/launch_vllm.py)."""

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


class ChainStateManager:
    """Read/write vllm_manager_state.json with file locking."""

    def __init__(self, project_root: Path):
        self.state_dir = project_root / "spatial_agent" / "logs"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "vllm_manager_state.json"
        self.lock_file = str(self.state_file) + ".lock"

    def _read(self) -> List[dict]:
        if not self.state_file.exists():
            return []
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _write(self, chains: List[dict]) -> None:
        with open(self.state_file, "w") as f:
            json.dump(chains, f, indent=2)

    def add_chain(self, chain: ChainState) -> None:
        with FileLock(self.lock_file):
            chains = self._read()
            chains.append(asdict(chain))
            self._write(chains)

    def remove_chain(self, chain_id: str) -> None:
        with FileLock(self.lock_file):
            chains = self._read()
            chains = [c for c in chains if c.get("chain_id") != chain_id]
            self._write(chains)

    def update_chain_jobs(self, chain_id: str, slurm_job_ids: List[str]) -> None:
        with FileLock(self.lock_file):
            chains = self._read()
            for c in chains:
                if c.get("chain_id") == chain_id:
                    c["slurm_job_ids"] = slurm_job_ids
                    break
            self._write(chains)

    def list_chains(self) -> List[ChainState]:
        with FileLock(self.lock_file):
            raw = self._read()
        result = []
        for c in raw:
            try:
                result.append(ChainState(**c))
            except TypeError:
                continue
        return result

    def is_chain_alive(self, chain: ChainState) -> bool:
        """Check if the manager process PID is still running."""
        try:
            os.kill(chain.pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def cleanup_dead_chains(self) -> List[ChainState]:
        """Remove chains whose manager PID is dead. Returns removed chains."""
        with FileLock(self.lock_file):
            chains = self._read()
            alive = []
            dead = []
            for c in chains:
                pid = c.get("pid", 0)
                try:
                    os.kill(pid, 0)
                    alive.append(c)
                except (ProcessLookupError, PermissionError, OSError):
                    dead.append(c)
            self._write(alive)
        return [ChainState(**c) for c in dead if _is_valid_chain(c)]


def _is_valid_chain(c: dict) -> bool:
    try:
        ChainState(**c)
        return True
    except TypeError:
        return False
