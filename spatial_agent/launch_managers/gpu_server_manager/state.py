"""Persistent state for active GPU server chain processes."""

import fcntl
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from ..lifecycle import (
    compare_process_identities,
    process_identity as lifecycle_process_identity,
    process_identity_is_valid,
    process_identity_record_is_recognized,
    process_identity_scope_is_local,
    process_pid_is_valid,
)
from ..slurm_utils import SchedulerJobIds


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
    # Optional lifecycle metadata.  Defaults preserve compatibility with
    # state files written before lifecycle-aware orchestration existed.
    pool_id: Optional[str] = None
    namespace: Optional[str] = None
    spec_digest: Optional[str] = None
    topology_digest: Optional[str] = None
    desired_instances: Optional[int] = None
    lifecycle_state: Optional[str] = None
    process_identity: Optional[dict] = None
    # Durable pre-sbatch fence; see the vLLM chain state for the lifecycle
    # contract.  Legacy records omit it and load with the safe None default.
    submission_intent: Optional[dict] = None


def _process_identity(pid: int) -> Optional[dict]:
    """Best-effort boot/start identity for local PID-reuse detection."""
    return lifecycle_process_identity(pid)


def _pid_record_status(
    pid: int, recorded_identity: Optional[dict],
) -> Optional[bool]:
    """Return exact local owner match, proven death/reuse, or uncertainty."""
    if not process_pid_is_valid(pid):
        return False
    if not process_identity_is_valid(recorded_identity, expected_pid=pid):
        return None
    if not process_identity_scope_is_local(recorded_identity):
        return None
    try:
        os.kill(pid, 0)
    except PermissionError:
        # EPERM proves numeric existence but not identity. Continue to the
        # readable kernel identity, while retaining on unreadability below.
        pass
    except (ProcessLookupError, OSError, ValueError):
        return False
    current = _process_identity(pid)
    return compare_process_identities(recorded_identity, current)


def _pid_record_alive(pid: int, recorded_identity: Optional[dict]) -> bool:
    """Conservatively retain uncertain owners as possibly alive."""
    return _pid_record_status(pid, recorded_identity) is not False


def _pid_identity_verified(pid: int, recorded_identity: Optional[dict]) -> bool:
    return _pid_record_status(pid, recorded_identity) is True


def _validate_process_authority(record: dict) -> None:
    pid = record.get("pid")
    identity = record.get("process_identity")
    if not process_pid_is_valid(pid):
        raise ValueError("manager PID must be a positive integer")
    if identity is not None and not process_identity_record_is_recognized(
        identity, expected_pid=pid,
    ):
        raise ValueError("manager process identity is malformed")


class StateFileCorruptionError(RuntimeError):
    """Raised when persistent manager state exists but is not trustworthy."""


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
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as exc:
            raise StateFileCorruptionError(
                f"Invalid JSON in {self.state_file}; refusing to treat "
                "manager state as empty"
            ) from exc
        if not isinstance(data, list):
            raise StateFileCorruptionError(
                f"Expected a JSON list in {self.state_file}; refusing to "
                "treat manager state as empty"
            )
        for index, record in enumerate(data):
            if not isinstance(record, dict):
                raise StateFileCorruptionError(
                    f"Invalid record {index} in {self.state_file}"
                )
            try:
                _server_from_dict(record)
                _validate_process_authority(record)
            except (TypeError, ValueError) as exc:
                raise StateFileCorruptionError(
                    f"Invalid record {index} in {self.state_file}"
                ) from exc
        return data

    def _write(self, servers: List[dict]) -> None:
        fd, temp_name = tempfile.mkstemp(
            dir=self.state_dir,
            prefix=f".{self.state_file.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(servers, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_name, self.state_file)
            directory_fd = os.open(
                self.state_dir,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def archive_corrupt_state(self) -> Optional[Path]:
        """Atomically preserve corrupt state and restore first-use behavior.

        This explicit recovery operation never treats corruption as verified
        empty state.  It retains the original bytes in a uniquely named
        sibling file for audit or manual repair.
        """
        with FileLock(self.lock_file):
            try:
                self._read()
            except StateFileCorruptionError:
                fd, archive_name = tempfile.mkstemp(
                    dir=self.state_dir,
                    prefix=f"{self.state_file.name}.corrupt.",
                    suffix=".json",
                )
                os.close(fd)
                archive_path = Path(archive_name)
                moved = False
                try:
                    os.replace(self.state_file, archive_path)
                    moved = True
                    directory_fd = os.open(
                        self.state_dir,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    )
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                finally:
                    if not moved:
                        archive_path.unlink(missing_ok=True)
                return archive_path
        return None

    def add_server(self, server: GPUServerState) -> None:
        replacement = asdict(server)
        _validate_process_authority(replacement)
        with FileLock(self.lock_file):
            servers = self._read()
            updated = []
            found = False
            for existing in servers:
                if existing.get("chain_id") != server.chain_id:
                    updated.append(existing)
                    continue
                if not found:
                    merged = dict(existing)
                    merged.update(replacement)
                    updated.append(merged)
                    found = True
            if not found:
                updated.append(replacement)
            for record in updated:
                _validate_process_authority(record)
            self._write(updated)

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

    def update_server_submission(
        self,
        chain_id: str,
        *,
        slurm_job_ids: List[str],
        submission_intent: Optional[dict],
    ) -> None:
        """Atomically persist exact job ownership and its submission fence."""
        with FileLock(self.lock_file):
            servers = self._read()
            for server in servers:
                if server.get("chain_id") == chain_id:
                    server["slurm_job_ids"] = list(slurm_job_ids)
                    server["submission_intent"] = submission_intent
                    self._write(servers)
                    return
            raise StateFileCorruptionError(
                f"Cannot update missing server submission authority {chain_id!r}"
            )

    def list_servers(self) -> List[GPUServerState]:
        with FileLock(self.lock_file):
            raw = self._read()
        return [_server_from_dict(s) for s in raw]

    def is_server_alive(self, server: GPUServerState) -> bool:
        """Check if the manager process PID is still running."""
        return _pid_record_alive(server.pid, server.process_identity)

    def is_server_identity_verified(self, server: GPUServerState) -> bool:
        """True only when PID, boot ID, and process start time all match."""
        return _pid_identity_verified(server.pid, server.process_identity)

    def cleanup_dead_servers(
        self,
        alive_job_ids: Optional[SchedulerJobIds] = None,
        job_liveness: Optional[
            Callable[[List[str]], SchedulerJobIds]
        ] = None,
    ) -> List[GPUServerState]:
        """Remove dead managers only when their recorded jobs are also dead.

        ``alive_job_ids`` must be the provenance-carrying result of
        ``SchedulerSnapshot.keys()``.  ``job_liveness`` is an optional batch
        resolver called once with all recorded job IDs and must return that
        same carrier.  Plain iterables are treated as unknown.  If neither
        trusted source is supplied, records with job IDs are conservatively
        retained and marked ``quarantined``.
        """
        with FileLock(self.lock_file):
            servers = self._read()
            known_alive = _authoritative_job_ids(alive_job_ids)
            if job_liveness is not None:
                all_job_ids = sorted(
                    {
                        str(job_id)
                        for server in servers
                        for job_id in server.get("slurm_job_ids", [])
                    }
                )
                callback_alive = _authoritative_job_ids(
                    job_liveness(all_job_ids)
                )
                if known_alive is None and callback_alive is not None:
                    known_alive = callback_alive
                elif callback_alive is not None:
                    known_alive.update(callback_alive)

            alive = []
            dead = []
            for s in servers:
                pid = s.get("pid", 0)
                owner_status = _pid_record_status(
                    pid, s.get("process_identity"),
                )
                if owner_status is True:
                    alive.append(s)
                elif owner_status is None:
                    s["lifecycle_state"] = "quarantined"
                    alive.append(s)
                else:
                    recorded_jobs = {
                        str(job_id) for job_id in s.get("slurm_job_ids", [])
                    }
                    jobs_may_be_alive = s.get("submission_intent") is not None or (
                        bool(recorded_jobs) and (
                            known_alive is None
                            or bool(recorded_jobs.intersection(known_alive))
                        )
                    )
                    if jobs_may_be_alive:
                        s["lifecycle_state"] = "quarantined"
                        alive.append(s)
                    else:
                        dead.append(s)
            self._write(alive)
        result = []
        for s in dead:
            try:
                result.append(_server_from_dict(s))
            except TypeError:
                continue
        return result


def _server_from_dict(s: dict) -> GPUServerState:
    """Load known fields while tolerating fields written by newer versions."""
    known = GPUServerState.__dataclass_fields__
    return GPUServerState(**{key: value for key, value in s.items() if key in known})


def _authoritative_job_ids(
    job_ids: Optional[SchedulerJobIds],
) -> Optional[set]:
    """Return IDs only when their iterable represents a verified snapshot."""
    if not isinstance(job_ids, SchedulerJobIds) or not job_ids.authoritative:
        return None
    return {str(job_id) for job_id in job_ids}
