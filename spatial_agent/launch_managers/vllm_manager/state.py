"""Persistent state for active server chain processes."""

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
    # Optional lifecycle metadata.  Defaults keep legacy state files and
    # callers valid while allowing newer managers to identify compatible
    # pools without relying on the served model name alone.
    pool_id: Optional[str] = None
    namespace: Optional[str] = None
    spec_digest: Optional[str] = None
    topology_digest: Optional[str] = None
    desired_instances: Optional[int] = None
    lifecycle_state: Optional[str] = None
    process_identity: Optional[dict] = None
    # Written before invoking sbatch and cleared only after the exact submitted
    # allocation is visible and durably adopted.  A surviving intent fences the
    # logical slot across manager crashes and scheduler visibility lag.
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
                _chain_from_dict(record)
                _validate_process_authority(record)
            except (TypeError, ValueError) as exc:
                raise StateFileCorruptionError(
                    f"Invalid record {index} in {self.state_file}"
                ) from exc
        return data

    def _write(self, chains: List[dict]) -> None:
        fd, temp_name = tempfile.mkstemp(
            dir=self.state_dir,
            prefix=f".{self.state_file.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(chains, f, indent=2)
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
            # os.replace removes the temporary path on success.  On a write
            # or replace failure, do not leave stale state fragments behind.
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

    def add_chain(self, chain: ChainState) -> None:
        replacement = asdict(chain)
        _validate_process_authority(replacement)
        with FileLock(self.lock_file):
            chains = self._read()
            updated = []
            found = False
            for existing in chains:
                if existing.get("chain_id") != chain.chain_id:
                    updated.append(existing)
                    continue
                if not found:
                    # Preserve forward-compatible fields that this version
                    # does not know about while updating the known record.
                    merged = dict(existing)
                    merged.update(replacement)
                    updated.append(merged)
                    found = True
            if not found:
                updated.append(replacement)
            for record in updated:
                _validate_process_authority(record)
            self._write(updated)

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

    def update_chain_submission(
        self,
        chain_id: str,
        *,
        slurm_job_ids: List[str],
        submission_intent: Optional[dict],
    ) -> None:
        """Atomically persist exact job ownership and its submission fence."""
        with FileLock(self.lock_file):
            chains = self._read()
            for chain in chains:
                if chain.get("chain_id") == chain_id:
                    chain["slurm_job_ids"] = list(slurm_job_ids)
                    chain["submission_intent"] = submission_intent
                    self._write(chains)
                    return
            raise StateFileCorruptionError(
                f"Cannot update missing chain submission authority {chain_id!r}"
            )

    def list_chains(self) -> List[ChainState]:
        with FileLock(self.lock_file):
            raw = self._read()
        return [_chain_from_dict(c) for c in raw]

    def is_chain_alive(self, chain: ChainState) -> bool:
        """Check if the manager process PID is still running."""
        return _pid_record_alive(chain.pid, chain.process_identity)

    def is_chain_identity_verified(self, chain: ChainState) -> bool:
        """True only when PID, boot ID, and process start time all match."""
        return _pid_identity_verified(chain.pid, chain.process_identity)

    def cleanup_dead_chains(
        self,
        alive_job_ids: Optional[SchedulerJobIds] = None,
        job_liveness: Optional[
            Callable[[List[str]], SchedulerJobIds]
        ] = None,
    ) -> List[ChainState]:
        """Remove dead managers only when their recorded jobs are also dead.

        ``alive_job_ids`` must be the provenance-carrying result of
        ``SchedulerSnapshot.keys()``.  Alternatively, ``job_liveness`` is
        called once with all recorded job IDs and must return that same typed
        carrier for the live subset.  Plain iterables are deliberately treated
        as unknown so they cannot authorize record deletion.  This keeps
        scheduler batching outside the per-record cleanup loop.

        With neither source, a dead manager that still records job IDs is
        conservatively retained and marked ``quarantined``.  This preserves
        the no-argument API without risking deletion of the only ownership
        record for a live allocation.
        """
        with FileLock(self.lock_file):
            chains = self._read()
            known_alive = _authoritative_job_ids(alive_job_ids)
            if job_liveness is not None:
                all_job_ids = sorted(
                    {
                        str(job_id)
                        for chain in chains
                        for job_id in chain.get("slurm_job_ids", [])
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
            for c in chains:
                pid = c.get("pid", 0)
                owner_status = _pid_record_status(
                    pid, c.get("process_identity"),
                )
                if owner_status is True:
                    alive.append(c)
                elif owner_status is None:
                    c["lifecycle_state"] = "quarantined"
                    alive.append(c)
                else:
                    recorded_jobs = {
                        str(job_id) for job_id in c.get("slurm_job_ids", [])
                    }
                    # A pre-sbatch intent means an accepted allocation may not
                    # yet be visible.  One empty scheduler snapshot must never
                    # erase the only durable recovery key (the exact comment).
                    jobs_may_be_alive = c.get("submission_intent") is not None or (
                        bool(recorded_jobs) and (
                            known_alive is None
                            or bool(recorded_jobs.intersection(known_alive))
                        )
                    )
                    if jobs_may_be_alive:
                        c["lifecycle_state"] = "quarantined"
                        alive.append(c)
                    else:
                        dead.append(c)
            self._write(alive)
        return [_chain_from_dict(c) for c in dead if _is_valid_chain(c)]


def _chain_from_dict(c: dict) -> ChainState:
    """Load known fields while tolerating fields written by newer versions."""
    known = ChainState.__dataclass_fields__
    return ChainState(**{key: value for key, value in c.items() if key in known})


def _authoritative_job_ids(
    job_ids: Optional[SchedulerJobIds],
) -> Optional[set]:
    """Return IDs only when their iterable represents a verified snapshot."""
    if not isinstance(job_ids, SchedulerJobIds) or not job_ids.authoritative:
        return None
    return {str(job_id) for job_id in job_ids}


def _is_valid_chain(c: dict) -> bool:
    try:
        _chain_from_dict(c)
        return True
    except TypeError:
        return False
