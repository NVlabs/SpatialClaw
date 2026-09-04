"""Persistent state for active experiment chain processes."""

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
class ExperimentState:
    experiment_id: str
    benchmark: str
    model_name: str          # e.g. "qwen3.5-122b-a10b"
    experiment_name: str     # user-given name
    pid: int                 # chain process PID
    # Kernel-backed launch identity (boot ID + /proc start ticks).  Legacy
    # records omit it and therefore cannot authorize PID signaling.
    process_identity: Optional[dict] = None
    slurm_job_ids: List[str] = field(default_factory=list)
    started_at: str = ""
    account: str = ""
    partition: str = ""
    gpus: int = 8
    concurrency: int = 8
    subsample: int = 0
    work_dir: str = ""       # absolute path to work_dir for this experiment
    total_samples: int = -1  # -1 = unknown yet
    # queue_deferred_retry_required is a managed, result-neutral terminal
    # handoff: the CPU evaluator exceeded its pending bound and must be
    # explicitly relaunched later after its service lease was released.
    status: str = "running"
    experiment_type: str = "agent"  # "agent" | "cot"
    scheduled_for: str = ""  # ISO timestamp; "" = start immediately
    # Optional ownership/lifecycle metadata.  Defaults keep legacy records
    # and callers valid until lease-aware orchestration is enabled.
    lease_id: Optional[str] = None
    cohort_id: Optional[str] = None
    workload_policy_digest: Optional[str] = None
    service_manifest_path: Optional[str] = None
    service_manifest_digest: Optional[str] = None
    lifecycle_state: Optional[str] = None
    # Crash fence for the single evaluator submission currently crossing the
    # sbatch acknowledgement boundary.  A prepared/accepted intent is retained
    # even when an authoritative squeue snapshot has no match: a successful
    # submission can be temporarily invisible to squeue.
    evaluator_submission_comment: Optional[str] = None
    evaluator_submission_phase: Optional[str] = None
    evaluator_submission_job_ids: List[str] = field(default_factory=list)
    evaluator_submission_observed: bool = False


def _process_identity(pid: int) -> Optional[dict]:
    """Return the kernel-backed identity for a local process, if readable."""
    return lifecycle_process_identity(pid)


def _pid_record_status(
    pid: int, recorded_identity: Optional[dict],
) -> Optional[bool]:
    """Return exact owner status: matched, proven dead/reused, or unknown.

    ``True`` requires an exact stored/current identity match. ``False`` means
    the PID is absent or has been reused. ``None`` preserves ownership when a
    numeric PID exists but either identity is missing or unreadable.
    """
    if not process_pid_is_valid(pid):
        return False
    if not process_identity_is_valid(recorded_identity, expected_pid=pid):
        return None
    if not process_identity_scope_is_local(recorded_identity):
        return None
    try:
        os.kill(pid, 0)
    except PermissionError:
        # EPERM establishes neither ownership nor death.  Continue to the
        # kernel identity check, which may still be readable.
        pass
    except (ProcessLookupError, OSError, ValueError):
        return False
    current_identity = _process_identity(pid)
    return compare_process_identities(recorded_identity, current_identity)


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


class ExperimentStateManager:
    """Read/write agent_manager_state.json with file locking."""

    def __init__(self, project_root: Path):
        self.state_dir = project_root / "spatial_agent" / "logs"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / "agent_manager_state.json"
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
                _experiment_from_dict(record)
                _validate_process_authority(record)
            except (TypeError, ValueError) as exc:
                raise StateFileCorruptionError(
                    f"Invalid record {index} in {self.state_file}"
                ) from exc
            comment = record.get("evaluator_submission_comment")
            phase = record.get("evaluator_submission_phase")
            intent_jobs = record.get("evaluator_submission_job_ids", [])
            observed = record.get("evaluator_submission_observed", False)
            if (
                (comment is None) != (phase is None)
                or (
                    comment is not None
                    and (not isinstance(comment, str) or not comment)
                )
                or phase not in {
                    None, "prepared", "accepted_unseen",
                    "accepted_unresolved", "accepted_exact", "observed",
                    "ambiguous_multiple",
                }
                or not isinstance(intent_jobs, list)
                or any(
                    not isinstance(job_id, str) or not job_id.isdigit()
                    for job_id in intent_jobs
                )
                or not isinstance(observed, bool)
            ):
                raise StateFileCorruptionError(
                    f"Invalid evaluator submission fence in record {index} "
                    f"of {self.state_file}"
                )
        return data

    def _write(self, experiments: List[dict]) -> None:
        fd, temp_name = tempfile.mkstemp(
            dir=self.state_dir,
            prefix=f".{self.state_file.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(experiments, f, indent=2)
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

        This is an explicit operator-recovery primitive, not an automatic
        fallback: valid state is never moved, and corrupt bytes are retained
        in a uniquely named sibling file for audit or manual repair.
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

    def add_experiment(self, exp: ExperimentState) -> None:
        replacement = asdict(exp)
        _validate_process_authority(replacement)
        with FileLock(self.lock_file):
            experiments = self._read()
            updated = []
            found = False
            for existing in experiments:
                if existing.get("experiment_id") != exp.experiment_id:
                    updated.append(existing)
                    continue
                if not found:
                    # Retain fields written by a newer version while updating
                    # the fields understood by this version.
                    merged = dict(existing)
                    merged.update(replacement)
                    # A detached child may submit/update before its launching
                    # parent persists the initial record.  Never overwrite that
                    # forward progress with the parent's empty placeholder.
                    if (
                        existing.get("slurm_job_ids")
                        and not replacement.get("slurm_job_ids")
                    ):
                        merged["slurm_job_ids"] = existing["slurm_job_ids"]
                    for lifecycle_key in (
                        "service_manifest_path",
                        "service_manifest_digest",
                        "workload_policy_digest",
                        "process_identity",
                        "evaluator_submission_comment",
                        "evaluator_submission_phase",
                        "evaluator_submission_job_ids",
                    ):
                        if (
                            existing.get(lifecycle_key)
                            and not replacement.get(lifecycle_key)
                        ):
                            if (
                                lifecycle_key == "process_identity"
                                and existing.get("pid") != replacement.get("pid")
                            ):
                                raise ValueError(
                                    "cannot transfer a process identity across PIDs"
                                )
                            merged[lifecycle_key] = existing[lifecycle_key]
                    updated.append(merged)
                    found = True
            if not found:
                updated.append(replacement)
            for record in updated:
                _validate_process_authority(record)
            self._write(updated)

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

    def update_experiment_submission(
        self,
        experiment_id: str,
        *,
        comment: Optional[str],
        phase: Optional[str],
        job_ids: List[str],
        observed: bool = False,
    ) -> None:
        """Atomically persist the evaluator sbatch fence and exact job IDs.

        The state-file rename and directory fsync make each transition durable.
        When an exact ID is known it is adopted into ``slurm_job_ids`` in the
        same write, so no crash can publish the ID without durable ownership.
        """
        allowed_phases = {
            None,
            "prepared",
            "accepted_unseen",
            "accepted_unresolved",
            "accepted_exact",
            "observed",
            "ambiguous_multiple",
        }
        if phase not in allowed_phases:
            raise ValueError(f"Invalid evaluator submission phase: {phase!r}")
        normalized_jobs = [str(job_id) for job_id in job_ids]
        if any(not job_id.isdigit() for job_id in normalized_jobs):
            raise ValueError("Evaluator submission job IDs must be numeric")
        if (comment is None) != (phase is None):
            raise ValueError("Evaluator submission comment and phase must clear together")
        if comment is not None and (not isinstance(comment, str) or not comment):
            raise ValueError("Evaluator submission comment must be non-empty")

        with FileLock(self.lock_file):
            experiments = self._read()
            found = False
            for experiment in experiments:
                if experiment.get("experiment_id") != experiment_id:
                    continue
                found = True
                existing_jobs = {
                    str(job_id)
                    for job_id in experiment.get("slurm_job_ids", [])
                    if job_id
                }
                experiment["slurm_job_ids"] = sorted(
                    existing_jobs | set(normalized_jobs), key=int,
                )
                experiment["evaluator_submission_comment"] = comment
                experiment["evaluator_submission_phase"] = phase
                experiment["evaluator_submission_job_ids"] = normalized_jobs
                experiment["evaluator_submission_observed"] = bool(observed)
                break
            if not found:
                raise KeyError(f"Unknown experiment {experiment_id!r}")
            self._write(experiments)

    def update_experiment_status(self, experiment_id: str, status: str) -> None:
        with FileLock(self.lock_file):
            experiments = self._read()
            for e in experiments:
                if e.get("experiment_id") == experiment_id:
                    e["status"] = status
                    break
            self._write(experiments)

    def update_experiment_lifecycle(self, experiment_id: str, **fields) -> None:
        """Update only lifecycle metadata on one experiment record.

        Keeping this as a partial update avoids replacing scheduler/job fields
        that may have been written by another manager callback.
        """
        allowed = {
            "lease_id",
            "cohort_id",
            "workload_policy_digest",
            "service_manifest_path",
            "service_manifest_digest",
            "lifecycle_state",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(
                f"Unknown experiment lifecycle field(s): {sorted(unknown)}"
            )
        with FileLock(self.lock_file):
            experiments = self._read()
            for experiment in experiments:
                if experiment.get("experiment_id") == experiment_id:
                    experiment.update(fields)
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
        return [_experiment_from_dict(e) for e in raw]

    def is_experiment_alive(self, exp: ExperimentState) -> bool:
        """True only when the recorded process identity still matches."""
        return self.process_owner_status(exp.pid, exp.process_identity) is True

    @staticmethod
    def process_owner_status(
        pid: int, recorded_identity: Optional[dict],
    ) -> Optional[bool]:
        """Return exact owner match, proven death/reuse, or uncertainty."""
        return _pid_record_status(pid, recorded_identity)

    def cleanup_dead_experiments(
        self,
        alive_job_ids: Optional[SchedulerJobIds] = None,
        job_liveness: Optional[
            Callable[[List[str]], SchedulerJobIds]
        ] = None,
    ) -> List[ExperimentState]:
        """Remove dead running experiments only after their jobs are dead.

        ``alive_job_ids`` must be the provenance-carrying result of
        ``SchedulerSnapshot.keys()``.  Alternatively, ``job_liveness`` is
        called once with every recorded job ID and must return that same typed
        carrier for the live subset.  Plain iterables are intentionally
        treated as unknown so an accidental empty ``set()`` cannot authorize
        deletion.  This keeps scheduler batching outside the per-record loop.

        Without either source, a dead running experiment that still records
        jobs is conservatively retained and marked ``quarantined``.  When a
        snapshot confirms a recorded job is live, it is retained and marked
        ``orphaned`` because its manager process has exited.
        """
        with FileLock(self.lock_file):
            experiments = self._read()
            known_alive = _authoritative_job_ids(alive_job_ids)
            if job_liveness is not None:
                all_job_ids = sorted(
                    {
                        str(job_id)
                        for experiment in experiments
                        for job_id in experiment.get("slurm_job_ids", [])
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
            for e in experiments:
                pid = e.get("pid", 0)
                status = e.get("status", "running")
                # Keep completed/failed experiments for history
                if status != "running":
                    alive.append(e)
                    continue
                owner_status = _pid_record_status(
                    pid, e.get("process_identity"),
                )
                if owner_status is True:
                    alive.append(e)
                    continue
                if owner_status is None:
                    # A numerically live PID without readable, matching
                    # identity is not permission to clean up or recover its
                    # allocations.  Preserve the ownership record visibly.
                    e["lifecycle_state"] = "quarantined"
                    alive.append(e)
                    continue
                recorded_jobs = {
                    str(job_id) for job_id in e.get("slurm_job_ids", [])
                }
                confirmed_live = (
                    known_alive is not None
                    and bool(recorded_jobs.intersection(known_alive))
                )
                submission_fenced = bool(
                    e.get("evaluator_submission_comment")
                    and e.get("evaluator_submission_phase")
                    and not e.get("evaluator_submission_observed", False)
                )
                unresolved = (
                    submission_fenced
                    or (bool(recorded_jobs) and known_alive is None)
                )
                if confirmed_live or unresolved:
                    e["lifecycle_state"] = (
                        "orphaned" if confirmed_live else "quarantined"
                    )
                    alive.append(e)
                else:
                    dead.append(e)
            self._write(alive)
        result = []
        for e in dead:
            try:
                result.append(_experiment_from_dict(e))
            except TypeError:
                continue
        return result


def _experiment_from_dict(e: dict) -> ExperimentState:
    """Load known fields while tolerating fields written by newer versions."""
    known = ExperimentState.__dataclass_fields__
    return ExperimentState(**{key: value for key, value in e.items() if key in known})


def _authoritative_job_ids(
    job_ids: Optional[SchedulerJobIds],
) -> Optional[set]:
    """Return IDs only when their iterable represents a verified snapshot."""
    if not isinstance(job_ids, SchedulerJobIds) or not job_ids.authoritative:
        return None
    return {str(job_id) for job_id in job_ids}
