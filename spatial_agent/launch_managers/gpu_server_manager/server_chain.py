"""
Background GPU server chain manager.

Each chain runs as an independent subprocess that manages a continuous loop
of 4-hour SLURM jobs with overlap for zero-downtime restarts.

Can be invoked directly:
    python -m spatial_agent.launch_managers.gpu_server_manager.server_chain --config '{"chain_id": ..., ...}'
"""

import argparse
import datetime
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from spatial_agent.launch_managers.slurm_utils import (
    cancel_job,
    check_slurm_available,
    get_exact_job_states,
    get_job_status,
    get_user_jobs_snapshot,
    is_job_alive,
    resolve_partitions,
    wait_for_job_visible,
)
from spatial_agent.launch_managers.lifecycle import (
    LifecycleError,
    LifecycleStore,
    format_service_job_comment,
    open_unique_text_artifact,
    process_identity,
    reconcile_runtime,
    write_unique_text_artifact,
    write_json_atomic,
)
from spatial_agent.launch_managers.gpu_server_manager.state import (
    GPUServerState,
    GPUServerStateManager,
)


# Stop a chain after this many consecutive jobs that died within
# _FAST_FAIL_SECONDS of starting (e.g. server crashes immediately).
_MAX_FAST_FAILURES = 3
_FAST_FAIL_SECONDS = 300
_CANCEL_VERIFY_ATTEMPTS = 60
_CANCEL_VERIFY_INTERVAL_SECONDS = 2.0
_RECONCILE_INTERVAL_SECONDS = 300.0
_RECONCILE_RETRY_SECONDS = 30.0
_AUTHORITY_VERIFY_ATTEMPTS = 5
_AUTHORITY_VERIFY_INTERVAL_SECONDS = 5.0
_AMBIGUOUS_SUBMIT_VERIFY_ATTEMPTS = 10
_AMBIGUOUS_SUBMIT_VERIFY_INTERVAL_SECONDS = 1.0
_ACCEPTED_ID_VISIBILITY_ATTEMPTS = 10
_ACCEPTED_ID_VISIBILITY_INTERVAL_SECONDS = 1.0
_JOB_START_TIMEOUT_SECONDS = 4 * 60 * 60
_FINALIZE_VERIFY_ATTEMPTS = 75
_FINALIZE_VERIFY_INTERVAL_SECONDS = 2.0
_REMOTE_DRAIN_TIMEOUT_SECONDS = 90.0


class ServerChain:
    """Manages continuous chain of 4-hour SLURM jobs for one GPU server."""

    def __init__(
        self,
        chain_id: str,
        account: str,
        partition: str,
        gpus: int = 1,
        reconstruct_backend: str = "pi3",
        time_limit: str = "4:00:00",
        restart_before_minutes: int = 20,
        project_root: str = ".",
        pool_id: str = "",
        namespace: str = "",
        tool_spec_digest: str = "",
        topology_digest: str = "",
        desired_instances: int = 1,
        auto_stop_when_unowned: bool = False,
        bootstrap_grace_minutes: int = 30,
    ):
        self.chain_id = chain_id
        self.account = account
        self.partition = resolve_partitions(partition, account)
        self.gpus = gpus
        self.reconstruct_backend = reconstruct_backend
        self.time_limit = time_limit
        if isinstance(restart_before_minutes, bool) or not isinstance(
            restart_before_minutes, int
        ):
            raise ValueError("restart_before_minutes must be an integer")
        self.restart_before_minutes = restart_before_minutes
        self.project_root = Path(project_root)
        self.pool_id = pool_id
        self.namespace = namespace
        self.tool_spec_digest = tool_spec_digest
        self.topology_digest = topology_digest
        self.desired_instances = int(desired_instances)
        self.auto_stop_when_unowned = bool(auto_stop_when_unowned)
        self.bootstrap_grace_seconds = max(0, int(bootstrap_grace_minutes) * 60)

        self.total_seconds = self._parse_time(time_limit)
        if self.total_seconds <= 0:
            raise ValueError("time_limit must be greater than zero")
        if restart_before_minutes < 0:
            raise ValueError("restart_before_minutes must be non-negative")
        if restart_before_minutes * 60 >= self.total_seconds:
            raise ValueError(
                "restart_before_minutes must be shorter than time_limit"
            )
        self.wait_seconds = self.total_seconds - (restart_before_minutes * 60)

        self.running = True
        self._stop_event = threading.Event()  # For fast shutdown
        self.job_counter = 1
        self.current_job_id: Optional[str] = None
        self.job_submit_time: Optional[float] = None
        self.submitted_job_ids: List[str] = []
        self._ever_had_owner = False
        self._chain_started_monotonic = time.monotonic()
        self._shutdown_drain_failed = False
        self._shutdown_requested = False
        self._shutdown_signal: Optional[int] = None
        self._last_reconcile_attempt = float("-inf")
        self._last_reconcile_ok = False
        self._replacement_blocked = False
        self._submission_intent: Optional[dict] = None

        self.log_dir = self.project_root / "spatial_agent" / "logs" / "slurm_gpu_server"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.gpu_server_script = Path(__file__).parent / "run_gpu_server.sh"
        self.state_manager = GPUServerStateManager(self.project_root)
        self.lifecycle_store = LifecycleStore(self.project_root)

    def _parse_time(self, time_str: str) -> int:
        if not isinstance(time_str, str):
            raise ValueError(f"Invalid time format: {time_str}")
        parts = time_str.split(":")
        try:
            if len(parts) == 3:
                h, m, s = map(int, parts)
            elif len(parts) == 2:
                h = 0
                m, s = map(int, parts)
            else:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid time format: {time_str}") from exc
        if h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
            raise ValueError(f"Invalid time format: {time_str}")
        return h * 3600 + m * 60 + s

    def _ts(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _format_time(self, seconds: int) -> str:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _log(self, msg: str, flush: bool = True):
        print(f"[{self._ts()}] {msg}", flush=flush)

    def _reconcile_runtime_authority(self, *, force: bool = False) -> bool:
        """Refresh ownership evidence at a bounded, fail-closed cadence."""
        if not self.pool_id:
            return True
        now = time.monotonic()
        minimum_age = (
            _RECONCILE_RETRY_SECONDS if force
            else _RECONCILE_INTERVAL_SECONDS
        )
        if now - self._last_reconcile_attempt < minimum_age:
            return self._last_reconcile_ok
        self._last_reconcile_attempt = now
        try:
            reconcile_runtime(self.project_root)
        except Exception as exc:
            self._last_reconcile_ok = False
            self._log(
                "Runtime ownership reconciliation unavailable; pausing "
                f"teardown/replacement: {exc}"
            )
            return False
        self._last_reconcile_ok = True
        return True

    def _await_runtime_authority(self) -> bool:
        for attempt in range(_AUTHORITY_VERIFY_ATTEMPTS):
            if not self.running or self._shutdown_requested:
                return False
            self._last_reconcile_attempt = float("-inf")
            if self._reconcile_runtime_authority(force=True):
                return True
            if attempt + 1 < _AUTHORITY_VERIFY_ATTEMPTS and self._stop_event.wait(
                timeout=_AUTHORITY_VERIFY_INTERVAL_SECONDS
            ):
                return False
        return False

    def _submission_authority_available(self) -> bool:
        """Require fresh scheduler authority immediately before each sbatch."""
        if not self.pool_id:
            # Preserve historical unmanaged availability during staged
            # rollout. Managed submissions still require fresh lifecycle and
            # scheduler authority; legacy renewal is not permanently stranded
            # by a short control-plane outage.
            return True
        for attempt in range(_AUTHORITY_VERIFY_ATTEMPTS):
            if not self.running or self._shutdown_requested:
                return False
            jobs = get_user_jobs_snapshot(force=True)
            scheduler_ok = bool(getattr(jobs, "authoritative", False))
            runtime_ok = False
            if scheduler_ok:
                self._last_reconcile_attempt = float("-inf")
                runtime_ok = self._reconcile_runtime_authority(force=True)
                if runtime_ok:
                    return True
            reason = (
                "runtime ownership reconciliation"
                if scheduler_ok and not runtime_ok
                else "authoritative Slurm state"
            )
            self._log(
                f"{reason.capitalize()} unavailable before submission "
                f"(attempt {attempt + 1}/{_AUTHORITY_VERIFY_ATTEMPTS}); "
                "allocation remains blocked."
            )
            if attempt + 1 < _AUTHORITY_VERIFY_ATTEMPTS and self._stop_event.wait(
                timeout=_AUTHORITY_VERIFY_INTERVAL_SECONDS
            ):
                return False
        return False

    def _fresh_owner_epoch_available(self) -> bool:
        """Recheck renewable demand at the final managed sbatch boundary."""
        if not self.pool_id:
            return True
        try:
            owner_present = bool(
                self.lifecycle_store.should_renew_pool(self.pool_id)
            )
        except Exception as exc:
            self._log(
                "Fresh owner epoch is unreadable immediately before sbatch; "
                f"submission refused: {exc}"
            )
            return False
        if not owner_present:
            self._log(
                "No fresh compatible owner epoch remains immediately before "
                "sbatch; submission refused."
            )
        return owner_present

    def _submission_comment(self, generation: int) -> str:
        return format_service_job_comment(
            pool_id=self.pool_id,
            slot_id=self.chain_id,
            generation=generation,
        )

    @staticmethod
    def _parse_parsable_job_id(stdout: str) -> Optional[str]:
        lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
        if len(lines) != 1:
            return None
        fields = lines[0].split(";")
        if len(fields) not in {1, 2}:
            return None
        if len(fields) == 2 and (
            not fields[1] or any(character.isspace() for character in fields[1])
        ):
            return None
        return fields[0] if fields[0].isdigit() else None

    def _submission_payload(
        self, generation: int, *, phase: str, job_ids: Iterable[str] = (),
    ) -> dict:
        return {
            "schema_version": 1,
            "comment": self._submission_comment(generation),
            "generation": int(generation),
            "phase": phase,
            "job_ids": list(dict.fromkeys(
                str(job_id) for job_id in job_ids if job_id
            )),
            "updated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        }

    def _persist_submission_state(
        self, submission_intent: Optional[dict],
    ) -> bool:
        """Durably write the pre-sbatch fence and all adopted exact IDs."""
        previous = self._submission_intent
        if submission_intent is not None:
            # Preserve an in-process fence even if either durable writer fails.
            self._submission_intent = dict(submission_intent)
        exact = list(dict.fromkeys(
            str(job_id) for job_id in self.submitted_job_ids if job_id
        ))
        try:
            if self.pool_id:
                self.lifecycle_store.update_chain_submission(
                    self.pool_id,
                    self.chain_id,
                    job_ids=exact,
                    submission_intent=submission_intent,
                )
            self.state_manager.update_server_submission(
                self.chain_id,
                slurm_job_ids=exact,
                submission_intent=submission_intent,
            )
        except Exception as exc:
            if submission_intent is None:
                self._submission_intent = previous or {
                    "schema_version": 1,
                    "phase": "clear_persistence_failed",
                    "comment": "",
                    "updated_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                }
            self._log(f"Could not persist service submission authority: {exc}")
            return False
        self._submission_intent = (
            dict(submission_intent) if submission_intent is not None else None
        )
        return True

    def _submission_pending(self) -> Optional[bool]:
        """Return pending-fence authority, or None when stores are unreadable."""
        if self._submission_intent is not None:
            return True
        try:
            for server in self.state_manager.list_servers():
                if (
                    server.chain_id == self.chain_id
                    and server.submission_intent is not None
                ):
                    return True
            if self.pool_id:
                pool = self.lifecycle_store.snapshot().get("pools", {}).get(
                    self.pool_id
                )
                if not isinstance(pool, dict):
                    return None
                chain = (pool.get("chains") or {}).get(self.chain_id)
                if not isinstance(chain, dict):
                    return None
                if chain.get("submission_intent") is not None:
                    return True
        except Exception as exc:
            self._log(f"Could not read submission-fence authority: {exc}")
            return None
        return False

    def _recover_ambiguous_submission(self, generation: int) -> Optional[str]:
        """Wait until an accepted submission has exact actionable ownership."""
        expected = self._submission_comment(generation)
        for attempt in range(_AMBIGUOUS_SUBMIT_VERIFY_ATTEMPTS):
            if not self.running or self._shutdown_requested:
                break
            jobs = get_user_jobs_snapshot(force=True)
            if getattr(jobs, "authoritative", False):
                matches = sorted(
                    str(job_id) for job_id, info in dict.items(jobs)
                    if isinstance(info, dict) and info.get("comment") == expected
                )
                if len(matches) == 1:
                    return matches[0]
                if len(matches) > 1:
                    for job_id in matches:
                        if job_id not in self.submitted_job_ids:
                            self.submitted_job_ids.append(job_id)
                    ambiguous = self._submission_payload(
                        generation,
                        phase="ambiguous_multiple",
                        job_ids=matches,
                    )
                    self._persist_submission_state(ambiguous)
                    self._replacement_blocked = True
                    self._retain_quarantined_ownership(
                        "multiple live allocations matched one exact accepted "
                        f"submission comment: {matches}"
                    )
                    return None
            self._log(
                "Accepted service submission is not yet exactly visible; "
                "retaining its durable comment fence "
                f"(attempt {attempt + 1}/"
                f"{_AMBIGUOUS_SUBMIT_VERIFY_ATTEMPTS})."
            )
            if attempt + 1 < _AMBIGUOUS_SUBMIT_VERIFY_ATTEMPTS and self._stop_event.wait(
                timeout=_AMBIGUOUS_SUBMIT_VERIFY_INTERVAL_SECONDS
            ):
                break
        self._replacement_blocked = True
        self._retain_quarantined_ownership(
            "accepted Slurm submission could not be resolved to exactly one "
            f"live allocation after {_AMBIGUOUS_SUBMIT_VERIFY_ATTEMPTS} "
            "bounded checks; durable comment fence retained"
        )
        return None

    def _generate_sbatch(self, job_number: int) -> str:
        job_name = f"gpu-{self.chain_id[:8]}"
        exclusive_line = "#SBATCH --exclusive\n" if self.gpus >= 8 else ""
        metadata = {
            "SPATIAL_AGENT_SERVICE_POOL_ID": self.pool_id,
            "SPATIAL_AGENT_SERVICE_NAMESPACE": self.namespace,
            "SPATIAL_AGENT_SERVICE_SLOT_ID": self.chain_id,
            "SPATIAL_AGENT_SERVICE_GENERATION": str(job_number),
            "SPATIAL_AGENT_TOOL_SPEC_DIGEST": self.tool_spec_digest,
            "SPATIAL_AGENT_TOPOLOGY_DIGEST": self.topology_digest,
        }
        if self.pool_id:
            metadata["SPATIAL_AGENT_ENABLE_SYNTHETIC_KEEPALIVE"] = "0"
        exports = "\n".join(
            f"export {key}={shlex.quote(str(value))}"
            for key, value in metadata.items()
        )
        # Preserve an exact, result-neutral recovery key for both managed and
        # legacy submissions. The empty legacy pool token cannot parse as a
        # managed lifecycle tag.
        comment = self._submission_comment(job_number)
        comment_line = f"#SBATCH --comment={comment}\n"
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={self.gpus}
#SBATCH --mem-per-gpu=228G
#SBATCH --time={self.time_limit}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
{comment_line}{exclusive_line}
echo "=========================================="
echo "GPU Server - Chain {self.chain_id[:8]} - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

cd {self.project_root}

{exports}

bash {self.gpu_server_script} {self.gpus} {self.reconstruct_backend}

echo "End time: $(date)"
"""

    def _wait_for_job_to_start(self, job_id: str) -> bool:
        self._log(f"Waiting for job {job_id} to start...")
        # Prime the cache, but do not treat the helper's conservative result as
        # affirmative visibility.  Only exact authoritative membership below
        # may clear the crash fence.
        wait_for_job_visible(job_id)
        visible = False
        start = time.time()
        last_status = None

        while self.running:
            if self._shutdown_requested:
                return False
            exact = get_exact_job_states([job_id], force=True)
            if exact is None:
                status = "UNKNOWN"
            elif job_id not in exact:
                status = "NOT_FOUND"
            else:
                value = exact[job_id]
                status = (
                    value if isinstance(value, str)
                    else str(value.get("status") or "UNKNOWN")
                ).upper()
                if not visible:
                    visible = True
                    expected_comment = self._submission_comment(self.job_counter)
                    actual_comment = (
                        str(value.get("comment") or "")
                        if isinstance(value, dict) else ""
                    )
                    if actual_comment != expected_comment:
                        self._replacement_blocked = True
                        reason = (
                            "accepted GPU-tool allocation failed exact scheduler-"
                            "comment attestation: "
                            f"job={job_id}, expected={expected_comment!r}, "
                            f"actual={actual_comment!r}"
                        )
                        self._log(reason)
                        self._retain_quarantined_ownership(reason)
                        return False
                    if self._submission_intent is not None:
                        if not self._persist_submission_state(None):
                            self._replacement_blocked = True
                            return False
            now = time.time()

            if status != last_status:
                elapsed_m = int((now - start) / 60)
                self._log(f"Job {job_id} status: {status} (waited {elapsed_m}m)")
                last_status = status

            if status == "RUNNING":
                return True
            elif status in (
                "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED",
            ):
                self._log(f"Job {job_id} failed: {status}")
                return False
            elif status == "NOT_FOUND" and visible:
                # Previously observed, now authoritatively absent from live
                # scheduler state: terminal for this exact job ID.
                return False

            if self.pool_id and now - start >= _JOB_START_TIMEOUT_SECONDS:
                self._log(
                    f"Job {job_id} did not reach RUNNING within "
                    f"{self._format_time(_JOB_START_TIMEOUT_SECONDS)}; "
                    "ending the bounded start wait."
                )
                return False

            if self._stop_event.wait(timeout=30):
                return False  # Signalled to stop

        return False

    def submit_job(self) -> bool:
        self._log(f"Submitting job #{self.job_counter}...")

        if not self._submission_authority_available():
            self._replacement_blocked = True
            return False

        job_name = f"gpu-{self.chain_id[:8]}"
        script_content = self._generate_sbatch(self.job_counter)
        script_path = write_unique_text_artifact(
            self.log_dir,
            identity=(
                f"{job_name}-{self.chain_id}-generation-{self.job_counter}"
            ),
            suffix=".sbatch",
            content=script_content,
        )

        prepared_intent = self._submission_payload(
            self.job_counter, phase="prepared",
        )
        if not self._persist_submission_state(prepared_intent):
            self._replacement_blocked = True
            return False


        if not self._fresh_owner_epoch_available():
            if not self._persist_submission_state(None):
                self._log("Could not clear the unsubmitted prepared fence.")
            self._replacement_blocked = True
            return False

        try:
            result = subprocess.run(
                ["sbatch", "--parsable", str(script_path)],
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            # An exec failure proves sbatch never ran.  Unknown subprocess
            # outcomes retain the prepared fence by propagating instead.
            if not self._persist_submission_state(None):
                self._replacement_blocked = True
            self._log(f"Failed to execute sbatch: {exc}")
            return False

        if result.returncode != 0:
            if not self._persist_submission_state(None):
                self._replacement_blocked = True
            self._log(f"Failed to submit job: {result.stderr}")
            return False

        job_id = self._parse_parsable_job_id(result.stdout) or ""
        if job_id:
            self.current_job_id = job_id
            if job_id not in self.submitted_job_ids:
                self.submitted_job_ids.append(job_id)
        accepted_intent = self._submission_payload(
            self.job_counter,
            phase=("accepted_unseen" if job_id else "accepted_unresolved"),
            job_ids=([job_id] if job_id else ()),
        )
        if not self._persist_submission_state(accepted_intent):
            if job_id:
                self._cancel_after_accepted_fence_failure(job_id)
            else:
                # There is no exact cancellation target.  Retain the unique
                # comment fence and prohibit a replacement submission.
                self._replacement_blocked = True
            return False

        if not job_id or not job_id.isdigit():
            self._log(
                "Ambiguous successful sbatch response; reconciling the exact "
                "exact job comment."
            )
            job_id = self._recover_ambiguous_submission(self.job_counter) or ""
            if not job_id:
                self._replacement_blocked = True
                self._log(
                    "Could not prove exactly one submitted allocation; "
                    "quarantining this chain without retry."
                )
                return False
            self.current_job_id = job_id
            if job_id not in self.submitted_job_ids:
                self.submitted_job_ids.append(job_id)
            # Exact comment recovery proves scheduler visibility.
            if not self._persist_submission_state(None):
                self._replacement_blocked = True
                return False

        self.current_job_id = job_id

        self._log(f"Job #{self.job_counter} submitted: SLURM ID {job_id}")
        self._log(f"Log: {self.log_dir}/{job_name}_{job_id}.out")

        if self._wait_for_job_to_start(job_id):
            self.job_submit_time = time.time()
            return True
        else:
            if self._shutdown_requested:
                return False
            self._log(f"Cancelling failed-to-start exact job {job_id}...")
            live = self._exact_live_jobs([job_id])
            if live is None:
                self._replacement_blocked = True
                return False
            if job_id in live:
                if not cancel_job(job_id) or not self._wait_for_exact_jobs_terminal(
                    [job_id]
                ):
                    self._replacement_blocked = True
            return False

    def _cancel_after_accepted_fence_failure(self, job_id: str) -> bool:
        """Reclaim an exact accepted job whose durable fence write failed.

        The parsed Slurm ID is already present in memory and lifecycle state.
        Cancellation is therefore exact; scheduler uncertainty still fails
        closed and retains the ownership fence.
        """
        self._log(
            "Accepted-job authority could not be persisted everywhere; "
            f"cancelling exact Slurm job {job_id} before any retry."
        )
        if not self._wait_for_accepted_job_visibility(job_id):
            self._replacement_blocked = True
            return False
        live = self._exact_live_jobs([job_id])
        if live is None:
            self._replacement_blocked = True
            return False
        if job_id in live and (
            not cancel_job(job_id)
            or not self._wait_for_exact_jobs_terminal([job_id])
        ):
            self._replacement_blocked = True
            return False

        previous_ids = list(self.submitted_job_ids)
        previous_current = self.current_job_id
        self.submitted_job_ids = [
            recorded for recorded in self.submitted_job_ids
            if recorded != job_id
        ]
        if self.current_job_id == job_id:
            self.current_job_id = None
        if not self._persist_submission_state(None):
            self.submitted_job_ids = previous_ids
            self.current_job_id = previous_current
            self._replacement_blocked = True
            return False
        return True

    def _wait_for_accepted_job_visibility(self, job_id: str) -> bool:
        """Never infer terminal state before an accepted ID was observed."""
        for attempt in range(_ACCEPTED_ID_VISIBILITY_ATTEMPTS):
            live = self._exact_live_jobs([job_id])
            if live is not None and job_id in live:
                return True
            if attempt + 1 < _ACCEPTED_ID_VISIBILITY_ATTEMPTS:
                time.sleep(_ACCEPTED_ID_VISIBILITY_INTERVAL_SECONDS)
        self._log(
            f"Accepted Slurm job {job_id} never became authoritatively "
            "visible; retaining its fence and blocking replacement."
        )
        return False

    def _resolve_accepted_fence_before_stop(self) -> bool:
        pending = self._submission_pending()
        if pending is False:
            return True
        if pending is None:
            return False
        job_ids = self._recorded_job_ids()
        if not job_ids:
            return False
        if not all(
            self._wait_for_accepted_job_visibility(job_id)
            for job_id in job_ids
        ):
            return False
        return self._persist_submission_state(None)

    def wait_with_progress(self) -> bool:
        wait_h = self.wait_seconds // 3600
        wait_m = (self.wait_seconds % 3600) // 60
        self._log(f"Waiting {wait_h}h {wait_m}m before overlap submission...")

        start = time.time()
        last_check = 0

        while self.running:
            if self._shutdown_requested:
                return False
            # Sleep in short bursts so SIGTERM is handled promptly
            if self._stop_event.wait(timeout=10):
                return False  # Signalled to stop

            elapsed = int(time.time() - start)
            if elapsed >= self.wait_seconds:
                break

            # Periodic checks every ~60s
            if elapsed - last_check >= 60:
                last_check = elapsed

                if self.pool_id:
                    if not self._reconcile_runtime_authority():
                        continue
                    owner_present = bool(
                        self.lifecycle_store.active_leases_for_pool(self.pool_id)
                    )
                    self._ever_had_owner = self._ever_had_owner or owner_present
                    grace_expired = (
                        time.monotonic() - self._chain_started_monotonic
                        >= self.bootstrap_grace_seconds
                    )
                    if (
                        self.auto_stop_when_unowned
                        and not owner_present
                        and (self._ever_had_owner or grace_expired)
                    ):
                        self._log(
                            "No compatible owner lease remains; attempting "
                            "verified graceful teardown."
                        )
                        stop_outcome = self._graceful_stop_all_jobs()
                        if stop_outcome == "stopped":
                            self.running = False
                            return False
                        if stop_outcome in {"owner_present", "busy"}:
                            self._ever_had_owner = (
                                self._ever_had_owner
                                or stop_outcome == "owner_present"
                            )
                            continue
                        self._log(
                            "Drain/cancellation could not be verified; retaining "
                            "every recorded allocation until its wall limit "
                            "(no successor)."
                        )
                        self.auto_stop_when_unowned = False

                if self.current_job_id and not is_job_alive(self.current_job_id):
                    elapsed_m = int((time.time() - self.job_submit_time) / 60) if self.job_submit_time else 0
                    self._log(f"Job {self.current_job_id} died after {elapsed_m}m!")
                    return False

                if elapsed % 600 < 60 and elapsed > 0:
                    remaining = self.wait_seconds - elapsed
                    self._log(f"Next submission in {self._format_time(remaining)}")
                    self._cleanup_gpu_registry()
                    self._update_state_jobs()

        return True

    def _set_chain_tool_records(self, *, draining: bool) -> Optional[List[dict]]:
        """Transition every tool generation owned by this exact chain.

        ``None`` is an unknown registry state and aborts teardown.  Returned
        records carry their prior local state for safe restoration when no
        remote drain was accepted.
        """
        reg_file = self.project_root / "spatial_agent" / "logs" / "gpu_server.json"
        lock_file = self.project_root / "spatial_agent" / "logs" / "gpu_server.json.lock"
        records: List[dict] = []
        lock_f = open(lock_file, "a+")
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                with open(reg_file, "r", encoding="utf-8") as handle:
                    registry = json.load(handle)
            except FileNotFoundError:
                return []
            except (json.JSONDecodeError, OSError) as exc:
                self._log(f"GPU registry transition failed: {exc}")
                return None
            if not isinstance(registry, dict):
                return None
            for uid, info in registry.items():
                if not isinstance(info, dict):
                    return None
                if info.get("pool_id") != self.pool_id:
                    continue
                if info.get("slot_id") != self.chain_id:
                    continue
                previous = {
                    key: (key in info, info.get(key))
                    for key in (
                        "state", "accepting", "draining", "state_change_time",
                    )
                }
                info.update({
                    "state": "draining" if draining else "ready",
                    "accepting": not draining,
                    "draining": draining,
                    "state_change_time": self._ts(),
                })
                records.append({
                    "uid": uid,
                    "job_id": str(info.get("slurm_job_id") or ""),
                    "previous": previous,
                    **info,
                })
            if records:
                write_json_atomic(reg_file, registry)
        except Exception as exc:
            self._log(f"GPU registry transition failed: {exc}")
            return None
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            lock_f.close()
        return records

    def _restore_chain_tool_records(self, records: Iterable[dict]) -> bool:
        """Restore records only when their remote server did not enter drain."""
        selected = list(records)
        if not selected:
            return True
        reg_file = self.project_root / "spatial_agent" / "logs" / "gpu_server.json"
        lock_file = self.project_root / "spatial_agent" / "logs" / "gpu_server.json.lock"
        lock_f = open(lock_file, "a+")
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            with open(reg_file, "r", encoding="utf-8") as handle:
                registry = json.load(handle)
            if not isinstance(registry, dict):
                return False
            for record in selected:
                info = registry.get(record["uid"])
                if not isinstance(info, dict):
                    continue
                if info.get("pool_id") != self.pool_id:
                    continue
                if info.get("slot_id") != self.chain_id:
                    continue
                if str(info.get("slurm_job_id") or "") != record["job_id"]:
                    continue
                for key, (existed, value) in record["previous"].items():
                    if existed:
                        info[key] = value
                    else:
                        info.pop(key, None)
            write_json_atomic(reg_file, registry)
            return True
        except Exception as exc:
            self._log(f"GPU registry restore failed: {exc}")
            return False
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            lock_f.close()

    @staticmethod
    def _tool_url(info: dict) -> Optional[str]:
        ip, port = info.get("ip"), info.get("http_port")
        if not ip or not port:
            return None
        host = (
            f"[{ip}]" if ":" in str(ip) and not str(ip).startswith("[")
            else str(ip)
        )
        return f"http://{host}:{port}"

    @classmethod
    def _begin_remote_drain(cls, info: dict) -> bool:
        endpoint = cls._tool_url(info)
        token = info.get("auth_token")
        if not endpoint or not token:
            return False
        try:
            request = urllib.request.Request(
                f"{endpoint}/drain",
                data=b"",
                headers={"Authorization": f"Bearer {token}"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                return 200 <= int(response.status) < 300
        except Exception:
            return False

    @classmethod
    def _remote_active_calls(cls, info: dict) -> Optional[int]:
        endpoint = cls._tool_url(info)
        if not endpoint:
            return None
        try:
            request = urllib.request.Request(f"{endpoint}/health", method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not payload.get("draining") or payload.get("accepting") is not False:
                return None
            return int(payload["active_requests"])
        except Exception:
            return None

    def _recorded_job_ids(self, extra: Iterable[str] = ()) -> Optional[List[str]]:
        job_ids = [
            str(job_id) for job_id in [*self.submitted_job_ids, self.current_job_id]
            if job_id
        ]
        job_ids.extend(str(job_id) for job_id in extra if job_id)
        try:
            for server in self.state_manager.list_servers():
                if server.chain_id == self.chain_id:
                    job_ids.extend(
                        str(job_id) for job_id in server.slurm_job_ids if job_id
                    )
            if self.pool_id:
                pool = (
                    self.lifecycle_store.snapshot().get("pools", {})
                    .get(self.pool_id)
                )
                if pool is None:
                    return None
                persisted = (pool.get("chains", {}).get(self.chain_id) or {})
                job_ids.extend(
                    str(job_id) for job_id in persisted.get("job_ids", [])
                    if job_id
                )
        except Exception as exc:
            self._log(f"Could not read complete chain ownership: {exc}")
            return None
        return list(dict.fromkeys(job_ids))

    def _owners_present(self) -> bool:
        if not self.pool_id:
            return False
        try:
            return bool(self.lifecycle_store.active_leases_for_pool(self.pool_id))
        except Exception as exc:
            self._log(f"Owner check failed; assuming an owner is present: {exc}")
            return True

    def _await_initial_owner(self) -> bool:
        """Hold the manager, not GPUs, until a fresh demand lease exists."""
        if not self.pool_id or not self.auto_stop_when_unowned:
            return True
        deadline = self._chain_started_monotonic + self.bootstrap_grace_seconds
        self._log(
            "Waiting for a fresh lifecycle owner before the first GPU "
            "allocation (demand-first launch)."
        )
        while self.running and not self._shutdown_requested:
            try:
                if self.lifecycle_store.should_renew_pool(self.pool_id):
                    self._ever_had_owner = True
                    return True
            except Exception as exc:
                self._log(
                    "Initial owner authority is unreadable; keeping GPU "
                    f"submission blocked: {exc}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._log(
                    "No fresh owner lease appeared during bootstrap grace; "
                    "exiting without submitting a GPU allocation."
                )
                return False
            if self._stop_event.wait(timeout=min(10.0, remaining)):
                return False
        return False

    def _cancellation_still_authorized(self, stop_token: Optional[str]) -> bool:
        """Final integration point for the atomic lifecycle stop fence."""
        if not self.pool_id:
            return stop_token is None
        if not stop_token or self._owners_present():
            return False
        try:
            return bool(
                self.lifecycle_store.validate_pool_stop(
                    self.pool_id, stop_token,
                )
            )
        except Exception as exc:
            self._log(f"Stop-token validation failed: {exc}")
            return False

    def _finish_stop_intent(self, stop_token: str, outcome: str) -> bool:
        try:
            return bool(
                self.lifecycle_store.end_pool_stop(
                    self.pool_id, stop_token, outcome=outcome,
                )
            )
        except Exception as exc:
            self._log(f"Could not close stop intent as {outcome}: {exc}")
            return False

    @staticmethod
    def _exact_live_jobs(job_ids: Iterable[str]) -> Optional[Dict[str, dict]]:
        return get_exact_job_states(job_ids, force=True)

    def _wait_for_exact_jobs_terminal(self, job_ids: Iterable[str]) -> bool:
        exact = list(dict.fromkeys(str(job_id) for job_id in job_ids if job_id))
        for attempt in range(_CANCEL_VERIFY_ATTEMPTS):
            live = self._exact_live_jobs(exact)
            if live is None:
                return False
            if not live:
                return True
            if attempt + 1 < _CANCEL_VERIFY_ATTEMPTS:
                time.sleep(_CANCEL_VERIFY_INTERVAL_SECONDS)
        return False

    def _cancel_exact_live_jobs(
        self, job_ids: Iterable[str], *, stop_token: Optional[str],
    ) -> Tuple[bool, List[str]]:
        exact = list(dict.fromkeys(str(job_id) for job_id in job_ids if job_id))
        live = self._exact_live_jobs(exact)
        if live is None:
            return False, []
        accepted: List[str] = []
        for job_id in exact:
            if job_id not in live:
                continue
            if not self._cancellation_still_authorized(stop_token):
                return False, accepted
            if not cancel_job(job_id):
                self._log(f"Slurm rejected cancellation of exact job {job_id}")
                return False, accepted
            accepted.append(job_id)
        if not self._wait_for_exact_jobs_terminal(exact):
            self._log(
                "Scheduler did not authoritatively confirm every exact job "
                "terminal; retaining ownership metadata."
            )
            return False, accepted
        return True, accepted

    def _wait_for_remote_idle(self, records: Iterable[dict]) -> bool:
        records = list(records)
        if not records:
            return True
        zero_observations = 0
        deadline = time.monotonic() + _REMOTE_DRAIN_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            active = [self._remote_active_calls(record) for record in records]
            if all(value == 0 for value in active):
                zero_observations += 1
                if zero_observations >= 3:
                    return True
            else:
                zero_observations = 0
            time.sleep(5)
        return False

    def _graceful_stop_all_jobs(self) -> str:
        """Drain and stop every generation, or retain all ownership metadata."""
        if not self._reconcile_runtime_authority(force=True):
            return "failed"
        if not self._resolve_accepted_fence_before_stop():
            return "failed"
        try:
            stop_token = self.lifecycle_store.begin_pool_stop(
                self.pool_id, chain_id=self.chain_id,
            )
        except Exception as exc:
            self._log(f"Could not establish an atomic stop intent: {exc}")
            return "failed"
        if not stop_token:
            return "owner_present" if self._owners_present() else "busy"

        records = self._set_chain_tool_records(draining=True)
        if records is None:
            self._finish_stop_intent(stop_token, "quarantined")
            return "failed"
        job_ids = self._recorded_job_ids(
            record.get("job_id") for record in records
        )
        if job_ids is None:
            self._restore_chain_tool_records(records)
            self._finish_stop_intent(stop_token, "quarantined")
            return "failed"
        live = self._exact_live_jobs(job_ids)
        if live is None:
            self._restore_chain_tool_records(records)
            self._finish_stop_intent(stop_token, "aborted")
            return "failed"
        # Missing registry records are uncertainty, not zero active calls.
        # Require exactly one authenticated, addressable tool record for every
        # live allocation before beginning remote drain or cancellation.
        live_record_matches = {
            job_id: [
                record for record in records
                if record.get("job_id") == job_id
            ]
            for job_id in live
        }
        if any(
            len(matches) != 1
            or not self._tool_url(matches[0])
            or not matches[0].get("auth_token")
            for matches in live_record_matches.values()
        ):
            self._log(
                "Cannot drain safely: every live GPU-tool allocation must map "
                "to exactly one owned authenticated registry endpoint; "
                "retaining jobs."
            )
            self._restore_chain_tool_records(records)
            self._finish_stop_intent(stop_token, "quarantined")
            return "failed"
        live_records = [
            record for record in records if record.get("job_id") in live
        ]
        if not self._cancellation_still_authorized(stop_token):
            self._restore_chain_tool_records(records)
            self._finish_stop_intent(stop_token, "aborted")
            return "owner_present" if self._owners_present() else "failed"

        remotely_drained: List[dict] = []
        failed_remote: List[dict] = []
        for record in live_records:
            if self._begin_remote_drain(record):
                remotely_drained.append(record)
            else:
                failed_remote.append(record)
        if failed_remote:
            # It is safe to republish only servers that did not acknowledge
            # remote drain.  A remotely draining process has no resume API.
            self._restore_chain_tool_records(failed_remote)
            outcome = "quarantined" if remotely_drained else "aborted"
            self._finish_stop_intent(stop_token, outcome)
            return "failed"
        if not self._wait_for_remote_idle(remotely_drained):
            self._finish_stop_intent(stop_token, "quarantined")
            return "failed"
        if not self._cancellation_still_authorized(stop_token):
            self._finish_stop_intent(stop_token, "quarantined")
            return "owner_present" if self._owners_present() else "failed"

        success, _accepted = self._cancel_exact_live_jobs(
            job_ids, stop_token=stop_token,
        )
        if not success:
            self._finish_stop_intent(stop_token, "quarantined")
            return "failed"
        if not self._finish_stop_intent(stop_token, "stopped"):
            return "failed"
        return "stopped"

    def _stop_legacy_jobs(self) -> bool:
        if not self._resolve_accepted_fence_before_stop():
            return False
        job_ids = self._recorded_job_ids()
        if job_ids is None:
            return False
        success, _accepted = self._cancel_exact_live_jobs(
            job_ids, stop_token=None,
        )
        return success

    def _wait_for_recorded_jobs_end(self) -> None:
        while not self._shutdown_requested:
            job_ids = self._recorded_job_ids()
            if job_ids is None:
                return
            live = self._exact_live_jobs(job_ids)
            if live is None:
                time.sleep(30)
                continue
            if not live:
                return
            time.sleep(30)
            self._update_state_jobs()

    def handle_shutdown(self, signum, frame):
        """Signal-safe handler: record intent; normal control flow tears down."""
        self._shutdown_requested = True
        self._shutdown_signal = int(signum) if signum is not None else None

    def _update_state_jobs(self):
        recorded = self._recorded_job_ids()
        if recorded is None:
            self._log(
                "Warning: ownership state is unreadable; refusing to overwrite "
                "its recorded job IDs"
            )
            return
        self.submitted_job_ids = recorded
        try:
            self.state_manager.update_server_jobs(
                self.chain_id, recorded,
            )
        except Exception as e:
            self._log(f"Warning: failed to update state: {e}")
        if self.pool_id:
            try:
                self.lifecycle_store.update_chain_jobs(
                    self.pool_id, self.chain_id, recorded,
                )
            except Exception as e:
                self._log(f"Warning: failed to update lifecycle jobs: {e}")

    def _cleanup_gpu_registry(self):
        """Remove dead entries from gpu_server.json."""
        reg_file = self.project_root / "spatial_agent" / "logs" / "gpu_server.json"
        lock_file = self.project_root / "spatial_agent" / "logs" / "gpu_server.json.lock"
        if not reg_file.exists():
            return

        try:
            # Batch-query all SLURM job IDs BEFORE taking the lock
            # (squeue can be slow; don't hold the lock during I/O).
            with open(reg_file, "r") as f:
                preview = json.load(f)

            slurm_ids = [
                info.get("slurm_job_id")
                for info in preview.values()
                if info.get("slurm_job_id")
            ]
            if not slurm_ids:
                return

            from spatial_agent.launch_managers.slurm_utils import batch_query_jobs
            job_status = batch_query_jobs(slurm_ids)

            # Failure snapshots contain truthy UNKNOWN placeholders. Require
            # provenance-carrying scheduler authority before deleting any
            # discovery record.
            if slurm_ids and not bool(
                getattr(job_status, "authoritative", False)
            ):
                return

            alive_statuses = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}

            # Take the file lock and re-read (another process may have
            # modified the file between our preview read and now).
            lock_f = open(lock_file, "a+")
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)

                try:
                    with open(reg_file, "r") as f:
                        registry = json.load(f)
                except (json.JSONDecodeError, OSError):
                    return

                removed = 0
                for uid in list(registry.keys()):
                    info = registry[uid]
                    slurm_id = info.get("slurm_job_id")
                    should_remove = False

                    if slurm_id:
                        if slurm_id not in set(slurm_ids):
                            # Entry was added between preview and lock — we
                            # didn't query this job, so leave it alone.
                            continue
                        ji = job_status.get(slurm_id)
                        if ji:
                            if ji["status"].upper() not in alive_statuses:
                                should_remove = True
                        else:
                            # Job not in squeue at all — likely finished.
                            # But only if we got SOME results back (guarded above).
                            should_remove = True
                    else:
                        pid = info.get("pid")
                        if pid:
                            try:
                                os.kill(pid, 0)
                            except (ProcessLookupError, PermissionError, OSError):
                                should_remove = True

                    if should_remove:
                        del registry[uid]
                        removed += 1

                if removed > 0:
                    write_json_atomic(reg_file, registry)
                    self._log(f"Cleaned gpu_server.json: removed {removed} dead entries")
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                lock_f.close()

        except Exception as e:
            self._log(f"Warning: registry cleanup failed: {e}")

    def _all_recorded_jobs_terminal(self) -> bool:
        # A successful submission without exact terminal resolution owns an
        # allocation-shaped uncertainty even when no job ID was parsed.
        if self._submission_pending() is not False:
            return False
        job_ids = self._recorded_job_ids()
        if job_ids is None:
            return False
        live = self._exact_live_jobs(job_ids)
        return live is not None and not live

    def _finalization_state(self) -> str:
        """Classify the exact final scheduler state without losing authority."""
        if self._submission_pending() is not False:
            return "unknown"
        job_ids = self._recorded_job_ids()
        if job_ids is None:
            return "unknown"
        live = self._exact_live_jobs(job_ids)
        if live is None:
            return "unknown"
        if not live:
            return "terminal"
        statuses = {
            str(info.get("status") or "").upper()
            for info in live.values()
            if isinstance(info, dict)
        }
        if statuses and statuses <= {"COMPLETING"}:
            return "completing"
        return "live"

    def _retain_quarantined_ownership(self, reason: str) -> None:
        self._shutdown_drain_failed = True
        self._update_state_jobs()
        if not self.pool_id:
            return
        try:
            self.lifecycle_store.set_pool_status(
                self.pool_id, "quarantined", reason=reason,
                chain_id=self.chain_id,
                job_ids=self._recorded_job_ids() or self.submitted_job_ids,
            )
        except Exception as exc:
            self._log(f"Could not persist pool quarantine: {exc}")

    def _finalize_ownership(self) -> None:
        terminal = False
        if not self._shutdown_drain_failed:
            for attempt in range(_FINALIZE_VERIFY_ATTEMPTS):
                state = self._finalization_state()
                if state == "terminal":
                    terminal = True
                    break
                if state != "completing":
                    break
                if attempt + 1 < _FINALIZE_VERIFY_ATTEMPTS:
                    self._stop_event.wait(
                        timeout=_FINALIZE_VERIFY_INTERVAL_SECONDS
                    )
        if self._shutdown_drain_failed or not terminal:
            self._retain_quarantined_ownership(
                "manager exited without authoritative terminal proof for every "
                "recorded Slurm job after bounded finalization retries"
            )
            return
        try:
            self.state_manager.remove_server(self.chain_id)
        except Exception as exc:
            self._log(f"Could not remove terminal chain ownership: {exc}")
            return
        if self.pool_id:
            try:
                self.lifecycle_store.remove_chain_record(
                    self.pool_id, self.chain_id,
                )
            except Exception as exc:
                self._log(
                    f"Could not release terminal lifecycle slot ownership: {exc}"
                )

    def run(self):
        if not check_slurm_available():
            self._log("ERROR: SLURM not available")
            sys.exit(1)

        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        self._log(f"Starting GPU server chain ({self.gpus} GPU(s), backend={self.reconstruct_backend})")
        self._log(f"Chain ID: {self.chain_id}")
        self._log(f"Job duration: {self.time_limit}, overlap: {self.restart_before_minutes}m")

        self._cleanup_gpu_registry()

        consecutive_fast_failures = 0

        if not self._await_initial_owner():
            self.running = False

        while self.running:
            if self._shutdown_requested:
                if self.pool_id:
                    stop_outcome = self._graceful_stop_all_jobs()
                    if stop_outcome == "owner_present":
                        self._log(
                            "Refusing shutdown: a compatible owner appeared "
                            "before cancellation."
                        )
                        self._shutdown_requested = False
                        self._shutdown_signal = None
                        continue
                    if stop_outcome == "busy":
                        self._log(
                            "Another chain owns the pool stop fence; waiting "
                            "before retrying shutdown."
                        )
                        time.sleep(5)
                        continue
                    self._shutdown_drain_failed = stop_outcome != "stopped"
                else:
                    self._shutdown_drain_failed = not self._stop_legacy_jobs()
                self.running = False
                break

            if (
                self.pool_id
                and self.submitted_job_ids
                and not self._await_runtime_authority()
            ):
                self._log(
                    "Runtime authority stayed unavailable after bounded "
                    "checks; suppressing all successor submissions and "
                    "retaining the current allocation only to its wall limit."
                )
                self._wait_for_recorded_jobs_end()
                break

            self._cleanup_gpu_registry()

            submit_t = time.time()
            if not self.submit_job():
                if self._shutdown_requested:
                    continue
                if self._replacement_blocked:
                    self._retain_quarantined_ownership(
                        "submission/replacement safety could not be established; "
                        "no automatic retry was made"
                    )
                    break
                consecutive_fast_failures += 1
                self._log(
                    f"Failed to submit/start "
                    f"({consecutive_fast_failures}/{_MAX_FAST_FAILURES} consecutive)."
                )
                if consecutive_fast_failures >= _MAX_FAST_FAILURES:
                    self._log(
                        f"Aborting chain: {_MAX_FAST_FAILURES} consecutive jobs "
                        f"failed to start. Check the latest job log."
                    )
                    break
                if self._stop_event.wait(timeout=60):
                    break
                continue

            self.job_counter += 1

            if self.running:
                if self.wait_with_progress():
                    consecutive_fast_failures = 0
                    if self.pool_id and not self._await_runtime_authority():
                        self._log(
                            "Runtime authority stayed unavailable at the "
                            "renewal boundary; suppressing the successor and "
                            "retaining the current allocation only to its wall "
                            "limit."
                        )
                        self._wait_for_recorded_jobs_end()
                        break
                    if self.pool_id and not self.lifecycle_store.should_renew_pool(
                        self.pool_id
                    ):
                        self._log(
                            "No active compatible owner at renewal boundary; "
                            "suppressing successor submission."
                        )
                        self._wait_for_recorded_jobs_end()
                        break
                    self._log(f"Overlap window: submitting next job ({self.restart_before_minutes}m overlap)")
                else:
                    if self._shutdown_requested:
                        continue
                    elapsed = time.time() - submit_t
                    if elapsed < _FAST_FAIL_SECONDS:
                        consecutive_fast_failures += 1
                        self._log(
                            f"Job died after {int(elapsed)}s "
                            f"({consecutive_fast_failures}/{_MAX_FAST_FAILURES} consecutive fast failures)."
                        )
                        if consecutive_fast_failures >= _MAX_FAST_FAILURES:
                            self._log(
                                f"Aborting chain: {_MAX_FAST_FAILURES} consecutive jobs "
                                f"died within {_FAST_FAIL_SECONDS}s. Check the latest job log."
                            )
                            break
                    else:
                        consecutive_fast_failures = 0
                    self._log("Job failed early, submitting replacement...")

        self._finalize_ownership()

        self._log("Chain manager stopped.")


def recover_managed_chain(
    server_state: GPUServerState, project_root: Path,
) -> str:
    """Safely drain a managed GPU-tool allocation after manager death."""
    state_manager = GPUServerStateManager(project_root)
    if not server_state.pool_id:
        return "not_managed"
    if state_manager.is_server_alive(server_state):
        return "manager_alive"

    chain = ServerChain(
        chain_id=server_state.chain_id,
        account=server_state.account,
        partition=server_state.partition,
        gpus=max(1, int(server_state.gpus or 1)),
        reconstruct_backend=server_state.reconstruct_backend,
        time_limit="4:00:00",
        restart_before_minutes=0,
        project_root=str(project_root),
        pool_id=str(server_state.pool_id),
        namespace=str(server_state.namespace or ""),
        tool_spec_digest=str(server_state.spec_digest or ""),
        topology_digest=str(server_state.topology_digest or ""),
        desired_instances=max(1, int(server_state.desired_instances or 1)),
        auto_stop_when_unowned=True,
        bootstrap_grace_minutes=0,
    )
    chain.submitted_job_ids = list(dict.fromkeys(
        str(job_id) for job_id in server_state.slurm_job_ids if job_id
    ))
    chain.current_job_id = (
        chain.submitted_job_ids[-1] if chain.submitted_job_ids else None
    )
    chain._submission_intent = server_state.submission_intent

    if server_state.submission_intent is not None:
        if not chain._reconcile_runtime_authority(force=True):
            chain._retain_quarantined_ownership(
                "operator recovery could not establish scheduler authority"
            )
            return "failed"
        pool = chain.lifecycle_store.snapshot().get("pools", {}).get(
            chain.pool_id, {}
        )
        persisted = (pool.get("chains") or {}).get(chain.chain_id, {})
        chain.submitted_job_ids = list(dict.fromkeys([
            *chain.submitted_job_ids,
            *(
                str(job_id) for job_id in persisted.get("job_ids", [])
                if job_id
            ),
        ]))
        if persisted.get("submission_intent") is not None:
            chain._retain_quarantined_ownership(
                "operator recovery retained an unresolved accepted-submission "
                "comment fence"
            )
            return "unresolved"
        chain._submission_intent = None
        if not chain._persist_submission_state(None):
            chain._retain_quarantined_ownership(
                "operator recovery could not synchronize adopted ownership"
            )
            return "failed"

    outcome = chain._graceful_stop_all_jobs()
    if outcome == "stopped":
        chain._finalize_ownership()
    elif outcome not in {"owner_present", "busy"}:
        chain._retain_quarantined_ownership(
            "operator recovery could not verify drain and exact cancellation"
        )
    return outcome


def start_chain_background(
    chain_id: str,
    account: str,
    partition: str,
    gpus: int = 1,
    reconstruct_backend: str = "pi3",
    time_limit: str = "4:00:00",
    restart_before_minutes: int = 20,
    project_root: Path = Path("."),
    pool_id: str = "",
    namespace: str = "",
    tool_spec_digest: str = "",
    topology_digest: str = "",
    desired_instances: int = 1,
    auto_stop_when_unowned: bool = False,
    bootstrap_grace_minutes: int = 30,
    launch_claim_id: str = "",
):
    """Start a GPU server chain after transactionally persisting ownership.

    Returns (GPUServerState, chain_log_path).
    """
    config = {
        "chain_id": chain_id,
        "account": account,
        "partition": partition,
        "gpus": gpus,
        "reconstruct_backend": reconstruct_backend,
        "time_limit": time_limit,
        "restart_before_minutes": restart_before_minutes,
        "project_root": str(project_root),
        "pool_id": pool_id,
        "namespace": namespace,
        "tool_spec_digest": tool_spec_digest,
        "topology_digest": topology_digest,
        "desired_instances": desired_instances,
        "auto_stop_when_unowned": auto_stop_when_unowned,
        "bootstrap_grace_minutes": bootstrap_grace_minutes,
    }

    lifecycle_store = LifecycleStore(project_root)
    if pool_id and pool_id not in lifecycle_store.snapshot().get("pools", {}):
        raise LifecycleError(
            f"Managed GPU-tool pool {pool_id!r} must be registered before launch"
        )

    log_dir = project_root / "spatial_agent" / "logs" / "slurm_gpu_server"
    log_dir.mkdir(parents=True, exist_ok=True)
    gate_read_fd, gate_write_fd = os.pipe()
    config["_startup_gate_fd"] = gate_read_fd
    chain_log, log_f = open_unique_text_artifact(
        log_dir,
        identity=f"chain-gpu-tools-{chain_id}",
        suffix=".log",
    )
    proc = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable, "-m",
                "spatial_agent.launch_managers.gpu_server_manager.server_chain",
                "--config", json.dumps(config),
            ],
            stdout=log_f,
            stderr=log_f,
            start_new_session=True,
            cwd=str(project_root),
            pass_fds=(gate_read_fd,),
        )
    except BaseException:
        os.close(gate_write_fd)
        log_f.close()
        raise
    finally:
        os.close(gate_read_fd)

    manager_identity = process_identity(proc.pid)
    if manager_identity is None:
        # The child remains behind the startup gate, so it cannot have
        # submitted an allocation.  Never persist an identity-less new owner.
        os.close(gate_write_fd)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_f.close()
        raise RuntimeError("Could not capture exact GPU manager process identity")

    started_at = datetime.datetime.now().isoformat()
    server_state = GPUServerState(
        chain_id=chain_id,
        pid=proc.pid,
        process_identity=manager_identity,
        slurm_job_ids=[],
        started_at=started_at,
        account=account,
        partition=partition,
        gpus=gpus,
        reconstruct_backend=reconstruct_backend,
        pool_id=pool_id or None,
        namespace=namespace or None,
        spec_digest=tool_spec_digest or None,
        topology_digest=topology_digest or None,
        desired_instances=desired_instances if pool_id else None,
        lifecycle_state="launching" if pool_id else None,
    )

    state_manager = GPUServerStateManager(project_root)
    lifecycle_recorded = False
    gate_release_attempted = False
    try:
        state_manager.add_server(server_state)
        if pool_id:
            lifecycle_store.record_chain(
                pool_id, chain_id, pid=proc.pid, job_ids=[],
                launch_claim_id=launch_claim_id or None,
            )
            lifecycle_recorded = True
        # A signal can arrive after the byte reaches the child but before the
        # parent returns.  From this point, rollback without scheduler proof
        # could erase ownership for an already accepted sbatch.
        gate_release_attempted = True
        if os.write(gate_write_fd, b"1") != 1:
            raise RuntimeError("Could not release persisted chain startup gate")
    except BaseException:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=10)
                except Exception:
                    pass
        if not gate_release_attempted:
            try:
                state_manager.remove_server(chain_id)
            except Exception:
                pass
            if lifecycle_recorded:
                try:
                    lifecycle_store.remove_chain_record(pool_id, chain_id)
                except Exception:
                    pass
        elif pool_id and lifecycle_recorded:
            try:
                lifecycle_store.set_pool_status(
                    pool_id,
                    "quarantined",
                    reason=(
                        "parent failed after attempting persisted startup-gate "
                        "release; exact submission outcome requires reconciliation"
                    ),
                    chain_id=chain_id,
                )
            except Exception:
                pass
        raise
    finally:
        os.close(gate_write_fd)
        log_f.close()

    return server_state, chain_log


def main():
    parser = argparse.ArgumentParser(description="GPU server chain subprocess")
    parser.add_argument("--config", type=str, required=True, help="JSON config string")
    args = parser.parse_args()

    config = json.loads(args.config)
    startup_gate_fd = config.pop("_startup_gate_fd", None)
    chain = ServerChain(**config)
    if startup_gate_fd is not None:
        try:
            gate_value = os.read(int(startup_gate_fd), 1)
        finally:
            os.close(int(startup_gate_fd))
        if gate_value != b"1":
            raise RuntimeError(
                "Parent exited before chain ownership was durably persisted"
            )
    chain.run()


if __name__ == "__main__":
    main()
