"""
Background server chain manager.

Each chain runs as an independent subprocess that manages a continuous loop
of 4-hour SLURM jobs with overlap for zero-downtime restarts.

Can be invoked directly:
    python -m spatial_agent.launch_managers.vllm_manager.server_chain --config '{"chain_id": ..., ...}'
"""

import argparse
import datetime
import json
import os
import re
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
    batch_query_jobs,
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

_ALIVE_STATUSES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}
from spatial_agent.launch_managers.vllm_manager.state import ChainState, ChainStateManager, FileLock


# Stop a chain after this many consecutive jobs that died within
# _FAST_FAIL_SECONDS of starting (e.g. model load crash).
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
_MANAGED_STARTUP_TIMEOUT_SECONDS = 12 * 60


class ServerChain:
    """Manages continuous chain of 4-hour SLURM jobs with overlap."""

    def __init__(
        self,
        chain_id: str,
        served_name: str,
        model_name: str,
        model_path: str,
        account: str,
        partition: str,
        max_model_len: int,
        max_num_seqs: int,
        tp_size: int,
        kv_cache_dtype: str = "auto",
        quantization: str = "none",
        gpus: int = 8,
        time_limit: str = "4:00:00",
        restart_before_minutes: int = 20,
        project_root: str = ".",
        pool_id: str = "",
        namespace: str = "",
        server_spec_digest: str = "",
        topology_digest: str = "",
        desired_instances: int = 1,
        auto_stop_when_unowned: bool = False,
        bootstrap_grace_minutes: int = 30,
        model_revision: Optional[str] = None,
    ):
        self.chain_id = chain_id
        self.served_name = served_name
        self.model_name = model_name
        self.model_path = model_path
        self.model_revision = model_revision or None
        self.account = account
        self.partition = resolve_partitions(partition, account)
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.tp_size = tp_size
        self.kv_cache_dtype = kv_cache_dtype
        self.quantization = quantization
        self.gpus = gpus
        self.time_limit = time_limit
        if isinstance(restart_before_minutes, bool) or not isinstance(
            restart_before_minutes, int
        ):
            raise ValueError("restart_before_minutes must be an integer")
        self.restart_before_minutes = restart_before_minutes
        self.project_root = Path(project_root)
        self.pool_id = pool_id
        self.namespace = namespace
        self.server_spec_digest = server_spec_digest
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
        self._stop_event = threading.Event()
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

        self.log_dir = self.project_root / "spatial_agent" / "logs" / "slurm_vllm"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.vllm_script = Path(__file__).parent / "run_vllm.sh"
        self.state_manager = ChainStateManager(self.project_root)
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
            # Each iteration must perform a fresh read rather than reusing the
            # short failure cache in ``_reconcile_runtime_authority``.
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
            # Preserve the historical unmanaged renewal path during staged
            # rollout. Managed capacity alone depends on lifecycle/scheduler
            # authority and fails closed; legacy managers retain their exact
            # durable submission fence instead of dying after a brief squeue
            # outage at the routine renewal boundary.
            return True
        for attempt in range(_AUTHORITY_VERIFY_ATTEMPTS):
            if not self.running or self._shutdown_requested:
                return False
            jobs = get_user_jobs_snapshot(force=True)
            scheduler_ok = bool(getattr(jobs, "authoritative", False))
            runtime_ok = False
            if scheduler_ok:
                # Never reuse a cached failure/success for a destructive
                # allocation transition: every attempt starts a fresh
                # ownership reconciliation.
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
            "job_ids": list(dict.fromkeys(str(job_id) for job_id in job_ids if job_id)),
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
            # Keep an in-process fence even if either durable writer fails.
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
            self.state_manager.update_chain_submission(
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
            for chain in self.state_manager.list_chains():
                if (
                    chain.chain_id == self.chain_id
                    and chain.submission_intent is not None
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
        job_name = f"vllm-{self.served_name}"
        exclusive_line = "#SBATCH --exclusive\n" if self.gpus >= 8 else ""
        metadata = {
            "SPATIAL_AGENT_SERVICE_POOL_ID": self.pool_id,
            "SPATIAL_AGENT_SERVICE_NAMESPACE": self.namespace,
            "SPATIAL_AGENT_SERVICE_SLOT_ID": self.chain_id,
            "SPATIAL_AGENT_SERVICE_GENERATION": str(job_number),
            "SPATIAL_AGENT_SERVER_SPEC_DIGEST": self.server_spec_digest,
            "SPATIAL_AGENT_TOPOLOGY_DIGEST": self.topology_digest,
        }
        if self.pool_id:
            metadata["SPATIAL_AGENT_ENABLE_SYNTHETIC_KEEPALIVE"] = "0"
            metadata["SPATIAL_AGENT_SERVICE_STARTUP_TIMEOUT_SECONDS"] = str(
                _MANAGED_STARTUP_TIMEOUT_SECONDS
            )
        exports = "\n".join(
            f"export {key}={shlex.quote(str(value))}"
            for key, value in metadata.items()
        )
        # Scheduler metadata is result-neutral and supplies the only recovery
        # key when sbatch is accepted before its numeric ID reaches the
        # manager. Legacy comments deliberately have an empty pool token, so
        # managed reconciliation cannot mistake them for pool ownership.
        comment = self._submission_comment(job_number)
        comment_line = f"#SBATCH --comment={comment}\n"
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
#SBATCH --gpus-per-node={self.gpus}
#SBATCH --mem-per-gpu=240G
#SBATCH --time={self.time_limit}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
{comment_line}{exclusive_line}
echo "=========================================="
echo "SLURM vLLM Server - Chain {self.chain_id[:8]} - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

cd {self.project_root}

{exports}

bash {self.vllm_script} "{self.model_path}" "{self.served_name}" {self.max_model_len} {self.max_num_seqs} {self.gpus} {self.tp_size} {self.kv_cache_dtype} {self.quantization} {shlex.quote(self.model_revision or "")}

echo "End time: $(date)"
"""

    def _wait_for_job_to_start(self, job_id: str) -> bool:
        self._log(f"Waiting for job {job_id} to start...")
        # Prime the cache, but do not treat this helper's conservative return
        # value as affirmative visibility.  Only exact authoritative
        # membership below may clear the crash fence.
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
                job_info = exact[job_id]
                status = str(job_info.get("status") or "UNKNOWN").upper()
                if not visible:
                    visible = True
                    expected_comment = self._submission_comment(self.job_counter)
                    actual_comment = str(job_info.get("comment") or "")
                    if actual_comment != expected_comment:
                        self._replacement_blocked = True
                        reason = (
                            "accepted vLLM allocation failed exact scheduler-"
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
            elif status in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED"):
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
                return False

        return False

    def submit_job(self) -> bool:
        self._log(f"Submitting job #{self.job_counter}...")

        if not self._submission_authority_available():
            self._replacement_blocked = True
            return False

        job_name = f"vllm-{self.served_name}"
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


        # The authority query above may be separated from sbatch by script and
        # fence persistence. Recheck the short-lived renewable owner epoch at
        # the last safe point; this also covers early-failure replacements.
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
            # Exec failure proves sbatch was never invoked. Other exceptions
            # retain the prepared fence because the subprocess outcome is not
            # known to be negative.
            if not self._persist_submission_state(None):
                self._replacement_blocked = True
            self._log(f"Failed to execute sbatch: {exc}")
            return False

        if result.returncode != 0:
            if not self._persist_submission_state(None):
                self._replacement_blocked = True
            self._log(f"Failed to submit job: {result.stderr}")
            return False

        # ``--parsable`` emits ``job_id`` or ``job_id;cluster``.  Never use a
        # loose first-number match: an unrecognized success response could
        # otherwise create an allocation the manager cannot own or cancel.
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
            # Exact comment recovery itself proves scheduler visibility.
            if not self._persist_submission_state(None):
                self._replacement_blocked = True
                return False

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
            if self._stop_event.wait(timeout=10):
                return False

            elapsed = int(time.time() - start)
            if elapsed >= self.wait_seconds:
                break

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
                    self._cleanup_serve_registry()
                    self._update_state_jobs()

        return True

    def _set_chain_registry_state(
        self, *, state: str, accepting: bool, draining: bool,
    ) -> Optional[List[dict]]:
        """Transition every registry generation owned by this exact chain.

        ``None`` means the registry could not be read or updated and teardown
        must fail closed.  Each returned record contains enough information to
        restore its previous discovery state if cancellation is aborted.
        """
        serve_file = self.project_root / "spatial_agent" / "logs" / "serve.json"
        lock_file = str(serve_file) + ".lock"
        records: List[dict] = []
        try:
            with FileLock(lock_file):
                if not serve_file.exists():
                    return []
                with open(serve_file, "r", encoding="utf-8") as handle:
                    registry = json.load(handle)
                if not isinstance(registry, dict):
                    return None
                for model_key, servers in registry.items():
                    if not isinstance(servers, dict):
                        return None
                    for server_id, info in servers.items():
                        if not isinstance(info, dict):
                            return None
                        if info.get("pool_id") != self.pool_id:
                            continue
                        if info.get("slot_id") != self.chain_id:
                            continue
                        previous = {
                            key: (key in info, info.get(key))
                            for key in (
                                "state", "accepting", "draining",
                                "state_change_time",
                            )
                        }
                        info.update({
                            "state": state,
                            "accepting": bool(accepting),
                            "draining": bool(draining),
                            "state_change_time": self._ts(),
                        })
                        ip, port = info.get("ip"), info.get("port")
                        endpoint = None
                        if ip and port:
                            host = (
                                f"[{ip}]"
                                if ":" in str(ip)
                                and not str(ip).startswith("[")
                                else str(ip)
                            )
                            endpoint = f"http://{host}:{port}"
                        records.append({
                            "model_key": model_key,
                            "server_id": server_id,
                            "job_id": str(info.get("slurm_job_id") or ""),
                            "endpoint": endpoint,
                            "previous": previous,
                        })
                write_json_atomic(serve_file, registry)
        except Exception as exc:
            self._log(f"Registry drain transition failed: {exc}")
            return None
        return records

    def _restore_chain_registry_records(
        self, records: Iterable[dict], *, job_ids: Optional[Iterable[str]] = None,
    ) -> bool:
        """Restore only unchanged records transitioned by this manager."""
        selected_jobs = None if job_ids is None else {
            str(job_id) for job_id in job_ids if job_id
        }
        selected = [
            record for record in records
            if selected_jobs is None or record.get("job_id") in selected_jobs
        ]
        if not selected:
            return True
        serve_file = self.project_root / "spatial_agent" / "logs" / "serve.json"
        lock_file = str(serve_file) + ".lock"
        try:
            with FileLock(lock_file):
                with open(serve_file, "r", encoding="utf-8") as handle:
                    registry = json.load(handle)
                for record in selected:
                    info = (
                        registry.get(record["model_key"], {})
                        .get(record["server_id"])
                    )
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
                write_json_atomic(serve_file, registry)
            return True
        except Exception as exc:
            self._log(f"Registry restore failed: {exc}")
            return False

    @staticmethod
    def _vllm_queue_counts(endpoint: str) -> Optional[tuple[float, float]]:
        """Read passive vLLM metrics; never sends an inference request."""
        try:
            request = urllib.request.Request(f"{endpoint}/metrics", method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                body = response.read().decode("utf-8", errors="replace")
        except Exception:
            return None

        def metric(name: str) -> Optional[float]:
            matches = re.findall(
                rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$",
                body,
                flags=re.MULTILINE,
            )
            if not matches:
                return None
            return sum(float(value) for value in matches)

        running = metric("vllm:num_requests_running")
        waiting = metric("vllm:num_requests_waiting")
        if running is None or waiting is None:
            return None
        return running, waiting

    def _recorded_job_ids(self, extra: Iterable[str] = ()) -> Optional[List[str]]:
        """Return every exact job ID persisted for this chain.

        Failure to read either owner store is an unknown ownership state, not
        proof that a generation is gone.
        """
        job_ids = [
            str(job_id) for job_id in [*self.submitted_job_ids, self.current_job_id]
            if job_id
        ]
        job_ids.extend(str(job_id) for job_id in extra if job_id)
        try:
            for chain in self.state_manager.list_chains():
                if chain.chain_id == self.chain_id:
                    job_ids.extend(
                        str(job_id) for job_id in chain.slurm_job_ids if job_id
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
        """Cancel exact live IDs one by one with a final owner fence."""
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

    def _wait_for_vllm_idle(self, records: Iterable[dict]) -> bool:
        endpoints = [record.get("endpoint") for record in records]
        if any(not endpoint for endpoint in endpoints):
            return False
        if not endpoints:
            return True
        zero_observations = 0
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            samples = [self._vllm_queue_counts(endpoint) for endpoint in endpoints]
            if all(
                sample is not None and sample == (0.0, 0.0)
                for sample in samples
            ):
                zero_observations += 1
                if zero_observations >= 3:
                    return True
            else:
                zero_observations = 0
            time.sleep(10)
        return False

    def _graceful_stop_all_jobs(self) -> str:
        """Drain and stop every generation, or retain all ownership metadata.

        Returns ``stopped``, ``owner_present``, ``busy``, or ``failed``.
        """
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

        records = self._set_chain_registry_state(
            state="draining", accepting=False, draining=True,
        )
        if records is None:
            self._finish_stop_intent(stop_token, "quarantined")
            return "failed"

        job_ids = self._recorded_job_ids(
            record.get("job_id") for record in records
        )
        if job_ids is None:
            self._restore_chain_registry_records(records)
            self._finish_stop_intent(stop_token, "quarantined")
            return "failed"

        live = self._exact_live_jobs(job_ids)
        if live is None:
            self._restore_chain_registry_records(records)
            self._finish_stop_intent(stop_token, "aborted")
            return "failed"
        # A registry transition is not itself proof that it covered every
        # live generation.  In particular, a missing/deleted registry used to
        # produce ``records=[]`` and the empty idle-probe set then authorized
        # cancellation of a still-live service.  Require a one-to-one mapping
        # from every live recorded allocation to an owned, addressable record
        # before treating passive metrics as drain evidence.  Terminal jobs do
        # not need a discovery record because Slurm has already proved that
        # they cannot serve requests.
        live_record_matches = {
            job_id: [
                record for record in records
                if record.get("job_id") == job_id
            ]
            for job_id in live
        }
        if any(
            len(matches) != 1 or not matches[0].get("endpoint")
            for matches in live_record_matches.values()
        ):
            self._log(
                "Cannot drain safely: every live vLLM allocation must map "
                "to exactly one owned registry endpoint; retaining jobs."
            )
            self._restore_chain_registry_records(records)
            self._finish_stop_intent(stop_token, "quarantined")
            return "failed"
        live_records = [
            record for record in records if record.get("job_id") in live
        ]
        if not self._wait_for_vllm_idle(live_records):
            self._restore_chain_registry_records(records)
            self._finish_stop_intent(stop_token, "aborted")
            return "failed"
        if not self._cancellation_still_authorized(stop_token):
            self._restore_chain_registry_records(records)
            self._finish_stop_intent(stop_token, "aborted")
            return "owner_present" if self._owners_present() else "failed"

        success, accepted = self._cancel_exact_live_jobs(
            job_ids, stop_token=stop_token,
        )
        if not success:
            accepted_set = set(accepted)
            self._restore_chain_registry_records(
                records,
                job_ids=[
                    record.get("job_id") for record in records
                    if record.get("job_id") not in accepted_set
                ],
            )
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
        """Retain every generation ID while an unrenewed chain reaches wall."""
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
            self.state_manager.update_chain_jobs(
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

    def _cleanup_serve_registry(self):
        serve_file = self.project_root / "spatial_agent" / "logs" / "serve.json"
        lock_file = str(serve_file) + ".lock"
        if not serve_file.exists():
            return

        try:
            # Phase 1: Read under lock to get a consistent snapshot
            with FileLock(lock_file):
                if not serve_file.exists():
                    return
                with open(serve_file, "r") as f:
                    registry = json.load(f)

            # Phase 2: Check liveness WITHOUT lock (squeue can be slow).
            # Collect every slurm_id first, do ONE batched squeue, then
            # decide what to remove — avoids one squeue per registry entry.
            slurm_ids = []
            for servers in registry.values():
                for info in servers.values():
                    sid = info.get("slurm_job_id")
                    if sid:
                        slurm_ids.append(sid)
            job_status = batch_query_jobs(slurm_ids) if slurm_ids else {}

            # An outage snapshot deliberately contains UNKNOWN placeholders
            # for requested IDs, so truthiness cannot prove authority. Never
            # delete discovery records unless the scheduler carrier itself
            # attests a successful parse.
            if slurm_ids and not bool(
                getattr(job_status, "authoritative", False)
            ):
                return

            to_remove = set()  # (model_name, server_id) pairs
            for model_name, servers in registry.items():
                for server_id, info in servers.items():
                    slurm_id = info.get("slurm_job_id")
                    should_remove = False

                    if slurm_id:
                        ji = job_status.get(str(slurm_id))
                        if ji is None or ji["status"].upper() not in _ALIVE_STATUSES:
                            should_remove = True
                    else:
                        create_str = info.get("create_time")
                        if create_str:
                            try:
                                ct = datetime.datetime.strptime(create_str, "%Y/%m/%d %H:%M:%S")
                                age_h = (datetime.datetime.now() - ct).total_seconds() / 3600
                                if age_h > 8:
                                    should_remove = True
                            except Exception:
                                should_remove = True

                    if should_remove:
                        to_remove.add((model_name, server_id))

            if not to_remove:
                return

            # Phase 3: Re-read under lock and apply only confirmed removals
            with FileLock(lock_file):
                if not serve_file.exists():
                    return
                with open(serve_file, "r") as f:
                    registry = json.load(f)

                removed = 0
                for model_name, server_id in to_remove:
                    if model_name in registry and server_id in registry[model_name]:
                        del registry[model_name][server_id]
                        removed += 1
                        if not registry[model_name]:
                            del registry[model_name]

                if removed > 0:
                    write_json_atomic(serve_file, registry)
                    self._log(f"Cleaned serve.json: removed {removed} dead entries")

        except Exception as e:
            self._log(f"Warning: serve.json cleanup failed: {e}")

    def _all_recorded_jobs_terminal(self) -> bool:
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
        """Remove manager ownership only after authoritative terminal proof."""
        terminal = False
        if not self._shutdown_drain_failed:
            for attempt in range(_FINALIZE_VERIFY_ATTEMPTS):
                state = self._finalization_state()
                if state == "terminal":
                    terminal = True
                    break
                # Only Slurm's normal COMPLETING/KillWait tail warrants the
                # full retry window. Unknown authority or a genuinely live job
                # is quarantined immediately rather than hiding a control-plane
                # fault for minutes.
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
            self.state_manager.remove_chain(self.chain_id)
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

        self._log(f"Starting server chain for {self.model_name} ({self.served_name})")
        self._log(f"Chain ID: {self.chain_id}")
        self._log(f"Job duration: {self.time_limit}, overlap: {self.restart_before_minutes}m")

        self._cleanup_serve_registry()

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

            if self.pool_id and self.submitted_job_ids and not self._await_runtime_authority():
                self._log(
                    "Runtime authority stayed unavailable after bounded "
                    "checks; suppressing all successor submissions and "
                    "retaining the current allocation only to its wall limit."
                )
                self._wait_for_recorded_jobs_end()
                break

            self._cleanup_serve_registry()

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
                # submit_job returns False on sbatch error or fail-to-start —
                # treat both as fast failures so a broken config doesn't
                # spawn jobs forever.
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
                            "No active compatible owner at the renewal boundary; "
                            "suppressing successor submission."
                        )
                        self._wait_for_recorded_jobs_end()
                        break
                    self._log(f"Overlap window: submitting next job ({self.restart_before_minutes}m overlap)")
                else:
                    if self._shutdown_requested:
                        continue
                    # Job died early. If it ran longer than the fast-fail
                    # threshold we treat it as a normal restart cycle.
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


def recover_managed_chain(chain_state: ChainState, project_root: Path) -> str:
    """Safely drain a managed allocation whose local manager has died.

    Recovery deliberately reuses the same atomic owner fence, passive vLLM
    queue checks, exact-ID cancellation, and terminal verification as the
    normal manager.  It never provides a broad job-name cancellation path.
    """
    state_manager = ChainStateManager(project_root)
    if not chain_state.pool_id:
        return "not_managed"
    if state_manager.is_chain_alive(chain_state):
        return "manager_alive"

    chain = ServerChain(
        chain_id=chain_state.chain_id,
        served_name=chain_state.served_name,
        model_name=chain_state.model_name,
        model_path=chain_state.model_path,
        account=chain_state.account,
        partition=chain_state.partition,
        max_model_len=max(1, int(chain_state.max_model_len or 1)),
        max_num_seqs=max(1, int(chain_state.max_num_seqs or 1)),
        tp_size=max(1, int(chain_state.tp_size or 1)),
        gpus=max(1, int(chain_state.gpus or 1)),
        time_limit="4:00:00",
        restart_before_minutes=0,
        project_root=str(project_root),
        pool_id=str(chain_state.pool_id),
        namespace=str(chain_state.namespace or ""),
        server_spec_digest=str(chain_state.spec_digest or ""),
        topology_digest=str(chain_state.topology_digest or ""),
        desired_instances=max(1, int(chain_state.desired_instances or 1)),
        auto_stop_when_unowned=True,
        bootstrap_grace_minutes=0,
    )
    chain.submitted_job_ids = list(dict.fromkeys(
        str(job_id) for job_id in chain_state.slurm_job_ids if job_id
    ))
    chain.current_job_id = (
        chain.submitted_job_ids[-1] if chain.submitted_job_ids else None
    )
    chain._submission_intent = chain_state.submission_intent

    # A dead manager may have crashed after sbatch accepted but before its
    # parsed ID was persisted. Reconcile the exact durable comment first and
    # synchronize any adopted ID back into both owner stores.
    if chain_state.submission_intent is not None:
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
    served_name: str,
    model_name: str,
    model_path: str,
    account: str,
    partition: str,
    max_model_len: int,
    max_num_seqs: int,
    tp_size: int,
    kv_cache_dtype: str = "auto",
    quantization: str = "none",
    gpus: int = 8,
    time_limit: str = "4:00:00",
    restart_before_minutes: int = 20,
    project_root: Path = Path("."),
    pool_id: str = "",
    namespace: str = "",
    server_spec_digest: str = "",
    topology_digest: str = "",
    desired_instances: int = 1,
    auto_stop_when_unowned: bool = False,
    bootstrap_grace_minutes: int = 30,
    launch_claim_id: str = "",
    model_revision: Optional[str] = None,
) -> ChainState:
    """Start a server chain after transactionally persisting ownership."""
    config = {
        "chain_id": chain_id,
        "served_name": served_name,
        "model_name": model_name,
        "model_path": model_path,
        "model_revision": model_revision or None,
        "account": account,
        "partition": partition,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "tp_size": tp_size,
        "kv_cache_dtype": kv_cache_dtype,
        "quantization": quantization,
        "gpus": gpus,
        "time_limit": time_limit,
        "restart_before_minutes": restart_before_minutes,
        "project_root": str(project_root),
        "pool_id": pool_id,
        "namespace": namespace,
        "server_spec_digest": server_spec_digest,
        "topology_digest": topology_digest,
        "desired_instances": desired_instances,
        "auto_stop_when_unowned": auto_stop_when_unowned,
        "bootstrap_grace_minutes": bootstrap_grace_minutes,
    }

    lifecycle_store = LifecycleStore(project_root)
    if pool_id and pool_id not in lifecycle_store.snapshot().get("pools", {}):
        raise LifecycleError(
            f"Managed vLLM pool {pool_id!r} must be registered before launch"
        )

    # Log file for the chain manager process itself
    log_dir = project_root / "spatial_agent" / "logs" / "slurm_vllm"
    log_dir.mkdir(parents=True, exist_ok=True)
    # The child blocks on this inherited pipe before it can submit.  EOF means
    # the parent crashed or persistence failed; only an explicit byte after
    # both ownership records are durable lets it proceed.
    gate_read_fd, gate_write_fd = os.pipe()
    config["_startup_gate_fd"] = gate_read_fd
    chain_log, log_f = open_unique_text_artifact(
        log_dir,
        identity=f"chain-vllm-{served_name}-{chain_id}",
        suffix=".log",
    )
    proc = None
    try:
        proc = subprocess.Popen(
            [
                sys.executable, "-m",
                "spatial_agent.launch_managers.vllm_manager.server_chain",
                "--config", json.dumps(config),
            ],
            stdout=log_f,
            stderr=log_f,
            start_new_session=True,  # Survives CLI exit
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
        # The child is still blocked on the startup gate and therefore cannot
        # have submitted work.  Refuse to create a durable record that could
        # later bind this PID to an unrelated reused process.
        os.close(gate_write_fd)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_f.close()
        raise RuntimeError("Could not capture exact vLLM manager process identity")

    started_at = datetime.datetime.now().isoformat()
    chain_state = ChainState(
        chain_id=chain_id,
        served_name=served_name,
        model_name=model_name,
        model_path=model_path,
        pid=proc.pid,
        process_identity=manager_identity,
        slurm_job_ids=[],
        started_at=started_at,
        account=account,
        partition=partition,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        tp_size=tp_size,
        gpus=gpus,
        pool_id=pool_id or None,
        namespace=namespace or None,
        spec_digest=server_spec_digest or None,
        topology_digest=topology_digest or None,
        desired_instances=desired_instances if pool_id else None,
        lifecycle_state="launching" if pool_id else None,
    )

    state_manager = ChainStateManager(project_root)
    lifecycle_recorded = False
    gate_release_attempted = False
    try:
        state_manager.add_chain(chain_state)
        if pool_id:
            lifecycle_store.record_chain(
                pool_id, chain_id, pid=proc.pid, job_ids=[],
                launch_claim_id=launch_claim_id or None,
            )
            lifecycle_recorded = True
        # From this point onward an asynchronous exception can race a byte that
        # already reached the child.  Treat every outcome as ambiguous and keep
        # durable ownership until scheduler reconciliation proves no sbatch was
        # accepted.
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
                state_manager.remove_chain(chain_id)
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
                # The chain record itself remains the conservative ownership
                # fence even if the supplementary status write fails.
                pass
        raise
    finally:
        os.close(gate_write_fd)
        log_f.close()

    return chain_state, chain_log


def main():
    parser = argparse.ArgumentParser(description="Server chain subprocess")
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
