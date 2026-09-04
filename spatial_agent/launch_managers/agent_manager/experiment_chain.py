"""
Background experiment chain manager.

Each chain runs as an independent subprocess that manages a continuous loop
of sequential 4-hour SLURM jobs with automatic resume for one benchmark.

Can be invoked directly:
    python -m spatial_agent.launch_managers.agent_manager.experiment_chain --config '{"experiment_id": ..., ...}'
"""

import argparse
import datetime
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import tempfile
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

from spatial_agent.launch_managers.slurm_utils import (
    cancel_job,
    cancel_pending_job,
    check_slurm_available,
    filter_alive_jobs,
    get_job_status,
    get_exact_job_states,
    get_user_jobs_snapshot,
    get_user_steps_snapshot,
    resolve_partitions,
    wait_for_job_visible,
)
from spatial_agent.launch_managers.agent_manager import dispatcher as agent_dispatcher
from spatial_agent.launch_managers.lifecycle import (
    LifecycleError,
    LifecycleStore,
    canonical_digest,
    compare_process_identities,
    evaluator_submission_evidence_token,
    file_sha256,
    format_evaluator_job_comment,
    format_evaluator_step_name,
    manifest_environment,
    open_unique_text_artifact,
    process_identity,
    process_identity_is_valid,
    wait_for_ready_manifest,
    write_unique_text_artifact,
)


# Stop a chain after this many consecutive jobs that ended without
# producing any new predictions (e.g. dataloader crash early in startup).
# 3 strikes tolerates transient SLURM issues (NODE_FAIL, preemption) on
# an otherwise healthy chain without killing a long-running experiment.
_MAX_NO_PROGRESS_JOBS = 3
_CANCEL_VERIFY_ATTEMPTS = 60
_CANCEL_VERIFY_INTERVAL_SECONDS = 2.0
_CLIENT_LIFECYCLE_ENV_KEYS = (
    "SPATIAL_AGENT_LEASE_ID", "SPATIAL_AGENT_COHORT_ID",
    "SPATIAL_AGENT_OWNER_ID", "SPATIAL_AGENT_WORKLOAD_POLICY_DIGEST",
    "SPATIAL_AGENT_SERVICE_MANIFEST_DIGEST",
    "SPATIAL_AGENT_VLLM_POOL_ID", "SPATIAL_AGENT_VLLM_NAMESPACE",
    "SPATIAL_AGENT_VLLM_SERVER_SPEC_DIGEST",
    "SPATIAL_AGENT_VLLM_TOPOLOGY_DIGEST",
    "SPATIAL_AGENT_VLLM_SLOT_ALLOWLIST",
    "SPATIAL_AGENT_VLLM_ENDPOINT_ALLOWLIST",
    "SPATIAL_AGENT_TOOL_POOL_ID", "SPATIAL_AGENT_TOOL_NAMESPACE",
    "SPATIAL_AGENT_TOOL_SPEC_DIGEST", "SPATIAL_AGENT_TOOL_TOPOLOGY_DIGEST",
    "SPATIAL_AGENT_GPU_SERVER_SLOT_ALLOWLIST",
    "SPATIAL_AGENT_GPU_SERVER_UID_ALLOWLIST",
)
_SUBMISSION_RECOVERY_ATTEMPTS = 3
_SUBMISSION_RECOVERY_INTERVAL_SECONDS = 2.0
_ACCEPTED_ID_VISIBILITY_ATTEMPTS = 10
_ACCEPTED_ID_VISIBILITY_INTERVAL_SECONDS = 1.0
_UNRESOLVED_RECONCILE_INTERVAL_SECONDS = 5.0
_QUARANTINE_RETENTION_TIMEOUT_SECONDS = 10 * 60
_SERVICE_FINALIZE_RETRY_SECONDS = 5.0
from spatial_agent.launch_managers.agent_manager.state import ExperimentState, ExperimentStateManager


def _fsync_directory(path: Path) -> None:
    """Durably publish directory entries or fail the lifecycle transition."""
    try:
        directory_fd = os.open(
            path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise LifecycleError(
            f"Could not durably publish service epoch directory {path}: {exc}"
        ) from exc


def _redact_sensitive_config(value: Any) -> Any:
    """Copy policy metadata without publishing credentials."""
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            normalized = str(key).strip().lower()
            sensitive = (
                normalized in {"api_key", "token", "secret", "password"}
                or normalized.endswith(
                    ("_api_key", "_token", "_secret", "_password")
                )
            )
            redacted[key] = (
                "<redacted>" if sensitive else _redact_sensitive_config(item)
            )
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_config(item) for item in value]
    return value


class ExperimentChain:
    """Manages continuous chain of sequential 4-hour SLURM jobs for one experiment."""

    def __init__(
        self,
        experiment_id: str,
        benchmark: str,
        model_name: str,
        experiment_name: str,
        account: str,
        partition: str,
        gpus: int,
        time_limit: str,
        concurrency: int,
        subsample: int,
        project_root: str,
        experiment_type: str = "agent",
        max_frames: int = 0,
        system_prompt: str = "cot",
        scheduled_for: str = "",
        enable_service_lifecycle: bool = False,
        service_readiness_timeout_minutes: int = 240,
        service_running_readiness_timeout_minutes: int = 15,
        evaluator_pending_wait_minutes: int = 15,
        service_renewal_heartbeat_ttl_minutes: int = 5,
        service_finalize_timeout_minutes: int = 5,
        lease_id: str = "",
        cohort_id: str = "",
        generation_seed: Optional[int] = None,
    ):
        self.experiment_id = experiment_id
        self.benchmark = benchmark
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.account = account
        self.partition = resolve_partitions(partition, account)
        self.gpus = gpus
        self.time_limit = time_limit
        self.concurrency = concurrency
        self.subsample = subsample
        self.project_root = Path(project_root)
        self.experiment_type = experiment_type
        self.max_frames = max_frames
        self.system_prompt = system_prompt
        self.scheduled_for = scheduled_for
        self.enable_service_lifecycle = bool(enable_service_lifecycle)
        self.service_readiness_timeout_seconds = max(
            1, int(service_readiness_timeout_minutes) * 60
        )
        self.service_running_readiness_timeout_seconds = max(
            1, int(service_running_readiness_timeout_minutes) * 60
        )
        self.evaluator_pending_wait_seconds = max(
            1, int(evaluator_pending_wait_minutes) * 60
        )
        self.service_renewal_heartbeat_ttl_seconds = max(
            1, int(service_renewal_heartbeat_ttl_minutes) * 60
        )
        self.service_finalize_timeout_seconds = max(
            0, int(service_finalize_timeout_minutes) * 60
        )
        self.lease_id = lease_id or uuid.uuid4().hex
        self.cohort_id = cohort_id or f"cohort-{experiment_id}"
        self.generation_seed = (
            None if generation_seed is None else int(generation_seed)
        )
        self.service_environment: dict[str, str] = {}
        self._lease_acquired = False
        self._lease_finalized = False
        self._lease_renewal_epoch_id = ""
        self._last_lease_heartbeat = 0.0

        self.total_seconds = self._parse_time(time_limit)

        self.running = True
        self._stop_event = threading.Event()
        self.job_counter = 1
        self.current_job_id: Optional[str] = None
        self.submitted_job_ids: List[str] = []
        self._observed_job_ids: set[str] = set()
        self._submission_blocked = False
        self._submission_quarantine_reason = ""
        self._scheduler_ownership_uncertain = False
        self._unresolved_submission_comments: List[str] = []
        self._ambiguous_submission_comments: List[str] = []
        self._active_submission_intent: Optional[dict] = None
        self._queue_deferred_retry_required = False

        # Paths
        self.model_config = f"spatial_agent/config/model/{model_name}.json"
        self.dataset_config = f"spatial_agent/config/dataset/{benchmark}.json"

        if experiment_type == "cot":
            self.log_dir = self.project_root / "spatial_agent" / "logs" / "slurm_cot"
        else:
            self.agent_script = self.project_root / "spatial_agent" / "scripts" / "agent" / "slurm" / "run.sh"
            self.log_dir = self.project_root / "spatial_agent" / "logs" / "slurm_agent"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        evidence_token = evaluator_submission_evidence_token(self.experiment_id)
        self._submission_evidence_path = (
            self.log_dir / f"evaluator_submission_{evidence_token}.json"
        )

        # Derive work_dir to match run.py + run.sh logic
        self.work_dir = self._derive_work_dir()

        self.state_manager = ExperimentStateManager(self.project_root)
        self.lifecycle_store = LifecycleStore(self.project_root)

        self._model_runtime = self._read_json(self.project_root / self.model_config)
        self.llm_model = str(self._model_runtime.get("llm_model") or model_name)
        self.reconstruct_backend = str(
            self._model_runtime.get("reconstruct_backend") or "da3"
        )
        configured_tools = self._model_runtime.get(
            "tools_to_use", ["Reconstruct", "SAM3"]
        )
        self.needs_gpu_tools = (
            self.experiment_type == "agent" and bool(configured_tools)
        )
        self.workload_policy = self._build_workload_policy()
        self.workload_policy_digest = canonical_digest(self.workload_policy)
        self.run_policy_path = Path(self.work_dir) / "run_policy.json"

        self._cached_total: Optional[int] = None

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            with open(path, encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _build_workload_policy(self) -> dict:
        """Semantic/client policy digest with lifecycle files excluded."""
        policy = {
            "schema_version": 1,
            "experiment_type": self.experiment_type,
            "benchmark": self.benchmark,
            "model_config": self.model_config,
            "dataset_config": self.dataset_config,
            "model_runtime": self._model_runtime,
            "concurrency": self.concurrency,
            "subsample": self.subsample,
            "max_frames": self.max_frames,
            "system_prompt": self.system_prompt,
            "generation_seed": self.generation_seed,
        }
        provenance_enabled = bool(
            self.enable_service_lifecycle
            or self._model_runtime.get("record_input_fingerprints") is True
        )
        if not provenance_enabled:
            # Preserve legacy manager startup and resume behavior exactly for
            # profiles that did not opt into lifecycle/provenance controls.
            return policy

        semantic_paths = {
            "spatial_agent/config.py",
            "spatial_agent/workflow.py",
            "spatial_agent/state.py",
            "spatial_agent/gpu_rpc.py",
            "spatial_agent/kernel/manager.py",
            "spatial_agent/evals/base.py",
            "spatial_agent/evals/factory.py",
            "spatial_agent/evals/scoring.py",
            f"spatial_agent/evals/{self.benchmark}.py",
            # This manager constructs the evaluator command (including CoT
            # frame preparation), so it is result-affecting policy code.
            "spatial_agent/launch_managers/agent_manager/experiment_chain.py",
            "spatial_agent/requirements/requirements-agent.txt",
            "spatial_agent/requirements/requirements-vllm.txt",
            (
                "spatial_agent/entrypoints/cot_baseline.py"
                if self.experiment_type == "cot"
                else "spatial_agent/entrypoints/run.py"
            ),
            (
                "spatial_agent/scripts/cot_baseline/run.sh"
                if self.experiment_type == "cot"
                else "spatial_agent/scripts/agent/slurm/run.sh"
            ),
        }
        # These modules can change requests, tool inputs/outputs, transport
        # semantics, or the state machine that selects a benchmark answer.
        # Pure lifecycle files remain excluded because they only schedule
        # identical work and must not invalidate result provenance.
        for relative_directory in (
            "spatial_agent/llm",
            "spatial_agent/nodes",
            "spatial_agent/tools",
            "spatial_agent/gpu_models",
            "spatial_agent/kernel",
            "spatial_agent/kernel_types",
        ):
            directory = self.project_root / relative_directory
            if directory.exists():
                semantic_paths.update(
                    str(path.relative_to(self.project_root))
                    for path in directory.rglob("*.py")
                )

        backend_requirements = {
            "da3": (
                "tools/third_party/Depth-Anything-3/pyproject.toml",
                "tools/third_party/Depth-Anything-3/requirements.txt",
            ),
            "pi3": (
                "tools/third_party/Pi3/pyproject.toml",
                "tools/third_party/Pi3/requirements.txt",
            ),
            "mapanything": (
                "tools/third_party/map-anything/pyproject.toml",
            ),
        }
        semantic_paths.update(
            backend_requirements.get(self.reconstruct_backend.lower(), ())
        )
        configured_tools = {
            str(tool).strip().lower()
            for tool in self._model_runtime.get(
                "tools_to_use", ["Reconstruct", "SAM3"]
            )
        }
        if "sam3" in configured_tools:
            semantic_paths.add("tools/third_party/sam3/pyproject.toml")
        source_hashes = {}
        for relative in sorted(semantic_paths):
            path = self.project_root / relative
            if path.exists():
                source_hashes[relative] = file_sha256(path)
        model_path = self.project_root / self.model_config
        dataset_path = self.project_root / self.dataset_config
        policy.update({
            "model_config_sha256": file_sha256(model_path),
            "dataset_config_sha256": file_sha256(dataset_path),
            "model_runtime": _redact_sensitive_config(self._model_runtime),
            "semantic_source_sha256": source_hashes,
        })
        return policy

    def _establish_run_policy(self) -> None:
        """Publish or verify immutable provenance before writing predictions."""
        # Provenance/fencing is opt-in. Legacy/default-off experiments
        # historically resumed partial outputs without this sidecar.
        if not self._durable_submission_tracking_enabled():
            return
        prediction_path = Path(self.work_dir) / "predictions.jsonl"
        try:
            predictions_exist = (
                prediction_path.exists() and prediction_path.stat().st_size > 0
            )
        except OSError as exc:
            raise LifecycleError(
                f"Cannot verify existing predictions before resume: {exc}"
            ) from exc
        expected = {
            "schema_version": 1,
            "workload_policy_digest": self.workload_policy_digest,
            "workload_policy": self.workload_policy,
        }
        try:
            self.run_policy_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LifecycleError(
                f"Cannot create run-policy directory: {exc}"
            ) from exc

        if not self.run_policy_path.exists():
            if predictions_exist:
                raise LifecycleError(
                    "Partial predictions exist without run_policy.json; refusing "
                    "to resume with unverifiable result provenance"
                )
            payload = {
                **expected,
                "created_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
            temporary: Optional[str] = None
            try:
                fd, temporary = tempfile.mkstemp(
                    prefix=".run_policy.", suffix=".tmp",
                    dir=self.run_policy_path.parent,
                )
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    # Atomic create-without-replace: concurrent managers can
                    # publish the same policy but cannot overwrite provenance.
                    os.link(temporary, self.run_policy_path)
                    _fsync_directory(self.run_policy_path.parent)
                except FileExistsError:
                    pass
            except OSError as exc:
                raise LifecycleError(
                    f"Cannot atomically publish run-policy provenance: {exc}"
                ) from exc
            finally:
                if temporary is not None:
                    try:
                        os.unlink(temporary)
                    except FileNotFoundError:
                        pass

        try:
            with open(self.run_policy_path, encoding="utf-8") as handle:
                observed = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            raise LifecycleError(
                f"Run-policy provenance is unreadable: {self.run_policy_path}: {exc}"
            ) from exc
        if not isinstance(observed, dict):
            raise LifecycleError("Run-policy provenance root is not an object")
        if (
            observed.get("schema_version") != expected["schema_version"]
            or observed.get("workload_policy_digest")
            != expected["workload_policy_digest"]
            or canonical_digest(observed.get("workload_policy"))
            != expected["workload_policy_digest"]
        ):
            raise LifecycleError(
                "Run-policy provenance does not match this evaluator workload; "
                "refusing a result-mixing resume"
            )

    def _parse_time(self, time_str: str) -> int:
        parts = time_str.split(":")
        if len(parts) == 3:
            h, m, s = map(int, parts)
        elif len(parts) == 2:
            h = 0
            m, s = map(int, parts)
        else:
            raise ValueError(f"Invalid time format: {time_str}")
        return h * 3600 + m * 60 + s

    def _ts(self) -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log(self, msg: str):
        print(f"[{self._ts()}] {msg}", flush=True)

    def _derive_work_dir(self) -> str:
        """Derive work_dir matching run.py / cot_baseline.py logic."""
        model_config_path = self.project_root / self.model_config
        model_short = "unknown"
        if model_config_path.exists():
            try:
                with open(model_config_path) as f:
                    cfg = json.load(f)
                model_short = (
                    cfg.get("model_profile")
                    or cfg["llm_model"].split("/")[-1][:30]
                )
            except Exception:
                pass

        prefix = "cot" if self.experiment_type == "cot" else "spatial"
        pkg_dir = self.project_root / "spatial_agent"
        work_dir = pkg_dir / "work_dir" / f"{prefix}_{self.benchmark}_{model_short}"
        if self.experiment_name:
            work_dir = Path(f"{work_dir}_{self.experiment_name}")
        return str(work_dir)

    def _count_predictions(self) -> int:
        """Count unique, well-formed completed samples from predictions.jsonl."""
        pred_file = os.path.join(self.work_dir, "predictions.jsonl")
        if not os.path.exists(pred_file):
            return 0
        try:
            sample_ids = set()
            with open(pred_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        sample_ids.add(str(entry["sample_id"]))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
            return len(sample_ids)
        except Exception:
            return 0

    def _get_total_samples(self) -> int:
        """Get total number of samples. Caches after first successful load."""
        if self._cached_total is not None:
            return self._cached_total

        # 1. Check if the CLI already computed total_samples in the state file
        for exp in self.state_manager.list_experiments():
            if exp.experiment_id == self.experiment_id and exp.total_samples > 0:
                self._cached_total = exp.total_samples
                self._log(f"Total samples (from state): {exp.total_samples}")
                return exp.total_samples

        # 2. Try computing from benchmark factory (may fail in background process)
        try:
            from spatial_agent.evals.factory import BenchmarkFactory
            benchmark = BenchmarkFactory.create_benchmark(self.benchmark)
            if benchmark is not None:
                total = len(benchmark)
                if self.subsample > 0:
                    total = min(total, self.subsample)
                self._cached_total = total
                self.state_manager.update_experiment_total(
                    self.experiment_id, total
                )
                self._log(f"Total samples: {total}")
                return total
        except Exception as e:
            self._log(f"Warning: could not determine total samples: {e}")

        # 3. If subsample is set, use it as best guess
        if self.subsample > 0:
            self._cached_total = self.subsample
            return self.subsample

        return -1

    def _is_experiment_complete(self) -> bool:
        """Check if all samples have been processed."""
        total = self._get_total_samples()
        if total <= 0:
            return False
        completed = self._count_predictions()
        return completed >= total

    def _prepare_service_epoch(self) -> None:
        if not self.enable_service_lifecycle:
            return
        llm_base_url = str(
            self._model_runtime.get("llm_base_url") or ""
        ).strip().lower()
        if llm_base_url != "vllm":
            raise LifecycleError(
                "Lifecycle-managed evaluation requires model llm_base_url='vllm'; "
                "an explicit external URL would bypass the reserved local vLLM pool"
            )
        # Publish demand before resolving or waiting for serving capacity.  The
        # service managers are still separate processes, but this closes the
        # local select/readiness window in which a declared pool had no owner.
        requirements = {
            "llm_model": self.llm_model,
            "needs_gpu_tools": self.needs_gpu_tools,
            "reconstruct_backend": (
                self.reconstruct_backend if self.needs_gpu_tools else None
            ),
        }
        lease = self.lifecycle_store.acquire_lease(
            lease_id=self.lease_id,
            cohort_id=self.cohort_id,
            owner_id=self.experiment_id,
            owner_pid=os.getpid(),
            requirements=requirements,
            workload_policy=self.workload_policy,
            renewal_ttl_seconds=self.service_renewal_heartbeat_ttl_seconds,
        )
        self._lease_acquired = True
        self._lease_renewal_epoch_id = str(
            lease.get("renewal_epoch_id") or ""
        )
        if not self._lease_renewal_epoch_id:
            raise LifecycleError(
                "Reserved service lease has no renewable owner epoch"
            )
        self._log(
            f"Reserved demand-first service lease {self.lease_id[:8]}; "
            "resolving its exact declared pools."
        )

        vllm_pool = self.lifecycle_store.resolve_declared_pool(
            service_kind="vllm",
            identity_value=self.llm_model,
            requested_pool_id=(
                os.getenv("SPATIAL_AGENT_REQUIRED_VLLM_POOL_ID") or None
            ),
        )
        bound_requirements = {
            "vllm_pool_id": vllm_pool["pool_id"],
            "vllm_namespace": vllm_pool["namespace"],
            "server_spec_digest": (vllm_pool.get("digests") or {}).get(
                "server_spec"
            ),
            "vllm_topology_digest": (vllm_pool.get("digests") or {}).get(
                "topology"
            ),
        }
        tool_pool = None
        if self.needs_gpu_tools:
            tool_pool = self.lifecycle_store.resolve_declared_pool(
                service_kind="gpu_tools",
                identity_value=self.reconstruct_backend,
                requested_pool_id=(
                    os.getenv("SPATIAL_AGENT_REQUIRED_TOOL_POOL_ID") or None
                ),
            )
            bound_requirements.update({
                "tool_pool_id": tool_pool["pool_id"],
                "tool_namespace": tool_pool["namespace"],
                "tool_spec_digest": (tool_pool.get("digests") or {}).get(
                    "tool_spec"
                ),
                "tool_topology_digest": (tool_pool.get("digests") or {}).get(
                    "topology"
                ),
            })
        self.lifecycle_store.bind_lease_requirements(
            self.lease_id, bound_requirements
        )
        self._log(
            f"Bound service lease {self.lease_id[:8]}; waiting for exact "
            "healthy pool before evaluator submission."
        )
        manifest = wait_for_ready_manifest(
            self.project_root,
            llm_model=self.llm_model,
            needs_gpu_tools=self.needs_gpu_tools,
            reconstruct_backend=self.reconstruct_backend,
            vllm_pool_id=vllm_pool["pool_id"],
            tool_pool_id=(tool_pool["pool_id"] if tool_pool else None),
            timeout_seconds=self.service_readiness_timeout_seconds,
            running_timeout_seconds=(
                self.service_running_readiness_timeout_seconds
            ),
            stop_event=self._stop_event,
            resources_running=lambda: self._service_resources_running(
                [
                    vllm_pool["pool_id"],
                    *([tool_pool["pool_id"]] if tool_pool else []),
                ]
            ),
            on_poll=lambda: self._heartbeat_service_lease(
                force=True, evidence="service_readiness_poll"
            ),
        )
        realized_requirements = {
            "vllm_pool_id": manifest["vllm_pool_id"],
            "vllm_namespace": manifest["vllm_namespace"],
            "server_spec_digest": manifest.get("vllm_server_spec_digest"),
            "vllm_topology_digest": manifest.get("vllm_topology_digest"),
        }
        if self.needs_gpu_tools:
            realized_requirements.update({
                "tool_pool_id": manifest["tool_pool_id"],
                "tool_namespace": manifest["tool_namespace"],
                "tool_spec_digest": manifest.get("tool_spec_digest"),
                "tool_topology_digest": manifest.get("tool_topology_digest"),
            })
        # Re-assert the realized manifest against the already-declared pool.
        # This is idempotent narrowing, not a late selection of a "newest" pool.
        self.lifecycle_store.bind_lease_requirements(
            self.lease_id, realized_requirements
        )

        epoch_dir = Path(self.work_dir) / "service_epochs"
        epoch_dir_preexisted = epoch_dir.exists()
        epoch_dir.mkdir(parents=True, exist_ok=True)
        if not epoch_dir_preexisted:
            # Persist the newly created service_epochs entry before relying on
            # a manifest inside it after a host crash.
            _fsync_directory(epoch_dir.parent)
        manifest_path = epoch_dir / f"{self.lease_id}.json"
        payload = {
            "schema_version": 1,
            "lease_id": self.lease_id,
            "cohort_id": self.cohort_id,
            "owner_id": self.experiment_id,
            "workload_policy_digest": self.workload_policy_digest,
            "service_manifest": manifest,
        }
        fd, temporary = tempfile.mkstemp(
            prefix=f".{self.lease_id}.", suffix=".tmp", dir=epoch_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest_path)
            # File fsync alone does not make the rename durable.  Publish the
            # new manifest entry before activating the lease that references it.
            _fsync_directory(epoch_dir)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

        self.lifecycle_store.activate_lease(self.lease_id, manifest)
        self.lifecycle_store.set_pool_status(
            manifest["vllm_pool_id"], "ready",
            manifest_digest=manifest["manifest_digest"],
        )
        if self.needs_gpu_tools:
            self.lifecycle_store.set_pool_status(
                manifest["tool_pool_id"], "ready",
                manifest_digest=manifest["manifest_digest"],
            )
        self.service_environment = manifest_environment(
            manifest,
            lease_id=self.lease_id,
            cohort_id=self.cohort_id,
            owner_id=self.experiment_id,
            workload_policy_digest=self.workload_policy_digest,
        )
        os.environ.update(self.service_environment)
        if hasattr(self.state_manager, "update_experiment_lifecycle"):
            self.state_manager.update_experiment_lifecycle(
                self.experiment_id,
                lease_id=self.lease_id,
                cohort_id=self.cohort_id,
                workload_policy_digest=self.workload_policy_digest,
                service_manifest_path=str(manifest_path),
                service_manifest_digest=manifest["manifest_digest"],
                lifecycle_state="active",
            )
        self._log(
            f"Service epoch READY: {len(manifest['vllm_endpoints'])} vLLM "
            f"endpoint(s), {len(manifest.get('tool_endpoints', []))} tool "
            f"endpoint(s); manifest={manifest['manifest_digest'][:12]}."
        )

    def _service_resources_running(
        self, pool_ids: List[str],
    ) -> Optional[bool]:
        """Distinguish free scheduler wait from a consuming partial pool.

        ``None`` means scheduler authority is unavailable.  The readiness
        waiter treats that uncertainty as possibly GPU-consuming and starts
        the shorter deadline; it never uses it as cancellation permission.
        """
        requested = {str(pool_id) for pool_id in pool_ids if pool_id}
        snapshot = self.lifecycle_store.snapshot()
        job_ids = []
        for pool_id in requested:
            pool = (snapshot.get("pools") or {}).get(pool_id)
            if not isinstance(pool, dict):
                raise LifecycleError(
                    f"Required lifecycle pool {pool_id!r} disappeared"
                )
            for chain in (pool.get("chains") or {}).values():
                if not isinstance(chain, dict):
                    raise LifecycleError(
                        f"Pool {pool_id!r} has invalid chain authority"
                    )
                job_ids.extend(
                    str(job_id) for job_id in chain.get("job_ids", [])
                    if job_id
                )

        # A matching registry record proves that at least one service process
        # reached startup, even if the scheduler record is briefly unavailable.
        def read_registry(path: Path) -> Optional[dict]:
            try:
                with open(path, encoding="utf-8") as handle:
                    value = json.load(handle)
            except FileNotFoundError:
                return {}
            except (json.JSONDecodeError, OSError):
                return None
            return value if isinstance(value, dict) else None

        serve = read_registry(
            self.project_root / "spatial_agent" / "logs" / "serve.json"
        )
        if serve is None:
            return None
        for instances in serve.values():
            if not isinstance(instances, dict):
                continue
            if any(
                isinstance(record, dict)
                and str(record.get("pool_id") or "") in requested
                for record in instances.values()
            ):
                return True
        tools = read_registry(
            self.project_root / "spatial_agent" / "logs" / "gpu_server.json"
        )
        if tools is None:
            return None
        if any(
            isinstance(record, dict)
            and str(record.get("pool_id") or "") in requested
            for record in tools.values()
        ):
            return True

        exact = get_exact_job_states(job_ids, force=True)
        if exact is None:
            return None
        for info in exact.values():
            status = str((info or {}).get("status") or "").upper()
            # PENDING consumes no requested GPU allocation. CONFIGURING and
            # every other live squeue state may retain allocated resources.
            if status and status != "PENDING":
                return True
        return False

    def _heartbeat_service_lease(
        self, force: bool = False, *, evidence: str = "manager_progress",
        renewable: bool = True,
    ) -> None:
        if not self._lease_acquired or self._lease_finalized:
            return
        now = time.monotonic()
        if not force and now - self._last_lease_heartbeat < 30:
            return
        heartbeat_kwargs = {
            "job_ids": getattr(self, "submitted_job_ids", []),
            "owner_pid": os.getpid(),
        }
        if renewable:
            heartbeat_kwargs.update({
                "renewal_epoch_id": self._lease_renewal_epoch_id,
                "renewal_evidence": {
                    "kind": evidence,
                    "manager_pid": os.getpid(),
                },
            })
        self.lifecycle_store.heartbeat_lease(
            self.lease_id, **heartbeat_kwargs,
        )
        self._last_lease_heartbeat = now

    def _dispatch_overlay_with_lease_fence(
        self, overlay_args: List[str],
    ) -> bool:
        """Dispatch without letting a blocked manager heartbeat forever.

        While the dispatcher is blocked in a real overlay run, only an exact
        lease-tagged Slurm step refreshes the renewable epoch.  A short startup
        allowance covers the bounded slot search and srun visibility lag.
        """
        result: dict[str, bool] = {}
        failure: dict[str, BaseException] = {}

        def dispatch() -> None:
            try:
                result["dispatched"] = agent_dispatcher.try_dispatch_overlay(
                    self.project_root,
                    str(self.agent_script),
                    overlay_args,
                    self.concurrency,
                    log_fn=self._log,
                    stop_event=self._stop_event,
                )
            except BaseException as exc:
                failure["error"] = exc

        worker = threading.Thread(
            target=dispatch,
            name=f"overlay-{self.experiment_id[:16]}",
            daemon=True,
        )
        started = time.monotonic()
        worker.start()
        step_name = format_evaluator_step_name(self.lease_id)
        while worker.is_alive():
            worker.join(timeout=30.0)
            if not worker.is_alive():
                break
            steps = get_user_steps_snapshot()
            tagged_step_live = steps is not None and any(
                info.get("name") == step_name for info in steps.values()
            )
            if tagged_step_live or time.monotonic() - started <= 60.0:
                self._heartbeat_service_lease(
                    force=True,
                    evidence=(
                        "live_overlay_step" if tagged_step_live
                        else "overlay_startup"
                    ),
                )
        if "error" in failure:
            raise failure["error"]
        return bool(result.get("dispatched", False))

    def _overlay_owned_by_this_process(self) -> Optional[bool]:
        """Return overlay ownership, or ``None`` when authority is unreadable."""
        path = self.project_root / "spatial_agent" / "logs" / "agent_reservations.json"
        try:
            with open(path, encoding="utf-8") as handle:
                reservations = json.load(handle)
        except FileNotFoundError:
            return False
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(reservations, dict):
            return None
        current_identity = process_identity(os.getpid())
        matching_pid_identity_unknown = False
        for slots in reservations.values():
            if not isinstance(slots, list):
                return None
            for slot in slots:
                if not isinstance(slot, dict):
                    return None
                try:
                    owner_pid = int(slot.get("pid") or 0)
                except (TypeError, ValueError):
                    return None
                if owner_pid == os.getpid():
                    recorded_identity = slot.get("process_identity")
                    if (
                        not process_identity_is_valid(
                            recorded_identity, expected_pid=owner_pid,
                        )
                        or current_identity is None
                    ):
                        matching_pid_identity_unknown = True
                        continue
                    comparison = compare_process_identities(
                        recorded_identity, current_identity,
                    )
                    if comparison is True:
                        return True
                    if comparison is None:
                        matching_pid_identity_unknown = True
                    # Exact mismatch proves this numeric PID was reused; this
                    # reservation belongs to the earlier process, not us.
        return None if matching_pid_identity_unknown else False

    def _finalize_service_lease(self, outcome: str) -> None:
        if not self._lease_acquired or self._lease_finalized:
            return
        try:
            self.lifecycle_store.suppress_lease_renewal(
                self.lease_id,
                renewal_epoch_id=self._lease_renewal_epoch_id,
                reason=f"manager_finalizing:{outcome}",
            )
        except Exception as exc:
            # Never refresh successor authority after finalization begins. If
            # this fence cannot persist, the last heartbeat still expires at
            # its short TTL and the current allocation is left untouched.
            self._log(f"Could not persist immediate renewal suppression: {exc}")
        timeout = max(0.0, float(getattr(
            self, "service_finalize_timeout_seconds", 0.0
        )))
        deadline = time.monotonic() + timeout
        first_reason: Optional[str] = None
        while True:
            try:
                reason, retryable = self._try_finalize_service_lease(outcome)
            except Exception as exc:
                reason = f"lifecycle finalization transition failed: {exc}"
                retryable = True
            if reason is None:
                return
            if first_reason is None:
                first_reason = reason
                self._log(f"Lease finalization waiting: {reason}")
            remaining = deadline - time.monotonic()
            if not retryable or remaining <= 0:
                self._log(
                    "Lease remains active after bounded finalization: "
                    f"{reason}"
                )
                return
            time.sleep(min(_SERVICE_FINALIZE_RETRY_SECONDS, remaining))

    def _try_finalize_service_lease(
        self, outcome: str,
    ) -> Tuple[Optional[str], bool]:
        """Attempt one authoritative release; return (blocker, retryable)."""
        if self._scheduler_ownership_uncertain:
            return (
                "an accepted submission has no resolved exact scheduler job ID",
                False,
            )
        try:
            self._heartbeat_service_lease(force=True, renewable=False)
        except Exception as exc:
            return (
                f"final ownership heartbeat failed: {exc}",
                True,
            )
        scheduler = get_exact_job_states(self.submitted_job_ids, force=True)
        steps = get_user_steps_snapshot()
        if scheduler is None or steps is None:
            return (
                "scheduler job/step authority is unavailable",
                True,
            )
        alive_jobs = sorted(str(job_id) for job_id in scheduler)
        step_name = format_evaluator_step_name(self.lease_id)
        live_owned_steps = sorted(
            str(step_id) for step_id, info in steps.items()
            if info.get("name") == step_name
        )
        overlay_owned = self._overlay_owned_by_this_process()
        if overlay_owned is None:
            return (
                "overlay reservation authority is unreadable",
                True,
            )
        if (
            alive_jobs
            or live_owned_steps
            or overlay_owned
        ):
            # Keep the owning lease active so a service generation cannot time
            # out underneath a still-live evaluator/overlay producer.
            self._heartbeat_service_lease(force=True, renewable=False)
            return (
                "evaluator jobs/overlay steps are not terminal: "
                f"jobs={alive_jobs}, "
                f"steps={live_owned_steps or 'reservation active'}",
                True,
            )
        lease = self.lifecycle_store.snapshot()["leases"].get(self.lease_id, {})
        if lease.get("status") == "released":
            self._lease_finalized = True
            return None, False
        if lease.get("status") in {"reserved", "active"}:
            self.lifecycle_store.begin_drain(self.lease_id)
        if lease.get("status") != "released":
            self.lifecycle_store.mark_terminal(
                self.lease_id,
                outcome,
                completed=self._count_predictions(),
                total=self._get_total_samples(),
                work_dir=self.work_dir,
            )
            self.lifecycle_store.release_lease(
                self.lease_id,
                scheduler_checked=True,
                overlay_checked=True,
                writer_process_terminal=True,
            )
        self._lease_finalized = True
        if hasattr(self.state_manager, "update_experiment_lifecycle"):
            try:
                self.state_manager.update_experiment_lifecycle(
                    self.experiment_id, lifecycle_state="released",
                )
            except Exception as exc:
                # The lifecycle authority is already released. Experiment
                # dashboard metadata is advisory and can be reconciled later.
                self._log(
                    "Warning: lease released but experiment lifecycle metadata "
                    f"could not be updated: {exc}"
                )
        return None, False

    def _service_export_block(self) -> str:
        """Render frozen lifecycle metadata as shell-safe, out-of-band env."""
        lines = ["unset " + " ".join(_CLIENT_LIFECYCLE_ENV_KEYS)]
        lines.extend(
            f"export {key}={shlex.quote(value)}"
            for key, value in sorted(self.service_environment.items())
        )
        return "\n".join(lines)

    def _durable_submission_tracking_enabled(self) -> bool:
        """Whether evaluator submission fencing is explicitly opted in."""
        return bool(
            self.enable_service_lifecycle
            or self._model_runtime.get("record_input_fingerprints") is True
        )

    def _persist_submission_evidence(self) -> bool:
        """Atomically retain ownership evidence outside the manager process."""
        if not self._durable_submission_tracking_enabled():
            return True
        payload = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "lease_id": self.lease_id if self.enable_service_lifecycle else "",
            "submitted_job_ids": list(dict.fromkeys(self.submitted_job_ids)),
            "observed_job_ids": sorted(self._observed_job_ids),
            "unresolved_submission_comments": list(dict.fromkeys(
                self._unresolved_submission_comments
            )),
            "ambiguous_submission_comments": list(dict.fromkeys(
                self._ambiguous_submission_comments
            )),
            "active_submission_intent": self._active_submission_intent,
            "scheduler_ownership_uncertain": self._scheduler_ownership_uncertain,
            "quarantine_reason": self._submission_quarantine_reason,
            "updated_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        }
        temporary: Optional[str] = None
        try:
            fd, temporary = tempfile.mkstemp(
                prefix=f".{self._submission_evidence_path.name}.",
                suffix=".tmp",
                dir=self._submission_evidence_path.parent,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._submission_evidence_path)
            return True
        except OSError as exc:
            self._log(f"Could not persist evaluator submission evidence: {exc}")
            return False
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def _new_submission_comment(self, job_number: int) -> str:
        """Return a unique, scheduler-searchable evaluator ownership tag."""
        return format_evaluator_job_comment(
            lease_id=(self.lease_id if self.enable_service_lifecycle else ""),
            experiment_id=self.experiment_id,
            job_number=job_number,
            nonce=uuid.uuid4().hex,
        )

    def _persist_evaluator_submission_transition(
        self,
        submission_comment: Optional[str],
        phase: Optional[str],
        job_ids: List[str],
        *,
        observed: bool = False,
    ) -> bool:
        """Publish one evaluator submission transition to every authority.

        Each underlying authority uses an atomic replace plus fsync.  The
        transition must reach ExperimentState, the lifecycle lease (when
        enabled), and the independent ownership-evidence file before sbatch is
        allowed to run.
        """
        if not self._durable_submission_tracking_enabled():
            return True
        intent = None
        if submission_comment is not None:
            intent = {
                "schema_version": 1,
                "comment": submission_comment,
                "phase": phase,
                "job_ids": [str(job_id) for job_id in job_ids],
                "observed": bool(observed),
                "updated_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
        state_persisted = False
        try:
            self.state_manager.update_experiment_submission(
                self.experiment_id,
                comment=submission_comment,
                phase=phase,
                job_ids=list(job_ids),
                observed=observed,
            )
            persisted = next(
                (
                    experiment
                    for experiment in self.state_manager.list_experiments()
                    if experiment.experiment_id == self.experiment_id
                ),
                None,
            )
            state_persisted = bool(
                persisted is not None
                and getattr(
                    persisted, "evaluator_submission_comment", None
                ) == submission_comment
                and getattr(
                    persisted, "evaluator_submission_phase", None
                ) == phase
                and set(str(job_id) for job_id in job_ids).issubset(
                    set(
                        str(job_id)
                        for job_id in getattr(persisted, "slurm_job_ids", [])
                    )
                )
            )
        except Exception as exc:
            self._log(
                "Could not persist evaluator submission transition in "
                f"ExperimentState: {exc}"
            )

        lease_persisted = not (
            self.enable_service_lifecycle and self._lease_acquired
        )
        if self.enable_service_lifecycle and self._lease_acquired:
            try:
                self.lifecycle_store.update_evaluator_submission(
                    self.lease_id,
                    owner_pid=os.getpid(),
                    submission_intent=intent,
                    job_ids=list(job_ids),
                )
                lease_persisted = True
            except Exception as exc:
                self._log(
                    "Could not persist evaluator submission transition in "
                    f"service lease: {exc}"
                )

        self._active_submission_intent = intent
        evidence_persisted = self._persist_submission_evidence()
        return state_persisted and lease_persisted and evidence_persisted

    def _mark_evaluator_submission_observed(self, job_id: str) -> None:
        """Record first authoritative visibility without changing results."""
        self._observed_job_ids.add(str(job_id))
        intent = self._active_submission_intent
        if not isinstance(intent, dict):
            self._persist_submission_evidence()
            return
        if str(job_id) not in {
            str(value) for value in intent.get("job_ids", [])
        }:
            self._persist_submission_evidence()
            return
        if not self._persist_evaluator_submission_transition(
            str(intent.get("comment") or ""),
            "observed",
            [str(job_id)],
            observed=True,
        ):
            self._log(
                f"Warning: evaluator job {job_id} was observed but its "
                "observed submission phase could not be durably refreshed"
            )

    def _generate_sbatch(
        self, job_number: int, submission_comment: str = "",
    ) -> str:
        if self.experiment_type == "cot":
            return self._generate_cot_sbatch(job_number, submission_comment)
        return self._generate_agent_sbatch(job_number, submission_comment)

    def _generate_agent_sbatch(
        self, job_number: int, submission_comment: str = "",
    ) -> str:
        job_name = f"spatial-{self.benchmark}"
        gpu_line = f"#SBATCH --gpus-per-node={self.gpus}" if self.gpus > 0 else ""
        exclusive = "" if ("interactive" in self.partition or self.gpus == 0) else "#SBATCH --exclusive"
        mem_gb = min(max(self.concurrency * 4, 16), 48)
        cpus = min(self.concurrency + 2, 8)
        lifecycle_exports = self._service_export_block()
        comment_line = (
            f"#SBATCH --comment={submission_comment}"
            if submission_comment else ""
        )
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
{gpu_line}
#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={self.time_limit}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
{comment_line}
{exclusive}

echo "=========================================="
echo "Spatial Agent Evaluation - Chain {self.experiment_id[:8]} - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Benchmark: {self.benchmark}"
echo "Concurrency: {self.concurrency}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

cd {self.project_root}

CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent

{lifecycle_exports}

bash {self.agent_script} "{self.benchmark}" {self.concurrency} \\
    "{self.model_config}" "{self.dataset_config}" "{self.experiment_name}" \\
    {self.subsample} "{'' if self.generation_seed is None else self.generation_seed}"

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
"""

    def _generate_cot_sbatch(
        self, job_number: int, submission_comment: str = "",
    ) -> str:
        job_name = f"cot-{self.benchmark}"
        exclusive = ""
        gpu_line = f"#SBATCH --gpus-per-node={self.gpus}" if self.gpus > 0 else ""
        mem_gb = 8
        cpus = 4
        lifecycle_exports = self._service_export_block()
        comment_line = (
            f"#SBATCH --comment={submission_comment}"
            if submission_comment else ""
        )

        # Build the python command
        cmd_parts = [
            "python -m spatial_agent.entrypoints.cot_baseline",
            f"--dataset {self.dataset_config}",
            f"--model {self.model_config}",
            f"--concurrency {self.concurrency}",
            "--resume",
        ]
        frame_preparation = self._model_runtime.get(
            "cot_frame_preparation", "legacy"
        )
        if frame_preparation not in {"legacy", "spatial-agent"}:
            raise ValueError(
                "Model config field 'cot_frame_preparation' must be "
                "'legacy' or 'spatial-agent'"
            )
        cmd_parts.append(f"--frame-preparation {frame_preparation}")
        if self.max_frames > 0:
            cmd_parts.append(f"--max_frames {self.max_frames}")
        if self.system_prompt:
            cmd_parts.append(f"--system_prompt {self.system_prompt}")
        if self.subsample > 0:
            cmd_parts.append(f"--subsample {self.subsample}")
        if self.generation_seed is not None:
            cmd_parts.append(f"--seed {self.generation_seed}")
        if self.work_dir:
            cmd_parts.append(f"--work_dir {self.work_dir}")

        cmd = " \\\n    ".join(cmd_parts)

        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={self.account}
#SBATCH --partition={self.partition}
#SBATCH --nodes=1
{gpu_line}
#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={self.time_limit}
#SBATCH --output={self.log_dir}/{job_name}_%j.out
#SBATCH --error={self.log_dir}/{job_name}_%j.err
{comment_line}
{exclusive}

echo "=========================================="
echo "CoT Baseline Evaluation - Chain {self.experiment_id[:8]} - Job #{job_number}"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Benchmark: {self.benchmark}"
echo "Concurrency: {self.concurrency}"
echo "Max Frames: {self.max_frames}"
echo "System Prompt: {self.system_prompt}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

cd {self.project_root}

# Disable core dumps
ulimit -c 0
export PYTHONUNBUFFERED=1

CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent

{lifecycle_exports}

{cmd}

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
"""

    def _persist_queue_deferred_retry_required(self, job_id: str) -> bool:
        """Durably retire one terminal pending job and request a later retry."""
        self.submitted_job_ids = [
            recorded for recorded in self.submitted_job_ids
            if str(recorded) != str(job_id)
        ]
        if not self._update_state_jobs():
            return False
        if not self._persist_evaluator_submission_transition(
            None, None, [], observed=False,
        ):
            return False
        self._observed_job_ids.discard(str(job_id))
        self._update_status("queue_deferred_retry_required")
        try:
            persisted = next(
                (
                    experiment
                    for experiment in self.state_manager.list_experiments()
                    if experiment.experiment_id == self.experiment_id
                ),
                None,
            )
        except Exception:
            return False
        if (
            persisted is None
            or getattr(persisted, "status", None)
            != "queue_deferred_retry_required"
        ):
            return False
        self._queue_deferred_retry_required = True
        self._persist_submission_evidence()
        return True

    def _defer_strictly_pending_evaluator(self, job_id: str) -> str:
        """State-filter cancel one pending job without touching an active race.

        Returns the authoritative state observed after the cancellation race,
        ``DEFERRED`` after terminal proof and durable deferral, or
        ``QUARANTINED`` when scheduler/persistence authority is uncertain.
        """
        states = get_exact_job_states([job_id], force=True)
        if states is None:
            self._quarantine_submission(
                "scheduler authority unavailable at managed evaluator pending "
                "bound; ownership retained",
                ownership_uncertain=True,
            )
            return "QUARANTINED"
        info = states.get(str(job_id))
        if info is None:
            if not self._persist_queue_deferred_retry_required(job_id):
                self._quarantine_submission(
                    "pending evaluator became terminal at its bound but "
                    "deferred-retry status/fence was not durably persisted"
                )
                return "QUARANTINED"
            self._log(
                f"Managed evaluator {job_id} was terminal when its pending "
                "bound was rechecked; explicit deferred retry required."
            )
            return "DEFERRED"
        exact_status = str(info.get("status") or "").upper()
        if not exact_status or exact_status == "UNKNOWN":
            self._quarantine_submission(
                "scheduler returned no authoritative state at managed "
                "evaluator pending bound; ownership retained",
                ownership_uncertain=True,
            )
            return "QUARANTINED"
        if exact_status in {"CONFIGURING", "RUNNING"}:
            return exact_status

        terminalizing_states = {
            "CANCELLED", "COMPLETED", "COMPLETING", "FAILED", "NODE_FAIL",
            "PREEMPTED", "TIMEOUT",
        }
        if exact_status not in {"PENDING", *terminalizing_states}:
            self._quarantine_submission(
                f"unexpected evaluator state {exact_status!r} at managed "
                "pending bound; ownership retained",
                ownership_uncertain=True,
            )
            return "QUARANTINED"

        # Slurm evaluates this predicate atomically with cancellation.  A job
        # that advances after the snapshot cannot be cancelled by this call.
        if exact_status == "PENDING":
            cancel_pending_job(str(job_id))
        for attempt in range(_CANCEL_VERIFY_ATTEMPTS):
            states = get_exact_job_states([job_id], force=True)
            if states is None:
                self._quarantine_submission(
                    "scheduler authority unavailable after state-filtered "
                    "pending cancellation; ownership retained",
                    ownership_uncertain=True,
                )
                return "QUARANTINED"
            info = states.get(str(job_id))
            if info is None:
                if not self._persist_queue_deferred_retry_required(job_id):
                    self._quarantine_submission(
                        "pending evaluator became terminal but deferred-retry "
                        "status/fence was not durably persisted"
                    )
                    return "QUARANTINED"
                self._log(
                    f"Managed evaluator {job_id} exceeded its pending bound; "
                    "released for an explicit deferred retry."
                )
                return "DEFERRED"
            exact_status = str(info.get("status") or "").upper()
            if not exact_status or exact_status == "UNKNOWN":
                self._quarantine_submission(
                    "scheduler returned no authoritative state after "
                    "state-filtered pending cancellation; ownership retained",
                    ownership_uncertain=True,
                )
                return "QUARANTINED"
            if exact_status in {"CONFIGURING", "RUNNING"}:
                self._log(
                    f"Evaluator {job_id} advanced to {exact_status} during "
                    "state-filtered pending cancellation; continuing normally."
                )
                return exact_status
            if exact_status not in {"PENDING", *terminalizing_states}:
                self._quarantine_submission(
                    f"unexpected evaluator state {exact_status!r} after "
                    "managed pending bound; ownership retained",
                    ownership_uncertain=True,
                )
                return "QUARANTINED"
            if attempt + 1 < _CANCEL_VERIFY_ATTEMPTS:
                time.sleep(_CANCEL_VERIFY_INTERVAL_SECONDS)

        self._quarantine_submission(
            "state-filtered evaluator cancellation was not terminal-proven; "
            "ownership retained",
            ownership_uncertain=True,
        )
        return "QUARANTINED"

    def _wait_for_job_to_start(self, job_id: str, timeout: int = 7200) -> bool:
        """Wait for a SLURM job to start running."""
        self._log(f"Waiting for job {job_id} to start...")
        # Force-refresh the squeue cache so a just-submitted job becomes
        # visible immediately (covers the brief sbatch→squeue lag).
        wait_for_job_visible(job_id)
        start = time.monotonic()
        last_status = None
        last_update_time = start
        pending_since: Optional[float] = None

        while self.running:
            status = get_job_status(job_id)
            if (
                status not in {"UNKNOWN", "NOT_FOUND"}
                and str(job_id) not in self._observed_job_ids
            ):
                self._mark_evaluator_submission_observed(job_id)
            if not self._heartbeat_owned_job_or_quarantine(
                job_id, job_status=status
            ):
                return False
            now = time.monotonic()

            if status != last_status:
                elapsed_m = int((now - start) / 60)
                self._log(f"Job {job_id} status: {status} (waited {elapsed_m}m)")
                last_status = status
                last_update_time = now

            if status in ("PENDING", "CONFIGURING") and now - last_update_time >= 300:
                elapsed_m = int((now - start) / 60)
                self._log(f"Still waiting for job {job_id}... ({elapsed_m}m)")
                last_update_time = now

            if status == "RUNNING":
                return True
            elif status in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "NOT_FOUND"):
                self._log(f"Job {job_id} failed: {status}")
                return False

            if status == "PENDING":
                if pending_since is None:
                    pending_since = now
                if (
                    self.enable_service_lifecycle
                    and now - pending_since
                    >= self.evaluator_pending_wait_seconds
                ):
                    outcome = self._defer_strictly_pending_evaluator(job_id)
                    if outcome == "RUNNING":
                        return True
                    if outcome in {"DEFERRED", "QUARANTINED"}:
                        return False
                    # CONFIGURING or another authoritative non-pending state
                    # won the race. Reset the pending clock and let normal
                    # startup handling continue without cancellation/retry.
                    pending_since = None
            elif status != "UNKNOWN":
                # Scheduler uncertainty is not proof that an observed pending
                # allocation left the queue; retain the accumulated bound so
                # the authoritative exact check fails closed when due.
                pending_since = None

            if now - start > timeout:
                if status == "UNKNOWN":
                    self._quarantine_submission(
                        "scheduler authority remained unknown through evaluator "
                        "startup timeout; replacement blocked"
                    )
                self._log(f"Timeout waiting for job {job_id} to start")
                return False

            if self._stop_event.wait(timeout=30):
                return False

        return False

    def _wait_for_job_completion(self, job_id: str) -> bool:
        """Wait for a SLURM job to complete. Returns True if completed."""
        self._log(f"Waiting for job {job_id} to complete...")
        start = time.time()
        last_update_time = start

        while self.running:
            status = get_job_status(job_id)
            if (
                status not in {"UNKNOWN", "NOT_FOUND"}
                and str(job_id) not in self._observed_job_ids
            ):
                self._mark_evaluator_submission_observed(job_id)
            if not self._heartbeat_owned_job_or_quarantine(
                job_id, job_status=status
            ):
                return False
            now = time.time()

            # Periodic progress updates every 10 minutes
            if now - last_update_time >= 600:
                elapsed_m = int((now - start) / 60)
                completed = self._count_predictions()
                total = self._get_total_samples()
                if total > 0:
                    self._log(f"Job {job_id} running ({elapsed_m}m) — progress: {completed}/{total}")
                else:
                    self._log(f"Job {job_id} running ({elapsed_m}m) — completed: {completed}")
                last_update_time = now

            if status == "RUNNING":
                if self._stop_event.wait(timeout=60):
                    return False
                continue
            elif status in ("COMPLETED", "NOT_FOUND"):
                # NOT_FOUND means job left the queue — check sacct
                if status == "NOT_FOUND":
                    cmd = ["sacct", "-j", str(job_id), "-n", "-o", "State"]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        if str(job_id) not in self._observed_job_ids:
                            self._mark_evaluator_submission_observed(job_id)
                        sacct_status = result.stdout.strip().split()[0].upper()
                        if "COMPLETED" not in sacct_status:
                            self._log(f"Job {job_id} ended: {sacct_status}")
                            return False
                elapsed_m = int((now - start) / 60)
                self._log(f"Job {job_id} completed (ran {elapsed_m}m)")
                return True
            elif status in ("FAILED", "TIMEOUT", "NODE_FAIL", "CANCELLED"):
                elapsed_m = int((now - start) / 60)
                self._log(f"Job {job_id} ended: {status} (after {elapsed_m}m)")
                return False
            else:
                if self._stop_event.wait(timeout=60):
                    return False

        return False

    def _build_agent_script_args(self) -> List[str]:
        """Args matching the `bash run.sh ...` invocation in _generate_agent_sbatch."""
        return [
            self.benchmark,
            str(self.concurrency),
            self.model_config,
            self.dataset_config,
            self.experiment_name,
            str(self.subsample),
            "" if self.generation_seed is None else str(self.generation_seed),
        ]

    @staticmethod
    def _parse_parsable_job_id(stdout: str) -> Optional[str]:
        """Strictly parse ``sbatch --parsable`` output."""
        lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
        if len(lines) != 1:
            return None
        fields = lines[0].split(";")
        if len(fields) not in {1, 2}:
            return None
        job_id = fields[0]
        if len(fields) == 2 and (
            not fields[1] or any(character.isspace() for character in fields[1])
        ):
            return None
        return job_id if job_id.isdigit() else None

    def _recover_submitted_job_by_comment(
        self, submission_comment: str,
    ) -> Optional[Tuple[str, Any]]:
        """Recover one exact ID and its already-attested scheduler record."""
        last_match_count: Optional[int] = None
        for attempt in range(_SUBMISSION_RECOVERY_ATTEMPTS):
            snapshot = get_user_jobs_snapshot(force=True)
            if not getattr(snapshot, "authoritative", False):
                last_match_count = None
            else:
                matches = [
                    (str(job_id), info)
                    for job_id, info in dict.items(snapshot)
                    if isinstance(info, dict)
                    and str(info.get("comment") or "") == submission_comment
                ]
                last_match_count = len(matches)
                if len(matches) == 1:
                    return matches[0]
                if len(matches) > 1:
                    break
            if attempt + 1 < _SUBMISSION_RECOVERY_ATTEMPTS:
                time.sleep(_SUBMISSION_RECOVERY_INTERVAL_SECONDS)

        if last_match_count is None:
            self._log(
                "Cannot recover malformed sbatch output: scheduler authority "
                "is unavailable."
            )
        else:
            self._log(
                "Cannot recover malformed sbatch output: expected exactly one "
                f"job with comment {submission_comment!r}, found "
                f"{last_match_count}."
            )
        return None

    @staticmethod
    def _matching_comment_job_ids(snapshot, submission_comment: str) -> List[str]:
        return sorted(
            str(job_id)
            for job_id, info in dict.items(snapshot)
            if isinstance(info, dict)
            and str(info.get("comment") or "") == submission_comment
        )

    def _heartbeat_recovered_ownership(self) -> bool:
        if not self._lease_acquired or self._lease_finalized:
            return False
        try:
            self._heartbeat_service_lease(force=True, renewable=False)
            return True
        except Exception as exc:
            self._log(
                "Could not persist recovered evaluator ownership in lease: "
                f"{exc}"
            )
            return False

    def _reconcile_unresolved_submission_comments(self) -> bool:
        """Resolve accepted submissions whose exact Slurm ID was unavailable.

        Returns ``True`` once every unresolved comment is either proven absent
        or converted into durable exact-job ownership. Scheduler uncertainty is
        never interpreted as absence.
        """
        if not self._unresolved_submission_comments:
            return True
        snapshot = get_user_jobs_snapshot(force=True)
        if not getattr(snapshot, "authoritative", False):
            self._persist_submission_evidence()
            return False

        still_unresolved: List[str] = []
        for submission_comment in list(dict.fromkeys(
            self._unresolved_submission_comments
        )):
            matches = self._matching_comment_job_ids(
                snapshot, submission_comment,
            )
            if not matches:
                self._log(
                    "Accepted evaluator comment is not currently visible; "
                    "retaining its durable fence because never-observed "
                    f"absence is not terminal proof: {submission_comment!r}."
                )
                still_unresolved.append(submission_comment)
                continue

            for job_id in matches:
                if job_id not in self.submitted_job_ids:
                    self.submitted_job_ids.append(job_id)
                self._observed_job_ids.add(job_id)
            transition_phase = (
                "observed" if len(matches) == 1 else "ambiguous_multiple"
            )
            transition_persisted = self._persist_evaluator_submission_transition(
                submission_comment,
                transition_phase,
                matches,
                observed=True,
            )
            state_persisted = self._update_state_jobs()
            lease_persisted = self._heartbeat_recovered_ownership()

            if len(matches) > 1:
                self._log(
                    "Multiple live evaluator jobs share one unique submission "
                    f"comment; retaining all exact IDs without replacement: {matches}"
                )
                if submission_comment not in self._ambiguous_submission_comments:
                    self._ambiguous_submission_comments.append(submission_comment)
                if not (
                    transition_persisted
                    or state_persisted
                    or lease_persisted
                ):
                    still_unresolved.append(submission_comment)
                continue

            job_id = matches[0]
            if submission_comment in self._ambiguous_submission_comments:
                self._ambiguous_submission_comments.remove(submission_comment)
            terminal = self._cancel_exact_job_and_confirm_terminal(job_id)
            if terminal:
                self._log(
                    f"Recovered evaluator job {job_id} and confirmed it terminal."
                )
                continue
            if state_persisted or lease_persisted:
                self._log(
                    f"Recovered evaluator job {job_id}; cancellation is not "
                    "terminal-confirmed, so exact ownership remains quarantined."
                )
                continue
            still_unresolved.append(submission_comment)

        self._unresolved_submission_comments = still_unresolved
        self._scheduler_ownership_uncertain = bool(
            self._unresolved_submission_comments
            or self._ambiguous_submission_comments
        )
        self._persist_submission_evidence()
        return not self._unresolved_submission_comments

    def _quarantine_submission(
        self, reason: str, *, ownership_uncertain: bool = False,
    ) -> None:
        self._submission_blocked = True
        self._submission_quarantine_reason = reason
        self._scheduler_ownership_uncertain = (
            self._scheduler_ownership_uncertain or ownership_uncertain
        )
        self._persist_submission_evidence()
        self._log(f"Evaluator submission quarantined: {reason}")
        self._update_status("quarantined")
        if self.enable_service_lifecycle and hasattr(
            self.state_manager, "update_experiment_lifecycle"
        ):
            try:
                self.state_manager.update_experiment_lifecycle(
                    self.experiment_id, lifecycle_state="quarantined",
                )
            except Exception as exc:
                self._log(f"Could not persist lifecycle quarantine: {exc}")

    def _wait_for_exact_job_terminal(self, job_id: str) -> bool:
        for attempt in range(_CANCEL_VERIFY_ATTEMPTS):
            states = get_exact_job_states([job_id], force=True)
            if states is None:
                return False
            if job_id not in states:
                return True
            if attempt + 1 < _CANCEL_VERIFY_ATTEMPTS:
                time.sleep(_CANCEL_VERIFY_INTERVAL_SECONDS)
        return False

    def _cancel_exact_job_and_confirm_terminal(self, job_id: str) -> bool:
        """Cancel one known evaluator allocation with authoritative proof."""
        if (
            job_id not in self._observed_job_ids
            and not self._wait_for_accepted_job_visibility(job_id)
        ):
            return False
        states = get_exact_job_states([job_id], force=True)
        if states is None:
            return False
        if job_id not in states:
            return True
        if not cancel_job(job_id):
            return False
        return self._wait_for_exact_job_terminal(job_id)

    def _wait_for_accepted_job_record(self, job_id: str) -> Optional[Any]:
        """Return one observed record under the existing pending bound.

        Slurm can acknowledge ``sbatch --parsable`` before a subsequent
        accounting query exposes the allocation.  Keep the durable exact-ID
        fence during that propagation window: an empty never-observed query is
        not terminal proof and must not turn into either a replacement job or
        an abandoned allocation.
        """
        fast_attempts = _ACCEPTED_ID_VISIBILITY_ATTEMPTS
        fast_window = max(
            0.0,
            (fast_attempts - 1) * _ACCEPTED_ID_VISIBILITY_INTERVAL_SECONDS,
        )
        slow_interval = _UNRESOLVED_RECONCILE_INTERVAL_SECONDS
        remaining_window = max(
            0.0,
            float(self.evaluator_pending_wait_seconds) - fast_window,
        )
        slow_attempts = int(
            (remaining_window + slow_interval - 1) // slow_interval
        )
        attempts = fast_attempts + slow_attempts
        for attempt in range(attempts):
            states = get_exact_job_states([job_id], force=True)
            if states is not None and job_id in states:
                self._observed_job_ids.add(job_id)
                self._persist_submission_evidence()
                return states[job_id]
            if attempt + 1 < attempts:
                delay = (
                    _ACCEPTED_ID_VISIBILITY_INTERVAL_SECONDS
                    if attempt + 1 < fast_attempts else slow_interval
                )
                time.sleep(delay)
        self._log(
            f"Accepted evaluator job {job_id} never became authoritatively "
            "visible; retaining exact ownership and blocking replacement."
        )
        return None

    def _wait_for_accepted_job_visibility(self, job_id: str) -> bool:
        """Boundedly prove that an accepted exact ID was once visible."""
        return self._wait_for_accepted_job_record(job_id) is not None

    def _quarantine_attestation_failure(
        self, *, job_id: str, submission_comment: str, reason: str,
    ) -> None:
        """Quarantine bad attestation without retaining a proved-dead owner.

        The exact parsed ID and authoritative visibility permit exact
        cancellation.  Scheduler ownership remains uncertain only when that
        cancellation cannot be terminal-confirmed.  When terminality is
        proved, clear the prepared comment fence so managed service leases can
        finalize instead of holding otherwise idle GPU services.
        """
        self._quarantine_submission(reason, ownership_uncertain=False)
        if not self._cancel_exact_job_and_confirm_terminal(job_id):
            self._scheduler_ownership_uncertain = True
            self._persist_submission_evidence()
            self._log(
                "Attestation-failed evaluator allocation could not be "
                "terminal-confirmed; exact ownership remains retained."
            )
            return

        if submission_comment in self._unresolved_submission_comments:
            self._unresolved_submission_comments.remove(submission_comment)
        if submission_comment in self._ambiguous_submission_comments:
            self._ambiguous_submission_comments.remove(submission_comment)
        cleared = self._persist_evaluator_submission_transition(
            None, None, [], observed=False,
        )
        self._scheduler_ownership_uncertain = bool(
            self._unresolved_submission_comments
            or self._ambiguous_submission_comments
        )
        self._persist_submission_evidence()
        if not cleared:
            self._log(
                "Evaluator job is terminal-confirmed, but one durable "
                "submission-fence authority could not be cleared; lease "
                "finalization will retry its independent lifecycle transition."
            )

    def _attest_exact_submission_comment(
        self, job_id: str, submission_comment: str,
        *, observed_record: Optional[Any] = None,
    ) -> bool:
        """Bind an accepted evaluator ID to its exact prepared scheduler tag."""
        info = (
            observed_record
            if observed_record is not None
            else self._wait_for_accepted_job_record(job_id)
        )
        if info is not None:
            actual_comment = (
                str(info.get("comment") or "")
                if isinstance(info, dict) else ""
            )
            if actual_comment != submission_comment:
                self._quarantine_attestation_failure(
                    job_id=job_id,
                    submission_comment=submission_comment,
                    reason=(
                        "accepted evaluator allocation failed exact scheduler-"
                        "comment attestation: "
                        f"job={job_id}, expected={submission_comment!r}, "
                        f"actual={actual_comment!r}"
                    ),
                )
                return False
            if observed_record is not None:
                # Malformed-output recovery already observed this exact ID and
                # comment in one authoritative snapshot.  Preserve that proof
                # rather than opening a second visibility/authority race.
                self._observed_job_ids.add(job_id)
                self._persist_submission_evidence()
            if not self._persist_evaluator_submission_transition(
                submission_comment,
                "observed",
                [job_id],
                observed=True,
            ):
                self._quarantine_attestation_failure(
                    job_id=job_id,
                    submission_comment=submission_comment,
                    reason=(
                        "exact evaluator scheduler-comment attestation could not "
                        f"be persisted; job_id={job_id}"
                    ),
                )
                return False
            return True
        self._quarantine_submission(
            f"accepted evaluator job {job_id} never became visible with its "
            "exact prepared scheduler comment",
            ownership_uncertain=True,
        )
        return False

    def _persist_submitted_job_ownership(self, job_id: str) -> bool:
        state_persisted = self._update_state_jobs()
        heartbeat_persisted = True
        try:
            self._heartbeat_service_lease(force=True, renewable=False)
        except Exception as exc:
            heartbeat_persisted = False
            self._log(f"Failed to heartbeat exact submitted job {job_id}: {exc}")
        if state_persisted and heartbeat_persisted:
            return True

        failures = []
        if not state_persisted:
            failures.append("experiment-state persistence failed")
        if not heartbeat_persisted:
            failures.append("service-lease heartbeat failed")
        terminal = self._cancel_exact_job_and_confirm_terminal(job_id)
        reason = "; ".join(failures)
        if terminal:
            reason += "; exact job is scheduler-confirmed terminal"
        else:
            reason += "; exact job remains owned/unconfirmed and no retry is allowed"
        self._quarantine_submission(reason)
        return False

    def _heartbeat_owned_job_or_quarantine(
        self, job_id: str, *, job_status: str = "UNKNOWN",
    ) -> bool:
        renewable = str(job_status).upper() == "RUNNING"
        try:
            self._heartbeat_service_lease(
                force=renewable,
                renewable=renewable,
                evidence=(
                    "exact_running_evaluator"
                    if renewable else "exact_job_ownership"
                ),
            )
            return True
        except Exception as exc:
            terminal = self._cancel_exact_job_and_confirm_terminal(job_id)
            reason = f"service-lease heartbeat failed for exact job {job_id}: {exc}"
            if terminal:
                reason += "; job is scheduler-confirmed terminal"
            else:
                reason += "; job remains owned/unconfirmed and no retry is allowed"
            self._quarantine_submission(reason)
            return False

    def _retain_blocked_exact_ownership(self) -> None:
        """Reconcile quarantine for a bounded window, then exit safely.

        Exact comments/job IDs and the lease are durable authorities; the
        manager PID is not.  On expiry, keep the quarantine evidence intact,
        submit no replacement, cancel nothing under uncertainty, and let the
        renewable heartbeat TTL suppress future service successors.
        """
        retention_seconds = max(0.0, float(getattr(
            self,
            "submission_quarantine_retention_seconds",
            _QUARANTINE_RETENTION_TIMEOUT_SECONDS,
        )))
        deadline = time.monotonic() + retention_seconds

        def wait_or_expire(delay: float) -> bool:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                persisted = self._persist_submission_evidence()
                self._log(
                    "Bounded evaluator-quarantine retention expired; exiting "
                    "without replacement submission or uncertain cancellation. "
                    f"Durable evidence persisted={persisted}."
                )
                return True
            time.sleep(min(float(delay), remaining))
            return False

        while True:
            if self._unresolved_submission_comments:
                if not self._reconcile_unresolved_submission_comments():
                    self._log(
                        "Accepted evaluator submission remains unresolved; "
                        "retaining manager and retrying."
                    )
                    if wait_or_expire(
                        _UNRESOLVED_RECONCILE_INTERVAL_SECONDS
                    ):
                        return
                    continue

            exact = list(dict.fromkeys(
                str(job_id) for job_id in self.submitted_job_ids if job_id
            ))
            if not exact:
                return
            states = get_exact_job_states(exact, force=True)
            if (
                states is not None
                and not states
                and set(exact).issubset(self._observed_job_ids)
            ):
                return

            state_persisted = self._update_state_jobs()
            lease_persisted = not self._lease_acquired
            if self._lease_acquired and not self._lease_finalized:
                try:
                    self._heartbeat_service_lease(
                        force=True, renewable=False
                    )
                    lease_persisted = True
                except Exception as exc:
                    lease_persisted = False
                    self._log(
                        "Retrying exact lease ownership persistence after "
                        f"heartbeat failure: {exc}"
                    )

            durable = lease_persisted if self._lease_acquired else state_persisted
            if durable:
                return
            self._log(
                "Exact evaluator job remains live or scheduler-unknown without "
                "durable ownership; retaining manager and retrying in 30s."
            )
            if wait_or_expire(30):
                return

    def submit_job(self) -> bool:
        """Submit a SLURM job."""
        self._log(f"Submitting job #{self.job_counter}...")

        prefix = "cot" if self.experiment_type == "cot" else "spatial"
        job_name = f"{prefix}-{self.benchmark}"

        durable_tracking = self._durable_submission_tracking_enabled()
        submission_comment = (
            self._new_submission_comment(self.job_counter)
            if durable_tracking else ""
        )
        script_content = self._generate_sbatch(
            self.job_counter, submission_comment,
        )
        script_path = write_unique_text_artifact(
            self.log_dir,
            identity=(
                f"{job_name}-{self.experiment_id}-generation-{self.job_counter}"
            ),
            suffix=".sbatch",
            content=script_content,
        )

        if not durable_tracking:
            # Preserve the original/default evaluator launch behavior exactly.
            # Durable comment fencing is opt-in because it adds synchronous
            # Lustre writes before sbatch and can turn an evidence-write outage
            # into an experiment submission failure.
            result = subprocess.run(
                ["sbatch", str(script_path)], capture_output=True, text=True,
            )
            if result.returncode != 0:
                self._log(f"Failed to submit job: {result.stderr}")
                return False

            job_id = None
            for word in result.stdout.split():
                if word.isdigit():
                    job_id = word
                    break
            self.current_job_id = job_id
            self.submitted_job_ids.append(job_id)
            self._update_state_jobs()
            self._log(
                f"Job #{self.job_counter} submitted: SLURM ID {job_id}"
            )
            self._log(f"Log: {self.log_dir}/{job_name}_{job_id}.out")
            return True

        # This comment is the only recovery key if the manager dies after
        # Slurm accepts sbatch but before its exact ID reaches local memory.
        # Publish it durably to every ownership authority before submission.
        if submission_comment not in self._unresolved_submission_comments:
            self._unresolved_submission_comments.append(submission_comment)
        if not self._persist_evaluator_submission_transition(
            submission_comment, "prepared", [], observed=False,
        ):
            self._quarantine_submission(
                "evaluator pre-sbatch ownership intent could not be durably "
                f"published; scheduler comment={submission_comment}",
                ownership_uncertain=True,
            )
            return False

        try:
            result = subprocess.run(
                ["sbatch", "--parsable", str(script_path)],
                capture_output=True,
                text=True,
            )
        except BaseException as exc:
            # The client may have raised after the RPC was accepted.  Keep the
            # prepared comment fence; replacement is unsafe until reconciled.
            self._quarantine_submission(
                "sbatch outcome is unknown after durable prepared intent: "
                f"{type(exc).__name__}: {exc}; comment={submission_comment}",
                ownership_uncertain=True,
            )
            return False

        if result.returncode != 0:
            self._log(f"Failed to submit job: {result.stderr}")
            # A synchronous rejected sbatch is the sole path allowed to clear
            # an unaccepted prepared intent without scheduler observation.
            if not self._persist_evaluator_submission_transition(
                None, None, [], observed=False,
            ):
                self._quarantine_submission(
                    "sbatch rejected the request but its prepared ownership "
                    "fence could not be durably cleared"
                )
                return False
            self._unresolved_submission_comments.remove(submission_comment)
            self._persist_submission_evidence()
            return False

        job_id = self._parse_parsable_job_id(result.stdout)
        if job_id is not None:
            self.current_job_id = job_id
            if job_id not in self.submitted_job_ids:
                self.submitted_job_ids.append(job_id)
            if not self._persist_evaluator_submission_transition(
                submission_comment,
                "accepted_exact",
                [job_id],
                observed=False,
            ):
                self._quarantine_submission(
                    "accepted evaluator ID could not be durably adopted; "
                    f"job_id={job_id}, comment={submission_comment}",
                    ownership_uncertain=True,
                )
                return False
        else:
            if not self._persist_evaluator_submission_transition(
                submission_comment,
                "accepted_unseen",
                [],
                observed=False,
            ):
                self._quarantine_submission(
                    "successful sbatch acknowledgement could not durably "
                    f"advance its comment fence; comment={submission_comment}",
                    ownership_uncertain=True,
                )
                return False
        recovered_record = None
        if job_id is None:
            recovered = self._recover_submitted_job_by_comment(
                submission_comment,
            )
            if recovered is not None:
                job_id, recovered_record = recovered
        if job_id is None:
            self._persist_evaluator_submission_transition(
                submission_comment,
                "accepted_unresolved",
                [],
                observed=False,
            )
            self._quarantine_submission(
                "sbatch returned success without a recoverable exact job ID; "
                f"scheduler comment={submission_comment}",
                ownership_uncertain=True,
            )
            return False

        self.current_job_id = job_id
        if job_id not in self.submitted_job_ids:
            self.submitted_job_ids.append(job_id)
        if not self._attest_exact_submission_comment(
            job_id, submission_comment, observed_record=recovered_record,
        ):
            return False
        if submission_comment in self._unresolved_submission_comments:
            self._unresolved_submission_comments.remove(submission_comment)
        self._persist_submission_evidence()
        if not self._persist_submitted_job_ownership(job_id):
            return False

        self._log(f"Job #{self.job_counter} submitted: SLURM ID {job_id}")
        self._log(f"Log: {self.log_dir}/{job_name}_{job_id}.out")

        return True

    def _cancel_tracked_jobs_and_confirm_terminal(self) -> bool:
        """Stop exact evaluator jobs without inferring absence on uncertainty."""
        if self._scheduler_ownership_uncertain:
            return False
        exact = list(dict.fromkeys(
            str(job_id) for job_id in self.submitted_job_ids if job_id
        ))
        for job_id in exact:
            if not self._cancel_exact_job_and_confirm_terminal(job_id):
                return False
        final_states = get_exact_job_states(exact, force=True)
        return final_states is not None and not final_states

    def handle_shutdown(self, signum, frame):
        """Signal-safe intent only; normal control flow performs Slurm I/O."""
        self.running = False
        self._stop_event.set()

    def _update_state_jobs(self) -> bool:
        if any(not isinstance(job_id, str) or not job_id for job_id in self.submitted_job_ids):
            self._log("Warning: refusing to persist an invalid evaluator job ID")
            return False
        try:
            self.state_manager.update_experiment_jobs(
                self.experiment_id, list(self.submitted_job_ids),
            )
            persisted = next(
                (
                    experiment for experiment in self.state_manager.list_experiments()
                    if experiment.experiment_id == self.experiment_id
                ),
                None,
            )
            expected = set(self.submitted_job_ids)
            if persisted is None or not expected.issubset(
                set(str(job_id) for job_id in persisted.slurm_job_ids)
            ):
                self._log(
                    "Warning: evaluator job ownership write was not durably "
                    "observable"
                )
                return False
            return True
        except Exception as e:
            self._log(f"Warning: failed to update state: {e}")
            return False

    def _update_status(self, status: str):
        try:
            self.state_manager.update_experiment_status(self.experiment_id, status)
        except Exception as e:
            self._log(f"Warning: failed to update status: {e}")

    def _wait_until_scheduled(self) -> bool:
        """Block until self.scheduled_for (ISO) is reached. Returns False if stopped."""
        if not self.scheduled_for:
            return True
        try:
            target = datetime.datetime.fromisoformat(self.scheduled_for).timestamp()
        except ValueError:
            self._log(f"Warning: unparseable scheduled_for={self.scheduled_for!r}, starting now.")
            return True

        remaining = target - time.time()
        if remaining <= 0:
            return True

        self._log(f"Scheduled for {self.scheduled_for} (in {int(remaining // 60)}m {int(remaining % 60)}s).")
        while self.running:
            remaining = target - time.time()
            if remaining <= 0:
                break
            if remaining > 60:
                self._log(f"Waiting for scheduled start: {int(remaining // 60)}m remaining.")
            if self._stop_event.wait(timeout=min(60.0, remaining)):
                return False
        if not self.running:
            return False
        self._log("Scheduled time reached — starting chain.")
        return True

    def run(self):
        """Run the continuous job submission loop."""
        if not check_slurm_available():
            self._log("ERROR: SLURM not available")
            sys.exit(1)

        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        type_label = "CoT baseline" if self.experiment_type == "cot" else "Agent"
        self._log(f"Starting {type_label} experiment chain: {self.benchmark} ({self.experiment_name})")
        self._log(f"Experiment ID: {self.experiment_id}")
        self._log(f"Model: {self.model_name}")
        self._log(f"Work dir: {self.work_dir}")
        self._log(f"Concurrency: {self.concurrency}, Subsample: {self.subsample or 'all'}")
        self._log(f"Job duration: {self.time_limit} (sequential, no overlap)")
        if self.experiment_type == "cot":
            self._log(f"Max frames: {self.max_frames}, System prompt: {self.system_prompt}")

        if not self._wait_until_scheduled():
            self._log("Chain manager stopped before scheduled start.")
            return

        try:
            self._establish_run_policy()
        except LifecycleError as exc:
            self._log(f"ERROR: run-policy provenance failed closed: {exc}")
            self._update_status("quarantined")
            return

        # Verify immutable result provenance before accepting even a fully
        # completed resumed work directory. For default-off legacy profiles,
        # _establish_run_policy is inert and historical behavior is unchanged.
        if self._is_experiment_complete():
            self._update_status("completed")
            self._log("Experiment was already complete; no service lease needed.")
            return

        try:
            self._prepare_service_epoch()
        except LifecycleError as exc:
            self._log(f"ERROR: service readiness failed: {exc}")
            self._update_status("failed")
            self._finalize_service_lease("readiness_failed")
            return

        consecutive_no_progress = 0
        terminal_status: Optional[str] = None

        while self.running:
            # Check if experiment is already complete
            if self._is_experiment_complete():
                self._update_status("completed")
                terminal_status = "completed"
                completed = self._count_predictions()
                total = self._get_total_samples()
                self._log(f"All samples completed! ({completed}/{total})")
                break

            predictions_before = self._count_predictions()

            # Try overlay first (agent type only). On success the agent ran
            # to completion as an `srun --overlap` step inside an existing
            # vLLM/gpu_server SLURM job — no new sbatch, no wait phases.
            # On failure (no slot within backoff window) fall back to the
            # existing sbatch path.
            overlay_dispatched = False
            job_completed = False
            overlays_disabled = os.getenv(
                "SPATIAL_AGENT_DISABLE_OVERLAY", ""
            ).strip().lower() in {"1", "true", "yes"}
            if self.experiment_type == "agent" and not overlays_disabled:
                self._heartbeat_service_lease(force=True)
                overlay_args = self._build_agent_script_args()
                if self.enable_service_lifecycle:
                    dispatched = self._dispatch_overlay_with_lease_fence(
                        overlay_args
                    )
                else:
                    dispatched = agent_dispatcher.try_dispatch_overlay(
                        self.project_root,
                        str(self.agent_script),
                        overlay_args,
                        self.concurrency,
                        log_fn=self._log,
                        stop_event=self._stop_event,
                    )
                if dispatched:
                    overlay_dispatched = True
                    job_completed = True
                self._heartbeat_service_lease(force=True)

            # If overlay returned False because stop_event was set during the
            # call, don't bother sbatching — we're shutting down.
            if not overlay_dispatched and not self.running:
                break

            if not overlay_dispatched:
                # Submit next job. Submission failures count toward the same
                # no-progress budget as jobs that produce zero predictions —
                # otherwise a bad account/partition would spin forever.
                if not self.submit_job():
                    if self._submission_blocked:
                        terminal_status = "quarantined"
                        break
                    consecutive_no_progress += 1
                    self._log(
                        f"Failed to submit "
                        f"({consecutive_no_progress}/{_MAX_NO_PROGRESS_JOBS} consecutive)."
                    )
                    if consecutive_no_progress >= _MAX_NO_PROGRESS_JOBS:
                        self._log(
                            f"Aborting chain: {_MAX_NO_PROGRESS_JOBS} consecutive "
                            f"submission failures. Check your account/partition/script."
                        )
                        self._update_status("failed")
                        terminal_status = "failed"
                        break
                    if self._stop_event.wait(timeout=60):
                        break
                    continue

                # Wait for job to start. Same policy — a job that never runs
                # can't make progress, so it counts against the budget.
                if not self._wait_for_job_to_start(self.current_job_id):
                    if self._submission_blocked:
                        terminal_status = "quarantined"
                        break
                    if self._queue_deferred_retry_required:
                        terminal_status = "queue_deferred_retry_required"
                        break
                    if not self.running:
                        break
                    consecutive_no_progress += 1
                    self._log(
                        f"Job failed to start "
                        f"({consecutive_no_progress}/{_MAX_NO_PROGRESS_JOBS} consecutive)."
                    )
                    if not self._cancel_exact_job_and_confirm_terminal(
                        self.current_job_id
                    ):
                        self._quarantine_submission(
                            "failed-to-start exact evaluator job could not be "
                            "scheduler-confirmed terminal; replacement blocked"
                        )
                        terminal_status = "quarantined"
                        break
                    self._prune_submitted_jobs()
                    if consecutive_no_progress >= _MAX_NO_PROGRESS_JOBS:
                        self._log(
                            f"Aborting chain: {_MAX_NO_PROGRESS_JOBS} consecutive jobs "
                            f"failed to start. Check the latest job log."
                        )
                        self._update_status("failed")
                        terminal_status = "failed"
                        break
                    if self._stop_event.wait(timeout=60):
                        break
                    continue

                # Wait for job to complete (sequential, no overlap)
                job_completed = self._wait_for_job_completion(self.current_job_id)

                if self._submission_blocked:
                    terminal_status = "quarantined"
                    break

                if not job_completed:
                    self._log("Job ended abnormally, retrying in 60s...")

            # Detect a stuck experiment: job ended (success or fail) but
            # produced no new predictions. Most often a dataloader / startup
            # error that fails fast and would loop forever otherwise.
            predictions_after = self._count_predictions()
            progress = predictions_after - predictions_before
            if progress <= 0:
                consecutive_no_progress += 1
                self._log(
                    f"Job made no progress "
                    f"({consecutive_no_progress}/{_MAX_NO_PROGRESS_JOBS} consecutive)."
                )
                if consecutive_no_progress >= _MAX_NO_PROGRESS_JOBS:
                    self._log(
                        f"Aborting chain: {_MAX_NO_PROGRESS_JOBS} consecutive jobs "
                        f"produced no new predictions. Check the latest job log "
                        f"for the underlying error."
                    )
                    self._update_status("failed")
                    terminal_status = "failed"
                    break
            else:
                consecutive_no_progress = 0

            # Prune the submitted-job list so we don't keep thousands of
            # finished IDs (used to be cancelled redundantly on stop and
            # to bloat dashboard squeue queries).
            self._prune_submitted_jobs()

            if not job_completed:
                if self._stop_event.wait(timeout=60):
                    break

            self.job_counter += 1

            if self.running:
                completed = predictions_after
                total = self._get_total_samples()
                if total > 0:
                    self._log(f"Progress: {completed}/{total} — submitting next job...")
                else:
                    self._log(f"Completed: {completed} — submitting next job...")

        if terminal_status == "quarantined":
            self._retain_blocked_exact_ownership()

        # Only remove from state on abnormal/in-progress exit. Keep
        # completed/failed entries so they're visible in the dashboard.
        if terminal_status is None:
            if self._cancel_tracked_jobs_and_confirm_terminal():
                try:
                    self.state_manager.remove_experiment(self.experiment_id)
                except Exception:
                    pass
            else:
                terminal_status = "quarantined"
                self._quarantine_submission(
                    "shutdown retained exact evaluator ownership because "
                    "cancellation or terminal scheduler proof was unavailable",
                    ownership_uncertain=self._scheduler_ownership_uncertain,
                )

        self._finalize_service_lease(terminal_status or "stopped")

        self._log("Chain manager stopped.")

    def _prune_submitted_jobs(self) -> None:
        """Drop finished job IDs from the tracked list to keep it small.

        Uses one batched squeue call instead of N serial is_job_alive() calls.
        """
        alive = filter_alive_jobs(self.submitted_job_ids)
        if len(alive) != len(self.submitted_job_ids):
            self.submitted_job_ids = alive
            self._update_state_jobs()


def start_experiment_background(
    experiment_id: str,
    benchmark: str,
    model_name: str,
    experiment_name: str,
    account: str,
    partition: str,
    gpus: int,
    time_limit: str,
    concurrency: int,
    subsample: int,
    project_root: Path,
    experiment_type: str = "agent",
    max_frames: int = 0,
    system_prompt: str = "cot",
    defer_minutes: int = 0,
    enable_service_lifecycle: bool = False,
    service_readiness_timeout_minutes: int = 240,
    service_running_readiness_timeout_minutes: int = 15,
    evaluator_pending_wait_minutes: int = 15,
    service_renewal_heartbeat_ttl_minutes: int = 5,
    service_finalize_timeout_minutes: int = 5,
    generation_seed: Optional[int] = None,
) -> Tuple[ExperimentState, Path]:
    """Start an experiment chain as an independent background process."""
    if defer_minutes > 0:
        scheduled_for = (
            datetime.datetime.now() + datetime.timedelta(minutes=defer_minutes)
        ).isoformat()
    else:
        scheduled_for = ""
    lease_id = uuid.uuid4().hex if enable_service_lifecycle else ""
    cohort_id = f"cohort-{experiment_id}" if enable_service_lifecycle else ""

    config = {
        "experiment_id": experiment_id,
        "benchmark": benchmark,
        "model_name": model_name,
        "experiment_name": experiment_name,
        "account": account,
        "partition": partition,
        "gpus": gpus,
        "time_limit": time_limit,
        "concurrency": concurrency,
        "subsample": subsample,
        "project_root": str(project_root),
        "experiment_type": experiment_type,
        "max_frames": max_frames,
        "system_prompt": system_prompt,
        "scheduled_for": scheduled_for,
        "enable_service_lifecycle": bool(enable_service_lifecycle),
        "service_readiness_timeout_minutes": int(
            service_readiness_timeout_minutes
        ),
        "service_running_readiness_timeout_minutes": int(
            service_running_readiness_timeout_minutes
        ),
        "evaluator_pending_wait_minutes": int(
            evaluator_pending_wait_minutes
        ),
        "service_renewal_heartbeat_ttl_minutes": int(
            service_renewal_heartbeat_ttl_minutes
        ),
        "service_finalize_timeout_minutes": int(
            service_finalize_timeout_minutes
        ),
        "lease_id": lease_id,
        "cohort_id": cohort_id,
        "generation_seed": generation_seed,
    }

    # Derive work_dir (same logic as ExperimentChain._derive_work_dir)
    model_config_path = project_root / f"spatial_agent/config/model/{model_name}.json"
    model_short = "unknown"
    if model_config_path.exists():
        try:
            with open(model_config_path) as f:
                cfg = json.load(f)
            model_short = (
                cfg.get("model_profile")
                or cfg["llm_model"].split("/")[-1][:30]
            )
        except Exception:
            pass
    prefix = "cot" if experiment_type == "cot" else "spatial"
    pkg_dir = project_root / "spatial_agent"
    work_dir = str(pkg_dir / "work_dir" / f"{prefix}_{benchmark}_{model_short}")
    if experiment_name:
        work_dir = f"{work_dir}_{experiment_name}"

    # Compute total_samples eagerly (CLI has access to benchmark data)
    total_samples = -1
    try:
        from spatial_agent.evals.factory import BenchmarkFactory
        bench = BenchmarkFactory.create_benchmark(benchmark)
        if bench is not None:
            total_samples = len(bench)
            if subsample > 0:
                total_samples = min(total_samples, subsample)
    except Exception:
        # Fall back to subsample as best guess
        if subsample > 0:
            total_samples = subsample

    # Log file for the chain process
    log_subdir = "slurm_cot" if experiment_type == "cot" else "slurm_agent"
    log_dir = project_root / "spatial_agent" / "logs" / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    # The child cannot reserve services or submit evaluator work until the
    # parent has durably persisted its ExperimentState. EOF means the parent
    # failed/crashed; only an explicit byte opens the startup gate.
    gate_read_fd, gate_write_fd = os.pipe()
    config["_startup_gate_fd"] = gate_read_fd
    chain_log, log_f = open_unique_text_artifact(
        log_dir,
        identity=(
            f"chain-{experiment_type}-{benchmark}-{experiment_name}-"
            f"{experiment_id}"
        ),
        suffix=".log",
    )
    proc = None
    try:
        # stdin closed so the chain (and any srun it spawns) can't steal
        # keystrokes from the interactive agent-manager CLI.
        proc = subprocess.Popen(
            [
                sys.executable, "-m",
                "spatial_agent.launch_managers.agent_manager.experiment_chain",
                "--config", json.dumps(config),
            ],
            stdin=subprocess.DEVNULL,
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
    if not process_identity_is_valid(manager_identity, expected_pid=proc.pid):
        # The child is still blocked behind the inherited startup gate, so it
        # cannot have reserved services or submitted evaluator work.  Refuse
        # to create a new identity-less authority record.
        os.close(gate_write_fd)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        log_f.close()
        raise RuntimeError(
            "Could not capture exact experiment manager process identity"
        )

    started_at = datetime.datetime.now().isoformat()
    exp_state = ExperimentState(
        experiment_id=experiment_id,
        benchmark=benchmark,
        model_name=model_name,
        experiment_name=experiment_name,
        pid=proc.pid,
        process_identity=manager_identity,
        slurm_job_ids=[],
        started_at=started_at,
        account=account,
        partition=partition,
        gpus=gpus,
        concurrency=concurrency,
        subsample=subsample,
        work_dir=work_dir,
        total_samples=total_samples,
        status="running",
        experiment_type=experiment_type,
        scheduled_for=scheduled_for,
        lease_id=lease_id or None,
        cohort_id=cohort_id or None,
        lifecycle_state=("starting" if enable_service_lifecycle else None),
    )

    state_manager = ExperimentStateManager(project_root)
    gate_release_attempted = False
    try:
        state_manager.add_experiment(exp_state)
        # An asynchronous exception can race the byte after it reaches the
        # detached child.  From this point onward the child may already have
        # acquired a service lease or submitted evaluator work, so deleting
        # its durable manager record would erase the only ownership fence.
        gate_release_attempted = True
        if os.write(gate_write_fd, b"1") != 1:
            raise RuntimeError(
                "Could not release persisted experiment startup gate"
            )
    except BaseException:
        # Never leave a detached manager able to submit untracked work when
        # initial authority persistence fails.
        if not gate_release_attempted:
            try:
                state_manager.remove_experiment(experiment_id)
            except Exception:
                pass
        else:
            # Retain the persisted identity even if supplementary status
            # updates fail.  Reconciliation must prove the child, lease, and
            # exact scheduler ownership terminal before removing it.
            try:
                state_manager.update_experiment_status(
                    experiment_id, "quarantined"
                )
            except Exception:
                pass
            if enable_service_lifecycle:
                try:
                    state_manager.update_experiment_lifecycle(
                        experiment_id, lifecycle_state="quarantined",
                    )
                except Exception:
                    pass
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
        raise
    finally:
        os.close(gate_write_fd)
        log_f.close()

    return exp_state, chain_log


def main():
    parser = argparse.ArgumentParser(description="Experiment chain subprocess")
    parser.add_argument("--config", type=str, required=True, help="JSON config string")
    args = parser.parse_args()

    config = json.loads(args.config)
    startup_gate_fd = config.pop("_startup_gate_fd", None)
    if startup_gate_fd is not None:
        try:
            gate_value = os.read(int(startup_gate_fd), 1)
        finally:
            os.close(int(startup_gate_fd))
        if gate_value != b"1":
            raise RuntimeError(
                "Parent exited before experiment ownership was durably persisted"
            )
    chain = ExperimentChain(**config)
    try:
        chain.run()
    except BaseException:
        # Preserve a draining ownership fence on unexpected manager failure;
        # reconciliation may release it only after jobs and overlay steps are
        # authoritatively absent.
        try:
            chain._finalize_service_lease("crashed")
        except Exception as cleanup_error:
            chain._log(f"Lifecycle crash cleanup failed closed: {cleanup_error}")
        raise


if __name__ == "__main__":
    main()
