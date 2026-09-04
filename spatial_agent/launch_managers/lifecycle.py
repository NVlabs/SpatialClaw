"""Crash-safe service ownership and readiness state for SpatialClaw.

This module deliberately contains *resource lifecycle* metadata only.  It does
not modify prompts, decoding arguments, benchmark inputs, or scoring.  Runtime
records live under ``spatial_agent/logs`` (which is ignored by Git); callers
may persist a frozen manifest in an experiment output directory when they need
an auditable service epoch.

The safety rule is fail closed: an expired heartbeat is evidence to
investigate, never sufficient permission to stop a service.  A lease becomes
non-owning only after an explicit terminal/released transition or a successful
reconciliation proving that its owner process and all recorded Slurm jobs are
gone.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
import socket
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, TextIO, Tuple


SCHEMA_VERSION = 1
ACTIVE_LEASE_STATES = frozenset({"reserved", "active", "draining"})
TERMINAL_LEASE_STATES = frozenset({"terminal", "released", "orphaned"})
READY_ENDPOINT_STATES = frozenset({"ready", "accepting"})
LAUNCH_ELIGIBLE_POOL_STATES = frozenset({"launching", "ready"})
DEFAULT_RENEWAL_HEARTBEAT_TTL_SECONDS = 5 * 60


class LifecycleError(RuntimeError):
    """Lifecycle state is incompatible, incomplete, or unsafe to use."""


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _parse_utc_timestamp(value: Any) -> Optional[_dt.datetime]:
    """Parse one lifecycle timestamp without treating corruption as fresh."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def canonical_digest(value: Any) -> str:
    """SHA-256 of a stable, compact JSON representation."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(
    path: os.PathLike[str] | str,
    value: Any,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Durably replace a JSON authority while an external lock is held.

    Readers of service registries are intentionally lockless. A same-directory
    temp file plus ``os.replace`` prevents them from observing a truncated
    drain/restore update; fsyncing both file and parent directory makes the
    ownership transition durable before returning.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                value, handle, indent=indent, ensure_ascii=ensure_ascii,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(destination.parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def component_digests(
    *,
    server_spec: Optional[Dict[str, Any]] = None,
    tool_spec: Optional[Dict[str, Any]] = None,
    workload_policy: Optional[Dict[str, Any]] = None,
    topology: Optional[Dict[str, Any]] = None,
    lifecycle_policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return independent digests instead of one ambiguous source-tree hash."""
    components = {
        "server_spec": server_spec,
        "tool_spec": tool_spec,
        "workload_policy": workload_policy,
        "topology": topology,
        "lifecycle_policy": lifecycle_policy,
    }
    return {
        name: canonical_digest(value)
        for name, value in components.items()
        if value is not None
    }


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        # EPERM proves neither death nor ownership loss.  Fail closed.
        return True
    except (ProcessLookupError, OSError, ValueError):
        return False


def _process_identity(pid: int) -> Optional[Dict[str, Any]]:
    """Best-effort host/namespace/boot/start identity for one local PID."""
    try:
        stat_text = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        # comm is parenthesized and may contain spaces; fields after the final
        # ')' resume at proc(5) field 3. starttime is field 22 => index 19 here.
        after_comm = stat_text.rsplit(")", 1)[1].strip().split()
        start_ticks = int(after_comm[19])
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
        pid_namespace = os.readlink(f"/proc/{int(pid)}/ns/pid")
        return {
            "pid": int(pid),
            "start_ticks": start_ticks,
            "boot_id": boot_id,
            "hostname": socket.gethostname(),
            "pid_namespace": pid_namespace,
        }
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        return None


def process_identity(pid: int) -> Optional[Dict[str, Any]]:
    """Return a stable local process identity when the kernel exposes one."""
    return _process_identity(pid)


def process_pid_is_valid(pid: Any) -> bool:
    """Return whether a persisted PID is an exact positive integer."""
    return type(pid) is int and pid > 0


def process_identity_is_valid(
    identity: Any, *, expected_pid: Optional[int] = None,
) -> bool:
    """Validate the complete kernel identity used as PID-reuse authority.

    A merely JSON-shaped mapping is not proof of process identity.  Treat
    missing identities as conservative legacy/uncertain state and malformed
    identities as corrupt authority; neither may prove PID reuse or authorize
    a signal/cancellation.
    """
    if not isinstance(identity, dict) or set(identity) != {
        "pid", "start_ticks", "boot_id", "hostname", "pid_namespace",
    }:
        return False
    identity_pid = identity.get("pid")
    start_ticks = identity.get("start_ticks")
    boot_id = identity.get("boot_id")
    hostname = identity.get("hostname")
    pid_namespace = identity.get("pid_namespace")
    if (
        not process_pid_is_valid(identity_pid)
        or isinstance(start_ticks, bool)
        or not isinstance(start_ticks, int)
        or start_ticks < 0
        or not isinstance(boot_id, str)
        or not boot_id.strip()
        or not isinstance(hostname, str)
        or not hostname.strip()
        or not isinstance(pid_namespace, str)
        or not pid_namespace.strip()
    ):
        return False
    if expected_pid is not None:
        if (
            not process_pid_is_valid(expected_pid)
            or identity_pid != expected_pid
        ):
            return False
    return True


def process_identity_record_is_recognized(
    identity: Any, *, expected_pid: Optional[int] = None,
) -> bool:
    """Accept an exact identity or the former 3-field record schema.

    Legacy records remain readable so historical state does not brick the
    control plane, but :func:`process_identity_is_valid` deliberately rejects
    them.  They therefore retain ownership as uncertainty and can never
    authorize a signal, dead-owner recovery, or a new ownership mutation.
    """
    if process_identity_is_valid(identity, expected_pid=expected_pid):
        return True
    if not isinstance(identity, dict) or set(identity) != {
        "pid", "start_ticks", "boot_id",
    }:
        return False
    identity_pid = identity.get("pid")
    start_ticks = identity.get("start_ticks")
    boot_id = identity.get("boot_id")
    return bool(
        process_pid_is_valid(identity_pid)
        and not isinstance(start_ticks, bool)
        and isinstance(start_ticks, int)
        and start_ticks >= 0
        and isinstance(boot_id, str)
        and boot_id.strip()
        and (
            expected_pid is None
            or (
                process_pid_is_valid(expected_pid)
                and identity_pid == expected_pid
            )
        )
    )


def process_identity_scope_is_local(identity: Mapping[str, Any]) -> bool:
    """Whether this process may authoritatively inspect the recorded PID."""
    if not process_identity_is_valid(identity):
        return False
    try:
        local_namespace = os.readlink("/proc/self/ns/pid")
    except OSError:
        return False
    return (
        identity.get("hostname") == socket.gethostname()
        and identity.get("pid_namespace") == local_namespace
    )


def compare_process_identities(
    recorded_identity: Any, current_identity: Any,
) -> Optional[bool]:
    """Compare identities without calling a different host/namespace dead.

    ``True`` is an exact owner match, ``False`` is reuse/reboot proven within
    the same host and PID namespace, and ``None`` is uncertainty.
    """
    if (
        not process_identity_is_valid(recorded_identity)
        or not process_identity_is_valid(current_identity)
    ):
        return None
    recorded_scope = (
        recorded_identity.get("hostname"),
        recorded_identity.get("pid_namespace"),
    )
    current_scope = (
        current_identity.get("hostname"),
        current_identity.get("pid_namespace"),
    )
    if recorded_scope != current_scope:
        return None
    return recorded_identity == current_identity


def signal_process_record(
    pid: int, recorded_identity: Optional[Dict[str, Any]], signal_number: int,
) -> bool:
    """Signal only a freshly reverified exact process identity.

    This interpreter lacks ``os.pidfd_open``, so identity verification and
    signaling cannot be one atomic kernel operation.  Rechecking immediately
    on the straight-line signal path reduces that residual sub-instruction
    PID-reuse race.  Any missing, malformed, unreadable, or changed identity
    fails closed without signaling.
    """
    if not process_identity_is_valid(recorded_identity, expected_pid=pid):
        return False
    current_identity = _process_identity(pid)
    if current_identity is None or current_identity != recorded_identity:
        return False
    try:
        os.kill(int(pid), int(signal_number))
    except (PermissionError, ProcessLookupError, OSError, TypeError, ValueError):
        return False
    return True


def process_record_alive(
    pid: int, recorded_identity: Optional[Dict[str, Any]] = None,
) -> bool:
    """Conservatively decide whether ``pid`` can still own a record.

    A start-identity mismatch is affirmative evidence of PID reuse.  Failure to
    read ``/proc`` while the numeric PID exists is uncertainty and therefore
    retains ownership.
    """
    if not process_pid_is_valid(pid):
        return True
    if not process_identity_is_valid(recorded_identity, expected_pid=pid):
        return True
    if not process_identity_scope_is_local(recorded_identity):
        return True
    if not _pid_alive(pid):
        return False
    current_identity = _process_identity(pid)
    comparison = compare_process_identities(recorded_identity, current_identity)
    return comparison is not False


def _dead_process_proof(
    pid: Any, recorded_identity: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return affirmative local death/PID-reuse evidence, never uncertainty."""
    if not process_pid_is_valid(pid):
        return None
    numeric_pid = pid
    identity = copy.deepcopy(dict(recorded_identity or {})) or None
    if not process_identity_is_valid(identity, expected_pid=numeric_pid):
        return None
    if not process_identity_scope_is_local(identity):
        return None
    if not _pid_alive(numeric_pid):
        return {
            "kind": "pid_absent",
            "pid": numeric_pid,
            "recorded_identity": identity,
        }
    current_identity = _process_identity(numeric_pid)
    comparison = compare_process_identities(identity, current_identity)
    if comparison is not False:
        return None
    return {
        "kind": "pid_reused",
        "pid": numeric_pid,
        "recorded_identity": identity,
        "current_identity": copy.deepcopy(current_identity),
    }


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self.handle

    def __exit__(self, *exc):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class LifecycleStore:
    """Lock-protected owner leases and desired service pools."""

    def __init__(self, project_root: os.PathLike[str] | str):
        self.project_root = Path(project_root).resolve()
        self.state_dir = self.project_root / "spatial_agent" / "logs"
        self.state_file = self.state_dir / "service_lifecycle.json"
        self.lock_file = Path(str(self.state_file) + ".lock")

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now(),
            "pools": {},
            "leases": {},
            "events": [],
        }

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.state_file.exists():
            return self._empty()
        try:
            with open(self.state_file, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except FileNotFoundError:
            # A first-use file may be absent.  If it disappeared after the
            # existence check, treat that race as authority uncertainty.
            raise LifecycleError(
                f"Lifecycle authority disappeared while reading {self.state_file}"
            )
        except json.JSONDecodeError as exc:
            raise LifecycleError(
                f"Lifecycle authority is corrupt: {self.state_file}: {exc}"
            ) from exc
        except OSError as exc:
            raise LifecycleError(
                f"Lifecycle authority is unreadable: {self.state_file}: {exc}"
            ) from exc
        if not isinstance(state, dict):
            raise LifecycleError(
                f"Lifecycle authority must contain a JSON object: {self.state_file}"
            )
        if state.get("schema_version") != SCHEMA_VERSION:
            raise LifecycleError(
                "Unsupported service_lifecycle.json schema: "
                f"{state.get('schema_version')!r}"
            )
        for field, expected_type, default in (
            ("pools", dict, {}),
            ("leases", dict, {}),
            ("events", list, []),
        ):
            if field not in state:
                state[field] = default
            elif not isinstance(state[field], expected_type):
                raise LifecycleError(
                    f"Lifecycle authority field {field!r} has invalid shape: "
                    f"expected {expected_type.__name__}"
                )
        _validate_reconciliation_lifecycle_shape(state)
        return state

    def _write_unlocked(self, state: Dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = utc_now()
        fd, temporary = tempfile.mkstemp(
            prefix=".service_lifecycle.", suffix=".tmp", dir=self.state_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_file)
            # fsyncing the file is insufficient to make the directory entry
            # durable across a host crash.  Persist the rename before callers
            # are told that an ownership transition succeeded.
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_fd = os.open(self.state_dir, directory_flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def _mutate(self, callback: Callable[[Dict[str, Any]], Any]) -> Any:
        with _FileLock(self.lock_file):
            state = self._read_unlocked()
            result = callback(state)
            # Keep the audit trail bounded; pool/lease records retain current
            # state and the latest 2,000 transitions are ample for diagnosis.
            state["events"] = state.get("events", [])[-2000:]
            self._write_unlocked(state)
            return result

    @staticmethod
    def _event(state: Dict[str, Any], kind: str, **details: Any) -> None:
        state.setdefault("events", []).append(
            {"at": utc_now(), "kind": kind, **details}
        )

    def snapshot(self) -> Dict[str, Any]:
        with _FileLock(self.lock_file):
            return copy.deepcopy(self._read_unlocked())

    def service_identity_launch_lock(
        self, service_kind: str, identity_value: str,
    ) -> _FileLock:
        """Serialize launch decisions for one logical service identity.

        Pool IDs and specification digests are deliberately absent from the
        lock key.  Two CLIs requesting different quantization/topology/desired
        totals for the same served model or tool backend must still serialize
        before either can create capacity.
        """
        if service_kind not in {"vllm", "gpu_tools"}:
            raise ValueError(f"Unsupported service_kind={service_kind!r}")
        identity = str(identity_value or "")
        if not identity:
            raise ValueError("Service identity must not be empty")
        token = canonical_digest({
            "schema_version": 1,
            "service_kind": service_kind,
            "identity_value": identity,
        })
        return _FileLock(
            self.state_dir / "service_identity_locks" / f"{service_kind}-{token}.lock"
        )

    @staticmethod
    def _pool_identity_value(pool: Mapping[str, Any]) -> str:
        kind = pool.get("service_kind")
        if kind == "vllm":
            return str((pool.get("server_spec") or {}).get("served_name") or "")
        if kind == "gpu_tools":
            return str(
                (pool.get("tool_spec") or {}).get("reconstruct_backend") or ""
            )
        return ""

    @classmethod
    def _reap_stale_launch_claims(
        cls,
        state: Dict[str, Any],
        pool_id: str,
        pool: Dict[str, Any],
    ) -> List[str]:
        """Release claims only with affirmative PID-death/reuse evidence."""
        slots = cls._ensure_slots(pool)
        released: List[str] = []
        claims = pool.setdefault("launch_claims", {})
        for claimant_id, claim in list(claims.items()):
            proof = _dead_process_proof(
                claim.get("pid"), claim.get("process_identity"),
            )
            if proof is None:
                continue
            for slot_id in claim.get("slots", []) or []:
                slot = slots.get(slot_id)
                if (
                    slot
                    and slot.get("status") == "claimed"
                    and slot.get("claimant_id") == claimant_id
                ):
                    slots[slot_id] = {
                        "slot_id": slot_id,
                        "status": "available",
                        "updated_at": utc_now(),
                    }
            claims.pop(claimant_id, None)
            released.append(str(claimant_id))
            cls._event(
                state,
                "stale_launch_claim_released",
                pool_id=pool_id,
                claimant_id=claimant_id,
                process_evidence=proof,
            )
        return released

    @classmethod
    def _pool_durable_launch_reasons(
        cls, state: Mapping[str, Any], pool: Mapping[str, Any],
    ) -> List[str]:
        """Describe authority that forbids superseding one same-identity pool."""
        reasons: List[str] = []
        chains = pool.get("chains") or {}
        if chains:
            reasons.append(f"chains={','.join(sorted(map(str, chains)))}")
        claims = pool.get("launch_claims") or {}
        if claims:
            reasons.append(f"launch_claims={','.join(sorted(map(str, claims)))}")
        slots = pool.get("slots") or {}
        nonavailable = sorted(
            str(slot_id) for slot_id, slot in slots.items()
            if not isinstance(slot, Mapping)
            or slot.get("status", "available") != "available"
        )
        if nonavailable:
            reasons.append(f"nonavailable_slots={','.join(nonavailable)}")
        if pool.get("stop_intent"):
            reasons.append("stop_intent=active")
        owners = sorted(
            str(lease_id) for lease_id, lease in (state.get("leases") or {}).items()
            if isinstance(lease, Mapping)
            and lease.get("status") in ACTIVE_LEASE_STATES
            and cls._lease_matches_pool(dict(lease), dict(pool))
        )
        if owners:
            reasons.append(f"active_leases={','.join(owners)}")
        return reasons

    def service_identity_conflicts(
        self,
        *,
        service_kind: str,
        identity_value: str,
        desired_pool_id: str,
    ) -> List[str]:
        """Return durable same-identity ownership outside the desired pool.

        This catches the cross-pool crash window that scheduler and endpoint
        scans cannot see yet: an accepted submission may have a durable pending
        comment while the allocation is still absent from ``squeue``.
        """
        def inspect(state: Dict[str, Any]) -> List[str]:
            conflicts: List[str] = []
            for pool_id, pool in sorted((state.get("pools") or {}).items()):
                if pool_id == desired_pool_id or not isinstance(pool, dict):
                    continue
                if pool.get("service_kind") != service_kind:
                    continue
                if self._pool_identity_value(pool) != str(identity_value):
                    continue
                # A crashed launcher may have left only a claim. Reap it under
                # the same lifecycle lock before deciding that the old pool is
                # durable conflicting ownership. Unreadable PID identity is
                # uncertainty and deliberately retains the claim.
                self._reap_stale_launch_claims(state, pool_id, pool)
                reasons = self._pool_durable_launch_reasons(state, pool)
                if reasons:
                    conflicts.append(
                        f"lifecycle_pool={pool_id} "
                        f"status={pool.get('status') or 'unknown'} "
                        + " ".join(reasons)
                    )
            return conflicts

        return list(self._mutate(inspect))

    def retire_abandoned_identity_pools(
        self,
        *,
        service_kind: str,
        identity_value: str,
        keep_pool_id: str,
    ) -> List[str]:
        """Atomically stop empty superseded pools after all live scans pass.

        Any chain, claim, non-available slot, stop intent, or matching active
        lease turns retirement into an error.  This recheck is atomic with the
        status updates, so a concurrently reserved lease cannot be archived.
        """
        retired: List[str] = []

        def update(state: Dict[str, Any]):
            candidates = []
            conflicts = []
            for pool_id, pool in sorted(state["pools"].items()):
                if pool_id == keep_pool_id:
                    continue
                if pool.get("service_kind") != service_kind:
                    continue
                if self._pool_identity_value(pool) != str(identity_value):
                    continue
                self._reap_stale_launch_claims(state, pool_id, pool)
                reasons = self._pool_durable_launch_reasons(state, pool)
                if reasons:
                    conflicts.append(f"{pool_id}: {' '.join(reasons)}")
                    continue
                candidates.append((pool_id, pool))
            if conflicts:
                raise LifecycleError(
                    "Cannot supersede same-identity lifecycle ownership: "
                    + "; ".join(conflicts)
                )
            now = utc_now()
            for pool_id, pool in candidates:
                if pool.get("status") == "stopped":
                    continue
                pool["status"] = "stopped"
                pool["status_at"] = now
                pool.setdefault("status_details", {}).update({
                    "reason": "superseded empty pool before identity launch",
                    "superseded_by_pool_id": keep_pool_id,
                })
                retired.append(pool_id)
                self._event(
                    state, "empty_pool_superseded", pool_id=pool_id,
                    superseded_by_pool_id=keep_pool_id,
                )
            return list(retired)

        return list(self._mutate(update))

    @staticmethod
    def _slot_ids(pool: Dict[str, Any]) -> List[str]:
        """Return deterministic logical slot IDs for one immutable pool."""
        pool_id = str(pool.get("pool_id") or "")
        desired = int(pool.get("desired_instances") or 0)
        return [f"{pool_id}-s{index:03d}" for index in range(desired)]

    @classmethod
    def _ensure_slots(cls, pool: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Lazily migrate a pool to durable logical-slot ownership.

        Existing chain records are conservatively assigned before any free slot
        can be claimed.  A chain record is durable ownership even when its
        process cannot currently be inspected; authoritative reconciliation or
        terminal chain cleanup must release it.
        """
        expected = cls._slot_ids(pool)
        slots = pool.setdefault("slots", {})
        for slot_id in expected:
            slots.setdefault(
                slot_id,
                {"slot_id": slot_id, "status": "available"},
            )

        unbound = []
        for chain_id, chain in sorted(pool.setdefault("chains", {}).items()):
            slot_id = chain.get("slot_id")
            if slot_id not in slots:
                if chain_id in slots:
                    slot_id = chain_id
                else:
                    unbound.append((chain_id, chain))
                    continue
            slot = slots[slot_id]
            owner = slot.get("chain_id")
            if slot.get("status") == "owned" and owner not in {None, chain_id}:
                raise LifecycleError(
                    f"Logical slot {slot_id!r} has conflicting chain owners"
                )
            chain["slot_id"] = slot_id
            slot.update({
                "slot_id": slot_id,
                "status": "owned",
                "chain_id": chain_id,
                "pid": int(chain.get("pid") or 0),
                "process_identity": copy.deepcopy(
                    chain.get("process_identity")
                ),
                "updated_at": chain.get("updated_at") or utc_now(),
            })
            for key in ("claimant_id", "claimant_pid", "claimed_at"):
                slot.pop(key, None)

        for chain_id, chain in unbound:
            available = next(
                (
                    slot_id for slot_id in expected
                    if slots[slot_id].get("status") == "available"
                ),
                None,
            )
            if available is None:
                raise LifecycleError(
                    f"Pool {pool.get('pool_id')!r} has more durable chain "
                    "owners than declared logical slots"
                )
            chain["slot_id"] = available
            slots[available].update({
                "slot_id": available,
                "status": "owned",
                "chain_id": chain_id,
                "pid": int(chain.get("pid") or 0),
                "process_identity": copy.deepcopy(
                    chain.get("process_identity")
                ),
                "updated_at": chain.get("updated_at") or utc_now(),
            })
        return slots

    def resolve_declared_pool(
        self,
        *,
        service_kind: str,
        identity_value: str,
        requested_pool_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve exactly one predeclared serving envelope, never "newest".

        Experiments use this before reserving a lease so a same-name pool with
        different quantization, dependencies, or topology cannot be kept alive
        or selected accidentally.  When multiple envelopes exist, the operator
        must name the intended pool explicitly through lifecycle configuration.
        """
        state = self.snapshot()
        candidates = []
        for pool in state["pools"].values():
            if pool.get("service_kind") != service_kind:
                continue
            if requested_pool_id and pool.get("pool_id") != requested_pool_id:
                continue
            if service_kind == "vllm":
                actual = (pool.get("server_spec") or {}).get("served_name")
            elif service_kind == "gpu_tools":
                actual = (pool.get("tool_spec") or {}).get("reconstruct_backend")
            else:
                raise ValueError(f"Unsupported service_kind={service_kind!r}")
            if str(actual or "") != str(identity_value):
                continue
            if pool.get("status") in {
                "failed", "quarantined", "stopped", "stopping", "draining",
            } or pool.get("stop_intent"):
                continue
            candidates.append(pool)
        if len(candidates) != 1:
            qualifier = (
                f" requested_pool_id={requested_pool_id!r}"
                if requested_pool_id else ""
            )
            raise LifecycleError(
                f"Expected exactly one declared {service_kind} pool for "
                f"{identity_value!r}{qualifier}; found {len(candidates)}"
            )
        return copy.deepcopy(candidates[0])

    def register_pool(
        self,
        *,
        pool_id: str,
        service_kind: str,
        namespace: str,
        desired_instances: int,
        server_spec: Optional[Dict[str, Any]] = None,
        tool_spec: Optional[Dict[str, Any]] = None,
        topology: Optional[Dict[str, Any]] = None,
        lifecycle_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if service_kind not in {"vllm", "gpu_tools"}:
            raise ValueError(f"Unsupported service_kind={service_kind!r}")
        if int(desired_instances) < 1:
            raise ValueError("desired_instances must be positive")
        effective_topology = topology or {
            "desired_instances": int(desired_instances)
        }
        digests = component_digests(
            server_spec=server_spec,
            tool_spec=tool_spec,
            topology=effective_topology,
            lifecycle_policy=lifecycle_policy,
        )
        identity = {
            "service_kind": service_kind,
            "namespace": namespace,
            "desired_instances": int(desired_instances),
            "digests": digests,
        }

        def update(state: Dict[str, Any]):
            pools = state["pools"]
            existing = pools.get(pool_id)
            if existing is not None:
                current = {
                    key: existing.get(key)
                    for key in ("service_kind", "namespace", "desired_instances", "digests")
                }
                if current != identity:
                    raise LifecycleError(
                        f"Pool {pool_id!r} already exists with a different specification"
                    )
                return copy.deepcopy(existing)
            record = {
                "pool_id": pool_id,
                **identity,
                "server_spec": server_spec,
                "tool_spec": tool_spec,
                "topology": effective_topology,
                "lifecycle_policy": lifecycle_policy or {},
                "status": "launching",
                "created_at": utc_now(),
                "chains": {},
            }
            record["slots"] = {
                slot_id: {"slot_id": slot_id, "status": "available"}
                for slot_id in self._slot_ids(record)
            }
            pools[pool_id] = record
            self._event(state, "pool_registered", pool_id=pool_id)
            return copy.deepcopy(record)

        return self._mutate(update)

    def ensure_pool(
        self,
        *,
        proposed_pool_id: str,
        service_kind: str,
        namespace: str,
        desired_instances: int,
        server_spec: Optional[Dict[str, Any]] = None,
        tool_spec: Optional[Dict[str, Any]] = None,
        topology: Optional[Dict[str, Any]] = None,
        lifecycle_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically reuse an identical desired pool or create it once."""
        effective_topology = topology or {
            "desired_instances": int(desired_instances)
        }
        digests = component_digests(
            server_spec=server_spec,
            tool_spec=tool_spec,
            topology=effective_topology,
            lifecycle_policy=lifecycle_policy,
        )

        def update(state: Dict[str, Any]):
            matches = [
                pool for pool in state["pools"].values()
                if pool.get("service_kind") == service_kind
                and int(pool.get("desired_instances") or 0) == int(desired_instances)
                and pool.get("digests") == digests
                and pool.get("status") not in {
                    "failed", "quarantined", "stopped", "stopping", "draining",
                }
                and not pool.get("stop_intent")
            ]
            if matches:
                return copy.deepcopy(
                    max(matches, key=lambda item: item.get("created_at", ""))
                )
            record = {
                "pool_id": proposed_pool_id,
                "service_kind": service_kind,
                "namespace": namespace,
                "desired_instances": int(desired_instances),
                "digests": digests,
                "server_spec": server_spec,
                "tool_spec": tool_spec,
                "topology": effective_topology,
                "lifecycle_policy": lifecycle_policy or {},
                "status": "launching",
                "created_at": utc_now(),
                "chains": {},
            }
            record["slots"] = {
                slot_id: {"slot_id": slot_id, "status": "available"}
                for slot_id in self._slot_ids(record)
            }
            state["pools"][proposed_pool_id] = record
            self._event(state, "pool_registered", pool_id=proposed_pool_id)
            return copy.deepcopy(record)

        return self._mutate(update)

    def record_chain(
        self,
        pool_id: str,
        chain_id: str,
        *,
        pid: int,
        job_ids: Iterable[str] = (),
        launch_claim_id: Optional[str] = None,
    ) -> None:
        """Persist a chain and atomically transfer its claimed logical slot."""
        if not process_pid_is_valid(pid):
            raise LifecycleError("Chain manager PID must be a positive integer")
        identity = _process_identity(pid)
        if not process_identity_is_valid(identity, expected_pid=pid):
            raise LifecycleError(
                "New managed chain lacks an exact kernel process identity"
            )

        def update(state: Dict[str, Any]):
            pool = state["pools"].get(pool_id)
            if pool is None:
                raise LifecycleError(f"Unknown pool {pool_id!r}")
            if pool.get("stop_intent") or pool.get("status") not in (
                LAUNCH_ELIGIBLE_POOL_STATES
            ):
                raise LifecycleError(
                    f"Pool {pool_id!r} is not eligible for new chain ownership"
                )
            slots = self._ensure_slots(pool)
            claims = pool.setdefault("launch_claims", {})
            slot_id = chain_id if chain_id in slots else None
            previous = pool.setdefault("chains", {}).get(chain_id, {})
            if slot_id is None and previous.get("slot_id") in slots:
                slot_id = previous["slot_id"]
            if launch_claim_id:
                claim = claims.get(launch_claim_id)
                if claim is None:
                    raise LifecycleError(
                        f"Launch claim {launch_claim_id!r} no longer exists"
                    )
                claimed_slots = list(claim.get("slots") or [])
                if slot_id not in claimed_slots:
                    raise LifecycleError(
                        f"Chain {chain_id!r} is not reserved by launch claim "
                        f"{launch_claim_id!r}"
                    )
                slot = slots[slot_id]
                if (
                    slot.get("status") != "claimed"
                    or slot.get("claimant_id") != launch_claim_id
                ):
                    raise LifecycleError(
                        f"Logical slot {slot_id!r} has incompatible ownership"
                    )
            elif slot_id is None:
                # Backward-compatible migration for a chain created before
                # logical-slot claims existed. It may consume only an explicitly
                # available slot; claimed/owned slots remain fenced.
                slot_id = next(
                    (
                        candidate for candidate in self._slot_ids(pool)
                        if slots[candidate].get("status") == "available"
                    ),
                    None,
                )
                if slot_id is None:
                    raise LifecycleError(
                        f"No unclaimed logical slot remains in pool {pool_id!r}"
                    )
            pool["chains"][chain_id] = {
                **previous,
                "chain_id": chain_id,
                "slot_id": slot_id,
                "pid": int(pid),
                "process_identity": copy.deepcopy(identity),
                "job_ids": [str(job) for job in job_ids if job],
                "updated_at": utc_now(),
            }
            slots[slot_id] = {
                "slot_id": slot_id,
                "status": "owned",
                "chain_id": chain_id,
                "pid": int(pid),
                "process_identity": copy.deepcopy(identity),
                "updated_at": utc_now(),
            }
            if launch_claim_id:
                claim = claims[launch_claim_id]
                claim["slots"] = [
                    value for value in claim.get("slots", [])
                    if value != slot_id
                ]
                claim["count"] = len(claim["slots"])
            self._event(
                state, "pool_chain_recorded", pool_id=pool_id, chain_id=chain_id,
                slot_id=slot_id, launch_claim_id=launch_claim_id,
            )

        self._mutate(update)

    def remove_chain_record(self, pool_id: str, chain_id: str) -> bool:
        """Release exact durable chain ownership after terminal proof/rollback."""
        removed = False

        def update(state: Dict[str, Any]):
            nonlocal removed
            pool = state["pools"].get(pool_id)
            if pool is None:
                return
            slots = self._ensure_slots(pool)
            chain = pool.setdefault("chains", {}).pop(chain_id, None)
            if chain is None:
                return
            removed = True
            slot_id = chain.get("slot_id")
            slot = slots.get(slot_id)
            if slot and slot.get("chain_id") == chain_id:
                slots[slot_id] = {
                    "slot_id": slot_id,
                    "status": "available",
                    "updated_at": utc_now(),
                }
            self._event(
                state, "pool_chain_removed", pool_id=pool_id,
                chain_id=chain_id, slot_id=slot_id,
            )

        self._mutate(update)
        return removed

    def claim_launch_slots(
        self,
        pool_id: str,
        *,
        claimant_id: str,
        claimant_pid: int,
        max_slots: Optional[int] = None,
    ) -> List[str]:
        """Atomically reserve every currently missing deterministic slot."""
        if not process_pid_is_valid(claimant_pid):
            raise LifecycleError("Launch claimant PID must be a positive integer")
        if max_slots is not None and int(max_slots) < 0:
            raise ValueError("max_slots must be non-negative")
        claimant_identity = _process_identity(claimant_pid)
        if not process_identity_is_valid(
            claimant_identity, expected_pid=claimant_pid,
        ):
            raise LifecycleError(
                "New launch claim lacks an exact kernel process identity"
            )

        def update(state: Dict[str, Any]):
            pool = state["pools"].get(pool_id)
            if pool is None:
                raise LifecycleError(f"Unknown pool {pool_id!r}")
            if pool.get("stop_intent") or pool.get("status") in {
                "stopping", "draining", "quarantined", "stopped", "failed",
            }:
                raise LifecycleError(
                    f"Pool {pool_id!r} is not eligible for new launch capacity"
                )
            slots = self._ensure_slots(pool)
            claims = pool.setdefault("launch_claims", {})
            self._reap_stale_launch_claims(state, pool_id, pool)

            existing = claims.get(claimant_id)
            if existing is not None:
                if int(existing.get("pid") or 0) != int(claimant_pid):
                    raise LifecycleError(
                        f"Launch claim {claimant_id!r} belongs to another process"
                    )
                recorded_identity = existing.get("process_identity")
                if not process_identity_is_valid(
                    recorded_identity, expected_pid=claimant_pid,
                ):
                    raise LifecycleError(
                        f"Launch claim {claimant_id!r} lacks exact process identity"
                    )
                if recorded_identity != claimant_identity:
                    raise LifecycleError(
                        f"Launch claim {claimant_id!r} PID identity changed"
                    )
                return list(existing.get("slots") or [])

            available = [
                slot_id for slot_id in self._slot_ids(pool)
                if slots[slot_id].get("status") == "available"
            ]
            if max_slots is not None:
                available = available[: int(max_slots)]
            for slot_id in available:
                slots[slot_id] = {
                    "slot_id": slot_id,
                    "status": "claimed",
                    "claimant_id": claimant_id,
                    "claimant_pid": int(claimant_pid),
                    "process_identity": copy.deepcopy(claimant_identity),
                    "claimed_at": utc_now(),
                }
            claims[claimant_id] = {
                "pid": int(claimant_pid),
                "process_identity": copy.deepcopy(claimant_identity),
                "slots": list(available),
                "count": len(available),
                "claimed_at": utc_now(),
            }
            self._event(
                state, "launch_capacity_claimed", pool_id=pool_id,
                claimant_id=claimant_id, count=len(available),
                slots=list(available),
            )
            return list(available)

        return list(self._mutate(update))

    def claim_launch_capacity(
        self,
        pool_id: str,
        *,
        claimant_id: str,
        claimant_pid: int,
        observed_compatible: int,
    ) -> int:
        """Compatibility wrapper; durable slot state is authoritative.

        ``observed_compatible`` only caps how many otherwise-free slots the
        compatibility caller requests. Durable slot ownership remains the
        authority, so a stale observation cannot recreate transferred slots.
        """
        desired = int(
            self.snapshot().get("pools", {}).get(pool_id, {})
            .get("desired_instances") or 0
        )
        return len(self.claim_launch_slots(
            pool_id,
            claimant_id=claimant_id,
            claimant_pid=claimant_pid,
            max_slots=max(0, desired - int(observed_compatible)),
        ))

    def release_launch_claim(self, pool_id: str, claimant_id: str) -> None:
        def update(state: Dict[str, Any]):
            pool = state["pools"].get(pool_id)
            if pool is None:
                return
            slots = self._ensure_slots(pool)
            removed = pool.setdefault("launch_claims", {}).pop(claimant_id, None)
            if removed is not None:
                for slot_id in removed.get("slots", []) or []:
                    slot = slots.get(slot_id)
                    if (
                        slot
                        and slot.get("status") == "claimed"
                        and slot.get("claimant_id") == claimant_id
                    ):
                        slots[slot_id] = {
                            "slot_id": slot_id,
                            "status": "available",
                            "updated_at": utc_now(),
                        }
                self._event(
                    state, "launch_capacity_released", pool_id=pool_id,
                    claimant_id=claimant_id,
                )

        self._mutate(update)

    def update_chain_jobs(
        self, pool_id: str, chain_id: str, job_ids: Iterable[str],
    ) -> None:
        def update(state: Dict[str, Any]):
            pool = state["pools"].get(pool_id)
            if pool is None:
                return
            chain = pool.setdefault("chains", {}).get(chain_id)
            if chain is None:
                raise LifecycleError(
                    f"Cannot update jobs for unowned chain {chain_id!r}"
                )
            chain["job_ids"] = [str(job) for job in job_ids if job]
            chain["updated_at"] = utc_now()
            slot = self._ensure_slots(pool).get(chain.get("slot_id"))
            if slot and slot.get("chain_id") == chain_id:
                slot["job_ids"] = list(chain["job_ids"])
                slot["updated_at"] = chain["updated_at"]

        self._mutate(update)

    def update_chain_submission(
        self,
        pool_id: str,
        chain_id: str,
        *,
        job_ids: Iterable[str],
        submission_intent: Optional[Mapping[str, Any]],
    ) -> None:
        """Atomically persist exact jobs and a pre-sbatch recovery fence.

        ``submission_intent`` is written before invoking ``sbatch`` and remains
        non-null until the resulting allocation has been observed by exact
        scheduler comment.  Reconciliation therefore cannot turn initial
        squeue visibility lag into permission to free the logical slot.
        """
        normalized_jobs = [str(job) for job in job_ids if job]
        normalized_intent = (
            copy.deepcopy(dict(submission_intent))
            if submission_intent is not None else None
        )

        def update(state: Dict[str, Any]):
            pool = state["pools"].get(pool_id)
            if pool is None:
                raise LifecycleError(f"Unknown pool {pool_id!r}")
            chain = pool.setdefault("chains", {}).get(chain_id)
            if chain is None:
                raise LifecycleError(
                    f"Cannot update submission for unowned chain {chain_id!r}"
                )
            if any(not job_id.isdigit() for job_id in normalized_jobs):
                raise LifecycleError("Managed chain job IDs must be numeric")
            if normalized_intent is not None:
                generation = normalized_intent.get("generation")
                tag = parse_managed_job_comment(
                    normalized_intent.get("comment")
                )
                intent_jobs = normalized_intent.get("job_ids", [])
                if (
                    isinstance(
                        normalized_intent.get("schema_version"), bool,
                    )
                    or normalized_intent.get("schema_version") != 1
                    or isinstance(generation, bool)
                    or not isinstance(generation, int)
                    or generation < 1
                    or normalized_intent.get("phase") not in {
                        "prepared", "accepted_unseen",
                        "accepted_unresolved", "ambiguous_multiple",
                    }
                    or not isinstance(intent_jobs, list)
                    or any(
                        not isinstance(job_id, str) or not job_id.isdigit()
                        for job_id in intent_jobs
                    )
                    or tag is None
                    or tag["pool_id"] != pool_id
                    or tag["slot_id"] != chain.get("slot_id")
                    or tag["generation"] != generation
                ):
                    raise LifecycleError(
                        f"Lifecycle chain {chain_id!r} submission intent "
                        "does not match its exact owned pool/slot generation"
                    )
            chain["job_ids"] = list(normalized_jobs)
            chain["submission_intent"] = normalized_intent
            chain["updated_at"] = utc_now()
            slot = self._ensure_slots(pool).get(chain.get("slot_id"))
            if slot and slot.get("chain_id") == chain_id:
                slot["job_ids"] = list(normalized_jobs)
                slot["submission_pending"] = normalized_intent is not None
                slot["updated_at"] = chain["updated_at"]
            self._event(
                state,
                "chain_submission_updated",
                pool_id=pool_id,
                chain_id=chain_id,
                job_ids=list(normalized_jobs),
                submission_pending=normalized_intent is not None,
            )

        self._mutate(update)

    def set_pool_status(self, pool_id: str, status: str, **details: Any) -> None:
        def update(state: Dict[str, Any]):
            pool = state["pools"].get(pool_id)
            if pool is None:
                raise LifecycleError(f"Unknown pool {pool_id!r}")
            pool["status"] = status
            pool["status_at"] = utc_now()
            if details:
                pool.setdefault("status_details", {}).update(details)
            self._event(state, "pool_status", pool_id=pool_id, status=status)

        self._mutate(update)

    def stop_empty_pool_if_unowned(self, pool_id: str) -> bool:
        """Atomically terminalize a declared pool that never gained capacity."""
        def update(state: Dict[str, Any]):
            pool = state["pools"].get(pool_id)
            if pool is None:
                return True
            slots = self._ensure_slots(pool)
            active_owners = [
                lease_id for lease_id, lease in state["leases"].items()
                if lease.get("status") in ACTIVE_LEASE_STATES
                and pool_id in {
                    str((lease.get("requirements") or {}).get(
                        "vllm_pool_id",
                    ) or ""),
                    str((lease.get("requirements") or {}).get(
                        "tool_pool_id",
                    ) or ""),
                }
            ]
            if (
                active_owners
                or pool.get("chains")
                or pool.get("launch_claims")
                or pool.get("stop_intent")
                or any(
                    slot.get("status") != "available"
                    for slot in slots.values()
                )
            ):
                return False
            pool["status"] = "stopped"
            pool["status_at"] = utc_now()
            pool.setdefault("status_details", {})[
                "empty_unowned_terminalization"
            ] = True
            self._event(
                state, "empty_pool_stopped", pool_id=pool_id,
            )
            return True

        return bool(self._mutate(update))

    def acquire_lease(
        self,
        *,
        lease_id: Optional[str],
        cohort_id: str,
        owner_id: str,
        owner_pid: int,
        requirements: Dict[str, Any],
        workload_policy: Dict[str, Any],
        owner_kind: str = "experiment_manager",
        renewal_ttl_seconds: float = DEFAULT_RENEWAL_HEARTBEAT_TTL_SECONDS,
    ) -> Dict[str, Any]:
        if not process_pid_is_valid(owner_pid):
            raise LifecycleError("Lease owner PID must be a positive integer")
        lease_id = lease_id or uuid.uuid4().hex
        workload_digest = canonical_digest(workload_policy)
        renewal_ttl = max(1, int(float(renewal_ttl_seconds)))

        def update(state: Dict[str, Any]):
            current_identity = _process_identity(owner_pid)
            if not process_identity_is_valid(
                current_identity, expected_pid=owner_pid,
            ):
                raise LifecycleError(
                    f"Lease {lease_id!r} owner process identity is unavailable"
                )
            leases = state["leases"]
            existing = leases.get(lease_id)
            if existing is not None:
                immutable = {
                    "cohort_id": cohort_id,
                    "owner_id": owner_id,
                    "requirements": requirements,
                    "workload_policy_digest": workload_digest,
                }
                current = {key: existing.get(key) for key in immutable}
                if current != immutable:
                    raise LifecycleError(
                        f"Lease {lease_id!r} already exists with a different identity"
                    )
                if int(existing.get("owner_pid") or 0) != int(owner_pid):
                    raise LifecycleError(
                        f"Lease {lease_id!r} is fenced to another owner process"
                    )
                recorded_identity = existing.get("owner_process_identity")
                if not process_identity_is_valid(
                    recorded_identity, expected_pid=owner_pid,
                ):
                    raise LifecycleError(
                        f"Lease {lease_id!r} has corrupt owner process identity"
                    )
                if (
                    recorded_identity != current_identity
                ):
                    raise LifecycleError(
                        f"Lease {lease_id!r} owner PID identity was reused"
                    )
                if int(existing.get("renewal_ttl_seconds") or 0) != renewal_ttl:
                    raise LifecycleError(
                        f"Lease {lease_id!r} already has a different renewal TTL"
                    )
                if existing.get("status") in ACTIVE_LEASE_STATES:
                    self._assert_requested_pools_eligible(state, existing)
                self._assert_no_matching_stop_intent(state, existing)
                return copy.deepcopy(existing)
            now = utc_now()
            record = {
                "lease_id": lease_id,
                "cohort_id": cohort_id,
                "owner_id": owner_id,
                "owner_kind": owner_kind,
                "owner_pid": owner_pid,
                "owner_process_identity": copy.deepcopy(current_identity),
                "requirements": copy.deepcopy(requirements),
                "workload_policy": copy.deepcopy(workload_policy),
                "workload_policy_digest": workload_digest,
                "status": "reserved",
                "created_at": now,
                "heartbeat_at": now,
                # Successor submission is authorized only by this explicit,
                # short-lived owner epoch.  Ordinary liveness/reconciliation
                # heartbeats deliberately do not refresh it.
                "renewal_epoch_id": uuid.uuid4().hex,
                "renewal_heartbeat_at": now,
                "renewal_heartbeat_sequence": 0,
                "renewal_ttl_seconds": renewal_ttl,
                "job_ids": [],
            }
            self._assert_requested_pools_eligible(state, record)
            self._assert_no_matching_stop_intent(state, record)
            leases[lease_id] = record
            self._event(state, "lease_reserved", lease_id=lease_id, cohort_id=cohort_id)
            return copy.deepcopy(record)

        return self._mutate(update)

    def heartbeat_lease(
        self, lease_id: str, *, job_ids: Optional[Iterable[str]] = None,
        owner_pid: Optional[int] = None,
        renewal_epoch_id: Optional[str] = None,
        renewal_evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if renewal_epoch_id is not None and owner_pid is None:
            raise LifecycleError(
                "Lease renewal heartbeat requires its exact owner PID"
            )
        if owner_pid is not None and not process_pid_is_valid(owner_pid):
            raise LifecycleError("Lease heartbeat owner PID is invalid")
        def update(state: Dict[str, Any]):
            lease = state["leases"].get(lease_id)
            if lease is None:
                raise LifecycleError(f"Unknown lease {lease_id!r}")
            if lease.get("status") in TERMINAL_LEASE_STATES:
                return
            if owner_pid is not None and int(lease.get("owner_pid") or 0) != int(
                owner_pid
            ):
                raise LifecycleError(
                    f"Lease {lease_id!r} heartbeat owner PID does not match"
                )
            if renewal_epoch_id is not None:
                if renewal_epoch_id != lease.get("renewal_epoch_id"):
                    raise LifecycleError(
                        f"Lease {lease_id!r} renewal epoch is stale or invalid"
                    )
                if owner_pid is not None and int(lease.get("owner_pid") or 0) != int(
                    owner_pid
                ):
                    raise LifecycleError(
                        f"Lease {lease_id!r} renewal owner PID does not match"
                    )
            lease["heartbeat_at"] = utc_now()
            if job_ids is not None:
                lease["job_ids"] = [str(job) for job in job_ids if job]
            if owner_pid is not None:
                identity = _process_identity(owner_pid)
                recorded_identity = lease.get("owner_process_identity")
                if not process_identity_is_valid(
                    identity, expected_pid=owner_pid,
                ):
                    raise LifecycleError(
                        f"Lease {lease_id!r} current owner identity is unavailable"
                    )
                if not process_identity_is_valid(
                    recorded_identity, expected_pid=owner_pid,
                ):
                    raise LifecycleError(
                        f"Lease {lease_id!r} has corrupt owner process identity"
                    )
                if (
                    recorded_identity != identity
                ):
                    raise LifecycleError(
                        f"Lease {lease_id!r} heartbeat PID identity was reused"
                    )
            if renewal_epoch_id is not None:
                lease["renewal_heartbeat_at"] = lease["heartbeat_at"]
                lease["renewal_heartbeat_sequence"] = int(
                    lease.get("renewal_heartbeat_sequence") or 0
                ) + 1
                lease["renewal_evidence"] = copy.deepcopy(
                    dict(renewal_evidence or {})
                )

        self._mutate(update)

    def update_evaluator_submission(
        self,
        lease_id: str,
        *,
        owner_pid: int,
        submission_intent: Optional[Mapping[str, Any]],
        job_ids: Iterable[str],
    ) -> None:
        """Durably fence one evaluator submission in its owning lease.

        The intent is published before ``sbatch``.  Exact IDs returned by
        Slurm are adopted into the lease in the same atomic lifecycle write.
        A non-null intent is deliberately not cleared merely because an
        authoritative scheduler snapshot has zero matches.
        """
        if not process_pid_is_valid(owner_pid):
            raise LifecycleError("Evaluator submission owner PID is invalid")
        normalized_jobs = [str(job_id) for job_id in job_ids if job_id]
        normalized_intent = (
            copy.deepcopy(dict(submission_intent))
            if submission_intent is not None else None
        )
        if any(not job_id.isdigit() for job_id in normalized_jobs):
            raise LifecycleError("Evaluator submission job IDs must be numeric")
        if normalized_intent is not None:
            comment = normalized_intent.get("comment")
            phase = normalized_intent.get("phase")
            intent_jobs = normalized_intent.get("job_ids", [])
            tag = parse_evaluator_job_comment(comment)
            if (
                normalized_intent.get("schema_version") != 1
                or phase not in {
                    "prepared", "accepted_unseen", "accepted_unresolved",
                    "accepted_exact", "observed", "ambiguous_multiple",
                }
                or tag is None
                or tag.get("lease_id") != lease_id
                or not isinstance(intent_jobs, list)
                or any(
                    not isinstance(job_id, str) or not job_id.isdigit()
                    for job_id in intent_jobs
                )
            ):
                raise LifecycleError(
                    "Evaluator submission intent does not match its owning lease"
                )

        def update(state: Dict[str, Any]) -> None:
            lease = state["leases"].get(lease_id)
            if lease is None:
                raise LifecycleError(f"Unknown lease {lease_id!r}")
            if lease.get("status") in TERMINAL_LEASE_STATES:
                raise LifecycleError(
                    f"Lease {lease_id!r} is terminal and cannot submit evaluators"
                )
            if int(lease.get("owner_pid") or 0) != int(owner_pid):
                raise LifecycleError(
                    f"Lease {lease_id!r} evaluator submission owner PID does not match"
                )
            current_identity = _process_identity(owner_pid)
            recorded_identity = lease.get("owner_process_identity")
            if (
                not process_identity_is_valid(
                    current_identity, expected_pid=owner_pid,
                )
                or not process_identity_is_valid(
                    recorded_identity, expected_pid=owner_pid,
                )
                or current_identity != recorded_identity
            ):
                raise LifecycleError(
                    f"Lease {lease_id!r} evaluator submission owner identity "
                    "is unavailable or changed"
                )
            existing_jobs = {
                str(job_id) for job_id in lease.get("job_ids", []) if job_id
            }
            lease["job_ids"] = sorted(
                existing_jobs | set(normalized_jobs),
                key=lambda value: int(value),
            )
            lease["evaluator_submission_intent"] = normalized_intent
            lease["heartbeat_at"] = utc_now()
            self._event(
                state,
                "evaluator_submission_updated",
                lease_id=lease_id,
                phase=(normalized_intent or {}).get("phase"),
                comment=(normalized_intent or {}).get("comment"),
                job_ids=list(normalized_jobs),
            )

        self._mutate(update)

    def bind_lease_requirements(
        self, lease_id: str, requirements: Dict[str, Any],
    ) -> None:
        """Freeze pool/digest requirements before a reserved lease activates."""
        def update(state: Dict[str, Any]):
            lease = state["leases"].get(lease_id)
            if lease is None:
                raise LifecycleError(f"Unknown lease {lease_id!r}")
            if lease.get("status") != "reserved":
                raise LifecycleError(
                    f"Lease {lease_id!r} requirements can only bind while reserved"
                )
            merged = dict(lease.get("requirements") or {})
            merged.update(copy.deepcopy(requirements))
            candidate = {**lease, "requirements": merged}
            self._assert_requested_pools_eligible(state, candidate)
            self._assert_no_matching_stop_intent(state, candidate)
            lease["requirements"] = merged
            lease["requirements_digest"] = canonical_digest(merged)
            self._event(state, "lease_requirements_bound", lease_id=lease_id)

        self._mutate(update)

    def suppress_lease_renewal(
        self, lease_id: str, *, renewal_epoch_id: str, reason: str,
    ) -> None:
        """Fence successor creation without cancelling current allocations."""
        def update(state: Dict[str, Any]):
            lease = state["leases"].get(lease_id)
            if lease is None:
                raise LifecycleError(f"Unknown lease {lease_id!r}")
            if lease.get("status") in TERMINAL_LEASE_STATES:
                return
            if renewal_epoch_id != lease.get("renewal_epoch_id"):
                raise LifecycleError(
                    f"Lease {lease_id!r} renewal epoch is stale or invalid"
                )
            if lease.get("renewal_suppressed_at"):
                return
            lease["renewal_suppressed_at"] = utc_now()
            lease["renewal_suppression_reason"] = str(reason)
            self._event(
                state, "lease_renewal_suppressed", lease_id=lease_id,
                reason=str(reason),
            )

        self._mutate(update)

    def activate_lease(self, lease_id: str, manifest: Dict[str, Any]) -> None:
        def update(state: Dict[str, Any]):
            lease = state["leases"].get(lease_id)
            if lease is None:
                raise LifecycleError(f"Unknown lease {lease_id!r}")
            if lease.get("status") != "reserved":
                raise LifecycleError(
                    f"Lease {lease_id!r} cannot activate from {lease.get('status')!r}"
                )
            self._assert_requested_pools_eligible(state, lease)
            self._assert_no_matching_stop_intent(state, lease)
            lease["status"] = "active"
            lease["activated_at"] = utc_now()
            lease["heartbeat_at"] = utc_now()
            lease["service_manifest"] = copy.deepcopy(manifest)
            lease["service_manifest_digest"] = canonical_digest(manifest)
            self._event(state, "lease_activated", lease_id=lease_id)

        self._mutate(update)

    def begin_drain(self, lease_id: str) -> None:
        def update(state: Dict[str, Any]):
            lease = state["leases"].get(lease_id)
            if lease is None:
                raise LifecycleError(f"Unknown lease {lease_id!r}")
            if lease.get("status") not in {"reserved", "active"}:
                raise LifecycleError(
                    f"Lease {lease_id!r} cannot drain from {lease.get('status')!r}"
                )
            lease["status"] = "draining"
            lease["draining_at"] = utc_now()
            self._event(state, "lease_draining", lease_id=lease_id)

        self._mutate(update)

    def mark_terminal(self, lease_id: str, outcome: str, **evidence: Any) -> None:
        def update(state: Dict[str, Any]):
            lease = state["leases"].get(lease_id)
            if lease is None:
                raise LifecycleError(f"Unknown lease {lease_id!r}")
            if lease.get("status") == "released":
                return
            lease["status"] = "terminal"
            lease["terminal_at"] = utc_now()
            lease["outcome"] = outcome
            lease["terminal_evidence"] = copy.deepcopy(evidence)
            self._event(
                state, "lease_terminal", lease_id=lease_id, outcome=outcome,
            )

        self._mutate(update)

    def release_lease(self, lease_id: str, **evidence: Any) -> None:
        def update(state: Dict[str, Any]):
            lease = state["leases"].get(lease_id)
            if lease is None:
                raise LifecycleError(f"Unknown lease {lease_id!r}")
            if lease.get("status") == "released":
                return
            if lease.get("status") not in {"terminal", "orphaned", "released"}:
                raise LifecycleError(
                    f"Lease {lease_id!r} must be terminal before release"
                )
            lease["status"] = "released"
            lease["released_at"] = utc_now()
            lease["release_evidence"] = copy.deepcopy(evidence)
            self._event(state, "lease_released", lease_id=lease_id)

        self._mutate(update)

    @staticmethod
    def _lease_matches_pool(lease: Dict[str, Any], pool: Dict[str, Any]) -> bool:
        requirements = lease.get("requirements") or {}
        kind = pool.get("service_kind")
        if kind == "vllm":
            served_name = (pool.get("server_spec") or {}).get("served_name")
            if requirements.get("llm_model") != served_name:
                return False
            requested_pool = requirements.get("vllm_pool_id")
            if requested_pool and requested_pool != pool.get("pool_id"):
                return False
            requested_namespace = requirements.get("vllm_namespace")
            if requested_namespace and requested_namespace != pool.get("namespace"):
                return False
            digests = pool.get("digests") or {}
            requested_spec = requirements.get("server_spec_digest")
            if requested_spec and requested_spec != digests.get("server_spec"):
                return False
            requested_topology = requirements.get("vllm_topology_digest")
            return (
                not requested_topology
                or requested_topology == digests.get("topology")
            )
        if kind == "gpu_tools":
            if not requirements.get("needs_gpu_tools"):
                return False
            requested_pool = requirements.get("tool_pool_id")
            if requested_pool and requested_pool != pool.get("pool_id"):
                return False
            requested_namespace = requirements.get("tool_namespace")
            if requested_namespace and requested_namespace != pool.get("namespace"):
                return False
            requested_backend = requirements.get("reconstruct_backend")
            pool_backend = (pool.get("tool_spec") or {}).get("reconstruct_backend")
            if requested_backend and requested_backend != pool_backend:
                return False
            digests = pool.get("digests") or {}
            requested_spec = requirements.get("tool_spec_digest")
            if requested_spec and requested_spec != digests.get("tool_spec"):
                return False
            requested_topology = requirements.get("tool_topology_digest")
            return (
                not requested_topology
                or requested_topology == digests.get("topology")
            )
        return False

    @classmethod
    def _assert_requested_pools_eligible(
        cls, state: Dict[str, Any], lease: Dict[str, Any],
    ) -> None:
        """Validate every explicitly pinned pool inside the lifecycle lock.

        Pool resolution happens before lease acquisition, so it cannot itself
        close a retire-vs-reserve race. Rechecking the exact pool, status,
        identity, namespace, and supplied digests in the lease mutation makes
        both orderings safe: an earlier reservation fences retirement, while
        an earlier retirement makes the later reservation fail closed.
        """
        requirements = lease.get("requirements") or {}
        if not isinstance(requirements, Mapping):
            raise LifecycleError("Lease requirements authority has invalid shape")
        for pool_key, expected_kind in (
            ("vllm_pool_id", "vllm"),
            ("tool_pool_id", "gpu_tools"),
        ):
            requested_pool_id = requirements.get(pool_key)
            if requested_pool_id is None or requested_pool_id == "":
                continue
            if not isinstance(requested_pool_id, str):
                raise LifecycleError(
                    f"Lease requirement {pool_key!r} must be a string"
                )
            pool = (state.get("pools") or {}).get(requested_pool_id)
            if not isinstance(pool, dict):
                raise LifecycleError(
                    f"Explicitly requested pool {requested_pool_id!r} does not exist"
                )
            if pool.get("service_kind") != expected_kind:
                raise LifecycleError(
                    f"Explicitly requested pool {requested_pool_id!r} has "
                    "the wrong service kind"
                )
            if (
                pool.get("status") not in LAUNCH_ELIGIBLE_POOL_STATES
                or pool.get("stop_intent")
            ):
                raise LifecycleError(
                    f"Explicitly requested pool {requested_pool_id!r} is not "
                    f"lease eligible (status={pool.get('status')!r})"
                )
            if not cls._lease_matches_pool(lease, pool):
                raise LifecycleError(
                    f"Explicitly requested pool {requested_pool_id!r} does not "
                    "match the lease identity, namespace, or digest requirements"
                )

    @classmethod
    def _assert_no_matching_stop_intent(
        cls, state: Dict[str, Any], lease: Dict[str, Any],
    ) -> None:
        """Reject a compatible lease while an exact pool stop owns the fence."""
        for pool in state.get("pools", {}).values():
            intent = pool.get("stop_intent")
            if not intent or not cls._lease_matches_pool(lease, pool):
                continue
            raise LifecycleError(
                f"Pool {pool.get('pool_id')!r} has an active stop intent; "
                f"compatible lease {lease.get('lease_id')!r} cannot acquire"
            )

    def begin_pool_stop(self, pool_id: str, *, chain_id: str) -> Optional[str]:
        """Atomically create a stop token only while the pool has no owner.

        Once created, the same lifecycle lock rejects later compatible lease
        acquisition/binding until the stop finishes.  Callers still revalidate
        immediately before destructive scheduler actions as defense in depth.
        """
        def update(state: Dict[str, Any]):
            pool = state["pools"].get(pool_id)
            if pool is None:
                raise LifecycleError(f"Unknown pool {pool_id!r}")
            owners = [
                lease for lease in state["leases"].values()
                if lease.get("status") in ACTIVE_LEASE_STATES
                and self._lease_matches_pool(lease, pool)
            ]
            if owners:
                return None
            existing = pool.get("stop_intent")
            if existing:
                if existing.get("chain_id") == chain_id:
                    return str(existing["token"])
                return None
            token = uuid.uuid4().hex
            chain = (pool.get("chains") or {}).get(chain_id) or {}
            manager_pid = int(chain.get("pid") or os.getpid())
            pool["stop_intent"] = {
                "token": token,
                "chain_id": chain_id,
                "slot_id": chain.get("slot_id") or chain_id,
                "manager_pid": manager_pid,
                "manager_process_identity": copy.deepcopy(
                    chain.get("process_identity")
                    or _process_identity(manager_pid)
                ),
                "job_ids": [
                    str(job_id) for job_id in chain.get("job_ids", []) if job_id
                ],
                "created_at": utc_now(),
                "previous_status": pool.get("status") or "ready",
            }
            pool["status"] = "stopping"
            pool["status_at"] = utc_now()
            self._event(
                state, "pool_stop_started", pool_id=pool_id,
                chain_id=chain_id, token=token,
            )
            return token

        return self._mutate(update)

    def validate_pool_stop(self, pool_id: str, token: str) -> bool:
        """Locked final owner/token fence for a destructive stop operation."""
        with _FileLock(self.lock_file):
            state = self._read_unlocked()
            pool = state["pools"].get(pool_id)
            if pool is None:
                return False
            intent = pool.get("stop_intent") or {}
            if intent.get("token") != token:
                return False
            return not any(
                lease.get("status") in ACTIVE_LEASE_STATES
                and self._lease_matches_pool(lease, pool)
                for lease in state["leases"].values()
            )

    def end_pool_stop(
        self, pool_id: str, token: str, *, outcome: str,
    ) -> bool:
        """Close an exact stop intent without clobbering a newer generation."""
        if outcome not in {"stopped", "aborted", "quarantined"}:
            raise ValueError(f"Unsupported pool stop outcome {outcome!r}")

        def update(state: Dict[str, Any]):
            pool = state["pools"].get(pool_id)
            if pool is None:
                return False
            intent = pool.get("stop_intent") or {}
            if intent.get("token") != token:
                return False
            completed = {
                **copy.deepcopy(intent),
                "finished_at": utc_now(),
                "outcome": outcome,
            }
            pool["last_stop_intent"] = completed
            pool.pop("stop_intent", None)
            if outcome == "aborted":
                pool["status"] = intent.get("previous_status") or "ready"
            else:
                pool["status"] = outcome
            pool["status_at"] = utc_now()
            self._event(
                state, "pool_stop_finished", pool_id=pool_id,
                token=token, outcome=outcome,
            )
            return True

        return bool(self._mutate(update))

    def active_leases_for_pool(self, pool_id: str) -> List[Dict[str, Any]]:
        state = self.snapshot()
        pool = state["pools"].get(pool_id)
        if pool is None:
            return []
        return [
            copy.deepcopy(lease)
            for lease in state["leases"].values()
            if lease.get("status") in ACTIVE_LEASE_STATES
            and self._lease_matches_pool(lease, pool)
        ]

    def should_renew_pool(self, pool_id: str) -> bool:
        """Authorize a successor only for a fresh, epoch-fenced owner.

        Expiry suppresses *future successor submission*; it never changes the
        lease state and is never permission to cancel a current allocation.
        That separation protects active work while preventing a live-but-hung
        manager record from renewing GPU services indefinitely.
        """
        now = _parse_utc_timestamp(utc_now())
        if now is None:
            return False
        for lease in self.active_leases_for_pool(pool_id):
            if lease.get("status") not in {"reserved", "active"}:
                continue
            if lease.get("renewal_suppressed_at"):
                continue
            if not lease.get("renewal_epoch_id"):
                continue
            heartbeat = _parse_utc_timestamp(lease.get("renewal_heartbeat_at"))
            if heartbeat is None:
                continue
            try:
                ttl = max(1, int(lease.get("renewal_ttl_seconds") or 0))
            except (TypeError, ValueError):
                continue
            age = max(0.0, (now - heartbeat).total_seconds())
            if age <= ttl:
                return True
        return False

    def reconcile_leases(
        self,
        *,
        alive_job_ids: Iterable[str],
        owner_alive: Callable[[int], bool] = _pid_alive,
        protected_owner_pids: Iterable[int] = (),
        live_step_lease_ids: Iterable[str] = (),
    ) -> List[str]:
        """Mark provably ownerless leases ``orphaned`` while retaining evidence.

        Callers must include live Slurm allocations/steps and overlay owners in
        ``alive_job_ids``/``protected_owner_pids``.  Records are not deleted.
        """
        alive_jobs = {str(job) for job in alive_job_ids if job}
        protected_pids = {int(pid) for pid in protected_owner_pids if pid}
        protected_leases = {
            str(lease_id) for lease_id in live_step_lease_ids if lease_id
        }
        changed: List[str] = []

        def update(state: Dict[str, Any]):
            for lease_id, lease in state["leases"].items():
                if lease.get("status") not in ACTIVE_LEASE_STATES:
                    continue
                pid = int(lease.get("owner_pid") or 0)
                if lease_id in protected_leases:
                    continue
                recorded_jobs = {str(job) for job in lease.get("job_ids", []) if job}
                if pid in protected_pids:
                    continue
                reported_alive = owner_alive(pid)
                dead_proof = _dead_process_proof(
                    pid, lease.get("owner_process_identity"),
                )
                if dead_proof is None:
                    continue
                if reported_alive and dead_proof.get("kind") == "pid_absent":
                    # Conflicting observers are uncertainty, never release
                    # authority (notably across hosts/PID namespaces).
                    continue
                if recorded_jobs & alive_jobs:
                    continue
                lease["status"] = "orphaned"
                lease["orphaned_at"] = utc_now()
                lease["orphan_reason"] = (
                    "owner PID absent and no recorded live Slurm job/step"
                )
                changed.append(lease_id)
                self._event(state, "lease_orphaned", lease_id=lease_id)

        self._mutate(update)
        return changed

    def reconcile_authoritative_runtime(
        self,
        *,
        alive_job_ids: Iterable[str],
        protected_owner_pids: Iterable[int],
        live_step_lease_ids: Iterable[str],
        tagged_lease_jobs: Mapping[str, Iterable[str]],
        tagged_evaluator_jobs: Mapping[str, Iterable[str]],
        tagged_service_jobs: Mapping[str, Iterable[str]],
        exact_service_tags: Iterable[tuple[str, str]],
        live_registry_slots: Iterable[tuple[str, str]],
        predead_lease_records: Mapping[str, Mapping[str, Any]],
        predead_chain_records: Mapping[tuple[str, str], Mapping[str, Any]],
        predead_stop_records: Mapping[str, Mapping[str, Any]],
        scheduler_refreshed_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Atomically adopt tagged jobs and reconcile proven-dead ownership.

        Candidate managers must have been proven dead or PID-reused *before*
        the scheduler query and are proved again under the lifecycle lock. This
        ordering closes the crash window where a manager submits immediately
        after an absence snapshot. A logical slot is freed only if recorded
        jobs, exact service comments, and matching live registry jobs are all
        absent from the same authoritative observation.
        """
        alive_jobs = {str(job_id) for job_id in alive_job_ids if job_id}
        protected_pids = {int(pid) for pid in protected_owner_pids if pid}
        protected_leases = {
            str(lease_id) for lease_id in live_step_lease_ids if lease_id
        }
        tagged_jobs = {
            str(lease_id): {str(job_id) for job_id in job_ids if job_id}
            for lease_id, job_ids in tagged_lease_jobs.items()
            if lease_id
        }
        evaluator_jobs_by_comment = {
            str(comment): {str(job_id) for job_id in job_ids if job_id}
            for comment, job_ids in tagged_evaluator_jobs.items()
            if comment
        }
        service_jobs_by_comment = {
            str(comment): {str(job_id) for job_id in job_ids if job_id}
            for comment, job_ids in tagged_service_jobs.items()
            if comment
        }
        service_tags = {
            (str(pool_id), str(slot_id))
            for pool_id, slot_id in exact_service_tags
            if pool_id and slot_id
        }
        registry_slots = {
            (str(pool_id), str(slot_id))
            for pool_id, slot_id in live_registry_slots
            if pool_id and slot_id
        }

        def update(state: Dict[str, Any]) -> Dict[str, Any]:
            _validate_reconciliation_lifecycle_shape(state)
            now = utc_now()
            result: Dict[str, Any] = {
                "adopted_evaluator_jobs": {},
                "ambiguous_evaluator_submissions": {},
                "adopted_service_jobs": {},
                "ambiguous_service_submissions": {},
                "quarantined_service_submission_pool_ids": [],
                "orphaned_lease_ids": [],
                "orphaned_chain_ids": [],
                "quarantined_stop_pool_ids": [],
            }

            # Determine unchanged candidates before adopting evaluator IDs into
            # the same lease records below.
            unchanged_leases = {
                lease_id
                for lease_id, candidate in predead_lease_records.items()
                if lease_id in state["leases"]
                and canonical_digest(state["leases"][lease_id])
                == candidate.get("record_digest")
            }

            # Resolve the exact pre-sbatch evaluator fence before considering
            # a dead lease owner for orphaning.  An empty match set retains a
            # never-observed intent: accepted jobs may be initially invisible,
            # and absence without prior observation is not terminal proof.
            for lease_id, lease in sorted(state["leases"].items()):
                if lease.get("status") not in ACTIVE_LEASE_STATES:
                    continue
                intent = lease.get("evaluator_submission_intent")
                if not isinstance(intent, Mapping):
                    continue
                comment = str(intent.get("comment") or "")
                matches = evaluator_jobs_by_comment.get(comment, set())
                if not matches:
                    intent_jobs = {
                        str(job_id)
                        for job_id in intent.get("job_ids", [])
                        if job_id
                    }
                    previously_observed = bool(intent.get("observed")) or (
                        intent.get("phase") == "observed"
                    )
                    if previously_observed and not (intent_jobs & alive_jobs):
                        lease["evaluator_submission_intent"] = None
                        self._event(
                            state,
                            "evaluator_submission_terminal_absence",
                            lease_id=lease_id,
                            comment=comment,
                            job_ids=sorted(intent_jobs),
                        )
                    else:
                        protected_leases.add(lease_id)
                    continue
                existing_jobs = {
                    str(job_id) for job_id in lease.get("job_ids", []) if job_id
                }
                adopted = sorted(matches - existing_jobs)
                lease["job_ids"] = sorted(
                    existing_jobs | matches, key=lambda value: int(value),
                )
                lease["heartbeat_at"] = now
                lease["scheduler_comment_heartbeat_at"] = now
                protected_leases.add(lease_id)
                if adopted:
                    result["adopted_evaluator_jobs"].setdefault(
                        lease_id, []
                    ).extend(adopted)
                if len(matches) == 1:
                    lease["evaluator_submission_intent"] = {
                        **copy.deepcopy(dict(intent)),
                        "phase": "observed",
                        "job_ids": sorted(matches),
                        "observed": True,
                        "observed_at": now,
                    }
                else:
                    lease["evaluator_submission_intent"] = {
                        **copy.deepcopy(dict(intent)),
                        "phase": "ambiguous_multiple",
                        "job_ids": sorted(matches),
                        "observed": True,
                        "observed_at": now,
                    }
                    lease["renewal_suppressed_at"] = now
                    lease["renewal_suppression_reason"] = (
                        "multiple evaluator jobs matched one unique submission comment"
                    )
                    result["ambiguous_evaluator_submissions"][comment] = sorted(
                        matches
                    )
                self._event(
                    state,
                    "evaluator_submission_adopted",
                    lease_id=lease_id,
                    comment=comment,
                    job_ids=sorted(matches),
                    ambiguous=len(matches) != 1,
                )

            for lease_id, job_ids in sorted(tagged_jobs.items()):
                lease = state["leases"].get(lease_id)
                if lease is None or lease.get("status") not in ACTIVE_LEASE_STATES:
                    continue
                adopted = sorted(job_ids - {
                    str(job_id) for job_id in lease.get("job_ids", []) if job_id
                })
                lease["job_ids"] = sorted({
                    str(job_id) for job_id in lease.get("job_ids", []) if job_id
                } | job_ids)
                lease["heartbeat_at"] = now
                lease["scheduler_comment_heartbeat_at"] = now
                protected_leases.add(lease_id)
                if adopted:
                    result["adopted_evaluator_jobs"].setdefault(
                        lease_id, []
                    ).extend(
                        job_id for job_id in adopted
                        if job_id not in result["adopted_evaluator_jobs"].get(
                            lease_id, []
                        )
                    )
                    self._event(
                        state, "lease_scheduler_jobs_adopted",
                        lease_id=lease_id, job_ids=adopted,
                    )

            # Adopt accepted service submissions from their exact durable
            # comments before considering a dead manager/slot for release.
            # Zero matches deliberately leaves the intent intact: a freshly
            # accepted sbatch may be absent from an otherwise authoritative
            # squeue snapshot for a short visibility window.
            for pool_id, pool in sorted(state["pools"].items()):
                for chain_id, chain in sorted((pool.get("chains") or {}).items()):
                    intent = chain.get("submission_intent")
                    if not isinstance(intent, Mapping):
                        continue
                    candidate = predead_chain_records.get((pool_id, chain_id))
                    manager_death_proof = None
                    if (
                        isinstance(candidate, Mapping)
                        and canonical_digest(chain) == candidate.get("record_digest")
                    ):
                        manager_death_proof = _dead_process_proof(
                            chain.get("pid"), chain.get("process_identity")
                        )
                    comment = str(intent.get("comment") or "")
                    if not comment:
                        continue
                    matches = service_jobs_by_comment.get(comment, set())
                    if not matches:
                        continue
                    existing_jobs = {
                        str(job_id) for job_id in chain.get("job_ids", []) if job_id
                    }
                    adopted = sorted(matches - existing_jobs)
                    chain["job_ids"] = sorted(existing_jobs | matches)
                    chain["updated_at"] = now
                    slot = self._ensure_slots(pool).get(chain.get("slot_id"))
                    if slot and slot.get("chain_id") == chain_id:
                        slot["job_ids"] = list(chain["job_ids"])
                        slot["updated_at"] = now
                    result["adopted_service_jobs"][comment] = list(
                        chain["job_ids"]
                    )
                    if len(matches) == 1:
                        chain["submission_intent"] = None
                        if slot and slot.get("chain_id") == chain_id:
                            slot["submission_pending"] = False
                    else:
                        ambiguous = {
                            **copy.deepcopy(dict(intent)),
                            "phase": "ambiguous_multiple",
                            "matched_job_ids": sorted(matches),
                            "observed_at": now,
                        }
                        chain["submission_intent"] = ambiguous
                        if slot and slot.get("chain_id") == chain_id:
                            slot["submission_pending"] = True
                        pool["status"] = "quarantined"
                        pool["status_at"] = now
                        result["ambiguous_service_submissions"][comment] = sorted(
                            matches
                        )
                    if manager_death_proof is not None:
                        chain["lifecycle_state"] = "quarantined"
                        chain["quarantine_reason"] = (
                            "accepted service allocation adopted after manager "
                            "process death; exact jobs require recovery"
                        )
                        chain["quarantine_evidence"] = {
                            "manager_process": manager_death_proof,
                            "scheduler_refreshed_at": scheduler_refreshed_at,
                            "exact_job_ids": sorted(matches),
                            "submission_comment": comment,
                        }
                        pool["status"] = "quarantined"
                        pool["status_at"] = now
                        if pool_id not in result[
                            "quarantined_service_submission_pool_ids"
                        ]:
                            result[
                                "quarantined_service_submission_pool_ids"
                            ].append(pool_id)
                    self._event(
                        state,
                        "service_submission_adopted",
                        pool_id=pool_id,
                        chain_id=chain_id,
                        comment=comment,
                        job_ids=sorted(matches),
                        ambiguous=len(matches) != 1,
                        manager_dead=manager_death_proof is not None,
                    )

            for lease_id in sorted(unchanged_leases):
                lease = state["leases"].get(lease_id)
                if lease is None or lease.get("status") not in ACTIVE_LEASE_STATES:
                    continue
                pid = int(lease.get("owner_pid") or 0)
                if lease_id in protected_leases or pid in protected_pids:
                    continue
                if {
                    str(job_id) for job_id in lease.get("job_ids", []) if job_id
                } & alive_jobs:
                    continue
                if lease.get("evaluator_submission_intent") is not None:
                    continue
                proof = _dead_process_proof(
                    pid, lease.get("owner_process_identity")
                )
                if proof is None:
                    continue
                lease["status"] = "orphaned"
                lease["orphaned_at"] = now
                lease["orphan_reason"] = (
                    "manager process absent/reused before authoritative Slurm "
                    "query and no exact live evaluator allocation/step"
                )
                lease["orphan_evidence"] = {
                    "manager_process": proof,
                    "scheduler_refreshed_at": scheduler_refreshed_at,
                    "recorded_jobs_absent": True,
                    "tagged_evaluator_job_absent": True,
                    "live_step_absent": True,
                }
                result["orphaned_lease_ids"].append(lease_id)
                self._event(state, "lease_orphaned", lease_id=lease_id)

            for (pool_id, chain_id), candidate in sorted(
                predead_chain_records.items()
            ):
                pool = state["pools"].get(pool_id)
                if pool is None:
                    continue
                chain = (pool.get("chains") or {}).get(chain_id)
                if not isinstance(chain, dict):
                    continue
                if canonical_digest(chain) != candidate.get("record_digest"):
                    continue
                proof = _dead_process_proof(
                    chain.get("pid"), chain.get("process_identity")
                )
                if proof is None:
                    continue
                slots = self._ensure_slots(pool)
                slot_id = str(chain.get("slot_id") or "")
                recorded_jobs = {
                    str(job_id) for job_id in chain.get("job_ids", []) if job_id
                }
                if recorded_jobs & alive_jobs:
                    continue
                if chain.get("submission_intent") is not None:
                    # The exact comment is the only recovery key during the
                    # acknowledged sbatch-to-squeue visibility window.  It is
                    # intentionally a manual fence if the manager died before
                    # Slurm ever exposed the allocation.
                    continue
                if (pool_id, slot_id) in service_tags:
                    continue
                if (pool_id, slot_id) in registry_slots:
                    continue
                archived = {
                    "state": "orphaned",
                    "orphaned_at": now,
                    "reason": (
                        "manager process absent/reused before authoritative "
                        "Slurm query and no exact live service capacity"
                    ),
                    "chain": copy.deepcopy(chain),
                    "evidence": {
                        "manager_process": proof,
                        "scheduler_refreshed_at": scheduler_refreshed_at,
                        "recorded_job_ids": sorted(recorded_jobs),
                        "recorded_jobs_absent": True,
                        "exact_scheduler_comment_absent": True,
                        "matching_live_registry_job_absent": True,
                    },
                }
                archive = pool.setdefault("chain_archive", [])
                archive.append(archived)
                pool["chain_archive"] = archive[-2000:]
                pool["chains"].pop(chain_id, None)
                slot = slots.get(slot_id)
                if slot and slot.get("chain_id") == chain_id:
                    slots[slot_id] = {
                        "slot_id": slot_id,
                        "status": "available",
                        "updated_at": now,
                    }
                result["orphaned_chain_ids"].append(
                    {"pool_id": pool_id, "chain_id": chain_id, "slot_id": slot_id}
                )
                self._event(
                    state, "pool_chain_orphaned", pool_id=pool_id,
                    chain_id=chain_id, slot_id=slot_id,
                )

            for pool_id, candidate in sorted(predead_stop_records.items()):
                pool = state["pools"].get(pool_id)
                if pool is None:
                    continue
                intent = pool.get("stop_intent")
                if not isinstance(intent, dict):
                    continue
                if canonical_digest(intent) != candidate.get("intent_digest"):
                    continue
                if any(
                    lease.get("status") in ACTIVE_LEASE_STATES
                    and self._lease_matches_pool(lease, pool)
                    for lease in state["leases"].values()
                ):
                    continue
                chain_id = str(intent.get("chain_id") or "")
                if chain_id and chain_id in (pool.get("chains") or {}):
                    continue
                proof = _dead_process_proof(
                    candidate.get("manager_pid"),
                    candidate.get("manager_process_identity"),
                )
                if proof is None:
                    continue
                slot_id = str(
                    intent.get("slot_id") or candidate.get("slot_id") or chain_id
                )
                recorded_jobs = {
                    str(job_id)
                    for job_id in candidate.get("job_ids", [])
                    if job_id
                } | {
                    str(job_id) for job_id in intent.get("job_ids", []) if job_id
                }
                if recorded_jobs & alive_jobs:
                    continue
                if (pool_id, slot_id) in service_tags:
                    continue
                if (pool_id, slot_id) in registry_slots:
                    continue
                completed = {
                    **copy.deepcopy(intent),
                    "finished_at": now,
                    "outcome": "quarantined",
                    "reason": "orphaned stop manager and exact capacity absent",
                    "evidence": {
                        "manager_process": proof,
                        "scheduler_refreshed_at": scheduler_refreshed_at,
                        "recorded_job_ids": sorted(recorded_jobs),
                        "recorded_jobs_absent": True,
                        "exact_scheduler_comment_absent": True,
                        "matching_live_registry_job_absent": True,
                    },
                }
                archive = pool.setdefault("stop_intent_archive", [])
                archive.append(completed)
                pool["stop_intent_archive"] = archive[-2000:]
                pool["last_stop_intent"] = completed
                pool.pop("stop_intent", None)
                pool["status"] = "quarantined"
                pool["status_at"] = now
                result["quarantined_stop_pool_ids"].append(pool_id)
                self._event(
                    state, "pool_stop_orphaned", pool_id=pool_id,
                    token=intent.get("token"), outcome="quarantined",
                )
            return result

        return dict(self._mutate(update))


def _read_json(path: Path, *, strict: bool = False) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            return value
        if strict:
            raise LifecycleError(f"JSON authority root is not an object: {path}")
        return {}
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        if strict:
            raise LifecycleError(f"JSON authority is unreadable: {path}: {exc}") from exc
        return {}


def _validate_reconciliation_lifecycle_shape(state: Mapping[str, Any]) -> None:
    """Validate every lifecycle container touched by runtime reconciliation."""
    pools = state.get("pools")
    leases = state.get("leases")
    events = state.get("events")
    if not isinstance(pools, Mapping) or not isinstance(leases, Mapping):
        raise LifecycleError("Lifecycle pools/leases authority has invalid shape")
    if not isinstance(events, list):
        raise LifecycleError("Lifecycle events authority has invalid shape")
    for lease_id, lease in pools.items():
        if not isinstance(lease_id, str) or not isinstance(lease, Mapping):
            raise LifecycleError("Lifecycle pool authority has invalid shape")
        for field in ("chains", "slots", "launch_claims"):
            value = lease.get(field, {})
            if not isinstance(value, Mapping):
                raise LifecycleError(
                    f"Lifecycle pool {lease_id!r} field {field!r} has invalid shape"
                )
        for chain_id, chain in (lease.get("chains") or {}).items():
            if not isinstance(chain_id, str) or not isinstance(chain, Mapping):
                raise LifecycleError(
                    f"Lifecycle pool {lease_id!r} chain authority is invalid"
                )
            if not isinstance(chain.get("job_ids", []), list):
                raise LifecycleError(
                    f"Lifecycle chain {chain_id!r} job_ids has invalid shape"
                )
            if not process_pid_is_valid(chain.get("pid")):
                raise LifecycleError(
                    f"Lifecycle chain {chain_id!r} manager PID is invalid"
                )
            chain_identity = chain.get("process_identity")
            if chain_identity is not None and not process_identity_record_is_recognized(
                chain_identity, expected_pid=chain.get("pid"),
            ):
                raise LifecycleError(
                    f"Lifecycle chain {chain_id!r} process identity is invalid"
                )
            intent = chain.get("submission_intent")
            if intent is not None and not isinstance(intent, Mapping):
                raise LifecycleError(
                    f"Lifecycle chain {chain_id!r} submission intent has invalid shape"
                )
            if isinstance(intent, Mapping):
                schema_version = intent.get("schema_version")
                generation = intent.get("generation")
                phase = intent.get("phase")
                comment = intent.get("comment")
                job_ids = intent.get("job_ids", [])
                tag = parse_managed_job_comment(comment)
                if (
                    isinstance(schema_version, bool)
                    or schema_version != 1
                    or isinstance(generation, bool)
                    or not isinstance(generation, int)
                    or generation < 1
                    or phase not in {
                        "prepared", "accepted_unseen",
                        "accepted_unresolved", "ambiguous_multiple",
                    }
                    or not isinstance(comment, str)
                    or not isinstance(job_ids, list)
                    or any(
                        not isinstance(job_id, str) or not job_id.isdigit()
                        for job_id in job_ids
                    )
                    or tag is None
                    or tag["pool_id"] != lease_id
                    or tag["slot_id"] != chain.get("slot_id")
                    or tag["generation"] != generation
                ):
                    raise LifecycleError(
                        f"Lifecycle chain {chain_id!r} submission intent "
                        "does not match its exact owned pool/slot generation"
                    )
        for claimant_id, claim in (lease.get("launch_claims") or {}).items():
            if not isinstance(claimant_id, str) or not isinstance(claim, Mapping):
                raise LifecycleError(
                    f"Lifecycle pool {lease_id!r} launch claim is invalid"
                )
            if not process_pid_is_valid(claim.get("pid")):
                raise LifecycleError(
                    f"Lifecycle launch claim {claimant_id!r} PID is invalid"
                )
            claim_identity = claim.get("process_identity")
            if claim_identity is not None and not process_identity_record_is_recognized(
                claim_identity, expected_pid=claim.get("pid"),
            ):
                raise LifecycleError(
                    f"Lifecycle launch claim {claimant_id!r} process identity "
                    "is invalid"
                )
        for slot_id, slot in (lease.get("slots") or {}).items():
            if not isinstance(slot_id, str) or not isinstance(slot, Mapping):
                raise LifecycleError(
                    f"Lifecycle pool {lease_id!r} slot authority is invalid"
                )
            slot_status = slot.get("status")
            if slot_status not in {"available", "claimed", "owned"}:
                raise LifecycleError(
                    f"Lifecycle slot {slot_id!r} status is invalid"
                )
            expected_slot_pid = None
            if slot_status == "claimed":
                expected_slot_pid = slot.get("claimant_pid")
            elif slot_status == "owned":
                expected_slot_pid = slot.get("pid")
            if expected_slot_pid is not None and not process_pid_is_valid(
                expected_slot_pid
            ):
                raise LifecycleError(
                    f"Lifecycle slot {slot_id!r} owner PID is invalid"
                )
            slot_identity = slot.get("process_identity")
            if slot_status == "available" and (
                slot_identity is not None
                or slot.get("pid") is not None
                or slot.get("claimant_pid") is not None
            ):
                raise LifecycleError(
                    f"Available lifecycle slot {slot_id!r} retains owner authority"
                )
            if slot_status != "available" and not process_pid_is_valid(
                expected_slot_pid
            ):
                raise LifecycleError(
                    f"Lifecycle slot {slot_id!r} lacks an exact owner PID"
                )
            if slot_identity is not None and not process_identity_record_is_recognized(
                slot_identity, expected_pid=expected_slot_pid,
            ):
                raise LifecycleError(
                    f"Lifecycle slot {slot_id!r} process identity is invalid"
                )
        stop_intent = lease.get("stop_intent")
        if stop_intent is not None and not isinstance(stop_intent, Mapping):
            raise LifecycleError(
                f"Lifecycle pool {lease_id!r} stop intent has invalid shape"
            )
        if isinstance(stop_intent, Mapping):
            if not process_pid_is_valid(stop_intent.get("manager_pid")):
                raise LifecycleError(
                    f"Lifecycle pool {lease_id!r} stop-manager PID is invalid"
                )
            stop_identity = stop_intent.get("manager_process_identity")
            if stop_identity is not None and not process_identity_record_is_recognized(
                stop_identity, expected_pid=stop_intent.get("manager_pid"),
            ):
                raise LifecycleError(
                    f"Lifecycle pool {lease_id!r} stop-manager identity is invalid"
                )
        for field in ("chain_archive", "stop_intent_archive"):
            value = lease.get(field, [])
            if not isinstance(value, list):
                raise LifecycleError(
                    f"Lifecycle pool {lease_id!r} field {field!r} has invalid shape"
                )
    for lease_id, lease in leases.items():
        if not isinstance(lease_id, str) or not isinstance(lease, Mapping):
            raise LifecycleError("Lifecycle lease authority has invalid shape")
        if not isinstance(lease.get("job_ids", []), list):
            raise LifecycleError(
                f"Lifecycle lease {lease_id!r} job_ids has invalid shape"
            )
        if not process_pid_is_valid(lease.get("owner_pid")):
            raise LifecycleError(
                f"Lifecycle lease {lease_id!r} owner PID is invalid"
            )
        owner_identity = lease.get("owner_process_identity")
        if owner_identity is not None and not process_identity_record_is_recognized(
            owner_identity, expected_pid=lease.get("owner_pid"),
        ):
            raise LifecycleError(
                f"Lifecycle lease {lease_id!r} owner process identity is invalid"
            )
        intent = lease.get("evaluator_submission_intent")
        if intent is not None and not isinstance(intent, Mapping):
            raise LifecycleError(
                f"Lifecycle lease {lease_id!r} evaluator intent has invalid shape"
            )
        if isinstance(intent, Mapping):
            tag = parse_evaluator_job_comment(intent.get("comment"))
            intent_jobs = intent.get("job_ids", [])
            if (
                intent.get("schema_version") != 1
                or intent.get("phase") not in {
                    "prepared", "accepted_unseen", "accepted_unresolved",
                    "accepted_exact", "observed", "ambiguous_multiple",
                }
                or tag is None
                or tag.get("lease_id") != lease_id
                or not isinstance(intent_jobs, list)
                or any(
                    not isinstance(job_id, str) or not job_id.isdigit()
                    for job_id in intent_jobs
                )
            ):
                raise LifecycleError(
                    f"Lifecycle lease {lease_id!r} evaluator intent is invalid"
                )


def _read_service_registry_authority(
    root: Path,
) -> tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """Read and fully shape-check both lockless service registries."""
    serve_path = root / "spatial_agent" / "logs" / "serve.json"
    tool_path = root / "spatial_agent" / "logs" / "gpu_server.json"
    serve = _read_json(serve_path, strict=True)
    tools = _read_json(tool_path, strict=True)
    records: List[Dict[str, Any]] = []
    for model, instances in serve.items():
        if not isinstance(model, str) or not isinstance(instances, dict):
            raise LifecycleError(
                f"Endpoint registry bucket has invalid shape: {serve_path}"
            )
        for uid, info in instances.items():
            if not isinstance(uid, str) or not isinstance(info, dict):
                raise LifecycleError(
                    f"Endpoint registry record has invalid shape: {serve_path}"
                )
            records.append({"registry": "vllm", "uid": uid, **info})
    for uid, info in tools.items():
        if not isinstance(uid, str) or not isinstance(info, dict):
            raise LifecycleError(
                f"GPU registry record has invalid shape: {tool_path}"
            )
        records.append({"registry": "gpu_tools", "uid": uid, **info})
    return serve, tools, records


_SERVICE_COMMENT_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")


def format_service_job_comment(
    *, pool_id: str, slot_id: str, generation: int,
) -> str:
    """Format one exact managed (nonempty pool) or legacy service identity."""
    pool_token = str(pool_id or "")
    slot_token = str(slot_id or "")
    if pool_token and not _SERVICE_COMMENT_TOKEN_RE.fullmatch(pool_token):
        raise ValueError("pool_id contains unsupported scheduler-comment characters")
    if not _SERVICE_COMMENT_TOKEN_RE.fullmatch(slot_token):
        raise ValueError("slot_id contains unsupported scheduler-comment characters")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise ValueError("generation must be a positive integer")
    if generation < 1:
        raise ValueError("generation must be a positive integer")
    return f"spatialclaw:v1:{pool_token}:{slot_token}:{generation}"


def parse_managed_job_comment(comment: Any) -> Optional[Dict[str, Any]]:
    """Parse an exact SpatialClaw service allocation identity comment."""
    parts = str(comment or "").split(":")
    if len(parts) != 5 or parts[:2] != ["spatialclaw", "v1"]:
        return None
    pool_id, slot_id, raw_generation = parts[2:]
    if (
        not _SERVICE_COMMENT_TOKEN_RE.fullmatch(pool_id)
        or not _SERVICE_COMMENT_TOKEN_RE.fullmatch(slot_id)
        or not re.fullmatch(r"[1-9][0-9]*", raw_generation)
    ):
        return None
    generation = int(raw_generation)
    return {
        "pool_id": pool_id,
        "slot_id": slot_id,
        "generation": generation,
    }


def parse_legacy_job_comment(comment: Any) -> Optional[Dict[str, Any]]:
    """Parse an exact unmanaged SpatialClaw service allocation comment.

    Legacy service jobs deliberately leave the pool field empty.  Keeping this
    parser separate from :func:`parse_managed_job_comment` lets generic
    reconciliation distinguish a valid legacy identity from both managed
    ownership and a malformed reserved-prefix comment.
    """
    parts = str(comment or "").split(":")
    if len(parts) != 5 or parts[:2] != ["spatialclaw", "v1"]:
        return None
    pool_id, slot_id, raw_generation = parts[2:]
    if (
        pool_id
        or not _SERVICE_COMMENT_TOKEN_RE.fullmatch(slot_id)
        or not re.fullmatch(r"[1-9][0-9]*", raw_generation)
    ):
        return None
    generation = int(raw_generation)
    return {
        "pool_id": "",
        "slot_id": slot_id,
        "generation": generation,
    }


def parse_service_job_comment(comment: Any) -> Optional[Dict[str, Any]]:
    """Parse a managed or legacy service identity without conflating them."""
    managed = parse_managed_job_comment(comment)
    if managed is not None:
        return {**managed, "ownership_mode": "managed"}
    legacy = parse_legacy_job_comment(comment)
    if legacy is not None:
        return {**legacy, "ownership_mode": "legacy"}
    return None


_COMMENT_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")
_COMMENT_NONCE_RE = re.compile(r"[0-9a-f]{32}")


def collision_safe_token(value: str, *, max_length: int = 80) -> str:
    """Return a path/Slurm-safe bounded token without lossy prefix aliases."""
    raw = str(value or "")
    if not raw:
        raise ValueError("Identity token must not be empty")
    if max_length < 22:
        raise ValueError("Identity token max_length must be at least 22")
    if _COMMENT_TOKEN_RE.fullmatch(raw) and len(raw) <= max_length:
        return raw
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    prefix_length = max(1, max_length - len(digest) - 1)
    return f"{normalized[:prefix_length]}-{digest}"


def _collision_safe_comment_token(value: str, *, max_length: int = 80) -> str:
    """Backward-compatible private name used by frozen validation tooling."""
    return collision_safe_token(value, max_length=max_length)


def open_unique_text_artifact(
    directory: os.PathLike[str] | str,
    *,
    identity: str,
    suffix: str,
    mode: int = 0o600,
) -> Tuple[Path, TextIO]:
    """Create one collision-free text artifact and return its open handle.

    ``tempfile.mkstemp`` uses ``O_CREAT|O_EXCL``.  The bounded digest-preserving
    prefix is for operators; uniqueness never depends on a wall-clock second or
    a shortened logical ID.
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    token = collision_safe_token(identity, max_length=80)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f"{token}_", suffix=suffix, dir=target_dir, text=True,
    )
    path = Path(raw_path)
    try:
        os.fchmod(descriptor, mode)
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path, handle


def write_unique_text_artifact(
    directory: os.PathLike[str] | str,
    *,
    identity: str,
    suffix: str,
    content: str,
    mode: int = 0o755,
) -> Path:
    """Write and close a unique artifact before returning its path."""
    path, handle = open_unique_text_artifact(
        directory, identity=identity, suffix=suffix, mode=mode,
    )
    try:
        with handle:
            handle.write(content)
            handle.flush()
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


def evaluator_submission_evidence_token(experiment_id: str) -> str:
    """Stable collision-safe filename token for evaluator fence sidecars."""
    return collision_safe_token(experiment_id, max_length=64)


def format_evaluator_step_name(lease_id: str) -> str:
    """Exact collision-safe Slurm step name for one evaluator lease."""
    return f"spatial-lease-{collision_safe_token(lease_id, max_length=80)}"


def format_evaluator_job_comment(
    *, lease_id: str, experiment_id: str, job_number: int, nonce: str,
) -> str:
    """Format a collision-safe evaluator allocation identity.

    Lifecycle lease IDs are deliberately not rewritten: reconciliation must
    compare the complete durable ID, not a lossy prefix. Unsupported lease IDs
    are rejected instead of being silently aliased. ``-`` denotes an evaluator
    that has no lifecycle lease.
    """
    raw_lease = str(lease_id or "")
    if raw_lease:
        if raw_lease == "-" or not _COMMENT_TOKEN_RE.fullmatch(raw_lease):
            raise ValueError(
                "lease_id must contain only ASCII letters, digits, '_' or '-'"
            )
        lease_token = raw_lease
    else:
        lease_token = "-"
    try:
        numeric_job = int(job_number)
    except (TypeError, ValueError) as exc:
        raise ValueError("job_number must be a positive integer") from exc
    if numeric_job < 1 or str(numeric_job) != str(job_number):
        raise ValueError("job_number must be a canonical positive integer")
    nonce_token = str(nonce or "")
    if not _COMMENT_NONCE_RE.fullmatch(nonce_token):
        raise ValueError("nonce must contain exactly 32 lowercase hex characters")
    experiment_token = _collision_safe_comment_token(str(experiment_id or ""))
    return (
        f"spatialclaw:eval:v2:{lease_token}:{experiment_token}:"
        f"{numeric_job}:{nonce_token}"
    )


def parse_evaluator_job_comment(comment: Any) -> Optional[Dict[str, Any]]:
    """Parse the exact v2 evaluator comment grammar, or return ``None``."""
    parts = str(comment or "").split(":")
    if len(parts) != 7 or parts[:3] != ["spatialclaw", "eval", "v2"]:
        return None
    lease_token, experiment_token, raw_job_number, nonce = parts[3:]
    if lease_token != "-" and not _COMMENT_TOKEN_RE.fullmatch(lease_token):
        return None
    if not _COMMENT_TOKEN_RE.fullmatch(experiment_token):
        return None
    if not re.fullmatch(r"[1-9][0-9]*", raw_job_number):
        return None
    if not _COMMENT_NONCE_RE.fullmatch(nonce):
        return None
    return {
        "lease_id": "" if lease_token == "-" else lease_token,
        "experiment_token": experiment_token,
        "job_number": int(raw_job_number),
        "nonce": nonce,
    }


def tagged_capacity_conflicts(
    scheduler_jobs: Mapping[str, Mapping[str, Any]],
    lifecycle_state: Mapping[str, Any],
    *,
    service_kind: str,
    identity_value: str,
    desired_pool_id: str,
    compatible_chain_ids: Iterable[str],
) -> List[str]:
    """Find live tagged capacity lacking exact manager ownership.

    The scheduler snapshot must affirmatively prove completeness.  Unknown
    controller state never authorizes replacement capacity.
    """
    if not getattr(scheduler_jobs, "authoritative", False):
        raise LifecycleError(
            "Authoritative Slurm state is required before launching capacity"
        )
    if service_kind not in {"vllm", "gpu_tools"}:
        raise ValueError(f"Unsupported service_kind={service_kind!r}")
    compatible = {str(chain_id) for chain_id in compatible_chain_ids}
    pools = lifecycle_state.get("pools", {}) or {}
    conflicts: List[str] = []
    for job_id, info in dict.items(scheduler_jobs):
        if not isinstance(info, Mapping):
            continue
        tag = parse_managed_job_comment(info.get("comment"))
        if tag is None:
            continue
        tagged_pool = pools.get(tag["pool_id"])
        if tagged_pool is None:
            job_name = str(info.get("name") or "")
            relevant = (
                job_name == f"vllm-{identity_value}"
                if service_kind == "vllm"
                else job_name.startswith("gpu-")
            )
        elif tagged_pool.get("service_kind") != service_kind:
            relevant = False
        elif service_kind == "vllm":
            relevant = str(
                (tagged_pool.get("server_spec") or {}).get("served_name") or ""
            ) == str(identity_value)
        else:
            relevant = str(
                (tagged_pool.get("tool_spec") or {}).get("reconstruct_backend")
                or ""
            ) == str(identity_value)
        if not relevant:
            continue
        if (
            tag["pool_id"] == desired_pool_id
            and tag["slot_id"] in compatible
        ):
            continue
        conflicts.append(
            f"scheduler_job={job_id} pool={tag['pool_id']} "
            f"slot={tag['slot_id']} generation={tag['generation']}"
        )
    return conflicts


def registry_record_attests_pool(
    info: Mapping[str, Any],
    pool: Mapping[str, Any],
    *,
    identity_value: str,
    slot_id: Optional[str] = None,
    require_ready: bool = False,
) -> bool:
    """Validate ownership, immutable digests, and effective service identity."""
    if not isinstance(info, Mapping) or not isinstance(pool, Mapping):
        return False
    service_kind = pool.get("service_kind")
    digests = pool.get("digests") or {}
    if (
        info.get("pool_id") != pool.get("pool_id")
        or info.get("namespace") != pool.get("namespace")
        or (slot_id is not None and info.get("slot_id") != slot_id)
        or info.get("topology_digest") != digests.get("topology")
    ):
        return False
    if service_kind == "vllm":
        if info.get("server_spec_digest") != digests.get("server_spec"):
            return False
        if info.get("effective_model") != identity_value:
            return False
        spec = pool.get("server_spec") or {}
        expected_revision = spec.get("resolved_model_revision")
        if (
            expected_revision is not None
            and info.get("requested_model_revision") != expected_revision
        ):
            return False
    elif service_kind == "gpu_tools":
        if info.get("tool_spec_digest") != digests.get("tool_spec"):
            return False
        if info.get("reconstruct_backend") != identity_value:
            return False
        spec = pool.get("tool_spec") or {}
    else:
        return False
    if (
        spec.get("synthetic_keepalive") is False
        and info.get("synthetic_keepalive_effective") is not False
    ):
        return False
    state = str(info.get("state") or "")
    ready = state in READY_ENDPOINT_STATES
    if require_ready and not ready:
        return False
    if not ready:
        return not info.get("draining") and info.get("accepting") is not True
    if info.get("draining") or info.get("accepting") is False:
        return False
    plain_info = dict(info)
    plain_pool = dict(pool)
    if service_kind == "vllm":
        return _vllm_identity_matches(plain_info, plain_pool, identity_value)
    return _tool_identity_matches(plain_info, plain_pool, identity_value)


def _url_host(ip: str) -> str:
    return f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip


def _healthy(url: str, timeout: float = 3.0) -> bool:
    try:
        request = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, OSError, ValueError, socket.timeout):
        return False


def _vllm_identity_matches(
    info: Dict[str, Any], pool: Dict[str, Any], llm_model: str,
) -> bool:
    spec = pool.get("server_spec") or {}
    health = info.get("health") or {}
    if not isinstance(health, dict):
        return False
    if (
        health.get("status") != "ok"
        or health.get("backend") != "vllm"
        or health.get("protocol") != "openai-v1"
        or health.get("effective_model") != llm_model
        or llm_model not in (health.get("served_models") or [])
    ):
        return False
    expected_revision = spec.get("resolved_model_revision")
    if expected_revision is not None and (
        info.get("requested_model_revision") != expected_revision
        or health.get("requested_model_revision") != expected_revision
    ):
        return False
    field_map = {
        "max_model_len": "max_model_len",
        "max_num_seqs": "max_num_seqs",
        "tp_size": "tensor_parallel_size",
        "kv_cache_dtype": "kv_cache_dtype",
        "quantization": "quantization",
    }
    for spec_key, health_key in field_map.items():
        if spec_key not in spec:
            continue
        expected, actual = spec[spec_key], health.get(health_key)
        if spec_key in {"max_model_len", "max_num_seqs", "tp_size"}:
            try:
                if int(actual) != int(expected):
                    return False
            except (TypeError, ValueError):
                return False
        elif str(actual) != str(expected):
            return False
    if spec.get("synthetic_keepalive") is False:
        keepalive = health.get("synthetic_keepalive") or {}
        if keepalive.get("effective") is not False:
            return False
    expected_gpus = spec.get("gpus_per_instance")
    if expected_gpus is not None and len(info.get("gpus") or []) != int(expected_gpus):
        return False
    return True


def _tool_identity_matches(
    info: Dict[str, Any], pool: Dict[str, Any], backend: Optional[str],
) -> bool:
    spec = pool.get("tool_spec") or {}
    health = info.get("health") or {}
    if not isinstance(health, dict):
        return False
    expected_tools = list(spec.get("tools") or [])
    protocol = health.get("protocol") or {}
    if (
        health.get("status") != "ok"
        or health.get("backend") != backend
        or protocol.get("name") != "spatial-gpu-rpc"
        or protocol.get("version") != spec.get("protocol_version")
        or list(health.get("expected_tools") or []) != expected_tools
        or list(health.get("deployed_tools") or []) != expected_tools
        or list(info.get("tools") or []) != expected_tools
    ):
        return False
    gpu = health.get("gpu") or {}
    expected_gpus = spec.get("gpus_per_instance")
    if expected_gpus is not None:
        try:
            if int(gpu.get("requested_count")) != int(expected_gpus):
                return False
            if int(info.get("num_gpus")) != int(expected_gpus):
                return False
        except (TypeError, ValueError):
            return False
    if spec.get("synthetic_keepalive") is False:
        keepalive = health.get("synthetic_keepalive") or {}
        if keepalive.get("effective") is not False:
            return False
    return True


def _select_pool(
    state: Dict[str, Any], *, service_kind: str, served_name: Optional[str] = None,
    backend: Optional[str] = None, requested_pool_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    pools = []
    for pool in state.get("pools", {}).values():
        if pool.get("service_kind") != service_kind:
            continue
        if requested_pool_id and pool.get("pool_id") != requested_pool_id:
            continue
        if service_kind == "vllm":
            if (pool.get("server_spec") or {}).get("served_name") != served_name:
                continue
        elif backend:
            if (pool.get("tool_spec") or {}).get("reconstruct_backend") != backend:
                continue
        if pool.get("status") in {"stopped", "failed", "quarantined"}:
            continue
        if pool.get("status") in {"stopping", "draining"} or pool.get("stop_intent"):
            continue
        pools.append(pool)
    if not pools:
        return None
    return max(pools, key=lambda item: item.get("created_at", ""))


def discover_ready_manifest(
    project_root: os.PathLike[str] | str,
    *,
    llm_model: str,
    needs_gpu_tools: bool,
    reconstruct_backend: Optional[str] = None,
    vllm_pool_id: Optional[str] = None,
    tool_pool_id: Optional[str] = None,
    health_timeout: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """Return one complete, healthy, pool-scoped manifest or ``None``.

    Untagged legacy endpoints are intentionally not adopted.  Their effective
    specification and ownership cannot be established from the old registry.
    """
    root = Path(project_root).resolve()
    store = LifecycleStore(root)
    state = store.snapshot()
    vllm_pool = _select_pool(
        state, service_kind="vllm", served_name=llm_model,
        requested_pool_id=vllm_pool_id,
    )
    if vllm_pool is None:
        return None

    serve = _read_json(
        root / "spatial_agent" / "logs" / "serve.json", strict=True,
    )
    instances = serve.get(llm_model, {})
    vllm_records = []
    expected_vllm_slots = set(LifecycleStore._slot_ids(vllm_pool))
    vllm_slots = vllm_pool.get("slots") or {}
    vllm_chains = vllm_pool.get("chains") or {}
    for uid, info in instances.items():
        if info.get("pool_id") != vllm_pool["pool_id"]:
            continue
        if info.get("namespace") != vllm_pool.get("namespace"):
            continue
        if info.get("state", "ready") not in READY_ENDPOINT_STATES:
            continue
        if info.get("draining") or info.get("accepting") is False:
            continue
        if not _vllm_identity_matches(info, vllm_pool, llm_model):
            continue
        expected_topology_digest = (vllm_pool.get("digests") or {}).get(
            "topology"
        )
        if expected_topology_digest and info.get("topology_digest") != expected_topology_digest:
            continue
        slot_id = str(info.get("slot_id") or "")
        slot = vllm_slots.get(slot_id)
        chain_id = slot.get("chain_id") if isinstance(slot, Mapping) else None
        chain = vllm_chains.get(chain_id) if chain_id else None
        registry_job_id = str(info.get("slurm_job_id") or "")
        if (
            slot_id not in expected_vllm_slots
            or not isinstance(slot, Mapping)
            or slot.get("status") != "owned"
            or not isinstance(chain, Mapping)
            or chain.get("slot_id") != slot_id
            or registry_job_id not in {
                str(job_id) for job_id in chain.get("job_ids", []) if job_id
            }
        ):
            return None
        ip, port = info.get("ip"), info.get("port")
        if not ip or not port:
            continue
        base = f"http://{_url_host(str(ip))}:{port}/v1"
        if not _healthy(f"{base}/models", timeout=health_timeout):
            continue
        vllm_records.append({
            "uid": uid,
            "endpoint": base,
            "slurm_job_id": registry_job_id,
            "pool_id": info.get("pool_id"),
            "namespace": info.get("namespace"),
            "server_spec_digest": info.get("server_spec_digest"),
            "slot_id": slot_id,
            "generation": int(info.get("generation") or 0),
        })

    expected_digest = (vllm_pool.get("digests") or {}).get("server_spec")
    if expected_digest and any(
        record.get("server_spec_digest") != expected_digest
        for record in vllm_records
    ):
        return None
    # A four-hour handoff may briefly publish two generations for one logical
    # slot. Select the newest healthy generation per slot, never count both as
    # additional desired capacity.
    vllm_by_slot: Dict[str, Dict[str, Any]] = {}
    for record in vllm_records:
        slot_id = str(record["slot_id"])
        previous = vllm_by_slot.get(slot_id)
        if previous is None or record["generation"] > previous["generation"]:
            vllm_by_slot[slot_id] = record
    vllm_records = list(vllm_by_slot.values())
    if len({record["endpoint"] for record in vllm_records}) != len(vllm_records):
        return None
    expected_vllm = int(vllm_pool.get("desired_instances") or 0)
    if len(vllm_records) != expected_vllm:
        return None

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now(),
        "llm_model": llm_model,
        "vllm_pool_id": vllm_pool["pool_id"],
        "vllm_namespace": vllm_pool["namespace"],
        "vllm_server_spec_digest": expected_digest,
        "vllm_topology_digest": (vllm_pool.get("digests") or {}).get("topology"),
        # Slot order, rather than physical URL order, preserves sticky-routing
        # identity when a four-hour successor changes host/port.
        "vllm_endpoints": sorted(vllm_records, key=lambda item: item["slot_id"]),
        "tool_endpoints": [],
    }

    if needs_gpu_tools:
        tool_pool = _select_pool(
            state, service_kind="gpu_tools", backend=reconstruct_backend,
            requested_pool_id=tool_pool_id,
        )
        if tool_pool is None:
            return None
        registry = _read_json(
            root / "spatial_agent" / "logs" / "gpu_server.json", strict=True,
        )
        tool_records = []
        expected_tool_slots = set(LifecycleStore._slot_ids(tool_pool))
        tool_slots = tool_pool.get("slots") or {}
        tool_chains = tool_pool.get("chains") or {}
        for uid, info in registry.items():
            if info.get("pool_id") != tool_pool["pool_id"]:
                continue
            if info.get("namespace") != tool_pool.get("namespace"):
                continue
            if info.get("state", "ready") not in READY_ENDPOINT_STATES:
                continue
            if info.get("draining") or info.get("accepting") is False:
                continue
            if not _tool_identity_matches(
                info, tool_pool, reconstruct_backend,
            ):
                continue
            expected_tool_topology = (tool_pool.get("digests") or {}).get(
                "topology"
            )
            if expected_tool_topology and info.get("topology_digest") != expected_tool_topology:
                continue
            slot_id = str(info.get("slot_id") or "")
            slot = tool_slots.get(slot_id)
            chain_id = slot.get("chain_id") if isinstance(slot, Mapping) else None
            chain = tool_chains.get(chain_id) if chain_id else None
            registry_job_id = str(info.get("slurm_job_id") or "")
            if (
                slot_id not in expected_tool_slots
                or not isinstance(slot, Mapping)
                or slot.get("status") != "owned"
                or not isinstance(chain, Mapping)
                or chain.get("slot_id") != slot_id
                or registry_job_id not in {
                    str(job_id) for job_id in chain.get("job_ids", []) if job_id
                }
            ):
                return None
            ip, port = info.get("ip"), info.get("http_port")
            if not ip or not port:
                continue
            endpoint = f"http://{_url_host(str(ip))}:{port}"
            if not _healthy(f"{endpoint}/health", timeout=health_timeout):
                continue
            tool_records.append({
                "uid": uid,
                "endpoint": endpoint,
                "slurm_job_id": registry_job_id,
                "pool_id": info.get("pool_id"),
                "namespace": info.get("namespace"),
                "tool_spec_digest": info.get("tool_spec_digest"),
                "slot_id": slot_id,
                "generation": int(info.get("generation") or 0),
            })
        expected_tool_digest = (tool_pool.get("digests") or {}).get("tool_spec")
        if expected_tool_digest and any(
            record.get("tool_spec_digest") != expected_tool_digest
            for record in tool_records
        ):
            return None
        tools_by_slot: Dict[str, Dict[str, Any]] = {}
        for record in tool_records:
            slot_id = str(record["slot_id"])
            previous = tools_by_slot.get(slot_id)
            if previous is None or record["generation"] > previous["generation"]:
                tools_by_slot[slot_id] = record
        tool_records = list(tools_by_slot.values())
        if len({record["endpoint"] for record in tool_records}) != len(tool_records):
            return None
        expected_tools = int(tool_pool.get("desired_instances") or 0)
        if len(tool_records) != expected_tools:
            return None
        manifest.update({
            "tool_pool_id": tool_pool["pool_id"],
            "tool_namespace": tool_pool["namespace"],
            "tool_spec_digest": expected_tool_digest,
            "tool_topology_digest": (tool_pool.get("digests") or {}).get("topology"),
            "tool_endpoints": sorted(tool_records, key=lambda item: item["slot_id"]),
        })

    manifest["manifest_digest"] = canonical_digest(
        {key: value for key, value in manifest.items() if key != "created_at"}
    )
    return manifest


def wait_for_ready_manifest(
    project_root: os.PathLike[str] | str,
    *,
    llm_model: str,
    needs_gpu_tools: bool,
    reconstruct_backend: Optional[str] = None,
    vllm_pool_id: Optional[str] = None,
    tool_pool_id: Optional[str] = None,
    timeout_seconds: float = 4 * 3600,
    running_timeout_seconds: Optional[float] = None,
    poll_seconds: float = 10.0,
    stop_event=None,
    resources_running: Optional[Callable[[], Optional[bool]]] = None,
    on_poll: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Wait for a complete manifest with separate queue and GPU deadlines.

    ``timeout_seconds`` bounds scheduler-pending/no-allocation time.  Once
    ``resources_running`` reports a consuming or uncertain allocation, the
    independent, shorter ``running_timeout_seconds`` deadline takes over and
    cannot be reset by later state changes.  ``None`` from the callback means
    scheduler authority is uncertain and is treated conservatively as possibly
    consuming GPU capacity.
    """
    pending_deadline = time.monotonic() + float(timeout_seconds)
    running_deadline: Optional[float] = None
    while True:
        manifest = discover_ready_manifest(
            project_root,
            llm_model=llm_model,
            needs_gpu_tools=needs_gpu_tools,
            reconstruct_backend=reconstruct_backend,
            vllm_pool_id=vllm_pool_id,
            tool_pool_id=tool_pool_id,
        )
        if manifest is not None:
            return manifest

        if on_poll is not None:
            on_poll()

        if resources_running is not None and running_deadline is None:
            try:
                running_state = resources_running()
            except LifecycleError:
                raise
            except Exception as exc:
                raise LifecycleError(
                    f"Could not determine service allocation phase: {exc}"
                ) from exc
            if running_state is not False:
                if running_timeout_seconds is None:
                    raise LifecycleError(
                        "Service allocation may be consuming resources but no "
                        "RUNNING-to-READY timeout was configured"
                    )
                running_deadline = (
                    time.monotonic() + float(running_timeout_seconds)
                )

        deadline = running_deadline or pending_deadline
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            phase = (
                "RUNNING/partial-pool" if running_deadline is not None
                else "scheduler-PENDING"
            )
            raise LifecycleError(
                f"Service readiness timed out in {phase} phase for model "
                f"{llm_model!r}"
            )
        delay = min(float(poll_seconds), remaining)
        if stop_event is not None:
            if stop_event.wait(timeout=delay):
                raise LifecycleError("Service readiness cancelled")
        else:
            time.sleep(delay)


def manifest_environment(
    manifest: Dict[str, Any], *, lease_id: str, cohort_id: str,
    owner_id: str, workload_policy_digest: str,
) -> Dict[str, str]:
    """Out-of-band environment used to freeze discovery for one epoch."""
    expected_manifest_digest = canonical_digest(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at", "manifest_digest"}
        }
    )
    if manifest.get("manifest_digest") != expected_manifest_digest:
        raise LifecycleError(
            "Service manifest was mutated or has an invalid manifest_digest"
        )
    required_digest_fields = [
        "vllm_server_spec_digest",
        "vllm_topology_digest",
    ]
    if manifest.get("tool_pool_id"):
        required_digest_fields.extend([
            "tool_spec_digest",
            "tool_topology_digest",
        ])
    missing_digests = [
        field for field in required_digest_fields if not manifest.get(field)
    ]
    if missing_digests:
        raise LifecycleError(
            "Service manifest is missing frozen digest field(s): "
            f"{', '.join(sorted(missing_digests))}"
        )
    return {
        "SPATIAL_AGENT_LEASE_ID": lease_id,
        "SPATIAL_AGENT_COHORT_ID": cohort_id,
        "SPATIAL_AGENT_OWNER_ID": owner_id,
        "SPATIAL_AGENT_WORKLOAD_POLICY_DIGEST": workload_policy_digest,
        "SPATIAL_AGENT_SERVICE_MANIFEST_DIGEST": str(manifest["manifest_digest"]),
        "SPATIAL_AGENT_VLLM_POOL_ID": str(manifest["vllm_pool_id"]),
        "SPATIAL_AGENT_VLLM_NAMESPACE": str(manifest["vllm_namespace"]),
        "SPATIAL_AGENT_VLLM_SERVER_SPEC_DIGEST": str(
            manifest["vllm_server_spec_digest"]
        ),
        "SPATIAL_AGENT_VLLM_TOPOLOGY_DIGEST": str(
            manifest["vllm_topology_digest"]
        ),
        # Permit only the declared logical slots.  Physical endpoints may roll
        # to a newer generation in the same pool/namespace; the clients select
        # the newest accepting generation per slot and retire the predecessor.
        "SPATIAL_AGENT_VLLM_SLOT_ALLOWLIST": json.dumps(
            [record["slot_id"] for record in manifest["vllm_endpoints"]],
            separators=(",", ":"),
        ),
        "SPATIAL_AGENT_GPU_SERVER_SLOT_ALLOWLIST": json.dumps(
            [record["slot_id"] for record in manifest.get("tool_endpoints", [])],
            separators=(",", ":"),
        ),
        "SPATIAL_AGENT_TOOL_POOL_ID": str(manifest.get("tool_pool_id", "")),
        "SPATIAL_AGENT_TOOL_NAMESPACE": str(manifest.get("tool_namespace", "")),
        "SPATIAL_AGENT_TOOL_SPEC_DIGEST": str(
            manifest.get("tool_spec_digest", "")
        ),
        "SPATIAL_AGENT_TOOL_TOPOLOGY_DIGEST": str(
            manifest.get("tool_topology_digest", "")
        ),
    }


def passive_snapshot(project_root: os.PathLike[str] | str) -> Dict[str, Any]:
    """Collect lifecycle/registry state without sending inference requests."""
    root = Path(project_root).resolve()
    store = LifecycleStore(root)
    lifecycle_state = store.snapshot()
    serve_registry = _read_json(
        root / "spatial_agent" / "logs" / "serve.json", strict=True,
    )
    tool_registry = _read_json(
        root / "spatial_agent" / "logs" / "gpu_server.json", strict=True,
    )
    alerts = []
    for pool_id, pool in lifecycle_state["pools"].items():
        owners = [
            lease["lease_id"] for lease in lifecycle_state["leases"].values()
            if lease.get("status") in ACTIVE_LEASE_STATES
            and LifecycleStore._lease_matches_pool(lease, pool)
        ]
        if pool.get("status") in {"ready", "launching"} and not owners:
            alerts.append({
                "severity": "warning",
                "kind": "unowned_pool",
                "pool_id": pool_id,
                "message": "pool has no compatible owner lease",
            })
        if pool.get("status") == "quarantined":
            alerts.append({
                "severity": "error",
                "kind": "quarantined_service_pool",
                "pool_id": pool_id,
                "status_details": copy.deepcopy(pool.get("status_details") or {}),
                "message": (
                    "pool ownership is quarantined; inspect exact chain/job "
                    "records before launching or cancelling capacity"
                ),
            })
        for chain_id, chain in sorted((pool.get("chains") or {}).items()):
            if not isinstance(chain, Mapping):
                continue
            exact_job_ids = sorted({
                str(job_id) for job_id in chain.get("job_ids", []) if job_id
            })
            intent = chain.get("submission_intent")
            if isinstance(intent, Mapping):
                alerts.append({
                    "severity": "error",
                    "kind": "service_submission_pending",
                    "pool_id": pool_id,
                    "chain_id": chain_id,
                    "slot_id": chain.get("slot_id"),
                    "comment": intent.get("comment"),
                    "phase": intent.get("phase"),
                    "exact_job_ids": exact_job_ids,
                    "message": (
                        "submission outcome is fenced; reconcile the exact "
                        "scheduler comment before releasing this slot"
                    ),
                })
            pid = int(chain.get("pid") or 0)
            if exact_job_ids and not process_record_alive(
                pid, chain.get("process_identity"),
            ):
                alerts.append({
                    "severity": "error",
                    "kind": "orphaned_service_jobs",
                    "pool_id": pool_id,
                    "chain_id": chain_id,
                    "slot_id": chain.get("slot_id"),
                    "manager_pid": pid,
                    "exact_job_ids": exact_job_ids,
                    "message": (
                        "manager is proven absent/reused but exact Slurm IDs "
                        "remain durably owned; use lifecycle reconciliation "
                        "and lease-fenced cleanup"
                    ),
                })
        if pool.get("service_kind") == "vllm":
            served_name = (pool.get("server_spec") or {}).get("served_name")
            records = list((serve_registry.get(served_name) or {}).values())
        else:
            records = list(tool_registry.values())
        matching = [
            record for record in records
            if record.get("pool_id") == pool_id
            and record.get("namespace") == pool.get("namespace")
            and record.get("state") in READY_ENDPOINT_STATES
            and not record.get("draining")
            and record.get("accepting") is not False
        ]
        slots = {str(record.get("slot_id")) for record in matching}
        if matching and len(slots) != int(pool.get("desired_instances") or 0):
            alerts.append({
                "severity": "error",
                "kind": "topology_mismatch",
                "pool_id": pool_id,
                "desired_instances": pool.get("desired_instances"),
                "accepting_logical_slots": len(slots),
            })
        for record in matching:
            if record.get("synthetic_keepalive_effective") is True:
                alerts.append({
                    "severity": "error",
                    "kind": "synthetic_keepalive_active",
                    "pool_id": pool_id,
                    "endpoint_uid": record.get("uid") or record.get("slot_id"),
                })
    return {
        "observed_at": utc_now(),
        "lifecycle": lifecycle_state,
        "serve_registry": serve_registry,
        "gpu_server_registry": tool_registry,
        "alerts": alerts,
    }


def reconcile_runtime(project_root: os.PathLike[str] | str) -> Dict[str, Any]:
    """Reconcile crashed evaluators, service chains, slots, and stop intents.

    This performs no inference, submission, cancellation, or registry write.
    Every release requires a manager-death proof obtained before a fresh Slurm
    query plus authoritative absence across recorded IDs, exact scheduler
    comments, live steps, and both service registries. Corrupt or incomplete
    authority raises before the single atomic lifecycle mutation.
    """
    from spatial_agent.launch_managers.slurm_utils import (
        get_user_jobs_snapshot,
        get_user_steps_snapshot,
    )

    root = Path(project_root).resolve()
    store = LifecycleStore(root)
    before = store.snapshot()
    _validate_reconciliation_lifecycle_shape(before)

    # Prove the manager was already absent/reused before querying Slurm. A
    # process that was alive at this point is never reconciled in this pass,
    # even if it exits while the controller query is running.
    predead_leases: Dict[str, Dict[str, Any]] = {}
    for lease_id, lease in before["leases"].items():
        if lease.get("status") not in ACTIVE_LEASE_STATES:
            continue
        proof = _dead_process_proof(
            lease.get("owner_pid"), lease.get("owner_process_identity")
        )
        if proof is not None:
            predead_leases[lease_id] = {
                "record_digest": canonical_digest(lease),
                "process_evidence": proof,
            }

    predead_chains: Dict[tuple[str, str], Dict[str, Any]] = {}
    predead_stops: Dict[str, Dict[str, Any]] = {}
    for pool_id, pool in before["pools"].items():
        chains = pool.get("chains") or {}
        for chain_id, chain in chains.items():
            proof = _dead_process_proof(
                chain.get("pid"), chain.get("process_identity")
            )
            if proof is not None:
                predead_chains[(pool_id, chain_id)] = {
                    "record_digest": canonical_digest(chain),
                    "process_evidence": proof,
                }
        intent = pool.get("stop_intent")
        if not intent:
            continue
        chain_id = str(intent.get("chain_id") or "")
        chain = chains.get(chain_id) if chain_id else None
        manager_pid = intent.get("manager_pid")
        manager_identity = intent.get("manager_process_identity")
        if not manager_pid and isinstance(chain, Mapping):
            manager_pid = chain.get("pid")
            manager_identity = chain.get("process_identity")
        proof = _dead_process_proof(manager_pid, manager_identity)
        if proof is not None:
            predead_stops[pool_id] = {
                "intent_digest": canonical_digest(intent),
                "manager_pid": manager_pid,
                "manager_process_identity": copy.deepcopy(manager_identity),
                "slot_id": intent.get("slot_id")
                or ((chain or {}).get("slot_id") if isinstance(chain, Mapping) else None)
                or chain_id,
                "job_ids": list(intent.get("job_ids") or [])
                + (
                    list((chain or {}).get("job_ids") or [])
                    if isinstance(chain, Mapping) else []
                ),
                "process_evidence": proof,
            }

    jobs = get_user_jobs_snapshot(force=True)
    steps = get_user_steps_snapshot()
    if not getattr(jobs, "authoritative", False) or steps is None:
        raise LifecycleError(
            "Slurm allocation/step authority unavailable; reconciliation aborted"
        )
    if not isinstance(jobs, dict) or any(
        not isinstance(job_id, str) or not isinstance(info, dict)
        for job_id, info in jobs.items()
    ):
        raise LifecycleError("Slurm allocation authority has invalid shape")
    if not isinstance(steps, dict) or any(
        not isinstance(step_id, str) or not isinstance(info, dict)
        for step_id, info in steps.items()
    ):
        raise LifecycleError("Slurm step authority has invalid shape")

    # Registry shape is validated in full before lifecycle state is touched.
    _serve_registry, _tool_registry, registry_records = (
        _read_service_registry_authority(root)
    )

    active_lease_ids = {
        lease_id for lease_id, lease in before["leases"].items()
        if lease.get("status") in ACTIVE_LEASE_STATES
    }
    exact_service_tags = set()
    tagged_service_jobs: Dict[str, set[str]] = {}
    tagged_lease_jobs: Dict[str, set[str]] = {}
    tagged_evaluator_jobs: Dict[str, set[str]] = {}
    for job_id, info in dict.items(jobs):
        comment = str(info.get("comment") or "")
        service_identity = parse_service_job_comment(comment)
        if comment.startswith("spatialclaw:v1:") and service_identity is None:
            raise LifecycleError(
                f"Malformed live SpatialClaw service comment on job {job_id}"
            )
        service_tag = (
            service_identity
            if service_identity is not None
            and service_identity["ownership_mode"] == "managed"
            else None
        )
        if service_tag is not None:
            exact_service_tags.add(
                (service_tag["pool_id"], service_tag["slot_id"])
            )
            tagged_service_jobs.setdefault(comment, set()).add(str(job_id))
        evaluator_tag = parse_evaluator_job_comment(comment)
        if comment.startswith("spatialclaw:eval:") and evaluator_tag is None:
            raise LifecycleError(
                f"Malformed live SpatialClaw evaluator comment on job {job_id}"
            )
        if (
            evaluator_tag is not None
            and evaluator_tag["lease_id"] in active_lease_ids
        ):
            tagged_lease_jobs.setdefault(
                evaluator_tag["lease_id"], set()
            ).add(str(job_id))
            tagged_evaluator_jobs.setdefault(comment, set()).add(str(job_id))

    live_registry_slots = set()
    known_pool_ids = set(before["pools"])
    for record in registry_records:
        job_id = str(record.get("slurm_job_id") or "")
        if not job_id or job_id not in jobs:
            continue
        pool_id = str(record.get("pool_id") or "")
        slot_id = str(record.get("slot_id") or "")
        if pool_id in known_pool_ids and not slot_id:
            raise LifecycleError(
                f"Live registry record for managed pool {pool_id!r} lacks slot_id"
            )
        if pool_id and slot_id:
            live_registry_slots.add((pool_id, slot_id))

    reservations_path = root / "spatial_agent" / "logs" / "agent_reservations.json"
    if reservations_path.exists():
        try:
            with open(reservations_path, encoding="utf-8") as handle:
                reservations = json.load(handle)
        except (json.JSONDecodeError, OSError) as exc:
            raise LifecycleError(
                f"Overlay reservation authority is unreadable: {reservations_path}"
            ) from exc
        if not isinstance(reservations, dict):
            raise LifecycleError(
                f"Overlay reservation authority has invalid shape: {reservations_path}"
            )
    else:
        reservations = {}
    protected_pids = []
    for slots in reservations.values():
        if not isinstance(slots, list):
            raise LifecycleError("Overlay reservation entry has invalid shape")
        for slot in slots:
            if not isinstance(slot, dict):
                raise LifecycleError("Overlay reservation slot has invalid shape")
            pid = slot.get("pid")
            identity = slot.get("process_identity")
            if (
                isinstance(pid, bool)
                or not isinstance(pid, int)
                or pid <= 0
                or (
                    identity is not None
                    and not process_identity_record_is_recognized(
                        identity, expected_pid=pid,
                    )
                )
            ):
                raise LifecycleError(
                    "Overlay reservation PID/process identity has invalid shape"
                )
            # Reservation files are durable hints, not immortal liveness
            # authority.  Only a currently live same-identity process can
            # protect a manager.  A live PID with unreadable/missing identity
            # stays conservative; affirmative death or PID reuse does not.
            if process_record_alive(pid, identity):
                protected_pids.append(pid)

    protected_lease_ids = set()
    step_names: Dict[str, set[str]] = {}
    for lease_id in active_lease_ids:
        step_names.setdefault(
            format_evaluator_step_name(lease_id), set(),
        ).add(lease_id)
    for info in steps.values():
        name = str(info.get("name") or "")
        protected_lease_ids.update(step_names.get(name, set()))

    reconciled = store.reconcile_authoritative_runtime(
        alive_job_ids=jobs.keys(),
        protected_owner_pids=protected_pids,
        live_step_lease_ids=protected_lease_ids,
        tagged_lease_jobs=tagged_lease_jobs,
        tagged_evaluator_jobs=tagged_evaluator_jobs,
        tagged_service_jobs=tagged_service_jobs,
        exact_service_tags=exact_service_tags,
        live_registry_slots=live_registry_slots,
        predead_lease_records=predead_leases,
        predead_chain_records=predead_chains,
        predead_stop_records=predead_stops,
        scheduler_refreshed_at=getattr(jobs, "refreshed_at", None),
    )
    return {
        "reconciled_at": utc_now(),
        **reconciled,
        "live_allocation_count": len(jobs),
        "live_step_count": len(steps),
        "protected_overlay_owner_pids": sorted(set(protected_pids)),
        "protected_overlay_lease_ids": sorted(
            protected_lease_ids | set(tagged_lease_jobs)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="SpatialClaw service lifecycle status")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "command", choices=["status", "reconcile"], default="status", nargs="?",
    )
    args = parser.parse_args()
    if args.command == "reconcile":
        result = reconcile_runtime(args.project_root)
    else:
        result = passive_snapshot(args.project_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
