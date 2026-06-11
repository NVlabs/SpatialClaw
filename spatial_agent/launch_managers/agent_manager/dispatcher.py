"""Agent overlay dispatcher.

Tries to attach a new agent run as an `srun --overlap` step inside an already-
running vLLM or GPU server SLURM job. If no slot is available within a backoff
window, returns False so the caller falls back to submitting a separate sbatch.

All SLURM lookups go through `slurm_utils.get_user_jobs_snapshot()` (30s cache);
this module never calls squeue/scontrol/sacct directly.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from spatial_agent.launch_managers.agent_manager.state import FileLock
from spatial_agent.launch_managers.slurm_utils import get_user_jobs_snapshot


# ---------------------------------------------------------------------------
# Resource model
# ---------------------------------------------------------------------------

# Each vLLM / gpu_server sbatch claims --mem-per-gpu=PER_GPU_TOTAL_GB.
# Of that, PER_GPU_RESERVED_GB is left for the host's own process; the rest
# is the budget for overlay agents.
PER_GPU_TOTAL_GB = 240
PER_GPU_RESERVED_GB = 5

# Mode B: lazy with backoff.
RETRY_WAIT_SEC = 10
MAX_RETRIES = 3                # total wait ≈ MAX_RETRIES × RETRY_WAIT_SEC
MIN_TIME_LEFT_SEC = 900        # avoid hosts with <15 min left

# Per-step srun resources unrelated to memory tracking.
OVERLAY_CPUS_PER_TASK = 4

# Reservation file lives next to other shared state JSONs.
_LOG_DIR_NAME = ("spatial_agent", "logs")
_RESERVATION_BASENAME = "agent_reservations.json"

# Per-server registries — the actual source of truth for "what is running":
# each running GPU/vLLM server registers itself here on startup. The
# `*_manager_state.json` files track chain manager processes (a different
# concern) and may be empty even when servers are running.
_GPU_SERVER_REGISTRY = "gpu_server.json"
_VLLM_SERVE_REGISTRY = "serve.json"


# ---------------------------------------------------------------------------
# Datatypes
# ---------------------------------------------------------------------------


@dataclass
class AgentSlot:
    slot_id: str
    jobid: str
    concurrency_gb: int
    pid: int
    started_at: float


@dataclass
class OverlayHost:
    jobid: str
    node: str
    gpus: int
    kind: str               # "vllm" | "gpu_server"
    seconds_left: int

    @property
    def agent_budget_gb(self) -> int:
        return (PER_GPU_TOTAL_GB - PER_GPU_RESERVED_GB) * self.gpus


# ---------------------------------------------------------------------------
# Reservation file helpers
# ---------------------------------------------------------------------------


def _reservation_paths(project_root: Path) -> Tuple[Path, str]:
    log_dir = project_root.joinpath(*_LOG_DIR_NAME)
    log_dir.mkdir(parents=True, exist_ok=True)
    state_file = log_dir / _RESERVATION_BASENAME
    lock_file = str(state_file) + ".lock"
    return state_file, lock_file


def _load_reservations(state_file: Path) -> Dict[str, List[AgentSlot]]:
    if not state_file.exists():
        return {}
    try:
        with open(state_file, "r") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return {}
    out: Dict[str, List[AgentSlot]] = {}
    if not isinstance(raw, dict):
        return {}
    for jid, slots in raw.items():
        if not isinstance(slots, list):
            continue
        valid: List[AgentSlot] = []
        for s in slots:
            if not isinstance(s, dict):
                continue
            try:
                valid.append(AgentSlot(**s))
            except TypeError:
                continue
        if valid:
            out[str(jid)] = valid
    return out


def _save_reservations(
    state_file: Path, reservations: Dict[str, List[AgentSlot]]
) -> None:
    serial = {jid: [asdict(s) for s in slots] for jid, slots in reservations.items() if slots}
    tmp = state_file.parent / (state_file.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(serial, f, indent=2)
    os.replace(tmp, state_file)


# ---------------------------------------------------------------------------
# Time parsing — squeue snapshot fields are strings ("4:00:00", "0:30", "2-00:00:00")
# ---------------------------------------------------------------------------


def _parse_slurm_duration(s: str) -> Optional[int]:
    """Parse a SLURM duration string into seconds. Returns None for UNLIMITED/invalid."""
    if not s:
        return None
    s = s.strip()
    if s.upper() in ("UNLIMITED", "INVALID", "NOT_SET", "N/A", ""):
        return None
    # Optional leading "D-"
    days = 0
    if "-" in s:
        d, rest = s.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            return None
        s = rest
    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, sec = nums
    elif len(nums) == 2:
        h, m, sec = 0, nums[0], nums[1]
    elif len(nums) == 1:
        h, m, sec = 0, 0, nums[0]
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + sec


def _seconds_left(time_limit: str, elapsed: str) -> Optional[int]:
    tl = _parse_slurm_duration(time_limit)
    el = _parse_slurm_duration(elapsed)
    if tl is None or el is None:
        return None
    return max(0, tl - el)


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def _read_json_safe(path: Path):
    """Read JSON file, return None on any error."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _build_jobid_meta_map(project_root: Path) -> Dict[str, Tuple[str, int]]:
    """jobid -> (kind, gpus).

    Reads the per-server registries that GPU servers and vLLM servers write
    to themselves on startup:
      - gpu_server.json: dict[chain_id] = {slurm_job_id, num_gpus, ...}
      - serve.json:      dict[served_name] = dict[uuid] = {slurm_job_id, gpus[], tp, ...}

    These files are the source of truth for "what's currently running" — the
    *_manager_state.json files track chain manager processes only and may
    be empty even when servers are running.
    """
    log_dir = project_root.joinpath(*_LOG_DIR_NAME)
    meta: Dict[str, Tuple[str, int]] = {}

    # GPU servers (Pi3/SAM3/etc.)
    raw = _read_json_safe(log_dir / _GPU_SERVER_REGISTRY)
    if isinstance(raw, dict):
        for entry in raw.values():
            if not isinstance(entry, dict):
                continue
            jid = entry.get("slurm_job_id")
            num_gpus = entry.get("num_gpus")
            try:
                num_gpus = int(num_gpus)
            except (TypeError, ValueError):
                num_gpus = 0
            if jid and num_gpus > 0:
                meta[str(jid)] = ("gpu_server", num_gpus)

    # vLLM servers (nested: served_name -> uuid -> entry)
    raw = _read_json_safe(log_dir / _VLLM_SERVE_REGISTRY)
    if isinstance(raw, dict):
        for servers in raw.values():
            if not isinstance(servers, dict):
                continue
            for entry in servers.values():
                if not isinstance(entry, dict):
                    continue
                jid = entry.get("slurm_job_id")
                gpus_field = entry.get("gpus")
                num_gpus = 0
                if isinstance(gpus_field, list):
                    num_gpus = len(gpus_field)
                if num_gpus == 0:
                    try:
                        num_gpus = int(entry.get("tp", 0))
                    except (TypeError, ValueError):
                        num_gpus = 0
                if jid and num_gpus > 0:
                    meta[str(jid)] = ("vllm", num_gpus)

    return meta


def list_overlay_candidates(
    project_root: Path,
    snapshot: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[OverlayHost]:
    """Build the list of currently-eligible overlay hosts.

    Eligibility: status RUNNING, listed in gpu_server or vllm state, and
    `seconds_left >= MIN_TIME_LEFT_SEC`.

    `snapshot` may be passed in to avoid an extra cache read when the caller
    already has one in hand.
    """
    if snapshot is None:
        snapshot = get_user_jobs_snapshot()
    meta = _build_jobid_meta_map(project_root)
    out: List[OverlayHost] = []
    for jid, info in snapshot.items():
        if jid not in meta:
            continue
        if info.get("status", "").upper() != "RUNNING":
            continue
        node = info.get("node") or ""
        if not node:
            continue
        sec_left = _seconds_left(info.get("time_limit", ""), info.get("elapsed", ""))
        if sec_left is None or sec_left < MIN_TIME_LEFT_SEC:
            continue
        kind, gpus = meta[jid]
        out.append(OverlayHost(
            jobid=jid, node=node, gpus=int(gpus), kind=kind, seconds_left=sec_left,
        ))
    return out


# ---------------------------------------------------------------------------
# Stale slot cleanup
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, OSError, ValueError, TypeError):
        return False


# Slots older than this are forcibly dropped — defense against a chain
# process dying without finally-releasing its slot, with PID later recycled
# to an unrelated process. Set well above the 4h sbatch time limit.
_MAX_SLOT_AGE_SEC = 5 * 3600


def _drop_stale_inplace(
    reservations: Dict[str, List[AgentSlot]],
    snapshot: Dict[str, Dict[str, str]],
) -> None:
    """In-place: drop slots for jobs no longer RUNNING, with dead PIDs, or
    whose age exceeds the per-job time limit."""
    now = time.time()
    for jid in list(reservations.keys()):
        info = snapshot.get(jid)
        if not info or info.get("status", "").upper() != "RUNNING":
            del reservations[jid]
            continue
        kept = [
            s for s in reservations[jid]
            if _pid_alive(s.pid) and (now - s.started_at) < _MAX_SLOT_AGE_SEC
        ]
        if kept:
            reservations[jid] = kept
        else:
            del reservations[jid]


# ---------------------------------------------------------------------------
# Reserve / release
# ---------------------------------------------------------------------------


def try_reserve_slot(
    project_root: Path,
    concurrency: int,
    excluded_jobids: Optional[Iterable[str]] = None,
    excluded_nodes: Optional[Iterable[str]] = None,
) -> Optional[Tuple[OverlayHost, AgentSlot]]:
    """Atomically pick a host with sufficient free budget and reserve a slot.

    Selection is randomized (uniform across hosts that have any free budget
    above `concurrency`) so concurrent dispatchers don't all pile onto the
    same node — important because Slurm caps `srun` step ports per node
    (typically ~25 ports → easy exhaustion under bursts).

    `excluded_jobids` / `excluded_nodes` are skipped — used by the caller to
    avoid hosts that just fast-failed in this dispatch.

    Returns None when no eligible candidate has at least `concurrency` GB free.
    """
    excluded_jids: Set[str] = set(excluded_jobids or ())
    excluded_nds: Set[str] = set(excluded_nodes or ())

    # Single snapshot read shared between candidate listing and stale cleanup.
    snapshot = get_user_jobs_snapshot()
    candidates = list_overlay_candidates(project_root, snapshot=snapshot)
    if not candidates:
        return None
    state_file, lock_file = _reservation_paths(project_root)

    with FileLock(lock_file):
        reservations = _load_reservations(state_file)
        _drop_stale_inplace(reservations, snapshot)

        eligible: List[OverlayHost] = []
        for host in candidates:
            if host.jobid in excluded_jids or host.node in excluded_nds:
                continue
            used = sum(s.concurrency_gb for s in reservations.get(host.jobid, []))
            free = host.agent_budget_gb - used
            if free >= concurrency:
                eligible.append(host)

        if not eligible:
            _save_reservations(state_file, reservations)  # persist cleanup
            return None

        # Random pick spreads load across hosts. Concurrent dispatchers each
        # roll independently, naturally distributing srun steps across nodes.
        host = random.choice(eligible)

        slot = AgentSlot(
            slot_id=uuid.uuid4().hex,
            jobid=host.jobid,
            concurrency_gb=int(concurrency),
            pid=os.getpid(),
            started_at=time.time(),
        )
        reservations.setdefault(host.jobid, []).append(slot)
        _save_reservations(state_file, reservations)
        return host, slot


def release_slot(project_root: Path, slot: AgentSlot) -> None:
    state_file, lock_file = _reservation_paths(project_root)
    with FileLock(lock_file):
        reservations = _load_reservations(state_file)
        bucket = reservations.get(slot.jobid)
        if bucket is not None:
            bucket = [s for s in bucket if s.slot_id != slot.slot_id]
            if bucket:
                reservations[slot.jobid] = bucket
            else:
                reservations.pop(slot.jobid, None)
        _save_reservations(state_file, reservations)


# ---------------------------------------------------------------------------
# srun execution
# ---------------------------------------------------------------------------


def _build_srun_cmd(
    host: OverlayHost,
    concurrency_gb: int,
    script_path: str,
    script_args: List[str],
    step_log_path: Path,
) -> List[str]:
    return [
        "srun",
        f"--jobid={host.jobid}",
        "--overlap",
        # The agent is a single python process, no MPI — skip the PMIx plugin.
        "--mpi=none",
        # Step output goes directly to a per-step file so each agent's logs
        # are isolated and easy to grep.
        "--output", str(step_log_path),
        "--error", str(step_log_path),
        # Don't forward stdin to the step. Without this, srun client tries
        # to read from its own stdin (inherited from the chain manager →
        # agent-manager CLI), stealing keystrokes the user is typing into
        # the interactive menu.
        "--input=none",
        "--ntasks=1",
        "--gpus=0",
        f"--mem={int(concurrency_gb)}G",
        f"--cpus-per-task={OVERLAY_CPUS_PER_TASK}",
        "bash", str(script_path), *script_args,
    ]


def _step_log_path(project_root: Path, jobid: str, slot_id: str) -> Path:
    """Per-overlay-step log file. Mirrors the slurm_agent/ convention."""
    log_dir = project_root.joinpath("spatial_agent", "logs", "slurm_agent")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"overlay_{jobid}_{slot_id[:8]}.out"


def _run_overlay(
    project_root: Path,
    host: OverlayHost,
    slot: AgentSlot,
    script_path: str,
    script_args: List[str],
    stop_event=None,
) -> Tuple[int, float, Path]:
    """Run the overlay step, returning (returncode, elapsed_seconds, log_path).

    The step's stdout/stderr go directly to a per-step file via slurmstepd
    (`--output --error`); srun's own messages are appended to the same file.

    If `stop_event` is provided and gets set during execution, the srun
    subprocess is terminated (SIGTERM, then SIGKILL after 10s) so the chain
    can shut down promptly during overlay.

    Returncode 127 indicates srun was not found on PATH; -1 indicates an
    unexpected error before the process could be launched.
    """
    step_log = _step_log_path(project_root, host.jobid, slot.slot_id)
    cmd = _build_srun_cmd(
        host, slot.concurrency_gb, script_path, script_args, step_log,
    )
    started = time.monotonic()
    # srun's own stdout/stderr (status messages, error reports) go to PIPE
    # so we can fold them into the step log on completion. stdin is closed
    # so srun can't compete with the agent-manager CLI for keystrokes.
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        print("[dispatcher] srun binary not found on PATH", file=sys.stderr)
        return 127, time.monotonic() - started, step_log
    except OSError as e:
        print(f"[dispatcher] srun launch failed: {e}", file=sys.stderr)
        return -1, time.monotonic() - started, step_log

    while True:
        try:
            rc = proc.wait(timeout=1.0)
            break
        except subprocess.TimeoutExpired:
            if stop_event is not None and stop_event.is_set():
                proc.terminate()
                try:
                    rc = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()
                break

    # Append srun's own messages (e.g. port errors, step abort notices) to
    # the step log so a single file has the full picture.
    try:
        srun_msgs = proc.stdout.read() if proc.stdout else b""
    except Exception:
        srun_msgs = b""
    if srun_msgs:
        try:
            with open(step_log, "ab") as f:
                f.write(b"\n--- srun stderr ---\n")
                f.write(srun_msgs)
        except OSError:
            pass

    return rc, time.monotonic() - started, step_log


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# If srun returns non-zero in less than this many seconds, treat as "step
# failed to start" (parent vanished, mem rejection, port exhaustion, etc.)
# rather than a real agent run — caller blacklists the host and retries.
# A real agent run with concurrency≥1 takes far longer than this even on
# a tiny subsample, so the threshold can be generous.
_FAST_FAIL_SECONDS = 30.0


def try_dispatch_overlay(
    project_root: Path,
    script_path: str,
    script_args: List[str],
    concurrency: int,
    log_fn=None,
    stop_event=None,
) -> bool:
    """Attempt overlay dispatch. Returns True if the agent ran inside an
    existing job (regardless of agent exit code), False if no slot was found
    within the backoff window OR the chosen step failed before running —
    caller should fall back to sbatch.

    `log_fn(msg: str)` is called with status messages; defaults to print.
    `stop_event` (threading.Event) — if set during the call, the active srun
    step is terminated and the function returns False immediately.
    """
    log = log_fn if log_fn is not None else print
    project_root = Path(project_root)
    concurrency = max(1, int(concurrency))   # --mem=0G means "all memory" — disallow

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    # Hosts/nodes that fast-failed in this dispatch. Avoid retrying the same
    # ones — port exhaustion or step-creation issues tend to be sticky.
    failed_jobids: Set[str] = set()
    failed_nodes: Set[str] = set()

    for attempt in range(MAX_RETRIES + 1):
        if _stopped():
            return False
        reserved = try_reserve_slot(
            project_root, concurrency,
            excluded_jobids=failed_jobids,
            excluded_nodes=failed_nodes,
        )
        if reserved is not None:
            host, slot = reserved
            log(
                f"[dispatcher] overlay attempt OK: jobid={host.jobid} "
                f"node={host.node} kind={host.kind} mem={concurrency}G "
                f"slot={slot.slot_id[:8]}"
            )
            try:
                rc, elapsed, step_log = _run_overlay(
                    project_root, host, slot, script_path, script_args,
                    stop_event=stop_event,
                )
                log(
                    f"[dispatcher] overlay finished: jobid={host.jobid} "
                    f"rc={rc} elapsed={elapsed:.1f}s log={step_log}"
                )
            finally:
                release_slot(project_root, slot)

            if _stopped():
                return False

            # Step failed before the agent could actually run (parent vanished,
            # memory rejection, port exhaustion, etc.) — blacklist this host
            # for the rest of this dispatch and try another one.
            if rc != 0 and elapsed < _FAST_FAIL_SECONDS:
                failed_jobids.add(host.jobid)
                failed_nodes.add(host.node)
                log(
                    f"[dispatcher] step failed to start (rc={rc} in "
                    f"{elapsed:.1f}s); blacklisting node={host.node} "
                    f"and retrying"
                )
                # Brief pause so concurrent dispatchers' colliding srun
                # ports get a chance to clear.
                time.sleep(1.0)
                continue
            return True

        if attempt < MAX_RETRIES:
            log(
                f"[dispatcher] no overlay slot for concurrency={concurrency}G "
                f"(attempt {attempt + 1}/{MAX_RETRIES + 1}); retrying in "
                f"{RETRY_WAIT_SEC}s"
            )
            # Sleep in small chunks so stop_event interrupts promptly.
            slept = 0.0
            while slept < RETRY_WAIT_SEC:
                if _stopped():
                    return False
                time.sleep(min(1.0, RETRY_WAIT_SEC - slept))
                slept += 1.0

    log(
        f"[dispatcher] no overlay slot after {MAX_RETRIES + 1} attempts; "
        f"falling back to sbatch"
    )
    return False
