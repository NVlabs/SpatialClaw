"""SLURM query and control helpers.

All status queries funnel through a cross-process JSON cache of `squeue -u $USER`
so we make at most one squeue call per TTL window regardless of how many chain
processes are running. Writes are lock-protected and atomic (temp file + rename);
reads are lockless because `os.replace` is atomic on POSIX/Lustre.
"""

import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

_DELIM = "\x1f"  # ASCII unit separator — safer than '|' in job names

# Located next to the other shared state JSONs.
_CACHE_FILE = (
    Path(__file__).resolve().parent.parent / "logs" / "squeue_cache.json"
)
_CACHE_LOCK = Path(str(_CACHE_FILE) + ".lock")


class SchedulerJobIds(frozenset):
    """Job-ID snapshot that carries scheduler-authority metadata.

    Manager cleanup APIs deliberately trust only this subtype, never an
    ordinary iterable.  This lets ``get_user_jobs_snapshot(...).keys()``
    preserve whether even an empty set came from a successful controller
    query while making accidental ``set()``/``list()`` conversions fail
    closed.
    """

    def __new__(
        cls,
        values=(),
        *,
        authoritative: bool,
        refreshed_at: Optional[float] = None,
        source: str = "unknown",
    ):
        instance = super().__new__(cls, values)
        instance.authoritative = bool(authoritative)
        instance.refreshed_at = refreshed_at
        instance.source = source
        return instance


def _unknown_job_info(job_id: str, last_known: Optional[dict] = None) -> dict:
    info = dict(last_known or {})
    last_status = str(info.get("status") or "")
    if last_status and last_status != "UNKNOWN":
        info["last_known_status"] = last_status
    info.update(
        {
            "name": info.get("name", ""),
            "status": "UNKNOWN",
            "node": info.get("node", ""),
            "elapsed": info.get("elapsed", ""),
            "time_limit": info.get("time_limit", ""),
            "comment": info.get("comment", ""),
            "query_state": "scheduler_unavailable",
            "job_id": str(job_id),
        }
    )
    return info


class SchedulerSnapshot(dict):
    """Dict-compatible Slurm snapshot with explicit authority provenance.

    A non-authoritative snapshot may contain stale last-known jobs for
    observability, but every status is projected to ``UNKNOWN``.  Missing-key
    membership is conservative so legacy ownership/reconciliation callers do
    not interpret a controller outage as verified absence.
    """

    def __init__(
        self,
        jobs=None,
        *,
        authoritative: bool,
        refreshed_at: Optional[float] = None,
        source: str = "unknown",
    ):
        raw_jobs = dict(jobs or {})
        if not authoritative:
            raw_jobs = {
                str(job_id): _unknown_job_info(str(job_id), info)
                for job_id, info in raw_jobs.items()
            }
        super().__init__(raw_jobs)
        self.authoritative = bool(authoritative)
        self.refreshed_at = refreshed_at
        self.source = source

    def keys(self) -> SchedulerJobIds:
        return SchedulerJobIds(
            dict.keys(self),
            authoritative=self.authoritative,
            refreshed_at=self.refreshed_at,
            source=self.source,
        )

    def __contains__(self, key) -> bool:
        if dict.__contains__(self, key):
            return True
        # During a controller outage, absence is unknown.  Treat a queried ID
        # as possibly present so legacy membership checks fail closed.
        return not self.authoritative and bool(str(key or ""))

    def __missing__(self, key):
        if not self.authoritative and key:
            return _unknown_job_info(str(key))
        raise KeyError(key)

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            return dict.get(self, key)
        if not self.authoritative and key:
            return _unknown_job_info(str(key))
        return default


def _cache_ttl_seconds() -> int:
    try:
        return max(1, int(os.environ.get("SPATIAL_AGENT_SQUEUE_TTL_SECONDS", "30")))
    except (TypeError, ValueError):
        return 30


def _ensure_cache_dir() -> None:
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


class _CacheLock:
    """fcntl EX lock — only writers take this; readers are lockless."""

    def __init__(self, path: Path):
        self._path = path
        self._f = None

    def __enter__(self):
        _ensure_cache_dir()
        self._f = open(self._path, "a+")
        fcntl.flock(self._f.fileno(), fcntl.LOCK_EX)
        return self._f

    def __exit__(self, *exc):
        if self._f:
            fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
            self._f.close()
            self._f = None


def check_slurm_available() -> bool:
    result = subprocess.run("which sbatch", shell=True, capture_output=True)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Cache read / write
# ---------------------------------------------------------------------------


def _read_cache() -> Optional[dict]:
    try:
        with open(_CACHE_FILE, "r") as f:
            data = json.load(f)
        if (
            isinstance(data, dict)
            and "refreshed_at" in data
            and data.get("comments_authoritative") is True
            and isinstance(data.get("jobs"), dict)
            and all(isinstance(info, dict) for info in data["jobs"].values())
        ):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return None


def _write_cache_atomic(snapshot: dict) -> None:
    _ensure_cache_dir()
    fd, tmp_path = tempfile.mkstemp(
        prefix=".squeue_cache.", suffix=".tmp", dir=str(_CACHE_FILE.parent),
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(snapshot, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, _CACHE_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _invalidate_cache() -> None:
    """Force the next snapshot read to refresh. Used after scancel so dashboards
    reflect cancellations immediately instead of within the TTL window.

    The TTL check reads the `refreshed_at` field *inside* the JSON, not the
    file mtime — so we must rewrite the snapshot with refreshed_at=0 (or
    unlink). Unlink is simpler and the lock-protected refresh path handles
    the missing-file case.
    """
    try:
        _CACHE_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# squeue invocation (user-scoped, one call returns every job)
# ---------------------------------------------------------------------------


def _run_squeue_user() -> Optional[Dict[str, Dict[str, str]]]:
    """Run one `squeue -u $USER` and parse. Returns None on controller failure
    (so callers can keep the previous cache). Returns {} if the user truly has
    no jobs."""
    fmt = _DELIM.join(["%i", "%j", "%T", "%N", "%M", "%l", "%k"])
    user = os.environ.get("USER") or subprocess.getoutput("whoami").strip()
    try:
        result = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", fmt],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    # Exact comments are ownership authority for service and evaluator crash
    # recovery. A commentless compatibility fallback would turn a formatter
    # limitation into false proof that no tagged allocation exists.
    if result.returncode != 0 or (getattr(result, "stderr", "") or "").strip():
        return None

    jobs: Dict[str, Dict[str, str]] = {}
    stdout = result.stdout or ""
    if not stdout.strip():
        return jobs  # genuinely empty

    for line in stdout.splitlines():
        parts = line.split(_DELIM)
        if len(parts) != 7:
            # A successful command with unparseable non-empty output is not
            # an authoritative empty snapshot.
            return None
        jid = parts[0].strip()
        if not jid:
            return None
        jobs[jid] = {
            "name": parts[1],
            "status": parts[2],
            "node": parts[3],
            "elapsed": parts[4],
            "time_limit": parts[5],
            "comment": parts[6],
        }
    return jobs


def get_user_steps_snapshot() -> Optional[Dict[str, Dict[str, str]]]:
    """Return live Slurm job steps for owner/overlay reconciliation.

    This is deliberately separate from the allocation cache because step
    churn is much faster than top-level job churn. ``None`` means the
    controller/query failed; callers must fail closed instead of treating it
    as an empty step set.
    """
    # This cluster's Slurm accepts %T/%t for allocation queries but rejects
    # them under --steps (and may still return rc=0).  squeue itself lists only
    # live steps, so step ID/name/node are sufficient ownership evidence.
    fmt = _DELIM.join(["%i", "%j", "%N"])
    user = os.environ.get("USER") or subprocess.getoutput("whoami").strip()
    try:
        result = subprocess.run(
            ["squeue", "--steps", "-u", user, "-h", "-o", fmt],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or (result.stderr or "").strip():
        return None
    steps: Dict[str, Dict[str, str]] = {}
    for line in (result.stdout or "").splitlines():
        parts = line.split(_DELIM)
        if len(parts) != 3:
            # Nonempty malformed output is uncertainty, never verified
            # absence.  This also protects against unsupported formatter
            # diagnostics emitted on stdout by unusual Slurm wrappers.
            return None
        step_id = parts[0].strip()
        if not step_id:
            return None
        steps[step_id] = {
            "name": parts[1],
            "status": "LIVE",
            "node": parts[2],
        }
    return steps


# ---------------------------------------------------------------------------
# Public cache accessor
# ---------------------------------------------------------------------------


def get_user_jobs_snapshot(
    ttl: Optional[int] = None, force: bool = False,
) -> SchedulerSnapshot:
    """Return all Slurm jobs with explicit scheduler authority metadata.

    At most one `squeue` call per TTL window is made across all processes.
    ttl=None uses the default (env-configurable) TTL.
    force=True bypasses the TTL check and accepts a collapsed concurrent query
    only when it completed after this call began.
    On squeue failure, a dict-compatible non-authoritative snapshot contains
    stale last-known records (if any) with ``UNKNOWN`` status.  Callers that
    can delete, replace, or release resources must check ``.authoritative`` or
    use :func:`get_exact_job_states`.
    """
    ttl_s = _cache_ttl_seconds() if ttl is None else ttl
    now = time.time()
    requested_at = now

    if not force:
        cached = _read_cache()
        if cached is not None and (now - cached.get("refreshed_at", 0)) < ttl_s:
            return SchedulerSnapshot(
                cached.get("jobs") or {},
                authoritative=True,
                refreshed_at=cached.get("refreshed_at"),
                source="cache",
            )

    # Acquire lock and re-check (double-checked locking) — collapses
    # thundering-herd refreshes to one squeue call.
    with _CacheLock(_CACHE_LOCK):
        now = time.time()
        cached = _read_cache()
        if (
            not force
            and cached is not None
            and (now - cached.get("refreshed_at", 0)) < ttl_s
        ):
            return SchedulerSnapshot(
                cached.get("jobs") or {},
                authoritative=True,
                refreshed_at=cached.get("refreshed_at"),
                source="cache",
            )

        # Collapse simultaneous force callers only when another process's
        # authoritative refresh completed after this call began. Reusing a
        # pre-call snapshot is unsafe for submission/crash reconciliation.
        if (
            force
            and cached is not None
            and cached.get("refreshed_at", 0) >= requested_at
        ):
            return SchedulerSnapshot(
                cached.get("jobs") or {},
                authoritative=True,
                refreshed_at=cached.get("refreshed_at"),
                source="concurrent-fresh-cache",
            )

        fresh = _run_squeue_user()
        if fresh is None:
            # Controller failure — keep previous cache if any.
            print(
                "[slurm_utils] squeue refresh failed; using last cached snapshot",
                file=sys.stderr,
            )
            return SchedulerSnapshot(
                (cached or {}).get("jobs") or {},
                authoritative=False,
                refreshed_at=(cached or {}).get("refreshed_at"),
                source="stale-cache" if cached is not None else "query-failed",
            )

        snapshot = {
            "refreshed_at": time.time(),
            "comments_authoritative": True,
            "jobs": fresh,
        }
        try:
            _write_cache_atomic(snapshot)
        except OSError as e:
            print(f"[slurm_utils] cache write failed: {e}", file=sys.stderr)
        return SchedulerSnapshot(
            fresh,
            authoritative=True,
            refreshed_at=snapshot["refreshed_at"],
            source="fresh",
        )


# ---------------------------------------------------------------------------
# Public query helpers — all filter the snapshot (no direct squeue calls)
# ---------------------------------------------------------------------------


def batch_query_jobs(
    job_ids: List[str], ttl: Optional[int] = None,
) -> SchedulerSnapshot:
    """Return requested states while preserving scheduler authority.

    On query uncertainty, every requested ID is returned with ``UNKNOWN``
    status so legacy callers cannot turn a missing dict entry into
    ``NOT_FOUND``.
    """
    if not job_ids:
        return SchedulerSnapshot({}, authoritative=True, source="no-job-ids")
    snap = get_user_jobs_snapshot(ttl=ttl)
    wanted = {str(jid) for jid in job_ids if jid}
    if not snap.authoritative:
        jobs = {
            jid: _unknown_job_info(jid, dict.get(snap, jid))
            for jid in wanted
        }
    else:
        jobs = {
            jid: info
            for jid, info in dict.items(snap)
            if jid in wanted
        }
    return SchedulerSnapshot(
        jobs,
        authoritative=snap.authoritative,
        refreshed_at=snap.refreshed_at,
        source=snap.source,
    )


def get_exact_job_states(
    job_ids: Iterable[str], *, force: bool = True,
) -> Optional[Dict[str, Dict[str, str]]]:
    """Return authoritative live states for exactly ``job_ids``.

    ``None`` means the controller/query is unknown and callers must not infer
    absence, release ownership, or launch replacement capacity.  An empty
    dict is therefore meaningful only after a successful query.
    """
    wanted = {str(job_id) for job_id in job_ids if job_id}
    if not wanted:
        return {}
    if force:
        # Destructive/replacement decisions need a query no older than this
        # call. Collapse callers only when another process refreshed *after*
        # this call began.
        requested_at = time.time()
        with _CacheLock(_CACHE_LOCK):
            cached = _read_cache()
            if cached is not None and cached.get("refreshed_at", 0) >= requested_at:
                snapshot = SchedulerSnapshot(
                    cached.get("jobs") or {},
                    authoritative=True,
                    refreshed_at=cached.get("refreshed_at"),
                    source="concurrent-fresh-cache",
                )
            else:
                fresh = _run_squeue_user()
                if fresh is None:
                    return None
                refreshed_at = time.time()
                cache_record = {
                    "refreshed_at": refreshed_at,
                    "comments_authoritative": True,
                    "jobs": fresh,
                }
                try:
                    _write_cache_atomic(cache_record)
                except OSError as exc:
                    print(
                        f"[slurm_utils] cache write failed: {exc}",
                        file=sys.stderr,
                    )
                snapshot = SchedulerSnapshot(
                    fresh,
                    authoritative=True,
                    refreshed_at=refreshed_at,
                    source="fresh-exact",
                )
    else:
        snapshot = get_user_jobs_snapshot(force=False)
    if not snapshot.authoritative:
        return None
    return {
        job_id: info
        for job_id, info in dict.items(snapshot)
        if job_id in wanted
    }


def get_job_status(job_id: str) -> str:
    """Return status, ``NOT_FOUND`` after proof, or ``UNKNOWN`` on outage."""
    snap = get_user_jobs_snapshot()
    if not snap.authoritative:
        return "UNKNOWN"
    info = dict.get(snap, str(job_id))
    if info:
        return info["status"].upper()
    return "NOT_FOUND"


def get_job_info(job_id: str) -> Optional[Dict[str, str]]:
    snap = get_user_jobs_snapshot()
    if not snap.authoritative:
        return _unknown_job_info(str(job_id), dict.get(snap, str(job_id)))
    return dict.get(snap, str(job_id))


_ALIVE_STATUSES = ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING")


def is_job_alive(job_id: str) -> bool:
    status = get_job_status(job_id)
    return status == "UNKNOWN" or status in _ALIVE_STATUSES


def filter_alive_jobs(job_ids: List[str]) -> List[str]:
    """Return the subset of job IDs still alive, preserving input order."""
    if not job_ids:
        return []
    snap = get_user_jobs_snapshot()
    if not snap.authoritative:
        # A controller outage cannot prove that any tracked job ended.
        return list(job_ids)
    alive_ids = {
        jid for jid, info in snap.items()
        if info["status"].upper() in _ALIVE_STATUSES
    }
    return [jid for jid in job_ids if str(jid) in alive_ids]


def wait_for_job_visible(
    job_id: str, attempts: int = 3, backoff_seconds: float = 2.0,
) -> bool:
    """Force-refresh the snapshot until a just-submitted job_id is visible.

    Covers the brief sbatch → squeue visibility lag. Returns True once the
    job appears in the snapshot, False if it never does within the budget.
    """
    jid = str(job_id)
    query_unknown = False
    for i in range(max(1, attempts)):
        snap = get_user_jobs_snapshot(force=True)
        if snap.authoritative and dict.__contains__(snap, jid):
            return True
        if not snap.authoritative:
            query_unknown = True
        if i < attempts - 1 and backoff_seconds > 0:
            time.sleep(backoff_seconds)
    # Preserve the bool API conservatively: unknown visibility is not proof
    # that submission failed.  Strict callers should use exact-job states.
    return query_unknown


# ---------------------------------------------------------------------------
# scancel (direct, no cache)
# ---------------------------------------------------------------------------


def cancel_job(job_id: str) -> bool:
    result = subprocess.run(
        ["scancel", str(job_id)], capture_output=True, text=True,
    )
    ok = result.returncode == 0
    if ok:
        _invalidate_cache()
    return ok


_CANCEL_CHUNK = 200


def cancel_jobs(job_ids: List[str]) -> int:
    """Cancel many SLURM jobs in batched scancel calls.

    Returns the count of unique IDs the controller accepted. scancel is
    idempotent so jobs already gone still return rc=0.
    """
    unique_ids = list(dict.fromkeys(str(jid) for jid in job_ids if jid))
    if not unique_ids:
        return 0
    sent = 0
    any_ok = False
    for i in range(0, len(unique_ids), _CANCEL_CHUNK):
        chunk = unique_ids[i : i + _CANCEL_CHUNK]
        result = subprocess.run(
            ["scancel", *chunk], capture_output=True, text=True,
        )
        if result.returncode == 0:
            sent += len(chunk)
            any_ok = True
    if any_ok:
        _invalidate_cache()
    return sent


def cancel_pending_job(job_id: str) -> bool:
    """Request cancellation of one exact job only while it is PENDING.

    The scheduler-side state predicate closes the query-to-cancel race: if the
    allocation advances to CONFIGURING or RUNNING after the caller's exact
    snapshot, this command cannot cancel it.
    """
    normalized = str(job_id or "")
    if not normalized.isdigit():
        return False
    result = subprocess.run(
        ["scancel", "--state=PENDING", normalized],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    _invalidate_cache()
    return True


# ---------------------------------------------------------------------------
# Partition resolution
# ---------------------------------------------------------------------------
#
# Configs list partition *candidates* across all clusters we run on
# (preference-ordered, comma-separated). sbatch hard-fails if any name in
# --partition does not exist on the local cluster, so the candidate list is
# filtered against `sinfo` (existence) and `scontrol show partition`
# AllowAccounts (accessibility) before it reaches an #SBATCH line.

_PARTITION_INFO_TTL_SECONDS = 300.0
_partition_cache: Dict[str, tuple] = {}  # key -> (timestamp, value)


def _cached(key: str, fetch) -> Optional[str]:
    now = time.monotonic()
    hit = _partition_cache.get(key)
    if hit is not None and now - hit[0] < _PARTITION_INFO_TTL_SECONDS:
        return hit[1]
    value = fetch()
    _partition_cache[key] = (now, value)
    return value


def _existing_partitions() -> Optional[set]:
    """Set of partition names on this cluster, or None if SLURM is unreachable."""

    def fetch():
        try:
            result = subprocess.run(
                ["sinfo", "-h", "-o", "%R"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        names = set(result.stdout.split())
        return names or None

    return _cached("sinfo", fetch)


def _partition_allows_account(partition: str, account: str) -> bool:
    """True unless the partition's AllowAccounts list provably excludes *account*.

    Fails open: any query error, missing field, or DenyAccounts-style setups
    keep the partition (sbatch remains the final authority).
    """

    def fetch():
        try:
            result = subprocess.run(
                ["scontrol", "show", "partition", partition, "-o"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout if result.returncode == 0 else None

    output = _cached(f"scontrol:{partition}", fetch)
    if not output:
        return True
    for token in output.split():
        if token.startswith("AllowAccounts="):
            allowed = token[len("AllowAccounts="):]
            if allowed.upper() == "ALL":
                return True
            return account in allowed.split(",")
    return True


def resolve_partitions(candidates, account: Optional[str] = None) -> str:
    """Filter a preference-ordered partition candidate list for this cluster.

    ``candidates`` may be a comma-separated string or a list of names.
    Returns a comma-separated string of the candidates that exist locally
    (and, when ``account`` is given, are not excluded by AllowAccounts).
    Order is preserved. Idempotent: already-valid strings pass through.

    SPATIAL_AGENT_SLURM_PARTITION overrides everything verbatim. If SLURM
    cannot be queried (e.g. non-SLURM machine) the candidates pass through
    unchanged. Raises RuntimeError when SLURM is reachable but none of the
    candidates are usable.
    """
    override = os.environ.get("SPATIAL_AGENT_SLURM_PARTITION", "").strip()
    if override:
        return override

    if isinstance(candidates, str):
        wanted = [p.strip() for p in candidates.split(",") if p.strip()]
    else:
        wanted = [str(p).strip() for p in candidates if str(p).strip()]
    if not wanted:
        raise RuntimeError("resolve_partitions: empty candidate list")

    existing = _existing_partitions()
    if existing is None:
        return ",".join(wanted)

    usable = [p for p in wanted if p in existing]
    if account:
        usable = [p for p in usable if _partition_allows_account(p, account)]

    if not usable:
        raise RuntimeError(
            f"None of the candidate partitions {wanted} are usable on this "
            f"cluster (available: {sorted(existing)}"
            + (f", account: {account}" if account else "")
            + "). Set SPATIAL_AGENT_SLURM_PARTITION to override."
        )
    return ",".join(usable)
