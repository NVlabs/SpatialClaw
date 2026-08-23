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
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

_DELIM = "\x1f"  # ASCII unit separator — safer than '|' in job names

# When force=True callers pile up (e.g. 50 chains all doing
# wait_for_job_visible at startup), only one of them should actually run
# squeue if a fresh result already exists. Window must be << sbatch→squeue
# lag so a caller whose job isn't yet visible still retries correctly.
_FORCE_COLLAPSE_SECONDS = 1.5

# Located next to the other shared state JSONs.
_CACHE_FILE = (
    Path(__file__).resolve().parent.parent / "logs" / "squeue_cache.json"
)
_CACHE_LOCK = Path(str(_CACHE_FILE) + ".lock")


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
        if isinstance(data, dict) and "refreshed_at" in data and "jobs" in data:
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
    fmt = _DELIM.join(["%i", "%j", "%T", "%N", "%M", "%l"])
    user = os.environ.get("USER") or subprocess.getoutput("whoami").strip()
    try:
        result = subprocess.run(
            ["squeue", "-u", user, "-h", "-o", fmt],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    jobs: Dict[str, Dict[str, str]] = {}
    stdout = result.stdout or ""
    if not stdout.strip():
        return jobs  # genuinely empty

    for line in stdout.splitlines():
        parts = line.split(_DELIM)
        if len(parts) >= 6:
            jid = parts[0].strip()
            if not jid:
                continue
            jobs[jid] = {
                "name": parts[1],
                "status": parts[2],
                "node": parts[3],
                "elapsed": parts[4],
                "time_limit": parts[5],
            }
    return jobs


# ---------------------------------------------------------------------------
# Public cache accessor
# ---------------------------------------------------------------------------


def get_user_jobs_snapshot(
    ttl: Optional[int] = None, force: bool = False,
) -> Dict[str, Dict[str, str]]:
    """Return a dict of all SLURM jobs for the current user, cached.

    At most one `squeue` call per TTL window is made across all processes.
    ttl=None uses the default (env-configurable) TTL.
    force=True bypasses the TTL check and unconditionally refreshes.
    On squeue failure, returns the last known cache rather than empty, so
    chains don't mis-classify jobs as NOT_FOUND due to controller blips.
    """
    ttl_s = _cache_ttl_seconds() if ttl is None else ttl
    now = time.time()

    if not force:
        cached = _read_cache()
        if cached is not None and (now - cached.get("refreshed_at", 0)) < ttl_s:
            return dict(cached.get("jobs") or {})

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
            return dict(cached.get("jobs") or {})

        # Even with force=True, if another process just refreshed within
        # _FORCE_COLLAPSE_SECONDS, reuse their result. This keeps 50
        # simultaneous wait_for_job_visible() calls from issuing 50 back-
        # to-back squeues. Window is small enough that an sbatch
        # immediately before the collapsed refresh still has time to
        # appear via wait_for_job_visible's retry loop.
        if (
            force
            and cached is not None
            and (now - cached.get("refreshed_at", 0)) < _FORCE_COLLAPSE_SECONDS
        ):
            return dict(cached.get("jobs") or {})

        fresh = _run_squeue_user()
        if fresh is None:
            # Controller failure — keep previous cache if any.
            print(
                "[slurm_utils] squeue refresh failed; using last cached snapshot",
                file=sys.stderr,
            )
            return dict((cached or {}).get("jobs") or {})

        snapshot = {"refreshed_at": time.time(), "jobs": fresh}
        try:
            _write_cache_atomic(snapshot)
        except OSError as e:
            print(f"[slurm_utils] cache write failed: {e}", file=sys.stderr)
        return dict(fresh)


# ---------------------------------------------------------------------------
# Public query helpers — all filter the snapshot (no direct squeue calls)
# ---------------------------------------------------------------------------


def batch_query_jobs(
    job_ids: List[str], ttl: Optional[int] = None,
) -> Dict[str, Dict[str, str]]:
    """Return info for the requested job IDs, filtered from the cached snapshot."""
    if not job_ids:
        return {}
    snap = get_user_jobs_snapshot(ttl=ttl)
    wanted = {str(jid) for jid in job_ids if jid}
    return {jid: info for jid, info in snap.items() if jid in wanted}


def get_job_status(job_id: str) -> str:
    """Return SLURM status for a job, or NOT_FOUND if the cache has no record."""
    snap = get_user_jobs_snapshot()
    info = snap.get(str(job_id))
    if info:
        return info["status"].upper()
    return "NOT_FOUND"


def get_job_info(job_id: str) -> Optional[Dict[str, str]]:
    snap = get_user_jobs_snapshot()
    return snap.get(str(job_id))


_ALIVE_STATUSES = ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING")


def is_job_alive(job_id: str) -> bool:
    return get_job_status(job_id) in _ALIVE_STATUSES


def filter_alive_jobs(job_ids: List[str]) -> List[str]:
    """Return the subset of job IDs still alive, preserving input order."""
    if not job_ids:
        return []
    snap = get_user_jobs_snapshot()
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
    for i in range(max(1, attempts)):
        snap = get_user_jobs_snapshot(force=True)
        if jid in snap:
            return True
        if i < attempts - 1 and backoff_seconds > 0:
            time.sleep(backoff_seconds)
    return False


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
