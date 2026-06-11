"""Background sampler thread.

Runs inside the chain-manager process (outside SLURM) so that history
collection survives the 24 h web-server restart.
"""

import argparse
import datetime
import sys
import threading
import time
from pathlib import Path

from . import agent_counter, collector
from .discovery import build_servers
from .storage import GpuDashboardDB


def _log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[sampler {ts}] {msg}", flush=True)


def sample_once(project_root: Path, db: GpuDashboardDB, timeout: int = 15) -> tuple[int, int]:
    """Run one sample tick. Returns (gpu_rows_written, agent_total)."""
    ts = int(time.time())
    servers = build_servers(project_root)
    gpu_rows = collector.sample_all(servers, ts=ts, timeout=timeout)
    written = db.insert_gpu_samples(gpu_rows)

    snap = agent_counter.snapshot(project_root)
    db.insert_agent_sample(
        ts=ts,
        total=snap.total,
        by_benchmark=snap.by_benchmark,
        by_model=snap.by_model,
        experiment_ids=snap.experiment_ids,
    )
    return written, snap.total


_PRUNE_EVERY_SEC = 60   # run prune at most once a minute


class SamplerThread(threading.Thread):
    def __init__(
        self,
        project_root: Path,
        db_path: str,
        interval_sec: int = 60,
        history_sec: int = 3600,
        node_timeout: int = 15,
    ):
        super().__init__(daemon=True, name="gpu-dashboard-sampler")
        self.project_root = Path(project_root)
        self.db_path = str(db_path)
        self.interval_sec = max(1, int(interval_sec))
        self.history_sec = max(60, int(history_sec))
        self.node_timeout = max(5, int(node_timeout))
        self._stop_evt = threading.Event()
        self._last_prune_ts = 0.0

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        db = GpuDashboardDB(self.db_path)
        _log(
            f"starting (interval={self.interval_sec}s, "
            f"history={self.history_sec}s, db={self.db_path})"
        )
        while not self._stop_evt.is_set():
            tick_start = time.time()
            try:
                written, agent_total = sample_once(
                    self.project_root, db, timeout=self.node_timeout
                )
                _log(f"tick: wrote {written} gpu row(s), agents_running={agent_total}")
            except Exception as e:
                _log(f"tick failed: {type(e).__name__}: {e}")

            self._maybe_prune(db)

            elapsed = time.time() - tick_start
            remaining = max(0.0, self.interval_sec - elapsed)
            if self._stop_evt.wait(remaining):
                break
        _log("stopped")

    def _maybe_prune(self, db: GpuDashboardDB) -> None:
        now = time.time()
        if now - self._last_prune_ts < _PRUNE_EVERY_SEC:
            return
        try:
            g, a = db.prune(self.history_sec)
            if g or a:
                _log(
                    f"pruned {g} gpu row(s), {a} agent row(s) "
                    f"older than {self.history_sec}s"
                )
        except Exception as e:
            _log(f"prune failed: {e}")
        self._last_prune_ts = now


def _main() -> int:
    parser = argparse.ArgumentParser(description="GPU dashboard sampler")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent.parent.parent),
    )
    parser.add_argument("--db", required=True, help="SQLite DB path")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between ticks")
    parser.add_argument("--history-sec", type=int, default=3600)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--once", action="store_true", help="Run a single tick and exit")
    args = parser.parse_args()

    project_root = Path(args.project_root)

    if args.once:
        db = GpuDashboardDB(args.db)
        written, total = sample_once(project_root, db, timeout=args.timeout)
        print(f"Wrote {written} GPU row(s); agents_running={total}")
        return 0

    thr = SamplerThread(
        project_root=project_root,
        db_path=args.db,
        interval_sec=args.interval,
        history_sec=args.history_sec,
        node_timeout=args.timeout,
    )
    thr.start()
    try:
        while thr.is_alive():
            thr.join(timeout=1.0)
    except KeyboardInterrupt:
        thr.stop()
        thr.join(timeout=10)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
