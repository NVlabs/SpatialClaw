"""SQLite storage for GPU samples and agent counts."""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS gpu_samples (
  ts            INTEGER NOT NULL,
  node          TEXT    NOT NULL,
  ip            TEXT,
  gpu_index     INTEGER NOT NULL,
  util_pct      REAL,
  mem_used_mb   INTEGER,
  mem_total_mb  INTEGER,
  temp_c        REAL,
  power_w       REAL,
  service_type  TEXT,
  service_id    TEXT,
  slurm_job_id  TEXT
);
CREATE INDEX IF NOT EXISTS ix_gpu_ts      ON gpu_samples(ts);
CREATE INDEX IF NOT EXISTS ix_gpu_node_ts ON gpu_samples(node, ts);

CREATE TABLE IF NOT EXISTS agent_samples (
  ts             INTEGER PRIMARY KEY,
  total          INTEGER NOT NULL,
  by_benchmark   TEXT NOT NULL,
  by_model       TEXT NOT NULL,
  experiment_ids TEXT
);
CREATE INDEX IF NOT EXISTS ix_agent_ts ON agent_samples(ts);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    # WAL mode is unusable on Lustre/NFS (shared-memory index file fails with
    # "locking protocol"). Fall back to TRUNCATE journal — writer holds a
    # brief fcntl lock during INSERT, readers wait via busy_timeout. Since we
    # have exactly one writer (the sampler) and the web server is read-only,
    # contention is negligible at 1 sample/min.
    try:
        conn.execute("PRAGMA journal_mode=TRUNCATE")
    except sqlite3.OperationalError:
        pass
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn


class GpuDashboardDB:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with _connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return _connect(self.db_path)

    def insert_gpu_samples(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        tuples = [
            (
                r["ts"], r["node"], r.get("ip"), r["gpu_index"],
                r.get("util_pct"), r.get("mem_used_mb"), r.get("mem_total_mb"),
                r.get("temp_c"), r.get("power_w"),
                r.get("service_type"), r.get("service_id"), r.get("slurm_job_id"),
            )
            for r in rows
        ]
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO gpu_samples"
                " (ts, node, ip, gpu_index, util_pct, mem_used_mb, mem_total_mb,"
                "  temp_c, power_w, service_type, service_id, slurm_job_id)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                tuples,
            )
        return len(tuples)

    def insert_agent_sample(
        self,
        ts: int,
        total: int,
        by_benchmark: dict[str, int],
        by_model: dict[str, int],
        experiment_ids: list[str],
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO agent_samples"
                " (ts, total, by_benchmark, by_model, experiment_ids)"
                " VALUES (?,?,?,?,?)",
                (
                    ts,
                    total,
                    json.dumps(by_benchmark, sort_keys=True),
                    json.dumps(by_model, sort_keys=True),
                    json.dumps(experiment_ids),
                ),
            )

    def latest_samples(self, max_age_sec: int = 600) -> list[dict]:
        """Most recent sample per (service_id, gpu_index), within max_age_sec.

        Keying by service_id (not node) keeps metrics attributed to the
        owning job even when multiple jobs share a physical host.
        """
        cutoff = int(time.time()) - max(0, int(max_age_sec))
        sql = """
            SELECT g.*
            FROM gpu_samples g
            JOIN (
                SELECT service_id, gpu_index, MAX(ts) AS mts
                FROM gpu_samples
                WHERE ts >= ?
                GROUP BY service_id, gpu_index
            ) last
              ON g.service_id = last.service_id
             AND g.gpu_index  = last.gpu_index
             AND g.ts         = last.mts
            ORDER BY g.service_id, g.gpu_index
        """
        with self._conn() as conn:
            rows = conn.execute(sql, [cutoff]).fetchall()
        return [dict(r) for r in rows]

    def gpu_history(
        self,
        since: int,
        until: Optional[int] = None,
        node: Optional[str] = None,
        service_type: Optional[str] = None,
        service_id: Optional[str] = None,
        bucket_sec: int = 300,
    ) -> list[dict]:
        """Time-bucketed GPU util averaged per server."""
        until = until if until is not None else int(time.time())
        bucket_sec = max(1, int(bucket_sec))
        params: list[Any] = [bucket_sec, bucket_sec, since, until]
        where = ["ts >= ?", "ts <= ?"]
        if node:
            where.append("node = ?")
            params.append(node)
        if service_type:
            where.append("service_type = ?")
            params.append(service_type)
        if service_id:
            where.append("service_id = ?")
            params.append(service_id)
        sql = f"""
            SELECT (ts / ?) * ? AS bucket_ts,
                   service_id,
                   node,
                   AVG(util_pct)     AS util_pct,
                   AVG(mem_used_mb)  AS mem_used_mb,
                   AVG(mem_total_mb) AS mem_total_mb,
                   AVG(temp_c)       AS temp_c,
                   AVG(power_w)      AS power_w,
                   COUNT(DISTINCT gpu_index) AS gpu_count
            FROM gpu_samples
            WHERE {' AND '.join(where)}
            GROUP BY bucket_ts, service_id, node
            ORDER BY bucket_ts, service_id
        """
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def gpu_history_per_gpu_server(
        self,
        since: int,
        until: Optional[int],
        service_id: str,
        bucket_sec: int = 300,
    ) -> list[dict]:
        """Per-GPU time-bucketed metrics for a single server."""
        until = until if until is not None else int(time.time())
        bucket_sec = max(1, int(bucket_sec))
        sql = """
            SELECT (ts / ?) * ? AS bucket_ts,
                   gpu_index,
                   AVG(util_pct)    AS util_pct,
                   AVG(mem_used_mb) AS mem_used_mb,
                   AVG(power_w)     AS power_w
            FROM gpu_samples
            WHERE ts >= ? AND ts <= ? AND service_id = ?
            GROUP BY bucket_ts, gpu_index
            ORDER BY bucket_ts, gpu_index
        """
        with self._conn() as conn:
            rows = conn.execute(
                sql, [bucket_sec, bucket_sec, since, until, service_id]
            ).fetchall()
        return [dict(r) for r in rows]

    def gpu_history_per_gpu(
        self,
        since: int,
        until: Optional[int],
        node: str,
        bucket_sec: int = 300,
    ) -> list[dict]:
        until = until if until is not None else int(time.time())
        bucket_sec = max(1, int(bucket_sec))
        sql = """
            SELECT (ts / ?) * ? AS bucket_ts,
                   gpu_index,
                   AVG(util_pct)    AS util_pct,
                   AVG(mem_used_mb) AS mem_used_mb,
                   AVG(power_w)     AS power_w
            FROM gpu_samples
            WHERE ts >= ? AND ts <= ? AND node = ?
            GROUP BY bucket_ts, gpu_index
            ORDER BY bucket_ts, gpu_index
        """
        with self._conn() as conn:
            rows = conn.execute(
                sql, [bucket_sec, bucket_sec, since, until, node]
            ).fetchall()
        return [dict(r) for r in rows]

    def agent_history(
        self, since: int, until: Optional[int] = None, bucket_sec: int = 300,
    ) -> list[dict]:
        until = until if until is not None else int(time.time())
        bucket_sec = max(1, int(bucket_sec))
        sql = """
            SELECT (ts / ?) * ? AS bucket_ts,
                   AVG(total)    AS total,
                   MAX(total)    AS total_max
            FROM agent_samples
            WHERE ts >= ? AND ts <= ?
            GROUP BY bucket_ts
            ORDER BY bucket_ts
        """
        with self._conn() as conn:
            rows = conn.execute(
                sql, [bucket_sec, bucket_sec, since, until]
            ).fetchall()
        return [dict(r) for r in rows]

    def agent_breakdown_series(
        self, since: int, until: Optional[int] = None,
    ) -> list[dict]:
        """Raw agent samples with JSON breakdowns (for stacked-area rendering)."""
        until = until if until is not None else int(time.time())
        sql = """
            SELECT ts, total, by_benchmark, by_model, experiment_ids
            FROM agent_samples
            WHERE ts >= ? AND ts <= ?
            ORDER BY ts
        """
        with self._conn() as conn:
            rows = conn.execute(sql, [since, until]).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for key in ("by_benchmark", "by_model", "experiment_ids"):
                try:
                    d[key] = json.loads(d[key]) if d.get(key) else None
                except (TypeError, json.JSONDecodeError):
                    d[key] = None
            out.append(d)
        return out

    def distinct_nodes(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT node FROM gpu_samples ORDER BY node"
            ).fetchall()
        return [r["node"] for r in rows]

    def distinct_servers(self, max_age_sec: int = 86400) -> list[dict]:
        """Recently-seen servers for the history-page dropdown."""
        cutoff = int(time.time()) - max(0, int(max_age_sec))
        sql = """
            SELECT service_id,
                   MAX(service_type) AS service_type,
                   MAX(node)         AS node,
                   MAX(ts)           AS last_ts
            FROM gpu_samples
            WHERE ts >= ? AND service_id IS NOT NULL AND service_id != ''
            GROUP BY service_id
            ORDER BY service_id
        """
        with self._conn() as conn:
            rows = conn.execute(sql, [cutoff]).fetchall()
        return [dict(r) for r in rows]

    def prune(self, older_than_sec: int) -> tuple[int, int]:
        cutoff = int(time.time()) - max(0, int(older_than_sec))
        with self._conn() as conn:
            g = conn.execute("DELETE FROM gpu_samples WHERE ts < ?", [cutoff]).rowcount
            a = conn.execute("DELETE FROM agent_samples WHERE ts < ?", [cutoff]).rowcount
        return (g or 0, a or 0)
