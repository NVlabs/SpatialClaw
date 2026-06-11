"""Sample GPU metrics via `ssh <node> nvidia-smi`, tagged per server.

One ssh per unique node (shared among co-located servers), with two
nvidia-smi queries: GPU metrics and compute-apps (PID -> GPU UUID). Each
emitted row is attributed to exactly one server so the dashboard never
blends metrics across jobs that happen to share the same host.
"""

import argparse
import getpass
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .discovery import Server, build_servers, servers_by_node


NVSMI_GPU_FIELDS = (
    "index,uuid,utilization.gpu,memory.used,memory.total,"
    "temperature.gpu,power.draw"
)
NVSMI_APPS_FIELDS = "pid,gpu_uuid"
PS_FIELDS = "pid,ppid"

# AF_UNIX sockets don't work reliably on NFS/Lustre, so keep control sockets
# on local /tmp. %C is a 16-hex hash of user@host:port — keeps the socket
# path well under the 108-byte sun_path limit regardless of node name length.
_SSH_CONTROL_DIR = f"/tmp/{getpass.getuser()}-gpu-dashboard-cm"
try:
    os.makedirs(_SSH_CONTROL_DIR, mode=0o700, exist_ok=True)
except OSError:
    pass

SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=no",
    "-o", "ConnectTimeout=10",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ControlMaster=auto",
    "-o", f"ControlPath={_SSH_CONTROL_DIR}/%C",
    "-o", "ControlPersist=10m",
]


def _run_ssh(host: str, remote_cmd: str, timeout: int) -> Optional[str]:
    cmd = ["ssh", *SSH_OPTS, host, remote_cmd]
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout


def _parse_gpu_csv(text: str) -> dict[int, dict]:
    """Return {gpu_index: {uuid, util_pct, mem_used_mb, mem_total_mb, temp_c, power_w}}."""
    out: dict[int, dict] = {}
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 7:
            continue
        try:
            idx = int(parts[0])
        except (ValueError, IndexError):
            continue

        def _f(v: str) -> Optional[float]:
            return float(v) if v and v != "[N/A]" else None

        def _i(v: str) -> Optional[int]:
            return int(float(v)) if v and v != "[N/A]" else None

        out[idx] = {
            "uuid": parts[1],
            "util_pct":    _f(parts[2]),
            "mem_used_mb": _i(parts[3]),
            "mem_total_mb": _i(parts[4]),
            "temp_c":      _f(parts[5]),
            "power_w":     _f(parts[6]),
        }
    return out


def _parse_apps_csv(text: str) -> dict[str, list[str]]:
    """Return {pid: [gpu_uuid, ...]} (a PID can occupy multiple GPUs)."""
    out: dict[str, list[str]] = {}
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        pid, uuid = parts[0], parts[1]
        if not pid or not uuid:
            continue
        out.setdefault(pid, []).append(uuid)
    return out


def _parse_ps(text: str) -> dict[str, str]:
    """Return {pid: ppid}. Tolerates a header line or extra whitespace."""
    out: dict[str, str] = {}
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        pid, ppid = parts[0], parts[1]
        if not pid.isdigit() or not ppid.isdigit():
            continue
        out[pid] = ppid
    return out


def _descendants(root_pid: str, pid_to_ppid: dict[str, str]) -> set[str]:
    """Return root_pid plus every PID whose ancestor chain includes it."""
    children: dict[str, list[str]] = {}
    for pid, ppid in pid_to_ppid.items():
        children.setdefault(ppid, []).append(pid)
    out = {root_pid}
    stack = [root_pid]
    while stack:
        cur = stack.pop()
        for c in children.get(cur, ()):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def _gpus_for_server(
    srv: Server,
    gpu_map: dict[int, dict],
    pid_to_uuids: dict[str, list[str]],
    pid_to_ppid: dict[str, str],
) -> list[int]:
    """Return the physical GPU indices on this host that belong to srv.

    Walks the process tree from srv.pid (the launcher recorded in
    serve.json / gpu_server.json) to catch subprocess workers (vLLM's
    VLLM::Worker_TP* processes hold the GPU context, not the parent).
    srv.gpus_hint is NOT trusted as physical indices: serve.json records
    the process-visible CUDA_VISIBLE_DEVICES values, which SLURM remaps
    to 0..N-1 regardless of which physical GPUs were assigned.
    """
    if not srv.pid:
        return []
    owned_pids = _descendants(str(srv.pid), pid_to_ppid)
    uuids: set[str] = set()
    for pid in owned_pids:
        uuids.update(pid_to_uuids.get(pid, []))
    if not uuids:
        return []
    uuid_to_idx = {g["uuid"]: i for i, g in gpu_map.items() if g.get("uuid")}
    return sorted({uuid_to_idx[u] for u in uuids if u in uuid_to_idx})


def sample_host(
    host: str, servers: list[Server], ts: int, timeout: int = 15,
) -> list[dict]:
    """SSH once; emit rows tagged per (server, owned GPU)."""
    gpu_text = _run_ssh(
        host,
        f"nvidia-smi --query-gpu={NVSMI_GPU_FIELDS} --format=csv,noheader,nounits",
        timeout,
    )
    if gpu_text is None:
        return []
    apps_text = _run_ssh(
        host,
        f"nvidia-smi --query-compute-apps={NVSMI_APPS_FIELDS} --format=csv,noheader,nounits",
        timeout,
    ) or ""
    ps_text = _run_ssh(host, f"ps -eo {PS_FIELDS}", timeout) or ""

    gpu_map = _parse_gpu_csv(gpu_text)
    pid_to_uuids = _parse_apps_csv(apps_text)
    pid_to_ppid = _parse_ps(ps_text)

    rows: list[dict] = []
    for srv in servers:
        for idx in _gpus_for_server(srv, gpu_map, pid_to_uuids, pid_to_ppid):
            g = gpu_map.get(idx)
            if not g:
                continue
            rows.append({
                "ts": ts,
                "node": srv.node,
                "ip": srv.ip,
                "gpu_index": idx,
                "util_pct": g["util_pct"],
                "mem_used_mb": g["mem_used_mb"],
                "mem_total_mb": g["mem_total_mb"],
                "temp_c": g["temp_c"],
                "power_w": g["power_w"],
                "service_type": srv.service_type,
                "service_id": srv.server_id,
                "slurm_job_id": srv.slurm_job_id,
            })
    return rows


def sample_all(
    servers: list[Server], ts: int, max_workers: int = 16, timeout: int = 15,
) -> list[dict]:
    by_node = servers_by_node(servers)
    if not by_node:
        return []
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(sample_host, host, srvs, ts, timeout): host
            for host, srvs in by_node.items()
        }
        for fut in as_completed(futs):
            try:
                rows = fut.result()
            except Exception:
                rows = []
            all_rows.extend(rows)
    return all_rows


def _main() -> int:
    parser = argparse.ArgumentParser(description="One-shot GPU sampler")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent.parent.parent),
    )
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    servers = build_servers(project_root)
    if not servers:
        print(
            "No servers registered in serve.json / gpu_server.json",
            file=sys.stderr,
        )
        return 1

    by_node = servers_by_node(servers)
    print(
        f"Discovered {len(servers)} server(s) on {len(by_node)} node(s):",
        file=sys.stderr,
    )
    for host, srvs in by_node.items():
        print(f"  {host}:", file=sys.stderr)
        for s in srvs:
            hint = s.gpus_hint if s.gpus_hint else f"pid={s.pid}"
            print(
                f"    [{s.service_type}] {s.server_id} "
                f"({s.display_label}) {hint}",
                file=sys.stderr,
            )

    rows = sample_all(servers, ts=int(time.time()), timeout=args.timeout)
    if not rows:
        print("No samples collected.", file=sys.stderr)
        return 1

    print(f"{'node':<20} {'server':<40} {'gpu':>3} {'util%':>6} {'mem_used':>10}")
    for r in rows:
        print(
            f"{r['node']:<20} {r['service_id']:<40} {r['gpu_index']:>3} "
            f"{r['util_pct']!s:>6} {r['mem_used_mb']!s:>10}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
