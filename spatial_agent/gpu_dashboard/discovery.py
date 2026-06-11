"""Discover per-server GPU monitoring entries.

The dashboard is *job-wise*: one entry per vLLM / GPU-tool server, each
owning a known set of GPUs on its SLURM node. Multiple servers may share a
physical host — the collector groups by node so each host is ssh'd exactly
once per tick, but the rows it emits are always tagged to a single server.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from spatial_agent.launch_managers import slurm_utils


@dataclass
class Server:
    server_id: str                # stable id for UI + DB
    service_type: str             # "vllm" | "gpu_tool"
    node: str                     # SLURM node name (preferred) or IP fallback
    ip: str
    slurm_job_id: str
    pid: str = ""                 # compute-apps lookup key when gpus_hint is missing
    model: str = ""               # for vllm
    tools: list[str] = field(default_factory=list)  # for gpu_tool
    gpus_hint: Optional[list[int]] = None           # explicit GPU indices if known
    num_gpus: int = 0             # expected count (for card layout)

    @property
    def display_label(self) -> str:
        if self.service_type == "vllm":
            return self.model or "vllm"
        return ",".join(self.tools) or "gpu_tool"


def _load_json(path: Path) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _resolve_node_names(job_ids: list[str]) -> dict[str, str]:
    """slurm_job_id -> node name via slurm_utils cached snapshot."""
    if not job_ids:
        return {}
    try:
        info = slurm_utils.batch_query_jobs(job_ids)
    except Exception:
        info = {}
    return {jid: (info.get(jid, {}).get("node") or "") for jid in job_ids}


def build_servers(project_root: Path) -> list[Server]:
    logs = project_root / "spatial_agent" / "logs"
    serve = _load_json(logs / "serve.json")
    gpusv = _load_json(logs / "gpu_server.json")

    job_ids: list[str] = []
    for _, servers in serve.items():
        for _, s in servers.items():
            jid = str(s.get("slurm_job_id") or "")
            if jid:
                job_ids.append(jid)
    for _, s in gpusv.items():
        jid = str(s.get("slurm_job_id") or "")
        if jid:
            job_ids.append(jid)
    node_for = _resolve_node_names(list(dict.fromkeys(job_ids)))

    out: list[Server] = []

    for model, servers in serve.items():
        for sid, s in servers.items():
            jid = str(s.get("slurm_job_id") or "")
            ip = str(s.get("ip") or "")
            node = node_for.get(jid, "") or ip
            if not node:
                continue
            gpus = [int(g) for g in (s.get("gpus") or [])]
            short = sid[:12] if sid else "00"
            out.append(Server(
                server_id=f"vllm:{model}:{short}",
                service_type="vllm",
                node=node,
                ip=ip,
                slurm_job_id=jid,
                pid=str(s.get("pid") or ""),
                model=model,
                tools=[],
                gpus_hint=gpus if gpus else None,
                num_gpus=len(gpus) or int(s.get("tp") or 0) or 1,
            ))

    for h, s in gpusv.items():
        jid = str(s.get("slurm_job_id") or "")
        ip = str(s.get("ip") or "")
        node = node_for.get(jid, "") or ip
        if not node:
            continue
        out.append(Server(
            server_id=f"gpu_tool:{h}",
            service_type="gpu_tool",
            node=node,
            ip=ip,
            slurm_job_id=jid,
            pid=str(s.get("pid") or ""),
            model="",
            tools=list(s.get("tools") or []),
            gpus_hint=None,
            num_gpus=int(s.get("num_gpus") or 1),
        ))

    return out


def servers_by_node(servers: list[Server]) -> dict[str, list[Server]]:
    out: dict[str, list[Server]] = {}
    for srv in servers:
        out.setdefault(srv.node, []).append(srv)
    return out


def summarize_servers(servers: list[Server]) -> list[dict]:
    return [
        {
            "server_id": s.server_id,
            "service_type": s.service_type,
            "node": s.node,
            "ip": s.ip,
            "slurm_job_id": s.slurm_job_id,
            "pid": s.pid,
            "model": s.model,
            "tools": s.tools,
            "gpus_hint": s.gpus_hint,
            "num_gpus": s.num_gpus,
            "display_label": s.display_label,
        }
        for s in servers
    ]
