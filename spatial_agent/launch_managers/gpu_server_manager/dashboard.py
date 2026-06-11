"""Dashboard: render status of all GPU servers."""

import json
from pathlib import Path
from typing import Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from spatial_agent.launch_managers.slurm_utils import batch_query_jobs
from spatial_agent.launch_managers.gpu_server_manager.state import (
    GPUServerState,
    GPUServerStateManager,
)

_ALIVE_STATUSES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}


class GPUServerDashboard:

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.registry_file = project_root / "spatial_agent" / "logs" / "gpu_server.json"
        self.state_manager = GPUServerStateManager(project_root)
        self.console = Console()

    def render(self) -> None:
        """Render the GPU server dashboard."""
        servers = self.state_manager.list_servers()
        registry = self._load_registry()

        # Clean up dead servers
        dead = self.state_manager.cleanup_dead_servers()
        if dead:
            servers = self.state_manager.list_servers()

        # Collect all SLURM job IDs
        all_job_ids = set()
        for srv in servers:
            all_job_ids.update(srv.slurm_job_ids)
        for uid, info in registry.items():
            sjid = info.get("slurm_job_id")
            if sjid:
                all_job_ids.add(sjid)

        job_info_map = batch_query_jobs(list(all_job_ids)) if all_job_ids else {}

        # Build endpoint map from registry
        endpoint_map: Dict[str, dict] = {}
        for uid, info in registry.items():
            sjid = info.get("slurm_job_id")
            if sjid:
                endpoint_map[sjid] = {
                    "ip": info.get("ip", "?"),
                    "http_port": info.get("http_port", "?"),
                    "num_gpus": info.get("num_gpus", "?"),
                    "tools": info.get("tools", []),
                    "backend": info.get("reconstruct_backend", "?"),
                }

        accounted_job_ids = set()

        if servers:
            table = Table(
                title="Managed GPU Servers",
                title_style="bold cyan",
                border_style="cyan",
                show_lines=True,
                padding=(0, 1),
            )
            table.add_column("#", style="dim", width=3, justify="right")
            table.add_column("Status", min_width=10, justify="center")
            table.add_column("GPUs", width=5, justify="center")
            table.add_column("Backend", min_width=6, justify="center")
            table.add_column("SLURM Job", min_width=10)
            table.add_column("Node", min_width=12)
            table.add_column("HTTP Endpoint", min_width=22)
            table.add_column("Uptime", min_width=8, justify="right")
            table.add_column("Next Job", min_width=10, justify="center")
            table.add_column("Account", style="dim", min_width=14)

            for idx, srv in enumerate(servers, 1):
                alive = self.state_manager.is_server_alive(srv)
                accounted_job_ids.update(srv.slurm_job_ids)

                running_jobs = []
                pending_jobs = []
                for jid in srv.slurm_job_ids:
                    info = job_info_map.get(jid)
                    if not info:
                        continue
                    st = info["status"].upper()
                    if st == "RUNNING":
                        running_jobs.append((jid, info))
                    elif st in _ALIVE_STATUSES:
                        pending_jobs.append((jid, info))

                if not alive and not running_jobs and not pending_jobs:
                    status = Text("DEAD", style="bold red")
                    table.add_row(
                        str(idx), status, str(srv.gpus), srv.reconstruct_backend,
                        "-", "-", "-", "-", "-", srv.account,
                    )
                    continue

                if running_jobs:
                    jid, info = running_jobs[0]
                    ep = endpoint_map.get(jid, {})
                    endpoint = f"{ep.get('ip', '?')}:{ep.get('http_port', '?')}" if ep else "-"
                    uptime = info.get("elapsed", "-")
                    status = Text("RUNNING", style="bold green")
                    node = info.get("node", "-")

                    next_job = "-"
                    if pending_jobs:
                        next_job = Text("PENDING", style="yellow")
                    elif len(running_jobs) > 1:
                        next_job = Text("OVERLAP", style="cyan")

                    table.add_row(
                        str(idx), status, str(srv.gpus), srv.reconstruct_backend,
                        jid, node, endpoint, uptime, next_job, srv.account,
                    )
                elif pending_jobs:
                    jid, info = pending_jobs[0]
                    status = Text("PENDING", style="bold yellow")
                    table.add_row(
                        str(idx), status, str(srv.gpus), srv.reconstruct_backend,
                        jid, info.get("node", "-"), "-", "-", "-", srv.account,
                    )
                else:
                    status = Text("STARTING", style="bold yellow")
                    table.add_row(
                        str(idx), status, str(srv.gpus), srv.reconstruct_backend,
                        "-", "-", "-", "-", "-", srv.account,
                    )

            self.console.print(table)
            self.console.print()

        # Unmanaged servers (in registry but not tracked by manager)
        unmanaged = []
        for uid, info in registry.items():
            sjid = info.get("slurm_job_id")
            if sjid and sjid not in accounted_job_ids:
                ji = job_info_map.get(sjid)
                if ji and ji["status"].upper() in _ALIVE_STATUSES:
                    unmanaged.append((uid, info, ji))

        if unmanaged:
            table = Table(
                title="Unmanaged GPU Servers",
                title_style="bold yellow",
                border_style="yellow",
                show_lines=True,
                padding=(0, 1),
            )
            table.add_column("#", style="dim", width=3, justify="right")
            table.add_column("Status", min_width=10, justify="center")
            table.add_column("GPUs", width=5, justify="center")
            table.add_column("SLURM Job", min_width=10)
            table.add_column("Node", min_width=12)
            table.add_column("HTTP Endpoint", min_width=22)
            table.add_column("Uptime", min_width=8, justify="right")

            for idx, (uid, info, ji) in enumerate(unmanaged, 1):
                sjid = info.get("slurm_job_id", "-")
                ep = f"{info.get('ip', '?')}:{info.get('http_port', '?')}"
                uptime = ji.get("elapsed", "-")
                node = ji.get("node", "-")
                status_str = ji.get("status", "UNKNOWN")

                if status_str == "RUNNING":
                    status = Text("RUNNING", style="bold green")
                else:
                    status = Text(status_str, style="yellow")

                table.add_row(str(idx), status, str(info.get("num_gpus", "?")), sjid, node, ep, uptime)

            self.console.print(table)
            self.console.print()

        if not servers and not unmanaged:
            self.console.print(
                Panel(
                    "[dim]No GPU servers running[/dim]",
                    title="GPU Server Dashboard",
                    border_style="dim",
                ),
            )

    def _load_registry(self) -> Dict:
        if not self.registry_file.exists():
            return {}
        try:
            with open(self.registry_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
