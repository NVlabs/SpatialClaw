"""Dashboard: render status of all vLLM servers."""

import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from spatial_agent.launch_managers.slurm_utils import batch_query_jobs
from spatial_agent.launch_managers.vllm_manager.state import ChainState, ChainStateManager

_ALIVE_STATUSES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}


class Dashboard:

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.serve_file = project_root / "spatial_agent" / "logs" / "serve.json"
        self.state_manager = ChainStateManager(project_root)
        self.console = Console()

    def render(self) -> None:
        """Render the full dashboard."""
        chains = self.state_manager.list_chains()
        registry = self._load_serve_registry()

        # Clean up dead chains silently
        dead = self.state_manager.cleanup_dead_chains()
        if dead:
            chains = self.state_manager.list_chains()

        # Collect ALL slurm job IDs we need to query, then do ONE squeue call
        all_job_ids = set()
        for chain in chains:
            all_job_ids.update(chain.slurm_job_ids)
        for model_name, servers in registry.items():
            for uid, info in servers.items():
                sjid = info.get("slurm_job_id")
                if sjid:
                    all_job_ids.add(sjid)

        # Single batched squeue call
        job_info_map = batch_query_jobs(list(all_job_ids)) if all_job_ids else {}

        # Build mapping: slurm_job_id → serve.json endpoint info
        endpoint_map: Dict[str, dict] = {}
        for model_name, servers in registry.items():
            for uid, info in servers.items():
                sjid = info.get("slurm_job_id")
                if sjid:
                    endpoint_map[sjid] = {
                        "ip": info.get("ip", "?"),
                        "port": info.get("port", "?"),
                        "create_time": info.get("create_time", ""),
                        "model_key": model_name,
                    }

        # Track which serve.json entries are accounted for by managed chains
        accounted_job_ids = set()

        # === Managed Servers Table ===
        if chains:
            table = Table(
                title="Managed Servers",
                title_style="bold cyan",
                border_style="cyan",
                show_lines=True,
                padding=(0, 1),
            )
            table.add_column("#", style="dim", width=3, justify="right")
            table.add_column("Model", style="bold white", min_width=18)
            table.add_column("Served Name", style="white", min_width=16)
            table.add_column("Status", min_width=10, justify="center")
            table.add_column("SLURM Job", min_width=10)
            table.add_column("Node", min_width=12)
            table.add_column("Endpoint", min_width=22)
            table.add_column("Uptime", min_width=8, justify="right")
            table.add_column("Next Job", min_width=10, justify="center")
            table.add_column("Account", style="dim", min_width=14)

            for idx, chain in enumerate(chains, 1):
                alive = self.state_manager.is_chain_alive(chain)
                accounted_job_ids.update(chain.slurm_job_ids)

                # Classify jobs using the cached batch result
                running_jobs = []
                pending_jobs = []
                for jid in chain.slurm_job_ids:
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
                        str(idx), chain.model_name, chain.served_name,
                        status, "-", "-", "-", "-", "-", chain.account,
                    )
                    continue

                if running_jobs:
                    jid, info = running_jobs[0]
                    ep = endpoint_map.get(jid, {})
                    endpoint = f"{ep.get('ip', '?')}:{ep.get('port', '?')}" if ep else "-"
                    uptime = info.get("elapsed", "-")
                    status = Text("RUNNING", style="bold green")
                    node = info.get("node", "-")

                    next_job = "-"
                    if pending_jobs:
                        next_job = Text("PENDING", style="yellow")
                    elif len(running_jobs) > 1:
                        next_job = Text("OVERLAP", style="cyan")

                    table.add_row(
                        str(idx), chain.model_name, chain.served_name,
                        status, jid, node, endpoint, uptime, next_job, chain.account,
                    )
                elif pending_jobs:
                    jid, info = pending_jobs[0]
                    status = Text("PENDING", style="bold yellow")
                    table.add_row(
                        str(idx), chain.model_name, chain.served_name,
                        status, jid, info.get("node", "-"), "-", "-", "-", chain.account,
                    )
                else:
                    status = Text("STARTING", style="bold yellow")
                    table.add_row(
                        str(idx), chain.model_name, chain.served_name,
                        status, "-", "-", "-", "-", "-", chain.account,
                    )

            self.console.print(table)
            self.console.print()

        # === Unmanaged Servers (in serve.json but not tracked by manager) ===
        unmanaged = []
        for model_name, servers in registry.items():
            for uid, info in servers.items():
                sjid = info.get("slurm_job_id")
                if sjid and sjid not in accounted_job_ids:
                    ji = job_info_map.get(sjid)
                    if ji and ji["status"].upper() in _ALIVE_STATUSES:
                        unmanaged.append((model_name, info, ji))

        if unmanaged:
            table = Table(
                title="Unmanaged Servers (not started by this tool)",
                title_style="bold yellow",
                border_style="yellow",
                show_lines=True,
                padding=(0, 1),
            )
            table.add_column("#", style="dim", width=3, justify="right")
            table.add_column("Served Name", style="white", min_width=16)
            table.add_column("Status", min_width=10, justify="center")
            table.add_column("SLURM Job", min_width=10)
            table.add_column("Node", min_width=12)
            table.add_column("Endpoint", min_width=22)
            table.add_column("Uptime", min_width=8, justify="right")

            for idx, (model_name, info, ji) in enumerate(unmanaged, 1):
                sjid = info.get("slurm_job_id", "-")
                ep = f"{info.get('ip', '?')}:{info.get('port', '?')}"
                uptime = ji.get("elapsed", "-")
                node = ji.get("node", "-")
                status_str = ji.get("status", "UNKNOWN")

                if status_str == "RUNNING":
                    status = Text("RUNNING", style="bold green")
                else:
                    status = Text(status_str, style="yellow")

                table.add_row(str(idx), model_name, status, sjid, node, ep, uptime)

            self.console.print(table)
            self.console.print()

        if not chains and not unmanaged:
            self.console.print(
                Panel(
                    "[dim]No servers running[/dim]",
                    title="Dashboard",
                    border_style="dim",
                ),
            )

    def _load_serve_registry(self) -> Dict:
        if not self.serve_file.exists():
            return {}
        try:
            with open(self.serve_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
