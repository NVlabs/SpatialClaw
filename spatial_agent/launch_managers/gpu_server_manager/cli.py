"""Interactive CLI for the GPU Server Manager."""

import os
import signal
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from spatial_agent.launch_managers.cli_utils import parse_range_selection
from spatial_agent.launch_managers.gpu_server_manager.config import load_config
from spatial_agent.launch_managers.gpu_server_manager.dashboard import GPUServerDashboard
from spatial_agent.launch_managers.gpu_server_manager.server_chain import start_chain_background
from spatial_agent.launch_managers.gpu_server_manager.state import GPUServerStateManager
from spatial_agent.launch_managers.slurm_utils import (
    batch_query_jobs,
    cancel_jobs,
    check_slurm_available,
)


class _Abort(Exception):
    """Raised when user types 'q' to cancel during a multi-step flow."""


class GPUServerManagerCLI:

    def __init__(self):
        # Project root: gpu_server_manager/ → launch_managers/ → spatial_agent/ → repo root
        self.project_root = Path(__file__).parent.parent.parent.parent.absolute()
        self.config = load_config()
        self.state_manager = GPUServerStateManager(self.project_root)
        self.dashboard = GPUServerDashboard(self.project_root)
        self.console = Console()

    def run(self) -> None:
        self.console.print()
        self.console.print(
            Panel(
                "[bold cyan]GPU Server Manager[/bold cyan]\n"
                "[dim]Manage SLURM-based GPU tool servers (Pi3/SAM3/EasyOCR) "
                "with automatic 4-hour rotation[/dim]",
                border_style="cyan",
                padding=(1, 2),
            ),
        )

        if not check_slurm_available():
            self.console.print("[bold red]Error: SLURM (sbatch) not found on this system.[/bold red]")
            return

        while True:
            self.console.print()
            self._show_quick_status()
            self.console.print()

            menu = Table(show_header=False, box=None, padding=(0, 2))
            menu.add_column(style="bold cyan", width=5)
            menu.add_column()
            menu.add_row("[1]", "Dashboard — view all GPU servers")
            menu.add_row("[2]", "Start GPU Server(s)")
            menu.add_row("[3]", "Stop GPU Server(s)")
            menu.add_row("[q]", "Quit")
            self.console.print(menu)
            self.console.print()

            choice = Prompt.ask(
                "[bold]Select",
                choices=["1", "2", "3", "q"],
                default="1",
            )

            if choice == "1":
                self._show_dashboard()
            elif choice == "2":
                self._start_gpu_servers()
            elif choice == "3":
                self._stop_gpu_servers()
            elif choice == "q":
                self.console.print("[dim]Goodbye.[/dim]")
                break

    def _show_quick_status(self) -> None:
        servers = self.state_manager.list_servers()
        alive = [s for s in servers if self.state_manager.is_server_alive(s)]
        if alive:
            total_gpus = sum(s.gpus for s in alive)
            self.console.print(
                f"[green]Active GPU servers ({len(alive)}):[/green] {total_gpus} GPU(s)",
            )
        else:
            self.console.print("[dim]No GPU servers currently managed.[/dim]")

    def _show_dashboard(self) -> None:
        self.console.print()
        self.dashboard.render()

    def _ask(self, prompt: str, default: str = "") -> str:
        val = Prompt.ask(f"{prompt} [dim](q to cancel)[/dim]", default=default)
        if val.strip().lower() == "q":
            raise _Abort()
        return val

    def _ask_int(self, prompt: str, default: int) -> int:
        val = self._ask(prompt, default=str(default))
        try:
            return int(val)
        except ValueError:
            self.console.print("[red]Invalid number.[/red]")
            raise _Abort()

    def _start_gpu_servers(self) -> None:
        self.console.print()
        try:
            self._start_gpu_servers_flow()
        except _Abort:
            self.console.print("[dim]Cancelled.[/dim]")

    def _start_gpu_servers_flow(self) -> None:
        num_servers = self._ask_int("[bold]How many GPU servers to launch", default=1)
        if num_servers < 1:
            self.console.print("[red]Must launch at least 1 server.[/red]")
            return

        gpus = self._ask_int("[bold]GPUs per server", default=self.config.default_slurm.gpus)

        self.console.print()
        backend = self._ask(
            "[bold]Reconstruct backend (pi3/da3/mapanything)",
            default=self.config.reconstruct_backend,
        )
        if backend not in ("pi3", "da3", "mapanything"):
            self.console.print("[red]Invalid backend. Use 'pi3', 'da3', or 'mapanything'.[/red]")
            return

        self.console.print()
        accounts = self.config.accounts
        for i, acc in enumerate(accounts, 1):
            self.console.print(f"  [bold cyan][{i}][/bold cyan] {acc}")
        self.console.print()
        acc_idx = self._ask_int("[bold]Select account", default=1)
        if 1 <= acc_idx <= len(accounts):
            account = accounts[acc_idx - 1]
        else:
            self.console.print("[red]Invalid selection.[/red]")
            return

        self.console.print()
        summary = (
            f"  Servers:    {num_servers}\n"
            f"  GPUs each:  {gpus}\n"
            f"  Backend:    {backend}\n"
            f"  Account:    {account}\n"
            f"  Partition:  {self.config.default_slurm.partition}\n"
            f"  Time limit: {self.config.default_slurm.time_limit}"
        )
        self.console.print(Panel(summary, title="Confirm GPU Server Launch", border_style="green"))

        confirm = Prompt.ask("[bold]Launch?", choices=["y", "n"], default="y")
        if confirm != "y":
            self.console.print("[dim]Cancelled.[/dim]")
            return

        self.console.print()
        for i in range(num_servers):
            chain_id = str(uuid.uuid4())
            self.console.print(f"[yellow]Starting GPU server {i + 1}/{num_servers}...[/yellow]")

            srv_state, chain_log = start_chain_background(
                chain_id=chain_id,
                account=account,
                partition=self.config.default_slurm.partition,
                gpus=gpus,
                reconstruct_backend=backend,
                time_limit=self.config.default_slurm.time_limit,
                restart_before_minutes=self.config.default_slurm.restart_before_minutes,
                project_root=self.project_root,
            )

            self.console.print(
                f"  [green]Started![/green] PID={srv_state.pid}, log={chain_log}"
            )

        self.console.print()
        self.console.print(
            f"[bold green]{num_servers} GPU server(s) launched![/bold green]\n"
            f"[dim]Use Dashboard to monitor.[/dim]"
        )

    def _stop_gpu_servers(self) -> None:
        self.console.print()

        dead = self.state_manager.cleanup_dead_servers()
        if dead:
            for d in dead:
                self.console.print(
                    f"[dim]Cleaned up dead GPU server (PID {d.pid})[/dim]"
                )

        servers = self.state_manager.list_servers()
        alive_servers = [
            (i, s) for i, s in enumerate(servers)
            if self.state_manager.is_server_alive(s)
        ]

        if not alive_servers:
            self.console.print("[dim]No active GPU servers to stop.[/dim]")
            return

        all_jids = []
        for _, s in alive_servers:
            all_jids.extend(s.slurm_job_ids)
        job_info_map = batch_query_jobs(all_jids) if all_jids else {}

        table = Table(
            title="Active GPU Servers",
            title_style="bold red",
            border_style="red",
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("GPUs", width=5, justify="center")
        table.add_column("Backend", min_width=6)
        table.add_column("PID", width=8)
        table.add_column("SLURM Jobs", min_width=16)
        table.add_column("Account", min_width=14)

        for display_idx, (_, srv) in enumerate(alive_servers, 1):
            active_jobs = [jid for jid in srv.slurm_job_ids if jid in job_info_map]
            jobs_str = ", ".join(active_jobs) if active_jobs else "[dim]-[/dim]"
            table.add_row(
                str(display_idx), str(srv.gpus), srv.reconstruct_backend,
                str(srv.pid), jobs_str, srv.account,
            )

        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Select with range notation (e.g. 1-3, 5), 'all', or 'c' to cancel[/dim]")

        selection = Prompt.ask(
            "[bold]Select GPU server(s) to stop",
            default="c",
        )

        if selection.lower() == "c":
            return

        if selection.lower() == "all":
            targets = [s for _, s in alive_servers]
        else:
            indices = parse_range_selection(selection, len(alive_servers))
            if not indices:
                self.console.print("[red]No valid selection.[/red]")
                return
            targets = [alive_servers[i - 1][1] for i in indices]

        all_jids = []
        all_pids = []
        for srv in targets:
            all_jids.extend(srv.slurm_job_ids)
            all_pids.append(srv.pid)

        for pid in all_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        if all_jids:
            self.console.print(f"  [yellow]Cancelling {len(all_jids)} SLURM job(s)...[/yellow]")
            cancel_jobs(all_jids)

        for srv in targets:
            self.state_manager.remove_server(srv.chain_id)

        self.console.print(f"[green]Stopped {len(targets)} GPU server(s).[/green]")
        self.console.print()
