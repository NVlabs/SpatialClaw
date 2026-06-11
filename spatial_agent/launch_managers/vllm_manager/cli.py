"""Interactive CLI for the vLLM Server Manager."""

import os
import signal
import time
import uuid
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from spatial_agent.launch_managers.cli_utils import parse_range_selection
from spatial_agent.launch_managers.vllm_manager.config import ManagerConfig, load_config
from spatial_agent.launch_managers.vllm_manager.dashboard import Dashboard
from spatial_agent.launch_managers.vllm_manager.server_chain import start_chain_background
from spatial_agent.launch_managers.slurm_utils import (
    batch_query_jobs,
    cancel_jobs,
    check_slurm_available,
)
from spatial_agent.launch_managers.vllm_manager.state import ChainStateManager


def _pid_alive(pid: int) -> bool:
    """Check if a PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


class _Abort(Exception):
    """Raised when user types 'q' to cancel during a multi-step flow."""


class VLLMManagerCLI:

    def __init__(self):
        # Walk up vllm_manager/ → launch_managers/ → spatial_agent/ → project root.
        self.project_root = Path(__file__).parent.parent.parent.parent.absolute()
        self.config = load_config()
        self.state_manager = ChainStateManager(self.project_root)
        self.dashboard = Dashboard(self.project_root)
        self.console = Console()

    def run(self) -> None:
        self.console.print()
        self.console.print(
            Panel(
                "[bold cyan]vLLM Server Manager[/bold cyan]\n"
                "[dim]Manage SLURM-based vLLM servers with automatic 4-hour rotation[/dim]",
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
            menu.add_row("[1]", "Dashboard — view all servers")
            menu.add_row("[2]", "Start Server")
            menu.add_row("[3]", "Stop Server")
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
                self._start_server()
            elif choice == "3":
                self._stop_server()
            elif choice == "q":
                self.console.print("[dim]Goodbye.[/dim]")
                break

    def _show_quick_status(self) -> None:
        """One-line status summary."""
        chains = self.state_manager.list_chains()
        alive = [c for c in chains if self.state_manager.is_chain_alive(c)]
        if alive:
            names = ", ".join(c.served_name for c in alive)
            self.console.print(
                f"[green]Active servers ({len(alive)}):[/green] {names}",
            )
        else:
            self.console.print("[dim]No servers currently managed.[/dim]")

    def _show_dashboard(self) -> None:
        self.console.print()
        self.dashboard.render()

    def _ask(self, prompt: str, default: str = "") -> str:
        """Prompt that accepts 'q' to abort. Returns value or raises _Abort."""
        val = Prompt.ask(f"{prompt} [dim](q to cancel)[/dim]", default=default)
        if val.strip().lower() == "q":
            raise _Abort()
        return val

    def _ask_int(self, prompt: str, default: int) -> int:
        """Int prompt that accepts 'q' to abort."""
        val = self._ask(prompt, default=str(default))
        try:
            return int(val)
        except ValueError:
            self.console.print("[red]Invalid number.[/red]")
            raise _Abort()

    def _start_server(self) -> None:
        self.console.print()

        try:
            self._start_server_inner()
        except _Abort:
            self.console.print("[dim]Cancelled.[/dim]")

    def _start_server_inner(self) -> None:
        # Step 1: Select model
        models = self.config.models
        table = Table(
            title="Available Models",
            title_style="bold cyan",
            border_style="cyan",
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Name", style="white", min_width=20)
        table.add_column("Served Name", style="cyan", min_width=18)
        table.add_column("TP", justify="center", width=4)
        table.add_column("KV dtype", justify="center", width=8)
        table.add_column("Quant", justify="center", width=6)
        table.add_column("Max Len", justify="right", width=8)
        table.add_column("Max Seqs", justify="right", width=9)
        table.add_column("Partition", style="dim", min_width=16)

        for i, m in enumerate(models, 1):
            table.add_row(
                str(i), m.name, m.served_name,
                str(m.tp_size), m.kv_cache_dtype, m.quantization,
                str(m.max_model_len), str(m.max_num_seqs),
                m.partition,
            )

        self.console.print(table)
        self.console.print()

        model_idx = self._ask_int("[bold]Select model", default=1)
        if model_idx < 1 or model_idx > len(models):
            self.console.print("[red]Invalid selection.[/red]")
            return

        model = models[model_idx - 1]

        # Step 2: Select account
        self.console.print()
        accounts = self.config.accounts
        for i, acc in enumerate(accounts, 1):
            self.console.print(f"  [bold cyan][{i}][/bold cyan] {acc}")
        self.console.print()

        acc_idx = self._ask_int("[bold]Select account", default=1)
        if acc_idx < 1 or acc_idx > len(accounts):
            self.console.print("[red]Invalid selection.[/red]")
            return
        account = accounts[acc_idx - 1]

        # Step 3: Editable parameters
        self.console.print()
        self.console.print("[bold]Parameters[/bold] [dim](press Enter for default)[/dim]")

        max_model_len = self._ask_int("  max_model_len", default=model.max_model_len)
        max_num_seqs = self._ask_int("  max_num_seqs", default=model.max_num_seqs)
        num_instances = max(1, min(32, self._ask_int("  num_instances", default=1)))
        partition = self._ask("  partition", default=model.partition)

        gpus = model.tp_size
        time_limit = self.config.default_slurm.time_limit
        restart_before = self.config.default_slurm.restart_before_minutes

        # Step 4: Confirm
        self.console.print()
        kv_info = f"  KV cache dtype: {model.kv_cache_dtype}\n" if model.kv_cache_dtype != "auto" else ""
        quant_info = f"  Quantization:  {model.quantization}\n" if model.quantization != "none" else ""
        summary = (
            f"[bold]{model.name}[/bold] ({model.served_name})\n"
            f"  Model:         {model.model}\n"
            f"  Account:       {account}\n"
            f"  Partition:     {partition}\n"
            f"  TP size:       {model.tp_size}\n"
            f"{kv_info}"
            f"{quant_info}"
            f"  GPUs/instance: {gpus}\n"
            f"  Instances:     {num_instances}\n"
            f"  Max model len: {max_model_len}\n"
            f"  Max num seqs:  {max_num_seqs}\n"
            f"  Time limit:    {time_limit} (overlap: {restart_before}m)"
        )
        self.console.print(Panel(summary, title="Confirm Launch", border_style="green"))

        confirm = Prompt.ask("[bold]Launch?", choices=["y", "n"], default="y")
        if confirm != "y":
            self.console.print("[dim]Cancelled.[/dim]")
            return

        # Step 5: Start chain(s)
        self.console.print()
        self.console.print(f"[yellow]Starting {num_instances} server chain(s)...[/yellow]")

        for i in range(num_instances):
            chain_id = str(uuid.uuid4())
            chain_state, chain_log = start_chain_background(
                chain_id=chain_id,
                served_name=model.served_name,
                model_name=model.name,
                model_path=model.model,
                account=account,
                partition=partition,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                tp_size=model.tp_size,
                kv_cache_dtype=model.kv_cache_dtype,
                quantization=model.quantization,
                gpus=gpus,
                time_limit=time_limit,
                restart_before_minutes=restart_before,
                project_root=self.project_root,
            )

            self.console.print(
                f"  [green]Instance {i + 1}/{num_instances}[/green] — "
                f"Chain {chain_id[:8]}, PID {chain_state.pid}",
            )

        self.console.print(
            f"\n[bold green]All {num_instances} server chain(s) started![/bold green]\n"
            f"[dim]Each chain will submit SLURM jobs automatically.[/dim]\n"
            f"[dim]Use Dashboard to monitor status.[/dim]",
        )

    def _stop_server(self) -> None:
        self.console.print()

        # Clean up dead chains first
        dead = self.state_manager.cleanup_dead_chains()
        if dead:
            for d in dead:
                self.console.print(f"[dim]Cleaned up dead chain: {d.served_name} (PID {d.pid})[/dim]")

        chains = self.state_manager.list_chains()
        alive_chains = [(i, c) for i, c in enumerate(chains) if self.state_manager.is_chain_alive(c)]

        if not alive_chains:
            self.console.print("[dim]No managed servers to stop.[/dim]")
            return

        # Single batch squeue call for all job IDs
        all_jids = []
        for _, c in alive_chains:
            all_jids.extend(c.slurm_job_ids)
        job_info_map = batch_query_jobs(all_jids) if all_jids else {}

        table = Table(
            title="Running Servers",
            title_style="bold red",
            border_style="red",
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Model", style="white", min_width=20)
        table.add_column("Served Name", style="cyan", min_width=18)
        table.add_column("PID", width=8)
        table.add_column("Account", style="dim", min_width=14)
        table.add_column("SLURM Jobs", min_width=16)

        for display_idx, (_, chain) in enumerate(alive_chains, 1):
            active_jobs = [jid for jid in chain.slurm_job_ids if jid in job_info_map]
            jobs_str = ", ".join(active_jobs) if active_jobs else "[dim]-[/dim]"
            table.add_row(
                str(display_idx), chain.model_name, chain.served_name,
                str(chain.pid), chain.account, jobs_str,
            )

        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Select with range notation (e.g. 1-3, 5), 'all', or 'c' to cancel[/dim]")

        selection = Prompt.ask(
            "[bold]Select server(s) to stop",
            default="c",
        )

        if selection.lower() == "c":
            return

        if selection.lower() == "all":
            targets = [c for _, c in alive_chains]
        else:
            indices = parse_range_selection(selection, len(alive_chains))
            if not indices:
                self.console.print("[red]No valid selection.[/red]")
                return
            targets = [alive_chains[i - 1][1] for i in indices]

        # Collect all SLURM job IDs and PIDs upfront
        all_jids = []
        all_pids = []
        for chain in targets:
            all_jids.extend(chain.slurm_job_ids)
            all_pids.append(chain.pid)

        # SIGTERM all chain manager processes (non-blocking, fire-and-forget)
        for pid in all_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        # Cancel all SLURM jobs in one batched scancel call.
        if all_jids:
            self.console.print(f"  [yellow]Cancelling {len(all_jids)} SLURM job(s)...[/yellow]")
            cancel_jobs(all_jids)

        # Clean up state immediately
        for chain in targets:
            self.state_manager.remove_chain(chain.chain_id)

        self.console.print(f"[green]Stopped {len(targets)} server(s).[/green]")
        self.console.print()
