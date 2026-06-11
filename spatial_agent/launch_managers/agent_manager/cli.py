"""Interactive CLI for the Agent Experiment Manager."""

import datetime
import json
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from spatial_agent.launch_managers.agent_manager.config import ManagerConfig, load_config
from spatial_agent.launch_managers.agent_manager.dashboard import Dashboard
from spatial_agent.launch_managers.agent_manager.experiment_chain import start_experiment_background
from spatial_agent.launch_managers.cli_utils import parse_range_selection
from spatial_agent.launch_managers.slurm_utils import (
    batch_query_jobs,
    cancel_jobs,
    check_slurm_available,
)
from spatial_agent.launch_managers.agent_manager.state import ExperimentStateManager


def _pid_alive(pid: int) -> bool:
    """Check if a PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


class _Abort(Exception):
    """Raised when user types 'q' to cancel during a multi-step flow."""


class AgentManagerCLI:

    RECENT_FILE = "agent_manager_recent.json"

    def __init__(self):
        # Walk up agent_manager/ → launch_managers/ → spatial_agent/ → project root.
        self.project_root = Path(__file__).parent.parent.parent.parent.absolute()
        self.config = load_config(self.project_root)
        self.state_manager = ExperimentStateManager(self.project_root)
        self.dashboard = Dashboard(self.project_root)
        self.console = Console()
        self._recent = self._load_recent()

    def _recent_file_path(self) -> Path:
        return self.project_root / "spatial_agent" / "logs" / self.RECENT_FILE

    def _load_recent(self) -> Dict[str, str]:
        """Load recent selections (model, account) from disk."""
        path = self._recent_file_path()
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_recent(self, model: str, account: str, experiment_name: str = "") -> None:
        """Persist recent selections."""
        self._recent = {"model": model, "account": account, "experiment_name": experiment_name}
        path = self._recent_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._recent, f)

    def run(self) -> None:
        self.console.print()
        self.console.print(
            Panel(
                "[bold cyan]Agent Experiment Manager[/bold cyan]\n"
                "[dim]Manage SLURM-based spatial agent experiments with automatic 4-hour rotation[/dim]",
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
            menu.add_row("[1]", "Dashboard — view all experiments")
            menu.add_row("[2]", "Start Agent Experiment(s)")
            menu.add_row("[3]", "Start CoT Experiment(s)")
            menu.add_row("[4]", "Stop Experiment(s)")
            menu.add_row("[q]", "Quit")
            self.console.print(menu)
            self.console.print()

            choice = Prompt.ask(
                "[bold]Select",
                choices=["1", "2", "3", "4", "q"],
                default="1",
            )

            if choice == "1":
                self._show_dashboard()
            elif choice == "2":
                self._start_experiments("agent")
            elif choice == "3":
                self._start_experiments("cot")
            elif choice == "4":
                self._stop_experiments()
            elif choice == "q":
                self.console.print("[dim]Goodbye.[/dim]")
                break

    def _show_quick_status(self) -> None:
        """One-line status summary."""
        experiments = self.state_manager.list_experiments()
        alive = [e for e in experiments if e.status == "running" and self.state_manager.is_experiment_alive(e)]
        completed = [e for e in experiments if e.status == "completed"]

        parts = []

        if alive:
            names = ", ".join(
                f"{'[CoT]' if e.experiment_type == 'cot' else '[Agent]'} {e.benchmark}/{e.experiment_name}"
                for e in alive
            )
            parts.append(f"[green]Active ({len(alive)}):[/green] {names}")
        if completed:
            names = ", ".join(
                f"{'[CoT]' if e.experiment_type == 'cot' else '[Agent]'} {e.benchmark}/{e.experiment_name}"
                for e in completed
            )
            parts.append(f"[bold green]Completed ({len(completed)}):[/bold green] {names}")

        if parts:
            self.console.print(" | ".join(parts))
        else:
            self.console.print("[dim]No experiments currently managed.[/dim]")

    def _show_dashboard(self) -> None:
        self.console.print()
        self.dashboard.render()

    def _ask(self, prompt: str, default: str = "") -> str:
        """Prompt that accepts 'q' to abort."""
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

    def _start_experiments(self, experiment_type: str) -> None:
        self.console.print()
        try:
            self._start_experiments_flow(experiment_type)
        except _Abort:
            self.console.print("[dim]Cancelled.[/dim]")

    def _start_experiments_flow(self, experiment_type: str) -> None:
        type_label = "CoT" if experiment_type == "cot" else "Agent"
        is_cot = experiment_type == "cot"

        # Pick SLURM defaults based on experiment type
        if is_cot:
            slurm = self.config.cot_slurm
            default_concurrency = self.config.cot_default_concurrency
        else:
            slurm = self.config.default_slurm
            default_concurrency = self.config.default_concurrency

        # Step 1: Select benchmark(s) — supports range notation
        benchmarks = self.config.benchmarks
        table = Table(
            title=f"Available Benchmarks ({type_label})",
            title_style="bold cyan",
            border_style="cyan",
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Benchmark", style="white", min_width=20)

        for i, b in enumerate(benchmarks, 1):
            table.add_row(str(i), b)

        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Select multiple with range notation: e.g. 1-5, 10, 15-16[/dim]")

        selection_str = self._ask("[bold]Select benchmark(s)")
        selected_indices = parse_range_selection(selection_str, len(benchmarks))
        if not selected_indices:
            self.console.print("[red]No valid benchmarks selected.[/red]")
            return

        selected_benchmarks = [benchmarks[i - 1] for i in selected_indices]
        self.console.print(
            f"[green]Selected:[/green] {', '.join(selected_benchmarks)}"
        )

        # Step 2: Select model
        self.console.print()
        models = self.config.model_configs
        recent_model = self._recent.get("model")
        table = Table(
            title="Available Models",
            title_style="bold cyan",
            border_style="cyan",
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Model", min_width=20)

        if recent_model and recent_model in models:
            table.add_row("[0]", f"[bold yellow]{recent_model}  (recent)[/bold yellow]")

        for i, m in enumerate(models, 1):
            table.add_row(str(i), m)

        self.console.print(table)
        self.console.print()

        default_model = 0 if (recent_model and recent_model in models) else 1
        model_idx = self._ask_int("[bold]Select model", default=default_model)
        if model_idx == 0:
            if recent_model and recent_model in models:
                model_name = recent_model
            else:
                self.console.print("[red]No recent model available.[/red]")
                return
        elif 1 <= model_idx <= len(models):
            model_name = models[model_idx - 1]
        else:
            self.console.print("[red]Invalid selection.[/red]")
            return

        # Step 3: Experiment name
        self.console.print()
        recent_exp = self._recent.get("experiment_name", "default")
        experiment_name = self._ask("[bold]Experiment name", default=recent_exp)

        # Step 4: Parameters
        self.console.print()
        self.console.print("[bold]Parameters[/bold] [dim](press Enter for default)[/dim]")
        concurrency = self._ask_int("  Concurrency", default=default_concurrency)
        subsample = self._ask_int("  Subsample (0=all)", default=self.config.default_subsample)

        # CoT-specific parameters
        max_frames = 0
        system_prompt = "cot"
        if is_cot:
            max_frames = self._ask_int("  Max frames per sample", default=self.config.cot_default_max_frames)
            system_prompt = self._ask(
                "  System prompt (cot/direct)", default=self.config.cot_default_system_prompt
            )
            if system_prompt not in ("cot", "direct"):
                self.console.print("[red]Invalid system prompt. Use 'cot' or 'direct'.[/red]")
                return

        # Step 5: Select account
        self.console.print()
        accounts = self.config.accounts
        recent_account = self._recent.get("account")

        if recent_account and recent_account in accounts:
            self.console.print(f"  [bold yellow][0][/bold yellow] [bold yellow]{recent_account}  (recent)[/bold yellow]")
        for i, acc in enumerate(accounts, 1):
            self.console.print(f"  [bold cyan][{i}][/bold cyan] {acc}")
        self.console.print()

        default_acc = 0 if (recent_account and recent_account in accounts) else 1
        acc_idx = self._ask_int("[bold]Select account", default=default_acc)
        if acc_idx == 0:
            if recent_account and recent_account in accounts:
                account = recent_account
            else:
                self.console.print("[red]No recent account available.[/red]")
                return
        elif 1 <= acc_idx <= len(accounts):
            account = accounts[acc_idx - 1]
        else:
            self.console.print("[red]Invalid selection.[/red]")
            return

        # Step 6: Confirm
        self.console.print()
        benchmark_list = "\n".join(f"    - {b}" for b in selected_benchmarks)
        summary = (
            f"[bold]{type_label} Experiment — Benchmarks ({len(selected_benchmarks)}):[/bold]\n{benchmark_list}\n\n"
            f"  Model:        {model_name}\n"
            f"  Experiment:   {experiment_name}\n"
            f"  Account:      {account}\n"
            f"  Partition:    {slurm.partition}\n"
            f"  GPUs:         {slurm.gpus}\n"
            f"  Concurrency:  {concurrency}\n"
            f"  Subsample:    {subsample or 'all'}\n"
            f"  Time limit:   {slurm.time_limit}"
        )
        if is_cot:
            summary += (
                f"\n  Max frames:   {max_frames}\n"
                f"  System prompt: {system_prompt}"
            )
        self.console.print(Panel(summary, title=f"Confirm {type_label} Launch", border_style="green"))

        confirm = Prompt.ask("[bold]Launch?", choices=["y", "n"], default="y")
        if confirm != "y":
            self.console.print("[dim]Cancelled.[/dim]")
            return

        # Step 6.5: Optional launch deferral
        defer_minutes = max(0, self._ask_int(
            "[bold]Defer launch by minutes (0 = immediate)", default=0
        ))
        if defer_minutes > 0:
            start_at = datetime.datetime.now() + datetime.timedelta(minutes=defer_minutes)
            self.console.print(
                f"[dim]Experiments will start at {start_at.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(in {defer_minutes} minute(s)).[/dim]"
            )

        # Save recent selections
        self._save_recent(model=model_name, account=account, experiment_name=experiment_name)

        # Step 7: Spawn one chain per benchmark
        self.console.print()
        for benchmark in selected_benchmarks:
            experiment_id = str(uuid.uuid4())

            self.console.print(f"[yellow]Starting {type_label} {benchmark}/{experiment_name}...[/yellow]")

            exp_state, chain_log = start_experiment_background(
                experiment_id=experiment_id,
                benchmark=benchmark,
                model_name=model_name,
                experiment_name=experiment_name,
                account=account,
                partition=slurm.partition,
                gpus=slurm.gpus,
                time_limit=slurm.time_limit,
                concurrency=concurrency,
                subsample=subsample,
                project_root=self.project_root,
                experiment_type=experiment_type,
                max_frames=max_frames,
                system_prompt=system_prompt,
                defer_minutes=defer_minutes,
            )

            self.console.print(
                f"  [green]Started![/green] PID={exp_state.pid}, log={chain_log}"
            )

        self.console.print()
        self.console.print(
            f"[bold green]{len(selected_benchmarks)} {type_label} experiment(s) launched![/bold green]\n"
            f"[dim]Use Dashboard to monitor progress.[/dim]"
        )

    # ------------------------------------------------------------------
    # Experiment Management
    # ------------------------------------------------------------------

    def _stop_experiments(self) -> None:
        self.console.print()

        # Clean up dead experiments first
        dead = self.state_manager.cleanup_dead_experiments()
        if dead:
            for d in dead:
                self.console.print(
                    f"[dim]Cleaned up dead experiment: {d.benchmark}/{d.experiment_name} (PID {d.pid})[/dim]"
                )

        experiments = self.state_manager.list_experiments()
        alive_exps = [
            (i, e) for i, e in enumerate(experiments)
            if e.status == "running" and self.state_manager.is_experiment_alive(e)
        ]

        if not alive_exps:
            self.console.print("[dim]No active experiments to stop.[/dim]")
            return

        # Batch query SLURM jobs
        all_jids = []
        for _, e in alive_exps:
            all_jids.extend(e.slurm_job_ids)
        job_info_map = batch_query_jobs(all_jids) if all_jids else {}

        table = Table(
            title="Active Experiments",
            title_style="bold red",
            border_style="red",
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Type", style="magenta", width=6, justify="center")
        table.add_column("Benchmark", style="white", min_width=14)
        table.add_column("Model", min_width=16)
        table.add_column("Experiment", style="cyan", min_width=14)
        table.add_column("PID", width=8)
        table.add_column("SLURM Jobs", min_width=16)

        for display_idx, (_, exp) in enumerate(alive_exps, 1):
            active_jobs = [jid for jid in exp.slurm_job_ids if jid in job_info_map]
            jobs_str = ", ".join(active_jobs) if active_jobs else "[dim]-[/dim]"
            type_label = "CoT" if exp.experiment_type == "cot" else "Agent"
            table.add_row(
                str(display_idx), type_label, exp.benchmark, exp.model_name,
                exp.experiment_name, str(exp.pid), jobs_str,
            )

        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Select with range notation (e.g. 1-3, 5), 'all', or 'c' to cancel[/dim]")

        selection = Prompt.ask(
            "[bold]Select experiment(s) to stop",
            default="c",
        )

        if selection.lower() == "c":
            return

        if selection.lower() == "all":
            targets = [e for _, e in alive_exps]
        else:
            indices = parse_range_selection(selection, len(alive_exps))
            if not indices:
                self.console.print("[red]No valid selection.[/red]")
                return
            targets = [alive_exps[i - 1][1] for i in indices]

        # Collect all SLURM job IDs and PIDs upfront
        all_jids = []
        all_pids = []
        for exp in targets:
            all_jids.extend(exp.slurm_job_ids)
            all_pids.append(exp.pid)

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
        for exp in targets:
            self.state_manager.remove_experiment(exp.experiment_id)

        self.console.print(f"[green]Stopped {len(targets)} experiment(s).[/green]")
        self.console.print()
