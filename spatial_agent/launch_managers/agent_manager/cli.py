"""Interactive CLI for the Agent Experiment Manager."""

import datetime
import json
import os
import signal
import tempfile
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
    get_exact_job_states,
    get_user_jobs_snapshot,
    get_user_steps_snapshot,
)
from spatial_agent.launch_managers.lifecycle import (
    LifecycleStore,
    _collision_safe_comment_token,
    compare_process_identities,
    evaluator_submission_evidence_token,
    format_evaluator_step_name,
    process_identity,
    process_identity_is_valid,
    process_identity_scope_is_local,
    process_pid_is_valid,
    parse_evaluator_job_comment,
    reconcile_runtime,
    signal_process_record,
)
from spatial_agent.launch_managers.agent_manager.state import (
    ExperimentStateManager,
    StateFileCorruptionError,
)


def _pid_alive(pid: int) -> bool:
    """Check if a PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, OSError):
        return False


class _Abort(Exception):
    """Raised when user types 'q' to cancel during a multi-step flow."""


class AgentManagerCLI:

    RECENT_FILE = "agent_manager_recent.json"
    STOP_VERIFY_ATTEMPTS = 10
    STOP_VERIFY_INTERVAL_SECONDS = 1.0

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

        try:
            self.state_manager.list_experiments()
        except StateFileCorruptionError as exc:
            self.console.print(
                "[bold red]Manager state is corrupt; all launch, cleanup, and "
                "stop actions are locked out.[/bold red]\n"
                f"[red]{exc}[/red]\n"
                f"[dim]State file: {self.state_manager.state_file}\n"
                "Inspect scheduler/lifecycle ownership first. The explicit "
                "archive_corrupt_state() recovery preserves the original "
                "bytes, but must only be used after that ownership audit.[/dim]"
            )
            return

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
        deferred = [
            e for e in experiments
            if e.status == "queue_deferred_retry_required"
        ]

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
        if deferred:
            names = ", ".join(
                f"{e.benchmark}/{e.experiment_name}" for e in deferred
            )
            parts.append(
                f"[bold yellow]Deferred ({len(deferred)}, manual resume; no "
                f"automatic retry):[/bold yellow] {names}"
            )

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
                enable_service_lifecycle=self.config.service_lifecycle_enabled,
                service_readiness_timeout_minutes=(
                    self.config.service_readiness_timeout_minutes
                ),
                service_running_readiness_timeout_minutes=(
                    self.config.service_running_readiness_timeout_minutes
                ),
                evaluator_pending_wait_minutes=(
                    self.config.evaluator_pending_wait_minutes
                ),
                service_renewal_heartbeat_ttl_minutes=(
                    self.config.service_renewal_heartbeat_ttl_minutes
                ),
                service_finalize_timeout_minutes=(
                    self.config.service_finalize_timeout_minutes
                ),
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

        experiments = self.state_manager.list_experiments()
        stoppable_exps = [
            (i, e) for i, e in enumerate(experiments)
            if (
                (
                    e.status in {"running", "quarantined"}
                    or (bool(e.lease_id) and e.lifecycle_state != "released")
                ) and (
                    e.status == "running"
                    or self.state_manager.is_experiment_alive(e)
                    or bool(e.lease_id)
                    or e.lifecycle_state in {"quarantined", "orphaned"}
                )
            )
        ]

        if not stoppable_exps:
            self.console.print("[dim]No active or recoverable experiments to stop.[/dim]")
            return

        # Batch query SLURM jobs
        all_jids = []
        for _, e in stoppable_exps:
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

        for display_idx, (_, exp) in enumerate(stoppable_exps, 1):
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
            targets = [e for _, e in stoppable_exps]
        else:
            indices = parse_range_selection(selection, len(stoppable_exps))
            if not indices:
                self.console.print("[red]No valid selection.[/red]")
                return
            targets = [stoppable_exps[i - 1][1] for i in indices]

        managed_reconciled = True
        if any(exp.lease_id for exp in targets):
            # Reconciliation is the only authority that may retire an
            # observed terminal evaluator fence after manager death. Its
            # failure blocks managed targets, but must not prevent an
            # independently exact-proven legacy stop in a mixed selection.
            try:
                reconcile_runtime(self.project_root)
            except Exception as exc:
                managed_reconciled = False
                self.console.print(
                    "[bold red]Managed stop reconciliation failed: "
                    f"{exc}. Managed ownership was retained.[/bold red]"
                )

        stopped = 0
        retained = 0
        for exp in targets:
            if exp.lease_id and not managed_reconciled:
                self._quarantine_stop(
                    exp, "authoritative lifecycle reconciliation failed",
                )
                retained += 1
            elif self._stop_one_experiment(exp):
                stopped += 1
            else:
                retained += 1

        if stopped:
            self.console.print(f"[green]Stopped {stopped} experiment(s).[/green]")
        if retained:
            self.console.print(
                f"[bold yellow]Retained {retained} experiment(s) because exact "
                "scheduler/lifecycle terminal proof was unavailable.[/bold yellow]"
            )
        self.console.print()

    @staticmethod
    def _submission_fenced(exp, lease: Optional[dict] = None) -> bool:
        state_fenced = bool(
            exp.evaluator_submission_comment and exp.evaluator_submission_phase
        )
        if lease is None:
            return state_fenced
        if lease.get("evaluator_submission_intent"):
            return True
        if not state_fenced:
            return False
        # Agent-manager state is an ownership mirror and may retain its last
        # observed marker after authoritative reconciliation orphaned the dead
        # lease. Only that observed+terminal combination supersedes the stale
        # mirror. Never-observed state remains a permanent fence.
        return not (
            bool(exp.evaluator_submission_observed)
            and lease.get("status") in {"orphaned", "terminal", "released"}
        )

    @staticmethod
    def _numeric_job_ids(*collections) -> Optional[List[str]]:
        job_ids = {
            str(job_id)
            for collection in collections
            for job_id in (collection or [])
            if job_id
        }
        if any(not job_id.isdigit() for job_id in job_ids):
            return None
        return sorted(job_ids, key=int)

    @staticmethod
    def _lease_step_ids(steps: dict, lease_id: str) -> List[str]:
        try:
            step_name = format_evaluator_step_name(str(lease_id))
        except ValueError:
            return []
        return sorted(
            str(step_id) for step_id, info in steps.items()
            if isinstance(info, dict) and info.get("name") == step_name
        )

    def _quarantine_stop(self, exp, reason: str) -> bool:
        try:
            self.state_manager.update_experiment_status(
                exp.experiment_id, "quarantined",
            )
            self.state_manager.update_experiment_lifecycle(
                exp.experiment_id, lifecycle_state="quarantined",
            )
        except Exception:
            pass
        self.console.print(
            f"  [yellow]{exp.benchmark}/{exp.experiment_name}: retained "
            f"(quarantined) — {reason}[/yellow]"
        )
        return False

    def _current_record(self, experiment_id: str):
        return next(
            (
                record for record in self.state_manager.list_experiments()
                if record.experiment_id == experiment_id
            ),
            None,
        )

    def _reservations_for_pid(self, pid: int) -> Optional[bool]:
        path = (
            self.project_root / "spatial_agent" / "logs"
            / "agent_reservations.json"
        )
        try:
            with open(path, encoding="utf-8") as handle:
                reservations = json.load(handle)
        except FileNotFoundError:
            return False
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(reservations, dict):
            return None
        identity_unknown = False
        for slots in reservations.values():
            if not isinstance(slots, list):
                return None
            for slot in slots:
                if not isinstance(slot, dict):
                    return None
                owner_pid = slot.get("pid")
                if not process_pid_is_valid(owner_pid):
                    return None
                if owner_pid != pid:
                    continue
                recorded_identity = slot.get("process_identity")
                if (
                    not process_identity_is_valid(
                        recorded_identity, expected_pid=owner_pid,
                    )
                    or not process_identity_scope_is_local(recorded_identity)
                ):
                    identity_unknown = True
                    continue
                alive = _pid_alive(owner_pid)
                current_identity = process_identity(owner_pid) if alive else None
                if not alive:
                    # A reservation belonging to a proven-dead process is
                    # stale and cannot block exact recovery.
                    continue
                comparison = compare_process_identities(
                    recorded_identity, current_identity,
                )
                if comparison is None:
                    identity_unknown = True
                    continue
                if comparison is True:
                    return True
                # Identity mismatch proves numeric PID reuse; the recorded
                # reservation belongs to the prior, dead process.
        return None if identity_unknown else False

    def _cancel_exact_and_verify(self, job_ids: List[str], lease_id: str = "") -> bool:
        states = get_exact_job_states(job_ids, force=True)
        steps = get_user_steps_snapshot() if lease_id else {}
        if states is None or steps is None:
            return False
        live_jobs = sorted(states)
        live_steps = self._lease_step_ids(steps, lease_id) if lease_id else []
        exact_targets = live_jobs + live_steps
        if exact_targets and cancel_jobs(exact_targets) != len(exact_targets):
            return False
        for attempt in range(self.STOP_VERIFY_ATTEMPTS):
            states = get_exact_job_states(job_ids, force=True)
            steps = get_user_steps_snapshot() if lease_id else {}
            if states is not None and steps is not None:
                live_steps = (
                    self._lease_step_ids(steps, lease_id) if lease_id else []
                )
                if not states and not live_steps:
                    return True
            if attempt + 1 < self.STOP_VERIFY_ATTEMPTS:
                time.sleep(self.STOP_VERIFY_INTERVAL_SECONDS)
        return False

    def _stop_one_experiment(self, exp) -> bool:
        """Stop one record only after exact, authoritative terminal proof."""
        managed = bool(exp.lease_id)
        lifecycle_store = LifecycleStore(self.project_root) if managed else None
        lease = None
        if managed:
            lease = lifecycle_store.snapshot().get("leases", {}).get(exp.lease_id)
            if not isinstance(lease, dict):
                return self._quarantine_stop(exp, "owning lifecycle lease is missing")
            if (
                lease.get("owner_id") != exp.experiment_id
                or int(lease.get("owner_pid") or 0) != int(exp.pid)
            ):
                return self._quarantine_stop(exp, "lifecycle owner identity does not match")
        recorded_identity = (
            lease.get("owner_process_identity") if managed
            else exp.process_identity
        )
        owner_status = self.state_manager.process_owner_status(
            exp.pid, recorded_identity,
        )
        if owner_status is None:
            return self._quarantine_stop(
                exp, "manager process ownership cannot be verified",
            )
        owner_live = owner_status is True

        if managed and not owner_live and self._submission_fenced(exp, lease):
            return self._quarantine_stop(
                exp, "an evaluator submission is accepted/pending but not terminal-proven",
            )

        if owner_live:
            if not signal_process_record(
                exp.pid, recorded_identity, signal.SIGTERM,
            ):
                return self._quarantine_stop(
                    exp, "manager process identity changed before signaling",
                )
            if managed:
                for attempt in range(self.STOP_VERIFY_ATTEMPTS):
                    current = self._current_record(exp.experiment_id)
                    snapshot = lifecycle_store.snapshot()
                    current_lease = snapshot.get("leases", {}).get(exp.lease_id, {})
                    ids = self._numeric_job_ids(
                        (current or exp).slurm_job_ids,
                        current_lease.get("job_ids", []),
                    )
                    if ids is None:
                        return self._quarantine_stop(exp, "recorded job IDs are not exact")
                    states = get_exact_job_states(ids, force=True)
                    steps = get_user_steps_snapshot()
                    if states is not None and steps is not None and (
                        not states
                        and not self._lease_step_ids(steps, exp.lease_id)
                        and current_lease.get("status") == "released"
                    ):
                        if current is not None:
                            self.state_manager.remove_experiment(exp.experiment_id)
                        return True
                    current_owner_status = self.state_manager.process_owner_status(
                        exp.pid, current_lease.get("owner_process_identity"),
                    )
                    if current_owner_status is None:
                        return self._quarantine_stop(
                            exp, "manager process ownership became uncertain",
                        )
                    if current_owner_status is False:
                        owner_live = False
                        break
                    if attempt + 1 < self.STOP_VERIFY_ATTEMPTS:
                        time.sleep(self.STOP_VERIFY_INTERVAL_SECONDS)
            else:
                # Legacy managers have no lifecycle finalizer authority; the
                # exact cancellation path below remains the sole cleanup path.
                pass

        if managed:
            return self._recover_dead_managed_experiment(
                exp, lifecycle_store, owner_status=owner_live,
            )
        return self._recover_legacy_experiment(exp)

    def _recover_legacy_experiment(self, exp) -> bool:
        ids = self._numeric_job_ids(exp.slurm_job_ids)
        if ids is None:
            return self._quarantine_stop(exp, "recorded job IDs are not exact")
        fenced = self._submission_fenced(exp)
        if fenced:
            comment = str(exp.evaluator_submission_comment or "")
            parsed = parse_evaluator_job_comment(comment)
            mirrored_ids = self._numeric_job_ids(
                exp.evaluator_submission_job_ids,
            )
            if (
                parsed is None
                or parsed["lease_id"]
                or parsed["experiment_token"]
                != _collision_safe_comment_token(exp.experiment_id)
                or exp.evaluator_submission_phase not in {
                    "prepared", "accepted_unseen", "accepted_unresolved",
                    "accepted_exact", "observed", "ambiguous_multiple",
                }
                or mirrored_ids is None
            ):
                return self._quarantine_stop(
                    exp, "malformed legacy evaluator submission fence",
                )
            snapshot = get_user_jobs_snapshot(force=True)
            if not getattr(snapshot, "authoritative", False):
                return self._quarantine_stop(
                    exp, "scheduler unavailable for exact comment recovery",
                )
            matches = sorted(
                str(job_id) for job_id, info in dict.items(snapshot)
                if isinstance(info, dict)
                and str(info.get("comment") or "") == comment
            )
            ids = self._numeric_job_ids(ids, mirrored_ids, matches)
            if ids is None or not ids:
                return self._quarantine_stop(
                    exp, "unseen evaluator submission fence has no exact target",
                )
        if not self._cancel_exact_and_verify(ids):
            return self._quarantine_stop(exp, "cancellation or terminal proof failed")
        owner_status = self.state_manager.process_owner_status(
            exp.pid, exp.process_identity,
        )
        if owner_status is None:
            return self._quarantine_stop(
                exp, "manager process ownership became uncertain",
            )
        if owner_status is True:
            return self._quarantine_stop(exp, "manager process did not terminate")
        if fenced:
            current = self._current_record(exp.experiment_id)
            if (
                current is None
                or current.evaluator_submission_comment
                != exp.evaluator_submission_comment
                or current.evaluator_submission_phase
                != exp.evaluator_submission_phase
            ):
                return self._quarantine_stop(
                    exp, "evaluator submission authority changed during recovery",
                )
            try:
                self.state_manager.update_experiment_submission(
                    exp.experiment_id,
                    comment=None,
                    phase=None,
                    job_ids=ids,
                    observed=True,
                )
            except Exception as exc:
                return self._quarantine_stop(
                    exp, f"could not clear terminal evaluator fence: {exc}",
                )
            if not self._persist_terminal_submission_evidence(exp, ids):
                return self._quarantine_stop(
                    exp, "could not clear independent evaluator evidence",
                )
            current = self._current_record(exp.experiment_id)
            if current is None or self._submission_fenced(current):
                return self._quarantine_stop(
                    exp, "terminal evaluator fence did not clear durably",
                )
        self.state_manager.remove_experiment(exp.experiment_id)
        return True

    def _persist_terminal_submission_evidence(self, exp, job_ids) -> bool:
        """Clear a legacy fence sidecar only after exact terminal proof."""
        try:
            token = evaluator_submission_evidence_token(exp.experiment_id)
        except ValueError:
            return False
        subdir = "slurm_cot" if exp.experiment_type == "cot" else "slurm_agent"
        path = (
            self.project_root / "spatial_agent" / "logs" / subdir
            / f"evaluator_submission_{token}.json"
        )
        if not path.exists():
            # Read-only compatibility for sidecars created before the token
            # gained a digest. Never create another artifact at this lossy path.
            legacy_token = "".join(
                character for character in exp.experiment_id
                if character.isalnum() or character in {"-", "_"}
            )[:64]
            legacy_path = path.with_name(
                f"evaluator_submission_{legacy_token}.json"
            ) if legacy_token else None
            if legacy_path is not None and legacy_path.exists():
                path = legacy_path
            else:
                # Some explicit test/legacy records predate the independent
                # sidecar; state and scheduler authority remain sufficient.
                return True
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(existing, dict)
                or existing.get("experiment_id") != exp.experiment_id
            ):
                return False
            payload = {
                **existing,
                "submitted_job_ids": list(job_ids),
                "observed_job_ids": list(job_ids),
                "unresolved_submission_comments": [],
                "ambiguous_submission_comments": [],
                "active_submission_intent": None,
                "scheduler_ownership_uncertain": False,
                "quarantine_reason": None,
                "terminal_cleanup": True,
                "updated_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                directory_fd = os.open(
                    path.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        return True

    def _recover_dead_managed_experiment(
        self, exp, lifecycle_store, *, owner_status: Optional[bool],
    ) -> bool:
        if owner_status is None:
            return self._quarantine_stop(
                exp, "manager process ownership cannot be verified",
            )
        if owner_status is True:
            return self._quarantine_stop(exp, "manager process is still live")
        snapshot = lifecycle_store.snapshot()
        lease = snapshot.get("leases", {}).get(exp.lease_id)
        if not isinstance(lease, dict) or (
            lease.get("owner_id") != exp.experiment_id
            or int(lease.get("owner_pid") or 0) != int(exp.pid)
        ):
            return self._quarantine_stop(exp, "lifecycle owner identity changed")
        current = self._current_record(exp.experiment_id)
        submission_fenced = (
            self._submission_fenced(current, lease)
            if current is not None
            else bool(lease.get("evaluator_submission_intent"))
        )
        if submission_fenced:
            return self._quarantine_stop(exp, "pending evaluator submission fence")
        ids = self._numeric_job_ids(
            (current or exp).slurm_job_ids, lease.get("job_ids", []),
        )
        if ids is None:
            return self._quarantine_stop(exp, "recorded job IDs are not exact")
        if not self._cancel_exact_and_verify(ids, exp.lease_id):
            return self._quarantine_stop(exp, "exact allocations/steps are not terminal")
        final_owner_status = self.state_manager.process_owner_status(
            exp.pid, lease.get("owner_process_identity"),
        )
        if final_owner_status is None:
            return self._quarantine_stop(
                exp, "manager process ownership became uncertain",
            )
        if final_owner_status is True:
            return self._quarantine_stop(exp, "manager process is still live")
        reservation_live = self._reservations_for_pid(exp.pid)
        if reservation_live is not False:
            return self._quarantine_stop(exp, "overlay reservation authority is not clear")

        # Re-check immutable ownership and the submission fence immediately
        # before releasing the lease; a concurrent writer cannot be ignored.
        lease = lifecycle_store.snapshot().get("leases", {}).get(exp.lease_id)
        current = self._current_record(exp.experiment_id)
        current_job_ids = self._numeric_job_ids(
            (current or exp).slurm_job_ids,
            lease.get("job_ids", []) if isinstance(lease, dict) else [],
        )
        if not isinstance(lease, dict) or (
            lease.get("owner_id") != exp.experiment_id
            or int(lease.get("owner_pid") or 0) != int(exp.pid)
            or lease.get("evaluator_submission_intent")
            or (
                current is not None
                and self._submission_fenced(current, lease)
            )
            or current_job_ids is None
            or set(current_job_ids) != set(ids)
        ):
            return self._quarantine_stop(exp, "lifecycle authority changed during recovery")
        try:
            if lease.get("status") in {"reserved", "active"}:
                lifecycle_store.begin_drain(exp.lease_id)
            if lease.get("status") != "released":
                lifecycle_store.mark_terminal(
                    exp.lease_id, "operator_stopped",
                    scheduler_checked=True,
                    overlay_checked=True,
                    dead_owner_recovery=True,
                )
                lifecycle_store.release_lease(
                    exp.lease_id,
                    scheduler_checked=True,
                    overlay_checked=True,
                    dead_owner_recovery=True,
                )
        except Exception as exc:
            return self._quarantine_stop(exp, f"lifecycle release failed: {exc}")
        released = lifecycle_store.snapshot().get("leases", {}).get(exp.lease_id, {})
        if released.get("status") != "released":
            return self._quarantine_stop(exp, "lifecycle release was not durable")
        self.state_manager.remove_experiment(exp.experiment_id)
        return True
