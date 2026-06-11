"""Dashboard: render status of all agent experiments."""

import datetime
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from spatial_agent.launch_managers.slurm_utils import batch_query_jobs
from spatial_agent.launch_managers.agent_manager.state import ExperimentState, ExperimentStateManager

_ALIVE_STATUSES = {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"}

_RESERVATIONS_FILE = "agent_reservations.json"


def _load_active_overlays(project_root: Path) -> Dict[int, List[Tuple[str, float]]]:
    """Map chain-manager pid → list of (parent_jobid, started_at).

    Reads agent_reservations.json (written by the dispatcher). Each slot's
    pid is the chain manager process that holds the overlay; we use it to
    correlate back to the experiment row in the dashboard.
    """
    path = project_root / "spatial_agent" / "logs" / _RESERVATIONS_FILE
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    out: Dict[int, List[Tuple[str, float]]] = {}
    if not isinstance(raw, dict):
        return {}
    for jid, slots in raw.items():
        if not isinstance(slots, list):
            continue
        for s in slots:
            if not isinstance(s, dict):
                continue
            try:
                pid = int(s.get("pid"))
                started_at = float(s.get("started_at", 0))
            except (TypeError, ValueError):
                continue
            out.setdefault(pid, []).append((str(jid), started_at))
    return out


def _format_elapsed(seconds: float) -> str:
    """Format seconds as '1:23:45' or '0:42'."""
    if seconds < 0:
        return "-"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _scheduled_remaining(scheduled_for: str):
    """Return a short 'in Xh Ym' label if scheduled_for is still in the future, else None."""
    if not scheduled_for:
        return None
    try:
        target = datetime.datetime.fromisoformat(scheduled_for)
    except ValueError:
        return None
    remaining = (target - datetime.datetime.now()).total_seconds()
    if remaining <= 0:
        return None
    hours, rem = divmod(int(remaining), 3600)
    minutes = rem // 60
    if hours > 0:
        return f"in {hours}h {minutes}m"
    if minutes > 0:
        return f"in {minutes}m"
    return f"in {int(remaining)}s"


def count_predictions(work_dir: str) -> int:
    """Count completed samples from predictions.jsonl."""
    pred_file = os.path.join(work_dir, "predictions.jsonl")
    if not os.path.exists(pred_file):
        return 0
    try:
        with open(pred_file) as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


class Dashboard:

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.state_manager = ExperimentStateManager(project_root)
        self.console = Console()

    def render(self) -> None:
        """Render the full experiment dashboard."""
        # Clean up dead chains
        dead = self.state_manager.cleanup_dead_experiments()
        if dead:
            for d in dead:
                self.console.print(
                    f"[dim]Cleaned up dead experiment: {d.benchmark}/{d.experiment_name} (PID {d.pid})[/dim]"
                )

        experiments = self.state_manager.list_experiments()

        if not experiments:
            self.console.print(
                Panel(
                    "[dim]No experiments running or completed[/dim]",
                    title="Agent Manager Dashboard",
                    border_style="dim",
                ),
            )
            return

        # Active overlay slots: chain-pid → [(parent_jobid, started_at)]
        overlay_by_pid = _load_active_overlays(self.project_root)

        # Collect all SLURM job IDs for a single batch query: experiments'
        # own sbatch jobs + parent jobids of any active overlays so we can
        # show their nodes.
        all_job_ids = set()
        for exp in experiments:
            all_job_ids.update(exp.slurm_job_ids)
        for slots in overlay_by_pid.values():
            for parent_jid, _ in slots:
                all_job_ids.add(parent_jid)
        job_info_map = batch_query_jobs(list(all_job_ids)) if all_job_ids else {}

        # Separate active and completed experiments. Completed/failed entries
        # are shown once and then removed from state so they don't accumulate.
        active_exps = [e for e in experiments if e.status == "running"]
        completed_exps = [e for e in experiments if e.status in ("completed", "failed")]

        if active_exps:
            self._render_active_table(active_exps, job_info_map, overlay_by_pid)

        if completed_exps:
            self._render_completed_table(completed_exps)
            self.state_manager.remove_experiments(
                [e.experiment_id for e in completed_exps]
            )

    def _render_active_table(
        self,
        experiments: List[ExperimentState],
        job_info_map: Dict,
        overlay_by_pid: Optional[Dict[int, List[Tuple[str, float]]]] = None,
    ) -> None:
        overlay_by_pid = overlay_by_pid or {}
        table = Table(
            title="Active Experiments",
            title_style="bold cyan",
            border_style="cyan",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("Type", style="magenta", width=6, justify="center")
        table.add_column("Benchmark", style="bold white", min_width=14)
        table.add_column("Model", style="white", min_width=16)
        table.add_column("Experiment", style="cyan", min_width=14)
        table.add_column("Status", min_width=10, justify="center")
        table.add_column("Progress", min_width=16, justify="center")
        table.add_column("SLURM Job", min_width=10)
        table.add_column("Node", min_width=12)
        table.add_column("Elapsed", min_width=8, justify="right")
        table.add_column("Account", style="dim", min_width=14)

        for idx, exp in enumerate(experiments, 1):
            alive = self.state_manager.is_experiment_alive(exp)
            type_label = "CoT" if exp.experiment_type == "cot" else "Agent"

            # Get progress
            completed = count_predictions(exp.work_dir)
            total = exp.total_samples
            progress = self._format_progress(completed, total)

            # Classify SLURM jobs
            running_jobs = []
            pending_jobs = []
            for jid in exp.slurm_job_ids:
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
                    str(idx), type_label, exp.benchmark, exp.model_name, exp.experiment_name,
                    status, progress, "-", "-", "-", exp.account,
                )
                continue

            if running_jobs:
                jid, info = running_jobs[0]
                status = Text("RUNNING", style="bold green")
                node = info.get("node", "-")
                elapsed = info.get("elapsed", "-")
                table.add_row(
                    str(idx), type_label, exp.benchmark, exp.model_name, exp.experiment_name,
                    status, progress, jid, node, elapsed, exp.account,
                )
            elif pending_jobs:
                jid, info = pending_jobs[0]
                status = Text("PENDING", style="bold yellow")
                table.add_row(
                    str(idx), type_label, exp.benchmark, exp.model_name, exp.experiment_name,
                    status, progress, jid, info.get("node", "-"), "-", exp.account,
                )
            elif overlay_by_pid.get(exp.pid):
                # Agent is running as an `srun --overlap` step inside an
                # existing vLLM/gpu_server job — no separate sbatch jobid,
                # but we can still show the parent job's id/node and the
                # slot's age.
                parent_jid, started_at = overlay_by_pid[exp.pid][0]
                parent_info = job_info_map.get(parent_jid, {})
                node = parent_info.get("node", "-")
                elapsed = _format_elapsed(time.time() - started_at) if started_at else "-"
                status = Text("OVERLAY", style="bold green")
                table.add_row(
                    str(idx), type_label, exp.benchmark, exp.model_name, exp.experiment_name,
                    status, progress, f"→{parent_jid}", node, elapsed, exp.account,
                )
            else:
                remaining_label = _scheduled_remaining(getattr(exp, "scheduled_for", ""))
                if remaining_label is not None:
                    status = Text("SCHEDULED", style="bold cyan")
                    table.add_row(
                        str(idx), type_label, exp.benchmark, exp.model_name, exp.experiment_name,
                        status, progress, "-", "-", remaining_label, exp.account,
                    )
                else:
                    status = Text("STARTING", style="bold yellow")
                    table.add_row(
                        str(idx), type_label, exp.benchmark, exp.model_name, exp.experiment_name,
                        status, progress, "-", "-", "-", exp.account,
                    )

        self.console.print(table)
        self.console.print()

    def _render_completed_table(self, experiments: List[ExperimentState]) -> None:
        table = Table(
            title="Completed Experiments",
            title_style="bold green",
            border_style="green",
            show_lines=True,
            padding=(0, 1),
        )
        table.add_column("#", style="dim", width=3, justify="right")
        table.add_column("Type", style="magenta", width=6, justify="center")
        table.add_column("Benchmark", style="bold white", min_width=14)
        table.add_column("Model", style="white", min_width=16)
        table.add_column("Experiment", style="cyan", min_width=14)
        table.add_column("Status", min_width=10, justify="center")
        table.add_column("Progress", min_width=16, justify="center")
        table.add_column("Started", min_width=16)

        for idx, exp in enumerate(experiments, 1):
            completed = count_predictions(exp.work_dir)
            total = exp.total_samples
            progress = self._format_progress(completed, total)
            type_label = "CoT" if exp.experiment_type == "cot" else "Agent"

            if exp.status == "completed":
                status = Text("COMPLETED", style="bold green")
            else:
                status = Text("FAILED", style="bold red")

            started = exp.started_at[:19] if exp.started_at else "-"
            table.add_row(
                str(idx), type_label, exp.benchmark, exp.model_name, exp.experiment_name,
                status, progress, started,
            )

        self.console.print(table)
        self.console.print()

    def _format_progress(self, completed: int, total: int) -> Text:
        """Format progress as '45/200 (22.5%)'."""
        if total <= 0:
            return Text(f"{completed}/?", style="dim")

        pct = (completed / total) * 100
        text = f"{completed}/{total} ({pct:.1f}%)"

        if completed >= total:
            return Text(text, style="bold green")
        elif pct >= 50:
            return Text(text, style="yellow")
        else:
            return Text(text, style="white")
