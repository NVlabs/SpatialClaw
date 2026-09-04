"""Interactive CLI for the vLLM Server Manager."""

import json
import os
import signal
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from spatial_agent.launch_managers.cli_utils import parse_range_selection
from spatial_agent.launch_managers.agent_manager.config import (
    load_config as load_agent_manager_config,
)
from spatial_agent.launch_managers.lifecycle import (
    LifecycleError,
    LifecycleStore,
    component_digests,
    file_sha256,
    reconcile_runtime,
    registry_record_attests_pool,
    signal_process_record,
    tagged_capacity_conflicts,
)
from spatial_agent.launch_managers.vllm_manager.config import ManagerConfig, load_config
from spatial_agent.launch_managers.vllm_manager.dashboard import Dashboard
from spatial_agent.launch_managers.vllm_manager.server_chain import (
    recover_managed_chain,
    start_chain_background,
)
from spatial_agent.launch_managers.slurm_utils import (
    batch_query_jobs,
    cancel_jobs,
    check_slurm_available,
    get_exact_job_states,
    get_user_jobs_snapshot,
    resolve_partitions,
)
from spatial_agent.launch_managers.vllm_manager.state import (
    ChainStateManager,
    StateFileCorruptionError,
)


def _resolved_hf_revision(model_id: str) -> str | None:
    """Best-effort local Hugging Face commit identity without network access."""
    if not model_id or "/" not in model_id:
        return None
    # ``huggingface_hub`` accepts both cache-specific variables below.  They
    # point at the hub cache itself, unlike ``HF_HOME`` which points one level
    # above it.  Mirror that precedence so the manager pins the exact local
    # revision that the worker will load, including on clusters with a shared
    # model cache outside the login node's home directory.
    cache_override = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
    )
    if cache_override:
        hub_cache = Path(cache_override)
    else:
        hf_home = Path(
            os.environ.get(
                "HF_HOME", str(Path.home() / ".cache" / "huggingface")
            )
        )
        hub_cache = hf_home / "hub"
    reference = (
        hub_cache / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
    )
    try:
        value = reference.read_text(encoding="utf-8").strip()
        return value or None
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _server_spec(project_root: Path, model, *, max_model_len: int,
                 max_num_seqs: int, gpus: int) -> dict:
    """Exact compatibility fields for a manager-launched vLLM service."""
    tracked_files = [
        project_root / "spatial_agent" / "entrypoints" / "launch_vllm.py",
        project_root / "spatial_agent" / "launch_managers" / "vllm_manager" / "run_vllm.sh",
        project_root / "spatial_agent" / "requirements" / "requirements-vllm.txt",
    ]
    return {
        "schema_version": 1,
        "model_name": model.name,
        "model": model.model,
        "resolved_model_revision": _resolved_hf_revision(model.model),
        "served_name": model.served_name,
        "max_model_len": int(max_model_len),
        "max_num_seqs": int(max_num_seqs),
        "tp_size": int(model.tp_size),
        "gpus_per_instance": int(gpus),
        "kv_cache_dtype": model.kv_cache_dtype,
        "quantization": model.quantization,
        "launcher_files": {
            str(path.relative_to(project_root)): file_sha256(path)
            for path in tracked_files
        },
        "synthetic_keepalive": False,
    }


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


class VLLMManagerCLI:

    STOP_VERIFY_ATTEMPTS = 10
    STOP_VERIFY_INTERVAL_SECONDS = 1.0

    def __init__(self):
        # Walk up vllm_manager/ → launch_managers/ → spatial_agent/ → project root.
        self.project_root = Path(__file__).parent.parent.parent.parent.absolute()
        self.config = load_config()
        self.service_lifecycle_enabled = load_agent_manager_config(
            self.project_root
        ).service_lifecycle_enabled
        self.state_manager = ChainStateManager(self.project_root)
        self.dashboard = Dashboard(self.project_root)
        self.console = Console()
        self._launch_stack = None

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

        try:
            self.state_manager.list_chains()
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
        for chain in chains:
            intent = chain.submission_intent
            if intent is None:
                continue
            comment = intent.get("comment") if isinstance(intent, dict) else None
            self.console.print(
                "[bold red]Submission fence:[/bold red] "
                f"chain={chain.chain_id} pool={chain.pool_id or 'legacy'} "
                f"comment={comment or 'invalid/unavailable'} "
                f"exact_jobs={','.join(map(str, chain.slurm_job_ids)) or '-'}"
            )
        try:
            pools = LifecycleStore(self.project_root).snapshot().get("pools", {})
            for pool_id, pool in sorted(pools.items()):
                pending = [
                    (chain_id, chain)
                    for chain_id, chain in (pool.get("chains") or {}).items()
                    if isinstance(chain, dict)
                    and chain.get("submission_intent") is not None
                ]
                if pool.get("status") == "quarantined" or pending:
                    exact_ids = sorted({
                        str(job_id)
                        for chain in (pool.get("chains") or {}).values()
                        if isinstance(chain, dict)
                        for job_id in chain.get("job_ids", [])
                        if job_id
                    })
                    self.console.print(
                        f"[bold red]Lifecycle attention:[/bold red] pool={pool_id} "
                        f"status={pool.get('status')} pending={len(pending)} "
                        f"exact_jobs={','.join(exact_ids) or '-'}"
                    )
        except Exception as exc:
            self.console.print(
                f"[bold red]Lifecycle status unavailable; launch is unsafe:[/bold red] {exc}"
            )

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
        with ExitStack() as launch_stack:
            self._launch_stack = launch_stack
            try:
                self._start_server_inner()
            except _Abort:
                self.console.print("[dim]Cancelled.[/dim]")
            except (LifecycleError, OSError, ValueError) as exc:
                self.console.print(
                    f"[bold red]Refusing launch: identity lock/lifecycle "
                    f"authority failed: {exc}[/bold red]"
                )
            finally:
                self._launch_stack = None

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
        try:
            partition_default = resolve_partitions(model.partition, account)
        except RuntimeError as exc:
            self.console.print(f"[yellow]{exc}[/yellow]")
            partition_default = model.partition
        partition = self._ask("  partition", default=partition_default)

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

        if not getattr(self, "service_lifecycle_enabled", False):
            pending_authority = [
                chain for chain in self.state_manager.list_chains()
                if chain.served_name == model.served_name
                and getattr(chain, "submission_intent", None) is not None
            ]
            if pending_authority:
                self.console.print(
                    "[bold red]Refusing legacy launch: a matching manager "
                    "retains unresolved submission authority.[/bold red]"
                )
                return
            self.console.print(
                "[bold yellow]Service lifecycle is disabled in the shared "
                "agent-manager config. Launching legacy capacity; it must be "
                "stopped explicitly after the experiment.[/bold yellow]"
            )
            for i in range(num_instances):
                chain_state, _chain_log = start_chain_background(
                    chain_id=str(uuid.uuid4()),
                    served_name=model.served_name,
                    model_name=model.name,
                    model_path=model.model,
                    # Historical unmanaged launch resolves the model after
                    # the worker sources its own environment/cache settings.
                    model_revision=None,
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
                    f"  [green]Legacy {i + 1}/{num_instances}[/green] — "
                    f"Chain {chain_state.chain_id[:8]}, PID {chain_state.pid}"
                )
            return

        # Step 5: reconcile the desired *total*. Repeating this flow is
        # idempotent; it no longer means "add N more".
        server_spec = _server_spec(
            self.project_root, model,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            gpus=gpus,
        )
        if (
            "/" in model.model
            and not Path(model.model).exists()
            and not server_spec.get("resolved_model_revision")
        ):
            raise LifecycleError(
                "Managed Hugging Face services require an exact locally "
                f"resolved revision; no cached refs/main was found for "
                f"{model.model!r}"
            )
        topology = {
            "schema_version": 1,
            "desired_instances": num_instances,
            "gpus_per_instance": gpus,
            "tp_size": model.tp_size,
            "target_hardware": "scheduler-selected; realized hardware recorded per epoch",
        }
        lifecycle_policy = {
            "schema_version": 1,
            "owner_gated_successor": True,
            "auto_stop_when_unowned": True,
            "bootstrap_grace_minutes": 30,
            "synthetic_keepalive": False,
            "restart_before_minutes": restart_before,
            "time_limit": time_limit,
        }
        digests = component_digests(
            server_spec=server_spec,
            topology=topology,
            lifecycle_policy=lifecycle_policy,
        )
        lifecycle_store = LifecycleStore(self.project_root)
        if self._launch_stack is None:
            raise LifecycleError("Launch flow is missing its identity lock scope")
        self._launch_stack.enter_context(
            lifecycle_store.service_identity_launch_lock(
                "vllm", model.served_name,
            )
        )
        try:
            reconcile_runtime(self.project_root)
        except LifecycleError as exc:
            self.console.print(
                "[bold red]Refusing launch: authoritative lifecycle "
                f"reconciliation failed: {exc}[/bold red]"
            )
            return
        proposed_pool_id = f"vllm-{uuid.uuid4().hex[:12]}"
        proposed_namespace = f"{model.served_name}-{uuid.uuid4().hex[:12]}"
        pool = lifecycle_store.ensure_pool(
            proposed_pool_id=proposed_pool_id,
            service_kind="vllm",
            namespace=proposed_namespace,
            desired_instances=num_instances,
            server_spec=server_spec,
            topology=topology,
            lifecycle_policy=lifecycle_policy,
        )

        jobs = get_user_jobs_snapshot(force=True)
        if not getattr(jobs, "authoritative", False):
            self.console.print(
                "[bold red]Refusing launch: authoritative Slurm state is "
                "unavailable. Existing capacity cannot be ruled out.[/bold red]"
            )
            return
        self.state_manager.cleanup_dead_chains(alive_job_ids=jobs.keys())
        chains = self.state_manager.list_chains()
        compatible = []
        conflicts = []
        for chain in chains:
            recorded_live_jobs = [
                job_id for job_id in chain.slurm_job_ids
                if str(job_id) in jobs
            ]
            manager_alive = self.state_manager.is_chain_alive(chain)
            identity_verified = self.state_manager.is_chain_identity_verified(chain)
            if chain.served_name != model.served_name:
                continue
            intent = chain.submission_intent
            if intent is not None:
                comment = intent.get("comment") if isinstance(intent, dict) else None
                conflicts.append(
                    f"chain={chain.chain_id[:8]} pool={chain.pool_id or 'legacy'} "
                    "has pending submission authority "
                    f"comment={comment or 'invalid/unavailable'} "
                    f"jobs={','.join(map(str, chain.slurm_job_ids)) or '-'}"
                )
                continue
            exact = (
                chain.pool_id == pool["pool_id"]
                and chain.namespace == pool["namespace"]
                and chain.spec_digest == digests["server_spec"]
                and chain.topology_digest == digests["topology"]
            )
            if (
                exact
                and manager_alive
                and identity_verified
                and chain.lifecycle_state != "quarantined"
            ):
                compatible.append(chain)
            elif manager_alive or recorded_live_jobs:
                conflicts.append(
                    f"chain={chain.chain_id[:8]} pool={chain.pool_id or 'legacy'} "
                    f"jobs={','.join(map(str, recorded_live_jobs)) or '-'} "
                    f"pid_identity={'verified' if identity_verified else 'unknown'}"
                )

        conflicts.extend(tagged_capacity_conflicts(
            jobs,
            lifecycle_store.snapshot(),
            service_kind="vllm",
            identity_value=model.served_name,
            desired_pool_id=pool["pool_id"],
            compatible_chain_ids=[chain.chain_id for chain in compatible],
        ))

        # The endpoint registry catches a live service whose detached manager
        # state was lost. Never adopt it by model name alone.
        serve_file = self.project_root / "spatial_agent" / "logs" / "serve.json"
        try:
            with open(serve_file, encoding="utf-8") as handle:
                registry = json.load(handle)
        except FileNotFoundError:
            registry = {}
        except (json.JSONDecodeError, OSError) as exc:
            lifecycle_store.set_pool_status(
                pool["pool_id"], "quarantined",
                registry_error=str(exc),
            )
            self.console.print(
                f"[bold red]Refusing launch: endpoint registry is unreadable "
                f"or corrupt: {exc}[/bold red]"
            )
            return
        if not isinstance(registry, dict):
            lifecycle_store.set_pool_status(
                pool["pool_id"], "quarantined",
                registry_error="registry root is not an object",
            )
            self.console.print(
                "[bold red]Refusing launch: endpoint registry root is invalid.[/bold red]"
            )
            return
        managed_jobs = {
            str(job_id): chain
            for chain in compatible
            for job_id in chain.slurm_job_ids
        }
        registry_shape_error = None
        for registry_model, records in registry.items():
            if not isinstance(records, dict):
                registry_shape_error = (
                    f"registry bucket {registry_model!r} is not an object"
                )
                break
            for uid, info in records.items():
                if not isinstance(info, dict):
                    registry_shape_error = (
                        f"registry record {registry_model!r}/{uid!r} is not an object"
                    )
                    break
                job_id = str(info.get("slurm_job_id") or "")
                if not job_id or job_id not in jobs:
                    continue
                managed_chain = managed_jobs.get(job_id)
                if managed_chain is not None:
                    if (
                        registry_model != model.served_name
                        or not registry_record_attests_pool(
                            info,
                            pool,
                            identity_value=model.served_name,
                            slot_id=managed_chain.chain_id,
                        )
                    ):
                        conflicts.append(
                            f"registry_uid={uid[:8]} job={job_id} has "
                            "incompatible managed-service attestation"
                        )
                elif registry_model == model.served_name:
                    conflicts.append(
                        f"registry_uid={uid[:8]} "
                        f"pool={info.get('pool_id') or 'legacy'} job={job_id}"
                    )
            if registry_shape_error:
                break
        if registry_shape_error:
            lifecycle_store.set_pool_status(
                pool["pool_id"], "quarantined",
                registry_error=registry_shape_error,
            )
            self.console.print(
                f"[bold red]Refusing launch: {registry_shape_error}.[/bold red]"
            )
            return

        conflicts.extend(lifecycle_store.service_identity_conflicts(
            service_kind="vllm",
            identity_value=model.served_name,
            desired_pool_id=pool["pool_id"],
        ))

        if conflicts:
            lifecycle_store.set_pool_status(
                pool["pool_id"], "quarantined", conflicts=conflicts,
            )
            self.console.print(
                "[bold red]Refusing launch: live capacity with incomplete or "
                "incompatible ownership was found.[/bold red]"
            )
            for conflict in conflicts:
                self.console.print(f"  [red]- {conflict}[/red]")
            self.console.print(
                "[dim]Reconcile those exact managers/jobs before retrying; "
                "nothing was cancelled.[/dim]"
            )
            return

        try:
            retired = lifecycle_store.retire_abandoned_identity_pools(
                service_kind="vllm",
                identity_value=model.served_name,
                keep_pool_id=pool["pool_id"],
            )
        except LifecycleError as exc:
            lifecycle_store.set_pool_status(
                pool["pool_id"], "quarantined",
                identity_retirement_error=str(exc),
            )
            self.console.print(
                f"[bold red]Refusing launch: same-identity lifecycle authority "
                f"changed during validation: {exc}[/bold red]"
            )
            return
        if retired:
            self.console.print(
                "[dim]Archived empty superseded pool(s): "
                + ", ".join(retired)
                + "[/dim]"
            )

        if len(compatible) > num_instances:
            self.console.print(
                f"[bold yellow]Pool already has {len(compatible)} compatible "
                f"chains, above desired {num_instances}. No mid-cohort "
                "scale-down was performed.[/bold yellow]"
            )
            return

        missing = num_instances - len(compatible)
        self.console.print()
        if compatible:
            self.console.print(
                f"[green]Adopting {len(compatible)} exact compatible chain(s).[/green]"
            )
        if missing == 0:
            self.console.print(
                f"[bold green]Desired total already satisfied: {num_instances}.[/bold green]"
            )
            return
        claim_id = f"cli-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        claimed_slots = lifecycle_store.claim_launch_slots(
            pool["pool_id"],
            claimant_id=claim_id,
            claimant_pid=os.getpid(),
            max_slots=missing,
        )
        if not claimed_slots:
            lifecycle_store.release_launch_claim(pool["pool_id"], claim_id)
            self.console.print(
                "[yellow]Missing capacity is already claimed or durably owned; "
                "no duplicate launch was made.[/yellow]"
            )
            return
        self.console.print(
            f"[yellow]Starting {len(claimed_slots)} missing chain(s) to reach desired "
            f"total {num_instances}...[/yellow]"
        )

        try:
            for i, chain_id in enumerate(claimed_slots):
                chain_state, chain_log = start_chain_background(
                    chain_id=chain_id,
                    served_name=model.served_name,
                    model_name=model.name,
                    model_path=model.model,
                    model_revision=server_spec.get(
                        "resolved_model_revision"
                    ),
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
                    pool_id=pool["pool_id"],
                    namespace=pool["namespace"],
                    server_spec_digest=digests["server_spec"],
                    topology_digest=digests["topology"],
                    desired_instances=num_instances,
                    auto_stop_when_unowned=True,
                    bootstrap_grace_minutes=30,
                    launch_claim_id=claim_id,
                )

                self.console.print(
                    f"  [green]New {i + 1}/{len(claimed_slots)}[/green] — "
                    f"Chain {chain_id[:8]}, PID {chain_state.pid}",
                )
        finally:
            lifecycle_store.release_launch_claim(pool["pool_id"], claim_id)

        self.console.print(
            f"\n[bold green]Desired pool total: {num_instances} "
            f"({len(claimed_slots)} newly started).[/bold green]\n"
            f"[dim]Pool: {pool['pool_id']}  namespace: {pool['namespace']}[/dim]\n"
            f"[dim]Each chain will submit SLURM jobs automatically.[/dim]\n"
            f"[dim]Use Dashboard to monitor status.[/dim]",
        )

    def _stop_server(self) -> None:
        self.console.print()

        # Stop-menu rendering is read-only. Managed cleanup requires lifecycle
        # reconciliation after explicit selection; legacy cleanup requires the
        # exact cancellation path below.
        chains = self.state_manager.list_chains()
        alive_chains = [
            (i, chain)
            for i, chain in enumerate(chains)
            if self.state_manager.is_chain_alive(chain)
            or (
                chain.pool_id
                and (
                    chain.slurm_job_ids
                    or chain.submission_intent is not None
                    or chain.lifecycle_state == "quarantined"
                )
            )
        ]

        if not alive_chains:
            self.console.print(
                "[dim]No running or quarantined managed servers to stop.[/dim]"
            )
            return

        # Single batch squeue call for all job IDs
        all_jids = []
        for _, c in alive_chains:
            all_jids.extend(c.slurm_job_ids)
        job_info_map = batch_query_jobs(all_jids) if all_jids else {}

        table = Table(
            title="Running / Recoverable Servers",
            title_style="bold red",
            border_style="red",
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("Model", style="white", min_width=20)
        table.add_column("Served Name", style="cyan", min_width=18)
        table.add_column("PID", width=8)
        table.add_column("Manager", width=12)
        table.add_column("Account", style="dim", min_width=14)
        table.add_column("SLURM Jobs", min_width=16)

        for display_idx, (_, chain) in enumerate(alive_chains, 1):
            active_jobs = [jid for jid in chain.slurm_job_ids if jid in job_info_map]
            jobs_str = ", ".join(active_jobs) if active_jobs else "[dim]-[/dim]"
            table.add_row(
                str(display_idx), chain.model_name, chain.served_name,
                str(chain.pid),
                (
                    "[green]alive[/green]"
                    if self.state_manager.is_chain_alive(chain)
                    else "[bold yellow]recovery[/bold yellow]"
                ),
                chain.account, jobs_str,
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

        managed_reconciled = True
        if any(chain.pool_id for chain in targets):
            try:
                reconcile_runtime(self.project_root)
            except Exception as exc:
                managed_reconciled = False
                self.console.print(
                    "[bold red]Managed stop reconciliation failed: "
                    f"{exc}. Managed ownership was retained.[/bold red]"
                )
        lifecycle_store = LifecycleStore(self.project_root)
        refused = []
        allowed = []
        for chain in targets:
            if chain.pool_id and not managed_reconciled:
                refused.append((chain, "lifecycle reconciliation is unavailable"))
            elif chain.pool_id and lifecycle_store.active_leases_for_pool(chain.pool_id):
                refused.append((chain, "an owner lease is still active"))
            else:
                allowed.append(chain)
        for chain, reason in refused:
            self.console.print(
                f"[bold red]Refused {chain.served_name}/{chain.chain_id[:8]}: "
                f"{reason}.[/bold red]"
            )
        if not allowed:
            return

        # Managed chains drain and cancel their exact jobs themselves. Legacy
        # chains use the same exact-ID/terminal-proof standard below.
        recovery_outcomes = []
        for chain in allowed:
            if not chain.pool_id:
                continue
            if self.state_manager.is_chain_alive(chain):
                if not self.state_manager.is_chain_identity_verified(chain):
                    recovery_outcomes.append((chain, "identity_unverified"))
                    continue
                if not signal_process_record(
                    chain.pid, chain.process_identity, signal.SIGTERM,
                ):
                    recovery_outcomes.append((chain, "identity_unverified"))
            elif chain.pool_id:
                recovery_outcomes.append(
                    (chain, recover_managed_chain(chain, self.project_root))
                )

        legacy = [chain for chain in allowed if not chain.pool_id]
        legacy_outcomes = [
            (chain, self._stop_legacy_chain_exact(chain)) for chain in legacy
        ]

        managed_count = len([
            chain for chain in allowed
            if chain.pool_id and self.state_manager.is_chain_alive(chain)
        ])
        if managed_count:
            self.console.print(
                f"[green]Requested graceful drain for {managed_count} managed server(s).[/green]"
            )
        for chain, outcome in recovery_outcomes:
            style = "green" if outcome == "stopped" else "yellow"
            self.console.print(
                f"[{style}]Recovery {chain.served_name}/"
                f"{chain.chain_id[:8]}: {outcome}.[/{style}]"
            )
        for chain, stopped in legacy_outcomes:
            style = "green" if stopped else "yellow"
            outcome = "stopped" if stopped else "retained/quarantined"
            self.console.print(
                f"[{style}]Legacy {chain.served_name}/"
                f"{chain.chain_id[:8]}: {outcome}.[/{style}]"
            )
        self.console.print()

    def _retain_legacy_chain(self, chain) -> bool:
        chain.lifecycle_state = "quarantined"
        self.state_manager.add_chain(chain)
        return False

    def _stop_legacy_chain_exact(self, chain) -> bool:
        """Stop a legacy chain without broad cancellation or assumed absence."""
        job_ids = sorted({str(job) for job in chain.slurm_job_ids if job})
        if any(not job_id.isdigit() for job_id in job_ids):
            return self._retain_legacy_chain(chain)

        if self.state_manager.is_chain_alive(chain):
            if not self.state_manager.is_chain_identity_verified(chain):
                return self._retain_legacy_chain(chain)
            if not signal_process_record(
                chain.pid, chain.process_identity, signal.SIGTERM,
            ):
                return self._retain_legacy_chain(chain)

        intent = chain.submission_intent
        if intent is not None:
            generation = (
                intent.get("generation") if isinstance(intent, dict) else None
            )
            intent_ids_value = (
                intent.get("job_ids") if isinstance(intent, dict) else None
            )
            comment = str(
                intent.get("comment") or ""
                if isinstance(intent, dict) else ""
            )
            if (
                not isinstance(intent, dict)
                or isinstance(intent.get("schema_version"), bool)
                or intent.get("schema_version") != 1
                or isinstance(generation, bool)
                or not isinstance(generation, int)
                or generation < 1
                or intent.get("phase") not in {
                    "prepared", "accepted_unseen", "accepted_unresolved",
                    "ambiguous_multiple",
                }
                or not isinstance(intent_ids_value, list)
                or any(
                    not isinstance(job_id, str) or not job_id.isdigit()
                    for job_id in intent_ids_value
                )
                or comment
                != f"spatialclaw:v1::{chain.chain_id}:{generation}"
            ):
                return self._retain_legacy_chain(chain)
            snapshot = get_user_jobs_snapshot(force=True)
            if not comment or not getattr(snapshot, "authoritative", False):
                return self._retain_legacy_chain(chain)
            matches = sorted(
                str(job_id) for job_id, info in dict.items(snapshot)
                if isinstance(info, dict)
                and str(info.get("comment") or "") == comment
            )
            intent_ids = [
                str(job_id) for job_id in intent_ids_value
                if str(job_id).isdigit()
            ]
            if not matches and not intent_ids:
                # A prepared/accepted-unresolved fence with no visible exact
                # allocation is uncertainty, never terminal proof.
                return self._retain_legacy_chain(chain)
            job_ids = sorted(set(job_ids) | set(matches) | set(intent_ids))

        states = get_exact_job_states(job_ids, force=True)
        if states is None:
            return self._retain_legacy_chain(chain)
        if not job_ids:
            snapshot = get_user_jobs_snapshot(force=True)
            if not getattr(snapshot, "authoritative", False):
                return self._retain_legacy_chain(chain)
        live_ids = sorted(states)
        if live_ids and cancel_jobs(live_ids) != len(live_ids):
            return self._retain_legacy_chain(chain)

        terminal = False
        for attempt in range(self.STOP_VERIFY_ATTEMPTS):
            states = get_exact_job_states(job_ids, force=True)
            if states is not None and not states:
                terminal = True
                if not self.state_manager.is_chain_alive(chain):
                    break
            if attempt + 1 < self.STOP_VERIFY_ATTEMPTS:
                time.sleep(self.STOP_VERIFY_INTERVAL_SECONDS)
        if not terminal or self.state_manager.is_chain_alive(chain):
            return self._retain_legacy_chain(chain)

        current = next(
            (
                record for record in self.state_manager.list_chains()
                if record.chain_id == chain.chain_id
            ),
            None,
        )
        if current is None:
            return True
        current_ids = {str(job) for job in current.slurm_job_ids if job}
        if current_ids - set(job_ids):
            return self._retain_legacy_chain(current)
        if current.submission_intent is not None:
            if current.submission_intent != intent:
                return self._retain_legacy_chain(current)
            try:
                self.state_manager.update_chain_submission(
                    current.chain_id,
                    slurm_job_ids=job_ids,
                    submission_intent=None,
                )
            except Exception:
                return self._retain_legacy_chain(current)
            current = next(
                (
                    record for record in self.state_manager.list_chains()
                    if record.chain_id == chain.chain_id
                ),
                None,
            )
            if current is None:
                return True
            if current.submission_intent is not None or {
                str(job) for job in current.slurm_job_ids if job
            } - set(job_ids):
                return self._retain_legacy_chain(current)
        self.state_manager.remove_chain(chain.chain_id)
        return True
