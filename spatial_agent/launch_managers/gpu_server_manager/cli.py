"""Interactive CLI for the GPU Server Manager."""

import ast
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
from spatial_agent.launch_managers.gpu_server_manager.config import load_config
from spatial_agent.launch_managers.gpu_server_manager.dashboard import GPUServerDashboard
from spatial_agent.launch_managers.gpu_server_manager.server_chain import (
    recover_managed_chain,
    start_chain_background,
)
from spatial_agent.launch_managers.gpu_server_manager.state import (
    GPUServerStateManager,
    StateFileCorruptionError,
)
from spatial_agent.launch_managers.slurm_utils import (
    batch_query_jobs,
    cancel_jobs,
    check_slurm_available,
    get_exact_job_states,
    get_user_jobs_snapshot,
    resolve_partitions,
)


def _protocol_version(project_root: Path) -> int:
    """Read the wire version without importing GPU/runtime dependencies.

    The manager CLI runs on a login node and should remain usable in a bare
    control-plane environment.  Importing ``gpu_rpc`` there transitively loads
    NumPy/Pillow and other worker-side packages, so extract the literal from
    its authoritative source file with the standard library instead.
    """
    source_path = project_root / "spatial_agent" / "gpu_rpc.py"
    try:
        module = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise LifecycleError(
            f"Cannot read GPU RPC protocol authority from {source_path}: {exc}"
        ) from exc
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "PROTOCOL_VERSION"
            for target in targets
        ):
            continue
        value_node = node.value
        if isinstance(value_node, ast.Constant) and type(value_node.value) is int:
            return int(value_node.value)
        break
    raise LifecycleError(
        f"PROTOCOL_VERSION must be an integer literal in {source_path}"
    )


def _tool_spec(project_root: Path, *, backend: str, gpus: int) -> dict:
    tracked_files = [
        project_root / "spatial_agent" / "entrypoints" / "launch_gpu_server.py",
        project_root / "spatial_agent" / "launch_managers" / "gpu_server_manager" / "run_gpu_server.sh",
        project_root / "spatial_agent" / "gpu_rpc.py",
    ]
    tracked_files.extend(sorted(
        (project_root / "spatial_agent" / "gpu_models").glob("*.py")
    ))
    tracked_files.extend([
        project_root / "spatial_agent" / "tools" / "base.py",
        project_root / "spatial_agent" / "tools" / "reconstruct_tool.py",
        project_root / "spatial_agent" / "tools" / "sam3_tool.py",
        project_root / "spatial_agent" / "requirements" / "requirements-agent.txt",
    ])
    return {
        "schema_version": 1,
        "tools": ["Reconstruct", "SAM3"],
        "reconstruct_backend": backend,
        "gpus_per_instance": int(gpus),
        "protocol_version": _protocol_version(project_root),
        "launcher_files": {
            str(path.relative_to(project_root)): file_sha256(path)
            for path in tracked_files
        },
        "synthetic_keepalive": False,
    }


class _Abort(Exception):
    """Raised when user types 'q' to cancel during a multi-step flow."""


class GPUServerManagerCLI:

    STOP_VERIFY_ATTEMPTS = 10
    STOP_VERIFY_INTERVAL_SECONDS = 1.0

    def __init__(self):
        # Project root: gpu_server_manager/ → launch_managers/ → spatial_agent/ → repo root
        self.project_root = Path(__file__).parent.parent.parent.parent.absolute()
        self.config = load_config()
        self.service_lifecycle_enabled = load_agent_manager_config(
            self.project_root
        ).service_lifecycle_enabled
        self.state_manager = GPUServerStateManager(self.project_root)
        self.dashboard = GPUServerDashboard(self.project_root)
        self.console = Console()
        self._launch_stack = None

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

        try:
            self.state_manager.list_servers()
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
        for server in servers:
            intent = server.submission_intent
            if intent is None:
                continue
            comment = intent.get("comment") if isinstance(intent, dict) else None
            self.console.print(
                "[bold red]Submission fence:[/bold red] "
                f"chain={server.chain_id} pool={server.pool_id or 'legacy'} "
                f"comment={comment or 'invalid/unavailable'} "
                f"exact_jobs={','.join(map(str, server.slurm_job_ids)) or '-'}"
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
        with ExitStack() as launch_stack:
            self._launch_stack = launch_stack
            try:
                self._start_gpu_servers_flow()
            except _Abort:
                self.console.print("[dim]Cancelled.[/dim]")
            except (LifecycleError, OSError, ValueError) as exc:
                self.console.print(
                    f"[bold red]Refusing launch: identity lock/lifecycle "
                    f"authority failed: {exc}[/bold red]"
                )
            finally:
                self._launch_stack = None

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
        try:
            partition_default = resolve_partitions(
                self.config.default_slurm.partition, account
            )
        except RuntimeError as exc:
            self.console.print(f"[yellow]{exc}[/yellow]")
            partition_default = self.config.default_slurm.partition
        partition = self._ask(
            "[bold]partition",
            default=partition_default,
        )

        self.console.print()
        summary = (
            f"  Servers:    {num_servers}\n"
            f"  GPUs each:  {gpus}\n"
            f"  Backend:    {backend}\n"
            f"  Account:    {account}\n"
            f"  Partition:  {partition}\n"
            f"  Time limit: {self.config.default_slurm.time_limit}"
        )
        self.console.print(Panel(summary, title="Confirm GPU Server Launch", border_style="green"))

        confirm = Prompt.ask("[bold]Launch?", choices=["y", "n"], default="y")
        if confirm != "y":
            self.console.print("[dim]Cancelled.[/dim]")
            return

        if not getattr(self, "service_lifecycle_enabled", False):
            pending_authority = [
                server for server in self.state_manager.list_servers()
                if server.reconstruct_backend == backend
                and getattr(server, "submission_intent", None) is not None
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
            for i in range(num_servers):
                server_state, _chain_log = start_chain_background(
                    chain_id=str(uuid.uuid4()),
                    account=account,
                    partition=partition,
                    gpus=gpus,
                    reconstruct_backend=backend,
                    time_limit=self.config.default_slurm.time_limit,
                    restart_before_minutes=(
                        self.config.default_slurm.restart_before_minutes
                    ),
                    project_root=self.project_root,
                )
                self.console.print(
                    f"  [green]Legacy {i + 1}/{num_servers}[/green] — "
                    f"Chain {server_state.chain_id[:8]}, PID {server_state.pid}"
                )
            return

        tool_spec = _tool_spec(self.project_root, backend=backend, gpus=gpus)
        topology = {
            "schema_version": 1,
            "desired_instances": num_servers,
            "gpus_per_instance": gpus,
            "target_hardware": "scheduler-selected; realized hardware recorded per epoch",
        }
        lifecycle_policy = {
            "schema_version": 1,
            "owner_gated_successor": True,
            "auto_stop_when_unowned": True,
            "bootstrap_grace_minutes": 30,
            "synthetic_keepalive": False,
            "restart_before_minutes": self.config.default_slurm.restart_before_minutes,
            "time_limit": self.config.default_slurm.time_limit,
        }
        digests = component_digests(
            tool_spec=tool_spec,
            topology=topology,
            lifecycle_policy=lifecycle_policy,
        )
        lifecycle_store = LifecycleStore(self.project_root)
        if self._launch_stack is None:
            raise LifecycleError("Launch flow is missing its identity lock scope")
        self._launch_stack.enter_context(
            lifecycle_store.service_identity_launch_lock(
                "gpu_tools", backend,
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
        pool = lifecycle_store.ensure_pool(
            proposed_pool_id=f"tools-{uuid.uuid4().hex[:12]}",
            service_kind="gpu_tools",
            namespace=f"tools-{backend}-{uuid.uuid4().hex[:12]}",
            desired_instances=num_servers,
            tool_spec=tool_spec,
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
        self.state_manager.cleanup_dead_servers(alive_job_ids=jobs.keys())
        servers = self.state_manager.list_servers()
        compatible = []
        conflicts = []
        for server in servers:
            live_jobs = [
                job_id for job_id in server.slurm_job_ids
                if str(job_id) in jobs
            ]
            manager_alive = self.state_manager.is_server_alive(server)
            identity_verified = self.state_manager.is_server_identity_verified(server)
            if server.reconstruct_backend != backend:
                continue
            intent = server.submission_intent
            if intent is not None:
                comment = intent.get("comment") if isinstance(intent, dict) else None
                conflicts.append(
                    f"chain={server.chain_id[:8]} pool={server.pool_id or 'legacy'} "
                    "has pending submission authority "
                    f"comment={comment or 'invalid/unavailable'} "
                    f"jobs={','.join(map(str, server.slurm_job_ids)) or '-'}"
                )
                continue
            exact = (
                server.pool_id == pool["pool_id"]
                and server.namespace == pool["namespace"]
                and server.spec_digest == digests["tool_spec"]
                and server.topology_digest == digests["topology"]
            )
            if (
                exact
                and manager_alive
                and identity_verified
                and server.lifecycle_state != "quarantined"
            ):
                compatible.append(server)
            elif manager_alive or live_jobs:
                conflicts.append(
                    f"chain={server.chain_id[:8]} pool={server.pool_id or 'legacy'} "
                    f"jobs={','.join(map(str, live_jobs)) or '-'} "
                    f"pid_identity={'verified' if identity_verified else 'unknown'}"
                )

        conflicts.extend(tagged_capacity_conflicts(
            jobs,
            lifecycle_store.snapshot(),
            service_kind="gpu_tools",
            identity_value=backend,
            desired_pool_id=pool["pool_id"],
            compatible_chain_ids=[server.chain_id for server in compatible],
        ))

        registry_file = self.project_root / "spatial_agent" / "logs" / "gpu_server.json"
        try:
            with open(registry_file, encoding="utf-8") as handle:
                registry = json.load(handle)
        except FileNotFoundError:
            registry = {}
        except (json.JSONDecodeError, OSError) as exc:
            lifecycle_store.set_pool_status(
                pool["pool_id"], "quarantined",
                registry_error=str(exc),
            )
            self.console.print(
                f"[bold red]Refusing launch: GPU registry is unreadable or "
                f"corrupt: {exc}[/bold red]"
            )
            return
        if not isinstance(registry, dict):
            lifecycle_store.set_pool_status(
                pool["pool_id"], "quarantined",
                registry_error="registry root is not an object",
            )
            self.console.print(
                "[bold red]Refusing launch: GPU registry root is invalid.[/bold red]"
            )
            return
        managed_jobs = {
            str(job_id): server
            for server in compatible
            for job_id in server.slurm_job_ids
        }
        registry_shape_error = None
        for uid, info in registry.items():
            if not isinstance(info, dict):
                registry_shape_error = f"registry record {uid!r} is not an object"
                break
            job_id = str(info.get("slurm_job_id") or "")
            if not job_id or job_id not in jobs:
                continue
            managed_server = managed_jobs.get(job_id)
            if managed_server is not None:
                if not registry_record_attests_pool(
                    info,
                    pool,
                    identity_value=backend,
                    slot_id=managed_server.chain_id,
                ):
                    conflicts.append(
                        f"registry_uid={uid[:8]} job={job_id} has "
                        "incompatible managed-service attestation"
                    )
            elif info.get("reconstruct_backend") == backend:
                conflicts.append(
                    f"registry_uid={uid[:8]} pool={info.get('pool_id') or 'legacy'} "
                    f"job={job_id}"
                )
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
            service_kind="gpu_tools",
            identity_value=backend,
            desired_pool_id=pool["pool_id"],
        ))

        if conflicts:
            lifecycle_store.set_pool_status(
                pool["pool_id"], "quarantined", conflicts=conflicts,
            )
            self.console.print(
                "[bold red]Refusing launch: live tool capacity has incomplete "
                "or incompatible ownership.[/bold red]"
            )
            for conflict in conflicts:
                self.console.print(f"  [red]- {conflict}[/red]")
            return
        try:
            retired = lifecycle_store.retire_abandoned_identity_pools(
                service_kind="gpu_tools",
                identity_value=backend,
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
        if len(compatible) > num_servers:
            self.console.print(
                f"[bold yellow]Pool already has {len(compatible)} compatible "
                f"servers, above desired {num_servers}; no active scale-down "
                "was performed.[/bold yellow]"
            )
            return

        missing = num_servers - len(compatible)
        if missing == 0:
            self.console.print(
                f"[bold green]Desired tool total already satisfied: {num_servers}.[/bold green]"
            )
            return
        claim_id = f"cli-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        claimed_slots = lifecycle_store.claim_launch_slots(
            pool["pool_id"], claimant_id=claim_id,
            claimant_pid=os.getpid(), max_slots=missing,
        )
        if not claimed_slots:
            lifecycle_store.release_launch_claim(pool["pool_id"], claim_id)
            self.console.print(
                "[yellow]Missing capacity is already claimed or durably owned; "
                "no duplicate launch was made.[/yellow]"
            )
            return

        self.console.print()
        try:
            for i, chain_id in enumerate(claimed_slots):
                self.console.print(
                    f"[yellow]Starting missing GPU server {i + 1}/"
                    f"{len(claimed_slots)}...[/yellow]"
                )
                srv_state, chain_log = start_chain_background(
                    chain_id=chain_id,
                    account=account,
                    partition=partition,
                    gpus=gpus,
                    reconstruct_backend=backend,
                    time_limit=self.config.default_slurm.time_limit,
                    restart_before_minutes=self.config.default_slurm.restart_before_minutes,
                    project_root=self.project_root,
                    pool_id=pool["pool_id"],
                    namespace=pool["namespace"],
                    tool_spec_digest=digests["tool_spec"],
                    topology_digest=digests["topology"],
                    desired_instances=num_servers,
                    auto_stop_when_unowned=True,
                    bootstrap_grace_minutes=30,
                    launch_claim_id=claim_id,
                )
                self.console.print(
                    f"  [green]Started![/green] PID={srv_state.pid}, log={chain_log}"
                )
        finally:
            lifecycle_store.release_launch_claim(pool["pool_id"], claim_id)

        self.console.print()
        self.console.print(
            f"[bold green]Desired tool pool total: {num_servers} "
            f"({len(claimed_slots)} newly started).[/bold green]\n"
            f"[dim]Pool: {pool['pool_id']}  namespace: {pool['namespace']}[/dim]\n"
            f"[dim]Use Dashboard to monitor.[/dim]"
        )

    def _stop_gpu_servers(self) -> None:
        self.console.print()

        # Rendering/selecting is read-only; selected managed and legacy
        # targets enter their separate authoritative stop paths below.
        servers = self.state_manager.list_servers()
        alive_servers = [
            (i, server) for i, server in enumerate(servers)
            if self.state_manager.is_server_alive(server)
            or (
                server.pool_id
                and (
                    server.slurm_job_ids
                    or server.submission_intent is not None
                    or server.lifecycle_state == "quarantined"
                )
            )
        ]

        if not alive_servers:
            self.console.print(
                "[dim]No running or quarantined managed GPU servers to "
                "stop.[/dim]"
            )
            return

        all_jids = []
        for _, s in alive_servers:
            all_jids.extend(s.slurm_job_ids)
        job_info_map = batch_query_jobs(all_jids) if all_jids else {}

        table = Table(
            title="Active / Recoverable GPU Servers",
            title_style="bold red",
            border_style="red",
            padding=(0, 1),
        )
        table.add_column("#", style="bold", width=4, justify="right")
        table.add_column("GPUs", width=5, justify="center")
        table.add_column("Backend", min_width=6)
        table.add_column("PID", width=8)
        table.add_column("Manager", width=12)
        table.add_column("SLURM Jobs", min_width=16)
        table.add_column("Account", min_width=14)

        for display_idx, (_, srv) in enumerate(alive_servers, 1):
            active_jobs = [jid for jid in srv.slurm_job_ids if jid in job_info_map]
            jobs_str = ", ".join(active_jobs) if active_jobs else "[dim]-[/dim]"
            table.add_row(
                str(display_idx), str(srv.gpus), srv.reconstruct_backend,
                str(srv.pid),
                (
                    "[green]alive[/green]"
                    if self.state_manager.is_server_alive(srv)
                    else "[bold yellow]recovery[/bold yellow]"
                ),
                jobs_str, srv.account,
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

        managed_reconciled = True
        if any(server.pool_id for server in targets):
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
        for server in targets:
            if server.pool_id and not managed_reconciled:
                refused.append((server, "lifecycle reconciliation is unavailable"))
            elif server.pool_id and lifecycle_store.active_leases_for_pool(server.pool_id):
                refused.append((server, "an owner lease is still active"))
            else:
                allowed.append(server)
        for server, reason in refused:
            self.console.print(
                f"[bold red]Refused {server.chain_id[:8]}: {reason}.[/bold red]"
            )
        if not allowed:
            return

        recovery_outcomes = []
        for server in allowed:
            if not server.pool_id:
                continue
            if self.state_manager.is_server_alive(server):
                if not self.state_manager.is_server_identity_verified(server):
                    recovery_outcomes.append((server, "identity_unverified"))
                    continue
                if not signal_process_record(
                    server.pid, server.process_identity, signal.SIGTERM,
                ):
                    recovery_outcomes.append((server, "identity_unverified"))
            elif server.pool_id:
                recovery_outcomes.append(
                    (server, recover_managed_chain(server, self.project_root))
                )

        legacy = [server for server in allowed if not server.pool_id]
        legacy_outcomes = [
            (server, self._stop_legacy_server_exact(server)) for server in legacy
        ]

        managed_count = len([
            server for server in allowed
            if server.pool_id and self.state_manager.is_server_alive(server)
        ])
        if managed_count:
            self.console.print(
                f"[green]Requested graceful drain for {managed_count} managed "
                "GPU server(s).[/green]"
            )
        for server, outcome in recovery_outcomes:
            style = "green" if outcome == "stopped" else "yellow"
            self.console.print(
                f"[{style}]Recovery {server.chain_id[:8]}: "
                f"{outcome}.[/{style}]"
            )
        for server, stopped in legacy_outcomes:
            style = "green" if stopped else "yellow"
            outcome = "stopped" if stopped else "retained/quarantined"
            self.console.print(
                f"[{style}]Legacy {server.chain_id[:8]}: "
                f"{outcome}.[/{style}]"
            )
        self.console.print()

    def _retain_legacy_server(self, server) -> bool:
        server.lifecycle_state = "quarantined"
        self.state_manager.add_server(server)
        return False

    def _stop_legacy_server_exact(self, server) -> bool:
        """Stop one legacy server only with exact authoritative proof."""
        job_ids = sorted({str(job) for job in server.slurm_job_ids if job})
        if any(not job_id.isdigit() for job_id in job_ids):
            return self._retain_legacy_server(server)

        if self.state_manager.is_server_alive(server):
            if not self.state_manager.is_server_identity_verified(server):
                return self._retain_legacy_server(server)
            if not signal_process_record(
                server.pid, server.process_identity, signal.SIGTERM,
            ):
                return self._retain_legacy_server(server)

        intent = server.submission_intent
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
                != f"spatialclaw:v1::{server.chain_id}:{generation}"
            ):
                return self._retain_legacy_server(server)
            snapshot = get_user_jobs_snapshot(force=True)
            if not comment or not getattr(snapshot, "authoritative", False):
                return self._retain_legacy_server(server)
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
                return self._retain_legacy_server(server)
            job_ids = sorted(set(job_ids) | set(matches) | set(intent_ids))

        states = get_exact_job_states(job_ids, force=True)
        if states is None:
            return self._retain_legacy_server(server)
        if not job_ids:
            snapshot = get_user_jobs_snapshot(force=True)
            if not getattr(snapshot, "authoritative", False):
                return self._retain_legacy_server(server)
        live_ids = sorted(states)
        if live_ids and cancel_jobs(live_ids) != len(live_ids):
            return self._retain_legacy_server(server)

        terminal = False
        for attempt in range(self.STOP_VERIFY_ATTEMPTS):
            states = get_exact_job_states(job_ids, force=True)
            if states is not None and not states:
                terminal = True
                if not self.state_manager.is_server_alive(server):
                    break
            if attempt + 1 < self.STOP_VERIFY_ATTEMPTS:
                time.sleep(self.STOP_VERIFY_INTERVAL_SECONDS)
        if not terminal or self.state_manager.is_server_alive(server):
            return self._retain_legacy_server(server)

        current = next(
            (
                record for record in self.state_manager.list_servers()
                if record.chain_id == server.chain_id
            ),
            None,
        )
        if current is None:
            return True
        current_ids = {str(job) for job in current.slurm_job_ids if job}
        if current_ids - set(job_ids):
            return self._retain_legacy_server(current)
        if current.submission_intent is not None:
            if current.submission_intent != intent:
                return self._retain_legacy_server(current)
            try:
                self.state_manager.update_server_submission(
                    current.chain_id,
                    slurm_job_ids=job_ids,
                    submission_intent=None,
                )
            except Exception:
                return self._retain_legacy_server(current)
            current = next(
                (
                    record for record in self.state_manager.list_servers()
                    if record.chain_id == server.chain_id
                ),
                None,
            )
            if current is None:
                return True
            if current.submission_intent is not None or {
                str(job) for job in current.slurm_job_ids if job
            } - set(job_ids):
                return self._retain_legacy_server(current)
        self.state_manager.remove_server(server.chain_id)
        return True
