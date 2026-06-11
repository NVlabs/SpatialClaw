"""Count currently running agent experiments with per-benchmark/model breakdown."""

from collections import Counter
from pathlib import Path
from typing import NamedTuple

from spatial_agent.launch_managers.agent_manager.state import ExperimentStateManager


class AgentSnapshot(NamedTuple):
    total: int
    by_benchmark: dict[str, int]
    by_model: dict[str, int]
    experiment_ids: list[str]


def snapshot(project_root: Path) -> AgentSnapshot:
    """Read agent_manager_state.json and count rows with status='running'."""
    try:
        mgr = ExperimentStateManager(project_root)
        experiments = mgr.list_experiments()
    except Exception:
        return AgentSnapshot(0, {}, {}, [])

    running = [e for e in experiments if (e.status or "").lower() == "running"]
    # Filter out stale entries whose chain process is dead.
    alive = []
    for e in running:
        try:
            if mgr.is_experiment_alive(e):
                alive.append(e)
        except Exception:
            alive.append(e)

    by_bench = Counter(e.benchmark or "unknown" for e in alive)
    by_model = Counter(e.model_name or "unknown" for e in alive)
    ids = [e.experiment_id for e in alive]
    return AgentSnapshot(len(alive), dict(by_bench), dict(by_model), ids)
