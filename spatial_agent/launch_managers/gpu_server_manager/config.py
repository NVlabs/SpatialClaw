"""Load and validate gpu_server_manager config."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class SlurmDefaults:
    gpus: int = 1
    time_limit: str = "4:00:00"
    restart_before_minutes: int = 20
    partition: str = "grizzly,polar,polar3,polar4"


@dataclass
class GPUServerManagerConfig:
    accounts: List[str] = field(default_factory=list)
    default_slurm: SlurmDefaults = field(default_factory=SlurmDefaults)
    reconstruct_backend: str = "da3"


def load_config(config_path: Path = None) -> GPUServerManagerConfig:
    """Load config.json from package directory (or custom path)."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"

    with open(config_path, "r") as f:
        raw = json.load(f)

    slurm_defaults = SlurmDefaults(**raw.get("default_slurm", {}))

    return GPUServerManagerConfig(
        accounts=raw.get("accounts", []),
        default_slurm=slurm_defaults,
        reconstruct_backend=raw.get("reconstruct_backend", "da3"),
    )
