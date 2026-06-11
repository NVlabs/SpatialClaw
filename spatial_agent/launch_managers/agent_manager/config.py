"""Load agent manager configuration and discover available benchmarks/models."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class SlurmDefaults:
    gpus: int = 0  # CPU-only (GPU tools are on external GPU server)
    time_limit: str = "4:00:00"
    partition: str = "cpu_short,cpu"


@dataclass
class CoTSlurmDefaults:
    gpus: int = 0
    time_limit: str = "4:00:00"
    partition: str = "cpu_short"


@dataclass
class ManagerConfig:
    accounts: List[str] = field(default_factory=list)
    default_slurm: SlurmDefaults = field(default_factory=SlurmDefaults)
    default_concurrency: int = 8
    default_subsample: int = 0
    benchmarks: List[str] = field(default_factory=list)
    model_configs: List[str] = field(default_factory=list)
    # CoT-specific defaults
    cot_slurm: CoTSlurmDefaults = field(default_factory=CoTSlurmDefaults)
    cot_default_concurrency: int = 32
    cot_default_max_frames: int = 32
    cot_default_system_prompt: str = "cot"


def load_config(project_root: Path = None) -> ManagerConfig:
    """Load config.json and discover available benchmarks/models."""
    if project_root is None:
        # agent_manager/ → launch_managers/ → spatial_agent/ → project root.
        project_root = Path(__file__).parent.parent.parent.parent.absolute()

    config_path = Path(__file__).parent / "config.json"

    with open(config_path, "r") as f:
        raw = json.load(f)

    slurm_defaults = SlurmDefaults(**raw.get("default_slurm", {}))
    cot_slurm = CoTSlurmDefaults(**raw.get("cot_slurm", {}))

    # Discover benchmarks from config/dataset/*.json
    dataset_dir = project_root / "spatial_agent" / "config" / "dataset"
    benchmarks = sorted(
        p.stem for p in dataset_dir.glob("*.json")
    ) if dataset_dir.exists() else []

    # Discover models from config/model/*.json
    model_dir = project_root / "spatial_agent" / "config" / "model"
    model_configs = sorted(
        p.stem for p in model_dir.glob("*.json")
    ) if model_dir.exists() else []

    return ManagerConfig(
        accounts=raw.get("accounts", []),
        default_slurm=slurm_defaults,
        default_concurrency=raw.get("default_concurrency", 8),
        default_subsample=raw.get("default_subsample", 0),
        benchmarks=benchmarks,
        model_configs=model_configs,
        cot_slurm=cot_slurm,
        cot_default_concurrency=raw.get("cot_default_concurrency", 32),
        cot_default_max_frames=raw.get("cot_default_max_frames", 32),
        cot_default_system_prompt=raw.get("cot_default_system_prompt", "cot"),
    )
