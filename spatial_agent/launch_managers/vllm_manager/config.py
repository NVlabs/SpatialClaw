"""Load and validate models.json configuration."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ModelConfig:
    name: str
    model: str
    served_name: str
    max_model_len: int
    max_num_seqs: int
    tp_size: int
    kv_cache_dtype: str = "auto"
    quantization: str = "none"
    partition: str = ""


@dataclass
class SlurmDefaults:
    gpus: int = 8
    time_limit: str = "4:00:00"
    restart_before_minutes: int = 20


@dataclass
class ManagerConfig:
    accounts: List[str] = field(default_factory=list)
    default_slurm: SlurmDefaults = field(default_factory=SlurmDefaults)
    models: List[ModelConfig] = field(default_factory=list)


def load_config(config_path: Path = None) -> ManagerConfig:
    """Load models.json from package directory (or custom path)."""
    if config_path is None:
        config_path = Path(__file__).parent / "models.json"

    with open(config_path, "r") as f:
        raw = json.load(f)

    slurm_defaults = SlurmDefaults(**raw.get("default_slurm", {}))

    models = []
    for m in raw.get("models", []):
        models.append(ModelConfig(
            name=m["name"],
            model=m["model"],
            served_name=m["served_name"],
            max_model_len=m["max_model_len"],
            max_num_seqs=m["max_num_seqs"],
            tp_size=m["tp_size"],
            kv_cache_dtype=m.get("kv_cache_dtype", "auto"),
            quantization=m.get("quantization", "none"),
            partition=m["partition"],
        ))

    return ManagerConfig(
        accounts=raw.get("accounts", []),
        default_slurm=slurm_defaults,
        models=models,
    )
