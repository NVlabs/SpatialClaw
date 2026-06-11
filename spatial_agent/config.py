"""Configuration for the Spatial Understanding Agent.

Loading priority: CLI Arguments > Model/Dataset JSON > Environment Variables > Defaults.
Env vars use prefix SPATIAL_AGENT_ (e.g., SPATIAL_AGENT_BENCHMARK=erqa).

Config is split into two orthogonal dimensions:
- Model config (config/model/<model>.json): LLM connection + per-role hyperparameters
- Dataset config (config/dataset/<benchmark>.json): benchmark, tools, agent loop params

String values loaded from JSON configs are passed through ``${VAR}`` env-var
expansion so that secrets such as API keys can live in the environment (or a
``.env`` file at the project root) rather than in version-controlled JSON.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# .env loading and ${VAR} expansion
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_DOTENV_LOADED = False


def _load_dotenv() -> None:
    """Populate os.environ from a project-root ``.env`` file if present.

    The lookup walks up from this file until it finds a ``.env`` or hits the
    filesystem root. Existing environment variables take precedence — values
    from ``.env`` only fill in what isn't already set, so an explicit
    ``KEY=... python ...`` invocation always wins.

    Format is the usual ``KEY=value`` per line with ``#`` comments and
    optional surrounding quotes. No expansion is done inside ``.env`` itself.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):  # walk up at most 6 levels
        candidate = os.path.join(here, ".env")
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("export "):
                            line = line[len("export "):].lstrip()
                        if "=" not in line:
                            continue
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip()
                        if (val.startswith('"') and val.endswith('"')) or (
                            val.startswith("'") and val.endswith("'")
                        ):
                            val = val[1:-1]
                        os.environ.setdefault(key, val)
            except OSError:
                pass
            return
        parent = os.path.dirname(here)
        if parent == here:
            return
        here = parent


def _expand_env_vars(value: Any) -> Any:
    """Expand ``${VAR}`` / ``${VAR:-default}`` references inside a string.

    Non-string inputs pass through unchanged. A missing variable with no
    default expands to an empty string (matches shell behavior); callers that
    treat the empty string as "no key" then fall back to their own defaults.
    """
    if not isinstance(value, str):
        return value

    def _sub(match: "re.Match[str]") -> str:
        name, default = match.group(1), match.group(2)
        return os.environ.get(name, default if default is not None else "")

    return _ENV_VAR_RE.sub(_sub, value)


@dataclass
class LLMRoleParams:
    """Per-role LLM hyperparameters.

    Supported by all OpenAI-compatible backends: max_tokens, temperature, top_p,
    presence_penalty. vLLM-specific: top_k, min_p, repetition_penalty,
    enable_thinking (sent via extra_body).
    """

    max_tokens: int = 16384
    temperature: float = 0.6
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    enable_thinking: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LLMRoleParams":
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys and v is not None})


@dataclass
class SpatialAgentConfig:
    """All configuration fields for the Spatial Understanding Agent."""

    # --- Benchmark ---
    benchmark: str = "erqa"
    question_type: Optional[List[str]] = None
    limit: Optional[int] = None
    sample_ids: Optional[List[str]] = None  # evaluate only these sample IDs

    # --- LLM Connection ---
    llm_model: str = ""
    llm_base_url: str = ""  # 'vllm' triggers serve.json auto-discovery
    llm_api_key: str = ""

    # --- LLM Role Parameters ---
    main_params: LLMRoleParams = field(default_factory=LLMRoleParams)
    vlm_params: LLMRoleParams = field(
        default_factory=lambda: LLMRoleParams(temperature=0.1, enable_thinking=True)
    )
    vlm_grounding_params: LLMRoleParams = field(
        default_factory=lambda: LLMRoleParams(
            max_tokens=16384, temperature=0.1, enable_thinking=True
        )
    )
    planning_params: LLMRoleParams = field(
        default_factory=lambda: LLMRoleParams(temperature=1.0)
    )
    general_params: LLMRoleParams = field(
        default_factory=lambda: LLMRoleParams(max_tokens=32768, enable_thinking=False)
    )

    # --- Agent Loop ---
    max_steps: int = 30
    max_failures: int = 30
    max_tool_calls: int = -1
    timeout_sec: int = 600  # per Jupyter cell execution

    # --- Executor Variant ---
    # "code"        : free-form Python per step (default, the SpatialClaw interface).
    # "react"       : one structured tool-call per step; translated to code.
    # "single_pass" : single complete program; pair with max_steps=1 and
    #                 enable_planning=false to match the paper's baseline.
    executor_type: str = "code"

    # --- Tools ---
    tools_to_use: List[str] = field(
        default_factory=lambda: ["Reconstruct", "SAM3"]
    )
    reconstruct_backend: str = "da3"  # "pi3" or "da3"
    reconstruct_max_frames: int = 64
    sam3_max_video_frames: int = 1000  # max frames for SAM3 video segmentation

    # --- VLM Queries ---
    vlm_query_timeout_sec: int = 600

    # --- Video ---
    video_max_fps: Optional[float] = None
    video_frame_resize_short_edge: Optional[int] = None
    image_max_long_edge: Optional[int] = 768  # None = no resize
    num_key_frames: int = 32  # key frames shown to main LLM (0 = blind mode)

    # --- Logging ---
    work_dir: Optional[str] = None
    enable_logging: bool = True
    generate_report: bool = True

    # --- Planning ---
    enable_planning: bool = True

    # --- Reflection ---
    enable_reflection: bool = False
    reflect_every_n_steps: int = 5  # reflect every N steps + ReturnAnswer steps (1 = every step)
    max_answer_blocks: int = -1  # max consecutive ReturnAnswer blocks before force-accept (-1 = unlimited)
    max_total_answer_attempts: int = 5  # max cumulative ReturnAnswer attempts before force-accept (-1 = unlimited)
    min_budget_for_answer_reject: int = 3  # reflection won't reject answers when budget <= this (-1 = always reject)
    reflection_params: LLMRoleParams = field(
        default_factory=lambda: LLMRoleParams(
            temperature=0.6, max_tokens=2048, enable_thinking=True
        )
    )

    # --- Runtime ---
    concurrency: int = 8

    # --- Sighted Feedback ---
    enable_sighted_feedback: bool = True  # show() sends images inline in feedback
    max_show_images_per_step: int = -1  # per-step cap (-1 = disabled)
    max_show_images_per_session: int = 250  # total show() images per session (-1 = unlimited)
    show_image_labels: bool = True  # white padding with label on top of show() images

    # --- GPU Server ---
    gpu_server: str = "auto"  # "auto" discovers from gpu_server.json; "http://host:port" for direct
    gpu_tool_max_retries: int = 3  # max retries per GPU tool call (-1 = no retry)

    # --- Context Management ---
    condense_errors: bool = True  # replace verbose AIMessages of errored steps with condensed versions

    # --- Prompt Ablation ---
    prompt_section_ablations: Dict[str, Any] = field(default_factory=dict)
    # Format: {"exclude": ["section_name", ...], "override": {"section_name": "path/to/file.md"}}
    # Main prompt sections (unprefixed): header, response_format, show_api, vlm_api,
    #   available_tools, coordinate_system, robust_computation, evidence_hierarchy,
    #   return_answer, code_rules, workflow, session_input
    # Planning prompt sections (planning_ prefix): planning_header, planning_available_tools,
    #   planning_show_api, planning_vlm_api, planning_coordinate_system, planning_tool_decision,
    #   planning_input, planning_task, planning_rules, planning_checklist
    # Reflection prompt sections (reflection_ prefix): reflection_base,
    #   reflection_what_to_check, reflection_checklist, reflection_output_format

    # --- Resource Guardrails ---
    max_variable_size_mb: int = 500  # warn and instruct del if exceeded

    def _load_from_envs(self) -> None:
        """Load configuration from environment variables with SPATIAL_AGENT_ prefix.

        Only handles connection params and simple dataclass fields.
        Role params come from model config files only.
        """
        for f in fields(self):
            # Skip LLMRoleParams fields — those come from model config files
            val = getattr(self, f.name, None)
            if isinstance(val, LLMRoleParams):
                continue
            env_name = f"SPATIAL_AGENT_{f.name.upper()}"
            env_val = os.getenv(env_name)
            if env_val is None:
                continue
            converted = self._convert_type(env_val, f.type, f.name)
            if converted is not None:
                setattr(self, f.name, converted)

    @staticmethod
    def _convert_type(value: str, type_hint: Any, field_name: str) -> Any:
        """Convert string env var to the appropriate Python type."""
        type_str = str(type_hint)
        if "bool" in type_str:
            return value.lower() in ("true", "1", "yes")
        if "int" in type_str and "Optional" not in type_str:
            return int(value)
        if "Optional[int]" in type_str:
            return int(value) if value else None
        if "float" in type_str and "Optional" not in type_str:
            return float(value)
        if "Optional[float]" in type_str:
            return float(value) if value else None
        if "List[str]" in type_str:
            return [s.strip() for s in value.split(",")]
        return value

    def update_from_model_json(self, config_path: str) -> None:
        """Load LLM connection params and per-role hyperparameters from a model config.

        Expected format::

            {
                "llm_model": "...",
                "llm_base_url": "vllm",
                "llm_api_key": "bearer",
                "roles": {
                    "main": {"max_tokens": 16384, "temperature": 0.6},
                    "vlm": {"max_tokens": 131072, "temperature": 0.1, "enable_thinking": true},
                    "vlm_grounding": {"max_tokens": 4096, "temperature": 0.1, "enable_thinking": false},
                    "planning": {"max_tokens": 16384, "temperature": 1.0},
                    "general": {"max_tokens": 32768, "temperature": 0.6}
                }
            }
        """
        with open(config_path, "r") as f:
            data = json.load(f)

        # Entrypoints construct SpatialAgentConfig() directly without going
        # through get_config(), so .env must be loaded here — before ${VAR}
        # expansion — or keys defined only in .env expand to "".
        _load_dotenv()

        # Connection params (expand ${VAR} references so api keys can come
        # from environment / .env rather than the JSON file itself)
        for key in ("llm_model", "llm_base_url", "llm_api_key"):
            if key in data and data[key] is not None:
                setattr(self, key, _expand_env_vars(data[key]))

        # Role params
        roles = data.get("roles", {})
        role_map = {
            "main": "main_params",
            "vlm": "vlm_params",
            "vlm_grounding": "vlm_grounding_params",
            "planning": "planning_params",
            "general": "general_params",
            "reflection": "reflection_params",
        }
        for role_name, attr_name in role_map.items():
            if role_name in roles:
                setattr(self, attr_name, LLMRoleParams.from_dict(roles[role_name]))

    def update_from_dataset_json(self, config_path: str) -> None:
        """Load dataset/benchmark overrides from a dataset config file."""
        with open(config_path, "r") as f:
            data = json.load(f)
        for key, value in data.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def update_from_args(self, args) -> None:
        """Load overrides from argparse namespace."""
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dict for logging."""
        result = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, LLMRoleParams):
                result[f.name] = val.to_dict()
            else:
                result[f.name] = val
        return result


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_config: Optional[SpatialAgentConfig] = None


def get_config() -> SpatialAgentConfig:
    """Return the global config singleton, creating it if needed."""
    global _config
    if _config is None:
        _load_dotenv()
        _config = SpatialAgentConfig()
        _config._load_from_envs()
    return _config


def set_config(config: SpatialAgentConfig) -> None:
    """Replace the global config singleton."""
    global _config
    _config = config
