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
    seed: Optional[int] = None

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
    # Optional model-profile identity used for output isolation.  It does not
    # change the served model name and therefore remains dataset-agnostic.
    model_profile: Optional[str] = None
    # Model configs may opt in to locking their declared model identity so a
    # later CLI override cannot move model-specific behavior to a backbone for
    # which it was not designed.
    lock_llm_model: bool = False

    # --- Opt-in LLM retry controls ---
    # ``None`` preserves the historical OpenAI SDK and outer-loop behavior.
    # Model profiles may disable hidden SDK retries and explicit outer
    # generation-deadline retries without changing transport/server retries.
    llm_sdk_max_retries: Optional[int] = None
    llm_timeout_retries: Optional[int] = None
    finalize_on_llm_timeout: bool = False
    # Explicit outer request budgets. ``None`` preserves the historical
    # environment-variable/default resolution. Model profiles pin these so
    # an inherited shell environment cannot silently change benchmark results.
    llm_request_timeout_sec: Optional[float] = None
    vlm_request_timeout_sec: Optional[float] = None

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
    direct_answer_params: LLMRoleParams = field(
        default_factory=lambda: LLMRoleParams(max_tokens=32768, enable_thinking=False)
    )

    # --- Agent Loop ---
    max_steps: int = 30
    max_failures: int = 30
    max_tool_calls: int = -1
    timeout_sec: int = 600  # per Jupyter cell execution

    # --- Opt-in Protocol Fixes ---
    # Disabled by default so existing backbone configs retain legacy behavior.
    # Model configs may enable these fixes independently.
    explicit_planner_frame_mapping: bool = False
    safe_multimodal_fallback: bool = False
    # Encode an entire image batch on a worker thread before constructing a
    # vision request.  The encoded bytes and ordering are unchanged; only the
    # CPU-bound work is moved off the asyncio event loop. This is model opt-in
    # because historical backbone profiles must retain their execution path.
    offload_multimodal_encoding: bool = False

    # Normalize schema-valid bare-name answer submissions such as
    # ``ReturnAnswer(A)`` to ``ReturnAnswer("A")``. The implementation is AST
    # guarded and leaves strings, comments, attribute calls, and known kernel
    # variables untouched. Only affects the agent loop; the CoT baseline never
    # executes code.
    normalize_answer_symbols: bool = False
    # Record exact, result-neutral RGB fingerprints of the visual inputs used
    # by an evaluation.  Disabled by default; model profiles opt in when
    # overlap-matched comparisons require auditable input parity.
    record_input_fingerprints: bool = False
    # Visual preparation used by the CoT comparator.  Keeping the historical
    # default here makes overlap-matched preparation a model-profile choice.
    cot_frame_preparation: str = "legacy"

    # Controller-managed, model-specific behavior. All fields are inert by
    # default so existing backbones keep the historical graph behavior.
    direct_answer_anchor: bool = False
    evidence_gated_override: bool = False
    restrict_adjudicator_to_candidates: bool = False
    require_novel_visual_override_evidence: bool = False
    # When the loop's proposal disagrees with the anchor and no override
    # evidence was collected, run the restricted adjudication instead of
    # silently returning the anchor. On inline-image datasets every input is a
    # key frame, so evidence_count is structurally zero and the anchor would
    # otherwise win every disagreement unexamined.
    adjudicate_when_no_evidence: bool = False
    # Count the first inspection of each inline input image as loop progress
    # (new_evidence_last_step) without crediting it as override evidence.
    # Prevents soft_step_limit from starving image-only samples whose only
    # possible visual action is re-inspecting inline inputs. Inert on video
    # inputs, where novel frames are acquirable and count normally.
    credit_inline_reinspection: bool = False
    # Accept a bare schema-valid answer token (e.g. a response that is only
    # "C") as an answer submission instead of a format-validation failure.
    accept_bare_letter_answers: bool = False
    # Recover embedded multiple-choice labels without mistaking label-like
    # option text for additional choices.  This parser correction is an
    # independent model opt-in so every other backbone retains its historical
    # answer-schema behavior.
    robust_embedded_choice_schema: bool = False
    reserve_final_answer_turn: bool = False
    no_progress_limit: int = 0  # repeated action+outcome transitions; 0 disables
    soft_step_limit: int = 0  # stop after this many steps without fresh evidence

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

        profile = data.get("model_profile")
        if profile is not None and (
            not isinstance(profile, str)
            or not profile
            or re.fullmatch(r"[A-Za-z0-9._-]+", profile) is None
        ):
            raise ValueError(
                "Model config field 'model_profile' must be a non-empty "
                "filesystem-safe string"
            )
        self.model_profile = profile

        lock_llm_model = data.get("lock_llm_model", False)
        if not isinstance(lock_llm_model, bool):
            raise ValueError("Model config field 'lock_llm_model' must be a boolean")
        declared_model = data.get("llm_model")
        expanded_declared_model = _expand_env_vars(declared_model)
        if lock_llm_model and (
            not isinstance(declared_model, str)
            or not isinstance(expanded_declared_model, str)
            or not expanded_declared_model
        ):
            raise ValueError(
                "Model config field 'lock_llm_model' requires this model JSON "
                "to declare a non-empty string 'llm_model'"
            )
        self.lock_llm_model = lock_llm_model
        self._locked_llm_model = (
            expanded_declared_model if lock_llm_model else None
        )

        # Optional model-specific protocol fixes. Keeping these default-off
        # preserves legacy behavior for every backbone that does not opt in.
        for key in (
            "explicit_planner_frame_mapping",
            "safe_multimodal_fallback",
            "offload_multimodal_encoding",
            "normalize_answer_symbols",
            "record_input_fingerprints",
            "enable_planning",
            "direct_answer_anchor",
            "evidence_gated_override",
            "restrict_adjudicator_to_candidates",
            "require_novel_visual_override_evidence",
            "adjudicate_when_no_evidence",
            "credit_inline_reinspection",
            "accept_bare_letter_answers",
            "robust_embedded_choice_schema",
            "reserve_final_answer_turn",
            "finalize_on_llm_timeout",
        ):
            if key in data and data[key] is not None:
                if not isinstance(data[key], bool):
                    raise ValueError(f"Model config field {key!r} must be a boolean")
                setattr(self, key, data[key])

        # Model-scoped retry and request-deadline controls.
        for key in ("llm_sdk_max_retries", "llm_timeout_retries"):
            if key not in data or data[key] is None:
                continue
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Model config field {key!r} must be an integer >= 0"
                )
            setattr(self, key, value)

        for key in ("llm_request_timeout_sec", "vlm_request_timeout_sec"):
            if key not in data or data[key] is None:
                continue
            value = data[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"Model config field {key!r} must be a number > 0")
            setattr(self, key, float(value))

        # Opt-in integer loop-policy overrides. Same rationale: absent keys
        # leave every other backbone's behavior untouched.
        for key in (
            "max_steps",
            "max_failures",
            "no_progress_limit",
            "soft_step_limit",
        ):
            if key in data and data[key] is not None:
                if isinstance(data[key], bool) or not isinstance(data[key], int):
                    raise ValueError(f"Model config field {key!r} must be an integer")
                setattr(self, key, data[key])

        for key in (
            "max_steps",
            "max_failures",
            "no_progress_limit",
            "soft_step_limit",
        ):
            value = getattr(self, key)
            minimum = 1 if key in ("max_steps", "max_failures") else 0
            if value < minimum:
                raise ValueError(
                    f"Model config field {key!r} must be >= {minimum}, got {value}"
                )
        if self.soft_step_limit and self.soft_step_limit >= self.max_steps:
            raise ValueError(
                "Model config field 'soft_step_limit' must be less than 'max_steps'"
            )

        if "executor_type" in data and data["executor_type"] is not None:
            executor_type = data["executor_type"]
            if executor_type not in {"code", "react", "single_pass"}:
                raise ValueError(
                    "Model config field 'executor_type' must be one of "
                    "'code', 'react', or 'single_pass'"
                )
            self.executor_type = executor_type

        if (
            "cot_frame_preparation" in data
            and data["cot_frame_preparation"] is not None
        ):
            cot_frame_preparation = data["cot_frame_preparation"]
            if cot_frame_preparation not in {"legacy", "spatial-agent"}:
                raise ValueError(
                    "Model config field 'cot_frame_preparation' must be "
                    "'legacy' or 'spatial-agent'"
                )
            self.cot_frame_preparation = cot_frame_preparation

        # Role params
        roles = data.get("roles", {})
        role_map = {
            "main": "main_params",
            "vlm": "vlm_params",
            "vlm_grounding": "vlm_grounding_params",
            "planning": "planning_params",
            "general": "general_params",
            "direct_answer": "direct_answer_params",
            "reflection": "reflection_params",
        }
        for role_name, attr_name in role_map.items():
            if role_name in roles:
                setattr(self, attr_name, LLMRoleParams.from_dict(roles[role_name]))

    def update_from_dataset_json(self, config_path: str) -> None:
        """Load dataset/benchmark overrides from a dataset config file."""
        with open(config_path, "r") as f:
            data = json.load(f)
        model_only_fields = {
            "adjudicate_when_no_evidence",
            "credit_inline_reinspection",
            "accept_bare_letter_answers",
            "robust_embedded_choice_schema",
            "cot_frame_preparation",
        }
        misplaced = sorted(model_only_fields.intersection(data))
        if misplaced:
            raise ValueError(
                "Dataset config may not set model-only field(s): "
                + ", ".join(misplaced)
            )
        for key, value in data.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def update_from_args(self, args) -> None:
        """Load overrides from argparse namespace."""
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def validate(self) -> None:
        """Validate the fully layered dataset/model/CLI configuration."""
        if not isinstance(self.lock_llm_model, bool):
            raise ValueError("Config field 'lock_llm_model' must be a boolean")
        if self.model_profile is not None and (
            not isinstance(self.model_profile, str)
            or not self.model_profile
            or re.fullmatch(r"[A-Za-z0-9._-]+", self.model_profile) is None
        ):
            raise ValueError(
                "Config field 'model_profile' must be a non-empty "
                "filesystem-safe string"
            )
        locked_model = getattr(self, "_locked_llm_model", None)
        if locked_model is not None and self.llm_model != locked_model:
            raise ValueError(
                "Model config locks llm_model="
                f"{locked_model!r}; got conflicting override {self.llm_model!r}"
            )
        for key in (
            "enable_planning",
            "direct_answer_anchor",
            "evidence_gated_override",
            "restrict_adjudicator_to_candidates",
            "require_novel_visual_override_evidence",
            "adjudicate_when_no_evidence",
            "credit_inline_reinspection",
            "accept_bare_letter_answers",
            "robust_embedded_choice_schema",
            "reserve_final_answer_turn",
            "normalize_answer_symbols",
            "record_input_fingerprints",
            "offload_multimodal_encoding",
            "finalize_on_llm_timeout",
        ):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"Config field {key!r} must be a boolean")
        for key in (
            "max_steps",
            "max_failures",
            "no_progress_limit",
            "soft_step_limit",
        ):
            value = getattr(self, key)
            minimum = 1 if key in ("max_steps", "max_failures") else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Config field {key!r} must be an integer >= {minimum}")
        if self.soft_step_limit and self.soft_step_limit >= self.max_steps:
            raise ValueError("Config field 'soft_step_limit' must be less than 'max_steps'")
        for key in ("llm_sdk_max_retries", "llm_timeout_retries"):
            value = getattr(self, key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"Config field {key!r} must be None or an integer >= 0")
        for key in ("llm_request_timeout_sec", "vlm_request_timeout_sec"):
            value = getattr(self, key)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value <= 0
            ):
                raise ValueError(f"Config field {key!r} must be None or a number > 0")
        if self.executor_type not in {"code", "react", "single_pass"}:
            raise ValueError(
                "Config field 'executor_type' must be one of 'code', 'react', or 'single_pass'"
            )
        if self.cot_frame_preparation not in {"legacy", "spatial-agent"}:
            raise ValueError(
                "Config field 'cot_frame_preparation' must be 'legacy' or "
                "'spatial-agent'"
            )

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
