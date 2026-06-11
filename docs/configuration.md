# Configuration & Benchmarks

Model/dataset JSON configs, environment-variable overrides, and the list of supported benchmarks.

> ⬅ [Main README](../README.md) &nbsp;·&nbsp; Prev: [« Monitoring & logs](monitoring.md) &nbsp;·&nbsp; Next: [Architecture »](architecture.md)

---

## Configuration Reference

Loading priority (highest first): **CLI args > Model/Dataset JSON > Environment variables > Defaults.**

### Model configs (`spatial_agent/config/model/*.json`)

```json
{
    "llm_model":    "qwen3.5-397b-a17b",
    "llm_base_url": "vllm",
    "llm_api_key":  "bearer",
    "roles": {
        "main":          { "max_tokens": 131072, "temperature": 0.6, "enable_thinking": true },
        "planning":      { "max_tokens": 131072, "temperature": 0.6, "enable_thinking": true },
        "vlm":           { "max_tokens": 32768,  "temperature": 0.6, "enable_thinking": true },
        "vlm_grounding": { "max_tokens": 16384,  "temperature": 0.6, "enable_thinking": true },
        "general":       { "max_tokens": 131072, "temperature": 0.6, "enable_thinking": false }
    }
}
```

* `llm_base_url: "vllm"` — auto-discover endpoints from `spatial_agent/logs/serve.json` (load-balanced, sticky-session for prefix cache hits). With this base URL, **`llm_model` must be the `served_name`** from `vllm_manager/models.json` (e.g. `qwen3.5-397b-a17b`), **not** the HuggingFace path.
* `llm_base_url: "https://..."` — call any OpenAI-compatible HTTP endpoint. Here `llm_model` is whatever model identifier that endpoint expects (e.g. `gcp/google/gemini-3-pro`).
* `llm_api_key` accepts `${ENV_VAR}` and `${ENV_VAR:-default}` substitution; secrets stay in `.env` (see [Installation → API Keys](installation.md#api-keys--env)).
* `roles.*` — per-role hyperparameters (`main`, `planning`, `vlm`, `vlm_grounding`, `general`, `reflection`).

### Dataset configs (`spatial_agent/config/dataset/*.json`)

Per-benchmark settings: data paths, question types, frame sampling, tool list, agent-loop caps.

### Common env-var overrides

Any `SpatialAgentConfig` field can be overridden via `SPATIAL_AGENT_<FIELD>=...`. Examples:

```bash
export SPATIAL_AGENT_CONCURRENCY=16
export SPATIAL_AGENT_MAX_STEPS=30
export SPATIAL_AGENT_RECONSTRUCT_BACKEND=da3   # or pi3
```

---

## Supported Benchmarks

All 20 paper benchmarks ship as ready-to-run dataset configs under `spatial_agent/config/dataset/`:

| Category                          | Benchmarks                                                                   |
|-----------------------------------|------------------------------------------------------------------------------|
| Single-image spatial reasoning    | ERQA, Omni3D, OmniSpatial, SPBench                                           |
| Multi-view spatial reasoning      | MindCube, MMSI, SPAR-Bench                                                   |
| General spatial reasoning         | BLINK, SpatialTree, ViewSpatial                                             |
| Video spatial & 4D reasoning      | MMSI-Video, OSI-Bench, PAI-Bench, VSI-Bench-U, VSTI-Bench, DSI-Bench         |
| General video understanding       | CV-Bench, PerceptComp, Video-MME, Video-MME-v2                              |

See `spatial_agent/evals/` for the corresponding loaders and scoring scripts.

---

> Next: [Architecture »](architecture.md)
