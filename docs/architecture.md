# Architecture

How the agentic loop works, the three-service runtime, and the repository layout.

> ⬅ [Main README](../README.md) &nbsp;·&nbsp; Prev: [« Configuration](configuration.md) &nbsp;·&nbsp; Next: [Troubleshooting »](troubleshooting.md)

---

## How It Works

### Agentic loop (Fig. 2 of the paper)

For every sample SpatialClaw runs a five-stage loop on top of a persistent Python kernel:

```
                +-----------------+
   Question --> | I. Planning     |  (separate LLM session, no images,
                +--------+--------+     produces a structured analysis plan)
                         |
                         v
   +--------> +-----------------+
   |          | II. Code Gen    |  (main VLM produces purpose / reasoning /
   |          +--------+--------+     next goal / code in a markdown response)
   |                   |
   |                   v
   |          +-----------------+
   |          | III. Execute    |  (AST safety check, then run cell in the
   |          +--------+--------+     persistent kernel; vars persist across steps)
   |                   |
   |                   v
   |          +-----------------+
   |          | IV. Feedback    |  (stdout, tracebacks, variable summaries,
   |          +--------+--------+     and any images registered via show())
   |                   |
   +-- not done <------+------> ReturnAnswer() ---> V. Answer Submission
```

The kernel exposes six entry points to the agent:

| Entry point         | Purpose                                                                              |
|---------------------|--------------------------------------------------------------------------------------|
| `InputImages`       | Sampled frames / images for the current question.                                    |
| `Metadata`          | FPS, duration, frame indices — used for temporal reasoning on video samples.         |
| `tools.*`           | Perception primitives: `Reconstruct` (DA3 / Pi3), `SAM3`, geometry helpers, drawing. |
| `show(...)`         | Register an image to be embedded into the agent's next observation.                  |
| `vlm.locate(...)` / `vlm.ask_with_thinking(...)` | Isolated VLM session for grounding / commonsense queries.    |
| `ReturnAnswer(...)` | Submit a candidate answer; terminates the loop when the format is valid.             |

### Runtime architecture

The system has three independent services, all managed via SLURM:

```
+------------------+       +------------------+       +------------------+
|   vLLM Server    |       |   GPU Server     |       |   Agent          |
|  (LLM inference) |       | (DA3, SAM3 tools)|       | (Jupyter kernels)|
|  .venv (uv)      |       |  conda env       |       |  conda env       |
|  H100 GPUs       |       |  H100 GPU        |       |  CPU only        |
+--------+---------+       +--------+---------+       +--------+---------+
         |                          |                          |
         |   logs/serve.json        |  logs/gpu_server.json    |
         +--------------------------+--------------------------+
                    Auto-discovery via shared JSON registries
```

* **vLLM Server** — serves the VLM backbone (e.g. Qwen3.5-397B-A17B, Gemma4-31B) via an OpenAI-compatible API.
* **GPU Server** — runs the heavy perception tools (Depth-Anything-3 / Pi3 reconstruction, SAM3 video segmentation) behind a FastAPI HTTP service.
* **Agent** — orchestrates the LangGraph loop and spawns a per-sample Jupyter kernel that executes the agent's code.

Each service runs as a chain of 4-hour SLURM jobs with automatic restart and 20-minute overlap, so long evaluations survive job-time limits with zero downtime.

> **Don't have a SLURM cluster?** Each service is also a normal Python entry point (`spatial_agent.entrypoints.launch_vllm`, `launch_gpu_server`, `run`) and can be started directly on any GPU machine — the SLURM managers are convenience wrappers, not requirements.

---

## Directory Structure

```
SpatialClaw/                               # Project root (must be cwd)
├── .env.example                           # Template for local secrets (copy to .env)
├── spatial_agent/                         # Main Python package
│   ├── workflow.py                        #   LangGraph orchestration (planning → code → execute → feedback)
│   ├── config.py                          #   Config dataclass + .env / ${VAR} expansion
│   ├── state.py                           #   Workflow state shared across nodes
│   ├── config/                            # JSON configs
│   │   ├── dataset/                       #   per-benchmark settings
│   │   ├── model/                         #   LLM backbone + per-role hyperparameters
│   ├── entrypoints/                       # CLI entry points
│   │   ├── run.py                         #   main agent evaluation
│   │   ├── cot_baseline.py                #   no-tool CoT baseline
│   │   ├── launch_vllm.py                 #   vLLM server launcher (runs in .venv)
│   │   └── launch_gpu_server.py           #   GPU tool server launcher
│   ├── launch_managers/                   # Interactive SLURM managers (one per service)
│   │   ├── vllm_manager/                  #   vLLM servers
│   │   ├── gpu_server_manager/            #   GPU tool server chains
│   │   └── agent_manager/                 #   agent + CoT experiment chains
│   ├── nodes/                             # LangGraph node implementations (planner, executor, reflector, …)
│   ├── kernel/                            # Persistent Jupyter kernel + AST safety check
│   ├── kernel_types/                      # Typed wrappers exposed to the agent (InputImages, Metadata, …)
│   ├── llm/                               # LLM client with vLLM auto-discovery and load balancing
│   ├── tools/                             # CPU + GPU tool implementations (SAM3, Reconstruct, geometry, drawing)
│   ├── gpu_models/                        # GPU model classes (Pi3, SAM3, DA3, MapAnything)
│   ├── evals/                             # Benchmark loaders + scoring
│   ├── visualization_server/              # Per-session HTML report viewer
│   ├── gpu_dashboard/                     # SLURM / GPU live-status TUI
│   ├── logging_utils/                     # Session logging + HTML report writers
│   ├── requirements/                      # Pinned dependency lists
│   └── scripts/                           # Setup + helper shell scripts
├── tools/third_party/                     # Third-party model repos (cloned by setup.sh)
│   ├── sam3/                              #   SAM3 (+ weights/)
│   ├── Pi3/                               #   Pi3 reconstruction (alt: --reconstruct_backend pi3)
│   ├── Depth-Anything-3/                  #   Depth Anything 3 (default Reconstruct backend)
│   └── map-anything/                      #   MapAnything (alt: --reconstruct_backend mapanything)
├── data/                                  # Benchmark datasets (downloaded separately)
└── .venv/                                 # vLLM virtual environment (uv)
```

---

> Next: [Troubleshooting »](troubleshooting.md)
