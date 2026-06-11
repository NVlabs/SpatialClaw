# Running Experiments

Configure SLURM (optional), pre-download model weights, then launch a run.

> ⬅ [Main README](../README.md) &nbsp;·&nbsp; Prev: [« Installation](installation.md) &nbsp;·&nbsp; Next: [Monitoring & logs »](monitoring.md)

All commands run from the project root (`SpatialClaw/`) with `conda activate spatialagent`.

---

## SLURM Account, Partition, and Memory

> Skip this section if you're running on a single machine without SLURM.

The shipped configs reference one specific cluster's partitions and accounts. Before submitting jobs, **edit these three files** to match your cluster:

| File | Fields to update |
|------|------------------|
| `spatial_agent/launch_managers/vllm_manager/models.json` | top-level `accounts`, and `partition` field of each model entry |
| `spatial_agent/launch_managers/agent_manager/config.json` | `accounts`, `default_slurm.partition` (agents themselves are CPU-only) |
| `spatial_agent/launch_managers/gpu_server_manager/config.json` | `accounts`, `default_slurm.partition` |

Check what your cluster offers:

```bash
sinfo -h -o "%P %f %G"          # available partitions and GPU types
sacctmgr show -P -n assoc where user=$USER format=Account | sort -u
```

**`--mem-per-gpu` hard cap.** vLLM's sbatch template requests `--mem-per-gpu=240G` and the GPU tool server requests `228G`. Some clusters refuse to allocate more than ~234 GB/GPU and abort submission. If you see:

```
sbatch: error: You requested 245760M RAM, but only 1 GPUs. For 1 GPUs, please only request MAX 234842M RAM
```

lower the `--mem-per-gpu` line in `spatial_agent/launch_managers/vllm_manager/server_chain.py` and `gpu_server_manager/server_chain.py` to fit your cluster's cap.

---

## Pre-download Model Weights

> ⚠️ **Pre-download is mandatory, not optional.** The vLLM SLURM script (`spatial_agent/launch_managers/vllm_manager/run_vllm.sh`) sets `HF_HUB_OFFLINE=1` so compute jobs cannot reach huggingface.co, and most HPC compute nodes have no outbound internet anyway. Anything you don't pre-fetch into the HF cache here will crash the SLURM job with `LocalEntryNotFoundError`.

All commands run from the login node with `conda activate spatialagent`.

### HuggingFace login

```bash
huggingface-cli login
```

### SAM3.1 weights (gated)

1. Accept the license at <https://huggingface.co/facebook/sam3.1>.
2. Download:
   ```bash
   mkdir -p tools/third_party/sam3/weights
   huggingface-cli download facebook/sam3.1 \
     --local-dir tools/third_party/sam3/weights \
     --include "*.pt" "*.txt.gz"
   ln -sf ../sam3/assets/bpe_simple_vocab_16e6.txt.gz \
       tools/third_party/sam3/weights/bpe_simple_vocab_16e6.txt.gz
   ```

### VLM backbones

The six paper backbones (Hopper / H100+ required for FP8):
```bash
huggingface-cli download Qwen/Qwen3.5-397B-A17B-FP8
huggingface-cli download Qwen/Qwen3.5-122B-A10B-FP8
huggingface-cli download Qwen/Qwen3.6-35B-A3B-FP8
huggingface-cli download Qwen/Qwen3.6-27B
huggingface-cli download google/gemma-4-26B-A4B-it
huggingface-cli download prithivMLmods/gemma-4-31B-it-FP8
```

On A100 / L40S, swap the FP8 entries for AWQ / GPTQ variants from `vllm_manager/models.json`:
```bash
huggingface-cli download cyankiwi/gemma-4-31B-it-AWQ-4bit       # for Gemma-4-31B-IT-AWQ
huggingface-cli download Qwen/Qwen3.5-397B-A17B-GPTQ-Int4       # for Qwen3.5-397B GPTQ
```

### Reconstruction backbones

The GPU tool server's `Reconstruct` ships with three backends — download only the one(s) you'll use:

```bash
# Depth-Anything-3 (default — DA3NESTED, requires the -1.1 suffix)
huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE-1.1

# Pi3 (alternative — fast, monocular + multi-view)
huggingface-cli download yyfz233/Pi3X

# MapAnything (multi-view, optional). Also requires the DINOv2-giant torch-hub
# cache; place it under tools/third_party/torch_hub/.
huggingface-cli download facebook/map-anything
```

Pick the backend at launch time via `--reconstruct_backend {pi3,da3,mapanything}` on the GPU server (see below). If you deviate from the default (`da3`), also set the agent-side `reconstruct_backend` (e.g. `SPATIAL_AGENT_RECONSTRUCT_BACKEND=pi3`) — the agent uses it to pick the input-image resizing rules matching the server's backend.

---

## Launching a Run

The three services are launched by **three independent managers** — start them in order: vLLM → GPU tool server → agent.

### Quickstart via launch managers (recommended)

```bash
# Terminal 1 — start a vLLM server (interactive menu)
python -m spatial_agent.launch_managers.vllm_manager
#   [1] Dashboard   [2] Start Server   [3] Stop Server   [q] Quit
# Pick [2], then choose the model, account, partition, and confirm.

# Terminal 2 — start the GPU tool server
python -m spatial_agent.launch_managers.gpu_server_manager
#   [1] Dashboard   [2] Start GPU Server(s)   [3] Stop GPU Server(s)   [q] Quit
# Pick [2], choose the number of GPUs and the Reconstruct backend (pi3 / da3 / mapanything).

# Terminal 3 — start the agent run
python -m spatial_agent.launch_managers.agent_manager
#   [1] Dashboard   [2] Start Agent Experiment(s)   [3] Start CoT Experiment(s)   [4] Stop   [q] Quit
# Pick [2], then choose benchmark(s), model, concurrency, and confirm.
```

Each run is checkpointed and auto-resumes when a 4-hour SLURM job rolls over.

> The GPU tool server is only required when the dataset config's `tools_to_use` is non-empty (i.e. uses `Reconstruct` / `SAM3`). For pure CoT runs, skip Terminal 2.

### Direct CLI (no SLURM)

Single-machine flow against either a self-hosted vLLM or a hosted endpoint:

```bash
# 1) (Optional) Start a local GPU tool server — only if your dataset config uses tools.
python -m spatial_agent.entrypoints.launch_gpu_server \
    --num_gpus 1 --reconstruct_backend da3

# 2) Run an experiment.
python -m spatial_agent.entrypoints.run \
    --dataset spatial_agent/config/dataset/erqa.json \
    --model   spatial_agent/config/model/gemini-3-pro.json \
    --concurrency 4
```

For a vLLM-served model, launch vLLM separately and ensure your model config has `"llm_base_url": "vllm"` and `"llm_model"` set to the **`served_name`** (e.g. `"gemma-4-31b"`, not the HF path). See [Configuration](configuration.md) for the config schema.

### Reproducing paper tables

* **Table 1 (main results)** — run each benchmark in `spatial_agent/config/dataset/` with the corresponding vLLM-served model in `spatial_agent/config/model/`. The same hyperparameters / system prompt / tool set are used everywhere; no per-benchmark overrides are required.
* **Table 2 (action-interface comparison)** — pass `--executor_type {code,react,single_pass}` (default `code` is SpatialClaw; `react` is the structured tool-call interface; `single_pass` is the one-shot code baseline).

### CoT (no-tool) baseline

```bash
python -m spatial_agent.launch_managers.agent_manager      # pick [3] Start CoT Experiment(s)
```

Or directly:
```bash
python -m spatial_agent.entrypoints.cot_baseline \
    --dataset spatial_agent/config/dataset/erqa.json \
    --model   spatial_agent/config/model/qwen3.5-397b-a17b.json
```

---

> Next: [Monitoring & logs »](monitoring.md) &nbsp;·&nbsp; Hitting errors? See [Troubleshooting](troubleshooting.md)
