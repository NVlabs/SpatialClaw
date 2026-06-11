# Installation

Prerequisites, environment setup, and API keys for SpatialClaw.

> ⬅ [Main README](../README.md) &nbsp;·&nbsp; Next: [Running experiments »](running.md)

---

## Prerequisites

* Linux with NVIDIA GPUs
  * **Hopper (H100) or newer** is required to serve the **FP8** model variants used in the paper (e.g. `prithivMLmods/gemma-4-31B-it-FP8`, `Qwen/Qwen3.5-397B-A17B-FP8`). On A100 / L40S, use the **AWQ** entries listed in `spatial_agent/launch_managers/vllm_manager/models.json` (e.g. `Gemma-4-31B-IT-AWQ`, `Qwen3.5-397B-A17B-GPTQ-Int4`) — they map to the same `served_name` so model configs need no changes.
* CUDA 12.x compatible driver
* `conda` (Miniconda or Anaconda) — verify with `conda --version`
* HuggingFace account with access to gated models (for SAM3.1 weights)
* SLURM cluster *(optional — only required to use the chain-job managers)*
* Network access to huggingface.co for initial model downloads

---

## Option A: Automated (recommended)

```bash
git clone --recursive https://github.com/NVlabs/SpatialClaw.git
cd SpatialClaw
bash spatial_agent/scripts/setup.sh
```

`--recursive` pulls the third-party model code (SAM3, Pi3, Depth-Anything-3, map-anything) at pinned commits — these are tracked as **git submodules** under `tools/third_party/`. If you cloned without `--recursive`, run:

```bash
git submodule update --init --recursive
```

Setup takes ~15–30 minutes on first run. Individual steps:

```bash
bash spatial_agent/scripts/setup.sh --agent    # agent env only (step 1 below)
bash spatial_agent/scripts/setup.sh --cuda     # CUDA env only  (step 2 below)
bash spatial_agent/scripts/setup.sh --vllm     # vLLM venv only (step 3 below)
```

---

## Option B: Manual

### 1. Agent conda environment

Used by the agent runtime, the SLURM managers, and the GPU server.

```bash
conda create -n spatialagent python=3.11 -y
conda activate spatialagent

pip install -r spatial_agent/requirements/requirements-agent.txt
conda install -y ffmpeg
pip install uv         # needed to build the vLLM venv in step 3
```

Verify:
```bash
python -c "import langgraph, openai, jupyter_client, fastapi; print('OK')"
ffmpeg -version | head -1
```

### 2. CUDA conda environment for vLLM

vLLM borrows CUDA shared libraries from a separate env to avoid version conflicts:

```bash
conda create -n spatialclaw-cuda -y
conda install -n spatialclaw-cuda -c nvidia cuda-toolkit=12.8 -y
```

### 3. vLLM virtual environment

Pinned to nightly + CUDA 12.9 for Gemma4 support:

```bash
conda activate spatialagent
uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

uv pip install transformers==5.5.0 pynvml pandas
deactivate
```

Then build **DeepGEMM** — required by the FP8 model variants used in the paper:

```bash
CONDA_BASE=$(conda info --base)
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /tmp/DeepGEMM
cd /tmp/DeepGEMM
export CUDA_HOME="$CONDA_BASE/envs/spatialclaw-cuda"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib"
/path/to/SpatialClaw/.venv/bin/python setup.py bdist_wheel
/path/to/SpatialClaw/.venv/bin/pip install dist/*.whl
cd - && rm -rf /tmp/DeepGEMM
```

> Skip this step only if you'll use BF16 / AWQ models exclusively — FP8 inference (`prithivMLmods/gemma-4-31B-it-FP8`, `Qwen/Qwen3.5-*-FP8`) will not work without `deep_gemm`.

### 4. Third-party model repos

The four third-party repos are tracked as git submodules. `git clone --recursive` (Option A) populates them automatically. To fetch them after a non-recursive clone:

```bash
git submodule update --init --recursive
```

This pulls the pinned commits of: SAM3 (segmentation), Pi3 (reconstruction), Depth-Anything-3 (default reconstruction backend), and map-anything (alt multi-view backend).

### 5. Install SAM3 as editable

```bash
conda activate spatialagent
pip install -e "tools/third_party/sam3[dev]"
```

---

## API Keys & `.env`

SpatialClaw can run against either **self-hosted vLLM** (no API key needed — pass the literal string `bearer`) or **OpenAI-compatible cloud endpoints**. Secrets are kept out of version control:

1. Copy the template:
   ```bash
   cp .env.example .env
   ```
2. Fill in the keys you need (only `NVIDIA_API_KEY` / `OPENAI_API_KEY` are common):
   ```dotenv
   NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. Reference them from any model config JSON via `${VAR}` expansion:
   ```json
   {
     "llm_model": "gcp/google/gemini-3-pro",
     "llm_base_url": "https://inference-api.nvidia.com/v1",
     "llm_api_key": "${NVIDIA_API_KEY}"
   }
   ```
   `llm_base_url` must be the OpenAI-compatible API **root** — the client
   appends `/chat/completions` to it, so for most gateways the URL ends in
   `/v1`.

The `.env` file is loaded automatically by `spatial_agent.config.get_config()`. Shell environment variables (e.g. `export NVIDIA_API_KEY=...`) always win over `.env`, so production deploys can inject secrets through any normal mechanism.

`${VAR:-default}` syntax is also supported, e.g. `"${OPENAI_API_KEY:-bearer}"`.

---

> Next: [Running experiments »](running.md) &nbsp;·&nbsp; See also [Configuration](configuration.md)
