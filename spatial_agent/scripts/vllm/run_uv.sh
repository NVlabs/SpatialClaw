#!/bin/bash
# SLURM vLLM Server Execution Script (uv/venv environment)
# This script runs INSIDE each SLURM job using a uv-managed Python environment
# Configuration is passed as arguments from the manager

set -e  # Exit on error

# Parse arguments
MODEL="$1"
SERVED_NAME="$2"
MAX_MODEL_LEN="$3"
MAX_NUM_SEQS="$4"
NUM_GPUS="$5"
TP_SIZE="${6:-$NUM_GPUS}"

if [ -z "$MODEL" ] || [ -z "$SERVED_NAME" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 <model> <served_name> <max_model_len> <max_num_seqs> <num_gpus> [tp_size]"
    exit 1
fi

echo "========================================"
echo "SLURM vLLM Server Startup (uv/venv)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-N/A}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "========================================"

# ---------------------------------------------------------------------------
# Step 1: Borrow CUDA shared libraries from spatialclaw-cuda conda environment.
# libcudart.so.12, libcublas.so, etc. live in the conda env's lib/ dir.
# ---------------------------------------------------------------------------
CUDA_CONDA_ENV="spatialclaw-cuda"
CONDA_BASE=$(conda info --base 2>/dev/null)
CUDA_PREFIX="$CONDA_BASE/envs/$CUDA_CONDA_ENV"
export CUDA_HOME="$CUDA_PREFIX"
export LD_LIBRARY_PATH="$CUDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
echo "CUDA borrowed from: $CUDA_PREFIX"

# ---------------------------------------------------------------------------
# Step 2: Add PyTorch's own shared libraries (libtorch_cuda.so, libtorch.so,
# etc.) to the linker search path.
#
# We use `find` instead of importing Python — importing torch itself requires
# libtorch_cuda.so in LD_LIBRARY_PATH, so calling `uv run python -c` or any
# Python here creates a circular dependency.
#
# We also can't simply use `uv run` for the server launch, because uv spawns
# Python as a managed subprocess and may not inherit LD_LIBRARY_PATH reliably.
# We call .venv/bin/python directly instead.
# ---------------------------------------------------------------------------
TORCH_LIB=$(find .venv/lib -name "libtorch_cuda.so" -print -quit 2>/dev/null | xargs -r dirname)
if [ -z "$TORCH_LIB" ]; then
    echo "ERROR: libtorch_cuda.so not found under .venv — is torch installed with CUDA support?"
    exit 1
fi
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
echo "PyTorch lib: $TORCH_LIB"

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# ---------------------------------------------------------------------------
# Step 3: Standard CUDA/vLLM environment flags
# ---------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
# Put .venv/bin on PATH so torch cpp_extension JIT can find `ninja` when it
# shells out by name. We call .venv/bin/python directly (no `source activate`),
# so PATH would otherwise not include the venv's scripts.
export PATH="$PWD/.venv/bin:$PATH"
# Default to offline HF mode: models are pre-downloaded to the user cache,
# and HF's API rate limit (429) can kill startup when vLLM tries to list repo
# files. Override with HF_HUB_OFFLINE=0 if you need to fetch a new model.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "Model: $MODEL"
echo "Served name: $SERVED_NAME"
echo "Max model length: $MAX_MODEL_LEN"
echo "Max num sequences: $MAX_NUM_SEQS"
echo "Number of GPUs: $NUM_GPUS"
echo "Tensor parallel size: $TP_SIZE"
echo "========================================"

# Start vLLM server using .venv/bin/python directly — bypasses uv run's
# subprocess layer so LD_LIBRARY_PATH is guaranteed to be inherited.
echo "Starting vLLM server..."
stdbuf -oL -eL .venv/bin/python -u -m spatial_agent.entrypoints.launch_vllm \
    --model "$MODEL" \
    --served_model_name "$SERVED_NAME" \
    --tp "$TP_SIZE" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_num_seqs "$MAX_NUM_SEQS"

# If we get here, the server stopped
echo "========================================"
echo "vLLM server stopped"
echo "End time: $(date)"
echo "========================================"
