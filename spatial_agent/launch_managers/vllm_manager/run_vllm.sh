#!/bin/bash
# vLLM SLURM Execution Script (runs INSIDE each SLURM job)
# Sets up CUDA/PyTorch paths and calls entrypoints/launch_vllm.py

set -e

# Parse arguments
MODEL="$1"
SERVED_NAME="$2"
MAX_MODEL_LEN="$3"
MAX_NUM_SEQS="$4"
NUM_GPUS="$5"
TP_SIZE="${6:-$NUM_GPUS}"
KV_CACHE_DTYPE="${7:-auto}"
QUANTIZATION="${8:-none}"

if [ -z "$MODEL" ] || [ -z "$SERVED_NAME" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 <model> <served_name> <max_model_len> <max_num_seqs> <num_gpus> [tp_size] [kv_cache_dtype] [quantization]"
    exit 1
fi

echo "========================================"
echo "vLLM Server Startup (vllm_manager)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-N/A}"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "========================================"

# Step 1: Borrow CUDA libs from spatialclaw-cuda conda env
CUDA_CONDA_ENV="spatialclaw-cuda"
CONDA_BASE=$(conda info --base 2>/dev/null)
CUDA_PREFIX="$CONDA_BASE/envs/$CUDA_CONDA_ENV"
export CUDA_HOME="$CUDA_PREFIX"
export LD_LIBRARY_PATH="$CUDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# CUDA headers in this conda env live under targets/x86_64-linux/include
# (standard CUDA SDK layout), not the flat include/. flashinfer's JIT
# compiler only sees include/ via -isystem and misses cublasLt.h etc.
# CPATH is honored by both gcc and nvcc for implicit header search.
CUDA_TARGETS="$CUDA_PREFIX/targets/x86_64-linux"
if [ -d "$CUDA_TARGETS/include" ]; then
    export CPATH="$CUDA_TARGETS/include:${CPATH:-}"
    export LIBRARY_PATH="$CUDA_TARGETS/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_TARGETS/lib:${LD_LIBRARY_PATH}"
fi
echo "CUDA borrowed from: $CUDA_PREFIX"

# Step 2: Add .venv's bundled NVIDIA libs (nvjitlink, cusparse, etc.) BEFORE
# the conda env's libs so the matching versions are found first.
NVIDIA_LIBS=$(find .venv/lib -path "*/nvidia/*/lib" -type d 2>/dev/null | tr '\n' ':')
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
    echo "NVIDIA .venv libs prepended"
fi

# Step 3: Add PyTorch shared libraries to linker path
TORCH_LIB=$(find .venv/lib -name "libtorch_cuda.so" -print -quit 2>/dev/null | xargs -r dirname)
if [ -z "$TORCH_LIB" ]; then
    echo "ERROR: libtorch_cuda.so not found under .venv"
    exit 1
fi
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
echo "PyTorch lib: $TORCH_LIB"

# Step 3b: CUDA forward-compat. torch cu129 / vLLM Marlin kernels emit PTX
# that requires a driver >= 575, but cluster nodes run 550. The cuda-compat
# package ships a userspace libcuda.so.1 (575.57.08) that handles the newer
# PTX while leaving the kernel driver untouched. Must be prepended so its
# libcuda.so.1 wins over /usr/lib/x86_64-linux-gnu/libcuda.so.1.
if [ -d "$CUDA_PREFIX/cuda-compat" ]; then
    export LD_LIBRARY_PATH="$CUDA_PREFIX/cuda-compat:$LD_LIBRARY_PATH"
    echo "CUDA forward-compat enabled: $CUDA_PREFIX/cuda-compat"
fi

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Step 4: Environment flags
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Marlin Int4 kernels: use atomic-add accumulation instead of a scratch
# reduction buffer. H100 has native bf16/fp16 atomics, so this is a perf
# win on MoE Int4 (less memory traffic for per-expert temps). No-op for
# BF16/FP8 paths, which don't enter Marlin kernels.
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
# Put .venv/bin on PATH so torch cpp_extension / flashinfer JIT can find
# `ninja` when they shell out by name. We call .venv/bin/python directly
# (no `source activate`), so PATH would otherwise not include venv scripts.
export PATH="$PWD/.venv/bin:$PATH"
# Default to offline HF mode: models are pre-downloaded to the user cache,
# and HF's API rate limit (429) can kill startup when vLLM lists repo files.
# Override with HF_HUB_OFFLINE=0 if you need to fetch a new model.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# HF_TOKEN etc. live in .env.local (gitignored, chmod 600) — raises our HF API
# rate-limit quota so tokenizer init doesn't 429 during restart bursts.
if [ -f .env.local ]; then
    set -a
    . ./.env.local
    set +a
fi

echo "Model: $MODEL"
echo "Served name: $SERVED_NAME"
echo "Max model length: $MAX_MODEL_LEN"
echo "Max num sequences: $MAX_NUM_SEQS"
echo "Number of GPUs: $NUM_GPUS"
echo "Tensor parallel size: $TP_SIZE"
echo "KV cache dtype: $KV_CACHE_DTYPE"
echo "Quantization: $QUANTIZATION"
echo "========================================"

# Start vLLM server
echo "Starting vLLM server..."
stdbuf -oL -eL .venv/bin/python -u -m spatial_agent.entrypoints.launch_vllm \
    --model "$MODEL" \
    --served_model_name "$SERVED_NAME" \
    --tp "$TP_SIZE" \
    --kv_cache_dtype "$KV_CACHE_DTYPE" \
    --quantization "$QUANTIZATION" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_num_seqs "$MAX_NUM_SEQS"

echo "========================================"
echo "vLLM server stopped"
echo "End time: $(date)"
echo "========================================"
