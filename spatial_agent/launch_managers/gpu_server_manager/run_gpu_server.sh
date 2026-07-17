#!/bin/bash
# GPU Server SLURM Execution Script (runs INSIDE each SLURM job)
# Activates the spatialagent conda env and launches GPU tools directly.

set -e

NUM_GPUS="${1:-1}"
RECONSTRUCT_BACKEND="${2:-da3}"

echo "========================================"
echo "GPU Server Startup"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURM_NODELIST:-N/A}"
echo "GPUs: $NUM_GPUS"
echo "Backend: $RECONSTRUCT_BACKEND"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "========================================"

ulimit -c 0
export PYTHONUNBUFFERED=1

# Step 1: Activate spatialagent conda env (has all GPU tool dependencies: sam3, iopath, etc.)
CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent
echo "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python: $(which python)"

# Add third-party paths so model imports can find sam3, pi3, da3, map-anything, etc.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GCA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
THIRD_PARTY="${GCA_ROOT}/tools/third_party"
export PYTHONPATH="${THIRD_PARTY}/sam3:${THIRD_PARTY}/Pi3:${THIRD_PARTY}/Depth-Anything-3/src:${THIRD_PARTY}/map-anything:${PYTHONPATH:-}"

# Step 2: Launch GPU tools with HTTP API (models loaded directly, no Ray)
echo "Loading GPU models..."
stdbuf -oL -eL python -u -m spatial_agent.entrypoints.launch_gpu_server \
    --num_gpus "$NUM_GPUS" \
    --reconstruct_backend "$RECONSTRUCT_BACKEND" \
    --host 0.0.0.0

echo "========================================"
echo "GPU server stopped"
echo "End time: $(date)"
echo "========================================"
