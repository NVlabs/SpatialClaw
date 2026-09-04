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

# Conda activation hooks may set environment-scoped variables.  Preserve the
# manager-authenticated lifecycle envelope so a managed allocation cannot be
# relabelled as another pool, slot, or tool specification.
LIFECYCLE_POOL_ID="${SPATIAL_AGENT_SERVICE_POOL_ID:-}"
LIFECYCLE_NAMESPACE="${SPATIAL_AGENT_SERVICE_NAMESPACE:-}"
LIFECYCLE_SLOT_ID="${SPATIAL_AGENT_SERVICE_SLOT_ID:-}"
LIFECYCLE_GENERATION="${SPATIAL_AGENT_SERVICE_GENERATION:-}"
LIFECYCLE_TOOL_SPEC_DIGEST="${SPATIAL_AGENT_TOOL_SPEC_DIGEST:-}"
LIFECYCLE_TOPOLOGY_DIGEST="${SPATIAL_AGENT_TOPOLOGY_DIGEST:-}"
readonly LIFECYCLE_POOL_ID LIFECYCLE_NAMESPACE LIFECYCLE_SLOT_ID
readonly LIFECYCLE_GENERATION LIFECYCLE_TOOL_SPEC_DIGEST
readonly LIFECYCLE_TOPOLOGY_DIGEST

# Step 1: Activate spatialagent conda env (has all GPU tool dependencies: sam3, iopath, etc.)
CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent
echo "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "Python: $(which python)"

# Conda activation hooks may set environment-scoped variables after the sbatch
# wrapper's exports. Lifecycle-managed services must enforce their declared
# effective policy at the final launch boundary.
if [ -n "$LIFECYCLE_POOL_ID" ]; then
    export SPATIAL_AGENT_SERVICE_POOL_ID="$LIFECYCLE_POOL_ID"
    export SPATIAL_AGENT_SERVICE_NAMESPACE="$LIFECYCLE_NAMESPACE"
    export SPATIAL_AGENT_SERVICE_SLOT_ID="$LIFECYCLE_SLOT_ID"
    export SPATIAL_AGENT_SERVICE_GENERATION="$LIFECYCLE_GENERATION"
    export SPATIAL_AGENT_TOOL_SPEC_DIGEST="$LIFECYCLE_TOOL_SPEC_DIGEST"
    export SPATIAL_AGENT_TOPOLOGY_DIGEST="$LIFECYCLE_TOPOLOGY_DIGEST"
    export SPATIAL_AGENT_ENABLE_SYNTHETIC_KEEPALIVE=0
else
    unset SPATIAL_AGENT_SERVICE_POOL_ID SPATIAL_AGENT_SERVICE_NAMESPACE
    unset SPATIAL_AGENT_SERVICE_SLOT_ID SPATIAL_AGENT_SERVICE_GENERATION
    unset SPATIAL_AGENT_TOOL_SPEC_DIGEST SPATIAL_AGENT_TOPOLOGY_DIGEST
fi

# CUDA forward-compat: torch cu130 in the spatialagent env needs a driver newer
# than the cluster nodes provide. Reuse the cuda-compat libs from the
# spatialclaw-cuda env (same mechanism as vllm_manager/run_vllm.sh).
CUDA_PREFIX="$CONDA_BASE/envs/spatialclaw-cuda"
if [ -d "$CUDA_PREFIX/cuda-compat" ]; then
    export LD_LIBRARY_PATH="$CUDA_PREFIX/cuda-compat:${LD_LIBRARY_PATH:-}"
    echo "CUDA forward-compat enabled: $CUDA_PREFIX/cuda-compat"
fi

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
