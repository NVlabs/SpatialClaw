#!/bin/bash
# User-facing launch script for SLURM spatial agent evaluation
# This script contains all configurable parameters

export RAY_SERVE_QUEUE_LENGTH_RESPONSE_DEADLINE_S=10.0

#=============================================================================
# Benchmark Configuration
#=============================================================================
BENCHMARK="${1}"  # Required: benchmark name (e.g. erqa, mmsi, mmsivideo)
EXPERIMENT_NAME="${2:-}"  # Optional: appended to work_dir folder name
MODEL_NAME="${3:-qwen3.5-397b-a17b}"  # Optional: model config filename (no .json)
CONCURRENCY="${4:-8}"  # Optional: number of parallel workers (default: 8)
SUBSAMPLE="${5:-}"  # Optional: deterministic random subsample size (e.g. 200)
MODEL_CONFIG="spatial_agent/config/model/${MODEL_NAME}.json"
DATASET_CONFIG="spatial_agent/config/dataset/${BENCHMARK}.json"

#=============================================================================
# SLURM Configuration
#=============================================================================
JOB_NAME="spatial-agent"
ACCOUNT="nvr_taiwan_rvos"
# PARTITION="grizzly,polar,polar3,polar4"
PARTITION="interactive"
GPUS=8
TIME_LIMIT="4:00:00"

#=============================================================================
# Advanced Configuration (optional)
#=============================================================================
OUTPUT_DIR=""  # Leave empty for default: spatial_agent/logs/slurm_agent

#=============================================================================
# Help message
#=============================================================================
if [ -z "$BENCHMARK" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    cat << EOF
SLURM Spatial Agent Evaluation Launcher

This script launches a continuous chain of 4-hour SLURM jobs for spatial agent
evaluation with automatic resume from checkpoints. Each job completes before
the next starts.

USAGE:
    bash spatial_agent/scripts/agent/slurm/launch.sh <benchmark> [experiment_name] [model_name] [concurrency] [subsample]

EXAMPLES:
    bash spatial_agent/scripts/agent/slurm/launch.sh erqa                                       # Run ERQA, default model, 8 workers
    bash spatial_agent/scripts/agent/slurm/launch.sh erqa run01                                 # Run ERQA with experiment name
    bash spatial_agent/scripts/agent/slurm/launch.sh erqa run01 qwen3.5-397b-a17b               # Use qwen3.5-397b-a17b model
    bash spatial_agent/scripts/agent/slurm/launch.sh erqa "" qwen3.5-397b-a17b 16               # 16 workers, no experiment name
    bash spatial_agent/scripts/agent/slurm/launch.sh erqa run01 "" "" 200                       # 200 random samples (deterministic)

AVAILABLE BENCHMARKS:
    See spatial_agent/config/dataset/ for the full list of paper benchmarks.

CONFIGURATION:
    Edit this file to modify:
    - Model config (MODEL_CONFIG) — LLM connection + hyperparameters
    - Dataset config (DATASET_CONFIG) — benchmark-specific settings
    - SLURM settings (ACCOUNT, PARTITION, GPUS, TIME_LIMIT)

MONITORING:
    - Job status: squeue -u \$USER
    - Logs: tail -f spatial_agent/logs/slurm_agent/spatial-agent-<benchmark>_<job_id>.out
    - Results: spatial_agent/work_dir/

STOP:
    Press Ctrl+C in this terminal (will cancel all jobs)
EOF
    exit 0
fi

#=============================================================================
# Launch the manager
#=============================================================================
echo "========================================"
echo "Launching SLURM Spatial Agent Evaluation"
echo "========================================"
echo "Benchmark:      $BENCHMARK"
echo "Experiment:     ${EXPERIMENT_NAME:-<default>}"
echo "Concurrency:    $CONCURRENCY"
echo "Subsample:      ${SUBSAMPLE:-<all>}"
echo "Model config:   $MODEL_CONFIG"
echo "Dataset config: $DATASET_CONFIG"
echo "GPUs:           $GPUS"
echo "Time limit:     $TIME_LIMIT"
echo "========================================"
echo ""
echo "This will run continuous 4-hour SLURM jobs"
echo "Each job will resume from the previous checkpoint"
echo "Press Ctrl+C to stop the chain"
echo "========================================"
echo ""

# Get the spatial_agent directory
SPATIAL_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Build arguments for manager
ARGS=(
    --benchmark "$BENCHMARK"
    --concurrency "$CONCURRENCY"
    --job-name "${JOB_NAME}-${BENCHMARK}"
    --account "$ACCOUNT"
    --partition "$PARTITION"
    --gpus "$GPUS"
    --time "$TIME_LIMIT"
    --agent-script "spatial_agent/scripts/agent/slurm/run.sh"
    --model-config "$MODEL_CONFIG"
    --dataset-config "$DATASET_CONFIG"
)

# Add experiment name as work_dir suffix
if [ -n "$EXPERIMENT_NAME" ]; then
    ARGS+=(--experiment-name "$EXPERIMENT_NAME")
fi

# Add subsample size
if [ -n "$SUBSAMPLE" ]; then
    ARGS+=(--subsample "$SUBSAMPLE")
fi

# Add optional output directory
if [ -n "$OUTPUT_DIR" ]; then
    ARGS+=(--output-dir "$OUTPUT_DIR")
fi

# Activate spatialagent environment and run manager
CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent

python "$SPATIAL_AGENT_DIR/scripts/agent/slurm/manager.py" "${ARGS[@]}"
