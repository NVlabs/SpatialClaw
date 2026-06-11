#!/bin/bash
# User-facing launch script for SLURM CoT baseline evaluation
# No GPUs needed — pure VLM inference via vLLM server

#=============================================================================
# Configuration
#=============================================================================
# MODEL_CONFIG="spatial_agent/config/model/qwen3.5-9b.json"
# MODEL_CONFIG="spatial_agent/config/model/qwen3.5-122b-a10b.json"
# MODEL_CONFIG="spatial_agent/config/model/qwen3.5-397b-a17b.json"

#=============================================================================
# Benchmark Configuration
#=============================================================================
BENCHMARK="${1}"  # Required: benchmark name (e.g. erqa, mmsivideo)
MODEL_NAME="${2:-qwen3.5-397b-a17b}"
CONCURRENCY="${3:-32}"  # Optional: number of parallel workers (default: 32)
MAX_FRAMES="${4:-32}"  # Optional: max frames per sample (default: 32)
SUBSAMPLE="${5:-}"  # Optional: deterministic random subsample size (e.g. 200)

MODEL_CONFIG="spatial_agent/config/model/${MODEL_NAME}.json"

#=============================================================================
# SLURM Configuration
#=============================================================================
JOB_NAME="cot-baseline"
ACCOUNT="nvr_lpr_nvgptvision"
PARTITION="cpu_short,cpu"
TIME_LIMIT="4:00:00"

#=============================================================================
# Help message
#=============================================================================
if [ -z "$BENCHMARK" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    cat << EOF
SLURM CoT Baseline Evaluation Launcher

This script launches a continuous chain of 4-hour SLURM jobs for CoT baseline
evaluation with automatic resume. No GPUs are requested — all VLM inference
goes through the vLLM server.

USAGE:
    bash spatial_agent/scripts/cot_baseline/launch.sh <benchmark> [concurrency] [max_frames] [subsample]

EXAMPLES:
    bash spatial_agent/scripts/cot_baseline/launch.sh erqa              # ERQA, 32 workers, 32 frames
    bash spatial_agent/scripts/cot_baseline/launch.sh erqa 64 16        # ERQA, 64 workers, 16 frames
    bash spatial_agent/scripts/cot_baseline/launch.sh mmsivideo 32      # MMSI-Video, 32 workers
    bash spatial_agent/scripts/cot_baseline/launch.sh erqa 32 32 200    # 200 random samples (deterministic)

AVAILABLE BENCHMARKS:
    Any benchmark with a config under spatial_agent/config/dataset/
    (e.g. erqa, mmsivideo, vsibench_unbiased — see docs/configuration.md)

CONFIGURATION:
    Edit this file to modify:
    - Model config (MODEL_CONFIG) — LLM connection + hyperparameters
    - SLURM settings (ACCOUNT, PARTITION, TIME_LIMIT)

MONITORING:
    - Job status: squeue -u \$USER
    - Logs: tail -f spatial_agent/logs/slurm_cot/cot-baseline-<benchmark>_<job_id>.out
    - Results: spatial_agent/work_dir/cot_<benchmark>_*/

STOP:
    Press Ctrl+C in this terminal (will cancel all jobs)
EOF
    exit 0
fi

#=============================================================================
# Launch the manager
#=============================================================================
echo "========================================"
echo "Launching SLURM CoT Baseline Evaluation"
echo "========================================"
echo "Benchmark:     $BENCHMARK"
echo "Concurrency:   $CONCURRENCY"
echo "Max frames:    $MAX_FRAMES"
echo "Subsample:     ${SUBSAMPLE:-<all>}"
echo "Model config:  $MODEL_CONFIG"
echo "========================================"
echo ""
echo "This will run continuous 4-hour SLURM jobs"
echo "Each job will resume from the previous checkpoint"
echo "Press Ctrl+C to stop the chain"
echo "========================================"
echo ""

# Get the spatial_agent directory
SPATIAL_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Build arguments for manager
ARGS=(
    --benchmark "$BENCHMARK"
    --concurrency "$CONCURRENCY"
    --max-frames "$MAX_FRAMES"
    --job-name "${JOB_NAME}-${BENCHMARK}"
    --account "$ACCOUNT"
    --partition "$PARTITION"
    --gpus 0
    --time "$TIME_LIMIT"
    --run-script "spatial_agent/scripts/cot_baseline/run.sh"
    --model-config "$MODEL_CONFIG"
)

# Add subsample size
if [ -n "$SUBSAMPLE" ]; then
    ARGS+=(--subsample "$SUBSAMPLE")
fi

# Activate spatialagent environment and run manager
CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent

python "$SPATIAL_AGENT_DIR/scripts/cot_baseline/manager.py" "${ARGS[@]}"
