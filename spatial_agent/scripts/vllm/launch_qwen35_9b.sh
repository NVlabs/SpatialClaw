#!/bin/bash
# User-facing launch script for SLURM vLLM server
# This script contains all configurable parameters

#=============================================================================
# vLLM Model Configuration
#=============================================================================
MODEL="Qwen/Qwen3.5-9B"
SERVED_NAME="qwen3.5-9b"
MAX_MODEL_LEN=65536  # 128K context
MAX_NUM_SEQS=64   # Concurrency=1 workload: minimize graph-capture pressure
TP_SIZE=8

#=============================================================================
# SLURM Configuration
#=============================================================================
JOB_NAME="vllm-qwen35-9b"
ACCOUNT="nvr_lpr_nvgptvision"
PARTITION="grizzly,polar,polar3,polar4"
GPUS=8
TIME_LIMIT="4:00:00"
RESTART_BEFORE_MIN=20  # Minutes before timeout to start next server (for overlap)

#=============================================================================
# Advanced Configuration (optional)
#=============================================================================
OUTPUT_DIR=""  # Leave empty for default: logs/slurm_vllm

#=============================================================================
# Help message
#=============================================================================
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    cat << EOF
SLURM vLLM Server Launcher

This script launches a continuous chain of 4-hour SLURM jobs running vLLM server
with automatic restarts and 20-minute overlaps for zero downtime.

USAGE:
    bash scripts/vllm/slurm/launch_qwen35_397b.sh

CONFIGURATION:
    Edit this file to modify:
    - Model settings (MODEL, SERVED_NAME, MAX_MODEL_LEN, MAX_NUM_SEQS)
    - SLURM settings (ACCOUNT, PARTITION, GPUS, TIME_LIMIT)
    - Restart timing (RESTART_BEFORE_MIN)

MONITORING:
    - Job status: squeue -u \$USER
    - Logs: tail -f logs/slurm_vllm/vllm-qwen35-397b_<job_id>.out
    - Server registry: cat logs/serve.json

STOP:
    Press Ctrl+C in this terminal (will cancel all jobs)

See scripts/vllm/slurm/README.md for detailed documentation.
EOF
    exit 0
fi

#=============================================================================
# Launch the manager
#=============================================================================
echo "========================================"
echo "Launching SLURM vLLM Server"
echo "========================================"
echo "Model:          $MODEL"
echo "Served name:    $SERVED_NAME"
echo "Max model len:  $MAX_MODEL_LEN"
echo "Max num seqs:   $MAX_NUM_SEQS"
echo "GPUs:           $GPUS"
echo "Time limit:     $TIME_LIMIT"
echo "Restart before: ${RESTART_BEFORE_MIN}m"
echo "========================================"
echo ""
echo "This will run continuous 4-hour SLURM jobs with"
echo "${RESTART_BEFORE_MIN}-minute overlaps for zero downtime."
echo "Press Ctrl+C to stop the chain"
echo "========================================"
echo ""

# Get the scripts directory
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Build arguments for manager
ARGS=(
    --job-name "$JOB_NAME"
    --account "$ACCOUNT"
    --partition "$PARTITION"
    --gpus "$GPUS"
    --time "$TIME_LIMIT"
    --restart-before "$RESTART_BEFORE_MIN"
    --vllm-script "scripts/vllm/run_uv.sh"
    --model "$MODEL"
    --served-name "$SERVED_NAME"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --tp-size "$TP_SIZE"
)

# Add optional output directory
if [ -n "$OUTPUT_DIR" ]; then
    ARGS+=(--output-dir "$OUTPUT_DIR")
fi

# Activate .venv environment and run manager
source .venv/bin/activate

python "$SCRIPTS_DIR/scripts/vllm/manager.py" "${ARGS[@]}"
