#!/bin/bash
# SLURM Agent Evaluation Execution Script
# This script runs INSIDE each SLURM job
# Configuration is passed as arguments from the manager

set -e  # Exit on error

# Parse arguments
BENCHMARK="$1"
CONCURRENCY="${2:-1}"
MODEL_CONFIG="${3:-}"
DATASET_CONFIG="${4:-}"
EXPERIMENT_NAME="${5:-}"
SUBSAMPLE="${6:-0}"

if [ -z "$BENCHMARK" ]; then
    echo "ERROR: Benchmark not specified"
    echo "Usage: $0 <benchmark> <concurrency> <model_config> <dataset_config>"
    exit 1
fi

# Disable core dumps — the Ray GCS server crashes periodically and writes 20MB
# core files into the working directory, wasting disk space.
ulimit -c 0

# Force unbuffered Python output so logs appear in real-time in SLURM .out files
export PYTHONUNBUFFERED=1

echo "========================================"
echo "SLURM Spatial Agent Evaluation"
echo "========================================"
echo "Benchmark:      $BENCHMARK"
echo "Concurrency:    $CONCURRENCY"
echo "Model config:   ${MODEL_CONFIG:-<none>}"
echo "Dataset config: ${DATASET_CONFIG:-<none>}"
echo "========================================"
echo ""

# Create logs directory
mkdir -p spatial_agent/logs/

# Derive dataset config from benchmark name if not provided
if [ -z "$DATASET_CONFIG" ]; then
    DATASET_CONFIG="spatial_agent/config/dataset/${BENCHMARK}.json"
fi

# Build command
CMD="python -m spatial_agent.entrypoints.run"
CMD+=" --dataset $DATASET_CONFIG"
CMD+=" --concurrency $CONCURRENCY"
CMD+=" --resume"

if [ -n "$MODEL_CONFIG" ]; then
    CMD+=" --model $MODEL_CONFIG"
fi
if [ -n "$EXPERIMENT_NAME" ]; then
    # Mirror run.py's default work_dir layout.
    MODEL_SHORT="unknown"
    if [ -n "$MODEL_CONFIG" ] && [ -f "$MODEL_CONFIG" ]; then
        MODEL_SHORT=$(python3 -c "import json; print(json.load(open('$MODEL_CONFIG'))['llm_model'].split('/')[-1][:30])" 2>/dev/null || echo "unknown")
    fi
    WORK_DIR="spatial_agent/work_dir/spatial_${BENCHMARK}_${MODEL_SHORT}_${EXPERIMENT_NAME}"
    CMD+=" --work_dir ${WORK_DIR}"
fi
if [ "$SUBSAMPLE" -gt 0 ] 2>/dev/null; then
    CMD+=" --subsample $SUBSAMPLE"
fi

# Run the agent with resume flag
echo "Starting spatial agent evaluation..."
echo "Command: $CMD"
echo ""
eval $CMD

echo ""
echo "Spatial agent evaluation completed or interrupted"
