#!/bin/bash
# CoT Baseline Execution Script
# This script runs INSIDE each SLURM job
# Configuration is passed as arguments from the manager

set -e

# Parse arguments
BENCHMARK="$1"
CONCURRENCY="${2:-32}"
MODEL_CONFIG="${3:-}"
MAX_FRAMES="${4:-8}"
SUBSAMPLE="${5:-0}"

if [ -z "$BENCHMARK" ]; then
    echo "ERROR: Benchmark not specified"
    echo "Usage: $0 <benchmark> <concurrency> <model_config> <max_frames>"
    exit 1
fi

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

echo "========================================"
echo "SLURM CoT Baseline Evaluation"
echo "========================================"
echo "Benchmark:     $BENCHMARK"
echo "Concurrency:   $CONCURRENCY"
echo "Max Frames:    $MAX_FRAMES"
echo "Subsample:     ${SUBSAMPLE:-<all>}"
echo "Model config:  ${MODEL_CONFIG:-<none>}"
echo "========================================"
echo ""

# Derive dataset config from benchmark name
DATASET_CONFIG="spatial_agent/config/dataset/${BENCHMARK}.json"

# Build command
CMD="python -m spatial_agent.entrypoints.cot_baseline"
CMD+=" --dataset $DATASET_CONFIG"
CMD+=" --concurrency $CONCURRENCY"
CMD+=" --max_frames $MAX_FRAMES"
CMD+=" --resume"

if [ -n "$MODEL_CONFIG" ]; then
    CMD+=" --model $MODEL_CONFIG"
fi
if [ "$SUBSAMPLE" -gt 0 ] 2>/dev/null; then
    CMD+=" --subsample $SUBSAMPLE"
fi

echo "Starting CoT baseline evaluation..."
echo "Command: $CMD"
echo ""
eval $CMD

echo ""
echo "CoT baseline evaluation completed or interrupted"
