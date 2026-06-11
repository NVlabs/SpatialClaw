#!/bin/bash
# User-facing launch script for the SLURM GPU dashboard.
# The manager process stays on the login node, runs the background sampler,
# and keeps a chain of 24h SLURM jobs alive that serve the web UI.

PORT="${1:-8502}"

JOB_NAME="gpu-dashboard"
ACCOUNT="llmservice_fm_vision"
PARTITION="cpu_short"
TIME_LIMIT="3:59:00"
SAMPLE_INTERVAL="5"
HISTORY_SEC="3600"

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    cat << EOF
SLURM GPU Dashboard Launcher

Launches a continuous chain of 24-hour SLURM jobs that host a web UI showing
vLLM + GPU-tool server utilization over time, correlated with the number of
running agent experiments.

USAGE:
    bash spatial_agent/scripts/gpu_dashboard/launch.sh [port]

EXAMPLES:
    bash spatial_agent/scripts/gpu_dashboard/launch.sh          # Default port 8502
    bash spatial_agent/scripts/gpu_dashboard/launch.sh 8600

ACCESS:
    Once running, open http://<node-hostname>:<port> or read the endpoint
    from spatial_agent/logs/gpu_dashboard_serve.json.

MONITORING:
    - Job status:  squeue -u \$USER
    - Web logs:    tail -f spatial_agent/logs/slurm_gpu_dashboard/gpu-dashboard_<job_id>.out
    - Sampler log: the sampler prints to THIS terminal (it runs outside SLURM).

STOP:
    Press Ctrl+C in this terminal (cancels jobs, stops the sampler).
EOF
    exit 0
fi

echo "========================================"
echo "Launching SLURM GPU Dashboard"
echo "========================================"
echo "Port:             $PORT"
echo "Partition:        $PARTITION"
echo "Time limit:       $TIME_LIMIT"
echo "Sample interval:  ${SAMPLE_INTERVAL}s"
echo "History window:   ${HISTORY_SEC}s"
echo "========================================"
echo ""
echo "Sampler runs on this login node (survives SLURM job rotations)."
echo "Web server runs as chained 24h SLURM jobs. Ctrl+C to stop."
echo "========================================"
echo ""

SPATIAL_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ARGS=(
    --port "$PORT"
    --job-name "$JOB_NAME"
    --account "$ACCOUNT"
    --partition "$PARTITION"
    --time "$TIME_LIMIT"
    --sample-interval "$SAMPLE_INTERVAL"
    --history-sec "$HISTORY_SEC"
)

CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent

python "$SPATIAL_AGENT_DIR/scripts/gpu_dashboard/manager.py" "${ARGS[@]}"
