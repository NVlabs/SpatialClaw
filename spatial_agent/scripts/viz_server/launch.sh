#!/bin/bash
# User-facing launch script for SLURM visualization server
# No GPUs needed — serves a web UI for browsing work_dir results

#=============================================================================
# Configuration
#=============================================================================
PORT="${1:-8501}"  # Optional: server port (default: 8501)

#=============================================================================
# SLURM Configuration
#=============================================================================
JOB_NAME="viz-server"
ACCOUNT="llmservice_fm_vision"
PARTITION="cpu_long"
TIME_LIMIT="23:59:00"

#=============================================================================
# Help message
#=============================================================================
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    cat << EOF
SLURM Visualization Server Launcher

This script launches a continuous chain of 24-hour SLURM jobs for the
results visualization web server with automatic restart.

USAGE:
    bash spatial_agent/scripts/viz_server/launch.sh [port]

EXAMPLES:
    bash spatial_agent/scripts/viz_server/launch.sh          # Default port 8501
    bash spatial_agent/scripts/viz_server/launch.sh 8080      # Custom port

ACCESS:
    Once running, open http://<node-hostname>:<port> in your browser.
    The allocated node hostname is printed in the job log.

MONITORING:
    - Job status: squeue -u \$USER
    - Logs: tail -f spatial_agent/logs/slurm_viz/viz-server_<job_id>.out

STOP:
    Press Ctrl+C in this terminal (will cancel all jobs)
EOF
    exit 0
fi

#=============================================================================
# Launch the manager
#=============================================================================
echo "========================================"
echo "Launching SLURM Visualization Server"
echo "========================================"
echo "Port:          $PORT"
echo "Partition:     $PARTITION"
echo "Time limit:    $TIME_LIMIT"
echo "========================================"
echo ""
echo "This will run continuous 24-hour SLURM jobs"
echo "Each job will restart the server automatically"
echo "Press Ctrl+C to stop the chain"
echo "========================================"
echo ""

# Get the spatial_agent directory
SPATIAL_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Build arguments for manager
ARGS=(
    --port "$PORT"
    --job-name "$JOB_NAME"
    --account "$ACCOUNT"
    --partition "$PARTITION"
    --time "$TIME_LIMIT"
)

# Activate spatialagent environment and run manager
CONDA_BASE="$(conda info --base 2>/dev/null)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spatialagent

python "$SPATIAL_AGENT_DIR/scripts/viz_server/manager.py" "${ARGS[@]}"
