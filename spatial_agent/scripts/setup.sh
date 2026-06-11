#!/usr/bin/env bash
# ============================================================================
# SpatialAgent — Environment Setup Script
#
# Creates three environments:
#   1. spatialagent  (conda)  — agent, managers, GPU server
#   2. .venv         (uv)     — vLLM inference server
#   3. spatialclaw-cuda      (conda)  — CUDA shared libraries borrowed by vLLM at runtime
#
# Usage:
#   bash setup.sh            # full install (all three environments)
#   bash setup.sh --agent    # agent conda env only
#   bash setup.sh --vllm     # vLLM .venv only
#   bash setup.sh --cuda     # spatialclaw-cuda CUDA env only
# ============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDA_ENV_NAME="spatialagent"
CUDA_ENV_NAME="spatialclaw-cuda"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.8"

# scripts/ -> spatial_agent/ -> SpatialAgent/
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIREMENTS_AGENT="$PROJECT_ROOT/spatial_agent/requirements/requirements-agent.txt"
REQUIREMENTS_VLLM="$PROJECT_ROOT/spatial_agent/requirements/requirements-vllm.txt"
THIRD_PARTY="$PROJECT_ROOT/tools/third_party"

# Third-party model code (SAM3, Pi3, Depth-Anything-3,
# map-anything) is tracked as git submodules — see .gitmodules at the
# project root. Pinned commits live there.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
fail()  { echo -e "\033[1;31m[FAIL]\033[0m  $*"; exit 1; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
check_conda() {
    if ! command -v conda &>/dev/null; then
        fail "conda not found. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    fi
    CONDA_BASE="$(conda info --base 2>/dev/null)"
    # Make sure conda shell functions (activate/deactivate) are available.
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    ok "conda found at $CONDA_BASE"
}

check_requirements() {
    [ -f "$REQUIREMENTS_AGENT" ] || fail "Missing $REQUIREMENTS_AGENT"
    [ -f "$REQUIREMENTS_VLLM" ]  || fail "Missing $REQUIREMENTS_VLLM"
    [ -d "$PROJECT_ROOT/spatial_agent" ] || fail "Missing spatial_agent/ package directory"
    ok "Project files present"
}

# ---------------------------------------------------------------------------
# 1. Agent conda environment
# ---------------------------------------------------------------------------
setup_agent() {
    info "=== Setting up agent conda environment ($CONDA_ENV_NAME) ==="

    if conda env list 2>/dev/null | grep -qw "$CONDA_ENV_NAME"; then
        info "Conda env '$CONDA_ENV_NAME' already exists — installing into it"
    else
        info "Creating conda env '$CONDA_ENV_NAME' (Python $PYTHON_VERSION)..."
        conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y -q
    fi

    conda activate "$CONDA_ENV_NAME"

    info "Installing agent dependencies..."
    pip install --no-user -q -r "$REQUIREMENTS_AGENT"

    # ffmpeg binary (needed for video frame extraction in evals)
    if ! command -v ffmpeg &>/dev/null; then
        info "Installing ffmpeg via conda..."
        conda install -y -q ffmpeg
    fi

    # Install sam3 as editable with all deps (pycocotools, timm, iopath, etc.)
    if [ -d "$THIRD_PARTY/sam3" ]; then
        info "Installing sam3 (editable, with dev extras)..."
        pip install --no-user -q -e "$THIRD_PARTY/sam3[dev]"
    fi

    # uv is needed to create the vLLM venv later
    if ! command -v uv &>/dev/null; then
        info "Installing uv..."
        pip install --no-user -q uv
    fi

    # Verify critical imports
    python -c "
import langgraph, openai, jupyter_client, fastapi, pynvml, numpy, cloudpickle, rich
print('All agent imports OK')
" 2>/dev/null || fail "Agent import verification failed"

    conda deactivate
    ok "Agent environment ready"
}

# ---------------------------------------------------------------------------
# 2. vLLM virtual environment (.venv via uv)
# ---------------------------------------------------------------------------
setup_vllm() {
    info "=== Setting up vLLM virtual environment (.venv) ==="

    # Ensure uv is available (may be in the agent env or on PATH)
    if ! command -v uv &>/dev/null; then
        if conda env list 2>/dev/null | grep -qw "$CONDA_ENV_NAME"; then
            conda activate "$CONDA_ENV_NAME"
        else
            fail "uv not found and agent env not set up. Run 'bash setup.sh --agent' first."
        fi
    fi

    if [ -d "$PROJECT_ROOT/.venv" ]; then
        info ".venv already exists — reinstalling packages"
    else
        info "Creating .venv (Python $PYTHON_VERSION)..."
        uv venv "$PROJECT_ROOT/.venv" --python "$PYTHON_VERSION"
    fi

    info "Installing nightly vLLM with CUDA 12.9 (required for Gemma4)..."
    uv pip install --python "$PROJECT_ROOT/.venv/bin/python" \
        -U vllm --pre \
        --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        --index-strategy unsafe-best-match

    info "Pinning transformers==5.5.0..."
    uv pip install --python "$PROJECT_ROOT/.venv/bin/python" transformers==5.5.0

    info "Installing additional dependencies (pynvml, pandas)..."
    uv pip install --python "$PROJECT_ROOT/.venv/bin/python" pynvml pandas

    # deep_gemm: required for FP8 models (e.g. Gemma-4-31B-it-FP8).
    # Must be built from source with git submodules (CUTLASS).
    if "$PROJECT_ROOT/.venv/bin/python" -c "import deep_gemm" 2>/dev/null; then
        ok "deep_gemm already installed"
    else
        info "Building deep_gemm from source (FP8 kernel support)..."
        local DG_DIR="${TMPDIR:-/tmp}/DeepGEMM"
        rm -rf "$DG_DIR"
        if git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git "$DG_DIR" 2>/dev/null; then
            local CUDA_HOME_BK="$CUDA_HOME"
            export CUDA_HOME="$CONDA_BASE/envs/$CUDA_ENV_NAME"
            export CPLUS_INCLUDE_PATH="$CUDA_HOME/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
            (cd "$DG_DIR" && "$PROJECT_ROOT/.venv/bin/python" setup.py bdist_wheel 2>/dev/null) && {
                # Rename wheel tag for compatibility (linux_x86_64 -> manylinux)
                local WHL
                WHL=$(ls "$DG_DIR"/dist/*.whl 2>/dev/null | head -1)
                if [ -n "$WHL" ]; then
                    local FIXED="${WHL/linux_x86_64/manylinux_2_35_x86_64}"
                    [ "$WHL" != "$FIXED" ] && cp "$WHL" "$FIXED" && WHL="$FIXED"
                    uv pip install --python "$PROJECT_ROOT/.venv/bin/python" "$WHL"
                    ok "deep_gemm installed"
                fi
            } || warn "deep_gemm build failed — FP8 models will not work"
            export CUDA_HOME="${CUDA_HOME_BK:-}"
            rm -rf "$DG_DIR"
        else
            warn "Failed to clone DeepGEMM — FP8 models will not work"
        fi
    fi

    # Verify
    "$PROJECT_ROOT/.venv/bin/python" -c "
import vllm, pynvml, transformers, pandas
print(f'vLLM {vllm.__version__}, transformers {transformers.__version__} OK')
" 2>/dev/null || fail "vLLM import verification failed"

    # Deactivate if we activated the agent env above
    conda deactivate 2>/dev/null || true
    ok "vLLM environment ready"
}

# ---------------------------------------------------------------------------
# 3. CUDA conda environment (spatialclaw-cuda) — shared libraries for vLLM
# ---------------------------------------------------------------------------
setup_cuda() {
    info "=== Setting up CUDA conda environment ($CUDA_ENV_NAME) ==="

    if conda env list 2>/dev/null | grep -qw "$CUDA_ENV_NAME"; then
        info "Conda env '$CUDA_ENV_NAME' already exists — checking CUDA toolkit"
    else
        info "Creating conda env '$CUDA_ENV_NAME'..."
        conda create -n "$CUDA_ENV_NAME" -y -q
    fi

    # Check if cuda-toolkit is already installed
    CUDA_LIB="$CONDA_BASE/envs/$CUDA_ENV_NAME/lib/libcudart.so"
    if [ -f "$CUDA_LIB" ]; then
        ok "CUDA toolkit already installed"
    else
        info "Installing CUDA toolkit $CUDA_VERSION (this may take several minutes)..."
        conda install -n "$CUDA_ENV_NAME" -c nvidia "cuda-toolkit=$CUDA_VERSION" -y -q
    fi

    # Verify
    if ls "$CONDA_BASE/envs/$CUDA_ENV_NAME/lib/libcudart"* &>/dev/null; then
        ok "CUDA environment ready"
    else
        fail "CUDA libs not found in $CUDA_ENV_NAME"
    fi
}

# ---------------------------------------------------------------------------
# 4. Initialize third-party submodules + download model weights
#
# All third-party model code (SAM3, Pi3, Depth-Anything-3,
# map-anything) lives under tools/third_party/ as git submodules pinned to
# the commits below. A normal `git clone --recursive ...` of this repo
# already populates them; this function only fills in submodules that were
# missed (e.g. someone cloned without --recursive).
# ---------------------------------------------------------------------------

setup_third_party() {
    info "=== Initializing third-party submodules and weights ==="

    if [ -f "$PROJECT_ROOT/.gitmodules" ]; then
        info "Running git submodule update --init --recursive..."
        git -C "$PROJECT_ROOT" submodule update --init --recursive
        ok "Submodules initialized"
    else
        warn "No .gitmodules at $PROJECT_ROOT — skipping submodule init"
    fi

    mkdir -p "$THIRD_PARTY"

    # --- SAM3.1 weights (gated HF repo — requires login + license) ---
    SAM3_WEIGHTS_DIR="$THIRD_PARTY/sam3/weights"
    SAM3_CHECKPOINT="$SAM3_WEIGHTS_DIR/sam3.1_multiplex.pt"
    SAM3_BPE="$SAM3_WEIGHTS_DIR/bpe_simple_vocab_16e6.txt.gz"
    SAM3_BPE_SRC="$THIRD_PARTY/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

    if [ -f "$SAM3_CHECKPOINT" ]; then
        ok "SAM3.1 checkpoint already present"
    else
        mkdir -p "$SAM3_WEIGHTS_DIR"
        info "Downloading SAM3.1 weights from huggingface.co/facebook/sam3.1..."
        info "(Requires HuggingFace login and license acceptance)"
        if command -v huggingface-cli &>/dev/null; then
            huggingface-cli download facebook/sam3.1 \
                --local-dir "$SAM3_WEIGHTS_DIR" \
                --include "*.pt" "*.txt.gz" || {
                warn "SAM3.1 download failed. You may need to:"
                warn "  1. Accept the license at https://huggingface.co/facebook/sam3.1"
                warn "  2. Run: huggingface-cli login"
            }
        else
            warn "huggingface-cli not found — install with: pip install huggingface-hub"
        fi
    fi

    # Symlink BPE vocab into weights/ if not already there
    if [ -f "$SAM3_CHECKPOINT" ] && [ ! -e "$SAM3_BPE" ] && [ -f "$SAM3_BPE_SRC" ]; then
        ln -sf ../sam3/assets/bpe_simple_vocab_16e6.txt.gz "$SAM3_BPE"
        ok "BPE vocab symlinked"
    fi

    # Pi3 weights auto-download from HuggingFace on first GPU server launch.
    # DA3 and MapAnything default to local cache use in their GPU wrappers.
    info "Pi3 weights will auto-download from HF (yyfz233/Pi3X) on first use"
    info "DA3 weights should be pre-cached from HF (depth-anything/DA3NESTED-GIANT-LARGE-1.1)"
    info "MapAnything weights should be pre-cached from HF (facebook/map-anything)"
    info "MapAnything also requires DINOv2 giant torch-hub cache under tools/third_party/torch_hub"
}

# ---------------------------------------------------------------------------
# 5. Create log directories
# ---------------------------------------------------------------------------
setup_dirs() {
    info "Creating log directories..."
    mkdir -p "$PROJECT_ROOT/spatial_agent/logs"/{slurm_vllm,slurm_agent,slurm_cot,slurm_gpu_server}
    ok "Directories ready"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    echo ""
    echo "============================================="
    echo "  SpatialAgent Environment Setup"
    echo "============================================="
    echo "  Project root: $PROJECT_ROOT"
    echo "============================================="
    echo ""

    check_conda
    check_requirements

    # Parse arguments
    local do_agent=false do_vllm=false do_cuda=false
    if [ $# -eq 0 ]; then
        do_agent=true; do_vllm=true; do_cuda=true
    else
        for arg in "$@"; do
            case "$arg" in
                --agent) do_agent=true ;;
                --vllm)  do_vllm=true ;;
                --cuda)  do_cuda=true ;;
                --help|-h)
                    echo "Usage: bash setup.sh [--agent] [--vllm] [--cuda]"
                    echo ""
                    echo "  --agent   Set up the spatialagent conda env (agent + managers + GPU server)"
                    echo "  --vllm    Set up the .venv for vLLM inference server"
                    echo "  --cuda    Set up the spatialclaw-cuda conda env with CUDA toolkit"
                    echo ""
                    echo "  No flags = install all three environments."
                    exit 0
                    ;;
                *) fail "Unknown argument: $arg (try --help)" ;;
            esac
        done
    fi

    $do_agent && setup_agent
    $do_cuda  && setup_cuda
    $do_vllm  && setup_vllm
    setup_third_party
    setup_dirs

    echo ""
    echo "============================================="
    echo "  Setup complete!"
    echo "============================================="
    echo ""
    echo "  Quick start:"
    echo "    conda activate spatialagent"
    echo "    cd $PROJECT_ROOT"
    echo ""
    echo "    # Start a vLLM server"
    echo "    python -m spatial_agent.launch_managers.vllm_manager"
    echo ""
    echo "    # Start an agent experiment"
    echo "    python -m spatial_agent.launch_managers.agent_manager"
    echo ""
    echo "  See spatial_agent/README.md for full documentation."
    echo "============================================="
}

main "$@"
