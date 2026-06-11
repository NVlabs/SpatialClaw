#!/usr/bin/env bash
# download_datasets.sh — Download all SpatialAgent evaluation datasets from HuggingFace.
#
# Usage:
#   bash download_datasets.sh <HF_TOKEN>
#   bash download_datasets.sh hf_XXXXXXXXXXXXXXXXXXXX
#
# Datasets are downloaded into data/ at the project root (SpatialClaw/data/),
# which is where the benchmark loaders look for them (evals/factory.py resolves
# data/<dir> relative to the project root — the required cwd for all commands).
# Zip/tar archives are extracted automatically.
#
# Requires: conda env "spatialagent" with huggingface_hub installed.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <HUGGINGFACE_TOKEN>"
    exit 1
fi

HF_TOKEN="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/data"

mkdir -p "$DATA_DIR"
echo "=== Downloading datasets to: $DATA_DIR ==="

# Maximum parallel downloads (adjust based on bandwidth/disk IO)
MAX_PARALLEL=${MAX_PARALLEL:-4}

# ─── Dataset registry ────────────────────────────────────────────────────────
# Format: "HF_REPO_ID|TARGET_FOLDER_NAME"
# TARGET_FOLDER_NAME must match factory.py data_dir values exactly.
# One entry per paper benchmark (VSI-Bench covers both vsibench and
# vsibench_unbiased — test_debiased.parquet ships in the same repo).
DATASETS=(
    # Single-image spatial reasoning
    "FlagEval/ERQA|ERQA"
    "dmarsili/Omni3D-Bench|Omni3D-Bench"
    "qizekun/OmniSpatial|OmniSpatial"
    "hongxingli/SPBench|SPBench"
    # Multi-view spatial reasoning
    "MLL-Lab/MindCube|MindCube"
    "RunsenXu/MMSI-Bench|MMSI-Bench"
    "jasonzhango/SPAR-Bench|SPAR-Bench"
    # General spatial reasoning
    "BLINK-Benchmark/BLINK|BLINK"
    "LongfeiLi/SpatialTree-Bench|SpatialTree-Bench"
    "lidingm/ViewSpatial-Bench|ViewSpatial-Bench"
    # Video spatial & 4D reasoning
    "rbler/MMSI-Video-Bench|MMSI-Video-Bench"
    "HarmlessSR07/OSI-Bench|OSI-Bench"
    "shi-labs/physical-ai-bench-understanding|PAI-Bench"
    "nyu-visionx/VSI-Bench|VSI-Bench"
    "Journey9ni/vstibench|vstibench"
    "Viglong/DSI-Bench|DSI-Bench"
    # General video understanding
    "Dongyh35/CVBench|CVBench"
    "hrinnnn/PerceptionComp|PerceptionComp"
    "lmms-lab/Video-MME|Video-MME"
    "MME-Benchmarks/Video-MME-v2|Video-MME-v2"
)

# ─── Helper: download one dataset ────────────────────────────────────────────
download_one() {
    local repo="$1"
    local folder="$2"
    local target="$DATA_DIR/$folder"

    if [[ -d "$target" ]] && [[ "$(find "$target" -type f 2>/dev/null | head -1)" != "" ]]; then
        echo "[SKIP] $folder — already exists at $target"
        return 0
    fi

    echo "[DOWN] $folder ← $repo"
    conda run --no-banner -n spatialagent \
        hf download --repo-type dataset --token "$HF_TOKEN" \
        "$repo" --local-dir "$target" \
        2>&1 | tail -3

    if [[ $? -ne 0 ]]; then
        echo "[FAIL] $folder — download failed"
        return 1
    fi
    echo "[DONE] $folder downloaded"
}

# ─── Helper: extract all archives inside a directory ─────────────────────────
extract_archives() {
    local dir="$1"
    local name="$(basename "$dir")"

    # Extract .zip files
    find "$dir" -maxdepth 3 -name "*.zip" -print0 2>/dev/null | while IFS= read -r -d '' zipfile; do
        local parent="$(dirname "$zipfile")"
        echo "[UNZIP] $name: $(basename "$zipfile")"
        unzip -o -q "$zipfile" -d "$parent" && rm "$zipfile"
    done

    # Extract .tar.gz files
    find "$dir" -maxdepth 3 -name "*.tar.gz" -print0 2>/dev/null | while IFS= read -r -d '' tarfile; do
        local parent="$(dirname "$tarfile")"
        echo "[UNTAR] $name: $(basename "$tarfile")"
        tar xzf "$tarfile" -C "$parent" && rm "$tarfile"
    done

    # Extract .tar files
    find "$dir" -maxdepth 3 -name "*.tar" -print0 2>/dev/null | while IFS= read -r -d '' tarfile; do
        local parent="$(dirname "$tarfile")"
        echo "[UNTAR] $name: $(basename "$tarfile")"
        tar xf "$tarfile" -C "$parent" && rm "$tarfile"
    done
}

# ─── Main: download all datasets with parallelism ────────────────────────────
echo ""
echo "Downloading ${#DATASETS[@]} datasets (max $MAX_PARALLEL parallel)..."
echo ""

running=0
for entry in "${DATASETS[@]}"; do
    repo="${entry%%|*}"
    folder="${entry##*|}"

    # Run download in background
    (
        download_one "$repo" "$folder"
        extract_archives "$DATA_DIR/$folder"
    ) &

    running=$((running + 1))
    if [[ $running -ge $MAX_PARALLEL ]]; then
        wait -n 2>/dev/null || wait
        running=$((running - 1))
    fi
done

# Wait for all remaining background jobs
wait

# ─── Final report ────────────────────────────────────────────────────────────
echo ""
echo "========================================="
echo "  Dataset Download Report"
echo "========================================="

total=0
done_count=0
missing_count=0

for entry in "${DATASETS[@]}"; do
    folder="${entry##*|}"
    target="$DATA_DIR/$folder"
    total=$((total + 1))

    if [[ -d "$target" ]]; then
        fcount=$(find "$target" -type f 2>/dev/null | wc -l)
        zcount=$(find "$target" -maxdepth 3 \( -name "*.zip" -o -name "*.tar.gz" \) 2>/dev/null | wc -l)
        if [[ $zcount -gt 0 ]]; then
            echo "  WARN  $folder ($fcount files, $zcount archives not extracted)"
        else
            echo "  OK    $folder ($fcount files)"
        fi
        done_count=$((done_count + 1))
    else
        echo "  MISS  $folder"
        missing_count=$((missing_count + 1))
    fi
done

echo "========================================="
echo "  Total: $total | Downloaded: $done_count | Missing: $missing_count"
echo "========================================="
echo ""
echo "Done."
