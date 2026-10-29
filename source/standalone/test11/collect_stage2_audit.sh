#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-quick}"
if [[ "$MODE" != "quick" && "$MODE" != "standard" ]]; then
    echo "Usage: $0 [quick|standard]"
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
COLLECTOR="$SCRIPT_DIR/collect_vision_data.py"
AUDITOR="$SCRIPT_DIR/speed_prediction/audit_stage2_data.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

POLICY="${POLICY:-$REPO_ROOT/logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt}"
OSTRACK_REPO="${OSTRACK_REPO:-$REPO_ROOT/source/standalone/test_lk_ostrack/ostrack/OSTrack-main}"
OSTRACK_CKPT="${OSTRACK_CKPT:-$OSTRACK_REPO/output/checkpoints/OSTrack_ep0300.pth.tar}"
RUN_TAG="$(date +%m%d_%H%M%S)"
AUDIT_ROOT="${AUDIT_ROOT:-$SCRIPT_DIR/data/stage2_static_nonideal/audit_$RUN_TAG}"

for required_path in "$COLLECTOR" "$AUDITOR" "$POLICY" "$OSTRACK_REPO" "$OSTRACK_CKPT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "[stage2-audit] Required path was not found: $required_path"
        exit 1
    fi
done

if ! "$PYTHON_BIN" -c "import torch, cv2; from isaaclab.app import AppLauncher"; then
    echo "[stage2-audit] Python environment check failed."
    echo "Activate the same IsaacLab environment used for Stage-1 collection, then retry."
    echo "You can override the executable with PYTHON_BIN=/path/to/python."
    exit 1
fi

if [[ "$MODE" == "quick" ]]; then
    NUM_ENVS="${NUM_ENVS:-1}"
    CLEAN_SS=8
    CLEAN_SD=8
    SINGLE_SD=8
    MIXED_SD=24
    MIXED_SS=16
else
    NUM_ENVS="${NUM_ENVS:-2}"
    CLEAN_SS=200
    CLEAN_SD=200
    SINGLE_SD=300
    MIXED_SD=800
    MIXED_SS=500
fi

NUM_STEPS="${NUM_STEPS:-50000}"
MAX_WINDOWS_PER_EPISODE="${MAX_WINDOWS_PER_EPISODE:-8}"
MIN_TIMESTAMP_DT_RATIO="${MIN_TIMESTAMP_DT_RATIO:-0.75}"
# GPU 0 is commonly occupied by training/desktop rendering on this machine.
# COLLECTION_DEVICE can be overridden, for example:
#   COLLECTION_DEVICE=cuda:0 bash collect_stage2_audit.sh quick
COLLECTION_DEVICE="${COLLECTION_DEVICE:-cuda:1}"
mkdir -p "$AUDIT_ROOT"

log_matches()
{
    local pattern="$1"
    local log_path="$2"
    if command -v rg >/dev/null 2>&1; then
        rg -q -- "$pattern" "$log_path"
    else
        grep -Eq -- "$pattern" "$log_path"
    fi
}

collect_one()
{
    local scenario="$1"
    local profile="$2"
    local severity="$3"
    local seed="$4"
    local target_samples="$5"
    local output_dir="$AUDIT_ROOT/$scenario/$profile"

    mkdir -p "$output_dir"
    echo
    echo "[stage2-audit] Collecting scenario=$scenario profile=$profile severity=$severity target=$target_samples"
    if ! "$PYTHON_BIN" "$COLLECTOR" \
            --device "$COLLECTION_DEVICE" \
            --scenario "$scenario" \
            --split train \
            --degradation_profile "$profile" \
            --severity "$severity" \
            --seed "$seed" \
            --degradation_seed "$((seed + 10000))" \
            --target_samples "$target_samples" \
            --max_windows_per_episode "$MAX_WINDOWS_PER_EPISODE" \
            --min_timestamp_dt_ratio "$MIN_TIMESTAMP_DT_RATIO" \
            --num_steps "$NUM_STEPS" \
            --num_envs "$NUM_ENVS" \
            --progress_every 100 \
            --headless \
            --out_dir "$output_dir" \
            --locomotion_policy "$POLICY" \
            --ostrack_repo "$OSTRACK_REPO" \
            --ostrack_checkpoint "$OSTRACK_CKPT" \
            2>&1 | tee "$output_dir/collector.log"; then
        echo "[stage2-audit] ERROR: collector returned a non-zero status."
        echo "[stage2-audit] See: $output_dir/collector.log"
        exit 1
    fi

    # Isaac Sim can occasionally print a fatal Python/PhysX error but still
    # return exit code 0.  Validate both the log and the actual data product so
    # the script cannot misreport a failed launch as a completed collection.
    if log_matches \
        "Traceback \(most recent call last\):|Failed to create simulation view backend|unable to create a PhysicsScene" \
        "$output_dir/collector.log"; then
        echo "[stage2-audit] ERROR: collector crashed for scenario=$scenario profile=$profile."
        if log_matches \
            "failed to allocate memory|Not enough GPU memory|CUDA out of memory" \
            "$output_dir/collector.log"; then
            echo "[stage2-audit] Cause: insufficient GPU memory on $COLLECTION_DEVICE."
            echo "[stage2-audit] Close the process using that GPU or retry with COLLECTION_DEVICE=cuda:N."
        fi
        echo "[stage2-audit] See: $output_dir/collector.log"
        exit 1
    fi

    if [[ -z "$(find "$output_dir" -type f -name 'taq_velocity_chunk_*.pt' -print -quit)" ]]; then
        echo "[stage2-audit] ERROR: collector produced no TAQ chunk for scenario=$scenario profile=$profile."
        echo "[stage2-audit] See: $output_dir/collector.log"
        exit 1
    fi

    if ! log_matches "\[taq-collect\] complete: saved_samples=" "$output_dir/collector.log"; then
        echo "[stage2-audit] WARNING: completion marker was not flushed, but a TAQ chunk exists; continuing."
    fi
}

echo "[stage2-audit] mode=$MODE num_envs=$NUM_ENVS max_windows_per_episode=$MAX_WINDOWS_PER_EPISODE min_timestamp_dt_ratio=$MIN_TIMESTAMP_DT_RATIO device=$COLLECTION_DEVICE output=$AUDIT_ROOT"

collect_one ss clean medium 3101 "$CLEAN_SS"
collect_one sd clean medium 3201 "$CLEAN_SD"
collect_one sd image medium 3301 "$SINGLE_SD"
collect_one sd tracking medium 3401 "$SINGLE_SD"
collect_one sd range medium 3501 "$SINGLE_SD"
collect_one sd timing medium 3601 "$SINGLE_SD"
collect_one sd proprio medium 3701 "$SINGLE_SD"
collect_one sd mixed random 3801 "$MIXED_SD"
collect_one ss mixed random 3901 "$MIXED_SS"

echo
echo "[stage2-audit] Collection complete. Running data audit..."
"$PYTHON_BIN" "$AUDITOR" "$AUDIT_ROOT" | tee "$AUDIT_ROOT/audit_report.txt"
echo
echo "[stage2-audit] Done: $AUDIT_ROOT"
