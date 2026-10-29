#!/usr/bin/env bash

# Formal Stage-2 non-ideal data collection.
#
# The campaign is intentionally split into independent train/val/test launches.
# Every split/scenario/profile receives unique simulation and degradation seeds.
# Existing output directories are never reused because the collector starts its
# episode numbering from zero on every process launch.

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "train" && "$MODE" != "val" && "$MODE" != "test" && "$MODE" != "all" ]]; then
    echo "Usage: $0 train|val|test|all"
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
COLLECTOR="$SCRIPT_DIR/collect_vision_data.py"
AUDITOR="$SCRIPT_DIR/speed_prediction/audit_stage2_data.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
COLLECTION_DEVICE="${COLLECTION_DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-2}"
NUM_STEPS="${NUM_STEPS:-2000000}"
MAX_WINDOWS_PER_EPISODE="${MAX_WINDOWS_PER_EPISODE:-8}"
MIN_TIMESTAMP_DT_RATIO="${MIN_TIMESTAMP_DT_RATIO:-0.75}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$SCRIPT_DIR/data/stage2_static_nonideal/formal_v1}"

POLICY="${POLICY:-$REPO_ROOT/logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt}"
OSTRACK_REPO="${OSTRACK_REPO:-$REPO_ROOT/source/standalone/test_lk_ostrack/ostrack/OSTrack-main}"
OSTRACK_CKPT="${OSTRACK_CKPT:-$OSTRACK_REPO/output/checkpoints/OSTrack_ep0300.pth.tar}"

for required_path in "$COLLECTOR" "$AUDITOR" "$POLICY" "$OSTRACK_REPO" "$OSTRACK_CKPT"; do
    if [[ ! -e "$required_path" ]]; then
        echo "[stage2-formal] Required path was not found: $required_path"
        exit 1
    fi
done

if ! "$PYTHON_BIN" -c "import torch, cv2; from isaaclab.app import AppLauncher"; then
    echo "[stage2-formal] Python environment check failed."
    echo "Activate isaac_lab_232 or set PYTHON_BIN=/path/to/its/python."
    exit 1
fi

if [[ "$NUM_ENVS" -lt 1 || "$NUM_STEPS" -lt 1 || "$MAX_WINDOWS_PER_EPISODE" -lt 1 ]]; then
    echo "[stage2-formal] NUM_ENVS, NUM_STEPS and MAX_WINDOWS_PER_EPISODE must be positive."
    exit 2
fi

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

profile_total()
{
    local scenario="$1"
    local profile="$2"
    case "$scenario:$profile" in
        ss:image|ss:tracking|ss:range|ss:mixed) echo 20000 ;;
        ss:timing|ss:proprio) echo 10000 ;;
        sd:image|sd:tracking|sd:range) echo 25000 ;;
        sd:timing) echo 15000 ;;
        sd:proprio) echo 10000 ;;
        sd:mixed) echo 20000 ;;
        *) echo "[stage2-formal] Unsupported quota key: $scenario/$profile" >&2; return 2 ;;
    esac
}

split_percent()
{
    case "$1" in
        train) echo 66 ;;
        val) echo 16 ;;
        test) echo 18 ;;
        *) return 2 ;;
    esac
}

split_seed_code()
{
    case "$1" in
        train) echo 1 ;;
        val) echo 2 ;;
        test) echo 3 ;;
        *) return 2 ;;
    esac
}

collect_one()
{
    local split="$1"
    local scenario="$2"
    local profile="$3"
    local profile_index="$4"
    local total percent target split_code scenario_code seed degradation_seed output_dir log_path
    total="$(profile_total "$scenario" "$profile")"
    percent="$(split_percent "$split")"
    target=$((total * percent / 100))
    split_code="$(split_seed_code "$split")"
    if [[ "$scenario" == "ss" ]]; then
        scenario_code=1
    else
        scenario_code=2
    fi
    seed=$((50000 + split_code * 1000 + scenario_code * 100 + profile_index * 10 + 1))
    degradation_seed=$((seed + 500000))
    output_dir="$CAMPAIGN_ROOT/$split/$scenario/$profile"
    log_path="$output_dir/collector.log"

    # A rerun into this directory would overwrite env_*/episode_000000.  A
    # fully completed group is safe to skip; a partial group must be inspected
    # and moved aside before retrying.
    if [[ -e "$output_dir" ]]; then
        if [[ -f "$log_path" ]] \
                && [[ -n "$(find "$output_dir" -type f -name 'taq_velocity_chunk_*.pt' -print -quit)" ]] \
                && log_matches "\[taq-collect\] complete: saved_samples=.*target_reached=True" "$log_path"; then
            echo "[stage2-formal] Already complete; skipping $split/$scenario/$profile"
            return
        fi
        echo "[stage2-formal] Refusing to overwrite incomplete output: $output_dir"
        echo "[stage2-formal] Inspect and move it aside before retrying, or use a new CAMPAIGN_ROOT."
        exit 1
    fi
    mkdir -p "$output_dir"

    echo
    echo "[stage2-formal] split=$split scenario=$scenario profile=$profile target=$target seed=$seed degradation_seed=$degradation_seed"
    if ! "$PYTHON_BIN" "$COLLECTOR" \
            --device "$COLLECTION_DEVICE" \
            --scenario "$scenario" \
            --split "$split" \
            --degradation_profile "$profile" \
            --severity random \
            --seed "$seed" \
            --degradation_seed "$degradation_seed" \
            --target_samples "$target" \
            --max_windows_per_episode "$MAX_WINDOWS_PER_EPISODE" \
            --min_timestamp_dt_ratio "$MIN_TIMESTAMP_DT_RATIO" \
            --num_steps "$NUM_STEPS" \
            --num_envs "$NUM_ENVS" \
            --progress_every 500 \
            --headless \
            --out_dir "$output_dir" \
            --locomotion_policy "$POLICY" \
            --ostrack_repo "$OSTRACK_REPO" \
            --ostrack_checkpoint "$OSTRACK_CKPT" \
            2>&1 | tee "$log_path"; then
        echo "[stage2-formal] ERROR: collector returned a non-zero status."
        echo "[stage2-formal] See: $log_path"
        exit 1
    fi

    if log_matches \
        "Traceback \(most recent call last\):|Failed to create simulation view backend|unable to create a PhysicsScene|CUDA out of memory|failed to allocate memory" \
        "$log_path"; then
        echo "[stage2-formal] ERROR: fatal collector message detected: $log_path"
        exit 1
    fi
    if [[ -z "$(find "$output_dir" -type f -name 'taq_velocity_chunk_*.pt' -print -quit)" ]]; then
        echo "[stage2-formal] ERROR: no TAQ chunks were produced: $output_dir"
        exit 1
    fi
    if ! log_matches "\[taq-collect\] complete: saved_samples=.*target_reached=True" "$log_path"; then
        echo "[stage2-formal] ERROR: collector stopped before reaching target=$target."
        echo "[stage2-formal] Increase NUM_STEPS and use a new CAMPAIGN_ROOT."
        exit 1
    fi
}

collect_split()
{
    local split="$1"
    local scenario profile profile_index
    local profiles=(image tracking range timing proprio mixed)
    for scenario in ss sd; do
        profile_index=0
        for profile in "${profiles[@]}"; do
            profile_index=$((profile_index + 1))
            collect_one "$split" "$scenario" "$profile" "$profile_index"
        done
    done
    echo
    echo "[stage2-formal] Auditing completed split=$split"
    "$PYTHON_BIN" "$AUDITOR" "$CAMPAIGN_ROOT/$split" | tee "$CAMPAIGN_ROOT/$split/audit_report.txt"
}

echo "[stage2-formal] mode=$MODE root=$CAMPAIGN_ROOT device=$COLLECTION_DEVICE num_envs=$NUM_ENVS num_steps=$NUM_STEPS"
echo "[stage2-formal] quota scale: train=66%, val=16%, test=18% of the 220,000-window plan"

if [[ "$MODE" == "all" ]]; then
    collect_split train
    collect_split val
    collect_split test
else
    collect_split "$MODE"
fi

echo
echo "[stage2-formal] Complete: $CAMPAIGN_ROOT"
