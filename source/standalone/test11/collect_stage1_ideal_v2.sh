#!/usr/bin/env bash

# Stage-1 ideal-v2 collection.  Each invocation uses an independent run ID;
# the collector itself enforces non-overlap, a 20 ms inter-window gap and the
# four-window episode cap.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "train" && "$MODE" != "val" && "$MODE" != "test" && "$MODE" != "all" ]]; then
    echo "Usage: $0 train|val|test|all"
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
COLLECTION_DEVICE="${COLLECTION_DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-2}"
NUM_STEPS="${NUM_STEPS:-1000000}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$SCRIPT_DIR/data/stage1_ideal_v2}"
POLICY="${POLICY:-$REPO_ROOT/logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt}"
OSTRACK_REPO="${OSTRACK_REPO:-$REPO_ROOT/source/standalone/test_lk_ostrack/ostrack/OSTrack-main}"
OSTRACK_CKPT="${OSTRACK_CKPT:-$OSTRACK_REPO/output/checkpoints/OSTrack_ep0300.pth.tar}"

for required_path in "$POLICY" "$OSTRACK_REPO" "$OSTRACK_CKPT"; do
    [[ -e "$required_path" ]] || { echo "Required path was not found: $required_path"; exit 1; }
done

target_windows()
{
    case "$1:$2" in
        train:ss) echo 4800 ;;
        train:sd) echo 7200 ;;
        val:ss|test:ss) echo 1200 ;;
        val:sd|test:sd) echo 1800 ;;
        *) return 2 ;;
    esac
}

collection_seed()
{
    case "$1:$2" in
        train:ss) echo 7001 ;;
        train:sd) echo 7002 ;;
        val:ss) echo 7101 ;;
        val:sd) echo 7102 ;;
        test:ss) echo 7201 ;;
        test:sd) echo 7202 ;;
        *) return 2 ;;
    esac
}

collect_split()
{
    local split="$1" scenario run_id out_dir target seed
    local -a stream_args
    for scenario in ss sd; do
        target="$(target_windows "$split" "$scenario")"
        seed="$(collection_seed "$split" "$scenario")"
        run_id="stage1_ideal_v2_${split}_${scenario}_$(date +%Y%m%dT%H%M%S)"
        out_dir="$CAMPAIGN_ROOT/$split/$scenario"
        stream_args=()
        if [[ "$split" == "val" || "$split" == "test" ]]; then
            stream_args+=(--save_episode_stream)
        fi
        [[ ! -e "$out_dir" ]] || { echo "Refusing to overwrite: $out_dir"; exit 1; }
        mkdir -p "$out_dir"
        "$PYTHON_BIN" "$SCRIPT_DIR/collect_vision_data.py" \
            --device "$COLLECTION_DEVICE" --scenario "$scenario" --split "$split" \
            --degradation_profile clean --severity medium --seed "$seed" \
            --collection_run_id "$run_id" --target_samples "$target" \
            --window_s 0.48 --max_sequence_steps 32 --max_windows_per_episode 4 \
            --min_window_gap_s 0.02 --supervision_burn_in_s 0.10 \
            --num_steps "$NUM_STEPS" --num_envs "$NUM_ENVS" --progress_every 500 \
            --headless --out_dir "$out_dir" --locomotion_policy "$POLICY" \
            "${stream_args[@]}" \
            --ostrack_repo "$OSTRACK_REPO" --ostrack_checkpoint "$OSTRACK_CKPT"
        "$PYTHON_BIN" "$SCRIPT_DIR/speed_prediction/audit_stage1_ideal_v2.py" "$out_dir" \
            | tee "$out_dir/audit_report.json"
    done
}

if [[ "$MODE" == "all" ]]; then
    collect_split train
    collect_split val
    collect_split test
else
    collect_split "$MODE"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/speed_prediction/audit_stage1_ideal_v2.py" "$CAMPAIGN_ROOT" \
    | tee "$CAMPAIGN_ROOT/audit_report.json"
