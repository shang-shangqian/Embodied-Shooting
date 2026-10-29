#!/usr/bin/env bash

# Collect clean SS/SD TAQ velocity training data on GPU 0.
# SD uses 3-D target speeds and reflects at the live camera frustum.
set -eo pipefail

COLLECT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COLLECT_REPO_ROOT="$(cd -- "$COLLECT_SCRIPT_DIR/../../.." && pwd)"
MODE="${1:-all}"
case "$MODE" in
    all|ss|sd) ;;
    *) echo "Usage: $0 [all|ss|sd]" >&2; exit 2 ;;
esac

source /home/jby/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_232
# This setup script uses variables that may be unset, so enable nounset after it.
source "$COLLECT_REPO_ROOT/_isaac_sim/setup_conda_env.sh"
set -u

cd "$COLLECT_REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$COLLECT_SCRIPT_DIR/data/velocity_train_fov3d_$RUN_STAMP}"
SEED_BASE="${SEED_BASE:-$(date +%s)}"
SEED_SS="$SEED_BASE"
SEED_SD="$((SEED_BASE + 1))"

COLLECTOR="$COLLECT_SCRIPT_DIR/collect_vision_data.py"
POLICY="$COLLECT_REPO_ROOT/logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt"
OSTRACK_REPO="$COLLECT_REPO_ROOT/source/standalone/test_lk_ostrack/ostrack/OSTrack-main"
OSTRACK_CKPT="$OSTRACK_REPO/output/checkpoints/OSTrack_ep0300.pth.tar"

for required_path in "$COLLECTOR" "$POLICY" "$OSTRACK_REPO" "$OSTRACK_CKPT"; do
    [[ -e "$required_path" ]] || { echo "Missing required path: $required_path" >&2; exit 1; }
done
if [[ "$MODE" == "all" && -e "$CAMPAIGN_ROOT" ]]; then
    echo "Refusing to overwrite existing output: $CAMPAIGN_ROOT" >&2
    exit 1
fi
if [[ "$MODE" != "all" && -e "$CAMPAIGN_ROOT/train/$MODE" ]]; then
    echo "Refusing to overwrite existing scenario: $CAMPAIGN_ROOT/train/$MODE" >&2
    exit 1
fi
mkdir -p "$CAMPAIGN_ROOT"

collect_scenario()
{
    local scenario="$1" seed="$2" target_samples="$3"
    local output_dir="$CAMPAIGN_ROOT/train/$scenario"
    local log_path="$CAMPAIGN_ROOT/$scenario.log"

    echo "Collecting $scenario: seed=$seed target_samples=$target_samples output=$output_dir"
    python "$COLLECTOR" \
        --headless --device cuda:0 --num_envs 4 --num_steps 1000000 \
        --scenario "$scenario" --split train --validation_condition none \
        --degradation_profile clean --seed "$seed" \
        --target_samples "$target_samples" \
        --window_s 0.48 --max_sequence_steps 32 \
        --max_windows_per_episode 4 --supervision_burn_in_s 0.10 \
        --progress_every 500 \
        --locomotion_policy "$POLICY" \
        --ostrack_repo "$OSTRACK_REPO" \
        --ostrack_checkpoint "$OSTRACK_CKPT" \
        --out_dir "$output_dir" 2>&1 | tee "$log_path"

    if ! rg -q 'target_reached=True' "$log_path"; then
        echo "Collection stopped before reaching $target_samples windows: $log_path" >&2
        exit 1
    fi
}

if [[ "$MODE" == "all" || "$MODE" == "ss" ]]; then
    collect_scenario ss "$SEED_SS" 4800
fi
if [[ "$MODE" == "all" || "$MODE" == "sd" ]]; then
    collect_scenario sd "$SEED_SD" 7200
fi

echo "Training data: $CAMPAIGN_ROOT"
