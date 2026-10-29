#!/usr/bin/env bash
# Clean LK/OSTrack + RayCaster episodes, with a fresh warmup after every reset.
set -eo pipefail

IDEAL_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IDEAL_REPO_ROOT="$(cd -- "$IDEAL_SCRIPT_DIR/../../.." && pwd)"
source /home/jby/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_232
source "$IDEAL_REPO_ROOT/_isaac_sim/setup_conda_env.sh"
set -u
cd "$IDEAL_REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

IDEAL_STAMP="$(date +%Y%m%d_%H%M%S)"
IDEAL_SPLIT="${SPLIT:-train}"
case "$IDEAL_SPLIT" in
    train) IDEAL_DEFAULT_SEED=20261012 ;;
    val) IDEAL_DEFAULT_SEED=20261013 ;;
    test) IDEAL_DEFAULT_SEED=20261014 ;;
    *) echo "SPLIT must be train, val, or test" >&2; exit 2 ;;
esac
IDEAL_SEED="${SEED:-$IDEAL_DEFAULT_SEED}"
IDEAL_RUN_ID="${COLLECTION_RUN_ID:-legacygru_full_episode_ideal_toi_${IDEAL_SPLIT}_${IDEAL_STAMP}}"
IDEAL_ROOT="${COLLECTION_ROOT:-/mnt/A/hust_jby/isaaclab_storage/test11/data/legacy_gru_full_episode_ideal_toi_20261012}"
IDEAL_OUT="${OUT_DIR:-$IDEAL_ROOT/$IDEAL_SPLIT/sd_clean_seed${IDEAL_SEED}_${IDEAL_STAMP}}"
IDEAL_POLICY="$IDEAL_REPO_ROOT/logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt"
IDEAL_OSTRACK="$IDEAL_REPO_ROOT/source/standalone/test_lk_ostrack/ostrack/OSTrack-main"
IDEAL_CKPT="$IDEAL_OSTRACK/output/checkpoints/OSTrack_ep0300.pth.tar"
for required_path in "$IDEAL_POLICY" "$IDEAL_CKPT"; do
    [[ -f "$required_path" ]] || { echo "Missing: $required_path" >&2; exit 1; }
done
[[ ! -e "$IDEAL_OUT" ]] || { echo "Refusing to overwrite: $IDEAL_OUT" >&2; exit 1; }
mkdir -p "$IDEAL_OUT"

python "$IDEAL_SCRIPT_DIR/collect_vision_data.py" \
    --headless --device cuda:0 --scenario sd --split "$IDEAL_SPLIT" \
    --validation_condition D --pair_id "$IDEAL_RUN_ID" \
    --seed "$IDEAL_SEED" --collection_run_id "$IDEAL_RUN_ID" \
    --degradation_profile clean --sensor_time_contract physical_capture_v1 \
    --capture_skip_probability 0 --episode_warmup_s "${EPISODE_WARMUP_S:-1.0}" \
    --target_speed_min 0 --target_speed_max 5 \
    --target_boundary_mode fixed_reference_toi --target_fov_margin_fraction 0.10 \
    --num_envs "${NUM_ENVS:-4}" --num_steps "${NUM_STEPS:-1000000}" \
    --target_samples "${TARGET_SAMPLES:-1200}" \
    --save_episode_stream --keep_measurement_discontinuities \
    --window_s 0.48 --max_sequence_steps 32 --max_windows_per_episode 4 \
    --supervision_burn_in_s 0.10 --progress_every 100 \
    --locomotion_policy "$IDEAL_POLICY" \
    --ostrack_repo "$IDEAL_OSTRACK" --ostrack_checkpoint "$IDEAL_CKPT" \
    --out_dir "$IDEAL_OUT" 2>&1 | tee "$IDEAL_OUT/collection.log"

if ! rg -q '\[taq-collect\] complete:' "$IDEAL_OUT/collection.log"; then
    echo "Collector did not finish successfully: $IDEAL_OUT/collection.log" >&2
    exit 1
fi
if ! compgen -G "$IDEAL_OUT/episodes/env_*/episode_*/taq_velocity_episode.taqstream" > /dev/null; then
    echo "No episode stream was recorded; check warmup and num_steps" >&2
    exit 1
fi
if [[ "${TARGET_SAMPLES:-1200}" != 0 ]] && ! rg -q 'target_reached=True' "$IDEAL_OUT/collection.log"; then
    echo "Collection stopped before reaching the requested window quota" >&2
    exit 1
fi
if [[ "${TARGET_SAMPLES:-1200}" != 0 ]]; then
    python - "$IDEAL_OUT" <<'PY'
from pathlib import Path
import sys
import torch

streams = list(Path(sys.argv[1]).rglob("*.taqstream"))
for path in streams:
    metadata = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
    if not metadata.get("episode_complete", False):
        raise SystemExit(f"Incomplete episode: {path}; increase NUM_STEPS")
print(f"Verified {len(streams)} complete episodes")
PY
fi
echo "Dataset: $IDEAL_OUT"
