#!/usr/bin/env bash
set -eo pipefail
source /home/jby/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_232
source /home/jby/IsaacLab-2.3.2/_isaac_sim/setup_conda_env.sh
set -u
cd /home/jby/IsaacLab-2.3.2
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
N0_OUTPUT=source/standalone/test11/velocity_error_validation/outputs/n0_capture_20261007
mkdir -p "$N0_OUTPUT"
mode="${1:-collect}"
if [[ "$mode" == status ]]; then
  python - <<'PY'
from pathlib import Path
root=Path('source/standalone/test11/velocity_error_validation/outputs/n0_capture_20261007')
for folder in sorted(root.iterdir()):
    if folder.is_dir():
        print(folder.name, 'complete', (folder/'COLLECTION_COMPLETE').exists(),
              'packet_streams', len(list(folder.rglob('timed_sensor_packets.pt'))))
PY
  exit 0
fi
if [[ "$mode" == smoke_evaluate ]]; then
  python source/standalone/test11/velocity_error_validation/tools/evaluate_n0_time_contract.py --data "$N0_OUTPUT" --device cuda:0 --smoke --out source/standalone/test11/velocity_error_validation/reports/n0_smoke_time_contract_20261007
  exit 0
fi
if [[ "$mode" == evaluate_sparse ]]; then
  python source/standalone/test11/velocity_error_validation/tools/evaluate_n0_time_contract.py --data "$N0_OUTPUT" --device cuda:0 --actual-sparse --out source/standalone/test11/velocity_error_validation/reports/n0_actual_sampling_20261007
  exit 0
fi
if [[ "$mode" == tests ]]; then
  python -m pytest source/standalone/test11/velocity_error_validation/tests source/standalone/test11/speed_prediction/tests -q -rs
  exit 0
fi
if [[ "$mode" == evaluate ]]; then
  python source/standalone/test11/velocity_error_validation/tools/evaluate_n0_time_contract.py --data "$N0_OUTPUT" --device cuda:0
  exit 0
fi
if [[ "$mode" == smoke ]]; then
  groups=(smoke); seeds=(26100779); nenv=2; nsteps=180
elif [[ "$mode" == actual_sparse ]]; then
  groups=(sd_sparse_standing sd_sparse_moving); seeds=(26100782 26100784); nenv=4; nsteps=650
elif [[ "$mode" == collect ]]; then
  groups=(ss_standing ss_standing sd_standing sd_standing sd_moving sd_moving)
  seeds=(26100780 26100781 26100782 26100783 26100784 26100785)
  nenv=4; nsteps=650
else
  echo 'Usage: run_n0_time_campaign.sh [smoke|collect|evaluate]' >&2; exit 2
fi
for index in "${!groups[@]}"; do
  group="${groups[$index]}"; seed="${seeds[$index]}"
  scenario=sd; vx=0; yaw=0; skip_probability=0
  if [[ "$group" == ss_standing ]]; then scenario=ss; fi
  if [[ "$group" == sd_moving || "$group" == sd_sparse_moving ]]; then vx=0.3; yaw=0.1; fi
  if [[ "$group" == sd_sparse* ]]; then skip_probability=0.08; fi
  output="$N0_OUTPUT/${group}_${seed}"
  if [[ -f "$output/COLLECTION_COMPLETE" ]]; then
    python source/standalone/test11/velocity_error_validation/tools/validate_n0_capture.py --folder "$output" --log "$N0_OUTPUT/${group}_${seed}.log" --envs "$nenv" --steps "$nsteps"
    continue
  fi
  if [[ -d "$output" ]]; then echo "Unfinished output exists: $output" >&2; exit 3; fi
  python source/standalone/test11/collect_vision_data.py \
    --headless --device cuda:0 --num_envs "$nenv" --num_steps "$nsteps" \
    --scenario "$scenario" --split test --seed "$seed" --trajectory_seed "$seed" \
    --validation_condition D --pair_id "n0_${seed}" \
    --collection_run_id "n0_${group}_${seed}" \
    --sensor_time_contract physical_capture_v1 --degradation_profile clean \
    --episode_warmup_s 0 \
    --target_boundary_mode legacy_projection \
    --capture_skip_probability "$skip_probability" \
    --save_episode_stream --keep_measurement_discontinuities \
    --target_samples 0 --max_windows_per_episode 1 --progress_every 100 \
    --locomotion_vx_mps "$vx" --locomotion_yaw_rate_radps "$yaw" \
    --locomotion_policy logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt \
    --ostrack_repo source/standalone/test_lk_ostrack/ostrack/OSTrack-main \
    --ostrack_checkpoint source/standalone/test_lk_ostrack/ostrack/OSTrack-main/output/checkpoints/OSTrack_ep0300.pth.tar \
    --out_dir "$output" > "$N0_OUTPUT/${group}_${seed}.log" 2>&1
  python source/standalone/test11/velocity_error_validation/tools/validate_n0_capture.py --folder "$output" --log "$N0_OUTPUT/${group}_${seed}.log" --envs "$nenv" --steps "$nsteps"
  touch "$output/COLLECTION_COMPLETE"
  echo "Completed $group seed=$seed"
done
