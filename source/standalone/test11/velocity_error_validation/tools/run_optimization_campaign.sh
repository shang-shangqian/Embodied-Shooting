#!/usr/bin/env bash
set -eo pipefail
source /home/jby/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_232
source /home/jby/IsaacLab-2.3.2/_isaac_sim/setup_conda_env.sh
set -u
cd /home/jby/IsaacLab-2.3.2
export CUDA_VISIBLE_DEVICES=0
CAMPAIGN_DIR=source/standalone/test11/velocity_error_validation/outputs/optimization_20261007
mkdir -p "$CAMPAIGN_DIR"
case "${1:-collect}" in
  smoke) splits=(smoke);;
  collect) splits=(train val test);;
  *) echo 'Usage: run_optimization_campaign.sh [smoke|collect]' >&2; exit 2;;
esac
for split in "${splits[@]}"; do
  case "$split" in
    smoke) seed_base=26100700; nenv=2; nsteps=240; actual_split=train;;
    train) seed_base=26100710; nenv=8; nsteps=1500; actual_split=train;;
    val) seed_base=26100720; nenv=8; nsteps=1000; actual_split=val;;
    test) seed_base=26100730; nenv=8; nsteps=1000; actual_split=test;;
  esac
  offset=0
  for scenario in ss sd; do
    for profile in clean mixed; do
      output="$CAMPAIGN_DIR/$split/${scenario}_${profile}"
      log="$CAMPAIGN_DIR/${split}_${scenario}_${profile}.log"
      if [[ -f "$output/COLLECTION_COMPLETE" ]]; then offset=$((offset+1)); continue; fi
      if [[ -d "$output" ]]; then echo "Unfinished output exists: $output" >&2; exit 3; fi
      seed=$((seed_base+offset)); offset=$((offset+1))
      python source/standalone/test11/collect_vision_data.py \
        --headless --device cuda:0 --num_envs "$nenv" --num_steps "$nsteps" \
        --scenario "$scenario" --split "$actual_split" --seed "$seed" \
        --validation_condition D --pair_id "optimization_${seed}" \
        --collection_run_id "optimization_${split}_${scenario}_${profile}_${seed}" \
        --degradation_profile "$profile" --severity random \
        --save_episode_stream --keep_measurement_discontinuities \
        --target_samples 0 --max_windows_per_episode 4 --progress_every 100 \
        --locomotion_policy logs/rsl_rl/deeprobotics_lite3_rtx_flat/2026-07-29_20-44-55_test7_2_weight_flat/exported/policy.pt \
        --ostrack_repo source/standalone/test_lk_ostrack/ostrack/OSTrack-main \
        --ostrack_checkpoint source/standalone/test_lk_ostrack/ostrack/OSTrack-main/output/checkpoints/OSTrack_ep0300.pth.tar \
        --out_dir "$output" > "$log" 2>&1
      touch "$output/COLLECTION_COMPLETE"
      echo "Completed $split $scenario $profile: $output"
      if [[ "$split" == smoke ]]; then exit 0; fi
    done
  done
done
