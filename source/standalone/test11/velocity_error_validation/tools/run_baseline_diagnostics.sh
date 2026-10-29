#!/usr/bin/env bash
set -eo pipefail
source /home/jby/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_232
source /home/jby/IsaacLab-2.3.2/_isaac_sim/setup_conda_env.sh
cd /home/jby/IsaacLab-2.3.2
for dataset in pilot_fixed_camera_v2_warmup_flowgate formal_fixed_camera_3090_20260929; do
  python source/standalone/test11/velocity_error_validation/tools/analyze_observation_baselines.py --data "source/standalone/test11/velocity_error_validation/outputs/$dataset" --out "source/standalone/test11/velocity_error_validation/reports/${dataset}_e2_cuda_20261007" --device cuda:0
 done
