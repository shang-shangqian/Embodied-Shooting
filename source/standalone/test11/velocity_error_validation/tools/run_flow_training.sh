#!/usr/bin/env bash
set -eo pipefail
source /home/jby/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab_232
source /home/jby/IsaacLab-2.3.2/_isaac_sim/setup_conda_env.sh
cd /home/jby/IsaacLab-2.3.2
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "${1:-train}" == development ]]; then
  python source/standalone/test11/velocity_error_validation/tools/prepare_development_inputs.py
  exit 0
fi
if [[ "${1:-train}" == smoke ]]; then
  python source/standalone/test11/velocity_error_validation/tools/smoke_flow_training.py
  exit 0
fi
if [[ "${1:-train}" == trainval ]]; then
  python source/standalone/test11/velocity_error_validation/tools/train_flow_ablations.py --train-val-only --data source/standalone/test11/velocity_error_validation/outputs/optimization_20261007 --out source/standalone/test11/velocity_error_validation/reports/optimization_flow_runs_20261007 --device cuda:0 --epochs 24 --batch-size 64 --seeds 41 42 43 > source/standalone/test11/velocity_error_validation/reports/optimization_flow_runs_20261007/training_campaign.log 2>&1
  exit 0
fi
python source/standalone/test11/velocity_error_validation/tools/train_flow_ablations.py --data source/standalone/test11/velocity_error_validation/outputs/optimization_20261007 --out source/standalone/test11/velocity_error_validation/reports/optimization_flow_runs_20261007 --device cuda:0 --epochs 24 --batch-size 64 --seeds 41 42 43
python source/standalone/test11/velocity_error_validation/tools/evaluate_optimization_campaign.py --runs source/standalone/test11/velocity_error_validation/reports/optimization_flow_runs_20261007 --device cuda:0

python source/standalone/test11/velocity_error_validation/tools/benchmark_flow_candidates.py --runs source/standalone/test11/velocity_error_validation/reports/optimization_flow_runs_20261007
