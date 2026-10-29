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
N6_REPORT=source/standalone/test11/velocity_error_validation/reports/n6_interface_20261007
mkdir -p "$N6_REPORT"
mode="${1:-evaluate}"
case "$mode" in
  tests)
    python -m pytest source/standalone/test11/velocity_error_validation/tests source/standalone/test11/speed_prediction/tests -q -rs | tee "$N6_REPORT/tests.log"
    ;;
  smoke)
    python source/standalone/test11/velocity_error_validation/tools/evaluate_n6_interface.py --device cuda:0 --smoke --out source/standalone/test11/velocity_error_validation/reports/n6_interface_smoke_20261007
    ;;
  evaluate)
    python source/standalone/test11/velocity_error_validation/tools/evaluate_n6_interface.py --device cuda:0 --out "$N6_REPORT" | tee "$N6_REPORT/evaluation.log"
    ;;
  report)
    python source/standalone/test11/velocity_error_validation/tools/write_n6_interface_report.py
    ;;
  *) echo 'Usage: run_n6_interface_campaign.sh [tests|smoke|evaluate|report]' >&2; exit 2 ;;
esac
