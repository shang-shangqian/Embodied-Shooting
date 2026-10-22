#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash source/standalone/integrated_shooting_chain/train_full_chain.sh LOCOMOTION_POLICY [training options]" >&2
    exit 2
fi

locomotion_policy="$1"
shift

if [[ -f "_isaac_sim/setup_conda_env.sh" ]]; then
    # Keep the active Conda Python, while adding Isaac Sim's extension/library
    # paths. This mirrors the repository's documented shell setup.
    set +u
    source "_isaac_sim/setup_conda_env.sh"
    set -u
fi

python_command="${PYTHON_BIN:-python}"

exec "$python_command" source/standalone/test7_2_weight/train_turret_mixed_gt.py \
    --training_mode full_chain \
    --headless \
    --num_envs 16 \
    --locomotion_policy "$locomotion_policy" \
    "$@"
