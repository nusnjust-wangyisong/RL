#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/experiment.yaml}
TIMESTEPS=${TIMESTEPS:-220000}

algorithms=(
  DDPG
  TD3
  SAC
  TQC
  SAC_HER
  TD3_HER
  TQC_HER
  TD3_HER_CURRICULUM
)

for algo in "${algorithms[@]}"; do
  python -m rl_reach.train --config "$CONFIG" --algo "$algo" --fixed-goal --timesteps "$TIMESTEPS"
  python -m rl_reach.train --config "$CONFIG" --algo "$algo" --random-goal --timesteps "$TIMESTEPS"
done
