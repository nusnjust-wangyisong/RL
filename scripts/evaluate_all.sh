#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/experiment.yaml}
EPISODES=${EPISODES:-50}
DISTURBANCE=${DISTURBANCE:-medium}

python -m rl_reach.run_suite \
  --config "$CONFIG" \
  --model-dir runs/models \
  --episodes "$EPISODES" \
  --disturbance "$DISTURBANCE"
