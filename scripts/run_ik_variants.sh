#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/experiment_irb120.yaml}
PYTHON=${PYTHON:-python}
TIMESTEPS=${TIMESTEPS:-80000}
EPISODES=${EPISODES:-30}
DISTURBANCE=${DISTURBANCE:-medium}
MODEL_DIR=${MODEL_DIR:-runs/irb120/models}

# Variant 2: train and evaluate IK base action plus RL residual.
for task in fixed random; do
  task_flag="--${task}-goal"
  "$PYTHON" -m rl_reach.train \
    --config "$CONFIG" \
    --algo TD3_HER_CURRICULUM \
    "$task_flag" \
    --timesteps "$TIMESTEPS" \
    --precision-servo-mode ik_residual \
    --train-precision-servo \
    --run-name "TD3_HER_IK_RESIDUAL_${task}"
done

# Variant 3: train and evaluate with IK delta appended to the observation.
for task in fixed random; do
  task_flag="--${task}-goal"
  "$PYTHON" -m rl_reach.train \
    --config "$CONFIG" \
    --algo TD3_HER_CURRICULUM \
    "$task_flag" \
    --timesteps "$TIMESTEPS" \
    --include-ik-observation \
    --run-name "TD3_HER_IK_OBS_${task}"
done

"$PYTHON" -m rl_reach.compare_ik_variants \
  --config "$CONFIG" \
  --model-dir "$MODEL_DIR" \
  --episodes "$EPISODES" \
  --disturbance "$DISTURBANCE" \
  --output "ik_variant_summary_${DISTURBANCE}.csv"
