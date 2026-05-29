#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/experiment.yaml}
PY=${PY:-/home/kaixin/anaconda3/envs/rl_reach/bin/python}
BASE_STEPS=${BASE_STEPS:-60000}
FULL_STEPS=${FULL_STEPS:-100000}
MAX_JOBS=${MAX_JOBS:-4}

mkdir -p runs/logs

tasks=(
  "DDPG fixed $BASE_STEPS"
  "TD3 fixed $BASE_STEPS"
  "SAC fixed $BASE_STEPS"
  "TQC fixed $BASE_STEPS"
  "SAC_HER fixed $BASE_STEPS"
  "TD3_HER fixed $BASE_STEPS"
  "TQC_HER fixed $BASE_STEPS"
  "TD3_HER_CURRICULUM fixed $FULL_STEPS"
  "DDPG random $BASE_STEPS"
  "TD3 random $BASE_STEPS"
  "SAC random $BASE_STEPS"
  "TQC random $BASE_STEPS"
  "SAC_HER random $BASE_STEPS"
  "TD3_HER random $BASE_STEPS"
  "TQC_HER random $BASE_STEPS"
  "TD3_HER_CURRICULUM random $FULL_STEPS"
)

running=0
idx=0
for task in "${tasks[@]}"; do
  read -r algo mode steps <<< "$task"
  model="runs/models/${algo}_${mode}.zip"
  if [[ -f "$model" ]]; then
    echo "Skip existing $model"
    continue
  fi
  gpu=$((idx % MAX_JOBS))
  idx=$((idx + 1))
  log="runs/logs/train_${algo}_${mode}.log"
  flag="--${mode}-goal"
  echo "Start $algo $mode on GPU $gpu for $steps steps"
  (
    export PYTHONNOUSERSITE=1
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PY" -m rl_reach.train \
      --config "$CONFIG" \
      --algo "$algo" \
      "$flag" \
      --timesteps "$steps" \
      --run-name "${algo}_${mode}" \
      > "$log" 2>&1
  ) &
  running=$((running + 1))
  if (( running >= MAX_JOBS )); then
    wait -n
    running=$((running - 1))
  fi
done
wait
