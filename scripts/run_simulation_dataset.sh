#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/setup_env.sh"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to a directory containing .ply files}"
MATERIAL_CONFIG="${MATERIAL_CONFIG:-$PHYS_RELEASE_DIR/configs/simulation/material_space_example.json}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$PHYS_RELEASE_DIR/configs/simulation/train_config_dataset_full.json}"

for var_name in MODEL_PATH MATERIAL_CONFIG TRAIN_CONFIG; do
  value="${!var_name}"
  if [[ "$value" != /* ]]; then
    printf -v "$var_name" '%s/%s' "$PHYS_RELEASE_DIR" "$value"
  fi
done

cd "$PHYS_RELEASE_DIR"
python simulation/auto_simulation_runner.py \
  --model_path "$MODEL_PATH" \
  --material_space_config "$MATERIAL_CONFIG" \
  --train_config "$TRAIN_CONFIG" \
  --num_gpus "${NUM_GPUS:-1}" \
  "$@"
