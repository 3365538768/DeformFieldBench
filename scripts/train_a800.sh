#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_HOME="${TORCH_HOME:-$PWD/weights/torch_cache}"
CONFIG="${1:?Usage: bash scripts/train_a800.sh configs/your_experiment.json}"
GPUS="${GPUS:-8}"
exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" \
    -m fieldbridge.train --config "$CONFIG"
