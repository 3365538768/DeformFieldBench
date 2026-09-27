#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_env.sh"
cd "$PHYS_RELEASE_DIR"
python -m vlm_benchmark.run_vlm_benchmark \
  --physgaussian_root "$PHYS_RELEASE_DIR" \
  --vlm_tag "${VLM_TAG:-qwen3.5-plus}" \
  --output_tag "${OUTPUT_TAG:-qwen3.5-plus}" \
  "$@"
