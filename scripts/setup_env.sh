#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PHYS_RELEASE_DIR="${PHYS_RELEASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PROJECT_ROOT="${PROJECT_ROOT:-$PHYS_RELEASE_DIR}"

GS_ROOT="$PHYS_RELEASE_DIR/simulation/gaussian-splatting"
GS_RASTERIZER="$GS_ROOT/submodules/diff-gaussian-rasterization"
GS_KNN="$GS_ROOT/submodules/simple-knn"
export PYTHONPATH="$PHYS_RELEASE_DIR:$PHYS_RELEASE_DIR/simulation:$GS_ROOT:$GS_RASTERIZER:$GS_KNN${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TORCH_HOME="${TORCH_HOME:-$PHYS_RELEASE_DIR/.torch}"
