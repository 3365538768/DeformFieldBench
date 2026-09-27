#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="${ENV_NAME:-deformfieldbench}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Create the environment from environment.yml first." >&2
  return 1 2>/dev/null || exit 1
fi

CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${ENV_PATH:-$ENV_NAME}"
source "$SCRIPT_DIR/setup_env.sh"

# Optional MetaX compatibility. Standard CUDA installations do not need this.
if [[ -n "${MACA_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"
  export MACA_CLANG_PATH="$MACA_PATH/mxgpu_llvm/bin"
  export CUDA_PATH="$MACA_PATH/tools/cu-bridge"
  export CUCC_PATH="$MACA_PATH/tools/cu-bridge"
  export PATH="$CUCC_PATH/tools:$CUCC_PATH/bin:$PATH"
fi

export PROJECT_ROOT="${PROJECT_ROOT:-$RELEASE_DIR}"
