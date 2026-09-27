#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/setup_env.sh"

SIM_TYPE="${SIM_TYPE:-press}"
case "$SIM_TYPE" in
  press|drop|shear|stretch|bend) ;;
  *) echo "SIM_TYPE must be press, drop, shear, stretch, or bend" >&2; exit 2 ;;
esac

PLY_PATH="${PLY_PATH:?set PLY_PATH to a Gaussian-splatting .ply file}"
if [[ "$PLY_PATH" != /* ]]; then
  PLY_PATH="$(cd "$(dirname "$PLY_PATH")" && pwd)/$(basename "$PLY_PATH")"
fi

CONFIG="${CONFIG:-$PHYS_RELEASE_DIR/simulation/config/${SIM_TYPE}_cube_jelly.json}"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$PHYS_RELEASE_DIR/$CONFIG"
fi
OUTPUT_PATH="${OUTPUT_PATH:-$PHYS_RELEASE_DIR/outputs/simulation_sample}"
if [[ "$OUTPUT_PATH" != /* ]]; then
  OUTPUT_PATH="$PHYS_RELEASE_DIR/$OUTPUT_PATH"
fi

cd "$PHYS_RELEASE_DIR/simulation"
python modified_simulation.py \
  --ply_path "$PLY_PATH" \
  --config "$CONFIG" \
  --output_path "$OUTPUT_PATH" \
  --sim_type "$SIM_TYPE" \
  --num_views "${NUM_VIEWS:-3}" \
  --num_render_views "${NUM_RENDER_VIEWS:-3}" \
  --render_outputs_per_sim_second "${RENDER_OUTPUTS_PER_SIM_SECOND:-16}" \
  --render_img \
  --output_view_stress_gaussian \
  --stress_gaussian_single_channel \
  --output_view_flow_gaussian \
  --output_view_force_mask \
  --force_mask_single_channel \
  --output_view_object_mask \
  --output_bc_info \
  --output_force_info \
  --output_initial_force_mask_arrow \
  --pack_sample_pack \
  --pack_sample_pack_include_object_mask \
  --arch4_lmdb_resize "${ARCH4_LMDB_RESIZE:-224}"
