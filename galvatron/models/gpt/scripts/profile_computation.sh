#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/profile.yaml"

export PROFILE_LAUNCHER="${PROFILE_LAUNCHER:-torchrun --nnodes 1 --nproc_per_node 1}"
export PROFILE_TRAINER="${PROFILE_TRAINER:-train_dist.py}"

python3 "${SCRIPT_DIR}/../profile.py" "${CONFIG_PATH}" \
  profiler.profile_type=computation \
  profiler.profile_mode=batch \
  profiler.profile_batch_size=1 \
  profiler.profile_min_batch_size=1 \
  profiler.profile_max_batch_size=8 \
  profiler.profile_batch_size_step=1 \
  profiler.layernum_min=2 \
  profiler.layernum_max=4 \
  profiler.profile_unit=all

