#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/profile.yaml"

export PROFILE_LAUNCHER="${PROFILE_LAUNCHER:-torchrun --nnodes 1 --nproc_per_node 8}"
export PROFILE_TRAINER="${PROFILE_TRAINER:-train_dist.py}"

python3 "${SCRIPT_DIR}/../profile.py" "${CONFIG_PATH}" \
  profiler.profile_type=memory \
  profiler.profile_mode=static \
  profiler.profile_batch_size=8 \
  profiler.profile_seq_length_list=1024 \
  profiler.layernum_min=1 \
  profiler.layernum_max=2 \
  profiler.max_tp_deg=8 \
  profiler.profile_dp_type=zero3 \
  profiler.profile_unit=all

