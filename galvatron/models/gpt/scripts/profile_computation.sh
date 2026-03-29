export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$RANK


export PROFILE_LAUNCHER="${PROFILE_LAUNCHER:-torchrun --nnodes ${NUM_NODES} --nproc_per_node ${NUM_GPUS_PER_NODE}}"
export PROFILE_TRAINER="${PROFILE_TRAINER:-train_dist.py scripts/train_dist.yaml}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${GPT_DIR}"

python3 profiler.py scripts/profile.yaml \
  profiler.profile_type=computation \
  profiler.profile_mode=batch \
  profiler.profile_batch_size=1 \
  profiler.profile_min_batch_size=1 \
  profiler.profile_max_batch_size=8 \
  profiler.profile_batch_size_step=1 \
  profiler.layernum_min=2 \
  profiler.layernum_max=4 \
  profiler.profile_unit=all

