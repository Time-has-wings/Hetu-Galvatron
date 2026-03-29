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
  profiler.profile_type=memory \
  profiler.profile_mode=static \
  profiler.profile_batch_size=8 \
  profiler.profile_seq_length_list=1024 \
  profiler.layernum_min=1 \
  profiler.layernum_max=2 \
  profiler.max_tp_deg=8 \
  profiler.profile_dp_type=zero3 \
  profiler.profile_unit=all

