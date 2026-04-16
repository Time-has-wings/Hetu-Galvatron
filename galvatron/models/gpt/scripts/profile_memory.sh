set -x
set -o pipefail

export NUM_NODES=${NUM_NODES:-1}
export NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=${RANK:-0}

log_dir="logs/profile_memory"
mkdir -p $log_dir

export RUNTIME_LAUNCHER="torchrun --nnodes ${NUM_NODES} --nproc_per_node ${NUM_GPUS_PER_NODE} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --node_rank ${NODE_RANK} train_dist.py "
python3 profiler.py scripts/profile_memory.yaml 2>&1 | tee $log_dir/profile_memory.log
