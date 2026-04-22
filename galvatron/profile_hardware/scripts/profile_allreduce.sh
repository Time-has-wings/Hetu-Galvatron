NCCL_DEBUG=WARN
NCCL_IB_DISABLE=0
NCCL_IB_HCA=mlx5_2,mlx5_5
export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$RANK
# Bandwidth sweep = legacy: while tp halves; each tp runs consec 1 then 0; skip tp==world_size with consec 0. Implemented in profile_allreduce.bandwidth_jobs_from_tp_degrees.
# Omit --local_batch_size here: profile_allreduce.py defaults to 32 (still used for message size).
mkdir -p logs/allreduce
echo "Running: torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    profile_allreduce.py \
    --global_tp_deg 8 4 2 \
    --profile_time 0 \
    2>&1 | tee logs/allreduce/allreduce_bandwidth_tp_time0.log
"
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    profile_allreduce.py \
    --global_tp_deg 8 4 2 \
    --profile_time 0 \
    2>&1 | tee logs/allreduce/allreduce_bandwidth_tp_time0.log
