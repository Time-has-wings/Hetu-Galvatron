NCCL_DEBUG=WARN
NCCL_IB_DISABLE=0
NCCL_IB_HCA=mlx5_2,mlx5_5
export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$RANK
mkdir -p logs/allreduce_sp
echo "Running: torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    profile_allreduce.py \
    --global_tp_deg 8 4 2 \
    --local_batch_size 1024 512 256 128 64 32 16 8 4 2 1 \
    --profile_time 1 \
    2>&1 | tee logs/allreduce_sp/allreduce_sp_time1.log
"
torchrun \
    --nnodes=$NUM_NODES \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    profile_allreduce.py \
    --global_tp_deg 8 4 2 \
    --local_batch_size 1024 512 256 128 64 32 16 8 4 2 1 \
    --profile_time 1 \
    2>&1 | tee logs/allreduce_sp/allreduce_sp_time1.log
