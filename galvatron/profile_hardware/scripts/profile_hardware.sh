NUM_NODES=1
NUM_GPUS_PER_NODE=8
ADDR='$MASTER_ADDR'
PORT='$MASTER_PORT'
NODE_RANK='$RANK'
ENVS="NCCL_DEBUG=WARN NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_2,mlx5_5"

PROFILE_ARGS="
    --num_nodes ${NUM_NODES} \
    --num_gpus_per_node ${NUM_GPUS_PER_NODE} \
    --max_pp_deg 16 \
    --master_addr ${ADDR} \
    --master_port ${PORT} \
    --envs ${ENVS} \
    --node_rank ${NODE_RANK} \
    --overlap_time_multiply 4"
python3 profile_hardware.py ${PROFILE_ARGS}