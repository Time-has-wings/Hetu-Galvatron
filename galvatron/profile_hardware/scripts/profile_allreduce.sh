NCCL_DEBUG=WARN
NCCL_IB_DISABLE=0
NCCL_IB_HCA=mlx5_2,mlx5_5
export NUM_NODES=1
export NUM_GPUS_PER_NODE=8
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NODE_RANK=$RANK
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 8 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 8 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
sleep 1
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 4 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 4 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 4 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 4 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
sleep 1
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 2 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 2 --global_tp_consec 1 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
echo "Running: python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 2 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
"
python -m torch.distributed.launch --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --node_rank=$NODE_RANK profile_allreduce.py --global_tp_deg 2 --global_tp_consec 0 --pp_deg 1 --nproc_per_node=$NUM_GPUS_PER_NODE 
sleep 1
