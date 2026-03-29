import os
import random
import argparse

import numpy as np
import torch
import torch.distributed as dist

from galvatron.utils import read_json_config, write_json_config
from galvatron.core.runtime.comm_groups import build_rank_to_parallel_coords, get_groups


def single_all_to_all(input:torch.Tensor, group:torch.distributed.ProcessGroup):
    seq_world_size = dist.get_world_size(group)
    input_t = input.reshape(seq_world_size, -1)
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    return output

def set_seed(rank):
    seed = 123 + rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args):
    torch.distributed.init_process_group(backend="nccl")

    # [Step 0] preparation
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    local_rank = int(os.environ['LOCAL_RANK'])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])  # Env vars set by torchrun
    nproc_per_node = local_world_size
    nnodes = world_size // nproc_per_node

    set_seed(rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)


    # [Step 1] build sp group
    name2size = {
        'pp': 1,
        'dp': world_size // args.global_tp_deg,
        'cp': 1,
        'tsp': args.global_tp_deg,
    }
    order = 'pp-dp-cp-tsp'
    degree_rank_dict = build_rank_to_parallel_coords(world_size, name2size, order=order)
    sp_group, _ = get_groups(degree_rank_dict, ignore_keys=['tsp'])


    # [Step 2] calculate info
    bf16_size = 2 # we use bf16 to profile
    all2all_message_size_per_layer = int(args.local_batch_size * args.seq_length * args.hidden_size * bf16_size // 1024 // 1024) # in MB
    if rank == 0:
        print(f'local_batch_size: {args.local_batch_size}, seq_length: {args.seq_length}, hidden_size: {args.hidden_size}')
        print(f'[all2all_message_size]: per_layer {all2all_message_size_per_layer} MB') # local_batch_size * 512 * 1024 * 2 // 1024 // 1024 = local_batch_size MB


    # [Step 3] warmup
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_list = []
    for _ in range(5):
        input = np.random.rand(*(args.local_batch_size, args.seq_length, args.hidden_size))
        input = torch.tensor(input, dtype=torch.bfloat16, device=device) # use bf16 to profile
        output = single_all_to_all(input, sp_group.group)
    

    # [Step 4] run profile
    # torch.cuda.cudart().cudaProfilerStart()
    for _ in range(20):
        input = np.random.rand(*(args.local_batch_size, args.seq_length, args.hidden_size))
        input = torch.tensor(input, dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=sp_group.group)

        start.record()
        output = single_all_to_all(input, sp_group.group)
        end.record()

        torch.cuda.synchronize()
        print(f"device: {local_rank}, time: {start.elapsed_time(end)} ms", flush=True) # ms
        time_list.append(start.elapsed_time(end))

    # torch.cuda.cudart().cudaProfilerStop()
    

    # [Step 5] process result
    per_comm_time = sum(time_list) / len(time_list)
    per_comm_time = torch.tensor([per_comm_time]).to(device)
    torch.distributed.all_reduce(per_comm_time, group=sp_group.group, op=torch.distributed.ReduceOp.SUM)
    per_comm_time = per_comm_time.cpu().numpy()[0] / sp_group.size
    if rank == 0:
        print(sum(time_list), len(time_list))
        print('**********')
        print(f'comm_time_{args.local_batch_size}MB_{args.global_tp_deg}: {per_comm_time} ms')
        print('**********')
        path = os.path.dirname(os.path.abspath(__file__))
        env_config_path = os.path.join(path, f'./hardware_configs/sp_time_{nnodes}nodes_{nproc_per_node}gpus_per_node.json')
        config = read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
        key = f'all2all_size_{args.global_tp_deg}_{args.local_batch_size}MB_time'
        config[key] = per_comm_time
        write_json_config(config, env_config_path)
        print('Already written all2all time into env config file %s!'%(env_config_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8,16,32,64,128,256], required=True)
    parser.add_argument("--local_batch_size", type=int, default=32, help="local training batch size", required=True)
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")

    args = parser.parse_args()
    train(args)