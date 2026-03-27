import argparse
import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from galvatron.utils import read_json_config, write_json_config
from galvatron.core.runtime.comm_groups import CommGroup, build_rank_to_parallel_coords, get_groups

class pre_sync_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, hidden_states:torch.Tensor):
        return hidden_states

class pre_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 1024)

    def forward(self, hidden_states:torch.Tensor):
        hidden_states = self.linear(hidden_states)
        return hidden_states

def _reduce(input_:torch.Tensor, group:torch.distributed.ProcessGroup):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_.contiguous(), group=group)

    return input_

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""
    
    @staticmethod
    def forward(ctx, input_:torch.Tensor, group:torch.distributed.ProcessGroup):
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        return grad_output, None

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    
    @staticmethod
    def forward(ctx, input_:torch.Tensor, group:torch.distributed.ProcessGroup):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output:torch.Tensor):
        return _reduce(grad_output, ctx.group), None

def reduce_from_tensor_model_parallel_region_group(input_:torch.Tensor, group:torch.distributed.ProcessGroup):
    return _ReduceFromModelParallelRegion.apply(input_, group)

def copy_to_tensor_model_parallel_region_group(input_:torch.Tensor, group:torch.distributed.ProcessGroup):
    return _CopyToModelParallelRegion.apply(input_, group)

class allreduce_block(nn.Module):
    def __init__(self, tp_group:CommGroup):
        super().__init__()
        self.tp_group = tp_group
        self.linear = nn.Linear(1024, 1024)

    def forward(self, hidden_states:torch.Tensor):
        hidden_states = copy_to_tensor_model_parallel_region_group(hidden_states, self.tp_group.group) # forward: nothing, backward: allreduce once
        hidden_states = reduce_from_tensor_model_parallel_region_group(hidden_states, self.tp_group.group) # forward: allreduce once, backward: nothing
        hidden_states = hidden_states.requires_grad_(True)
        return hidden_states

class DataLoaderRandom(Dataset):
    def __init__(self, dataset_size, profile_time, seq_length=512, hidden_size=1024):
        self.dataset_size = dataset_size
        self.input = np.random.rand(*(self.dataset_size, seq_length, hidden_size))
        self.profile_time = profile_time

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        if self.profile_time == 1:
            input = torch.tensor(self.input[idx], dtype=torch.bfloat16) # 512 * 1024 * 2 = 1MB
        else:
            input = torch.tensor(self.input[idx], dtype=torch.float32) # 512 * 1024 * 4 = 2MB
        return input

def fake_loss_func(output:torch.Tensor):
    loss = output.sum()
    loss = loss.requires_grad_(True)
    return loss

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

    # Initialize torch.profiler before initializing CUDA context to avoid the bug
    # Refer to https://github.com/pytorch/pytorch/issues/60158
    with torch.profiler.profile() as p:
        pass
    
    # [Step 1] build tp group
    name2size = {
        'pp': 1,
        'dp': world_size // args.global_tp_deg,
        'cp': 1,
        'tsp': args.global_tp_deg,
    }
    if args.global_tp_consec == 1:
        order = 'pp-dp-cp-tsp'
    else:
        order = 'pp-tsp-dp-cp'
    degree_rank_dict = build_rank_to_parallel_coords(world_size, name2size, order=order)
    tp_group, _ = get_groups(degree_rank_dict, ignore_keys=['tsp'])

    # [Step 2] build model
    model = nn.Sequential()
    model.add_module('pre_sync_module', pre_sync_module())
    model.add_module('pre_mlp', pre_mlp())
    for i in range(args.num_layers):
        module = allreduce_block(tp_group=tp_group)
        model.add_module(f'mlp_{i}', module)

    if args.profile_time == 1:
        model = model.bfloat16()
    model = model.to(device)
    
    # [Step 3] build dataset and dataloader
    warmup_num, active_num = 1, 10
    dataset = DataLoaderRandom(args.local_batch_size * (warmup_num + active_num), args.profile_time, args.seq_length, args.hidden_size)
    trainloader = DataLoader(dataset=dataset, batch_size=args.local_batch_size)

    # [Step 4] calculate info
    """
        When profile_time is 0, we decompose the AllReduce operation and use the sum of the communication volumes of AllGather 
        and ReduceScatter as the communication volume of AllReduce. 
        The data volume transmitted per millisecond (MB/ms) is characterized by dividing the communication volume by time. 
        Therefore, when applying this value in the Cost Model, AllReduce communication needs to be decomposed.
        
        When profile_time is 1, we no longer decompose the AllReduce operation or look into its internal communication details. 
        Instead, we directly record the time required for AllReduce to process a specific data size and use fitting to derive the model.
    """
    tp_size = args.global_tp_deg
    if args.profile_time == 0:
        allreduce_numbers_per_layer = 2
        fp32_size = 4 # when args.profile_time == 0, we use fp32 to profile
        activation_size = args.local_batch_size * args.seq_length * args.hidden_size * fp32_size / 1024 / 1024
        
        # 2 means one allgather and one reduce_scatter.
        # tp_size - 1 means allgather tp_size - 1 chunks.
        # activation / tp_size means the size of each chunk.
        allreduce_message_size_per_layer = allreduce_numbers_per_layer * 2 * (tp_size - 1) * (activation_size / tp_size)
        allreduce_message_size_total = allreduce_message_size_per_layer * args.num_layers

        if rank == 0:
            print(f'Strategy: tp_size{tp_size}, consecutive{args.global_tp_consec}')
            print(f'[allreduce_message_size]: per_layer {allreduce_message_size_per_layer} MB, total {allreduce_message_size_total} MB')
    else:
        bf16_size = 2 # when args.profile_time == 1, we use bf16 to profile
        activation_size = int(args.local_batch_size * args.seq_length * args.hidden_size * bf16_size // 1024 // 1024) # 1 * 512 * 1024 * 2 = 1MB

    # [Step 5] build trace handler
    def trace_handler(prof):
        try:
            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5)
            if rank == 0:
                print(table)
            table = table.split('\n')
            def split_line(line):
                line = line.split('  ')
                ls = []
                for s in line:
                    if len(s):
                        ls.append(s.strip())
                return ls
            def str2time(s):
                if 'ms' in s:
                    return float(s[:-2])
                elif 'us' in s:
                    return float(s[:-2]) * 1e-3
                else:
                    return float(s[:-1]) * 1e3
            for line in table:
                if 'Name' in line:
                    title = split_line(line)
                if 'AllReduce' in line: # ncclKernel_AllReduce or ncclDevKernel_AllReduce
                    result = split_line(line)
            for i in range(len(title)):
                # print('%s: %s'%(title[i],result[i]))
                if 'CUDA total' in title[i]:
                    cuda_total_idx = i
                if "Calls" in title[i]:
                    comm_num = int(result[i])
            comm_time = str2time(result[cuda_total_idx])
            
            if args.profile_time == 0:
                allreduce_time_single_step = comm_time / active_num # unit: ms
                comm_coe = allreduce_message_size_total / allreduce_time_single_step # unit: MB/ms
                comm_coe = torch.tensor([comm_coe]).to(device)
                torch.distributed.all_reduce(comm_coe, group=tp_group.group, op=torch.distributed.ReduceOp.SUM)
                comm_coe = comm_coe.cpu().numpy()[0] / tp_group.size # unit: MB/ms
                if rank == 0:
                    print('**********')
                    print(f'comm_coe_{args.global_tp_deg}_{args.global_tp_consec}: {comm_coe} MB/ms')
                    print('**********')
                    path = os.path.dirname(os.path.abspath(__file__))
                    env_config_path = os.path.join(path, f'./hardware_configs/allreduce_bandwidth_{nnodes}nodes_{nproc_per_node}gpus_per_node.json')
                    config = read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
                    key = f'allreduce_size_{args.global_tp_deg}_consec_{args.global_tp_consec}'
                    config[key] = comm_coe # unit: MB/ms
                    write_json_config(config, env_config_path)
                    print(f'Already written allreduce bandwidth into env config file {env_config_path}!')
            else:
                per_comm_time = comm_time / comm_num # unit: ms
                per_comm_time = torch.tensor([per_comm_time]).to(device)
                torch.distributed.all_reduce(per_comm_time, group=tp_group.group, op=torch.distributed.ReduceOp.SUM)
                per_comm_time = per_comm_time.cpu().numpy()[0] / tp_group.size # unit: ms
                if rank == 0:
                    print('**********')
                    print(f'comm_time_{activation_size}MB_{args.global_tp_deg}_{args.global_tp_consec}: {per_comm_time} ms')
                    print('**********')
                    path = os.path.dirname(os.path.abspath(__file__))
                    env_config_path = os.path.join(path, f'./hardware_configs/allreduce_bandwidth_{nnodes}nodes_{nproc_per_node}gpus_per_node.json')
                    config = read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
                    key = f'allreduce_size_{args.global_tp_deg}_{activation_size}MB_time'
                    config[key] = per_comm_time # unit: ms
                    write_json_config(config, env_config_path)
                    print(f'Already written allreduce bandwidth into env config file {env_config_path}!')
        except Exception as e:
            print(f"Profiler error: {e}")
            return

    # [Step 6] run
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0,warmup=warmup_num,active=active_num),
        on_trace_ready=trace_handler
    ) as p:
        for i, input in enumerate(tqdm(trainloader)):
            input = input.to(device)
            output = model(input)
            loss = fake_loss_func(output)
            loss.backward()
            p.step()
        
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8,16,32,64,128,256], required=True)
    parser.add_argument("--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag.", choices=[0,1], required=True)
    parser.add_argument("--profile_time", type=int, default=0, help="Profile time", required=True)
    parser.add_argument("--local_batch_size", type=int, default=32, help="local training batch size")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=24, help="Number of layers")

    args = parser.parse_args()
    train(args)