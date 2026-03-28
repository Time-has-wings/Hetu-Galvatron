import os
import random
import time
import argparse
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from galvatron.utils import read_json_config, write_json_config
from galvatron.core.runtime.comm_groups import build_rank_to_parallel_coords, get_groups, CommGroup

class DataLoaderRandom(Dataset):
    def __init__(self, dataset_size, seq_length, hidden_size):
        self.dataset_size = dataset_size
        self.input = np.random.rand(*(self.dataset_size, seq_length, hidden_size))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        input = torch.FloatTensor(self.input[idx]) # fp32
        return input

def set_seed(rank):
    seed = 123 + rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

Shape = Union[List[int], torch.Size]

PP_GROUP:CommGroup = None

def set_global_pp_group(pp_group:CommGroup):
    global PP_GROUP
    assert PP_GROUP is None, f'pp_group is initialized'
    PP_GROUP = pp_group

def get_global_pp_group():
    global PP_GROUP
    assert PP_GROUP is not None, f'pp_group is not initialized'
    return PP_GROUP

def get_pipeline_model_parallel_prev_rank():
    pp_group = get_global_pp_group()
    curr_idx = pp_group.ranks.index(torch.distributed.get_rank())
    assert curr_idx != 0, f'rank {torch.distributed.get_rank()} is the first rank'
    return pp_group.ranks[curr_idx - 1]

def get_pipeline_model_parallel_next_rank():
    pp_group = get_global_pp_group()
    curr_idx = pp_group.ranks.index(torch.distributed.get_rank())
    assert curr_idx != len(pp_group.ranks) - 1, f'rank {torch.distributed.get_rank()} is the last rank'
    return pp_group.ranks[curr_idx + 1]

def _run_p2pops(
    tensor_send_prev: Union[torch.Tensor, None],
    tensor_send_next: Union[torch.Tensor, None],
    tensor_recv_prev: Union[torch.Tensor, None],
    tensor_recv_next: Union[torch.Tensor, None],
):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_prev,
            get_pipeline_model_parallel_prev_rank(),
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_prev,
            get_pipeline_model_parallel_prev_rank(),
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_next,
            get_pipeline_model_parallel_next_rank(),
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_next,
            get_pipeline_model_parallel_next_rank(),
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

def _communicate(
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Optional[Shape] = None,
    dtype_: Optional[torch.dtype] = None,
    *,
    params_dtype: Optional[torch.dtype] = None,
    fp32_residual_connection: bool = False,
) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
    # Create placeholder tensors for receive in forward and backward directions if needed.
    tensor_recv_prev = None
    tensor_recv_next = None
    
    if tensor_shape is None:
        raise RuntimeError("`tensor_shape` must be specified. Common `tensor_shape` is `(seq_length, micro_batch_size, hidden_size)`")

    tensor_chunk_shape = tensor_shape

    dtype = params_dtype or torch.float
    if fp32_residual_connection:
        dtype = torch.float
    requires_grad = True
    if dtype_ is not None:
        dtype = dtype_
        requires_grad = False

    if recv_prev:
        tensor_recv_prev = torch.empty(
            tensor_chunk_shape,
            requires_grad=requires_grad,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )
    if recv_next:
        tensor_recv_next = torch.empty(
            tensor_chunk_shape,
            requires_grad=requires_grad,
            device=torch.cuda.current_device(),
            dtype=dtype,
        )

    _run_p2pops(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next)

    torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next

def send_forward(
    output_tensor: torch.Tensor,
    tensor_shape: Shape = None,
    dtype: Optional[torch.dtype] = None
):
    _communicate(
        tensor_send_next=output_tensor.contiguous(),
        tensor_send_prev=None,
        recv_prev=False,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype_=dtype,
    )

def recv_forward(
    tensor_shape: Shape,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    input_tensor, _ = _communicate(
        tensor_send_next=None,
        tensor_send_prev=None,
        recv_prev=True,
        recv_next=False,
        tensor_shape=tensor_shape,
        dtype_=dtype,
    )
    return input_tensor

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

    # [Step 1] build pp group
    name2size = {
        'pp': args.pp_deg,
        'dp': 1,
        'cp': 1,
        'tsp': world_size // args.pp_deg
    }
    order = 'pp-dp-cp-tsp'
    degree_rank_dict = build_rank_to_parallel_coords(world_size, name2size, order=order)
    pp_group, _ = get_groups(degree_rank_dict, ignore_keys=['pp'])
    tp_group, _ = get_groups(degree_rank_dict, ignore_keys=['tsp'])
    set_global_pp_group(pp_group)

    # [Step 3] build dataset and dataloader
    warmup_num, active_num = 1, 10
    dataset = DataLoaderRandom(args.local_batch_size * (warmup_num + active_num), args.seq_length, args.hidden_size)
    trainloader = DataLoader(dataset=dataset, batch_size=args.local_batch_size)


    # [Step 4] calculate info
    fp32_size = 4
    p2p_message_size = args.local_batch_size * args.seq_length * args.hidden_size * fp32_size / 1024 / 1024
    if local_rank == 0:
        print(f'Strategy: pp_deg = {args.pp_deg} local_batch_size = {args.local_batch_size} seq_length = {args.seq_length} hidden_size = {args.hidden_size}')
        print(f'[p2p_message_size]: total {p2p_message_size} MB')


    # [Step 5] build trace handler
    def trace_handler(prof):
        try:
            if rank == inter_node_send_rank:
                table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5)
                time.sleep(0.5 * pp_group.ranks[0])
                print('Results of p2p from rank %d to rank %d:'%(inter_node_send_rank, inter_node_recv_rank))
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
                        return float(s[:-2])*1e-3
                    else:
                        return float(s[:-1])*1e3
                for line in table:
                    if 'Name' in line:
                        title = split_line(line)
                    if 'SendRecv' in line: # ncclKernel_SendRecv or ncclDevKernel_SendRecv
                        result = split_line(line)
                for i in range(len(title)):
                    # print('%s: %s'%(title[i],result[i]))
                    if 'CUDA total' in title[i]:
                        cuda_total_idx = i

                p2p_time = str2time(result[cuda_total_idx]) / active_num
                comm_coe = p2p_message_size / p2p_time

                comm_coe = torch.tensor([comm_coe]).to(device)
                torch.distributed.all_reduce(comm_coe, group=tp_group.group, op=torch.distributed.ReduceOp.SUM)
                comm_coe = comm_coe.cpu().numpy()[0] / tp_group.size
                if 0 in pp_group.ranks:
                    print('**********')
                    print(f'p2p_coe_pp_deg_{args.pp_deg}: {comm_coe} MB/ms')
                    print('**********')
                    path = os.path.dirname(os.path.abspath(__file__))
                    env_config_path = os.path.join(path, f'./hardware_configs/p2p_bandwidth_{nnodes}nodes_{nproc_per_node}gpus_per_node.json')
                    config = read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
                    key = f'pp_size_{args.pp_deg}'
                    config[key] = comm_coe
                    write_json_config(config, env_config_path)
                    print('Already written p2p bandwidth into env config file %s!'%(env_config_path))
        except Exception as e:
            print(f"Profiler error: {e}")
            return

    # [Step 6] run profiler
    inter_node_send_rank = pp_group.ranks[pp_group.size // 2 - 1]
    inter_node_recv_rank = pp_group.ranks[pp_group.size // 2]

    if rank == inter_node_send_rank:
        print(f'p2p comm from rank {inter_node_send_rank} to rank {inter_node_recv_rank}.')

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0,warmup=1,active=10),
        on_trace_ready=trace_handler
    ) as p:
        for i, input in enumerate(tqdm(trainloader)):
            input = input.to(device)
            if rank == inter_node_send_rank:
                send_forward(input, tensor_shape=[args.local_batch_size, args.seq_length, args.hidden_size])
            if rank == inter_node_recv_rank:
                out = recv_forward(tensor_shape=[args.local_batch_size, args.seq_length, args.hidden_size])
            p.step()

    torch.distributed.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp_deg", type=int, default=2, help="Pipeline parallel degree.", required=True)
    parser.add_argument("--local_batch_size", type=int, default=32, help="local training batch size")
    parser.add_argument("--num_layers", type=int, default=48, help="Number of layers")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")

    args = parser.parse_args()
    train(args)