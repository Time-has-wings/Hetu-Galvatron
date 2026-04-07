import torch.distributed as dist
import os
import torch

def init_dist_env():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    """Initialize distributed environment and return rank and world_size"""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://"
        )
    return dist.get_rank(), dist.get_world_size()