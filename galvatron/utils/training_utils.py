import os
import json
import torch
import numpy as np
import random
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

@dataclass
class ColorSet:
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m" 
    RESET = "\033[0m"

def set_seed(seed = 1234):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def distributed_dataloader(dataset, global_bsz, shuffle = True, args = None, group = None, collate_fn=None):
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    # pp_deg = args.pp_deg if args is not None and 'pp_deg' in args else 1
    # data_num_replicas = world_size // pp_deg
    train_batch_size_input = global_bsz // world_size
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            sampler=DistributedSampler(dataset,shuffle=shuffle,num_replicas=world_size,rank=rank),
                            collate_fn=collate_fn)
    return trainloader

def print_loss(args, loss, ep, iter):
    if args.check_loss or args.profile:
        if loss is None:
            return
        if isinstance(loss, (list, tuple)): # Average loss of each microbatch
            if len(loss) == 0:
                return
            if isinstance(loss[0], torch.Tensor):
                loss = np.mean([l.item() for l in loss])
            else:
                loss = np.mean(loss)
        else:
            loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        if ep == -1:
            print('(Iteration %d): Loss = %.3f'% (iter,loss))
        else:
            print('[Epoch %d] (Iteration %d): Loss = %.3f'% (ep,iter,loss))

def print_single_rank(message, rank=0):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank:
            print(message, flush=True)
    else:
        print(message, flush=True)

def store_single_rank(info,):
    chunk = info['chunk']
    # expert_id_list = info['expert_id_list']
    layer_id = info['layer_id']
    token_num_per_expert_list = info['token_num_per_expert_list']

    store_file_name = f'./route_info/rank_{torch.distributed.get_rank()}.json'
    os.makedirs(os.path.dirname(store_file_name), exist_ok=True)

    if os.path.exists(store_file_name):
        with open(store_file_name, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    if str(layer_id) not in existing_data.keys():
        existing_data[str(layer_id)] = {}
    existing_data[str(layer_id)][str(chunk)] = token_num_per_expert_list

    with open(store_file_name, 'w') as f:
        json.dump(existing_data, f, indent=4)

def store_expert_tendency(info):
    chunk = info['chunk']
    layer_id = info['layer_id']
    tendency = info['tendency']

    store_file_name = f'./route_info/rank_{torch.distributed.get_rank()}_tendency.json'
    os.makedirs(os.path.dirname(store_file_name), exist_ok=True)

    if os.path.exists(store_file_name):
        with open(store_file_name, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    if str(chunk) not in existing_data.keys():
        existing_data[str(chunk)] = {}
    existing_data[str(chunk)][str(layer_id)] = tendency

    with open(store_file_name, 'w') as f:
        json.dump(existing_data, f, indent=4)
