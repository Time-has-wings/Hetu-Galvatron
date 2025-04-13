import os

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

from galvatron.core import initialize_galvatron
from galvatron.models.llama_hf.arguments import model_args
from galvatron.models.llama_hf.dataloader import DataLoaderForLlama, random_collate_fn
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_llama_config, get_runtime_profiler, llama_model_hp
from galvatron.utils import distributed_dataloader, print_loss, set_seed
from megatron.training.arguments import _print_args
import nvtx

def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = get_llama_config(args)
    model = llama_model_hp(config, args)

    if local_rank == 0:
        print("Creating Dataset...")

    trainloader = distributed_dataloader(
        dataset=DataLoaderForLlama(args, device),
        global_bsz=args.global_train_batch_size,
        shuffle=True,
        args=args,
        group=model.dp_groups_whole[0].group,
        collate_fn=random_collate_fn,
    )
    
    if local_rank == 0:
        _print_args("arguments", args)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    # path = os.path.dirname(os.path.abspath(__file__))
    # profiler = get_runtime_profiler(args, path, config)

    # profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")

    if args.profile_forward:
        torch.set_grad_enabled(False)

    iter_remove_second = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for ep in range(1):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        
        for iter, batch in enumerate(trainloader):
            if(iter == 21):
                print(f"[rank {rank}] ealry stopping")
                break
            
            if iter == 8:
                torch.cuda.cudart().cudaProfilerStart() 
            
            tokens, kwargs, loss_func = batch
            input_ids = tokens
            batch = [input_ids]
            torch.cuda.synchronize()
            
            with nvtx.annotate(f"galvatron_rank{rank}_iter{iter}"):
                start_event.record()
                with nvtx.annotate(f'forward_backward'):
                    loss = model.forward_backward(batch, iter, profiler=None, loss_func=loss_func, **kwargs)
                optimizer.step()
                optimizer.zero_grad()
                
                end_event.record()
                torch.cuda.synchronize()
            
            iter_remove_second.append(start_event.elapsed_time(end_event) / 1e3)
            if (rank == 0):
                print(f"[rank {rank}]  iter {iter}, iter_time: {iter_remove_second[-1]}")
            
            torch.distributed.barrier()
            if (iter == 13):
                torch.cuda.cudart().cudaProfilerStop()
        
        iter_remove_second = iter_remove_second[2:] if iter_remove_second else [] # remove the first item
        total_time = sum(iter_remove_second) if iter_remove_second else 0 
        avg_time = total_time / len(iter_remove_second) if iter_remove_second else 0
        min_time = min(iter_remove_second) if iter_remove_second else 0
        max_time = max(iter_remove_second) if iter_remove_second else 0
        print(f"Epoch {1} [rank {rank}]summary: total time {total_time:.4f}, avg time {avg_time:.4f}, min time {min_time:.4f}, max time {max_time:.4f}")
    
    if rank == 0:
        print("Training finished.")

if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args)
