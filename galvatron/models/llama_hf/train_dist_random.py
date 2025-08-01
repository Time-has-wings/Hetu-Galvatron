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
    
    print(f'[linguangming] model is {model}')
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, shape: {param.shape}, dtype: {param.dtype}")

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

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config)

    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")

    if args.profile_forward:
        torch.set_grad_enabled(False)

    for ep in range(args.epochs):
        
        if ep == 1:
            break
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
         
        iter_nsys_list = [6, 7, 8]
        for iter, batch in enumerate(trainloader):
            
            if iter in iter_nsys_list:
                if iter == iter_nsys_list[0]:
                    print(f'Starting profiling at iteration {iter} for epoch {ep}.')
                    torch.cuda.cudart().cudaProfilerStart()

                with nvtx.annotate(f"rank{torch.distributed.get_rank()}_Iteration{iter}", domain="torchTraining"):
                    tokens, kwargs, loss_func = batch
                    profiler.profile_time_start(iter)
                    profiler.profile_memory(iter, "Before Forward")

                    input_ids = tokens
                    batch = [input_ids]

                    loss = model.forward_backward(batch, iter, profiler, loss_func=loss_func, **kwargs)

                    profiler.profile_memory(iter, "After Backward")

                    optimizer.step()

                    profiler.profile_memory(iter, "After optimizer_step")

                    optimizer.zero_grad()

                    print_loss(args, loss, ep, iter)

                    profiler.post_profile_memory(iter)
                    profiler.profile_time_end(iter)

                torch.distributed.barrier()
                
                if iter == iter_nsys_list[-1]:
                    print(f'Stopping after {iter} iterations for epoch {ep}.')
                    torch.cuda.cudart().cudaProfilerStop()
            
            else:      
                tokens, kwargs, loss_func = batch
                profiler.profile_time_start(iter)
                profiler.profile_memory(iter, "Before Forward")

                input_ids = tokens
                batch = [input_ids]

                loss = model.forward_backward(batch, iter, profiler, loss_func=loss_func, **kwargs)

                profiler.profile_memory(iter, "After Backward")

                optimizer.step()

                profiler.profile_memory(iter, "After optimizer_step")

                optimizer.zero_grad()

                print_loss(args, loss, ep, iter)

                profiler.post_profile_memory(iter)
                profiler.profile_time_end(iter)

                torch.distributed.barrier()
            

if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args)
