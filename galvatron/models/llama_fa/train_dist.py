import os

import torch
from flash_attn.models.gpt import GPTLMHeadModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from galvatron.core import initialize_galvatron, set_megatron_args_for_dataset
from galvatron.models.llama_fa.arguments import model_args
from galvatron.models.llama_fa.dataloader import DataLoaderForLlama, get_batch, get_train_valid_test_data_iterators
from galvatron.models.llama_fa.LlamaModel_hybrid_parallel import get_llama_config, get_runtime_profiler, llama_model_hp
from galvatron.utils import distributed_dataloader, print_loss, set_seed


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
    set_megatron_args_for_dataset(
        args, model, model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], model.dp_groups_whole[0]
    )
    train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config)

    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")
    for iter in range(args.train_iters):
        tokens, kwargs, loss_func = get_batch(train_data_iterator)
        profiler.profile_time_start(iter)
        profiler.profile_memory(iter, "Before Forward")

        input_ids = tokens
        batch = [input_ids]

        loss = model.forward_backward(batch, iter, profiler, loss_func=loss_func, **kwargs)

        profiler.profile_memory(iter, "After Backward")

        optimizer.step()

        profiler.profile_memory(iter, "After optimizer_step")

        optimizer.zero_grad()

        # print_loss(args, loss, ep, iter)

        profiler.post_profile_memory(iter)
        profiler.profile_time_end(iter, loss)

        torch.distributed.barrier()


if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args)
