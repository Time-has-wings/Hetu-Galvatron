"""Distributed training entry point for GPT.

Usage:
    torchrun ... train_dist.py scripts/train_dist.yaml [overrides...]
"""

import os
import sys

import torch

from galvatron.core.arguments import load_with_hydra
from galvatron.core.runtime.optimizer.utils import clip_grad_norm, get_optimizer_and_param_scheduler
from galvatron.core.runtime.models.builder import build_model, get_runtime_profiler
from galvatron.core.runtime.dataloader import get_batch, get_train_valid_test_data_iterators
from galvatron.core.runtime.utils.utils import set_megatron_args_for_dataset
from galvatron.core.runtime.initialize import initialize_galvatron, _print_args
from galvatron.utils.hf_config_adapter import resolve_model_config


def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    resolve_model_config(args)
    model = build_model(args)

    if local_rank == 0:
        print("Creating Dataset...")

    set_megatron_args_for_dataset(args)

    _print_args(args)

    train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()
    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, start_iter=args.train.iteration, end_iter=args.train.train_iters)
    profiler.profile_memory(0, "After creating model")

    if local_rank == 0:
        print("Start training...")

    for iter_idx in range(getattr(args.train, "iteration", 0), args.train.train_iters):
        tokens, kwargs, loss_func = get_batch(train_data_iterator)

        profiler.profile_time_start(iter_idx)
        profiler.profile_memory(iter_idx, "Before Forward")

        loss = model.forward_backward([tokens], iter_idx, profiler, loss_func=loss_func, **kwargs)

        profiler.profile_memory(iter_idx, "After Backward")

        grad_norm = clip_grad_norm(model, args.train.clip_grad)
        optimizer.step()
        opt_param_scheduler.step(increment=args.train.global_batch_size)

        profiler.profile_memory(iter_idx, "After optimizer_step")
        optimizer.zero_grad()
        profiler.post_profile_memory(iter_idx)

        lr = optimizer.param_groups[0]["lr"]
        profiler.profile_time_end(iter_idx, loss, lr, grad_norm)

        torch.distributed.barrier()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].endswith((".yaml", ".yml")):
        config_path, overrides = sys.argv[1], sys.argv[2:]
        sys.argv = [sys.argv[0]]
        args = load_with_hydra(config_path, overrides=overrides, mode="train_dist")
    else:
        raise ValueError("Usage: python train_dist.py <config_path> [overrides...]")
    initialize_galvatron(args)
    train(args)
