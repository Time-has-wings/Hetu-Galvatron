import os
import torch
from galvatron.core import (
    clip_grad_norm,
    get_optimizer_and_param_scheduler,
    initialize_galvatron,
    set_megatron_args_for_dataset,
)
from galvatron.models.llama_hf.arguments import model_args
from galvatron.models.llama_hf.dataloader import (
    GeneralDataLoader,
    general_collate_fn,
)
from galvatron.models.llama_hf.LlamaModel_checkpoint import save_llama_module
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_llama_config, get_runtime_profiler, llama_model_hp
from galvatron.utils import distributed_dataloader, set_seed
from megatron.training.arguments import _print_args
from galvatron.models.llama_hf.dataloader import init_loguru


def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    import time
    torch.distributed.barrier()
    if rank == 0:
        timestamp = int(time.time())
    else:
        timestamp = 0
    timestamp_tensor = torch.tensor([timestamp], dtype=torch.long, device=device)
    torch.distributed.all_reduce(timestamp_tensor, op=torch.distributed.ReduceOp.MAX, group=None)
    torch.distributed.barrier()
    timestamp = int(timestamp_tensor.item())
    init_loguru(msg=f'train_dist_varlen/{timestamp}')
    torch.distributed.barrier()

    config = get_llama_config(args)
    model = llama_model_hp(config, args)

    if local_rank == 0:
        print("Creating Dataset...")

    set_megatron_args_for_dataset(
        args, model, model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], 
        model.dp_groups_whole[0], model.cp_groups_whole[0])
    if local_rank == 0:
        _print_args("arguments", args)

    trainloader = distributed_dataloader(
        dataset=GeneralDataLoader(args, device),
        global_bsz=args.global_train_batch_size,
        shuffle=True,
        args=args,
        group=model.dp_groups_whole[0].group,
        collate_fn=general_collate_fn,
    )

    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config, start_iter=0, end_iter=args.train_iters)

    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")

    for iter, batch in enumerate(trainloader):
        input_ids, kwargs, loss_func = batch

        profiler.profile_time_start(iter)
        profiler.profile_memory(iter, "Before Forward")

        loss = model.forward_backward(input_ids, iter, profiler, loss_func=loss_func, **kwargs)

        profiler.profile_memory(iter, "After Backward")

        # for name, weight in model.named_parameters():
        #     if torch.cuda.current_device() == 0:
        #         print(f"final grad {name},{weight.grad}")
        total_norm = clip_grad_norm(model, args.clip_grad)
        # total_norm = 0.0
        optimizer.step()
        opt_param_scheduler.step(increment=args.global_batch_size)

        profiler.profile_memory(iter, "After optimizer_step")

        optimizer.zero_grad()

        # print_loss(args, loss, ep, iter)

        profiler.post_profile_memory(iter)
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        profiler.profile_time_end(iter, loss, learning_rate, total_norm)

        torch.distributed.barrier()

        if args.save != None and (iter + 1) % args.save_interval == 0:
            save_llama_module(args.save, model, optimizer, opt_param_scheduler, iter + 1, args)


if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args)
