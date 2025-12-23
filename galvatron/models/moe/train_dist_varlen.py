import os
import torch
from galvatron.core import (
    clip_grad_norm,
    get_optimizer_and_param_scheduler,
    initialize_galvatron,
    set_megatron_args_for_dataset,
)
from galvatron.models.moe.arguments import model_args
from galvatron.models.moe.dataloader import random_collate_fn, RealDataLoaderForMoE
from galvatron.models.moe.MoEModel_checkpoint import save_moe_module
from galvatron.models.moe.MoEModel_hybrid_parallel import get_moe_config, get_runtime_profiler, moe_model_hp
from galvatron.utils import distributed_dataloader, set_seed
from megatron.training.arguments import _print_args
import torch.profiler as torch_profiler
from datetime import datetime

def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config = get_moe_config(args)
    model = moe_model_hp(config, args)

    if local_rank == 0:
        print("Creating Dataset...")

    set_megatron_args_for_dataset(
        args, model, model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], model.dp_groups_whole[0], model.cp_groups_whole[0]
    )
    if local_rank == 0:
        _print_args("arguments", args)

    trainloader = distributed_dataloader(
        dataset=RealDataLoaderForMoE(args, device),
        global_bsz=args.global_train_batch_size,
        shuffle=True,
        args=args,
        group=model.dp_groups_whole[0].group,
        collate_fn=random_collate_fn,
    )

    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config, start_iter=0, end_iter=args.train_iters - 1)

    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")

    trace = int(os.getenv('trace', default=0))
    current_time = datetime.now()
    current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    if not trace:
        for iter, batch in enumerate(trainloader):
            tokens, kwargs, loss_func = batch
            profiler.profile_time_start(iter)
            profiler.profile_memory(iter, "Before Forward")

            input_ids = tokens
            batch = [input_ids]

            loss = model.forward_backward(batch, iter, profiler, loss_func=loss_func, **kwargs)

            profiler.profile_memory(iter, "After Backward")

            total_norm = clip_grad_norm(model, args.clip_grad)

            optimizer.step()
            opt_param_scheduler.step(increment=args.global_batch_size)

            profiler.profile_memory(iter, "After optimizer_step")

            optimizer.zero_grad()

            profiler.post_profile_memory(iter)
            for param_group in optimizer.param_groups:
                learning_rate = param_group["lr"]
            profiler.profile_time_end(iter, loss, learning_rate, total_norm)

            torch.distributed.barrier()

            if args.save is not None and (iter + 1) % args.save_interval == 0:
                save_moe_module(args.save, model, optimizer, opt_param_scheduler, iter + 1, args)
    else:
        warmup_times = args.train_iters - 20
        active_times = 4
        export_iter = warmup_times + active_times + 2
        os.makedirs('./traces', exist_ok=True)
        
        with torch_profiler.profile(
            activities=[torch_profiler.ProfilerActivity.CPU, torch_profiler.ProfilerActivity.CUDA],
            schedule=torch_profiler.schedule(
                wait=0,
                warmup=warmup_times,
                active=active_times,
                repeat=1
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for iter, batch in enumerate(trainloader):
                tokens, kwargs, loss_func = batch
                profiler.profile_time_start(iter)
                profiler.profile_memory(iter, "Before Forward")

                input_ids = tokens
                batch = [input_ids]

                loss = model.forward_backward(batch, iter, profiler, loss_func=loss_func, **kwargs)

                profiler.profile_memory(iter, "After Backward")

                total_norm = clip_grad_norm(model, args.clip_grad)

                optimizer.step()
                opt_param_scheduler.step(increment=args.global_batch_size)

                profiler.profile_memory(iter, "After optimizer_step")

                optimizer.zero_grad()

                profiler.post_profile_memory(iter)
                for param_group in optimizer.param_groups:
                    learning_rate = param_group["lr"]
                profiler.profile_time_end(iter, loss, learning_rate, total_norm)

                torch.distributed.barrier()

                prof.step()
                if iter == export_iter:
                    print('[linguangming] start to export chrome_trace')
                    prof.export_chrome_trace(f"./traces/moe_trace_rank{rank}_iter{args.train_iters - 20}_to_{args.train_iters - 1}_{current_time}.json")
                elif iter == export_iter + 1:
                    if rank == 0:
                        print('[linguangming] rank 0 start to modify folder name')
                        os.rename("./traces", f"./traces_{current_time}")
                        print('[linguangming] rank 0 finish to modify folder name')

                if args.save is not None and (iter + 1) % args.save_interval == 0:
                    save_moe_module(args.save, model, optimizer, opt_param_scheduler, iter + 1, args)


if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args)
