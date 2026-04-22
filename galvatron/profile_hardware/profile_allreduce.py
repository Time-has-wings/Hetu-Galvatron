import torch
import torch.distributed as dist
import os
import argparse

import galvatron
from galvatron.core.runtime.pipeline import PipelineParallel, PipeSequential
from galvatron.core.runtime.comm_groups import gen_comm_groups
from galvatron.utils import read_json_config, write_json_config
from galvatron.utils.training_utils import gen_profiling_groups

# Constants
SEQ_LEN = 512
HIDDEN_SIZE = 1024
BYTES_PER_FLOAT16 = 2
MB_TO_BYTES = 1024 * 1024
WARMUP_ITERATIONS = 5
PROFILE_ITERATIONS = 20
ITERATIONS_PER_MEASUREMENT = 10
TRIM_EDGES = 5  # Trim first and last N measurements for stability


def single_all_reduce(input_tensor, group):
    """Perform all-reduce operation on the input tensor"""
    dist.all_reduce(input_tensor.contiguous(), group=group)
    return input_tensor


def set_seed(rank):
    seed = 123 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def bandwidth_jobs_from_tp_degrees(world_size, tp_degrees: list[int]):
    """For each tp in list, run consec 1 then 0 (skip full-world consec=0, same as old shell loop)."""
    jobs = []
    for s in tp_degrees:
        if world_size % s != 0:
            raise SystemExit(f"--global_tp_deg value {s} must divide world_size {world_size}")
        for c in (1, 0):
            if world_size == s and c == 0:
                continue
            jobs.append((s, c))
    return jobs


def allreduce_work_items(
    world_size: int,
    tp_list: list[int],
    batch_list: list[int],
    profile_time: int,
    global_tp_consec: int | None,
) -> list[tuple[int, int, int]]:
    """Build (tp_size, global_tp_consec, local_batch) jobs.

    Bandwidth (profile_time==0): sweep tp×consec via bandwidth_jobs; exactly one batch.
    Otherwise (SP): sweep over batch_list; multi-tp uses consec=1, single-tp uses ``global_tp_consec``.
    """
    if len(tp_list) > 1 and profile_time not in (0, 1):
        raise SystemExit("multiple --global_tp_deg only supports --profile_time 0 or 1")

    if profile_time == 0:
        if len(batch_list) != 1:
            raise SystemExit("--profile_time 0 (bandwidth) requires exactly one --local_batch_size")
        bs0 = batch_list[0]
        if len(tp_list) > 1:
            return [(tp, c, bs0) for tp, c in bandwidth_jobs_from_tp_degrees(world_size, tp_list)]
        return [(tp_list[0], int(global_tp_consec), bs0)]

    if len(tp_list) > 1:
        out: list[tuple[int, int, int]] = []
        for tp_size in tp_list:
            if world_size % tp_size != 0:
                raise SystemExit(f"--global_tp_deg value {tp_size} must divide world_size {world_size}")
            for bs in batch_list:
                out.append((tp_size, 1, bs))
        return out

    tp_size = tp_list[0]
    if world_size % tp_size != 0:
        raise SystemExit(f"--global_tp_deg value {tp_size} must divide world_size {world_size}")
    c = int(global_tp_consec)
    return [(tp_size, c, bs) for bs in batch_list]


def _profile_allreduce_one(
    rank,
    local_rank,
    device,
    world_size,
    node_num,
    nproc_per_node,
    batch_size,
    seq_len,
    hidden_size,
    tp_size,
    global_tp_consec,
    profile_time,
    save_config,
    comm_group=None,
):
    if comm_group is None:
        comm_group = gen_profiling_groups(tp_size, bool(global_tp_consec))
    allreduce_message_size = (
        2
        * (tp_size - 1)
        / tp_size
        * (batch_size * seq_len * hidden_size * BYTES_PER_FLOAT16 / MB_TO_BYTES)
    )
    if local_rank == 0:
        print(f"Strategy: {tp_size}_{global_tp_consec}")
        print(f"[allreduce_message_size]: per_layer {allreduce_message_size:.2f} MB")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_list = []
    for _ in range(WARMUP_ITERATIONS):
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
        single_all_reduce(input_tensor, comm_group)
    for _ in range(PROFILE_ITERATIONS):
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=comm_group)
        start.record()
        for __ in range(ITERATIONS_PER_MEASUREMENT):
            single_all_reduce(input_tensor, comm_group)
        end.record()
        torch.cuda.synchronize()
        time_list.append(start.elapsed_time(end) / ITERATIONS_PER_MEASUREMENT)

    time_list = sorted(time_list)
    per_comm_time = sum(time_list[TRIM_EDGES:-TRIM_EDGES]) / len(time_list[TRIM_EDGES:-TRIM_EDGES])
    per_comm_time = torch.tensor([per_comm_time]).to(device)
    torch.distributed.all_reduce(per_comm_time, group=comm_group, op=torch.distributed.ReduceOp.SUM)
    per_comm_time = per_comm_time.cpu().numpy()[0] / tp_size

    if profile_time == 0:
        throughput_mb_per_ms = allreduce_message_size / per_comm_time
        if rank == 0:
            comm_coe = allreduce_message_size / per_comm_time * (1.024**2)
            print(f"{per_comm_time:.4f} ms, {comm_coe:.4f} GB/s")
            print("**********")
            print(f"comm_coe_{tp_size}_{global_tp_consec}: {throughput_mb_per_ms:.4f} MB/ms")
            print("**********")
            key = f"allreduce_size_{tp_size}_consec_{global_tp_consec}"
            env_config_path = save_config(
                "./hardware_configs/allreduce_bandwidth_%dnodes_%dgpus_per_node.json", key, throughput_mb_per_ms
            )
            print(f"Already written allreduce bandwidth into env config file {env_config_path}!")
    else:
        if rank == 0:
            print(f"Total time: {sum(time_list):.4f} ms, Measurements: {len(time_list)}")
            print("**********")
            print(f"comm_time_{batch_size}MB_{tp_size}: {per_comm_time:.4f} ms")
            print("**********")
            key = f"allreduce_size_{tp_size}_{batch_size}MB_time"
            env_config_path = save_config(
                "./hardware_configs/sp_time_%dnodes_%dgpus_per_node.json", key, per_comm_time
            )
            print(f"Already written allreduce SP time into env config file {env_config_path}!")
    dist.barrier(device_ids=[local_rank])


def train(args):
    if hasattr(args, "local_rank") and args.local_rank >= 0:
        local_rank = args.local_rank
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device_id = local_rank
    torch.cuda.set_device(device_id)
    device = torch.device("cuda", device_id)

    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    set_seed(rank)
    world_size = torch.distributed.get_world_size()
    nproc_per_node_arg = getattr(args, "nproc_per_node", -1)
    nproc_per_node = nproc_per_node_arg if nproc_per_node_arg and nproc_per_node_arg > 0 else int(
        os.environ.get("LOCAL_WORLD_SIZE", 1)
    )
    node_num = world_size // nproc_per_node

    tp_list = args.global_tp_deg
    batch_list = list(args.local_batch_size)
    seq_len = int(getattr(args, "seq_length", SEQ_LEN))
    hidden_size = int(getattr(args, "hidden_size", HIDDEN_SIZE))
    profile_time = int(args.profile_time)

    if rank == 0:
        print(f"local_bsz list = {batch_list}")

    def save_config(filename_template, key, value):
        path = os.path.dirname(os.path.abspath(__file__))
        env_config_path = os.path.join(path, filename_template % (node_num, nproc_per_node))
        config = read_json_config(env_config_path) if os.path.exists(env_config_path) else {}
        config[key] = value
        write_json_config(config, env_config_path)
        return env_config_path

    work = allreduce_work_items(world_size, tp_list, batch_list, profile_time, args.global_tp_consec)

    if rank == 0:
        print(
            f"[allreduce jobs] world_size={world_size}, profile_time={profile_time}, "
            f"{len(work)} configs (tp, consec, local_bsz): {work}"
        )

    comm_cache = {}

    def comm_for(tp_size: int, global_tp_consec: int):
        key = (tp_size, bool(global_tp_consec))
        if key not in comm_cache:
            comm_cache[key] = gen_profiling_groups(tp_size, bool(global_tp_consec))
        return comm_cache[key]

    for tp_size, global_tp_consec, bs in work:
        torch.cuda.synchronize()
        dist.barrier(device_ids=[local_rank])
        _profile_allreduce_one(
            rank,
            local_rank,
            device,
            world_size,
            node_num,
            nproc_per_node,
            bs,
            seq_len,
            hidden_size,
            tp_size,
            global_tp_consec,
            profile_time,
            save_config,
            comm_group=comm_for(tp_size, global_tp_consec),
        )

    torch.distributed.barrier(device_ids=[local_rank])
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global_tp_deg",
        nargs="+",
        type=int,
        required=True,
        metavar="DEG",
        help="Tensor parallel degree(s), e.g. 8 4 2. One value needs --global_tp_consec; multiple tp: bandwidth (profile_time 0) or SP (profile_time 1) per --local_batch_size rules below.",
    )
    parser.add_argument(
        "--global_tp_consec",
        type=int,
        default=None,
        help="Required when exactly one --global_tp_deg is given. Ignored when multiple DEG values are passed (SP uses consec=1; bandwidth sweep uses 1/0 per tp).",
        choices=[0, 1],
    )
    parser.add_argument(
        "--local_batch_size",
        nargs="+",
        type=int,
        default=[32],
        metavar="N",
        help="Local batch size(s). profile_time 0: exactly one (bandwidth, no batch sweep). "
        "profile_time 1: one or many (SP sweep over batch). Default: 32.",
    )
    parser.add_argument("--profile_time", type=int, default=0, help="Profile time", required=True)
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=24, help="Number of layers")

    args = parser.parse_args()
    train(args)