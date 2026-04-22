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


def single_all_to_all(input_tensor, group):
    seq_world_size = dist.get_world_size(group)
    input_t = input_tensor.reshape(seq_world_size, -1)
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)
    return output


def set_seed(rank):
    seed = 123 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _profile_all2all_one(
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
    comm_group,
    save_config,
):
    tp_consec = 1
    all2all_message_size = (
        (batch_size * seq_len * hidden_size * BYTES_PER_FLOAT16 / MB_TO_BYTES) * (tp_size - 1) / tp_size
    )

    if local_rank == 0:
        print(f"Strategy: {tp_size}_{tp_consec}")
        print(f"[all2all_message_size]: per_layer {all2all_message_size:.2f} MB")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_list = []

    for _ in range(WARMUP_ITERATIONS):
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
        single_all_to_all(input_tensor, comm_group)

    for _ in range(PROFILE_ITERATIONS):
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=comm_group)
        start.record()
        for __ in range(ITERATIONS_PER_MEASUREMENT):
            single_all_to_all(input_tensor, comm_group)
        end.record()
        torch.cuda.synchronize()
        time_list.append(start.elapsed_time(end) / ITERATIONS_PER_MEASUREMENT)

    time_list = sorted(time_list)
    per_comm_time = sum(time_list[TRIM_EDGES:-TRIM_EDGES]) / len(time_list[TRIM_EDGES:-TRIM_EDGES])
    per_comm_time = torch.tensor([per_comm_time]).to(device)
    torch.distributed.all_reduce(per_comm_time, group=comm_group, op=torch.distributed.ReduceOp.SUM)
    per_comm_time = per_comm_time.cpu().numpy()[0] / tp_size

    if rank == 0:
        print(f"Total time: {sum(time_list):.4f} ms, Measurements: {len(time_list)}")
        print("**********")
        print(f"comm_time_{batch_size}MB_{tp_size}: {per_comm_time:.4f} ms")
        print("**********")
        key = f"all2all_size_{tp_size}_{batch_size}MB_time"
        env_config_path = save_config("./hardware_configs/sp_time_%dnodes_%dgpus_per_node.json", key, per_comm_time)
        print(f"Already written all2all time into env config file {env_config_path}!")
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

    seq_len = int(getattr(args, "seq_length", SEQ_LEN))
    hidden_size = int(getattr(args, "hidden_size", HIDDEN_SIZE))
    tp_list = args.global_tp_deg
    batch_list = args.local_batch_size

    def save_config(filename_template, key, value):
        path = os.path.dirname(os.path.abspath(__file__))
        env_config_path = os.path.join(path, filename_template % (node_num, nproc_per_node))
        config = read_json_config(env_config_path) if os.path.exists(env_config_path) else {}
        config[key] = value
        write_json_config(config, env_config_path)
        return env_config_path

    if rank == 0:
        jobs = [(t, b) for t in tp_list for b in batch_list]
        print(f"[global_tp_deg x local_batch_size] world_size={world_size}, {len(jobs)} configs: {jobs}")

    comm_by_tp = {}

    def comm_for_tp(tp_size: int):
        if tp_size not in comm_by_tp:
            comm_by_tp[tp_size] = gen_profiling_groups(tp_size, 1)
        return comm_by_tp[tp_size]

    for tp_size in tp_list:
        if world_size % tp_size != 0:
            raise SystemExit(f"--global_tp_deg value {tp_size} must divide world_size {world_size}")
        comm_group = comm_for_tp(tp_size)
        for batch_size in batch_list:
            torch.cuda.synchronize()
            dist.barrier(device_ids=[local_rank])
            _profile_all2all_one(
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
                comm_group,
                save_config,
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
        help="Tensor parallel degree(s), e.g. 8 4 2, for a Cartesian sweep with --local_batch_size.",
    )
    parser.add_argument(
        "--local_batch_size",
        nargs="+",
        type=int,
        required=True,
        metavar="N",
        help="Local batch size(s), e.g. 32 or 1024 512 ....",
    )
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")

    args = parser.parse_args()
    train(args)
