import torch
import torch.distributed as dist
import os
import argparse

from galvatron.utils import read_json_config, write_json_config

# Constants
SEQ_LEN = 512
HIDDEN_SIZE = 1024
BYTES_PER_FLOAT16 = 2
MB_TO_BYTES = 1024 * 1024
WARMUP_ITERATIONS = 5
PROFILE_ITERATIONS = 20
ITERATIONS_PER_MEASUREMENT = 10
TRIM_EDGES = 5  # Trim first and last N measurements for stability


def single_p2p_send_recv(input_tensor, prev_rank, next_rank, rank, pp_rank_in_group, pp_size):
    """Perform point-to-point communication using async P2P ops."""
    ops = []

    # Send to next stage (if not last stage)
    if next_rank is not None:
        send_op = dist.P2POp(
            dist.isend,
            input_tensor.contiguous(),
            next_rank,
        )
        ops.append(send_op)

    # Receive from previous stage (if not first stage)
    if prev_rank is not None:
        output = torch.empty_like(input_tensor)
        recv_op = dist.P2POp(
            dist.irecv,
            output,
            prev_rank,
        )
        ops.append(recv_op)
    else:
        output = None

    # Execute all P2P operations
    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return output


def set_seed(rank):
    seed = 123 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _profile_p2p_one(
    rank,
    local_rank,
    device,
    world_size,
    node_num,
    nproc_per_node,
    batch_size,
    seq_len,
    hidden_size,
    pp_size,
    save_config,
):
    if world_size % pp_size != 0:
        raise SystemExit(f"pp_deg {pp_size} must divide world_size {world_size}")

    p2p_message_size = batch_size * seq_len * hidden_size * BYTES_PER_FLOAT16 / MB_TO_BYTES

    num_pp_groups = world_size // pp_size
    pp_rank_in_group = rank // num_pp_groups

    if pp_rank_in_group == 0:
        prev_rank = None
    else:
        prev_rank = rank - num_pp_groups

    if pp_rank_in_group == pp_size - 1:
        next_rank = None
    else:
        next_rank = rank + num_pp_groups

    if local_rank == 0:
        print(f"Strategy: pp_deg = {pp_size}")
        print(f"[p2p_message_size]: {p2p_message_size:.2f} MB")
        print(f"Pipeline stages: {pp_size}, Current rank {rank} is stage {pp_rank_in_group}")
        if prev_rank is not None:
            print(f"  Receives from rank {prev_rank}")
        if next_rank is not None:
            print(f"  Sends to rank {next_rank}")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_list = []

    for _ in range(WARMUP_ITERATIONS):
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
        single_p2p_send_recv(input_tensor, prev_rank, next_rank, rank, pp_rank_in_group, pp_size)

    for _ in range(PROFILE_ITERATIONS):
        input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize()
        torch.distributed.barrier(device_ids=[local_rank])
        start.record()
        for __ in range(ITERATIONS_PER_MEASUREMENT):
            single_p2p_send_recv(input_tensor, prev_rank, next_rank, rank, pp_rank_in_group, pp_size)
        end.record()
        torch.cuda.synchronize()
        if prev_rank is not None or next_rank is not None:
            time_list.append(start.elapsed_time(end) / ITERATIONS_PER_MEASUREMENT)

    if prev_rank is not None or next_rank is not None:
        time_list = sorted(time_list)
        per_comm_time = sum(time_list[TRIM_EDGES:-TRIM_EDGES]) / len(time_list[TRIM_EDGES:-TRIM_EDGES])
        per_comm_time = torch.tensor([per_comm_time]).to(device)
        torch.distributed.all_reduce(per_comm_time, op=torch.distributed.ReduceOp.SUM)
        per_comm_time = per_comm_time.cpu().numpy()[0] / world_size
        throughput_mb_per_ms = p2p_message_size / per_comm_time
    else:
        per_comm_time = 0.0
        throughput_mb_per_ms = 0.0

    if rank == 0:
        if prev_rank is not None or next_rank is not None:
            approx_gb_s = throughput_mb_per_ms * (1.024**2)
            print(
                f"{per_comm_time:.4f} ms, throughput {throughput_mb_per_ms:.4f} MB/ms (~{approx_gb_s:.4f} GB/s)"
            )
        print("**********")
        print(f"p2p_throughput_pp_deg_{pp_size}: {throughput_mb_per_ms:.4f} MB/ms")
        print("**********")
        key = f"pp_size_{pp_size}"
        env_config_path = save_config(
            "./hardware_configs/p2p_bandwidth_%dnodes_%dgpus_per_node.json",
            key,
            throughput_mb_per_ms,
        )
        print(f"Already written p2p bandwidth into env config file {env_config_path}!")
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

    batch_size = int(args.local_batch_size)
    seq_len = int(getattr(args, "seq_length", SEQ_LEN))
    hidden_size = int(getattr(args, "hidden_size", HIDDEN_SIZE))
    pp_list = args.pp_deg

    if rank == 0:
        print(f"local_bsz = {batch_size}")

    def save_config(filename_template, key, value):
        path = os.path.dirname(os.path.abspath(__file__))
        env_config_path = os.path.join(path, filename_template % (node_num, nproc_per_node))
        config = read_json_config(env_config_path) if os.path.exists(env_config_path) else {}
        config[key] = value
        write_json_config(config, env_config_path)
        return env_config_path

    if rank == 0:
        print(f"[pp_deg] world_size={world_size}, order: {pp_list}")
    for pp_size in pp_list:
        torch.cuda.synchronize()
        dist.barrier(device_ids=[local_rank])
        _profile_p2p_one(
            rank,
            local_rank,
            device,
            world_size,
            node_num,
            nproc_per_node,
            batch_size,
            seq_len,
            hidden_size,
            pp_size,
            save_config,
        )

    torch.distributed.barrier(device_ids=[local_rank])
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pp_deg",
        nargs="+",
        type=int,
        required=True,
        metavar="DEG",
        help="Pipeline parallel degree(s), e.g. 2 4 8 (each >= 2).",
    )
    parser.add_argument("--local_batch_size", type=int, default=32, help="local training batch size")
    parser.add_argument("--num_layers", type=int, default=48, help="Number of layers")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size")
    args = parser.parse_args()
    if any(d < 2 for d in args.pp_deg):
        parser.error("--pp_deg values must be >= 2")
    train(args)
