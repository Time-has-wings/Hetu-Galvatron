"""Distributed dataloader + subgroup sanity checks using the Galvatron runtime dataset/collate."""

import json
import sys

import pytest
import torch
import torch.distributed as dist

from galvatron.core.runtime.datasets import RandomTokenDataset, random_collate_fn
from galvatron.core.runtime.parallel_state import set_args
from galvatron.utils.training_utils import distributed_dataloader, set_seed
from tests.utils.init_dist import init_dist_env
from tests.utils.runtime_args import make_test_args


def _run_test(args: dict):
    rank, world_size = init_dist_env()
    group_size = args["group_size"]
    seed = args["seed"]
    small_model_config = args["small_model_config"]

    if world_size < group_size:
        pytest.skip(f"Test requires at least {group_size} processes")

    torch.cuda.set_device(rank)

    num_groups = world_size // group_size
    group_id = rank // group_size
    groups = []
    for i in range(num_groups):
        ranks_in_group = list(range(i * group_size, (i + 1) * group_size))
        groups.append(dist.new_group(ranks=ranks_in_group))

    current_group = groups[group_id]

    set_seed(seed)

    rt_args = make_test_args(
        rank=rank,
        world_size=world_size,
        seq_length=small_model_config["seq_length"],
        vocab_size=small_model_config["vocab_size"],
        hidden_size=small_model_config["hidden_size"],
        num_layers=small_model_config["num_layers"],
        num_attention_heads=small_model_config["num_attention_heads"],
        use_flash_attn=True,
    )
    set_args(rt_args)

    dataset = RandomTokenDataset(
        rt_args.model.vocab_size,
        rt_args.train.seq_length,
        size=64,
    )

    global_bsz = 16
    loader = distributed_dataloader(
        dataset=dataset,
        global_bsz=global_bsz,
        shuffle=True,
        group=current_group,
        collate_fn=random_collate_fn,
    )

    assert loader is not None
    assert isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler)

    expected_local_bsz = global_bsz // group_size
    assert loader.batch_size == expected_local_bsz

    first_batch = None
    for batch in loader:
        first_batch = batch
        break

    assert first_batch[0].shape == (expected_local_bsz, small_model_config["seq_length"])
    assert isinstance(first_batch[1], dict)
    assert first_batch[1]["attention_mask"] is None
    assert first_batch[1]["labels"].shape == (expected_local_bsz, small_model_config["seq_length"])
    assert first_batch[2] is None

    rank_in_group = rank % group_size
    all_position_groups = []
    for pos in range(group_size):
        ranks_with_same_position = [i * group_size + pos for i in range(num_groups)]
        all_position_groups.append(ranks_with_same_position)

    pos_groups = []
    for ranks_in_group in all_position_groups:
        pos_groups.append(dist.new_group(ranks=ranks_in_group))

    my_group = pos_groups[rank_in_group]

    assert rank in all_position_groups[rank_in_group]

    same_rank_samples = [torch.zeros_like(first_batch[0]) for _ in range(num_groups)]
    dist.all_gather(same_rank_samples, first_batch[0], group=my_group)
    assert all(torch.equal(same_rank_samples[0], sample) for sample in same_rank_samples), (
        "Same rank index across DP groups should see identical samples"
    )


@pytest.mark.distributed
@pytest.mark.parametrize("group_size", [2])
def test_distributed_dataloader_with_groups(run_distributed, small_model_config, seed, group_size):
    run_distributed(
        func_name="_run_test",
        world_size=8,
        args={
            "group_size": group_size,
            "seed": seed,
            "small_model_config": small_model_config,
        },
        script=__file__,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_file.py <function_name> <json_args>")
        sys.exit(1)

    func_name = sys.argv[1]
    payload = json.loads(sys.argv[2])

    if func_name == "_run_test":
        _run_test(payload)
    else:
        print(f"Unknown function: {func_name}")
        sys.exit(1)
