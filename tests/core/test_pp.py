"""Pipeline-parallel correctness vs HF baseline (Galvatron runtime)."""

import json
import sys
from typing import Any, Dict

import pytest
import torch
from torch.amp import autocast
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import GPT2Config, GPT2LMHeadModel

from galvatron.core.runtime.datasets import RandomTokenDataset, random_collate_fn
from galvatron.core.runtime.models.builder import build_model
from galvatron.core.runtime.parallel_state import set_global_memory_buffer, set_args
from galvatron.tools.checkpoint_convert_h2g import convert_checkpoints_gpt
from galvatron.utils.training_utils import distributed_dataloader, set_seed
from tests.utils.init_dist import init_dist_env
from tests.utils.runtime_args import make_test_args

_NUM_LAYERS = 4


def _pp_parallel_config(pp_size: int, batch: int, chunks: int, pipeline_type: str) -> Dict[str, Any]:
    if pp_size == 2:
        pp_div = "2,2"
    elif pp_size == 4:
        pp_div = "1,1,1,1"
    else:
        raise ValueError(pp_size)
    enc = ",".join(["1"] * _NUM_LAYERS)
    zeros = ",".join(["0"] * _NUM_LAYERS)
    return {
        "pp_deg": pp_size,
        "tp_sizes_enc": enc,
        "tp_consecutive_flags": enc,
        "cp_sizes_enc": enc,
        "dp_types_enc": zeros,
        "use_sp": zeros,
        "checkpoint": zeros,
        "global_bsz": batch,
        "chunks": chunks,
        "pp_division": pp_div,
        "pipeline_type": pipeline_type,
        "default_dp_type": "zero2",
        "vtp": 1,
        "vsp": 0,
    }


def _run_test(test_args: Dict[str, Any]):
    rank, world_size = init_dist_env()
    pp_size = test_args["pp_size"]
    pipeline_type = test_args["pipeline_type"]
    dp_size = world_size // pp_size
    batch_size = test_args["batch_size"]
    chunks = test_args["chunks"]
    num_steps = test_args["num_steps"]
    seed = test_args["seed"]
    checkpoint_dir = test_args["checkpoint_dir"]
    parallel_config = test_args["parallel_config"]

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    set_seed(seed)

    args = make_test_args(
        rank=rank,
        world_size=world_size,
        checkpoint_load=checkpoint_dir["converted"],
        mixed_precision="bf16",
        async_grad_reduce=False,
        galvatron_config_path=parallel_config,
        global_batch_size=batch_size,
        chunks=chunks,
        seed=seed,
    )
    set_args(args)
    set_global_memory_buffer()

    hf_config = GPT2Config(
        n_embd=args.model.hidden_size,
        n_layer=args.model.num_layers,
        n_head=args.model.num_attention_heads,
        n_positions=args.train.seq_length,
        n_inner=args.model.ffn_hidden_size,
        vocab_size=args.model.vocab_size,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

    if rank == world_size - 1:
        baseline_model = GPT2LMHeadModel(hf_config)
        baseline_optimizer = Adam(
            baseline_model.parameters(),
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
        )
        baseline_model.save_pretrained(checkpoint_dir["baseline"])
        convert_checkpoints_gpt(checkpoint_dir["baseline"], checkpoint_dir["converted"])
        baseline_model = baseline_model.to(device)

    torch.distributed.barrier()

    model = build_model(args)
    optimizer = Adam(model.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay)

    trainloader = distributed_dataloader(
        dataset=RandomTokenDataset(args.model.vocab_size, args.train.seq_length, size=256),
        global_bsz=batch_size,
        shuffle=True,
        group=model.dp_groups_whole[0].group,
        collate_fn=random_collate_fn,
    )

    for i, batch in enumerate(trainloader):
        tokens, kwargs, loss_func = batch
        input_ids = tokens
        fwd_batch = [input_ids]

        dp_group = model.dp_groups_whole[0].group
        gathered_input_ids = [torch.zeros_like(input_ids) for _ in range(dp_size)]
        gathered_labels = [torch.zeros_like(kwargs["labels"]) for _ in range(dp_size)]
        torch.distributed.all_gather(gathered_input_ids, input_ids, group=dp_group)
        torch.distributed.all_gather(gathered_labels, kwargs["labels"], group=dp_group)

        loss = model.forward_backward(fwd_batch, i, None, loss_func=loss_func, **kwargs)
        optimizer.step()
        optimizer.zero_grad()

        if loss is not None:
            loss = torch.tensor(loss, device=device, dtype=torch.float)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)

        if rank == world_size - 1:
            full_batch = torch.cat(gathered_input_ids, dim=0)
            full_labels = torch.cat(gathered_labels, dim=0)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = baseline_model(input_ids=full_batch).logits
                baseline_loss = CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    full_labels.view(-1).to(logits.device),
                )
            baseline_loss.backward()
            baseline_optimizer.step()
            baseline_optimizer.zero_grad()
        else:
            baseline_loss = torch.tensor(0.0, device=device, dtype=torch.float)
            loss = torch.tensor(0.0, device=device, dtype=torch.float)

        torch.distributed.broadcast(baseline_loss, src=world_size - 1)
        torch.distributed.broadcast(loss, src=world_size - 1)

        assert torch.allclose(loss, baseline_loss, rtol=5e-3), (
            f"Loss mismatch at iteration {i}: {loss} vs {baseline_loss}"
        )

        torch.distributed.barrier()
        if i == num_steps - 1:
            break


@pytest.mark.distributed
@pytest.mark.parallel
@pytest.mark.parametrize("world_size", [8])
@pytest.mark.parametrize("pp_size", [2, 4])
@pytest.mark.parametrize("pipeline_type", ["gpipe", "pipedream_flush"])
@pytest.mark.parametrize("chunks", [2, 8])
def test_pp(run_distributed, world_size, pp_size, pipeline_type, chunks, checkpoint_dir):
    """Pipeline parallel (8 GPUs): compare losses to HF on the last global rank."""
    parallel_config = _pp_parallel_config(pp_size, batch=32, chunks=chunks, pipeline_type=pipeline_type)
    config = {
        "pp_size": pp_size,
        "pipeline_type": pipeline_type,
        "parallel_config": parallel_config,
        "batch_size": 32,
        "chunks": chunks,
        "num_steps": 3,
        "seed": 42,
        "checkpoint_dir": checkpoint_dir,
    }
    run_distributed(
        func_name="_run_test",
        world_size=world_size,
        args=config,
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
