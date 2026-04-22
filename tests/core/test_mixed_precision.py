"""Mixed-precision DP correctness vs HF baseline (Galvatron runtime)."""

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


def _dp_parallel_config(batch: int, chunks: int) -> Dict[str, Any]:
    enc = ",".join(["1"] * _NUM_LAYERS)
    return {
        "pp_deg": 1,
        "tp_sizes_enc": enc,
        "tp_consecutive_flags": enc,
        "cp_sizes_enc": enc,
        "dp_types_enc": ",".join(["0"] * _NUM_LAYERS),
        "use_sp": enc.replace("1", "0"),
        "checkpoint": enc.replace("1", "0"),
        "global_bsz": batch,
        "chunks": chunks,
        "pp_division": str(_NUM_LAYERS),
        "pipeline_type": "pipedream_flush",
        "default_dp_type": "zero2",
        "vtp": 1,
        "vsp": 0,
    }


def _run_test(test_args: Dict[str, Any]):
    rank, world_size = init_dist_env()
    dp_size = test_args["dp_size"]
    assert dp_size == world_size, "world_size must equal dp_size for this test"

    mixed_precision = test_args["mixed_precision"]
    use_flash_attn = test_args["use_flash_attn"]
    checkpoint_dir = test_args["checkpoint_dir"]
    num_steps = test_args["num_steps"]
    seed = test_args["seed"]
    batch_size = test_args["batch_size"]
    chunks = test_args["chunks"]
    parallel_config = test_args["parallel_config"]

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    set_seed(seed)

    args = make_test_args(
        rank=rank,
        world_size=world_size,
        checkpoint_load=checkpoint_dir["converted"],
        mixed_precision=mixed_precision,
        async_grad_reduce=False,
        galvatron_config_path=parallel_config,
        global_batch_size=batch_size,
        chunks=chunks,
        seed=seed,
        use_flash_attn=use_flash_attn,
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

    if rank == 0:
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

    cast_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16

    for i, batch in enumerate(trainloader):
        tokens, kwargs, loss_func = batch
        input_ids = tokens
        fwd_batch = [input_ids]

        dp_group = model.dp_groups_whole[0].group
        if rank == 0:
            gathered_input_ids = [torch.zeros_like(input_ids) for _ in range(world_size)]
            gathered_labels = [torch.zeros_like(kwargs["labels"]) for _ in range(world_size)]
        else:
            gathered_input_ids = None
            gathered_labels = None
        torch.distributed.gather(input_ids, gathered_input_ids, dst=0, group=dp_group)
        torch.distributed.gather(kwargs["labels"], gathered_labels, dst=0, group=dp_group)

        loss = model.forward_backward(fwd_batch, i, None, loss_func=loss_func, **kwargs)
        loss = torch.tensor(loss, device=device, dtype=torch.float)
        optimizer.step()
        optimizer.zero_grad()

        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)

        if rank == 0:
            full_batch = torch.cat(gathered_input_ids, dim=0)
            full_labels = torch.cat(gathered_labels, dim=0)
            with autocast(device_type="cuda", dtype=cast_dtype):
                logits = baseline_model(input_ids=full_batch).logits
                baseline_loss = CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    full_labels.view(-1).to(logits.device),
                )
            baseline_loss.backward()
            baseline_optimizer.step()
            baseline_optimizer.zero_grad()
            assert torch.allclose(loss, baseline_loss, rtol=5e-3), (
                f"Loss mismatch at iteration {i}: {loss} vs {baseline_loss}"
            )

        if i == num_steps - 1:
            break


@pytest.mark.distributed
@pytest.mark.model
@pytest.mark.parametrize("mixed_precision", ["fp16", "bf16"])
@pytest.mark.parametrize("use_flash_attn", [True])
def test_dp_correctness(run_distributed, mixed_precision, use_flash_attn, checkpoint_dir):
    """DP training with fp16/bf16; runtime attention requires FlashAttention (``use_flash_attn=True``)."""
    parallel_config = _dp_parallel_config(batch=16, chunks=2)
    config = {
        "dp_size": 8,
        "parallel_config": parallel_config,
        "batch_size": 16,
        "chunks": 2,
        "num_steps": 3,
        "seed": 42,
        "checkpoint_dir": checkpoint_dir,
        "mixed_precision": mixed_precision,
        "use_flash_attn": use_flash_attn,
    }
    run_distributed(
        func_name="_run_test",
        world_size=8,
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
