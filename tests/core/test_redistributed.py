import pytest
import torch
import sys
import json
from typing import Dict, Any

from torch.optim import Adam
from torch.amp import autocast
from torch.nn import CrossEntropyLoss

from tests.utils.init_dist import init_dist_env
from tests.utils.runtime_args import make_test_args
from tests.models.configs.get_config_json import ConfigFactory

from galvatron.core.runtime.parallel_state import set_args, _set_global_memory_buffer
from galvatron.core.runtime.models.builder import build_model
from galvatron.core.runtime.datasets import RandomTokenDataset, random_collate_fn
from galvatron.utils.training_utils import set_seed, distributed_dataloader
from galvatron.tools.checkpoint_convert_h2g import convert_checkpoints_gpt
from transformers import GPT2Config, GPT2LMHeadModel

def _run_test(test_args: Dict[str, Any]):
    rank, world_size = init_dist_env()
    tp_list = test_args["tp_size"]
    model_type = test_args["model_type"]
    batch_size = test_args["batch_size"]
    chunks = test_args["chunks"]
    num_steps = test_args["num_steps"]
    seed = test_args["seed"]
    checkpoint_dir = test_args["checkpoint_dir"]

    # Galvatron runtime: currently flash-attn path requires sequence parallel.
    mixed_precision = "bf16"
    async_grad_reduce = False

    device = torch.device("cuda", rank)
    set_seed(seed)

    # Derive model sizes (gpt / gpt256) to match HF baseline.
    cfg = ConfigFactory.get_config_json(model_type)
    hidden_size = cfg["n_embd"]
    num_layers = cfg["n_layer"]
    num_attention_heads = cfg["n_head"]
    seq_length = cfg["n_positions"]
    vocab_size = cfg["vocab_size"]
    ffn_hidden_size = hidden_size * 4

    parallel_config = {
        "pp_deg": 1,
        "tp_sizes_enc": ",".join(str(x) for x in tp_list["tp"]),
        "tp_consecutive_flags": ",".join(["1"] * len(tp_list["tp"])),
        "cp_sizes_enc": ",".join(["1"] * len(tp_list["tp"])),
        "dp_types_enc": ",".join(["0"] * len(tp_list["tp"])),
        "use_sp": ",".join(["0"] * len(tp_list["tp"])),
        "checkpoint": ",".join(["0"] * len(tp_list["tp"])),
        "global_bsz": batch_size,
        "chunks": chunks,
        "pp_division": str(num_layers),
        "pipeline_type": "pipedream_flush",
        "default_dp_type": "zero2",
        "vtp": tp_list["vocab_tp"],
        "vsp": 0,
    }

    args = make_test_args(
        rank=rank,
        world_size=world_size,
        checkpoint_load=checkpoint_dir["converted"],
        mixed_precision=mixed_precision,
        async_grad_reduce=async_grad_reduce,
        galvatron_config_path=parallel_config,
        global_batch_size=batch_size,
        chunks=chunks,
        seed=seed,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        ffn_hidden_size=ffn_hidden_size,
        vocab_size=vocab_size,
    )
    set_args(args)
    _set_global_memory_buffer()

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
    optimizer = Adam(
        model.parameters(),
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
    )

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
        batch = [input_ids]

        dp_group = model.dp_groups_whole[0].group
        dp_world_size = torch.distributed.get_world_size(dp_group)

        if input_ids is not None:
            gathered_input_ids = [torch.zeros_like(input_ids) for _ in range(dp_world_size)]
            gathered_labels = [torch.zeros_like(kwargs["labels"]) for _ in range(dp_world_size)]
            torch.distributed.all_gather(gathered_input_ids, input_ids, group=dp_group)
            torch.distributed.all_gather(gathered_labels, kwargs["labels"], group=dp_group)

        loss = model.forward_backward(batch, i, None, loss_func=loss_func, **kwargs)
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
@pytest.mark.parametrize("model_type", ["gpt256"])
@pytest.mark.parametrize("world_size", [8])
@pytest.mark.parametrize("tp_size", (
    {"tp":[1,2,4,8], "vocab_tp":8},
    {"tp":[2,8,2,1], "vocab_tp":4},
    {"tp":[8,4,1,2], "vocab_tp":2}
))
def test_redistributed(run_distributed, model_type, world_size, tp_size, checkpoint_dir):
    """Test redistributed correctness (adapted to Galvatron runtime)."""
    config = {
        "model_type": model_type,
        "tp_size": tp_size,
        "batch_size": 32,
        "chunks": 2,
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
    """Entry point for distributed processes"""
    if len(sys.argv) != 3:
        print("Usage: python test_file.py <function_name> <json_args>")
        sys.exit(1)
        
    func_name = sys.argv[1]
    args = json.loads(sys.argv[2])
    
    if func_name == "_run_test":
        _run_test(args)
    else:
        print(f"Unknown function: {func_name}")
        sys.exit(1)