"""Cross-stack model correctness: Galvatron runtime vs HuggingFace (DP, 8 ranks).

Runtime ``args.model.model_type`` is always ``gpt`` (same stack). Param ``hf_arch``
only picks the HF baseline / checkpoint layout: ``gpt`` (GPT-2), ``llama``, ``llama2`` (GQA).
"""

import json
import sys
from typing import Any, Dict

import pytest
import torch
from torch.amp import autocast
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from galvatron.core.runtime.datasets import RandomTokenDataset, random_collate_fn
from galvatron.core.runtime.models.builder import build_model
from galvatron.core.runtime.parallel_state import set_args, set_global_memory_buffer
from galvatron.tools.checkpoint_convert_h2g import convert_checkpoints_gpt, convert_checkpoints_llama
from galvatron.utils.training_utils import distributed_dataloader, set_seed
from tests.models.configs.get_config_json import ConfigFactory
from tests.utils.init_dist import init_dist_env
from tests.utils.runtime_args import make_test_args


def _dp_parallel_config(num_layers: int, batch: int, chunks: int) -> Dict[str, Any]:
    enc = ",".join(["1"] * num_layers)
    zeros = ",".join(["0"] * num_layers)
    return {
        "pp_deg": 1,
        "tp_sizes_enc": enc,
        "tp_consecutive_flags": enc,
        "cp_sizes_enc": enc,
        "dp_types_enc": zeros,
        "use_sp": zeros,
        "checkpoint": zeros,
        "global_bsz": batch,
        "chunks": chunks,
        "pp_division": str(num_layers),
        "pipeline_type": "pipedream_flush",
        "default_dp_type": "zero2",
        "vtp": 1,
        "vsp": 0,
    }


def _run_test(test_args: Dict[str, Any]):
    rank, world_size = init_dist_env()
    dp_size = test_args["dp_size"]
    assert dp_size == world_size

    hf_arch = test_args["hf_arch"]
    assert hf_arch in ("gpt", "llama", "llama2")

    batch_size = test_args["batch_size"]
    chunks = test_args["chunks"]
    num_steps = test_args["num_steps"]
    checkpoint_dir = test_args["checkpoint_dir"]
    seed = test_args["seed"]
    last = world_size - 1

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    set_seed(seed)

    cfg = ConfigFactory.get_config_json(hf_arch)

    if hf_arch == "gpt":
        n_layer = cfg["n_layer"]
        parallel_config = _dp_parallel_config(n_layer, batch_size, chunks)
        args = make_test_args(
            hf_arch="gpt",
            rank=rank,
            world_size=world_size,
            checkpoint_load=checkpoint_dir["converted"],
            mixed_precision="bf16",
            async_grad_reduce=False,
            galvatron_config_path=parallel_config,
            global_batch_size=batch_size,
            chunks=chunks,
            seed=seed,
            seq_length=cfg["n_positions"],
            hidden_size=cfg["n_embd"],
            num_layers=n_layer,
            num_attention_heads=cfg["n_head"],
            ffn_hidden_size=cfg["n_embd"] * 4,
            vocab_size=cfg["vocab_size"],
        )
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
        if rank == last:
            baseline_model = GPT2LMHeadModel(hf_config)
            baseline_optimizer = Adam(
                baseline_model.parameters(),
                lr=args.train.lr,
                weight_decay=args.train.weight_decay,
            )
            baseline_model.save_pretrained(checkpoint_dir["baseline"])
            convert_checkpoints_gpt(checkpoint_dir["baseline"], checkpoint_dir["converted"])
            baseline_model = baseline_model.to(device)
    else:
        n_layer = cfg["n_layers"]
        n_heads = cfg["n_heads"]
        n_kv = cfg.get("n_kv_heads", n_heads)
        gqa = n_kv < n_heads
        parallel_config = _dp_parallel_config(n_layer, batch_size, chunks)
        hf_config = LlamaConfig(
            hidden_size=cfg["dim"],
            num_hidden_layers=n_layer,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            intermediate_size=cfg["dim"] * 4,
            vocab_size=cfg["vocab_size"],
            max_position_embeddings=cfg["n_positions"],
            rms_norm_eps=cfg["norm_eps"],
        )
        args = make_test_args(
            hf_arch=hf_arch,
            rank=rank,
            world_size=world_size,
            checkpoint_load=checkpoint_dir["converted"],
            mixed_precision="bf16",
            async_grad_reduce=False,
            galvatron_config_path=parallel_config,
            global_batch_size=batch_size,
            chunks=chunks,
            seed=seed,
            seq_length=cfg["n_positions"],
            hidden_size=cfg["dim"],
            num_layers=n_layer,
            num_attention_heads=n_heads,
            ffn_hidden_size=hf_config.intermediate_size,
            vocab_size=cfg["vocab_size"],
            group_query_attention=gqa,
            num_query_groups=n_kv if gqa else None,
            norm_epsilon=cfg["norm_eps"],
        )
        if rank == last:
            baseline_model = LlamaForCausalLM(hf_config)
            baseline_optimizer = Adam(
                baseline_model.parameters(),
                lr=args.train.lr,
                weight_decay=args.train.weight_decay,
            )
            baseline_model.save_pretrained(checkpoint_dir["baseline"])
            convert_checkpoints_llama(checkpoint_dir["baseline"], checkpoint_dir["converted"])
            baseline_model = baseline_model.to(device)

    set_args(args)
    set_global_memory_buffer()

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

    dp_group = model.dp_groups_whole[0].group
    dp_world_size = torch.distributed.get_world_size(dp_group)

    for i, batch in enumerate(trainloader):
        tokens, kwargs, loss_func = batch
        input_ids = tokens
        fwd_batch = [input_ids]

        gathered_input_ids = [torch.zeros_like(input_ids) for _ in range(dp_world_size)]
        gathered_labels = [torch.zeros_like(kwargs["labels"]) for _ in range(dp_world_size)]
        torch.distributed.all_gather(gathered_input_ids, input_ids, group=dp_group)
        torch.distributed.all_gather(gathered_labels, kwargs["labels"], group=dp_group)

        loss = model.forward_backward(fwd_batch, i, None, loss_func=loss_func, **kwargs)
        optimizer.step()
        optimizer.zero_grad()

        if loss is not None:
            loss = torch.tensor(loss, device=device, dtype=torch.float)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)

        if rank == last:
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

        torch.distributed.broadcast(baseline_loss, src=last)
        torch.distributed.broadcast(loss, src=last)

        assert torch.allclose(loss, baseline_loss, rtol=5e-3), (
            f"Loss mismatch at iteration {i}: {loss} vs {baseline_loss}"
        )

        torch.distributed.barrier()
        if i == num_steps - 1:
            break


@pytest.mark.distributed
@pytest.mark.model
@pytest.mark.parametrize("hf_arch", ["gpt", "llama", "llama2"])
@pytest.mark.parametrize("backend", ["hf"])
@pytest.mark.parametrize("dp_size", [8])
def test_dp_correctness(run_distributed, hf_arch, backend, dp_size, checkpoint_dir):
    run_distributed(
        func_name="_run_test",
        world_size=dp_size,
        args={
            "hf_arch": hf_arch,
            "backend": backend,
            "dp_size": dp_size,
            "batch_size": 16,
            "chunks": 2,
            "num_steps": 3,
            "seed": 42,
            "checkpoint_dir": checkpoint_dir,
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
