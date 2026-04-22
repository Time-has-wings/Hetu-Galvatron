"""Cross-stack MoE correctness: Galvatron runtime vs HuggingFace Mixtral (DP only)."""

import json
import sys
from typing import Any, Dict

try:
    import pytest
except ImportError:  # pragma: no cover
    class _PytestMarkStub:
        def skipif(self, *args, **kwargs):
            return None

        def parametrize(self, *args, **kwargs):
            def decorator(obj):
                return obj
            return decorator

        def __getattr__(self, _name):
            def decorator(obj):
                return obj
            return decorator

    class _PytestStub:
        mark = _PytestMarkStub()

    pytest = _PytestStub()

import torch
from torch.amp import autocast
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

try:
    from transformers import MixtralConfig, MixtralForCausalLM
except ImportError:  # pragma: no cover
    MixtralConfig = None
    MixtralForCausalLM = None

from galvatron.core.runtime.datasets import RandomTokenDataset, random_collate_fn
from galvatron.core.runtime.models.builder import build_model
from galvatron.core.runtime.parallel_state import set_args, set_global_memory_buffer
from galvatron.tools.checkpoint_convert_h2g import convert_checkpoints_mixtral
from galvatron.utils.training_utils import distributed_dataloader, set_seed
from tests.utils.model_utils import ModelFactory
from tests.utils.init_dist import init_dist_env
from tests.utils.runtime_args import make_test_args

if hasattr(pytest.mark, "skipif"):
    pytestmark = pytest.mark.skipif(
        MixtralConfig is None or MixtralForCausalLM is None,
        reason="Mixtral support is unavailable in the installed transformers package.",
    )
else:  # pragma: no cover
    pytestmark = None


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
        "ep_sizes_enc": enc,
        "tp_of_ep_sizes_enc": enc,
    }


def _run_test(test_args: Dict[str, Any]):
    rank, world_size = init_dist_env()
    dp_size = test_args["dp_size"]
    assert dp_size == world_size

    batch_size = test_args["batch_size"]
    chunks = test_args["chunks"]
    num_steps = test_args["num_steps"]
    checkpoint_dir = test_args["checkpoint_dir"]
    seed = test_args["seed"]
    last = world_size - 1

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    set_seed(seed)

    cfg = ModelFactory.get_test_config("mixtral")
    n_layer = cfg["num_layers"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_query_groups"]
    gqa = n_kv < n_heads
    parallel_config = _dp_parallel_config(n_layer, batch_size, chunks)

    hf_config = MixtralConfig(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["ffn_hidden_size"],
        num_hidden_layers=n_layer,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        num_local_experts=cfg["num_moe_experts"],
        num_experts_per_tok=cfg["moe_router_topk"],
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["seq_length"],
        rms_norm_eps=cfg["norm_epsilon"],
        hidden_act="silu",
        attention_dropout=0.0,
    )

    args = make_test_args(
        hf_arch="mixtral",
        rank=rank,
        world_size=world_size,
        checkpoint_load=checkpoint_dir["converted"],
        mixed_precision="bf16",
        async_grad_reduce=False,
        galvatron_config_path=parallel_config,
        global_batch_size=batch_size,
        chunks=chunks,
        seed=seed,
        seq_length=cfg["seq_length"],
        hidden_size=cfg["hidden_size"],
        num_layers=n_layer,
        num_attention_heads=n_heads,
        ffn_hidden_size=cfg["ffn_hidden_size"],
        vocab_size=cfg["vocab_size"],
        group_query_attention=gqa,
        num_query_groups=n_kv if gqa else None,
        norm_epsilon=cfg["norm_epsilon"],
        num_moe_experts=cfg["num_moe_experts"],
        moe_ffn_hidden_size=cfg["ffn_hidden_size"],
        moe_router_topk=cfg["moe_router_topk"],
        moe_router_load_balancing_type="none",
        moe_router_score_function="softmax",
        moe_permute_fusion=False,
    )

    if rank == last:
        baseline_model = MixtralForCausalLM(hf_config)
        baseline_optimizer = Adam(
            baseline_model.parameters(),
            lr=args.train.lr,
            weight_decay=args.train.weight_decay,
        )
        baseline_model.save_pretrained(checkpoint_dir["baseline"])
        convert_checkpoints_mixtral(checkpoint_dir["baseline"], checkpoint_dir["converted"])
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
@pytest.mark.parametrize("dp_size", [2])
def test_dp_correctness(run_distributed, dp_size, checkpoint_dir):
    run_distributed(
        func_name="_run_test",
        world_size=dp_size,
        args={
            "dp_size": dp_size,
            "batch_size": 8,
            "chunks": 2,
            "num_steps": 2,
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
