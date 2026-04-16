"""Expert Parallelism correctness: Galvatron EP vs HuggingFace Mixtral (single-device baseline)."""

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
from tests.models.configs.get_config_json import ConfigFactory
from tests.utils.init_dist import init_dist_env
from tests.utils.runtime_args import make_test_args

if hasattr(pytest.mark, "skipif"):
    pytestmark = pytest.mark.skipif(
        MixtralConfig is None or MixtralForCausalLM is None,
        reason="Mixtral support is unavailable in the installed transformers package.",
    )
else:  # pragma: no cover
    pytestmark = None


def _ep_parallel_config(
    num_layers: int,
    ep_size: int,
    batch: int,
    chunks: int,
    dispatcher: str = "alltoall",
) -> Dict[str, Any]:
    """Build a JSON parallel config with Expert Parallelism enabled.

    TP=1, PP=1, CP=1.  EP = *ep_size* so that experts are sharded across
    ``ep_size`` ranks and the remaining ranks form the DP dimension.
    """
    ones = ",".join(["1"] * num_layers)
    zeros = ",".join(["0"] * num_layers)
    ep_enc = ",".join([str(ep_size)] * num_layers)

    return {
        "pp_deg": 1,
        "tp_sizes_enc": ones,
        "tp_consecutive_flags": ones,
        "cp_sizes_enc": ones,
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
        "ep_sizes_enc": ep_enc,
        "tp_of_ep_sizes_enc": ones,
        "dispatcher": dispatcher,
    }


def _run_test(test_args: Dict[str, Any]):
    rank, world_size = init_dist_env()
    ep_size = test_args["ep_size"]
    dispatcher = test_args["dispatcher"]
    batch_size = test_args["batch_size"]
    chunks = test_args["chunks"]
    num_steps = test_args["num_steps"]
    checkpoint_dir = test_args["checkpoint_dir"]
    seed = test_args["seed"]
    last = world_size - 1

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    set_seed(seed)

    cfg = ConfigFactory.get_config_json("mixtral")
    n_layer = cfg["n_layers"]
    n_heads = cfg["n_heads"]
    n_kv = cfg["n_kv_heads"]
    gqa = n_kv < n_heads
    num_experts = max(cfg["num_local_experts"], ep_size)
    parallel_config = _ep_parallel_config(
        n_layer, ep_size, batch_size, chunks, dispatcher
    )

    hf_config = MixtralConfig(
        hidden_size=cfg["dim"],
        intermediate_size=cfg["hidden_dim"],
        num_hidden_layers=n_layer,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        num_local_experts=num_experts,
        num_experts_per_tok=cfg["num_experts_per_tok"],
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["n_positions"],
        rms_norm_eps=cfg["norm_eps"],
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
        seq_length=cfg["n_positions"],
        hidden_size=cfg["dim"],
        num_layers=n_layer,
        num_attention_heads=n_heads,
        ffn_hidden_size=cfg["hidden_dim"],
        vocab_size=cfg["vocab_size"],
        group_query_attention=gqa,
        num_query_groups=n_kv if gqa else None,
        norm_epsilon=cfg["norm_eps"],
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=cfg["hidden_dim"],
        moe_router_topk=cfg["num_experts_per_tok"],
        moe_router_load_balancing_type="none",
        moe_router_score_function="softmax",
        moe_permute_fusion=False,
        moe_token_dispatcher_type=dispatcher,
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
            f"[EP={ep_size}, dispatcher={dispatcher}] "
            f"Loss mismatch at iteration {i}: {loss} vs {baseline_loss}"
        )

        torch.distributed.barrier()
        if i == num_steps - 1:
            break


@pytest.mark.distributed
@pytest.mark.moe
@pytest.mark.parametrize("ep_size", [2, 4, 8])
@pytest.mark.parametrize("dispatcher", ["allgather", "alltoall"])
def test_ep_correctness(run_distributed, ep_size, dispatcher, checkpoint_dir):
    """Expert Parallelism on 8 GPUs with varying EP degrees and dispatchers."""
    run_distributed(
        func_name="_run_test",
        world_size=8,
        args={
            "ep_size": ep_size,
            "dispatcher": dispatcher,
            "batch_size": 16,
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
