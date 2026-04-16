import json
import os
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from galvatron.core.runtime.parallel_state import get_args
from galvatron.core.runtime.tensor_parallel.utils import VocabUtility
from galvatron.core.runtime.hybrid_parallel_config import get_hybrid_parallel_configs_api

from ..models.modules import (
    GalvatronEmbedding,
    GalvatronFinalNorm,
    GalvatronCausalLMHead,
)
from ..models.moe_modules import (
    GalvatronMoEAttention,
    GalvatronMoERouter,
    GalvatronMoEMLP,
    GalvatronMoEDecoderLayer,
)

embedding_name = "model_embed_tokens.pt"
ln_f_name = "model_norm.pt"
cls_name = "lm_head.pt"
attention_name = "model_layers_%d_attention.pt"
router_name = "model_layers_%d_router.pt"
mlp_name = "model_layers_%d_mlp.pt"


def _runtime_args():
    args = get_args()
    model_args = getattr(args, "model", args)
    ckpt_args = getattr(args, "ckpt", args)
    parallel_args = getattr(args, "parallel", args)
    return args, model_args, ckpt_args, parallel_args


def _load_file(path):
    return torch.load(path, mmap=True, map_location="cpu")


def _copy_module_state(checkpoint, name, submodule):
    weight_key = f"{name}.weight"
    if hasattr(submodule, "weight") and weight_key in checkpoint:
        submodule.weight.copy_(checkpoint[weight_key].to(device="cuda", dtype=torch.float32))
    bias_key = f"{name}.bias"
    if getattr(submodule, "bias", None) is not None and bias_key in checkpoint:
        submodule.bias.copy_(checkpoint[bias_key].to(device="cuda", dtype=torch.float32))


def load_distributed_checkpoint(load, tp_groups, name, submodule, module, ep_groups):
    args, _, ckpt_args, _ = _runtime_args()
    load = os.path.join(load, f"iter_{ckpt_args.load_iteration}")

    if isinstance(module, GalvatronEmbedding):
        rank = dist.get_rank(tp_groups)
        checkpoint = _load_file(os.path.join(load, embedding_name[:-3], f"{rank}.pt"))
        _copy_module_state(checkpoint, name, submodule)
        return

    if isinstance(module, GalvatronFinalNorm):
        checkpoint = _load_file(os.path.join(load, ln_f_name))
        _copy_module_state(checkpoint, name, submodule)
        return

    if isinstance(module, GalvatronCausalLMHead):
        rank = dist.get_rank(tp_groups)
        checkpoint = _load_file(os.path.join(load, cls_name[:-3], f"{rank}.pt"))
        _copy_module_state(checkpoint, name, submodule)
        return

    if isinstance(module, GalvatronMoEAttention):
        rank = dist.get_rank(tp_groups)
        checkpoint = _load_file(os.path.join(load, (attention_name % module.layer_idx)[:-3], f"{rank}.pt"))
        _copy_module_state(checkpoint, name, submodule)
        return

    if isinstance(module, GalvatronMoERouter):
        checkpoint = _load_file(os.path.join(load, router_name % module.layer_idx))
        module.router.weight.copy_(checkpoint["router.weight"].to(device="cuda", dtype=torch.float32))
        if getattr(module.router, "expert_bias", None) is not None and "router.expert_bias" in checkpoint:
            module.router.expert_bias.copy_(checkpoint["router.expert_bias"].to(device="cuda", dtype=torch.float32))
        return

    if isinstance(module, GalvatronMoEMLP):
        rank = dist.get_rank(tp_groups)
        ep_rank = dist.get_rank(ep_groups)
        checkpoint = _load_file(os.path.join(load, (mlp_name % module.layer_idx)[:-3], f"{ep_rank}_{rank}.pt"))
        _copy_module_state(checkpoint, name, submodule)
        return

    raise ValueError(f"moe_adapter: unhandled distributed checkpoint module {type(module).__name__}")


def _load_embedding_from_hf(load, tp_groups, submodule):
    _, model_args, _, _ = _runtime_args()
    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)
    checkpoint = _load_file(os.path.join(load, embedding_name))
    vocab_size = checkpoint["embed_tokens.weight"].shape[0]
    padding_size = model_args.padded_vocab_size - vocab_size
    padded_weight = F.pad(
        checkpoint["embed_tokens.weight"].to(device="cuda", dtype=torch.float32),
        (0, 0, padding_size, 0),
        mode="constant",
        value=0,
    )
    vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
        model_args.padded_vocab_size,
        rank,
        world_size,
    )
    submodule.weight.copy_(padded_weight[vocab_start_index:vocab_end_index])


def _load_lm_head_from_hf(load, tp_groups, submodule):
    _, model_args, _, _ = _runtime_args()
    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)
    checkpoint = _load_file(os.path.join(load, cls_name))
    vocab_size = checkpoint["weight"].shape[0]
    padding_size = model_args.padded_vocab_size - vocab_size
    padded_weight = F.pad(
        checkpoint["weight"].to(device="cuda", dtype=torch.float32),
        (0, 0, padding_size, 0),
        mode="constant",
        value=0,
    )
    vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
        model_args.padded_vocab_size,
        rank,
        world_size,
    )
    submodule.weight.copy_(padded_weight[vocab_start_index:vocab_end_index].contiguous())


def _load_attention_from_hf(checkpoint, tp_groups, name, submodule):
    _, model_args, _, _ = _runtime_args()
    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)

    if "input_layernorm" in name:
        submodule.weight.copy_(checkpoint["input_layernorm.weight"].to(device="cuda", dtype=torch.float32))
        return

    if "linear_qkv" in name:
        nh = model_args.num_attention_heads
        ng = model_args.num_query_groups if model_args.num_query_groups is not None else model_args.num_attention_heads
        dim = model_args.kv_channels
        assert nh % ng == 0
        weight = torch.cat(
            [
                checkpoint["self_attn.q_proj.weight"].reshape((ng, dim * nh // ng, -1)),
                checkpoint["self_attn.k_proj.weight"].reshape((ng, dim, -1)),
                checkpoint["self_attn.v_proj.weight"].reshape((ng, dim, -1)),
            ],
            dim=1,
        ).reshape((-1, model_args.hidden_size))
        start, end = VocabUtility.vocab_range_from_global_vocab_size(weight.shape[0], rank, world_size)
        submodule.weight.copy_(weight[start:end].contiguous())
        return

    if "linear_proj" in name:
        weight = checkpoint["self_attn.o_proj.weight"].to(device="cuda", dtype=torch.float32)
        start, end = VocabUtility.vocab_range_from_global_vocab_size(weight.shape[1], rank, world_size)
        submodule.weight.copy_(weight[:, start:end].contiguous())
        if getattr(submodule, "bias", None) is not None and "self_attn.o_proj.bias" in checkpoint:
            submodule.bias.copy_(checkpoint["self_attn.o_proj.bias"].to(device="cuda", dtype=torch.float32))
        return

    if "pre_router_norm" in name:
        submodule.weight.copy_(checkpoint["post_attention_layernorm.weight"].to(device="cuda", dtype=torch.float32))
        return

    raise ValueError(f"moe_adapter: unhandled MoE attention submodule {name!r}")


def _load_router_from_hf(checkpoint, submodule):
    router = submodule.router if hasattr(submodule, "router") else submodule
    router.weight.copy_(checkpoint["block_sparse_moe.gate.weight"].to(device="cuda", dtype=torch.float32))
    if getattr(router, "expert_bias", None) is not None and "block_sparse_moe.expert_bias" in checkpoint:
        router.expert_bias.copy_(checkpoint["block_sparse_moe.expert_bias"].to(device="cuda", dtype=torch.float32))


def _load_mlp_from_hf(checkpoint, tp_groups, name, submodule, module):
    if "local_experts" not in name:
        return
    if not hasattr(module.experts, "local_experts"):
        raise NotImplementedError("moe_adapter: grouped GEMM checkpoints are not supported yet")

    match = re.search(r"local_experts\.(\d+)\.(linear_fc1|linear_fc2)$", name)
    if match is None:
        return

    local_idx = int(match.group(1))
    proj_name = match.group(2)
    global_idx = module.local_expert_indices[local_idx]

    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)

    if proj_name == "linear_fc1":
        w1 = checkpoint[f"block_sparse_moe.experts.{global_idx}.w1.weight"]
        w3 = checkpoint[f"block_sparse_moe.experts.{global_idx}.w3.weight"]
        start, end = VocabUtility.vocab_range_from_global_vocab_size(w1.shape[0], rank, world_size)
        weight = torch.cat([
            w1[start:end].contiguous(),
            w3[start:end].contiguous(),
        ], dim=0)
        submodule.weight.copy_(weight.to(device="cuda", dtype=torch.float32).contiguous())
        return

    weight = checkpoint[f"block_sparse_moe.experts.{global_idx}.w2.weight"].to(device="cuda", dtype=torch.float32)
    start, end = VocabUtility.vocab_range_from_global_vocab_size(weight.shape[1], rank, world_size)
    submodule.weight.copy_(weight[:, start:end].contiguous())


def load_hf_checkpoint(load, tp_groups, name, submodule, module, ep_groups):
    if name.endswith("embed_tokens"):
        _load_embedding_from_hf(load, tp_groups, submodule)
        return

    if name == "norm":
        checkpoint = _load_file(os.path.join(load, ln_f_name))
        submodule.weight.copy_(checkpoint["weight"].to(device="cuda", dtype=torch.float32))
        return

    if name == "lm_head":
        _load_lm_head_from_hf(load, tp_groups, submodule)
        return

    if isinstance(module, GalvatronMoEAttention):
        checkpoint = _load_file(os.path.join(load, f"model_layers_{module.layer_idx}.pt"))
        _load_attention_from_hf(checkpoint, tp_groups, name, submodule)
        return

    if isinstance(module, GalvatronMoERouter):
        checkpoint = _load_file(os.path.join(load, f"model_layers_{module.layer_idx}.pt"))
        _load_router_from_hf(checkpoint, submodule)
        return

    if isinstance(module, GalvatronMoEMLP):
        checkpoint = _load_file(os.path.join(load, f"model_layers_{module.layer_idx}.pt"))
        _load_mlp_from_hf(checkpoint, tp_groups, name, submodule, module)
        return

    raise ValueError(f"moe_adapter: unhandled HF checkpoint module {type(module).__name__} name={name!r}")


@torch.no_grad()
def load_moe_module(load, tp_groups, name, submodule, module, distributed_checkpoint, ep_groups=None):
    if distributed_checkpoint:
        load_distributed_checkpoint(load, tp_groups, name, submodule, module, ep_groups)
    else:
        load_hf_checkpoint(load, tp_groups, name, submodule, module, ep_groups)


@torch.no_grad()
def save_moe_module(save_path, model, optimizer, opt_param_scheduler, iter_num, args):
    rank = torch.distributed.get_rank()
    pipeline_model = model.model if hasattr(model, "model") else model
    hybrid_parallel_configs = getattr(model, "hybrid_parallel_configs", None)
    if hybrid_parallel_configs is None and hasattr(model, "args"):
        hybrid_parallel_configs = get_hybrid_parallel_configs_api(model.args)

    if rank == 0:
        print("Begin to save ckpt")
        os.makedirs(save_path, exist_ok=True)
        if hybrid_parallel_configs is not None:
            json.dump(hybrid_parallel_configs, open(os.path.join(save_path, "hybrid_parallel_configs.json"), "w"))

        os.makedirs(os.path.join(save_path, f"iter_{iter_num}"), exist_ok=True)
        json.dump(
            opt_param_scheduler.state_dict(),
            open(os.path.join(save_path, f"iter_{iter_num}", "opt_param_scheduler.json"), "w"),
        )

    assert args.parallel.default_dp_type != "ddp", "Save / Load distributed checkpoint is not supported for DDP"

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        iter_path = os.path.join(save_path, f"iter_{iter_num}")
        for block in pipeline_model.model_cur_stage:
            block_module = block
            if isinstance(block_module, CheckpointWrapper):
                block_module = block_module._checkpoint_wrapped_module

            for m in block.modules():
                if not isinstance(m, FSDP):
                    continue

                wrapped_module = m._fsdp_wrapped_module
                if isinstance(wrapped_module, CheckpointWrapper):
                    wrapped_module = wrapped_module._checkpoint_wrapped_module

                state_dict = m.state_dict()
                if not state_dict:
                    continue

                if isinstance(wrapped_module, GalvatronEmbedding):
                    tp_rank = dist.get_rank(wrapped_module.tp_group)
                    os.makedirs(os.path.join(iter_path, embedding_name[:-3]), exist_ok=True)
                    torch.save(state_dict, os.path.join(iter_path, embedding_name[:-3], f"{tp_rank}.pt"))
                elif isinstance(wrapped_module, GalvatronFinalNorm):
                    torch.save(state_dict, os.path.join(iter_path, ln_f_name))
                elif isinstance(wrapped_module, GalvatronCausalLMHead):
                    tp_rank = dist.get_rank(wrapped_module.tp_group)
                    os.makedirs(os.path.join(iter_path, cls_name[:-3]), exist_ok=True)
                    torch.save(state_dict, os.path.join(iter_path, cls_name[:-3], f"{tp_rank}.pt"))
                elif isinstance(wrapped_module, GalvatronMoEAttention):
                    tp_rank = dist.get_rank(wrapped_module.attn.tp_group)
                    os.makedirs(os.path.join(iter_path, (attention_name % wrapped_module.layer_idx)[:-3]), exist_ok=True)
                    torch.save(
                        state_dict,
                        os.path.join(iter_path, (attention_name % wrapped_module.layer_idx)[:-3], f"{tp_rank}.pt"),
                    )
                    if hasattr(block_module, "router") and tp_rank == 0:
                        router_state_dict = {
                            key: value.detach().cpu() if torch.is_tensor(value) else value
                            for key, value in block_module.router.state_dict().items()
                        }
                        torch.save(router_state_dict, os.path.join(iter_path, router_name % wrapped_module.layer_idx))
                elif isinstance(wrapped_module, GalvatronMoEMLP):
                    tp_rank = dist.get_rank(wrapped_module.tp_of_ep_group)
                    ep_rank = dist.get_rank(wrapped_module.ep_group)
                    os.makedirs(os.path.join(iter_path, (mlp_name % wrapped_module.layer_idx)[:-3]), exist_ok=True)
                    torch.save(
                        state_dict,
                        os.path.join(iter_path, (mlp_name % wrapped_module.layer_idx)[:-3], f"{ep_rank}_{tp_rank}.pt"),
                    )

    optimizer_state_dict = optimizer.state_dict()
    os.makedirs(os.path.join(save_path, f"iter_{iter_num}", "optimizer"), exist_ok=True)
    torch.save(optimizer_state_dict, os.path.join(save_path, f"iter_{iter_num}", "optimizer", f"{rank}.pt"))

    torch.distributed.barrier()
    if rank == 0:
        print("Finish saving ckpt")
