"""High-level model construction API.

Provides functions to build hybrid-parallel models from a declarative
architecture list, as well as convenience helpers for profiling.

Key entry points:
    - ``build_model(args)``:  one-call model builder (resolve → arch → HP model)
    - ``build_sequential_from_arch(...)``:  lower-level PipeSequential builder
    - ``build_causal_lm_arch(args)``:  generate arch list for decoder-only LMs
    - ``get_hybrid_parallel_configs(args)``:  auto-derive HP configs
    - ``get_runtime_profiler(args, path)``:  create a RuntimeProfiler
"""

from typing import List
import torch.nn as nn
from dataclasses import dataclass

from galvatron.core.runtime.pipeline import PipeSequential
from galvatron.core.runtime.tensor_parallel.reset import colummn_row_reset_parameters

from .modules import (
    GalvatronEmbedding,
    GalvatronDecoderLayer,
    GalvatronAttention,
    GalvatronMLP,
    GalvatronFinalNorm,
    GalvatronCausalLMHead,
)
from .arch import (
    MODULE_REGISTRY,
    _LAYER_MODULE_TYPES,
    ArchModelInfo,
)


@dataclass
class BlockNames:
    wrap_block_name: List[nn.Module]
    wrap_checkpoint_block_name: List[nn.Module]
    wrap_other_block_name: List[nn.Module]
    all_block_name: List[nn.Module]


def build_sequential_from_arch(
    arch_list: List[str],
    args,
    tp_groups: List,
    sp_groups: List,
    cp_groups: List,
    ep_groups: List | None = None,
    tp_of_ep_groups: List | None = None,
    tp_and_ep_groups: List | None = None,
) -> PipeSequential:
    """Build a ``PipeSequential`` model directly from an architecture list.

    Each element in *arch_list* is mapped to a TP-aware module via
    ``MODULE_REGISTRY``.  Layer-type modules (``decoder``, ``moe_decoder``)
    receive an incrementing ``layer_number``; other modules do not.

    Args:
        arch_list: e.g. ``["embedding", "decoder", ..., "prenorm", "lm_head"]``
        args: Galvatron args (with ``args.model``, ``args.train``, ``args.parallel``)
        tp_groups: per-position TP comm groups
        sp_groups: per-position SP comm groups
        cp_groups: per-position CP comm groups

    Returns:
        A ``PipeSequential`` ready for pipeline-parallel wrapping.
    """
    seq = PipeSequential()
    layer_idx = 0

    for i, module_type in enumerate(arch_list):
        if module_type not in MODULE_REGISTRY:
            raise ValueError(
                f"Unknown module type '{module_type}'. "
                f"Available: {list(MODULE_REGISTRY.keys())}"
            )
        cls = MODULE_REGISTRY[module_type]

        if module_type in _LAYER_MODULE_TYPES:
            module = cls(
                args,
                layer_idx,
                tp_group=tp_groups[i],
                sp_group=sp_groups[i],
                cp_group=cp_groups[i],
            )
            layer_idx += 1
        elif module_type in ("embedding", "lm_head"):
            module = cls(
                args,
                tp_group=tp_groups[i],
                sp_group=sp_groups[i],
                cp_group=cp_groups[i],
            )
        elif module_type in ("prenorm"):
            module = cls(
                args,
            )
        else:
            assert False, "Unknown module type: " + module_type

        seq.add_module(f"{module_type}_{i}", module)
    return seq


def build_causal_lm_arch(args, model_type="gpt") -> List[str]:
    """Build architecture list for a standard decoder-only causal LM."""
    if model_type == "gpt":
        num_layers = args.model.num_layers
        return ["embedding"] + ["decoder"] * num_layers + ["prenorm", "lm_head"]
    else:
        assert False, "Unknown model type: " + model_type

def get_hybrid_parallel_configs(args):
    """Get hybrid parallel configs using auto-derived ModelInfo from args."""
    from galvatron.core.runtime.hybrid_parallel_config import get_hybrid_parallel_configs_api

    arch_list = build_causal_lm_arch(args, args.model.model_type)
    model_info = ArchModelInfo(arch_list, args, args.model.model_type)
    return get_hybrid_parallel_configs_api(args, model_info), model_info


def get_block_names(args):
    """Derive FSDP/checkpoint wrapping class lists from model type."""
    if args.model.model_type == "gpt":
        if args.model.is_moe_model:
            pass
            # TODO: add MoE block names
            # return BlockNames(
            #     wrap_block_name=[MoEAttention_tp, MoEMLP_tp],
            #     wrap_checkpoint_block_name=[MoEAttention_tp, MoEMLP_tp],
            #     wrap_other_block_name=[MoEEmbeddings_, MoEPreNorm_, MoECls_],
            #     all_block_name=[MoEEmbeddings_, MoEAttention_tp, MoEMLP_tp, MoEPreNorm_, MoECls_],
            # )
        else:
            # When profiling attention/MLP units separately, wrap the
            # attention and MLP blocks directly; otherwise wrap the whole
            # decoder layer as a unit.
            if args.profile.profile_unit in ("attention", "mlp"):
                return BlockNames(
                    wrap_block_name=[GalvatronAttention, GalvatronMLP],
                    wrap_checkpoint_block_name=[GalvatronAttention, GalvatronMLP],
                    wrap_other_block_name=[GalvatronEmbedding, GalvatronFinalNorm, GalvatronCausalLMHead],
                    all_block_name=[GalvatronEmbedding, GalvatronAttention, GalvatronMLP, GalvatronFinalNorm, GalvatronCausalLMHead],
                )
            else:
                return BlockNames(
                    wrap_block_name=[GalvatronDecoderLayer],
                    wrap_checkpoint_block_name=[GalvatronDecoderLayer],
                    wrap_other_block_name=[GalvatronEmbedding, GalvatronFinalNorm, GalvatronCausalLMHead],
                    all_block_name=[GalvatronEmbedding, GalvatronDecoderLayer, GalvatronFinalNorm, GalvatronCausalLMHead],
                )

def build_model(args):
    """One-call model builder: arch_list → hybrid-parallel model.

    Call ``resolve_model_config(args)`` before this to populate
    ``args.model.*`` from YAML / HF sources, or set them directly.
    """
    from galvatron.core.runtime.hybrid_parallel_model import construct_hybrid_parallel_model_api

    arch_list = build_causal_lm_arch(args)
    hp_configs, model_info = get_hybrid_parallel_configs(args)
    block_names = get_block_names(args)

    if args.local_rank == 0:
        print("Creating Model...")

    return construct_hybrid_parallel_model_api(
        arch_list=arch_list,
        args=args,
        hybrid_parallel_configs=hp_configs,
        model_info=model_info,
        layernorm_name=["input_layernorm" ,"post_attention_layernorm", "norm"],
        tied_wte_attr_names=["embed_tokens", "lm_head"] if args.model.untie_embeddings_and_output_weights else None,
        block_names=block_names,

    )


def get_runtime_profiler(args, path, start_iter=10, end_iter=20):
    """Create a ``RuntimeProfiler`` with model info derived from args."""
    from galvatron.core.profiler import RuntimeProfiler
    from galvatron.utils.hf_config_adapter import model_layer_configs, model_name

    profiler = RuntimeProfiler(args)
    profiler.set_profiler_dist(
        path, model_layer_configs(args), model_name(args),
        start_iter=start_iter, end_iter=end_iter,
    )
    return profiler
