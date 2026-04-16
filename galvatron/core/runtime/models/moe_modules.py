import torch
import torch.nn as nn

from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
from galvatron.core.runtime.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from galvatron.core.runtime.transformer.mlp import MLPSubmodules
from galvatron.core.runtime.transformer.norm import GalvatronNorm
from galvatron.core.runtime.moe.router import TopKRouter
from galvatron.core.runtime.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from galvatron.core.runtime.moe.mlp import GroupedMLP, SequentialMLP

from .modules import GalvatronAttention


class GalvatronMoEAttention(nn.Module):
    def __init__(self, args: GalvatronRuntimeArgs, layer_idx, tp_group=None, sp_group=None, cp_group=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = GalvatronAttention(args, layer_idx, tp_group, sp_group, cp_group)
        self.pre_router_norm = GalvatronNorm(args.model, args.model.hidden_size, args.model.norm_epsilon)

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None):
        hidden_states = self.attn(hidden_states, position_ids, attention_mask, rotary_embedding)
        mlp_residual = hidden_states
        hidden_states = self.pre_router_norm(hidden_states)
        return hidden_states, mlp_residual


class GalvatronMoERouter(nn.Module):
    def __init__(self, args: GalvatronRuntimeArgs, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.init_method_std = args.train.init_method_std
        self.router = TopKRouter(config=args.model)
        self.router.set_layer_idx(layer_idx)
        if not self.router.weight.is_meta:
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=self.init_method_std)
        if getattr(self.router, "expert_bias", None) is not None:
            self.router.expert_bias.zero_()
        if getattr(self.router, "local_tokens_per_expert", None) is not None:
            self.router.local_tokens_per_expert.zero_()

    def forward(self, hidden_states):
        probs, routing_map = self.router(hidden_states)
        return probs, routing_map


# TODO: Add shared expert support
class GalvatronMoEMLP(nn.Module):
    def __init__(self, args: GalvatronRuntimeArgs, layer_idx, ep_group=None, tp_of_ep_group=None, tp_and_ep_group=None):
        super().__init__()
        self.layer_idx = layer_idx

        m = args.model

        self.ep_group = ep_group.group if ep_group is not None else None
        self.tp_of_ep_group = tp_of_ep_group.group if tp_of_ep_group is not None else None
        self.tp_and_ep_group = tp_and_ep_group.group if tp_and_ep_group is not None else None

        self.expert_parallel_size = torch.distributed.get_world_size(self.ep_group)
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        self.expert_parallel_rank = torch.distributed.get_rank(self.ep_group)
        assert self.expert_parallel_rank >= 0, "Expected non-negative expert parallel rank"

        assert m.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = m.num_moe_experts // self.expert_parallel_size

        local_expert_indices_offset = self.expert_parallel_rank * self.num_local_experts
        self.local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]
        assert all(map(lambda x: x < m.num_moe_experts, self.local_expert_indices))

        token_dispatcher_kwargs = {
            "num_local_experts": self.num_local_experts,
            "local_expert_indices": self.local_expert_indices,
            "config": m,
            "ep_group": self.ep_group,
            "tp_of_ep_group": self.tp_of_ep_group,
            "tp_and_ep_group": self.tp_and_ep_group,
            "layer_idx": self.layer_idx,
        }

        if m.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(**token_dispatcher_kwargs)
        elif m.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(**token_dispatcher_kwargs)
        elif m.moe_token_dispatcher_type == "alltoall_seq":
            assert False, "alltoall_seq is deprecated"
        elif m.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(**token_dispatcher_kwargs)
        else:
            raise ValueError(f"Unsupported MoE dispatcher type: {m.moe_token_dispatcher_type}")

        if m.moe_grouped_gemm:
            self.experts = GroupedMLP(
                num_local_experts=self.num_local_experts,
                config=m,
                tp_of_ep_group=self.tp_of_ep_group,
                layer_idx=self.layer_idx,
            )
        else:
            self.experts = SequentialMLP(
                num_local_experts=self.num_local_experts,
                config=m,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                ),
                tp_of_ep_group=self.tp_of_ep_group,
                tp_and_ep_group=self.tp_and_ep_group,
                layer_idx=self.layer_idx,
            )

    def forward(self, hidden_states, mlp_residual, probs, routing_map):
        dispatched_input, tokens_per_expert = self.token_dispatcher.token_permutation(
            hidden_states, probs, routing_map
        )
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        hidden_states, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        hidden_states = hidden_states + mlp_residual
        return hidden_states


class GalvatronMoEDecoderLayer(nn.Module):
    """Pre-norm decoder block = attention + router + MoE MLP."""

    def __init__(
        self,
        args: GalvatronRuntimeArgs,
        layer_idx,
        tp_group=None,
        sp_group=None,
        cp_group=None,
        ep_group=None,
        tp_of_ep_group=None,
        tp_and_ep_group=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = GalvatronMoEAttention(args, layer_idx, tp_group, sp_group, cp_group)
        self.router = GalvatronMoERouter(args, layer_idx)
        self.ffn = GalvatronMoEMLP(args, layer_idx, ep_group, tp_of_ep_group, tp_and_ep_group)

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None, rotary_embedding=None):
        hidden_states, mlp_residual = self.attn(hidden_states, position_ids, attention_mask, rotary_embedding)
        probs, routing_map = self.router(hidden_states)
        hidden_states = self.ffn(hidden_states, mlp_residual, probs, routing_map)
        return hidden_states
