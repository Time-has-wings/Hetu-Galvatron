# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import warnings
from copy import deepcopy
from math import ceil

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parameter import Parameter

from galvatron.core.runtime.parallel_state import get_parallel_world_size, get_parallel_rank
from galvatron.core.runtime.utils.utils import is_torch_min_version
from galvatron.core.runtime.args_schema import GalvatronModelArgs
from galvatron.core.runtime.tensor_parallel.utils import divide
from galvatron.core.runtime.moe import grouped_gemm_util as gg
from galvatron.core.runtime.transformer.fused_kernels import bias_geglu_impl, bias_gelu_impl, bias_swiglu_impl
from galvatron.core.runtime.tensor_parallel.mlp import MLP, MLPSubmodules
from galvatron.core.runtime.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    copy_to_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
)

class GroupedMLP(torch.nn.Module):
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(self, num_local_experts: int, config: GalvatronModelArgs, tp_of_ep_group: dist.ProcessGroup = None):
        super().__init__()
        self.config: GalvatronModelArgs = config
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        assert (
            config.add_bias_linear == False
        ), "bias not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        # self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            @torch.compile
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = get_parallel_world_size(tp_of_ep_group)
        tp_rank = get_parallel_rank(tp_of_ep_group)

        fc1_output_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        self.weight1 = Parameter(
            torch.empty(
                self.config.hidden_size,
                fc1_output_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )
        self.weight2 = Parameter(
            torch.empty(
                fc2_input_size_per_partition,
                self.config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )

    def forward(self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        """Forward step of the GroupedMLP."""
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )

            intermediate_parallel = self.activation_func(fc1_output)

            fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func(h)
            h = torch.matmul(h, w2)

            fc2_output = h

        return fc2_output, None

class SequentialMLP(torch.nn.Module):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    def __init__(
        self, 
        num_local_experts, 
        config: GalvatronModelArgs, 
        submodules: MLPSubmodules, 
        tp_of_ep_group: dist.ProcessGroup = None,
        tp_and_ep_group: dist.ProcessGroup = None,
    ):

        if config.moe_ffn_hidden_size == config.ffn_hidden_size:
            expert_config = config
        else:
            # Local SequentialMLP can still be used here by overriding the ffn_hidden_size
            # with a deepcopied config.
            expert_config = deepcopy(config)
            expert_config.ffn_hidden_size = config.moe_ffn_hidden_size
        super().__init__()

        self.config = expert_config
        self.add_bias = config.add_bias_linear
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()

        for _ in range(self.num_local_experts):
            expert = MLP(expert_config, submodules, is_expert=True, tp_group = tp_of_ep_group, tp_and_ep_group = tp_and_ep_group)
            self.local_experts.append(expert)

    def _pad_tensor_for_fp8(self, hidden):
        """Padding tensor shape to multiples of 16."""
        actual_num_tokens = hidden.shape[0]
        divisor = 16
        padded_num_tokens = ceil(actual_num_tokens / divisor) * divisor - actual_num_tokens
        if padded_num_tokens > 0:
            pad_tensor = torch.zeros(
                padded_num_tokens, hidden.shape[1], dtype=hidden.dtype, device=hidden.device
            )
            hidden = torch.cat((hidden, pad_tensor), dim=0)
        return hidden

    def forward(self, permuted_local_hidden_states: torch.Tensor, tokens_per_expert: torch.Tensor):
        """Forward step of the SequentialMLP."""
        if self.num_local_experts == 1:
            # if self.config.fp8:
            #     hidden = self._pad_tensor_for_fp8(permuted_local_hidden_states)
            #     output, output_bias = self.local_experts[0](hidden)
            #     output = output[: permuted_local_hidden_states.shape[0]]
            # else:
            output, output_bias = self.local_experts[0](permuted_local_hidden_states)

            return output, output_bias
        else:
            tokens_per_expert = tokens_per_expert.tolist()
            tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert)

            output_local_list = []
            output_bias_list = []

            for expert, tokens in zip(self.local_experts, tokens_list):
                # if self.config.fp8:
                #     hidden = self._pad_tensor_for_fp8(tokens)
                #     output, output_bias = expert(hidden)
                #     output = output[: tokens.shape[0]]
                # else:
                output, output_bias = expert(tokens)
                output_local_list.append(output)
                if self.add_bias:
                    output_bias_list.append(output_bias.expand_as(output))

            output_local = torch.cat(output_local_list, dim=0)
            if self.add_bias:
                output_bias_local = torch.cat(output_bias_list, dim=0)
            else:
                output_bias_local = None

            return output_local, output_bias_local


# TODO: Test correctness of shared expert MLP
class SharedExpertMLP(MLP):
    """
    MLP layer for Shared Experts.
    """

    # This stream is used when '--moe-shared-expert-overlap' is set.
    # The shared experts are scheduled into this stream to be overlapped with the dispatcher.
    stream = None

    def __init__(self, config: GalvatronModelArgs, submodules: MLPSubmodules, gate: bool, tp_group: dist.ProcessGroup = None, attn_tp_group: dist.ProcessGroup = None):
        self.tp_group = tp_group
        config = deepcopy(config)
        assert config.add_bias_linear == False, "bias is not supported in the shared experts, "
        "please set '--disable-bias-linear' instead."

        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        super().__init__(config=config, submodules=submodules, tp_group=tp_group)

        self.use_shared_expert_gate = gate
        if self.use_shared_expert_gate:
            # TODO: Add support for GPU initialization, which requires updating the golden values.
            self.gate_weight = torch.nn.Parameter(torch.empty((1, self.config.hidden_size)))
            self.gate_weight.data = self.gate_weight.data.to(dtype=config.params_dtype)
            # setattr(self.gate_weight, 'sequence_parallel', self.config.sequence_parallel)
        else:
            self.gate_weight = None

        if self.config.moe_shared_expert_overlap:
            # disable TP related AG/RS communications in the linear module
            for linear in [self.linear_fc1, self.linear_fc2]:
                if hasattr(linear, 'parallel_mode'):
                    # TELinear
                    linear.parallel_mode = None
                else:
                    # MCore legacy Linear
                    linear.explicit_expert_comm = True

            # The overlapped version is splitted into some separated functions and is put inside
            # the token dispatcher. These functions should be called in this order and no one can
            # be skipped:
            #     pre_forward_comm(input)
            #     linear_fc1_forward_and_act()
            #     linear_fc2_forward()
            #     post_forward_comm()
            #     output = get_output()
            #
            # We use cached intermediate results to avoid messy arg passing in the dispatcher.
            self.cached_fc1_input = None
            self.cached_fc2_input = None
            self.cached_fc2_output = None
            self.cached_output = None
            self.gate_score = None

            if self.stream is None:
                self.stream = torch.cuda.Stream()

    def forward(self, hidden_states):
        """Forward function"""
        output, _ = super().forward(hidden_states)
        if self.use_shared_expert_gate:
            logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
            gate_score = torch.nn.functional.sigmoid(logits)
            output = output * gate_score
        return output

    def pre_forward_comm(self, input):
        """
        All Gather for SP before forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_output is None
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            if self.use_shared_expert_gate:
                logits = torch.nn.functional.linear(input, self.gate_weight)
                self.gate_score = torch.nn.functional.sigmoid(logits)
            if self.config.sequence_parallel:
                self.cached_fc1_input = gather_from_sequence_parallel_region(
                    input, tensor_parallel_output_grad=True
                )
            else:
                self.cached_fc1_input = copy_to_tensor_model_parallel_region(input)
            set_tensor_grad_fn_sequence_sr(self.cached_fc1_input, torch.iinfo(torch.int).max)

    def linear_fc1_forward_and_act(self, overlapped_comm_output=None):
        """
        Do Linear FC1 and activation function forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc1_input is not None
        if overlapped_comm_output is not None:
            set_tensor_grad_fn_sequence_sr(overlapped_comm_output, torch.iinfo(torch.int).max)
        with torch.cuda.stream(self.stream):
            # [s, b, 4 * h/p]
            intermediate_parallel, bias_parallel = self.linear_fc1(self.cached_fc1_input)
            self.cached_fc1_input = None

            if self.config.bias_activation_fusion:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
            else:
                if bias_parallel is not None:
                    intermediate_parallel = intermediate_parallel + bias_parallel
                if self.config.gated_linear_unit:

                    def glu(x):
                        x = torch.chunk(x, 2, dim=-1)
                        return self.config.activation_func(x[0]) * x[1]

                    intermediate_parallel = glu(intermediate_parallel)
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)

            self.cached_fc2_input = intermediate_parallel

    def linear_fc2_forward(self, overlapped_comm_output=None):
        """
        Do Linear FC2 forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc2_input is not None
        if overlapped_comm_output is not None:
            set_tensor_grad_fn_sequence_sr(overlapped_comm_output, torch.iinfo(torch.int).max)
        with torch.cuda.stream(self.stream):
            # [s, b, h]
            self.cached_fc2_output, _ = self.linear_fc2(self.cached_fc2_input)
            self.cached_fc2_input = None

    def post_forward_comm(self):
        """
        Reduce scatter for SP after forward.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_fc2_output is not None
        with torch.cuda.stream(self.stream):
            if self.config.sequence_parallel:
                self.cached_output = reduce_scatter_to_sequence_parallel_region(
                    self.cached_fc2_output
                )
            else:
                self.cached_output = reduce_from_tensor_model_parallel_region(
                    self.cached_fc2_output
                )
            self.cached_fc2_output = None
            set_tensor_grad_fn_sequence_sr(self.cached_output, torch.iinfo(torch.int).max)

    def get_output(self):
        """
        Gets the module forward output.
        This function is used to overlap shared experts with the dispatcher.
        It is only useful when --moe-shared-expert-overlap is set and may be changed.
        """
        assert self.config.moe_shared_expert_overlap
        assert self.cached_output is not None
        with torch.cuda.stream(self.stream):
            if self.use_shared_expert_gate:
                assert self.gate_score is not None
                output = self.cached_output * self.gate_score
                self.gate_score = None
            else:
                output = self.cached_output
            self.cached_output = None
        torch.cuda.current_stream().wait_stream(self.stream)
        return output


def set_tensor_grad_fn_sequence_sr(tensor, value):
    """
    Set sequence_sr for the grad_fn of a tensor to control the backward order.
    For older PyTorch version, do nothing (backward order is not changed).
    The bigger the value is, the earlier the grad_fn is scheduled.
    """
    if is_torch_min_version("2.2.0"):
        if tensor is not None and tensor.grad_fn is not None:
            tensor.grad_fn._set_sequence_nr(value)
    else:
        warnings.warn(
            "WARNING : PyTorch is too old to set sequence_sr and the performance may not "
            "be optimal. Please use PyTorch >= 2.2.0 for better performance."
        )
