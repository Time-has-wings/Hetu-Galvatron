# modify from te 2.1

# TODO: update kernel to latest version of te
import torch
import triton
import triton.language as tl
from typing import Union, Tuple
import warnings

def moe_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: torch.Tensor = None,
    restore_shape: torch.Tensor = None,
    map_type: str = "mask",
    probs: torch.Tensor = None,
) -> torch.Tensor:
    """
    Unpermute a tensor with permuted tokens, and optionally merge the tokens with their
    corresponding probabilities.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor with permuted tokens of shape `[num_tokens, hidden_size]` to be unpermuted.
    row_id_map: torch.Tensor
        The tensor of a mapping table for sorted indices used to unpermute the tokens,
        which is the second output tensor of `Permute`.
    merging_probs: torch.Tensor, default = None
        The tensor of probabilities corresponding to the permuted tokens. If provided,
        the unpermuted tokens will be merged with their respective probabilities.
        By default, set to an empty tensor, which means that the tokens are directly merged by accumulation.
    restore_shape: torch.Tensor
        The output shape after the unpermute operation.
    map_type: str, default = 'mask'
        Type of the routing map tensor. Should be the same as the value passed to moe_permute.
        Options are: 'mask', 'index'.
    probs: torch.Tensor, default = None
        Renamed to merging_probs. Keep for backward compatibility.
    """
    if probs is not None:
        if merging_probs is not None:
            raise ValueError(
                "Both merging_probs and probs kwarg are provided. probs is deprecated."
            )
        warnings.warn("probs kwarg is deprecated. Use merging_probs kwarg instead.")
        merging_probs = probs
    if map_type == "index":
        assert False, "index type not support yet!"
        # return _moe_unpermute_index_map.apply(inp, row_id_map, merging_probs)
    if map_type == "mask":
        return _moe_unpermute_mask_map.apply(inp, row_id_map, merging_probs, restore_shape)
    raise ValueError("map_type should be one of 'mask' or 'index'")

class _moe_unpermute_mask_map(torch.autograd.Function):
    """functional Unpermute with mask router map"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        merging_probs: torch.Tensor,
        restore_shape: torch.Size,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            ctx.merging_probs = merging_probs
            return inp

        if restore_shape is None:
            restore_shape = inp.shape
        num_tokens, hidden_size = restore_shape
        num_experts = row_id_map.size(0)

        with_probs = merging_probs is not None
        if with_probs:
            assert merging_probs.is_cuda, "TransformerEngine needs CUDA."

        # Device check
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert row_id_map.is_cuda, "TransformerEngine needs CUDA."

        unpermuted_output, _ = triton_unpermute_with_mask_map(
            inp,
            row_id_map,
            merging_probs,
            None,
            num_tokens,
            num_experts,
            hidden_size,
        )
        if with_probs:
            ctx.save_for_backward(inp, row_id_map, merging_probs)
        else:
            ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.num_permuted_tokens = inp.size(0)
        ctx.hidden_size = hidden_size
        ctx.with_probs = with_probs
        return unpermuted_output

    @staticmethod
    def backward(ctx, unpermuted_act_grad):
        # pylint: disable=missing-function-docstring
        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, None, ctx.merging_probs, None

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            if ctx.with_probs:
                fwd_input, row_id_map, merging_probs = ctx.saved_tensors
            else:
                (row_id_map,) = ctx.saved_tensors

            if ctx.with_probs:
                act_grad, probs_grad = (
                    triton_unpermute_with_mask_map_bwd_with_merging_probs(
                        unpermuted_act_grad,
                        row_id_map,
                        fwd_input,
                        merging_probs,
                        ctx.num_tokens,
                        ctx.num_experts,
                        ctx.num_permuted_tokens,
                        ctx.hidden_size,
                    )
                )
            else:
                assert False, "no probs not support yet!"
                # act_grad, _ = triton_permute_with_mask_map(
                #     unpermuted_act_grad,
                #     row_id_map,
                #     None,
                #     ctx.num_tokens,
                #     ctx.num_experts,
                #     ctx.num_permuted_tokens,
                #     ctx.hidden_size,
                # )

        if not ctx.needs_input_grad[2]:
            probs_grad = None
        return act_grad, None, probs_grad, None

def triton_unpermute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Union[torch.Tensor, None],
    permuted_probs: Union[torch.Tensor, None],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
):
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if permuted_probs is not None:
        unpermuted_probs = torch.empty(
            (num_tokens, num_experts), dtype=permuted_probs.dtype, device="cuda"
        )
    else:
        unpermuted_probs = None
    grid = (num_tokens,)
    _unpermute_kernel[grid](
        inp,
        output,
        row_id_map,
        merging_probs,
        permuted_probs,
        unpermuted_probs,
        num_tokens,
        num_experts,
        hidden_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        merging_probs.stride(0) if merging_probs is not None else None,
        merging_probs.stride(1) if merging_probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        unpermuted_probs.stride(0) if unpermuted_probs is not None else None,
        unpermuted_probs.stride(1) if unpermuted_probs is not None else None,
        WITH_MERGING_PROBS=merging_probs is not None,
        PERMUTE_PROBS=permuted_probs is not None,
    )
    return output, unpermuted_probs

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["hidden_size"],
)
@triton.jit
def _unpermute_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    merging_probs_ptr,
    permuted_probs_ptr,
    unpermuted_probs_ptr,
    # sizes
    num_tokens,
    num_experts,
    hidden_size,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_permuted_probs_token,
    stride_unpermuted_probs_token,
    stride_unpermuted_probs_expert,
    # metas
    WITH_MERGING_PROBS: tl.constexpr,
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type = input_ptr.dtype.element_ty
    compute_type = tl.float32

    pid = tl.program_id(0)
    current_start = 0
    while current_start < hidden_size:
        current_offset = current_start + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < hidden_size
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
        for expert_idx in range(num_experts):
            src_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid)
            if src_row != -1:
                input_off = src_row * stride_input_token + current_offset * stride_input_hidden
                inp = tl.load(input_ptr + input_off, mask=mask)
                inp = inp.to(compute_type)
                if WITH_MERGING_PROBS:

                    merging_prob_off = (
                        pid * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
                    )
                    merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
                    inp *= merging_prob
                accumulator += inp
            if PERMUTE_PROBS:
                if current_start == 0:
                    unpermuted_prob_off = (
                        pid * stride_unpermuted_probs_token
                        + expert_idx * stride_unpermuted_probs_expert
                    )
                    if src_row != -1:
                        permuted_prob_off = src_row * stride_permuted_probs_token
                        prob = tl.load(permuted_probs_ptr + permuted_prob_off)
                        tl.store(unpermuted_probs_ptr + unpermuted_prob_off, prob)
                    else:
                        tl.store(unpermuted_probs_ptr + unpermuted_prob_off, 0.0)
        accumulator = accumulator.to(data_type)
        output_off = pid * stride_output_token + current_offset * stride_output_hidden
        tl.store(output_ptr + output_off, accumulator, mask=mask)
        current_start += BLOCK_SIZE

def triton_unpermute_with_mask_map_bwd_with_merging_probs(
    fwd_output_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    fwd_input: torch.Tensor,
    merging_probs: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
):
    act_grad = torch.empty(
        (num_out_tokens, hidden_size), dtype=fwd_output_grad.dtype, device="cuda"
    )
    merging_probs_grad = torch.empty(
        (num_tokens, num_experts), dtype=merging_probs.dtype, device="cuda"
    )
    grid = (num_tokens,)
    _unpermute_bwd_with_merging_probs_kernel[grid](
        fwd_output_grad,
        act_grad,
        fwd_input,
        merging_probs,
        merging_probs_grad,
        row_id_map,
        num_tokens,
        num_experts,
        hidden_size,
        fwd_output_grad.stride(0),
        fwd_output_grad.stride(1),
        act_grad.stride(0),
        act_grad.stride(1),
        fwd_input.stride(0),
        fwd_input.stride(1),
        merging_probs.stride(0),
        merging_probs.stride(1),
        merging_probs_grad.stride(0),
        merging_probs_grad.stride(1),
    )
    return act_grad, merging_probs_grad

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["hidden_size"],
)
@triton.jit
def _unpermute_bwd_with_merging_probs_kernel(
    # pointers
    fwd_output_grad_ptr,
    fwd_input_grad_ptr,
    fwd_input_ptr,
    merging_probs_ptr,
    merging_probs_grad_ptr,
    row_id_map_ptr,
    # sizes
    num_tokens,
    num_experts,
    hidden_size,
    # strides
    stride_fwd_output_grad_token,
    stride_fwd_output_grad_hidden,
    stride_fwd_input_grad_token,
    stride_fwd_input_grad_hidden,
    stride_fwd_input_token,
    stride_fwd_input_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_merging_probs_grad_token,
    stride_merging_probs_grad_expert,
    # metas
    BLOCK_SIZE: tl.constexpr,
):
    data_type = fwd_output_grad_ptr.dtype.element_ty
    compute_type = tl.float32

    pid = tl.program_id(0)

    # add zero tensor
    zero_tensor = tl.zeros((1,), dtype=merging_probs_grad_ptr.dtype.element_ty)
    zero_val = tl.sum(zero_tensor).to(merging_probs_grad_ptr.dtype.element_ty)

    for expert_idx in range(num_experts):
        dst_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid)
        if dst_row != -1:
            prob_grad_accum = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
            current_start = 0
            while current_start < hidden_size:
                current_offset = current_start + tl.arange(0, BLOCK_SIZE)
                mask = current_offset < hidden_size
                input_off = (
                    pid * stride_fwd_output_grad_token
                    + current_offset * stride_fwd_output_grad_hidden
                )
                inp = tl.load(fwd_output_grad_ptr + input_off, mask=mask)
                inp = inp.to(compute_type)
                merging_prob_off = (
                    pid * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
                )
                merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
                output = inp * merging_prob
                output = output.to(data_type)
                output_off = (
                    dst_row * stride_fwd_input_grad_token
                    + current_offset * stride_fwd_input_grad_hidden
                )
                tl.store(fwd_input_grad_ptr + output_off, output, mask=mask)

                fwd_input_off = (
                    dst_row * stride_fwd_input_token + current_offset * stride_fwd_input_hidden
                )
                fwd_input = tl.load(fwd_input_ptr + fwd_input_off, mask=mask)
                prob_grad_accum += fwd_input.to(compute_type) * inp
                current_start += BLOCK_SIZE
            probs_grad = tl.sum(prob_grad_accum).to(merging_probs_grad_ptr.dtype.element_ty)
            probs_grad_off = (
                pid * stride_merging_probs_grad_token
                + expert_idx * stride_merging_probs_grad_expert
            )
            tl.store(merging_probs_grad_ptr + probs_grad_off, probs_grad)
        else:
            probs_grad_off = (
                pid * stride_merging_probs_grad_token
                + expert_idx * stride_merging_probs_grad_expert
            )
            # Modify 0.0 -> zero_val
            tl.store(merging_probs_grad_ptr + probs_grad_off, zero_val)

def moe_permute(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
    max_token_num: int = -1,
    map_type: str = "mask",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens based on the routing_map. Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    routing_map: torch.Tensor
        The token to expert mapping tensor.
        If map_type is 'mask', routing_map is of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
        If map_type is 'index', routing_map is of shape [num_tokens, topK] and dtype 'int32'.
        The values in it are the routed expert indices.
    num_out_tokens: int, default = -1
        The effective output token count, representing the number of tokens not dropped.
        By default, set to '-1', meaning no tokens are dropped.
    max_token_num: int, default = -1
        The maximum number of tokens, used for workspace allocation.
        By default, set to '-1', meaning the calculation of the size of workspace is
        automatically taken over by the operator.
    map_type: str, default = 'mask'
        Type of the routing map tensor.
        Options are: 'mask', 'index'.
        Refer to `routing_map` for more details.
    """
    if map_type == "index":
        assert False, "index type not support yet!"
        # return _moe_permute_index_map.apply(inp, routing_map, num_out_tokens, max_token_num)
    if map_type == "mask":
        output, row_id_map, _ = _moe_permute_mask_map.apply(inp, routing_map, num_out_tokens, None)
        return output, row_id_map
    raise ValueError("map_type should be one of 'mask' or 'index'")

class _moe_permute_mask_map(torch.autograd.Function):
    """functional Permute with mask router map"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        routing_map: torch.Tensor,
        num_out_tokens: int,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            ctx.probs = probs
            return inp, torch.tensor([], device=inp.device), torch.tensor([], device=inp.device)

        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert routing_map.is_cuda, "TransformerEngine needs CUDA."
        if probs is not None:
            assert probs.is_cuda, "TransformerEngine needs CUDA."

        assert inp.size(0) == routing_map.size(0), "Permute not possible"
        num_tokens, hidden_size = inp.size()
        num_experts = routing_map.size(1)
        assert (
            num_out_tokens is not None
        ), "num_out_tokens must be provided to the fused permute function."

        row_id_map = triton_make_row_id_map(routing_map, num_tokens, num_experts)

        output, permuted_probs = triton_permute_with_mask_map(
            inp,
            row_id_map,
            probs,
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
        )

        ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        return output, row_id_map, permuted_probs

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        _,
        permuted_probs_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, ctx.probs

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            (row_id_map,) = ctx.saved_tensors
            act_grad, probs_grad = triton_unpermute_with_mask_map(
                permuted_act_grad,
                row_id_map,
                None,
                permuted_probs_grad,
                ctx.num_tokens,
                ctx.num_experts,
                ctx.hidden_size,
            )
        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad

def triton_make_row_id_map(
    routing_map: torch.Tensor,
    num_tokens: int,
    num_experts: int,
):
    # pylint: disable=missing-function-docstring
    row_id_map = torch.empty((num_experts, num_tokens), dtype=torch.int64, device="cuda")
    block_size = 256
    grid = (num_experts, triton.cdiv(num_tokens, block_size))
    workspace_tensor = torch.empty(grid, dtype=torch.int64, device="cuda")
    # block cumsum
    _row_id_map_pass_1_kernel[grid](
        routing_map,
        row_id_map,
        workspace_tensor,
        num_tokens,
        routing_map.stride(0),
        routing_map.stride(1),
        block_size,
    )
    # cumsum all and process the mask
    _row_id_map_pass_2_kernel[grid](
        row_id_map,
        workspace_tensor,
        num_tokens,
        triton.next_power_of_2(num_experts * triton.cdiv(num_tokens, block_size)),
        block_size,
    )
    return row_id_map

@triton.jit
def _row_id_map_pass_1_kernel(
    # pointers
    routing_map_ptr,
    row_id_map_ptr,
    workspace_ptr,
    # sizes
    num_tokens,
    # strides
    stride_routing_map_token,
    stride_routing_map_expert,
    # metas
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert_token_mask = tl.load(
        routing_map_ptr + pid_m * stride_routing_map_expert + offset * stride_routing_map_token,
        mask=(offset < num_tokens),
        other=0,
    ).to(tl.int64)
    row_id_within_token_block = tl.cumsum(expert_token_mask) * expert_token_mask
    tl.store(
        row_id_map_ptr + pid_m * num_tokens + offset,
        row_id_within_token_block,
        mask=offset < num_tokens,
    )
    n_tokens_per_block = tl.sum(expert_token_mask)
    tl.store(workspace_ptr + pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n, n_tokens_per_block)

@triton.jit
def _row_id_map_pass_2_kernel(
    # pointers
    row_id_map_ptr,
    workspace_ptr,
    # sizes
    num_tokens,
    # metas
    WORKSPACE_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    chunk_idx = pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_id_within_token_block = tl.load(
        row_id_map_ptr + pid_m * num_tokens + offset, mask=(offset < num_tokens), other=0
    )

    workspace_off = tl.arange(0, WORKSPACE_LOAD_WIDTH)
    n_tokens_per_chunk = tl.load(workspace_ptr + workspace_off, mask=workspace_off < chunk_idx)
    row_id = tl.where(
        row_id_within_token_block == 0,
        -1,
        row_id_within_token_block + tl.sum(n_tokens_per_chunk) - 1,
    )
    tl.store(
        row_id_map_ptr + pid_m * num_tokens + offset,
        row_id,
        mask=(offset < num_tokens),
    )

def triton_permute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
):
    # pylint: disable=missing-function-docstring
    output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if probs is not None:
        permuted_probs = torch.empty((num_out_tokens,), dtype=probs.dtype, device="cuda")
    else:
        permuted_probs = None
    grid = (num_tokens,)
    _permute_kernel[grid](
        inp,
        output,
        row_id_map,
        probs,
        permuted_probs,
        num_tokens,
        num_experts,
        hidden_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        probs.stride(1) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        PERMUTE_PROBS=probs is not None,
    )
    return output, permuted_probs

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["hidden_size"],
)
@triton.jit
def _permute_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    probs_ptr,
    permuted_probs_ptr,
    # sizes
    num_tokens,
    num_experts,
    hidden_size,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_probs_expert,
    stride_permuted_probs_token,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_pos = 0
    while cur_pos < hidden_size:
        cur_off = cur_pos + tl.arange(0, BLOCK_SIZE)
        mask = cur_off < hidden_size
        input_off = pid * stride_input_token + cur_off * stride_input_hidden
        inp = tl.load(input_ptr + input_off, mask=mask)
        for expert_idx in range(num_experts):
            dst_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid)
            if dst_row != -1:
                output_off = dst_row * stride_output_token + cur_off * stride_output_hidden
                tl.store(output_ptr + output_off, inp, mask=mask)
                if PERMUTE_PROBS:
                    if cur_pos == 0:
                        prob_off = pid * stride_probs_token + expert_idx * stride_probs_expert
                        prob = tl.load(probs_ptr + prob_off)
                        permuted_prob_off = dst_row * stride_permuted_probs_token
                        tl.store(permuted_probs_ptr + permuted_prob_off, prob)
        cur_pos += BLOCK_SIZE


class _moe_chunk_sort(torch.autograd.Function):
    """functional MoE chunk permute"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        split_sizes: torch.Tensor,
        sorted_idxs: torch.Tensor,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            return inp, probs

        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert split_sizes.is_cuda, "TransformerEngine needs CUDA."
        assert sorted_idxs.is_cuda, "TransformerEngine needs CUDA."
        if probs is not None:
            assert probs.is_cuda, "TransformerEngine needs CUDA."

        num_tokens, hidden_size = inp.shape
        num_splits = split_sizes.size(0)
        assert num_splits == sorted_idxs.size(0)
        output, row_id_map, permuted_probs = sort_chunks_by_idx(
            inp,
            split_sizes,
            sorted_idxs,
            probs,
            num_tokens,
            hidden_size,
            num_splits,
        )
        ctx.save_for_backward(row_id_map)
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        return output, permuted_probs

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        permuted_probs_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, permuted_probs_grad

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            (row_id_map,) = ctx.saved_tensors
            act_grad, probs_grad = sort_chunks_by_map(
                permuted_act_grad,
                row_id_map,
                permuted_probs_grad,
                ctx.num_tokens,
                ctx.hidden_size,
            )
        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad


def moe_sort_chunks_by_index(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split and sort the input tensor based on the split_sizes and sorted indices.
    The inp tensor is splitted along dim-0 according to the split_sizes list and then sorted
    according to the sorted_indices.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    split_sizes: torch.Tensor
        Chunk sizes of the inp tensor along the 0-th dimension.
    sorted_indices: torch.Tensor
        Chunk indices used to permute the chunks.
    """
    output, _ = _moe_chunk_sort.apply(inp, split_sizes, sorted_index, None)
    return output


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["hidden_size"],
)
@triton.jit
def _sort_chunks_by_idxs_kernel(
    # pointers
    input_ptr,
    split_sizes_ptr,
    sorted_indices_ptr,
    output_ptr,
    dst_rows_ptr,
    probs_ptr,
    permuted_probs_ptr,
    # sizes
    num_splits,
    hidden_size,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_permuted_probs_token,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    IDX_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    load_split_offset = tl.arange(0, IDX_LOAD_WIDTH)
    sorted_indices = tl.load(
        sorted_indices_ptr + load_split_offset, mask=load_split_offset < num_splits
    )

    # get chunk idx of the current token in the input tensor
    input_chunk_idx = -1
    in_chunk_offset = tl.zeros([], dtype=tl.int64)
    acc_chunk_sizes = tl.zeros([], dtype=tl.int64)
    cursor = 0
    while cursor < num_splits:
        cur_chunk_size = tl.load(split_sizes_ptr + cursor).to(tl.int64)
        acc_chunk_sizes += cur_chunk_size
        if input_chunk_idx == -1 and acc_chunk_sizes > pid:
            input_chunk_idx = cursor
            in_chunk_offset = pid - (acc_chunk_sizes - cur_chunk_size)
        cursor += 1

    # get chunk idx of the current token in the output tensor
    output_chunk_idx = 0
    cursor = 0
    while cursor < num_splits:
        cur_input_idx = tl.load(sorted_indices_ptr + cursor)
        if cur_input_idx == input_chunk_idx:
            output_chunk_idx = cursor
        cursor += 1

    # make row_id_map
    output_split_sizes = tl.load(
        split_sizes_ptr + sorted_indices, mask=load_split_offset < num_splits
    ).to(tl.int64)
    output_pre_split_sizes = tl.where(load_split_offset < output_chunk_idx, output_split_sizes, 0)
    dst_row = tl.sum(output_pre_split_sizes) + in_chunk_offset
    tl.store(dst_rows_ptr + pid, dst_row)

    current_start = 0
    while current_start < hidden_size:
        current_offset = current_start + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < hidden_size
        input_offsets = pid * stride_input_token + current_offset * stride_input_hidden
        output_offsets = dst_row * stride_output_token + current_offset * stride_output_hidden
        inp = tl.load(input_ptr + input_offsets, mask=mask)
        tl.store(output_ptr + output_offsets, inp, mask=mask)
        current_start += BLOCK_SIZE

    if PERMUTE_PROBS:
        prob_off = pid * stride_probs_token
        prob = tl.load(probs_ptr + prob_off)
        permuted_prob_off = dst_row * stride_permuted_probs_token
        tl.store(permuted_probs_ptr + permuted_prob_off, prob)


def sort_chunks_by_idx(
    inp: torch.Tensor,
    split_sizes: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    num_splits: int,
):
    # pylint: disable=missing-function-docstring
    row_id_map = torch.empty((num_tokens,), dtype=torch.int64, device="cuda")
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if probs is not None:
        permuted_probs = torch.empty((num_tokens,), dtype=probs.dtype, device="cuda")
    else:
        permuted_probs = None
    grid = (num_tokens,)
    _sort_chunks_by_idxs_kernel[grid](
        inp,
        split_sizes,
        sorted_indices,
        output,
        row_id_map,
        probs,
        permuted_probs,
        num_splits,
        hidden_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        PERMUTE_PROBS=probs is not None,
        IDX_LOAD_WIDTH=triton.next_power_of_2(num_splits),
    )
    return output, row_id_map, permuted_probs


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["hidden_size"],
)
@triton.jit
def _sort_chunks_by_map(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    probs_ptr,
    permuted_probs_ptr,
    # sizes
    hidden_size,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_permuted_probs_token,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    dst_row = tl.load(row_id_map_ptr + pid)
    current_start = 0
    while current_start < hidden_size:
        current_offset = current_start + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < hidden_size
        input_offsets = dst_row * stride_input_token + current_offset * stride_input_hidden
        output_offsets = pid * stride_output_token + current_offset * stride_output_hidden
        inp = tl.load(input_ptr + input_offsets, mask=mask)
        tl.store(output_ptr + output_offsets, inp, mask=mask)
        current_start += BLOCK_SIZE
    if PERMUTE_PROBS:
        prob_off = dst_row * stride_probs_token
        prob = tl.load(probs_ptr + prob_off)
        permuted_prob_off = pid * stride_permuted_probs_token
        tl.store(permuted_probs_ptr + permuted_prob_off, prob)


def sort_chunks_by_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
):
    # pylint: disable=missing-function-docstring
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if probs is not None:
        permuted_probs = torch.empty((num_tokens,), dtype=probs.dtype, device="cuda")
    else:
        permuted_probs = None
    grid = (num_tokens,)
    _sort_chunks_by_map[grid](
        inp,
        output,
        row_id_map,
        probs,
        permuted_probs,
        hidden_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        PERMUTE_PROBS=probs is not None,
    )
    return output, permuted_probs
