"""Triton-fused vocab-parallel cross-entropy kernels.

Migrated from ``galvatron/site_package/megatron/core/fusions/triton_fused_cross_entropy.py``
so that the implementation lives inside the Galvatron runtime rather than the
vendored Megatron tree.  The Megatron file now re-exports from here.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

from galvatron.core.runtime.tensor_parallel.utils import VocabUtility


# ============================================================================
# Triton Kernels for Memory-Optimized Cross Entropy
# ============================================================================

@triton.jit
def _tiled_max_kernel(
    logits_ptr,      # [S, B, V] bf16
    max_ptr,         # [S, B] fp32
    seq_len,
    batch_size,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Tile-wise max reduction.

    bf16 → fp32 conversion only happens in SRAM; no full fp32 tensor is created
    in global memory.
    """
    pid = tl.program_id(0)
    batch_idx = pid % batch_size
    seq_idx   = pid // batch_size

    if seq_idx >= seq_len:
        return

    max_val = float('-inf')

    for vocab_offset in range(0, vocab_size, BLOCK_SIZE):
        vocab_indices = vocab_offset + tl.arange(0, BLOCK_SIZE)
        mask = vocab_indices < vocab_size
        logits_offset = seq_idx * batch_size * vocab_size + batch_idx * vocab_size + vocab_indices
        logits_bf16 = tl.load(logits_ptr + logits_offset, mask=mask, other=float('-inf'))
        logits_fp32 = logits_bf16.to(tl.float32)
        tile_max = tl.max(logits_fp32)
        max_val = tl.maximum(max_val, tile_max)

    token_offset = seq_idx * batch_size + batch_idx
    tl.store(max_ptr + token_offset, max_val)


@triton.jit
def _tiled_cross_entropy_forward_kernel(
    logits_ptr,           # [S, B, V] bf16
    target_ptr,           # [S, B] int64
    logits_max_ptr,       # [S, B] fp32 (already all-reduced)
    predicted_logits_ptr, # [S, B] fp32
    sum_exp_logits_ptr,   # [S, B] fp32
    seq_len,
    batch_size,
    vocab_size,
    vocab_start_idx,
    vocab_end_idx,
    BLOCK_SIZE: tl.constexpr,
):
    """Tile-wise forward: compute statistics without storing full fp32 exp_logits."""
    pid = tl.program_id(0)
    batch_idx = pid % batch_size
    seq_idx   = pid // batch_size

    if seq_idx >= seq_len:
        return

    token_offset = seq_idx * batch_size + batch_idx
    target      = tl.load(target_ptr + token_offset)
    logits_max  = tl.load(logits_max_ptr + token_offset)

    sum_exp        = 0.0
    predicted_logit = 0.0

    for vocab_offset in range(0, vocab_size, BLOCK_SIZE):
        vocab_indices = vocab_offset + tl.arange(0, BLOCK_SIZE)
        mask = vocab_indices < vocab_size
        logits_offset = seq_idx * batch_size * vocab_size + batch_idx * vocab_size + vocab_indices
        logits_bf16 = tl.load(logits_ptr + logits_offset, mask=mask, other=0.0)
        logits_fp32 = logits_bf16.to(tl.float32)
        exp_logits  = tl.exp(logits_fp32 - logits_max)
        sum_exp    += tl.sum(tl.where(mask, exp_logits, 0.0))
        global_vocab_indices = vocab_start_idx + vocab_indices
        target_in_tile = (global_vocab_indices == target) & mask
        predicted_logit += tl.sum(tl.where(target_in_tile, logits_fp32 - logits_max, 0.0))

    tl.store(predicted_logits_ptr + token_offset, predicted_logit)
    tl.store(sum_exp_logits_ptr   + token_offset, sum_exp)


@triton.jit
def _tiled_cross_entropy_backward_kernel(
    logits_ptr,        # [S, B, V] bf16
    target_ptr,        # [S, B] int64
    logits_max_ptr,    # [S, B] fp32
    sum_exp_logits_ptr,# [S, B] fp32 (all-reduced)
    grad_output_ptr,   # [S, B] fp32
    grad_logits_ptr,   # [S, B, V] bf16
    seq_len,
    batch_size,
    vocab_size,
    vocab_start_idx,
    vocab_end_idx,
    BLOCK_SIZE: tl.constexpr,
):
    """Tile-wise backward: recompute exp, compute grad = grad_out*(softmax - onehot)."""
    pid = tl.program_id(0)
    batch_idx = pid % batch_size
    seq_idx   = pid // batch_size

    if seq_idx >= seq_len:
        return

    token_offset = seq_idx * batch_size + batch_idx
    target     = tl.load(target_ptr        + token_offset)
    logits_max = tl.load(logits_max_ptr    + token_offset)
    sum_exp    = tl.load(sum_exp_logits_ptr + token_offset)
    grad_out   = tl.load(grad_output_ptr   + token_offset)

    for vocab_offset in range(0, vocab_size, BLOCK_SIZE):
        vocab_indices = vocab_offset + tl.arange(0, BLOCK_SIZE)
        mask = vocab_indices < vocab_size
        logits_offset = seq_idx * batch_size * vocab_size + batch_idx * vocab_size + vocab_indices
        logits_bf16 = tl.load(logits_ptr + logits_offset, mask=mask, other=0.0)
        logits_fp32 = logits_bf16.to(tl.float32)
        exp_logits  = tl.exp(logits_fp32 - logits_max)
        softmax     = exp_logits / sum_exp
        global_vocab_indices = vocab_start_idx + vocab_indices
        onehot = (global_vocab_indices == target).to(tl.float32)
        grad    = grad_out * (softmax - onehot)
        grad_bf16 = grad.to(tl.bfloat16)
        tl.store(grad_logits_ptr + logits_offset, grad_bf16, mask=mask)


# ============================================================================
# Python wrappers around Triton kernels
# ============================================================================

def tiled_max_reduction(
    vocab_parallel_logits: torch.Tensor,   # [S, B, V/TP] bf16
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:                          # [S, B] fp32
    """Tile-wise max reduction (bf16 → fp32 only in SRAM)."""
    seq_len, batch_size, vocab_size = vocab_parallel_logits.shape
    device = vocab_parallel_logits.device
    logits_max = torch.empty(seq_len, batch_size, dtype=torch.float32, device=device)
    grid = (seq_len * batch_size,)
    _tiled_max_kernel[grid](
        vocab_parallel_logits, logits_max,
        seq_len, batch_size, vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return logits_max


def tiled_cross_entropy_forward(
    vocab_parallel_logits: torch.Tensor,   # [S, B, V/TP] bf16
    target: torch.Tensor,                  # [S, B] int64
    logits_max: torch.Tensor,              # [S, B] fp32
    vocab_start_idx: int,
    vocab_end_idx: int,
    BLOCK_SIZE: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tile-wise forward; returns (predicted_logits, sum_exp_logits) in fp32."""
    seq_len, batch_size, vocab_size = vocab_parallel_logits.shape
    device = vocab_parallel_logits.device
    predicted_logits = torch.zeros(seq_len, batch_size, dtype=torch.float32, device=device)
    sum_exp_logits   = torch.zeros(seq_len, batch_size, dtype=torch.float32, device=device)
    grid = (seq_len * batch_size,)
    _tiled_cross_entropy_forward_kernel[grid](
        vocab_parallel_logits, target, logits_max,
        predicted_logits, sum_exp_logits,
        seq_len, batch_size, vocab_size,
        vocab_start_idx, vocab_end_idx,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return predicted_logits, sum_exp_logits


def tiled_cross_entropy_backward(
    vocab_parallel_logits: torch.Tensor,   # [S, B, V/TP] bf16
    target: torch.Tensor,                  # [S, B] int64
    logits_max: torch.Tensor,              # [S, B] fp32
    sum_exp_logits: torch.Tensor,          # [S, B] fp32
    grad_output: torch.Tensor,             # [S, B] fp32
    vocab_start_idx: int,
    vocab_end_idx: int,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:                          # [S, B, V/TP] bf16
    """Tile-wise backward: recomputes exp tile-by-tile, outputs bf16 gradients."""
    seq_len, batch_size, vocab_size = vocab_parallel_logits.shape
    device = vocab_parallel_logits.device
    grad_logits = torch.empty_like(vocab_parallel_logits)
    grid = (seq_len * batch_size,)
    _tiled_cross_entropy_backward_kernel[grid](
        vocab_parallel_logits, target, logits_max, sum_exp_logits, grad_output, grad_logits,
        seq_len, batch_size, vocab_size,
        vocab_start_idx, vocab_end_idx,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return grad_logits


# ============================================================================
# AutoGrad function & public API
# ============================================================================

class _VocabParallelCrossEntropyTritonFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, tp_group):
        logits_max = tiled_max_reduction(vocab_parallel_logits, BLOCK_SIZE=1024)
        torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)

        partition_vocab_size = vocab_parallel_logits.size()[-1]
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, tp_group.rank(), tp_group.size()
        )

        predicted_logits, sum_exp_logits = tiled_cross_entropy_forward(
            vocab_parallel_logits, target, logits_max,
            vocab_start_index, vocab_end_index, BLOCK_SIZE=1024,
        )
        torch.distributed.all_reduce(predicted_logits, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        torch.distributed.all_reduce(sum_exp_logits,   op=torch.distributed.ReduceOp.SUM, group=tp_group)

        loss = torch.log(sum_exp_logits) - predicted_logits

        ctx.save_for_backward(vocab_parallel_logits, target, logits_max, sum_exp_logits)
        ctx.vocab_start_index = vocab_start_index
        ctx.vocab_end_index   = vocab_end_index
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        vocab_parallel_logits, target, logits_max, sum_exp_logits = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_logits = tiled_cross_entropy_backward(
            vocab_parallel_logits, target, logits_max, sum_exp_logits,
            grad_output, ctx.vocab_start_index, ctx.vocab_end_index, BLOCK_SIZE=1024,
        )
        return grad_logits, None, None


def triton_fused_vocab_parallel_cross_entropy(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    tp_group,
) -> torch.Tensor:
    """Memory-optimised TP cross-entropy using Triton tile kernels.

    Args:
        vocab_parallel_logits: ``[S, B, V/TP]`` bf16
        target:                 ``[S, B]`` int64
        tp_group:               tensor-parallel process group
    Returns:
        loss: ``[S, B]`` fp32
    """
    return _VocabParallelCrossEntropyTritonFused.apply(vocab_parallel_logits, target, tp_group)
