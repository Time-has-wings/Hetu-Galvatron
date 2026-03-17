"""Megatron-LM Utilities for models."""

import math
from typing import Sequence

import torch


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class VocabUtility:
    """Split the vocabulary into `world_size` chunks and return the first
    and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [fist, last)

    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        """Vocab range from per partition vocab size."""
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> Sequence[int]:
        """Vocab range from global vocab size."""
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )


def prepare_input_tensors_for_wgrad_compute(grad_output, all_gathered_input):
    """Ensure grad_output is stored in a contiguous buffer."""
    # Doing gather + slicing during the NeMo forward pass can make this tensor
    # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
    # clones it if it's not contiguous:
    # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
    grad_output = grad_output.contiguous()
    all_gathered_input = all_gathered_input.contiguous()
    # Convert the tensor shapes to 2D for execution compatibility
    if grad_output.dim() == 3:
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        all_gathered_input = all_gathered_input.view(
            all_gathered_input.shape[0] * all_gathered_input.shape[1], all_gathered_input.shape[2]
        )

    return grad_output, all_gathered_input