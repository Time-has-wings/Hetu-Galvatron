"""Random-token dataset and collate function for testing / debugging.

Generates random integer sequences that can be used as causal-LM inputs
without any real data or tokenizer dependency.
"""

import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset):
    """Dataset that produces random token sequences on GPU.

    Each sample has length ``seq_length + 1`` so that the collate function
    can split it into an input slice ``[:seq_length]`` and a label slice
    ``[1:]`` for next-token prediction.

    Args:
        vocab_size: Token vocabulary size (exclusive upper bound).
        seq_length: Model sequence length.  Stored samples are one token
            longer to allow the shift-by-one split in ``random_collate_fn``.
        size: Number of samples in the dataset.
    """

    def __init__(self, vocab_size: int, seq_length: int, size: int = 256):
        self.data = torch.randint(0, vocab_size, (size, seq_length + 1))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx].cuda()


def random_collate_fn(batch):
    """Collate for ``RandomTokenDataset``.

    Returns:
        tokens: ``(B, S)`` input ids.
        kwargs: dict with ``labels (B, S)`` and ``attention_mask = None``.
        loss_func: ``None`` — the Galvatron model uses its built-in loss.
    """
    tokens_ = torch.stack(batch, dim=0)
    tokens = tokens_[:, :-1].contiguous()
    labels = tokens_[:, 1:].contiguous()
    return tokens, {"labels": labels, "attention_mask": None}, None
