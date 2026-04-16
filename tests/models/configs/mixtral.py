from dataclasses import dataclass
from typing import Dict


@dataclass
class MixtralConfig:
    dim: int = 128
    hidden_dim: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2
    n_layers: int = 2
    norm_eps: float = 1e-5
    vocab_size: int = 1000
    n_positions: int = 32
    num_local_experts: int = 4
    num_experts_per_tok: int = 2

    def to_dict(self) -> Dict:
        return {
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "n_layers": self.n_layers,
            "norm_eps": self.norm_eps,
            "vocab_size": self.vocab_size,
            "n_positions": self.n_positions,
            "num_local_experts": self.num_local_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
        }
