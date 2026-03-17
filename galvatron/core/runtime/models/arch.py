"""Module registry and architecture metadata.

Central registry that maps declarative module type names (e.g. ``"decoder"``)
to their concrete ``nn.Module`` classes, plus ``ArchModelInfo`` which
auto-derives ModelInfo from an architecture list.
"""

from typing import Dict, List, Type

import torch.nn as nn

from galvatron.core.runtime.hybrid_parallel_config import ModelInfo, mixed_precision_dtype

from .modules import (
    GalvatronEmbedding,
    GalvatronDecoderLayer,
    GalvatronFinalNorm,
    GalvatronCausalLMHead,
)


# =========================================================================
# Type constants
# =========================================================================

_LAYER_MODULE_TYPES = {"decoder", "moe_decoder"}
"""Module types that count as repeating "layers" for parallel config."""

_MODULE_TYPE_SUFFIX = {
    "embedding": "embed",
    "decoder": "gpt_dec",
    "moe_decoder": "gpt_dec",
    "prenorm": "norm",
    "lm_head": "cls",
}
"""Module type → suffix used by ``hp_config_whole_model``."""

MODULE_REGISTRY: Dict[str, Type[nn.Module]] = {
    "embedding": GalvatronEmbedding,
    "decoder": GalvatronDecoderLayer,
    "prenorm": GalvatronFinalNorm,
    "lm_head": GalvatronCausalLMHead,
}
"""Module type → concrete class."""


# =========================================================================
# Helpers
# =========================================================================

def arch_to_module_types(arch_list: List[str]) -> List[str]:
    """Convert an architecture list to the ``module_types`` format expected by Galvatron."""
    return [_MODULE_TYPE_SUFFIX.get(t, t) for t in arch_list]


# =========================================================================
# Auto-derived ModelInfo
# =========================================================================

class ArchModelInfo(ModelInfo):
    """``ModelInfo`` automatically derived from *arch_list* + *args*."""

    def __init__(self, arch_list: List[str], args, model_type="gpt"):
        super().__init__()
        m = args.model
        if model_type == "gpt":
            num_layers = sum(1 for t in arch_list if t in _LAYER_MODULE_TYPES)
            seq_len = args.train.seq_length
            hidden_size = m.hidden_size
            mp_dtype = mixed_precision_dtype(args.parallel.mixed_precision)

            if m.shape_order == "SBH":
                layer_shapes = [[[seq_len, -1, hidden_size]]]
            else:
                layer_shapes = [[[-1, seq_len, hidden_size]]]

            module_types = arch_to_module_types(arch_list)

            self.set_layernums([num_layers])
            self.set_shapes(layer_shapes)
            self.set_dtypes([[mp_dtype]])
            self.set_module_types(module_types)
        else:
            assert False, "Unknown model type: " + model_type
