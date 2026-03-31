"""Universal HuggingFace config <-> GalvatronModelArgs adapter.

Provides three ways to configure a model, all converging to ``args.model.*``:

1. **HF auto-detection**: set ``args.model.hf_model_name_or_path``
   → calls ``AutoConfig`` → fills ``args.model.*`` + auto-detects architecture.

2. **YAML template**: set ``args.model.model_config_path``
   → loads a YAML file whose field names match ``GalvatronModelArgs``
   → fills ``args.model.*``.  If the YAML also contains ``hf_model_name_or_path``,
   HF auto-detection runs first, then YAML fields override.

3. **Inline YAML**: fill ``runtime.model.*`` fields directly in the training YAML.

All three can be combined; priority (highest → lowest):
    inline YAML  >  model_config YAML  >  HF auto-detection  >  schema defaults

Single entry point: ``resolve_model_config(args)``
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Callable
from pydantic import ImportString
import torch
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs

if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = logging.getLogger(__name__)

# =========================================================================
# HF attribute alias table
# =========================================================================
_ATTR_ALIASES: Dict[str, List[str]] = {
    "hidden_size":           ["hidden_size", "n_embd", "d_model"],
    "num_layers":            ["num_hidden_layers", "n_layer", "num_layers"],
    "num_attention_heads":   ["num_attention_heads", "n_head", "num_heads"],
    "ffn_hidden_size":       ["intermediate_size", "n_inner", "ffn_dim", "d_ff"],
    "vocab_size":            ["vocab_size"],
    "num_key_value_heads":   ["num_key_value_heads"],
    "max_position_embeddings": ["max_position_embeddings", "n_positions",
                                "max_seq_len", "max_sequence_length"],
    "norm_eps":              ["rms_norm_eps", "layer_norm_epsilon",
                              "layer_norm_eps", "norm_epsilon", "norm_eps"],
}


def get_hf_attr(config, canonical_name: str, default=None):
    """Read a canonical attribute from any HF config by trying known aliases."""
    for alias in _ATTR_ALIASES.get(canonical_name, [canonical_name]):
        val = getattr(config, alias, None)
        if val is not None:
            return val
    return default


def set_hf_attr(config, canonical_name: str, value):
    """Write a value to whichever HF attribute name the config actually has."""
    for alias in _ATTR_ALIASES.get(canonical_name, [canonical_name]):
        if hasattr(config, alias):
            setattr(config, alias, value)
            return
    setattr(config, _ATTR_ALIASES[canonical_name][0], value)


# =========================================================================
# Architecture auto-detection from HF config
# =========================================================================

_ACTIVATION_MAP: Dict[Callable, tuple] = {
    "silu":       (torch.nn.functional.silu, True),
    "swiglu":     (torch.nn.functional.silu, True),
    "gelu":       (torch.nn.functional.gelu, False),
    "torch.nn.functional.silu": (torch.nn.functional.silu, True),
    "torch.nn.functional.gelu": (torch.nn.functional.gelu, False),
}


def _detect_normalization(hf_config) -> str:
    if hasattr(hf_config, "rms_norm_eps"):
        return "RMSNorm"
    return "LayerNorm"


def _detect_activation(hf_config) -> tuple:
    act_name = getattr(hf_config, "hidden_act", None) or \
               getattr(hf_config, "activation_function", None) or "gelu"
    act_name = act_name.lower().replace("-", "_")
    return _ACTIVATION_MAP.get(act_name, (torch.nn.functional.gelu, False))


def _detect_position_embedding_type(hf_config) -> str:
    pe_type = getattr(hf_config, "position_embedding_type", None)
    if pe_type == "rope" or hasattr(hf_config, "rope_theta") or hasattr(hf_config, "rope_scaling"):
        return "rope"
    if pe_type == "mrope":
        return "mrope"
    if pe_type == "relative":
        return "relative"
    if hasattr(hf_config, "n_positions") and not hasattr(hf_config, "rope_theta"):
        return "learned_absolute"
    if hasattr(hf_config, "max_position_embeddings") and hasattr(hf_config, "rotary_pct"):
        return "rope"
    if hasattr(hf_config, "max_position_embeddings"):
        return "rope"
    return "none"


# =========================================================================
# YAML model config loading
# =========================================================================

# Fields from YAML template that map directly to args.model.*
_YAML_TO_MODEL_FIELDS = {
    "model_size",
    "hidden_size", "num_layers", "num_attention_heads", "num_query_groups",
    "ffn_hidden_size", "vocab_size", "kv_channels",
    "normalization", "norm_epsilon", "activation_func", "gated_linear_unit",
    "position_embedding_type", "rotary_base", "rotary_percent",
    "rotary_interleaved", "apply_rope_fusion",
    "add_bias_linear", "add_qkv_bias", "qk_layernorm",
    "untie_embeddings_and_output_weights", "make_vocab_size_divisible_by",
    # MoE fields
    "num_moe_experts", "moe_ffn_hidden_size", "moe_router_topk",
    "moe_shared_expert_intermediate_size",
}


def _load_yaml_model_config(yaml_path: str) -> dict:
    """Load a YAML model config file and return as dict."""
    import yaml
    resolved = os.path.expanduser(os.path.expandvars(yaml_path))
    if not os.path.isabs(resolved):
        resolved = os.path.abspath(resolved)
    with open(resolved, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def _apply_yaml_to_model_args(args:GalvatronRuntimeArgs, yaml_data: dict):
    """Apply non-null YAML values onto ``args.model.*``.

    Only overwrites fields that are still at their default (None) in args.model,
    unless the field is an architecture-type field (normalization, activation, etc.)
    which always gets written.
    """
    m = args.model

    # Architecture fields that should always be written from YAML
    _always_write = {
        "normalization", "activation_func", "gated_linear_unit",
        "position_embedding_type", "apply_rope_fusion",
        "add_bias_linear", "add_qkv_bias",
        "untie_embeddings_and_output_weights",
    }

    for key, val in yaml_data.items():
        if val is None:
            continue
        if key not in _YAML_TO_MODEL_FIELDS:
            continue
        current = getattr(m, key, None)
        if key in _always_write or current is None:
            setattr(m, key, val)


# =========================================================================
# HF config → args.model.* population
# =========================================================================

def populate_model_args_from_hf(args) -> "PretrainedConfig":
    """Load HF config from ``args.model.hf_model_name_or_path`` and populate ``args.model.*``.

    Returns the loaded ``PretrainedConfig``.
    """
    from transformers import AutoConfig

    path = args.model.hf_model_name_or_path
    if path is None:
        raise ValueError("args.model.hf_model_name_or_path must be set.")
    hf_config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    _fill_model_args_from_hf(args, hf_config)
    return hf_config


def _fill_model_args_from_hf(args, hf_config):
    """Internal: populate ``args.model.*`` from an HF PretrainedConfig."""
    m = args.model

    if m.hidden_size is None:
        m.hidden_size = get_hf_attr(hf_config, "hidden_size")
    if m.num_layers is None:
        m.num_layers = get_hf_attr(hf_config, "num_layers")
    if m.num_attention_heads is None:
        m.num_attention_heads = get_hf_attr(hf_config, "num_attention_heads")
    if m.ffn_hidden_size is None:
        m.ffn_hidden_size = get_hf_attr(hf_config, "ffn_hidden_size")
    if m.vocab_size is None:
        m.vocab_size = get_hf_attr(hf_config, "vocab_size")
    if m.num_query_groups is None:
        kv_heads = get_hf_attr(hf_config, "num_key_value_heads")
        if kv_heads is not None and kv_heads != m.num_attention_heads:
            m.num_query_groups = kv_heads
    if m.norm_epsilon is None:
        m.norm_epsilon = get_hf_attr(hf_config, "norm_eps", 1e-5)
    if m.kv_channels is None and m.hidden_size and m.num_attention_heads:
        m.kv_channels = m.hidden_size // m.num_attention_heads

    if hasattr(args, "train") and args.train.seq_length is None:
        seq = get_hf_attr(hf_config, "max_position_embeddings")
        if seq is not None:
            args.train.seq_length = seq

    # Architecture-detection: always auto-detect from HF
    m.normalization = _detect_normalization(hf_config)
    act_func, gated = _detect_activation(hf_config)
    m.activation_func = act_func
    m.gated_linear_unit = gated
    m.position_embedding_type = _detect_position_embedding_type(hf_config)

    if m.position_embedding_type == "rope":
        m.apply_rope_fusion = True
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is not None:
            m.rotary_base = int(rope_theta)

    bias = getattr(hf_config, "attention_bias", None)
    if bias is not None:
        m.add_qkv_bias = bias
    mlp_bias = getattr(hf_config, "mlp_bias", None)
    if mlp_bias is not None:
        m.add_bias_linear = mlp_bias

    tie_word = getattr(hf_config, "tie_word_embeddings", True)
    m.untie_embeddings_and_output_weights = not tie_word

    hf_model_type = getattr(hf_config, "model_type", None)
    if hf_model_type and m.model_size is None:
        m.model_size = hf_model_type

    logger.info(
        "Populated args.model from HF config (%s): hidden=%s, layers=%s, heads=%s, "
        "ffn=%s, vocab=%s, norm=%s, act=%s, pos=%s",
        type(hf_config).__name__, m.hidden_size, m.num_layers,
        m.num_attention_heads, m.ffn_hidden_size, m.vocab_size,
        m.normalization, act_func, m.position_embedding_type,
    )


# =========================================================================
# Unified entry point
# =========================================================================

def resolve_model_config(args:GalvatronRuntimeArgs) -> Optional["PretrainedConfig"]:
    """One-call entry point: resolve model config from all sources.

    Priority (highest wins):
        1. Inline fields already set in ``args.model.*`` (from training YAML)
        2. ``args.model.model_config_path`` (YAML template file)
        3. ``args.model.hf_model_name_or_path`` (HuggingFace auto-detection)
        4. Schema defaults

    Returns the HF ``PretrainedConfig`` if HF auto-detection was used,
    otherwise ``None``.
    """
    hf_config = None
    m = args.model

    # --- Step 1: Load YAML template (if specified) ---
    yaml_data = {}
    if m.model_config_path is not None:
        yaml_data = _load_yaml_model_config(m.model_config_path)
        # If YAML contains hf_model_name_or_path, use it (unless inline already set)
        if m.hf_model_name_or_path is None and yaml_data.get("hf_model_name_or_path"):
            m.hf_model_name_or_path = yaml_data["hf_model_name_or_path"]

    # --- Step 2: HF auto-detection (if hf path is set) ---
    if m.hf_model_name_or_path is not None:
        hf_config = populate_model_args_from_hf(args)

    # --- Step 3: Apply YAML template fields (overrides HF defaults for arch fields) ---
    if yaml_data:
        _apply_yaml_to_model_args(args, yaml_data)

    # --- Step 4: Derive computed fields ---
    if m.kv_channels is None and m.hidden_size and m.num_attention_heads:
        m.kv_channels = m.hidden_size // m.num_attention_heads

    if m.model_size is None and m.hf_model_name_or_path:
        m.model_size = m.hf_model_name_or_path.split("/")[-1]
    
    if isinstance(m.activation_func, str):
        m.activation_func = _ACTIVATION_MAP.get(m.activation_func, (torch.nn.functional.gelu, False))[0]

    return hf_config


# =========================================================================
# Reconstruct HF config from args.model.*
# =========================================================================

def create_hf_config(args, hf_config_class=None) -> "PretrainedConfig":
    """Reconstruct an HF ``PretrainedConfig`` from ``args.model.*``.

    If ``hf_model_name_or_path`` is set, loads the base HF config and overrides.
    Otherwise uses *hf_config_class* to build from scratch.
    """
    from transformers import AutoConfig

    m = args.model
    if m.hf_model_name_or_path is not None:
        hf_config = AutoConfig.from_pretrained(m.hf_model_name_or_path, trust_remote_code=True)
    elif hf_config_class is not None:
        hf_config = hf_config_class()
    else:
        raise ValueError("Either hf_model_name_or_path or hf_config_class must be provided.")

    if m.hidden_size is not None:
        set_hf_attr(hf_config, "hidden_size", m.hidden_size)
    if m.num_layers is not None:
        set_hf_attr(hf_config, "num_layers", m.num_layers)
    if m.num_attention_heads is not None:
        set_hf_attr(hf_config, "num_attention_heads", m.num_attention_heads)
    if m.ffn_hidden_size is not None:
        set_hf_attr(hf_config, "ffn_hidden_size", m.ffn_hidden_size)
    if m.vocab_size is not None:
        set_hf_attr(hf_config, "vocab_size", m.vocab_size)
    if hasattr(args, "train") and args.train.seq_length is not None:
        set_hf_attr(hf_config, "max_position_embeddings", args.train.seq_length)

    hf_config.use_cache = False
    return hf_config


# =========================================================================
# Convenience helpers
# =========================================================================

def model_name(args) -> str:
    """Return a human-readable model identifier from ``args.model``."""
    name = args.model.model_size or args.model.hf_model_name_or_path or "unknown"
    name = name.split("/")[-1]
    if hasattr(args, "profile"):
        if getattr(args.profile, "profile_mode", "sequence") != "sequence":
            seq = args.train.seq_length or 0
            # return f"{name}_seqlen{seq}"
    return str(name)


def model_layer_configs(args) -> List[Dict[str, Any]]:
    """Return layer metadata expected by the Galvatron planner."""
    return [
        {
            "hidden_size": args.model.hidden_size,
            "seq_len": args.train.seq_length,
            "layer_num": args.model.num_layers,
        }
    ]
