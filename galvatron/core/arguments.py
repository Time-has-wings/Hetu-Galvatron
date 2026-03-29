from pathlib import Path
from typing import Any, Dict, List, Optional

from galvatron.core.args_schema import CoreArgs
from galvatron.core.runtime.args_schema import (
    CommonTrainArgs,
    GalvatronModelArgs,
    GalvatronParallelArgs,
    GalvatronProfileArgs,
)
from omegaconf import OmegaConf
import torch


def _coerce_cli_value(raw: str) -> Any:
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _legacy_cli_to_flat_map(tokens: List[str]) -> Dict[str, Any]:
    """Parse `--key value` / `--flag` legacy argv tail."""
    out: Dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:].replace("-", "_")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            out[key] = _coerce_cli_value(tokens[i + 1])
            i += 2
        else:
            out[key] = True
            i += 1
    return out


def _runtime_subsection_for_key(key: str) -> Optional[str]:
    if key in GalvatronParallelArgs.model_fields:
        return "parallel"
    if key in GalvatronModelArgs.model_fields:
        return "model"
    if key in GalvatronProfileArgs.model_fields:
        return "profile"
    if key in CommonTrainArgs.model_fields:
        return "train"
    return None


def _legacy_cli_to_hydra_overrides(tokens: List[str]) -> List[str]:
    """Convert legacy `--key value` args to Hydra `runtime.x.y=value` overrides."""
    flat = _legacy_cli_to_flat_map(tokens)
    aliases = {
        "global_train_batch_size": ("train", "global_batch_size"),
        "adam_weight_decay": ("train", "weight_decay"),
    }
    skip = {"model_name", "epochs"}
    converted: List[str] = []
    for key, value in flat.items():
        if key in skip:
            continue
        if key in aliases:
            section, field = aliases[key]
        else:
            section = _runtime_subsection_for_key(key)
            field = key
        if section is None:
            continue
        # Use `++` so Hydra can both override existing keys and add missing keys.
        converted.append(f"++runtime.{section}.{field}={value}")
    return converted


def _normalize_runtime_model_dtype(config_dict: Dict[str, Any]) -> None:
    """Normalize runtime.model.params_dtype from string to torch.dtype."""
    runtime = config_dict.get("runtime")
    if not isinstance(runtime, dict):
        return
    model = runtime.get("model")
    if not isinstance(model, dict):
        return
    raw = model.get("params_dtype")
    if not isinstance(raw, str):
        return
    mapping = {
        "torch.float32": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "torch.float16": torch.float16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = raw.strip().lower()
    if key in mapping:
        model["params_dtype"] = mapping[key]


def _normalize_profiler_fields(config_dict: Dict[str, Any]) -> None:
    """Normalize profiler fields that may be auto-typed by Hydra."""
    profiler = config_dict.get("profiler")
    if not isinstance(profiler, dict):
        return
    seq_list = profiler.get("profile_seq_length_list")
    if isinstance(seq_list, int):
        profiler["profile_seq_length_list"] = str(seq_list)


def load_with_hydra(
    config_path: str,
    overrides: Optional[List[str]] = None,
    mode: Optional[str] = None,
    **hydra_kwargs: Any,
) -> CoreArgs:
    from hydra import compose, initialize_config_dir

    normalized_overrides = list(overrides or [])
    if mode == "train_dist" and normalized_overrides and normalized_overrides[0].startswith("--"):
        normalized_overrides = _legacy_cli_to_hydra_overrides(normalized_overrides)

    path = Path(config_path).resolve()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        cfg = compose(config_name=path.name, overrides=normalized_overrides, **hydra_kwargs)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    _normalize_runtime_model_dtype(config_dict)
    _normalize_profiler_fields(config_dict)
    args = CoreArgs(**config_dict)
    if mode == "train_dist":
        args = args.runtime
    elif mode == "profile":
        args = args.profiler
    elif mode == "search":
        args = args.search_engine
    return args