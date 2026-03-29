import os
import sys
from typing import Dict, List

from galvatron.core.arguments import load_with_hydra
from galvatron.core.profiler import ModelProfiler
from galvatron.core.profiler.args_schema import ProfilerArgs
from galvatron.utils.hf_config_adapter import model_name, resolve_model_config


def _build_profile_args(config_path: str, overrides: List[str]) -> ProfilerArgs:
    core_args = load_with_hydra(config_path, overrides=overrides)
    if core_args.runtime is None:
        raise ValueError("Missing `runtime` section in profile config.")
    if core_args.profiler is None:
        raise ValueError("Missing `profiler` section in profile config.")

    runtime = core_args.runtime
    profiler = core_args.profiler
    resolve_model_config(runtime)

    merged: Dict[str, object] = {}
    merged.update(runtime.model.model_dump(exclude_none=True))
    merged.update(runtime.train.model_dump(exclude_none=True))
    merged.update(runtime.parallel.model_dump(exclude_none=True))
    merged.update(profiler.model_dump(exclude_none=True))

    # Runtime uses `global_batch_size`, profiler expects `profile_batch_size`.
    if merged.get("profile_batch_size") is None and runtime.train.global_batch_size is not None:
        merged["profile_batch_size"] = runtime.train.global_batch_size

    # Use runtime sequence length as default static profile sequence.
    if merged.get("profile_seq_length_list") is None and runtime.train.seq_length is not None:
        merged["profile_seq_length_list"] = str(runtime.train.seq_length)

    # Keep training/runtime semantics consistent.
    merged["mixed_precision"] = runtime.parallel.mixed_precision
    merged["sequence_parallel"] = runtime.train.sequence_parallel
    merged["use_flash_attn"] = runtime.train.use_flash_attn
    # Keep nested structure for BaseProfiler path conventions.
    merged["parallel"] = runtime.parallel
    merged["profile"] = runtime.profile

    merged["model_name"] = model_name(runtime)
    return ProfilerArgs.from_source(merged)


if __name__ == "__main__":
    profile_args = _build_profile_args(sys.argv[1], sys.argv[2:])

    profiler = ModelProfiler(profile_args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_profiler_launcher(path, layernum_arg_names=["num_layers"], model_name=profile_args.model_name)
    profiler.launch_profiling_scripts()
    profiler.process_profiled_data()
