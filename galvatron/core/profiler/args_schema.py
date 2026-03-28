"""Pydantic models for Galvatron profiler arguments. Merged view: galvatron.core.args_schema."""
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field


def _source_to_dict(source: Any) -> Dict[str, Any]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        return dict(source)
    if hasattr(source, "_get_kwargs"):
        kwargs = source._get_kwargs()
        if isinstance(kwargs, Mapping):
            return dict(kwargs)
        if isinstance(kwargs, Iterable):
            return {k: v for k, v in kwargs}
    if hasattr(source, "__dict__"):
        return dict(vars(source))
    raise TypeError(f"Unsupported args source type: {type(source)}")


class ProfilerArgs(BaseModel):
    """Galvatron profiling (memory/computation) args."""
    model_config = ConfigDict(extra="allow")

    profile_type: Literal["memory", "computation"] = Field(default="memory", description="Galvatron profiling type.")
    set_model_config_manually: int = Field(
        default=0,
        description="Whether to set model config manually. If set to 1, model config set by 'model_size' will be overwritten.",
    )
    set_layernum_manually: int = Field(
        default=1,
        description="Whether to set layernum config manually (doesn't overwrite other model configs).",
    )
    set_seqlen_manually: int = Field(
        default=0,
        description="Whether to set sequence length config manually (doesn't overwrite other model configs).",
    )
    set_experts_manually: int = Field(
        default=0,
        description="Whether to set experts config manually (doesn't overwrite other model configs).",
    )
    profile_mode: Literal["static", "batch", "sequence"] = Field(
        default="static",
        description="Galvatron profiling mode.",
    )
    profile_batch_size: Optional[int] = Field(default=None, description="Galvatron profiling batch size.")
    profile_min_batch_size: Optional[int] = Field(default=None, description="Galvatron profiling min batch size.")
    profile_max_batch_size: Optional[int] = Field(default=None, description="Galvatron profiling max batch size.")
    profile_batch_size_step: int = Field(default=1, description="Galvatron profiling batch size step.")
    profile_seq_length_list: Optional[str] = Field(
        default=None,
        description="Galvatron profiling sequence length list.",
    )
    profile_min_seq_length: Optional[int] = Field(
        default=None,
        description="Galvatron profiling min sequence length.",
    )
    profile_max_seq_length: Optional[int] = Field(
        default=None,
        description="Galvatron profiling max sequence length.",
    )
    profile_seq_length_step: int = Field(
        default=128,
        description="Galvatron profiling sequence length step.",
    )
    layernum_min: int = Field(default=1, description="Layernum min for profiling.")
    layernum_max: int = Field(default=2, description="Layernum max for profiling.")
    max_tp_deg: int = Field(default=8, description="Maximum tensor parallel degree to profile.")
    profile_dp_type: Literal["zero3", "ddp"] = Field(default="zero3", description="Use zero3 or ddp to profile.")
    mixed_precision: Literal["fp32", "fp16", "bf16"] = Field(default="bf16", description="Mixed precision option.")
    use_flash_attn: bool = Field(default=False, description="Use FlashAttention implementation of attention.")
    extra_args_str: str = Field(default="", description="Extra arguments for megatron initialization.")
    sequence_parallel: bool = Field(default=False, description="Whether to use sequence parallel.")
    shape_order: Literal["SBH", "BSH"] = Field(default="SBH", description="Model shape order.")
    make_vocab_size_divisible_by: int = Field(
        default=128,
        description="Pad the vocab size to be divisible by this value. For computational efficiency.",
    )
    profile_unit: Literal["attention", "mlp", "all"] = Field(default="all", description="Profile granularity.")
    profile_flow_control: Literal["all", "scripts_only", "launch_only", "data_only"] = Field(
        default="all",
        description="Control profiling flow: all steps, data processing only, or script generation only.",
    )

    @classmethod
    def from_source(cls, source: Any) -> "ProfilerArgs":
        if isinstance(source, cls):
            return source
        return cls.model_validate(_source_to_dict(source))

    def _get_kwargs(self):
        return list(self.model_dump().items())


class ProfilerHardwareArgs(BaseModel):
    """Galvatron profiling hardware args."""
    model_config = ConfigDict(extra="allow")

    num_nodes: int = Field(default=1, description="Number of nodes.")
    num_gpus_per_node: int = Field(default=8, description="Number of GPUs per node.")
    master_addr: str = Field(default="$MASTER_ADDR", description="Master address.")
    master_port: str = Field(default="$MASTER_PORT", description="Master port.")
    node_rank: str = Field(default="$RANK", description="Node rank.")
    max_tp_size: int = Field(default=8, description="Maximum tensor parallel size.")
    envs: list[str] = Field(
        default_factory=list,
        description="Additional environment variables in format KEY=VALUE.",
    )
    max_pp_deg: int = Field(default=8, description="Maximum pipeline parallel degree to search.")
    overlap_time_multiply: int = Field(
        default=4,
        description="The multiple of communication time and computation time when overlapped.",
    )

    @classmethod
    def from_source(cls, source: Any) -> "ProfilerHardwareArgs":
        if isinstance(source, cls):
            return source
        return cls.model_validate(_source_to_dict(source))

    def _get_kwargs(self):
        return list(self.model_dump().items())


class RuntimeProfilerArgs(BaseModel):
    """Runtime profiler args wrapper (keeps original nested profile/train/parallel logic)."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    profile: Any
    train: Any
    parallel: Any
    local_rank: int = 0

    @classmethod
    def from_source(cls, source: Any) -> "RuntimeProfilerArgs":
        if isinstance(source, cls):
            return source

        if hasattr(source, "profile") and hasattr(source, "train") and hasattr(source, "parallel"):
            data = {
                "profile": source.profile,
                "train": source.train,
                "parallel": source.parallel,
                "local_rank": getattr(source, "local_rank", 0),
            }
            return cls.model_validate(data)

        raw = _source_to_dict(source)
        profile = raw.get(
            "profile",
            SimpleNamespace(
                profile=bool(raw.get("profile", False)),
                exit_after_profiling=bool(raw.get("exit_after_profiling", True)),
                save_profiled_memory=bool(raw.get("save_profiled_memory", False)),
                profile_forward=bool(raw.get("profile_forward", False)),
                profile_unit=raw.get("profile_unit", "all"),
            ),
        )
        train = raw.get(
            "train",
            SimpleNamespace(
                global_batch_size=raw.get("global_train_batch_size", raw.get("global_batch_size", 1)),
                sequence_parallel=bool(raw.get("sequence_parallel", False)),
                lr=raw.get("lr", 1e-4),
            ),
        )
        parallel = raw.get(
            "parallel",
            SimpleNamespace(
                pp_deg=raw.get("pp_deg", 1),
                global_tp_deg=raw.get("global_tp_deg", 1),
                global_checkpoint=raw.get("global_checkpoint", 0),
                vocab_tp=raw.get("vocab_tp", 1),
                pipeline_type=raw.get("pipeline_type", "gpipe"),
                mixed_precision=raw.get("mixed_precision", "bf16"),
            ),
        )

        data = {
            "profile": profile,
            "train": train,
            "parallel": parallel,
            "local_rank": raw.get("local_rank", 0),
        }
        return cls.model_validate(data)


# Backward-compatible aliases.
ModelProfilerArgs = ProfilerArgs
HardwareProfilerArgs = ProfilerHardwareArgs
