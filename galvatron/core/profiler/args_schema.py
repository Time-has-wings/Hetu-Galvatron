"""Pydantic models for Galvatron profiler arguments. Merged view: galvatron.core.args_schema."""
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, List

from pydantic import BaseModel, ConfigDict, Field

from galvatron.core.runtime.args_schema import GalvatronModelArgs

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


class GalvatronModelProfilerArgs(BaseModel):
    profile_type: Literal["memory", "computation"] = Field(default="memory", description="Galvatron profiling type.")
    profile_mode: Literal["static", "batch", "sequence"] = Field(default="static", description="Galvatron profiling mode.")
    profile_unit: Literal["attention", "mlp", "all"] = Field(default="all", description="Profile granularity.")
    profile_flow_control: Literal["all", "scripts_only", "launch_only", "data_only"] = Field(default="all", description="Control profiling flow: all steps, data processing only, or script generation only.")

    profile_mixed_precision: Literal["fp32", "fp16", "bf16"] = Field(default="bf16", description="Mixed precision option.")

    profile_fixed_batch_size: Optional[int] = Field(default=None, description="Galvatron profiling batch size.")
    profile_min_batch_size: Optional[int] = Field(default=None, description="Galvatron profiling min batch size.")
    profile_max_batch_size: Optional[int] = Field(default=None, description="Galvatron profiling max batch size.")
    profile_batch_size_step: Optional[int] = Field(default=None, description="Galvatron profiling batch size step.")
    
    profile_fixed_seq_length_list: Optional[List[int]] = Field(default=None, description="Galvatron profiling sequence length list. Length should be 1 for encoder-only or decoder-only models, and 2 for encoder-decoder models.")
    profile_min_seq_length: Optional[int] = Field(default=None, description="Galvatron profiling min sequence length.")
    profile_max_seq_length: Optional[int] = Field(default=None, description="Galvatron profiling max sequence length.")
    profile_seq_length_step: Optional[int] = Field(default=None, description="Galvatron profiling sequence length step.")

    profile_layernum_min: int = Field(default=1, description="Layernum min for profiling.")
    profile_layernum_max: int = Field(default=2, description="Layernum max for profiling.")

    profile_max_tp_deg: int = Field(default=8, description="Maximum tensor parallel degree to profile.")
    profile_dp_type: Literal["zero3", "ddp"] = Field(default="zero3", description="Use zero3 or ddp to profile.")

    runtime_yaml_template_path: Optional[str] = Field(default=None, description="Runtime yaml template path.")

    model_info:GalvatronModelArgs = Field(default_factory=GalvatronModelArgs, description="Model args.")

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

# Backward-compatible aliases.
ModelProfilerArgs = GalvatronModelProfilerArgs
HardwareProfilerArgs = ProfilerHardwareArgs
