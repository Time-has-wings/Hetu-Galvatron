"""Pydantic models for Galvatron profiler arguments. Merged view: galvatron.core.args_schema."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ProfilerArgs(BaseModel):
    """Galvatron profiling (memory/computation) args."""

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


class ProfilerHardwareArgs(BaseModel):
    """Galvatron profiling hardware (nccl-tests) args."""

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
    backend: Literal["nccl", "torch"] = Field(default="nccl", description="Backend of nccl-tests.")
    nccl_test_dir: str = Field(default="nccl-tests", description="Directory of nccl-tests.")
    mpi_path: str = Field(default="/usr/local/mpi/", description="MPI path.")
    start_mb: int = Field(default=16, description="Starting communication size in MB.")
    end_mb: int = Field(default=512, description="Ending communication size in MB.")
    scale: int = Field(default=2, description="Memory scale of nccl-tests.")
    hostfile: str = Field(default="hostfile", description="Hostfile for nccl-tests.")
    avg_or_min_or_first: Literal["first", "min", "avg"] = Field(
        default="first",
        description="For a given group size: 'first' only profile first group; 'min' profile group with minimum bandwidth; 'avg' profile all and take average.",
    )
    max_pp_deg: int = Field(default=8, description="Maximum pipeline parallel degree to search.")
    overlap_time_multiply: int = Field(
        default=4,
        description="The multiple of communication time and computation time when overlapped.",
    )
