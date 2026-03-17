"""Pydantic models for Galvatron search engine arguments. Merged view: galvatron.core.args_schema."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SearchEngineArgs(BaseModel):
    """Galvatron parallelism search engine args."""

    set_model_config_manually: int = Field(
        default=0,
        description="Whether to set model config manually. If set to 1, model config set by 'model_size' will be overwritten.",
    )
    set_layernum_manually: int = Field(
        default=0,
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
    num_nodes: int = Field(default=1, description="Number of nodes.")
    num_gpus_per_node: int = Field(default=8, description="Number of GPUs per node.")
    memory_constraint: int = Field(default=24, description="Memory constraint of Galvatron.")
    min_bsz: int = Field(default=8, description="Min batch size for searching.")
    max_bsz: int = Field(default=10240, description="Max batch size for searching.")
    recommend_min_bsz: int = Field(
        default=0,
        description="If 1, start searching from a recommended bsz to accelerate optimization.",
    )
    settle_bsz: int = Field(default=-1, description="If > 1, only search bsz=settle_bsz.")
    settle_chunk: int = Field(default=-1, description="If > 1, only search chunk=settle_chunk.")
    bsz_scale: int = Field(default=8, description="Bsz scale for searching.")
    search_space: Literal["full", "dp+tp", "dp+pp", "3d", "dp", "sdp", "tp", "pp"] = Field(
        default="full",
        description="Galvatron parallelism optimization type.",
    )
    sp_space: Literal["tp+sp", "tp", "sp"] = Field(
        default="tp",
        description="Galvatron sequence parallelism optimization type.",
    )
    disable_dp: int = Field(default=0, description="Whether to disable dp.")
    disable_tp: int = Field(default=0, description="Whether to disable tp.")
    disable_vtp: int = Field(default=0, description="Whether to disable vocab tp.")
    disable_pp: int = Field(default=0, description="Whether to disable pp.")
    disable_sdp: int = Field(default=0, description="Whether to disable sdp.")
    disable_ckpt: int = Field(default=0, description="Whether to disable checkpoint.")
    disable_tp_consec: int = Field(default=0, description="Whether to disable tp_consec.")
    max_tp_deg: int = Field(default=8, description="Maximum tensor parallel degree to search.")
    max_pp_deg: int = Field(default=8, description="Maximum pipeline parallel degree to search.")
    default_dp_type: Literal["ddp", "zero2"] = Field(default="ddp", description="Default data parallel type.")
    mixed_precision: Literal["fp32", "fp16", "bf16"] = Field(default="bf16", description="Mixed precision option.")
    pipeline_type: Literal["gpipe", "pipedream_flush"] = Field(default="gpipe", description="Galvatron pipeline type.")
    use_pipeline_costmodel: int = Field(default=1, description="Whether to use pipeline cost model.")
    costmodel_coe: float = Field(
        default=1.0,
        description="Multiply the outcome of time cost model by this coefficient. Only for fine-tuning time cost model, should be 1.0 in default.",
    )
    sequence_parallel: bool = Field(default=False, description="Whether to use sequence parallel.")
    global_memory_buffer: bool = Field(
        default=True,
        description="Enable estimation of global memory for allgather buffer when using Megatron-SP.",
    )
    async_grad_reduce: bool = Field(
        default=True,
        description="If False, gradient will be reduced every micro batch. Ensure Zero3 memory cost when chunk > 1.",
    )
    memory_profiling_path: Optional[str] = Field(default=None, description="Path to memory profiling config.")
    time_profiling_path: Optional[str] = Field(default=None, description="Path to time profiling config.")
    allreduce_bandwidth_config_path: Optional[str] = Field(
        default=None,
        description="Path to allreduce bandwidth config.",
    )
    p2p_bandwidth_config_path: Optional[str] = Field(default=None, description="Path to p2p bandwidth config.")
    overlap_coe_path: Optional[str] = Field(default=None, description="Path to overlap coefficient config.")
    sp_time_path: Optional[str] = Field(
        default=None,
        description="Path to sequence parallelism time config.",
    )
    output_config_path: Optional[str] = Field(default=None, description="Path to output config.")
    make_vocab_size_divisible_by: int = Field(
        default=128,
        description="Pad the vocab size to be divisible by this value. For computational efficiency.",
    )
    fine_grained_mode: int = Field(default=1, description="Enable fine-grained search.")
    time_profile_mode: Literal["static", "batch", "sequence", "hybrid"] = Field(
        default="static",
        description="Galvatron time profiling mode.",
    )
    memory_profile_mode: Literal["static", "batch", "sequence", "hybrid"] = Field(
        default="static",
        description="Galvatron memory profiling mode.",
    )
    parallel_search: bool = Field(default=False, description="Enable parallel search for faster execution.")
    worker: int = Field(
        default=0,
        description="Number of worker threads for parallel search. Default is 2x CPU cores.",
    )
    log_dir: str = Field(default="logs", description="Log directory for search engine.")
