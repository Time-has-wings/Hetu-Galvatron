from typing import Literal, Optional

from pydantic import BaseModel, Field

from galvatron.core.runtime.args_schema import GalvatronModelArgs, GalvatronParallelArgs, CommonTrainArgs

__all__ = [
    "GalvatronSearchArgs",
]


class SearchEngineBatchSizeArgs(BaseModel):
    min_bsz: int = Field(default=8, ge=1, description="Minimum batch size for searching.")
    max_bsz: int = Field(default=8, ge=1, description="Maximum batch size for searching.")
    recommend_min_bsz: int = Field(default=0, description="If 1, start searching from a recommended bsz to accelerate optimization.")
    settle_bsz: int = Field(default=-1, description="If > 1, only search bsz=settle_bsz.")
    settle_chunk: int = Field(default=-1, description="If > 1, only search chunk=settle_chunk.")
    bsz_scale: int = Field(default=8, ge=1, description="Batch size scale for searching.")


class SearchEngineHardwareInfoArgs(BaseModel):
    num_nodes: int = Field(default=1, ge=1, description="Number of nodes.")
    num_gpus_per_node: int = Field(default=8, ge=1, description="Number of GPUs per node.")
    memory_constraint: int = Field(default=24, ge=1, description="Memory constraint of Galvatron (GB).")

class SearchEngineSearchSpaceArgs(BaseModel):
    disable_dp: int = Field(default=0, description="Whether to disable data parallelism (DP).")
    disable_tp: int = Field(default=0, description="Whether to disable tensor parallelism (TP).")
    disable_cp: int = Field(default=1, description="Whether to disable context parallelism (CP).")
    disable_sp: int = Field(default=0, description="Whether to disable sequence parallelism (SP).")
    disable_embedding_lmhead_tp: int = Field(default=0, description="Whether to disable embedding / LM-head tensor parallelism.")
    disable_embedding_lmhead_sp: int = Field(default=0, description="Whether to disable embedding / LM-head sequence parallelism.")
    disable_pp: int = Field(default=0, description="Whether to disable pipeline parallelism (PP).")
    disable_ckpt: int = Field(default=0, description="Whether to disable activation checkpointing.")
    disable_fsdp: int = Field(default=0, description="Whether to disable FSDP.")
    max_tp_deg: int = Field(default=8, ge=1, description="Maximum tensor parallel degree to search.")
    max_pp_deg: int = Field(default=8, ge=1, description="Maximum pipeline parallel degree to search.")
    max_sp_deg: int = Field(default=8, ge=1, description="Maximum sequence parallel degree to search.")
    max_cp_deg: int = Field(default=8, ge=1, description="Maximum context parallel degree to search.")


class SearchEngineProfilingArgs(BaseModel):
    memory_profiling_path: Optional[str] = Field(default=None, description="Path to memory profiling config.")
    time_profiling_path: Optional[str] = Field(default=None, description="Path to time profiling config.")
    allreduce_bandwidth_config_path: Optional[str] = Field(default=None, description="Path to all-reduce bandwidth config.")
    p2p_bandwidth_config_path: Optional[str] = Field(default=None, description="Path to point-to-point bandwidth config.")
    overlap_coe_path: Optional[str] = Field(default=None, description="Path to overlap coefficient config.")
    sp_time_path: Optional[str] = Field(default=None, description="Path to sequence parallelism time config.")
    time_profile_mode: Literal["static", "batch", "sequence", "hybrid"] = Field(default="static", description="Galvatron time profiling mode.")
    memory_profile_mode: Literal["static", "batch", "sequence", "hybrid"] = Field(default="static", description="Galvatron memory profiling mode.")


class SearchEngineOptionsArgs(BaseModel):
    parallel_search: bool = Field(default=False, description="Enable parallel search for faster execution.")
    worker: int = Field(default=0, ge=0, description="Number of worker threads for parallel search. Default 0 means 2× CPU cores.")
    log_dir: str = Field(default="logs", description="Log directory for the search engine.")
    output_config_path: Optional[str] = Field(default=None, description="Path to output config.")
    fine_grained_mode: int = Field(default=1, description="Enable fine-grained search.")


class SearchEngineDebugArgs(BaseModel):
    debug_costmodel_coe: float = Field(default=1.0, description="Multiply the outcome of the time cost model by this coefficient. Only for fine-tuning the time cost model; should be 1.0 by default.")


class GalvatronSearchArgs(BaseModel):
    model_info:GalvatronModelArgs = Field(default=None, description="Model information.")
    parallelism_info:GalvatronParallelArgs = Field(default=None, description="Parallelism information.")
    common_train_info:CommonTrainArgs = Field(default=None, description="Common training information.")
    hardware_info:SearchEngineHardwareInfoArgs = Field(default=None, description="Hardware information.")
    batch_size_info:SearchEngineBatchSizeArgs = Field(default=None, description="Batch size information.")
    search_space_info:SearchEngineSearchSpaceArgs = Field(default=SearchEngineSearchSpaceArgs(), description="Search space information.")
    profiling_info:SearchEngineProfilingArgs = Field(default=None, description="Profiling information.")
    options_info:SearchEngineOptionsArgs = Field(default=SearchEngineOptionsArgs(), description="Options information.")
    debug_info:SearchEngineDebugArgs = Field(default=SearchEngineDebugArgs(), description="Debug information.")
