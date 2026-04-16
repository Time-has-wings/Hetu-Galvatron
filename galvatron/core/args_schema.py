"""
Merged Pydantic args for Galvatron core: runtime, profiler, search_engine, and tools.
Import from here for a single entry point; or use submodules for per-domain schemas.
"""
from typing import Optional

from pydantic import BaseModel, Field

# Runtime (training) args
from .runtime.args_schema import (
    CommonCkptArgs,
    CommonDataArgs,
    CommonTrainArgs,
    GalvatronModelArgs,
    GalvatronParallelArgs,
    GalvatronProfileArgs,
    GalvatronRuntimeArgs,
    GalvatronTrainingArgs,
)

# Profiler args
from .profiler.args_schema import ProfilerHardwareArgs, GalvatronModelProfilerArgs

# Search engine args
from .search_engine.args_schema import GalvatronSearchArgs

__all__ = [
    # Runtime
    "GalvatronParallelArgs",
    "GalvatronModelArgs",
    "GalvatronProfileArgs",
    "GalvatronRuntimeArgs",
    "GalvatronTrainingArgs",
    "CommonTrainArgs",
    "CommonDataArgs",
    "CommonCkptArgs",
    # Profiler
    "ProfilerHardwareArgs",
    "GalvatronModelProfilerArgs",
    # Search engine
    "SearchEngineArgs",
    # Merged
    "CoreArgs",
]


class CoreArgs(BaseModel):
    """Combined args: one of runtime, profiler, search, or tools is typically used per run."""

    runtime: Optional[GalvatronRuntimeArgs] = Field(default=None, description="Training/runtime args")
    profiler_hardware: Optional[ProfilerHardwareArgs] = Field(default=None, description="Hardware profiler args")
    search_engine: Optional[GalvatronSearchArgs] = Field(default=None, description="Search engine args")
    model_profiler: Optional[GalvatronModelProfilerArgs] = Field(default=None, description="Model profiler args")
