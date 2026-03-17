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
from .profiler.args_schema import ProfilerArgs, ProfilerHardwareArgs

# Search engine args
from .search_engine.args_schema import SearchEngineArgs

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
    "ProfilerArgs",
    "ProfilerHardwareArgs",
    # Search engine
    "SearchEngineArgs",
    # Merged
    "CoreArgs",
]


class CoreArgs(BaseModel):
    """Combined args: one of runtime, profiler, search, or tools is typically used per run."""

    runtime: Optional[GalvatronRuntimeArgs] = Field(default=None, description="Training/runtime args")
    profiler: Optional[ProfilerArgs] = Field(default=None, description="Profiler args")
    profiler_hardware: Optional[ProfilerHardwareArgs] = Field(default=None, description="Hardware profiler args")
    search_engine: Optional[SearchEngineArgs] = Field(default=None, description="Search engine args")
