import os
import sys

from galvatron.core.arguments import load_with_hydra
from galvatron.core.profiler import HardwareProfiler

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1].endswith((".yaml", ".yml")):
        config_path, overrides = sys.argv[1], sys.argv[2:]
        sys.argv = [sys.argv[0]]
        args = load_with_hydra(config_path, overrides=overrides, mode="profiler_hardware")
    else:
        raise ValueError("Usage: python profile_hardware.py <config_path> [overrides...]")

    profiler = HardwareProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_path(path)

    profiler.profile_bandwidth()
    profiler.profile_sp_bandwidth()
    profiler.profile_overlap()
