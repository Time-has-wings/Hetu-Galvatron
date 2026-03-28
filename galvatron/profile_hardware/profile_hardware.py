import argparse
import os

from galvatron.core import HardwareProfiler
from galvatron.core.profiler import galvatron_profile_hardware_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = galvatron_profile_hardware_args(parser)
    args = parser.parse_args()
    print(args)
    profiler = HardwareProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_path(path)
    
    # profile allreduce & p2p bandwidth
    profiler.profile_bandwidth()
    # profile allreduce & a2a bandwidth in different communication size
    profiler.profile_sp_bandwidth()
    # profile overlapping slowdown coefficient
    profiler.profile_overlap()