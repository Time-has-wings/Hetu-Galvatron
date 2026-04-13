import os

from galvatron.utils.config_utils import read_json_config, write_json_config

from .args_schema import ProfilerHardwareArgs
from .base_profiler import BaseProfiler


class HardwareProfiler(BaseProfiler):
    """Hardware profiler for generating communication profiling scripts."""

    def __init__(self, args: ProfilerHardwareArgs):
        super().__init__()
        self.args = args
        self.path = None

    def set_path(self, path: str) -> None:
        """Root directory for `scripts/` and generated logs (same layout as repo `profile_hardware/`)."""
        self.path = path

    def get_env(self) -> str:
        """Get environment configuration as string

        Returns:
            str: Environment configuration string with export commands
        """
        env = {
            "NUM_NODES": self.args.num_nodes,
            "NUM_GPUS_PER_NODE": self.args.num_gpus_per_node,
            "MASTER_ADDR": self.args.master_addr,
            "MASTER_PORT": self.args.master_port,
            "NODE_RANK": self.args.node_rank,
        }
        env_str = "\n".join([k for k in self.args.envs]) + "\n"
        env_str += "\n".join([f"export {k}={v}" for k, v in env.items()]) + "\n"

        return env_str

    def generate_script(self, num_nodes: int, num_gpus_per_node: int) -> None:
        """Generate test scripts for allreduce and p2p communication

        Args:
            num_nodes: Number of nodes to use
            num_gpus_per_node: Number of GPUs per node
        """
        world_size = num_nodes * num_gpus_per_node
        env = self.get_env()

        print("Generating allreduce test script...")

        torchrun_prefix = (
            "torchrun \\\n"
            "    --nnodes=$NUM_NODES \\\n"
            "    --nproc_per_node=$NUM_GPUS_PER_NODE \\\n"
            "    --master_addr=$MASTER_ADDR \\\n"
            "    --master_port=$MASTER_PORT \\\n"
            "    --node_rank=$NODE_RANK"
        )

        # One torchrun: bandwidth sweep logic (halving tp, consec 1 then 0, skip full-world consec=0)
        # lives in profile_allreduce.bandwidth_jobs_from_tp_degrees — same as legacy shell nested loops.
        log_name = "logs/allreduce/allreduce_bandwidth_tp_time0.log"
        script = (
            f"{torchrun_prefix} \\\n"
            "    profile_allreduce.py \\\n"
            f"    --global_tp_deg {_shell_int_list(_halving_tp_degrees(world_size, world_size))} \\\n"
            "    --profile_time 0 \\\n"
            f"    2>&1 | tee {log_name}\n"
        )

        config_dir = os.path.join(self.path, "./scripts")
        with open(os.path.join(config_dir, "profile_allreduce.sh"), "w") as f:
            f.write(env)
            f.write(
                "# Bandwidth sweep = legacy: while tp halves; each tp runs consec 1 then 0; "
                "skip tp==world_size with consec 0. Implemented in profile_allreduce.bandwidth_jobs_from_tp_degrees.\n"
                "# Omit --local_batch_size here: profile_allreduce.py defaults to 32 (still used for message size).\n"
            )
            f.write("mkdir -p logs/allreduce\n")
            f.write(f'echo "Running: {script}"\n')
            f.write(script)

        print("Generating p2p test script...")

        log_name = "logs/p2p/p2p_pp.log"
        script = (
            f"{torchrun_prefix} \\\n"
            "    profile_p2p.py \\\n"
            f"    --pp_deg {_shell_int_list(_p2p_pp_deg_sweep(world_size, self.args.max_pp_deg))} \\\n"
            f"    2>&1 | tee {log_name}\n"
        )

        with open(os.path.join(config_dir, "profile_p2p.sh"), "w") as f:
            f.write(env)
            f.write("mkdir -p logs/p2p\n")
            f.write(f'echo "Running: {script}"\n')
            f.write(script)

    def generate_sp_script(self, num_nodes: int, num_gpus_per_node: int) -> None:
        """Generate test scripts for allreduce and all2all communication

        Args:
            num_nodes: Number of nodes to use
            num_gpus_per_node: Number of GPUs per node
        """
        env = self.get_env()

        print("Generating allreduce test script...")

        torchrun_prefix = (
            "torchrun \\\n"
            "    --nnodes=$NUM_NODES \\\n"
            "    --nproc_per_node=$NUM_GPUS_PER_NODE \\\n"
            "    --master_addr=$MASTER_ADDR \\\n"
            "    --master_port=$MASTER_PORT \\\n"
            "    --node_rank=$NODE_RANK"
        )

        args = self.args
        config_dir = os.path.join(self.path, "./scripts")
        world_size = num_nodes * num_gpus_per_node
        log_name = "logs/allreduce_sp/allreduce_sp_time1.log"
        script = (
            f"{torchrun_prefix} \\\n"
            "    profile_allreduce.py \\\n"
            f"    --global_tp_deg {_shell_int_list(_halving_tp_degrees(world_size, args.max_tp_size))} \\\n"
            f"    --local_batch_size {_shell_int_list(_halving_batch_sizes(1024))} \\\n"
            "    --profile_time 1 \\\n"
            f"    2>&1 | tee {log_name}\n"
        )

        # Write allreduce test script with sequence parallelism (one torchrun)
        with open(os.path.join(config_dir, "profile_allreduce_sp.sh"), "w") as f:
            f.write(env)
            f.write("mkdir -p logs/allreduce_sp\n")
            f.write(f'echo "Running: {script}"\n')
            f.write(script)

        print("Generating all2all test script...")

        log_name = "logs/all2all_sp/all2all_sp.log"
        script = (
            f"{torchrun_prefix} \\\n"
            "    profile_all2all.py \\\n"
            f"    --global_tp_deg {_shell_int_list(_halving_tp_degrees(world_size, args.max_tp_size))} \\\n"
            f"    --local_batch_size {_shell_int_list(_halving_batch_sizes(1024))} \\\n"
            f"    2>&1 | tee {log_name}\n"
        )

        with open(os.path.join(config_dir, "profile_all2all_sp.sh"), "w") as f:
            f.write(env)
            f.write("mkdir -p logs/all2all_sp\n")
            f.write(f'echo "Running: {script}"\n')
            f.write(script)

    def profile_bandwidth(self) -> None:
        """Generate allreduce/p2p profiling scripts."""
        args = self.args
        self.generate_script(args.num_nodes, args.num_gpus_per_node)

    def profile_sp_bandwidth(self):
        """Generate sequence-parallel profiling scripts."""
        args = self.args
        self.generate_sp_script(args.num_nodes, args.num_gpus_per_node)

    def write_config(self, hardware_config_path: str, key: str, bandwidth: float) -> None:
        """Write bandwidth/time results to hardware config file

        Args:
            hardware_config_path: Path to the hardware config file
            key: Key for the bandwidth/time result
            bandwidth: Measured bandwidth or time value
        """
        config = read_json_config(hardware_config_path) if os.path.exists(hardware_config_path) else dict()
        config[key] = bandwidth
        write_json_config(config, hardware_config_path)
        print("Already written bandwidth/time %s into hardware config file %s!" % (key, hardware_config_path))

    # =============== For Launching Scripts for Profiling Overlap Slowdown Coefficient ===============
    def profile_overlap(self):
        """Profile overlap slowdown coefficient

        This method launches scripts to profile the overlap between computation and communication
        """
        args = self.args
        ARGS = ""
        ARGS += "USE_EXPORT_VARIABLE=1 "
        ARGS += "NUM_GPUS_PER_NODE=%d " % args.num_gpus_per_node
        ARGS += "OVERLAP_TIME_MULTIPLY=%d " % args.overlap_time_multiply
        os.system(ARGS + "sh %s" % (os.path.join(self.path, "scripts/profile_overlap.sh")))





def _halving_tp_degrees(world_size: int, max_tp: int) -> list[int]:
    """8,4,2,... down from min(world_size, max_tp), same order as legacy shell loops."""
    out = []
    s = min(world_size, max_tp)
    while s > 1:
        out.append(s)
        s //= 2
    return out


def _halving_batch_sizes(start: int = 1024) -> list[int]:
    """1024, 512, ... 1."""
    out = []
    b = start
    while b >= 1:
        out.append(b)
        b //= 2
    return out


def _p2p_pp_deg_sweep(world_size: int, max_pp_deg: int) -> list[int]:
    """2, 4, 8, ... up to world_size and max_pp_deg (same as legacy profile_p2p.sh loop)."""
    out = []
    pp_deg = 2
    while pp_deg <= world_size and pp_deg <= max_pp_deg:
        out.append(pp_deg)
        pp_deg *= 2
    return out


def _shell_int_list(xs: list[int]) -> str:
    """Space-separated ints for ``nargs='+'`` flags in generated shell, e.g. ``8 4 2``."""
    return " ".join(str(x) for x in xs)

