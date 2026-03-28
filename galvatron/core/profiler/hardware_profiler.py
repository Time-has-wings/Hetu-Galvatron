import os

from galvatron.utils.config_utils import read_json_config, write_json_config

from .args_schema import HardwareProfilerArgs
from .base_profiler import BaseProfiler


class HardwareProfiler(BaseProfiler):
    """Hardware profiler for generating communication profiling scripts."""

    def __init__(self, args):
        super().__init__(HardwareProfilerArgs.from_source(args))
        self.path = None

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

        # Generate allreduce test script
        def allreduce_script(allreduce_size: int, allreduce_consec: int) -> str:
            log_name = f"allreduce_world_size{allreduce_size}_consec{allreduce_consec}_profile_time0.log"
            return (
                f"{torchrun_prefix} \\\n"
                "    profile_allreduce.py \\\n"
                f"    --global_tp_deg {allreduce_size} \\\n"
                f"    --global_tp_consec {allreduce_consec} \\\n"
                "    --profile_time 0 \\\n"
                f"    2>&1 | tee {log_name}\n"
            )

        # Write allreduce test script
        config_dir = os.path.join(self.path, "./scripts")
        with open(os.path.join(config_dir, "profile_allreduce.sh"), "w") as f:
            f.write(env)
            allreduce_size = num_nodes * num_gpus_per_node
            while allreduce_size > 1:
                for allreduce_consec in [1, 0]:
                    if world_size == allreduce_size and allreduce_consec == 0:
                        continue
                    script = allreduce_script(allreduce_size, allreduce_consec)
                    f.write(f'echo "Running: {script}"\n')
                    f.write(script)
                allreduce_size //= 2
                f.write("sleep 1\n")

        print("Generating p2p test script...")

        # Generate p2p test script
        def p2p_script(pp_deg: int) -> str:
            log_name = f"p2p_pp_deg{pp_deg}.log"
            return (
                f"{torchrun_prefix} \\\n"
                "    profile_p2p.py \\\n"
                f"    --pp_deg {pp_deg} \\\n"
                f"    2>&1 | tee {log_name}\n"
            )

        # Write p2p test script
        with open(os.path.join(config_dir, "profile_p2p.sh"), "w") as f:
            f.write(env)
            pp_deg = 2
            while pp_deg <= world_size and pp_deg <= self.args.max_pp_deg:
                script = p2p_script(pp_deg)
                f.write(f'echo "Running: {script}"\n')
                f.write(script)
                pp_deg *= 2
                f.write("sleep 1\n")

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

        def allreduce_script(allreduce_size: int, allreduce_consec: int, buffer_size: int) -> str:
            log_name = (
                f"allreduce_sp_world_size{allreduce_size}_consec{allreduce_consec}"
                f"_local_batch_size{buffer_size}_profile_time1.log"
            )
            return (
                f"{torchrun_prefix} \\\n"
                "    profile_allreduce.py \\\n"
                f"    --global_tp_deg {allreduce_size} \\\n"
                f"    --global_tp_consec {allreduce_consec} \\\n"
                f"    --local_batch_size {buffer_size} \\\n"
                "    --profile_time 1 \\\n"
                f"    2>&1 | tee {log_name}\n"
            )

        args = self.args
        config_dir = os.path.join(self.path, "./scripts")

        # Write allreduce test script with sequence parallelism
        with open(os.path.join(config_dir, "profile_allreduce_sp.sh"), "w") as f:
            f.write(env)
            allreduce_size = min(num_nodes * num_gpus_per_node, args.max_tp_size)
            while allreduce_size > 1:
                buffer_size = 1024
                while buffer_size >= 1:
                    script = allreduce_script(allreduce_size, 1, buffer_size)
                    f.write(f'echo "Running: {script}"\n')
                    f.write(script)
                    f.write("sleep 1\n")
                    buffer_size //= 2
                allreduce_size //= 2

        print("Generating all2all test script...")

        def all2all_script(allreduce_size: int, buffer_size: int) -> str:
            log_name = (
                f"all2all_sp_world_size{allreduce_size}"
                f"_local_batch_size{buffer_size}.log"
            )
            return (
                f"{torchrun_prefix} \\\n"
                "    profile_all2all.py \\\n"
                f"    --global_tp_deg {allreduce_size} \\\n"
                f"    --local_batch_size {buffer_size} \\\n"
                f"    2>&1 | tee {log_name}\n"
            )

        # Write all-to-all test script
        with open(os.path.join(config_dir, "profile_all2all_sp.sh"), "w") as f:
            f.write(env)
            all2all_size = min(num_nodes * num_gpus_per_node, args.max_tp_size)
            while all2all_size > 1:
                buffer_size = 1024
                while buffer_size >= 1:
                    script = all2all_script(all2all_size, buffer_size)
                    f.write(f'echo "Running: {script}"\n')
                    f.write(script)
                    f.write("sleep 1\n")
                    buffer_size //= 2
                all2all_size //= 2

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
