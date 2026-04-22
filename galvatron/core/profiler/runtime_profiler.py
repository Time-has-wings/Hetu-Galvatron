import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base_profiler import BaseProfiler
from .utils import print_peak_memory, save_profiled_memory, save_profiled_time
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs


class RuntimeProfiler(BaseProfiler):
    """Runtime profiler for monitoring memory usage and computation time during model execution."""

    def __init__(self, args: GalvatronRuntimeArgs):
        """Initialize runtime profiler

        Args:
            args: Arguments containing profiling configuration
        """
        super().__init__()
        self.args = args

    def set_profiler_dist(
        self,
        path: Optional[str] = None,
        model_layer_configs: Optional[List[Dict]] = None,
        model_name: Optional[str] = None,
        profile_ranks: Optional[List[int]] = None,
        start_iter: int = 10,
        end_iter: int = 20,
        rank: Optional[int] = None,
    ) -> None:
        """Configure distributed profiling settings

        Args:
            path: Path to save profiling results
            model_layer_configs: List of layer configurations containing:
                - hidden_size: Hidden dimension size
                - layer_num: Number of layers
                - seq_len: Sequence length
            model_name: Name of the model being profiled
            profile_ranks: List of ranks to profile (default: [0, world_size-1])
            start_iter: Starting iteration for profiling
            end_iter: Ending iteration for profiling
            rank: Current process rank (default: get from torch.distributed)
        """
        args = self.args
        rank = torch.distributed.get_rank() if rank is None else rank
        if profile_ranks is None:
            world_size = torch.distributed.get_world_size()
            profile_ranks = [0, world_size - 1]

        self.set_work_dir(path)
        self.set_model_name(model_name)
        self.set_profile_unit(args.profile.profile_unit)
        self.set_mixed_precision(args.parallel.mixed_precision)

        self.set_model_layer_configs(model_layer_configs)

        self.set_memory_profiler(rank, profile_ranks)
        self.set_time_profiler(start_iter=start_iter, end_iter=end_iter, exit=bool(args.profile.exit_after_profiling))

    def set_profiler_single(self, start_iter=10, end_iter=20):
        """
        Set profiler for single process

        Args:
            start_iter: Starting iteration for profiling
            end_iter: Ending iteration for profiling
        """
        self.set_memory_profiler(0)
        exit_ = bool(self.args.profile.exit_after_profiling)
        self.set_time_profiler(start_iter=start_iter, end_iter=end_iter, exit=exit_)
    
    def set_model_layer_configs(self, model_layer_configs: Optional[List[Dict]]) -> None:
        """Set model layer configurations

        Args:
            model_layer_configs: List of layer configurations containing:
                - hidden_size: Hidden dimension size
                - layer_num: Number of layers
                - seq_len: Sequence length
        """
        if model_layer_configs is None:
            return
        self.hiddensize_list = [config["hidden_size"] for config in model_layer_configs]
        self.layernum_list = [config["layer_num"] for config in model_layer_configs]
        self.seqlen_list = [config["seq_len"] for config in model_layer_configs]

    # =============== Memory Profiling ===============
    def set_memory_profiler(self, rank: int, profile_ranks: List[int] = [], max_profile_iter: int = 5) -> None:
        """Configure memory profiler settings

        Args:
            rank: Current process rank
            profile_ranks: List of ranks to profile
            max_profile_iter: Maximum number of iterations to profile
        """
        self.rank = rank
        self.profile_ranks = profile_ranks if len(profile_ranks) > 0 else [rank]
        self.mem_dict = {}
        self.max_profile_iter = max_profile_iter

    def profile_memory(self, iter: int, stage: str = "") -> None:
        """Profile memory usage at different stages of training

        Args:
            iter: Current iteration number
            stage: Profiling stage ("Before Forward", "After Forward", "After Backward")
        """
        args, rank = self.args, self.rank
        profile_ranks, mem_dict = self.profile_ranks, self.mem_dict
        max_profile_iter = self.max_profile_iter

        if args.profile.profile and rank in profile_ranks and iter <= max_profile_iter:
            local_rank = args.local_rank
            profile_type = "allocated"

            if stage == "Before Forward":
                torch.cuda.reset_peak_memory_stats(local_rank)
                _, cur_mem = print_peak_memory("\n" + stage, local_rank, profile_type)
                mem_dict[f"iter_{iter}_before_forward"] = cur_mem
            elif stage == "After Forward":
                _, cur_mem = print_peak_memory(stage, local_rank, profile_type)
                mem_dict[f"iter_{iter}_after_forward"] = cur_mem
            elif stage == "After Backward":
                max_mem, cur_mem = print_peak_memory(stage, local_rank, profile_type)
                mem_dict[f"iter_{iter}_after_backward"] = cur_mem
                mem_dict[f"iter_{iter}_after_backward_max"] = max_mem
            else:
                print_peak_memory(stage, local_rank, profile_type)

    def post_profile_memory(self, iter: int) -> None:
        """Post-process and save memory profiling results

        Args:
            iter: Current iteration number
        """
        args, rank = self.args, self.rank
        profile_ranks, mem_dict = self.profile_ranks, self.mem_dict
        max_profile_iter = self.max_profile_iter

        if args.profile.profile and iter == max_profile_iter:
            save_mem = bool(args.profile.save_profiled_memory)
            if rank in profile_ranks:
                # Calculate memory statistics
                mem_dict["model_states"] = mem_dict[f"iter_{max_profile_iter-1}_after_backward"]

                pipeline_type = args.parallel.pipeline_type
                if pipeline_type == "gpipe":
                    mem_dict["model_states_and_activation"] = mem_dict[f"iter_{max_profile_iter-1}_after_forward"]
                    mem_dict["activation"] = (
                        mem_dict[f"iter_{max_profile_iter-1}_after_forward"]
                        - mem_dict[f"iter_{max_profile_iter-1}_before_forward"]
                    )

                mem_dict["model_states_and_peak_activation"] = mem_dict[f"iter_{max_profile_iter-1}_after_backward_max"]
                mem_dict["peak_activation"] = (
                    mem_dict[f"iter_{max_profile_iter-1}_after_backward_max"]
                    - mem_dict[f"iter_{max_profile_iter-1}_after_backward"]
                )

                # Print results
                time.sleep(0.2 * rank)
                print(f"[Profiled memory for rank {rank}]:")
                for key, val in mem_dict.items():
                    print(f"\t{key}: {val:.2f} MB")

                # Save results if requested
                if save_mem:
                    assert self.layernum_list is not None
                    world_size = torch.distributed.get_world_size()
                    memory_config_path = self.memory_profiling_path()

                    save_profiled_memory(
                        memory_config_path,
                        args.parallel.pp_deg,
                        args.parallel.global_tp_deg,
                        world_size,
                        self.layernum_list,
                        args.train.global_batch_size,
                        rank,
                        mem_dict["model_states"],
                        mem_dict["activation"],
                        mem_dict["peak_activation"],
                        args.parallel.global_checkpoint,
                        args.train.sequence_parallel,
                        args.parallel.vocab_tp,
                        self.seqlen_list,
                    )

            if save_mem:
                exit(0)

    # =============== Time Profiling ===============
    def set_time_profiler(self, start_iter: int, end_iter: int, exit: bool = False) -> None:
        """Configure time profiler settings

        Args:
            start_iter: Starting iteration for profiling
            end_iter: Ending iteration for profiling
            exit: Whether to exit after profiling
        """
        self.start_iter = start_iter
        self.end_iter = end_iter
        assert end_iter > start_iter, "End iteration must be greater than start iteration"

        self.exit = exit
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.time_list = []
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_size = 1

    def profile_time_start(self, iter: int) -> None:
        """Start timing for current iteration

        Args:
            iter: Current iteration number
        """
        if not self.args.profile.profile:
            return

        if iter >= self.start_iter and iter < self.end_iter:
            torch.cuda.synchronize()
            self.start.record()
        elif iter == self.end_iter:
            self._process_time_results()

    def profile_time_end(
        self,
        iter: int,
        loss: Optional[torch.Tensor] = None,
        learning_rate: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> None:
        """End timing for current iteration and log results

        Args:
            iter: Current iteration number
            loss: Training loss value
            learning_rate: Current learning rate
            grad_norm: Gradient norm
        """
        if not self.args.profile.profile:
            return

        if iter >= self.start_iter and iter < self.end_iter:
            self.end.record()
            torch.cuda.synchronize()
            iter_time = self.start.elapsed_time(self.end) / 1e3
            self.time_list.append(iter_time)

            if self.rank == self.world_size - 1:
                self._log_iteration_stats(iter, iter_time, loss, learning_rate, grad_norm)

    def profile_time_python(self, iter: int) -> None:
        """Profile time using Python's time module (coarse timing)

        Args:
            iter: Current iteration number
        """
        if not self.args.profile.profile:
            return

        if iter == self.start_iter:
            self.total_start_time = time.time()
        elif iter == self.end_iter:
            self.total_end_time = time.time()
            avg_time = (self.total_end_time - self.total_start_time) / (self.end_iter - self.start_iter)
            print(f"Average iteration time is: {avg_time:.4f} s")

            args = self.args
            if args.profile.profile_forward:
                assert self.layernum_list is not None
                time_config_path = self.time_profiling_path()
                save_profiled_time(
                    time_config_path, avg_time, args.train.global_batch_size, self.layernum_list, self.seqlen_list
                )

            if self.exit:
                exit(0)
            else:
                self.start_iter, self.end_iter = self.end_iter, (self.end_iter - self.start_iter + self.end_iter)
                self.total_start_time = time.time()

    def _process_time_results(self) -> None:
        """Process and save time profiling results"""
        valid_samples = self._filtered_time_samples()
        avg_time = sum(valid_samples) / len(valid_samples)
        print(f"Average iteration time is: {avg_time:.4f} s")

        args = self.args
        if args.profile.profile_forward:
            assert self.layernum_list is not None
            time_config_path = self.time_profiling_path()
            save_profiled_time(
                time_config_path, avg_time * 1e3, args.train.global_batch_size, self.layernum_list, self.seqlen_list
            )

        if self.exit:
            exit(0)
        else:
            self.time_list = []
            self.start_iter, self.end_iter = self.end_iter, (self.end_iter - self.start_iter + self.end_iter)
            torch.cuda.synchronize()
            self.start.record()

    def _filtered_time_samples(self) -> List[float]:
        """Apply iter0 warmup removal and 3-sigma filtering."""
        if len(self.time_list) == 0:
            raise RuntimeError("No timing samples are available for processing.")

        samples = list(self.time_list)
        if self.start_iter == 0 and len(samples) > 1:
            samples = samples[1:]

        if len(samples) <= 2:
            return samples

        mean = float(np.mean(samples))
        std = float(np.std(samples))
        if std == 0:
            return samples

        lower, upper = mean - 3 * std, mean + 3 * std
        filtered = [x for x in samples if lower <= x <= upper]
        return filtered if len(filtered) > 0 else samples

    def _log_iteration_stats(
        self,
        iter: int,
        iter_time: float,
        loss: Optional[torch.Tensor],
        learning_rate: Optional[float],
        grad_norm: Optional[float],
    ) -> None:
        """Log iteration statistics

        Args:
            iter: Current iteration number
            iter_time: Iteration time in seconds
            loss: Training loss value
            learning_rate: Current learning rate
            grad_norm: Gradient norm
        """
        if loss is None:
            print(iter_time)
        else:
            log_parts = [
                "| Iteration: {:6d} | Consumed samples: {:12d} | ",
                "Elapsed time per iteration (ms): {:.1f} | ",
                "Learning rate: {:.6e} | Loss: {:.6e} | ",
                "grad norm: {:.2f} |",
            ]
            message = "".join(log_parts)
            args = self.args
            print(
                message.format(
                    iter + 1,
                    (iter + 1) * args.train.global_batch_size,
                    iter_time * 1e3,
                    (args.train.lr or 0.0) if learning_rate is None else learning_rate,
                    loss.item(),
                    0.0 if grad_norm is None else grad_norm,
                )
            )
