from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum
from logging import Logger
from typing import List

@dataclass
class ModelArgsOptimize:
    hidden_size: int = 4096
    n_kv_heads:int = 4
    n_heads: int = 32

    # head_dim = hidden_size // n_heads
    # num_query_groups = n_heads // n_kv_heads

    @property
    def head_dim(self):
        return self.hidden_size // self.n_heads
    
    @property
    def num_query_groups(self):
        return self.n_heads // self.n_kv_heads


@dataclass
class TrainArgsOptimize:
    """Basic training arguments"""
    seq_length: int = 1024
    hidden_size: int = 4096
    sequence_length_list: List[int] = field(default_factory=lambda: [1024])
    
    """Optimization related arguments"""
    mixed_precision: bool = False
    
    """Data parallel related arguments"""
    use_zero2_for_dp: bool = False
    async_grad_reduce: bool = True
    
    """Tensor/Sequence parallel related arguments"""
    disable_vtp: bool = False
    sequence_parallel: bool = False
    
    """Pipeline parallel related arguments"""
    pipeline_type: str = 'gpipe'
    
    """Experts parallel related arguments"""
    num_experts: int = 8
    top_k:int = 2
    moe_grouped_gemm: bool = False
    # moe_profile_seq_length:int = 1024
    
    
@dataclass
class ProfileModelArgsOptimize:
    # all / attention / mlp
    forward_computation_time: Optional[Union[float, list]] = 35 / 24
    parameter_memory: float = 2025.0928
    tp_activation_per_bsz_dict: dict = field(default_factory=lambda: {1:85, 2:47, 4:28, 8:18.5})
    
    # other
    other_time_profiled: Optional[Union[float, list]] = 0
    other_memory_pp_off: dict = field(default_factory=lambda: {'model_states': 640, 'activation': 320})
    other_memory_pp_on: dict = field(default_factory=lambda: {'model_states': 640, 'activation': 320})

@dataclass
class ProfileHardwareArgsOptimize:
    """Communication Coefficients related arguments"""
    bct_fct_coe: float = 2
    overlap_slowdown_coe: float = 1.3

    """Communication related arguments"""
    allreduce_fixed_dict: dict = field(default_factory=lambda:{})
    allreduce_fit_dict: dict = field(default_factory=lambda: {})
    all_gather_fixed_dict: dict = field(default_factory=lambda: {})
    all_gather_fit_dict: dict = field(default_factory=lambda: {})
    reduce_scatter_fixed_dict: dict = field(default_factory=lambda: {})
    reduce_scatter_fit_dict: dict = field(default_factory=lambda: {})

    p2p_comm_coe_dict: dict = field(default_factory=lambda: {})

    all2all_fit_dict: dict = field(default_factory=lambda: {})

@dataclass
class UtilsArgsOptimize:
    """Utility related arguments"""
    extra_overhead: float = 0
    costmodel_coe: float = 1.0
    dummy_layer_num: int = 24
    pytorch_context_mem: int = 1024

class EstimateTPTimeType(Enum):
    FIXED = 1 
    FIT = 2        

@dataclass
class VersionOptionArgsOptimize:
    """Version and Iteration related arguments"""
    estimate_tp_time_type: EstimateTPTimeType = EstimateTPTimeType.FIXED
    zero_with_slight_noise: bool = True

class LogPrint:
    def __init__(self):
        self.logger:Logger = None
        self.class_name = self.__class__.__name__
        
    def logger_print(self, message):
        if self.logger is None:
            print(f'[{self.class_name}], {message}')
        else:
            self.logger.info(f'[{self.class_name}] {message}')
