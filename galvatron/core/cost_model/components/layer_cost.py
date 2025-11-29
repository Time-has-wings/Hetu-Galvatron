import numpy as np
from logging import Logger
from types import SimpleNamespace
from galvatron.utils.strategy_utils import GalvatronStrategy
from ..cost_model_args_optimize import TrainArgsOptimize, ProfileModelArgsOptimize, ProfileHardwareArgsOptimize, UtilsArgsOptimize, VersionOptionArgsOptimize
from ..cost_model_args_optimize import EstimateTPTimeType
from ..cost_model_args_optimize import LogPrint


time_args_list = {
    'TrainArgsOptimize': ['seq_length', 'hidden_size', 'mixed_precision', 'sequence_parallel'],
    'ProfileModelArgsOptimize': ['forward_computation_time', 'parameter_memory'],
    'ProfileHardwareArgsOptimize': ['bct_fct_coe', 'overlap_slowdown_coe', 
                                    'p2p_comm_coe_dict',
                                    'allreduce_fixed_dict', 'allreduce_fit_dict',
                                    'all_gather_fixed_dict', 'all_gather_fit_dict',
                                    'reduce_scatter_fixed_dict', 'reduce_scatter_fit_dict',
                                    'all2all_fit_dict'],
    'UtilsArgsOptimize': ['extra_overhead', 'costmodel_coe', 'dummy_layer_num'],
    'VersionOptionArgsOptimize': ['estimate_tp_time_type'],
}

class TimeCostModelBase(LogPrint):
    def __init__(self,
                strategy: GalvatronStrategy,
                global_batch_size:int = 8,
                chunks: int = 1,
                no_sync_gradient: bool = False,
                logger: Logger = None,
                train_args: TrainArgsOptimize = None,
                profile_model_args: ProfileModelArgsOptimize = None,
                profile_hardware_args: ProfileHardwareArgsOptimize = None,
                utils_args: UtilsArgsOptimize = None,
                version_option_args: VersionOptionArgsOptimize = None,
                ):
        # [Step 0] init parent class
        super().__init__()
        
        # [Step 1] assign attributes
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.no_sync_gradient = no_sync_gradient
        self.logger = logger
        
        # [Step 2] gather all args into a single namespace
        self.args = SimpleNamespace()
        components = {
                    'TrainArgsOptimize': train_args, 
                    'ProfileModelArgsOptimize': profile_model_args, 
                    'ProfileHardwareArgsOptimize': profile_hardware_args, 
                    'UtilsArgsOptimize': utils_args, 
                    'VersionOptionArgsOptimize': version_option_args
        }
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in time_args_list[class_name]:
                    setattr(self.args, key, value) 
        
        # [Step 3] initialize and estimate time  
        self.initialize()
        self.estimate_computation_time()
        self.estimate_dp_communication_time()
        self.estimate_tp_communication_time()
    
    def initialize(self):
        args = self.args
        
        # [Step 1] initialize strategy related attributes
        self.pp_size = self.strategy.pp_size
        self.tp_size = self.strategy.tp_size
        self.dp_size = self.strategy.dp_size
        self.cp_size = self.strategy.cp_size
        self.use_ulysses = self.strategy.use_ulysses
        self.checkpoint = self.strategy.checkpoint
        self.dp_type = self.strategy.dp_type
        self.sdp_size = self.tp_size * self.dp_size if self.use_ulysses else self.dp_size

        # [Step 2] calculate some information
        self.mbsz = self.global_batch_size // self.chunks // self.dp_size # NOTE still use dp_size rather than sdp_size.
        self.parameter_memory = args.parameter_memory if self.use_ulysses else args.parameter_memory / self.tp_size # used for calculating gradient reduce message size

        # [Step 3] copy some attributes for easy access
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size        
        
    def estimate_computation_time(self):
        """
            Estimate computation time including forward and backward time.
        """
        args = self.args
        
        # [Step 1] estimate forward computation time
        if isinstance(args.forward_computation_time, np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            self.fct = linear_func(self.mbsz / self.tp_size, *args.forward_computation_time) # (self.mbsz / self.tp_size) means computation split over tp devices.
        else:
            self.fct = args.forward_computation_time * self.mbsz / self.tp_size # (self.mbsz / self.tp_size) means computation split over tp devices.

        # [Step 2] estimate backward computation time
        self.bct = self.fct * args.bct_fct_coe
        if self.checkpoint:
            self.bct += self.fct
                    
    def estimate_dp_communication_time(self):
        """
            Estimate data parallel gradient message size and communication coefficients.
            Estimate fsdp communication time if necessary.    
        """
        args = self.args
        
        if self.sdp_size == 1:
            self.dc = 0
            self.dc_slowdown = 0
            self.gradient_message_size = 0
            self.fsdp_allgather_message_size = 0
            self.fsdp_allgather_time = 0
        else:
            # [Step 1] get dc and dc_slowdown
            if self.use_ulysses:
                self.dc = args.allreduce_fixed_dict[f'{self.sdp_size}_1']
            else:
                if self.tp_size == 1:
                    self.dc = args.allreduce_fixed_dict[f'{self.sdp_size}_1']
                else:
                    self.dc = args.allreduce_fixed_dict[f'{self.sdp_size}_0']
            
            self.dc_slowdown = self.dc * args.overlap_slowdown_coe
            
            # [Step 2] calculate gradient reduction message size
            # 2 * (self.sdp_size - 1) indicates that ring-allreduce requires (sdp_size - 1) reduce-scatters and (sdp_size - 2) all-gathers,
            # self.parameter_memory / self.sdp_size represents the amount sent or received each time
            self.gradient_message_size = 2 * (self.sdp_size - 1) * (self.parameter_memory / self.sdp_size) # in MB
            if args.mixed_precision:
                self.gradient_message_size /= 2
            if self.no_sync_gradient:
                self.gradient_message_size = 0
            
            # [Step 3] calculate fsdp message size and time if necessary
            self.fsdp_allgather_message_size = (self.sdp_size - 1) * (self.parameter_memory / self.sdp_size) if self.dp_type == 'zero3' else 0
            self.fsdp_allgather_time = self.fsdp_allgather_message_size * self.dc # NOTE consider overlap
            if self.checkpoint:
                self.fsdp_allgather_time *= 2
        
    def estimate_tp_communication_time(self):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the method 'estimate_tp_communication_time'.")

    def bct_dp_overlap(self, gradient_message_size, bct):
        args = self.args
        dp_slowdown_time = gradient_message_size * self.dc_slowdown
        bct_slowdown_time = bct * args.overlap_slowdown_coe
        
        if dp_slowdown_time > bct_slowdown_time:
            overlap_part = bct_slowdown_time
            rest_part = (gradient_message_size - overlap_part / self.dc_slowdown) * self.dc
            rest_dp_flag = True
        elif dp_slowdown_time < bct_slowdown_time:
            overlap_part = dp_slowdown_time
            rest_part = bct - overlap_part / args.overlap_slowdown_coe
            rest_dp_flag = False
        else:
            overlap_part = dp_slowdown_time
            rest_part = 0
            rest_dp_flag = False
        return overlap_part, rest_part, rest_dp_flag 
    
    def gen_result(self):
        args = self.args
        
        if self.tp_size == 1 and self.dp_size > 1: # without tp
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.gradient_message_size, self.bct)
            result = self.fct + overlap_part + rest_part
        elif self.tp_size > 1 and self.dp_size == 1: # without dp
            result = self.fct + self.bct + self.tp_communication_time
        elif self.tp_size == 1 and self.dp_size == 1: # without dp and tp
            result = self.fct + self.bct
        else: # with dp and tp
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.gradient_message_size, self.bct)
            result = self.fct + overlap_part + rest_part + self.tp_communication_time
        
        if self.dp_type == 'zero3':
            result += self.fsdp_allgather_time
            
        ms_to_s = 1e-3
        result = result * ms_to_s * args.costmodel_coe
        
        return result

class LayerTimeCostModelOptimize(TimeCostModelBase):
    def estimate_tp_communication_time(self):
        args = self.args
        
        if self.tp_size == 1:
            self.tp_communication_time = 0
        else:
            if args.estimate_tp_time_type == EstimateTPTimeType.FIXED:
                # forward: <all_gather, hidden_states>, <reduce_scatter, hidden_states>, <all_gather, hidden_states>, <reduce_scatter, hidden_states>
                # backward: <all_gather, data_grad>, <all_gather hidden_states>, <reduce_scatter param_grad>, <all_gather, data_grad>, <all_gather hidden_states>, <reduce_scatter param_grad>
                # NOTE In the backward pass, <allgather hidden_states> and <reduce_scatter param_grad> can overlap with the computation. 
                if args.sequence_parallel:
                    dtype_size = 2 if args.mixed_precision else 4
                    byte_to_MB = 1024 * 1024
                    per_tp_message_size_in_MB = (self.tp_size - 1) * self.mbsz * (self.seq_length / self.tp_size) * self.hidden_size * dtype_size / byte_to_MB
                    
                    self.fwd_tp_communication_time = per_tp_message_size_in_MB * args.all_gather_fixed_dict[f'{self.tp_size}_1'] * 2 + per_tp_message_size_in_MB * args.reduce_scatter_fixed_dict[f'{self.tp_size}_1'] * 2
                    self.bwd_tp_communication_time = per_tp_message_size_in_MB * args.all_gather_fixed_dict[f'{self.tp_size}_1'] * 2

                    if self.checkpoint == False:
                        self.tp_communication_time = self.fwd_tp_communication_time + self.bwd_tp_communication_time
                    else:
                        self.tp_communication_time = self.fwd_tp_communication_time * 2 + self.bwd_tp_communication_time
                else: # use allreduce
                    raise NotImplemented('allreduce not implement')
            elif args.estimate_tp_time_type == EstimateTPTimeType.FIT:
                if args.sequence_parallel:
                    dtype_size = 2 if args.mixed_precision else 4
                    byte_to_MB = 1024 * 1024
                    
                    per_tp_message_size_in_byte = (self.tp_size - 1) * self.mbsz * (self.seq_length / self.tp_size) * self.hidden_size * dtype_size
                    selected_dict = args.all2all_fit_dict[self.tp_size] if self.use_ulysses else args.all_gather_fit_dict[self.tp_size]

                    if per_tp_message_size_in_byte in selected_dict:
                        per_operation_time = selected_dict[per_tp_message_size_in_byte]
                    else:
                        def linear_func(x, m, c):
                            return m * x + c
                        per_operation_time = linear_func(per_tp_message_size_in_byte / byte_to_MB, *selected_dict['popt'])
                    
                    self.fwd_tp_communication_time = per_operation_time * 4
                    self.bwd_tp_communication_time = per_operation_time * 2
                    
                    if self.checkpoint == False:
                        self.tp_communication_time = self.fwd_tp_communication_time + self.bwd_tp_communication_time
                    else:
                        self.tp_communication_time = self.fwd_tp_communication_time * 2 + self.bwd_tp_communication_time
                else: # use allreduce
                    raise NotImplemented('allreduce not implement')


memory_args_list = {
        'TrainArgsOptimize': ['mixed_precision', 'async_grad_reduce', 'use_zero2_for_dp', 'sequence_parallel', 'pipeline_type'],
        'ProfileModelArgsOptimize': ['parameter_memory', 'tp_activation_per_bsz_dict' ],
        'VersionOptionArgsOptimize': ['zero_with_slight_noise'],
    }


class MemoryCostModelBase(LogPrint):
    def __init__(self,
                 strategy:GalvatronStrategy,
                 global_batch_size:int,
                 chunks:int,
                 stage_idx:int = 0,
                 logger:Logger=None,
                 train_args:TrainArgsOptimize=None,
                 profile_model_args:ProfileModelArgsOptimize=None,
                 version_option_args:VersionOptionArgsOptimize=None,
                 ):
        super().__init__()

        # [Step 1] assign attributes
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.stage_idx = stage_idx
        self.logger = logger
        
        # [Step 2] gather all args into a single namespace
        self.args = SimpleNamespace()
        components = {
                    'TrainArgsOptimize': train_args, 
                    'ProfileModelArgsOptimize': profile_model_args, 
                    'VersionOptionArgsOptimize': version_option_args,
        }
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in memory_args_list[class_name]:
                    setattr(self.args, key, value) 

        # [Step 3] initialize and estimate 
        self.initialize()
        self.estimate_model_states_memory()
        self.estimate_activation_memory()
    
    def initialize(self):
        args = self.args

        # [Step 1] initialize strategy related attributes
        self.pp_size = self.strategy.pp_size
        self.tp_size = self.strategy.tp_size
        self.dp_size = self.strategy.dp_size
        self.cp_size = self.strategy.cp_size
        self.use_ulysses = self.strategy.use_ulysses
        self.checkpoint = self.strategy.checkpoint
        self.dp_type = self.strategy.dp_type
        self.sdp_size = self.tp_size * self.dp_size if self.use_ulysses else self.dp_size

        # [Step 2] calculate some informations
        self.mbsz = self.global_batch_size // self.chunks // self.dp_size # NOTE still use dp_size rather than sdp_size.
        if self.pp_size == 1:
            self.local_bsz = self.mbsz
        else:
            if args.pipeline_type == 'pipedream_flush':
                assert self.chunks >= self.pp_size
                self.local_bsz = self.mbsz * (self.pp_size - self.stage_idx)
            elif args.pipeline_type == 'gpipe':
                self.local_bsz = self.mbsz * self.chunks
            
        # [Step 3] calculate zero ratio
        if args.zero_with_slight_noise:
            if self.chunks == 1:
                self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                self.zero3_ratio = lambda d: (1/d + 0.003)
            else:
                # self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                # self.zero3_ratio = lambda d: (1/d + 0.003)
                if args.async_grad_reduce:
                    self.zero2_ratio = (lambda d: (6/8 * (1/d + 0.003) + 2/8)) if args.mixed_precision else (lambda d: (2/4 * (1/d + 0.003) + 2/4))
                    self.zero3_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                else:
                    self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                    self.zero3_ratio = lambda d: (1/d + 0.003)
                    # *5/4: for fp32 grad 
        else:
            if self.chunks == 1:
                self.zero2_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                self.zero3_ratio = lambda d: (1/d)
            else:
                # print('use this')
                # self.zero2_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                # self.zero3_ratio = lambda d: (1/d)
                if args.async_grad_reduce:
                    self.zero2_ratio = (lambda d: (6/8 * (1/d) + 2/8)) if args.mixed_precision else (lambda d: (2/4 * (1/d) + 2/4))
                    self.zero3_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                else:
                    self.zero2_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                    self.zero3_ratio = lambda d: (1/d)
    
    def estimate_model_states_memory(self):
        args = self.args

        if self.use_ulysses:
            self.parameter_memory = args.parameter_memory
        else:
            self.parameter_memory = args.parameter_memory / self.tp_size

        self.model_states_memory = 4 * self.parameter_memory
        if self.dp_type == 'zero3':
            self.model_states_memory *= self.zero3_ratio(self.sdp_size)
        elif self.dp_type == 'zero2':
            self.model_states_memory *= self.zero2_ratio(self.sdp_size)
    
    def estimate_activation_memory(self):
        args = self.args
        if self.checkpoint:
            assert(args.tp_activation_per_bsz_dict['checkpoint'] is not None)
            self.activation_memory = args.tp_activation_per_bsz_dict['checkpoint'] * self.local_bsz
            if args.sequence_parallel:
                self.activation_memory /= self.tp_size
        else:
            self.activation_memory = args.tp_activation_per_bsz_dict[self.tp_size] * self.local_bsz

    def get_memory_cost(self):
        result = dict()
        result['parameter_memory'] = self.parameter_memory
        result['model_states_memory'] = self.model_states_memory
        result['activation_memory'] = self.activation_memory
        result['total_memory'] = self.model_states_memory + self.activation_memory
        return result
    
class LayerMemoryCostModelOptimize(MemoryCostModelBase):
    pass