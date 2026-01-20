import numpy as np
from logging import Logger
from typing import Union
from types import SimpleNamespace
from ..cost_model_args import TrainArgs, ProfileModelArgs, ProfileHardwareArgs, UtilsArgs, VersionOptionArgs, EstimateTPTimeType
from galvatron.utils.strategy_utils import LayerStrategy, byte_to_MB, model_states_to_param_size_ratio, AttentionStrategy, FFNStrategy, DPType


class TimeCostModelBase:
    time_args_list = {
        'TrainArgs': ['seq_length', 'hidden_size', 'mixed_precision', 'sequence_parallel'],
        'ProfileModelArgs': ['forward_computation_time', 'parameter_memory'],
        'ProfileHardwareArgs': ['bct_fct_coe', 'overlap_slowdown_coe', 
                                        'p2p_comm_coe_dict',
                                        'allreduce_fixed_dict', 'allreduce_fit_dict',
                                        'all_gather_fixed_dict', 'all_gather_fit_dict',
                                        'reduce_scatter_fixed_dict', 'reduce_scatter_fit_dict',
                                        'all2all_fit_dict'],
        'UtilsArgs': ['extra_overhead', 'costmodel_coe', 'dummy_layer_num'],
        'VersionOptionArgs': ['estimate_tp_time_type'],
    }
    
    def __init__(
        self,
        strategy:Union[LayerStrategy, AttentionStrategy, FFNStrategy],
        global_batch_size:int = 8,
        chunks:int = 1,
        no_sync_gradient:bool = False,
        logger:Logger = None,
        train_args: TrainArgs = None,
        profile_model_args: ProfileModelArgs = None,
        profile_hardware_args: ProfileHardwareArgs = None,
        utils_args: UtilsArgs = None,
        version_option_args: VersionOptionArgs = None
    ):
        # [Step 1] assign attibutes
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.no_sync_gradient = no_sync_gradient
        self.logger = logger

        # [Step 2] gather all args into a single namespace
        self.args: SimpleNamespace = SimpleNamespace()
        components = {
            'TrainArgs': train_args, 
            'ProfileModelArgs': profile_model_args, 
            'ProfileHardwareArgs': profile_hardware_args, 
            'UtilsArgs': utils_args, 
            'VersionOptionArgs': version_option_args
        }
        for class_name, instance in components.items():
            assert instance is not None, f'{class_name} is None'
            for key, value in instance.__dict__.items():
                if key in self.time_args_list[class_name]:
                    setattr(self.args, key, value) 
        
        # [Step 3] initialize and estimate time  
        self.initialize()
        self.estimate_computation_time()
        self.estimate_dp_communication_time()
        self.estimate_tp_communication_time()
        self.estimate_sp_communication_time()

    def initialize(self):
        args = self.args

        # [Step 1] initialize strategy related attributes
        strategy = self.strategy
        self.pp_size = strategy.pp_size
        self.tp_size = strategy.tp_size
        self.sp_size = strategy.sp_size
        self.cp_size = strategy.cp_size
        self.dp_size = strategy.dp_size
        self.dp_type:DPType = strategy.dp_type
        self.sdp_size = strategy.sdp_size
        self.checkpoint = strategy.checkpoint
        
        # [Step 2] calculate some information
        self.mbsz = self.global_batch_size // self.chunks // self.dp_size # NOTE still use dp_size rather than sdp_size.
        self.parameter_memory_in_MB = args.parameter_memory / self.tp_size
        
        # [Step 3] copy some attributes for easy access
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size
        if self.tp_size > 1:
            assert self.sp_size == 1, f'When tp_size > 1, sp_size must be 1, but got {self.sp_size}'
            if args.sequence_parallel:
                self.seq_length_per_rank = self.seq_length / self.cp_size / self.tp_size
            else:
                self.seq_length_per_rank = self.seq_length / self.cp_size
        else:
            self.seq_length_per_rank = self.seq_length / self.cp_size / self.sp_size
    
    def estimate_computation_time(self):
        """ Estimate computation time including forward and backward time. """
        args = self.args
        
        # [Step 1] estimate forward computation time
        if isinstance(args.forward_computation_time, np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            self.fct = linear_func(self.mbsz / self.tp_size / self.sp_size / self.cp_size, *args.forward_computation_time)
        else:
            self.fct = args.forward_computation_time * self.mbsz / self.tp_size / self.sp_size / self.cp_size

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
            self.gradient_message_size_in_MB = 0
            self.fsdp_allgather_message_size_in_MB = 0
            self.fsdp_allgather_time = 0
        else:
            # [Step 1] get dc and dc_slowdown
            if self.tp_size != 1:
                self.dc = args.allreduce_fixed_dict[f'{self.sdp_size}_0']
            else:
                self.dc = args.allreduce_fixed_dict[f'{self.sdp_size}_1']
            self.dc_slowdown = self.dc * args.overlap_slowdown_coe
            
            # [Step 2] calculate gradient reduction message size
            # 2 * (self.sdp_size - 1) indicates that ring-allreduce requires (sdp_size - 1) reduce-scatters and (sdp_size - 2) all-gathers,
            # self.parameter_memory / self.sdp_size represents the amount sent or received each time
            self.gradient_message_size_in_MB = 2 * (self.sdp_size - 1) * (self.parameter_memory_in_MB / self.sdp_size) # in MB
            if args.mixed_precision:
                self.gradient_message_size_in_MB /= 2
            if self.no_sync_gradient:
                self.gradient_message_size_in_MB = 0
            
            # [Step 3] calculate fsdp message size and time if necessary
            self.fsdp_allgather_message_size_in_MB = (self.sdp_size - 1) * (self.parameter_memory_in_MB / self.sdp_size) if self.dp_type == DPType.ZERO3 else 0
            self.fsdp_allgather_coe = self.dc / 2 # allreduce = allgather + reduce_scatter
            self.fsdp_allgather_time = self.fsdp_allgather_message_size_in_MB * self.fsdp_allgather_coe
            self.fsdp_allgather_time *= 2 # forward once and backward once
            if self.checkpoint:
                self.fsdp_allgather_time *= 1.5

    def estimate_tp_communication_time(self):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the method 'estimate_tp_communication_time'.")

    def estimate_sp_communication_time(self):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the method 'estimate_sp_communication_time'.")

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
        
        if self.dp_size == 1: # without dp
            result = self.fct + self.bct + self.tp_communication_time + self.sp_communication_time
        elif self.tp_size == 1 and self.sp_size == 1 and self.dp_size > 1: # only dp
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.gradient_message_size_in_MB, self.bct)
            result = self.fct + overlap_part + rest_part
        else:
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.gradient_message_size_in_MB, self.bct)
            result = self.fct + overlap_part + rest_part + self.tp_communication_time + self.sp_communication_time
        
        if self.dp_type == DPType.ZERO3:
            result += self.fsdp_allgather_time
            
        ms_to_s = 1e-3
        result = result * ms_to_s * args.costmodel_coe
        
        return result

class LayerTimeCostModel(TimeCostModelBase):
    def estimate_tp_communication_time(self):
        args = self.args
        
        if self.tp_size == 1:
            self.tp_communication_time = 0
        else:
            if args.estimate_tp_time_type == EstimateTPTimeType.FIXED:
                print(f'\nWarning: Fixed version of TP communication time is deprecated. Please use the FIT version instead.')
                if args.sequence_parallel:
                    # forward: <all_gather, hidden_states>, <reduce_scatter, hidden_states>, <all_gather, hidden_states>, <reduce_scatter, hidden_states>
                    # backward: <all_gather, data_grad>, <all_gather hidden_states>, <reduce_scatter param_grad>, <all_gather, data_grad>, <all_gather hidden_states>, <reduce_scatter param_grad>
                    # In the backward pass, <all_gather hidden_states> and <reduce_scatter param_grad> can overlap with the computation. 
                    # In summary, 
                    # forward: 2 <all_gather, hidden_states>, 2 <reduce_scatter, hidden_states>
                    # backward: 2 <all_gather, data_grad> (data_grad.shape is the same as hidden_states.shape)
                    dtype_size = 2 if args.mixed_precision else 4
                    per_tp_message_size_in_MB = self.mbsz * (self.seq_length_per_rank * (self.tp_size - 1)) * self.hidden_size * dtype_size / byte_to_MB
                    
                    self.fwd_tp_communication_time = (per_tp_message_size_in_MB * args.all_gather_fixed_dict[f'{self.tp_size}_1'] * 2 + 
                                                      per_tp_message_size_in_MB * args.reduce_scatter_fixed_dict[f'{self.tp_size}_1'] * 2)
                    self.bwd_tp_communication_time = per_tp_message_size_in_MB * args.all_gather_fixed_dict[f'{self.tp_size}_1'] * 2

                    if not self.checkpoint:
                        self.tp_communication_time = self.fwd_tp_communication_time + self.bwd_tp_communication_time
                    else:
                        self.tp_communication_time = self.fwd_tp_communication_time * 2 + self.bwd_tp_communication_time
                else:
                    # Roughly estimate using 4 allreduce operations, or 6 allreduce operations if checkpointing is enabled
                    all_reduce_nums = 4 if not self.checkpoint else 6
                    dtype_size = 2 if args.mixed_precision else 4
                    per_tp_message_size_in_MB = self.mbsz * (self.seq_length_per_rank * (self.tp_size - 1)) * self.hidden_size * dtype_size / byte_to_MB
                    self.tp_communication_time = per_tp_message_size_in_MB * args.allreduce_fixed_dict[f'{self.tp_size}_1'] * all_reduce_nums
            elif args.estimate_tp_time_type == EstimateTPTimeType.FIT:
                if args.sequence_parallel:
                    # forward: <all_gather, hidden_states>, <reduce_scatter, hidden_states>, <all_gather, hidden_states>, <reduce_scatter, hidden_states>
                    # backward: <all_gather, data_grad>, <all_gather hidden_states>, <reduce_scatter param_grad>, <all_gather, data_grad>, <all_gather hidden_states>, <reduce_scatter param_grad>
                    # In the backward pass, <all_gather hidden_states> and <reduce_scatter param_grad> can overlap with the computation. 
                    # In summary, 
                    # forward: 2 <all_gather, hidden_states>, 2 <reduce_scatter, hidden_states>
                    # backward: 2 <all_gather, data_grad> (data_grad.shape is the same as hidden_states.shape)
                    dtype_size = 2 if args.mixed_precision else 4
                    per_tp_message_size_in_byte = self.mbsz * (self.seq_length_per_rank * self.tp_size) * self.hidden_size * dtype_size

                    tp_dict = args.all_gather_fit_dict[self.tp_size]
                    if per_tp_message_size_in_byte in tp_dict:
                        per_tp_operation_time = tp_dict[per_tp_message_size_in_byte]
                    else:
                        def linear_func(x, m, c):
                            return m * x + c
                        per_tp_operation_time = linear_func(per_tp_message_size_in_byte / byte_to_MB, *tp_dict['popt'])

                    self.fwd_tp_communication_time = per_tp_operation_time * 4
                    self.bwd_tp_communication_time = per_tp_operation_time * 2
                    
                    if self.checkpoint == False:
                        self.tp_communication_time = self.fwd_tp_communication_time + self.bwd_tp_communication_time
                    else:
                        self.tp_communication_time = self.fwd_tp_communication_time * 2 + self.bwd_tp_communication_time
                else:
                    # Roughly estimate using 4 allreduce operations, or 6 allreduce operations if checkpointing is enabled
                    all_reduce_nums = 4 if not self.checkpoint else 6
                    dtype_size = 2 if args.mixed_precision else 4
                    per_tp_message_size_in_byte = self.mbsz * (self.seq_length_per_rank * self.tp_size) * self.hidden_size * dtype_size

                    tp_dict = args.allreduce_fit_dict[self.tp_size]
                    if per_tp_message_size_in_byte in tp_dict:
                        per_tp_operation_time = tp_dict[per_tp_message_size_in_byte]
                    else:
                        def linear_func(x, m, c):
                            return m * x + c
                        per_tp_operation_time = linear_func(per_tp_message_size_in_byte / byte_to_MB, *tp_dict['popt'])

                    self.tp_communication_time = per_tp_operation_time * all_reduce_nums

    def estimate_sp_communication_time(self):
        args = self.args

        if self.sp_size == 1:
            self.per_sp_message_size_in_byte = 0
            self.sp_per_operation_time = 0
            self.sp_communication_num = 0
            self.sp_communication_time = 0
        else:
            sp_dict = args.all2all_fit_dict[self.sp_size]
            dtype_size = 2 if args.mixed_precision else 4

            self.per_sp_message_size_in_byte = self.mbsz * (self.seq_length_per_rank * self.sp_size) * self.hidden_size * dtype_size
            if self.per_sp_message_size_in_byte in sp_dict:
                self.sp_per_operation_time = sp_dict[self.per_sp_message_size_in_byte]
            else:
                def linear_func(x, m, c):
                    return m * x + c
                self.sp_per_operation_time = linear_func(self.per_sp_message_size_in_byte / byte_to_MB, *sp_dict['popt'])

            if self.checkpoint:
                self.sp_communication_num = 6
            else:
                # 4 all_to_all ops in SP (forward: 2, backward: 2)
                self.sp_communication_num = 4
            self.sp_communication_time = self.sp_communication_num * self.sp_per_operation_time

class MemoryCostModelBase:
    layer_memory_args_list = {
        'TrainArgs': ['mixed_precision', 'async_grad_reduce', 'use_zero2_for_dp', 'sequence_parallel', 'pipeline_type'],
        'ProfileModelArgs': ['parameter_memory', 'tp_activation_per_bsz_dict' ],
        'VersionOptionArgs': ['zero_with_slight_noise'],
    }

    def __init__(
        self,
        strategy:Union[LayerStrategy, AttentionStrategy, FFNStrategy],
        global_batch_size:int,
        chunks:int,
        stage_idx:int = 0,
        logger:Logger=None,
        train_args:TrainArgs=None,
        profile_model_args:ProfileModelArgs=None,
        version_option_args:VersionOptionArgs=None,
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
            'TrainArgs': train_args, 
            'ProfileModelArgs': profile_model_args, 
            'VersionOptionArgs': version_option_args,
        }
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in self.layer_memory_args_list[class_name]:
                    setattr(self.args, key, value) 

        # [Step 3] initialize and estimate 
        self.initialize()
        self.estimate_model_states_memory()
        self.estimate_activation_memory()
    
    def initialize(self):
        args = self.args

        # [Step 1] initialize strategy related attributes
        strategy = self.strategy
        self.pp_size = strategy.pp_size
        self.tp_size = strategy.tp_size
        self.sp_size = strategy.sp_size
        self.cp_size = strategy.cp_size
        self.dp_size = strategy.dp_size
        self.checkpoint = strategy.checkpoint
        self.dp_type = strategy.dp_type
        self.sdp_size = strategy.sdp_size

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
        self.parameter_memory_in_MB = args.parameter_memory / self.tp_size
            
        # [Step 3] calculate zero ratio
        if args.zero_with_slight_noise:
            if self.chunks == 1:
                self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                self.zero3_ratio = lambda d: (1/d + 0.003)
            else:
                if args.async_grad_reduce:
                    self.zero2_ratio = (lambda d: (6/8 * (1/d + 0.003) + 2/8)) if args.mixed_precision else (lambda d: (2/4 * (1/d + 0.003) + 2/4))
                    self.zero3_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                else:
                    self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                    self.zero3_ratio = lambda d: (1/d + 0.003)
        else:
            if self.chunks == 1:
                self.zero2_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                self.zero3_ratio = lambda d: (1/d)
            else:
                if args.async_grad_reduce:
                    self.zero2_ratio = (lambda d: (6/8 * (1/d) + 2/8)) if args.mixed_precision else (lambda d: (2/4 * (1/d) + 2/4))
                    self.zero3_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                else:
                    self.zero2_ratio = (lambda d: (7/8 * (1/d) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d) + 1/4))
                    self.zero3_ratio = lambda d: (1/d)
    
    def estimate_model_states_memory(self):
        self.model_states_memory = self.parameter_memory_in_MB * model_states_to_param_size_ratio
        if self.dp_type == DPType.ZERO3:
            self.model_states_memory *= self.zero3_ratio(self.sdp_size)
        elif self.dp_type == DPType.ZERO2:
            self.model_states_memory *= self.zero2_ratio(self.sdp_size)
    
    def estimate_activation_memory(self):
        args = self.args
        if self.checkpoint:
            assert(args.tp_activation_per_bsz_dict['checkpoint'] is not None)
            self.activation_memory = args.tp_activation_per_bsz_dict['checkpoint'] * self.local_bsz
            if self.tp_size > 1:
                if args.sequence_parallel:
                    self.activation_memory = self.activation_memory / self.cp_size / self.tp_size
                else:
                    self.activation_memory = self.activation_memory / self.cp_size
            else:
                self.activation_memory = self.activation_memory / self.cp_size / self.sp_size
            if args.sequence_parallel:
                self.activation_memory /= self.tp_size
        else:
            if self.tp_size > 1:
                if args.sequence_parallel:
                    activation_idx = self.cp_size * self.tp_size
                else:
                    activation_idx = self.cp_size
            else:
                activation_idx = self.cp_size * self.sp_size
            self.activation_memory = args.tp_activation_per_bsz_dict[activation_idx] * self.local_bsz

    def get_memory_cost(self) -> dict:
        result = dict()
        result['parameter_memory'] = self.parameter_memory_in_MB
        result['model_states_memory'] = self.model_states_memory
        result['activation_memory'] = self.activation_memory
        result['total_memory'] = self.model_states_memory + self.activation_memory
        return result

class LayerMemoryCostModel(MemoryCostModelBase):
    pass