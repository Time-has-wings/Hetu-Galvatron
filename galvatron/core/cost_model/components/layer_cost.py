import numpy as np
from typing import Union
from logging import Logger
from types import SimpleNamespace

from galvatron.core.cost_model.cost_model_args import ModelArgs, TrainArgs, ParallelArgs, ProfileModelArgs, ProfileHardwareArgs
from galvatron.utils.strategy_utils import DPType, LayerStrategy, AttentionStrategy, FFNStrategy

class TimeCostModelBase:
    time_args_list = {
        'ModelArgs':['parameter_size', 'seq_length', 'hidden_size', 'layer_num'],
        'TrainArgs':['mixed_precision', 'async_grad_reduce'],
        'ParallelArgs':['optimal_chunk_func'],
        'ProfileModelArgs': ['forward_computation_time'],
        'ProfileHardwareArgs':['bct_fct_coe', 'extra_overhead', 'comm_coe_dict', 'dp_overlap_coe', 'bct_overlap_coe', 'p2p_comm_coe_dict', 'costmodel_coe', 'allreduce_dict', 'all2all_dict', 'allgather_message_size_to_latency_dict_dict', 'all2all_message_size_to_latency_dict_dict', 'allreduce_latency_per_MB_dict']
    }
    
    def __init__(
        self,
        strategy:Union[LayerStrategy, AttentionStrategy, FFNStrategy], 
        global_batch_size:int = 8, 
        chunks:int = 1,
        model_args: ModelArgs=None, 
        train_args:TrainArgs = None,
        parallel_args:ParallelArgs = None, 
        profile_model_args:ProfileModelArgs = None,
        profile_hardware_args:ProfileHardwareArgs = None,
        logger:Logger = None
    ):
        # [Step 1] assign attibutes
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.logger = logger

        # [Step 2] gather all args into a single namespace
        self.args: SimpleNamespace = SimpleNamespace()
        components = {
            'ModelArgs': model_args, 
            'TrainArgs': train_args, 
            'ParallelArgs': parallel_args, 
            'ProfileModelArgs': profile_model_args, 
            'ProfileHardwareArgs': profile_hardware_args
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
        self.estimate_pp_communication_time()

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
        self.tp_sp_size = strategy.tp_sp_size
        self.checkpoint = strategy.checkpoint
        
        # [Step 2] calculate some information
        self.lbsz = self.global_batch_size // self.chunks // self.dp_size # NOTE still use dp_size rather than sdp_size.
        self.parameter_memory_in_MB = args.parameter_size / self.tp_size
        
        # [Step 3] copy some attributes for easy access
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size
        self.layer_num = args.layer_num # TODO: remove this variable

        if self.tp_sp_size > 1:
            if self.tp_size > 1:
                self.tp_sp_dict = args.allreduce_dict[self.tp_size]
            else:
                self.tp_sp_dict = args.all2all_dict[self.sp_size]
    
    def estimate_computation_time(self):
        """ Estimate computation time including forward and backward time. """
        args = self.args
        
        # [Step 1] estimate forward computation time
        if isinstance(args.forward_computation_time, np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            self.fct = linear_func(self.lbsz / self.tp_sp_size, *args.forward_computation_time) * self.layer_num
        else:
            self.fct = args.forward_computation_time * self.lbsz / self.tp_sp_size * self.layer_num

        # [Step 2] estimate backward computation time
        self.bct = self.fct * args.bct_fct_coe
        if self.checkpoint:
            self.bct += self.fct  
    
    def estimate_dp_communication_time(self):
        args = self.args

        self.dp_message_size = 2 * (self.sdp_size - 1) * (self.parameter_memory_in_MB / self.sdp_size) * self.layer_num
        if args.mixed_precision:
            self.dp_message_size /= 2
        
        self.fsdp_allgather_message_size = self.dp_message_size * 0.5 # TODO: check correctness

        key = f'{self.sdp_size}_0' if self.tp_size != 1 else f'{self.sdp_size}_1'
        self.dc = args.allreduce_latency_per_MB_dict[key]        
        self.dc_overlap = self.dc * args.dp_overlap_coe

        
    def estimate_tp_communication_time(self): # TODO: split tp and sp to different functions
        args = self.args

        if self.tp_sp_size == 1:
            self.tp_communication_time = 0
        else:
            if self.tp_size == 1: # ulysses-sp
                self.tp_sp_comm_num = 4 * self.layer_num # all-to-all fwd 2, bwd 2
                if self.checkpoint:
                    self.tp_sp_comm_num *= 1.5
                select_dict = args.all2all_message_size_to_latency_dict_dict[self.sp_size]
            else: # tensor parallel
                # forward: <all_gather, hidden_states>, <reduce_scatter, hidden_states>
                # backward: <all_gather, data_grad>, <all_gather hidden_states>, <reduce_scatter param_grad>
                # In the backward pass, <all_gather hidden_states> and <reduce_scatter param_grad> can overlap with the computation. 
                # In summary, 
                # forward: 1 <all_gather, hidden_states>, 1 <reduce_scatter, hidden_states>
                # backward: 1 <all_gather, data_grad> (data_grad.shape is the same as hidden_states.shape)
                self.tp_sp_comm_num = 6 * self.layer_num # attention 3 allgather, mlp 3 allgather
                if self.checkpoint:
                    self.tp_sp_comm_num *= 1.5 # TODO: check correctness
                select_dict = args.allgather_message_size_to_latency_dict_dict[self.tp_size]

            message_size_in_MB = self.lbsz * self.seq_length * self.hidden_size * (2 if args.mixed_precision else 4) / 1024 / 1024
            if message_size_in_MB in select_dict:
                message_time = select_dict[message_size_in_MB]
            else:
                def linear_func(x, m, c):
                    return m * x + c
                message_time = linear_func(message_size_in_MB, *select_dict["popt"])

            self.tp_communication_time = message_time * self.tp_sp_comm_num
  
    def estimate_pp_communication_time(self):
        args = self.args
        self.p2p_comm_coe = None
        if self.pp_size > 1 and args.p2p_comm_coe_dict is not None:
            self.p2p_comm_coe = args.p2p_comm_coe_dict[self.pp_size]
            self.p2p_message_size = self.pp_size * 2 * self.lbsz * self.seq_length * self.hidden_size * 4 / 1024 / 1024
            if args.mixed_precision:
                self.p2p_message_size = self.p2p_message_size / 2

    def bct_dp_overlap(self, dp_message_size, bct):
        args = self.args
        dp_overlap_time = dp_message_size * self.dc_overlap
        bct_overlap_time = bct * args.bct_overlap_coe
        if dp_overlap_time > bct_overlap_time:
            overlap_part = bct_overlap_time
            rest_part = (dp_message_size - bct_overlap_time / self.dc_overlap) * self.dc
            rest_dp_flag = True
        elif dp_overlap_time < bct_overlap_time:
            overlap_part = dp_overlap_time
            rest_part = (bct - dp_overlap_time / args.bct_overlap_coe) 
            rest_dp_flag = False
        else:
            overlap_part = bct_overlap_time
            rest_part = 0
            rest_dp_flag = False
        rest_dp_flag = False
        return overlap_part, rest_part, rest_dp_flag
    
    def get_result(self, no_gradient_sync:bool = False):
        factor = 1 if not no_gradient_sync else 0
        args = self.args
        if self.tp_sp_size == 1 and self.dp_size > 1: # pp+dp
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size * factor, self.bct)
            overall_overhead = self.fct + overlap_part + rest_part + args.extra_overhead
            result = overall_overhead
        elif self.dp_size == 1 and self.tp_sp_size > 1: # pp+tp
            result = self.fct + self.bct + self.tp_communication_time
        elif self.dp_size == 1 and self.tp_sp_size == 1: # pure pp
            result = self.fct + self.bct
        else: # pp+dp+tp
            overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size * factor, self.bct)
            overall_overhead = self.fct + overlap_part + rest_part + self.tp_communication_time + args.extra_overhead
            result = overall_overhead

        # For fsdp, add allgather time of forward and backward
        # TODO: add overlap when fsdp is used
        if self.dp_type == DPType.ZERO3:
            forward_allgather_time = self.fsdp_allgather_message_size * self.dc
            result = result + forward_allgather_time

        if self.pp_size > 1 and self.p2p_comm_coe is not None: # TODO: split mode pp communication time to a new estimation file
            result = result + self.p2p_message_size * self.p2p_comm_coe
        
        coe = 0.001 * args.costmodel_coe
        result = result * coe
        result = result / self.layer_num
        return result

    def gen_result(self) -> tuple[float, float]:
        result = self.get_result(no_gradient_sync=False)
        result_no_comm = self.get_result(no_gradient_sync=True)
        return result, result_no_comm

class MemoryCostModelBase:
    memory_args_list = {
        'ModelArgs':['parameter_size'], 
        'TrainArgs':['mixed_precision', 'async_grad_reduce', 'pytorch_context_mem'], 
        'ParallelArgs':['use_zero2_for_dp', 'max_tp_deg', 'sequence_parallel', 'pipeline_type', 'optimal_chunk_func', 'chunks'], 
        'ProfileModelArgs':['tp_activation_per_bsz_dict', 'other_memory_pp_off', 'other_memory_pp_on']
    }
    
    def __init__(
        self, 
        strategy:Union[LayerStrategy, AttentionStrategy, FFNStrategy], 
        global_batch_size:int = 8, 
        chunks:int = 1,
        stage_idx: int = 0,
        logger:Logger = None,
        model_args: ModelArgs = None,
        train_args: TrainArgs = None,
        parallel_args: ParallelArgs = None,
        profile_model_args: ProfileModelArgs = None,
    ):
        assert all(x is not None for x in (model_args, train_args, parallel_args, profile_model_args)), "One or more variables are None"

        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.stage_idx = stage_idx
        self.logger = logger

        # Aggregate all arguments
        self.args = SimpleNamespace()
        components = {
            'ProfileModelArgs': profile_model_args, 
            'ModelArgs': model_args, 
            'TrainArgs': train_args, 
            'ParallelArgs': parallel_args
        }
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in self.memory_args_list[class_name]:
                    setattr(self.args, key, value)
        
        self.initialize()
        self.estimate_parameter_size()
        self.estimate_model_states_size()
        self.estimate_activation_size()

    def initialize(self):
        args = self.args
        
        # [initialize]:initialize strategy
        strategy = self.strategy
        self.pp_size = strategy.pp_size
        self.tp_size = strategy.tp_size
        self.sp_size = strategy.sp_size
        self.cp_size = strategy.cp_size
        self.dp_size = strategy.dp_size
        self.dp_type:DPType = strategy.dp_type
        self.sdp_size = strategy.sdp_size
        self.tp_sp_size = strategy.tp_sp_size
        self.checkpoint = strategy.checkpoint
    
        # [initialize]:initialize local batch size and cumulative local batch size
        self.lbsz = self.global_batch_size // self.chunks // self.dp_size
        if self.pp_size == 1:
            self.cumulative_num = 1
        else:
            if args.pipeline_type == 'pipedream_flush':
                assert self.chunks >= self.pp_size, f'chunks {self.chunks} must be greater than or equal to pp_size {self.pp_size}'
                self.cumulative_num = self.pp_size - self.stage_idx
            elif args.pipeline_type == 'gpipe':
                assert self.chunks >= self.pp_size, f'chunks {self.chunks} must be greater than or equal to pp_size {self.pp_size}'
                self.cumulative_num = self.chunks
        self.cumulative_lbsz = self.cumulative_num * self.lbsz

        # [initialize]:initialize zero2 and zero3 ratio
        if self.chunks == 1:
            self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            self.zero3_ratio = lambda d: (1/d + 0.003)
        else:
            if args.async_grad_reduce:
                self.zero2_ratio = (lambda d: (6/8 * (1/d + 0.003) + 2/8)) if args.mixed_precision else (lambda d: (2/4 * (1/d + 0.003) + 2/4))
                self.zero3_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            else:
                self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8) * 5/4) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                self.zero3_ratio = lambda d: (1/d + 0.003) * 5/4
                # *5/4: for fp32 grad 
    
    def estimate_parameter_size(self):
        args = self.args
        self.parameter_memory = args.parameter_size / self.tp_size
        
    def estimate_model_states_size(self):
        self.model_states_size = 4 * self.parameter_memory
        if self.dp_type == DPType.ZERO3:
            self.model_states_size *= self.zero3_ratio(self.sdp_size)
        elif self.dp_type == DPType.ZERO2:
            self.model_states_size *= self.zero2_ratio(self.sdp_size)
        
    def estimate_activation_size(self):
        args = self.args
        if self.checkpoint:
            self.activation_size = args.tp_activation_per_bsz_dict['checkpoint'] * self.cumulative_lbsz
            if self.sp_size > 1 or (self.tp_size > 1 and args.sequence_parallel):
                self.activation_size /= self.tp_sp_size
        else:
            self.activation_size = args.tp_activation_per_bsz_dict[self.tp_sp_size] * self.cumulative_lbsz
    
    def get_memory_cost(self):
        result = dict()
        result['parameter'] = self.parameter_memory
        result['model_states'] = self.model_states_size
        result['activation'] = self.activation_size
        result['enc_total'] = self.model_states_size + self.activation_size
        return result 

# class LayerTimeCostModel(TimeCostModelBase):
#     pass

# class LayerMemoryCostModel(MemoryCostModelBase):
#     pass