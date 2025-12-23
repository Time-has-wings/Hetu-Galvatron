from .layer_cost import TimeCostModelBase, MemoryCostModelBase
from galvatron.utils.strategy_utils import GalvatronStrategy
from ..cost_model_args_optimize import TrainArgsOptimize, ProfileModelArgsOptimize, ProfileHardwareArgsOptimize, UtilsArgsOptimize, VersionOptionArgsOptimize
from ..cost_model_args_optimize import EstimateTPTimeType
from logging import Logger
from types import SimpleNamespace
import numpy as np
from ..cost_model_args_optimize import LogPrint


moe_ffn_time_args_list = {
    'TrainArgsOptimize': ['seq_length', 'hidden_size', 'mixed_precision', 'num_experts', 'top_k', 'moe_grouped_gemm'],
    'ProfileModelArgsOptimize': ['forward_computation_time', 'parameter_memory'],
    'ProfileHardwareArgsOptimize': ['bct_fct_coe', 'overlap_slowdown_coe', 'comm_coe_dict', 'p2p_comm_coe_dict', 'allreduce_fit_dict', 'all2all_fit_dict', 'allreduce_fixed_dict'],
    'UtilsArgsOptimize': ['extra_overhead', 'costmodel_coe', 'dummy_layer_num'],
    'VersionOptionArgsOptimize': ['estimate_tp_time_type'],
}

moe_ffn_memory_args_list = {
        'TrainArgsOptimize': ['seq_length', 'hidden_size', 'mixed_precision', 'async_grad_reduce', 'use_zero2_for_dp', 'sequence_parallel', 'pipeline_type', 'num_experts', 'top_k', 'moe_grouped_gemm'],
        'ProfileModelArgsOptimize': ['parameter_memory', 'tp_activation_per_bsz_dict' ],
        'VersionOptionArgsOptimize': ['zero_with_slight_noise'],
    }


class MoEFFNTimeCostModelOptimize(LogPrint):
    def __init__(self,
                strategy:GalvatronStrategy,
                global_batch_size:int = 8,
                chunks:int = 1,
                logger:Logger = None,
                train_args: TrainArgsOptimize = None,
                profile_model_args: ProfileModelArgsOptimize = None,
                profile_hardware_args: ProfileHardwareArgsOptimize = None,
                utils_args: UtilsArgsOptimize = None,
                version_option_args: VersionOptionArgsOptimize = None,
        ):
        super().__init__()
        
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.logger = logger
        
        self.args = SimpleNamespace()
        components = {'TrainArgsOptimize': train_args, 'ProfileModelArgsOptimize': profile_model_args, 'ProfileHardwareArgsOptimize': profile_hardware_args, 'UtilsArgsOptimize': utils_args, 'VersionOptionArgsOptimize': version_option_args}
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in moe_ffn_time_args_list[class_name]:
                    setattr(self.args, key, value)
        
        self.initialize()
        self.estimate_computation_time()
        self.estimate_dispatch_communication_time()
        self.estimate_combine_communication_time()
    
    def initialize(self):
        args = self.args
        
        self.pp_size = self.strategy.pp_size
        self.ep_size = self.strategy.ep_size
        self.dp_of_ep_size = self.strategy.dp_of_ep_size
        self.tp_of_ep_size = self.strategy.tp_of_ep_size
        self.tp_ep_size = self.ep_size * self.tp_of_ep_size
        self.checkpoint = self.strategy.checkpoint
        self.world_size = self.strategy.world_size
        
        self.bsz = self.global_batch_size // self.chunks // self.dp_of_ep_size
        
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size
        
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        self.moe_grouped_gemm = args.moe_grouped_gemm
        
        # In the current modeling, we assume that experts are uniformly distributed and their loads are balanced.
        self.local_expert_num = self.num_experts // self.ep_size
        self.token_num_per_expert = self.seq_length * self.top_k / self.num_experts
        self.seq_length_scale_ratio = self.token_num_per_expert / self.seq_length
        
        self.origin_token_num_per_device = self.global_batch_size // self.chunks * self.seq_length // self.world_size
        self.token_num_per_expert_group = self.origin_token_num_per_device * self.ep_size * self.top_k
        dtype_size = 2 if args.mixed_precision else 4
        byte_to_MB = 1024 * 1024
        self.all_to_all_message_size_in_MB = (self.ep_size - 1) * (self.origin_token_num_per_device * self.top_k / self.ep_size) * self.hidden_size * dtype_size / byte_to_MB

        self.token_num_per_device_after_all_to_all = self.origin_token_num_per_device * self.top_k
        self.tp_message_size_in_MB = (self.tp_of_ep_size - 1) * self.token_num_per_device_after_all_to_all * self.hidden_size * dtype_size / byte_to_MB

        self.little_all_gather_message_size_in_MB = (self.tp_ep_size - 1) * self.num_experts * 8 / byte_to_MB # long -> 8 bytes

        # Explanation of the above modeling
        # First, there are a total of (global_batch_size // chunks) * seq_length tokens, and each token needs to select top_k experts.
        # Due to load balancing, the number of tokens per expert is (global_batch_size // chunks) * seq_length // num_experts * top_k.
        # Therefore, in the case of tensor parallelism (tp), the number of tokens per expert is (global_batch_size // chunks) * seq_length // num_experts * top_k // tp_of_ep_size.
        # When using tp, ep_size equals world_size // tp_of_ep_size.
        # At this time, the number of experts on one device is num_experts // ep_size = expert_num // world_size * tp_of_ep_size.
        # Thus, the number of tokens on one device is (num_experts // world_size * tp_of_ep_size) * (global_batch_size // chunks * seq_length // num_experts * top_k // tp_of_ep_size) =
        # (global_batch_size // chunks) * seq_length // world_size * top_k.
        # So the number of tokens on one device is a constant!!!
        # When tp increases, the volume of all_gather communication will increase. This is because an increase in tp leads to a decrease in ep_size,
        # which in turn increases the total number of experts on one device. Since each expert requires complete token data, the total volume of all_gather communication increases accordingly.
        # Therefore, the larger the tp, the longer the communication time.
        # What about the communication volume of all_to_all?
        # When ep_size is determined, the ep_group is also determined, and all_to_all communication will only be performed on the tokens of ep_size devices.
        # The initial number of tokens on one device is (global_batch_size // chunks) * seq_length // world_size (this is true regardless of the strategy adopted in the attention layer).
        # It should be noted that we also need to multiply by top_k, because each token needs to find top_k experts.
        # A single device will split the initial tokens into ep_size parts, and the size of each part is (global_batch_size // chunks) * seq_length // world_size // ep_size.
        # The communication will be performed (ep_size - 1) times according to the above communication volume.

        self.token_num_per_expert = self.global_batch_size // self.chunks * self.seq_length * self.top_k // self.num_experts
        self.equal_bsz = self.token_num_per_expert / self.seq_length
    

    def estimate_computation_time(self):
        args = self.args

        if isinstance(args.forward_computation_time, np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            if self.moe_grouped_gemm == False:
                self.fct = linear_func(self.equal_bsz / self.tp_of_ep_size * 1.5, *args.forward_computation_time) / 1.5
                self.fct *= self.local_expert_num
            else:
                raise NotImplementedError("This function is not yet implemented.")
        elif isinstance(args.forward_computation_time, dict):
            assert self.moe_grouped_gemm == False
            popt = args.forward_computation_time['popt']
            def linear_func(x, m, c):
                return m * x + c
            real_seq_length = self.global_batch_size // self.chunks * self.seq_length * self.top_k // self.num_experts
            self.fct = linear_func(real_seq_length / self.tp_of_ep_size * 2, *popt)
            self.fct *= self.local_expert_num
        else:
            raise NotImplementedError("This function is not yet implemented.")

        # [Step 2] estimate backward computation time
        self.bct = self.fct * args.bct_fct_coe
        if self.checkpoint:
            self.bct += self.fct
            
    def estimate_dispatch_communication_time(self):
        args = self.args

        def linear_func(x, m, c):
            return m * x + c
        litter_time = linear_func(self.little_all_gather_message_size_in_MB, *args.allreduce_fit_dict[self.tp_ep_size]['popt'])

        if self.ep_size != 1:
            fwd_all2all = linear_func(self.all_to_all_message_size_in_MB, *args.all2all_fit_dict[self.ep_size]['popt'])
        else:
            fwd_all2all = 0

        if self.tp_of_ep_size != 1:
            fwd_all_gather = linear_func(self.tp_message_size_in_MB, *args.allreduce_fit_dict[self.tp_of_ep_size]['popt'])
        else:
            fwd_all_gather = 0
        
        if self.tp_of_ep_size != 1:
            bwd_reduce_scatter = linear_func(self.tp_message_size_in_MB, *args.allreduce_fit_dict[self.tp_of_ep_size]['popt'])
        else:
            bwd_reduce_scatter = 0

        if self.ep_size != 1:
            bwd_all_to_all = linear_func(self.all_to_all_message_size_in_MB, *args.all2all_fit_dict[self.ep_size]['popt'])
        else:
            bwd_all_to_all = 0

        self.dispatch_communication_time = litter_time + fwd_all2all + fwd_all_gather + bwd_reduce_scatter + bwd_all_to_all 
    
    def estimate_combine_communication_time(self):
        args = self.args

        def linear_func(x, a, b):
            return a * x + b
        
        if self.tp_of_ep_size != 1:
            fwd_reduce_scatter = linear_func(self.tp_message_size_in_MB, *args.allreduce_fit_dict[self.tp_of_ep_size]['popt'])
        else:
            fwd_reduce_scatter = 0

        if self.ep_size != 1:
            fwd_all2all = linear_func(self.all_to_all_message_size_in_MB, *args.all2all_fit_dict[self.ep_size]['popt'])
        else:
            fwd_all2all = 0

        if self.ep_size != 1:
            bwd_all_to_all = linear_func(self.all_to_all_message_size_in_MB, *args.all2all_fit_dict[self.ep_size]['popt'])  
        else:
            bwd_all_to_all = 0

        if self.tp_of_ep_size != 1:
            bwd_all_gather = linear_func(self.tp_message_size_in_MB, *args.allreduce_fit_dict[self.tp_of_ep_size]['popt'])
        else:
            bwd_all_gather = 0

        self.combine_communication_time = fwd_reduce_scatter + fwd_all2all + bwd_all_to_all + bwd_all_gather
    
    def gen_result(self):
        result = self.fct + self.bct + self.dispatch_communication_time + self.combine_communication_time
        ms_to_s = 1e-3
        result = result * ms_to_s * self.args.costmodel_coe
        return result
    
    
class MoEFFNMemoryCostModelOptimize:
    def __init__(self, 
                 strategy: GalvatronStrategy,
                 global_batch_size: int,
                 chunks: int,
                 stage_idx: int = 0,
                 logger: Logger = None,
                 train_args: TrainArgsOptimize = None,
                 profile_model_args: ProfileModelArgsOptimize = None,
                 version_option_args: VersionOptionArgsOptimize = None,):
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.stage_idx = stage_idx
        self.logger = logger

        self.args = SimpleNamespace()
        components = {
                    'TrainArgsOptimize': train_args, 
                    'ProfileModelArgsOptimize': profile_model_args, 
                    'VersionOptionArgsOptimize': version_option_args,
        }
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in moe_ffn_memory_args_list[class_name]:
                    setattr(self.args, key, value) 
    
        self.initialize()
        self.estimate_model_states_memory()
        self.estimate_activation_memory()
        self.estimate_communication_buffer_memory()

    def initialize(self):
        args = self.args

        # strategy related
        self.pp_size = self.strategy.pp_size
        self.ep_size = self.strategy.ep_size
        self.tp_of_ep_size = self.strategy.tp_of_ep_size
        self.dp_of_ep_size = self.strategy.dp_of_ep_size
        self.use_ulysses = self.strategy.use_ulysses
        self.checkpoint = self.strategy.checkpoint
        self.world_size = self.strategy.world_size
        self.sdp_size = self.dp_of_ep_size * self.tp_of_ep_size if self.use_ulysses else self.dp_of_ep_size
        self.dp_type = self.strategy.dp_type

        # moe related
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        self.local_expert_num = self.num_experts // self.ep_size
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size
        self.scale_seq_length = self.seq_length * self.top_k // self.num_experts
        self.scale_ratio = self.scale_seq_length // self.seq_length

        # batch size related
        self.mbsz = self.global_batch_size // self.chunks // self.dp_of_ep_size
        if self.pp_size == 1:
            self.local_bsz = self.mbsz
        else:
            if args.pipeline_type == 'pipeline':
                assert self.chunks >= self.pp_size
                self.local_bsz = self.mbsz * (self.pp_size - self.stage_idx) // self.pp_size
            else:
                self.local_bsz = self.mbsz * self.chunks

        # zero ratio related
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

        # sequence scale related
        # seq_length -> seqlen * top_k

    def estimate_model_states_memory(self):
        args = self.args

        if self.use_ulysses:
            self.parameter_memory = args.parameter_memory
        else:
            self.parameter_memory = args.parameter_memory / self.tp_of_ep_size
        self.parameter_memory *= self.local_expert_num

        self.model_states_memory = 4 * self.parameter_memory
        if self.dp_type == 'zero3':
            self.model_states_memory *= self.zero3_ratio(self.sdp_size)
        else:
            self.model_states_memory *= self.zero2_ratio(self.sdp_size)

    def estimate_activation_memory(self):
        args = self.args

        if self.checkpoint:
            assert(args.tp_activation_per_bsz_dict['checkpoint'] is not None)
            self.activation_memory = args.tp_activation_per_bsz_dict['checkpoint'] * self.local_bsz
            self.activation_memory *= self.scale_ratio
            self.activation_memory *= self.local_expert_num
            if args.sequence_parallel:
                self.activation_memory /= self.tp_of_ep_size
        else:
            self.activation_memory = args.tp_activation_per_bsz_dict[self.tp_of_ep_size] * self.local_bsz
            self.activation_memory *= self.scale_ratio
            self.activation_memory *= self.local_expert_num

    def estimate_communication_buffer_memory(self):
        self.communication_buffer_memory = 0

    def get_memory_cost(self):
        result = dict()
        result['parameter_memory'] = self.parameter_memory
        result['model_states_memory'] = self.model_states_memory
        result['activation_memory'] = self.activation_memory
        result['communication_buffer_memory'] = self.communication_buffer_memory
        result['total_memory'] = self.model_states_memory + self.activation_memory + self.communication_buffer_memory
        return result


class FFNTimeCostModelOptimize(TimeCostModelBase):
    def estimate_tp_communication_time(self):
        args = self.args
        
        if self.tp_size == 1:
            self.tp_communication_time = 0
        else:
            if args.estimate_tp_time_type == EstimateTPTimeType.FIXED:
                # forward: <all_gather, hidden_states>, <reduce_scatter, hidden_states>
                # backward: <all_gather, data_grad>, <all_gather hidden_states>, <reduce_scatter param_grad>
                # NOTE In the backward pass, <allgather hidden_states> and <reduce_scatter param_grad> can overlap with the computation. 
                if args.sequence_parallel:
                    dtype_size = 2 if args.mixed_precision else 4
                    byte_to_MB = 1024 * 1024
                    per_tp_message_size_in_MB = (self.tp_size - 1) * self.mbsz * (self.seq_length / self.tp_size) * self.hidden_size * dtype_size / byte_to_MB
                    
                    self.fwd_tp_communication_time = per_tp_message_size_in_MB * args.all_gather_fixed_dict[f'{self.tp_size}_1'] * 1 + per_tp_message_size_in_MB * args.reduce_scatter_fixed_dict[f'{self.tp_size}_1'] * 1
                    self.bwd_tp_communication_time = per_tp_message_size_in_MB * args.all_gather_fixed_dict[f'{self.tp_size}_1'] * 1
                    
                    if not self.checkpoint:
                        self.tp_communication_time = self.fwd_tp_communication_time + self.bwd_tp_communication_time
                    else:
                        self.tp_communication_time = self.fwd_tp_communication_time * 2 + self.bwd_tp_communication_time
                else: # use allreduce
                    raise NotImplementedError('allreduce not implement')
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
                    
                    self.fwd_tp_communication_time = per_operation_time * 2
                    self.bwd_tp_communication_time = per_operation_time * 1
                    
                    if not self.checkpoint:
                        self.tp_communication_time = self.fwd_tp_communication_time + self.bwd_tp_communication_time
                    else:
                        self.tp_communication_time = self.fwd_tp_communication_time * 2 + self.bwd_tp_communication_time
                else: # use allreduce
                    raise NotImplementedError('allreduce not implement')

class FFNMemoryCostModelOptimize(MemoryCostModelBase):
    pass