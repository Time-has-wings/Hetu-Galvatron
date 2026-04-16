import numpy as np
from logging import Logger
from types import SimpleNamespace
from typing import Tuple, List

from galvatron.utils.strategy_utils import EmbeddingLMHeadStrategy, DPType
from galvatron.core.cost_model.cost_model_args import ModelArgs, TrainArgs, ParallelArgs, ProfileModelArgs, ProfileHardwareArgs

class EmbeddingLMHeadTimeCostModel:
    embedding_lmhead_time_args_list = {
        'ModelArgs': ['hidden_size'],
        'TrainArgs': ['mixed_precision'],
        'ParallelArgs': ['sequence_parallel'],
        'ProfileModelArgs': ['other_memory_pp_on', 'other_memory_pp_off', 'other_time_profiled'],
        'ProfileHardwareArgs':['comm_coe_dict', 'allreduce_dict', 'dp_overlap_coe', 'bct_overlap_coe', 'bct_fct_coe', 'allreduce_latency_per_MB_dict', 'allreduce_message_size_to_latency_dict_dict', 'allgather_message_size_to_latency_dict_dict', 'all2all_message_size_to_latency_dict_dict']
    }

    def __init__(
        self,
        strategy:EmbeddingLMHeadStrategy,
        global_batch_size:int = 8,
        chunks:int = 1,
        logger:Logger = None,
        sequence_length_list:List[int] = [512],
        model_args:ModelArgs = None, 
        train_args:TrainArgs = None, 
        parallel_args:ParallelArgs = None, 
        profile_model_args:ProfileModelArgs = None, 
        profile_hardware_args:ProfileHardwareArgs = None,
    ):
        # [Step 1] assign attributes
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.logger = logger
        self.sequence_length_list = sequence_length_list
        
        # [Step 2] gather all args into a single namespace
        self.args: SimpleNamespace = SimpleNamespace()
        components = {
            'ModelArgs': model_args,
            'TrainArgs': train_args,
            'ParallelArgs': parallel_args,
            'ProfileModelArgs': profile_model_args,
            'ProfileHardwareArgs': profile_hardware_args,
        }
        for class_name, instance in components.items():
            assert instance is not None, f'{class_name} is None'
            for key, value in instance.__dict__.items():
                if key in self.embedding_lmhead_time_args_list[class_name]:
                    setattr(self.args, key, value)
                    
        # [Step 3] initialize and estimate time  
        self.initialize()
        self.estimate_computation_time()
        self.estimate_dp_communication_time()
        self.estimate_tp_communication_time()    

    def initialize(self):
        args = self.args

        # [Step 1] initialize strategy related attributes
        strategy:EmbeddingLMHeadStrategy = self.strategy
        self.pp_size = strategy.pp_size
        self.tp_size = strategy.tp_size
        self.sp_size = strategy.sp_size
        self.cp_size = strategy.cp_size
        self.dp_size = strategy.dp_size
        self.dp_type = strategy.dp_type
        self.sdp_size = strategy.sdp_size
        self.tp_sp_size = strategy.tp_sp_size
        
        # [Step 2] calculate some information
        self.lbsz = self.global_batch_size // self.chunks // self.dp_size # NOTE still use dp_size rather than sdp_size

        # [Step 3] get hardware related attributes
        self.allreduce_latency_per_MB_dict = args.allreduce_latency_per_MB_dict
        self.allgather_message_size_to_latency_dict = args.allgather_message_size_to_latency_dict_dict[self.tp_size] if self.tp_size != 1 else None
        self.all2all_message_size_to_latency_dict = args.all2all_message_size_to_latency_dict_dict[self.sp_size] if self.sp_size != 1 else None

    def estimate_computation_time(self):
        args = self.args

        self.fct = [0] * self.pp_size

        if isinstance(args.other_time_profiled, np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            fct_time = linear_func(self.lbsz / self.tp_sp_size / self.cp_size, *args.other_time_profiled)
        else:
            fct_time = args.other_time_profiled * self.lbsz / self.tp_sp_size / self.cp_size

        if self.pp_size == 1:
            self.fct[0] = fct_time
        else:
            self.fct[0] = fct_time / 2
            self.fct[-1] = fct_time / 2

    def estimate_dp_communication_time(self):
        args = self.args
        
        self.dp_message_size = [0] * self.pp_size

        key = f'{self.sdp_size}_0' if self.tp_size != 1 else f'{self.sdp_size}_1'
        self.dp_coe = self.allreduce_latency_per_MB_dict[key] * (self.sdp_size - 1) / self.sdp_size 

        if args.mixed_precision:
            factor = 0.5
        else:
            factor = 1.0
        
        if self.pp_size == 1:
            self.dp_message_size[0] = args.other_memory_pp_off['model_states'][self.tp_size] / 4 * factor
        else:
            self.dp_message_size[0] = args.other_memory_pp_on['first_stage']['model_states'][self.tp_size] / 4 * factor
            self.dp_message_size[-1] = args.other_memory_pp_on['last_stage']['model_states'][self.tp_size] / 4 * factor

        if self.dp_type == DPType.ZERO3: # TODO: check correctness
            self.fwd_factor = 0.5
            self.bwd_factor = 1.0
        else:
            self.fwd_factor = 0.0
            self.bwd_factor = 0.5

    def estimate_tp_communication_time(self):
        args = self.args

        self.tp_sp_time = [0] * self.pp_size
        tp_sp_time_per_seq_len = []

        for seq_len in self.sequence_length_list:
            if self.tp_sp_size == 1:
                tp_sp_time_per_seq_len.append(0)
            else:
                if self.tp_size == 1:
                    tp_sp_time_per_seq_len.append(0)
                else: # self.sp == 1 and self.tp_size > 1
                    message_size_in_MB = self.lbsz * seq_len * args.hidden_size * (2 if args.mixed_precision else 4) / 1024 / 1024
                    assert args.sequence_parallel, f'sequence_parallel must be True when tp_size > 1'
                    if message_size_in_MB in self.allgather_message_size_to_latency_dict:
                        message_time = self.allgather_message_size_to_latency_dict[message_size_in_MB]
                    else:
                        def linear_func(x, m, c):
                            return m * x + c
                        message_time = linear_func(message_size_in_MB, *self.allgather_message_size_to_latency_dict["popt"])
                    tp_sp_time_per_seq_len.append(message_time)
            
        if self.pp_size == 1:
            self.tp_sp_time[0] = tp_sp_time_per_seq_len[0] + tp_sp_time_per_seq_len[-1]
        else:
            self.tp_sp_time[0] = tp_sp_time_per_seq_len[0]
            self.tp_sp_time[-1] = tp_sp_time_per_seq_len[-1]

    # In new vesion, we assume that comm overlap_coe(bct_overlap_coe)=1, so we only need to calculate comp overlap time
    def get_overlap_time(self, forward_comm_time, forward_comp_time, backward_comm_time, backward_comp_time, tp_sp_time):
        forward_comp_time = forward_comp_time * self.args.dp_overlap_coe
        backward_comp_time = backward_comp_time * self.args.dp_overlap_coe
        if forward_comp_time > forward_comm_time:
            forward_time = forward_comm_time + (forward_comp_time - forward_comm_time) / self.args.dp_overlap_coe
        else:
            forward_time = forward_comm_time
        if backward_comp_time > backward_comm_time:
            backward_time = backward_comm_time + (backward_comp_time - backward_comm_time) / self.args.dp_overlap_coe
        else:
            backward_time = backward_comm_time
        return forward_time + backward_time + tp_sp_time

    def gen_result(self) -> Tuple[List[float], List[float]]:
        ms_to_s = 0.001

        other_time_cost = [0] * self.pp_size
        other_time_cost_no_grad_sync = [0] * self.pp_size

        if self.pp_size == 1:
            other_time_cost[0] = ms_to_s * self.get_overlap_time(self.dp_message_size[0] * self.dp_coe * self.fwd_factor, self.fct[0], self.dp_message_size[0] * self.dp_coe * self.bwd_factor, self.fct[0] * self.args.bct_fct_coe, self.tp_sp_time[0])
            other_time_cost_no_grad_sync[0] = ms_to_s * self.get_overlap_time(self.dp_message_size[0] * self.dp_coe * self.fwd_factor, self.fct[0], self.dp_message_size[0] * self.dp_coe * (self.bwd_factor - 0.5), self.fct[0] * self.args.bct_fct_coe, self.tp_sp_time[0])
        else:
            dp_coe = self.dp_coe
            other_time_cost[0] = ms_to_s * self.get_overlap_time(self.dp_message_size[0] * dp_coe * self.fwd_factor, self.fct[0], self.dp_message_size[0] * dp_coe * self.bwd_factor, self.fct[0] * self.args.bct_fct_coe, self.tp_sp_time[0])
            other_time_cost[-1] = ms_to_s * self.get_overlap_time(self.dp_message_size[-1] * dp_coe * self.fwd_factor, self.fct[-1], self.dp_message_size[-1] * dp_coe * self.bwd_factor, self.fct[-1] * self.args.bct_fct_coe, self.tp_sp_time[-1])
            other_time_cost_no_grad_sync[0] = ms_to_s * self.get_overlap_time(self.dp_message_size[0] * dp_coe * self.fwd_factor, self.fct[0], self.dp_message_size[0] * dp_coe * (self.bwd_factor - 0.5), self.fct[0] * self.args.bct_fct_coe, self.tp_sp_time[0])
            other_time_cost_no_grad_sync[-1] = ms_to_s * self.get_overlap_time(self.dp_message_size[-1] * dp_coe * self.fwd_factor, self.fct[-1], self.dp_message_size[-1] * dp_coe * (self.bwd_factor - 0.5), self.fct[-1] * self.args.bct_fct_coe, self.tp_sp_time[-1])

        return other_time_cost, other_time_cost_no_grad_sync


class EmbeddingLMHeadMemoryCostModel:
    memory_args_list = {
        'ModelArgs':['parameter_size'], 
        'TrainArgs':['mixed_precision', 'async_grad_reduce', 'pytorch_context_mem'], 
        'ParallelArgs':['use_zero2_for_dp', 'max_tp_deg', 'sequence_parallel', 'pipeline_type', 'optimal_chunk_func', 'chunks'], 
        'ProfileModelArgs':['tp_activation_per_bsz_dict', 'other_memory_pp_off', 'other_memory_pp_on']
    }
    
    def __init__(
        self, 
        strategy:EmbeddingLMHeadStrategy, 
        global_batch_size:int = 8, 
        chunks:int = 1,
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

        # [initialize]: initialize local batch size
        self.lbsz = self.global_batch_size // self.chunks // self.dp_size

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
        
    def estimate_model_states_size(self):
        args = self.args
        
        self.model_states_size = [0] * self.pp_size

        if self.dp_type == DPType.ZERO3:
            self.zero_scale_factor = self.zero3_ratio(self.sdp_size)
        elif self.dp_type == DPType.ZERO2:
            self.zero_scale_factor = self.zero2_ratio(self.sdp_size)
        else:
            self.zero_scale_factor = 1.0

        if self.pp_size == 1:
            self.model_states_size[0] = args.other_memory_pp_off['model_states'][self.tp_size] * self.zero_scale_factor
        else:
            self.model_states_size[0] = args.other_memory_pp_on['first_stage']['model_states'][self.tp_size] * self.zero_scale_factor
            self.model_states_size[-1]= args.other_memory_pp_on['last_stage']['model_states'][self.tp_size] * self.zero_scale_factor
            

    def estimate_activation_size(self):
        args = self.args
        self.activation_size = [0] * self.pp_size
        self.cumulative_num = [0] * self.pp_size
        self.cumulative_lbsz = [0] * self.pp_size

        if self.pp_size == 1:
            self.cumulative_num[0] = 1
            self.cumulative_lbsz[0] = self.cumulative_num[0] * self.lbsz
            self.activation_size[0] = args.other_memory_pp_off['activation'][self.tp_sp_size] * self.cumulative_lbsz[0]
        else:
            if args.pipeline_type == 'pipedream_flush':
                assert self.chunks >= self.pp_size, f'chunks {self.chunks} must be greater than or equal to pp_size {self.pp_size}'
                self.cumulative_num[0], self.cumulative_num[-1] = self.pp_size, 1
                self.cumulative_lbsz[0], self.cumulative_lbsz[-1] = self.cumulative_num[0] * self.lbsz, self.cumulative_num[-1] * self.lbsz
            elif args.pipeline_type == 'gpipe':
                assert self.chunks >= self.pp_size, f'chunks {self.chunks} must be greater than or equal to pp_size {self.pp_size}'
                self.cumulative_num[0], self.cumulative_num[-1] = self.chunks, self.chunks
                self.cumulative_lbsz[0], self.cumulative_lbsz[-1] = self.cumulative_num[0] * self.lbsz, self.cumulative_num[-1] * self.lbsz
            self.activation_size[0] = args.other_memory_pp_on['first_stage']['activation'][self.tp_sp_size] * self.cumulative_lbsz[0]
            self.activation_size[-1] = args.other_memory_pp_on['last_stage']['activation'][self.tp_sp_size] * self.cumulative_lbsz[-1]
    
    def get_memory_cost(self):
        args = self.args
        
        self.pytorch_context_mem = [args.pytorch_context_mem] * self.pp_size # TODO: add more correct estimation

        result = dict()
        result['model_states'] = self.model_states_size
        result['activation'] = self.activation_size
        result['pytorch_context_mem'] = self.pytorch_context_mem
        result['enc_total'] = [sum(x) for x in zip(self.model_states_size, self.activation_size, self.pytorch_context_mem)]
        
        return result