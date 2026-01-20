import numpy as np
from logging import Logger
from types import SimpleNamespace
from typing import Tuple
from ..cost_model_args import TrainArgs, ProfileModelArgs, ProfileHardwareArgs, VersionOptionArgs, EstimateTPTimeType
from galvatron.utils.strategy_utils import EmbeddingLMHeadStrategy, DPType, byte_to_MB, model_states_to_param_size_ratio

class EmbeddingLMHeadTimeCostModel:
    embedding_lmhead_time_args_list = {
        'TrainArgs': ['hidden_size', 'mixed_precision', 'sequence_length_list', 'sequence_parallel'],
        'ProfileModelArgs': ['other_memory_pp_on', 'other_memory_pp_off', 'other_time_profiled'],
        'ProfileHardwareArgs': ['overlap_slowdown_coe', 'bct_fct_coe', 
                    'allreduce_fixed_dict', 'allreduce_fit_dict',
                    'all_gather_fixed_dict', 'all_gather_fit_dict',
                    'reduce_scatter_fixed_dict', 'reduce_scatter_fit_dict'],
        'VersionOptionArgs': ['estimate_tp_time_type'],
    }

    def __init__(
        self,
        strategy:EmbeddingLMHeadStrategy,
        global_batch_size:int = 8,
        chunks:int = 1,
        no_sync_gradient:bool = False,
        logger:Logger = None,
        train_args: TrainArgs = None,
        profile_model_args: ProfileModelArgs = None,
        profile_hardware_args: ProfileHardwareArgs = None,
        version_option_args: VersionOptionArgs = None,
    ):
        # [Step 1] assign attributes
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
            'VersionOptionArgs': version_option_args
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
        
        # [Step 2] calculate some information
        self.mbsz = self.global_batch_size // self.chunks // self.dp_size # NOTE still use dp_size rather than sdp_size
        if self.pp_size > 1:
            self.embedding_parameter_memory_in_MB = args.other_memory_pp_on['first_stage']['model_states'][self.tp_size] / model_states_to_param_size_ratio # actually model_states_to_param_size_ratio is 4.
            self.lmhead_parameter_memory_in_MB = args.other_memory_pp_on['last_stage']['model_states'][self.tp_size] / model_states_to_param_size_ratio # actually model_states_to_param_size_ratio is 4.
        else:
            self.embedding_parameter_memory_in_MB = args.other_memory_pp_off['model_states'][self.tp_size] / model_states_to_param_size_ratio / 2
            self.lmhead_parameter_memory_in_MB = args.other_memory_pp_off['model_states'][self.tp_size] / model_states_to_param_size_ratio / 2

        # [Step 3] copy some attributes for easy access
        self.sequence_length_list = args.sequence_length_list 
        self.hidden_size = args.hidden_size
        assert len(self.sequence_length_list) == 1 or len(self.sequence_length_list) == 2, f'sequence_length_list should be 1 or 2, but got {len(self.sequence_length_list)}'
        self.embedding_seq_length = self.sequence_length_list[0]
        self.lmhead_seq_length = self.sequence_length_list[-1]

        if self.tp_size > 1:
            if args.sequence_parallel:
                self.embedding_seq_length_per_rank = self.embedding_seq_length / self.cp_size / self.tp_size
                self.lmhead_seq_length_per_rank = self.lmhead_seq_length / self.cp_size / self.tp_size
            else:
                self.embedding_seq_length_per_rank = self.embedding_seq_length / self.cp_size
                self.lmhead_seq_length_per_rank = self.lmhead_seq_length / self.cp_size
        else:
            self.embedding_seq_length_per_rank = self.embedding_seq_length / self.cp_size / self.sp_size
            self.lmhead_seq_length_per_rank = self.lmhead_seq_length / self.cp_size / self.sp_size

        # [Step 4] initialize dictionaries for easy access
        self.embedding_dict = {}
        self.lmhead_dict = {}

    def estimate_computation_time(self):
        args = self.args

        # [Step 1] estimate forward computation time
        if isinstance(args.other_time_profiled, np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            fct_time = linear_func(self.mbsz / self.tp_size / self.sp_size / self.cp_size, *args.other_time_profiled)
        else:
            fct_time = args.other_time_profiled * self.mbsz / self.tp_size / self.sp_size / self.cp_size

        # [Step 2] estimate backward computation time
        self.embedding_dict['fct'] = fct_time / 2
        self.lmhead_dict['fct'] = fct_time / 2
        self.embedding_dict['bct'] = fct_time / 2 * args.bct_fct_coe
        self.lmhead_dict['bct'] = fct_time / 2 * args.bct_fct_coe

    def estimate_dp_communication_time(self):
        args = self.args
        
        if self.sdp_size == 1:
            self.dc = 0
            self.embedding_dict['gradient_message_size_in_MB'] = 0
            self.lmhead_dict['gradient_message_size_in_MB'] = 0
            self.fsdp_allgather_coe = 0
            self.embedding_dict['fsdp_allgather_message_size_in_MB'] = 0
            self.lmhead_dict['fsdp_allgather_message_size_in_MB'] = 0
            self.embedding_dict['fsdp_allgather_time_once'] = 0
            self.lmhead_dict['fsdp_allgather_time_once'] = 0
        else:
            # [Step 1] Get dc and dc_slowdown
            if self.tp_size != 1:
                self.dc = args.allreduce_fixed_dict[f'{self.sdp_size}_0']
            else:
                self.dc = args.allreduce_fixed_dict[f'{self.sdp_size}_1']
            
            # [Step 2] calculate gradient message size
            # 2 * (self.sdp_size - 1) indicates that ring-allreduce requires (sdp_size - 1) reduce-scatters and (sdp_size - 2) all-gathers,
            # self.parameter_memory / self.sdp_size represents the amount sent or received each time
            self.embedding_dict['gradient_message_size_in_MB'] = 2 * (self.sdp_size - 1) * (self.embedding_parameter_memory_in_MB / self.sdp_size)
            self.lmhead_dict['gradient_message_size_in_MB'] = 2 * (self.sdp_size - 1) * (self.lmhead_parameter_memory_in_MB / self.sdp_size)

            if args.mixed_precision:
                self.embedding_dict['gradient_message_size_in_MB'] /= 2
                self.lmhead_dict['gradient_message_size_in_MB'] /= 2
            if self.no_sync_gradient:
                self.embedding_dict['gradient_message_size_in_MB'] = 0
                self.lmhead_dict['gradient_message_size_in_MB'] = 0

            # [Step 3] calculate fsdp allgather message size and time if necessary
            if self.dp_type == DPType.ZERO3:
                self.embedding_dict['fsdp_allgather_message_size_in_MB'] = (self.sdp_size - 1) * (self.embedding_parameter_memory_in_MB / self.sdp_size)
                self.lmhead_dict['fsdp_allgather_message_size_in_MB'] = (self.sdp_size - 1) * (self.lmhead_parameter_memory_in_MB / self.sdp_size)
                self.fsdp_allgather_coe = self.dc / 2 # allreduce = allgather + reduce_scatter
                self.embedding_dict['fsdp_allgather_time_once'] = self.embedding_dict['fsdp_allgather_message_size_in_MB'] * self.fsdp_allgather_coe
                self.lmhead_dict['fsdp_allgather_time_once'] = self.lmhead_dict['fsdp_allgather_message_size_in_MB'] * self.fsdp_allgather_coe
            else:
                self.embedding_dict['fsdp_allgather_message_size_in_MB'] = 0
                self.lmhead_dict['fsdp_allgather_message_size_in_MB'] = 0
                self.fsdp_allgather_coe = 0
                self.embedding_dict['fsdp_allgather_time_once'] = 0
                self.lmhead_dict['fsdp_allgather_time_once'] = 0

    def estimate_tp_communication_time(self):
        args = self.args

        if self.tp_size == 1:
            self.embedding_dict['tp_communication_time'] = 0
            self.lmhead_dict['tp_communication_time'] = 0
        else:
            if args.estimate_tp_time_type == EstimateTPTimeType.FIXED:
                print(f'\nWarning: Fixed version of TP communication time is deprecated. Please use the FIT version instead.')
                if args.sequence_parallel:
                    # forward: <embedding, reduce_scatter hidden_states> <lmhead, all_gather hidden_states>
                    # backward: <lmhead, all_gather hidden_states> <embedding, all_gather hidden_states>
                    dtype_size = 2 if args.mixed_precision else 4
                    self.embedding_dict['per_tp_message_size_in_MB'] = self.mbsz * (self.embedding_seq_length_per_rank * (self.tp_size - 1)) * self.hidden_size * dtype_size / byte_to_MB
                    self.lmhead_dict['per_tp_message_size_in_MB'] = self.mbsz * (self.lmhead_seq_length_per_rank * (self.tp_size - 1)) * self.hidden_size * dtype_size / byte_to_MB

                    self.embedding_dict['tp_communication_time'] = (
                        self.embedding_dict['per_tp_message_size_in_MB'] *
                        args.reduce_scatter_fixed_dict[f'{self.tp_size}_1'] + 
                        self.embedding_dict['per_tp_message_size_in_MB'] *
                        args.all_gather_fixed_dict[f'{self.tp_size}_1']
                    )
                    self.lmhead_dict['tp_communication_time'] = (
                        self.lmhead_dict['per_tp_message_size_in_MB'] *
                        args.all_gather_fixed_dict[f'{self.tp_size}_1'] +
                        self.lmhead_dict['per_tp_message_size_in_MB'] *
                        args.all_gather_fixed_dict[f'{self.tp_size}_1']
                    )
                else:
                    # Roughly estimate using 2 allreduce operations
                    dtype_size = 2 if args.mixed_precision else 4
                    self.embedding_dict['per_tp_message_size_in_MB'] = self.mbsz * (self.embedding_seq_length_per_rank * (self.tp_size - 1)) * self.hidden_size * dtype_size / byte_to_MB
                    self.lmhead_dict['per_tp_message_size_in_MB'] = self.mbsz * (self.lmhead_seq_length_per_rank * (self.tp_size - 1)) * self.hidden_size * dtype_size / byte_to_MB

                    self.embedding_dict['tp_communication_time'] = (
                        self.embedding_dict['per_tp_message_size_in_MB'] *
                        args.allreduce_fixed_dict[f'{self.tp_size}_1']
                    )
                    self.lmhead_dict['tp_communication_time'] = (
                        self.lmhead_dict['per_tp_message_size_in_MB'] *
                        args.allreduce_fixed_dict[f'{self.tp_size}_1']
                    )

            elif args.estimate_tp_time_type == EstimateTPTimeType.FIT:
                if args.sequence_parallel:
                    # forward: <embedding, reduce_scatter hidden_states> <lmhead, all_gather hidden_states>
                    # backward: <lmhead, all_gather hidden_states> <embedding, all_gather hidden_states>
                    dtype_size = 2 if args.mixed_precision else 4
                    self.embedding_dict['per_tp_message_size_in_byte'] = self.mbsz * (self.embedding_seq_length_per_rank * self.tp_size) * self.hidden_size * dtype_size
                    self.lmhead_dict['per_tp_message_size_in_byte'] = self.mbsz * (self.lmhead_seq_length_per_rank * self.tp_size) * self.hidden_size * dtype_size

                    tp_dict = args.all_gather_fit_dict[self.tp_size]
                    if self.embedding_dict['per_tp_message_size_in_byte'] in tp_dict:
                        self.embedding_dict['per_tp_operation_time'] = tp_dict[self.embedding_dict['per_tp_message_size_in_byte']]
                    else:
                        def linear_func(x, m, c):
                            return m * x + c
                        self.embedding_dict['per_tp_operation_time'] = linear_func(self.embedding_dict['per_tp_message_size_in_byte'] / byte_to_MB, *tp_dict['popt'])
                    if self.lmhead_dict['per_tp_message_size_in_byte'] in tp_dict:
                        self.lmhead_dict['per_tp_operation_time'] = tp_dict[self.lmhead_dict['per_tp_message_size_in_byte']]
                    else:
                        def linear_func(x, m, c):
                            return m * x + c
                        self.lmhead_dict['per_tp_operation_time'] = linear_func(self.lmhead_dict['per_tp_message_size_in_byte'] / byte_to_MB, *tp_dict['popt'])

                    self.embedding_dict['tp_communication_time'] = self.embedding_dict['per_tp_operation_time'] * 2 # all_gather and reduce_scatter
                    self.lmhead_dict['tp_communication_time'] = self.lmhead_dict['per_tp_operation_time'] * 2 # all_gather and all_gather
                else:
                    # Roughly estimate using 2 allreduce operations
                    dtype_size = 2 if args.mixed_precision else 4
                    self.embedding_dict['per_tp_message_size_in_byte'] = self.mbsz * (self.embedding_seq_length_per_rank * self.tp_size) * self.hidden_size * dtype_size
                    self.lmhead_dict['per_tp_message_size_in_byte'] = self.mbsz * (self.lmhead_seq_length_per_rank * self.tp_size) * self.hidden_size * dtype_size  

                    tp_dict = args.allreduce_fit_dict[self.tp_size]
                    if self.embedding_dict['per_tp_message_size_in_byte'] in tp_dict:
                        self.embedding_dict['per_tp_operation_time'] = tp_dict[self.embedding_dict['per_tp_message_size_in_byte']]
                    else:
                        def linear_func(x, m, c):
                            return m * x + c
                        self.embedding_dict['per_tp_operation_time'] = linear_func(self.embedding_dict['per_tp_message_size_in_byte'] / byte_to_MB, *tp_dict['popt'])
                    if self.lmhead_dict['per_tp_message_size_in_byte'] in tp_dict:
                        self.lmhead_dict['per_tp_operation_time'] = tp_dict[self.lmhead_dict['per_tp_message_size_in_byte']]
                    else:
                        def linear_func(x, m, c):
                            return m * x + c
                        self.lmhead_dict['per_tp_operation_time'] = linear_func(self.lmhead_dict['per_tp_message_size_in_byte'] / byte_to_MB, *tp_dict['popt'])
                    
                    self.embedding_dict['tp_communication_time'] = self.embedding_dict['per_tp_operation_time'] # allreduce once
                    self.lmhead_dict['tp_communication_time'] = self.lmhead_dict['per_tp_operation_time'] # allreduce once

    def get_overlap_time(self, forward_comm_time, forward_comp_time, backward_comm_time, backward_comp_time, tp_time):
        forward_comp_time = forward_comp_time * self.args.overlap_slowdown_coe
        backward_comp_time = backward_comp_time * self.args.overlap_slowdown_coe
        if forward_comp_time > forward_comm_time:
            forward_time = forward_comm_time + (forward_comp_time - forward_comm_time) / self.args.overlap_slowdown_coe
        else:
            forward_time = forward_comm_time
        if backward_comp_time > backward_comm_time:
            backward_time = backward_comm_time + (backward_comp_time - backward_comm_time) / self.args.overlap_slowdown_coe
        else:
            backward_time = backward_comm_time
        return forward_time + backward_time + tp_time

    def gen_result(self) -> Tuple[float, float]:
        ms_to_s = 0.001
    
        embedding_time_cost = ms_to_s * self.get_overlap_time(
            forward_comm_time=self.embedding_dict['fsdp_allgather_time_once'],
            forward_comp_time=self.embedding_dict['fct'],
            backward_comm_time=self.embedding_dict['gradient_message_size_in_MB'] * self.dc + self.embedding_dict['fsdp_allgather_time_once'],
            backward_comp_time=self.embedding_dict['bct'],
            tp_time=self.embedding_dict['tp_communication_time']
        )
        lmhead_time_cost = ms_to_s * self.get_overlap_time(
            forward_comm_time=self.lmhead_dict['fsdp_allgather_time_once'],
            forward_comp_time=self.lmhead_dict['fct'],
            backward_comm_time=self.lmhead_dict['gradient_message_size_in_MB'] * self.dc + self.lmhead_dict['fsdp_allgather_time_once'],
            backward_comp_time=self.lmhead_dict['bct'],
            tp_time=self.lmhead_dict['tp_communication_time']
        )

        return embedding_time_cost, lmhead_time_cost

class EmbeddingLMHeadMemoryCostModel:
    embedding_lmhead_memory_args_list = {
        'TrainArgs': ['mixed_precision', 'async_grad_reduce', 'pipeline_type', 'sequence_parallel'],
        'ProfileModelArgs': ['other_memory_pp_off', 'other_memory_pp_on' ],
        # 'UtilsArgs': ['pytorch_context_mem' ],
        'VersionOptionArgs': ['zero_with_slight_noise' ],
    }

    def __init__(
        self,
        strategy:EmbeddingLMHeadStrategy,
        global_batch_size:int,
        chunks:int,
        logger:Logger=None,
        train_args:TrainArgs=None,
        profile_model_args:ProfileModelArgs=None,
        version_option_args:VersionOptionArgs=None
    ):
        
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.logger = logger
        
        self.args = SimpleNamespace()
        components = {
            'TrainArgs': train_args, 
            'ProfileModelArgs': profile_model_args, 
            'VersionOptionArgs': version_option_args,
        }
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in self.embedding_lmhead_memory_args_list[class_name]:
                    setattr(self.args, key, value) 

        self.initialize()
        self.estimate_model_states_memory()
        self.estimate_activation_memory()

    def initialize(self):
        args = self.args

        # [Step 1] initialize strategy related attributes
        self.pp_size = self.strategy.pp_size
        self.tp_size = self.strategy.tp_size
        self.cp_size = self.strategy.cp_size
        self.sp_size = self.strategy.sp_size
        self.dp_size = self.strategy.dp_size
        self.dp_type = self.strategy.dp_type
        self.sdp_size = self.strategy.sdp_size

        # [Step 2] calculate some informations
        self.mbsz = self.global_batch_size // self.chunks // self.dp_size
        if self.pp_size == 1:
            self.embedding_local_bsz = self.mbsz
            self.lmhead_local_bsz = self.mbsz
        else:
            if args.pipeline_type == 'pipedream_flush':
                assert self.chunks >= self.pp_size
                self.embedding_local_bsz = self.mbsz * self.pp_size
                self.lmhead_local_bsz = self.mbsz * 1
            elif args.pipeline_type == 'gpipe':
                self.embedding_local_bsz = self.mbsz * self.pp_size
                self.lmhead_local_bsz = self.mbsz * self.pp_size

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
                    # *5/4: for fp32 grad 
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

        # [Step 4] initialize dictionaries for easy access
        self.embedding_dict = {}
        self.lmhead_dict = {}

    def estimate_model_states_memory(self):
        args = self.args

        if self.pp_size == 1:
            model_states_memory = args.other_memory_pp_off['model_states'][self.tp_size]
            if self.dp_type == DPType.ZERO3:
                model_states_memory *= self.zero3_ratio(self.sdp_size)
            elif self.dp_type == DPType.ZERO2:
                model_states_memory *= self.zero2_ratio(self.sdp_size)
            self.embedding_dict['model_states_memory'] = model_states_memory / 2
            self.lmhead_dict['model_states_memory'] = model_states_memory / 2
        else:
            self.embedding_dict['model_states_memory'] = args.other_memory_pp_on['first_stage']['model_states'][self.tp_size]
            self.lmhead_dict['model_states_memory'] = args.other_memory_pp_on['last_stage']['model_states'][self.tp_size]
            if self.dp_type == DPType.ZERO3:
                self.embedding_dict['model_states_memory'] *= self.zero3_ratio(self.sdp_size)
                self.lmhead_dict['model_states_memory'] *= self.zero3_ratio(self.sdp_size)
            elif self.dp_type == DPType.ZERO2:
                self.embedding_dict['model_states_memory'] *= self.zero2_ratio(self.sdp_size)
                self.lmhead_dict['model_states_memory'] *= self.zero2_ratio(self.sdp_size)

    def estimate_activation_memory(self):
        args = self.args
        
        if self.tp_size > 1:
            if args.sequence_parallel:
                activation_idx = self.cp_size * self.tp_size
            else:
                activation_idx = self.cp_size
        else:
            activation_idx = self.cp_size * self.sp_size

        if self.pp_size == 1:
            self.embedding_dict['activation_memory'] = self.embedding_local_bsz * args.other_memory_pp_off['activation'][activation_idx]
            self.lmhead_dict['activation_memory'] = self.lmhead_local_bsz * args.other_memory_pp_off['activation'][activation_idx]
        else:
            self.embedding_dict['activation_memory'] = self.embedding_local_bsz * args.other_memory_pp_on['first_stage']['activation'][activation_idx]
            self.lmhead_dict['activation_memory'] = self.lmhead_local_bsz * args.other_memory_pp_on['last_stage']['activation'][activation_idx]

    def get_memory_cost(self)->Tuple[dict, dict]:
        self.embedding_dict['total_memory'] = self.embedding_dict['model_states_memory'] + self.embedding_dict['activation_memory']
        self.lmhead_dict['total_memory'] = self.lmhead_dict['model_states_memory'] + self.lmhead_dict['activation_memory']
        return self.embedding_dict, self.lmhead_dict