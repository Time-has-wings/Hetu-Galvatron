import numpy as np
from logging import Logger
from types import SimpleNamespace
from ..cost_model_args_optimize import TrainArgsOptimize, ProfileModelArgsOptimize, ProfileHardwareArgsOptimize, VersionOptionArgsOptimize
from ..cost_model_args_optimize import EstimateTPTimeType
from galvatron.utils.strategy_utils import GalvatronStrategy

embedding_lmhead_time_args_list = {
        'TrainArgsOptimize': ['hidden_size', 'mixed_precision', 'sequence_length_list', 'sequence_parallel'],
        'ProfileModelArgsOptimize': ['other_memory_pp_on', 'other_memory_pp_off', 'other_time_profiled'],
        'ProfileHardwareArgsOptimize': ['overlap_slowdown_coe', 'bct_fct_coe', 
                                        'allreduce_fixed_dict', 'allreduce_fit_dict',
                                        'all_gather_fixed_dict', 'all_gather_fit_dict'
                                        'reduce_scatter_fixed_dict', 'reduce_scatter_fit_dict'],
        'VersionOptionArgsOptimize': ['estimate_tp_time_type'],
    }

class EmbeddingLMHeadTimeCostModelOptimize:
    def __init__(self,
                 strategy:GalvatronStrategy,
                 global_batch_size: int = 32,
                 chunks:int = 1,
                 logger: Logger = None,
                 train_args: TrainArgsOptimize = None,
                 profile_model_args: ProfileModelArgsOptimize = None,
                 profile_hardware_args: ProfileHardwareArgsOptimize = None,
                 version_option_args: VersionOptionArgsOptimize = None,
                ):
        assert all(x is not None for x in [train_args, profile_model_args, profile_hardware_args, version_option_args]), "All argument instances must be provided."
        
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.logger = logger
        
        self.args = SimpleNamespace()
        components = {'TrainArgsOptimize': train_args, 'ProfileModelArgsOptimize': profile_model_args, 'ProfileHardwareArgsOptimize': profile_hardware_args, 'VersionOptionArgsOptimize': version_option_args}
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in embedding_lmhead_time_args_list[class_name]:
                    setattr(self.args, key, value)

        self.args.all_gather_fixed_dict = self.args.allreduce_fixed_dict 
        self.args.reduce_scatter_fixed_dict = self.args.allreduce_fixed_dict       
        self.args.all_gather_fit_dict = self.args.allreduce_fit_dict 
        self.args.reduce_scatter_fit_dict = self.args.allreduce_fit_dict     
        
        self.initialize()
        self.estimate_computation_time()
        self.estimate_dp_communication_time()
        self.estimate_tp_communcation_cost()    
        
    def initialize(self):
        args = self.args
        
        self.pp_size = self.strategy.pp_size
        self.tp_size = self.strategy.tp_size
        self.dp_size = self.strategy.dp_size
        self.use_ulysses = self.strategy.use_ulysses
        self.dp_type = self.strategy.dp_type
        self.sdp_size = self.tp_size * self.dp_size if self.use_ulysses else self.dp_size
        
        self.mbsz = self.global_batch_size // self.chunks // self.dp_size # NOTE still use dp_size rather than sdp_size
        
        self.sequence_length_list = args.sequence_length_list
        self.hidden_size = args.hidden_size
    
    def estimate_computation_time(self):
        args = self.args
        
        if isinstance(args.other_time_profiled, np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            fct_time = linear_func(self.mbsz / self.tp_size, *args.other_time_profiled)
        else:
            fct_time = args.other_time_profiled * self.mbsz / self.tp_size
        
        self.fct = {'embedding': fct_time / 2, 'lmhead': fct_time / 2}
        self.bct = {'embedding': fct_time / 2 * args.bct_fct_coe, 'lmhead': fct_time / 2 * args.bct_fct_coe}
        
    def estimate_dp_communication_time(self):
        args = self.args
        
        if self.sdp_size == 1:
            self.dc = 0
            self.gradient_message_size = {'embedding': 0, 'lmhead': 0}
        else:
            # [Step 1] Get dc
            if self.use_ulysses:
                self.dc = args.allreduce_fixed_dict[f'{self.sdp_size}_1']
            else:
                if self.tp_size == 1:
                    self.dc = args.allreduce_fixed_dict[f'{self.sdp_size}_1']
                else:
                    self.dc = args.allreduce_fixed_dict[f'{self.sdp_size}_0']
            
            # [Step 2] Get gradient message size
            self.gradient_message_size = {}
            if self.sdp_size == 1:
                self.gradient_message_size = {'embedding': 0, 'lmhead': 0}
            else:
                message_size_divide_param_size = 2 * (self.sdp_size - 1) / self.sdp_size
                if self.pp_size == 1:
                    if self.use_ulysses == False:
                        message_size = args.other_memory_pp_off['model_states'][self.tp_size] / 4 * message_size_divide_param_size
                    else:
                        message_size = args.other_memory_pp_off['model_states'][1] / 4 * message_size_divide_param_size
                    self.gradient_message_size = {'embedding': message_size / 2, 'lmhead': message_size / 2}        
                else:
                    if self.use_ulysses == False:
                        self.gradient_message_size['embedding'] = args.other_memory_pp_on['first_stage']['model_states'][self.tp_size] / 4 * message_size_divide_param_size
                        self.gradient_message_size['lmhead'] = args.other_memory_pp_on['last_stage']['model_states'][self.tp_size] / 4 * message_size_divide_param_size
                    else:
                        self.gradient_message_size['embedding'] = args.other_memory_pp_on['first_stage']['model_states'][1] / 4 * message_size_divide_param_size
                        self.gradient_message_size['lmhead'] = args.other_memory_pp_on['last_stage']['model_states'][1] / 4 * message_size_divide_param_size
                
        # TODO Verify whether it is correct
        if self.dp_type == 'zero3':
            self.fwd_factor = 0.5 # fsdp allgather param(0.5)
            self.bwd_factor = 1.0 # fsdp allgather param(0.5) + dp reduce gradient(0.5)
        else:
            self.fwd_factor = 0.0 # nothing
            self.bwd_factor = 0.5 # dp reduce gradient(0.5)
            
    def estimate_tp_communcation_cost(self):
        args = self.args
        
        if self.tp_size == 1:
            self.tp_communication_time = {'embedding':0, 'lmhead': 0}
        else:
            if args.sequence_parallel:
                # forward: <embedding, reduce_scatter hidden_states> <lmhead, all_gather hidden_states>
                # backward: <lmhead, all_gather hidden_states> <embedding, all_gather hidden_states>
                if args.estimate_tp_time_type == EstimateTPTimeType.FIXED:
                    dtype_size = 2 if args.mixed_precision else 4
                    byte_to_MB = 1024 * 1024

                    all_gather_coe = args.all_gather_fixed_dict[f'{self.tp_size}_1']
                    reduce_scatter_coe = args.reduce_scatter_fixed_dict[f'{self.tp_size}_1']
                    
                    embedding_message_size_in_MB = (self.tp_size - 1) * (self.mbsz * self.sequence_length_list[0] / self.tp_size * self.hidden_size * dtype_size) / byte_to_MB
                    lmhead_message_size_in_MB = (self.tp_size - 1) * (self.mbsz * self.sequence_length_list[-1] / self.tp_size * self.hidden_size * dtype_size) / byte_to_MB

                    self.tp_communication_time = {
                        'embedding': embedding_message_size_in_MB * reduce_scatter_coe + embedding_message_size_in_MB * all_gather_coe,
                        'lmhead': lmhead_message_size_in_MB * all_gather_coe + lmhead_message_size_in_MB * all_gather_coe
                    }
                elif args.estimate_tp_time_type == EstimateTPTimeType.FIT:
                    dtype_size = 2 if args.mixed_precision else 4
                    byte_to_MB = 1024 * 1024

                    all_gather_dict = args.all_gather_fit_dict[self.tp_size]
                    reduce_scatter_dict = args.reduce_scatter_fit_dict[self.tp_size]
                    
                    embedding_message_size_in_byte = (self.tp_size - 1) * self.mbsz * (self.sequence_length_list[0] / self.tp_size) * self.hidden_size * dtype_size
                    lmhead_messgae_size_in_byte = (self.tp_size - 1) * self.mbsz * (self.sequence_length_list[-1] / self.tp_size) * self.hidden_size * dtype_size

                    self.tp_communication_time = {}
                    if embedding_message_size_in_byte in all_gather_dict:
                        self.tp_communication_time['embedding'] = reduce_scatter_dict[embedding_message_size_in_byte] + all_gather_dict[embedding_message_size_in_byte]
                    else:
                        def linear_func(x, m, c):
                            return m * x + c
                        self.tp_communication_time['embedding'] = linear_func(embedding_message_size_in_byte / byte_to_MB, * reduce_scatter_dict["popt"]) + \
                                                                linear_func(embedding_message_size_in_byte / byte_to_MB, *all_gather_dict["popt"])
                    if lmhead_messgae_size_in_byte in all_gather_dict:
                        self.tp_communication_time['lmhead'] = reduce_scatter_dict[lmhead_messgae_size_in_byte] + all_gather_dict[lmhead_messgae_size_in_byte]
                    else:
                        def linear_func(x, m, c):
                            return m * x + c
                        self.tp_communication_time['lmhead'] = linear_func(lmhead_messgae_size_in_byte / byte_to_MB, * reduce_scatter_dict["popt"]) + \
                                                                linear_func(lmhead_messgae_size_in_byte / byte_to_MB, *all_gather_dict["popt"])
            else:
                raise NotImplemented("not implement")
            
    # In new vesion, we assume that comm overlap_coe(bct_overlap_coe)=1, so we only need to calculate comp overlap time
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
    
    def gen_result(self):
        time_cost = {'embedding': 0, 'lmhead': 0}
        time_no_sync_gradient_cost = {'embedding': 0, 'lmhead': 0}
        ms_to_s = 0.001

        for key in ['embedding', 'lmhead']:
            time_cost[key] = ms_to_s * self.get_overlap_time(
                forward_comm_time=self.gradient_message_size[key] * self.dc * self.fwd_factor,
                forward_comp_time=self.fct[key],
                backward_comm_time=self.gradient_message_size[key] * self.dc * self.bwd_factor,
                backward_comp_time=self.bct[key],
                tp_time=self.tp_communication_time[key]
            )
            time_no_sync_gradient_cost[key] = ms_to_s * self.get_overlap_time(
                forward_comm_time=self.gradient_message_size[key] * self.dc * self.fwd_factor,
                forward_comp_time=self.fct[key],
                backward_comm_time=self.gradient_message_size[key] * self.dc * (self.bwd_factor - 0.5),
                backward_comp_time=self.bct[key],
                tp_time=self.tp_communication_time[key]
            )

        return time_cost, time_no_sync_gradient_cost


embedding_lmhead_memory_args_list = {
        'TrainArgsOptimize': ['disable_vtp', 'mixed_precision', 'async_grad_reduce', 'use_zero2_for_dp', 'pipeline_type'],
        'ProfileModelArgsOptimize': ['other_memory_pp_off', 'other_memory_pp_on' ],
        # 'UtilsArgsOptimize': ['pytorch_context_mem' ],
        'VersionOptionArgsOptimize': ['zero_with_slight_noise' ],
    }

class EmbeddingLMHeadMemoryCostModelOptimize:
    def __init__(self,
                 strategy:GalvatronStrategy,
                 global_batch_size:int,
                 chunks:int,
                 logger:Logger=None,
                 train_args:TrainArgsOptimize=None,
                 profile_model_args:ProfileModelArgsOptimize=None,
                 version_option_args:VersionOptionArgsOptimize=None):
        
        self.strategy = strategy
        self.global_batch_size = global_batch_size
        self.chunks = chunks
        self.logger = logger
        
        self.args = SimpleNamespace()
        components = {
                    'TrainArgsOptimize': train_args, 
                    'ProfileModelArgsOptimize': profile_model_args, 
                    'VersionOptionArgsOptimize': version_option_args,
        }
        for class_name, instance in components.items():
            for key, value in instance.__dict__.items():
                if key in embedding_lmhead_memory_args_list[class_name]:
                    setattr(self.args, key, value) 

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
        self.dp_type = self.strategy.dp_type
        self.sdp_size = self.tp_size * self.dp_size if self.use_ulysses else self.dp_size

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

        if self.pp_size == 1:
            if self.use_ulysses:
                model_states_memory = args.other_memory_pp_off['model_states'][1]
            else:
                model_states_memory = args.other_memory_pp_off['model_states'][self.tp_size]
            
            if self.dp_type == 'zero3':
                model_states_memory *= self.zero3_ratio(self.sdp_size)
            elif self.dp_type == 'zero2':
                model_states_memory *= self.zero2_ratio(self.sdp_size)

            self.model_states_memory = {
                'embedding': model_states_memory / 2,
                'lmhead': model_states_memory / 2
            }
        else:
            if self.use_ulysses:
                embedding_model_states_memory = args.other_memory_pp_on['first_stage']['model_states'][1]
                lmhead_model_states_memory = args.other_memory_pp_on['last_stage']['model_states'][1]
            else:
                embedding_model_states_memory = args.other_memory_pp_on['first_stage']['model_states'][self.tp_size]
                lmhead_model_states_memory = args.other_memory_pp_on['last_stage']['model_states'][self.tp_size]
            
            if self.dp_type == 'zero3':
                embedding_model_states_memory *= self.zero3_ratio(self.sdp_size)
                lmhead_model_states_memory *= self.zero3_ratio(self.sdp_size)
            elif self.dp_type == 'zero2':
                embedding_model_states_memory *= self.zero2_ratio(self.sdp_size)
                lmhead_model_states_memory *= self.zero2_ratio(self.sdp_size)

            self.model_states_memory = {
                'embedding': embedding_model_states_memory,
                'lmhead': lmhead_model_states_memory
            }

    def estimate_activation_memory(self):
        args = self.args
        
        if self.pp_size == 1:
            self.activation_memory = {
                'embedding': self.embedding_local_bsz * args.other_memory_pp_off['activation'][self.tp_size],
                'lmhead': self.lmhead_local_bsz * args.other_memory_pp_off['activation'][self.tp_size]
            }
        else:
            self.activation_memory = {
                'embedding': self.embedding_local_bsz * args.other_memory_pp_on['first_stage']['activation'][self.tp_size],
                'lmhead': self.lmhead_local_bsz * args.other_memory_pp_on['last_stage']['activation'][self.tp_size],
            }

    def get_memory_cost(self):
        result = dict()
        result['model_states_memory'] = self.model_states_memory
        result['activation_memory'] = self.activation_memory
        result['total_memory'] = {
            'embedding': self.model_states_memory['embedding'] + self.activation_memory['embedding'],
            'lmhead': self.model_states_memory['lmhead'] + self.activation_memory['lmhead']
        }
        return result