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
        # self.estimate_tp_communication_time()
        self.estimate_dispatch_communication_time()
        self.estimate_combine_communication_time()
    
    def initialize(self):
        args = self.args
        
        self.pp_size = self.strategy.pp_size
        self.ep_size = self.strategy.ep_size
        self.tp_of_ep_size = self.strategy.tp_of_ep_size
        self.tp_ep_size = self.ep_size * self.tp_of_ep_size
        print(f'[DEBUG] self.tp_ep_size: {self.tp_ep_size}')
        self.checkpoint = self.strategy.checkpoint
        self.world_size = self.strategy.world_size
        
        self.bsz = self.global_batch_size // self.chunks # NOTE no dp_size
        
        self.seq_length = args.seq_length
        self.hidden_size = args.hidden_size
        
        self.num_experts = args.num_experts
        self.top_k = args.top_k
        self.moe_grouped_gemm = args.moe_grouped_gemm
        
        # In the current modeling, we assume that experts are uniformly distributed and their loads are balanced.
        self.local_expert_num = self.num_experts // self.ep_size
        self.token_num_per_expert = self.seq_length * self.top_k / self.num_experts
        self.seq_length_scale_ratio = self.token_num_per_expert / self.seq_length
        
        # 开始分析
        # 首先，不管attention层是如何设置dp,tp，每个设备的初始token数量一定为global_batch_size // chunks  * seq_length / world_size
        self.origin_token_num_per_device = self.global_batch_size // self.chunks * self.seq_length // self.world_size
        # 之后，all_to_all的通信量就是在ep_size中将上述token进行分发。
        self.token_num_per_expert_group = self.origin_token_num_per_device * self.ep_size * self.top_k
        dtype_size = 2 if args.mixed_precision else 4
        byte_to_MB = 1024 * 1024
        self.all_to_all_message_size_in_MB = (self.ep_size - 1) * (self.origin_token_num_per_device * self.top_k / self.ep_size) * self.hidden_size * dtype_size / byte_to_MB

        # 接下来，构建一下tp通信的大小
        # 对于每个设备，其在all-to-all之后，其token数量一定origin_token_num_per_device * ep_size * top_k / ep_size
        self.token_num_per_device_after_all_to_all = self.origin_token_num_per_device * self.top_k
        self.tp_message_size_in_MB = (self.tp_of_ep_size - 1) * self.token_num_per_device_after_all_to_all * self.hidden_size * dtype_size / byte_to_MB

        # 最后是关于最开始很小的all_gather的通信量
        self.little_all_gather_message_size_in_MB = (self.tp_ep_size - 1) * self.num_experts * 8 / byte_to_MB # long -> 8 bytes

        # 对上述建模的解释
        # 首先是总共有global_batch_size // chunks * seq_length个token，每个token需要选择top_k个专家。
        # 由于负载均衡。所以每个专家的token数量为global_batch_size // chunks * seq_length // num_experts * topk
        # 所以对于有tp的情形下，每个专家的token数量为global_batch_size // chunks * seq_length // num_experts * topk // tp_of_ep_size
        # 而在tp的时候，ep_size就是world_size // tp_of_ep_size
        # 此时一个设备的专家数为 num_experts // ep_size = expert_num // world_size * tp_of_ep_size
        # 所以一个设备的token数量为 num_experts // world_size * tp_of_ep_size * global_batch_size // chunks * seq_length // num_experts * top_k // tp_of_ep_size = 
            # global_batch_size // chunks * seq_length // world_size * top_k
        # 所以一个设备的token的数量是常量!!!
        # 而当tp增大时，那么就说明 all_gather的量会增大。这一点是因为tp增大了，ep就减少了，此时一个设备上的专家总数也就变大了，而每个专家都需要完整的token，所以总的all_gather的量就变大了。
        # 那么tp越大，这个通信时间也就会越长。
        # 那么，对于all_to-all的通信量呢？
        # 当ep_size确定时，ep_group也就确定了，此时只会对ep_size个设备的token进行all_to_all通信。
        # 而一个设备的初始token数量为global_batch_size // chunks * seq_length // world_size(这一点是无论attention层采用了何种策略，都会是这样的)
        # 需要注意，还需要乘以top_k，这是因为，每个token都需要找top_k个专家，
        # 单个设备会将初始token分为ep_size份，每一份的大小是global_batch_size // chunks * seq_length // world_size // ep_size 
        # 按照上述通信量进行ep_size - 1次通信

        self.token_num_per_expert = self.global_batch_size // self.chunks * self.seq_length * self.top_k // self.num_experts
        self.equal_bsz = self.token_num_per_expert / self.seq_length
        # 这个equal_bsz是不会发生变化的。那么，tp对计算时间的影响应该是怎么样的呢？
        # 当tp_of_ep_size增大时，ep_size减少，单个设备上的专家数量增加
        # 此时就是需要liear_fun计算出来的值会不一样
        # return args.forward_computation_time * args.num_layers
    

    def estimate_computation_time(self):
        args = self.args

        if isinstance(args.forward_computation_time, np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            if self.moe_grouped_gemm == False:
                self.fct = linear_func(self.equal_bsz / self.tp_of_ep_size, *args.forward_computation_time)
                # self.fct = linear_func(self.global_batch_size // self.chunks // self.tp_of_ep_size, *args.forward_computation_time)
                # self.fct = self.fct * self.top_k / self.num_experts
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
        
        # if self.tp_of_ep_size == 8:
        #     self.fct = 5.013
        # elif self.tp_of_ep_size == 1:
            # self.fct = 2.747

        # [Step 2] estimate backward computation time
        self.bct = self.fct * args.bct_fct_coe
        if self.checkpoint:
            self.bct += self.fct
            
        # [DEBUG]
        ms_to_s = 1
        self.logger_print(f'moe fct time:{self.fct * ms_to_s * args.costmodel_coe}, bct time: {self.bct * ms_to_s * args.costmodel_coe}')
    
    def estimate_dispatch_communication_time(self):
        # [forward] tp_ep_group进行非常小规模的all_gather(说实话，此处的通信开销有点大啊，感觉可以和之前的什么操作进行合并?但是不能算是论文的创新点)
        # [forward] 如果ep!= 1, all_to_all通信 (已经构建了all_to_all的通信量)
        # [forward] 如果etp != 1，需要all_gather
        # [backward] 如果etp!= 1，需要reduce_scatter
        # [backward] 如果ep!= 1, all_to_all通信
        args = self.args

        # print(f'[DEBUG] args.allreduce_fit_dict: {args.allreduce_fit_dict}')

        def linear_func(x, m, c):
            return m * x + c
        litter_time = linear_func(self.little_all_gather_message_size_in_MB, *args.allreduce_fit_dict[self.tp_ep_size]['popt'])
        # litter_time = 0

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

        print(f'Litter time: {litter_time}')
        print(f'Fwd all2all time: {fwd_all2all}')
        print(f'Fwd all_gather time: {fwd_all_gather}')
        print(f'Bwd reduce_scatter time: {bwd_reduce_scatter}')
        print(f'Bwd all_to_all time: {bwd_all_to_all}')

        self.dispatch_communication_time = litter_time + fwd_all2all + fwd_all_gather + bwd_reduce_scatter + bwd_all_to_all 
        print(f'Dispatch communication time: {self.dispatch_communication_time}')
        
        return

        YELLOW = '\033[93m'
        RESET = '\033[0m'
        print(f'{YELLOW}Warning: The dispatch communication time modeling for MoE layers is not yet considered.{RESET}')
        return 0
    
    def estimate_combine_communication_time(self):
        # [forward] 如果etp!=1，reduce_scatter
        # [forward] 如果ep!=1, all_to_all通信 (已经构建了all_to_all的通信量)
        # [backward] 如果ep!=1, all_to_all通信
        # [backward] 如果etp!=1，all_gather
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
        print(f'Forward reduce scatter: {fwd_reduce_scatter}')
        print(f'Forward all2all: {fwd_all2all}')
        print(f'Backward all to all: {bwd_all_to_all}')
        print(f'Backward all gather: {bwd_all_gather}')
        self.combine_communication_time = fwd_reduce_scatter + fwd_all2all + bwd_all_to_all + bwd_all_gather
        print(f'Combine communication time: {self.combine_communication_time}')

        return

        YELLOW = '\033[93m'
        RESET = '\033[0m'
        print(f'{YELLOW}Warning: The combine communication time modeling for MoE layers is not yet considered.{RESET}')
        return 0
    
    def gen_result(self):
        result = self.fct + self.bct
        # 以下假设dispatch时间=combine时间=computation时间
        # computation_time = self.fct + self.bct
        # dispatch_time = computation_time
        # combine_time = computation_time
        # result = computation_time + dispatch_time + combine_time
        result = self.fct + self.bct + self.dispatch_communication_time + self.combine_communication_time
        ms_to_s = 1e-3
        result = result * ms_to_s * self.args.costmodel_coe
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
                    
                    self.fwd_tp_communication_time = per_operation_time * 2
                    self.bwd_tp_communication_time = per_operation_time * 1
                    
                    if self.checkpoint == False:
                        self.tp_communication_time = self.fwd_tp_communication_time + self.bwd_tp_communication_time
                    else:
                        self.tp_communication_time = self.fwd_tp_communication_time * 2 + self.bwd_tp_communication_time
                else: # use allreduce
                    raise NotImplemented('allreduce not implement')

class FFNMemoryCostModelOptimize(MemoryCostModelBase):
    pass