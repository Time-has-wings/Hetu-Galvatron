from .layer_cost import TimeCostModelBase
from .layer_cost import MemoryCostModelBase
from ..cost_model_args import EstimateTPTimeType
from galvatron.utils.strategy_utils import byte_to_MB


class AttentionTimeCostModel(TimeCostModelBase):
    def estimate_tp_communication_time(self):
        args = self.args
        
        if self.tp_size == 1:
            self.tp_communication_time = 0
        else:
            if args.estimate_tp_time_type == EstimateTPTimeType.FIXED:
                print(f'\nWarning: Fixed version of TP communication time is deprecated. Please use the FIT version instead.')
                if args.sequence_parallel:
                    # forward: <all_gather, hidden_states>, <reduce_scatter, hidden_states>
                    # backward: <all_gather, data_grad>, <all_gather hidden_states>, <reduce_scatter param_grad>
                    # In the backward pass, <all_gather hidden_states> and <reduce_scatter param_grad> can overlap with the computation. 
                    # In summary, 
                    # forward: 1 <all_gather, hidden_states>, 1 <reduce_scatter, hidden_states>
                    # backward: 1 <all_gather, data_grad> (data_grad.shape is the same as hidden_states.shape)
                    dtype_size = 2 if args.mixed_precision else 4
                    per_tp_message_size_in_MB = self.mbsz * (self.seq_length_per_rank * (self.tp_size - 1)) * self.hidden_size * dtype_size / byte_to_MB
                    
                    self.fwd_tp_communication_time = (per_tp_message_size_in_MB * args.all_gather_fixed_dict[f'{self.tp_size}_1'] * 1 + 
                                                      per_tp_message_size_in_MB * args.reduce_scatter_fixed_dict[f'{self.tp_size}_1'] * 1)
                    self.bwd_tp_communication_time = per_tp_message_size_in_MB * args.all_gather_fixed_dict[f'{self.tp_size}_1'] * 1

                    if not self.checkpoint:
                        self.tp_communication_time = self.fwd_tp_communication_time + self.bwd_tp_communication_time
                    else:
                        self.tp_communication_time = self.fwd_tp_communication_time * 2 + self.bwd_tp_communication_time
                else:
                    # Roughly estimate using 2 allreduce operations, or 3 allreduce operations if checkpointing is enabled
                    all_reduce_nums = 2 if not self.checkpoint else 3
                    dtype_size = 2 if args.mixed_precision else 4
                    per_tp_message_size_in_MB = self.mbsz * (self.seq_length_per_rank * (self.tp_size - 1)) * self.hidden_size * dtype_size / byte_to_MB
                    self.tp_communication_time = per_tp_message_size_in_MB * args.allreduce_fixed_dict[f'{self.tp_size}_1'] * all_reduce_nums
            elif args.estimate_tp_time_type == EstimateTPTimeType.FIT:
                if args.sequence_parallel:
                    # forward: <all_gather, hidden_states>, <reduce_scatter, hidden_states>
                    # backward: <all_gather, data_grad>, <all_gather hidden_states>, <reduce_scatter param_grad>
                    # In the backward pass, <all_gather hidden_states> and <reduce_scatter param_grad> can overlap with the computation. 
                    # In summary, 
                    # forward: 1 <all_gather, hidden_states>, 1 <reduce_scatter, hidden_states>
                    # backward: 1 <all_gather, data_grad> (data_grad.shape is the same as hidden_states.shape)
                    dtype_size = 2 if args.mixed_precision else 4
                    per_tp_message_size_in_byte = self.mbsz * (self.seq_length_per_rank * self.tp_size) * self.hidden_size * dtype_size

                    tp_dict = args.all_gather_fit_dict[self.tp_size]
                    if per_tp_message_size_in_byte in tp_dict:
                        per_tp_operation_time = tp_dict[per_tp_message_size_in_byte]
                    else:
                        def linear_func(x, m, c):
                            return m * x + c
                        per_tp_operation_time = linear_func(per_tp_message_size_in_byte / byte_to_MB, *tp_dict['popt'])

                    self.fwd_tp_communication_time = per_tp_operation_time * 2
                    self.bwd_tp_communication_time = per_tp_operation_time * 1
                    
                    if self.checkpoint == False:
                        self.tp_communication_time = self.fwd_tp_communication_time + self.bwd_tp_communication_time
                    else:
                        self.tp_communication_time = self.fwd_tp_communication_time * 2 + self.bwd_tp_communication_time
                else:
                    # Roughly estimate using 2 allreduce operations, or 3 allreduce operations if checkpointing is enabled
                    all_reduce_nums = 2 if not self.checkpoint else 3
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
            sp_dict = args.all2all_dict[self.sp_size]
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

class AttentionMemoryCostModel(MemoryCostModelBase):
    pass