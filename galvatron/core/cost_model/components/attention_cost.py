from .layer_cost import TimeCostModelBase
from ..cost_model_args_optimize import EstimateTPTimeType

class AttentionTimeCostModelOptimize(TimeCostModelBase):
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