import numpy as np
import copy
from typing import Optional, Callable, Union
from dataclasses import dataclass, field
class MemoryCostModel:
    def __init__(self,
            strategy,
            global_batch_size = 8,
            parameter_size = 48,
            tp_activation_per_bsz_dict = {1:85, 2:47, 4:28, 8:18.5},
            other_memory_pp_off = {'model_states': 640, 'activation': 320},
            other_memory_pp_on = {'first_stage':{'model_states': 640, 'activation': 320}, 'last_stage':{'model_states': 640, 'activation': 320}},
            peak_reduction_with_chunks=None,
            microbatch=True,
            optimal_chunk_func=None,
            pytorch_context_mem = 1024,
            model_type='bert',
            checkpoint=0,
            use_zero2_for_dp=0,
            use_zero3_for_embed=0,
            mixed_precision=False,
            pipeline_type='gpipe', 
            disable_vtp=0,
            max_tp_deg=8,
            stage_idx=0,
            mbsz=-1,
            min_tp = -1,
            gpu_num = 8,
            chunks=None,
            async_grad_reduce=True,
            sequence_parallel=True,
            vsp=0):
        assert mbsz > -1
        assert min_tp > -1
        self.strategy = strategy
        self.pp_size = self.strategy[0]
        self.tp_size = self.strategy[1]
        self.dp_size = self.strategy[2]
        if 'sp' in self.strategy[-1].keys() and self.strategy[-1]['sp'] == 1:
            self.sdp_size = self.tp_size * self.dp_size
            self.parameter_size = parameter_size
        else:
            self.sdp_size = self.dp_size
            self.parameter_size = parameter_size/self.tp_size
        
        self.model_states_size = 4 * self.parameter_size
        self.max_tp_deg = max_tp_deg
        self.gpu_num = gpu_num
        self.disable_vtp = disable_vtp

        self.bsz = global_batch_size/self.dp_size
        if chunks is None:
            chunks = optimal_chunk_func(global_batch_size//self.dp_size, strategy, mbsz, min_tp) # if microbatch else 1
        max_chunks = global_batch_size // (self.tp_size*self.dp_size // min_tp)
        max_chunks = 1 if max_chunks == 0 else max_chunks
        self.chunks = max_chunks if chunks > max_chunks else chunks
        self.chunks = int(self.chunks)
        
        if (pipeline_type == 'pipedream_flush' and self.pp_size > 1) or self.pp_size==1:
            microbatches = [t.shape[0] for t in chunk_like_torch(int(global_batch_size/self.dp_size/(self.tp_size//min_tp)), self.chunks)]
            assert self.chunks == len(microbatches)
            end = self.pp_size-stage_idx if self.pp_size-stage_idx <= self.chunks else self.chunks
            act_1f1b_ratio = np.sum(microbatches[:end]) / np.sum(microbatches)
            act_1f1b_ratio_first = np.sum(microbatches[:min(self.pp_size, self.chunks)]) / np.sum(microbatches)
            act_1f1b_ratio_last = microbatches[0] / np.sum(microbatches)
            self.bsz = act_1f1b_ratio * self.bsz
        else:
            microbatches = [t.shape[0] for t in chunk_like_torch(int(global_batch_size/self.dp_size/(self.tp_size//min_tp)), self.chunks)]
            self.bsz = microbatches[0]
        
        if 'cpt' in self.strategy[-1].keys() and self.strategy[-1]['cpt']:
            assert(tp_activation_per_bsz_dict['checkpoint'] is not None)
            self.activation_size = tp_activation_per_bsz_dict['checkpoint'] * self.bsz
            if sequence_parallel:
                self.activation_size /= self.tp_size
        else:
            self.activation_size = tp_activation_per_bsz_dict[self.tp_size] * self.bsz
        
        if self.chunks == 1:
            zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            zero3_ratio = lambda d: (1/d+0.003)
        else:
            if async_grad_reduce:
                zero2_ratio = (lambda d: (6/8 * (1/d + 0.003) + 2/8)) if mixed_precision else (lambda d: (2/4 * (1/d + 0.003) + 2/4))
                zero3_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            else:
                zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8) * 5/4) if mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                zero3_ratio = lambda d: (1/d+0.003) * 5/4
                # *5/4: for fp32 grad 
        
        if 'fsdp' in self.strategy[-1].keys() and self.strategy[-1]['fsdp']:
            # fsdp_model_states memory is slightly larger than dp_model_states/dp_size
            # we add a small bias to ensure the predicted fsdp memory NOT smaller than real value
            # Actually, this bias barely affect search result.
            self.model_states_size  *= zero3_ratio(self.sdp_size)
        elif 'fsdp' in self.strategy[-1].keys() and self.strategy[-1]['fsdp']==0 and use_zero2_for_dp:
            self.model_states_size *= zero2_ratio(self.sdp_size)
        
        self.total = self.model_states_size + self.activation_size
        
        total_min_tp = []
        i = min_tp
        while i * self.pp_size <= self.gpu_num and i <= self.max_tp_deg:
            total_min_tp.append(i)
            i *= 2
        if self.disable_vtp:
            total_min_tp = [1]
        
        self.other_memcosts = dict()
        for tp in total_min_tp:
            tp_other_memcosts = [0] * self.pp_size
            other_layers_bsz = global_batch_size * tp /self.tp_size/self.dp_size
            
            # print(self.tp_size, self.dp_size, tp)
            if vsp:
                model_tp = 1
                other_ms_zero2_ratio = zero3_ratio(self.tp_size*self.dp_size) if use_zero3_for_embed else (zero2_ratio(self.tp_size*self.dp_size) if use_zero2_for_dp else 1.0)
            else:
                model_tp = tp
                other_ms_zero2_ratio = zero3_ratio(self.tp_size*self.dp_size//tp) if use_zero3_for_embed else (zero2_ratio(self.tp_size*self.dp_size//tp) if use_zero2_for_dp else 1.0)
            
            model_type = 'gpt' if model_type not in ['bert', 't5', 'vit', 'swin', 'gpt'] else model_type
            
            # print(other_memory_pp_off['model_states'])
            if self.pp_size == 1:
                tp_other_memcosts[0] += (
                    other_memory_pp_off['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    other_memory_pp_off['activation'][tp] * 
                    other_layers_bsz * 
                    act_1f1b_ratio
                )
            else:
                if pipeline_type == 'pipedream_flush':
                    other_layers_bsz_first = other_layers_bsz * act_1f1b_ratio_first
                    other_layers_bsz_last = other_layers_bsz * act_1f1b_ratio_last
                else:
                    other_layers_bsz_first = other_layers_bsz_last = other_layers_bsz
                # TODO: check the correctness of other memory cost for first stage and last stage
                tp_other_memcosts[0] += (
                    other_memory_pp_on['first_stage']['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    other_memory_pp_on['first_stage']['activation'][tp] * 
                    other_layers_bsz_first
                )
                tp_other_memcosts[-1] += (
                    other_memory_pp_on['last_stage']['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    other_memory_pp_on['last_stage']['activation'][tp] * 
                    other_layers_bsz_last
                )
            # if checkpoint:
            #     for i in range(len(tp_other_memcosts)):
            #         tp_other_memcosts[i] += tp_activation_per_bsz_dict[self.tp_size] * mbsz

            for i in range(len(tp_other_memcosts)):
                tp_other_memcosts[i] += pytorch_context_mem
                
            self.other_memcosts[tp] = tp_other_memcosts

    def get_memory_cost(self):
        result = dict()
        result['parameter'] = self.parameter_size
        result['model_states'] = self.model_states_size
        result['activation'] = self.activation_size
        result['enc_total'] = self.total
        result['other'] = self.other_memcosts
        return result


class TimeCostModel:
    def __init__(self,
            strategy,
            global_batch_size,
            parameter_size = 48,
            microbatch=True,
            optimal_chunk_func = None,
            sequence_length=512,
            hidden_size=1024,
            vocab_size=32000,
            forward_computation_time=35 / 24,
            bct_fct_coe=2,
            extra_overhead=0,
            comm_coe_dict={},
            dp_overlap_coe=1.3,
            bct_overlap_coe=1.3,
            p2p_comm_coe_dict=None,
            layer_num=None,
            use_zero2_for_dp=0,
            mixed_precision=False,
            no_comm=False,
            costmodel_coe=1.0,
            async_grad_reduce=True,
            allreduce_dict = None,
            all2all_dict = None,
            sp_space = 'tp'):
        # TODO: align time cost model when async_grad_reduce is False
        assert microbatch == False
        self.s = strategy[:3]
        self.sl = sequence_length
        self.hs = hidden_size
        self.microbatch = microbatch
        self.pp_size = self.s[0]
        self.tp_size = self.s[1]
        self.dp_size = self.s[2]
        self.sp_space = sp_space
        if 'sp' in strategy[-1].keys() and strategy[-1]['sp'] == 1:
            self.sdp_size = self.tp_size * self.dp_size
            self.parameter_size = parameter_size
            if self.tp_size == 1:
                self.sp_dict = np.inf
            else:
                self.sp_dict = all2all_dict[self.tp_size]
        else:
            self.sdp_size = self.dp_size
            self.parameter_size = parameter_size/self.tp_size
            if self.tp_size == 1:
                self.sp_dict = np.inf
            else:
                self.sp_dict = allreduce_dict[self.tp_size]

        self.comm_coe_dict = comm_coe_dict
        self.costmodel_coe = costmodel_coe
        if 'sp' in strategy[-1].keys() and strategy[-1]['sp'] == 1:
            self.dc = self.comm_coe_dict['%d'%self.sdp_size] if '%d'%self.sdp_size in self.comm_coe_dict.keys() else self.comm_coe_dict['%d_1'%self.sdp_size]
            if self.tp_size == 1 or self.dp_size == 1:
                self.tc = self.comm_coe_dict['%d'%self.tp_size] if '%d'%self.tp_size in self.comm_coe_dict.keys() else self.comm_coe_dict['%d_1'%self.tp_size]
            else:
                # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
                info = strategy[-1]
                assert 'tp' in info.keys() and info['tp'] in [0, 1]
                tp_consecutive_flag = info['tp']
                if tp_consecutive_flag:
                    self.tc = self.comm_coe_dict['%d_1'%self.tp_size]
                else:
                    self.tc = self.comm_coe_dict['%d_0'%self.tp_size]
        else:
            if self.tp_size == 1 or self.dp_size == 1:
                self.dc = self.comm_coe_dict['%d'%self.dp_size] if '%d'%self.dp_size in self.comm_coe_dict.keys() else self.comm_coe_dict['%d_1'%self.dp_size]
                self.tc = self.comm_coe_dict['%d'%self.tp_size] if '%d'%self.tp_size in self.comm_coe_dict.keys() else self.comm_coe_dict['%d_1'%self.tp_size]
            else:
                # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
                info = strategy[-1]
                assert 'tp' in info.keys() and info['tp'] in [0, 1]
                tp_consecutive_flag = info['tp']
                if tp_consecutive_flag:
                    self.dc = self.comm_coe_dict['%d_0'%self.dp_size]
                    self.tc = self.comm_coe_dict['%d_1'%self.tp_size]
                else:
                    self.dc = self.comm_coe_dict['%d_1'%self.dp_size]
                    self.tc = self.comm_coe_dict['%d_0'%self.tp_size]
        self.fsdp = False
        if 'fsdp' in strategy[-1].keys() and strategy[-1]['fsdp']:
            self.fsdp = True
        self.dp_overlap_coe = dp_overlap_coe
        self.dc_overlap = self.dc*dp_overlap_coe

        self.bs = global_batch_size/self.dp_size 
        self.optimal_microbatch = optimal_chunk_func(self.bs, self.s) if microbatch else 1

        # Dummy layer_num, can be any multiple of 8.
        # We estimate the time cost of single layer by averaging the time of whole model to deal with pipeline parallel
        self.layer_num = 24 if layer_num is None else layer_num

        self.checkpoint = False
        if 'cpt' in strategy[-1].keys() and strategy[-1]['cpt']:
            self.checkpoint = True

        # forward & backward computation time of whole model (depending on dummy layer_num)
        if isinstance(forward_computation_time,np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            self.fct = linear_func(self.bs / self.tp_size, *forward_computation_time) * self.layer_num
        else:
            self.fct = forward_computation_time * self.bs / self.tp_size * self.layer_num 
        self.bct = self.fct * bct_fct_coe
        self.bct_overlap_coe = bct_overlap_coe
        self.bct_overlap = self.bct*bct_overlap_coe
        self.eo = extra_overhead

        # dp & tp message size of whole model (depending on dummy layer_num)
        self.dp_message_size = (2*(self.dp_size-1)/self.dp_size*self.parameter_size) * self.layer_num
        
        if self.sp_space == 'tp+sp':
            self.per_tp_message_size = self.bs*self.sl*self.hs * (2 if mixed_precision else 4)
            self.tp_comm_num = 4
            self.tp_comm_num *= self.layer_num

            if self.tp_size == 1:
                self.per_tp_message_time = 0
            else:
                if self.per_tp_message_size in self.sp_dict:
                    self.per_tp_message_time = self.sp_dict[self.per_tp_message_size // self.tp_size]
                else:
                    def linear_func(x, m, c):
                        return m * x + c
                    self.per_tp_message_time = linear_func( 1 / 1024 / 1024 * self.per_tp_message_size / self.tp_size, *self.sp_dict["popt"] )
        else:
            tp_comm_times = 4
            self.tp_message_size = 2*(self.tp_size-1)/self.tp_size*(self.bs*self.sl*self.hs*tp_comm_times*4/1024/1024) * self.layer_num

        # if self.fsdp:
        #     self.dp_message_size = self.dp_message_size * 0.5

        # if self.fsdp:
        #     self.dp_message_size_ori = self.dp_message_size
        #     self.dp_message_size = self.dp_message_size * 1.5

        self.p2p_comm_coe = None
        if self.pp_size > 1 and p2p_comm_coe_dict is not None:
            self.p2p_comm_coe = p2p_comm_coe_dict[self.pp_size]
            self.p2p_meg_size = self.pp_size*2*self.bs*self.sl*self.hs*4/1024/1024
            if mixed_precision:
                self.p2p_meg_size = self.p2p_meg_size/2

        self.use_zero2_for_dp = use_zero2_for_dp
        if self.checkpoint:
            # self.fct *= 2
            self.bct += self.fct #  * 0.5
            if self.sp_space == 'tp+sp':
                self.tp_comm_num *= 1.5
            else:
                self.tp_message_size *= 1.5

        if mixed_precision:
            self.dp_message_size = self.dp_message_size/2
            if self.sp_space == 'tp+sp':
                pass
            else:
                self.tp_message_size = self.tp_message_size/2

        self.fsdp_allgather_message_size = self.dp_message_size * 0.5
        
        if no_comm:
            self.dp_message_size = 0

        if self.sp_space == 'tp+sp':
            self.tp_time = self.tp_comm_num * self.per_tp_message_time
        else:
            self.tp_time = self.tp_message_size*self.tc
        
        

    def bct_dp_overlap(self, dp_message_size, bct):
        dp_overlap_time = dp_message_size * self.dc_overlap
        bct_overlap_time = bct * self.bct_overlap_coe
        if dp_overlap_time > bct_overlap_time:
            overlap_part = bct_overlap_time
            rest_part = (dp_message_size - bct_overlap_time / self.dc_overlap) * self.dc
            rest_dp_flag = True
        elif dp_overlap_time < bct_overlap_time:
            overlap_part = dp_overlap_time
            rest_part = (bct - dp_overlap_time / self.bct_overlap_coe) 
            rest_dp_flag = False
        else:
            overlap_part = bct_overlap_time
            rest_part = 0
            rest_dp_flag = False
        rest_dp_flag = False
        return overlap_part, rest_part, rest_dp_flag

    def pipe_with_microbatch(self, computation_overhead, communication_overhead):
        result = computation_overhead*(self.pp_size+self.optimal_microbatch-1)/(self.pp_size*self.optimal_microbatch)+communication_overhead
        return result

    def gen_result(self):
        if self.pp_size >= 1:
            if self.tp_size == 1 and self.dp_size > 1: # pp+dp
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                overall_overhead = self.fct + overlap_part + rest_part + self.eo
                if self.microbatch == False:
                    result = overall_overhead
                else:
                    computation_overhead = self.fct + self.bct
                    communication_overhead = overall_overhead-computation_overhead
                    result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
            elif self.dp_size == 1 and self.tp_size > 1: # pp+tp
                if self.microbatch == False:
                    result = self.fct + self.bct + self.tp_time
                else:
                    overall_overhead = self.fct + self.bct + self.tp_time
                    result = self.pipe_with_microbatch(overall_overhead, 0)
            elif self.dp_size == 1 and self.tp_size == 1: # pure pp
                if self.microbatch == False:
                    result = self.fct + self.bct
                else:
                    overall_overhead = self.fct + self.bct
                    result = self.pipe_with_microbatch(overall_overhead, 0)
            else: # pp+dp+tp
                if self.tp_size < self.tp_size * self.dp_size // 2:
                    overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                    overall_overhead = self.fct + overlap_part + rest_part + self.tp_time + self.eo
                    if self.microbatch == False:
                        result = overall_overhead
                    else:
                        computation_overhead = self.fct + self.bct + self.tp_time
                        communication_overhead = overall_overhead-computation_overhead
                        result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
                else:
                    overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct*1/2)
                    overall_overhead = self.fct + 1/2*self.bct + overlap_part + rest_part + self.tp_time + self.eo
                    if self.microbatch == False:
                        result = overall_overhead
                    else:
                        computation_overhead = self.fct + self.bct + self.tp_time
                        communication_overhead = overall_overhead-computation_overhead
                        result = self.pipe_with_microbatch(computation_overhead, communication_overhead)

        # For fsdp, add allgather time of forward and backward
        if self.fsdp:
            forward_allgather_time = self.fsdp_allgather_message_size * self.dc
            result = result + forward_allgather_time*self.optimal_microbatch

        if self.pp_size > 1 and self.p2p_comm_coe is not None:
            result = result + self.p2p_meg_size * self.p2p_comm_coe
        
        coe = 0.001 * self.costmodel_coe
        result = result*coe
        result = result / self.layer_num
        return result

class OtherTimeCostModel:
    def __init__(
            self,
            mbsz,
            pp_deg,
            world_size,
            sequence_length,
            hidden_size,
            mixed_precision,
            comm_coe_dict,
            allreduce_dict,
            sp_space,
            vsp,
            min_tp,
            max_tp,
            other_memory_pp_on,
            other_memory_pp_off,
            other_time_profiled_list,
    ):
        self.mbsz = mbsz
        self.sl = sequence_length
        self.hs = hidden_size
        self.vsp = vsp
        self.min_tp = min_tp
        self.max_tp = max_tp
        self.comm_coe_dict = comm_coe_dict
        self.dp_coe = dict()
        self.pp_deg = pp_deg
        
        self.fct = dict()
        self.tp_time = dict()
        self.sp_size = dict()
        self.dp_size = dict()
        self.comm_factor = dict()
        # calc tp comm size
        k = min_tp
        while k <= max_tp and world_size // pp_deg >= k:
            if self.vsp == 0:
                if sp_space == 'tp+sp':
                    self.per_tp_message_size = self.mbsz*self.sl*self.hs * (2 if mixed_precision else 4)
                    if k == 1:
                        self.per_tp_message_time = 0
                    else:
                        if self.per_tp_message_size in allreduce_dict:
                            self.per_tp_message_time = allreduce_dict[self.per_tp_message_size]
                        else:
                            def linear_func(x, m, c):
                                return m * x + c
                            self.per_tp_message_time = linear_func( 1 / 1024 / 1024 * self.per_tp_message_size, *allreduce_dict[k]["popt"] )
                else:
                    dp_size = world_size // pp_deg // k
                    if k == 1 or dp_size == 1:
                        tp_coe = self.comm_coe_dict['%d'%k] if '%d'%k in self.comm_coe_dict.keys() else self.comm_coe_dict['%d_1'%k]
                    else:
                        tp_coe = self.comm_coe_dict['%d_0'%k]

                    self.tp_message_size = (k-1)/k*(self.mbsz*self.sl*self.hs/1024/1024) * (2 if mixed_precision else 4)
                    self.per_tp_message_time = self.tp_message_size * tp_coe
            else:
                self.per_tp_message_time = 0
            if pp_deg == 1:
                self.tp_time[k] = 2 * self.per_tp_message_time
            else:
                self.tp_time[k] = (self.per_tp_message_time, self.per_tp_message_time)
            k *= 2
        # calc calc time (ms)
        k = min_tp
        while k <= max_tp and world_size // pp_deg >= k:
            def linear_func(x, m, c):
                return m * x + c
            if pp_deg == 1:
                if isinstance(other_time_profiled_list ,np.ndarray):
                    self.fct[k] = linear_func(mbsz / min_tp, *other_time_profiled_list)
                else:
                    self.fct[k] = mbsz / min_tp * k * other_time_profiled_list
            else:
                if isinstance(other_time_profiled_list, np.ndarray):
                    self.fct[k] = (linear_func(mbsz / min_tp, *other_time_profiled_list) / 2, \
                                linear_func(mbsz / min_tp, *other_time_profiled_list) / 2)
                else:
                    self.fct[k] = (mbsz / min_tp * k * other_time_profiled_list / 2, \
                                mbsz / min_tp * k * other_time_profiled_list / 2)
            k *= 2
        # calc dp comm size
        k = min_tp
        while k <= max_tp and world_size // pp_deg >= k:
            if vsp == 0:
                dp_size = world_size // pp_deg // k
                if k == 1 or dp_size == 1:
                    self.dp_coe[k] = self.comm_coe_dict['%d'%dp_size] if '%d'%dp_size in self.comm_coe_dict.keys() else self.comm_coe_dict['%d_1'%dp_size]
                else:
                    self.dp_coe[k] = self.comm_coe_dict['%d_0'%dp_size]
                self.dp_coe[k] *= (dp_size - 1) / dp_size # bus -> alg
            else:
                dp_size = world_size // pp_deg
                self.dp_coe[k] = self.comm_coe_dict['%d'%dp_size] if '%d'%dp_size in self.comm_coe_dict.keys() else self.comm_coe_dict['%d_1'%dp_size]
                self.dp_coe[k] *= (dp_size - 1) / dp_size # bus -> alg
            if pp_deg == 1:
                if vsp == 0:
                    self.dp_size[k] = other_memory_pp_off['model_states'][k] / 4
                else:
                    self.dp_size[k] = other_memory_pp_off['model_states'][1] / 4
            else:
                if vsp == 0:
                    self.dp_size[k] = (other_memory_pp_on['first_stage']['model_states'][k] / 4, other_memory_pp_on['first_stage']['model_states'][k] / 4)
                else:
                    self.dp_size[k] = (other_memory_pp_on['last_stage']['model_states'][1] / 4, other_memory_pp_on['last_stage']['model_states'][1] / 4)
            k *= 2

    def gen_result(self):
        
        other_time_cost = dict()
        k = self.min_tp
        for k in self.dp_size.keys():
            other_time_cost[k] = [0] * self.pp_deg 
            if self.pp_deg  == 1:
                other_time_cost[k][0] = 0.001 * (self.dp_size[k] * self.dp_coe[k] + self.fct[k] * 3 + self.tp_time[k]) # + 4 * self.sp_time[k] # fwd + bwd
            else:
                other_time_cost[k][0] = 0.001 * (self.dp_size[k][0] * self.dp_coe[k] + self.fct[k][0] * 3 + self.tp_time[k][0]) # + 2 * self.sp_time[k]
                other_time_cost[k][-1] = 0.001 * (self.dp_size[k][-1] * self.dp_coe[k] + self.fct[k][-1] * 3 + self.tp_time[k][-1]) # + 2 * self.sp_time[k]
        return other_time_cost

def chunk_like_torch(size, chunks):
    """Implement torch.arange(size).chunk(chunks) behavior using numpy"""
    if chunks <= 0:
        raise ValueError("chunks must be positive")
    
    # Calculate chunk size like PyTorch does
    chunk_size = (size + chunks - 1) // chunks  # ceiling division
    
    # Create splits
    splits = []
    for i in range(chunks):
        start = i * chunk_size
        if start >= size:
            break
        end = min(start + chunk_size, size)
        splits.append(np.arange(start, end))
    
    return splits

def get_real_chunk(local_bsz, chunk):
    if chunk == 1:
        return 1
    chunk = int(chunk)
    re = [t.shape[0] for t in chunk_like_torch(int(local_bsz), chunk)]
    return len(re)

def get_time_cost_all_stages(layer_timecosts, pp_stage_division):
    assert(np.sum(pp_stage_division)==len(layer_timecosts))
    stage_timecosts = []
    for stage_id in range(len(pp_stage_division)):
        layer_start_id, layer_end_id = int(np.sum(pp_stage_division[:stage_id])), int(np.sum(pp_stage_division[:stage_id+1]))
        stage_timecosts.append(np.sum(layer_timecosts[layer_start_id:layer_end_id]))
    return stage_timecosts

def pipeline_costmodel(timecostmodel, layer_num_list, timecostmodel_args_list, strategies, partition, chunks, bsz, min_tp, other_time_cost, return_stage_cost=False):
    if strategies is None:
        if return_stage_cost:
            return [np.inf] * len(partition), np.inf
        else:
            return np.inf
    layer_type_ids = []
    # print(layer_num_list)
    for layer_type_id in range(len(layer_num_list)):
        layer_type_ids += [layer_type_id] * layer_num_list[layer_type_id]
    if isinstance(chunks, list):
        chunks = [get_real_chunk(int(bsz/(strategies[0][1] * strategies[0][2] // min_tp)), chunks_) for chunks_ in chunks]
        bsz_chunked = [bsz / chunks_ for chunks_ in chunks]
        max_chunk = np.max(chunks)
        # print('Detected multi chunks!', chunks, 'Using %d as chunks!'%max_chunk)
    else:
        chunks = get_real_chunk(int(bsz/(strategies[0][1] * strategies[0][2] // min_tp)), chunks)
        bsz_chunked = [bsz / chunks] * len(layer_num_list)
        # print(bsz, bsz/chunks, chunks)
        max_chunk = chunks
         
    pp_deg = len(partition)
    layer_num = len(strategies)
    from galvatron.utils import form_strategy, strategy_str2list
    strategies_set = list(set([form_strategy(s) for s in strategies]))
    timecosts_dict_bsz_chunked, timecosts_dict_compute = {}, {}
    for layer_type_id in range(len(layer_num_list)):
        timecosts_dict_bsz_chunked[layer_type_id], timecosts_dict_compute[layer_type_id] = {}, {}
        for s in strategies_set:
            timecosts_dict_bsz_chunked[layer_type_id][s] = timecostmodel(strategy=strategy_str2list(s), global_batch_size=bsz_chunked[layer_type_id], **timecostmodel_args_list[layer_type_id]).gen_result()
            timecosts_dict_compute[layer_type_id][s] = timecostmodel(strategy=strategy_str2list(s), global_batch_size=bsz_chunked[layer_type_id], no_comm=True, **timecostmodel_args_list[layer_type_id]).gen_result()
    timecosts_bsz_chunked = [timecosts_dict_bsz_chunked[layer_type_ids[i]][form_strategy(strategies[i])] for i in range(layer_num)]
    timecosts_bsz_compute = [timecosts_dict_compute[layer_type_ids[i]][form_strategy(strategies[i])] for i in range(layer_num)]
    stage_costs_bsz_chunked = get_time_cost_all_stages(timecosts_bsz_chunked, partition)
    stage_costs_compute = get_time_cost_all_stages(timecosts_bsz_compute, partition)
    assert(len(other_time_cost) == len(stage_costs_compute))
    for i in range(len(other_time_cost)):
        stage_costs_compute[i] += other_time_cost[i]
    # print(timecosts_bsz_chunked, stage_costs_bsz_chunked, np.sum(stage_costs_bsz_chunked))
    # print(stage_costs_compute, np.max(stage_costs_compute))
    # print(np.sum(stage_costs_bsz_chunked), np.max(stage_costs_compute), np.max(stage_costs_compute) * (max_chunk-1))
    
    # # p2p & reduce sync
    # result = np.sum(stage_costs_bsz_chunked) + np.max(stage_costs_compute) * (max_chunk-1)
    
    # p2p & reduce async
    stage_costs_reduce = [total for total in stage_costs_bsz_chunked]
    # print(stage_costs_compute, stage_costs_reduce, stage_costs_bsz_chunked)
    result = np.sum(stage_costs_compute) + stage_costs_compute[-1] * (max_chunk - 1)
    # assume t_rank0 > t_rank1 > ... , warmup and cool down bubble can be overlapped
    result = max( result,
            max( min(pp_deg - 1, max_chunk - 1) * stage_costs_compute[0] * 1/3, np.sum(stage_costs_compute[1:]) * 1/3) + 
            max( min(pp_deg - 1, max_chunk - 1) * stage_costs_compute[0] * 2/3, np.sum(stage_costs_compute[1:]) * 2/3) + 
            stage_costs_compute[0] * max(0, max_chunk + 1 - pp_deg))
    
    # result += max(np.max(stage_costs_compute) * 2/3 * (max_chunk - 1), stage_costs_compute[-1] * (max_chunk - 1))
    # result = np.max(stage_costs_compute) * (max_chunk-1+pp_deg)
    for i in range(pp_deg):
        stage_costs_reduce[i] -= np.sum(stage_costs_compute[:i+1])
    reduce_time = np.max(stage_costs_reduce)
    reduce_time = reduce_time if reduce_time > 0 else 0
    
    # print(result,reduce_time)
    result += reduce_time
    
    if return_stage_cost:
        return stage_costs_bsz_chunked, result
    return result

@dataclass
class MemoryCostArgs:
    strategy: list  # for example: [pp_deg, tp_deg, dp_deg, {'sp': 1, 'cpt': 1}]
    global_batch_size: int = 8
    parameter_size: float = 48  # single layer parameter size
    tp_activation_per_bsz_dict: dict = field(default_factory=lambda: {1: 85, 2: 47, 4: 28, 'checkpoint': 28.0})
    other_memory_pp_off: dict = field(default_factory=lambda: {'model_states': {1: 16, 2:83, 4:41}, 'activation': {1: 28, 2: 11, 4: 5}})
    other_memory_pp_on: dict = field(default_factory=lambda: {'first_stage': {'model_states': {1: 83, 2: 41, 4: 20}, 'activation': {1: 36, 2: 11, 4: 8}}, 'last_stage': {'model_states': {1: 83, 2: 41, 4: 20}, 'activation': {1: 23, 2: 11, 4: 5}}})
    peak_reduction_with_chunks: Optional[Union[dict, int, float]] = None  # dict or numeric or None
    optimal_chunk_func: Optional[Callable] = None
    pytorch_context_mem: float = 1024
    model_type: str = 'bert'
    use_zero2_for_dp: int = 0
    use_zero3_for_embed: int = 0
    mixed_precision: bool = False
    pipeline_type: str = 'gpipe'
    disable_vtp: int = 0
    max_tp_deg: int = 8
    stage_idx: int = 0
    mbsz: int = -1
    min_tp: int = -1
    gpu_num: int = 8
    chunks: Optional[int] = None
    async_grad_reduce: bool = True
    sequence_parallel: bool = True
    vsp: int = 0

class MemoryCost:
    def __init__(self, **kwargs):
        self.args = MemoryCostArgs(**kwargs)
        self._validate()
        self.initialize()
        self.estimate_parameter_size()
        self.estimate_model_states_size()
        self.estimate_activation_size()
        self.estimate_other_memcosts()
    
    def _validate(self):
        assert self.args.mbsz > -1, f'Invalid mbsz: {self.args.mbsz}'
        assert self.args.min_tp > -1, f'Invalid min_tp: {self.args.min_tp}'
        
    def initialize(self):
        args = self.args
        
        # [initialize]:initialize strategy
        self.pp_size = args.strategy[0]
        self.tp_size = args.strategy[1]
        self.dp_size = args.strategy[2]
        if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
            self.sdp_size = self.tp_size * self.dp_size
        else:
            self.sdp_size = self.dp_size
    
        # [adjust]:adjust args.chunks
        if args.chunks is None:
            args.chunks = args.optimal_chunk_func(args.global_batch_size // self.dp_size, args.strategy, args.mbsz, args.min_tp)
        max_chunks = args.global_batch_size // (self.tp_size * self.dp_size // args.min_tp)
        max_chunks = 1 if max_chunks == 0 else max_chunks
        args.chunks = max_chunks if args.chunks > max_chunks else args.chunks
        args.chunks = int(args.chunks)
        
        # [initialize]:initialize zero2 and zero3 ratio
        if args.chunks == 1:
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
        
        # [initialize]:initialize local batch size and pp stage act_1f1b_ratio
        self.bsz = args.global_batch_size / self.dp_size
        if (args.pipeline_type == 'pipedream_flush' and self.pp_size > 1) or self.pp_size == 1:
            microbatches = [t.shape[0] for t in chunk_like_torch(int(args.global_batch_size / self.dp_size / (self.tp_size // args.min_tp)), args.chunks)]
            assert args.chunks == len(microbatches)
            end = self.pp_size - args.stage_idx if self.pp_size - args.stage_idx <= args.chunks else args.chunks
            self.act_1f1b_ratio = np.sum(microbatches[:end]) / np.sum(microbatches)
            self.act_1f1b_ratio_first = np.sum(microbatches[:min(self.pp_size, args.chunks)]) / np.sum(microbatches)
            self.act_1f1b_ratio_last = microbatches[0] / np.sum(microbatches)
            self.bsz = self.act_1f1b_ratio * self.bsz
        else:
            microbatches = [t.shape[0] for t in chunk_like_torch(int(args.global_batch_size / self.dp_size / (self.tp_size // args.min_tp)), args.chunks)]
            self.bsz = microbatches[0]
            
    def estimate_parameter_size(self):
        args = self.args
        if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
            self.parameter_size = copy.deepcopy(args.parameter_size)
        else:
            self.parameter_size = args.parameter_size / self.tp_size
        
    def estimate_model_states_size(self):
        args = self.args
        self.model_states_size = 4 * self.parameter_size
        if 'fsdp' in args.strategy[-1].keys() and args.strategy[-1]['fsdp']:
            # fsdp_model_states memory is slightly larger than dp_model_states/dp_size
            # we add a small bias to ensure the predicted fsdp memory NOT smaller than real value
            # Actually, this bias barely affect search result.
            self.model_states_size *= self.zero3_ratio(self.sdp_size)
        elif 'fsdp' in args.strategy[-1].keys() and args.strategy[-1]['fsdp'] == 0 and args.use_zero2_for_dp:
            self.model_states_size *= self.zero2_ratio(self.sdp_size)
        
    def estimate_activation_size(self):
        args = self.args
        if 'cpt' in args.strategy[-1].keys() and args.strategy[-1]['cpt']:
            assert(args.tp_activation_per_bsz_dict['checkpoint'] is not None)
            self.activation_size = args.tp_activation_per_bsz_dict['checkpoint'] * self.bsz
            if args.sequence_parallel:
                self.activation_size /= self.tp_size
        else:
            self.activation_size = args.tp_activation_per_bsz_dict[self.tp_size] * self.bsz
            
    def estimate_other_memcosts(self):
        args = self.args
        
        # [initialize]:initialize total_min_tp
        if args.disable_vtp:
            total_min_tp = [1]
        else:
            total_min_tp, tp = [], copy.deepcopy(args.min_tp)
            while tp * self.pp_size <= args.gpu_num and tp <= args.max_tp_deg:
                total_min_tp.append(tp)
                tp *= 2
        total_min_tp = [tp for tp in total_min_tp  # TODO: consider more cases
            if tp in args.other_memory_pp_off['model_states'].keys() and 
               tp in args.other_memory_pp_on['first_stage']['model_states'] and 
               tp in args.other_memory_pp_on['last_stage']['model_states']]
        
        # [calculate]:calculate other memory costs
        self.other_memcosts = dict()
        for tp in total_min_tp:
            tp_other_memcosts = [0] * self.pp_size
            other_layers_bsz = args.global_batch_size * tp / self.tp_size / self.dp_size
            
            # Determine the memory ratio for Zero optimization
            if args.vsp:
                model_tp = 1
                other_ms_zero2_ratio = self.zero3_ratio(self.tp_size * self.dp_size) if args.use_zero3_for_embed else (self.zero2_ratio(self.tp_size * self.dp_size) if args.use_zero2_for_dp else 1.0)
            else:
                model_tp = tp
                other_ms_zero2_ratio = self.zero3_ratio(self.tp_size * self.dp_size // tp) if args.use_zero3_for_embed else (self.zero2_ratio(self.tp_size * self.dp_size // tp) if args.use_zero2_for_dp else 1.0)
            
            args.model_type = 'gpt' if args.model_type not in ['bert', 't5', 'vit', 'swin', 'gpt'] else args.model_type
            
            # Handle different memory consumption scenarios based on pipeline size (PP Size)
            if self.pp_size == 1:
                tp_other_memcosts[0] += (
                    args.other_memory_pp_off['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    args.other_memory_pp_off['activation'][tp] * 
                    other_layers_bsz * 
                    self.act_1f1b_ratio)
            else:
                if args.pipeline_type == 'pipedream_flush':
                    other_layers_bsz_first = other_layers_bsz * self.act_1f1b_ratio_first
                    other_layers_bsz_last = other_layers_bsz * self.act_1f1b_ratio_last
                else:
                    other_layers_bsz_first = other_layers_bsz_last = other_layers_bsz
                # TODO: check the correctness of other memory cost for first stage and last stage
                tp_other_memcosts[0] += (
                    args.other_memory_pp_on['first_stage']['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    args.other_memory_pp_on['first_stage']['activation'][tp] * 
                    other_layers_bsz_first
                )
                tp_other_memcosts[-1] += (
                    args.other_memory_pp_on['last_stage']['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    args.other_memory_pp_on['last_stage']['activation'][tp] * 
                    other_layers_bsz_last
                )

            # if checkpoint:
            #     for i in range(len(tp_other_memcosts)):
            #         tp_other_memcosts[i] += tp_activation_per_bsz_dict[self.tp_size] * mbsz

            for i in range(len(tp_other_memcosts)):
                tp_other_memcosts[i] += args.pytorch_context_mem
                
            self.other_memcosts[tp] = tp_other_memcosts
    
    def get_memory_cost(self):
        result = dict()
        result['parameter'] = self.parameter_size
        result['model_states'] = self.model_states_size
        result['activation'] = self.activation_size
        result['enc_total'] = self.model_states_size + self.activation_size
        result['other'] = self.other_memcosts
        return result
            
@dataclass
class TimeCostArgs:
    strategy: str  # for example: [pp_deg, tp_deg, dp_deg, {'sp': 1, 'cpt': 1}]
    global_batch_size: int = 8
    parameter_size: float = 48
    microbatch: bool = True
    optimal_chunk_func: Optional[Callable] = None
    sequence_length: int = 512
    hidden_size: int = 1024,
    forward_computation_time: Optional[Union[float, np.ndarray]] = 35 / 24
    bct_fct_coe: float = 2
    extra_overhead: float = 0
    comm_coe_dict: dict = field(default_factory=lambda: {'8': 0.0062326653993580354, '4_0': 0.006042551648710218, '4_1': 0.006087464692704782, '2_0': 0.006496332820123041, '2_1': 0.006424794567193714, '1': 0})
    dp_overlap_coe: float = 1.3
    bct_overlap_coe: float = 1.3
    p2p_comm_coe_dict: dict = field(default_factory=lambda: {2: 0.006787944610371979, 4: 0.0074923765069042254, 8: 0.00920674670398468})
    layer_num: Optional[int] = None
    use_zero2_for_dp: int = 0
    mixed_precision: bool = False
    no_comm: bool = False
    costmodel_coe: float = 1.0
    async_grad_reduce: bool = True
    allreduce_dict: dict = field(default_factory=lambda: {})
    all2all_dict: dict = field(default_factory=lambda: {})
    sp_space: str = 'tp'
    
class TimeCost:
    def __init__(self, **kwargs):
        self.args = TimeCostArgs(**kwargs)
        self._validate_and_correct_args()
        self.initialize()
        self.estimate_computation_time()
        self.estimate_dp_communication_cost()
        self.estimate_tp_communication_cost()
        self.estimate_pp_communication_cost()
        
    def _validate_and_correct_args(self):
        args = self.args
        assert args.microbatch == False
        # correct layer_num. (Dummy layer_num, can be any multiple of 8. We estimate the time cost of single layer by averaging the time of whole model to deal with pipeline parallel)
        args.layer_num = 24 if args.layer_num is None else args.layer_num
    
    def initialize(self):
        args = self.args
        
        # [initialize]:initialize strategy
        self.pp_size = args.strategy[0]
        self.tp_size = args.strategy[1]
        self.dp_size = args.strategy[2]
        self.fsdp = True if 'fsdp' in args.strategy[-1].keys() and args.strategy[-1]['fsdp'] else False
        self.checkpoint = True if 'cpt' in args.strategy[-1].keys() and args.strategy[-1]['cpt'] else False
        if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
            self.sdp_size = self.tp_size * self.dp_size
            if self.tp_size == 1:
                self.sp_dict = np.inf
            else:
                self.sp_dict = args.all2all_dict[self.tp_size]
        else:
            self.sdp_size = self.dp_size
            if self.tp_size == 1:
                self.sp_dict = np.inf
            else:
                self.sp_dict = args.allreduce_dict[self.tp_size]
                
        # [initialize]:initialize local batch size, optimal_microbatch, parameter_size
        self.bsz = args.global_batch_size / self.dp_size
        self.optimal_microbatch = args.optimal_chunk_func(self.bsz, args.strategy) if args.microbatch else 1
        if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
            self.parameter_size = copy.deepcopy(args.parameter_size)
        else:
            self.parameter_size = args.parameter_size / self.tp_size

    def estimate_computation_time(self):
        # forward & backward computation time of whole model (depending on dummy layer_num)
        args = self.args
        if isinstance(args.forward_computation_time, np.ndarray):
            def linear_func(x, m, c):
                return m * x + c
            self.fct = linear_func(self.bsz / self.tp_size, *args.forward_computation_time) * args.layer_num
        else:
            self.fct = args.forward_computation_time * self.bsz / self.tp_size * args.layer_num 
            
        self.bct = self.fct * args.bct_fct_coe
        if self.checkpoint:
            self.bct += self.fct #  * 0.5
  
    def estimate_dp_communication_cost(self):
        args = self.args
        # [calculate]:calculate dp message size of whole model (depending on dummy layer_num)
        self.dp_message_size = (2 * (self.dp_size - 1) / self.dp_size * self.parameter_size) * args.layer_num
        if args.mixed_precision:
            self.dp_message_size /= 2
        if args.no_comm:
            self.dp_message_size = 0
            
        # [calculate]:calculate fsdp_allgather_message_size 
        self.fsdp_allgather_message_size = self.dp_message_size * 0.5
            
        # [calculate]:calculate dc
        if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
            self.dc = args.comm_coe_dict['%d'%self.sdp_size] if '%d'%self.sdp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%self.sdp_size]
        else:
            if self.tp_size == 1 or self.dp_size == 1:
                self.dc = args.comm_coe_dict['%d'%self.dp_size] if '%d'%self.dp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%self.dp_size]
            else:
                # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
                info = args.strategy[-1]
                assert 'tp' in info.keys() and info['tp'] in [0, 1]
                tp_consecutive_flag = info['tp']
                if tp_consecutive_flag:
                    self.dc = args.comm_coe_dict['%d_0'%self.dp_size]
                else:
                    self.dc = args.comm_coe_dict['%d_1'%self.dp_size]
        
        # [calculate]:calculate dc_overlap
        self.dc_overlap = self.dc * args.dp_overlap_coe 
    
    def estimate_tp_communication_cost(self):
        args = self.args
        if args.sp_space == 'tp+sp':
            # [calculate]:calculate tp comm time
            tp_comm_num = 4 * args.layer_num
            if self.checkpoint:
                tp_comm_num *= 1.5
            
            # [calculate]:calculate per_tp_message_time
            if self.tp_size == 1:
                per_tp_message_time = 0
            else:
                per_tp_message_size = self.bsz * args.sequence_length * args.hidden_size * (2 if args.mixed_precision else 4)
                if per_tp_message_size in self.sp_dict:
                    per_tp_message_time = self.sp_dict[per_tp_message_size // self.tp_size]  # NOTE use '/' or '//'?
                else:
                    def linear_func(x, m, c):
                        return m * x + c
                    per_tp_message_time = linear_func(1 / 1024 / 1024 * per_tp_message_size / self.tp_size, *self.sp_dict["popt"])  # NOTE use '/' or '//'?
            
            # [calculate]:calculate tp time
            self.tp_communication_time = tp_comm_num * per_tp_message_time
        else:
            # [calculate]:calculate tp message size of whole model (depending on dummy layer_num)
            tp_comm_num = 4 
            tp_message_size = 2 * (self.tp_size - 1) / self.tp_size * (self.bsz * args.sequence_length * args.hidden_size * tp_comm_num * 4 / 1024 / 1024) * args.layer_num
            if self.checkpoint:
                tp_message_size *= 1.5
            if args.mixed_precision:
                tp_message_size /= 2
            
            # [calculate]:calculate tc
            if 'sp' in args.strategy[-1].keys() and args.strategy[-1]['sp'] == 1:
                if self.tp_size == 1 or self.dp_size == 1:
                    tc = args.comm_coe_dict['%d'%self.tp_size] if '%d'%self.tp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%self.tp_size]
                else:
                    # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
                    info = args.strategy[-1]
                    assert 'tp' in info.keys() and info['tp'] in [0, 1]
                    tp_consecutive_flag = info['tp']
                    if tp_consecutive_flag:
                        tc = args.comm_coe_dict['%d_1'%self.tp_size]
                    else:
                        tc = args.comm_coe_dict['%d_0'%self.tp_size]
            else:
                if self.tp_size == 1 or self.dp_size == 1:
                    tc = args.comm_coe_dict['%d'%self.tp_size] if '%d'%self.tp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%self.tp_size]
                else:
                    # In this case, strategy[-1]['tp'] represents tp_consecutive_flag
                    info = args.strategy[-1]
                    assert 'tp' in info.keys() and info['tp'] in [0, 1]
                    tp_consecutive_flag = info['tp']
                    if tp_consecutive_flag:
                        tc = args.comm_coe_dict['%d_1'%self.tp_size]
                    else:
                        tc = args.comm_coe_dict['%d_0'%self.tp_size]  
                                      
            # [calculate]:calculate tp time
            self.tp_communication_time = tp_message_size * tc
  
    def estimate_pp_communication_cost(self):
        args = self.args
        if self.pp_size > 1 and args.p2p_comm_coe_dict is not None:
            self.p2p_message_size = self.pp_size * 2 * self.bsz * args.sequence_length * args.hidden_size * 4 / 1024 / 1024
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

    def pipe_with_microbatch(self, computation_overhead, communication_overhead):
        result = computation_overhead * (self.pp_size + self.optimal_microbatch - 1) / (self.pp_size * self.optimal_microbatch) + communication_overhead
        return result
    
    def gen_result(self):
        args = self.args
        if self.pp_size >= 1:
            if self.tp_size == 1 and self.dp_size > 1: # pp+dp
                overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                overall_overhead = self.fct + overlap_part + rest_part + args.extra_overhead
                if args.microbatch == False:
                    result = overall_overhead
                else:
                    computation_overhead = self.fct + self.bct
                    communication_overhead = overall_overhead - computation_overhead
                    result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
            elif self.dp_size == 1 and self.tp_size > 1: # pp+tp
                if args.microbatch == False:
                    result = self.fct + self.bct + self.tp_communication_time
                else:
                    overall_overhead = self.fct + self.bct + self.tp_communication_time
                    result = self.pipe_with_microbatch(overall_overhead, 0)
            elif self.dp_size == 1 and self.tp_size == 1: # pure pp
                if args.microbatch == False:
                    result = self.fct + self.bct
                else:
                    overall_overhead = self.fct + self.bct
                    result = self.pipe_with_microbatch(overall_overhead, 0)
            else: # pp+dp+tp
                if self.tp_size < self.tp_size * self.dp_size // 2:
                    overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct)
                    overall_overhead = self.fct + overlap_part + rest_part + self.tp_communication_time + args.extra_overhead
                    if args.microbatch == False:
                        result = overall_overhead
                    else:
                        computation_overhead = self.fct + self.bct + self.tp_communication_time
                        communication_overhead = overall_overhead - computation_overhead
                        result = self.pipe_with_microbatch(computation_overhead, communication_overhead)
                else:
                    overlap_part, rest_part, _ = self.bct_dp_overlap(self.dp_message_size, self.bct * 1 / 2)
                    overall_overhead = self.fct + 1 / 2 * self.bct + overlap_part + rest_part + self.tp_communication_time + args.extra_overhead
                    if args.microbatch == False:
                        result = overall_overhead
                    else:
                        computation_overhead = self.fct + self.bct + self.tp_communication_time
                        communication_overhead = overall_overhead - computation_overhead
                        result = self.pipe_with_microbatch(computation_overhead, communication_overhead)

        # For fsdp, add allgather time of forward and backward
        if self.fsdp:
            forward_allgather_time = self.fsdp_allgather_message_size * self.dc
            result = result + forward_allgather_time * self.optimal_microbatch

        if self.pp_size > 1 and args.p2p_comm_coe_dict is not None:
            result = result + self.p2p_message_size * args.p2p_comm_coe_dict[self.pp_size]
        
        coe = 0.001 * args.costmodel_coe
        result = result * coe
        result = result / args.layer_num
        return result
  
@dataclass
class OtherTimeCostArgs:
    mbsz: int = 1
    pp_deg: int = 2
    world_size: int = 8
    sequence_length: int = 2048
    hidden_size: int = 4096
    mixed_precision: bool = False
    comm_coe_dict: dict = field(default_factory=lambda: {})
    allreduce_dict: dict = field(default_factory=lambda: {})
    sp_space: str = 'tp+sp'
    vsp: int = 0
    min_tp: int = 1
    max_tp: int = 8
    other_memory_pp_on: dict = field(default_factory=lambda: {})
    other_memory_pp_off: dict = field(default_factory=lambda: {})
    other_time_profiled_list: list = field(default_factory=lambda: [])
    
class OtherTimeCost:
    def __init__(self, **kwargs):
        self.args = OtherTimeCostArgs(**kwargs)
        self.dp_coe = dict()
        self.fct = dict()
        self.tp_time = dict()
        self.sp_size = dict()
        self.dp_size = dict()
        self.comm_factor = dict()
        
        args = self.args
        # calc tp comm size
        k = args.min_tp
        while k <=  args.max_tp and args.world_size // args.pp_deg >= k:
            if args.vsp == 0:
                if args.sp_space == 'tp+sp':
                    self.per_tp_message_size = args.mbsz * args.sequence_length * args.hidden_size * (2 if args.mixed_precision else 4)
                    if k == 1:
                        self.per_tp_message_time = 0
                    else:
                        if self.per_tp_message_size in args.allreduce_dict:
                            self.per_tp_message_time = args.allreduce_dict[self.per_tp_message_size]
                        else:
                            def linear_func(x, m, c):
                                return m * x + c
                            self.per_tp_message_time = linear_func( 1 / 1024 / 1024 * self.per_tp_message_size, *args.allreduce_dict[k]["popt"] )
                else:
                    dp_size = args.world_size // args.pp_deg // k
                    if k == 1 or dp_size == 1:
                        tp_coe = args.comm_coe_dict['%d'%k] if '%d'%k in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%k]
                    else:
                        tp_coe = args.comm_coe_dict['%d_0'%k]

                    self.tp_message_size = (k - 1) / k * (args.mbsz * args.sequence_length * args.hidden_size / 1024 / 1024) * (2 if args.mixed_precision else 4)
                    self.per_tp_message_time = self.tp_message_size * tp_coe
            else:
                self.per_tp_message_time = 0
            if args.pp_deg == 1:
                self.tp_time[k] = 2 * self.per_tp_message_time
            else:
                self.tp_time[k] = (self.per_tp_message_time, self.per_tp_message_time)
            k *= 2
            
        # calc calc time (ms)
        k = args.min_tp
        while k <= args.max_tp and args.world_size // args.pp_deg >= k:
            def linear_func(x, m, c):
                return m * x + c
            if args.pp_deg == 1:
                if isinstance(args.other_time_profiled_list ,np.ndarray):
                    self.fct[k] = linear_func(args.mbsz / args.min_tp, *args.other_time_profiled_list)
                else:
                    self.fct[k] = args.mbsz / args.min_tp * k * args.other_time_profiled_list
            else:
                if isinstance(args.other_time_profiled_list, np.ndarray):
                    self.fct[k] = (linear_func(args.mbsz / args.min_tp, *args.other_time_profiled_list) / 2, \
                                linear_func(args.mbsz / args.min_tp, *args.other_time_profiled_list) / 2)
                else:
                    self.fct[k] = (args.mbsz / args.min_tp * k * args.other_time_profiled_list / 2, \
                                args.mbsz / args.min_tp * k * args.other_time_profiled_list / 2)
            k *= 2
            
        # calc dp comm size
        k = args.min_tp
        while k <= args.max_tp and args.world_size //args.pp_deg >= k:
            if args.vsp == 0:
                dp_size = args.world_size // args.pp_deg // k
                if k == 1 or dp_size == 1:
                    self.dp_coe[k] = args.comm_coe_dict['%d'%dp_size] if '%d'%dp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%dp_size]
                else:
                    self.dp_coe[k] = args.comm_coe_dict['%d_0'%dp_size]
                self.dp_coe[k] *= (dp_size - 1) / dp_size # bus -> alg
            else:
                dp_size = args.world_size // args.pp_deg
                self.dp_coe[k] = args.comm_coe_dict['%d'%dp_size] if '%d'%dp_size in args.comm_coe_dict.keys() else args.comm_coe_dict['%d_1'%dp_size]
                self.dp_coe[k] *= (dp_size - 1) / dp_size # bus -> alg
            if args.pp_deg == 1:
                if args.vsp == 0:
                    self.dp_size[k] = args.other_memory_pp_off['model_states'][k] / 4
                else:
                    self.dp_size[k] = args.other_memory_pp_off['model_states'][1] / 4
            else:
                if args.vsp == 0:
                    self.dp_size[k] = (args.other_memory_pp_on['first_stage']['model_states'][k] / 4, args.other_memory_pp_on['first_stage']['model_states'][k] / 4)
                else:
                    self.dp_size[k] = (args.other_memory_pp_on['last_stage']['model_states'][1] / 4, args.other_memory_pp_on['last_stage']['model_states'][1] / 4)
            k *= 2

    def gen_result(self):
        args = self.args
        other_time_cost = dict()
        k = args.min_tp
        for k in self.dp_size.keys():
            other_time_cost[k] = [0] * args.pp_deg 
            if args.pp_deg  == 1:
                other_time_cost[k][0] = 0.001 * (self.dp_size[k] * self.dp_coe[k] + self.fct[k] * 3 + self.tp_time[k]) # + 4 * self.sp_time[k] # fwd + bwd
            else:
                other_time_cost[k][0] = 0.001 * (self.dp_size[k][0] * self.dp_coe[k] + self.fct[k][0] * 3 + self.tp_time[k][0]) # + 2 * self.sp_time[k]
                other_time_cost[k][-1] = 0.001 * (self.dp_size[k][-1] * self.dp_coe[k] + self.fct[k][-1] * 3 + self.tp_time[k][-1]) # + 2 * self.sp_time[k]
        return other_time_cost