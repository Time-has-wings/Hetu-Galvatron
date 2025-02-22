import numpy as np
import copy
from typing import Optional, Callable, Union
from dataclasses import dataclass, field

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

@dataclass
class ParallelConfigArgs:
    strategy: list = field(default_factory=lambda: [1, 2, 4])  # for example: [pp_deg, tp_deg, dp_deg, {'sp': 1, 'cpt': 1}]
    sequence_parallel: bool = True
    pipeline_type: str = 'gpipe'
    use_zero2_for_dp: int = 0
    use_zero3_for_embed: int = 0
    max_tp_deg: int = 8
    vsp: int = 0 
    disable_vtp: bool = False
    
@dataclass 
class ModelConfigArgs:
    parameter_size: float = 48
    model_type: str = 'bert'
    
@dataclass
class TrainConfigArgs:
    global_batch_size: int = 8
    mixed_precision: bool = False 
    async_grad_reduce: bool = True
    
@dataclass
class DynamicConfigArgs:
    optimal_chunk_func: Optional[Callable] = None
    stage_idx: int = 0
    mbsz: int = -1
    min_tp: int = -1
    chunks: Optional[int] = None
    
@dataclass 
class HardwareConfigArgs:
    tp_activation_per_bsz_dict: dict = field(default_factory=lambda: {1: 85, 2: 47, 4: 28, 'checkpoint': 28.0})
    other_memory_pp_off: dict = field(default_factory=lambda: {'model_states': {1: 16, 2:83, 4:41}, 'activation': {1: 28, 2: 11, 4: 5}}) 
    other_memory_pp_on: dict = field(default_factory=lambda: {'first_stage': {'model_states': {1: 83, 2: 41, 4: 20}, 'activation': {1: 36, 2: 11, 4: 8}}, 'last_stage': {'model_states': {1: 83, 2: 41, 4: 20}, 'activation': {1: 23, 2: 11, 4: 5}}})
    pytorch_context_mem: float = 1024

class MemoryCost:
    def __init__(self, pca:ParallelConfigArgs, tca:TrainConfigArgs, dca:DynamicConfigArgs, mca:ModelConfigArgs, hca:HardwareConfigArgs):
        self.pca, self.tca, self.dca, self.mca, self.hca = copy.deepcopy(pca), copy.deepcopy(tca), copy.deepcopy(dca), copy.deepcopy(mca), copy.deepcopy(hca)  # TODO: 先进行深拷贝,之后再来看之后有bug
        self.initialize()
        self.estimate_parameter_size()
        self.estimate_model_states_size()
        self.estimate_activation_size()
        self.estimate_other_memory_cost()
    
    def _validate(self):
        assert self.dca.mbsz > -1, f'Invalid mbsz: {self.dca.mbsz}'
        assert self.dca.min_tp > -1, f'Invalid min_tp: {self.dca.min_tp}'
        
    def initialize(self):
        pca, tca, dca = self.pca, self.tca, self.dca
        
        # [initialize]:initialize strategy
        self.pp_size = pca.strategy[0]
        self.tp_size = pca.strategy[1]
        self.dp_size = pca.strategy[2]
        if 'sp' in pca.strategy[-1].keys() and pca.strategy[-1]['sp'] == 1:
            self.sdp_size = self.tp_size * self.dp_size
        else:
            self.sdp_size = self.dp_size
    
        # [adjust]:adjust args.chunks
        # TODO: args.chunks其实可以直接在输入时就进行确定,而不是在本函数内部进行确定,如此可以避免optimal_chunk_func函数的调用,以及args,mbsz的输入,减少函数参数
        if dca.chunks is None:
            dca.chunks = dca.optimal_chunk_func(tca.global_batch_size // self.dp_size, pca.strategy, dca.mbsz, dca.min_tp)  # FIXME:没有看懂这个函数的意义 以及这个函数的参数
        max_chunks = tca.global_batch_size // (self.tp_size * self.dp_size // dca.min_tp)
        max_chunks = 1 if max_chunks == 0 else max_chunks
        dca.chunks = max_chunks if dca.chunks > max_chunks else dca.chunks
        dca.chunks = int(dca.chunks)
        
        # [initialize]:initialize zero2 and zero3 ratio
        if dca.chunks == 1:
            self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if tca.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            self.zero3_ratio = lambda d: (1/d + 0.003)
        else:
            if tca.async_grad_reduce:
                self.zero2_ratio = (lambda d: (6/8 * (1/d + 0.003) + 2/8)) if tca.mixed_precision else (lambda d: (2/4 * (1/d + 0.003) + 2/4))
                self.zero3_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if tca.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            else:
                self.zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8) * 5/4) if tca.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
                self.zero3_ratio = lambda d: (1/d + 0.003) * 5/4
                # *5/4: for fp32 grad 
        
        # [initialize]:initialize local batch size and pp stage act_1f1b_ratio
        self.bsz = tca.global_batch_size / self.dp_size
        if (pca.pipeline_type == 'pipedream_flush' and self.pp_size > 1) or self.pp_size == 1:
            microbatches = [t.shape[0] for t in chunk_like_torch(int(tca.global_batch_size / self.dp_size / (self.tp_size // dca.min_tp)), dca.chunks)]
            assert dca.chunks == len(microbatches)
            end = self.pp_size - dca.stage_idx if self.pp_size - dca.stage_idx <= dca.chunks else dca.chunks
            self.act_1f1b_ratio = np.sum(microbatches[:end]) / np.sum(microbatches)
            self.act_1f1b_ratio_first = np.sum(microbatches[:min(self.pp_size, dca.chunks)]) / np.sum(microbatches)
            self.act_1f1b_ratio_last = microbatches[0] / np.sum(microbatches)
            self.bsz = self.act_1f1b_ratio * self.bsz
        else:
            microbatches = [t.shape[0] for t in chunk_like_torch(int(tca.global_batch_size / self.dp_size / (self.tp_size // dca.min_tp)), dca.chunks)]
            self.bsz = microbatches[0]
            
    def estimate_parameter_size(self):
        pca, mca = self.pca, self.mca
        if 'sp' in pca.strategy[-1].keys() and pca.strategy[-1]['sp'] == 1:
            self.parameter_size = copy.deepcopy(mca.parameter_size)
        else:
            self.parameter_size = mca.parameter_size / self.tp_size
        
    def estimate_model_states_size(self):
        pca = self.pca
        self.model_states_size = 4 * self.parameter_size
        if 'fsdp' in pca.strategy[-1].keys() and pca.strategy[-1]['fsdp']:
            # fsdp_model_states memory is slightly larger than dp_model_states/dp_size
            # we add a small bias to ensure the predicted fsdp memory NOT smaller than real value
            # Actually, this bias barely affect search result.
            self.model_states_size *= self.zero3_ratio(self.sdp_size)
        elif 'fsdp' in pca.strategy[-1].keys() and pca.strategy[-1]['fsdp'] == 0 and pca.use_zero2_for_dp:
            self.model_states_size *= self.zero2_ratio(self.sdp_size)
        
    def estimate_activation_size(self):
        pca, hca = self.pca, self.hca
        if 'cpt' in pca.strategy[-1].keys() and pca.strategy[-1]['cpt']:
            assert(hca.tp_activation_per_bsz_dict['checkpoint'] is not None)
            self.activation_size = hca.tp_activation_per_bsz_dict['checkpoint'] * self.bsz
            if pca.sequence_parallel:
                self.activation_size /= self.tp_size
        else:
            self.activation_size = hca.tp_activation_per_bsz_dict[self.tp_size] * self.bsz
    
    def estimate_other_memory_cost(self):
        pca, tca, dca, mca, hca = self.pca, self.tca, self.dca, self.mca, self.hca
        
        # [initialize]:initialize total_min_tp
        if pca.disable_vtp:
            total_min_tp = [1]
        else:
            total_min_tp, tp = [], copy.deepcopy(dca.min_tp)
            gpu_num = pca.strategy[0] * pca.strategy[1] * pca.strategy[2]
            while tp * self.pp_size <= gpu_num and tp <= pca.max_tp_deg:
                total_min_tp.append(tp)
                tp *= 2
        total_min_tp = [tp for tp in total_min_tp  # TODO: consider more cases
            if tp in hca.other_memory_pp_off['model_states'].keys() and 
               tp in hca.other_memory_pp_on['first_stage']['model_states'] and 
               tp in hca.other_memory_pp_on['last_stage']['model_states']]
        
        # [calculate]:calculate other memory costs
        self.other_memory_cost = dict()
        for tp in total_min_tp:
            tp_other_memory_cost = [0] * self.pp_size
            other_layers_bsz = tca.global_batch_size * tp / self.tp_size / self.dp_size
            
            # Determine the memory ratio for Zero optimization
            if pca.vsp:
                model_tp = 1
                other_ms_zero2_ratio = self.zero3_ratio(self.tp_size * self.dp_size) if pca.use_zero3_for_embed else (self.zero2_ratio(self.tp_size * self.dp_size) if pca.use_zero2_for_dp else 1.0)
            else:
                model_tp = tp
                other_ms_zero2_ratio = self.zero3_ratio(self.tp_size * self.dp_size // tp) if pca.use_zero3_for_embed else (self.zero2_ratio(self.tp_size * self.dp_size // tp) if pca.use_zero2_for_dp else 1.0)
            
            mca.model_type = 'gpt' if mca.model_type not in ['bert', 't5', 'vit', 'swin', 'gpt'] else mca.model_type
            
            # Handle different memory consumption scenarios based on pipeline size (PP Size)
            if self.pp_size == 1:
                tp_other_memory_cost[0] += (
                    hca.other_memory_pp_off['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    hca.other_memory_pp_off['activation'][tp] * 
                    other_layers_bsz * 
                    self.act_1f1b_ratio)
            else:
                if pca.pipeline_type == 'pipedream_flush':
                    other_layers_bsz_first = other_layers_bsz * self.act_1f1b_ratio_first
                    other_layers_bsz_last = other_layers_bsz * self.act_1f1b_ratio_last
                else:
                    other_layers_bsz_first = other_layers_bsz_last = other_layers_bsz
                # TODO: check the correctness of other memory cost for first stage and last stage
                tp_other_memory_cost[0] += (
                    hca.other_memory_pp_on['first_stage']['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    hca.other_memory_pp_on['first_stage']['activation'][tp] * 
                    other_layers_bsz_first
                )
                tp_other_memory_cost[-1] += (
                    hca.other_memory_pp_on['last_stage']['model_states'][model_tp] * 
                    other_ms_zero2_ratio + 
                    hca.other_memory_pp_on['last_stage']['activation'][tp] * 
                    other_layers_bsz_last
                )

            # if checkpoint:
            #     for i in range(len(tp_other_memory_cost)):
            #         tp_other_memory_cost[i] += tp_activation_per_bsz_dict[self.tp_size] * mbsz

            for i in range(len(tp_other_memory_cost)):
                tp_other_memory_cost[i] += hca.pytorch_context_mem
                
            self.other_memory_cost[tp] = tp_other_memory_cost
    
    def get_memory_cost(self):
        result = dict()
        result['parameter'] = self.parameter_size
        result['model_states'] = self.model_states_size
        result['activation'] = self.activation_size
        result['enc_total'] = self.model_states_size + self.activation_size
        result['other'] = self.other_memory_cost
        return result