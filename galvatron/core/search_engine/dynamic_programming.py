import math
import copy
import numpy as np
from typing import List, Any

from galvatron.core.cost_model.components.layer_cost import TimeCostModelBase, MemoryCostModelBase
from galvatron.core.cost_model.components.embedding_lmhead_cost import EmbeddingLMHeadTimeCostModel, EmbeddingLMHeadMemoryCostModel
from galvatron.utils.strategy_utils import EmbeddingLMHeadStrategy, LayerStrategy, DPType, print_strategy_list
from galvatron.core.cost_model.cost_model_handler import pipeline_costmodel

class DPAlg():
    def __init__(self, max_mem=8200, other_mem_cost=None, other_time_cost = None, layer_num=24, layer_strategy_num=4, strategy_set=None, fine_grained_mode=True, use_cpp_core=True) -> None:
        assert(other_mem_cost != None)
        self.max_mem = max_mem + 1
        self.layer_num = layer_num
        self.layer_strategy_num = layer_strategy_num
        self.other_mem_cost = other_mem_cost
        self.other_time_cost = other_time_cost

        self._f = np.full((self.max_mem, layer_strategy_num), 0, dtype=np.float64)
        
        self.v_data = None
        self.inter_cost = None
        self.intra_cost = None

        self._mark = np.full((layer_num, self.max_mem, layer_strategy_num), -1, dtype=np.int32)
        self.use_cpp_core = use_cpp_core
        self.strategy_set = strategy_set
        self.fine_grained_mode = fine_grained_mode
    
    def set_v_and_cost(self, v: np.ndarray, intra_layer_cost: np.ndarray, inter_layer_cost: np.ndarray):
        assert v.ndim == 2
        assert inter_layer_cost.ndim == 3
        assert intra_layer_cost.ndim == 2

        assert v.shape[0] == self.layer_num
        assert v.shape[1] == self.layer_strategy_num

        assert inter_layer_cost.shape[0] == self.layer_num
        assert inter_layer_cost.shape[1] == self.layer_strategy_num and inter_layer_cost.shape[2] == self.layer_strategy_num

        assert intra_layer_cost.shape[0] == self.layer_num
        assert intra_layer_cost.shape[1] == self.layer_strategy_num

        self.v_data = v.astype(np.int32)
        self.inter_cost = inter_layer_cost
        self.intra_cost = intra_layer_cost

    def fit(self):
        # if not self.fine_grained_mode:
        #     res_list = {k:np.full((self.layer_num), -1, dtype=np.int32) for k,v in self.other_mem_cost.items()}
        #     total_cost = {k:np.inf for k,v in self.other_mem_cost.items()}
        #     remaining_mem = {k:-1 for k,v in self.other_mem_cost.items()}
        #     for k,v in self.other_mem_cost.items():
        #         for i in range(self.layer_strategy_num):
        #             if self.strategy_set[i][1]==k:
        #                 time_cost = sum(self.intra_cost[:,i]) + sum(self.inter_cost[:,i,i]) + self.other_time_cost[k]
        #                 mem_cost = sum(self.v_data[:,i]) + self.other_mem_cost[k]
        #                 if self.max_mem - 1 - mem_cost >= 0 and total_cost[k] > time_cost:
        #                     remaining_mem[k] = self.max_mem - 1 - mem_cost
        #                     total_cost[k] = time_cost
        #                     res_list[k] = np.full((self.layer_num), i, dtype=np.int32)
        #     return total_cost, res_list, remaining_mem       
        if self.use_cpp_core:
            import galvatron_dp_core
            res_list = {k:np.full((self.layer_num), -1, dtype=np.int32) for k,v in self.other_mem_cost.items()}
            total_cost, remaining_mem = galvatron_dp_core.dynamic_programming_core(
                self.layer_num, 
                self.max_mem, 
                self.layer_strategy_num, 
                self.v_data, 
                self._mark, 
                self._f, 
                self.inter_cost, 
                self.intra_cost,
                self.other_mem_cost,
                self.other_time_cost,
                res_list,
                )
            res_list = {k:list(v) for k,v in res_list.items()}

            return total_cost, res_list, remaining_mem

        for i in range(self.layer_num):
            for v in range(self.max_mem - 1, -1, -1):
                for s in range(self.layer_strategy_num):

                    if v < self.v_data[i, s]:
                        self._mark[i, v, s] = -1
                        self._f[v, s] = np.inf
                        continue

                    candidates = [self._f[v - self.v_data[i, s], si] + self.inter_cost[i, si, s] for si in range(self.layer_strategy_num)]
                    candidates = np.array(candidates) + self.intra_cost[i, s]

                    min_index = np.argmin(candidates)

                    self._mark[i, v, s] = min_index
                    self._f[v, s] = candidates[min_index]
        
        next_index, next_v = np.argmin(self._f[-1, :]), self.max_mem - 1
        total_cost = self._f[-1, next_index]

        if not total_cost < np.inf:
            return np.inf, None, -1

        res_list = [-1] * self.layer_num
        res_list[-1] = next_index

        for i in range(self.layer_num - 1, 0, -1):
            next_index, next_v = self._mark[i, next_v, next_index], next_v - self.v_data[i, next_index]
            res_list[i - 1] = next_index

        return total_cost, res_list, next_v - self.v_data[0, next_index]

class DpOnModel:
    def __init__(   
        self, 
        model_args_list = None,
        train_args_list = None,
        parallel_args_list = None,
        profile_model_args_list = None,
        profile_hardware_args_list = None,
        max_mem = 8192, 
        layer_num = [24],
        sequence_len = [512],
        comm_coe_dict = {},
        world_size = 8,
        mem_cache = True,
        pipeline_type = 'gpipe',
        config = None,
        logger = None
    ):
        assert(isinstance(layer_num, list))
        assert(isinstance(model_args_list, list) and len(layer_num) == len(model_args_list))
        assert(isinstance(train_args_list, list) and len(layer_num) == len(train_args_list))
        assert(isinstance(parallel_args_list, list) and len(layer_num) == len(parallel_args_list))
        assert(isinstance(profile_model_args_list, list) and len(layer_num) == len(profile_model_args_list))
        assert(isinstance(profile_hardware_args_list, list) and len(layer_num) == len(profile_hardware_args_list))

        self.model_args_list = model_args_list
        self.train_args_list = train_args_list
        self.parallel_args_list = parallel_args_list
        self.profile_model_args_list = profile_model_args_list
        self.profile_hardware_args_list = profile_hardware_args_list
        self.max_mem = max_mem
        self.layer_num = layer_num
        self.sequence_len = sequence_len
        self.comm_coe_dict = comm_coe_dict
        self.config = config
        self.logger = logger
        self.world_size = world_size
        self.mem_cache = 0
        if max_mem // 1024 > 20 and mem_cache:
            self.mem_cache = int(max_mem * 0.2) # reserved memory for pytorch memory cache
            self.mem_sub_cache = self.max_mem - self.mem_cache
            self.max_mem -= self.mem_cache
        self.pipeline_type = pipeline_type
    
    def match_strategy(self, former:LayerStrategy, latter:LayerStrategy, diff_keys=[]):
        diff_keys = sorted(diff_keys)

        def is_all_key_same(keys):
            for key in keys:
                if key == 'pp_size' and former.pp_size != latter.pp_size:
                    return False
                if key == 'tp_sp_size' and former.tp_sp_size != latter.tp_sp_size:
                    return False
                if key == 'dp_size' and former.dp_size != latter.dp_size:
                    return False
                if key == 'checkpoint' and former.checkpoint != latter.checkpoint:
                    return False
                if key == 'dp_type' and former.dp_type != latter.dp_type:
                    return False
                if key == 'sp_size' and former.sp_size != latter.sp_size:
                    return False
                if key == 'tp_size' and former.tp_size != latter.tp_size:
                    return False
            return True

        if diff_keys == sorted(['sp']):
            must_be_same_keys = ['pp_size', 'tp_sp_size', 'dp_size', 'checkpoint', 'dp_type']
            if not is_all_key_same(must_be_same_keys):
                return False
            cannot_be_exactly_same_keys = ['sp_size']
            if is_all_key_same(cannot_be_exactly_same_keys):
                return False
        elif diff_keys == sorted(['fsdp']):
            must_be_same_keys = ['pp_size', 'tp_size', 'sp_size',  'dp_size', 'checkpoint']
            if not is_all_key_same(must_be_same_keys):
                return False
            cannot_be_exactly_same_keys = ['dp_type']
            if is_all_key_same(cannot_be_exactly_same_keys):
                return False
        elif diff_keys == sorted(['cpt']):
            must_be_same_keys = ['pp_size', 'tp_size', 'sp_size', 'dp_size', 'dp_type']
            if not is_all_key_same(must_be_same_keys):
                return False
            cannot_be_exactly_same_keys = ['checkpoint']
            if is_all_key_same(cannot_be_exactly_same_keys):
                return False
        elif diff_keys == sorted(['fsdp', 'cpt']):
            must_be_same_keys = ['pp_size', 'tp_size', 'sp_size', 'dp_size']
            if not is_all_key_same(must_be_same_keys):
                return False
            cannot_be_exactly_same_keys = ['dp_type', 'checkpoint']
            if is_all_key_same(cannot_be_exactly_same_keys):
                return False
        return True
    
    def _build_dp_and_run_multi_layer_type(
        self, 
        gbsz:int,
        chunks:int,
        pp_size:int,
        pp_stage_list:list[int],
        global_buffer_tp_size:int,
        tp_sp_mode:str,
    ) -> dict[str, Any]:
        # [Step 1] Preparation Works
        num_layertype = len(self.layer_num)
        total_layer_num = sum(self.layer_num)

        assert self.input_layer_strategy_list is not None and self.input_embedding_lmhead_strategy_list is not None
        layer_strategy_list = self.input_layer_strategy_list
        embedding_lmhead_strategy_list = self.input_embedding_lmhead_strategy_list
        embedding_lmhead_strategy_list = sorted(embedding_lmhead_strategy_list)  # Sort for easier debugging
        layer_strategy_num = len(layer_strategy_list)

        # [Step 2] Calculate some extra memory cost
        if self.config.sequence_parallel and self.config.global_memory_buffer and tp_sp_mode != 'sp_only':
            cur_dp = self.world_size // pp_size // global_buffer_tp_size
            cur_lbsz = gbsz / chunks / cur_dp
            global_memory = cur_lbsz * self.config.hidden_size * max(self.sequence_len) * 4 / 1024 / 1024
            if self.config.mixed_precision:
                global_memory = global_memory / 2
        else:
            global_memory = 0
        # if tp_sp_mode != 'tp_only:
        #     global_memory += 8192 # reserved memory for efficient all2all communication
        
        if self.config.fine_grained_mode == 0:
            # [Step 3] Solve the coarse-grained parallel strategy
            # [Step 3.1] Initialize the optimal solution
            optimal = {
                'time_cost': np.inf,
                'memory_used': [-1 for _ in range(pp_size)],
                'memory_remain': [-1 for _ in range(pp_size)],
                'strategy_list': None,
                'embedding_lmhead_tp_sp_size': -1,
                'embedding_lmhead_sp': -1,
                'embedding_lmhead_sdp': -1,
                'pp_size': pp_size,
            }
            # [Step 3.2] Solve the coarse-grained parallel strategy for each layer strategy
            for layer_strategy_idx, layer_strategy in enumerate(layer_strategy_list):
                embedding_lmhead_strategy = layer_strategy.to_embedding_lmhead_strategy()

                # [Step 3.2.1] Calculate the embedding_lmhead time cost
                embedding_lmhead_time_cost_obj = EmbeddingLMHeadTimeCostModel(
                    strategy=embedding_lmhead_strategy,
                    global_batch_size=gbsz,
                    chunks=chunks,
                    sequence_length_list=self.sequence_len,
                    model_args=self.model_args_list[0],
                    train_args=self.train_args_list[0],
                    parallel_args=self.parallel_args_list[0],
                    profile_model_args=self.profile_model_args_list[0],
                    profile_hardware_args=self.profile_hardware_args_list[0],
                    logger=self.logger
                )
                _, embedding_lmhead_time_cost_no_grad_sync = embedding_lmhead_time_cost_obj.gen_result() # embedding_lmhead_time_cost: List[float], embedding_lmhead_time_cost_no_grad_sync: List[float]
                
                # [Step 3.2.2] Calculate the embedding_lmhead memory cost
                embedding_lmhead_memory_cost_obj = EmbeddingLMHeadMemoryCostModel(
                    strategy=embedding_lmhead_strategy,
                    global_batch_size=gbsz,
                    chunks=chunks,
                    logger=self.logger,
                    model_args=self.model_args_list[0],
                    train_args=self.train_args_list[0],
                    parallel_args=self.parallel_args_list[0],
                    profile_model_args=self.profile_model_args_list[0],
                )
                embedding_lmhead_memory_cost = embedding_lmhead_memory_cost_obj.get_memory_cost()
                embedding_lmhead_memory_cost = embedding_lmhead_memory_cost['enc_total']

                # [Step 3.2.3] Calculate the layer memory cost
                layer_memory_cost_dict = {key:[] for key in range(pp_size)} # key:stage_idx, value: List[int]]
                for stage_idx in range(pp_size):
                    for layertype_idx in range(num_layertype):
                        layer_memory_cost_obj = MemoryCostModelBase(
                            strategy=layer_strategy,
                            global_batch_size=gbsz,
                            chunks=chunks,
                            stage_idx=stage_idx,
                            logger=self.logger,
                            model_args=self.model_args_list[layertype_idx],
                            train_args=self.train_args_list[layertype_idx],
                            parallel_args=self.parallel_args_list[layertype_idx],
                            profile_model_args=self.profile_model_args_list[layertype_idx],
                        )
                        layer_memory_cost = layer_memory_cost_obj.get_memory_cost()
                        layer_memory_cost = layer_memory_cost['enc_total']
                        layer_memory_cost_dict[stage_idx].extend([layer_memory_cost for _ in range(self.layer_num[layertype_idx])])
                
                # [Step 3.2.4] Calculate the memory cost for each strategy and check if it is out of memory
                strategy_OOM = False
                memory_used = [0 for _ in range(pp_size)]
                memory_remain = [0 for _ in range(pp_size)]

                start_layer = 0
                for stage_idx in range(pp_size):
                    used = 0
                    used += math.ceil(global_memory)
                    used += math.ceil(embedding_lmhead_memory_cost[stage_idx])
                    for layer_idx in range(start_layer, start_layer + pp_stage_list[stage_idx]):
                        used += math.ceil(layer_memory_cost_dict[stage_idx][layer_idx])
                    memory_used[stage_idx] = used
                    start_layer += pp_stage_list[stage_idx]

                    if used > self.mem_sub_cache:
                        strategy_OOM = True
                        break

                # [Step 3.2.5] Calculate the pipeline cost
                if not strategy_OOM:
                    memory_remain = [self.mem_sub_cache - memory_used[i] for i in range(pp_size)]
                    memory_used = [item + self.mem_cache for item in memory_used]
                    strategy_list = [layer_strategy for _ in range(total_layer_num)]
                    pipeline_cost = pipeline_costmodel(
                        layer_num_list=self.layer_num,
                        model_args_list=self.model_args_list,
                        train_args_list=self.train_args_list,
                        parallel_args_list=self.parallel_args_list,
                        profile_model_args_list=self.profile_model_args_list,
                        profile_hardware_args_list=self.profile_hardware_args_list,
                        strategy_list=strategy_list,
                        partition=pp_stage_list,
                        chunks=chunks,
                        pp_size=pp_size,
                        gbsz=gbsz,
                        other_time_cost=embedding_lmhead_time_cost_no_grad_sync, # TODO: check this
                        logger=self.logger,
                        return_stage_cost=False
                    )
                    if optimal['time_cost'] > pipeline_cost:
                        optimal['time_cost'] = pipeline_cost
                        optimal['memory_used'] = copy.deepcopy(memory_used)
                        optimal['memory_remain'] = copy.deepcopy(memory_remain)
                        optimal['strategy_list'] = copy.deepcopy(strategy_list)
                        optimal['embedding_lmhead_tp_sp_size'] = embedding_lmhead_strategy.tp_sp_size
                        optimal['embedding_lmhead_sp'] = 1 if embedding_lmhead_strategy.sp_size > 1 else 0
                        optimal['embedding_lmhead_sdp'] = 1 if embedding_lmhead_strategy.dp_type == DPType.ZERO3 else 0
                    self.log(f'layer_strategy_idx: {layer_strategy_idx}, strategy: {layer_strategy}, pipeline_cost: {pipeline_cost}, memory_used: {memory_used}, memory_remain: {memory_remain}')
                else:
                    self.log(f'layer_strategy_idx: {layer_strategy_idx}, strategy: {layer_strategy}, strategy_OOM')

            return optimal
        else:
            # [Step 3] Calculate the intra layer cost
            # intra_layer_cost: dtype:np.float64 shape:(total_layer_num, layer_strategy_num)
            intra_layer_cost = np.zeros((sum(self.layer_num), layer_strategy_num))
            for layertype_idx in range(num_layertype):
                all_strategy_time_cost:List[float] = []
                for layer_strategy in layer_strategy_list:
                    obj = TimeCostModelBase(
                        strategy=layer_strategy,
                        global_batch_size=gbsz,
                        chunks=chunks,
                        model_args=self.model_args_list[layertype_idx],
                        train_args=self.train_args_list[layertype_idx],
                        parallel_args=self.parallel_args_list[layertype_idx],
                        profile_model_args=self.profile_model_args_list[layertype_idx],
                        profile_hardware_args=self.profile_hardware_args_list[layertype_idx],
                        logger=self.logger,
                    )
                    res_with_grad_sync, _ = obj.gen_result()
                    all_strategy_time_cost.append(res_with_grad_sync)
                intra_layer_cost[sum(self.layer_num[:layertype_idx]):sum(self.layer_num[:layertype_idx+1]), :] = np.array(all_strategy_time_cost, dtype=np.float64).reshape(1, -1).repeat(self.layer_num[layertype_idx], axis=0)
            
            # [Step 4] Calculate embedding_lmhead time cost
            # embedding_lmhead_time_cost: dict[int, tuple[float, float]]
            # key: embedding_lmhead_strategy_idx
            # value: (time_with_grad_sync, time_without_grad_sync)
            embedding_lmhead_time_cost = {} # dict[int, tuple[float, float]]
            for embedding_lmhead_strategy_idx, embedding_lmhead_strategy in enumerate(embedding_lmhead_strategy_list):
                obj = EmbeddingLMHeadTimeCostModel(
                    strategy=embedding_lmhead_strategy,
                    global_batch_size=gbsz,
                    chunks=chunks,
                    sequence_length_list=self.sequence_len,
                    model_args=self.model_args_list[0],
                    train_args=self.train_args_list[0],
                    parallel_args=self.parallel_args_list[0],
                    profile_model_args=self.profile_model_args_list[0],
                    profile_hardware_args=self.profile_hardware_args_list[0],
                    logger=self.logger
                )
                res_with_grad_sync, res_no_grad_sync = obj.gen_result() # res: float, res_no_grad_sync: float
                embedding_lmhead_time_cost[embedding_lmhead_strategy_idx] = (res_with_grad_sync, res_no_grad_sync)
            
            # [Step 5] Calculate the layer-wise memory cost
            # memory_cost: List[np.ndarray]. len(memory_cost) == pp_size
            # memory_cost[stage_idx]: shape: (layer_strategy_num, total_layer_num), dtype:np.int32
            memory_cost = [np.zeros((sum(self.layer_num), layer_strategy_num)) for _ in range(pp_size)]  # List[np.ndarray] - shape: (layer_strategy_num, total_layer_num) - each row: one strategy, each column: one layer
            if self.pipeline_type == "gpipe":
                for layertype_idx in range(num_layertype):
                    all_strategy_memory_cost = []
                    for layer_strategy in layer_strategy_list:
                        obj = MemoryCostModelBase( # stage_idx is not used
                            strategy=layer_strategy,
                            global_batch_size=gbsz,
                            chunks=chunks,
                            logger=self.logger,
                            model_args=self.model_args_list[layertype_idx],
                            train_args=self.train_args_list[layertype_idx],
                            parallel_args=self.parallel_args_list[layertype_idx],
                            profile_model_args=self.profile_model_args_list[layertype_idx],
                        )
                        res = obj.get_memory_cost() # res:dict[str, float]
                        all_strategy_memory_cost.append(res['enc_total'])
                    all_strategy_memory_cost = np.ceil(np.array(all_strategy_memory_cost)).astype(np.int32)
                    for stage_idx in range(pp_size): # when gpipe, memory cost is the same for all stages
                        memory_cost[stage_idx][sum(self.layer_num[:layertype_idx]):sum(self.layer_num[:layertype_idx+1]), :] = all_strategy_memory_cost.reshape(1, -1).repeat(self.layer_num[layertype_idx], axis=0)
            elif self.pipeline_type == "pipedream_flush":
                for stage_idx in range(pp_size):
                    for layertype_idx in range(num_layertype):
                        all_strategy_memory_cost = []
                        for layer_strategy in layer_strategy_list:
                            obj = MemoryCostModelBase(
                                strategy=layer_strategy,
                                global_batch_size=gbsz,
                                chunks=chunks,
                                stage_idx=stage_idx,
                                logger=self.logger,
                                model_args=self.model_args_list[layertype_idx],
                                train_args=self.train_args_list[layertype_idx],
                                parallel_args=self.parallel_args_list[layertype_idx],
                                profile_model_args=self.profile_model_args_list[layertype_idx],
                            )
                            res = obj.get_memory_cost() # res:dict[str, float]
                            all_strategy_memory_cost.append(res['enc_total'])
                        all_strategy_memory_cost = np.ceil(np.array(all_strategy_memory_cost)).astype(np.int32)
                        memory_cost[stage_idx][sum(self.layer_num[:layertype_idx]):sum(self.layer_num[:layertype_idx+1]), :] = all_strategy_memory_cost.reshape(1, -1).repeat(self.layer_num[layertype_idx], axis=0)
            
            # [Step 6] Calculate embedding_lmhead memory cost
            # embedding_lmhead_memory_cost: dict[int, np.ndarray]. 
            # key: embedding_lmhead_strategy_idx
            # value: dtype:int shape:(pp_size,)
            embedding_lmhead_memory_cost = {} # dict[int, list[int]]
            for embedding_lmhead_strategy_idx, embedding_lmhead_strategy in enumerate(embedding_lmhead_strategy_list):
                embedding_lmhead_memory_cost_obj = EmbeddingLMHeadMemoryCostModel(
                    strategy=embedding_lmhead_strategy,
                    global_batch_size=gbsz,
                    chunks=chunks,
                    logger=self.logger,
                    model_args=self.model_args_list[0],
                    train_args=self.train_args_list[0],
                    parallel_args=self.parallel_args_list[0],
                    profile_model_args=self.profile_model_args_list[0],
                )
                res = embedding_lmhead_memory_cost_obj.get_memory_cost()
                embedding_lmhead_memory_cost[embedding_lmhead_strategy_idx] = np.ceil(res['enc_total']).astype(int) # NOTE check astype(int) or astype(np.int32)
                
            # [Step 7] Calculate the inter-layer cost
            # NEW VERSION: inter-layer timecost model
            # inter_layer_cost: dtype:np.float64 shape:(total_layer_num, layer_strategy_num, layer_strategy_num)
            inter_layer_cost = np.zeros((total_layer_num, layer_strategy_num, layer_strategy_num))
            for layertype_idx in range(num_layertype):
                res = np.zeros((layer_strategy_num, layer_strategy_num))
                for former_idx in range(layer_strategy_num):
                    for latter_idx in range(layer_strategy_num):
                        if former_idx == latter_idx: # the same strategy has no inter-layer cost
                            continue
                        former = layer_strategy_list[former_idx]
                        latter = layer_strategy_list[latter_idx]
                        if self.config.sequence_parallel and former.tp_sp_size != latter.tp_sp_size:
                            # sequence parallel and tp_sp_size is different
                            greater_tp_sp_size = max(former.tp_sp_size, latter.tp_sp_size)
                            cur_dp_size = self.world_size // pp_size // greater_tp_sp_size
                            cur_lbsz = gbsz / chunks / cur_dp_size
                            single_sample_size = self.sequence_len[layertype_idx] * self.config.hidden_size * (4 if self.config.mixed_precision == "fp32" else 2)
                            res[former_idx, latter_idx] = (greater_tp_sp_size - 1) / greater_tp_sp_size * cur_lbsz * single_sample_size
                            if greater_tp_sp_size == 1 or cur_dp_size == 1:
                                coe = self.comm_coe_dict['%d'%greater_tp_sp_size] if '%d'%greater_tp_sp_size in self.comm_coe_dict.keys() else self.comm_coe_dict['%d_1'%greater_tp_sp_size]
                            else:
                                coe = self.comm_coe_dict['%d_1'%greater_tp_sp_size]
                            res[former_idx, latter_idx] *= coe * 1e-7
                        else:
                            # add a small bias to sort fsdp and dp
                            # tp -> sp
                            if self.match_strategy(former, latter, diff_keys=['sp']):
                                if latter.sp_size > 1:
                                    res[former_idx, latter_idx] = 1e-10
                            # ->f     c -> fc 
                            if self.match_strategy(former, latter, diff_keys=['fsdp']):
                                if latter.dp_type == DPType.ZERO3:
                                    res[former_idx, latter_idx] = 1e-9
                            # ->c  f -> cf
                            if self.match_strategy(former, latter, diff_keys=['cpt']):
                                if latter.checkpoint:
                                    res[former_idx, latter_idx] = 2e-9
                            # ->fc
                            if self.match_strategy(former, latter, diff_keys=['fsdp','cpt']):
                                if latter.dp_type == DPType.ZERO3 and latter.checkpoint:
                                    res[former_idx, latter_idx] = 3e-9
                            # f->c
                            if self.match_strategy(former, latter, diff_keys=['fsdp','cpt']) \
                                and not self.match_strategy(former, latter, diff_keys=['fsdp']) \
                                and not self.match_strategy(former, latter, diff_keys=['cpt']):
                                    if former.dp_type == DPType.ZERO3 and latter.checkpoint:
                                        res[former_idx, latter_idx] = 1e-9
                            
                inter_layer_cost[sum(self.layer_num[:layertype_idx]):sum(self.layer_num[:layertype_idx+1]), :, :] = res
            inter_layer_cost[0, :, :] = 0 # no inter-layer communication cost in first layer

            # [Step 8] Solve the optimization problem
            # [Step 8.1] Initialize the optimal solution
            optimal = {
                'time_cost': np.inf,
                'memory_used': [-1 for _ in range(pp_size)],
                'memory_remain': [-1 for _ in range(pp_size)],
                'strategy_list': None,
                'embedding_lmhead_tp_sp_size': -1,
                'embedding_lmhead_sp': -1,
                'embedding_lmhead_sdp': -1,
                'pp_size': pp_size,
            }
            # [Step 8.2] Solve the optimization problem for each embedding_lmhead_strategy
            for embedding_lmhead_strategy_idx, embedding_lmhead_strategy in enumerate(embedding_lmhead_strategy_list):
                embedding_lmhead_tp = embedding_lmhead_strategy.tp_sp_size # to fit the old version DPAlg

                start_layer = 0

                # len(res_list_list) == len(mem_remain_list) == len(mem_used_list) == pp_size
                strategy_list_list, mem_remain_list, mem_used_list = [], [], []
                
                for stage_idx in range(pp_size):
                    cur_other_memory_cost = { # to fit the old version DPAlg
                        embedding_lmhead_tp: embedding_lmhead_memory_cost[embedding_lmhead_strategy_idx][stage_idx] + int(global_memory)
                    }
                    cur_other_time_cost = { # to fit the old version DPAlg
                        embedding_lmhead_tp: embedding_lmhead_time_cost[embedding_lmhead_strategy_idx][0][stage_idx]  # 0: grad sync
                    }

                    dp = DPAlg(
                        max_mem=self.max_mem,
                        other_mem_cost=cur_other_memory_cost,
                        other_time_cost=cur_other_time_cost,
                        layer_num=pp_stage_list[stage_idx],
                        layer_strategy_num=layer_strategy_num,
                        fine_grained_mode=self.config.fine_grained_mode,
                    )
                    dp.set_v_and_cost(
                        v=memory_cost[stage_idx][start_layer:start_layer+pp_stage_list[stage_idx]],
                        intra_layer_cost=intra_layer_cost[start_layer:start_layer+pp_stage_list[stage_idx]],
                        inter_layer_cost=inter_layer_cost[start_layer:start_layer+pp_stage_list[stage_idx]]
                    )
                    time_cost_this_stage, strategy_list_this_stage, mem_remain_this_stage = dp.fit() # time_cost_this_stage: float, strategy_list_this_stage: dict[int, list[int]], mem_remain_this_stage: dict[int, int]
                    
                    # to fit the old version DPAlg
                    strategy_list_this_stage = strategy_list_this_stage[embedding_lmhead_tp] # strategy_list_this_stage: list[int]
                    mem_remain_this_stage = mem_remain_this_stage[embedding_lmhead_tp] # mem_remain_this_stage: int

                    if mem_remain_this_stage == -1:
                        strategy_list_this_stage = None
                        mem_used_this_stage = np.inf
                    else:
                        strategy_list_this_stage = list(map(lambda x: layer_strategy_list[x], strategy_list_this_stage)) # list[new_strategy]
                        mem_used_this_stage = self.max_mem - mem_remain_this_stage + self.mem_cache

                    strategy_list_list.append(strategy_list_this_stage)
                    mem_remain_list.append(mem_remain_this_stage)
                    mem_used_list.append(mem_used_this_stage)
                    start_layer += pp_stage_list[stage_idx]

                if None not in strategy_list_list:
                    strategy_list = [] # list[new_strategy]
                    for item in strategy_list_list:
                        strategy_list.extend(item)
                    pipeline_cost = pipeline_costmodel(
                        layer_num_list=self.layer_num,
                        model_args_list=self.model_args_list,
                        train_args_list=self.train_args_list,
                        parallel_args_list=self.parallel_args_list,
                        profile_model_args_list=self.profile_model_args_list,
                        profile_hardware_args_list=self.profile_hardware_args_list,
                        strategy_list=strategy_list,
                        partition=pp_stage_list,
                        chunks=chunks,
                        gbsz=gbsz,
                        pp_size=pp_size,
                        other_time_cost=embedding_lmhead_time_cost[embedding_lmhead_strategy_idx][1], # TODO: check this
                        logger=self.logger,
                        return_stage_cost=False
                    )
                    if optimal['time_cost'] > pipeline_cost:
                        optimal['time_cost'] = pipeline_cost
                        optimal['memory_used'] = copy.deepcopy(mem_used_list)
                        optimal['memory_remain'] = copy.deepcopy(mem_remain_list)
                        optimal['strategy_list'] = copy.deepcopy(strategy_list)
                        optimal['embedding_lmhead_tp_sp_size'] = embedding_lmhead_tp
                        optimal['embedding_lmhead_sp'] = 1 if embedding_lmhead_strategy.sp_size > 1 else 0
                        optimal['embedding_lmhead_sdp'] = 1 if embedding_lmhead_strategy.dp_type == DPType.ZERO3 else 0
                    self.log(f'embedding_lmhead_strategy: {embedding_lmhead_strategy}\npipeline_cost: {pipeline_cost}')
                else:
                    self.log(f'embedding_lmhead_strategy: {embedding_lmhead_strategy}\nno solution')
            return optimal

    def log(self, msg) -> None:
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg, flush=True)

    def fit(
        self, 
        gbsz:int, 
        chunks:int, 
        pp_size:int, 
        pp_stage_list:list[int],
        global_buffer_tp_size:int, 
        tp_sp_mode:str,
        layer_strategy_list:List[LayerStrategy] = None,
        embedding_lmhead_strategy_list:List[EmbeddingLMHeadStrategy] = None
    ) -> dict[str, Any]:
        self.log(f'\n{"="*50}Enter DpOnModel{"="*50}')

        self.input_layer_strategy_list = layer_strategy_list
        self.input_embedding_lmhead_strategy_list = embedding_lmhead_strategy_list

        print_strategy_list(self.input_layer_strategy_list, logger=self.logger)
        print_strategy_list(self.input_embedding_lmhead_strategy_list, logger=self.logger)

        optimal = self._build_dp_and_run_multi_layer_type(
            gbsz=gbsz,
            chunks=chunks,
            pp_size=pp_size,
            pp_stage_list=pp_stage_list,
            global_buffer_tp_size=global_buffer_tp_size,
            tp_sp_mode=tp_sp_mode,
        )

        self.log(f'{"="*50}Exit DpOnModel{"="*50}\n')
        return optimal

