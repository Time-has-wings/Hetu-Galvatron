from logging import Logger
from galvatron.utils.strategy_utils import GalvatronStrategy,strategy_list_to_csv
from galvatron.utils.training_utils import ColorSet
import copy
from typing import List
from galvatron.core.search_engine.utils import ensure_log_dir, get_thread_logger_optimize
from galvatron.core.cost_model import GalvatronCostModelHandler
import numpy as np
from galvatron.core.search_engine.dynamic_programming import DPAlgOptimize
from galvatron.utils.strategy_utils import is_power_of_two

class StageInfo:
    def __init__(self, stage_idx):
        self.stage_idx = stage_idx
    
    def set_basic_info(self, layernum, layertype_id_list, layer_desc_list):
        self.layernum = layernum
        self.layertype_id_list = layertype_id_list
        self.layer_desc_list = layer_desc_list
        
    def set_dp_result(self, dp_time_cost, dp_memory_cost, dp_memory_remain, strategy_idx_select_list):
        self.dp_time_cost = dp_time_cost
        self.dp_memory_cost = dp_memory_cost
        self.dp_memory_remain = dp_memory_remain
        self.strategy_idx_select_list = strategy_idx_select_list

    def set_strategy_select_list(self, strategy_select_list):
        self.strategy_select_list = strategy_select_list

    def set_stage_cost(self, time_cost_list, time_no_sync_cost_list, memory_cost_list,
                            time_cost, time_no_sync_cost, memory_cost, global_memory_buffer_size_in_MB):
        assert len(time_cost_list) == self.layernum and len(time_no_sync_cost_list) == self.layernum and len(memory_cost_list) == self.layernum
        
        self.time_cost_list = time_cost_list
        self.time_no_sync_cost_list = time_no_sync_cost_list
        self.memory_cost_list = memory_cost_list

        self.time_cost = time_cost
        self.time_no_sync_cost = time_no_sync_cost
        self.memory_cost = memory_cost
        self.global_memory_buffer_size_in_MB = global_memory_buffer_size_in_MB

    def get_info(self):
        info = {
            'stage_idx': self.stage_idx,
            'layernum': self.layernum,
            'layertype_id_list': self.layertype_id_list,
            'layer_desc_list': self.layer_desc_list,
            'strategy_select_list': self.strategy_select_list,
            'time_cost': self.time_cost,
            'time_no_sync_cost': self.time_no_sync_cost,
            'memory_cost': self.memory_cost,
        }
        return info

class GalvatronSearchEngineOptimize:
    def __init__(self, args=None, logger:Logger=None):
        self.args = args
        self.logger = logger
        self.prompt_color = ColorSet.GREEN
        self.handler:GalvatronCostModelHandler = GalvatronCostModelHandler(args)
        self.memory_constraint = 32 * 1024 # 32GB
        
    def initialize_search_engine_optimize(self):
        pass
    
    def generate_strategy_list_optimize(self, world_size, is_moe=False):
        print(f'{self.prompt_color}[GalvatronSearchEngineOptimize] Generating strategy list for world_size={world_size} and is_moe={is_moe}...{ColorSet.RESET}')
        
        # [Step 0] 
        rank_range, avail = [], 1
        while avail <= world_size:
            rank_range.append(avail)
            avail *= 2
        
        # [Step 1] generate all possible strategies of layer
        strategy_list:List[GalvatronStrategy] = []
        for pp_size in rank_range:
            for tp_size in rank_range:
                if pp_size * tp_size > world_size:
                    continue
                dp_size = world_size // pp_size // tp_size # TODO consider cp_size
                dp_type_list = ['zero0'] if dp_size == 1 else ['zero0', 'zero2', 'zero3']
                for dp_type in dp_type_list:
                    for use_ulysses in [False, True]:
                        for checkpoint in [False, True]:
                            strategy = GalvatronStrategy( # The variables cp_size, ep_size, tp_of_ep_size, and tp_consecutive_flag are not set
                                pp_size=pp_size,
                                tp_size=tp_size,
                                dp_size=dp_size,
                                use_ulysses=use_ulysses,
                                dp_type=dp_type,
                                checkpoint=checkpoint,
                                unit='all',
                                is_moe=False
                            )
                            strategy_list.append(strategy)
        strategy_list = sorted(list(set(strategy_list)))
        self.strategy_list = strategy_list
        print(f'layer strategy list (Size {len(self.strategy_list)}) :')
        print(strategy_list_to_csv(self.strategy_list))
        
        # [Step 2] generate attention strategy list
        self.attention_strategy_list:List[GalvatronStrategy] = copy.deepcopy(strategy_list)
        for strategy in self.attention_strategy_list:
            strategy.unit = 'attention'
        self.attention_strategy_list = sorted(self.attention_strategy_list)
        print(f'attention strategy list (Size {len(self.attention_strategy_list)}) :')
        print(strategy_list_to_csv(self.attention_strategy_list))
        
        # [Step 3] generate ffn strategy list
        if is_moe:
            ffn_strategy_list:List[GalvatronStrategy] = []
            for pp_size in rank_range:
                for tp_of_ep_size in rank_range:
                    if pp_size * tp_of_ep_size > world_size:
                        continue
                    ep_size = world_size // pp_size // tp_of_ep_size
                    for checkpoint in [False, True]:
                        strategy = GalvatronStrategy(
                            pp_size=pp_size,
                            tp_of_ep_size=tp_of_ep_size,
                            ep_size=ep_size,
                            checkpoint=checkpoint,
                            unit='ffn',
                            is_moe=True,
                        )
                        ffn_strategy_list.append(strategy)
            ffn_strategy_list = list(set(ffn_strategy_list))
            self.ffn_strategy_list = ffn_strategy_list
        else:
            self.ffn_strategy_list:List[GalvatronStrategy] = copy.deepcopy(strategy_list)
            for strategy in self.ffn_strategy_list:
                strategy.unit = "ffn"
        self.ffn_strategy_list = sorted(self.ffn_strategy_list)
        print(f'ffn strategy list (Size {len(self.ffn_strategy_list)}) :')
        print(strategy_list_to_csv(self.ffn_strategy_list))
        
        # [Step 4] generate embedding_lmhead strategy list
        embedding_lmhead_strategy_list:List[GalvatronStrategy] = []
        for pp_size in rank_range:
            for tp_size in rank_range:
                if pp_size * tp_size > world_size:
                    continue
                dp_size = world_size // pp_size // tp_size
                dp_type_list = ['zero0'] if dp_size == 1 else ['zero0', 'zero2', 'zero3']
                for dp_type in dp_type_list:
                    for use_ulysses in [False, True]:
                        strategy = GalvatronStrategy( # The variables cp_size, ep_size, tp_of_ep_size, and tp_consecutive_flag are not set
                            pp_size=pp_size,
                            tp_size=tp_size,
                            dp_size=dp_size,
                            use_ulysses=use_ulysses,
                            dp_type=dp_type,
                            checkpoint=False,
                            unit='embedding_lmhead',
                            is_moe=False
                        )
                        embedding_lmhead_strategy_list.append(strategy)
        embedding_lmhead_strategy_list = sorted(list(set(embedding_lmhead_strategy_list)))
        self.embedding_lmhead_strategy_list = embedding_lmhead_strategy_list
        print(f'layer strategy list (Size {len(self.embedding_lmhead_strategy_list)}) :')
        print(strategy_list_to_csv(self.embedding_lmhead_strategy_list))

        print(f'{self.prompt_color}[GalvatronSearchEngineOptimize] Generating strategy list done.{ColorSet.RESET}')
        
    def parallelism_optimization(self, parallel_search=False, world_size=8, gbsz_list=[], layer_num=24):
        print(f'{self.prompt_color}[GalvatronSearchEngineOptimize] {"=" * 25} Galvatron Search Engine Start Searching {"=" * 25}{ColorSet.RESET}' )
        
        # [DEBUG] set some value manully
        self.world_size = world_size
        self.BSZs = gbsz_list
        avail = 1
        self.pp_set = []
        while avail <= world_size:
            self.pp_set.append(avail)
            avail *= 2
        self.sp_mode_set = ['tp_only']

        self.num_layertype = 1 # 临时测试
        self.pipeline_type = 'pipedream_flush'
        # print(f'[linguangming] self.pp_range: {self.pp_range}')        

        # [Step 1] Enumerate all scenarios and build the task set.
        results = dict()
        all_tasks = []
        for gbsz in self.BSZs:
            results[gbsz] = dict()
            chunk_list = range(1, gbsz + 1)
            for chunks in chunk_list:
                if gbsz % chunks != 0:
                    continue
                results[gbsz][chunks] = dict()
                for pp_size in self.pp_set:
                    if chunks < pp_size: # TODO check 
                        continue
                    results[gbsz][chunks][pp_size] = dict()

                    min_dp_size = 1
                    max_dp_size = min(gbsz // chunks, world_size // pp_size)
                    min_tp_size = world_size // pp_size // max_dp_size
                    max_tp_size = world_size // pp_size // min_dp_size

                    for sp_mode in self.sp_mode_set:
                        results[gbsz][chunks][pp_size][sp_mode] = dict()

                        consider_global_memory_buffer = True if self.handler.args.sequence_parallel and self.args.global_memory_buffer and sp_mode != 'ulysses-only' else False
                        if consider_global_memory_buffer:
                            lower_tp, upper_tp = min_tp_size, max_tp_size
                        else:
                            lower_tp, upper_tp = max_tp_size, max_tp_size

                        for tp_of_gmb in range(lower_tp, upper_tp + 1): # tp_of_gmb means tp of global memory buffer
                            if is_power_of_two(tp_of_gmb) == False:
                                continue
                            results[gbsz][chunks][pp_size][sp_mode][tp_of_gmb] = {'throughput': -1,'detailed_info': {}}

                            ctx = { 'min_tp_size': min_tp_size, } # ctx: Used to provide information of non-for-loop variables
                            all_tasks.append((gbsz, chunks, pp_size, sp_mode, tp_of_gmb, copy.deepcopy(ctx)))

        # [Step 2] Solve for a single task
        def search_for_single_task(gbsz, chunks, pp_size, sp_mode, tp_of_gmb, ctx:dict):
            # [Step 2.1] parse ctx
            min_tp_size = ctx['min_tp_size']

            # [Step 2.2] Filter the global strategies and form optional strategy pool.
            layer_strategy_pool:List[GalvatronStrategy] = []
            for strategy in self.strategy_list:
                if strategy.pp_size != pp_size:
                    continue
                if strategy.tp_size < min_tp_size or strategy.tp_size > tp_of_gmb:
                    continue
                if sp_mode == 'tp_only' and strategy.use_ulysses == True:
                    continue
                if sp_mode == 'ulysses_only' and strategy.use_ulysses == False:
                    continue
                layer_strategy_pool.append(copy.deepcopy(strategy))

            embedding_lmhead_strategy_pool:List[GalvatronStrategy] = []
            for strategy in self.embedding_lmhead_strategy_list:
                if strategy.pp_size != pp_size:
                    continue
                if strategy.tp_size < min_tp_size or strategy.tp_size > tp_of_gmb:
                    continue
                if sp_mode == 'tp_only' and strategy.use_ulysses == True:
                    continue
                if sp_mode == 'ulysses_only' and strategy.use_ulysses == False:
                    continue
                embedding_lmhead_strategy_pool.append(copy.deepcopy(strategy))
            
            layer_strategy_pool = sorted(layer_strategy_pool)
            embedding_lmhead_strategy_pool = sorted(embedding_lmhead_strategy_pool)

            # [Step 2.3] Prepare extra information
            if consider_global_memory_buffer:
                dtype_size = 2 if self.handler.args.mixed_precision else 4
                byte_to_MB = 1024 * 1024
                current_dp_size = self.world_size // pp_size // tp_of_gmb # TODO When considering CP, adjust this accordingly.
                global_memory_buffer_size_in_MB = (gbsz // chunks // current_dp_size) * max(self.handler.seqlen_list) * self.handler.args.hidden_size * dtype_size / byte_to_MB
            else:
                global_memory_buffer_size_in_MB = 0

            extra_info = {
                'naming': { # Naming: Used for log file naming
                    'sp_mode': sp_mode,
                    'tp_of_gmb': tp_of_gmb
                },
                'info': {
                    'consider_global_memory_buffer': consider_global_memory_buffer,
                    'global_memory_buffer_size_in_MB': global_memory_buffer_size_in_MB
                },
            }

            # [Step 2.4] Solve using dynamic programming
            throughput, detailed_info = self.search_for_fix_situation(gbsz=gbsz, chunks=chunks, pp_size=pp_size, layer_strategy_pool=layer_strategy_pool, embedding_lmhead_strategy_pool=embedding_lmhead_strategy_pool, extra_info=extra_info,)
            return throughput, detailed_info

        # [Step 3] Start solving the task set (supports parallel and sequential search)
        if parallel_search:
            import concurrent.futures
            import threading
            import multiprocessing

            if hasattr(self.args, 'worker') and self.args.worker > 0:
                num_threads = min(self.args.worker, len(all_tasks))
            else:
                num_threads = min(multiprocessing.cpu_count() * 2, len(all_tasks))
            print(f"Starting parallel search with {num_threads} threads for {len(all_tasks)} tasks...")

            results_lock = threading.Lock()
            def process_task(gbsz, chunks, pp_size, sp_mode, tp_of_gmb, ctx):
                thread_id = threading.get_ident() % 1000
                print(f"[Thread {thread_id:03d}] Start processing: gbsz={gbsz}, chunks={chunks}, pp_size={pp_size}, sp_mode={sp_mode}, tp_of_gmb={tp_of_gmb}, ctx={ctx}", flush=True)

                throughput, detailed_info = search_for_single_task(gbsz, chunks, pp_size, sp_mode, tp_of_gmb, ctx)
                with results_lock:
                    results[gbsz][chunks][pp_size][sp_mode][tp_of_gmb] = {
                        'throughput': throughput,
                        'detailed_info': copy.deepcopy(detailed_info)
                    }
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_task, gbsz, chunks, pp_size, sp_mode, tp_of_gmb, ctx) for gbsz, chunks, pp_size, sp_mode, tp_of_gmb, ctx in all_tasks]
                concurrent.futures.wait(futures)
        else:
            for task in all_tasks:
                gbsz, chunks, pp_size, sp_mode, tp_of_gmb, ctx = task
                throughput, detailed_info = search_for_single_task(gbsz, chunks, pp_size, sp_mode, tp_of_gmb, ctx)
                results[gbsz][chunks][pp_size][sp_mode][tp_of_gmb] = {
                    'throughput': throughput,
                    'detailed_info': detailed_info
                }
                        
        # [Step 4] Select the optimal solution and save results
        max_throughput, optimal_detailed_info = -1, None
        for gbsz in results:
            for chunks in results[gbsz]:
                for pp_size in results[gbsz][chunks]:
                    for sp_mode in results[gbsz][chunks][pp_size]:
                        for tp_of_gmb in results[gbsz][chunks][pp_size][sp_mode]:
                            res = results[gbsz][chunks][pp_size][sp_mode][tp_of_gmb]
                            if res['throughput'] > max_throughput:
                                max_throughput = res['throughput']
                                optimal_detailed_info = res['detailed_info']

        if max_throughput > 0:
            print(f'please print something and store config file')
            self.save_results(optimal_detailed_info)
        else:
            print("No valid configuration found.")
        
        print("-----------------------------------------")
        print(f'{self.prompt_color}[GalvatronSearchEngineOptimize] {"=" * 25} Galvatron Search Engine End Searching {"=" * 25}{ColorSet.RESET}' )
        
    def search_for_fix_situation(self, gbsz, chunks, pp_size, layer_strategy_pool:List[GalvatronStrategy], embedding_lmhead_strategy_pool:List[GalvatronStrategy], extra_info=None):
        print(f"{self.prompt_color}[GalvatronSearchEngineOptimize]{ColorSet.RESET} Starting search for gbsz={gbsz}, chunks={chunks}, pp_size={pp_size}, extra_info={extra_info}")

        log_dir = "search_engine_logs"
        log_dir = ensure_log_dir(log_dir)
        logger = get_thread_logger_optimize(gbsz, chunks, pp_size, extra_info['naming'], log_dir)
        logger.info(f"Starting search for gbsz={gbsz}, chunks={chunks}, pp_size={pp_size}, extra_info={extra_info}")
        logger.info(f'strategy_pool:\n {strategy_list_to_csv(layer_strategy_pool)}')
        logger.info(f'embedding_lmhead_strategy_pool:\n {strategy_list_to_csv(embedding_lmhead_strategy_pool)}')

        if len(layer_strategy_pool) == 0 or len(embedding_lmhead_strategy_pool) == 0:
            logger.info(f'Strategy pool empty, exit.')
            througput, detailed_info = -1, {}
            return througput, detailed_info

        # [Step 1] Create and maintain information of all stage. [OK]
        info_per_stage = [StageInfo(stage_idx) for stage_idx in range(pp_size)]
        layertype_id_list_per_stage, layer_desc_list_per_stage = self.get_pp_division(pp_size)
        for stage_idx in range(pp_size):
            info_per_stage[stage_idx].set_basic_info(layernum=len(layertype_id_list_per_stage[stage_idx]),
                                                     layertype_id_list=layertype_id_list_per_stage[stage_idx],
                                                     layer_desc_list=layer_desc_list_per_stage[stage_idx])
        
        # [Step 2] Get time cost of layer_strategy_pool. [OK]
        layer_time_cost_pool = [[] for _ in range(self.num_layertype)]
        layer_time_no_sync_cost_pool = [[] for _ in range(self.num_layertype)]
        for layertype_id in range(self.num_layertype):
            for strategy in layer_strategy_pool:
                time_cost, time_no_sync_cost = self.handler.get_time_cost(global_batch_size=gbsz, chunks=chunks, strategy=strategy, layertype_id=layertype_id)
                layer_time_cost_pool[layertype_id].append(time_cost)
                layer_time_no_sync_cost_pool[layertype_id].append(time_no_sync_cost)
        
        # [Step 3] Get time cost of embedding_lmhead_strategy_pool. [OK]
        embedding_lmhead_time_cost_pool = []
        embedding_lmhead_no_sync_time_cost_pool = []
        for strategy in embedding_lmhead_strategy_pool:
            time_cost, time_no_sync_cost = self.handler.get_time_cost(global_batch_size=gbsz, chunks=chunks, strategy=strategy, layertype_id=layertype_id)
            embedding_lmhead_time_cost_pool.append(time_cost)
            embedding_lmhead_no_sync_time_cost_pool.append(time_no_sync_cost)

        # [Step 4] Simulate inter-layer time overhead based on empirical rules. [OK]
        inter_layer_cost_list = []
        layer_strategy_pool_size = len(layer_strategy_pool)
        for layertype_id in range(self.num_layertype):
            inter_layer_cost = np.zeros((layer_strategy_pool_size, layer_strategy_pool_size))
            inter_layer_cost_list.append(inter_layer_cost)
        
        # [Step 5] Get memory cost of layer_strategy_pool. [OK]
        if self.pipeline_type == "gpipe":
            raise NotImplemented('not implemented')
        elif self.pipeline_type == 'pipedream_flush':
            layer_memory_cost_pool = [[[] for _ in range(pp_size)] for _ in range(self.num_layertype)]
            for layertype_id in range(self.num_layertype):
                for stage_idx in range(pp_size):
                    for strategy in layer_strategy_pool:
                        memory_cost = self.handler.get_memory_cost(global_batch_size=gbsz, chunks=chunks, strategy=strategy, layertype_id=layertype_id, stage_idx=stage_idx)
                        layer_memory_cost_pool[layertype_id][stage_idx].append(memory_cost)
        
        # [Step 6] Get memory cost of embedding_lmhead_memory_pool. [OK]
        embedding_lmhead_memory_cost_pool = [] 
        for strategy in embedding_lmhead_strategy_pool:
            memory_cost = self.handler.get_memory_cost(global_batch_size=gbsz, chunks=chunks, strategy=strategy)
            embedding_lmhead_memory_cost_pool.append(memory_cost)
    
        # [Step 7] Enumerate all strategy in embedding_lmhead_strategy_pool to ensure both the embedding and lmhead use the same strategy [OK]
        max_throughput, detailed_info = -1, {}
        for embedding_lmhead_strategy_idx in range(len(embedding_lmhead_strategy_pool)):
            # [Step 7.1] solve using dynamic program for each stage
            for stage_idx in range(pp_size):
                dp_layernum = info_per_stage[stage_idx].layernum
                strategy_num_list = []
                layer_time_no_sync_cost_list = []
                layer_memory_cost_list = []
                extra_memory_occupy = extra_info['info']['global_memory_buffer_size_in_MB'] # consider global memory buffer for tp.

                for layertype_id in info_per_stage[stage_idx].layertype_id_list:
                    if layertype_id == -1: # embedding
                        dp_layernum -= 1
                        extra_memory_occupy += embedding_lmhead_memory_cost_pool[embedding_lmhead_strategy_idx]['embedding']
                    elif layertype_id == -2: # lmhead
                        dp_layernum -= 1
                        extra_memory_occupy += embedding_lmhead_memory_cost_pool[embedding_lmhead_strategy_idx]['lmhead']
                    else:
                        layer_time_no_sync_cost_list.append(layer_time_no_sync_cost_pool[layertype_id])
                        layer_memory_cost_list.append(layer_memory_cost_pool[layertype_id][stage_idx])
                        strategy_num_list.append(len(layer_time_no_sync_cost_pool[layertype_id]))
                
                dp_solver = DPAlgOptimize(dp_layernum, strategy_num_list, layer_time_no_sync_cost_list, layer_memory_cost_list, extra_memory_occupy, self.memory_constraint) # use no_sync time cost
                dp_time_cost, dp_memory_cost, dp_memory_remain, strategy_idx_select_list = dp_solver.fit()  
                info_per_stage[stage_idx].set_dp_result(dp_time_cost, dp_memory_cost, dp_memory_remain, strategy_idx_select_list)
                
            # [Step 7.2] Get stage cost
            stage_time_list, stage_time_no_sync_list, stage_memory_list = [], [], []
            for stage_idx in range(pp_size):
                strategy_select_list = []
                time_cost, time_no_sync_cost, memory_cost = 0, 0, 0
                time_cost_list, time_no_sync_cost_list, memory_cost_list = [], [], []
                idx = -1
                for layertype_id in info_per_stage[stage_idx].layertype_id_list:
                    if layertype_id == -1: # embedding
                        strategy_select_list.append(copy.deepcopy(embedding_lmhead_strategy_pool[embedding_lmhead_strategy_idx]))
                        time_cost_list.append(embedding_lmhead_time_cost_pool[embedding_lmhead_strategy_idx]['embedding'])
                        time_no_sync_cost_list.append(embedding_lmhead_no_sync_time_cost_pool[embedding_lmhead_strategy_idx]['embedding'])
                        memory_cost_list.append(embedding_lmhead_memory_cost_pool[embedding_lmhead_strategy_idx]['embedding'])
                    elif layertype_id == -2: # lmhead
                        strategy_select_list.append(copy.deepcopy(embedding_lmhead_strategy_pool[embedding_lmhead_strategy_idx]))
                        time_cost_list.append(embedding_lmhead_time_cost_pool[embedding_lmhead_strategy_idx]['lmhead'])
                        time_no_sync_cost_list.append(embedding_lmhead_no_sync_time_cost_pool[embedding_lmhead_strategy_idx]['lmhead'])
                        memory_cost_list.append(embedding_lmhead_memory_cost_pool[embedding_lmhead_strategy_idx]['lmhead'])
                    else:
                        idx += 1
                        strategy_idx = info_per_stage[stage_idx].strategy_idx_select_list[idx]
                        strategy_select_list.append(copy.deepcopy(layer_strategy_pool[strategy_idx]))
                        time_cost_list.append(layer_time_cost_pool[layertype_id][strategy_idx])
                        time_no_sync_cost_list.append(layer_time_no_sync_cost_pool[layertype_id][strategy_idx])
                        memory_cost_list.append(layer_memory_cost_pool[layertype_id][stage_idx][strategy_idx])

                time_cost = sum(time_cost_list)
                time_no_sync_cost = sum(time_no_sync_cost_list)
                memory_cost = sum(memory_cost_list)
                memory_cost += extra_info['info']['global_memory_buffer_size_in_MB']
                info_per_stage[stage_idx].set_stage_cost(time_cost_list, time_no_sync_cost_list, memory_cost_list,
                                                        time_cost, time_no_sync_cost, memory_cost, extra_info['info']['global_memory_buffer_size_in_MB'])
                info_per_stage[stage_idx].set_strategy_select_list(strategy_select_list)

                stage_time_list.append(time_cost)
                stage_time_no_sync_list.append(time_no_sync_cost)
                stage_memory_list.append(memory_cost)

            # [Step 7.3] pipeline stage cost
            if pp_size == 1:
                fix_stage_idx = 0
                pipeline_time_cost = info_per_stage[fix_stage_idx].time_no_sync_cost * (chunks - 1) + info_per_stage[fix_stage_idx].time_cost
                throughput = gbsz / pipeline_time_cost

                if throughput > max_throughput:
                    max_throughput = throughput
                    detailed_info = {
                        'gbsz': gbsz,
                        'chunks': chunks,
                        'pp_size': pp_size,
                        'pipeline_time_cost': pipeline_time_cost,
                        'info_per_stage': [stage.get_info() for stage in info_per_stage]
                    }
            else:
                last_stage_idx = pp_size - 1
                pipeline_time_cost = info_per_stage[last_stage_idx].time_no_sync_cost * (chunks - 1)
                for stage_idx in range(pp_size):
                    pipeline_time_cost += info_per_stage[stage_idx].time_cost
                p2p_cost = self.handler.get_p2p_cost()
                pipeline_time_cost += 2 * (pp_size - 1) * p2p_cost
                throughput = gbsz / pipeline_time_cost

                if throughput > max_throughput:
                    max_throughput = throughput
                    detailed_info = {
                        'gbsz': gbsz,
                        'chunks': chunks,
                        'pp_size': pp_size,
                        'pipeline_time_cost': pipeline_time_cost,
                        'info_per_stage': [stage.get_info() for stage in info_per_stage]
                    }
        
        logger.info(f"End search for gbsz={gbsz}, chunks={chunks}, pp_size={pp_size}, extra_info={extra_info}")
        return max_throughput, detailed_info
    
    def get_pp_division(self, pp_size):
        layernum_list = copy.deepcopy(self.handler.layernum_list)
        layertype_id_each_layer = []
        layer_desc_each_layer = []
        if len(layernum_list) == 1: # decoder-only
            layertype_id_each_layer = [0 for _ in range(layernum_list[0])]
            layer_desc_each_layer = [f'decoder{i}' for i in range(layernum_list[0])]
        else: # encoder + decoder
            layertype_id_each_layer = [0 for _ in range(layernum_list[0])] + [1 for _ in range(layernum_list[1])]
            layer_desc_each_layer = [f'encoder{i}' for i in range(layernum_list[0])] + [f'decoder{i + layernum_list[0]}' for i in range(layernum_list[1])]
        
        # split evenly
        total_layer_num = sum(layernum_list)
        avg_layer_num = int(total_layer_num // pp_size)
        
        layertype_id_list_per_stage = []
        layer_desc_list_per_stage = []
        for stage_idx in range(pp_size):
            layertype_id_list = []
            layer_desc_list = []
            
            start = avg_layer_num * (stage_idx - 1)
            end = min(avg_layer_num * stage_idx, total_layer_num)
            
            if stage_idx == 0:
                layertype_id_list.append(-1)
                layer_desc_list.append('embedding')
            for i in range(start, end):
                layertype_id_list.append(layertype_id_each_layer[i])
                layer_desc_list.append(layer_desc_each_layer[i])
            if stage_idx == pp_size - 1:
                layertype_id_list.append(-2)
                layer_desc_list.append('lmhead')
            
            layertype_id_list_per_stage.append(layertype_id_list)
            layer_desc_list_per_stage.append(layer_desc_list)
        
        return layertype_id_list_per_stage, layer_desc_list_per_stage
    
    def save_results(self, optimal_detailed_info):
        pass
    
    