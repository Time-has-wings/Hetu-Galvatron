import copy
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from types import SimpleNamespace
from typing import List, Tuple, Union

import numpy as np
from loguru import logger as loguru_logger
from rich.pretty import pretty_repr

from galvatron.core.cost_model import GalvatronCostModelHandler
from galvatron.core.search_engine.dynamic_program import DynamicProgramming
from galvatron.core.search_engine.utils import get_thread_logger
from galvatron.utils.strategy_utils import (
    AttentionStrategy,
    byte_to_MB,
    DPType,
    EmbeddingLMHeadStrategy,
    FFNStrategy,
    is_power_of_two,
    LayerStrategy,
    MoEFFNStrategy,
)
from galvatron.utils.training_utils import ColorSet

@dataclass 
class StageInfo:
    stage_idx: int = 0
    layernum: int = 1
    layertype_id_list: List[int] = field(default_factory=list)
    layer_desc_list: List[str] = field(default_factory=list)
    best_time: float = -1
    best_memory:float = -1
    best_strategy_list:List[Union[LayerStrategy, AttentionStrategy, FFNStrategy]] = field(default_factory=list)
    stage_time_cost:float = 0
    stage_time_cost_no_sync:float = 0

    def get_info(self):
        return self.best_strategy_list

class GalvatronSearchEngine:
    def __init__(self, args=None):
        self.args = args if args is not None else SimpleNamespace(log_dir='./search_engine_logs')
        self.initialize_main_logger()

        self.world_size = None
        self.degree_range = None
        self.gbsz_list = None
    
    # ==========================initialize function=====================
    def initialize_main_logger(self) -> None:
        assert self.args.log_dir is not None, f'When initialize main logger, log_dir should not be None'
        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(self.args.log_dir, time_str)
        log_path = os.path.join(self.log_dir, "controller.log")
        if os.path.exists(log_path):
            os.remove(log_path)
        loguru_logger.add(log_path, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}", level="INFO") 
        loguru_logger.info("main logger initialized")
        loguru_logger.info(f'log path:{log_path}')
        loguru_logger.info(f'args:{self.args}')

    def initialize_cost_model_handler(self, path,  model_layer_configs, model_name):
        loguru_logger.info(f'[WORKFLOW] initialize cost model handler...')

        self.handler = GalvatronCostModelHandler(self.args)
        self.handler.set_cost_model_handler_info(path, model_layer_configs, model_name)
        self.handler.initialize_cost_model_handler(show_handler_info=False)
        self.handler.loguru_logger_handler_info()

    def initialize_search_engine(self):
        loguru_logger.info(f'[WORKFLOW] initialize search engine...')

        # set values
        self.hiddensize_list = self.handler.hiddensize_list
        self.layernum_list = self.handler.layernum_list
        self.total_layernum = sum(self.layernum_list)
        self.seqlen_list = self.handler.seqlen_list
        self.num_layertype = self.handler.num_layertype
        self.hidden_size = self.handler.hiddensize_list[0]
        self.profile_granularity = self.handler.args.profile_granularity
        self.world_size = self.handler.world_size
        self.mixed_precision = self.handler.args.mixed_precision
        self.memory_constraint_in_MB = self.args.memory_constraint * 1024

        self.initialize_degree_range()
        self.generate_strategy_list()
        self.filter_strategy_list()
        self.initialize_gbsz_list()

    def initialize_degree_range(self) -> None:
        loguru_logger.info('[WORKFLOW] Initializing degree info...')

        assert self.world_size is not None, f'When initialize_degree_range, world_size should not be None'
        self.degree_range = []
        deg = 1
        while deg <= self.world_size:
            self.degree_range.append(deg)
            deg *= 2
            
    def initialize_gbsz_list(self) -> None:
        loguru_logger.info('[WORKFLOW] Initializing global batch size list...')

        args = self.args

        # Set Searching BSZs
        if args.settle_bsz is not None and args.settle_bsz > 0:
            self.min_bsz = self.max_bsz = args.settle_bsz
            self.bsz_scale = 0
            self.gbsz_list = [args.settle_bsz]
            loguru_logger.info(f'-----, [Searching Batch Sizes Info], Settle bsz:, {args.settle_bsz}, -----')
        else:
            self.bsz_scale = args.bsz_scale
            self.min_bsz = max(args.min_bsz, self.bsz_scale)
            self.min_bsz = self.min_bsz // self.bsz_scale * self.bsz_scale
            self.max_bsz = int(np.ceil(args.max_bsz / self.bsz_scale) * self.bsz_scale) if args.max_bsz % self.bsz_scale else (args.max_bsz + self.bsz_scale)
            self.gbsz_list = list(range(self.min_bsz, self.max_bsz, self.bsz_scale))
            self.max_bsz = self.gbsz_list[-1]
            loguru_logger.info(f'-----, [Searching Batch Sizes Info], Min bsz: {self.min_bsz}, Max bsz: {self.max_bsz}, bsz_scale: {self.bsz_scale}, -----')

        loguru_logger.info(f'Global batch size list: {self.gbsz_list}')

    # ==========================debug function==========================
    def debug_set_values(self, kv_pair:dict) -> None:
        for key, value in kv_pair.items():
            setattr(self, key, value)

    def debug_set_args(self, kv_pair:dict) -> None:
        for key, value in kv_pair.items():
            setattr(self.args, key, value)

    # ==========================Generate Strategy List==========================
    def generate_strategy_list(self) -> None:
        loguru_logger.info(f'[WORKFLOW] Generating strategy list for world_size={self.world_size}...')
        
        assert self.world_size is not None, f'When generate_strategy_list, world_size should not be None'
        assert self.degree_range is not None, f'When generate_strategy_list, degree_range should not be None'
        assert hasattr(self.args, 'max_tp_deg'), f'When generate_strategy_list, max_tp_size should be set.'
        assert hasattr(self.args, 'default_dp_type'), f'When generate_strategy_list, default_dp_type should be set.'
        assert hasattr(self.args, 'max_sp_deg'), f'When generate_strategy_list, max_sp_size should be set.'
        assert hasattr(self.args, 'max_cp_deg'), f'When generate_strategy_list, max_cp_size should be set.'

        args = self.args
        default_dp_type = args.default_dp_type
        max_tp_deg = args.max_tp_deg
        max_sp_deg = args.max_sp_deg
        max_cp_deg = args.max_cp_deg
        world_size = self.world_size
        degree_range = self.degree_range

        attention_strategy_list:List[AttentionStrategy] = []
        ffn_strategy_list:List[FFNStrategy] = []
        embedding_lmhead_strategy_list:List[EmbeddingLMHeadStrategy] = []
        layer_strategy_list:List[LayerStrategy] = []
        moe_ffn_strategy_list:List[MoEFFNStrategy] = []

        # generate attention strategy list
        for pp_size in degree_range:
            if pp_size > self.total_layernum:
                continue
            for tp_size in degree_range:
                if max_tp_deg != -1 and tp_size > max_tp_deg:
                    continue
                if tp_size * pp_size > world_size:
                    continue
                for sp_size in degree_range:
                    if max_sp_deg != -1 and sp_size > max_sp_deg:
                        continue
                    if sp_size != 1 and tp_size != 1:
                        continue
                    if pp_size * tp_size * sp_size > world_size:
                        continue
                    for cp_size in degree_range:
                        if max_cp_deg != -1 and cp_size > max_cp_deg:
                            continue
                        if pp_size * tp_size * sp_size * cp_size > world_size:
                            continue
                        dp_size = world_size // pp_size // tp_size // sp_size // cp_size
                        dp_type_list = [DPType.DDP] if dp_size == 1 else ([DPType.DDP, DPType.ZERO3] if default_dp_type == 'ddp' else [DPType.ZERO2, DPType.ZERO3])
                        for dp_type in dp_type_list:
                            for checkpoint in [False, True]:
                                strategy = AttentionStrategy(
                                    pp_size=pp_size,
                                    tp_size=tp_size,
                                    sp_size=sp_size,
                                    cp_size=cp_size,
                                    dp_size=dp_size,
                                    dp_type=dp_type,
                                    checkpoint=checkpoint,
                                )
                                attention_strategy_list.append(strategy)
        attention_strategy_list = sorted(list(set(attention_strategy_list)))

        # generate ffn/embedding_lmhead/layer strategy list from attention strategy list
        for strategy in attention_strategy_list:
            ffn_strategy_list.append(strategy.to_ffn_strategy())
            embedding_lmhead_strategy_list.append(strategy.to_embedding_lmhead_strategy())
            layer_strategy_list.append(strategy.to_layer_strategy())
        ffn_strategy_list = sorted(list(set(ffn_strategy_list)))
        embedding_lmhead_strategy_list = sorted(list(set(embedding_lmhead_strategy_list)))
        layer_strategy_list = sorted(list(set(layer_strategy_list)))

        # generate moe_ffn strategy list
        for pp_size in degree_range:
            if pp_size > self.total_layernum:
                continue
            for ep_size in degree_range:
                if pp_size * ep_size > world_size:
                    continue
                for tp_size in degree_range:
                    if pp_size * ep_size * tp_size > world_size:
                        continue
                    dp_size = world_size // pp_size // tp_size // ep_size
                    dp_type_list = [DPType.DDP] if dp_size == 1 else ([DPType.DDP, DPType.ZERO3] if default_dp_type == 'ddp' else [DPType.ZERO2, DPType.ZERO3])
                    for dp_type in dp_type_list:
                        for checkpoint in [False, True]:
                            strategy = MoEFFNStrategy(
                                pp_size=pp_size,
                                ep_size=ep_size,
                                tp_size=tp_size,
                                dp_size=dp_size,
                                dp_type=dp_type,
                                checkpoint=checkpoint,
                            )
                            moe_ffn_strategy_list.append(strategy)
        moe_ffn_strategy_list = sorted(list(set(moe_ffn_strategy_list)))
        
        self.embedding_lmhead_strategy_list = embedding_lmhead_strategy_list
        self.attention_strategy_list = attention_strategy_list
        self.ffn_strategy_list = ffn_strategy_list
        self.layer_strategy_list = layer_strategy_list
        self.moe_ffn_strategy_list = moe_ffn_strategy_list

        # logger
        loguru_logger.info(f'attention_strategy_list.size:{len(self.attention_strategy_list)}')
        loguru_logger.info(f'ffn_strategy_list.size:{len(self.ffn_strategy_list)}')
        loguru_logger.info(f'embedding_lmhead_strategy_list.size:{len(self.embedding_lmhead_strategy_list)}')
        loguru_logger.info(f'layer_strategy_list.size:{len(self.layer_strategy_list)}')
        loguru_logger.info(f'moe_ffn_strategy_list.size:{len(self.moe_ffn_strategy_list)}')

        loguru_logger.info(f'attention:\n{pretty_repr(self.attention_strategy_list, max_width=1024)}')
        loguru_logger.info(f'ffn:\n{pretty_repr(self.ffn_strategy_list, max_width=1024)}')
        loguru_logger.info(f'embedding_lmhead:\n{pretty_repr(self.embedding_lmhead_strategy_list, max_width=1024)}')
        loguru_logger.info(f'layer:\n{pretty_repr(self.layer_strategy_list, max_width=1024)}')
        loguru_logger.info(f'moe:\n{pretty_repr(self.moe_ffn_strategy_list, max_width=1024)}')

        loguru_logger.info(f'Generate strategy list done.')

    def filter_strategy_list(self, disable_pp=None, disable_tp=None, disable_sp=None, disable_cp=None, disable_dp=None, disable_ckpt=None, disable_fsdp=None, disable_embedding_lmhead_tp=None, disable_embedding_lmhead_sp=None):
        loguru_logger.info(f'[WORKFLOW] Filtering Strategy...')
        args = self.args

        params = {
            "disable_pp": disable_pp,
            "disable_tp": disable_tp,
            "disable_sp": disable_sp,
            "disable_cp": disable_cp,
            "disable_dp": disable_dp,
            "disable_ckpt": disable_ckpt,
            "disable_fsdp": disable_fsdp,
            "disable_embedding_lmhead_tp": disable_embedding_lmhead_tp,
            "disable_embedding_lmhead_sp": disable_embedding_lmhead_sp
        }
        
        for name, value in params.items():
            if value is not None:
                setattr(args, name, value)

        if args.disable_pp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.pp_size == 1]
            self.attention_strategy_list = [strategy for strategy in self.attention_strategy_list if strategy.pp_size == 1]
            self.ffn_strategy_list = [strategy for strategy in self.ffn_strategy_list if strategy.pp_size == 1]
            self.layer_strategy_list = [strategy for strategy in self.layer_strategy_list if strategy.pp_size == 1]
            self.moe_ffn_strategy_list = [strategy for strategy in self.moe_ffn_strategy_list if strategy.pp_size == 1]
        if args.disable_tp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.tp_size == 1]
            self.attention_strategy_list = [strategy for strategy in self.attention_strategy_list if strategy.tp_size == 1]
            self.ffn_strategy_list = [strategy for strategy in self.ffn_strategy_list if strategy.tp_size == 1]
            self.layer_strategy_list = [strategy for strategy in self.layer_strategy_list if strategy.tp_size == 1]
            self.moe_ffn_strategy_list = [strategy for strategy in self.moe_ffn_strategy_list if strategy.tp_size == 1]
        if args.disable_sp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.sp_size == 1]
            self.attention_strategy_list = [strategy for strategy in self.attention_strategy_list if strategy.sp_size == 1]
            self.layer_strategy_list = [strategy for strategy in self.layer_strategy_list if strategy.sp_size == 1]
            self.ffn_strategy_list = [strategy for strategy in self.ffn_strategy_list if strategy.sp_size == 1]
        if args.disable_cp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.cp_size == 1]
            self.attention_strategy_list = [strategy for strategy in self.attention_strategy_list if strategy.cp_size == 1]
            self.layer_strategy_list = [strategy for strategy in self.layer_strategy_list if strategy.cp_size == 1]
            self.ffn_strategy_list = [strategy for strategy in self.ffn_strategy_list if strategy.cp_size == 1]
        if args.disable_dp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.dp_size == 1]
            self.attention_strategy_list = [strategy for strategy in self.attention_strategy_list if strategy.dp_size == 1]
            self.ffn_strategy_list = [strategy for strategy in self.ffn_strategy_list if strategy.dp_size == 1]
            self.layer_strategy_list = [strategy for strategy in self.layer_strategy_list if strategy.dp_size == 1]
            self.moe_ffn_strategy_list = [strategy for strategy in self.moe_ffn_strategy_list if strategy.dp_size == 1]
        if args.disable_ckpt:
            self.attention_strategy_list = [strategy for strategy in self.attention_strategy_list if strategy.checkpoint == False]
            self.ffn_strategy_list = [strategy for strategy in self.ffn_strategy_list if strategy.checkpoint == False]
            self.layer_strategy_list = [strategy for strategy in self.layer_strategy_list if strategy.checkpoint == False]
            self.moe_ffn_strategy_list = [strategy for strategy in self.moe_ffn_strategy_list if strategy.checkpoint == False]
        if args.disable_fsdp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.dp_type != DPType.ZERO3]
            self.attention_strategy_list = [strategy for strategy in self.attention_strategy_list if strategy.dp_type != DPType.ZERO3]
            self.ffn_strategy_list = [strategy for strategy in self.ffn_strategy_list if strategy.dp_type != DPType.ZERO3]
            self.layer_strategy_list = [strategy for strategy in self.layer_strategy_list if strategy.dp_type != DPType.ZERO3]
            self.moe_ffn_strategy_list = [strategy for strategy in self.moe_ffn_strategy_list if strategy.dp_type != DPType.ZERO3]
        if args.disable_embedding_lmhead_tp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.tp_size == 1]
        if args.disable_embedding_lmhead_sp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.sp_size == 1]

        self.embedding_lmhead_strategy_list = sorted(list(set(self.embedding_lmhead_strategy_list)))
        self.attention_strategy_list = sorted(list(set(self.attention_strategy_list)))
        self.ffn_strategy_list = sorted(list(set(self.ffn_strategy_list)))
        self.layer_strategy_list = sorted(list(set(self.layer_strategy_list)))
        self.moe_ffn_strategy_list = sorted(list(set(self.moe_ffn_strategy_list)))

        loguru_logger.info(f'After filter')
        loguru_logger.info(f'attention_strategy_list.size:{len(self.attention_strategy_list)}')
        loguru_logger.info(f'ffn_strategy_list.size:{len(self.ffn_strategy_list)}')
        loguru_logger.info(f'embedding_lmhead_strategy_list.size:{len(self.embedding_lmhead_strategy_list)}')
        loguru_logger.info(f'layer_strategy_list.size:{len(self.layer_strategy_list)}')
        loguru_logger.info(f'moe_ffn_strategy_list.size:{len(self.moe_ffn_strategy_list)}')

        loguru_logger.info(f'attention:\n{pretty_repr(self.attention_strategy_list, max_width=1024)}')
        loguru_logger.info(f'ffn:\n{pretty_repr(self.ffn_strategy_list, max_width=1024)}')
        loguru_logger.info(f'embedding_lmhead:\n{pretty_repr(self.embedding_lmhead_strategy_list, max_width=1024)}')
        loguru_logger.info(f'layer:\n{pretty_repr(self.layer_strategy_list, max_width=1024)}')
        loguru_logger.info(f'moe:\n{pretty_repr(self.moe_ffn_strategy_list, max_width=1024)}')

        loguru_logger.info(f'Filter strategy done.')

    # ==========================Parallelism Optimization==========================
    def get_pp_size_range(self) -> None:
        loguru_logger.info(f'[WORKFLOW] Geting pp size range...')
        self.pp_size_range = []
        assert hasattr(self, 'embedding_lmhead_strategy_list'), f"{ColorSet.RED}[ERROR] [{self.__class__.__name__}] embedding_lmhead_strategy_list is not set.{ColorSet.RESET}"
        for strategy in self.embedding_lmhead_strategy_list:
            self.pp_size_range.append(strategy.pp_size)
        self.pp_size_range = sorted(list(set(self.pp_size_range)))
        loguru_logger.info(f'pp size range: {self.pp_size_range}')

    def parallelism_optimization(self):
        loguru_logger.info(f'{"=" * 25} Galvatron Search Engine Start Searching {"=" * 25}')
            
        # [Step 1] Preparation Works
        args = self.args
        self.get_pp_size_range()
        self.tp_sp_space = ['tp', 'sp', 'tp+sp']
        max_tp_deg = args.max_tp_deg

        # [Step 2] get all possible 
        results = dict()
        all_tasks = []
        for gbsz in self.gbsz_list:
            results[gbsz] = dict()
            chunk_list = range(1, gbsz + 1)
            for chunks in chunk_list:
                if gbsz % chunks != 0:
                    continue
                results[gbsz][chunks] = dict()
                for pp_size in self.pp_size_range:
                    if chunks < pp_size:
                        loguru_logger.info(f'chunks({chunks}) < pp_size({pp_size}), skip')
                        continue
                    results[gbsz][chunks][pp_size] = dict()
                    for tp_sp_mode in self.tp_sp_space:
                        results[gbsz][chunks][pp_size][tp_sp_mode] = dict()
                        if tp_sp_mode == 'tp' or tp_sp_mode == 'tp+sp': # tp only
                            # Actually, the strategy set for 'tp+sp' is a superset of the above 'tp-only' cases and 'sp-only' case,
                            # but the above strategies all require that SP and TP cannot coexist,
                            # i.e., TP cannot be used in this layer and SP in the next layer.
                            # Therefore, the 'tp-only' cases and 'sp-only' case are simpler, and considering them separately
                            # would be more convenient for debugging and observation,
                            # so the above two methods are retained.
                            theoretical_max_tp_size = self.world_size // pp_size
                            if max_tp_deg != -1 and theoretical_max_tp_size > max_tp_deg:
                                theoretical_max_tp_size = max_tp_deg
                            for tp_size_limit in range(1, theoretical_max_tp_size + 1):
                                if not is_power_of_two(tp_size_limit):
                                    continue
                                results[gbsz][chunks][pp_size][tp_sp_mode][tp_size_limit] = dict()
                                all_tasks.append((gbsz, chunks, pp_size, tp_sp_mode, tp_size_limit))
                        elif tp_sp_mode == 'sp': # sp only
                            tp_size_limit = -1
                            results[gbsz][chunks][pp_size][tp_sp_mode][tp_size_limit] = dict()
                            all_tasks.append((gbsz, chunks, pp_size, tp_sp_mode, tp_size_limit))

        # [Step 3] search
        if args.parallel_search:
            import concurrent.futures
            import threading
            import multiprocessing
            import copy
            
            if hasattr(self.args, 'worker') and self.args.worker > 0:
                num_threads = min(self.args.worker, len(all_tasks))
            else:
                num_threads = min(multiprocessing.cpu_count() * 2, len(all_tasks))
                print(f"Starting parallel search with {num_threads} threads for {len(all_tasks)} tasks...")
                
                results_lock = threading.Lock()
                def process_task(gbsz, chunk, pp_size, tp_sp_mode, tp_size_limit):
                    thread_id = threading.get_ident() % 1000
                    print(f"[Thread {thread_id:03d}] Start processing: gbsz={gbsz}, chunk={chunk}, pp_size={pp_size}, tp_sp_mode={tp_sp_mode}, tp_size_limit={tp_size_limit}", flush=True)
                    try:
                        throughput, detailed_info = self.parallelism_optimization_for_single_task(gbsz, chunk, pp_size, tp_sp_mode, tp_size_limit)
                        with results_lock:
                            results[gbsz][chunk][pp_size][tp_sp_mode][tp_size_limit] = {
                                'throughput': throughput,
                                'detailed_info': copy.deepcopy(detailed_info)  # Deep copy to avoid thread safety issues
                            }
                        print(f"[Thread {thread_id:03d}] Completed: gbsz={gbsz}, chunk={chunk}, pp_size={pp_size}, tp_sp_mode={tp_sp_mode}, tp_size_limit={tp_size_limit}, throughput={throughput}", flush=True)
                    except Exception as e:
                        error_msg = f"[Thread {thread_id:03d}] Error processing gbsz={gbsz}, chunk={chunk}, pp_size={pp_size}, tp_sp_mode={tp_sp_mode}, tp_size_limit={tp_size_limit}: {str(e)}"
                        print(error_msg, flush=True)
                        loguru_logger.error(error_msg, exc_info=True)
                        with results_lock:
                            # Mark as failed task
                            results[gbsz][chunk][pp_size][tp_sp_mode][tp_size_limit] = {
                                'throughput': -1,
                                'detailed_info': None,
                                'error': str(e)
                            }
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(process_task, gbsz, chunk, pp_size, tp_sp_mode, tp_size_limit) for gbsz, chunk, pp_size, tp_sp_mode, tp_size_limit in all_tasks]
                    # Wait for all tasks and check for exceptions
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result()  # This will raise any exception that occurred
                        except Exception as e:
                            loguru_logger.error(f"Task failed with exception: {str(e)}", exc_info=True)
        else:
            for task in all_tasks:
                gbsz, chunk, pp_size, tp_sp_mode, tp_size_limit = task
                throughput, detailed_info = self.parallelism_optimization_for_single_task(gbsz, chunk, pp_size, tp_sp_mode, tp_size_limit)
                results[gbsz][chunk][pp_size][tp_sp_mode][tp_size_limit] = {
                    'throughput': throughput,
                    'detailed_info': detailed_info
                }
        
        # [Step 4] Select the optimal solution and save results
        max_throughput, optimal_detailed_info = -1, None
        for gbsz in results:
            for chunk in results[gbsz]:
                for pp_size in results[gbsz][chunk]:
                    for tp_sp_mode in results[gbsz][chunk][pp_size]:
                        for tp_size_limit in results[gbsz][chunk][pp_size][tp_sp_mode]:
                            res = results[gbsz][chunk][pp_size][tp_sp_mode][tp_size_limit]
                            # Skip empty dicts (initialized but not processed) and failed tasks
                            if not res or res.get('throughput', -1) < 0:
                                if 'error' in res:
                                    loguru_logger.warning(f'gbsz={gbsz}, chunk={chunk}, pp_size={pp_size}, tp_sp_mode={tp_sp_mode}, tp_size_limit={tp_size_limit} failed: {res.get("error")}')
                                continue
                            loguru_logger.info(f'gbsz={gbsz}, chunk={chunk}, pp_size={pp_size}, tp_sp_mode={tp_sp_mode}, tp_size_limit={tp_size_limit}, throughput={res.get("throughput")}')
                            if 'throughput' in res and res['throughput'] > max_throughput:
                                max_throughput = res['throughput']
                                optimal_detailed_info = res['detailed_info']
        
        if max_throughput > 0:
            print(f'please print something and store config file')
            self.save_results(optimal_detailed_info)
        else:
            print("No valid configuration found.")
        
        loguru_logger.info("-----------------------------------------")
        loguru_logger.info(f'{"=" * 25} Galvatron Search Engine End Searching {"=" * 25}')

    def parallelism_optimization_for_single_task(self, gbsz, chunks, pp_size, tp_sp_mode, tp_size_limit):
        loguru_logger.info(f'start processing: gbsz={gbsz}, chunks={chunks}, pp_size={pp_size}, tp_sp_mode={tp_sp_mode}, tp_size_limit={tp_size_limit}...')

        thread_logger = get_thread_logger(gbsz, chunks, pp_size, tp_sp_mode, tp_size_limit, log_dir=self.log_dir)
        thread_logger.info(f"start processing: gbsz={gbsz}, chunks={chunks}, pp_size={pp_size}, tp_sp_mode={tp_sp_mode}, tp_size_limit={tp_size_limit}...")

        assert self.seqlen_list is not None, f'When parallelism_optimization_for_single_task, seqlen_list should not be None'
        assert self.mixed_precision is not None, f'When parallelism_optimization_for_single_task, mixed_precision should not be None'
        assert self.hidden_size is not None, f'When parallelism_optimization_for_single_task, hidden_size should not be None'

        # calculate other limitation
        theoretical_max_dp_size = min(gbsz // chunks, self.world_size // pp_size) # gbsz = chunk * dp_size * 1, where 1 is mbsz

        def invalid_strategy(strategy:Union[EmbeddingLMHeadStrategy, LayerStrategy, AttentionStrategy, FFNStrategy]):
            if strategy.pp_size != pp_size or strategy.dp_size > theoretical_max_dp_size:
                return True
            if tp_size_limit != -1 and strategy.tp_size > tp_size_limit:
                return True
            if tp_sp_mode == 'tp' and strategy.sp_size != 1:
                return True
            if tp_sp_mode == 'sp' and strategy.tp_size != 1:
                return True

        # calculate global_memory_buffer_size_in_MB
        if tp_size_limit == -1:
            global_memory_buffer_size_in_MB = 0
        else:
            # For layers with tp_size = tp_size_limit, we have:
            curr_dp_cp_size = self.world_size // pp_size // tp_size_limit
            dtype_size = 2 if self.mixed_precision else 4
            global_memory_buffer_size_in_MB = (gbsz // chunks) * max(self.seqlen_list) * self.hidden_size / curr_dp_cp_size * dtype_size / byte_to_MB

        # filter embedding_lmhead strategy
        embedding_lmhead_strategy_pool:List[EmbeddingLMHeadStrategy] = []
        for strategy in self.embedding_lmhead_strategy_list:
            if invalid_strategy(strategy):
                continue
            embedding_lmhead_strategy_pool.append(copy.deepcopy(strategy))
        embedding_lmhead_strategy_pool = sorted(list(set(embedding_lmhead_strategy_pool)))

        if self.profile_granularity == 'split':
            # filter strategy
            attention_strategy_pool:List[AttentionStrategy] = []
            ffn_strategy_pool:List[FFNStrategy] = []
            for strategy in self.attention_strategy_list:
                if invalid_strategy(strategy):
                    continue
                attention_strategy_pool.append(copy.deepcopy(strategy))
            attention_strategy_pool = sorted(list(set(attention_strategy_pool)))

            for strategy in self.ffn_strategy_list:
                if invalid_strategy(strategy):
                    continue
                ffn_strategy_pool.append(copy.deepcopy(strategy))
            ffn_strategy_pool = sorted(list(set(ffn_strategy_pool)))

            throughput, detailed_info = self.search_for_fix_situation(
                gbsz, chunks, pp_size,
                embedding_lmhead_strategy_pool=embedding_lmhead_strategy_pool,
                layer_strategy_pool=None,
                attention_strategy_pool=attention_strategy_pool,
                ffn_strategy_pool=ffn_strategy_pool,
                global_memory_buffer_size_in_MB=global_memory_buffer_size_in_MB, 
                thread_logger=thread_logger,
            )
        elif self.profile_granularity == 'together':
            layer_strategy_pool:List[LayerStrategy] = []
            for strategy in self.layer_strategy_list:
                if invalid_strategy(strategy):
                    continue
                layer_strategy_pool.append(copy.deepcopy(strategy))
            layer_strategy_pool = sorted(list(set(layer_strategy_pool)))
            throughput, detailed_info = self.search_for_fix_situation(
                gbsz, chunks, pp_size,
                embedding_lmhead_strategy_pool=embedding_lmhead_strategy_pool,
                layer_strategy_pool=layer_strategy_pool,
                attention_strategy_pool=None, ffn_strategy_pool=None,
                global_memory_buffer_size_in_MB=global_memory_buffer_size_in_MB,
                thread_logger=thread_logger
            )

        return throughput, detailed_info

    def search_for_fix_situation(self, gbsz:int, chunks:int, pp_size:int, 
        embedding_lmhead_strategy_pool:List[EmbeddingLMHeadStrategy], 
        layer_strategy_pool:List[LayerStrategy], 
        attention_strategy_pool:List[AttentionStrategy], 
        ffn_strategy_pool:List[FFNStrategy], 
        global_memory_buffer_size_in_MB:float, 
        thread_logger=None
    ):
        assert thread_logger is not None, f'thread_logger should not be none.'

        thread_logger.info(f'enter search_for_fix_situation')
        thread_logger.info(f'embedding_lmhead_strategy_pool:\n {pretty_repr(embedding_lmhead_strategy_pool, max_width=1024)}')
        thread_logger.info(f'layer_strategy_pool:\n {pretty_repr(layer_strategy_pool, max_width=1024)}')
        thread_logger.info(f'attention_strategy_pool:\n {pretty_repr(attention_strategy_pool, max_width=1024)}')
        thread_logger.info(f'ffn_strategy_pool:\n {pretty_repr(ffn_strategy_pool, max_width=1024)}')
        
        if len(embedding_lmhead_strategy_pool) == 0:
            thread_logger.info(f'embedding_lmhead_strategy_pool is empty, exit')
            return -1, {}
        elif layer_strategy_pool is not None and len(layer_strategy_pool) == 0:
            thread_logger.info(f'layer_strategy_pool is empty, exit')
            return -1, {}
        elif attention_strategy_pool is not None and len(attention_strategy_pool) == 0:
            thread_logger.info(f'attention_strategy_pool is empty, exit')
            return -1, {}
        elif ffn_strategy_pool is not None and len(ffn_strategy_pool) == 0:
            thread_logger.info(f'ffn_strategy_pool is empty, exit')
            return -1, {}

        max_throughput, detailed_info = -1, {}

        # [Step 1] prepare info_per_stage
        info_per_stage:List[StageInfo] = []
        layertype_id_list_per_stage, layer_desc_list_per_stage = get_pp_division(pp_size, self.layernum_list)

        for stage_idx in range(pp_size):
            info = StageInfo()
            info.stage_idx = stage_idx
            info.layernum = len(layertype_id_list_per_stage[stage_idx])
            info.layertype_id_list = layertype_id_list_per_stage[stage_idx]
            info.layer_desc_list = layer_desc_list_per_stage[stage_idx]
            info_per_stage.append(info)

        # [Step 2] Get time cost of embedding_lmhead_strategy_pool.
        embedding_lmhead_time_cost_pool:List[Tuple[float, float]] = []
        embedding_lmhead_no_sync_time_cost_pool:List[Tuple[float, float]] = []
        for strategy in embedding_lmhead_strategy_pool:
            time_cost, time_no_sync_cost = self.handler.get_time_cost(strategy=strategy, global_batch_size=gbsz, chunks=chunks) # tuple[tuple[float, float], tuple[float, float]]
            embedding_lmhead_time_cost_pool.append(time_cost)
            embedding_lmhead_no_sync_time_cost_pool.append(time_no_sync_cost)

        # [Step 3] Get time cost of layer_strategy_pool.
        layer_time_cost_pool:List[List[float]] = [[] for _ in range(self.num_layertype)]
        layer_time_no_sync_cost_pool:List[List[float]] = [[] for _ in range(self.num_layertype)]
        for layertype_id in range(self.num_layertype):
            for strategy in layer_strategy_pool:
                time_cost, time_no_sync_cost = self.handler.get_time_cost(strategy=strategy, global_batch_size=gbsz, chunks=chunks, layertype_id=layertype_id)
                layer_time_cost_pool[layertype_id].append(time_cost)
                layer_time_no_sync_cost_pool[layertype_id].append(time_no_sync_cost)

        # [Step 4] Simulate inter-layer time overhead based on empirical rules.  # TODO Reconsider the overhead of switching between different strategies
        inter_layer_cost_list:List[List[List[float]]] = []
        layer_strategy_pool_size = len(layer_strategy_pool)
        for layertype_id in range(self.num_layertype):
            inter_layer_cost = [[0 for _ in range(layer_strategy_pool_size)] for _ in range(layer_strategy_pool_size)]
            inter_layer_cost_list.append(inter_layer_cost)

        # [Step 5] Get memory cost of embedding_lmhead_strategy_pool.
        embedding_memory_cost_pool:List[int] = []
        lmhead_memory_cost_pool:List[int] = []
        for strategy in embedding_lmhead_strategy_pool:
            memory_cost:Tuple[float, float] = self.handler.get_memory_cost(strategy=strategy, global_batch_size=gbsz, chunks=chunks) # tuple[float, float]
            embedding_memory_cost_pool.append(math.ceil(memory_cost[0]))
            lmhead_memory_cost_pool.append(math.ceil(memory_cost[1]))

        # [Step 6] Get memory cost of layer_strategy_pool.
        layer_memory_cost_pool:List[List[List[int]]] = [[[] for _ in range(self.num_layertype)] for _ in range(pp_size)] # dim(stage_idx, layertype_id, strategy_idx)
        for stage_idx in range(pp_size):
            for layertype_id in range(self.num_layertype):
                for strategy in layer_strategy_pool:
                    memory_cost:float = self.handler.get_memory_cost(strategy, global_batch_size=gbsz, chunks=chunks, layertype_id=layertype_id, stage_idx=stage_idx)
                    layer_memory_cost_pool[stage_idx][layertype_id].append(math.ceil(memory_cost))

        # [Step 7] Enumerate all strategy in embedding_lmhead_strategy_pool to ensure both the embedding and lmhead use the same strategy
        for embedding_lmhead_strategy_idx in range(len(embedding_lmhead_strategy_pool)):
            no_solution = False

            # [Step 7.1] solve using dynamic program for each stage
            for stage_idx in range(pp_size):
                info:StageInfo = info_per_stage[stage_idx]

                strategy_num_list_:List[int] = []
                layer_time_cost_list_:List[List[float]] = []
                layer_memory_cost_list_:List[List[int]] = []
                inter_layer_time_cost_list_:List[List[List[float]]] = []
                
                for layertype_id in info.layertype_id_list:
                    strategy_num_list_.append(len(layer_time_cost_pool[layertype_id]))
                    layer_time_cost_list_.append(layer_time_cost_pool[layertype_id])
                    layer_memory_cost_list_.append(layer_memory_cost_pool[stage_idx][layertype_id])
                    inter_layer_time_cost_list_.append(inter_layer_cost_list[layertype_id])

                extra_memory_cost_, extra_time_cost_ = global_memory_buffer_size_in_MB, 0
                if stage_idx == 0:
                    extra_memory_cost_ += embedding_memory_cost_pool[embedding_lmhead_strategy_idx]
                    extra_time_cost_ += embedding_lmhead_time_cost_pool[embedding_lmhead_strategy_idx][0] # 0:embedding
                if stage_idx == pp_size - 1:
                    extra_memory_cost_ += lmhead_memory_cost_pool[embedding_lmhead_strategy_idx]
                    extra_time_cost_ += embedding_lmhead_time_cost_pool[embedding_lmhead_strategy_idx][0] # 1:lmhead
                extra_memory_cost_ = math.ceil(extra_memory_cost_)

                dp_solver = DynamicProgramming(
                    layernum=info.layernum,
                    strategy_num_list=strategy_num_list_,
                    layer_time_cost_list=layer_time_cost_list_,
                    layer_memory_cost_list=layer_memory_cost_list_,
                    inter_layer_time_cost_list=inter_layer_time_cost_list_,
                    extra_memory_cost=extra_memory_cost_,
                    extra_time_cost=extra_time_cost_,
                    memory_constraint_in_MB=int(self.memory_constraint_in_MB * 0.8),
                )

                best_time, best_strategy_idx_list, best_memory = dp_solver.fit()

                if best_memory == -1:
                    thread_logger.info(f'When embedding_lmhead_strategy_idx is {embedding_lmhead_strategy_idx}, stage {stage_idx} have no solution.')
                    no_solution = True
                    break # early stop when no solution
                else:
                    info.best_time = best_time
                    info.best_memory = best_memory
                    info.best_strategy_list = []
                    for idx in best_strategy_idx_list:
                        info.best_strategy_list.append(layer_strategy_pool[idx])

                    info.stage_time_cost = 0
                    info.stage_time_cost_no_sync = 0
                    if stage_idx == 0:
                        info.stage_time_cost += embedding_lmhead_time_cost_pool[embedding_lmhead_strategy_idx][0] # 0:embedding
                        info.stage_time_cost_no_sync += embedding_lmhead_no_sync_time_cost_pool[embedding_lmhead_strategy_idx][0] # 0:embedding
                    if stage_idx == pp_size - 1:
                        info.stage_time_cost += embedding_lmhead_time_cost_pool[embedding_lmhead_strategy_idx][1] # 0:embedding
                        info.stage_time_cost_no_sync += embedding_lmhead_no_sync_time_cost_pool[embedding_lmhead_strategy_idx][1] # 0:embedding
                    for i, layertype_id in enumerate(info.layertype_id_list):
                        idx = best_strategy_idx_list[i]
                        info.stage_time_cost += layer_time_cost_pool[layertype_id][idx]
                        info.stage_time_cost_no_sync += layer_time_no_sync_cost_pool[layertype_id][idx]

            if no_solution:
                continue
            else:
                # [Step 7.2] pipeline stage cost
                if pp_size == 1:
                    info = info_per_stage[0]
                    total_time = info.stage_time_cost_no_sync * (chunks - 1) + info.stage_time_cost
                else:
                    p2p_communication_time = self.handler.get_p2p_time(global_batch_size=gbsz, chunks=chunks, pp_size=pp_size)
                    total_time = info_per_stage[pp_size - 1].stage_time_cost_no_sync * (chunks - 1)
                    for stage_idx in range(pp_size):
                        total_time += info_per_stage[stage_idx].stage_time_cost
                        if stage_idx != pp_size - 1:
                            total_time += p2p_communication_time
                
                # [Step 7.3] calculate throughput and update detailed_info
                throughput = gbsz / total_time
                thread_logger.info(f'embedding_lmhead_strategy_idx={embedding_lmhead_strategy_idx}, throughput={throughput}, total_time={total_time}')
                if throughput > max_throughput:
                    thread_logger.info(f'update max_throughput from {max_throughput} to {throughput}')
                    max_throughput = throughput
                    detailed_info = {
                        'gbsz': gbsz,
                        'chunks': chunks,
                        'pp_size': pp_size,
                        'throughput': throughput,
                        'time': total_time,
                        'embedding_lmhead_strategy': copy.deepcopy(embedding_lmhead_strategy_pool[embedding_lmhead_strategy_idx]),
                        'info_per_stage': [copy.deepcopy(info.get_info()) for info in info_per_stage]
                    }

        thread_logger.info('exit search_for_fix_situation')
        return max_throughput, detailed_info

    def save_results(self, optimal_detailed_info):
        data = {}
        data['global_bsz'] = optimal_detailed_info['gbsz']
        data['chunks'] = optimal_detailed_info['chunks']
        data['pp_size'] = optimal_detailed_info['pp_size']
        data['throughput'] = optimal_detailed_info['throughput']
        data['time'] = optimal_detailed_info['time']

        embedding_lmhead_strategy:EmbeddingLMHeadStrategy = optimal_detailed_info['embedding_lmhead_strategy']
        data['embedding_lmhead_dp_size'] = embedding_lmhead_strategy.dp_size
        data['embedding_lmhead_tp_size'] = embedding_lmhead_strategy.tp_size
        data['embedding_lmhead_sp_size'] = embedding_lmhead_strategy.sp_size
        data['embedding_lmhead_cp_size'] = embedding_lmhead_strategy.cp_size
        if embedding_lmhead_strategy.dp_type == DPType.DDP:
            data['embedding_lmhead_dp_type'] = 'ddp'
        elif embedding_lmhead_strategy.dp_type == DPType.ZERO2:
            data['embedding_lmhead_dp_type'] = 'zero2'
        elif embedding_lmhead_strategy.dp_type == DPType.ZERO3:
            data['embedding_lmhead_dp_type'] = 'zero3'

        data['default_dp_type'] = self.args.default_dp_type

        dp_size_enc_list = []
        tp_size_enc_list = []
        sp_size_enc_list = []
        cp_size_enc_list = []
        checkpoint_enc_list = []
        dp_type_enc_list = []
        for strategy_list in optimal_detailed_info['info_per_stage']:
            for idx in range(len(strategy_list)):
                strategy:List[Union[LayerStrategy, AttentionStrategy, FFNStrategy]] = strategy_list[idx]
                dp_size_enc_list.append(strategy.dp_size)
                tp_size_enc_list.append(strategy.tp_size)
                sp_size_enc_list.append(strategy.sp_size)
                cp_size_enc_list.append(strategy.cp_size)
                checkpoint_enc_list.append(1 if strategy.checkpoint is True else 0)
                if strategy.dp_type == DPType.ZERO3:
                    dp_type_enc_list.append(1)
                else:
                    dp_type_enc_list.append(0)

        data['dp_size_enc'] = ','.join(map(str, dp_size_enc_list))
        data['tp_size_enc'] = ','.join(map(str, tp_size_enc_list))
        data['sp_size_enc'] = ','.join(map(str, sp_size_enc_list))
        data['cp_size_enc'] = ','.join(map(str, cp_size_enc_list))
        data['checkpoint_enc'] = ','.join(map(str, checkpoint_enc_list))
        data['dp_type_enc'] = ','.join(map(str, dp_type_enc_list))
        loguru_logger.info(f'data:\n{pretty_repr(data, max_width=1024)}')

        file_name = f'{self.handler.model_name}_{self.args.num_nodes}nodes_{self.args.num_gpus_per_node}gpus_{self.memory_constraint_in_MB}GB'
        json_file_path = os.path.join(self.args.output_config_path, f'galvatron_config_{file_name}.json')
        json.dump(data, open(json_file_path, 'w'), indent=4)
        loguru_logger.info(f'save results to {json_file_path}')
    
def get_pp_division(pp_size:int, layernum_list:List[int]) -> Tuple[List[List[int]], List[List[str]]]:
    num_layertype = len(layernum_list)

    if num_layertype == 1: # decoder only
        layertype_id_list = [0 for _ in range(layernum_list[0])]
        layer_desc_list = [f'decoder_{i}' for i in range(layernum_list[0])]
    elif num_layertype == 2:
        layertype_id_list = [0 for _ in range(layernum_list[0])] + [1 for _ in range(layernum_list[1])]
        layer_desc_list = [f'encoder_{i}' for i in range(layernum_list[0])] + [f'decoder_{i}' for i in range(layernum_list[1])]
    else:
        raise ValueError(f'Invalid num_layertype: {num_layertype}')

    total_layer_num = sum(layernum_list)

    # (4, [17]) 17 // 4 = 4, 17 = 4 + 4 + 4 + 5
    # (4, [15]) 15 // 4 = 3, 15 = 3 + 3 + 3 + 6
    avg_layer_num = int(total_layer_num // pp_size)
    last_layer_num = total_layer_num - avg_layer_num * (pp_size - 1)

    # # (4, [17]) 17 + 4 - 1 = 20 20 // 4 = 5, 17 = 5 + 5 + 5 + 2
    # # (4, [15]) 15 + 4 - 1 = 18 18 // 4 = 4, 15 = 4 + 4 + 4 + 3
    # avg_layer_num = (total_layer_num + pp_size - 1) // pp_size
    # last_layer_num = total_layer_num - avg_layer_num * (pp_size - 1)

    layertype_id_list_per_stage = [[] for _ in range(pp_size)]
    layer_desc_list_per_stage = [[] for _ in range(pp_size)]
    for stage_idx in range(pp_size):
        start_idx = avg_layer_num * stage_idx
        last_idx = start_idx + avg_layer_num if stage_idx != pp_size - 1 else start_idx + last_layer_num
        layertype_id_list_per_stage[stage_idx] = layertype_id_list[start_idx:last_idx]
        layer_desc_list_per_stage[stage_idx] = layer_desc_list[start_idx:last_idx]
    return layertype_id_list_per_stage, layer_desc_list_per_stage