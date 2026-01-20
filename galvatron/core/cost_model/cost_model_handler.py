from galvatron.utils import read_json_config, num2str
from scipy.optimize import curve_fit
import os
from galvatron.utils import (
    read_allreduce_bandwidth_config, 
    read_json_config, 
    read_p2p_bandwidth_config, 
    remap_config,
    num2str
)
from typing import List
import re
import logging
import copy
from .cost_model_args import TrainArgs, ProfileModelArgs, ProfileHardwareArgs, UtilsArgs, VersionOptionArgs, EstimateTPTimeType
from .components.attention_cost import AttentionTimeCostModel, AttentionMemoryCostModel
from .components.ffn_cost import FFNTimeCostModel, FFNMemoryCostModel
from .components.layer_cost import LayerTimeCostModel, LayerMemoryCostModel
from .components.embedding_lmhead_cost import EmbeddingLMHeadTimeCostModel, EmbeddingLMHeadMemoryCostModel
from .components.pipeline_cost import PipelineTimeCostModel
from typing import Union, Tuple
from galvatron.utils.strategy_utils import EmbeddingLMHeadStrategy, AttentionStrategy, FFNStrategy, MoEFFNStrategy, LayerStrategy, DPType

class GalvatronCostModelHandler:
    def __init__(self, args=None):
        assert args is not None
        self.args = args
        args.gpu_num = args.num_nodes * args.num_gpus_per_node
        self.layernum_arg_names = None
        self.mem_path = None
        self.time_path = None
        self.model_name = None
        self.time_config = None
        self.memory_config = None
        self.param_sizes = None
        self.act_sizes = None
        self.other_memory_pp_off = None
        self.other_memory_pp_on = None
        self.time_profiled_list = None
        self.other_time_profiled_list = None
        self.attention_time_profiled_list = None
        self.ffn_time_profiled_list = None
        self.param_memory_list = None
        self.act_memory_list = None
        self.attention_param_memory_list = None
        self.attention_act_memory_list = None
        self.ffn_param_memory_list = None
        self.ffn_act_memory_list = None
        self.model_type = 'gpt'
        self.logger = None
        self.world_size = args.num_nodes * args.num_gpus_per_node

    def set_cost_model_handler_info(self, path,  model_layer_configs, model_name):
        self.set_model_layer_configs(model_layer_configs)
        self.set_path(path)
        self.set_model_name(model_name)
        self.memory_profiling_path()
        self.time_profiling_path()
    
    def set_path(self, path):
        self.path = path

    def set_model_type(self, model_type):
        self.model_type = model_type

    def set_model_name(self, name):
        self.model_name = name
        
    def memory_profiling_path(self):
        if self.mem_path is not None:
            return self.mem_path
        assert self.model_name is not None, 'Should specify the model name!'
        args = self.args
        if args.profile_granularity == 'split':
            memory_config_name = [
                f'memory_profiling_{args.mixed_precision}_{self.model_name}_attention.json',
                f'memory_profiling_{args.mixed_precision}_{self.model_name}_mlp.json'
            ]
        elif args.profile_granularity == 'together':
            memory_config_name = [f'memory_profiling_{args.mixed_precision}_{self.model_name}_all.json']
        else:
            raise ValueError("Unsupported profile granularity: %s"%(args.profile_granularity))
        if args.memory_profiling_path is None:
            memory_config_path = os.path.join(self.path, 'configs')
        else:
            memory_config_path = args.memory_profiling_path
        self.mem_path = [
            os.path.join(memory_config_path, item) for item in memory_config_name
        ]
        return self.mem_path
    
    def time_profiling_path(self):
        if self.time_path is not None:
            return self.time_path
        assert self.model_name is not None, 'Should specify the model name!'
        args = self.args
        if args.profile_granularity == 'split':
            time_config_name = [
                f'computation_profiling_{args.mixed_precision}_{self.model_name}_attention.json',
                f'computation_profiling_{args.mixed_precision}_{self.model_name}_mlp.json'
            ]
        elif args.profile_granularity == 'together':
            time_config_name = [f'computation_profiling_{args.mixed_precision}_{self.model_name}_all.json']
        else:
            raise ValueError("Unsupported profile granularity: %s"%(args.profile_granularity))
        if args.time_profiling_path is None:
            time_config_path = os.path.join(self.path, 'configs')
        else:
            time_config_path = args.memory_profiling_path
        self.time_path = [
            os.path.join(time_config_path, item) for item in time_config_name
        ]
        return self.time_path
    
    def set_model_layer_configs(self, model_layer_configs):
        if model_layer_configs is None:
            return
        self.hiddensize_list = [config['hidden_size'] for config in model_layer_configs]
        self.layernum_list = [config['layer_num'] for config in model_layer_configs]
        self.seqlen_list = [config['seq_len'] for config in model_layer_configs]
        self.num_layertype = len(self.layernum_list)
        
    def initialize_cost_model_handler(self, show_handler_info=True):
        self.handler_init_logger()
        self.get_profiled_model_configs()
        self.get_profiled_hardware_configs()
        self.set_cost_model_args()
        if show_handler_info:
            self.show_handler_info()
    
    def get_profiled_model_configs(self):
        args = self.args
        
        # [Step1] Profile Model Computation Configs
        if args.profile_granularity == 'together':
            time_path = self.time_profiling_path()
            self.time_profiled_list, self.other_time_profiled_list = parse_computaion_profile(time_path[0], args.time_profile_mode[0], self.num_layertype, self.seqlen_list, unit='all')
        elif args.profile_granularity == 'split':
            time_path = self.time_profiling_path()
            self.attention_time_profiled_list, self.other_time_profiled_list = parse_computaion_profile(time_path[0], args.time_profile_mode[0], self.num_layertype, self.seqlen_list, unit='attention')
            self.ffn_time_profiled_list, _ = parse_computaion_profile(time_path[1], args.time_profile_mode[1], self.num_layertype, self.seqlen_list, unit='ffn')
        else:
            raise ValueError("Unsupported profile granularity: %s"%(args.profile_granularity))
        
        # [Step2] Profile Model Memory Configs
        if args.profile_granularity == 'together':
            memory_path = self.memory_profiling_path()
            self.param_memory_list, self.act_memory_list, self.other_memory_pp_off, self.other_memory_pp_on = parse_memory_profile(memory_path[0], args.memory_profile_mode[0], self.num_layertype, self.seqlen_list, args.sequence_parallel)
        elif args.profile_granularity == 'split':
            memory_path = self.memory_profiling_path()
            self.attention_param_memory_list, self.attention_act_memory_list, self.other_memory_pp_off, self.other_memory_pp_on = parse_memory_profile(memory_path[0], args.memory_profile_mode[0], self.num_layertype, self.seqlen_list, args.sequence_parallel)
            # if args.is_MoE == False: # TODO currently, don't parse the MoE memory,
            self.ffn_param_memory_list, self.ffn_act_memory_list, _, _ = parse_memory_profile(memory_path[1], args.memory_profile_mode[1], self.num_layertype, self.seqlen_list, args.sequence_parallel)
        else:
            raise ValueError("Unsupported profile granularity: %s"%(args.profile_granularity))
            
    def get_profiled_hardware_configs(self):
        args = self.args
        
        # allreduce fixed
        if args.allreduce_bandwidth_config_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            allreduce_bandwidth_config_path = os.path.join(self.path, hardware_configs_dir)
        else:
            allreduce_bandwidth_config_path = args.allreduce_bandwidth_config_path
        allreduce_bandwidth_config_name = 'allreduce_bandwidth_%dnodes_%dgpus_per_node.json'%(args.num_nodes, args.num_gpus_per_node)
        args.allreduce_bandwidth_config_path  = os.path.join(allreduce_bandwidth_config_path, allreduce_bandwidth_config_name)
        self.allreduce_bandwidth, self.allreduce_fixed_dict = read_allreduce_bandwidth_config(args.allreduce_bandwidth_config_path, gpu_num=args.gpu_num)
        self.all_gather_fixed_dict = copy.deepcopy(self.allreduce_fixed_dict)
        for key, value in self.allreduce_fixed_dict.items():
            self.all_gather_fixed_dict[key] = value / 2
        self.reduce_scatter_fixed_dict = self.all_gather_fixed_dict

        # p2p
        if args.p2p_bandwidth_config_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            p2p_bandwidth_config_path = os.path.join(self.path, hardware_configs_dir)
        else:
            p2p_bandwidth_config_path = args.p2p_bandwidth_config_path
        p2p_bandwidth_config_name = 'p2p_bandwidth_%dnodes_%dgpus_per_node.json'%(args.num_nodes, args.num_gpus_per_node)
        args.p2p_bandwidth_config_path  = os.path.join(p2p_bandwidth_config_path, p2p_bandwidth_config_name)
        self.p2p_bandwidth, self.p2p_comm_coe = read_p2p_bandwidth_config(args.p2p_bandwidth_config_path)
        
        # overlap slowdown
        if args.overlap_coe_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            overlap_coe_path = os.path.join(self.path, hardware_configs_dir)
        else:
            overlap_coe_path = args.overlap_coe_path
        overlap_coe_name = 'overlap_coefficient.json'
        args.overlap_coe_path = os.path.join(overlap_coe_path, overlap_coe_name)
        self.overlap_coe = read_json_config(args.overlap_coe_path)['overlap_coe']
        
        # allreduce_fit && all2all_fit
        if args.sp_time_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            sp_time_path = os.path.join(self.path, hardware_configs_dir)
        else:
            sp_time_path = args.sp_time_path
        sp_time_config_name = 'sp_time_%dnodes_%dgpus_per_node.json'%(args.num_nodes, args.num_gpus_per_node)
        args.sp_time_path = os.path.join(sp_time_path, sp_time_config_name)
        sp_config = read_json_config(args.sp_time_path)
        self.allreduce_fit_dict = remap_config(sp_config, "allreduce") # 此时已经除以了2
        self.all_gather_fit_dict = copy.deepcopy(self.allreduce_fit_dict)
        self.reduce_scatter_fit_dict = copy.deepcopy(self.allreduce_fit_dict)
        self.all2all_fit_dict = remap_config(sp_config, "all2all")

        return self.allreduce_bandwidth, self.p2p_bandwidth, self.overlap_coe, self.allreduce_fit_dict, self.all2all_fit_dict

    def set_cost_model_args(self):
        self.train_args_list, self.profile_hardware_args_list, self.utils_args_list, self.version_option_args_list = [], [], [], []
        self.profile_model_args_list, self.profile_attention_args_list, self.profile_ffn_args_list = [], [], []
        for i in range(self.num_layertype):
            train_args = TrainArgs(
                seq_length=self.seqlen_list[i],
                hidden_size=self.hiddensize_list[i],
                sequence_length_list=self.seqlen_list,
                mixed_precision=False if self.args.mixed_precision == 'fp32' else True,
                async_grad_reduce=self.args.async_grad_reduce,
                sequence_parallel=self.args.sequence_parallel,
                pipeline_type=self.args.pipeline_type,
                num_experts=self.args.num_experts,
                top_k=self.args.top_k,
                moe_grouped_gemm=self.args.moe_grouped_gemm
            )
            profile_hardware_args = ProfileHardwareArgs(
                bct_fct_coe=2,
                overlap_slowdown_coe=self.overlap_coe,
                allreduce_fixed_dict=self.allreduce_fixed_dict,
                allreduce_fit_dict=self.allreduce_fit_dict,
                all_gather_fixed_dict=self.all_gather_fixed_dict,
                all_gather_fit_dict=self.all_gather_fit_dict,
                reduce_scatter_fixed_dict=self.reduce_scatter_fixed_dict,
                reduce_scatter_fit_dict=self.reduce_scatter_fit_dict,
                p2p_comm_coe_dict=self.p2p_comm_coe,
                all2all_fit_dict=self.all2all_fit_dict,
            )
            utils_args = UtilsArgs()
            version_option_args = VersionOptionArgs(
                estimate_tp_time_type=EstimateTPTimeType.FIXED if self.args.estimate_tp_time_type == 'fixed' else EstimateTPTimeType.FIT,
                zero_with_slight_noise=True if self.args.zero_with_slight_noise else False
            )
            
            self.train_args_list.append(train_args)
            self.profile_hardware_args_list.append(profile_hardware_args)
            self.utils_args_list.append(utils_args)
            self.version_option_args_list.append(version_option_args)

        for i in range(self.num_layertype):
            if self.args.profile_granularity == "together":
                profile_model_args = ProfileModelArgs(
                    forward_computation_time=self.time_profiled_list[i],
                    parameter_memory=self.param_memory_list[i],
                    tp_activation_per_bsz_dict=self.act_memory_list[i],
                    other_time_profiled=self.other_time_profiled_list[0],
                    other_memory_pp_off=self.other_memory_pp_off,
                    other_memory_pp_on=self.other_memory_pp_on
                )
                self.profile_model_args_list.append(profile_model_args)
            elif self.args.profile_granularity == "split":
                profile_attention_args = ProfileModelArgs(
                    forward_computation_time=self.attention_time_profiled_list[i],
                    parameter_memory=self.attention_param_memory_list[i],
                    tp_activation_per_bsz_dict=self.attention_act_memory_list[i],
                    other_time_profiled=self.other_time_profiled_list[0],
                    other_memory_pp_off=self.other_memory_pp_off,
                    other_memory_pp_on=self.other_memory_pp_on
                )
                if self.args.is_MoE == False:
                    profile_ffn_args = ProfileModelArgs(
                        forward_computation_time=self.ffn_time_profiled_list[i],
                        parameter_memory=self.attention_param_memory_list[i],
                        tp_activation_per_bsz_dict=self.attention_act_memory_list[i],
                        other_time_profiled=self.other_time_profiled_list[0],
                        other_memory_pp_off=self.other_memory_pp_off,
                        other_memory_pp_on=self.other_memory_pp_on
                    )
                else:
                    profile_ffn_args = ProfileModelArgs(
                        forward_computation_time=self.ffn_time_profiled_list[i],
                        parameter_memory=self.ffn_param_memory_list[i],
                        tp_activation_per_bsz_dict=self.ffn_act_memory_list[i],
                        other_time_profiled=self.other_time_profiled_list[0],
                        other_memory_pp_off=self.other_memory_pp_off,
                        other_memory_pp_on=self.other_memory_pp_on
                    )
                self.profile_attention_args_list.append(profile_attention_args)
                self.profile_ffn_args_list.append(profile_ffn_args)
            
    def show_handler_info(self):
        print('================================================================================')
        print('--- Optimization Configs ----')
        print('Pipeline Type:', self.args.pipeline_type)
        print('Mixed Precision:', self.args.mixed_precision)
        print('================================================================================')
        print('---- Environment Configs ----')
        print('Allreduce Bandwidth (GB/s):', self.allreduce_bandwidth)
        print('Allreduce Communication Coefficient (ms/MB):', self.allreduce_fixed_dict)
        print('P2P Bandwidth (GB/s):', self.p2p_bandwidth)
        print('P2P Communication Coefficient (ms/MB):', self.p2p_comm_coe)
        print('Overlap coefficient:', self.overlap_coe)
        print('================================================================================')
        print('------- Model Configs -------')
        print('Model Name:', self.model_name)
        print('Num layertype:', self.num_layertype)
        print('Layer_num:', self.layernum_list)
        print('Hidden_size:', self.hiddensize_list)
        print('Seq_len:', self.seqlen_list)
        print('================================================================================')
        print('--- Model Computation Configs ---')
        print('Forward computation time:', self.time_profiled_list)
        print('================================================================================')
        print('--- Model Memory Configs ---')
        print('Parameter Memory Cost:', self.param_memory_list)
        print('Activation Memory Cost of Different TP degree (per bsz):')
        print(self.act_memory_list)
        print('Other Memory Cost (pp = 1):')
        print(self.other_memory_pp_off)
        print('Other Memory Cost (pp > 1):')
        print(self.other_memory_pp_on)
        print('================================================================================')
        print('Train Args List:')
        print(self.train_args_list)
        print('================================================================================')
        print('Profile Model Args List:')
        print(self.profile_model_args_list)
        print('================================================================================')
        print('Profile Hardware Args List:')
        print(self.profile_hardware_args_list)
        print('================================================================================')
        print('Utils Args List:')
        print(self.utils_args_list)
        print('================================================================================')
        print('Version Option Args List:')
        print(self.version_option_args_list)
        print('================================================================================')
    
    def loguru_logger_handler_info(self):
        from loguru import logger as loguru_logger
        from rich.pretty import pretty_repr
        loguru_logger.info('================================================================================')
        loguru_logger.info('--- Optimization Configs ----')
        loguru_logger.info(f'Pipeline Type: {self.args.pipeline_type}')
        loguru_logger.info(f'Mixed Precision: {self.args.mixed_precision}')
        loguru_logger.info('================================================================================')
        loguru_logger.info('---- Environment Configs ----')
        loguru_logger.info(f'Allreduce Bandwidth (GB/s): {self.allreduce_bandwidth}')
        loguru_logger.info(f'Allreduce Communication Coefficient (ms/MB): {self.allreduce_fixed_dict}')
        loguru_logger.info(f'P2P Bandwidth (GB/s): {self.p2p_bandwidth}')
        loguru_logger.info(f'P2P Communication Coefficient (ms/MB): {self.p2p_comm_coe}')
        loguru_logger.info('Overlap coefficient:', self.overlap_coe)
        loguru_logger.info('================================================================================')
        loguru_logger.info('------- Model Configs -------')
        loguru_logger.info(f'Model Name: {self.model_name}')
        loguru_logger.info(f'Num layertype: {self.num_layertype}')
        loguru_logger.info(f'Layer_num: {self.layernum_list}')
        loguru_logger.info(f'Hidden_size: {self.hiddensize_list}')
        loguru_logger.info(f'Seq_len: {self.seqlen_list}')
        loguru_logger.info('================================================================================')
        loguru_logger.info('--- Model Computation Configs ---')
        loguru_logger.info(f'Forward computation time: {self.time_profiled_list}')
        loguru_logger.info('================================================================================')
        loguru_logger.info('--- Model Memory Configs ---')
        loguru_logger.info(f'Parameter Memory Cost: {self.param_memory_list}')
        loguru_logger.info('Activation Memory Cost of Different TP degree (per bsz):')
        loguru_logger.info(pretty_repr(self.act_memory_list, max_width=256))
        loguru_logger.info('Other Memory Cost (pp = 1):')
        loguru_logger.info(pretty_repr(self.other_memory_pp_off, max_width=256))
        loguru_logger.info('Other Memory Cost (pp > 1):')
        loguru_logger.info(pretty_repr(self.other_memory_pp_on, max_width=256))
        loguru_logger.info('================================================================================')
        loguru_logger.info('Train Args List:')
        loguru_logger.info(pretty_repr(self.train_args_list, max_width=256))
        loguru_logger.info('================================================================================')
        loguru_logger.info('Profile Model Args List:')
        loguru_logger.info(pretty_repr(self.profile_model_args_list, max_width=256))
        loguru_logger.info('================================================================================')
        loguru_logger.info('Profile Hardware Args List:')
        loguru_logger.info(pretty_repr(self.profile_hardware_args_list, max_width=256))
        loguru_logger.info('================================================================================')
        loguru_logger.info('Utils Args List:')
        loguru_logger.info(pretty_repr(self.utils_args_list, max_width=256))
        loguru_logger.info('================================================================================')
        loguru_logger.info('Version Option Args List:')
        loguru_logger.info(pretty_repr(self.version_option_args_list, max_width=256))
        loguru_logger.info('================================================================================')

    def get_time_cost_for_specific_strategy_together(self, strategy:LayerStrategy, global_batch_size, chunks):
        pp_size = strategy.pp_size

        layer_time_cost_dict, layer_time_cost_no_sync_dict = {}, {}
        for layertype_id in range(self.num_layertype):
            time_cost, time_cost_no_sync = self.get_time_cost(strategy, global_batch_size, chunks, layertype_id) # tuple(float, float)
            layer_time_cost_dict[layertype_id] = time_cost
            layer_time_cost_no_sync_dict[layertype_id] = time_cost_no_sync

        embedding_lmhead_strategy = strategy.to_embedding_lmhead_strategy()
        embedding_lmhead_cost, embedding_lmhead_no_sync_cost = self.get_time_cost(embedding_lmhead_strategy, global_batch_size, chunks) # tuple(float, float)

        if pp_size == 1:
            time_cost_per_chunk = embedding_lmhead_cost[0] + embedding_lmhead_cost[1] # 0:embedding 1:lmhead
            time_cost_no_sync_per_chunk = embedding_lmhead_no_sync_cost[0] + embedding_lmhead_no_sync_cost[1] # 0:embedding 1:lmhead
            for i in range(self.num_layertype):
                time_cost_per_chunk += layer_time_cost_dict[i] * self.layernum_list[i]
                time_cost_no_sync_per_chunk += layer_time_cost_no_sync_dict[i] * self.layernum_list[i]
            
            result = time_cost_no_sync_per_chunk * (chunks - 1)  + time_cost_per_chunk
            return result
        else:
            # p2p_communication_time = self.get_p2p_time()
            raise NotImplementedError("pp not implement")
        
    def get_memory_cost_for_specific_strategy_together(self, strategy:LayerStrategy, global_batch_size:int, chunks:int):
        pp_size = strategy.pp_size

        # [Step 1]
        layer_memory_cost_dict = [{} for _ in range(self.num_layertype)]
        for layertype_id in range(self.num_layertype):
            for stage_idx in range(pp_size):
                memory_cost:dict = self.get_memory_cost(strategy, global_batch_size, chunks, layertype_id, stage_idx, detailed=True) # dict
                layer_memory_cost_dict[layertype_id][stage_idx] = memory_cost
        
        # [Step 2]
        embedding_lmhead_strategy = strategy.to_embedding_lmhead_strategy()
        embedding_memory_cost, lmhead_memory_cost = self.get_memory_cost(embedding_lmhead_strategy, global_batch_size, chunks, detailed=True) # tuple[dict, dict]
        
        # [Step 3]
        if pp_size == 1:
            fix_stage_idx = 0
            memory_cost = embedding_memory_cost['total_memory'] + lmhead_memory_cost['total_memory']
            for layertype_id in range(self.num_layertype):
                memory_cost += layer_memory_cost_dict[layertype_id][fix_stage_idx]['total_memory'] * self.layernum_list[layertype_id]
            
            if strategy.dp_type != DPType.DDP:
                pytorch_context_memory = 2048 * strategy.dp_size / 8 # [trick]
            else:
                pytorch_context_memory = 0
            result = memory_cost + pytorch_context_memory
            return result
        else:
            raise NotImplementedError("pp not implement")

    def get_time_cost_for_specific_strategy_split(self, attention_strategy:AttentionStrategy, ffn_strategy:FFNStrategy, embedding_lmhead_strategy:EmbeddingLMHeadStrategy, global_batch_size:int, chunks:int):
        assert attention_strategy.pp_size == ffn_strategy.pp_size, f'attention pp_size {attention_strategy.pp_size} not equal ffn pp_size {ffn_strategy.pp_size}'
        assert attention_strategy.pp_size == embedding_lmhead_strategy.pp_size, f'attention pp_size {attention_strategy.pp_size} not equal embedding_lmhead pp_size {embedding_lmhead_strategy.pp_size}'
        pp_size = attention_strategy.pp_size
        
        # [Step 1]
        attention_time_cost_dict, attention_time_cost_no_sync_dict = {}, {}
        ffn_time_cost_dict, ffn_time_cost_no_sync_dict = {}, {}
        for layertype_id in range(self.num_layertype):
            attention_time_cost, attention_time_cost_no_sync = self.get_time_cost(attention_strategy, global_batch_size, chunks, layertype_id) # tuple(float, float)
            ffn_time_cost, ffn_time_cost_no_sync = self.get_time_cost(ffn_strategy, global_batch_size, chunks, layertype_id) # tuple(float, float)
            attention_time_cost_dict[layertype_id] = attention_time_cost
            attention_time_cost_no_sync_dict[layertype_id] = attention_time_cost_no_sync
            ffn_time_cost_dict[layertype_id] = ffn_time_cost
            ffn_time_cost_no_sync_dict[layertype_id] = ffn_time_cost_no_sync
         
        # [Step 2]
        embedding_lmhead_cost, embedding_lmhead_no_sync_cost = self.get_time_cost(embedding_lmhead_strategy, global_batch_size, chunks) # tuple(tuple(float, float), tuple(float, float))
        
        # [Step 3]
        if pp_size == 1:
            time_cost_per_chunk = embedding_lmhead_cost[0] + embedding_lmhead_cost[1] # 0:embedding 1:lmhead
            time_cost_no_sync_per_chunk = embedding_lmhead_no_sync_cost[0] + embedding_lmhead_no_sync_cost[1] # 0:embedding 1:lmhead
            for i in range(self.num_layertype):
                time_cost_per_chunk += attention_time_cost_dict[i] * self.layernum_list[i] + ffn_time_cost_dict[i] * self.layernum_list[i]
                time_cost_no_sync_per_chunk += attention_time_cost_no_sync_dict[i] * self.layernum_list[i] + ffn_time_cost_no_sync_dict[i] * self.layernum_list[i]
            
            result = time_cost_no_sync_per_chunk * (chunks - 1)  + time_cost_per_chunk
            return result
        else:
            raise NotImplemented("pp not implement")

    def get_time_cost(
        self, 
        strategy:Union[EmbeddingLMHeadStrategy, AttentionStrategy, FFNStrategy, MoEFFNStrategy, LayerStrategy], 
        global_batch_size:int, 
        chunks:int, 
        layertype_id:int=0
    )->Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]:
        if type(strategy) is EmbeddingLMHeadStrategy:
            time_cost:Tuple[float, float] = EmbeddingLMHeadTimeCostModel(
                strategy=strategy,
                global_batch_size=global_batch_size,
                chunks=chunks,
                no_sync_gradient=False,
                logger=self.logger,
                train_args=self.train_args_list[0],
                profile_model_args=self.profile_model_args_list[0] if len(self.profile_model_args_list) > 0 else self.profile_attention_args_list[0],
                profile_hardware_args=self.profile_hardware_args_list[0],
                version_option_args=self.version_option_args_list[0]
            ).gen_result()
            time_cost_no_sync:Tuple[float, float] = EmbeddingLMHeadTimeCostModel(
                strategy=strategy,
                global_batch_size=global_batch_size,
                chunks=chunks,
                no_sync_gradient=True,
                logger=self.logger,
                train_args=self.train_args_list[0],
                profile_model_args=self.profile_model_args_list[0] if len(self.profile_model_args_list) > 0 else self.profile_attention_args_list[0],
                profile_hardware_args=self.profile_hardware_args_list[0],
                version_option_args=self.version_option_args_list[0]
            ).gen_result()
        elif type(strategy) is AttentionStrategy or type(strategy) is FFNStrategy or type(strategy) is LayerStrategy:
            if type(strategy) is AttentionStrategy:
                cls = AttentionTimeCostModel
                profile_model_args=self.profile_attention_args_list[layertype_id]
            elif type(strategy) is FFNStrategy:
                cls = FFNTimeCostModel
                profile_model_args=self.profile_ffn_args_list[layertype_id]
            elif type(strategy) is LayerStrategy:
                cls = LayerTimeCostModel
                profile_model_args=self.profile_model_args_list[layertype_id]
            time_cost:float = cls(
                strategy=strategy,
                global_batch_size=global_batch_size,
                chunks=chunks,
                no_sync_gradient=False,
                logger=self.logger,
                train_args=self.train_args_list[layertype_id],
                profile_model_args=profile_model_args,
                profile_hardware_args=self.profile_hardware_args_list[layertype_id],
                utils_args=self.utils_args_list[layertype_id],
                version_option_args=self.version_option_args_list[layertype_id]
            ).gen_result()
            time_cost_no_sync:float = cls(
                strategy=strategy,
                global_batch_size=global_batch_size,
                chunks=chunks,
                no_sync_gradient=True,
                logger=self.logger,
                train_args=self.train_args_list[layertype_id],
                profile_model_args=profile_model_args,
                profile_hardware_args=self.profile_hardware_args_list[layertype_id],
                utils_args=self.utils_args_list[layertype_id],
                version_option_args=self.version_option_args_list[layertype_id]
            ).gen_result()
        elif type(strategy) is MoEFFNStrategy:
            pass
        
        return time_cost, time_cost_no_sync

    def get_p2p_time(self, global_batch_size, chunks, pp_size):
        p2p_communication_time = PipelineTimeCostModel(
            global_batch_size=global_batch_size,
            chunks=chunks,
            pp_size=pp_size,
            world_size=self.world_size,
            logger=self.logger,
            train_args=self.train_args_list[0],
            profile_hardware_args=self.profile_hardware_args_list[0]
        ).gen_result()
        return p2p_communication_time

    def get_memory_cost(
        self,
        strategy:Union[EmbeddingLMHeadStrategy, AttentionStrategy, FFNStrategy, MoEFFNStrategy, LayerStrategy],
        global_batch_size:int,
        chunks:int,
        layertype_id:int=0,
        stage_idx:int=0,
        detailed:bool=False,
    )->Union[float, dict, Tuple[float, float], Tuple[dict, dict]]:
        if type(strategy) is EmbeddingLMHeadStrategy:
            memory_cost:Tuple[dict, dict] = EmbeddingLMHeadMemoryCostModel(
                strategy=strategy,
                global_batch_size=global_batch_size,
                chunks=chunks,
                logger=self.logger,
                train_args=self.train_args_list[0],
                profile_model_args=self.profile_model_args_list[0] if len(self.profile_model_args_list) > 0 else self.profile_attention_args_list[0],
                version_option_args=self.version_option_args_list[0]
            ).get_memory_cost()
            if not detailed:
                embedding_memory_cost, lmhead_memory_cost = memory_cost
                memory_cost:Tuple[float, float] = (embedding_memory_cost['total_memory'], lmhead_memory_cost['total_memory'])
        elif type(strategy) is AttentionStrategy or type(strategy) is FFNStrategy or type(strategy) is LayerStrategy:
            if type(strategy) is AttentionStrategy:
                cls = AttentionMemoryCostModel
                profile_model_args=self.profile_attention_args_list[layertype_id]
            elif type(strategy) is FFNStrategy:
                cls = FFNMemoryCostModel
                profile_model_args=self.profile_ffn_args_list[layertype_id]
            elif type(strategy) is LayerStrategy:
                cls = LayerMemoryCostModel
                profile_model_args=self.profile_model_args_list[layertype_id]
            memory_cost:dict = cls(
                strategy=strategy,
                global_batch_size=global_batch_size,
                chunks=chunks,
                stage_idx=stage_idx,
                logger=self.logger,
                train_args=self.train_args_list[layertype_id],
                profile_model_args=profile_model_args,
                version_option_args=self.version_option_args_list[layertype_id]
            ).get_memory_cost()
            if not detailed:
                memory_cost:float = memory_cost['total_memory']
        elif type(strategy) is MoEFFNStrategy:
            pass

        return memory_cost

    # def get_time_cost_for_specific_strategy_moe(self, attention_strategy:AttentionStrategy, ffn_strategy:MoEFFNStrategy, embedding_lmhead_strategy:EmbeddingLMHeadStrategy, global_batch_size:int, chunks:int):
    #     # [Step 1]
    #     attention_time_cost_dict, attention_time_cost_no_sync_dict = {}, {}
    #     ffn_time_cost_dict = {}
    #     for layertype_id in range(self.num_layertype):
    #         attention_time_cost, attention_time_cost_no_sync = self.get_time_cost(global_batch_size, chunks, attention_strategy, layertype_id)
    #         ffn_time_cost, _ = self.get_time_cost(global_batch_size, chunks, ffn_strategy, layertype_id)
        
    #         attention_time_cost_dict[layertype_id] = attention_time_cost
    #         attention_time_cost_no_sync_dict[layertype_id] = attention_time_cost_no_sync
    #         ffn_time_cost_dict[layertype_id] = ffn_time_cost
        
    #     self.handler_log(f'[linguangming] attention_time_cost_dict: { {k: v*1e3 for k, v in attention_time_cost_dict.items()}}')
    #     self.handler_log(f'[linguangming] attention_time_cost_no_sync_dict: { {k: v*1e3 for k, v in attention_time_cost_no_sync_dict.items()}}')
    #     self.handler_log(f'[linguangming] ffn_time_cost_dict: { {k: v*1e3 for k, v in ffn_time_cost_dict.items()}}') 
        
    #     # [Step 2]
    #     embedding_lmhead_cost, embedding_lmhead_no_sync_cost = self.get_time_cost(global_batch_size, chunks, embedding_lmhead_strategy)
    #     self.handler_log(f'[linguangming] embedding_lmhead_cost: {embedding_lmhead_cost}')
    #     self.handler_log(f'[linguangming] embedding_lmhead_no_sync_cost: {embedding_lmhead_no_sync_cost}')
        
    #     # [Step 3]
    #     if embedding_lmhead_strategy.pp_size == 1:
    #         time_cost_per_chunk = embedding_lmhead_cost['embedding'] + embedding_lmhead_cost['lmhead']
    #         time_cost_no_sync_per_chunk = embedding_lmhead_no_sync_cost['embedding'] + embedding_lmhead_no_sync_cost['lmhead']
    #         for i in range(self.num_layertype):
    #             time_cost_per_chunk += attention_time_cost_dict[i] * self.layernum_list[i] + ffn_time_cost_dict[i] * self.layernum_list[i]
    #             time_cost_no_sync_per_chunk += attention_time_cost_no_sync_dict[i] * self.layernum_list[i] + ffn_time_cost_dict[i] * self.layernum_list[i] # MoE layers do not require gradient synchronization.
    #         result = time_cost_no_sync_per_chunk * (chunks - 1)  + time_cost_per_chunk
    #         return result
    #     else:
    #         raise NotImplemented("pp not implement")

    # def get_memory_cost_for_specific_strategy_moe(self, attention_strategy:AttentionStrategy, ffn_strategy:MoEFFNStrategy, embedding_lmhead_strategy:EmbeddingLMHeadStrategy, global_batch_size:int, chunks:int):
    #     pp_size = attention_strategy.pp_size

    #     # [Step 1]
    #     attention_memory_cost_dict = [{} for _ in range(self.num_layertype)]
    #     for layertype_id in range(self.num_layertype):
    #         for stage_idx in range(pp_size):
    #             memory_cost = self.get_memory_cost(global_batch_size, chunks, attention_strategy, layertype_id, stage_idx, detailed=True)
    #             attention_memory_cost_dict[layertype_id][stage_idx] = memory_cost

    #     moe_memory_cost_dict = [{} for _ in range(self.num_layertype)]
    #     for layertype_id in range(self.num_layertype):
    #         for stage_idx in range(pp_size):
    #             memory_cost = self.get_memory_cost(global_batch_size, chunks, ffn_strategy, layertype_id, stage_idx, detailed=True)
    #             moe_memory_cost_dict[layertype_id][stage_idx] = memory_cost

    #     # [Step 2]
    #     embedding_lmhead_memory_cost = self.get_memory_cost(global_batch_size, chunks, embedding_lmhead_strategy, detailed=True)
        
    #     # [Step 3]
    #     if attention_strategy.pp_size == 1:
    #         # [DEBUG]
    #         model_states = embedding_lmhead_memory_cost['model_states_memory']['embedding'] + embedding_lmhead_memory_cost['model_states_memory']['lmhead']
    #         model_states += attention_memory_cost_dict[0][0]['model_states_memory'] * self.layernum_list[0]
    #         model_states += moe_memory_cost_dict[0][0]['model_states_memory'] * self.layernum_list[0]
    #         print(f'[DEBUG] model_states is {model_states}')

    #         fix_stage_idx = 0
    #         memory_cost = embedding_lmhead_memory_cost['total_memory']['embedding'] + embedding_lmhead_memory_cost['total_memory']['lmhead']
    #         for layertype_id in range(self.num_layertype):
    #             memory_cost += attention_memory_cost_dict[layertype_id][fix_stage_idx]['total_memory'] * self.layernum_list[layertype_id]
    #             memory_cost += moe_memory_cost_dict[layertype_id][fix_stage_idx]['total_memory'] * self.layernum_list[layertype_id]
            
    #         if attention_strategy.dp_type != 'zero0':
    #             pytorch_context_memory = 1024 * 2
    #         else:
    #             pytorch_context_memory = 0
    #         result = memory_cost + pytorch_context_memory
    #         return result
    #     else:
    #         raise NotImplemented("pp not implement")

    # =============== Utils Functions ===============
    def handler_init_logger(self):
        """Initialize a compact, colored console logger without timestamps.

        Uses ANSI color codes for levels and avoids adding duplicate handlers
        if the logger already has a stream handler.
        """
        logger = logging.getLogger('CostModelHandler')
        logger.setLevel(logging.INFO)

        # Avoid adding multiple handlers in case this is called repeatedly
        has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        if has_stream_handler:
            # Make sure messages don't propagate to root handlers (avoid duplicate prints)
            logger.propagate = False
            self.logger = logger
            return

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Compact format: [LEVEL] message
        # No timestamp, no logger name
        class _LevelColorFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\u001b[36m',    # cyan
                'INFO': '\u001b[32m',     # green
                'WARNING': '\u001b[33m',  # yellow
                'ERROR': '\u001b[31m',    # red
                'CRITICAL': '\u001b[35m', # magenta
            }
            RESET = '\u001b[0m'

            def format(self, record):
                levelname = record.levelname
                color = self.COLORS.get(levelname, '')
                msg = super().format(record)
                return f"{color}[{levelname}] [CostModelHandler]{self.RESET} {msg}"

        formatter = _LevelColorFormatter('%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # Prevent propagation to root logger to avoid duplicate lines
        logger.propagate = False
        self.logger = logger
    
    def handler_log(self, message):
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(f'[CostModelHandler] {message}', flush=True)
        
def convert_keys_to_int(d):
    if isinstance(d, dict):
        new_dict = {}
        for k, v in d.items():
            if isinstance(k, str) and k.isdigit():
                new_dict[int(k)] = convert_keys_to_int(v)
            else:
                new_dict[k] = convert_keys_to_int(v)
        return new_dict
    return d
            
def parse_computaion_profile(computation_config_path:str, time_profile_mode:str, num_layertype:int, seqlen_list:List[int], unit='all'):
    time_config = read_json_config(computation_config_path)
    time_profiled_list = []
    other_time_profiled_list = []
    
    if time_profile_mode == 'static':
        for i in range(num_layertype):
            for key, value in time_config.items():
                if key.startswith(f'layertype_{i}_'):
                    time_profiled_list.append(value)
                if key.startswith(f'layertype_other_'):
                    other_time_profiled_list.append(value)
                    
    elif time_profile_mode == 'batch':
        def linear_func(x, m, c):
            return m * x + c
        for i in range(num_layertype):
            x_data, y_data = [], []
            template = rf'layertype_{i}_bsz(\d+)_seq{seqlen_list[i]}'
            for key, value in time_config.items():
                match = re.match(template, key)
                if match:
                    bsz = int(match.group(1))
                    x_data.append(bsz)
                    y_data.append(bsz * value)
            assert len(x_data) >= 8, f"Different batch size in computation profile of layertype_{i} should not be lower than 8."
            popt, _ = curve_fit(linear_func, x_data, y_data)
            print(f'Fitted parameters: {popt}')
            time_profiled_list.append(popt)
            
        for i in range(num_layertype):
            x_data, y_data = [], []
            template = rf'layertype_other_bsz(\d+)_seq{seqlen_list[i]}'
            for key, value in time_config.items():
                match = re.match(template, key)
                if match:
                    bsz = int(match.group(1))
                    x_data.append(bsz)
                    y_data.append(bsz * value)
            assert len(x_data) >= 8, f"Different batch size in computation profile of layertype_other should not be lower than 8."
            popt, _ = curve_fit(linear_func, x_data, y_data)
            print(f'Fitted parameters: {popt}')
            other_time_profiled_list.append(popt)
            
    elif time_profile_mode == 'sequence':
        for i in range(num_layertype):
            x_data, y_data = [], []
            template = rf'layertype_{i}_bsz1_seq(\d+)'
            for key, value in time_config.items():
                match = re.match(template, key)
                if match:
                    seq_len = int(match.group(1))
                    x_data.append(seq_len)
                    y_data.append(value)
            def quadratic_func(x, a, b, c):
                return a * x * x + b * x + c
            def linear_func(x, a, b):
                return a * x + b

            if unit == 'all' or unit == 'attention':
                fit_func = quadratic_func
            elif unit == 'ffn':
                fit_func = linear_func
            else:
                raise ValueError(f'Invalid unit: {unit}')
            
            popt, _ = curve_fit(fit_func, x_data, y_data)
            print("Fitted parameters:", popt)

            if unit == 'all' or unit == 'attention':
                time_profiled_list.append(fit_func(seqlen_list[i], *popt))
            elif unit == 'ffn':
                time_profiled_list.append({'popt': popt})

        for i in range(num_layertype):
            x_data, y_data = [], []
            template = rf'layertype_other_bsz1_seq(\d+)'
            for key, value in time_config.items():
                match = re.match(template, key)
                if match:
                    seq_len = int(match.group(1))
                    x_data.append(seq_len)
                    y_data.append(value)
            def linear_func(x, m, c):
                return m * x + c
            popt, _ = curve_fit(linear_func, x_data, y_data)
            print("Fitted parameters:", popt)
            other_time_profiled_list.append(linear_func(seqlen_list[i], *popt))            
    else:
        raise ValueError(f"Unsupported time profile mode: {time_profile_mode}")
    
    return time_profiled_list, other_time_profiled_list
    
    
def parse_memory_profile(memory_config_path:str, memory_profile_mode:str, num_layertype:int, seqlen_list:List[int], sequence_parallel:bool):
    memory_config = read_json_config(memory_config_path)
    memory_config = convert_keys_to_int(memory_config)
    
    param_sizes = [0 for _ in range(num_layertype)]
    act_sizes = [{} for _ in range(num_layertype)]
    other_memory_pp_off, other_memory_pp_on = None, None
    
    if memory_profile_mode == "sequence":
        assert sequence_parallel, "Sequence parallel is required for sequence memory profiling."
        assert num_layertype == 1, "Only support num(layertype) == 1 for sequence memory profiling."
        
        raise NotImplemented('暂未实现')
    elif memory_profile_mode == "static":
        if sequence_parallel:
            for i in range(num_layertype):
                layer_mem_config = memory_config[f'layertype_{i}_sp']
                parameter_memory = layer_mem_config[seqlen_list[i]]['parameter_size']
                tp_activation_per_bsz_dict = layer_mem_config[seqlen_list[i]]['tp_activation_per_bsz_dict'].copy()
                param_sizes[i] = parameter_memory
                act_sizes[i] = tp_activation_per_bsz_dict
            seq_info = num2str(seqlen_list, 'seq')[3:] # TODO modify this line
            if seq_info.isdigit():
                seq_info = int(seq_info)
            other_memory_pp_off = memory_config['other_memory_pp_off_sp'][seq_info]
            other_memory_pp_on = {
                'first_stage': memory_config['other_memory_pp_on_first_sp'][seq_info],
                'last_stage': memory_config['other_memory_pp_on_last_sp'][seq_info]
            }
        else:
            for i in range(num_layertype):
                layer_mem_config = memory_config[f'layertype_{i}']
                parameter_memory = layer_mem_config[seqlen_list[i]]['parameter_size']
                tp_activation_per_bsz_dict = layer_mem_config[seqlen_list[i]]['tp_activation_per_bsz_dict'].copy()
                param_sizes[i] = parameter_memory
                act_sizes[i] = tp_activation_per_bsz_dict
            seq_info = num2str(seqlen_list, 'seq')[3:] # TODO modify this line
            if seq_info.isdigit():
                seq_info = int(seq_info)
            other_memory_pp_off = memory_config['other_memory_pp_off'][seq_info]
            other_memory_pp_on = {
                'first_stage': memory_config['other_memory_pp_on_first'][seq_info],
                'last_stage': memory_config['other_memory_pp_on_last'][seq_info]
            }
    else:
        raise ValueError(f"Unsupported memory profile mode: {memory_profile_mode}")
    return param_sizes, act_sizes, other_memory_pp_off, other_memory_pp_on