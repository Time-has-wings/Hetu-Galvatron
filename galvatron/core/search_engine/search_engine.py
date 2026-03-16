import os
import copy
import numpy as np
from typing import List, Any, Union
from rich.pretty import pretty_repr
from scipy.optimize import curve_fit

from galvatron.utils import read_allreduce_bandwidth_config, read_json_config, read_p2p_bandwidth_config, array2str, write_json_config, remap_config, num2str
from galvatron.utils.strategy_utils import AttentionStrategy, FFNStrategy, EmbeddingLMHeadStrategy, LayerStrategy, DPType, ColorSet, is_power_of_two, print_strategy_list, strategy_list2config

from galvatron.core.cost_model.cost_model_handler import pipeline_costmodel
from galvatron.core.cost_model.components.embedding_lmhead_cost import EmbeddingLMHeadTimeCostModel, EmbeddingLMHeadMemoryCostModel
from galvatron.core.cost_model.components.layer_cost import MemoryCostModelBase
from galvatron.core.cost_model.cost_model_args import ModelArgs, ParallelArgs, TrainArgs, ProfileModelArgs, ProfileHardwareArgs

from galvatron.core.search_engine.utils import get_thread_logger_single_task, ensure_log_dir
from galvatron.core.search_engine.dynamic_programming import DpOnModel

class GalvatronSearchEngine():
    def __init__(self, args):
        self.args = args
        args.gpu_num = args.num_nodes * args.num_gpus_per_node
        self.world_size = args.gpu_num
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
        self.use_pipeline_costmodel = args.use_pipeline_costmodel
        self.model_type = 'gpt'
        self.optimal_chunk_func = optimal_chunk_func_default
        self.memory_constraint = args.memory_constraint * 1024
        
    # =============== Setting Galvatron Search Engine Basic Information ===============
    def set_search_engine_info(self, path, model_layer_configs, model_name):
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
        
    def memory_profiling_path(self): # TODO: add split mode profile path
        if self.mem_path is not None:
            return self.mem_path
        assert self.model_name is not None, 'Should specify the model name!'
        args = self.args
        memory_config_name = 'memory_profiling_%s_%s_all.json'%(args.mixed_precision, self.model_name) # TODO: dynamic parse profile file
        if args.memory_profiling_path is None:
            memory_config_path = os.path.join(self.path, 'configs')
        else:
            memory_config_path = args.memory_profiling_path
        self.mem_path = os.path.join(memory_config_path, memory_config_name)
        return self.mem_path
    
    def time_profiling_path(self): # TODO: add split mode profile path
        if self.time_path is not None:
            return self.time_path
        assert self.model_name is not None, 'Should specify the model name!'
        args = self.args
        time_config_name = "computation_profiling_%s_%s_all.json"%(args.mixed_precision, self.model_name) # TODO: dynamic parse profile file
        if args.time_profiling_path is None:
            self.time_path = os.path.join(self.path, "configs")
        else:
            self.time_path = args.time_profiling_path

        self.time_path = os.path.join(self.time_path, time_config_name)
        return self.time_path
    
    def set_microbatch_func(self, microbatch_size, max_chunk):
        self.optimal_chunk_func = lambda local_bsz, strategy: optimal_chunk_func_default(local_bsz, strategy, microbatch_size)
    
    def set_model_layer_configs(self, model_layer_configs):
        if model_layer_configs is None:
            return
        self.hiddensize_list = [config['hidden_size'] for config in model_layer_configs]
        self.layernum_list = [config['layer_num'] for config in model_layer_configs]
        self.seqlen_list = [config['seq_len'] for config in model_layer_configs]
        self.num_layertype = len(self.layernum_list)
        self.total_layernum = sum(self.layernum_list)
    
    # =============== Initializing Galvatron Search Engine ===============
    # Generating Strategies, Loading Profiled Memory & Time Config, Setting Memory & Time Cost Models
    def initialize_search_engine(self):
        self.generate_strategy_list()
        self.filter_strategy_list()
        self.show_all_strategy_list()

        self.get_profiled_model_configs()
        self.get_profiled_hardware_configs()
        self.set_cost_models()

        self.show_search_info()

    # =========================== Generating Strategy List ===========================
    def generate_strategy_list(self) -> None:
        print(f'{"="*25}Enter generate_strategy_list{"="*25}')

        args = self.args
        default_dp_type = args.default_dp_type
        max_pp_deg = args.max_pp_deg
        max_tp_deg = args.max_tp_deg
        max_sp_deg = args.max_sp_deg
        max_cp_deg = args.max_cp_deg
        world_size = self.world_size

        degree_range = []
        tmp = 1
        while tmp <= self.world_size:
            degree_range.append(tmp)
            tmp *= 2

        print(f'generate_strategy_list: world_size={world_size}, degree_range={degree_range}, max_pp_deg={max_pp_deg}, max_tp_deg={max_tp_deg}, max_sp_deg={max_sp_deg}, max_cp_deg={max_cp_deg}, default_dp_type={default_dp_type}')

        attention_strategy_list:List[AttentionStrategy] = []
        ffn_strategy_list:List[FFNStrategy] = []
        embedding_lmhead_strategy_list:List[EmbeddingLMHeadStrategy] = []
        layer_strategy_list:List[LayerStrategy] = []

        # generate attention strategy list
        for pp_size in degree_range:
            if pp_size > self.total_layernum: # pp_size cannot be greater than total_layernum
                continue
            if pp_size > max_pp_deg:
                continue
            for tp_or_sp in ['tp', 'sp']:
                for tp_sp_size in degree_range:
                    if tp_or_sp == 'tp' and max_tp_deg != -1 and tp_sp_size > max_tp_deg:
                        continue
                    if tp_or_sp == 'sp' and max_sp_deg != -1 and tp_sp_size > max_sp_deg:
                        continue
                    if tp_sp_size * pp_size > world_size:
                        continue
                    for cp_size in degree_range:
                        if max_cp_deg != -1 and cp_size > max_cp_deg:
                            continue
                        if pp_size * tp_sp_size * cp_size > world_size:
                            continue
                        dp_size = world_size // pp_size // tp_sp_size // cp_size
                        dp_type_list = [DPType.DDP] if dp_size == 1 else ([DPType.DDP, DPType.ZERO3] if default_dp_type == 'ddp' else [DPType.ZERO2, DPType.ZERO3])
                        for dp_type in dp_type_list:
                            for checkpoint in [False, True]:
                                tp_size = tp_sp_size if tp_or_sp == 'tp' else 1
                                sp_size = tp_sp_size if tp_or_sp == 'sp' else 1
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
        
        self.embedding_lmhead_strategy_list = embedding_lmhead_strategy_list
        self.attention_strategy_list = attention_strategy_list
        self.ffn_strategy_list = ffn_strategy_list
        self.layer_strategy_list = layer_strategy_list

        print(f'{"="*25}Exit generate_strategy_list{"="*25}')

    def filter_strategy_list(self, disable_pp=None, disable_tp=None, disable_sp=None, disable_cp=None, disable_dp=None, disable_ckpt=None, disable_fsdp=None, disable_embedding_lmhead_tp=None, disable_embedding_lmhead_sp=None):
        print(f'{"="*25}Enter filter_strategy_list{"="*25}')

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
        
        disable_string = 'disbale'
        for name, value in params.items():
            if value is not None:
                setattr(args, name, value)
            if getattr(args, name) != 0:
                name_remove_disable = name.replace('disable_', '')
                disable_string += f'-{name_remove_disable}'
        
        print(f'filter_strategy_list: {disable_string}')

        if args.disable_pp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.pp_size == 1]
            self.attention_strategy_list = [strategy for strategy in self.attention_strategy_list if strategy.pp_size == 1]
            self.ffn_strategy_list = [strategy for strategy in self.ffn_strategy_list if strategy.pp_size == 1]
            self.layer_strategy_list = [strategy for strategy in self.layer_strategy_list if strategy.pp_size == 1]
        if args.disable_tp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.tp_size == 1]
            self.attention_strategy_list = [strategy for strategy in self.attention_strategy_list if strategy.tp_size == 1]
            self.ffn_strategy_list = [strategy for strategy in self.ffn_strategy_list if strategy.tp_size == 1]
            self.layer_strategy_list = [strategy for strategy in self.layer_strategy_list if strategy.tp_size == 1]
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
        if args.disable_ckpt:
            self.attention_strategy_list = [strategy for strategy in self.attention_strategy_list if strategy.checkpoint == False]
            self.ffn_strategy_list = [strategy for strategy in self.ffn_strategy_list if strategy.checkpoint == False]
            self.layer_strategy_list = [strategy for strategy in self.layer_strategy_list if strategy.checkpoint == False]
        if args.disable_fsdp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.dp_type != DPType.ZERO3]
            self.attention_strategy_list = [strategy for strategy in self.attention_strategy_list if strategy.dp_type != DPType.ZERO3]
            self.ffn_strategy_list = [strategy for strategy in self.ffn_strategy_list if strategy.dp_type != DPType.ZERO3]
            self.layer_strategy_list = [strategy for strategy in self.layer_strategy_list if strategy.dp_type != DPType.ZERO3]
        if args.disable_embedding_lmhead_tp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.tp_size == 1]
        if args.disable_embedding_lmhead_sp:
            self.embedding_lmhead_strategy_list = [strategy for strategy in self.embedding_lmhead_strategy_list if strategy.sp_size == 1]

        self.embedding_lmhead_strategy_list = sorted(list(set(self.embedding_lmhead_strategy_list)))
        self.attention_strategy_list = sorted(list(set(self.attention_strategy_list)))
        self.ffn_strategy_list = sorted(list(set(self.ffn_strategy_list)))
        self.layer_strategy_list = sorted(list(set(self.layer_strategy_list)))

        print(f'{"="*25}Exit filter_strategy_list{"="*25}')

    def show_all_strategy_list(self):
        print(f'{"="*25}Enter show_all_strategy_list{"="*25}')

        print(f'attention_strategy_list.size:{len(self.attention_strategy_list)}')
        print(f'ffn_strategy_list.size:{len(self.ffn_strategy_list)}')
        print(f'embedding_lmhead_strategy_list.size:{len(self.embedding_lmhead_strategy_list)}')
        print(f'layer_strategy_list.size:{len(self.layer_strategy_list)}')

        print()

        print(f'attention_strategy_list:\n{pretty_repr(self.attention_strategy_list, max_width=1024)}')
        print(f'ffn_strategy_list:\n{pretty_repr(self.ffn_strategy_list, max_width=1024)}')
        print(f'embedding_lmhead_strategy_list:\n{pretty_repr(self.embedding_lmhead_strategy_list, max_width=1024)}')
        print(f'layer_strategy_list:\n{pretty_repr(self.layer_strategy_list, max_width=1024)}')

        print(f'{"="*25}Exit show_all_strategy_list{"="*25}')

    # =========================== Parsing Profiled Configurations ===========================
    def convert_keys_to_int(self, d):
        if isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                if isinstance(k, str) and k.isdigit():
                    new_dict[int(k)] = self.convert_keys_to_int(v)
                else:
                    new_dict[k] = self.convert_keys_to_int(v)
            return new_dict
        return d
    
    def get_profiled_model_configs(self): # TODO: add split mode profile configs
        self.time_config = read_json_config(self.time_profiling_path())
        self.memory_config = read_json_config(self.memory_profiling_path())
        self.memory_config = self.convert_keys_to_int(self.memory_config)
        if self.args.time_profile_mode=='static':
            self.time_profiled_list = []
            self.other_time_profiled_list = []
            for i in range(self.num_layertype):
                for s,t in self.time_config.items():
                    if s.startswith('layertype_%d_'%i):
                        self.time_profiled_list.append(t)
                    if s.startswith('layertype_other_'):
                        self.other_time_profiled_list.append(t)
        elif self.args.time_profile_mode == "batch":
            self.time_profiled_list = []
            for i in range(self.num_layertype):
                x_data = []
                y_data = []
                for s,t in self.time_config.items():
                    if s.startswith('layertype_%d_'%i) and '_seq%d'%self.seqlen_list[i] in s:
                        x_data.append(int(s.split('_')[-2][3:]))
                        y_data.append(t * x_data[-1])
                assert len(x_data) >= 8, "Different bsz in computation profile of layertype_%d should not be lower than 8."%i
                
                def linear_func(x, m, c):
                    return m * x + c
                popt, pcov = curve_fit(linear_func, x_data, y_data)
                print("Fitted parameters:", popt)
                self.time_profiled_list.append(popt)
            self.other_time_profiled_list = []
            for i in range(self.num_layertype):
                x_data = []
                y_data = []
                for s,t in self.time_config.items():
                    if s.startswith('layertype_other_') and '_seq%d'%self.seqlen_list[i] in s:
                        x_data.append(int(s.split('_')[-2][3:]))
                        y_data.append(t * x_data[-1])
                assert len(x_data) >= 8, "Different bsz in computation profile of layertype_other_%d should not be lower than 8."%i
                
                def linear_func(x, m, c):
                    return m * x + c
                popt, pcov = curve_fit(linear_func, x_data, y_data)
                
                print("Fitted parameters other:", popt)
                self.other_time_profiled_list.append(popt)
        elif self.args.time_profile_mode == "sequence":
            self.time_profiled_list = []
            for i in range(self.num_layertype):
                x_data = []
                y_data = []
                for s,t in self.time_config.items():
                    if s.startswith('layertype_%d_'%i) and "_bsz1_" in s:
                        x_data.append(int(s.split('seq')[-1]))
                        y_data.append(t)
                # assert len(x_data) >= 8, "Different bsz in computation profile of layertype_%d should not be lower than 8."%i
                
                def quadratic_func(x, a, b, c):
                    return a * x * x + b * x + c
                popt, pcov = curve_fit(quadratic_func, x_data, y_data)
                print("Fitted parameters:", popt)
                self.time_profiled_list.append(quadratic_func(self.seqlen_list[i],*popt))
            self.other_time_profiled_list = []
            for i in range(self.num_layertype):
                x_data = []
                y_data = []
                for s,t in self.time_config.items():
                    if s.startswith('layertype_other_') and "_bsz1_" in s:
                        x_data.append(int(s.split('seq')[-1]))
                        y_data.append(t)
                # assert len(x_data) >= 8, "Different bsz in computation profile of layertype_other_%d should not be lower than 8."%i
                
                def linear_func(x, m, c):
                    return m * x + c
                popt, pcov = curve_fit(linear_func, x_data, y_data)
                print("Fitted parameters other:", popt)
                self.other_time_profiled_list.append(linear_func(self.seqlen_list[i],*popt))
        self.param_sizes = [0] * self.num_layertype
        self.act_sizes = [{} for _ in range(self.num_layertype)]
        if self.args.memory_profile_mode == "sequence":

            assert self.args.sequence_parallel, "Sequence parallel is required for sequence memory profiling."
            assert self.num_layertype == 1, "Only support num(layertype) == 1 for sequence memory profiling."
            maxseq_list = []
            for i in range(self.num_layertype):
                layer_mem_config = self.memory_config['layertype_%d_sp'%i]
                seqs = layer_mem_config.keys()
                maxseq = max([int(seq) for seq in seqs])
                minseq = min([int(seq) for seq in seqs])
                maxseq_list.append(maxseq)
                parameter_size = layer_mem_config[minseq]['parameter_size']
                tp_activation_per_bsz_dict = layer_mem_config[maxseq]['tp_activation_per_bsz_dict'].copy()
                self.param_sizes[i] = parameter_size
                self.act_sizes[i] = tp_activation_per_bsz_dict
                for tp in self.act_sizes[i]:
                    self.act_sizes[i][tp] = self.act_sizes[i][tp] / maxseq * self.seqlen_list[i]
            self.other_memory_pp_off = self.memory_config['other_memory_pp_off_sp'][maxseq_list[0]]
            self.other_memory_pp_on = {'first_stage':self.memory_config['other_memory_pp_on_first_sp'][maxseq_list[0]], 'last_stage':self.memory_config['other_memory_pp_on_last_sp'][maxseq_list[-1]]}
            # for tp in self.other_memory_pp_off['activation']:
            #     self.other_memory_pp_off['activation'][tp] = 2/3 * self.other_memory_pp_off['activation'][tp] + 1/3 * self.other_memory_pp_off['activation'][tp] / maxseq_list[0] * self.seqlen_list[0] # TODO: reasonable scaling when len(seqlen_list) > 1
            #     self.other_memory_pp_on['first_stage']['activation'][tp] = self.other_memory_pp_on['first_stage']['activation'][tp] # / maxseq_list[0] * self.seqlen_list[0] # first stage is not scaled
            #     self.other_memory_pp_on['last_stage']['activation'][tp] = self.other_memory_pp_on['last_stage']['activation'][tp] / maxseq_list[-1] * self.seqlen_list[-1] # last stage is scaled
            for tp in self.other_memory_pp_off['activation']:
                self.other_memory_pp_off['activation'][tp] = self.other_memory_pp_off['activation'][tp] / maxseq_list[0] * self.seqlen_list[0] # TODO: reasonable scaling when len(seqlen_list) > 1
                self.other_memory_pp_on['first_stage']['activation'][tp] = self.other_memory_pp_on['first_stage']['activation'][tp] / maxseq_list[0] * self.seqlen_list[0] # first stage is not scaled
                self.other_memory_pp_on['last_stage']['activation'][tp] = self.other_memory_pp_on['last_stage']['activation'][tp] / maxseq_list[-1] * self.seqlen_list[-1] # last stage is scaled
        elif self.args.memory_profile_mode == "static":
            if self.args.sequence_parallel:
                for i in range(self.num_layertype):
                    layer_mem_config = self.memory_config['layertype_%d_sp'%i]
                    parameter_size = layer_mem_config[self.seqlen_list[i]]['parameter_size']
                    tp_activation_per_bsz_dict = layer_mem_config[self.seqlen_list[i]]['tp_activation_per_bsz_dict'].copy()
                    self.param_sizes[i] = parameter_size
                    self.act_sizes[i] = tp_activation_per_bsz_dict
                seq_info = num2str(self.seqlen_list, 'seq')[3:]
                if seq_info.isdigit():
                    seq_info = int(seq_info)
                self.other_memory_pp_off = self.memory_config['other_memory_pp_off_sp'][int(seq_info)]
                self.other_memory_pp_on = {'first_stage':self.memory_config['other_memory_pp_on_first_sp'][seq_info], 'last_stage':self.memory_config['other_memory_pp_on_last_sp'][seq_info]}
            else:
                for i in range(self.num_layertype):
                    layer_mem_config = self.memory_config['layertype_%d'%i]
                    parameter_size = layer_mem_config[self.seqlen_list[i]]['parameter_size']
                    tp_activation_per_bsz_dict = layer_mem_config[self.seqlen_list[i]]['tp_activation_per_bsz_dict'].copy()
                    self.param_sizes[i] = parameter_size
                    self.act_sizes[i] = tp_activation_per_bsz_dict
                seq_info = num2str(self.seqlen_list, 'seq')[3:]
                if seq_info.isdigit():
                    seq_info = int(seq_info)
                self.other_memory_pp_off = self.memory_config['other_memory_pp_off'][seq_info]
                self.other_memory_pp_on = {'first_stage':self.memory_config['other_memory_pp_on_first'][seq_info], 'last_stage':self.memory_config['other_memory_pp_on_last'][seq_info]}
        
        return self.time_config, self.memory_config
        
    def get_profiled_hardware_configs(self):
        args = self.args
        if args.allreduce_bandwidth_config_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            allreduce_bandwidth_config_path = os.path.join(self.path, hardware_configs_dir)
        else:
            allreduce_bandwidth_config_path = args.allreduce_bandwidth_config_path
        allreduce_bandwidth_config_name = 'allreduce_bandwidth_%dnodes_%dgpus_per_node.json'%(args.num_nodes, args.num_gpus_per_node)
        args.allreduce_bandwidth_config_path  = os.path.join(allreduce_bandwidth_config_path, allreduce_bandwidth_config_name)
        self.allreduce_bandwidth, self.allreduce_comm_coe = read_allreduce_bandwidth_config(args.allreduce_bandwidth_config_path, gpu_num=args.gpu_num)
        
        if args.p2p_bandwidth_config_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            p2p_bandwidth_config_path = os.path.join(self.path, hardware_configs_dir)
        else:
            p2p_bandwidth_config_path = args.p2p_bandwidth_config_path
        p2p_bandwidth_config_name = 'p2p_bandwidth_%dnodes_%dgpus_per_node.json'%(args.num_nodes, args.num_gpus_per_node)
        args.p2p_bandwidth_config_path  = os.path.join(p2p_bandwidth_config_path, p2p_bandwidth_config_name)
        self.p2p_bandwidth, self.p2p_comm_coe = read_p2p_bandwidth_config(args.p2p_bandwidth_config_path)
        
        if args.overlap_coe_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            overlap_coe_path = os.path.join(self.path, hardware_configs_dir)
        else:
            overlap_coe_path = args.overlap_coe_path
        overlap_coe_name = 'overlap_coefficient.json'
        args.overlap_coe_path = os.path.join(overlap_coe_path, overlap_coe_name)
        self.overlap_coe = read_json_config(args.overlap_coe_path)['overlap_coe']
        if args.sp_time_path is None:
            hardware_configs_dir = '../../profile_hardware/hardware_configs/'
            sp_time_path = os.path.join(self.path, hardware_configs_dir)
        else:
            sp_time_path = args.sp_time_path
        sp_time_config_name = 'sp_time_%dnodes_%dgpus_per_node.json'%(args.num_nodes, args.num_gpus_per_node)
        args.sp_time_path = os.path.join(sp_time_path, sp_time_config_name)
        sp_config = read_json_config(args.sp_time_path)
        self.sp_allreduce = remap_config(sp_config, "allreduce")
        self.sp_all2all = remap_config(sp_config, "all2all")

        return self.allreduce_bandwidth, self.p2p_bandwidth, self.overlap_coe, self.sp_allreduce, self.sp_all2all

    def set_cost_models(self): # TODO: add split mode cost models
        self.model_args_list, self.train_args_list, self.parallel_args_list, self.profile_model_args_list, self.profile_hardware_args_list = [], [], [], [], []
        for i in range(self.num_layertype):
            model_args = ModelArgs(
                parameter_size=self.param_sizes[i],
                seq_length=self.seqlen_list[i],
                hidden_size=self.hiddensize_list[i],
                layer_num=self.layernum_list[i],
            )
            train_args = TrainArgs(
                mixed_precision=False if self.args.mixed_precision == 'fp32' else True,
                async_grad_reduce=self.args.async_grad_reduce,
            )
            parallel_args = ParallelArgs(
                use_zero2_for_dp=True if self.args.default_dp_type == 'zero2' else False,
                disable_vtp=self.args.disable_vtp,
                sequence_parallel=self.args.sequence_parallel,
                sp_space=self.args.sp_space,
                pipeline_type=self.args.pipeline_type,
                optimal_chunk_func=self.optimal_chunk_func,
            )
            profile_model_args = ProfileModelArgs(
                tp_activation_per_bsz_dict=self.act_sizes[i],
                other_memory_pp_off=self.other_memory_pp_off,
                other_memory_pp_on=self.other_memory_pp_on,
                forward_computation_time=self.time_profiled_list[i],
                other_time_profiled=self.other_time_profiled_list[0],
            )
            profile_hardware_args = ProfileHardwareArgs(
                bct_fct_coe=2,
                extra_overhead=0,
                comm_coe_dict=self.allreduce_comm_coe,
                dp_overlap_coe=self.overlap_coe,
                bct_overlap_coe=self.overlap_coe,
                p2p_comm_coe_dict=self.p2p_comm_coe,
                costmodel_coe=self.args.costmodel_coe,
                allreduce_dict=self.sp_allreduce,
                all2all_dict=self.sp_all2all,
            )
            self.model_args_list.append(model_args)
            self.train_args_list.append(train_args)
            self.parallel_args_list.append(parallel_args)
            self.profile_model_args_list.append(profile_model_args)
            self.profile_hardware_args_list.append(profile_hardware_args)
    
    # =============== For Galvatron Search Engine Parallelism Optimization ===============
    def get_pp_size_range(self) -> None:
        self.pp_size_range = []
        assert hasattr(self, 'embedding_lmhead_strategy_list'), f"{ColorSet.RED}[ERROR] [{self.__class__.__name__}] embedding_lmhead_strategy_list is not set.{ColorSet.RESET}"
        for strategy in self.embedding_lmhead_strategy_list:
            self.pp_size_range.append(strategy.pp_size)
        self.pp_size_range = sorted(list(set(self.pp_size_range)))
        print(f'pp size range: {self.pp_size_range}')

    def parallelism_optimization(self):
        print('='*25, 'Galvatron Search Engine Start Searching','='*25)
        print('-----', '[Searching Memory Info]', 'Memory constraint:', self.memory_constraint, 'MB', '-----')
        
        # [Step 1] Preparation Works
        results = dict()
        self.get_pp_size_range()
        self.tp_sp_mode_space = ['tp_only', 'sp_only', 'tp_with_sp']
        self.set_searching_bsz()

        # [Step 2] Get all possible
        all_tasks = []
        for gbsz in self.BSZs:
            results[gbsz] = dict()
            chunk_list = range(1, gbsz+1)
            if self.args.settle_chunk != -1:
                chunk_list = [self.args.settle_chunk]
            
            for chunks in chunk_list:
                if gbsz % chunks != 0:
                    continue
                results[gbsz][chunks] = dict()

                for pp_size in self.pp_size_range:
                    if pp_size > chunks:
                        print(f'pp_size({pp_size}) > chunks({chunks}), skip')
                        continue
                    if pp_size > self.total_layernum:
                        print(f'pp_size({pp_size}) > total_layernum({self.total_layernum}), skip')
                        continue
                    results[gbsz][chunks][pp_size] = dict()

                    theoretical_max_tp_size = self.args.gpu_num // pp_size
                    theoretical_max_tp_size = max(theoretical_max_tp_size, 1)
                    if self.args.max_tp_deg != -1 and theoretical_max_tp_size > self.args.max_tp_deg:
                        theoretical_max_tp_size = self.args.max_tp_deg

                    theoretical_max_dp_size = min(gbsz // chunks, self.args.gpu_num // pp_size)
                    theoretical_max_dp_size = max(theoretical_max_dp_size, 1)
                    theoretical_min_tp_size = self.args.gpu_num // pp_size // theoretical_max_dp_size
                    theoretical_min_tp_size = max(theoretical_min_tp_size, 1)

                    for tp_sp_mode in self.tp_sp_mode_space:
                        results[gbsz][chunks][pp_size][tp_sp_mode] = dict()
                        
                        if tp_sp_mode == 'sp_only':
                            consider_max_tp_size_list = [theoretical_max_tp_size]
                        else:
                            consider_max_tp_size_list = []
                            for i in range(theoretical_min_tp_size, theoretical_max_tp_size + 1):
                                if is_power_of_two(i) and i * pp_size <= self.args.gpu_num:
                                    consider_max_tp_size_list.append(i)
                        
                        for global_buffer_tp_size in consider_max_tp_size_list:
                            results[gbsz][chunks][pp_size][tp_sp_mode][global_buffer_tp_size] = dict()
                            all_tasks.append((gbsz, chunks, pp_size, tp_sp_mode, global_buffer_tp_size))

        # [Step 3] Search
        print(f'self.args.parallel_search: {self.args.parallel_search}')
        if self.args.parallel_search:
            import concurrent.futures
            import threading
            import multiprocessing
            
            results_lock = threading.Lock()
            if hasattr(self.args, 'worker') and self.args.worker > 0:
                num_threads = min(self.args.worker, len(all_tasks))
            else:
                num_threads = min(multiprocessing.cpu_count() * 2, len(all_tasks))
            print(f"Starting parallel search with {num_threads} threads for {len(all_tasks)} tasks...")
            
            def process_task(gbsz, chunks, pp_size, tp_sp_mode, global_buffer_tp_size):
                thread_id = threading.get_ident() % 1000
                print(f"[Thread {thread_id:03d}] Start processing: gbsz={gbsz}, chunks={chunks}, pp_size={pp_size}, tp_sp_mode={tp_sp_mode}, global_buffer_tp_size={global_buffer_tp_size}", flush=True)
                try:
                    chunk_results = self.search_for_single_task(gbsz, chunks, pp_size, global_buffer_tp_size, tp_sp_mode)
                except Exception as e:
                    print(f"[Thread {thread_id:03d}] Task failed (gbsz={gbsz}, chunks={chunks}, pp_size={pp_size}, tp_sp_mode={tp_sp_mode}, global_buffer_tp_size={global_buffer_tp_size}): {e}")
                    raise e
                with results_lock:
                    results[gbsz][chunks][pp_size][tp_sp_mode][global_buffer_tp_size] = copy.deepcopy(chunk_results)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_task, gbsz, chunks, pp_size, tp_sp_mode, global_buffer_tp_size) for gbsz, chunks, pp_size, tp_sp_mode, global_buffer_tp_size in all_tasks]
                concurrent.futures.wait(futures)
        else:
            print(f"Starting sequential search with {len(all_tasks)} tasks...")
            for task_idx, task in enumerate(all_tasks):
                gbsz, chunks, pp_size, tp_sp_mode, global_buffer_tp_size = task
                print(f"Start processing: {task_idx}-th task, gbsz={gbsz}, chunks={chunks}, pp_size={pp_size}, tp_sp_mode={tp_sp_mode}, global_buffer_tp_size={global_buffer_tp_size}", flush=True)
                results[gbsz][chunks][pp_size][tp_sp_mode][global_buffer_tp_size] = self.search_for_single_task(gbsz, chunks, pp_size, global_buffer_tp_size, tp_sp_mode)
        
        # [Step 4] Select the optimal solution and save results
        max_throughput, optimal_bsz = -1, -1
        for bsz in results:
            for chunk in results[bsz]:
                for pp_size in results[bsz][chunk]:
                    for tp_sp_mode in results[bsz][chunk][pp_size]:
                        for global_buffer_tp_size in results[bsz][chunk][pp_size][tp_sp_mode]:
                            throughput = results[bsz][chunk][pp_size][tp_sp_mode][global_buffer_tp_size]['throughput']
                            if throughput > max_throughput:
                                max_throughput = throughput
                                optimal_bsz = bsz
                                optimal_chunk = chunk
                                optimal_pp_size = pp_size
                                optimal_global_buffer_tp_size = global_buffer_tp_size
                                optimal_tp_sp_mode = tp_sp_mode

        if max_throughput > 0:
            print('\nFinal results of max memory %d MB:'%self.memory_constraint)
            optimal = results[optimal_bsz][optimal_chunk][optimal_pp_size][optimal_tp_sp_mode][optimal_global_buffer_tp_size]
            
            print(f'Optimal gbsz = {optimal_bsz} Optimal chunk = {optimal_chunk} Optimal pp_size = {optimal_pp_size} Optimal tp_sp_mode = {optimal_tp_sp_mode} Optimal global_buffer_tp_size = {optimal_global_buffer_tp_size}')
            print(f"Minized timecost = {optimal['time_cost']} Memory remaining = {optimal['memory_remain']} Memory cost = {optimal['memory_cost']}")
            print(f"Embedding LMHead tp_sp_size = {optimal['embedding_lmhead_tp_sp_size']} Embedding LMHead sp = {optimal['embedding_lmhead_sp']} Embedding LMHead sdp = {optimal['embedding_lmhead_sdp']}")
            print_strategy_list(optimal['strategy_list'])

            self.save_results(optimal, optimal_bsz, optimal_chunk)
        else:
            print("No valid configuration found.")
        
        print("-----------------------------------------")
        print('='*25, 'Galvatron Search Engine End Searching','='*25)

        return max_throughput

    def search_for_single_task(self, gbsz, chunks, pp_size, global_buffer_tp_size, tp_sp_mode) -> dict[str, Any]:
        args = self.args

        # [Step 1] log initialization
        log_dir = self.args.log_dir + '/%s_%dnodes_%dgpus_%dGB'%(self.model_name, self.args.num_nodes, self.args.num_gpus_per_node, self.memory_constraint//1024)
        log_dir = ensure_log_dir(log_dir)
        logger = get_thread_logger_single_task(gbsz, chunks, pp_size, global_buffer_tp_size, tp_sp_mode, log_dir)
        logger.info(f"Starting search for gbsz={gbsz}, chunks={chunks}, pp_size={pp_size}, global_buffer_tp_size={global_buffer_tp_size}, tp_sp_mode={tp_sp_mode}")

        # [Step 2] filter strategies
        theoretical_max_dp_size = min(gbsz // chunks, self.args.gpu_num // pp_size)
        theoretical_max_dp_size = max(theoretical_max_dp_size, 1)
        
        def filter_strategies_for_single_task(original_strategy_list:Union[List[LayerStrategy], List[EmbeddingLMHeadStrategy]], pp_size, max_tp_size, max_dp_size, tp_sp_mode):
            strategy_list:List[Union[LayerStrategy, EmbeddingLMHeadStrategy]] = [strategy for strategy in original_strategy_list if strategy.pp_size == pp_size]
            strategy_list = [strategy for strategy in strategy_list if strategy.tp_sp_size <= max_tp_size] 
            strategy_list = [strategy for strategy in strategy_list if strategy.dp_size <= max_dp_size]
            if tp_sp_mode == 'tp_only':
                strategy_list = [strategy for strategy in strategy_list if strategy.sp_size == 1]
            elif tp_sp_mode == 'sp_only':
                strategy_list = [strategy for strategy in strategy_list if strategy.tp_size == 1]
            elif tp_sp_mode == 'tp_with_sp':
                pass
            return strategy_list
        
        filter_layer_strategy_list = filter_strategies_for_single_task(self.layer_strategy_list, pp_size, global_buffer_tp_size, theoretical_max_dp_size, tp_sp_mode)
        filter_embedding_lmhead_strategy_list = filter_strategies_for_single_task(self.embedding_lmhead_strategy_list, pp_size, global_buffer_tp_size, theoretical_max_dp_size, tp_sp_mode)
        if len(filter_layer_strategy_list) == 0 or len(filter_embedding_lmhead_strategy_list) == 0:
            logger.info(f"No strategies found for gbsz={gbsz}, chunks={chunks}, pp_size={pp_size}, global_buffer_tp_size={global_buffer_tp_size}, tp_sp_mode={tp_sp_mode}")
            return {'throughput': -1}

        # [Step 3] get pp_stage_list # TODO: Consider a more flexible splitting method.
        pp_stage_list = pp_division_even(self.layernum_list, pp_size) # List[int]

        # [Step 4] dynamic programming
        dp_on_model = DpOnModel(
            model_args_list=self.model_args_list,
            train_args_list=self.train_args_list,
            parallel_args_list=self.parallel_args_list,
            profile_model_args_list=self.profile_model_args_list,
            profile_hardware_args_list=self.profile_hardware_args_list,
            max_mem=self.memory_constraint,
            layer_num=self.layernum_list,
            sequence_len = self.seqlen_list,
            comm_coe_dict=self.allreduce_comm_coe,
            world_size=args.gpu_num,
            pipeline_type=args.pipeline_type,
            config = self.args,
            logger=logger
        )
        
        optimal = dp_on_model.fit(
            gbsz=gbsz, 
            chunks=chunks, 
            pp_size=pp_size,
            pp_stage_list=pp_stage_list,
            global_buffer_tp_size=global_buffer_tp_size, 
            tp_sp_mode=tp_sp_mode,
            layer_strategy_list=filter_layer_strategy_list,
            embedding_lmhead_strategy_list=filter_embedding_lmhead_strategy_list
        )

        # [Step 5] gather info
        throughput = gbsz / optimal['time_cost'] # if no solution, optimal['time_cost'] is np.inf
        logger.info(f'optimal: {optimal}')
        logger.info(f"Max throughput={throughput} samples/s")
        print_strategy_list(optimal['strategy_list'], logger)

        result = {
            'throughput': throughput,
            'time_cost': optimal['time_cost'],
            'strategy_list': optimal['strategy_list'],
            'pp_size': pp_size,
            'pp_stage_list': pp_stage_list,
            'memory_remain': optimal['memory_remain'],
            'memory_cost': optimal['memory_used'],
            'embedding_lmhead_tp_sp_size': optimal['embedding_lmhead_tp_sp_size'],
            'embedding_lmhead_sp': optimal['embedding_lmhead_sp'],
            'embedding_lmhead_sdp': optimal['embedding_lmhead_sdp'],
        }

        return result

    def set_searching_bsz(self):
        args = self.args

        if args.settle_bsz is not None and args.settle_bsz > 0:
            self.min_bsz = self.max_bsz = args.settle_bsz
            self.bsz_scale = 0
            self.BSZs = [args.settle_bsz]
            print('-----', '[Searching Batch Sizes Info]', 'Settle bsz:', args.settle_bsz, '-----')
            print('-----', '[Searching Batch Sizes Info]', 'BSZs:', self.BSZs, '-----')
        else:
            assert args.min_bsz is not None and args.max_bsz is not None and args.bsz_scale is not None
            assert args.min_bsz > 0 and args.max_bsz > 0 and args.bsz_scale > 0
            assert args.max_bsz >= args.min_bsz
            self.min_bsz = args.min_bsz
            self.max_bsz = args.max_bsz
            self.bsz_scale = args.bsz_scale
            self.BSZs = list(range(self.min_bsz, self.max_bsz + 1, self.bsz_scale))
            print('-----', '[Searching Batch Sizes Info]', 'Min bsz:', self.min_bsz, 'Max bsz:', self.max_bsz, 'bsz_scale:', self.bsz_scale, '-----')
            print('-----', '[Searching Batch Sizes Info]', 'BSZs:', self.BSZs, '-----')

    def save_results(self, optimal, optimal_bsz, chunk):
        args = self.args

        result_strategy = optimal['strategy_list']
        config = strategy_list2config(result_strategy)
        config['global_bsz'] = optimal_bsz
        config['chunks'] = chunk
        config['pp_division'] = array2str(optimal['pp_stage_list'])
        config['pipeline_type'] = args.pipeline_type
        config['default_dp_type'] = args.default_dp_type
        config['vtp'] = optimal['embedding_lmhead_tp_sp_size']
        config['vsp'] = optimal['embedding_lmhead_sp']
        config['embed_sdp'] = optimal['embedding_lmhead_sdp']
        
        mixed_precision = '_%s'%args.mixed_precision
        settle_bsz = '_bsz%d'%args.settle_bsz if args.settle_bsz > 0 else ''
        off_options = []
        if args.disable_dp:
            off_options.append('dp')
        if args.disable_tp:
            off_options.append('tp')
        if args.disable_pp:
            off_options.append('pp')
        if args.disable_sdp:
            off_options.append('sdp')
        if args.disable_ckpt:
            off_options.append('ckpt')
        if args.disable_tp_consec:
            off_options.append('tpconsec')
        off_options_str = '_[%s_off]'%('_'.join(off_options))if len(off_options) else ''
        config_path = args.output_config_path
        if config_path is None:
            config_path = os.path.join(self.path, 'configs/')
        output_config_name = 'galvatron_config_%s_%dnodes_%dgpus_per_node_%dGB'%(self.model_name, args.num_nodes, args.num_gpus_per_node, self.memory_constraint//1024)
        output_config_name = output_config_name + mixed_precision + settle_bsz + off_options_str + '.json'
        config_path = os.path.join(config_path, output_config_name)
        print(config_path)
        write_json_config(config, config_path)
        print('Already written optimized parallelism config into galvatron config file %s!'%(config_path))

    # =========================== Checking Cost Model (For Developer)===========================
    def check_cost_model(self, gbsz, chunks, specific_strategy_list:List[LayerStrategy] = None):
        print(f'=============== Checking Cost Model for gbsz={gbsz}, chunks={chunks} ==================')
        assert self.num_layertype == 1 # # NOTE only for decode-only model
        assert hasattr(self, 'layer_strategy_list'), f"{ColorSet.RED}[ERROR] [{self.__class__.__name__}] layer_strategy_list is not set.{ColorSet.RESET}"
        assert gbsz % chunks == 0, f"{ColorSet.RED}[ERROR] [{self.__class__.__name__}] gbsz {gbsz} is not divisible by chunks {chunks}.{ColorSet.RESET}"

        total_layernum = self.total_layernum

        if specific_strategy_list is not None:
            layer_strategy_list = specific_strategy_list
        else:
            layer_strategy_list = self.layer_strategy_list
        layer_strategy_num = len(layer_strategy_list)
        time_cost_each_strategy = [-1 for _ in range(layer_strategy_num)]
        memory_cost_each_strategy = [None for _ in range(layer_strategy_num)]

        for layer_strategy_idx, layer_strategy in enumerate(layer_strategy_list):
            print(f'start check layer_strategy: {layer_strategy_idx}-th, strategy: {layer_strategy}')
            embedding_lmhead_strategy = layer_strategy.to_embedding_lmhead_strategy()

            pp_size = layer_strategy.pp_size
            dp_size = layer_strategy.dp_size
            if pp_size > chunks:
                print(f'pp_size {pp_size} is greater than chunks {chunks}, skip')
                continue
            if gbsz // chunks < dp_size:
                print(f'gbsz // chunks {gbsz // chunks} is less than dp_size {dp_size}, skip')
                continue
            
            partition = pp_division_even(self.layernum_list, pp_size) # len(partition) == pp_size. partition[stage_idx] means the number of layers in the stage_idx-th stage

            # =========================== Time Cost Model ===========================
            embedding_lmhead_time_obj = EmbeddingLMHeadTimeCostModel(
                strategy=embedding_lmhead_strategy,
                global_batch_size=gbsz,
                chunks=chunks,
                logger=None,
                sequence_length_list=self.seqlen_list,
                model_args=self.model_args_list[0],
                train_args=self.train_args_list[0],
                parallel_args=self.parallel_args_list[0],
                profile_model_args=self.profile_model_args_list[0],
                profile_hardware_args=self.profile_hardware_args_list[0]
            )
            embedding_lmhead_time, embedding_lmhead_time_no_grad_sync = embedding_lmhead_time_obj.gen_result()
            strategy_list = [layer_strategy for _ in range(total_layernum)] # 每一层都采用此策略
            
            pipeline_time = pipeline_costmodel(
                layer_num_list=self.layernum_list,
                model_args_list=self.model_args_list,
                train_args_list=self.train_args_list,
                parallel_args_list=self.parallel_args_list,
                profile_model_args_list=self.profile_model_args_list,
                profile_hardware_args_list=self.profile_hardware_args_list,
                strategy_list=strategy_list,
                partition=partition,
                chunks=chunks,
                gbsz=gbsz,
                pp_size=pp_size,
                other_time_cost=embedding_lmhead_time_no_grad_sync,
                logger=None,
                return_stage_cost=False
            )
            time_cost_each_strategy[layer_strategy_idx] = pipeline_time

            # =========================== Memory Cost Model ===========================
            memory_cost = [0 for _ in range(pp_size)]
            embedding_lmhead_memory_cost_obj = EmbeddingLMHeadMemoryCostModel(
                strategy=embedding_lmhead_strategy,
                global_batch_size=gbsz,
                chunks=chunks,
                logger=None,
                model_args=self.model_args_list[0],
                train_args=self.train_args_list[0],
                parallel_args=self.parallel_args_list[0],
                profile_model_args=self.profile_model_args_list[0]
            )
            embedding_lmhead_memory_cost = embedding_lmhead_memory_cost_obj.get_memory_cost()
            embedding_lmhead_memory_cost = embedding_lmhead_memory_cost['enc_total']

            for stage_idx in range(pp_size):
                memory_cost[stage_idx] += embedding_lmhead_memory_cost[stage_idx]
                layer_memory_cost_obj = MemoryCostModelBase(
                    strategy=layer_strategy,
                    global_batch_size=gbsz,
                    chunks=chunks,
                    stage_idx=stage_idx,
                    logger=None,
                    model_args=self.model_args_list[0], # because only one layertype
                    train_args=self.train_args_list[0], # because only one layertype
                    parallel_args=self.parallel_args_list[0], # because only one layertype
                    profile_model_args=self.profile_model_args_list[0] # because only one layertype
                )
                layer_memory_cost = layer_memory_cost_obj.get_memory_cost()
                layer_memory_cost = layer_memory_cost['enc_total']
                memory_cost[stage_idx] += layer_memory_cost * partition[stage_idx]

            memory_cost_each_strategy[layer_strategy_idx] = memory_cost
        
        # =========================== Print Time Cost ===========================
        print()
        for layer_strategy_idx in range(layer_strategy_num):
            strategy_string = layer_strategy_list[layer_strategy_idx].to_simple_string()
            print(f'{strategy_string}: {time_cost_each_strategy[layer_strategy_idx]}')

        # =========================== Print Memory Cost ===========================
        print()
        for layer_strategy_idx in range(layer_strategy_num):
            strategy_string = layer_strategy_list[layer_strategy_idx].to_simple_string()
            print(f'{strategy_string}: {memory_cost_each_strategy[layer_strategy_idx]}')

    # =============== Search Engine Info Utils ===============
    def show_search_info(self):
        print('================================================================================')
        print('--- Optimization Configs ----')
        print('Memory constraint: %d GB'%self.args.memory_constraint)
        print('Pipeline Type:', self.args.pipeline_type)
        print('Default DP Type:', self.args.default_dp_type)
        print('Mixed Precision:', self.args.mixed_precision)
        print('================================================================================')
        print('---- Environment Configs ----')
        print('Allreduce Bandwidth (GB/s):', self.allreduce_bandwidth)
        print('Allreduce Communication Coefficient (ms/MB):', self.allreduce_comm_coe)
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
        print('Parameter Memory Cost:', self.param_sizes)
        print('Activation Memory Cost of Different TP degree (per bsz):')
        print(self.act_sizes)
        print('Other Memory Cost (pp = 1):')
        print(self.other_memory_pp_off)
        print('Other Memory Cost (pp > 1):')
        print(self.other_memory_pp_on)
        print('================================================================================')
        print('Model Args List:')
        print(self.model_args_list)
        print('================================================================================')
        print('Train Args List:')
        print(self.train_args_list)
        print('================================================================================')
        print('Parallel Args List:')
        print(self.parallel_args_list)
        print('================================================================================')
        print('Profile Model Args List:')
        print(self.profile_model_args_list)
        print('================================================================================')
        print('Profile Hardware Args List:')
        print(self.profile_hardware_args_list)
        print('================================================================================')


# ========================== Pipeline Division & Pipeline Cost Utils ==========================
def pp_division_memory_balanced(model_args_list, train_args_list, parallel_args_list, profile_model_args_list, layer_num, pp_deg, bsz, mbsz, strategies):
    return None, None
    # model_args_list, train_args_list= [copy.deepcopy(model_args_list[i]) for i in range(len(layer_num))], [copy.deepcopy(train_args_list[i]) for i in range(len(layer_num))]
    # parallel_args_list, profile_model_args_list = [copy.deepcopy(parallel_args_list[i]) for i in range(len(layer_num))], [copy.deepcopy(profile_model_args_list[i]) for i in range(len(layer_num))]
    # for i in range(len(parallel_args_list)):
    #     parallel_args_list[i].pipeline_type = 'gpipe'
    # assert(len(model_args_list) == len(layer_num) and len(train_args_list) == len(layer_num) and len(parallel_args_list) == len(layer_num) and len(profile_model_args_list) == len(layer_num))
    # if pp_deg == 1:
    #     return [np.sum(layer_num)], None
    # layer_type_num = len(layer_num)
    # layer_min_memcost = []
    # strategies = list(filter(lambda s: s[0] == pp_deg, strategies))
    # if len(strategies)==0:
    #     return None, None
    # gpu_num = strategies[0][0] * strategies[0][1] * strategies[0][2]
    # for i in range(layer_type_num):
    #     # memcosts = [MemoryCostModel(strategy, global_batch_size=bsz, model_args=model_args_list[i], train_args=train_args_list[i], parallel_args=parallel_args_list[i], profile_model_args=profile_model_args_list[i]).get_memory_cost()['enc_total'] for strategy in strategies]
    #     # layer_min_memcost.append(np.min(memcosts))
    #     memcost = MemoryCostModel([pp_deg, 1, gpu_num//pp_deg, {}], global_batch_size=bsz, mbsz = mbsz, min_tp = 1, max_tp = 1,
    #                               model_args=model_args_list[i], train_args=train_args_list[i], parallel_args=parallel_args_list[i], profile_model_args=profile_model_args_list[i]).get_memory_cost()['enc_total']
    #     layer_min_memcost.append(np.min(memcost))
    # other_cost = MemoryCostModel(strategies[0], global_batch_size=bsz, mbsz = mbsz, min_tp = 1, max_tp = 1,
    #                              model_args=model_args_list[0], train_args=train_args_list[0], parallel_args=parallel_args_list[0], profile_model_args=profile_model_args_list[0]).get_memory_cost()['other'][1]
    # # print(other_cost)
    # # print(layer_min_memcost, other_cost)
    # min_memcost_all_layers = []
    # for i in range(layer_type_num):
    #     min_memcost_all_layers += [layer_min_memcost[i]] * layer_num[i]
    # # print(min_memcost_all_layers)
    # avg_mem_cost = (np.sum(min_memcost_all_layers) + np.sum(other_cost)) / pp_deg
    # # print(min_memcost_all_layers, other_cost)
    # # print('Avg memcost:', avg_mem_cost)

    # pp_divide = [0] * pp_deg
    # mem_cost_per_stage = other_cost.copy()
    # idx = 0
    # for i in range(pp_deg):
    #     while True:
    #         if idx >= len(min_memcost_all_layers):
    #             break
    #         if i < pp_deg - 1 and avg_mem_cost - mem_cost_per_stage[i] < 0.5 * min_memcost_all_layers[idx]:
    #             break
    #         else:
    #             mem_cost_per_stage[i] += min_memcost_all_layers[idx]
    #             idx += 1
    #             pp_divide[i] += 1

    # # Avoid too much memory cost on previous stages
    # for i in range(pp_deg - 1):
    #     left, right = int(np.sum(pp_divide[:i])), int(np.sum(pp_divide[:i+1]))
    #     mem_cost_cur_stage = np.sum(min_memcost_all_layers[left:right]) + other_cost[i]
    #     while mem_cost_cur_stage > avg_mem_cost * 1.3:
    #         pp_divide[i] -= 1
    #         pp_divide[i+1] += 1
    #         right -= 1
    #         mem_cost_cur_stage -= min_memcost_all_layers[right]

    # # Avoid no layers on previous stages
    # for i in range(pp_deg-1):
    #     while pp_divide[i] <= 0:
    #         pp_divide[i] += 1
    #         pp_divide[i+1] -= 1

    # # Avoid no layers on last stage
    # for i in range(pp_deg-1, 0, -1):
    #     while pp_divide[i] <= 0:
    #         pp_divide[i] += 1
    #         pp_divide[i-1] -= 1
    
    # mem_cost_per_stage_adjusted = other_cost.copy()
    # # print(pp_divide)
    # # print(other_cost, avg_mem_cost)
    # for i in range(pp_deg):
    #     left, right = int(np.sum(pp_divide[:i])), int(np.sum(pp_divide[:i+1]))
    #     mem_cost_per_stage_adjusted[i] +=  np.sum(min_memcost_all_layers[left:right])
    # # print(mem_cost_per_stage,mem_cost_per_stage_adjusted)
    # return pp_divide, mem_cost_per_stage_adjusted

def get_pp_stage_for_bsz(strategies, model_args_list, train_args_list, parallel_args_list, profile_model_args_list, layer_num_list, bsz, mbsz_dict, single_layer_even=True):
    pp_stage_dict = dict()
    pp_deg_list = sorted(list(set([s[0] for s in strategies])))
    for pp_deg in pp_deg_list:
        if single_layer_even and len(layer_num_list) == 1:
            pp_divide = pp_division_even(layer_num_list, pp_deg)
        else:
            pp_divide, mem_cost_per_stage = pp_division_memory_balanced(model_args_list, train_args_list, parallel_args_list, profile_model_args_list, layer_num_list, pp_deg, bsz, mbsz_dict[pp_deg], strategies)
            #print(bsz, pp_deg, pp_divide, mem_cost_per_stage)
        pp_stage_dict[pp_deg] = pp_divide
    return pp_stage_dict

def get_cost_all_stages(layer_memcosts, pp_stage_division):
    pp_stage_division = copy.deepcopy(pp_stage_division)
    # include other memory on first stage
    if np.sum(pp_stage_division) + 1 == len(layer_memcosts):
        pp_stage_division[0] += 1
    elif np.sum(pp_stage_division) + 2 == len(layer_memcosts):
        pp_stage_division[0] += 1
        pp_stage_division[-1] += 1
        dist_costmodel = True
    assert(np.sum(pp_stage_division)==len(layer_memcosts))
    stage_memcosts = []
    for stage_id in range(len(pp_stage_division)):
        layer_start_id, layer_end_id = int(np.sum(pp_stage_division[:stage_id])), int(np.sum(pp_stage_division[:stage_id+1]))
        stage_memcosts.append(np.sum(layer_memcosts[layer_start_id:layer_end_id]))
    return stage_memcosts

def get_layer_costs(layernum_list, layer_costs):
    layer_memcosts = []
    for i in range(len(layernum_list)):
        layer_memcosts += [layer_costs[i]]*layernum_list[i]
    return layer_memcosts
    
def pp_division_even(layernum_list, pp_deg):
    total_layer_num = np.sum(layernum_list)
    avg_layer_num = int(total_layer_num // pp_deg)
    last_layer_num = total_layer_num - avg_layer_num * (pp_deg-1)
    pp_division = [avg_layer_num] * (pp_deg-1) + [last_layer_num]
    return pp_division
    
def optimal_chunk_func_default(local_bsz, strategy, microbatch_size, min_tp):
    # if strategy[0] == 1:
    #     return 1
    assert(strategy[1] % min_tp == 0)
    local_bsz = local_bsz // (strategy[1] // min_tp)
    chunk = np.ceil(local_bsz / microbatch_size)
    chunk = 1 if chunk == 0 else chunk
    # chunk = int(min(max_chunk,chunk))
    return chunk

def check_optimal_chunks(world_size, strategies, optimal_chunk_func, bsz, mbsz_dict, min_tp):
    chunk_dict = {}
    for pp_deg in sorted(set([s[0] for s in strategies])):
        chunk_dict[pp_deg] = optimal_chunk_func(bsz / (world_size // pp_deg // min_tp), [pp_deg, min_tp, world_size // pp_deg, {'fsdp':0, 'cpt':0}], mbsz_dict[pp_deg], min_tp)
    return chunk_dict