import numpy as np
from typing import List

from galvatron.utils.strategy_utils import LayerStrategy
from galvatron.core.cost_model.components.layer_cost import TimeCostModelBase


def get_time_cost_all_stages(layer_timecosts, pp_stage_division):
    assert(np.sum(pp_stage_division) == len(layer_timecosts))
    stage_timecosts = []
    for stage_id in range(len(pp_stage_division)):
        layer_start_id, layer_end_id = int(np.sum(pp_stage_division[:stage_id])), int(np.sum(pp_stage_division[:stage_id+1]))
        stage_timecosts.append(np.sum(layer_timecosts[layer_start_id:layer_end_id]))
    return stage_timecosts

def pipeline_costmodel(
    layer_num_list, 
    model_args_list, 
    train_args_list, 
    parallel_args_list, 
    profile_model_args_list, 
    profile_hardware_args_list, 
    strategy_list:List[LayerStrategy], 
    partition, 
    chunks, 
    gbsz,
    pp_size,
    other_time_cost, 
    logger=None, 
    return_stage_cost=False
):
    num_layertype = len(layer_num_list)
    total_layer_num = sum(layer_num_list)
    layertype_ids = []
    for layertype_id in range(num_layertype):
        layertype_ids.extend([layertype_id for _ in range(layer_num_list[layertype_id])])
    
    strategy_num = len(strategy_list)
    assert strategy_num == total_layer_num, f"strategy_num != total_layer_num, {strategy_num} != {total_layer_num}"
    strategy_set = list(set(strategy_list))  # Deduplicate strategies to avoid duplicate calculation

    timecosts_dict_bsz_chunked, timecosts_dict_compute = {}, {}
    for layertype_id in range(num_layertype):
        timecosts_dict_bsz_chunked[layertype_id], timecosts_dict_compute[layertype_id] = {}, {}
        for strategy in strategy_set:
            string = strategy.to_string()
            obj = TimeCostModelBase(
                strategy=strategy,
                global_batch_size=gbsz,
                chunks=chunks,
                model_args=model_args_list[layertype_id],
                train_args=train_args_list[layertype_id],
                parallel_args=parallel_args_list[layertype_id],
                profile_model_args=profile_model_args_list[layertype_id],
                profile_hardware_args=profile_hardware_args_list[layertype_id],
                logger=logger,
            )
            res_with_grad_sync, res_without_grad_sync = obj.gen_result()
            timecosts_dict_bsz_chunked[layertype_id][string] = res_with_grad_sync
            timecosts_dict_compute[layertype_id][string] = res_without_grad_sync

    timecosts_bsz_chunked = [timecosts_dict_bsz_chunked[layertype_ids[i]][strategy_list[i].to_string()] for i in range(total_layer_num)]
    timecosts_bsz_compute = [timecosts_dict_compute[layertype_ids[i]][strategy_list[i].to_string()] for i in range(total_layer_num)]

    stage_costs_bsz_chunked = get_time_cost_all_stages(timecosts_bsz_chunked, partition)
    stage_costs_compute = get_time_cost_all_stages(timecosts_bsz_compute, partition)
    assert(len(other_time_cost) == len(stage_costs_compute))
    for i in range(len(other_time_cost)):
        stage_costs_compute[i] += other_time_cost[i] # no comm
    # print(timecosts_bsz_chunked, stage_costs_bsz_chunked, np.sum(stage_costs_bsz_chunked))
    # print(stage_costs_compute, np.max(stage_costs_compute))
    # print(np.sum(stage_costs_bsz_chunked), np.max(stage_costs_compute), np.max(stage_costs_compute) * (max_chunk-1))
    
    # # p2p & reduce sync
    # result = np.sum(stage_costs_bsz_chunked) + np.max(stage_costs_compute) * (max_chunk-1)
    
    # p2p & reduce async
    stage_costs_reduce = [total for total in stage_costs_bsz_chunked]
    # print(stage_costs_compute, stage_costs_reduce, stage_costs_bsz_chunked)
    result = np.sum(stage_costs_compute) + stage_costs_compute[-1] * (chunks - 1)
    # assume t_rank0 > t_rank1 > ... , warmup and cool down bubble can be overlapped
    result = max( result,
            max( min(pp_size - 1, chunks - 1) * stage_costs_compute[0] * 1/3, np.sum(stage_costs_compute[1:]) * 1/3) + 
            max( min(pp_size - 1, chunks - 1) * stage_costs_compute[0] * 2/3, np.sum(stage_costs_compute[1:]) * 2/3) + 
            stage_costs_compute[0] * max(0, chunks + 1 - pp_size))

    # result += max(np.max(stage_costs_compute) * 2/3 * (max_chunk - 1), stage_costs_compute[-1] * (max_chunk - 1))
    # result = np.max(stage_costs_compute) * (max_chunk-1+pp_deg)
    for i in range(pp_size):
        stage_costs_reduce[i] -= np.sum(stage_costs_compute[:i+1])
    reduce_time = np.max(stage_costs_reduce)
    reduce_time = reduce_time if reduce_time > 0 else 0
    
    # print(result,reduce_time)
    result += reduce_time
    
    if return_stage_cost:
        return stage_costs_bsz_chunked, result
    return result