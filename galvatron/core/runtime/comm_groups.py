from typing import List, Dict
import torch

class CommGroup(object):
    def __init__(self, ranks:List[int]):
        self.ranks = sorted(ranks)
        self.size = len(self.ranks)
        self.group = torch.distributed.new_group(self.ranks) if torch.distributed.is_initialized() else None

    def has_rank(self, rank):
        return rank in self.ranks

    def print(self):
        print(self.ranks, end=" ")


def show_groups(groups:List[CommGroup]):
    for group in groups:
        if group is None:
            print("None", end=" ")
        else:
            group.print()
    print()


def build_rank_to_parallel_coords(world_size, name2size, order='pp-dp-cp-tp-sp'):
    assert sorted(name2size.keys()) == sorted(['pp', 'dp', 'cp', 'tp', 'sp']) or sorted(name2size.keys()) == sorted(['pp', 'ep', 'edp', 'etp']), f'name2size keys must be pp, dp, cp, tp, sp or pp, ep, edp, etp'
    
    name_list = order.split('-')
    stride_list = [1] * len(name_list)
    for i in range(len(name_list) - 2, -1, -1):
        stride_list[i] = stride_list[i + 1] * name2size[name_list[i + 1]]

    res: Dict[int, Dict[str, int]] = {}
    for rank in range(world_size):
        info = {}
        for i, name in enumerate(name_list):
            info[name] = (rank // stride_list[i]) % name2size[name]
        res[rank] = info
    
    return res 


def get_groups(degree_rank_dict:Dict[int, Dict[str, int]], ignore_keys=[], manual_global_rank=-1) -> tuple[CommGroup, List[CommGroup]]:
    global_rank = manual_global_rank if manual_global_rank != -1 else torch.distributed.get_rank()

    same_deg_dict:Dict[str, List[int]] = {}
    for rank, info in degree_rank_dict.items():
        string_key = ''.join(f"{k}{v}" for k, v in info.items() if k not in ignore_keys)
        if string_key not in same_deg_dict:
            same_deg_dict[string_key] = []
        same_deg_dict[string_key].append(rank)

    all_groups:List[CommGroup] = []
    owner_group:CommGroup = None
    
    for ranks in same_deg_dict.values():
        group = CommGroup(ranks)
        all_groups.append(group)
        if group.has_rank(global_rank):
            owner_group = group

    return owner_group, all_groups


def get_embedding_group(pp_size, pp_group:CommGroup, manual_global_rank=-1) -> CommGroup:
    global_rank = manual_global_rank if manual_global_rank != -1 else torch.distributed.get_rank()
    embedding_ranks = [pp_group.ranks[0], pp_group.ranks[-1]] if pp_size > 1 else [pp_group.ranks[0]]
    return CommGroup(embedding_ranks) if global_rank in embedding_ranks else None


# TODO: Check correctness
def merge_redistributed_group(split_tp_sp_cp_group:CommGroup, allgather_tp_sp_cp_group:CommGroup):
    assert split_tp_sp_cp_group is not None and allgather_tp_sp_cp_group is not None, "split_tp_sp_cp_group and allgather_tp_sp_cp_group must not be None"

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    split_tp_sp_cp_size = split_tp_sp_cp_group.size
    allgather_tp_sp_cp_size = allgather_tp_sp_cp_group.size

    if split_tp_sp_cp_size > allgather_tp_sp_cp_size:
        num_tp_sp_cp_groups = world_size // split_tp_sp_cp_size
        # mul = split_tp_sp_cp_size // allgather_tp_sp_cp_size
        for i in range(num_tp_sp_cp_groups):
            for j in range(allgather_tp_sp_cp_size):
                ranks = range(i * split_tp_sp_cp_size + j, (i + 1) * split_tp_sp_cp_size + j, allgather_tp_sp_cp_size)
                group = CommGroup(ranks)
                if group.has_rank(rank):
                    fused_group = group
        return fused_group, None
    elif split_tp_sp_cp_size < allgather_tp_sp_cp_size:
        num_tp_sp_cp_groups = world_size // allgather_tp_sp_cp_size
        # mul = allgather_tp_sp_cp_size // split_tp_sp_cp_size
        for i in range(num_tp_sp_cp_groups):
            for j in range(split_tp_sp_cp_size):
                ranks = range(i * allgather_tp_sp_cp_size + j, (i + 1) * allgather_tp_sp_cp_size + j, split_tp_sp_cp_size)
                group = CommGroup(ranks)
                if group.has_rank(rank):
                    fused_group = group
        return None, fused_group
    elif split_tp_sp_cp_size == allgather_tp_sp_cp_size:
        return None, None
    else:
        assert False, "merge_redistributed_group error!"


def gen_comm_groups(
    all_tp_sizes:List[int], 
    all_sp_sizes:List[int], 
    all_cp_sizes:List[int], 
    all_ep_sizes:List[int], 
    all_tp_of_ep_sizes:List[int], 
    pp_size:int,
    is_moe_model:bool=False, 
    show_rank=-1, 
):
    # [Step 1] Input Check and Some Preparations
    assert all(not (tp > 1 and sp > 1) for tp, sp in zip(all_tp_sizes, all_sp_sizes)), "DeepSpeed Ulysses is not compatible with Megatron Tensor Parallel!"

    world_size = torch.distributed.get_world_size()
    total_num = len(all_tp_sizes)

    # [Step 2] build rank to parallel coords
    pp_group:CommGroup = None
    embedding_group:CommGroup = None
    tp_groups:List[CommGroup] = []
    sp_groups:List[CommGroup] = []
    cp_groups:List[CommGroup] = []
    dp_groups:List[CommGroup] = []
    sdp_groups:List[CommGroup] = []
    tsp_cp_groups:List[CommGroup] = []

    for i in range(total_num):
        dp_size = world_size // pp_size // all_tp_sizes[i] // all_sp_sizes[i] // all_cp_sizes[i]
        name2size = {
            'pp': pp_size,
            'dp': dp_size,
            'cp': all_cp_sizes[i],
            'tp': all_tp_sizes[i],
            'sp': all_sp_sizes[i],
        }
        degree_rank_dict = build_rank_to_parallel_coords(world_size, name2size, order='pp-dp-cp-tp-sp')
        
        if i == 0:
            pp_group, _ = get_groups(degree_rank_dict, ignore_keys=['pp'])
            embedding_group = get_embedding_group(pp_size, pp_group)

        tp_group, _ = get_groups(degree_rank_dict, ignore_keys=['tp'])
        sp_group, _ = get_groups(degree_rank_dict, ignore_keys=['sp'])
        sdp_group, _ = get_groups(degree_rank_dict, ignore_keys=['dp', 'sp'])
        cp_group, _ = get_groups(degree_rank_dict, ignore_keys=['cp'])
        dp_group, _ = get_groups(degree_rank_dict, ignore_keys=['dp'])
        tsp_cp_group, _ = get_groups(degree_rank_dict, ignore_keys=['tp', 'sp', 'cp'])

        tp_groups.append(tp_group)
        sp_groups.append(sp_group)
        cp_groups.append(cp_group)
        dp_groups.append(dp_group)
        sdp_groups.append(sdp_group)
        tsp_cp_groups.append(tsp_cp_group)
        
    # [Step 3] build rank to parallel coords for moe layer
    if is_moe_model:
        ep_groups:List[CommGroup] = []
        tp_of_ep_groups:List[CommGroup] = []
        tp_and_ep_groups:List[CommGroup] = []
        dp_of_ep_groups:List[CommGroup] = []

        for i in range(total_num):
            edp_size = world_size // pp_size // all_ep_sizes[i] // all_tp_of_ep_sizes[i]
            name2size = {
                'pp': pp_size,
                'ep': all_ep_sizes[i],
                'edp': edp_size,
                'etp': all_tp_of_ep_sizes[i],
            }
            degree_rank_dict = build_rank_to_parallel_coords(world_size, name2size, order='pp-ep-edp-etp')
            ep_group, _ = get_groups(degree_rank_dict, ignore_keys=['ep'])
            tp_of_ep_group, _ = get_groups(degree_rank_dict, ignore_keys=['etp'])
            tp_and_ep_group, _ = get_groups(degree_rank_dict, ignore_keys=['ep', 'etp'])
            dp_of_ep_group, _ = get_groups(degree_rank_dict, ignore_keys=['edp'])
            ep_groups.append(ep_group)
            tp_of_ep_groups.append(tp_of_ep_group)
            tp_and_ep_groups.append(tp_and_ep_group)
            dp_of_ep_groups.append(dp_of_ep_group)
    else:
        ep_groups, tp_of_ep_groups, tp_and_ep_groups, dp_of_ep_groups = None, None, None, None

    # [Step 4] build redistribution communication groups
    allgather_cp_groups, split_cp_groups = [None], [None]
    allgather_tp_sp_cp_groups, split_tp_sp_cp_groups = [None], [None]
    fused_split_groups, fused_allgather_groups = [None], [None]

    for i in range(1, total_num):
        former_tsp_size = all_sp_sizes[i - 1] if all_sp_sizes[i - 1] > 1 else all_tp_sizes[i - 1]
        former_cp_size = all_cp_sizes[i - 1]
        latter_tsp_size = all_sp_sizes[i] if all_sp_sizes[i] > 1 else all_tp_sizes[i]
        latter_cp_size = all_cp_sizes[i]
        
        if former_tsp_size == latter_tsp_size and former_cp_size == latter_cp_size:
            split_cp_group = None
            allgather_cp_group = None
            split_tp_sp_cp_group = None
            allgather_tp_sp_cp_group = None
            fused_split_group = None
            fused_allgather_group = None
        else:
            split_cp_group = None if former_cp_size == 1 else cp_groups[i - 1]
            allgather_cp_group = None if latter_cp_size == 1 else cp_groups[i]
            split_tp_sp_cp_group = tsp_cp_groups[i - 1]
            allgather_tp_sp_cp_group = tsp_cp_groups[i]
            fused_split_group, fused_allgather_group = merge_redistributed_group(split_tp_sp_cp_group, allgather_tp_sp_cp_group)

        allgather_cp_groups.append(allgather_cp_group)
        split_cp_groups.append(split_cp_group)
        allgather_tp_sp_cp_groups.append(allgather_tp_sp_cp_group)
        split_tp_sp_cp_groups.append(split_tp_sp_cp_group)
        fused_split_groups.append(fused_split_group)
        fused_allgather_groups.append(fused_allgather_group)

    # [Step 5] Show Communication Groups
    show_rank = 0
    if show_rank >= 0 and torch.distributed.get_rank() == show_rank:
        print("====================== Galvatron Communication Group ===========================")
        print("Embedding group for rank %d:" % show_rank)
        show_groups([embedding_group])
        print("TP groups for rank %d (all layers):" % show_rank)
        show_groups(tp_groups)
        print("SP groups for rank %d (all layers):" % show_rank)
        show_groups(sp_groups)
        print("CP groups for rank %d (all layers):" % show_rank)
        show_groups(cp_groups)
        print("DP groups for rank %d (all layers):" % show_rank)
        show_groups(dp_groups)
        print("SDP groups for rank %d (all layers):" % show_rank)
        show_groups(sdp_groups)
        print("Split CP groups for rank %d:" % show_rank)
        show_groups(split_cp_groups)
        print("AllGather CP groups for rank %d:" % show_rank)
        show_groups(allgather_cp_groups)
        print("Split TP/SP/CP groups for rank %d:" % show_rank)
        show_groups(split_tp_sp_cp_groups)
        print("AllGather TP/SP/CP groups for rank %d:" % show_rank)
        show_groups(allgather_tp_sp_cp_groups)
        if is_moe_model:
            print("EP groups for rank %d (all layers)" % show_rank)
            show_groups(ep_groups)
            print("TP of EP groups for rank %d (all layers)" % show_rank)
            show_groups(tp_of_ep_groups)
            print("TP and EP groups for rank %d (all layers)" % show_rank)
            show_groups(tp_and_ep_groups)
            print("DP of EP groups for rank %d (all layers)" % show_rank)
            show_groups(dp_of_ep_groups)
        print("Fused split groups for rank %d:" % show_rank)
        show_groups(fused_split_groups)
        print("Fused allgather groups for rank %d:" % show_rank)
        show_groups(fused_allgather_groups)
        print("================================================================================")

    return (
        pp_group,
        tp_groups,
        sp_groups,
        cp_groups,
        dp_groups,
        sdp_groups,
        ep_groups,
        tp_of_ep_groups,
        tp_and_ep_groups,
        dp_of_ep_groups,
        allgather_cp_groups,
        split_cp_groups,
        allgather_tp_sp_cp_groups,
        split_tp_sp_cp_groups,
        fused_allgather_groups,
        fused_split_groups,
        embedding_group,
    )
