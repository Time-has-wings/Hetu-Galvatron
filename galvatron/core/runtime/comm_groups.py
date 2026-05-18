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
    """
    Map every global rank to its multi-dimensional parallel coordinate, i.e.
    "which slice of each parallel dim does this rank belong to".

    Args:
        world_size:  total number of ranks. Must equal the product of all
                     dim sizes in `name2size`.
        name2size:   {dim_name: degree}. Currently supports either the
                     {pp, dp, cp, tp, sp} family or the MoE {pp, ep, edp, etp}
                     family.
        order:       hyphen-separated dim names, **left = slowest-varying,
                     right = fastest-varying** in the flat rank index.
                     Example: order='pp-dp-cp-tp-sp' means that as `rank`
                     grows by 1, `sp` changes first, then `tp` once `sp`
                     wraps, ..., and `pp` is the outermost block.

    Returns:
        dict[int, dict[str, int]] — for each rank, a dict of its coord on
        every named dim.

    Strides:
        For order='pp-dp-cp-tp-sp' with sizes (pp, dp, cp, tp, sp),
            stride(sp) = 1
            stride(tp) = sp
            stride(cp) = sp * tp
            stride(dp) = sp * tp * cp
            stride(pp) = sp * tp * cp * dp
        and  coord_of_dim(rank) = (rank // stride(dim)) % size(dim).

    --------------------------------------------------------------------------
    Example — world_size=16, pp=2 / dp=2 / cp=2 / tp=2 / sp=1,
    order='pp-dp-cp-tp-sp' → strides: sp=1, tp=1, cp=2, dp=4, pp=8.

        >>> coords = build_rank_to_parallel_coords(
        ...     16, {'pp':2, 'dp':2, 'cp':2, 'tp':2, 'sp':1},
        ...     order='pp-dp-cp-tp-sp')
        >>> coords[0]
        {'pp': 0, 'dp': 0, 'cp': 0, 'tp': 0, 'sp': 0}
        >>> coords[5]    # 5 = 0*8 + 1*4 + 0*2 + 1*1 + 0
        {'pp': 0, 'dp': 1, 'cp': 0, 'tp': 1, 'sp': 0}
        >>> coords[11]   # 11 = 1*8 + 0*4 + 1*2 + 1*1 + 0
        {'pp': 1, 'dp': 0, 'cp': 1, 'tp': 1, 'sp': 0}

    Full layout (note pp splits the 16 ranks into the two halves 0-7 / 8-15):

        rank | pp dp cp tp sp     rank | pp dp cp tp sp
        -----+----------------    -----+----------------
          0  |  0  0  0  0  0       8  |  1  0  0  0  0
          1  |  0  0  0  1  0       9  |  1  0  0  1  0
          2  |  0  0  1  0  0      10  |  1  0  1  0  0
          3  |  0  0  1  1  0      11  |  1  0  1  1  0
          4  |  0  1  0  0  0      12  |  1  1  0  0  0
          5  |  0  1  0  1  0      13  |  1  1  0  1  0
          6  |  0  1  1  0  0      14  |  1  1  1  0  0
          7  |  0  1  1  1  0      15  |  1  1  1  1  0

    The `order` string also dictates **rank adjacency**: ranks that are
    neighbours along the fastest dim differ by 1 in rank id (here that's TP),
    which is normally what you want for the most bandwidth-hungry collective.
    Changing `order` reshuffles which ranks sit on the same NVLink island
    without changing the per-dim degrees.
    """
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
    """
    Group ranks that share the same parallel coordinates **after dropping the
    dimensions listed in `ignore_keys`**. The intuition: a "TP group" is the
    set of ranks that differ only along the TP axis (so we ignore the 'tp'
    coordinate when forming the grouping key); same for DP / PP / CP / SP.

    Args:
        degree_rank_dict: rank -> {dim_name: coord_along_that_dim},
                          typically produced by `build_rank_to_parallel_coords`.
        ignore_keys:      dimension names whose coordinate should NOT be part
                          of the grouping key. Ranks that match on every OTHER
                          coordinate land in the same group.
        manual_global_rank: override `torch.distributed.get_rank()` for tests.

    Returns:
        (owner_group, all_groups):
            owner_group — the CommGroup that contains this process's rank
                          (None if the rank is not in any group, which should
                          not happen for a well-formed layout).
            all_groups  — every CommGroup produced from the layout. The full
                          list is returned because `new_group()` is collective
                          and must be called in the same order on all ranks.

    --------------------------------------------------------------------------
    Example — world_size=16, pp=2 / dp=2 / cp=2 / tp=2 / sp=1
    with order='pp-dp-cp-tp-sp'. Strides become sp=1, tp=1, cp=2, dp=4, pp=8,
    so coords for each rank are:

        rank | pp dp cp tp sp     rank | pp dp cp tp sp
        -----+----------------    -----+----------------
          0  |  0  0  0  0  0       8  |  1  0  0  0  0
          1  |  0  0  0  1  0       9  |  1  0  0  1  0
          2  |  0  0  1  0  0      10  |  1  0  1  0  0
          3  |  0  0  1  1  0      11  |  1  0  1  1  0
          4  |  0  1  0  0  0      12  |  1  1  0  0  0
          5  |  0  1  0  1  0      13  |  1  1  0  1  0
          6  |  0  1  1  0  0      14  |  1  1  1  0  0
          7  |  0  1  1  1  0      15  |  1  1  1  1  0

    TP groups   (ignore 'tp'): adjacent pairs along the TP axis (stride 1)
        -> [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]]
    CP groups   (ignore 'cp'): stride 2
        -> [[0,2], [1,3], [4,6], [5,7], [8,10], [9,11], [12,14], [13,15]]
    DP groups   (ignore 'dp'): stride 4
        -> [[0,4], [1,5], [2,6], [3,7], [8,12], [9,13], [10,14], [11,15]]
    PP groups   (ignore 'pp'): stride 8
        -> [[0,8], [1,9], [2,10], [3,11], [4,12], [5,13], [6,14], [7,15]]

    Fused TP+SP+CP groups (ignore 'tp','sp','cp'): share (pp,dp), size 4
        -> [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]
    Fused DP+SP+CP groups, i.e. SDP (ignore 'dp','sp','cp'): share (pp,tp)
        -> [[0,2,4,6], [1,3,5,7], [8,10,12,14], [9,11,13,15]]

    Edge case — `ignore_keys=[]`: every rank gets a unique key, so each group
    has size 1 (i.e. all_groups = [[0], [1], ..., [15]]).
    """
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


def get_embedding_group(pp_size, pp_groups:List[CommGroup], manual_global_rank=-1) -> tuple[CommGroup, List[CommGroup]]:
    global_rank = manual_global_rank if manual_global_rank != -1 else torch.distributed.get_rank()

    all_groups:List[CommGroup] = []
    owner_group:CommGroup = None

    for pp_group in pp_groups:
        embedding_ranks = [pp_group.ranks[0], pp_group.ranks[-1]] if pp_size > 1 else [pp_group.ranks[0]]
        group = CommGroup(embedding_ranks)
        all_groups.append(group)
        if group.has_rank(global_rank):
            owner_group = group

    return owner_group, all_groups


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
    manual_world_size=-1,
):
    # [Step 1] Input Check and Some Preparations
    assert all(not (tp > 1 and sp > 1) for tp, sp in zip(all_tp_sizes, all_sp_sizes)), "DeepSpeed Ulysses is not compatible with Megatron Tensor Parallel!"

    world_size = torch.distributed.get_world_size() if manual_world_size == -1 else manual_world_size
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

    # Collect unique (tp, sp, cp) configs preserving first-seen order (deterministic across ranks).
    unique_keys: List[tuple] = []
    seen = set()
    for i in range(total_num):
        key = (all_tp_sizes[i], all_sp_sizes[i], all_cp_sizes[i])
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)

    # Build comm groups once per unique config.
    config_to_groups: Dict[tuple, Dict[str, CommGroup]] = {}
    for idx, key in enumerate(unique_keys):
        tp_size, sp_size, cp_size = key
        dp_size = world_size // pp_size // tp_size // sp_size // cp_size
        name2size = {
            'pp': pp_size,
            'dp': dp_size,
            'cp': cp_size,
            'tp': tp_size,
            'sp': sp_size,
        }
        degree_rank_dict = build_rank_to_parallel_coords(world_size, name2size, order='pp-dp-cp-tp-sp')

        if idx == 0:
            pp_group, pp_groups = get_groups(degree_rank_dict, ignore_keys=['pp'])
            # embedding_group = get_embedding_group(pp_size, pp_group)
            embedding_group, _ = get_embedding_group(pp_size, pp_groups)

        groups: Dict[str, CommGroup] = {}
        groups['tp'], _ = get_groups(degree_rank_dict, ignore_keys=['tp'])
        groups['sp'], _ = get_groups(degree_rank_dict, ignore_keys=['sp'])
        groups['sdp'], _ = get_groups(degree_rank_dict, ignore_keys=['dp', 'sp', 'cp'])
        groups['cp'], _ = get_groups(degree_rank_dict, ignore_keys=['cp'])
        groups['dp'], _ = get_groups(degree_rank_dict, ignore_keys=['dp'])
        groups['tsp_cp'], _ = get_groups(degree_rank_dict, ignore_keys=['tp', 'sp', 'cp'])
        config_to_groups[key] = groups

    # Populate per-layer lists by lookup.
    for i in range(total_num):
        key = (all_tp_sizes[i], all_sp_sizes[i], all_cp_sizes[i])
        groups = config_to_groups[key]
        tp_groups.append(groups['tp'])
        sp_groups.append(groups['sp'])
        cp_groups.append(groups['cp'])
        dp_groups.append(groups['dp'])
        sdp_groups.append(groups['sdp'])
        tsp_cp_groups.append(groups['tsp_cp'])
        
    # [Step 3] build rank to parallel coords for moe layer
    if is_moe_model:
        ep_groups:List[CommGroup] = []
        tp_of_ep_groups:List[CommGroup] = []
        tp_and_ep_groups:List[CommGroup] = []
        dp_of_ep_groups:List[CommGroup] = []

        moe_unique_keys: List[tuple] = []
        moe_seen = set()
        for i in range(total_num):
            key = (all_ep_sizes[i], all_tp_of_ep_sizes[i])
            if key not in moe_seen:
                moe_seen.add(key)
                moe_unique_keys.append(key)

        moe_config_to_groups: Dict[tuple, Dict[str, CommGroup]] = {}
        for key in moe_unique_keys:
            ep_size, etp_size = key
            edp_size = world_size // pp_size // ep_size // etp_size
            name2size = {
                'pp': pp_size,
                'ep': ep_size,
                'edp': edp_size,
                'etp': etp_size,
            }
            degree_rank_dict = build_rank_to_parallel_coords(world_size, name2size, order='pp-ep-edp-etp')
            groups: Dict[str, CommGroup] = {}
            groups['ep'], _ = get_groups(degree_rank_dict, ignore_keys=['ep'])
            groups['tp_of_ep'], _ = get_groups(degree_rank_dict, ignore_keys=['etp'])
            groups['tp_and_ep'], _ = get_groups(degree_rank_dict, ignore_keys=['ep', 'etp'])
            groups['dp_of_ep'], _ = get_groups(degree_rank_dict, ignore_keys=['edp'])
            moe_config_to_groups[key] = groups

        for i in range(total_num):
            key = (all_ep_sizes[i], all_tp_of_ep_sizes[i])
            groups = moe_config_to_groups[key]
            ep_groups.append(groups['ep'])
            tp_of_ep_groups.append(groups['tp_of_ep'])
            tp_and_ep_groups.append(groups['tp_and_ep'])
            dp_of_ep_groups.append(groups['dp_of_ep'])
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


if __name__ == "__main__":
    # Standalone test: no real torch.distributed init.
    # CommGroup skips new_group() because is_initialized() is False;
    # we monkey-patch get_rank / get_world_size so the helpers that
    # read them work for the simulated rank.
    import torch.distributed as dist

    world_size = 16
    pp_size = 2
    is_moe_model = False

    # 4 layers, varied (tp, sp, cp) to exercise dedup + redistribution paths.
    all_tp_sizes       = [2, 2, 1, 1]
    all_sp_sizes       = [1, 1, 1, 1]
    all_cp_sizes       = [1, 1, 2, 2]
    all_ep_sizes       = [1, 1, 1, 1]
    all_tp_of_ep_sizes = [1, 1, 1, 1]

    dist.get_world_size = lambda: world_size

    # Iterate over every rank to print each rank's owner groups.
    # Set show_ranks_to_print to a subset (e.g. [0]) if output is too verbose.
    show_ranks_to_print = list(range(world_size))

    for test_rank in show_ranks_to_print:
        dist.get_rank = lambda r=test_rank: r
        print(f"\n########## simulated rank {test_rank} ##########")
        gen_comm_groups(
            all_tp_sizes,
            all_sp_sizes,
            all_cp_sizes,
            all_ep_sizes,
            all_tp_of_ep_sizes,
            pp_size=pp_size,
            is_moe_model=is_moe_model,
            show_rank=test_rank,
            manual_world_size=world_size,
        )
