import pytest
import numpy as np
import copy
from galvatron.core.search_engine.search_engine import pp_division_memory_balanced, get_pp_stage_for_bsz, check_optimal_chunks, optimal_chunk_func_default
from tests.utils.cost_args import MemoryModelArgs, TimeModelArgs, create_model_args_from_dict
from galvatron.utils.strategy_utils import LayerStrategy, DPType

@pytest.fixture
def memory_model_args():
    """Create memory model args"""
    return MemoryModelArgs.from_mock_config()

@pytest.fixture
def time_model_args():
    """Create time model args"""
    return TimeModelArgs.from_mock_config()

@pytest.mark.search_engine
def test_pp_division_memory_balanced(memory_model_args):
    """Test pipeline division based on memory balance"""
    # Prepare test data
    memory_args_dicts = [copy.deepcopy(memory_model_args.to_dict()) for _ in range(2)]
    
    # Convert config dictionaries to list of five parameter objects
    model_args_list = []
    train_args_list = []
    parallel_args_list = []
    profile_model_args_list = []
    profile_hardware_args_list = []
    for args_dict in memory_args_dicts:
        model_args, train_args, parallel_args, profile_model_args, profile_hardware_args = create_model_args_from_dict(args_dict)
        # Combine five parameter objects into a tuple and add to list
        model_args_list.append(model_args)
        train_args_list.append(train_args)
        parallel_args_list.append(parallel_args)
        profile_model_args_list.append(profile_model_args)
        profile_hardware_args_list.append(profile_hardware_args)
    
    layer_num = [16, 16]
    pp_deg = 4
    bsz = 32
    mbsz = 8
    strategies = [
        LayerStrategy(pp_size=4, tp_size=1, sp_size=1, dp_size=8, dp_type=DPType.ZERO2, checkpoint=False),
        LayerStrategy(pp_size=4, tp_size=2, sp_size=1, dp_size=4, dp_type=DPType.ZERO2, checkpoint=False),
        LayerStrategy(pp_size=4, tp_size=4, sp_size=1, dp_size=2, dp_type=DPType.ZERO2, checkpoint=False),
    ]

    pp_divide, mem_costs = pp_division_memory_balanced(
        model_args_list,
        train_args_list,
        parallel_args_list,
        profile_model_args_list,
        layer_num,
        pp_deg,
        bsz,
        mbsz,
        strategies
    )

    # Validate results
    assert pp_divide is not None
    assert len(pp_divide) == pp_deg
    assert sum(pp_divide) == sum(layer_num)
    assert all(count > 0 for count in pp_divide)
    
    if mem_costs is not None:
        max_mem = max(mem_costs)
        min_mem = min(mem_costs)
        imbalance = (max_mem - min_mem) / max_mem
        print(f"PP divide: {pp_divide}")
        print(f"Memory costs per stage: {mem_costs}")
        print(f"Memory imbalance: {imbalance:.2%}")
        assert imbalance < 0.3

@pytest.mark.search_engine
@pytest.mark.parametrize("single_layer_even", [True, False])
def test_get_pp_stage_for_bsz(memory_model_args, single_layer_even):
    """Test getting pipeline stages for different batch sizes"""
    memory_args_dicts = [copy.deepcopy(memory_model_args.to_dict()) for _ in range(2)]
    
    # Convert config dictionaries to list of five parameter objects
    model_args_list = []
    train_args_list = []
    parallel_args_list = []
    profile_model_args_list = []
    profile_hardware_args_list = []
    for args_dict in memory_args_dicts:
        model_args, train_args, parallel_args, profile_model_args, profile_hardware_args = create_model_args_from_dict(args_dict)
        # Combine five parameter objects into a tuple and add to list
        model_args_list.append(model_args)
        train_args_list.append(train_args)
        parallel_args_list.append(parallel_args)
        profile_model_args_list.append(profile_model_args)
        profile_hardware_args_list.append(profile_hardware_args)
    
    layer_num_list = [16, 16]
    bsz = 32
    mbsz_dict = {1: 8, 2: 8, 4: 8}
    strategies = [
        LayerStrategy(pp_size=4, tp_size=1, sp_size=1, dp_size=8, dp_type=DPType.ZERO2, checkpoint=False),
        LayerStrategy(pp_size=4, tp_size=2, sp_size=1, dp_size=4, dp_type=DPType.ZERO2, checkpoint=False),
        LayerStrategy(pp_size=4, tp_size=4, sp_size=1, dp_size=2, dp_type=DPType.ZERO2, checkpoint=False),
    ]

    pp_stage_dict = get_pp_stage_for_bsz(
        strategies,
        model_args_list,
        train_args_list,
        parallel_args_list,
        profile_model_args_list,
        layer_num_list,
        bsz,
        mbsz_dict,
        single_layer_even
    )

    assert isinstance(pp_stage_dict, dict)
    for pp_deg in [4]:
        assert pp_deg in pp_stage_dict
        stages = pp_stage_dict[pp_deg]
        assert sum(stages) == sum(layer_num_list)
        print(f"PP={pp_deg} stage division: {stages}")
