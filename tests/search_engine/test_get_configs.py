import pytest
from tests.utils.search_configs import (
    write_time_config,
    write_memory_config,
    write_hardware_config
)
from galvatron.core.search_engine.args_schema import GalvatronSearchArgs
from tests.utils.model_utils_new import ModelFactory
from galvatron.core.search_engine.search_engine import GalvatronSearchEngine
from galvatron.utils.hf_config_adapter import model_layer_configs, model_name


def _build_hf_test_args(config_json, time_mode):
    model_ns = SimpleNamespace(
        model_size=config_json.get("model_size", "llama2-7b"),
        hf_model_name_or_path=config_json.get("hf_model_name_or_path"),
        hidden_size=config_json.get("hidden_size"),
        num_layers=config_json.get("num_hidden_layers", config_json.get("num_layers")),
        num_attention_heads=config_json.get("num_attention_heads"),
        ffn_hidden_size=config_json.get("intermediate_size", config_json.get("ffn_hidden_size")),
        vocab_size=config_json.get("vocab_size"),
    )
    train_ns = SimpleNamespace(seq_length=config_json.get("seq_length", 4096))
    profile_ns = SimpleNamespace(profile_mode=time_mode)
    return SimpleNamespace(model=model_ns, train=train_ns, profile=profile_ns)


def _promote_profile_filenames_to_all(configs_dir: Path, precision: str, model: str):
    time_src = configs_dir / f"computation_profiling_{precision}_{model}.json"
    time_dst = configs_dir / f"computation_profiling_{precision}_{model}_all.json"
    mem_src = configs_dir / f"memory_profiling_{precision}_{model}.json"
    mem_dst = configs_dir / f"memory_profiling_{precision}_{model}_all.json"
    shutil.copyfile(time_src, time_dst)
    shutil.copyfile(mem_src, mem_dst)

# ============= Model Config Tests =============
@pytest.mark.search_engine
@pytest.mark.parametrize("model_type", ["gpt"])
@pytest.mark.parametrize("backend", ["hf"])
@pytest.mark.parametrize("time_mode,memory_mode,sp_enabled", [
    ("static", "static", False),
    ("batch", "static", False),
    ("sequence", "static", False),
    ("static", "static", True),
    ("batch", "static", True),
    ("sequence", "static", True),
    ("static", "sequence", True),
    ("batch", "sequence", True),
    ("sequence", "sequence", True),
])
def test_config_loading(base_config_dirs, model_type, backend, time_mode, memory_mode, sp_enabled):
    """Test loading both time and memory configs with different modes"""
    _, configs_dir, _ = base_config_dirs

    # Setup search engine
    args = GalvatronSearchArgs()
    # model_layer_configs, model_name = ModelFactory.get_meta_configs(model_type, backend)
    # config_json = ConfigFactory.get_config_json(model_type)
    # args.model_info.model_size = config_json
    # config = ModelFactory.create_config(model_type, backend, args)
    

    args.profiling_info.time_profiling_path = str(configs_dir)
    args.profiling_info.memory_profiling_path = str(configs_dir)
    args.profiling_info.time_profile_mode = time_mode
    args.profiling_info.memory_profile_mode = memory_mode
    args.common_train_info.sequence_parallel = sp_enabled

    ModelFactory.resolve_model_config(args, model_type, backend)
    model_layer_configs_func = ModelFactory.get_model_layer_configs_func()
    model_name_func = ModelFactory.get_model_name_func()
    
    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(str(configs_dir.parent), model_layer_configs_func(args), model_name_func(args))
    if model_type == "gpt":
        search_engine.seqlen_list = [4096]

    # Write both config files
    write_time_config(configs_dir, profile_mode=time_mode, model_name=model_name_func(args))
    write_memory_config(configs_dir, profile_mode=memory_mode, sp_mode=sp_enabled, model_name=model_name_func(args))
    
    # Get configs and verify
    time_config, memory_config = search_engine.get_profiled_model_configs()
    
    # Verify time configs
    if time_mode == "static":
        assert "layertype_0_bsz8_seq4096" in time_config
        assert abs(time_config["layertype_0_bsz8_seq4096"] - 11.219752883911134) < 1e-6
    elif time_mode == "batch":
        assert "layertype_0_bsz4_seq4096" in time_config
        assert abs(time_config["layertype_0_bsz4_seq4096"] - 11.152996063232425) < 1e-6
    else:  # sequence
        assert "layertype_0_bsz1_seq32768" in time_config
        assert abs(time_config["layertype_0_bsz1_seq32768"] - 123.1998901367187) < 1e-6
    
    # Verify memory configs
    key_prefix = "layertype_0_sp" if sp_enabled else "layertype_0"
    assert key_prefix in memory_config
    
    if memory_mode == "sequence":
        assert 512 in memory_config[key_prefix]
        assert 2048 in memory_config[key_prefix]
    else:
        assert 4096 in memory_config[key_prefix]
    
    if sp_enabled:
        if memory_mode == "static":
            assert "tp_activation_per_bsz_dict" in memory_config[key_prefix][4096]
            assert abs(memory_config[key_prefix][4096]["tp_activation_per_bsz_dict"][8] - 79.5704345703125) < 1e-6
        else:
            assert "tp_activation_per_bsz_dict" in memory_config[key_prefix][4096]
            assert abs(memory_config[key_prefix][4096]["tp_activation_per_bsz_dict"][8] - 130.5587158203125) < 1e-6
    else:
        assert "tp_activation_per_bsz_dict" in memory_config[key_prefix][4096]
        assert abs(memory_config[key_prefix][4096]["tp_activation_per_bsz_dict"][8] - 191.6251220703125) < 1e-6

# ============= Hardware Config Tests =============
@pytest.mark.search_engine
@pytest.mark.parametrize("num_nodes,gpus_per_node", [
    (1, 8),
])
def test_hardware_config_loading(base_config_dirs, num_nodes, gpus_per_node):
    """Test loading hardware configs with different cluster configurations"""
    _, hardware_dir, _ = base_config_dirs
    write_hardware_config(hardware_dir, num_nodes=num_nodes, gpus_per_node=gpus_per_node)
    
    args = GalvatronSearchArgs()
    args.hardware_info.num_nodes = num_nodes
    args.hardware_info.num_gpus_per_node = gpus_per_node
    args.profiling_info.allreduce_bandwidth_config_path = str(hardware_dir)
    args.profiling_info.p2p_bandwidth_config_path = str(hardware_dir)
    args.profiling_info.overlap_coe_path = str(hardware_dir)
    args.profiling_info.sp_time_path = str(hardware_dir)
    engine = GalvatronSearchEngine(args)
    engine.set_path(str(hardware_dir.parent))
    allreduce_bandwidth, p2p_bandwidth, overlap_coe, sp_allreduce, sp_all2all = engine.get_profiled_hardware_configs()
    
    assert abs(allreduce_bandwidth['2_0'] - 153.933) < 1e-3
    assert abs(allreduce_bandwidth['4_1'] - 164.272) < 1e-3
    assert abs(p2p_bandwidth[2] - 147.32) < 1e-3
    assert abs(overlap_coe - 1.1534195950157762) < 1e-6
    assert abs(sp_allreduce[8][8*1024*1024] - 0.1827 / 2) < 1e-4
    assert abs(sp_allreduce[8][16*1024*1024] - 0.29410000000000003 / 2) < 1e-4
    assert abs(sp_all2all[4][8*1024*1024] - 0.1255) < 1e-4
    assert abs(sp_all2all[4][16*1024*1024] - 0.1514) < 1e-4