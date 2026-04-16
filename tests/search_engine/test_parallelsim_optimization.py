import pytest
import os
import glob
import json
from tests.utils.search_configs import (
    initialize_search_engine
)
from galvatron.utils.strategy_utils import config2strategy

@pytest.mark.search_engine
@pytest.mark.parametrize("idx, model_type,backend,time_mode,memory_mode,sp_enabled,settle_bsz, settle_chunk, memory_constraint, seqlen_list, fine_grained_mode", [
    (0, "llama_search", "hf", "sequence", "sequence", True, 64, 32, 36, [8192], 1),
    (1, "llama_search", "hf", "sequence", "sequence", True, 64, 8, 36, [8192], 0),
])
def test_basic_search_flow(base_config_dirs, base_log_dirs, idx, model_type, backend, time_mode, memory_mode, sp_enabled, settle_bsz, settle_chunk, memory_constraint, seqlen_list, fine_grained_mode):
    
    kwargs = {
        "settle_bsz": settle_bsz,
        "settle_chunk": settle_chunk,
        "memory_constraint": memory_constraint,
        "default_dp_type": "zero2",
        "pipeline_type": "pipedream_flush",
        "async_grad_reduce": False,
        "sequence_parallel": True,
        "fine_grained_mode": fine_grained_mode,
        'num_layers': 28,
    }

    search_engine = initialize_search_engine(base_config_dirs, base_log_dirs, model_type, backend, time_mode, memory_mode, sp_enabled, seqlen_list, **kwargs)
    

    
    throughput = search_engine.parallelism_optimization()

    if idx == 0:
        assert abs(throughput - 2.6485091403918064) < 1e-6, f'idx: {idx}, throughput: {throughput}'

        output_dir = base_config_dirs[2]
        json_files = glob.glob(os.path.join(output_dir, '*.json'))
        assert len(json_files) == 1, f"Expected exactly one JSON file, found {len(json_files)}"
        output_file = json_files[0]
        
        filename = os.path.basename(output_file)
        assert filename.startswith('galvatron_config_')
        assert filename.endswith('.json')

        with open(output_file, 'r') as f:
            config = json.load(f)

        expected_fields = [
                "pp_deg", "tp_sizes_enc", "tp_consecutive_flags", 
                "dp_types_enc", "use_sp", "checkpoint", "global_bsz",
                "chunks", "pp_division", "pipeline_type", 
                "default_dp_type", "vtp", "vsp"
            ]
        for field in expected_fields:
            assert field in config, f"Missing field: {field}"

        assert config["pp_deg"] == 1
        assert config["global_bsz"] == 64
        assert config["chunks"] == 32
        assert config["pp_division"] == "28", f'idx: {idx}, pp_division: {config["pp_division"]}'
        assert config["pipeline_type"] == "pipedream_flush"
        assert config["default_dp_type"] == "zero2"
        assert config["vtp"] == 8
        assert config["vsp"] == 0
        assert config["embed_sdp"] == 0

        layer_strategy_list = config2strategy(config, default_dp_type="zero2")
        string_list = [strategy.to_simple_string() for strategy in layer_strategy_list]
        string_list = ', '.join(string_list)
        assert string_list == "1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f-c, 1-4*-2f, 1-4*-2f, 1-4*-2f, 1-4*-2f, 1-4*-2f, 1-4*-2f, 1-4*-2f, 1-4*-2f, 1-4*-2f, 1-4*-2f, 1-4*-2f, 1-4*-2f, 1-4*-2, 1-4*-2"
    else:
        assert abs(throughput - 2.5246283459057333) < 1e-6, f'idx: {idx}, throughput: {throughput}'

        output_dir = base_config_dirs[2]
        json_files = glob.glob(os.path.join(output_dir, '*.json'))
        assert len(json_files) == 1, f"Expected exactly one JSON file, found {len(json_files)}"
        output_file = json_files[0]
        
        filename = os.path.basename(output_file)
        assert filename.startswith('galvatron_config_')
        assert filename.endswith('.json')

        with open(output_file, 'r') as f:
            config = json.load(f)

        expected_fields = [
                "pp_deg", "tp_sizes_enc", "tp_consecutive_flags", 
                "dp_types_enc", "use_sp", "checkpoint", "global_bsz",
                "chunks", "pp_division", "pipeline_type", 
                "default_dp_type", "vtp", "vsp"
            ]
        for field in expected_fields:
            assert field in config, f"Missing field: {field}"

        assert config["pp_deg"] == 1
        assert config["global_bsz"] == 64
        assert config["chunks"] == 8
        assert config["pp_division"] == "28", f'idx: {idx}, pp_division: {config["pp_division"]}'
        assert config["pipeline_type"] == "pipedream_flush"
        assert config["default_dp_type"] == "zero2"
        assert config["vtp"] == 1
        assert config["vsp"] == 0
        assert config["embed_sdp"] == 1

        layer_strategy_list = config2strategy(config, default_dp_type="zero2")
        string_list = [strategy.to_simple_string() for strategy in layer_strategy_list]
        string_list = ', '.join(string_list)
        assert string_list == "1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c, 1-1-8f-c"