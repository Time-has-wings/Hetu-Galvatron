import os
import sys
import time

from galvatron.core.arguments import load_with_hydra
from galvatron.core.search_engine.search_engine import GalvatronSearchEngine
from galvatron.core.search_engine.args_schema import GalvatronSearchArgs
from galvatron.utils.hf_config_adapter import model_name, model_layer_configs, resolve_model_config
from galvatron.utils.print_utils import print_args_rank0, print_single_rank

if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1].endswith((".yaml", ".yml")):
        config_path, overrides = sys.argv[1], sys.argv[2:]
        sys.argv = [sys.argv[0]]
        args: GalvatronSearchArgs = load_with_hydra(config_path, overrides=overrides, mode="search")
    else:
        raise ValueError("Usage: python profiler.py <config_path> [overrides...]")

    search_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print_single_rank(f"Search started at {search_time}")

    resolve_model_config(args)
    print_args_rank0(args, title="Galvatron Search Arguments")

    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(
        path=os.path.dirname(os.path.abspath(__file__)),
        model_layer_configs=model_layer_configs(args), 
        model_name=model_name(args)
    )
    
    search_engine.initialize_search_engine(show_all_strategy_list=True)
    search_engine.parallelism_optimization()
