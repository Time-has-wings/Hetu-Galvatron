import os
from galvatron.core import initialize_galvatron
from galvatron.models.llama_hf.arguments import model_args
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_llama_config
from galvatron.models.llama_hf.meta_configs import model_layer_configs, model_name
from galvatron.core.search_engine.search_engine_version2 import GalvatronSearchEngineOptimize

if __name__ ==  '__main__':
    args = initialize_galvatron(model_args, mode="cost_model")
    config = get_llama_config(args)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)

    search_engine = GalvatronSearchEngineOptimize(args)
    search_engine.handler.set_cost_model_handler_info(path, model_layer_configs(config), model_name(config))
    search_engine.handler.initialize_cost_model_handler()
    
    search_engine.generate_strategy_list_optimize(world_size=8, is_moe=False)
    
    search_engine.parallelism_optimization(
            world_size=8,
            gbsz_list=[64]
        )