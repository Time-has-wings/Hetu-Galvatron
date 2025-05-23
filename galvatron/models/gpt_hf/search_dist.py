import os

from galvatron.core import GalvatronSearchEngine, initialize_galvatron
from galvatron.models.gpt_hf.arguments import model_args
from galvatron.models.gpt_hf.GPTModel_hybrid_parallel import get_gpt_config
from galvatron.models.gpt_hf.meta_configs import model_layer_configs, model_name

if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="search")
    config = get_gpt_config(args)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)

    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(path, model_layer_configs(config), model_name(config))
    # search_engine.set_microbatch_func(microbatch_size=4, max_chunk=8) # Optional
    search_engine.set_model_type("gpt")  # Optional

    search_engine.initialize_search_engine()
    # search_engine.check_cost_model(bsz=16 ,chunk=1, min_tp=1)
    search_engine.parallelism_optimization()
