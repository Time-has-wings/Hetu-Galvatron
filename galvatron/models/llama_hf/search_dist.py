import os
from galvatron.core import initialize_galvatron
from galvatron.models.llama_hf.arguments import model_args
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_llama_config
from galvatron.models.llama_hf.meta_configs import model_layer_configs, model_name
from galvatron.core.search_engine.search_engine import GalvatronSearchEngine
from rich.pretty import pretty_repr

if __name__ ==  '__main__':
    args = initialize_galvatron(model_args, mode="search")
    config = get_llama_config(args)
    path = os.path.dirname(os.path.abspath(__file__))

    print(pretty_repr(args))
    print(pretty_repr(config))

    search_engine = GalvatronSearchEngine(args=args)
    search_engine.initialize_cost_model_handler(path, model_layer_configs(config), model_name(config))
    search_engine.initialize_search_engine()
    search_engine.parallelism_optimization()