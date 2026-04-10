import pytest
from galvatron.core.search_engine.search_engine import GalvatronSearchEngine
from galvatron.core.search_engine.args_schema import GalvatronSearchArgs
from galvatron.utils.strategy_utils import print_strategy_list
from tests.utils.model_utils_new import ModelFactory

@pytest.mark.search_engine
@pytest.mark.parametrize("model_type", ["llama_search"])
@pytest.mark.parametrize("backend", ["hf"])
@pytest.mark.parametrize("disables", [['cp']])
def test_generate_strategies(model_type, backend, tmp_path, disables, capsys):

    args = GalvatronSearchArgs()

    for disable in disables:
        setattr(args.search_space_info, f"disable_{disable}", 1)
    args.parallelism_info.default_dp_type = 'zero2'

    ModelFactory.resolve_model_config(args, model_type, backend)
    model_layer_configs_func = ModelFactory.get_model_layer_configs_func()
    model_name_func = ModelFactory.get_model_name_func()
    
    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(tmp_path, model_layer_configs_func(args), model_name_func(args))

    search_engine.generate_strategy_list()
    search_engine.filter_strategy_list()

    if disables == ['cp']:
        assert len(search_engine.layer_strategy_list) == 50
        capsys.readouterr()
        print_strategy_list(search_engine.layer_strategy_list)
        captured = capsys.readouterr()
        assert captured.out.strip() == "1-1-8, 1-1-8-c, 1-1-8f, 1-1-8f-c, 1-2*-4-sp, 1-2*-4-c-sp, 1-2*-4f-sp, 1-2*-4f-c-sp, 1-4*-2-sp, 1-4*-2-c-sp, 1-4*-2f-sp, 1-4*-2f-c-sp, 1-8*-1-sp, 1-8*-1-c-sp, 1-2*-4, 1-2*-4-c, 1-2*-4f, 1-2*-4f-c, 1-4*-2, 1-4*-2-c, 1-4*-2f, 1-4*-2f-c, 1-8*-1, 1-8*-1-c, 2-1-4, 2-1-4-c, 2-1-4f, 2-1-4f-c, 2-2*-2-sp, 2-2*-2-c-sp, 2-2*-2f-sp, 2-2*-2f-c-sp, 2-4*-1-sp, 2-4*-1-c-sp, 2-2*-2, 2-2*-2-c, 2-2*-2f, 2-2*-2f-c, 2-4*-1, 2-4*-1-c, 4-1-2, 4-1-2-c, 4-1-2f, 4-1-2f-c, 4-2*-1-sp, 4-2*-1-c-sp, 4-2*-1, 4-2*-1-c, 8-1-1, 8-1-1-c"
    else:
        assert len(search_engine.layer_strategy_list) > 0
