import pytest
import sys
from io import StringIO
from galvatron.core.search_engine.search_engine import GalvatronSearchEngine
from galvatron.utils.strategy_utils import print_strategy_list
from tests.utils.search_args import SearchArgs
from tests.utils.model_utils import ModelFactory
from tests.models.configs.get_config_json import ConfigFactory

@pytest.mark.search_engine
@pytest.mark.parametrize("model_type", ["llama_search"])
@pytest.mark.parametrize("backend", ["hf"])
@pytest.mark.parametrize("disables", [['cp']])
def test_generate_strategies(model_type, backend, tmp_path, disables, monkeypatch):

    args = SearchArgs()
    args.disables = disables

    for disable in disables:
        setattr(args, f"disable_{disable}", 1)

    model_layer_configs, model_name = ModelFactory.get_meta_configs(model_type, backend)
    config_json = ConfigFactory.get_config_json(model_type)
    args.model_size = config_json
    args.local_rank = 0
    config = ModelFactory.create_config(model_type, backend, args)
    
    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(tmp_path, model_layer_configs(config), model_name(config))

    search_engine.generate_strategy_list()
    search_engine.filter_strategy_list()

    if disables == ['cp']:
        assert len(search_engine.layer_strategy_list) == 50
        original_write = sys.stdout.write
        capture_flag = {'enabled': False}
        output_buffer = StringIO()
        def custom_write(text):
            original_write(text)
            if capture_flag['enabled']:
                output_buffer.write(text)
        monkeypatch.setattr(sys.stdout, 'write', custom_write)
        capture_flag['enabled'] = True
        print_strategy_list(search_engine.layer_strategy_list)
        capture_flag['enabled'] = False
        assert output_buffer.getvalue().strip() == "1-1-8, 1-1-8-c, 1-1-8f, 1-1-8f-c, 1-2*-4-sp, 1-2*-4-c-sp, 1-2*-4f-sp, 1-2*-4f-c-sp, 1-4*-2-sp, 1-4*-2-c-sp, 1-4*-2f-sp, 1-4*-2f-c-sp, 1-8*-1-sp, 1-8*-1-c-sp, 1-2*-4, 1-2*-4-c, 1-2*-4f, 1-2*-4f-c, 1-4*-2, 1-4*-2-c, 1-4*-2f, 1-4*-2f-c, 1-8*-1, 1-8*-1-c, 2-1-4, 2-1-4-c, 2-1-4f, 2-1-4f-c, 2-2*-2-sp, 2-2*-2-c-sp, 2-2*-2f-sp, 2-2*-2f-c-sp, 2-4*-1-sp, 2-4*-1-c-sp, 2-2*-2, 2-2*-2-c, 2-2*-2f, 2-2*-2f-c, 2-4*-1, 2-4*-1-c, 4-1-2, 4-1-2-c, 4-1-2f, 4-1-2f-c, 4-2*-1-sp, 4-2*-1-c-sp, 4-2*-1, 4-2*-1-c, 8-1-1, 8-1-1-c"
    else:
        assert len(search_engine.layer_strategy_list) > 0
