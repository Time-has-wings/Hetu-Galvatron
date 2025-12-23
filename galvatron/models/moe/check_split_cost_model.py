import os
import copy
from galvatron.core import initialize_galvatron
from galvatron.core.cost_model import GalvatronCostModelHandler
from galvatron.models.moe.arguments import model_args
from galvatron.models.moe.MoEModel_hybrid_parallel import get_moe_config
from galvatron.models.moe.meta_configs import model_layer_configs, model_name
from galvatron.utils.strategy_utils import GalvatronStrategy

if __name__ ==  '__main__':
    args = initialize_galvatron(model_args, mode="cost_model")
    config = get_moe_config(args)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)

    cost_model_handler = GalvatronCostModelHandler(args)
    cost_model_handler.set_cost_model_handler_info(path, model_layer_configs(config), model_name(config))
    cost_model_handler.initialize_cost_model_handler()
    
    cases = [
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": GalvatronStrategy(pp_size=1, tp_size=8, dp_size=1, dp_type='zero3'),
            "moe_strategy":GalvatronStrategy(pp_size=1, ep_size=8, tp_of_ep_size=1)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": GalvatronStrategy(pp_size=1, tp_size=8, dp_size=1, dp_type='zero3'),
            "moe_strategy":GalvatronStrategy(pp_size=1, ep_size=4, tp_of_ep_size=2)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": GalvatronStrategy(pp_size=1, tp_size=8, dp_size=1, dp_type='zero3'),
            "moe_strategy":GalvatronStrategy(pp_size=1, ep_size=2, tp_of_ep_size=4)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": GalvatronStrategy(pp_size=1, tp_size=8, dp_size=1, dp_type='zero3'),
            "moe_strategy":GalvatronStrategy(pp_size=1, ep_size=1, tp_of_ep_size=8)
        },
        # checkpoint
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": GalvatronStrategy(pp_size=1, tp_size=8, dp_size=1, dp_type='zero3', checkpoint=True),
            "moe_strategy":GalvatronStrategy(pp_size=1, ep_size=8, tp_of_ep_size=1, checkpoint=True)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": GalvatronStrategy(pp_size=1, tp_size=8, dp_size=1, dp_type='zero3', checkpoint=True),
            "moe_strategy":GalvatronStrategy(pp_size=1, ep_size=4, tp_of_ep_size=2, checkpoint=True)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": GalvatronStrategy(pp_size=1, tp_size=8, dp_size=1, dp_type='zero3', checkpoint=True),
            "moe_strategy":GalvatronStrategy(pp_size=1, ep_size=2, tp_of_ep_size=4, checkpoint=True)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": GalvatronStrategy(pp_size=1, tp_size=8, dp_size=1, dp_type='zero3', checkpoint=True),
            "moe_strategy":GalvatronStrategy(pp_size=1, ep_size=1, tp_of_ep_size=8, checkpoint=True)
        },
    ]
    
    time_cost_list = []
    for case in cases:
        global_batch_size = case["global_batch_size"]
        chunks = case["chunks"]
        attention_strategy:GalvatronStrategy = copy.deepcopy(case["strategy"])
        attention_strategy.unit = 'attention'
        ffn_strategy:GalvatronStrategy = copy.deepcopy(case["moe_strategy"])
        ffn_strategy.unit = 'ffn'
        ffn_strategy.is_moe = True
        embedding_lmhead_strategy:GalvatronStrategy = copy.deepcopy(case["strategy"])
        embedding_lmhead_strategy.unit = 'embedding_lmhead'
        embedding_lmhead_strategy.checkpoint = False

        print(f"\n=== Check Cost for Global_batch_size: {global_batch_size}, Chunks: {chunks}, attention_strategy: {attention_strategy}, ffn_strategy: {ffn_strategy}, embedding_lmhead_strategy: {embedding_lmhead_strategy} ===")
        time_cost = cost_model_handler.get_time_cost_for_specific_strategy_moe(attention_strategy, ffn_strategy, embedding_lmhead_strategy, global_batch_size, chunks)
        print(f'Time Cost: {time_cost}')
        time_cost_list.append(time_cost * 1000)

    print('all cases done')
    for time_cost in time_cost_list:
        print(f'{time_cost:.1f}')


    memory_cost_list = []
    for case in cases:
        global_batch_size = case["global_batch_size"]
        chunks = case["chunks"]
        attention_strategy:GalvatronStrategy = copy.deepcopy(case["strategy"])
        attention_strategy.unit = 'attention'
        ffn_strategy:GalvatronStrategy = copy.deepcopy(case["moe_strategy"])
        ffn_strategy.unit = 'ffn'
        ffn_strategy.is_moe = True
        embedding_lmhead_strategy:GalvatronStrategy = copy.deepcopy(case["strategy"])
        embedding_lmhead_strategy.unit = 'embedding_lmhead'
        embedding_lmhead_strategy.checkpoint = False

        print(f"\n=== Check Cost for Global_batch_size: {global_batch_size}, Chunks: {chunks}, attention_strategy: {attention_strategy}, ffn_strategy: {ffn_strategy}, embedding_lmhead_strategy: {embedding_lmhead_strategy} ===")
        memory_cost = cost_model_handler.get_memory_cost_for_specific_strategy_moe(attention_strategy, ffn_strategy, embedding_lmhead_strategy, global_batch_size, chunks)
        print(f'Memory Cost: {memory_cost}')
        memory_cost_list.append(memory_cost)

    print('all cases done')
    for memory_cost in memory_cost_list:
        print(f'{memory_cost:.1f}')