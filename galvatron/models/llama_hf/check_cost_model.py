import os
from galvatron.core import initialize_galvatron
from galvatron.core.cost_model import GalvatronCostModelHandler
from galvatron.models.llama_hf.arguments import model_args
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_llama_config
from galvatron.models.llama_hf.meta_configs import model_layer_configs, model_name
from galvatron.utils.strategy_utils import LayerStrategy, DPType
from galvatron.models.llama_hf.dataloader import init_loguru

if __name__ ==  '__main__':
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_file = f'check_cost_model_llama/{timestamp}.log'
    init_loguru(log_file)

    args = initialize_galvatron(model_args, mode="cost_model")
    config = get_llama_config(args)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)

    cost_model_handler = GalvatronCostModelHandler(args)
    cost_model_handler.set_cost_model_handler_info(path, model_layer_configs(config), model_name(config))
    cost_model_handler.initialize_cost_model_handler()
    
    time_cases = [
        {
            "global_batch_size": 128, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=1, dp_size=8, dp_type=DPType.ZERO2)
        },
        {
            "global_batch_size": 128, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        },
        {
            "global_batch_size": 128, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=4, dp_size=2, dp_type=DPType.ZERO2)
        },
        {
            "global_batch_size": 128, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=8, dp_size=1, dp_type=DPType.ZERO2)
        },
        {
            "global_batch_size": 128, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=1, dp_size=8, dp_type=DPType.ZERO2, checkpoint=True)
        },
        {
            "global_batch_size": 128, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=2, dp_size=4, dp_type=DPType.ZERO2, checkpoint=True)
        },
        {
            "global_batch_size": 128, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=4, dp_size=2, dp_type=DPType.ZERO2, checkpoint=True)
        },
        {
            "global_batch_size": 128, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=8, dp_size=1, dp_type=DPType.ZERO2, checkpoint=True)
        },
    ]
    
    memory_cases = [
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=1, dp_size=8, dp_type=DPType.DDP)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=1, dp_size=8, dp_type=DPType.ZERO2)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=1, dp_size=8, dp_type=DPType.ZERO3)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=2, dp_size=4, dp_type=DPType.DDP)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=2, dp_size=4, dp_type=DPType.ZERO2)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=2, dp_size=4, dp_type=DPType.ZERO3)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=4, dp_size=2, dp_type=DPType.DDP)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=4, dp_size=2, dp_type=DPType.ZERO2)
        },
        {
            "global_batch_size": 64, "chunks":8,
            "strategy": LayerStrategy(pp_size=1, tp_size=4, dp_size=2, dp_type=DPType.ZERO3)
        },
        
    ]

    time_cost_list = []
    for case in time_cases:
        global_batch_size = case["global_batch_size"]
        chunks = case["chunks"]
        strategy = case["strategy"]
        print(f"\n=== Check Cost for Global_batch_size: {global_batch_size}, Chunks: {chunks}, Strategy: {strategy} ===")
        time_cost = cost_model_handler.get_time_cost_for_specific_strategy_together(strategy, global_batch_size, chunks)
        print(f'Time Cost: {time_cost * 1000:.1f} ms')
        time_cost_list.append(time_cost * 1000)

    memory_cost_list = []
    for case in memory_cases:
        global_batch_size = case["global_batch_size"]
        chunks = case["chunks"]
        strategy = case["strategy"]
        print(f"\n=== Check Cost for Global_batch_size: {global_batch_size}, Chunks: {chunks}, Strategy: {strategy} ===")
        memory_cost = cost_model_handler.get_memory_cost_for_specific_strategy_together(strategy, global_batch_size, chunks)
        print(f'Memory Cost: {memory_cost:.2f} MB')
        memory_cost_list.append(memory_cost)

    print('all cases done')
    for time_cost in time_cost_list:
        print(f'{time_cost:.1f}')
    
    print('\n')

    for idx, memory_cost in enumerate(memory_cost_list):
        print(f'{memory_cost:.2f}')
        if idx % 3 == 2:
            print('\n')