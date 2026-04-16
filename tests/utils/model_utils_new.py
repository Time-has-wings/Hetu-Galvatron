import os
from dataclasses import dataclass
from typing import Callable, Type, NamedTuple, Optional, Union
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
from galvatron.core.search_engine.args_schema import GalvatronSearchArgs


class ModelComponents(NamedTuple):
    ModelClass: Type
    get_model_config: Callable
    get_model: Callable
    convert_checkpoints: Optional[Callable]
    DatasetClass: Type
    collate_fn: Callable

class ModelFactory:
    @staticmethod
    # def get_components(model_type: str, backend: str) -> ModelComponents:
    #     """Get model components based on model type and backend.
        
    #     Args:
    #         model_type: "gpt", "llama", or "llama2"
    #         backend: "hf" or "fa"
            
    #     Returns:
    #         ModelComponents containing all necessary model components
    #     """
    #     if model_type.startswith("gpt") and backend == "hf":
    #         from galvatron.models.gpt_hf.GPTModel_hybrid_parallel import (
    #             get_gpt_config as get_model_config,
    #             gpt_model_hp as get_model,
    #         )
    #         from transformers import GPT2LMHeadModel as ModelClass
    #         from galvatron.tools.checkpoint_convert_h2g import convert_checkpoints_gpt as convert_checkpoints
    #         from galvatron.models.gpt_hf.dataloader import (
    #             DataLoaderForGPT as DatasetClass,
    #             random_collate_fn as collate_fn
    #         )
        
    #     elif model_type.startswith("llama") and backend == "hf":
    #         from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import(
    #             get_llama_config as get_model_config,
    #             llama_model_hp as get_model,
    #         )
    #         from transformers import LlamaForCausalLM as ModelClass
    #         from galvatron.tools.checkpoint_convert_h2g import convert_checkpoints_llama as convert_checkpoints
    #         from galvatron.models.llama_hf.dataloader import (
    #             DataLoaderForLlama as DatasetClass,
    #             random_collate_fn as collate_fn
    #         )

    #     elif model_type.startswith("gpt") and backend == "fa":
    #         from galvatron.models.gpt_fa.GPTModel_hybrid_parallel import (
    #             get_gpt_config as get_model_config,
    #             gpt_model_hp as get_model,
    #         )
    #         from flash_attn.models.gpt import GPTLMHeadModel as ModelClass
    #         from galvatron.models.gpt_fa.dataloader import (
    #             DataLoaderForGPT as DatasetClass,
    #             random_collate_fn as collate_fn
    #         )
    #         convert_checkpoints = None
            
    #     elif model_type.startswith("llama") and backend == "fa":
    #         from galvatron.models.llama_fa.LlamaModel_hybrid_parallel import (
    #             get_llama_config as get_model_config,
    #             llama_model_hp as get_model,
    #         )
    #         from flash_attn.models.gpt import GPTLMHeadModel as ModelClass
    #         from galvatron.models.llama_fa.dataloader import (
    #             DataLoaderForLlama as DatasetClass,
    #             random_collate_fn as collate_fn
    #         )
    #         convert_checkpoints = None
            
    #     else:
    #         raise ValueError(f"Unsupported model type: {model_type} with backend: {backend}")

    #     return ModelComponents(
    #         ModelClass=ModelClass,
    #         get_model_config=get_model_config,
    #         get_model=get_model,
    #         convert_checkpoints=convert_checkpoints,
    #         DatasetClass=DatasetClass,
    #         collate_fn=collate_fn
    #     )

    # @staticmethod
    # def get_meta_configs(model_type: str, backend: str):
    #     from galvatron.utils.hf_config_adapter import model_layer_configs, model_name
    #     return model_layer_configs, model_name

    @staticmethod
    def resolve_model_config(args:Union[GalvatronRuntimeArgs, GalvatronSearchArgs], model_type: str, backend: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_yaml_dir = os.path.join(current_dir, "./model_configs")
        if model_type.startswith("gpt") and backend == "hf":
            model_yaml_path = os.path.join(model_yaml_dir, "gpt2-small.yaml")
        elif model_type.startswith("gpt") and backend == "fa":
            model_yaml_path = os.path.join(model_yaml_dir, "gpt2-small.yaml")
        elif model_type.startswith("llama") and backend == "hf":
            model_yaml_path = os.path.join(model_yaml_dir, "llama2-7b.yaml")
        elif model_type.startswith("llama") and backend == "fa":
            model_yaml_path = os.path.join(model_yaml_dir, "llama2-7b.yaml")
        else:
            raise ValueError(f"Unsupported model type: {model_type} with backend: {backend}")
        
        if isinstance(args, GalvatronSearchArgs):
            args.model_info.model_config_path = model_yaml_path
        elif isinstance(args, GalvatronRuntimeArgs):
            args.model.model_config_path = model_yaml_path
        else:
            raise ValueError(f"Unsupported args type: {type(args)}")  
        
        from galvatron.utils.hf_config_adapter import resolve_model_config
        resolve_model_config(args)
    
    def get_model_layer_configs_func() -> Callable:
        from galvatron.utils.hf_config_adapter import model_layer_configs as model_layer_configs_func
        return model_layer_configs_func


    def get_model_name_func() -> Callable:
        from galvatron.utils.hf_config_adapter import model_name as model_name_func
        return model_name_func

    
    @staticmethod
    def create_model(model_type: str, backend: str, config, args):
        """Factory method to create a model instance.
        
        Args:
            model_type: Type of model
            backend: Backend framework
            config: Model configuration
            args: Training arguments
            
        Returns:
            Instantiated model
        """
        components = ModelFactory.get_components(model_type, backend)
        return components.get_model(config, args)

    @staticmethod
    def create_config(model_type: str, backend: str, args, overwrite_args=True):
        """Factory method to create model configuration.
        
        Args:
            model_type: Type of model
            backend: Backend framework
            args: Training arguments
            
        Returns:
            Model configuration
        """
        components = ModelFactory.get_components(model_type, backend)
        return components.get_model_config(args, overwrite_args)