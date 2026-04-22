import os
from typing import Callable, List, Dict, Any, Optional, Union
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
from galvatron.core.search_engine.args_schema import GalvatronSearchArgs


class ModelFactory:
    """Unified model config factory for all Galvatron tests.

    All model configs live as YAML files under ``tests/utils/model_configs/``.
    Production-size configs (e.g. llama2-7b.yaml) are used by search/profiler tests.
    Small test configs (e.g. gpt-test.yaml) are used by core/models correctness tests.
    """

    # Production-size YAML mapping (for search/profiler tests)
    _YAML_MAP = {
        "gpt": "gpt2-small.yaml",
        "llama": "llama2-7b.yaml",
        "mixtral": "mistral-7b.yaml",
    }

    # Small test YAML mapping (for core/models correctness tests)
    _TEST_YAML_MAP = {
        "gpt": "gpt-test.yaml",
        "gpt256": "gpt-test-256.yaml",
        "llama": "llama-test.yaml",
        "llama2": "llama2-test.yaml",
        "mixtral": "mixtral-test.yaml",
    }

    @staticmethod
    def _get_yaml_dir() -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_configs")

    @staticmethod
    def _resolve_yaml_path(model_type: str) -> str:
        """Resolve production YAML config path based on model_type prefix."""
        yaml_dir = ModelFactory._get_yaml_dir()
        for prefix, yaml_file in ModelFactory._YAML_MAP.items():
            if model_type.startswith(prefix):
                return os.path.join(yaml_dir, yaml_file)
        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def resolve_model_config(args: Union[GalvatronRuntimeArgs, GalvatronSearchArgs], model_type: str):
        """Resolve model config from production YAML based on model_type."""
        model_yaml_path = ModelFactory._resolve_yaml_path(model_type)

        if isinstance(args, GalvatronSearchArgs):
            args.model_info.model_config_path = model_yaml_path
        elif isinstance(args, GalvatronRuntimeArgs):
            args.model.model_config_path = model_yaml_path
        else:
            raise ValueError(f"Unsupported args type: {type(args)}")

        from galvatron.utils.hf_config_adapter import resolve_model_config
        resolve_model_config(args)

    @staticmethod
    def get_test_config(model_type: str) -> Dict[str, Any]:
        """Load small test model config from YAML, returning a flat dict.

        Keys use Galvatron-standard
        names: hidden_size, num_layers, num_attention_heads, ffn_hidden_size,
        vocab_size, seq_length, norm_epsilon, etc.
        """
        import yaml

        if model_type not in ModelFactory._TEST_YAML_MAP:
            raise ValueError(f"Unsupported test model type: {model_type}. "
                             f"Available: {list(ModelFactory._TEST_YAML_MAP.keys())}")

        yaml_path = os.path.join(ModelFactory._get_yaml_dir(), ModelFactory._TEST_YAML_MAP[model_type])
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Ensure seq_length has a default (32 for small tests)
        if "seq_length" not in data:
            data["seq_length"] = 32

        return data

    @staticmethod
    def get_model_layer_configs(args: Union[GalvatronRuntimeArgs, GalvatronSearchArgs]) -> List[Dict[str, Any]]:
        """Get model layer configs from resolved args."""
        from galvatron.utils.hf_config_adapter import model_layer_configs
        return model_layer_configs(args)

    @staticmethod
    def get_model_name(args: Union[GalvatronRuntimeArgs, GalvatronSearchArgs]) -> str:
        """Get model name from resolved args."""
        from galvatron.utils.hf_config_adapter import model_name
        return model_name(args)

    @staticmethod
    def get_model_layer_configs_func() -> Callable:
        """Return the model_layer_configs function reference."""
        from galvatron.utils.hf_config_adapter import model_layer_configs as func
        return func

    @staticmethod
    def get_model_name_func() -> Callable:
        """Return the model_name function reference."""
        from galvatron.utils.hf_config_adapter import model_name as func
        return func
