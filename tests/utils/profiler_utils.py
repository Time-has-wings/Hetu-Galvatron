from galvatron.core.profiler import HardwareProfiler, ModelProfiler, RuntimeProfiler
from galvatron.core.profiler.args_schema import GalvatronModelProfilerArgs, ProfilerHardwareArgs
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs, GalvatronModelArgs
from tests.utils.model_utils import ModelFactory


def initialize_model_profile_profiler(profiler_model_configs_dir, model_type, **kwargs):
    """Build a ModelProfiler with Pydantic args matching production (Hydra / args_schema)."""
    _ = model_type  # fixture API compatibility
    defaults = dict(
        profile_type="memory",
        profile_mode="static",
        profile_unit="all",
        profile_flow_control="all",
        profile_mixed_precision="bf16",
        profile_fixed_batch_size=8,
        profile_fixed_seq_length_list=[4096],
        profile_layernum_min=1,
        profile_layernum_max=2,
        profile_batch_size_step=1,
        profile_seq_length_step=128,
        profile_max_tp_deg=8,
        runtime_yaml_template_path="scripts/profile_runtime.yaml",
        model_info=GalvatronModelArgs(model_size="test_model"),
    )
    defaults.update(kwargs)
    args = GalvatronModelProfilerArgs(**defaults)
    profiler = ModelProfiler(args)
    profiler.set_profiler_launcher(str(profiler_model_configs_dir.parent), model_name="test")
    return profiler


def initialize_hardware_profile_profiler(profiler_hardware_configs_dir):
    """Initialize hardware profiler."""
    args = ProfilerHardwareArgs()
    profiler = HardwareProfiler(args)
    profiler.set_path(profiler_hardware_configs_dir)
    return profiler


def initialize_runtime_profile_profiler(profiler_model_configs_dir, model_type, **kwargs):
    """Initialize runtime profiler via ModelFactory."""
    args = GalvatronRuntimeArgs()
    args.profile.profile = True

    # Resolve model config (loads from YAML via ModelFactory)
    ModelFactory.resolve_model_config(args, model_type)

    # Get layer configs and model name via ModelFactory
    layer_configs = ModelFactory.get_model_layer_configs(args)
    name = ModelFactory.get_model_name(args)

    # Initialize profiler
    profiler = RuntimeProfiler(args)
    profiler.set_profiler_dist(
        str(profiler_model_configs_dir.parent),
        layer_configs,
        name,
        rank=0,
        profile_ranks=[0],
        **kwargs,
    )
    return profiler
