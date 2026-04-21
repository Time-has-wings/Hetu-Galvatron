"""Tests for argument loading and Pydantic schemas (Hydra + CoreArgs).

Historically this module tested ``galvatron_training_args`` and related **argparse**
builders; those entry points were removed in favor of ``load_with_hydra`` and
``galvatron.core.args_schema``. Coverage is therefore split between:

- **YAML + Hydra**: ``train_dist.yaml`` → ``GalvatronRuntimeArgs`` (``mode="train_dist"``).
- **Standalone schemas**: defaults of ``ProfilerArgs``, ``ProfilerHardwareArgs``,
  ``GalvatronSearchArgs`` mirror the old argparse default assertions where the schema
  still matches.
"""

from pathlib import Path

import pytest

from galvatron.core.arguments import load_with_hydra
from galvatron.core.args_schema import ProfilerHardwareArgs, GalvatronSearchArgs
from galvatron.core.profiler.args_schema import GalvatronModelProfilerArgs

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_DIST_YAML = _REPO_ROOT / "galvatron" / "models" / "gpt" / "scripts" / "train_dist.yaml"


@pytest.mark.utils
def test_load_with_hydra_train_dist_runtime_matches_yaml():
    """Values resolved from ``train_dist.yaml`` (plus schema defaults)."""
    args = load_with_hydra(str(_TRAIN_DIST_YAML), mode="train_dist")

    assert args.parallel.pp_deg == 1
    assert args.parallel.global_tp_deg == 2
    assert args.parallel.default_dp_type == "ddp"
    assert args.parallel.pipeline_type == "gpipe"
    assert args.parallel.mixed_precision == "bf16"

    assert args.model.model_type == "llama"
    assert args.model.model_size == "llama2-7b"

    assert args.profile.profile == 1
    assert args.profile.profile_mode == "static"
    assert args.profile.profile_unit == "all"
    assert args.profile.save_profiled_memory == 0
    assert args.profile.exit_after_profiling == 0

    assert args.train.train_iters == 20
    assert args.train.eval_iters == 1
    assert args.train.lr == pytest.approx(6.0e-4)
    assert args.train.min_lr == pytest.approx(6.0e-5)
    assert args.train.global_batch_size == 32
    assert args.train.micro_batch_size == 1
    assert args.train.seq_length == 4096

    assert args.data.split == "949,50,1"
    assert args.data.tokenizer_type == "HuggingFaceTokenizer"
    assert args.data.shared_storage is True

    assert args.ckpt.load is None
    assert args.ckpt.distributed_checkpoint is False


@pytest.mark.utils
def test_load_with_hydra_train_dist_overrides():
    """Hydra overrides apply on top of the composed config (keys match YAML nesting)."""
    args = load_with_hydra(
        str(_TRAIN_DIST_YAML),
        mode="train_dist",
        overrides=["runtime.train.lr=1e-5", "runtime.parallel.pp_deg=2"],
    )
    assert args.train.lr == pytest.approx(1e-5)
    assert args.parallel.pp_deg == 2


@pytest.mark.utils
def test_profiler_args_defaults():
    """Defaults aligned with former ``galvatron_profile_args`` expectations."""
    args = GalvatronModelProfilerArgs()
    assert args.profile_type == "memory"
    assert args.profile_mode == "static"
    assert args.profile_batch_size_step is None
    assert args.profile_seq_length_step is None
    assert args.profile_layernum_min == 1
    assert args.profile_layernum_max == 2
    assert args.profile_max_tp_deg == 8
    assert args.profile_dp_type == "zero3"
    assert args.profile_mixed_precision == "bf16"


@pytest.mark.utils
def test_profiler_hardware_args_defaults():
    """Defaults aligned with former ``galvatron_profile_hardware_args`` expectations."""
    args = ProfilerHardwareArgs()
    assert args.num_nodes == 1
    assert args.num_gpus_per_node == 8
    assert args.master_addr == "$MASTER_ADDR"
    assert args.master_port == "$MASTER_PORT"
    assert args.node_rank == "$RANK"
    assert args.max_tp_size == 8
    assert args.envs == []
    assert args.max_pp_deg == 8
    assert args.overlap_time_multiply == 4



@pytest.mark.utils
def test_search_engine_args_defaults():
    """Defaults aligned with former ``galvatron_search_args`` expectations."""
    args = GalvatronSearchArgs()
    assert args.hardware_info.num_nodes == 1
    assert args.hardware_info.num_gpus_per_node == 8
    assert args.hardware_info.memory_constraint == 24
    assert args.batch_size_info.min_bsz == 8
    assert args.batch_size_info.max_bsz == 8
    assert args.batch_size_info.bsz_scale == 8
    assert args.search_space_info.max_tp_deg == 8
    assert args.search_space_info.max_pp_deg == 8
    assert args.parallelism_info.default_dp_type == "ddp"
    assert args.parallelism_info.mixed_precision == "bf16"
    assert args.parallelism_info.pipeline_type == "gpipe"
    assert args.debug_info.debug_costmodel_coe == 1.0
    assert args.options_info.fine_grained_mode == 1
