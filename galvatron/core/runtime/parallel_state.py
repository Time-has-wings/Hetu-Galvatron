import os
from galvatron.core.runtime.utils.utils import GlobalMemoryBuffer
from galvatron.core.runtime.datasets.megatron.tokenizer import build_tokenizer
import torch
import torch.distributed
from datetime import timedelta

_GLOBAL_MEMORY_BUFFER = None

def _set_global_memory_buffer():
    """Initialize global buffer."""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None

_GLOBAL_ARGS = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TENSORBOARD_WRITER = None
_GLOBAL_WANDB_WRITER = None


def get_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return _GLOBAL_ARGS


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return _GLOBAL_TOKENIZER


def set_args(args):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def _build_tokenizer(args):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def _set_tensorboard_writer(args):
    """Set tensorboard writer. *args* is the full GalvatronRuntimeArgs."""
    global _GLOBAL_TENSORBOARD_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_TENSORBOARD_WRITER,
                                   'tensorboard writer')
    log_cfg = args.logging
    if getattr(log_cfg, 'tensorboard_dir', None) and \
       args.rank == (args.world_size - 1):
        try:
            from torch.utils.tensorboard import SummaryWriter
            print('> setting tensorboard ...')
            _GLOBAL_TENSORBOARD_WRITER = SummaryWriter(
                log_dir=log_cfg.tensorboard_dir,
                max_queue=log_cfg.tensorboard_queue_size)
        except ModuleNotFoundError:
            print('WARNING: TensorBoard writing requested but is not '
                  'available (are you using PyTorch 1.1.0 or later?), '
                  'no TensorBoard logs will be written.', flush=True)


def _set_wandb_writer(args):
    """Set wandb writer. *args* is the full GalvatronRuntimeArgs."""
    global _GLOBAL_WANDB_WRITER
    _ensure_var_is_not_initialized(_GLOBAL_WANDB_WRITER,
                                   'wandb writer')
    log_cfg = args.logging
    if getattr(log_cfg, 'wandb_project', '') and args.rank == (args.world_size - 1):
        if log_cfg.wandb_exp_name == '':
            raise ValueError("Please specify the wandb experiment name!")

        import wandb
        if log_cfg.wandb_save_dir:
            save_dir = log_cfg.wandb_save_dir
        else:
            save_dir = os.path.join(args.ckpt.save, 'wandb')
        wandb_kwargs = {
            'dir': save_dir,
            'name': log_cfg.wandb_exp_name,
            'project': log_cfg.wandb_project,
            'config': args.model_dump()}
        os.makedirs(wandb_kwargs['dir'], exist_ok=True)
        wandb.init(**wandb_kwargs)
        _GLOBAL_WANDB_WRITER = wandb


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)


def set_global_variables(args):
    """Set global variables."""
    set_args(args)
    _build_tokenizer(args)
    _set_tensorboard_writer(args)
    _set_wandb_writer(args)

def _initialize_distributed(args):
    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print(
                "torch distributed is already initialized, " "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            torch.cuda.set_device(args.local_rank)
            device_id = torch.device(f'cuda:{args.local_rank}')
        else:
            device_id = None

        # Call the init process
        init_process_group_kwargs = {
            'backend': args.distributed_backend,
            'world_size': args.world_size,
            'rank': args.rank,
            # 'device_id': device_id,
            'timeout': timedelta(minutes=args.distributed_timeout_minutes),
        }

        torch.distributed.init_process_group(**init_process_group_kwargs)


# TODO: Add vpp support
def get_virtual_pipeline_model_parallel_rank():
    return None


_GLOBAL_VOCAB_TENSOR_PARALLEL_GROUP = None
_GLOBAL_VOCAB_CONTEXT_PARALLEL_GROUP = None
_GLOBAL_VOCAB_DATA_PARALLEL_GROUP = None
_GLOBAL_PIPELINE_PARALLEL_GROUP = None
_GLOBAL_TENSOR_MODEL_PARALLEL_SRC_RANK = None


def set_tensor_model_parallel_src_rank(rank):
    global _GLOBAL_TENSOR_MODEL_PARALLEL_SRC_RANK
    _GLOBAL_TENSOR_MODEL_PARALLEL_SRC_RANK = rank


def get_tensor_model_parallel_src_rank():
    global _GLOBAL_TENSOR_MODEL_PARALLEL_SRC_RANK
    return _GLOBAL_TENSOR_MODEL_PARALLEL_SRC_RANK


def set_vocab_tensor_parallel_group(group):
    global _GLOBAL_VOCAB_TENSOR_PARALLEL_GROUP
    _GLOBAL_VOCAB_TENSOR_PARALLEL_GROUP = group


def set_vocab_context_parallel_group(group):
    global _GLOBAL_VOCAB_CONTEXT_PARALLEL_GROUP
    _GLOBAL_VOCAB_CONTEXT_PARALLEL_GROUP = group


def set_vocab_data_parallel_group(group):
    global _GLOBAL_VOCAB_DATA_PARALLEL_GROUP
    _GLOBAL_VOCAB_DATA_PARALLEL_GROUP = group


def set_pipeline_parallel_group(group):
    global _GLOBAL_PIPELINE_PARALLEL_GROUP
    _GLOBAL_PIPELINE_PARALLEL_GROUP = group


def get_vocab_tensor_parallel_group():
    global _GLOBAL_VOCAB_TENSOR_PARALLEL_GROUP
    return _GLOBAL_VOCAB_TENSOR_PARALLEL_GROUP


def get_vocab_context_parallel_group():
    global _GLOBAL_VOCAB_CONTEXT_PARALLEL_GROUP
    return _GLOBAL_VOCAB_CONTEXT_PARALLEL_GROUP


def get_vocab_data_parallel_group():
    global _GLOBAL_VOCAB_DATA_PARALLEL_GROUP
    return _GLOBAL_VOCAB_DATA_PARALLEL_GROUP

def get_pipeline_parallel_group():
    global _GLOBAL_PIPELINE_PARALLEL_GROUP
    return _GLOBAL_PIPELINE_PARALLEL_GROUP


def get_vocab_tensor_parallel_world_size():
    global _GLOBAL_VOCAB_TENSOR_PARALLEL_GROUP
    assert _GLOBAL_VOCAB_TENSOR_PARALLEL_GROUP is not None, 'pipeline parallel group is not initialized'
    return torch.distributed.get_world_size(group=_GLOBAL_VOCAB_TENSOR_PARALLEL_GROUP)


def get_vocab_tensor_parallel_rank():
    global _GLOBAL_VOCAB_TENSOR_PARALLEL_GROUP
    assert _GLOBAL_VOCAB_TENSOR_PARALLEL_GROUP is not None, 'pipeline parallel group is not initialized'
    return torch.distributed.get_rank(group=_GLOBAL_VOCAB_TENSOR_PARALLEL_GROUP)


def get_vocab_context_parallel_world_size():
    global _GLOBAL_VOCAB_CONTEXT_PARALLEL_GROUP
    assert _GLOBAL_VOCAB_CONTEXT_PARALLEL_GROUP is not None, 'pipeline parallel group is not initialized'
    return torch.distributed.get_world_size(group=_GLOBAL_VOCAB_CONTEXT_PARALLEL_GROUP)


def get_vocab_context_parallel_rank():
    global _GLOBAL_VOCAB_CONTEXT_PARALLEL_GROUP
    assert _GLOBAL_VOCAB_CONTEXT_PARALLEL_GROUP is not None, 'pipeline parallel group is not initialized'
    return torch.distributed.get_rank(group=_GLOBAL_VOCAB_CONTEXT_PARALLEL_GROUP)


def get_vocab_data_parallel_world_size():
    global _GLOBAL_VOCAB_DATA_PARALLEL_GROUP
    assert _GLOBAL_VOCAB_DATA_PARALLEL_GROUP is not None, 'pipeline parallel group is not initialized'
    return torch.distributed.get_world_size(group=_GLOBAL_VOCAB_DATA_PARALLEL_GROUP)


def get_vocab_data_parallel_rank():
    global _GLOBAL_VOCAB_DATA_PARALLEL_GROUP
    assert _GLOBAL_VOCAB_DATA_PARALLEL_GROUP is not None, 'pipeline parallel group is not initialized'
    return torch.distributed.get_rank(group=_GLOBAL_VOCAB_DATA_PARALLEL_GROUP)


def get_pipeline_parallel_world_size():
    global _GLOBAL_PIPELINE_PARALLEL_GROUP
    assert _GLOBAL_PIPELINE_PARALLEL_GROUP is not None, 'pipeline parallel group is not initialized'
    return torch.distributed.get_world_size(group=_GLOBAL_PIPELINE_PARALLEL_GROUP)


def get_pipeline_parallel_rank():
    global _GLOBAL_PIPELINE_PARALLEL_GROUP
    assert _GLOBAL_PIPELINE_PARALLEL_GROUP is not None, 'pipeline parallel group is not initialized'
    return torch.distributed.get_rank(group=_GLOBAL_PIPELINE_PARALLEL_GROUP)


def get_parallel_world_size(group):
    return torch.distributed.get_world_size(group=group)


def get_parallel_rank(group):
    return torch.distributed.get_rank(group=group)


def is_pipeline_first_stage():
    return get_pipeline_parallel_rank() == 0


def is_pipeline_last_stage():
    return get_pipeline_parallel_rank() == get_pipeline_parallel_world_size() - 1