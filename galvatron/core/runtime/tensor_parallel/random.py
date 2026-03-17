# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib
import logging
from typing import Union

import torch
from torch import _C
from torch.cuda import _lazy_call, _lazy_init
from torch.cuda import device as device_ctx_manager
from torch.utils.checkpoint import detach_variable


# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'
_EXPERT_PARALLEL_RNG_TRACKER_NAME = 'expert-parallel-rng'
_DATA_PARALLEL_RNG_TRACKER_NAME = 'data-parallel-rng'


def _get_cuda_rng_state(
    device: Union[int, str, torch.device] = "cuda", clone: bool = False, graph_safe: bool = False
) -> torch.Tensor:
    """Return the random number generator state of the specified GPU.

    Arguments:
        device (int): The gpu to retrieve the rng state
        clone (bool): Whether to also clone the retrieved RNG state
        graph_safe (bool): Get the rng state in a graph safe manner.

    This function is adapted from torch.cuda.random.get_rng_state()"""

    # if not using cuda graphs, just use the builtin pytorch function
    if not graph_safe:
        return torch.cuda.random.get_rng_state(device=device)

    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()

    default_generator = torch.cuda.default_generators[idx]
    if clone:
        return default_generator.clone_state()
    return default_generator.graphsafe_get_state()


def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1, graph_safe: bool = False):
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
        device (int): The gpu to retrieve the rng state
        graph_safe (bool): Set the rng state in a graph safe manner.

    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]

            # if graph capturing, set the rng state in a cudagraphable way
            if graph_safe:
                default_generator.graphsafe_set_state(new_state)
            else:
                default_generator.set_state(new_state)

    _lazy_call(cb)


def get_expert_parallel_rng_tracker_name(group=None):
    """Get the expert parallel rng tracker name"""
    global _EXPERT_PARALLEL_RNG_TRACKER_NAME
    if group == None:
        return _EXPERT_PARALLEL_RNG_TRACKER_NAME
    else:
        return _EXPERT_PARALLEL_RNG_TRACKER_NAME + "-%d"%torch.distributed.get_world_size(group)

def get_tensor_parallel_rng_tracker_name(group=None):
    """Get the tensor parallel rng tracker name"""
    global _MODEL_PARALLEL_RNG_TRACKER_NAME
    if group == None:
        return _MODEL_PARALLEL_RNG_TRACKER_NAME
    else:
        return _MODEL_PARALLEL_RNG_TRACKER_NAME + "-%d"%torch.distributed.get_world_size(group)



def get_data_parallel_rng_tracker_name():
    """Get the data parallel rng tracker name"""
    global _DATA_PARALLEL_RNG_TRACKER_NAME
    return _DATA_PARALLEL_RNG_TRACKER_NAME


class CudaRNGStatesTracker:
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self, use_cudagraphable_rng=False, is_inference_rng_tracker=False):
        self.reset()
        self.use_cudagraphable_rng = use_cudagraphable_rng
        self.is_inference_rng_tracker = is_inference_rng_tracker

        if self.use_cudagraphable_rng:
            assert (
                hasattr(torch.cuda.CUDAGraph, "register_generator_state")
                and hasattr(torch.Generator, "graphsafe_set_state")
                and hasattr(torch.Generator, "graphsafe_get_state")
                and hasattr(torch.Generator, "clone_state")
            ), "Tried using cudagraphs with RNG, however not detected in pytorch!"

    def is_initialized(self):
        """Checks if the internal RNG state has been set wirth set_states()."""
        return self._is_initialized

    def reset(self):
        """Set to the initial state (no tracker)."""

        # Track if initialized.
        self._is_initialized = False

        # Map from a string name to the cuda rng state.
        self.states_ = {}

        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self._is_initialized = True
        self.states_ = states
    
    def check(self, name):
        if name not in self.states_:
            return True
        return False

    def add(self, name, seed):
        """Track the rng state."""
        self._is_initialized = True
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))

        # If available, create the state in a graph safe manner
        if self.use_cudagraphable_rng:
            new_state = _get_cuda_rng_state(clone=True, graph_safe=True)
            new_state.manual_seed(seed)
            self.states_[name] = new_state
        else:
            # Get the current rng state.
            orig_rng_state = torch.cuda.get_rng_state()
            # Set the new state and store it.
            torch.cuda.manual_seed(seed)
            self.states_[name] = torch.cuda.get_rng_state()
            # Reset rng state to what it was.
            _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        # Store current rng state.
        orig_cuda_rng_state = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name], graph_safe=self.use_cudagraphable_rng)
        # Record cpu RNG state
        cpu_rng_state = torch.get_rng_state()
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Throw a warning if cpu RNG state changed
            if not torch.all(cpu_rng_state == torch.get_rng_state()).item():
                logging.getLogger(__name__).warning('CPU RNG state changed within GPU RNG context')
            # Update the current rng state for later use.
            self.states_[name] = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state, graph_safe=self.use_cudagraphable_rng)


# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = None
_CUDA_RNG_STATE_TRACKER_INITIALIZED = False


def initialize_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Create the RNG tracker. 'use_te_rng_tracker' determines whether to use
    Megatron or TransformerEngine's implementation.
    In particular, TransformerEngine's implementation is cudagraphable and supports FP8.
    """
    global _CUDA_RNG_STATE_TRACKER
    global _CUDA_RNG_STATE_TRACKER_INITIALIZED
    if _CUDA_RNG_STATE_TRACKER_INITIALIZED:
        return

    # Get the base tracker class
    base_tracker = CudaRNGStatesTracker
    tracker_kwargs = {
        "use_cudagraphable_rng": use_cudagraphable_rng,
        "is_inference_rng_tracker": inference_rng_tracker,
    }

    if inference_rng_tracker:

        class InferenceCudaRNGStatesTracker(base_tracker):
            """RNG tracker for inference."""

            def add(self, name, seed):
                """Mirrors the interface from the training RNG tracker."""
                pass

            def set_states(self, states):
                """Mirrors the interface from the training RNG tracker."""
                pass

            def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
                """Mirrors the interface from the training RNG tracker."""
                return contextlib.nullcontext()

        tracker_class = InferenceCudaRNGStatesTracker
    else:
        tracker_class = base_tracker

    _CUDA_RNG_STATE_TRACKER = tracker_class(**tracker_kwargs)
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True


def set_seed_with_group(
    tp_groups: list = None,  
    tp_and_ep_groups: list = None, 
    seed: int = 1234,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    ):
    # 917 is just for fun and any POSITIVE value will work.
    data_parallel_seed = seed
    offset = seed + 917
    initialize_rng_tracker(te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)
    _CUDA_RNG_STATE_TRACKER.reset()

    torch.cuda.manual_seed(data_parallel_seed)
    _CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)

    for group in tp_groups:
        rank = torch.distributed.get_rank(group.group)
        world_size = torch.distributed.get_world_size(group.group)
        if _CUDA_RNG_STATE_TRACKER.check(_MODEL_PARALLEL_RNG_TRACKER_NAME + "-%d"%world_size):
            _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME + "-%d"%world_size, offset + rank)
            offset += 100

    if tp_and_ep_groups is not None:
        for group in tp_and_ep_groups:
            rank = torch.distributed.get_rank(group.group)
            world_size = torch.distributed.get_world_size(group.group)
            if _CUDA_RNG_STATE_TRACKER.check(_EXPERT_PARALLEL_RNG_TRACKER_NAME + "-%d"%world_size):
                _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME + "-%d"%world_size, offset + rank)
                offset += 100

    # Add defalut state.
    # _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, offset + get_tensor_model_parallel_rank())

    # expert_parallel_seed = (
    #     offset + 1024 + 100 * get_expert_model_parallel_rank() + get_expert_tensor_parallel_rank()
    # )
    # _CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)

def get_cuda_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Get cuda rng tracker."""
    initialize_rng_tracker(use_te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng)
    return _CUDA_RNG_STATE_TRACKER