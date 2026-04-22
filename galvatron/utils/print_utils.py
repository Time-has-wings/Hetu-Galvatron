import torch
import json
import pydantic
from dataclasses import dataclass

@dataclass
class ColorSet:
    YELLOW = "\033[33m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m" 
    RESET = "\033[0m"


def print_args_rank0(args: pydantic.BaseModel, title: str = "arguments"):
    """Print Pydantic args as indented JSON. Only rank 0 prints."""
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    d = args.model_dump()
    s = json.dumps(d, indent=2, default=str)
    print(f"\n=== {title} ===\n{s}\n", flush=True)


def print_single_rank(message, rank=0):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank:
            print(f'[rank{rank}] {message}', flush=True)
    else:
        print(f'[cpu] {message}', flush=True)