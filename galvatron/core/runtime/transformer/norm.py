from galvatron.core.runtime.args_schema import GalvatronModelArgs
import torch
from flash_attn.ops.rms_norm import RMSNorm
from flash_attn.ops.layer_norm import DropoutAddLayerNorm

class GalvatronNorm:
    """
    A conditional wrapper to initialize an instance of PyTorch's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__(cls, config: GalvatronModelArgs, hidden_size: int, eps: float = 1e-5):
        if config.normalization == "LayerNorm":
            instance = DropoutAddLayerNorm(
                hidden_size=hidden_size,
                eps=eps,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        elif config.normalization == "RMSNorm":
            instance = RMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        else:
            raise Exception('Only LayerNorm and RMSNorm are curently supported')

        return instance