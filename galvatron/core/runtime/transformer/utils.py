import warnings


def deprecate_inference_params(inference_context, inference_params):
    """Print warning for deprecated `inference_params`."""
    if inference_context is None and inference_params is not None:
        warnings.warn(
            "`inference_params` renamed to `inference_context`, and will be "
            "removed in `megatron-core` 0.13."
        )
        return inference_params
    return inference_context
