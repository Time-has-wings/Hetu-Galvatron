"""Pydantic models for Galvatron tool arguments (checkpoint conversion). Merged view: galvatron.core.args_schema."""
from pydantic import BaseModel, Field


class CheckpointConvertH2GArgs(BaseModel):
    """HuggingFace -> Galvatron checkpoint conversion."""

    model_type: str = Field(..., description="Model type")
    input_checkpoint: str = Field(..., description="Input checkpoint path")
    output_dir: str = Field(..., description="Output directory")


class CheckpointConvertG2HArgs(BaseModel):
    """Galvatron -> HuggingFace checkpoint conversion."""

    load_iteration: int = Field(..., description="Iteration to load.")
    input_checkpoint: str = Field(..., description="Path to the input Galvatron checkpoint.")
    output_dir: str = Field(..., description="Path to save the HuggingFace checkpoint.")
    model_config: str = Field(..., description="Path to model config file.")
    model_type: str = Field(..., description="Model type.")
