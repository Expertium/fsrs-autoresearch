from typing import Any, Dict, Union, List, Optional, cast
import torch
from src.prepare.prepare_config import ModelName, Config

from src.models import *


MODEL_REGISTRY: dict[ModelName, Any] = {
    "FSRS-7": FSRS7,
}

def create_model(
    config: Config,
    model_params: Optional[Union[List[float], Dict[str, torch.Tensor], float]] = None,
) -> TrainableModel:
    """
    Creates and returns an instance of the specified model.

    Args:
        PrepareConfig: The application configuration object.
        model_params: Optional parameters for model initialization.
                      - List[float]: For FSRS-like models' 'w' parameter.
                      - Dict[str, Tensor]: For neural models' state_dict.
                      - float: For ConstantModel's value.
                      If None, default initialization is used.

    Returns:
        An initialized nn.Module instance, moved to the device specified in config.

    Raises:
        ValueError: If model_name is not supported by the factory.
        TypeError: If model_params are of an incorrect type for the model.
    """
    model_name = config.model_name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not supported by the model factory. "
            f"Supported models: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_name]

    # Common arguments for all model constructors
    constructor_kwargs = {"config": config}

    if hasattr(
        model_cls, "init_w"
    ):  # FSRS-like models, HLR, ACT_R, DASH, SM2Trainable, Anki
        if model_params is not None:
            if not isinstance(model_params, list) or not all(
                isinstance(p, (float, int)) for p in model_params
            ):
                raise TypeError(
                    f"For {model_name}, model_params must be a List[float] or None, got {type(model_params)}"
                )
            constructor_kwargs["w"] = model_params  # type: ignore
        # Don't add 'w' to constructor_kwargs when model_params is None
        # This allows models to use their default parameter values
        instance = model_cls(**constructor_kwargs)  # type: ignore

    else:
        instance = model_cls(**constructor_kwargs)  # type: ignore

    return cast(TrainableModel, instance.to(config.device))
