from .base import BaseFeatureEngineer
from .fsrs_engineer import FSRSFeatureEngineer
from src.prepare.prepare_config import ModelName, Config
from typing import Type


FEATURE_ENGINEER_REGISTRY: dict[ModelName, Type[BaseFeatureEngineer]] = {
    "FSRS-7": FSRSFeatureEngineer,
}


def create_feature_engineer(config: Config) -> BaseFeatureEngineer:
    """
    Factory function to create the appropriate feature engineer based on model name from PrepareConfig

    Args:
        PrepareConfig: Configuration object containing model_name and other settings

    Returns:
        Appropriate feature engineer instance

    Raises:
        ValueError: If config.model_name is not supported
    """
    model_name = config.model_name

    # Create and return the appropriate feature engineer
    feature_engineer_cls = FEATURE_ENGINEER_REGISTRY[model_name]
    return feature_engineer_cls(config)


def get_supported_models() -> tuple[str, ...]:
    """
    Get list of all supported model names

    Returns:
        List of supported model names
    """
    return tuple(FEATURE_ENGINEER_REGISTRY.keys())
