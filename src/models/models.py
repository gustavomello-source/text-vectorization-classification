"""
ML Models module - imports from the models package for backward compatibility
"""

# Import all classes from the models package to maintain backward compatibility
from src.models import (
    LinearSVCStrategy,
    LogisticRegressionStrategy,
    MLModel,
    ModelStrategy,
    RandomForestStrategy,
)

# Make all classes available at module level
__all__ = [
    "ModelStrategy",
    "LinearSVCStrategy",
    "LogisticRegressionStrategy",
    "RandomForestStrategy",
    "MLModel",
]
