"""
Abstract base class for ML model strategies
"""
from abc import ABC, abstractmethod
from pandas import DataFrame

class ModelStrategy(ABC):
    """Abstract base class for ML model strategies"""
    
    @abstractmethod
    def train(self, X_train: DataFrame, y_train: DataFrame, params: dict):
        """Train the model and return the fitted model"""
        pass
    
    @abstractmethod
    def predict(self, model, X_test: DataFrame):
        """Make predictions using the fitted model"""
        pass