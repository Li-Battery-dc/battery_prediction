"""
Base model class for battery cycle life prediction
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error


class BaseModel(ABC):
    """Abstract base class for all prediction models"""
    
    def __init__(self, config=None):
        """Initialize model with configuration
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        self.is_fitted = False
        self.model = None
        
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, 
            X_val: pd.DataFrame = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary containing training information
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions
        
        Args:
            X: Feature data
            
        Returns:
            Predicted values
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance
        
        Args:
            X: Feature data
            y: True target values
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': np.mean(np.abs(y - y_pred)),
            'mape': np.mean(np.abs(y - y_pred) / y) * 100,
            'r2': 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        }
        
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'config': self.config.__dict__ if self.config else None
        }