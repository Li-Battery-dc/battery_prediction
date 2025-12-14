"""
Base model class for battery cycle life prediction
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error


class BaseModel(ABC):
    """Abstract base class for all prediction models"""
    
    def __init__(self, config=None, model_config=None):
        """Initialize model with configuration
        
        Args:
            config: General configuration object
            model_config: Model-specific configuration object
        """
        self.config = config
        self.model_config = model_config
        self.is_fitted = False
        self.model = None
        self.result_dir = None  # Will be set when saving results
        
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
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray, feature_extractor=None) -> Dict[str, float]:
        """Evaluate model performance
        
        Args:
            X: Feature data
            y: True target values (may be transformed, e.g., log scale)
            feature_extractor: Optional feature extractor for inverse transforming predictions and targets
            
        Returns:
            Dictionary containing evaluation metrics (always computed on original scale)
        """
        y_pred = self.predict(X)
        
        # Apply inverse transformation if feature extractor is provided
        if feature_extractor is not None and feature_extractor.log_transform_target:
            y_pred = feature_extractor.inverse_transform_target(y_pred)
            y = feature_extractor.inverse_transform_target(y)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mpe': np.mean(np.abs(y - y_pred) / np.abs(y)) * 100,
            'r2': 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        }
        
        return metrics
    
    def plot_feature_importance(self, feature_names: list = None, top_n: int = None, save: bool = True):
        """Plot feature importance
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display (None for all)
            save: Whether to save the plot (requires result_dir to be set)
        """
        if not hasattr(self, 'get_feature_importance'):
            print("   - Warning: Model does not support feature importance")
            return
        
        try:
            # Get feature importance
            feature_importance = self.get_feature_importance(feature_names)
            
            if not feature_importance:
                print("   - Warning: No feature importance available")
                return
            
            # Sort by importance
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Select top N features if specified
            if top_n:
                sorted_importance = sorted_importance[:top_n]
            
            features = [item[0] for item in sorted_importance]
            importances = [item[1] for item in sorted_importance]
            
            # Create plot
            plt.figure(figsize=(10, max(6, len(features) * 0.4)))
            plt.barh(range(len(features)), importances, align='center')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Importance', fontsize=12, fontweight='bold')
            plt.ylabel('Features', fontsize=12, fontweight='bold')
            plt.title(f'Feature Importance - {self.__class__.__name__}', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()  # Highest importance on top
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            # Save if requested and result_dir is set
            if save and self.result_dir:
                plot_path = os.path.join(self.result_dir, 'feature_importance.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"   - Feature importance plot saved to: {plot_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"   - Warning: Failed to plot feature importance: {e}")
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str = "Prediction vs Actual", save: bool = True):
        """Plot predicted vs actual cycle life
        
        Args:
            y_true: True cycle life values
            y_pred: Predicted cycle life values
            title: Plot title
            save: Whether to save the plot (requires result_dir to be set)
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate metrics for display
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs(y_true - y_pred) / y_true) * 100
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        # Plot scatter points
        plt.scatter(y_true, y_pred, alpha=0.6, s=100, edgecolors='k', linewidths=0.5)
        
        # Plot perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Add labels and title
        plt.xlabel('Actual Cycle Life', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Cycle Life', fontsize=12, fontweight='bold')
        plt.title(f'{title}\nRMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, R²: {r2:.3f}', 
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save if requested and result_dir is set
        if save and self.result_dir:
            plot_path = os.path.join(self.result_dir, 'prediction_plot.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   - Plot saved to: {plot_path}")
        
        plt.close()
    
    def save_results(self, metrics: Dict[str, float], training_info: Dict[str, Any] = None,
                     feature_names: list = None, y_true: np.ndarray = None, y_pred: np.ndarray = None,
                     train_metrics: Dict[str, float] = None):
        """Save model results to a timestamped directory
        
        Args:
            metrics: Dictionary of evaluation metrics (test set)
            training_info: Dictionary of training information (optional)
            feature_names: List of feature names (optional)
            y_true: True target values for visualization (optional)
            y_pred: Predicted values for visualization (optional)
            train_metrics: Dictionary of training set evaluation metrics (optional)
        """
        if not self.config:
            print("   - Warning: No config provided, cannot save results")
            return
        
        # Create timestamped result directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = self.__class__.__name__.replace('Model', '').lower()
        self.result_dir = os.path.join(self.config.RESULT_DIR, f'{model_name}_{timestamp}')
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Prepare result dictionary
        results = {
            'model_type': self.__class__.__name__,
            'timestamp': timestamp,
            'test_metrics': metrics
        }
        
        if train_metrics:
            results['train_metrics'] = train_metrics
        
        if training_info:
            results['training_info'] = training_info
        
        if feature_names:
            results['feature_names'] = feature_names
            
            # Get and save feature importance if available
            if hasattr(self, 'get_feature_importance'):
                try:
                    feature_importance = self.get_feature_importance(feature_names)
                    sorted_importance = sorted(feature_importance.items(), 
                                              key=lambda x: x[1], reverse=True)
                    results['feature_importance'] = dict(sorted_importance)
                except:
                    pass
        
        # Save model parameters if available
        if hasattr(self, 'get_model_params'):
            try:
                results['model_params'] = self.get_model_params()
            except:
                pass
        
        # Save results to JSON file
        results_path = os.path.join(self.result_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        print(f"   - Results saved to: {self.result_dir}")
        print(f"   - Metrics file: {results_path}")
        
        # Generate and save prediction plot if data provided
        if y_true is not None and y_pred is not None:
            self.plot_predictions(y_true, y_pred, "Test Set", save=True)
        else:
            print("   - Note: No prediction data provided, skipping visualization")
        
        # Generate and save feature importance plot if feature names provided
        if feature_names:
            self.plot_feature_importance(feature_names, save=True)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'config': self.config.__dict__ if self.config else None,
            'model_config': self.model_config.__dict__ if self.model_config else None
        }