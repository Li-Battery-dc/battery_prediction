"""
Elastic Net model implementation for battery cycle life prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from models.base_model import BaseModel
from config import Config


class ElasticNetModel(BaseModel):
    """Elastic Net regression model for battery cycle life prediction"""
    
    def __init__(self, config: Config = None):
        """Initialize Elastic Net model
        
        Args:
            config: Configuration object containing model parameters
        """
        super().__init__(config)
        self.config = config if config else Config()
        self.best_alpha = None
        self.best_lambda = None
        self.best_coefficients = None
        self.best_intercept = None
        self.training_history = []
        
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, 
            X_val: pd.DataFrame = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the Elastic Net model with hyperparameter search
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary containing training information
        """
        print("Training Elastic Net model with hyperparameter search...")
        
        # Get hyperparameter ranges
        alpha_vec = self.config.get_alpha_vec()
        lambda_vec = self.config.get_lambda_vec()
        
        # Store results for all alpha values
        rmse_list = []
        min_lambda_list = []
        coef_list = []
        intercept_list = []
        
        # Hyperparameter search
        for alpha in alpha_vec:
            # Use ElasticNetCV for automatic lambda selection with cross-validation
            model = ElasticNetCV(
                l1_ratio=alpha, 
                alphas=lambda_vec, 
                cv=self.config.CROSS_VALIDATION_FOLDS,
                max_iter=self.config.MAX_ITER,
                random_state=self.config.RANDOM_STATE
            )
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Make predictions on training set
            y_pred_train = model.predict(X_train)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            
            # Store results
            rmse_list.append(rmse)
            min_lambda_list.append(model.alpha_)  # Best lambda for this alpha
            coef_list.append(model.coef_)
            intercept_list.append(model.intercept_)
        
        # Convert to numpy arrays
        rmse_list = np.array(rmse_list)
        min_lambda_list = np.array(min_lambda_list)
        coef_list = np.array(coef_list)
        intercept_list = np.array(intercept_list)
        
        # Select top models for validation
        num_val_models = min(self.config.NUM_VALIDATION_MODELS, len(rmse_list))
        best_indices = np.argsort(rmse_list)[:num_val_models]
        
        # Use validation data to select final hyperparameters
        if X_val is not None and y_val is not None:
            val_rmse_list = []
            for idx in best_indices:
                y_pred_val = np.dot(X_val, coef_list[idx]) + intercept_list[idx]
                rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
                val_rmse_list.append(rmse_val)
            
            # Select best model based on validation RMSE
            best_val_idx = best_indices[np.argmin(val_rmse_list)]
        else:
            # If no validation data, use best training RMSE
            best_val_idx = best_indices[0]
        
        # Store best hyperparameters
        self.best_alpha = alpha_vec[best_val_idx]
        self.best_lambda = min_lambda_list[best_val_idx]
        self.best_coefficients = coef_list[best_val_idx]
        self.best_intercept = intercept_list[best_val_idx]
        
        # Create final model with best hyperparameters
        self.model = ElasticNetCV(
            l1_ratio=self.best_alpha,
            alphas=[self.best_lambda],
            cv=self.config.CROSS_VALIDATION_FOLDS,
            max_iter=self.config.MAX_ITER,
            random_state=self.config.RANDOM_STATE
        )
        self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        
        # Store training information
        training_info = {
            'best_alpha': self.best_alpha,
            'best_lambda': self.best_lambda,
            'coefficients': self.best_coefficients,
            'intercept': self.best_intercept,
            'num_features': len(self.best_coefficients),
            'train_rmse': rmse_list[best_val_idx]
        }
        
        if X_val is not None and y_val is not None:
            training_info['val_rmse'] = val_rmse_list[np.argmin(val_rmse_list)]
        
        print(f"Best hyperparameters: alpha={self.best_alpha:.4f}, lambda={self.best_lambda:.4f}")
        print(f"Training complete!")
        
        return training_info
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model
        
        Args:
            X: Feature data
            
        Returns:
            Predicted cycle life values
            
        Raises:
            RuntimeError: If model is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Use the stored coefficients for prediction
        return np.dot(X, self.best_coefficients) + self.best_intercept
    
    def get_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        """Get feature importance based on model coefficients
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.best_coefficients))]
        
        # Use absolute values of coefficients as importance
        importance = np.abs(self.best_coefficients)
        
        return dict(zip(feature_names, importance))
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters
        
        Returns:
            Dictionary containing model parameters
        """
        if not self.is_fitted:
            return {'fitted': False}
        
        return {
            'fitted': True,
            'alpha': self.best_alpha,
            'lambda': self.best_lambda,
            'intercept': self.best_intercept,
            'coefficients': self.best_coefficients.tolist(),
            'num_features': len(self.best_coefficients)
        }