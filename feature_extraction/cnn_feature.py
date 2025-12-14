"""
CNN Feature extraction module for battery cycle life prediction
Extracts raw time-series features for deep learning models
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler
from config import Config

class CNNFeatureExtractor():
    """CNN feature extractor for raw time-series data"""
    
    def __init__(self, config: Config = None):
        """Initialize CNN feature extractor
        
        Args:
            config: Configuration object
        """
        self.config = config if config else Config()
        self.normalize = self.config.NORMALIZE_FEATURES
        self.log_transform_target = self.config.LOG_TRANSFORM_TARGET
        self.feature_cycle_end = self.config.FEATURE_CYCLE_END
        self.scaler = StandardScaler() if self.normalize else None
        self.is_fitted = False
        self.feature_names = config.CNN_FEATURE_NAMES
        self.num_points = self.config.CNN_NUM_POINTS_PER_CYCLE
    
    def extract_features(self, original_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract raw time-series features from battery data
        
        Args:
            original_data: Dictionary containing battery measurement data for multiple cells
            
        Returns:
            Tuple of (feature_array, cycle_life_array)
            - feature_array: shape (N, sequence_length) containing raw time-series data
            - cycle_life_array: shape (N,) containing cycle life targets
        """
        N = len(original_data)
        X_data = []
        cycle_lives = []
        
        for cell_key, cell_data in original_data.items():
            battery_data = []
            # 提取原始数据信息
            for cycle in range(1, self.feature_cycle_end + 1):
                cycle_str = str(cycle)
                cycle_data = cell_data['cycles'][cycle_str]
                cycle_features = []
                for feature in self.feature_names:
                    original_data = np.array(cycle_data[feature])
                    sampled_data = np.interp(
                        np.linspace(0, len(original_data) - 1, self.num_points),
                        np.arange(len(original_data)),
                        original_data
                    )
                    cycle_features.append(sampled_data)
                
                battery_data.append(np.array(cycle_features))

            X_data.append(np.concatenate(battery_data, axis=-1))  # [channels, num_cycles * num_points]
            cycle_lives.append(cell_data['cycle_life'])
        
        # 将列表转换为numpy数组
        X = np.array(X_data)  # Shape: (N, sequence_length)
        y = np.array(cycle_lives).reshape(-1)  # Shape: (N,)
        
        # Apply log transformation to target if configured
        if self.log_transform_target:
            y = np.log10(y)
        
        # Apply normalization to features if configured
        if self.normalize and not self.is_fitted:
            # For time-series data, normalize across the feature dimension
            original_shape = X.shape
            X_reshaped = X.reshape(-1, 1)  # Reshape to (N * sequence_length, 1)
            self.scaler.fit(X_reshaped)
            X = self.scaler.transform(X_reshaped).reshape(original_shape)
            self.is_fitted = True
        elif self.normalize and self.is_fitted:
            original_shape = X.shape
            X_reshaped = X.reshape(-1, 1)
            X = self.scaler.transform(X_reshaped).reshape(original_shape)
        else:
            self.is_fitted = True
        
        return X, y
    
    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform the target variable (from log scale back to original scale)
        
        Args:
            y: Target values in transformed scale
            
        Returns:
            Target values in original scale
        """
        if self.log_transform_target:
            return np.power(10, y)
        return y
    
    def get_feature_names(self):
        """Get feature names
        
        Returns:
            List of feature names
        """
        return self.feature_names
