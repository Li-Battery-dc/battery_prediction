"""
Feature extraction module for battery cycle life prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler
from config import Config

class StandardFeatureExtractor():
    """Standard feature extractor based on the original research paper"""
    
    def __init__(self, config: Config = None):
        """Initialize standard feature extractor
        
        Args:
            config: Configuration object containing feature extraction parameters,
                   including NORMALIZE_FEATURES and LOG_TRANSFORM_TARGET settings
        """
        self.config = config if config else Config()
        self.normalize = self.config.NORMALIZE_FEATURES
        self.log_transform_target = self.config.LOG_TRANSFORM_TARGET
        self.scaler = StandardScaler() if self.normalize else None
        self.is_fitted = False
        self.feature_names = [
            'DeltaQ_var', 'DeltaQ_min', 'CapFadeCycle2Slope', 
            'CapFadeCycle2Intercept', 'Qd2', 'AvgChargeTime',
            'IntegralTemp', 'MinIR', 'IRDiff2And100'
        ]
    
    def extract_features(self, battery_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract standard features from battery data
        
        Args:
            battery_data: Dictionary containing battery measurement data for multiple cells
            
        Returns:
            Tuple of (feature_array, cycle_life_array)
            - feature_array: shape (N, num_features)
            - cycle_life_array: shape (N,)
        """
        N = len(battery_data)
        
        # Initialize feature arrays
        y = np.zeros(N)
        DeltaQ_var = np.zeros(N)
        DeltaQ_min = np.zeros(N)
        CapFadeCycle2Slope = np.zeros(N)
        CapFadeCycle2Intercept = np.zeros(N)
        Qd2 = np.zeros(N)
        AvgChargeTime = np.zeros(N)
        IntegralTemp = np.zeros(N)
        MinIR = np.zeros(N)
        IRDiff2And100 = np.zeros(N)
        
        for i, key in enumerate(battery_data.keys()):
            cell = battery_data[key]
            
            # Extract cycle life (target variable)
            y[i] = cell['cycle_life'].reshape(-1)[0]
            
            # 1. DeltaQ_var and DeltaQ_min: variance and minimum of Q_100 - Q_10
            Qd_100 = cell['cycles'][str(self.config.FEATURE_CYCLE_100)]['Qdlin']
            Qd_10 = cell['cycles'][str(self.config.FEATURE_CYCLE_10)]['Qdlin']
            delta_Q = Qd_100 - Qd_10
            valid_deltaQ = -delta_Q[delta_Q < 0]
            valid_deltaQ = valid_deltaQ + self.config.EPSILON  # To avoid log(0)
            DeltaQ_var[i] = np.log(np.var(valid_deltaQ))
            DeltaQ_min[i] = np.log(np.min(valid_deltaQ))
            
            # 2. CapFadeCycle2Slope and CapFadeCycle2Intercept: linear fit to capacity fade curve cycles 2-100
            cycles_range = range(self.config.FEATURE_CYCLE_START, self.config.FEATURE_CYCLE_END + 1)
            cycles_2_to_100 = list(cycles_range)
            Qd_2_to_100 = cell['summary']['QD'][self.config.FEATURE_CYCLE_START:self.config.FEATURE_CYCLE_END + 1]
            
            # Linear fit: Qd = slope * cycle + intercept
            coeffs = np.polyfit(cycles_2_to_100, Qd_2_to_100, 1)
            CapFadeCycle2Slope[i] = coeffs[0]
            CapFadeCycle2Intercept[i] = coeffs[1]
            
            # 3. Qd2: discharge capacity at cycle 2
            Qd2[i] = cell['summary']['QD'][self.config.FEATURE_CYCLE_START]
            
            # 4. AvgChargeTime: average charge time over first 5 cycles
            charge_end_idx = self.config.AVG_CHARGE_CYCLES + 1
            AvgChargeTime[i] = np.mean(cell['summary']['chargetime'][1:charge_end_idx])
            
            # 5. IntegralTemp: integral of temperature over time from cycles 2 to 100
            tempIntT = 0
            for jCycle in cycles_range:
                if str(jCycle) in cell['cycles']:
                    cycle_data = cell['cycles'][str(jCycle)]
                    if 't' in cycle_data and 'T' in cycle_data:
                        tempIntT += np.trapezoid(cycle_data['T'], cycle_data['t'])
            IntegralTemp[i] = tempIntT
            
            # 6. MinIR and IRDiff2And100: minimum internal resistance cycles 2-100, and difference between cycles 2 and 100
            IR_2 = cell['summary']['IR'][self.config.FEATURE_CYCLE_START]
            IR_100 = cell['summary']['IR'][self.config.FEATURE_CYCLE_100]
            IR_data = cell['summary']['IR'][self.config.FEATURE_CYCLE_START:self.config.FEATURE_CYCLE_END + 1]
            valid_IR = IR_data[IR_data > 0]
            MinIR[i] = np.min(valid_IR) if len(valid_IR) > 0 else 0
            IRDiff2And100[i] = IR_2 - IR_100
        
        # Stack features into numpy array
        X = np.column_stack([
            DeltaQ_var,
            DeltaQ_min,
            CapFadeCycle2Slope,
            CapFadeCycle2Intercept,
            Qd2,
            AvgChargeTime,
            IntegralTemp,
            MinIR,
            IRDiff2And100
        ])  # Shape: (N, 9)
        
        # Apply standardization if enabled
        if self.normalize:
            if not self.is_fitted:
                # Fit and transform on training data
                X = self.scaler.fit_transform(X)
                self.is_fitted = True
            else:
                # Only transform on validation/test data
                X = self.scaler.transform(X)
        
        # Apply log transformation to target if enabled
        if self.log_transform_target:
            y = np.log10(y)
        
        return X, y
    
    def get_feature_names(self) -> list:
        """Get names of extracted features
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy()
    
    def inverse_transform_target(self, y_transformed: np.ndarray) -> np.ndarray:
        """Apply inverse transformation to target variable
        
        Args:
            y_transformed: Transformed target values (log10 if log_transform_target=True)
            
        Returns:
            Original scale target values
        """
        if self.log_transform_target:
            return 10 ** y_transformed
        return y_transformed
    
    def extract_single_cell_features(self, cell_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from a single cell
        
        Args:
            cell_data: Dictionary containing single cell measurement data
            
        Returns:
            Dictionary of feature values
        """
        # Create a temporary batch with single cell
        temp_batch = {'temp_cell': cell_data}
        X, y = self.extract_features(temp_batch)
        
        # Convert to dictionary
        features = {name: X[0, i] for i, name in enumerate(self.feature_names)}
        features['cycle_life'] = y[0]
        
        return features