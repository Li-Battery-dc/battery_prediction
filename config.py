"""
Configuration file for battery cycle life prediction project
"""
import os

class Config:
    """Configuration class for all project parameters"""
    
    # Data paths
    DATA_DIR = './Data'
    BATCH1_PATH = os.path.join(DATA_DIR, 'batch1.pkl')
    BATCH2_PATH = os.path.join(DATA_DIR, 'batch2.pkl')
    BATCH3_PATH = os.path.join(DATA_DIR, 'batch3.pkl')
    
    # Feature extraction parameters
    FEATURE_CYCLE_10 = 10
    FEATURE_CYCLE_100 = 100
    FEATURE_CYCLE_START = 2
    FEATURE_CYCLE_END = 100
    AVG_CHARGE_CYCLES = 5
    EPSILON = 1e-8  # To avoid log(0)
    
    # Model parameters
    ALPHA_RANGE = (0.01, 1.0, 0.01)  # start, stop, step
    LAMBDA_RANGE = (0.01, 1.0, 0.01)  # start, stop, step
    CROSS_VALIDATION_FOLDS = 4
    MAX_ITER = 10000
    NUM_VALIDATION_MODELS = 10
    
    # Training parameters
    RANDOM_STATE = 42
    
    @classmethod
    def get_alpha_vec(cls):
        """Get alpha vector for hyperparameter search"""
        import numpy as np
        start, stop, step = cls.ALPHA_RANGE
        return np.arange(start, stop, step)
    
    @classmethod
    def get_lambda_vec(cls):
        """Get lambda vector for hyperparameter search"""
        import numpy as np
        start, stop, step = cls.LAMBDA_RANGE
        return np.arange(start, stop, step)