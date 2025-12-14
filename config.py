"""
Configuration file for battery cycle life prediction project
"""
import os
import numpy as np


class Config:
    """Configuration class for all project parameters"""
    
    # Data paths
    DATA_DIR = './Data'
    BATCH1_PATH = os.path.join(DATA_DIR, 'batch1.pkl')
    BATCH2_PATH = os.path.join(DATA_DIR, 'batch2.pkl')
    BATCH3_PATH = os.path.join(DATA_DIR, 'batch3.pkl')
    
    # Result directory
    RESULT_DIR = './results'
    
    # Feature extraction parameters
    FEATURE_CYCLE_10 = 10
    FEATURE_CYCLE_100 = 100
    FEATURE_CYCLE_START = 2
    FEATURE_CYCLE_END = 100
    AVG_CHARGE_CYCLES = 5
    EPSILON = 1e-8  # To avoid log(0)
    
    # Training parameters
    RANDOM_STATE = 42


class ElasticNetConfig:
    """Configuration class for Elastic Net model parameters"""
    
    # Hyperparameter search ranges
    ALPHA_RANGE = (0.01, 1.0, 0.01)  # start, stop, step (L1/L2 ratio)
    LAMBDA_RANGE = (0.01, 1.0, 0.01)  # start, stop, step (regularization strength)
    
    # Cross-validation and training
    CROSS_VALIDATION_FOLDS = 4
    MAX_ITER = 10000
    NUM_VALIDATION_MODELS = 10
    
    @classmethod
    def get_alpha_vec(cls):
        """Get alpha vector for hyperparameter search"""
        start, stop, step = cls.ALPHA_RANGE
        return np.arange(start, stop, step)
    
    @classmethod
    def get_lambda_vec(cls):
        """Get lambda vector for hyperparameter search"""
        start, stop, step = cls.LAMBDA_RANGE
        return np.arange(start, stop, step)


class XGBConfig:
    """Configuration class for XGBoost model parameters"""
    
    # Parameter loading and saving
    LOAD_PARAMS = None  # Path to load best parameters JSON file (e.g., './params_best.json')
    SAVE_PARAMS = './params/params_best.json'  # Path to save best parameters after search
    
    # Default parameters (used when LOAD_PARAMS is None and no search is performed)
    DEFAULT_PARAMS = {
        'n_estimators': 200,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    # Early stopping
    EARLY_STOPPING_ROUNDS = 20
    
    # Hyperparameter search configuration
    N_ITER_SEARCH = 100  # RandomizedSearchCV iterations
    CV_FOLDS = 5  
    SCORING = 'neg_mean_squared_error'  # Scoring metric for CV
    
    # Hyperparameter search space for RandomizedSearchCV
    PARAM_DISTRIBUTIONS = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
        'reg_lambda': [0.1, 0.5, 1.0, 5.0, 10.0]
    }