"""
Configuration file for battery cycle life prediction project
"""
import os
import numpy as np
from scipy.stats import randint, uniform, loguniform


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
    
    # Feature preprocessing options
    NORMALIZE_FEATURES = True  # Apply z-score normalization to features
    LOG_TRANSFORM_TARGET = True  # Apply log10 transformation to target (cycle life)
    
    # CNN feature extraction
    CNN_NUM_POINTS_PER_CYCLE = 1000  # Number of points to sample per cycle
    CNN_FEATURE_NAMES = ['V', 'I', 'T']

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
    LOAD_PARAMS = './params/1214_1926.json'  # Path to load best parameters JSON file (e.g., './params_best.json')
    SAVE_PARAMS = './params/1214_1926.json'  # Path to save best parameters after search
    
    # Default parameters (used when LOAD_PARAMS is None and no search is performed)
    DEFAULT_PARAMS = {
        "subsample": 0.6,
        "reg_lambda": 0.1,
        "reg_alpha": 0.01,
        "n_estimators": 250,
        "max_depth": 7,
        "learning_rate": 0.01,
        "colsample_bytree": 0.9
    }
    
    # Early stopping
    EARLY_STOPPING_ROUNDS = 20
    
    # Hyperparameter search configuration
    USE_GRID_SEARCH = False  # True: GridSearchCV, False: RandomizedSearchCV
    CV_FOLDS = 5  
    SCORING = 'neg_mean_squared_error'  # Scoring metric for CV
    
    # -------------------------------------------------------------------------
    # 1. RandomizedSearchCV (Broad Exploration)
    # -------------------------------------------------------------------------
    N_ITER_SEARCH = 10000  # Number of iterations for RandomizedSearchCV
    PARAM_DISTRIBUTIONS = {
        'n_estimators': randint(low=100, high=800),
        'max_depth': randint(low=2, high=5),
        'learning_rate': loguniform(0.005, 0.2),
        'min_child_weight': randint(low=1, high=10),
        'subsample': uniform(0.60, 0.40), # uniform(loc, scale) -> loc到loc+scale
        'colsample_bytree': uniform(0.60, 0.40),
        'reg_alpha': loguniform(0.01, 10.0),
        'reg_lambda': loguniform(0.1, 100.0),
        'gamma': uniform(0.0, 1.0)
    }
    
    # -------------------------------------------------------------------------
    # 2. GridSearchCV (Refined Search)
    # -------------------------------------------------------------------------
    # 注意：运行代码时，通常建议根据 RandomSearch 的最优结果动态调整这里的范围
    # 下面提供的是基于经验的“高概率”微调范围
    PARAM_GRID = {
        'n_estimators': [300, 400, 600, 800],
        'max_depth': [2, 3, 5, 7],              # 极大概率最佳深度是 2 或 3
        'learning_rate': [0.01, 0.03, 0.05, 0.07],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        # 'reg_alpha': [0.01, 0.05 ,0.1],
        'reg_lambda': [0.05, 0.1, 0.2],
        # 'gamma': [0, 0.01, 0.02]
    }


class RandomForestConfig:
    """Configuration class for Random Forest model parameters"""
    
    # Parameter loading and saving
    LOAD_PARAMS = None  # Path to load best parameters JSON file (e.g., './params/rf_best.json')
    SAVE_PARAMS = './params/rf_best.json'  # Path to save best parameters after search
    
    # Default parameters (used when LOAD_PARAMS is None and no search is performed)
    DEFAULT_PARAMS = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'max_samples': 0.8
    }
    
    # Hyperparameter search configuration
    CV_FOLDS = 5
    SCORING = 'neg_mean_squared_error'  # Scoring metric for CV
    
    # RandomizedSearchCV configuration
    N_ITER_SEARCH = 2000  # Number of iterations for RandomizedSearchCV
    PARAM_DISTRIBUTIONS = {
        'n_estimators': randint(low=100, high=500),
        'max_depth': [2, 4, 6 , 8, None],
        'min_samples_split': randint(low=1, high=10),
        'min_samples_leaf': randint(low=1, high=10),
        'max_features': ['sqrt', 'log2', None, 0.5, 0.7, 0.9],
        'bootstrap': [True],
        'max_samples': uniform(0.6, 0.4),  # uniform(loc, scale) -> 0.6 to 1.0
        'min_impurity_decrease': uniform(0.0, 0.01)  # 剪枝参数
    }


class CNNConfig:
    """Configuration class for CNN-BLSTM deep learning model parameters"""
    
    class ModelConfig:
        """Inner class for model-specific configuration"""
        # Training parameters
        batch_size = 16
        epochs = 200
        learning_rate = 0.001

        # Network architecture (can be extended)
        cnn_filters = [32, 64]
        lstm_hidden_size = 64
        dropout = 0.5
    
    # Parameter loading and saving
    LOAD_PARAMS = None  # Path to load model weights (e.g., './params/cnn_best.pth')
    SAVE_PARAMS = './params/cnn_best.pth'  # Path to save model weights
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 15
    
    @classmethod
    def get_model_config(cls):
        """Get model configuration object"""
        return cls.ModelConfig()