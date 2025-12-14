"""
Random Forest model implementation for battery cycle life prediction
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
import joblib
import contextlib

from models.base import BaseModel
from config import Config, RandomForestConfig


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class RandomForestModel(BaseModel):
    """Random Forest model for battery cycle life prediction"""
    
    def __init__(self, config: Config = None, model_config: RandomForestConfig = None):
        """Initialize Random Forest model"""
        super().__init__(config, model_config)
        self.config = config if config else Config()
        self.model_config = model_config if model_config else RandomForestConfig()
        self.model_params = {}
        
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, 
            X_val: pd.DataFrame = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the Random Forest model with parameter loading or hyperparameter search
        
        Note: y_train and y_val should already be log-transformed if using log transformation.
              Validation set will be merged with training set for cross-validation.
        """
        
        # 1. 合并训练集和验证集用于交叉验证（RF不需要early stopping）
        if X_val is not None and y_val is not None:
            X_train_full = pd.concat([X_train, X_val])
            y_train_full = np.concatenate([y_train, y_val])
            print(f"   - Merged training and validation sets for cross-validation")
            print(f"   - Total training samples: {len(y_train_full)}")
        else:
            X_train_full = X_train
            y_train_full = y_train
        
        # 2. 确定使用的参数
        if self.model_config.LOAD_PARAMS and os.path.exists(self.model_config.LOAD_PARAMS):
            # 加载已保存的最佳参数
            print(f"Loading parameters from {self.model_config.LOAD_PARAMS}...")
            with open(self.model_config.LOAD_PARAMS, 'r') as f:
                loaded_data = json.load(f)
                best_params = loaded_data.get('best_params', self.model_config.DEFAULT_PARAMS)
            print(f"   - Loaded parameters: {best_params}")
            
        else:
            # 执行超参数搜索
            print("No parameter file found. Performing hyperparameter search...")
            
            # 基础模型
            base_model = RandomForestRegressor(
                random_state=self.config.RANDOM_STATE,
                n_jobs=1  # 并行由 joblib 管理
            )
            
            # RandomizedSearchCV - 随机采样搜索
            print(f"   - Using RandomizedSearchCV (random sampling)")
            print(f"   - Number of iterations: {self.model_config.N_ITER_SEARCH}")
            print(f"   - Cross-validation folds: {self.model_config.CV_FOLDS}")
            total_fits = self.model_config.N_ITER_SEARCH * self.model_config.CV_FOLDS
            print(f"   - Total fits: {total_fits}")
            
            search = RandomizedSearchCV(
                base_model,
                param_distributions=self.model_config.PARAM_DISTRIBUTIONS,
                n_iter=self.model_config.N_ITER_SEARCH,
                cv=self.model_config.CV_FOLDS,
                scoring=self.model_config.SCORING,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                verbose=0  # 静默模式，使用tqdm显示进度
            )
            
            print("\nStarting hyperparameter search...")
            # 使用 tqdm_joblib 上下文管理器
            with tqdm_joblib(tqdm(desc="Hyperparameter Search", total=total_fits)) as pbar:
                search.fit(X_train_full, y_train_full)
            
            best_params = search.best_params_
            best_cv_score = -search.best_score_  # 转换为RMSE
            
            print(f"\n   - Search completed!")
            print(f"   - Best CV RMSE: {best_cv_score:.4f}")
            print(f"   - Best parameters: {best_params}")
            
            # 保存最佳参数
            if self.model_config.SAVE_PARAMS:
                os.makedirs(os.path.dirname(self.model_config.SAVE_PARAMS) or '.', exist_ok=True)
    
                save_data = {
                    'best_params': best_params,
                    'best_cv_score': float(best_cv_score),
                    'search_method': 'RandomizedSearchCV'
                }
                with open(self.model_config.SAVE_PARAMS, 'w') as f:
                    json.dump(save_data, f, indent=4)
                print(f"   - Saved best parameters to: {self.model_config.SAVE_PARAMS}")
        
        # 3. 使用最佳参数训练最终模型
        print("\nTraining final Random Forest model with best parameters...")
        
        self.model_params = {
            **best_params,
            'random_state': self.config.RANDOM_STATE,
            'n_jobs': -1
        }
        
        self.model = RandomForestRegressor(**self.model_params)
        
        # 4. 训练模型
        self.model.fit(X_train_full, y_train_full)
        
        self.is_fitted = True
        
        # 5. 存储训练信息
        training_info = {
            'best_params': best_params,
            'n_estimators': self.model.n_estimators,
            'search_method': 'Loaded from file' if (self.model_config.LOAD_PARAMS and os.path.exists(self.model_config.LOAD_PARAMS)) else 'RandomizedSearchCV'
        }
        
        print(f"Training complete! Number of trees: {training_info['n_estimators']}")
        
        return training_info
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model
        
        Note: Predictions are returned in the same scale as the training targets.
        If log transformation was used during training, predictions will be in log scale.
        Use feature_extractor.inverse_transform_target() to convert back if needed.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        y_pred = self.model.predict(X)
        
        # 确保预测值非负
        y_pred[y_pred < 0] = 0
        
        return y_pred
    
    def get_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        """Get feature importance from Random Forest model (based on impurity reduction)"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        importance = self.model.feature_importances_
        
        if feature_names:
            feature_importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
            return feature_importance_dict
        
        # 如果没有提供特征名，使用索引
        return {f'feature_{i}': float(imp) for i, imp in enumerate(importance)}
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.model_params
