"""
XGBoost model implementation for battery cycle life prediction
"""
import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from models.base import BaseModel
from config import Config, XGBConfig

class XGBoostModel(BaseModel):
    """Extreme Gradient Boosting (XGBoost) model for battery cycle life prediction"""
    
    def __init__(self, config: Config = None, model_config: XGBConfig = None):
        """Initialize XGBoost model"""
        super().__init__(config, model_config)
        self.config = config if config else Config()
        self.model_config = model_config if model_config else XGBConfig()
        self.model_params = {}
        
    def _transform_target(self, y: np.ndarray) -> np.ndarray:
        """Apply log10 transformation to the target variable"""
        # 目标变换：y_log = log10(y)
        # 对数变换可以稳定方差，并使误差指标 (如 MAPE) 更合理
        return np.log10(y)

    def _inverse_transform_target(self, y_log: np.ndarray) -> np.ndarray:
        """Apply inverse transformation to the predicted log targets"""
        # 逆变换：y_pred = 10^(y_log)
        return 10**y_log
        
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, 
            X_val: pd.DataFrame = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the XGBoost model with parameter loading or hyperparameter search"""
        
        # 1. 目标变量对数变换 (Log Transformation)
        y_train_log = self._transform_target(y_train)
        
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
            print(f"   - Using RandomizedSearchCV with {self.model_config.N_ITER_SEARCH} iterations")
            print(f"   - Cross-validation folds: {self.model_config.CV_FOLDS}")
            
            # 合并训练集和验证集用于交叉验证
            if X_val is not None and y_val is not None:
                X_train_full = pd.concat([X_train, X_val])
                y_train_full = np.concatenate([y_train, y_val])
            else:
                X_train_full = X_train
                y_train_full = y_train
            
            y_train_full_log = self._transform_target(y_train_full)
            
            # 基础模型
            base_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1
            )
            
            # RandomizedSearchCV
            search = RandomizedSearchCV(
                base_model,
                param_distributions=self.model_config.PARAM_DISTRIBUTIONS,
                n_iter=self.model_config.N_ITER_SEARCH,
                cv=self.model_config.CV_FOLDS,
                scoring=self.model_config.SCORING,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                verbose=1
            )
            
            search.fit(X_train_full, y_train_full_log)
            
            best_params = search.best_params_
            best_cv_score = -search.best_score_
            
            print(f"   - Best CV RMSE: {best_cv_score:.4f}")
            print(f"   - Best parameters: {best_params}")
            
            # 保存最佳参数
            if self.model_config.SAVE_PARAMS:
                os.makedirs(os.path.dirname(self.model_config.SAVE_PARAMS) or '.', exist_ok=True)
                save_data = {
                    'best_params': best_params,
                    'best_cv_score': float(best_cv_score),
                    'search_space': self.model_config.PARAM_DISTRIBUTIONS,
                    'n_iter': self.model_config.N_ITER_SEARCH,
                    'cv_folds': self.model_config.CV_FOLDS
                }
                with open(self.model_config.SAVE_PARAMS, 'w') as f:
                    json.dump(save_data, f, indent=4)
                print(f"   - Saved best parameters to: {self.model_config.SAVE_PARAMS}")
        
        # 3. 使用最佳参数训练最终模型
        print("\nTraining final XGBoost model with best parameters...")
        
        # 准备训练参数（包含early_stopping_rounds）
        self.model_params = {
            'objective': 'reg:squarederror',
            **best_params,
            'random_state': self.config.RANDOM_STATE,
            'n_jobs': -1
        }
        
        # 如果有验证集，添加early_stopping_rounds到模型参数中
        if X_val is not None and y_val is not None:
            self.model_params['early_stopping_rounds'] = self.model_config.EARLY_STOPPING_ROUNDS
        
        self.model = xgb.XGBRegressor(**self.model_params)
        
        # 4. 训练模型
        if X_val is not None and y_val is not None:
            y_val_log = self._transform_target(y_val)
            eval_set = [(X_val, y_val_log)]
            
            self.model.fit(
                X_train, 
                y_train_log, 
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(
                X_train, 
                y_train_log,
                verbose=False
            )
        
        self.is_fitted = True
        
        # 5. 存储训练信息
        training_info = {
            'best_params': best_params,
            'n_estimators': self.model.n_estimators if hasattr(self.model, 'n_estimators') else best_params.get('n_estimators', 'N/A'),
        }
        if hasattr(self.model, 'best_iteration'):
            training_info['best_iteration'] = self.model.best_iteration
             
        print(f"Training complete! Best iteration: {training_info.get('best_iteration', 'N/A')}")
        
        return training_info
        return training_info
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # 1. 在对数空间进行预测
        y_pred_log = self.model.predict(X)
        
        # 2. 逆变换回实际的循环寿命值 (Inverse Transformation)
        y_pred = self._inverse_transform_target(y_pred_log)
        
        # 确保预测值非负
        y_pred[y_pred < 0] = 0
        
        return y_pred
    
    def get_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        """Get feature importance from XGBoost model (gain)"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        importance = self.model.get_booster().get_score(importance_type='gain')
        
        # XGBoost 的特征名是 f0, f1, f2... 需要映射
        if feature_names:
            feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
            mapped_importance = {feature_map.get(k, k): v for k, v in importance.items()}
            return mapped_importance
        
        return importance
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return self.model_params