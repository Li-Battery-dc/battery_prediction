"""
Main script for battery cycle life prediction
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from config import Config, ElasticNetConfig, XGBConfig, RandomForestConfig, ExtraTreesConfig, CNNConfig
from data_preprocess.data_loader import BatteryDataLoader

from feature_extraction.standard import StandardFeatureExtractor
from feature_extraction.extended import ExtendedFeatureExtractor
from feature_extraction.cnn_feature import CNNFeatureExtractor

from models.elastic_net import ElasticNetModel
from models.xgb import XGBoostModel
from models.rf import RandomForestModel
from models.extra_trees import ExtraTreesModel
from models.CNN import CNNModel

def load_data(config):
    print("\n1. Loading and preparing data...")
    data_loader = BatteryDataLoader(config, apply_outlier_removal=False)
    train_data, val_data, test_data = data_loader.split_data()
    return train_data, val_data, test_data

def extract_features(config):
    """Load data and extract features
    
    Args:
        config: Configuration object    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, feature_extractor, train_data, val_data, test_data)
    """
    train_data, val_data, test_data = load_data(config)
    # Step 2: Extract features
    print("\n2. Extracting features...")
    if config.USE_EXTENDED_FEATURES:
        feature_extractor = ExtendedFeatureExtractor(config)
    else:
        feature_extractor = StandardFeatureExtractor(config)
    
    X_train, y_train = feature_extractor.extract_features(train_data)
    X_val, y_val = feature_extractor.extract_features(val_data)
    X_test, y_test = feature_extractor.extract_features(test_data)
    
    feature_names = feature_extractor.get_feature_names()
    print(f"   - Feature extractor: {feature_extractor.__class__.__name__}")
    print(f"   - Extracted {len(feature_names)} features: {feature_names}")
    if config.NORMALIZE_FEATURES:
        print(f"   - Applied standardization (z-score normalization)")
    if config.LOG_TRANSFORM_TARGET:
        print(f"   - Applied log10 transformation to target variable")
    print(f"   - Training set shape: {X_train.shape}")
    print(f"   - Validation set shape: {X_val.shape}")
    print(f"   - Test set shape: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_extractor


def run_elasticnet():
    """Train and evaluate Elastic Net model for battery cycle life prediction"""
    
    print("=" * 60)
    print("Elastic Net Model")
    print("=" * 60)
    
    # Initialize configurations
    config = Config()
    model_config = ElasticNetConfig()
    
    # Load data and extract features
    X_train, y_train, X_val, y_val, X_test, y_test, feature_extractor = \
        extract_features(config)
    
    feature_names = feature_extractor.get_feature_names()
    
    # Step 3: Train model
    print("\n3. Training Elastic Net model...")
    model = ElasticNetModel(config, model_config)
    training_info = model.fit(X_train, y_train, X_val, y_val)
    
    print(f"   - Best alpha (L1/L2 ratio): {training_info['best_alpha']:.4f}")
    print(f"   - Best lambda (regularization): {training_info['best_lambda']:.4f}")
    print(f"   - Training RMSE: {training_info['train_rmse']:.4f}")
    if 'val_rmse' in training_info:
        print(f"   - Validation RMSE: {training_info['val_rmse']:.4f}")
    
    # Step 4: Evaluate on train and test sets
    print("\n4. Evaluating on train and test sets...")
    train_metrics = model.evaluate(X_train, y_train, feature_extractor)
    test_metrics = model.evaluate(X_test, y_test, feature_extractor)

    print(f"   - Train MSE: {train_metrics['mse']:.4f}")
    print(f"   - Train MPE: {train_metrics['mpe']:.2f}%")
    print(f"   - Train R²: {train_metrics['r2']:.4f}")
    print(f"   - Test MSE: {test_metrics['mse']:.4f}")
    print(f"   - Test MPE: {test_metrics['mpe']:.2f}%")
    print(f"   - Test R²: {test_metrics['r2']:.4f}")
    
    # Step 5: Save results and generate visualization
    # 获取原始尺度的预测值和真实值用于可视化
    y_pred_test = feature_extractor.inverse_transform_target(model.predict(X_test))
    y_test_original = feature_extractor.inverse_transform_target(y_test)
    print("\n5. Saving results and generating visualization...")
    model.save_results(test_metrics, training_info, feature_names, y_test_original, y_pred_test, train_metrics=train_metrics)


def run_xgboost():
    """Train and evaluate XGBoost model for battery cycle life prediction"""
    
    print("=" * 60)
    print("XGBoost Model")
    print("=" * 60)
    
    # Initialize configurations
    config = Config()
    model_config = XGBConfig()
    
    # Load data and extract features
    X_train, y_train, X_val, y_val, X_test, y_test, feature_extractor= \
        extract_features(config)
    
    feature_names = feature_extractor.get_feature_names()
    
    # Step 3: Train model (auto load params or perform search)
    print("\n3. Training XGBoost model...")
    model = XGBoostModel(config, model_config)
    training_info = model.fit(X_train, y_train, X_val, y_val)
    
    print(f"   - Model trained with {training_info.get('n_estimators', 'N/A')} estimators")
    if 'best_iteration' in training_info:
        print(f"   - Best iteration: {training_info['best_iteration']}")
    
    # Step 4: Evaluate on train and test sets
    print("\n4. Evaluating on train and test sets...")
    # evaluate方法内部会自动进行逆转换
    train_metrics = model.evaluate(X_train, y_train, feature_extractor)
    test_metrics = model.evaluate(X_test, y_test, feature_extractor)
    
    print(f"   - Train MSE: {train_metrics['mse']:.4f}")
    print(f"   - Train MPE: {train_metrics['mpe']:.2f}%")
    print(f"   - Train R²: {train_metrics['r2']:.4f}")
    print(f"   - Test MSE: {test_metrics['mse']:.4f}")
    print(f"   - Test MPE: {test_metrics['mpe']:.2f}%")
    print(f"   - Test R²: {test_metrics['r2']:.4f}")
    
    # Step 5: Save results and generate visualization
    print("\n5. Saving results and generating visualizations...")
    # 获取原始尺度的预测值和真实值用于可视化
    y_pred_test = feature_extractor.inverse_transform_target(model.predict(X_test))
    y_test_original = feature_extractor.inverse_transform_target(y_test)
    model.save_results(test_metrics, training_info, feature_names, y_test_original, y_pred_test, train_metrics=train_metrics)


def run_rf():
    """Train and evaluate Random Forest model for battery cycle life prediction"""
    
    print("=" * 60)
    print("Random Forest Model")
    print("=" * 60)
    
    # Initialize configurations
    config = Config()
    model_config = RandomForestConfig()
    
    # Load data and extract features
    X_train, y_train, X_val, y_val, X_test, y_test, feature_extractor= \
        extract_features(config)
    
    feature_names = feature_extractor.get_feature_names()
    
    # Step 3: Train model (auto load params or perform search)
    print("\n3. Training Random Forest model...")
    model = RandomForestModel(config, model_config)
    training_info = model.fit(X_train, y_train, X_val, y_val)
    
    print(f"   - Model trained with {training_info.get('n_estimators', 'N/A')} trees")
    
    # Step 4: Evaluate on train and test sets
    print("\n4. Evaluating on train and test sets...")
    # evaluate方法内部会自动进行逆转换
    train_metrics = model.evaluate(X_train, y_train, feature_extractor)
    test_metrics = model.evaluate(X_test, y_test, feature_extractor)
    
    print(f"   - Train MSE: {train_metrics['mse']:.4f}")
    print(f"   - Train MPE: {train_metrics['mpe']:.2f}%")
    print(f"   - Train R²: {train_metrics['r2']:.4f}")
    print(f"   - Test MSE: {test_metrics['mse']:.4f}")
    print(f"   - Test MPE: {test_metrics['mpe']:.2f}%")
    print(f"   - Test R²: {test_metrics['r2']:.4f}")
    
    # Step 5: Save results and generate visualization
    print("\n5. Saving results and generating visualizations...")
    # 获取原始尺度的预测值和真实值用于可视化
    y_pred_test = feature_extractor.inverse_transform_target(model.predict(X_test))
    y_test_original = feature_extractor.inverse_transform_target(y_test)
    model.save_results(test_metrics, training_info, feature_names, y_test_original, y_pred_test, train_metrics=train_metrics)

def run_extratrees():
    """Train and evaluate Extra Trees model for battery cycle life prediction"""
    
    print("=" * 60)
    print("Extra Trees Model")
    print("=" * 60)
    
    # Initialize configurations
    config = Config()
    model_config = ExtraTreesConfig()
    
    # Load data and extract features
    X_train, y_train, X_val, y_val, X_test, y_test, feature_extractor= \
        extract_features(config)
    
    feature_names = feature_extractor.get_feature_names()
    
    # Step 3: Train model (auto load params or perform search)
    print("\n3. Training Extra Trees model...")
    model = ExtraTreesModel(config, model_config)
    training_info = model.fit(X_train, y_train, X_val, y_val)
    
    print(f"   - Model trained with {training_info.get('n_estimators', 'N/A')} trees")
    
    # Step 4: Evaluate on train and test sets
    print("\n4. Evaluating on train and test sets...")
   
    train_metrics = model.evaluate(X_train, y_train, feature_extractor)
    test_metrics = model.evaluate(X_test, y_test, feature_extractor)
    
    print(f"   - Train MSE: {train_metrics['mse']:.4f}")
    print(f"   - Train MPE: {train_metrics['mpe']:.2f}%")
    print(f"   - Train R²: {train_metrics['r2']:.4f}")
    print(f"   - Test MSE: {test_metrics['mse']:.4f}")
    print(f"   - Test MPE: {test_metrics['mpe']:.2f}%")
    print(f"   - Test R²: {test_metrics['r2']:.4f}")
    
    # Step 5: Save results and generate visualization
    print("\n5. Saving results and generating visualizations...")
    # 获取原始尺度的预测值和真实值用于可视化
    y_pred_test = feature_extractor.inverse_transform_target(model.predict(X_test))
    y_test_original = feature_extractor.inverse_transform_target(y_test)
    model.save_results(test_metrics, training_info, feature_names, y_test_original, y_pred_test, train_metrics=train_metrics)

def run_CNN():
    """Train and evaluate CNN-BLSTM model for battery cycle life prediction"""
    
    print("=" * 60)
    print("CNN Deep Learning Model")
    print("=" * 60)
    
    # Initialize configurations
    config = Config()
    config.LOG_TRANSFORM_TARGET = False  # CNN 不进行 log 转换
    model_config = CNNConfig()
    
    train_data, val_data, test_data = load_data(config)
    
    # Step 2: Extract raw time-series features using CNN feature extractor
    print(f"\n2. Extracting features maps for CNN...")
    feature_extractor = CNNFeatureExtractor(config)
    
    X_train, y_train = feature_extractor.extract_features(train_data, split='train')
    X_val, y_val = feature_extractor.extract_features(val_data, split='val')
    X_test, y_test = feature_extractor.extract_features(test_data, split='test')
    
    print(f"   - Feature extractor: {feature_extractor.__class__.__name__}")
    if config.NORMALIZE_FEATURES:
        print(f"   - Applied standardization to time-series data")
    if config.LOG_TRANSFORM_TARGET:
        print(f"   - Applied log10 transformation to target variable")
    print(f"   - Training set: {len(X_train)} samples")
    print(f"   - Validation set: {len(X_val)} samples")
    print(f"   - Test set: {len(X_test)} samples")
    
    # Step 3: Train model
    print("\n3. Training CNN model...")
    model = CNNModel(config, model_config)
    training_info = model.fit(X_train, y_train, X_val, y_val)
    
    print(f"   - Training completed")
    print(f"   - Final training loss: {training_info['train_loss'][-1]:.4f}")
    if 'val_loss' in training_info and len(training_info['val_loss']) > 0:
        print(f"   - Final validation loss: {training_info['val_loss'][-1]:.4f}")
    
    # Step 4: Evaluate on train and test sets
    print("\n4. Evaluating on train and test sets...")
    train_metrics = model.evaluate(X_train, y_train, feature_extractor)
    test_metrics = model.evaluate(X_test, y_test, feature_extractor)
    
    print(f"   - Train MSE: {train_metrics['mse']:.4f}")
    print(f"   - Train MPE: {train_metrics['mpe']:.2f}%")
    print(f"   - Train R²: {train_metrics['r2']:.4f}")
    print(f"   - Test MSE: {test_metrics['mse']:.4f}")
    print(f"   - Test MPE: {test_metrics['mpe']:.2f}%")
    print(f"   - Test R²: {test_metrics['r2']:.4f}")
    
    # Step 5: Save results and generate visualization
    print("\n5. Saving results and generating visualizations...")
    y_pred_test = feature_extractor.inverse_transform_target(model.predict(X_test))
    y_test_original = feature_extractor.inverse_transform_target(y_test)
    feature_names = feature_extractor.get_feature_names()
    model.save_results(test_metrics, training_info, feature_names, y_test_original, y_pred_test, train_metrics=train_metrics)

    # 将最佳模型权重复制到本次结果目录，便于归档
    saved_weight_src = training_info.get('saved_model_path') if isinstance(training_info, dict) else None
    if saved_weight_src and model.result_dir:
        dst_path = os.path.join(model.result_dir, os.path.basename(saved_weight_src))
        try:
            shutil.copy2(saved_weight_src, dst_path)
            print(f"   - Copied best model weights to: {dst_path}")
        except Exception as e:
            print(f"   - Warning: failed to copy weights to results dir: {e}")

def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Battery Cycle Life Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model elasticnet
  python main.py --model xgboost
  python main.py --model rf
  python main.py --model extratrees
  python main.py --model cnn
  
Note: XGBoost, Random Forest and Extra Trees will auto-load parameters from config.LOAD_PARAMS if specified,
      otherwise they will perform hyperparameter search and save to config.SAVE_PARAMS.
      CNN uses raw time-series features (e.g., Vdlin) instead of engineered features.
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='elasticnet',
        choices=['elasticnet', 'xgboost', 'rf', 'extratrees', 'cnn'],
        help='Model type to use for prediction (default: elasticnet)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate model function
    if args.model == 'elasticnet':
        run_elasticnet()
    elif args.model == 'xgboost':
        run_xgboost()
    elif args.model == 'rf':
        run_rf()
    elif args.model == 'extratrees':
        run_extratrees()
    elif args.model == 'cnn':
        run_CNN()
    else:
        print(f"Error: Model '{args.model}' is not yet implemented.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())