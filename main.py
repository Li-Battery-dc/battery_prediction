"""
Main script for battery cycle life prediction
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config, ElasticNetConfig, XGBConfig
from data_preprocess.data_loader import BatteryDataLoader
from feature_extraction.standard import StandardFeatureExtractor
from models.elastic_net import ElasticNetModel
from models.xgb import XGBoostModel


def run_elasticnet():
    """Train and evaluate Elastic Net model for battery cycle life prediction"""
    
    print("=" * 60)
    print("Elastic Net Model")
    print("=" * 60)
    
    # Initialize configurations
    config = Config()
    model_config = ElasticNetConfig()
    
    # Step 1: Load and prepare data
    data_loader = BatteryDataLoader(config)
    train_data, val_data, test_data = data_loader.split_data()

    # Step 2: Extract features
    print("\n2. Extracting features...")
    feature_extractor = StandardFeatureExtractor(config)
    
    X_train, y_train = feature_extractor.extract_features(train_data)
    X_val, y_val = feature_extractor.extract_features(val_data)
    X_test, y_test = feature_extractor.extract_features(test_data)
    
    feature_names = feature_extractor.get_feature_names()
    print(f"   - Extracted {len(feature_names)} features: {feature_names}")
    print(f"   - Training set shape: {X_train.shape}")
    print(f"   - Validation set shape: {X_val.shape}")
    print(f"   - Test set shape: {X_test.shape}")
    
    # Step 3: Train model
    print("\n3. Training Elastic Net model...")
    model = ElasticNetModel(config, model_config)
    training_info = model.fit(X_train, y_train, X_val, y_val)
    
    print(f"   - Best alpha (L1/L2 ratio): {training_info['best_alpha']:.4f}")
    print(f"   - Best lambda (regularization): {training_info['best_lambda']:.4f}")
    print(f"   - Training RMSE: {training_info['train_rmse']:.4f}")
    if 'val_rmse' in training_info:
        print(f"   - Validation RMSE: {training_info['val_rmse']:.4f}")
    
    # Step 4: Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_metrics = model.evaluate(X_test, y_test)
    y_pred_test = model.predict(X_test)
    
    print(f"   - Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"   - Test MAE: {test_metrics['mae']:.4f}")
    print(f"   - Test MAPE: {test_metrics['mape']:.2f}%")
    print(f"   - Test R²: {test_metrics['r2']:.4f}")
    
    # Step 5: Save results and generate visualization
    print("\n5. Saving results and generating visualization...")
    model.save_results(test_metrics, training_info, feature_names, y_test, y_pred_test)


def run_xgboost():
    """Train and evaluate XGBoost model for battery cycle life prediction"""
    
    print("=" * 60)
    print("Battery Cycle Life Prediction - XGBoost Model")
    print("=" * 60)
    
    # Initialize configurations
    config = Config()
    model_config = XGBConfig()
    
    # Step 1: Load and prepare data
    data_loader = BatteryDataLoader(config)
    train_data, val_data, test_data = data_loader.split_data()
    
    # Step 2: Extract features (with standardization for XGBoost)
    print("\n2. Extracting features...")
    feature_extractor = StandardFeatureExtractor(config, normalize=True)
    
    X_train, y_train = feature_extractor.extract_features(train_data)
    X_val, y_val = feature_extractor.extract_features(val_data)
    X_test, y_test = feature_extractor.extract_features(test_data)
    
    feature_names = feature_extractor.get_feature_names()
    print(f"   - Extracted {len(feature_names)} features: {feature_names}")
    print(f"   - Applied standardization (z-score normalization)")
    print(f"   - Training set shape: {X_train.shape}")
    print(f"   - Validation set shape: {X_val.shape}")
    print(f"   - Test set shape: {X_test.shape}")
    
    # Step 3: Train model (auto load params or perform search)
    print("\n3. Training XGBoost model...")
    model = XGBoostModel(config, model_config)
    training_info = model.fit(X_train, y_train, X_val, y_val)
    
    print(f"   - Model trained with {training_info.get('n_estimators', 'N/A')} estimators")
    if 'best_iteration' in training_info:
        print(f"   - Best iteration: {training_info['best_iteration']}")
    
    # Step 4: Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_metrics = model.evaluate(X_test, y_test)
    y_pred_test = model.predict(X_test)
    
    print(f"   - Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"   - Test MAE: {test_metrics['mae']:.4f}")
    print(f"   - Test MAPE: {test_metrics['mape']:.2f}%")
    print(f"   - Test R²: {test_metrics['r2']:.4f}")
    
    # Step 5: Save results and generate visualization
    print("\n5. Saving results and generating visualizations...")
    model.save_results(test_metrics, training_info, feature_names, y_test, y_pred_test)


def main():
    """Main function with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Battery Cycle Life Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model elasticnet
  python main.py --model xgboost
  
Note: XGBoost will auto-load parameters from config.LOAD_PARAMS if specified,
      otherwise it will perform hyperparameter search and save to config.SAVE_PARAMS
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='elasticnet',
        choices=['elasticnet', 'xgboost'],
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
    else:
        print(f"Error: Model '{args.model}' is not yet implemented.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())