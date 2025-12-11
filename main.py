"""
Main script for battery cycle life prediction
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from data_preprocess.data_loader import BatteryDataLoader
from feature_extraction.feature_extractor import FeatureExtractorFactory
from models.elastic_net_model import ElasticNetModel


def main():
    """Main function for training and evaluating battery cycle life prediction model"""
    
    print("=" * 60)
    print("Battery Cycle Life Prediction")
    print("=" * 60)
    
    # Initialize configuration
    config = Config()
    
    # Step 1: Load and prepare data
    print("\n1. Loading and preparing data...")
    data_loader = BatteryDataLoader(config)
    train_data, val_data, test_data = data_loader.split_data()
    
    # Print data information
    data_info = data_loader.get_data_info()
    print(f"   - Train: {data_info['train_cells']} cells")
    print(f"   - Validation: {data_info['val_cells']} cells") 
    print(f"   - Test: {data_info['test_cells']} cells")
    print(f"   - Total: {data_info['total_cells']} cells")
    
    # Step 2: Extract features
    print("\n2. Extracting features...")
    feature_extractor = FeatureExtractorFactory.create_extractor('standard', config)
    
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
    model = ElasticNetModel(config)
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
    
    # Step 5: Feature importance analysis
    print("\n5. Feature importance analysis...")
    feature_importance = model.get_feature_importance(feature_names)
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("   - Most important features:")
    for i, (feature, importance) in enumerate(sorted_importance[:5]):
        print(f"     {i+1}. {feature}: {importance:.4f}")
    
    # Step 7: Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model Type: Elastic Net Regression")
    print(f"Features: {len(feature_names)} engineered features")
    print(f"Training Data: {len(train_data)} cells")
    print(f"Test Performance:")
    print(f"  - RMSE: {test_metrics['rmse']:.2f} cycles")
    print(f"  - MAPE: {test_metrics['mape']:.1f}%")
    print(f"  - R²: {test_metrics['r2']:.3f}")
    print(f"Best Hyperparameters:")
    print(f"  - Alpha (L1/L2 ratio): {training_info['best_alpha']:.4f}")
    print(f"  - Lambda (regularization): {training_info['best_lambda']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()