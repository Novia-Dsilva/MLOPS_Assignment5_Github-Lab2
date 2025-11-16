# import pickle, os, json, random
# from sklearn.metrics import f1_score
# import joblib, glob, sys
# import argparse
# from sklearn.datasets import make_classification

# sys.path.insert(0, os.path.abspath('..'))

# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
#     args = parser.parse_args()
    
#     # Access the timestamp
#     timestamp = args.timestamp
#     try:
#         model_version = f'model_{timestamp}_dt_model'  # Use a timestamp as the version
#         model = joblib.load(f'{model_version}.joblib')
#     except:
#         raise ValueError('Failed to catching the latest model')
        
#     try:
#         # Check if the file exists within the folder
#         X, y = make_classification(
#                             n_samples=random.randint(0, 2000),
#                             n_features=6,
#                             n_informative=3,
#                             n_redundant=0,
#                             n_repeated=0,
#                             n_classes=2,
#                             random_state=0,
#                             shuffle=True,
#                         )
#     except:
#         raise ValueError('Failed to catching the data')
    
#     y_predict = model.predict(X)
#     metrics = {"F1_Score":f1_score(y, y_predict)}
    
#     # Save metrics to a JSON file

#     if not os.path.exists('metrics/'): 
#         # then create it.
#         os.makedirs("metrics/")
        
#     with open(f'{timestamp}_metrics.json', 'w') as metrics_file:
#         json.dump(metrics, metrics_file, indent=4)
               
    

"""
Comprehensive Model Evaluation with Visualizations
"""
import json
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

sys.path.insert(0, os.path.abspath('..'))

from data_loader import HousingDataLoader
from preprocessing import HousingPreprocessor


def load_model_and_preprocessor(timestamp, model_type):
    """Load trained model and preprocessor"""
    model_path = f'model_{timestamp}_{model_type}.joblib'
    preprocessor_path = f'models/preprocessor_{timestamp}.pkl'
    
    model = joblib.load(model_path)
    preprocessor = HousingPreprocessor.load_preprocessor(preprocessor_path)
    
    return model, preprocessor


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2_Score': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'Max_Error': np.max(np.abs(y_true - y_pred)),
        'Mean_Prediction': np.mean(y_pred),
        'Std_Prediction': np.std(y_pred)
    }
    return metrics


def create_evaluation_plots(y_true, y_pred, timestamp, model_type):
    """Create comprehensive evaluation plots"""
    os.makedirs('plots', exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Price ($100k)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Price ($100k)', fontsize=12)
    axes[0, 0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price ($100k)', fontsize=12)
    axes[0, 1].set_ylabel('Residuals', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error distribution
    errors = np.abs(residuals)
    axes[1, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Absolute Error', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Absolute Errors', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'plots/evaluation_{timestamp}_{model_type}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Evaluation plots saved: {plot_path}")
    return plot_path


def create_feature_importance_plot(model, feature_names, timestamp, model_type):
    """Create feature importance plot if model supports it"""
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 features
        
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot_path = f'plots/feature_importance_{timestamp}_{model_type}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" Feature importance plot saved: {plot_path}")
        return plot_path
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="gradient_boosting")
    args = parser.parse_args()
    
    timestamp = args.timestamp
    model_type = args.model_type
    
    print(f" Evaluating Housing Price Model - {timestamp}")
    print("="*60)
    
    # Load data
    print("\n Loading test data...")
    loader = HousingDataLoader()
    df, feature_names, _ = loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data(df)
    
    # Load model and preprocessor
    print("\n Loading model and preprocessor...")
    try:
        model, preprocessor = load_model_and_preprocessor(timestamp, model_type)
    except Exception as e:
        print(f" Error loading model: {e}")
        print("Trying alternative loading method...")
        model = joblib.load(f'model_{timestamp}_{model_type}.joblib')
        preprocessor = HousingPreprocessor.load_preprocessor(
            f'models/preprocessor_{timestamp}.pkl'
        )
    
    # Preprocess test data
    print("\n⚙️ Preprocessing test data...")
    X_test_features = preprocessor.create_features(X_test)
    X_test_processed = preprocessor.transform(X_test_features)
    
    # Make predictions
    print("\n Making predictions...")
    y_pred = model.predict(X_test_processed)
    
    # Calculate metrics
    print("\n Calculating metrics...")
    metrics = calculate_metrics(y_test.values, y_pred)
    
    # Print metrics
    print("\n" + "="*60)
    print(" EVALUATION RESULTS")
    print("="*60)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Create visualizations
    print("\n Creating visualizations...")
    create_evaluation_plots(y_test.values, y_pred, timestamp, model_type)
    
    # Feature importance (if available)
    create_feature_importance_plot(model, X_test_processed.columns, 
                                   timestamp, model_type)
    
    # Save metrics to JSON
    os.makedirs('metrics', exist_ok=True)
    metrics_filename = f'{timestamp}_metrics.json'
    
    with open(f'metrics/{metrics_filename}', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"\n Metrics saved: metrics/{metrics_filename}")
    print("\n Evaluation complete!")

