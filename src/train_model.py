# # from sklearn.datasets import fetch_rcv1
# import mlflow, datetime, os, pickle, random
# # import sklearn
# from joblib import dump
# from sklearn.datasets import make_classification
# from sklearn.metrics import accuracy_score, f1_score
# import sys
# from sklearn.ensemble import RandomForestClassifier
# import argparse

# sys.path.insert(0, os.path.abspath('..'))


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
#     args = parser.parse_args()
    
#     # Access the timestamp
#     timestamp = args.timestamp
    
#     # Use the timestamp in your script
#     print(f"Timestamp received from GitHub Actions: {timestamp}")
    
#     # Check if the file exists within the folder
#     X, y = make_classification(
#                             n_samples=random.randint(0, 2000),
#                             n_features=6,
#                             n_informative=3,
#                             n_redundant=0,
#                             n_repeated=0,
#                             n_classes=2,
#                             random_state=0,
#                             shuffle=True,
#                         )
#     if os.path.exists('data'): 
#         with open('data/data.pickle', 'wb') as data:
#             pickle.dump(X, data)
            
#         with open('data/target.pickle', 'wb') as data:
#             pickle.dump(y, data)  
#     else:
#         os.makedirs('data/')
#         with open('data/data.pickle', 'wb') as data:
#             pickle.dump(X, data)
            
#         with open('data/target.pickle', 'wb') as data:
#             pickle.dump(y, data)  
            
#     mlflow.set_tracking_uri("./mlruns")
#     dataset_name = "Reuters Corpus Volume"
#     current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
#     experiment_name = f"{dataset_name}_{current_time}"    
#     experiment_id = mlflow.create_experiment(f"{experiment_name}")

#     with mlflow.start_run(experiment_id=experiment_id,
#                         run_name= f"{dataset_name}"):
        
#         params = {
#                     "dataset_name": dataset_name,
#                     "number of datapoint": X.shape[0],
#                     "number of dimensions": X.shape[1]}
        
#         mlflow.log_params(params)
            
        
#         forest = RandomForestClassifier(random_state=0)
#         forest.fit(X, y)
        
#         y_predict = forest.predict(X)
#         mlflow.log_metrics({'Accuracy': accuracy_score(y, y_predict),
#                             'F1 Score': f1_score(y, y_predict)})
        
#         if not os.path.exists('models/'): 
#             # then create it.
#             os.makedirs("models/")
            
#         # After retraining the model
#         model_version = f'model_{timestamp}'  # Use a timestamp as the version
#         model_filename = f'{model_version}_dt_model.joblib'
#         dump(forest, model_filename)
                    



"""
Train Housing Price Prediction Model with MLflow Tracking
"""
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import argparse
import os
import sys
from joblib import dump
import json

sys.path.insert(0, os.path.abspath('..'))

from data_loader import HousingDataLoader
from preprocessing import preprocess_pipeline


def train_ridge_model(X_train, y_train, alpha=1.0):
    """Train Ridge Regression model"""
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, 
                           n_estimators=100, 
                           learning_rate=0.1,
                           max_depth=3):
    """Train Gradient Boosting model"""
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'r2': r2_score(y, y_pred),
        'mape': np.mean(np.abs((y - y_pred) / y)) * 100
    }
    
    return metrics, y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="gradient_boosting",
                       choices=['ridge', 'gradient_boosting'])
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    
    timestamp = args.timestamp
    print(f" Training Housing Price Model - Timestamp: {timestamp}")
    print("="*60)
    
    # Load and prepare data
    print("\n Loading data...")
    loader = HousingDataLoader()
    df, feature_names, _ = loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data(df)
    
    # Preprocess data
    print("\n⚙️ Preprocessing data...")
    X_train_p, X_test_p, y_train, y_test, preprocessor = preprocess_pipeline(
        X_train, X_test, y_train, y_test,
        create_features=True,
        scaling_method='standard',
        handle_outliers=True
    )
    
    # Save preprocessor
    preprocessor.save_preprocessor(f'models/preprocessor_{timestamp}.pkl')
    
    # MLflow setup
    mlflow.set_tracking_uri("./mlruns")
    experiment_name = f"Housing_Price_Prediction_{timestamp}"
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, 
                         run_name=f"Housing_{args.model_type}"):
        
        # Log parameters
        params = {
            "model_type": args.model_type,
            "n_samples_train": X_train_p.shape[0],
            "n_features": X_train_p.shape[1],
            "preprocessing": "standard_scaling + feature_engineering",
            "timestamp": timestamp
        }
        
        if args.model_type == "gradient_boosting":
            params.update({
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth
            })
        else:
            params["alpha"] = args.alpha
        
        mlflow.log_params(params)
        
        # Train model
        print(f"\nTraining {args.model_type} model...")
        if args.model_type == "gradient_boosting":
            model = train_gradient_boosting(
                X_train_p, y_train,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth
            )
        else:
            model = train_ridge_model(X_train_p, y_train, alpha=args.alpha)
        
        # Evaluate on training set
        print("\nEvaluating model...")
        train_metrics, _ = evaluate_model(model, X_train_p, y_train)
        test_metrics, test_predictions = evaluate_model(model, X_test_p, y_test)
        
        # Log metrics
        for key, value in train_metrics.items():
            mlflow.log_metric(f"train_{key}", value)
        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)
        
        # Print results
        print("\n" + "="*60)
        print("📊 MODEL PERFORMANCE")
        print("="*60)
        print("\nTraining Metrics:")
        for key, value in train_metrics.items():
            print(f"  {key.upper()}: {value:.4f}")
        
        print("\nTest Metrics:")
        for key, value in test_metrics.items():
            print(f"  {key.upper()}: {value:.4f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_filename = f'model_{timestamp}_{args.model_type}.joblib'
        dump(model, model_filename)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\nModel saved: {model_filename}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")
        
        # Save feature names for later use
        feature_info = {
            'feature_names': list(X_train_p.columns),
            'n_features': X_train_p.shape[1]
        }
        
        with open(f'models/feature_info_{timestamp}.json', 'w') as f:
            json.dump(feature_info, f, indent=4)
    
    print("\n🎉 Training complete!")