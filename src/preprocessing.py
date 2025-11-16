"""
Data Preprocessing for Housing Price Prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import pickle
import os


class HousingPreprocessor:
    """Preprocessing pipeline for housing data"""
    
    def __init__(self, scaling_method='standard'):
        """
        Args:
            scaling_method: 'standard', 'robust', or 'none'
        """
        self.scaling_method = scaling_method
        self.scaler = None
        self.feature_names = None
        
    def create_features(self, X):
        """Create additional features"""
        X_new = X.copy()
        
        # Rooms per household
        X_new['RoomsPerHousehold'] = X_new['AveRooms'] * X_new['AveOccup']
        
        # Bedrooms ratio
        X_new['BedroomsRatio'] = X_new['AveBedrms'] / X_new['AveRooms']
        
        # Population per household
        X_new['PopulationPerHousehold'] = X_new['Population'] / X_new['AveOccup']
        
        print(f"Created {len(X_new.columns) - len(X.columns)} new features")
        return X_new
    
    def fit_scaler(self, X_train):
        """Fit scaler on training data"""
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            return X_train
        
        self.scaler.fit(X_train)
        print(f"Scaler fitted: {self.scaling_method}")
        return self
    
    def transform(self, X):
        """Transform data using fitted scaler"""
        if self.scaler is None:
            return X
        
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def fit_transform(self, X_train):
        """Fit and transform training data"""
        self.fit_scaler(X_train)
        return self.transform(X_train)
    
    def handle_outliers(self, X, y, threshold=3):
        """Remove outliers using z-score method"""
        from scipy import stats
        
        z_scores = np.abs(stats.zscore(X))
        mask = (z_scores < threshold).all(axis=1)
        
        X_clean = X[mask]
        y_clean = y[mask]
        
        removed = len(X) - len(X_clean)
        print(f"Removed {removed} outliers ({removed/len(X)*100:.2f}%)")
        
        return X_clean, y_clean
    
    def save_preprocessor(self, filepath='models/preprocessor.pkl'):
        """Save fitted preprocessor"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load_preprocessor(filepath='models/preprocessor.pkl'):
        """Load fitted preprocessor"""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def preprocess_pipeline(X_train, X_test, y_train, y_test, 
                       create_features=True, 
                       scaling_method='standard',
                       handle_outliers=False):
    """Complete preprocessing pipeline"""
    
    preprocessor = HousingPreprocessor(scaling_method=scaling_method)
    
    # Create features
    if create_features:
        X_train = preprocessor.create_features(X_train)
        X_test = preprocessor.create_features(X_test)
    
    # Handle outliers (only on training data)
    if handle_outliers:
        X_train, y_train = preprocessor.handle_outliers(X_train, y_train)
    
    # Scale data
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor


if __name__ == '__main__':
    # Test preprocessing
    from data_loader import HousingDataLoader
    
    loader = HousingDataLoader()
    df, _, _ = loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data(df)
    
    print("\n" + "="*50)
    print("Testing Preprocessing Pipeline")
    print("="*50)
    
    X_train_p, X_test_p, y_train_p, y_test_p, preprocessor = preprocess_pipeline(
        X_train, X_test, y_train, y_test,
        create_features=True,
        scaling_method='standard',
        handle_outliers=True
    )
    
    print(f"\nOriginal features: {X_train.shape[1]}")
    print(f"Processed features: {X_train_p.shape[1]}")
    print(f"Training samples: {X_train.shape[0]} -> {X_train_p.shape[0]}")
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\nPreprocessing complete!")