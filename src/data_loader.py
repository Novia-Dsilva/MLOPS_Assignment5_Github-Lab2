"""
Data Loader for California Housing Dataset
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pickle
import os


class HousingDataLoader:
    """Load and split California Housing dataset"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        
    def load_data(self):
        """Load California Housing dataset"""
        print("Loading California Housing dataset...")
        housing = fetch_california_housing()
        
        # Create DataFrame
        df = pd.DataFrame(
            housing.data,
            columns=housing.feature_names
        )
        df['MedHouseVal'] = housing.target
        
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"Features: {housing.feature_names}")
        
        return df, housing.feature_names, housing.target_names
    
    def split_data(self, df):
        """Split data into train and test sets"""
        X = df.drop('MedHouseVal', axis=1)
        y = df['MedHouseVal']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def save_data(self, X_train, X_test, y_train, y_test, data_dir='data'):
        """Save processed data to disk"""
        os.makedirs(data_dir, exist_ok=True)
        
        with open(f'{data_dir}/X_train.pkl', 'wb') as f:
            pickle.dump(X_train, f)
        with open(f'{data_dir}/X_test.pkl', 'wb') as f:
            pickle.dump(X_test, f)
        with open(f'{data_dir}/y_train.pkl', 'wb') as f:
            pickle.dump(y_train, f)
        with open(f'{data_dir}/y_test.pkl', 'wb') as f:
            pickle.dump(y_test, f)
            
        print(f"Data saved to {data_dir}/")
    
    def load_saved_data(self, data_dir='data'):
        """Load previously saved data"""
        with open(f'{data_dir}/X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open(f'{data_dir}/X_test.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open(f'{data_dir}/y_train.pkl', 'rb') as f:
            y_train = pickle.load(f)
        with open(f'{data_dir}/y_test.pkl', 'rb') as f:
            y_test = pickle.load(f)
            
        return X_train, X_test, y_train, y_test
    
    def get_data_statistics(self, df):
        """Get basic statistics about the dataset"""
        stats = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,
            'target_mean': df['MedHouseVal'].mean(),
            'target_std': df['MedHouseVal'].std(),
            'target_min': df['MedHouseVal'].min(),
            'target_max': df['MedHouseVal'].max(),
        }
        return stats


if __name__ == '__main__':
    # Test the data loader
    loader = HousingDataLoader()
    df, feature_names, target_names = loader.load_data()
    
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Split and save
    X_train, X_test, y_train, y_test = loader.split_data(df)
    loader.save_data(X_train, X_test, y_train, y_test)
    
    print("\nData loading and splitting complete!")