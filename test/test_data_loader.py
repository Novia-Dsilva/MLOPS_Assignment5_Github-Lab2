"""
Unit tests for data loading functionality
"""
import pytest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('../src'))

from src.data_loader import HousingDataLoader


class TestHousingDataLoader:
    """Test suite for HousingDataLoader class"""
    
    @pytest.fixture
    def loader(self):
        """Create a HousingDataLoader instance"""
        return HousingDataLoader(test_size=0.2, random_state=42)
    
    def test_loader_initialization(self, loader):
        """Test that loader initializes with correct parameters"""
        assert loader.test_size == 0.2
        assert loader.random_state == 42
    
    def test_load_data(self, loader):
        """Test data loading returns correct format"""
        df, feature_names, target_names = loader.load_data()
        
        # Check DataFrame is not empty
        assert not df.empty
        
        # Check expected number of features (8 original + 1 target)
        assert df.shape[1] == 9
        
        # Check target column exists
        assert 'MedHouseVal' in df.columns
        
        # Check feature names
        assert len(feature_names) == 8
        assert 'MedInc' in feature_names
        assert 'HouseAge' in feature_names
    
    def test_data_types(self, loader):
        """Test that loaded data has correct types"""
        df, _, _ = loader.load_data()
        
        # All columns should be numeric
        assert df.select_dtypes(include=[np.number]).shape[1] == df.shape[1]
    
    def test_split_data(self, loader):
        """Test data splitting functionality"""
        df, _, _ = loader.load_data()
        X_train, X_test, y_train, y_test = loader.split_data(df)
        
        # Check splits are not empty
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        
        # Check split ratio is approximately correct
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert 0.15 < test_ratio < 0.25  # Should be around 0.2
        
        # Check feature count
        assert X_train.shape[1] == 8
        assert X_test.shape[1] == 8
    
    def test_no_data_leakage(self, loader):
        """Test that train and test sets don't overlap"""
        df, _, _ = loader.load_data()
        X_train, X_test, _, _ = loader.split_data(df)
        
        # Convert to sets of tuples for comparison
        train_set = set(map(tuple, X_train.values))
        test_set = set(map(tuple, X_test.values))
        
        # Check no overlap
        assert len(train_set.intersection(test_set)) == 0
    
    def test_data_statistics(self, loader):
        """Test data statistics calculation"""
        df, _, _ = loader.load_data()
        stats = loader.get_data_statistics(df)
        
        # Check all required statistics are present
        assert 'n_samples' in stats
        assert 'n_features' in stats
        assert 'target_mean' in stats
        assert 'target_std' in stats
        assert 'target_min' in stats
        assert 'target_max' in stats
        
        # Check values are reasonable
        assert stats['n_samples'] > 0
        assert stats['n_features'] == 8
        assert stats['target_mean'] > 0
        assert stats['target_std'] > 0
    
    def test_reproducibility(self):
        """Test that same random_state gives same split"""
        loader1 = HousingDataLoader(random_state=42)
        loader2 = HousingDataLoader(random_state=42)
        
        df1, _, _ = loader1.load_data()
        df2, _, _ = loader2.load_data()
        
        X_train1, X_test1, y_train1, y_test1 = loader1.split_data(df1)
        X_train2, X_test2, y_train2, y_test2 = loader2.split_data(df2)
        
        # Check that splits are identical
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])