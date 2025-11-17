"""
Unit tests using unittest framework for data loading functionality
"""
import unittest
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath('../src'))

from src.data_loader import HousingDataLoader


class TestHousingDataLoaderUnittest(unittest.TestCase):
    """Test suite for HousingDataLoader using unittest"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.loader = HousingDataLoader(test_size=0.2, random_state=42)
    
    def tearDown(self):
        """Clean up after each test method"""
        self.loader = None
    
    def test_loader_initialization(self):
        """Test that loader initializes with correct parameters"""
        self.assertEqual(self.loader.test_size, 0.2)
        self.assertEqual(self.loader.random_state, 42)
    
    def test_load_data_returns_dataframe(self):
        """Test data loading returns DataFrame"""
        df, feature_names, target_names = self.loader.load_data()
        
        # Check DataFrame is returned
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        
        # Check shape
        self.assertEqual(df.shape[1], 9)  # 8 features + 1 target
        
        # Check target column exists
        self.assertIn('MedHouseVal', df.columns)
    
    def test_feature_names(self):
        """Test that correct feature names are returned"""
        df, feature_names, target_names = self.loader.load_data()
        
        # Check number of features
        self.assertEqual(len(feature_names), 8)
        
        # Check specific features exist
        expected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                           'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        for feature in expected_features:
            self.assertIn(feature, feature_names)
    
    def test_data_types(self):
        """Test that all columns are numeric"""
        df, _, _ = self.loader.load_data()
        
        # All columns should be numeric
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        total_cols = df.shape[1]
        
        self.assertEqual(numeric_cols, total_cols)
    
    def test_split_data_shapes(self):
        """Test data splitting produces correct shapes"""
        df, _, _ = self.loader.load_data()
        X_train, X_test, y_train, y_test = self.loader.split_data(df)
        
        # Check splits are not empty
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)
        self.assertGreater(len(y_train), 0)
        self.assertGreater(len(y_test), 0)
        
        # Check feature count
        self.assertEqual(X_train.shape[1], 8)
        self.assertEqual(X_test.shape[1], 8)
        
        # Check that train and test sizes match
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
    
    def test_split_ratio(self):
        """Test that split ratio is approximately correct"""
        df, _, _ = self.loader.load_data()
        X_train, X_test, _, _ = self.loader.split_data(df)
        
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        
        # Should be around 0.2 (with some tolerance)
        self.assertGreater(test_ratio, 0.15)
        self.assertLess(test_ratio, 0.25)
    
    def test_no_data_leakage(self):
        """Test that train and test sets don't overlap"""
        df, _, _ = self.loader.load_data()
        X_train, X_test, _, _ = self.loader.split_data(df)
        
        # Convert to sets for comparison
        train_set = set(map(tuple, X_train.values))
        test_set = set(map(tuple, X_test.values))
        
        # Check no overlap
        intersection = train_set.intersection(test_set)
        self.assertEqual(len(intersection), 0)
    
    def test_data_statistics(self):
        """Test data statistics calculation"""
        df, _, _ = self.loader.load_data()
        stats = self.loader.get_data_statistics(df)
        
        # Check all required keys are present
        required_keys = ['n_samples', 'n_features', 'target_mean', 
                        'target_std', 'target_min', 'target_max']
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        # Check values are reasonable
        self.assertGreater(stats['n_samples'], 0)
        self.assertEqual(stats['n_features'], 8)
        self.assertGreater(stats['target_mean'], 0)
        self.assertGreater(stats['target_std'], 0)
    
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
    
    def test_no_missing_values(self):
        """Test that loaded data has no missing values"""
        df, _, _ = self.loader.load_data()
        
        # Check for NaN values
        self.assertFalse(df.isnull().any().any())
    
    def test_target_values_positive(self):
        """Test that target values are positive (house prices)"""
        df, _, _ = self.loader.load_data()
        
        # All target values should be positive
        self.assertTrue((df['MedHouseVal'] > 0).all())


class TestHousingDataLoaderEdgeCases(unittest.TestCase):
    """Test edge cases for HousingDataLoader"""
    
    def test_different_test_sizes(self):
        """Test different test size parameters"""
        for test_size in [0.1, 0.2, 0.3, 0.5]:
            loader = HousingDataLoader(test_size=test_size)
            df, _, _ = loader.load_data()
            X_train, X_test, _, _ = loader.split_data(df)
            
            total = len(X_train) + len(X_test)
            actual_ratio = len(X_test) / total
            
            # Allow 5% tolerance
            self.assertAlmostEqual(actual_ratio, test_size, delta=0.05)
    
    def test_different_random_states(self):
        """Test that different random states give different splits"""
        loader1 = HousingDataLoader(random_state=42)
        loader2 = HousingDataLoader(random_state=123)
        
        df1, _, _ = loader1.load_data()
        df2, _, _ = loader2.load_data()
        
        X_train1, _, _, _ = loader1.split_data(df1)
        X_train2, _, _, _ = loader2.split_data(df2)
        
        # Splits should be different
        with self.assertRaises(AssertionError):
            pd.testing.assert_frame_equal(X_train1, X_train2)


def suite():
    """Create test suite"""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestHousingDataLoaderUnittest))
    suite.addTest(unittest.makeSuite(TestHousingDataLoaderEdgeCases))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())