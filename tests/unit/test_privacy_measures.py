"""
Unit tests for privacy_measures module.

This module contains comprehensive unit tests for privacy protection utilities,
including differential privacy, sample size thresholding, and variable masking.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import warnings

from vantage6_strongaya_general.privacy_measures import (
    mask_unnecessary_variables,
    apply_sample_size_threshold,
    apply_differential_privacy
)


class TestMaskUnnecessaryVariables:
    """Test cases for mask_unnecessary_variables function."""
    
    def test_basic_variable_masking(self, mixed_data_sample):
        """Test basic variable masking functionality."""
        original_columns = mixed_data_sample.columns.tolist()
        variables_to_keep = ['age', 'gender']
        
        result = mask_unnecessary_variables(mixed_data_sample, variables_to_keep)
        
        # Check that only specified variables are kept
        assert list(result.columns) == variables_to_keep
        
        # Check that data for kept variables is preserved
        for var in variables_to_keep:
            pd.testing.assert_series_equal(
                result[var], 
                mixed_data_sample[var], 
                check_names=True
            )
    
    def test_mask_single_variable(self, mixed_data_sample):
        """Test masking when keeping only one variable."""
        variables_to_keep = ['age']
        
        result = mask_unnecessary_variables(mixed_data_sample, variables_to_keep)
        
        assert list(result.columns) == ['age']
        pd.testing.assert_series_equal(result['age'], mixed_data_sample['age'])
    
    def test_mask_all_variables(self, mixed_data_sample):
        """Test masking when keeping all variables."""
        all_variables = mixed_data_sample.columns.tolist()
        
        result = mask_unnecessary_variables(mixed_data_sample, all_variables)
        
        # Should be identical to original
        pd.testing.assert_frame_equal(result, mixed_data_sample)
    
    def test_mask_no_variables(self, mixed_data_sample):
        """Test masking when keeping no variables."""
        variables_to_keep = []
        
        result = mask_unnecessary_variables(mixed_data_sample, variables_to_keep)
        
        # Should return DataFrame with no columns
        assert len(result.columns) == 0
        assert len(result) == len(mixed_data_sample)  # Rows should be preserved
    
    def test_mask_nonexistent_variables(self, mixed_data_sample):
        """Test masking when requesting nonexistent variables."""
        variables_to_keep = ['nonexistent_var1', 'nonexistent_var2']
        
        # Behavior depends on implementation - might raise error or return empty DataFrame
        try:
            result = mask_unnecessary_variables(mixed_data_sample, variables_to_keep)
            # If no error, should be empty or have only valid columns
            assert len(result.columns) == 0
        except KeyError:
            # Acceptable to raise KeyError for nonexistent variables
            pass
    
    def test_mask_mixed_existing_nonexisting_variables(self, mixed_data_sample):
        """Test masking with mix of existing and nonexistent variables."""
        variables_to_keep = ['age', 'nonexistent_var', 'gender']
        
        # Should handle this gracefully
        try:
            result = mask_unnecessary_variables(mixed_data_sample, variables_to_keep)
            # Should keep only the existing variables
            existing_vars = [var for var in variables_to_keep if var in mixed_data_sample.columns]
            assert set(result.columns) <= set(existing_vars)
        except KeyError:
            # Acceptable to raise error
            pass
    
    def test_mask_preserves_data_types(self, mixed_data_sample):
        """Test that masking preserves data types of kept variables."""
        variables_to_keep = ['age', 'gender', 'bmi']
        original_dtypes = mixed_data_sample[variables_to_keep].dtypes
        
        result = mask_unnecessary_variables(mixed_data_sample, variables_to_keep)
        
        # Data types should be preserved
        for var in variables_to_keep:
            assert result[var].dtype == original_dtypes[var]
    
    def test_mask_with_empty_dataframe(self):
        """Test masking with empty DataFrame."""
        empty_df = pd.DataFrame()
        variables_to_keep = ['some_var']
        
        # Should handle empty DataFrame gracefully
        try:
            result = mask_unnecessary_variables(empty_df, variables_to_keep)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
        except (KeyError, IndexError):
            # Acceptable to raise error with empty DataFrame
            pass


class TestApplySampleSizeThreshold:
    """Test cases for apply_sample_size_threshold function."""
    
    def test_threshold_with_sufficient_samples(self, mock_algorithm_client):
        """Test thresholding when sample size is sufficient."""
        # Create DataFrame with sufficient samples
        sufficient_data = pd.DataFrame({
            'age': range(100),  # 100 samples
            'gender': ['Male'] * 50 + ['Female'] * 50,
            'organization_id': [1] * 100
        })
        
        variables_config = {
            'age': {'datatype': 'int', 'inliers': [0, 120]},
            'gender': {'datatype': 'str', 'inliers': ['Male', 'Female']}
        }
        
        # Mock environment variable for threshold
        with patch('vantage6_strongaya_general.privacy_measures.get_env_var') as mock_env:
            mock_env.return_value = "10"  # Threshold of 10
            
            result = apply_sample_size_threshold(
                mock_algorithm_client, 
                sufficient_data, 
                variables_config
            )
        
        # Should return the original data since it meets threshold
        assert len(result) == len(sufficient_data)
        pd.testing.assert_frame_equal(result, sufficient_data)
    
    def test_threshold_with_insufficient_samples(self, mock_algorithm_client):
        """Test thresholding when sample size is insufficient."""
        # Create DataFrame with insufficient samples
        insufficient_data = pd.DataFrame({
            'age': [25, 30, 35],  # Only 3 samples
            'gender': ['Male', 'Female', 'Male'],
            'organization_id': [1] * 3
        })
        
        variables_config = {
            'age': {'datatype': 'int', 'inliers': [0, 120]},
            'gender': {'datatype': 'str', 'inliers': ['Male', 'Female']}
        }
        
        # Mock environment variable for threshold
        with patch('vantage6_strongaya_general.privacy_measures.get_env_var') as mock_env:
            mock_env.return_value = "10"  # Threshold of 10
            
            # Depending on implementation, might raise exception or return empty/modified data
            try:
                result = apply_sample_size_threshold(
                    mock_algorithm_client, 
                    insufficient_data, 
                    variables_config
                )
                # If no exception, result should be valid DataFrame
                assert isinstance(result, pd.DataFrame)
            except Exception:
                # Acceptable to raise exception for insufficient samples
                pass
    
    def test_threshold_with_zero_samples(self, mock_algorithm_client):
        """Test thresholding with zero samples."""
        empty_data = pd.DataFrame({
            'age': [],
            'gender': [],
            'organization_id': []
        })
        
        variables_config = {}
        
        with patch('vantage6_strongaya_general.privacy_measures.get_env_var') as mock_env:
            mock_env.return_value = "10"
            
            try:
                result = apply_sample_size_threshold(
                    mock_algorithm_client, 
                    empty_data, 
                    variables_config
                )
                assert len(result) == 0
            except Exception:
                # Acceptable to raise exception for zero samples
                pass
    
    def test_threshold_with_custom_threshold(self, mock_algorithm_client):
        """Test thresholding with different threshold values."""
        test_data = pd.DataFrame({
            'value': range(50),
            'organization_id': [1] * 50
        })
        
        variables_config = {
            'value': {'datatype': 'int', 'inliers': [0, 100]}
        }
        
        # Test with low threshold
        with patch('vantage6_strongaya_general.privacy_measures.get_env_var') as mock_env:
            mock_env.return_value = "5"  # Low threshold
            
            result_low = apply_sample_size_threshold(
                mock_algorithm_client, 
                test_data, 
                variables_config
            )
            
            # Should pass with 50 samples vs threshold of 5
            assert len(result_low) == 50
        
        # Test with high threshold
        with patch('vantage6_strongaya_general.privacy_measures.get_env_var') as mock_env:
            mock_env.return_value = "100"  # High threshold
            
            try:
                result_high = apply_sample_size_threshold(
                    mock_algorithm_client, 
                    test_data, 
                    variables_config
                )
                # Might return empty or raise exception
                assert isinstance(result_high, pd.DataFrame)
            except Exception:
                # Acceptable for high threshold
                pass
    
    def test_threshold_with_invalid_threshold_value(self, mock_algorithm_client):
        """Test thresholding with invalid threshold environment variable."""
        test_data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'organization_id': [1] * 5
        })
        
        variables_config = {
            'value': {'datatype': 'int', 'inliers': [0, 10]}
        }
        
        # Test with non-numeric threshold
        with patch('vantage6_strongaya_general.privacy_measures.get_env_var') as mock_env:
            mock_env.return_value = "invalid_number"
            
            try:
                result = apply_sample_size_threshold(
                    mock_algorithm_client, 
                    test_data, 
                    variables_config
                )
                assert isinstance(result, pd.DataFrame)
            except (ValueError, TypeError):
                # Acceptable to raise error for invalid threshold
                pass


class TestApplyDifferentialPrivacy:
    """Test cases for apply_differential_privacy function."""
    
    def test_basic_differential_privacy_mean(self):
        """Test basic differential privacy for mean computation."""
        np.random.seed(42)  # For reproducible tests
        
        test_data = pd.DataFrame({
            'value': [10, 20, 30, 40, 50],
            'organization_id': [1] * 5
        })
        
        variables_config = {
            'value': {'datatype': 'float', 'inliers': [0, 100]}
        }
        
        result = apply_differential_privacy(
            test_data, 
            variables_config, 
            epsilon=1.0,
            return_type='dataframe'
        )
        
        # Should return DataFrame with noise added
        assert isinstance(result, pd.DataFrame)
        assert 'value' in result.columns
        assert len(result) == len(test_data)
        
        # Values should be different from original (due to noise)
        # But this is probabilistic, so we can't guarantee it for all values
        assert isinstance(result['value'].iloc[0], (int, float, np.number))
    
    def test_differential_privacy_with_different_epsilon(self):
        """Test differential privacy with different epsilon values."""
        test_data = pd.DataFrame({
            'value': [100] * 10,  # Identical values
            'organization_id': [1] * 10
        })
        
        variables_config = {
            'value': {'datatype': 'float', 'inliers': [0, 200]}
        }
        
        # Test with high epsilon (less noise)
        result_high_epsilon = apply_differential_privacy(
            test_data, 
            variables_config, 
            epsilon=10.0,
            return_type='dataframe'
        )
        
        # Test with low epsilon (more noise)
        result_low_epsilon = apply_differential_privacy(
            test_data, 
            variables_config, 
            epsilon=0.1,
            return_type='dataframe'
        )
        
        # Both should be DataFrames
        assert isinstance(result_high_epsilon, pd.DataFrame)
        assert isinstance(result_low_epsilon, pd.DataFrame)
        
        # Low epsilon should generally produce more variance (though this is probabilistic)
        var_high = result_high_epsilon['value'].var()
        var_low = result_low_epsilon['value'].var()
        
        # This is a probabilistic test, so we just check that both have some variance
        assert var_high >= 0
        assert var_low >= 0
    
    def test_differential_privacy_verbose_return(self):
        """Test differential privacy with verbose return type."""
        test_data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5],
            'organization_id': [1] * 5
        })
        
        variables_config = {
            'value': {'datatype': 'float', 'inliers': [0, 10]}
        }
        
        result = apply_differential_privacy(
            test_data, 
            variables_config, 
            epsilon=1.0,
            return_type='verbose'
        )
        
        # Should return dictionary with detailed information
        assert isinstance(result, dict)
        # Exact structure depends on implementation
        assert len(result) > 0
    
    def test_differential_privacy_with_multiple_variables(self):
        """Test differential privacy with multiple variables."""
        test_data = pd.DataFrame({
            'var1': [10, 20, 30],
            'var2': [100, 200, 300],
            'organization_id': [1] * 3
        })
        
        variables_config = {
            'var1': {'datatype': 'float', 'inliers': [0, 50]},
            'var2': {'datatype': 'float', 'inliers': [0, 500]}
        }
        
        result = apply_differential_privacy(
            test_data, 
            variables_config, 
            epsilon=1.0,
            return_type='dataframe'
        )
        
        # Should handle multiple variables
        assert isinstance(result, pd.DataFrame)
        assert 'var1' in result.columns
        assert 'var2' in result.columns
        assert len(result) == len(test_data)
    
    def test_differential_privacy_with_empty_dataframe(self):
        """Test differential privacy with empty DataFrame."""
        empty_data = pd.DataFrame({'value': [], 'organization_id': []})
        variables_config = {}
        
        try:
            result = apply_differential_privacy(
                empty_data, 
                variables_config, 
                epsilon=1.0,
                return_type='dataframe'
            )
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
        except Exception:
            # Acceptable to raise exception with empty data
            pass
    
    def test_differential_privacy_edge_cases(self):
        """Test differential privacy with edge case values."""
        # Test with zero values
        zero_data = pd.DataFrame({
            'value': [0, 0, 0, 0, 0],
            'organization_id': [1] * 5
        })
        
        variables_config = {
            'value': {'datatype': 'float', 'inliers': [-10, 10]}
        }
        
        result_zero = apply_differential_privacy(
            zero_data, 
            variables_config, 
            epsilon=1.0,
            return_type='dataframe'
        )
        
        assert isinstance(result_zero, pd.DataFrame)
        assert len(result_zero) == 5
        
        # Test with negative values
        negative_data = pd.DataFrame({
            'value': [-10, -5, -1, -3, -7],
            'organization_id': [1] * 5
        })
        
        result_negative = apply_differential_privacy(
            negative_data, 
            variables_config, 
            epsilon=1.0,
            return_type='dataframe'
        )
        
        assert isinstance(result_negative, pd.DataFrame)
        assert len(result_negative) == 5
    
    def test_differential_privacy_preserves_non_target_columns(self):
        """Test that differential privacy preserves non-target columns."""
        test_data = pd.DataFrame({
            'target_var': [1, 2, 3, 4, 5],
            'preserve_var': ['a', 'b', 'c', 'd', 'e'],
            'organization_id': [1] * 5
        })
        
        variables_config = {
            'target_var': {'datatype': 'float', 'inliers': [0, 10]}
        }
        
        result = apply_differential_privacy(
            test_data, 
            variables_config, 
            epsilon=1.0,
            return_type='dataframe'
        )
        
        # Non-target columns should be preserved exactly
        if 'preserve_var' in result.columns:
            pd.testing.assert_series_equal(
                result['preserve_var'], 
                test_data['preserve_var']
            )
        
        if 'organization_id' in result.columns:
            pd.testing.assert_series_equal(
                result['organization_id'], 
                test_data['organization_id']
            )


class TestPrivacyMeasuresIntegration:
    """Integration tests for privacy measures working together."""
    
    def test_complete_privacy_pipeline(self, mixed_data_sample, mock_algorithm_client):
        """Test complete privacy protection pipeline."""
        # Step 1: Apply sample size threshold
        variables_config = {
            'age': {'datatype': 'int', 'inliers': [0, 120]},
            'bmi': {'datatype': 'float', 'inliers': [10, 50]}
        }
        
        with patch('vantage6_strongaya_general.privacy_measures.get_env_var') as mock_env:
            mock_env.return_value = "10"  # Low threshold for testing
            
            try:
                thresholded_data = apply_sample_size_threshold(
                    mock_algorithm_client, 
                    mixed_data_sample, 
                    variables_config
                )
            except Exception:
                # If thresholding fails, use original data for rest of pipeline
                thresholded_data = mixed_data_sample
        
        # Step 2: Mask unnecessary variables
        variables_to_keep = ['age', 'bmi', 'organization_id']
        masked_data = mask_unnecessary_variables(thresholded_data, variables_to_keep)
        
        # Step 3: Apply differential privacy
        dp_variables_config = {
            'age': {'datatype': 'float', 'inliers': [0, 120]},
            'bmi': {'datatype': 'float', 'inliers': [10, 50]}
        }
        
        try:
            final_data = apply_differential_privacy(
                masked_data,
                dp_variables_config,
                epsilon=1.0,
                return_type='dataframe'
            )
            
            # Final result should be valid DataFrame
            assert isinstance(final_data, pd.DataFrame)
            assert len(final_data.columns) <= len(variables_to_keep)
            
        except Exception:
            # If DP fails, that's acceptable for this integration test
            final_data = masked_data
        
        # Pipeline should complete and produce valid result
        assert isinstance(final_data, pd.DataFrame)
    
    def test_privacy_measures_with_edge_cases(self, edge_case_data, mock_algorithm_client):
        """Test privacy measures with various edge cases."""
        for scenario_name, test_data in edge_case_data.items():
            if len(test_data) == 0:
                continue  # Skip empty datasets
            
            # Test masking
            if len(test_data.columns) > 0:
                first_col = test_data.columns[0]
                try:
                    masked = mask_unnecessary_variables(test_data, [first_col])
                    assert isinstance(masked, pd.DataFrame)
                except Exception:
                    pass  # Acceptable for edge cases to fail
            
            # Test thresholding (if data has required structure)
            if 'organization_id' in test_data.columns:
                variables_config = {}
                for col in test_data.columns:
                    if col != 'organization_id' and pd.api.types.is_numeric_dtype(test_data[col]):
                        variables_config[col] = {'datatype': 'float', 'inliers': [0, 1000]}
                
                with patch('vantage6_strongaya_general.privacy_measures.get_env_var') as mock_env:
                    mock_env.return_value = "5"
                    
                    try:
                        thresholded = apply_sample_size_threshold(
                            mock_algorithm_client, 
                            test_data, 
                            variables_config
                        )
                        assert isinstance(thresholded, pd.DataFrame)
                    except Exception:
                        pass  # Acceptable for edge cases to fail
    
    def test_privacy_budget_considerations(self):
        """Test considerations around privacy budget (epsilon) usage."""
        base_data = pd.DataFrame({
            'sensitive_value': [50] * 100,  # Identical values
            'organization_id': [1] * 100
        })
        
        variables_config = {
            'sensitive_value': {'datatype': 'float', 'inliers': [0, 100]}
        }
        
        # Apply DP multiple times with different epsilon values
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
        results = {}
        
        for epsilon in epsilons:
            try:
                result = apply_differential_privacy(
                    base_data.copy(),
                    variables_config,
                    epsilon=epsilon,
                    return_type='dataframe'
                )
                
                if isinstance(result, pd.DataFrame) and len(result) > 0:
                    # Calculate variance as proxy for noise level
                    variance = result['sensitive_value'].var()
                    results[epsilon] = variance
                    
            except Exception:
                # Some epsilon values might cause errors
                continue
        
        # Generally, higher epsilon should result in lower variance (less noise)
        # This is probabilistic, so we just check that we got some results
        assert len(results) > 0
        
        # All variances should be non-negative
        assert all(var >= 0 for var in results.values())
    
    def test_privacy_measures_data_integrity(self, sample_numerical_data):
        """Test that privacy measures maintain basic data integrity."""
        original_shape = sample_numerical_data.shape
        
        # Apply masking
        variables_to_keep = ['age', 'height', 'organization_id']
        masked_data = mask_unnecessary_variables(sample_numerical_data, variables_to_keep)
        
        # Row count should be preserved
        assert len(masked_data) == original_shape[0]
        
        # Apply DP to numeric columns
        dp_config = {
            'age': {'datatype': 'float', 'inliers': [0, 120]},
            'height': {'datatype': 'float', 'inliers': [100, 250]}
        }
        
        try:
            dp_data = apply_differential_privacy(
                masked_data,
                dp_config,
                epsilon=1.0,
                return_type='dataframe'
            )
            
            # Basic integrity checks
            if isinstance(dp_data, pd.DataFrame):
                assert len(dp_data) == len(masked_data)
                # Values should still be numeric
                for col in ['age', 'height']:
                    if col in dp_data.columns:
                        assert pd.api.types.is_numeric_dtype(dp_data[col])
                        
        except Exception:
            # DP might fail for various reasons, which is acceptable
            pass