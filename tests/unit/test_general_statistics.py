"""
Unit tests for general_statistics module.

This module contains comprehensive unit tests for all functions in the
general_statistics module, including local and aggregate statistical computations.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO
import json

from vantage6_strongaya_general.general_statistics import (
    compute_aggregate_general_statistics,
    compute_aggregate_adjusted_deviation,
    compute_local_general_statistics,
    compute_local_adjusted_deviation
)


class TestComputeLocalGeneralStatistics:
    """Test cases for compute_local_general_statistics function."""
    
    def test_basic_numerical_statistics(self, sample_numerical_data):
        """Test basic numerical statistics computation."""
        # Select only numerical columns for testing
        numerical_cols = ['age', 'height', 'weight', 'score_1']
        test_data = sample_numerical_data[numerical_cols + ['organization_id']].copy()
        
        # Remove any missing values for this test
        test_data = test_data.dropna()
        
        result = compute_local_general_statistics(test_data)
        
        # Check structure
        assert isinstance(result, dict)
        assert 'numerical' in result
        assert 'categorical' in result
        
        # Parse numerical results
        numerical_df = pd.read_json(StringIO(result['numerical']))
        
        # Check that all numerical variables are included
        variables_in_result = numerical_df['variable'].unique()
        assert all(col in variables_in_result for col in numerical_cols)
        
        # Check statistics for each variable
        for var in numerical_cols:
            var_stats = numerical_df[numerical_df['variable'] == var]
            stats_dict = dict(zip(var_stats['statistic'], var_stats['value']))
            
            # Verify expected statistics are present
            expected_stats = ['count', 'mean', 'std', 'min', 'max']
            assert all(stat in stats_dict for stat in expected_stats)
            
            # Verify statistics are reasonable
            assert stats_dict['count'] > 0
            assert stats_dict['min'] <= stats_dict['max']
            assert stats_dict['std'] >= 0
    
    def test_basic_categorical_statistics(self, sample_categorical_data):
        """Test basic categorical statistics computation."""
        # Select categorical columns for testing
        categorical_cols = ['gender', 'treatment_group', 'severity']
        test_data = sample_categorical_data[categorical_cols + ['organization_id']].copy()
        
        # Remove any missing values for this test
        test_data = test_data.dropna()
        
        result = compute_local_general_statistics(test_data)
        
        # Check structure
        assert isinstance(result, dict)
        assert 'categorical' in result
        
        # Parse categorical results
        categorical_df = pd.read_json(StringIO(result['categorical']))
        
        # Check that all categorical variables are included
        variables_in_result = categorical_df['variable'].unique()
        assert all(col in variables_in_result for col in categorical_cols)
        
        # Check counts for each variable
        for var in categorical_cols:
            var_counts = categorical_df[categorical_df['variable'] == var]
            
            # Verify counts are non-negative integers
            assert all(var_counts['count'] >= 0)
            assert all(isinstance(count, (int, np.integer)) for count in var_counts['count'])
            
            # Verify total count matches expected
            total_count = var_counts['count'].sum()
            expected_count = test_data[var].notna().sum()
            assert total_count == expected_count
    
    def test_mixed_data_types(self, mixed_data_sample):
        """Test statistics computation with mixed data types."""
        result = compute_local_general_statistics(mixed_data_sample)
        
        # Both numerical and categorical results should be present
        assert 'numerical' in result
        assert 'categorical' in result
        
        # Parse results
        numerical_df = pd.read_json(StringIO(result['numerical']))
        categorical_df = pd.read_json(StringIO(result['categorical']))
        
        # Check that numerical variables are properly classified
        numerical_vars = numerical_df['variable'].unique()
        expected_numerical = ['age', 'bmi', 'biomarker_1', 'biomarker_2']
        assert all(var in expected_numerical for var in numerical_vars)
        
        # Check that categorical variables are properly classified
        categorical_vars = categorical_df['variable'].unique()
        expected_categorical = ['gender', 'treatment', 'response', 'site_id']
        assert any(var in expected_categorical for var in categorical_vars)
    
    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        result = compute_local_general_statistics(empty_df)
        
        # Should return empty results but with proper structure
        assert isinstance(result, dict)
        assert 'numerical' in result
        assert 'categorical' in result
        
        # Parse results - should be empty DataFrames
        numerical_df = pd.read_json(StringIO(result['numerical']))
        categorical_df = pd.read_json(StringIO(result['categorical']))
        
        assert len(numerical_df) == 0
        assert len(categorical_df) == 0
    
    def test_single_row_dataframe(self):
        """Test behavior with single row DataFrame."""
        single_row_df = pd.DataFrame({
            'numeric_col': [42.5],
            'categorical_col': ['Category_A'],
            'organization_id': [1]
        })
        
        result = compute_local_general_statistics(single_row_df)
        
        # Parse numerical results
        numerical_df = pd.read_json(StringIO(result['numerical']))
        numeric_stats = numerical_df[numerical_df['variable'] == 'numeric_col']
        stats_dict = dict(zip(numeric_stats['statistic'], numeric_stats['value']))
        
        # With single value, mean should equal the value, std should be 0
        assert stats_dict['count'] == 1
        assert stats_dict['mean'] == 42.5
        assert stats_dict['std'] == 0
        assert stats_dict['min'] == stats_dict['max'] == 42.5
        
        # Parse categorical results
        categorical_df = pd.read_json(StringIO(result['categorical']))
        cat_counts = categorical_df[categorical_df['variable'] == 'categorical_col']
        
        # Should have one category with count of 1
        assert len(cat_counts) == 1
        assert cat_counts['count'].iloc[0] == 1
        assert cat_counts['value'].iloc[0] == 'Category_A'
    
    def test_all_nan_columns(self):
        """Test behavior with columns containing only NaN values."""
        nan_df = pd.DataFrame({
            'all_nan_numeric': [np.nan, np.nan, np.nan],
            'all_nan_categorical': [None, None, None],
            'valid_col': [1, 2, 3],
            'organization_id': [1, 1, 1]
        })
        
        result = compute_local_general_statistics(nan_df)
        
        # Parse results
        numerical_df = pd.read_json(StringIO(result['numerical']))
        categorical_df = pd.read_json(StringIO(result['categorical']))
        
        # Check that all-NaN columns are handled appropriately
        # They might be excluded or have count=0
        valid_numeric_stats = numerical_df[numerical_df['variable'] == 'valid_col']
        assert len(valid_numeric_stats) > 0
        
        valid_stats_dict = dict(zip(valid_numeric_stats['statistic'], valid_numeric_stats['value']))
        assert valid_stats_dict['count'] == 3
        assert valid_stats_dict['mean'] == 2.0
    
    def test_with_stratification(self, mixed_data_sample):
        """Test statistics computation with stratification variables."""
        stratification_dict = {
            'gender': ['Male', 'Female']
        }
        
        result = compute_local_general_statistics(mixed_data_sample, stratification_dict)
        
        # Should still return proper structure
        assert isinstance(result, dict)
        assert 'numerical' in result
        assert 'categorical' in result
        
        # Results should be stratified (exact format depends on implementation)
        numerical_df = pd.read_json(StringIO(result['numerical']))
        categorical_df = pd.read_json(StringIO(result['categorical']))
        
        # Should have results for both strata
        assert len(numerical_df) > 0
        assert len(categorical_df) > 0


class TestComputeAggregateGeneralStatistics:
    """Test cases for compute_aggregate_general_statistics function."""
    
    def test_basic_aggregation(self):
        """Test basic aggregation of results from multiple organizations."""
        # Create mock results from different organizations
        org1_numerical = pd.DataFrame({
            'variable': ['age', 'age', 'height', 'height'],
            'statistic': ['count', 'mean', 'count', 'mean'],
            'value': [100, 45.0, 100, 170.0]
        })
        
        org1_categorical = pd.DataFrame({
            'variable': ['gender', 'gender'],
            'value': ['Male', 'Female'],
            'count': [60, 40]
        })
        
        org2_numerical = pd.DataFrame({
            'variable': ['age', 'age', 'height', 'height'],
            'statistic': ['count', 'mean', 'count', 'mean'],
            'value': [150, 50.0, 150, 175.0]
        })
        
        org2_categorical = pd.DataFrame({
            'variable': ['gender', 'gender'],
            'value': ['Male', 'Female'],
            'count': [80, 70]
        })
        
        # Create federated results structure
        results = [
            {
                'numerical': org1_numerical.to_json(),
                'categorical': org1_categorical.to_json()
            },
            {
                'numerical': org2_numerical.to_json(),
                'categorical': org2_categorical.to_json()
            }
        ]
        
        aggregate_result = compute_aggregate_general_statistics(results)
        
        # Check structure
        assert isinstance(aggregate_result, dict)
        assert 'numerical' in aggregate_result
        assert 'categorical' in aggregate_result
        
        # Parse aggregated results
        agg_numerical = pd.read_json(StringIO(aggregate_result['numerical']))
        agg_categorical = pd.read_json(StringIO(aggregate_result['categorical']))
        
        # Check aggregated counts
        age_count = agg_numerical[(agg_numerical['variable'] == 'age') & 
                                 (agg_numerical['statistic'] == 'count')]['value'].iloc[0]
        assert age_count == 250  # 100 + 150
        
        height_count = agg_numerical[(agg_numerical['variable'] == 'height') & 
                                   (agg_numerical['statistic'] == 'count')]['value'].iloc[0]
        assert height_count == 250  # 100 + 150
        
        # Check aggregated categorical counts
        male_count = agg_categorical[(agg_categorical['variable'] == 'gender') & 
                                   (agg_categorical['value'] == 'Male')]['count'].iloc[0]
        assert male_count == 140  # 60 + 80
        
        female_count = agg_categorical[(agg_categorical['variable'] == 'gender') & 
                                     (agg_categorical['value'] == 'Female')]['count'].iloc[0]
        assert female_count == 110  # 40 + 70
    
    def test_empty_results_list(self):
        """Test aggregation with empty results list."""
        aggregate_result = compute_aggregate_general_statistics([])
        
        # Should return empty but properly structured result
        assert isinstance(aggregate_result, dict)
        assert 'numerical' in aggregate_result
        assert 'categorical' in aggregate_result
    
    def test_single_organization_result(self):
        """Test aggregation with results from single organization."""
        single_numerical = pd.DataFrame({
            'variable': ['age', 'age'],
            'statistic': ['count', 'mean'],
            'value': [100, 45.0]
        })
        
        single_categorical = pd.DataFrame({
            'variable': ['gender', 'gender'],
            'value': ['Male', 'Female'],
            'count': [60, 40]
        })
        
        results = [{
            'numerical': single_numerical.to_json(),
            'categorical': single_categorical.to_json()
        }]
        
        aggregate_result = compute_aggregate_general_statistics(results)
        
        # Result should be similar to input (no aggregation needed)
        assert isinstance(aggregate_result, dict)
        assert 'numerical' in aggregate_result
        assert 'categorical' in aggregate_result
        
        # Parse results
        agg_numerical = pd.read_json(StringIO(aggregate_result['numerical']))
        agg_categorical = pd.read_json(StringIO(aggregate_result['categorical']))
        
        # Check that values are preserved
        age_count = agg_numerical[(agg_numerical['variable'] == 'age') & 
                                 (agg_numerical['statistic'] == 'count')]['value'].iloc[0]
        assert age_count == 100
    
    def test_inconsistent_variables_across_organizations(self):
        """Test aggregation when organizations have different variables."""
        # Org1 has age, Org2 has height
        org1_numerical = pd.DataFrame({
            'variable': ['age', 'age'],
            'statistic': ['count', 'mean'],
            'value': [100, 45.0]
        })
        
        org2_numerical = pd.DataFrame({
            'variable': ['height', 'height'],
            'statistic': ['count', 'mean'],
            'value': [150, 175.0]
        })
        
        results = [
            {'numerical': org1_numerical.to_json(), 'categorical': pd.DataFrame().to_json()},
            {'numerical': org2_numerical.to_json(), 'categorical': pd.DataFrame().to_json()}
        ]
        
        aggregate_result = compute_aggregate_general_statistics(results)
        
        # Should handle different variables gracefully
        agg_numerical = pd.read_json(StringIO(aggregate_result['numerical']))
        
        # Both variables should be present in final result
        variables_in_result = agg_numerical['variable'].unique()
        assert 'age' in variables_in_result
        assert 'height' in variables_in_result


class TestComputeLocalAdjustedDeviation:
    """Test cases for compute_local_adjusted_deviation function."""
    
    def test_basic_adjusted_deviation(self, sample_numerical_data):
        """Test basic adjusted deviation computation."""
        # Prepare test data
        test_data = sample_numerical_data[['age', 'height', 'organization_id']].dropna()
        
        # Mock global statistics
        global_stats = {
            'age': {'mean': 45.0, 'std': 15.0},
            'height': {'mean': 170.0, 'std': 10.0}
        }
        
        result = compute_local_adjusted_deviation(test_data, global_stats)
        
        # Check result structure
        assert isinstance(result, dict)
        
        # Should contain adjusted deviations for both variables
        assert 'age' in result
        assert 'height' in result
        
        # Each variable should have proper statistics
        for var in ['age', 'height']:
            var_result = result[var]
            assert 'adjusted_deviation' in var_result
            assert isinstance(var_result['adjusted_deviation'], (int, float))
    
    def test_single_variable_deviation(self):
        """Test adjusted deviation for single variable."""
        test_data = pd.DataFrame({
            'test_var': [10, 20, 30, 40, 50],
            'organization_id': [1, 1, 1, 1, 1]
        })
        
        global_stats = {
            'test_var': {'mean': 25.0, 'std': 10.0}
        }
        
        result = compute_local_adjusted_deviation(test_data, global_stats)
        
        assert 'test_var' in result
        assert 'adjusted_deviation' in result['test_var']
        
        # Deviation should be calculated properly
        deviation = result['test_var']['adjusted_deviation']
        assert isinstance(deviation, (int, float))
        assert deviation >= 0  # Deviation should be non-negative


class TestComputeAggregateAdjustedDeviation:
    """Test cases for compute_aggregate_adjusted_deviation function."""
    
    def test_basic_aggregate_deviation(self):
        """Test aggregation of adjusted deviations."""
        # Mock local deviation results from multiple organizations
        local_results = [
            {
                'age': {'adjusted_deviation': 2.5, 'count': 100},
                'height': {'adjusted_deviation': 1.8, 'count': 100}
            },
            {
                'age': {'adjusted_deviation': 3.2, 'count': 150},
                'height': {'adjusted_deviation': 2.1, 'count': 150}
            }
        ]
        
        aggregate_result = compute_aggregate_adjusted_deviation(local_results)
        
        # Check structure
        assert isinstance(aggregate_result, dict)
        assert 'age' in aggregate_result
        assert 'height' in aggregate_result
        
        # Check that aggregated deviations are computed
        for var in ['age', 'height']:
            var_result = aggregate_result[var]
            assert 'aggregate_adjusted_deviation' in var_result
            assert 'total_count' in var_result
            
            # Total count should be sum of local counts
            if var == 'age':
                assert var_result['total_count'] == 250  # 100 + 150
    
    def test_empty_local_results(self):
        """Test aggregation with empty local results."""
        aggregate_result = compute_aggregate_adjusted_deviation([])
        
        # Should handle empty input gracefully
        assert isinstance(aggregate_result, dict)
    
    def test_single_organization_deviation(self):
        """Test aggregation with single organization."""
        local_results = [{
            'test_var': {'adjusted_deviation': 1.5, 'count': 100}
        }]
        
        aggregate_result = compute_aggregate_adjusted_deviation(local_results)
        
        assert 'test_var' in aggregate_result
        assert aggregate_result['test_var']['total_count'] == 100
        assert 'aggregate_adjusted_deviation' in aggregate_result['test_var']


class TestStatisticsIntegration:
    """Integration tests for statistics functions working together."""
    
    def test_local_to_aggregate_pipeline(self, mixed_data_sample):
        """Test complete pipeline from local computation to aggregation."""
        # Split data into multiple organizations
        org_data = {}
        unique_orgs = mixed_data_sample['organization_id'].unique()
        
        for org_id in unique_orgs[:3]:  # Use first 3 organizations
            org_data[org_id] = mixed_data_sample[
                mixed_data_sample['organization_id'] == org_id
            ].copy()
        
        # Compute local statistics for each organization
        local_results = []
        for org_id, data in org_data.items():
            local_result = compute_local_general_statistics(data)
            local_results.append(local_result)
        
        # Aggregate results
        aggregate_result = compute_aggregate_general_statistics(local_results)
        
        # Check that aggregation produces valid results
        assert isinstance(aggregate_result, dict)
        assert 'numerical' in aggregate_result
        assert 'categorical' in aggregate_result
        
        # Parse aggregated results
        agg_numerical = pd.read_json(StringIO(aggregate_result['numerical']))
        agg_categorical = pd.read_json(StringIO(aggregate_result['categorical']))
        
        # Should have some results
        assert len(agg_numerical) > 0 or len(agg_categorical) > 0
    
    def test_statistics_consistency_across_organizations(self, quantile_test_data):
        """Test that statistics are consistent when computed across different organization splits."""
        # Use known quantiles dataset for predictable results
        test_data = quantile_test_data['known_quantiles'].copy()
        
        # Split into two organizations
        mid_point = len(test_data) // 2
        org1_data = test_data.iloc[:mid_point].copy()
        org2_data = test_data.iloc[mid_point:].copy()
        
        org1_data['organization_id'] = 1
        org2_data['organization_id'] = 2
        
        # Compute local statistics
        org1_result = compute_local_general_statistics(org1_data)
        org2_result = compute_local_general_statistics(org2_data)
        
        # Aggregate
        aggregate_result = compute_aggregate_general_statistics([org1_result, org2_result])
        
        # Parse results
        agg_numerical = pd.read_json(StringIO(aggregate_result['numerical']))
        
        # Check that total count matches original data
        value_count = agg_numerical[(agg_numerical['variable'] == 'value') & 
                                   (agg_numerical['statistic'] == 'count')]['value'].iloc[0]
        assert value_count == len(test_data)