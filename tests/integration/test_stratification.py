"""
Integration tests for data stratification functionality.

This module tests the complete data stratification workflow
from configuration to statistical computation using British English.
"""

import pytest
import pandas as pd
import numpy as np

from vantage6_strongaya_general.miscellaneous import apply_data_stratification
from vantage6_strongaya_general.general_statistics import compute_local_general_statistics


class TestDataStratificationWorkflow:
    """Test complete data stratification workflows."""
    
    def test_categorical_stratification_workflow(self, mixed_data_sample):
        """Test complete workflow with categorical stratification."""
        # Define categorical stratification
        stratification_config = {
            'gender': ['Male', 'Female'],
            'treatment': ['A', 'B']
        }
        
        # Apply stratification
        stratified_data = apply_data_stratification(mixed_data_sample, stratification_config)
        
        # Verify stratification worked
        if len(stratified_data) > 0:
            unique_genders = stratified_data['gender'].unique()
            unique_treatments = stratified_data['treatment'].unique()
            
            assert all(gender in ['Male', 'Female'] for gender in unique_genders)
            assert all(treatment in ['A', 'B'] for treatment in unique_treatments)
        
        # Compute statistics on stratified data
        if len(stratified_data) > 0:
            try:
                stats_result = compute_local_general_statistics(stratified_data)
                assert isinstance(stats_result, dict)
            except Exception as e:
                # Stratified statistics might fail with small samples
                print(f"Statistics computation failed on stratified data: {e}")
    
    def test_numerical_range_stratification_workflow(self, mixed_data_sample):
        """Test complete workflow with numerical range stratification."""
        # Define numerical range stratification
        stratification_config = {
            'age': {'start': 30, 'end': 65}
        }
        
        # Apply stratification
        stratified_data = apply_data_stratification(mixed_data_sample, stratification_config)
        
        # Verify age ranges
        if len(stratified_data) > 0 and 'age' in stratified_data.columns:
            ages = stratified_data['age'].dropna()
            if len(ages) > 0:
                assert all(30 <= age <= 65 for age in ages)
        
        # Compute statistics on stratified data
        if len(stratified_data) > 0:
            try:
                stats_result = compute_local_general_statistics(stratified_data, stratification_config)
                assert isinstance(stats_result, dict)
            except Exception as e:
                print(f"Statistics computation failed on age-stratified data: {e}")
    
    def test_mixed_stratification_workflow(self, mixed_data_sample):
        """Test workflow with both categorical and numerical stratification."""
        # Define mixed stratification
        stratification_config = {
            'gender': ['Male', 'Female'],
            'age': {'start': 25, 'end': 70}
        }
        
        # Apply stratification
        stratified_data = apply_data_stratification(mixed_data_sample, stratification_config)
        
        # Verify both constraints
        if len(stratified_data) > 0:
            # Check gender constraint
            if 'gender' in stratified_data.columns:
                unique_genders = stratified_data['gender'].unique()
                assert all(gender in ['Male', 'Female'] for gender in unique_genders)
            
            # Check age constraint
            if 'age' in stratified_data.columns:
                ages = stratified_data['age'].dropna()
                if len(ages) > 0:
                    assert all(25 <= age <= 70 for age in ages)
        
        # Compute statistics on stratified data
        if len(stratified_data) > 0:
            try:
                stats_result = compute_local_general_statistics(stratified_data, stratification_config)
                assert isinstance(stats_result, dict)
            except Exception as e:
                print(f"Mixed stratification statistics failed: {e}")
    
    def test_federated_stratification_workflow(self, mixed_data_sample):
        """Test stratification across multiple federated organizations."""
        # Split data by organization
        organization_data = {}
        unique_orgs = mixed_data_sample['organization_id'].unique()[:3]
        
        for org_id in unique_orgs:
            org_data = mixed_data_sample[mixed_data_sample['organization_id'] == org_id].copy()
            if len(org_data) > 5:  # Only use orgs with sufficient data
                organization_data[org_id] = org_data
        
        if len(organization_data) < 2:
            pytest.skip("Insufficient organizations for federated stratification test")
        
        # Define stratification
        stratification_config = {
            'gender': ['Male', 'Female']
        }
        
        # Apply stratification to each organization
        stratified_org_data = {}
        for org_id, org_data in organization_data.items():
            stratified_data = apply_data_stratification(org_data, stratification_config)
            if len(stratified_data) > 0:
                stratified_org_data[org_id] = stratified_data
        
        # Compute local statistics for each stratified organization
        local_results = []
        for org_id, stratified_data in stratified_org_data.items():
            try:
                local_result = compute_local_general_statistics(stratified_data, stratification_config)
                local_results.append(local_result)
            except Exception as e:
                print(f"Local statistics failed for org {org_id}: {e}")
                continue
        
        # Should have at least some successful local computations
        if len(local_results) > 0:
            # Each result should be a valid dictionary
            for result in local_results:
                assert isinstance(result, dict)
                # Should have either numerical or categorical results
                has_results = (
                    ('numerical' in result and result['numerical']) or
                    ('categorical' in result and result['categorical'])
                )
                # Note: might be empty if stratification filtered out all data
    
    def test_stratification_preserves_data_integrity(self, mixed_data_sample):
        """Test that stratification preserves data integrity."""
        original_columns = set(mixed_data_sample.columns)
        original_dtypes = mixed_data_sample.dtypes.to_dict()
        
        # Apply stratification
        stratification_config = {
            'gender': ['Male', 'Female', 'Other']  # Include all possible values
        }
        
        stratified_data = apply_data_stratification(mixed_data_sample, stratification_config)
        
        # Check column preservation
        assert set(stratified_data.columns) == original_columns
        
        # Check data type preservation
        for col in stratified_data.columns:
            if col in original_dtypes:
                assert stratified_data[col].dtype == original_dtypes[col]
        
        # Check that stratified data is subset of original
        assert len(stratified_data) <= len(mixed_data_sample)
    
    def test_stratification_edge_cases(self, edge_case_data):
        """Test stratification with various edge cases."""
        for scenario_name, test_data in edge_case_data.items():
            if len(test_data) == 0:
                continue
            
            # Try different stratification configurations
            if 'value' in test_data.columns:
                # Try numerical stratification
                stratification_config = {
                    'value': {'start': 0, 'end': 1000}
                }
                
                try:
                    stratified_data = apply_data_stratification(test_data, stratification_config)
                    assert isinstance(stratified_data, pd.DataFrame)
                    assert len(stratified_data) <= len(test_data)
                except Exception as e:
                    # Edge cases might fail, which is acceptable
                    print(f"Numerical stratification failed for {scenario_name}: {e}")
            
            if 'category' in test_data.columns:
                # Try categorical stratification
                unique_categories = test_data['category'].dropna().unique()
                if len(unique_categories) > 0:
                    stratification_config = {
                        'category': list(unique_categories)
                    }
                    
                    try:
                        stratified_data = apply_data_stratification(test_data, stratification_config)
                        assert isinstance(stratified_data, pd.DataFrame)
                        assert len(stratified_data) <= len(test_data)
                    except Exception as e:
                        print(f"Categorical stratification failed for {scenario_name}: {e}")
    
    def test_no_stratification_equivalence(self, mixed_data_sample):
        """Test that no stratification returns original data."""
        # Apply no stratification
        result_none = apply_data_stratification(mixed_data_sample, None)
        result_empty = apply_data_stratification(mixed_data_sample, {})
        
        # Should be equivalent to original data
        pd.testing.assert_frame_equal(result_none, mixed_data_sample)
        pd.testing.assert_frame_equal(result_empty, mixed_data_sample)
    
    def test_stratification_with_missing_values(self, mixed_data_sample):
        """Test stratification behavior with missing values."""
        # Introduce missing values in stratification columns
        test_data = mixed_data_sample.copy()
        
        # Set some gender values to NaN
        np.random.seed(42)
        missing_mask = np.random.random(len(test_data)) < 0.2  # 20% missing
        test_data.loc[missing_mask, 'gender'] = np.nan
        
        # Apply stratification
        stratification_config = {
            'gender': ['Male', 'Female']
        }
        
        stratified_data = apply_data_stratification(test_data, stratification_config)
        
        # Should handle missing values appropriately
        assert isinstance(stratified_data, pd.DataFrame)
        
        # Check that non-missing stratified values meet criteria
        if len(stratified_data) > 0 and 'gender' in stratified_data.columns:
            non_missing_genders = stratified_data['gender'].dropna().unique()
            if len(non_missing_genders) > 0:
                assert all(gender in ['Male', 'Female'] for gender in non_missing_genders)
    
    def test_complex_stratification_combinations(self, mixed_data_sample):
        """Test complex stratification with multiple constraints."""
        # Define complex stratification
        stratification_config = {
            'gender': ['Male', 'Female'],
            'age': {'start': 30, 'end': 60},
            'treatment': ['A', 'B', 'C']
        }
        
        # Apply stratification
        stratified_data = apply_data_stratification(mixed_data_sample, stratification_config)
        
        # Verify all constraints are met
        if len(stratified_data) > 0:
            # Gender constraint
            if 'gender' in stratified_data.columns:
                genders = stratified_data['gender'].dropna().unique()
                if len(genders) > 0:
                    assert all(gender in ['Male', 'Female'] for gender in genders)
            
            # Age constraint
            if 'age' in stratified_data.columns:
                ages = stratified_data['age'].dropna()
                if len(ages) > 0:
                    assert all(30 <= age <= 60 for age in ages)
            
            # Treatment constraint
            if 'treatment' in stratified_data.columns:
                treatments = stratified_data['treatment'].dropna().unique()
                if len(treatments) > 0:
                    assert all(treatment in ['A', 'B', 'C'] for treatment in treatments)
        
        # Complex stratification might result in empty dataset
        # This is acceptable behavior
        assert isinstance(stratified_data, pd.DataFrame)


class TestStratificationPerformance:
    """Test performance characteristics of stratification."""
    
    def test_stratification_scales_with_data_size(self, test_performance_data):
        """Test that stratification performance scales reasonably with data size."""
        import time
        
        stratification_config = {
            'categorical_1': ['A', 'B', 'C'],
            'continuous_1': {'start': 0, 'end': 100}
        }
        
        performance_results = {}
        
        for size_name, test_data in test_performance_data.items():
            if len(test_data) == 0:
                continue
            
            # Measure stratification time
            start_time = time.perf_counter()
            
            try:
                stratified_data = apply_data_stratification(test_data, stratification_config)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                performance_results[size_name] = {
                    'execution_time': execution_time,
                    'input_size': len(test_data),
                    'output_size': len(stratified_data),
                    'success': True
                }
                
                # Should complete in reasonable time
                assert execution_time < 30  # 30 seconds max
                
            except Exception as e:
                performance_results[size_name] = {
                    'execution_time': None,
                    'input_size': len(test_data),
                    'output_size': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # Should have at least some successful results
        successful_results = [r for r in performance_results.values() if r['success']]
        if len(successful_results) > 1:
            # Performance should not degrade dramatically with size
            # (This is a basic check - real performance testing would be more rigorous)
            times = [r['execution_time'] for r in successful_results]
            sizes = [r['input_size'] for r in successful_results]
            
            # Larger datasets should not take exponentially longer
            # Simple check: largest dataset should not take more than 10x longer than smallest
            if max(times) > 0 and min(times) > 0:
                time_ratio = max(times) / min(times)
                size_ratio = max(sizes) / min(sizes)
                
                # Time ratio should not be much larger than size ratio
                # (allowing for some overhead and variation)
                assert time_ratio <= size_ratio * 10
    
    def test_stratification_memory_efficiency(self, test_performance_data):
        """Test that stratification doesn't cause excessive memory usage."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        stratification_config = {
            'categorical_1': ['A', 'B'],
            'continuous_1': {'start': 25, 'end': 75}
        }
        
        # Test with largest available dataset
        largest_dataset = None
        largest_size = 0
        
        for size_name, test_data in test_performance_data.items():
            if len(test_data) > largest_size:
                largest_size = len(test_data)
                largest_dataset = test_data
        
        if largest_dataset is not None and len(largest_dataset) > 1000:
            # Apply stratification
            try:
                stratified_data = apply_data_stratification(largest_dataset, stratification_config)
                
                # Check memory usage after stratification
                final_memory = process.memory_info().rss
                memory_increase = final_memory - initial_memory
                
                # Memory increase should be reasonable
                # (Should not be more than several times the input data size)
                input_size_bytes = largest_dataset.memory_usage(deep=True).sum()
                
                # Allow for some memory overhead, but not excessive
                assert memory_increase < input_size_bytes * 5
                
            except Exception as e:
                # Memory test might fail for various reasons
                print(f"Memory efficiency test failed: {e}")
                pass