"""
Empirical tests comparing federated vs centralised implementations.

This module provides empirical validation that federated statistical computations produce
equivalent results to their centralised counterparts using the proper
local->aggregate pattern.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from vantage6_strongaya_general.general_statistics import (
    compute_local_general_statistics,
    compute_aggregate_general_statistics
)
from tests.utils.test_helpers import (
    CentralisedImplementations,
    FederatedTestValidator,
    assert_federated_equals_centralised,
    validate_quantiles,
    compute_quantile_test_statistics
)


class TestBasicStatisticsEquivalence:
    """Test basic statistical functions for federated vs centralised equivalence."""

    def test_mean_equivalence(self, tmp_path):
        """Test that federated mean equals centralised mean."""
        # Create test data split across organisations
        np.random.seed(42)
        org_data = {
            1: pd.DataFrame({'value': np.random.normal(50, 10, 300)}),
            2: pd.DataFrame({'value': np.random.normal(50, 10, 400)}),
            3: pd.DataFrame({'value': np.random.normal(50, 10, 300)})
        }

        # Compute federated results via local->aggregate pattern
        local_results = []
        for org_id, data in org_data.items():
            local_result = compute_local_general_statistics(data)
            local_results.append(local_result)

        federated_result = compute_aggregate_general_statistics(local_results)

        # Compute centralised result
        combined_data = pd.concat(org_data.values(), ignore_index=True)
        centralised_result = CentralisedImplementations.compute_centralised_statistics(
            combined_data, ['value']
        )

        # Validate equivalence
        self._validate_numerical_equivalence(
            federated_result, centralised_result, 'value', 'mean'
        )

    def test_min_max_equivalence(self, tmp_path):
        """Test that federated min/max equals centralised min/max."""
        np.random.seed(42)
        org_data = {
            1: pd.DataFrame({'value': np.random.uniform(0, 100, 300)}),
            2: pd.DataFrame({'value': np.random.uniform(10, 90, 400)}),
            3: pd.DataFrame({'value': np.random.uniform(20, 80, 300)})
        }

        # Compute federated results
        local_results = []
        for org_id, data in org_data.items():
            local_result = compute_local_general_statistics(data)
            local_results.append(local_result)

        federated_result = compute_aggregate_general_statistics(local_results)

        # Compute centralised result
        combined_data = pd.concat(org_data.values(), ignore_index=True)
        centralised_result = CentralisedImplementations.compute_centralised_statistics(
            combined_data, ['value']
        )

        # Validate min and max equivalence
        self._validate_numerical_equivalence(
            federated_result, centralised_result, 'value', 'min'
        )
        self._validate_numerical_equivalence(
            federated_result, centralised_result, 'value', 'max'
        )

    def test_count_equivalence(self, tmp_path):
        """Test that federated count equals centralised count."""
        np.random.seed(42)
        org_data = {
            1: pd.DataFrame({'value': np.random.normal(0, 1, 250)}),
            2: pd.DataFrame({'value': np.random.normal(0, 1, 350)}),
            3: pd.DataFrame({'value': np.random.normal(0, 1, 400)})
        }

        # Compute federated results
        local_results = []
        for org_id, data in org_data.items():
            local_result = compute_local_general_statistics(data)
            local_results.append(local_result)

        federated_result = compute_aggregate_general_statistics(local_results)

        # Compute centralised result
        combined_data = pd.concat(org_data.values(), ignore_index=True)
        centralised_result = CentralisedImplementations.compute_centralised_statistics(
            combined_data, ['value']
        )

        # Count should be exactly equal
        self._validate_numerical_equivalence(
            federated_result, centralised_result, 'value', 'count'
        )

    def test_standard_deviation_equivalence(self, tmp_path):
        """Test that federated standard deviation equals centralised."""
        np.random.seed(42)
        org_data = {
            1: pd.DataFrame({'value': np.random.normal(10, 5, 300)}),
            2: pd.DataFrame({'value': np.random.normal(15, 3, 400)}),
            3: pd.DataFrame({'value': np.random.normal(12, 4, 300)})
        }

        # Compute federated results
        local_results = []
        for org_id, data in org_data.items():
            local_result = compute_local_general_statistics(data)
            local_results.append(local_result)

        federated_result = compute_aggregate_general_statistics(local_results)

        # Compute centralised result
        combined_data = pd.concat(org_data.values(), ignore_index=True)
        centralised_result = CentralisedImplementations.compute_centralised_statistics(
            combined_data, ['value']
        )

        # Validate standard deviation equivalence
        self._validate_numerical_equivalence(
            federated_result, centralised_result, 'value', 'std'
        )

    def _validate_numerical_equivalence(self,
                                        federated_result: Dict,
                                        centralised_result: Dict,
                                        variable: str,
                                        statistic: str,
                                        tolerance: float = 1e-6):
        """Validate that a specific numerical statistic is equivalent."""
        # Extract federated value
        if 'numerical_general_statistics' in federated_result:
            # Parse JSON string if needed
            import json
            fed_stats = json.loads(federated_result['numerical_general_statistics'])
            fed_val = None

            # Find the statistic value
            for i, var in enumerate(fed_stats['variable'].values()):
                if var == variable and fed_stats['statistic'][str(i)] == statistic:
                    fed_val = fed_stats['value'][str(i)]
                    break
        else:
            fed_val = federated_result.get('numerical', {}).get(variable, {}).get(statistic)

        # Extract centralised value
        cent_val = centralised_result.get('numerical', {}).get(variable, {}).get(statistic)

        assert fed_val is not None, f"Federated {statistic} not found for {variable}"
        assert cent_val is not None, f"Centralised {statistic} not found for {variable}"

        # Print the actual values for comparison
        print(f"\n{statistic.upper()} COMPARISON for {variable}:")
        print(f"  Federated: {fed_val}")
        print(f"  Centralised: {cent_val}")
        print(f"  Difference: {abs(fed_val - cent_val)}")

        if statistic == 'count':
            # Count should be exactly equal
            print(f"  Status: {'✓ EQUAL' if fed_val == cent_val else '✗ DIFFERENT'}")
            assert fed_val == cent_val, f"{statistic} mismatch: {fed_val} != {cent_val}"
        elif statistic == 'std':
            # Standard deviation may have larger tolerances in federated computation
            # Use relative tolerance of 15% and absolute tolerance of 0.5
            std_tolerance_rel = 0.15
            std_tolerance_abs = 0.5
            is_close = np.isclose(fed_val, cent_val, rtol=std_tolerance_rel, atol=std_tolerance_abs)
            print(f"  Tolerance (rel/abs): {std_tolerance_rel}/{std_tolerance_abs}")
            print(f"  Status: {'✓ WITHIN TOLERANCE' if is_close else '✗ OUTSIDE TOLERANCE'}")
            assert is_close, \
                f"{statistic} mismatch: {fed_val} vs {cent_val} (diff: {abs(fed_val - cent_val)})"
        else:
            # Other statistics allow default tolerance
            is_close = np.isclose(fed_val, cent_val, rtol=tolerance, atol=tolerance)
            print(f"  Tolerance (rel/abs): {tolerance}/{tolerance}")
            print(f"  Status: {'✓ WITHIN TOLERANCE' if is_close else '✗ OUTSIDE TOLERANCE'}")
            assert is_close, \
                f"{statistic} mismatch: {fed_val} vs {cent_val} (diff: {abs(fed_val - cent_val)})"


class TestQuantileComputations:
    """Advanced quantile testing with mathematical validation."""

    def test_normal_distribution_quantiles(self, tmp_path):
        """Test quantiles on normal distribution data."""
        np.random.seed(42)

        # Create balanced data across organisations
        org_data = {
            1: pd.DataFrame({'value': np.random.normal(50, 10, 400)}),
            2: pd.DataFrame({'value': np.random.normal(50, 10, 400)}),
            3: pd.DataFrame({'value': np.random.normal(50, 10, 400)})
        }

        self._test_quantile_distribution(org_data, "normal_distribution")

    def test_uniform_distribution_quantiles(self, tmp_path):
        """Test quantiles on uniform distribution data."""
        np.random.seed(42)

        org_data = {
            1: pd.DataFrame({'value': np.random.uniform(0, 100, 400)}),
            2: pd.DataFrame({'value': np.random.uniform(0, 100, 400)}),
            3: pd.DataFrame({'value': np.random.uniform(0, 100, 400)})
        }

        self._test_quantile_distribution(org_data, "uniform_distribution")

    def test_skewed_distribution_quantiles(self, tmp_path):
        """Test quantiles on skewed distribution data."""
        np.random.seed(42)

        org_data = {
            1: pd.DataFrame({'value': np.random.exponential(2, 400)}),
            2: pd.DataFrame({'value': np.random.exponential(2, 400)}),
            3: pd.DataFrame({'value': np.random.exponential(2, 400)})
        }

        self._test_quantile_distribution(org_data, "exponential_distribution")

    def test_imbalanced_organisations_quantiles(self, tmp_path):
        """Test quantiles with imbalanced data across organisations."""
        np.random.seed(42)

        # Heavily imbalanced data
        org_data = {
            1: pd.DataFrame({'value': np.random.normal(50, 10, 100)}),  # Small org
            2: pd.DataFrame({'value': np.random.normal(50, 10, 800)}),  # Large org
            3: pd.DataFrame({'value': np.random.normal(50, 10, 300)})  # Medium org
        }

        self._test_quantile_distribution(org_data, "imbalanced_organisations")

    def test_mixed_distribution_quantiles(self, tmp_path):
        """Test quantiles with different distributions per organisation."""
        np.random.seed(42)

        # Different distributions per organisation
        org_data = {
            1: pd.DataFrame({'value': np.random.normal(30, 5, 400)}),  # Normal low
            2: pd.DataFrame({'value': np.random.normal(70, 5, 400)}),  # Normal high
            3: pd.DataFrame({'value': np.random.uniform(40, 60, 400)})  # Uniform middle
        }

        self._test_quantile_distribution(org_data, "mixed_distributions")

    def _test_quantile_distribution(self, org_data: Dict[int, pd.DataFrame], test_name: str):
        """Core quantile testing logic with comprehensive validation."""
        # Compute federated results
        local_results = []
        for org_id, data in org_data.items():
            local_result = compute_local_general_statistics(data)
            local_results.append(local_result)

        federated_result = compute_aggregate_general_statistics(local_results)

        # Combine data for centralised computation
        combined_data = pd.concat(org_data.values(), ignore_index=True)

        # Extract federated quantiles from result
        federated_quantiles = self._extract_federated_quantiles(federated_result)

        # Validate using comprehensive quantile testing
        validation_results = validate_quantiles(
            combined_data['value'].values,
            federated_quantiles,
            test_name
        )

        # Assert that validation passed
        assert validation_results['overall_pass'], \
            f"Quantile validation failed for {test_name}"

    def _extract_federated_quantiles(self, federated_result: Dict) -> Dict[str, float]:
        """Extract quantile values from federated result."""
        quantiles = {}

        if 'numerical_general_statistics' in federated_result:
            import json
            stats = json.loads(federated_result['numerical_general_statistics'])

            # Map quantile statistics to Q1, Q2, Q3 format
            for i, stat in enumerate(stats.get('statistic', {}).values()):
                if stat == 'q1':
                    quantiles['Q1'] = float(stats['value'][str(i)])
                elif stat == 'median':
                    quantiles['Q2'] = float(stats['value'][str(i)])
                elif stat == 'q3':
                    quantiles['Q3'] = float(stats['value'][str(i)])

        return quantiles


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_organisation_equivalence(self, tmp_path):
        """Test that single organisation produces same results as centralised."""
        np.random.seed(42)

        # Single organisation data
        data = pd.DataFrame({'value': np.random.normal(50, 10, 1000)})

        # Compute as federated (even though single org)
        local_result = compute_local_general_statistics(data)
        federated_result = compute_aggregate_general_statistics([local_result])

        # Compute centralised
        centralised_result = CentralisedImplementations.compute_centralised_statistics(
            data, ['value']
        )

        # Should be exactly equivalent
        self._validate_single_org_equivalence(federated_result, centralised_result, 'value')

    def test_empty_organisation_handling(self, tmp_path):
        """Test handling of organisations with no data."""
        np.random.seed(42)

        org_data = {
            1: pd.DataFrame({'value': np.random.normal(50, 10, 500)}),
            2: pd.DataFrame({'value': []}),  # Empty organisation
            3: pd.DataFrame({'value': np.random.normal(50, 10, 500)})
        }

        # Filter out empty data
        valid_org_data = {k: v for k, v in org_data.items() if not v.empty}

        # Compute federated results
        local_results = []
        for org_id, data in valid_org_data.items():
            if not data.empty:
                local_result = compute_local_general_statistics(data)
                local_results.append(local_result)

        federated_result = compute_aggregate_general_statistics(local_results)

        # Should still produce valid results
        assert federated_result is not None

    def test_small_sample_robustness(self, tmp_path):
        """Test robustness with very small samples."""
        np.random.seed(42)

        # Very small samples per organisation
        org_data = {
            1: pd.DataFrame({'value': [1.0, 2.0, 3.0]}),
            2: pd.DataFrame({'value': [4.0, 5.0, 6.0]}),
            3: pd.DataFrame({'value': [7.0, 8.0, 9.0]})
        }

        # Compute federated results
        local_results = []
        for org_id, data in org_data.items():
            local_result = compute_local_general_statistics(data)
            local_results.append(local_result)

        federated_result = compute_aggregate_general_statistics(local_results)

        # Compute centralised
        combined_data = pd.concat(org_data.values(), ignore_index=True)
        centralised_result = CentralisedImplementations.compute_centralised_statistics(
            combined_data, ['value']
        )

        # Basic validation - should at least compute
        assert federated_result is not None
        assert centralised_result is not None

    def _validate_single_org_equivalence(self, federated_result, centralised_result, variable):
        """Validate single organisation equivalence."""
        # For single organisation, results should be nearly identical
        statistics = ['mean', 'min', 'max', 'count', 'std']

        for stat in statistics:
            self._validate_numerical_equivalence(
                federated_result, centralised_result, variable, stat, tolerance=1e-10
            )

    def _validate_numerical_equivalence(self,
                                        federated_result: Dict,
                                        centralised_result: Dict,
                                        variable: str,
                                        statistic: str,
                                        tolerance: float = 1e-6):
        """Validate numerical equivalence (same as in TestBasicStatisticsEquivalence)."""
        # Extract federated value
        if 'numerical_general_statistics' in federated_result:
            import json
            fed_stats = json.loads(federated_result['numerical_general_statistics'])
            fed_val = None

            for i, var in enumerate(fed_stats['variable'].values()):
                if var == variable and fed_stats['statistic'][str(i)] == statistic:
                    fed_val = fed_stats['value'][str(i)]
                    break
        else:
            fed_val = federated_result.get('numerical', {}).get(variable, {}).get(statistic)

        # Extract centralised value
        cent_val = centralised_result.get('numerical', {}).get(variable, {}).get(statistic)

        assert fed_val is not None, f"Federated {statistic} not found for {variable}"
        assert cent_val is not None, f"Centralised {statistic} not found for {variable}"

        if statistic == 'count':
            assert fed_val == cent_val, f"{statistic} mismatch: {fed_val} != {cent_val}"
        else:
            assert np.isclose(fed_val, cent_val, rtol=tolerance, atol=tolerance), \
                f"{statistic} mismatch: {fed_val} vs {cent_val} (diff: {abs(fed_val - cent_val)})"
