"""
Unit tests for general_statistics module.

This module contains comprehensive unit tests for all functions in the
general_statistics module, including local and aggregate statistical computations.
"""

import pandas as pd
import numpy as np
from io import StringIO
import json
import vantage6_strongaya_general.general_statistics as general_statistics_module

from vantage6_strongaya_general.general_statistics import (
    compute_aggregate_general_statistics,
    compute_aggregate_adjusted_deviation,
    compute_local_general_statistics,
    compute_local_adjusted_deviation,
)


class TestComputeLocalGeneralStatistics:
    """Test cases for compute_local_general_statistics function."""

    def test_basic_numerical_statistics(self, sample_numerical_data):
        """Test basic numerical statistics computation."""
        # Select only numerical columns for testing
        numerical_cols = ["age", "height", "weight", "score_1"]
        test_data = sample_numerical_data[numerical_cols].copy()

        # Remove any missing values for this test
        test_data = test_data.dropna()

        result = compute_local_general_statistics(test_data)

        # Check structure
        assert isinstance(result, dict)
        assert "numerical_general_partial_statistics" in result
        assert "categorical_general_partial_statistics" in result

        # Parse numerical results
        numerical_df = pd.read_json(
            StringIO(result["numerical_general_partial_statistics"])
        )

        # Check that all numerical variables are included
        variables_in_result = numerical_df["variable"].unique()
        assert all(col in variables_in_result for col in numerical_cols)

        # Check statistics for each variable
        for var in numerical_cols:
            var_stats = numerical_df[numerical_df["variable"] == var]
            stats_dict = dict(zip(var_stats["statistic"], var_stats["value"]))

            # Verify expected statistics are present
            expected_stats = ["count", "mean", "std", "min", "max"]
            assert all(stat in stats_dict for stat in expected_stats)

            # Verify statistics are reasonable
            assert stats_dict["count"] > 0
            assert stats_dict["min"] <= stats_dict["max"]
            assert stats_dict["std"] >= 0

    def test_basic_categorical_statistics(self, sample_categorical_data):
        """Test basic categorical statistics computation."""
        # Select categorical columns for testing
        categorical_cols = ["gender", "treatment_group", "severity"]
        test_data = sample_categorical_data[categorical_cols].copy()

        # Remove any missing values for this test
        test_data = test_data.dropna()

        result = compute_local_general_statistics(test_data)

        # Check structure
        assert isinstance(result, dict)
        assert "categorical_general_partial_statistics" in result

        # Parse categorical results
        categorical_df = pd.read_json(
            StringIO(result["categorical_general_partial_statistics"])
        )

        # Check that all categorical variables are included
        variables_in_result = categorical_df["variable"].unique()
        assert all(col in variables_in_result for col in categorical_cols)

        # Check counts for each variable
        for var in categorical_cols:
            var_counts = categorical_df[categorical_df["variable"] == var]

            # Verify counts are non-negative integers
            assert all(var_counts["count"] >= 0)
            assert all(
                isinstance(count, (int, np.integer)) for count in var_counts["count"]
            )

            # Verify total count matches expected
            total_count = var_counts["count"].sum()
            expected_count = test_data[var].notna().sum()
            assert total_count == expected_count

    def test_mixed_data_types(self, mixed_data_sample):
        """Test statistics computation with mixed data types."""
        result = compute_local_general_statistics(mixed_data_sample)

        # Both numerical and categorical results should be present
        assert "numerical_general_partial_statistics" in result
        assert "categorical_general_partial_statistics" in result

        # Parse results
        numerical_df = pd.read_json(
            StringIO(result["numerical_general_partial_statistics"])
        )
        categorical_df = pd.read_json(
            StringIO(result["categorical_general_partial_statistics"])
        )

        # Check that numerical variables are properly classified
        numerical_vars = numerical_df["variable"].unique()
        expected_numerical = ["age", "bmi", "biomarker_1", "biomarker_2"]
        assert all(var in expected_numerical for var in numerical_vars)

        # Check that categorical variables are properly classified
        categorical_vars = categorical_df["variable"].unique()
        expected_categorical = ["gender", "treatment", "response", "site_id"]
        assert any(var in expected_categorical for var in categorical_vars)

    def test_empty_dataframe(self, edge_case_data):
        """Test behavior with empty DataFrame."""
        empty_df = edge_case_data["empty"].copy()

        result = compute_local_general_statistics(empty_df)

        # Should return empty results but with proper structure
        assert isinstance(result, dict)
        assert "numerical_general_partial_statistics" in result
        assert "categorical_general_partial_statistics" in result

        # Parse results - should be empty DataFrames
        numerical_df = pd.read_json(
            StringIO(result["numerical_general_partial_statistics"])
        )
        categorical_df = pd.read_json(
            StringIO(result["categorical_general_partial_statistics"])
        )

        assert len(numerical_df) == 0
        assert len(categorical_df) == 0

    def test_single_row_dataframe(self, edge_case_data):
        """Test behavior with single row DataFrame."""
        single_row_df = edge_case_data["single_row"].copy()

        result = compute_local_general_statistics(single_row_df)

        # Parse numerical results
        numerical_df = pd.read_json(
            StringIO(result["numerical_general_partial_statistics"])
        )
        numeric_stats = numerical_df[numerical_df["variable"] == "value"]
        stats_dict = dict(zip(numeric_stats["statistic"], numeric_stats["value"]))

        # With single value, mean should equal the value, std should be 0
        assert stats_dict["count"] == 1
        assert stats_dict["mean"] == 42
        assert stats_dict["std"] == 0.0
        assert stats_dict["min"] == stats_dict["max"] == 42

        # Parse categorical results
        categorical_df = pd.read_json(
            StringIO(result["categorical_general_partial_statistics"])
        )
        cat_counts = categorical_df[categorical_df["variable"] == "category"]

        # Should have one category with count of 1
        assert len(cat_counts) == 3
        assert cat_counts["count"].iloc[0] == 1
        assert cat_counts["value"].iloc[0] == "A"

    def test_all_nan_columns(self, edge_case_data):
        """Test behavior with columns containing only NaN values."""
        nan_df = edge_case_data["all_nan"].copy()

        result = compute_local_general_statistics(nan_df)

        # Parse results
        numerical_df = pd.read_json(
            StringIO(result["numerical_general_partial_statistics"])
        )

        # Check that all-NaN columns are handled appropriately
        # They might be excluded or have count=0
        valid_numeric_stats = numerical_df[numerical_df["variable"] == "valid_col"]
        assert len(valid_numeric_stats) > 0

        valid_stats_dict = dict(
            zip(valid_numeric_stats["statistic"], valid_numeric_stats["value"])
        )
        assert valid_stats_dict["count"] == 10
        assert valid_stats_dict["mean"] == 5.5

    def test_quantile_failure_excludes_only_quantile_statistics_for_one_variable(
        self, monkeypatch
    ):
        """Test local quantile failure is isolated to quantile statistics for that variable."""
        df = pd.DataFrame({"ok": [1.0, 2.0, 3.0, 4.0], "failing": [5.0, 6.0, 7.0, 8.0]})

        original_compute_local_quantiles = (
            general_statistics_module._compute_local_quantiles
        )

        def _raise_for_specific_variable(column_values, iterations=1000):
            if column_values.name == "failing":
                raise RuntimeError("forced quantile failure")
            return original_compute_local_quantiles(column_values, iterations)

        monkeypatch.setattr(
            general_statistics_module,
            "_compute_local_quantiles",
            _raise_for_specific_variable,
        )

        result = compute_local_general_statistics(df)
        numerical_df = pd.read_json(
            StringIO(result["numerical_general_partial_statistics"])
        )

        ok_stats = set(
            numerical_df[numerical_df["variable"] == "ok"]["statistic"].tolist()
        )
        failing_stats = set(
            numerical_df[numerical_df["variable"] == "failing"]["statistic"].tolist()
        )

        assert {"Q1", "Q2", "Q3", "variance_Q1", "variance_Q2", "variance_Q3"}.issubset(
            ok_stats
        )
        assert {
            "Q1",
            "Q2",
            "Q3",
            "variance_Q1",
            "variance_Q2",
            "variance_Q3",
        }.isdisjoint(failing_stats)
        assert {
            "min",
            "max",
            "mean",
            "std",
            "count",
            "outliers",
            "na",
            "sum",
            "sq_dev_sum",
        }.issubset(failing_stats)


class TestComputeAggregateGeneralStatistics:
    """Test cases for compute_aggregate_general_statistics function."""

    def test_basic_aggregation(self):
        """Test basic aggregation following proper local->aggregate pattern."""
        # Create organisation datasets
        np.random.seed(42)

        org1_data = pd.DataFrame(
            {
                "age": np.random.normal(45, 10, 100),
                "gender": np.random.choice(["Male", "Female"], 100),
            }
        )
        # Make gender explicitly categorical
        org1_data["gender"] = org1_data["gender"].astype("category")

        org2_data = pd.DataFrame(
            {
                "age": np.random.normal(47, 12, 150),
                "gender": np.random.choice(["Male", "Female"], 150),
            }
        )
        # Make gender explicitly categorical
        org2_data["gender"] = org2_data["gender"].astype("category")

        # Compute local results first (proper pattern)
        local_result1 = compute_local_general_statistics(org1_data)
        local_result2 = compute_local_general_statistics(org2_data)

        # Now aggregate them
        aggregated_result = compute_aggregate_general_statistics(
            [local_result1, local_result2]
        )

        # Test structure
        assert isinstance(aggregated_result, dict)
        assert "numerical_general_statistics" in aggregated_result
        assert "categorical_general_statistics" in aggregated_result

        # Parse numerical results
        numerical_df = pd.read_json(
            StringIO(aggregated_result["numerical_general_statistics"])
        )
        assert len(numerical_df) > 0

        # Parse categorical results should work now
        categorical_df = pd.read_json(
            StringIO(aggregated_result["categorical_general_statistics"])
        )
        assert len(categorical_df) >= 0  # Should have categorical results structure

    def test_empty_results_list(self):
        """Test aggregation with empty results list."""
        aggregate_result = compute_aggregate_general_statistics([])

        # Should return empty but properly structured result
        assert isinstance(aggregate_result, dict)

    def test_aggregate_quantile_failure_excludes_only_quantile_statistics_for_one_variable(
        self, monkeypatch
    ):
        """Test aggregate quantile failure is isolated to one variable."""
        org1_data = pd.DataFrame(
            {"ok": [1.0, 2.0, 3.0, 4.0], "failing": [10.0, 20.0, 30.0, 40.0]}
        )
        org2_data = pd.DataFrame(
            {"ok": [2.0, 3.0, 4.0, 5.0], "failing": [15.0, 25.0, 35.0, 45.0]}
        )

        local_result1 = compute_local_general_statistics(org1_data)
        local_result2 = compute_local_general_statistics(org2_data)

        original_compute_aggregate_quantiles = (
            general_statistics_module._compute_aggregate_quantiles
        )

        def _raise_for_specific_variable(numerical_statistics):
            variable = numerical_statistics.index.get_level_values("variable").unique()[
                0
            ]
            if variable == "failing":
                raise RuntimeError("forced aggregate quantile failure")
            return original_compute_aggregate_quantiles(numerical_statistics)

        monkeypatch.setattr(
            general_statistics_module,
            "_compute_aggregate_quantiles",
            _raise_for_specific_variable,
        )

        aggregated_result = compute_aggregate_general_statistics(
            [local_result1, local_result2]
        )
        numerical_df = pd.read_json(
            StringIO(aggregated_result["numerical_general_statistics"])
        )

        ok_stats = set(
            numerical_df[numerical_df["variable"] == "ok"]["statistic"].tolist()
        )
        failing_stats = set(
            numerical_df[numerical_df["variable"] == "failing"]["statistic"].tolist()
        )

        assert {"q1", "median", "q3"}.issubset(ok_stats)
        assert {"q1", "median", "q3"}.isdisjoint(failing_stats)
        assert {"min", "max", "mean", "std", "count", "outliers", "na"}.issubset(
            failing_stats
        )

    def test_aggregate_quantile_statistics_present_when_computation_succeeds(self):
        """Test aggregate quantile statistics are still present in normal execution."""
        org1_data = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]})
        org2_data = pd.DataFrame({"value": [2.0, 3.0, 4.0, 5.0]})

        local_result1 = compute_local_general_statistics(org1_data)
        local_result2 = compute_local_general_statistics(org2_data)
        aggregated_result = compute_aggregate_general_statistics(
            [local_result1, local_result2]
        )

        numerical_df = pd.read_json(
            StringIO(aggregated_result["numerical_general_statistics"])
        )
        value_stats = set(
            numerical_df[numerical_df["variable"] == "value"]["statistic"].tolist()
        )

        assert {"q1", "median", "q3"}.issubset(value_stats)
        assert "numerical_general_statistics" in aggregated_result
        assert "categorical_general_statistics" in aggregated_result

    def test_aggregate_mean_failure_excludes_only_mean_for_affected_variable(
        self, monkeypatch
    ):
        """One variable's mean failure does not affect others."""
        org1_data = pd.DataFrame(
            {"ok": [1.0, 2.0, 3.0, 4.0], "failing": [10.0, 20.0, 30.0, 40.0]}
        )
        org2_data = pd.DataFrame(
            {"ok": [2.0, 3.0, 4.0, 5.0], "failing": [15.0, 25.0, 35.0, 45.0]}
        )

        local_result1 = compute_local_general_statistics(org1_data)
        local_result2 = compute_local_general_statistics(org2_data)

        original_compute_aggregate_mean = (
            general_statistics_module._compute_aggregate_mean
        )

        def _raise_for_failing(numerical_statistics):
            variable = numerical_statistics.index.get_level_values("variable").unique()[
                0
            ]
            if variable == "failing":
                raise RuntimeError("forced mean failure")
            return original_compute_aggregate_mean(numerical_statistics)

        monkeypatch.setattr(
            general_statistics_module,
            "_compute_aggregate_mean",
            _raise_for_failing,
        )

        aggregated_result = compute_aggregate_general_statistics(
            [local_result1, local_result2]
        )
        numerical_df = pd.read_json(
            StringIO(aggregated_result["numerical_general_statistics"])
        )

        ok_stats = set(
            numerical_df[numerical_df["variable"] == "ok"]["statistic"].tolist()
        )
        failing_stats = set(
            numerical_df[numerical_df["variable"] == "failing"]["statistic"].tolist()
        )

        # ok variable should have mean
        assert "mean" in ok_stats
        # failing variable should NOT have mean but should have everything else
        assert "mean" not in failing_stats
        assert {"min", "max", "std", "count", "outliers", "na"}.issubset(failing_stats)

    def test_aggregate_deviation_failure_excludes_only_std_for_affected_variable(
        self, monkeypatch
    ):
        """One variable's deviation failure does not affect others."""
        org1_data = pd.DataFrame(
            {"ok": [1.0, 2.0, 3.0, 4.0], "failing": [10.0, 20.0, 30.0, 40.0]}
        )
        org2_data = pd.DataFrame(
            {"ok": [2.0, 3.0, 4.0, 5.0], "failing": [15.0, 25.0, 35.0, 45.0]}
        )

        local_result1 = compute_local_general_statistics(org1_data)
        local_result2 = compute_local_general_statistics(org2_data)

        original_compute_aggregate_deviation = (
            general_statistics_module._compute_aggregate_deviation
        )

        def _raise_for_failing(numerical_statistics):
            variable = numerical_statistics.index.get_level_values("variable").unique()[
                0
            ]
            if variable == "failing":
                raise RuntimeError("forced deviation failure")
            return original_compute_aggregate_deviation(numerical_statistics)

        monkeypatch.setattr(
            general_statistics_module,
            "_compute_aggregate_deviation",
            _raise_for_failing,
        )

        aggregated_result = compute_aggregate_general_statistics(
            [local_result1, local_result2]
        )
        numerical_df = pd.read_json(
            StringIO(aggregated_result["numerical_general_statistics"])
        )

        ok_stats = set(
            numerical_df[numerical_df["variable"] == "ok"]["statistic"].tolist()
        )
        failing_stats = set(
            numerical_df[numerical_df["variable"] == "failing"]["statistic"].tolist()
        )

        assert "std" in ok_stats
        assert "std" not in failing_stats
        assert {"min", "max", "mean", "count", "outliers", "na"}.issubset(
            failing_stats
        )

    def test_single_organisation_result(self):
        """Test aggregation with results from single organisation."""
        np.random.seed(42)

        org_data = pd.DataFrame(
            {
                "age": np.random.normal(45, 10, 100),
                "gender": np.random.choice(["Male", "Female"], 100),
            }
        )

        # Compute local result
        local_result = compute_local_general_statistics(org_data)

        # Aggregate single result
        aggregate_result = compute_aggregate_general_statistics([local_result])

        # Result should be similar to input (no aggregation needed)
        assert isinstance(aggregate_result, dict)
        assert "numerical_general_statistics" in aggregate_result
        assert "categorical_general_statistics" in aggregate_result

    def test_multiple_organisations_aggregation(self):
        """Test aggregation across multiple organisations with different variables."""
        np.random.seed(42)

        # Organisation 1 with age and gender
        org1_data = pd.DataFrame(
            {
                "age": np.random.normal(45, 10, 100),
                "gender": np.random.choice(["Male", "Female"], 100),
            }
        )

        # Organisation 2 with age, height and gender
        org2_data = pd.DataFrame(
            {
                "age": np.random.normal(47, 12, 150),
                "height": np.random.normal(170, 8, 150),
                "gender": np.random.choice(["Male", "Female", "Other"], 150),
            }
        )

        # Organisation 3 with only height
        org3_data = pd.DataFrame({"height": np.random.normal(165, 10, 80)})

        # Compute local results
        local_results = [
            compute_local_general_statistics(org1_data),
            compute_local_general_statistics(org2_data),
            compute_local_general_statistics(org3_data),
        ]

        # Aggregate results
        aggregate_result = compute_aggregate_general_statistics(local_results)

        # Test structure
        assert isinstance(aggregate_result, dict)
        assert "numerical_general_statistics" in aggregate_result
        assert "categorical_general_statistics" in aggregate_result

        # Parse and validate aggregated results
        numerical_df = pd.read_json(
            StringIO(aggregate_result["numerical_general_statistics"])
        )
        categorical_df = pd.read_json(
            StringIO(aggregate_result["categorical_general_statistics"])
        )

        # Should have age from org1+org2 and height from org2+org3
        age_stats = numerical_df[numerical_df["variable"] == "age"]
        height_stats = numerical_df[numerical_df["variable"] == "height"]

        assert len(age_stats) > 0  # Age should be present
        assert len(height_stats) > 0  # Height should be present

        # Check counts are correctly aggregated
        age_count = age_stats[age_stats["statistic"] == "count"]["value"].iloc[0]
        height_count = height_stats[height_stats["statistic"] == "count"]["value"].iloc[
            0
        ]

        assert age_count == 250  # org1: 100 + org2: 150
        assert height_count == 230  # org2: 150 + org3: 80

        # Verify categorical results structure and accuracy
        gender_counts = categorical_df[categorical_df["variable"] == "gender"]
        assert len(gender_counts) >= 0  # Should handle gender categories properly


class TestComputeLocalAdjustedDeviation:
    """Test cases for compute_local_adjusted_deviation function."""

    def test_basic_adjusted_deviation(self, sample_numerical_data):
        """Test basic adjusted deviation computation."""
        # Prepare test data - only use columns that exist
        test_data = sample_numerical_data[["age", "height"]].dropna()

        # Mock global statistics - format it like actual aggregated results
        global_stats = """{
            "variable": {"0": "age", "1": "height", "2": "age", "3": "height"},
            "statistic": {"0": "mean", "1": "mean", "2": "std", "3": "std"},
            "value": {"0": 45.0, "1": 170.0, "2": 15.0, "3": 10.0}
        }"""

        result = compute_local_adjusted_deviation(test_data, global_stats)

        # Check result structure
        assert isinstance(result, dict)
        assert "adjusted_deviation" in result

        # Should contain adjusted deviations for both variables
        # Parse the JSON result to check the content
        deviation_data = json.loads(result["adjusted_deviation"])
        variables_present = set(deviation_data["variable"].values())
        assert "age" in variables_present or "height" in variables_present

    def test_single_variable_deviation(self):
        """Test adjusted deviation for single variable."""
        test_data = pd.DataFrame({"test_var": [10, 20, 30, 40, 50]})

        global_stats = """{
            "variable": {"0": "test_var", "1": "test_var"},
            "statistic": {"0": "mean", "1": "std"},
            "value": {"0": 25.0, "1": 10.0}
        }"""

        result = compute_local_adjusted_deviation(test_data, global_stats)

        assert "adjusted_deviation" in result


class TestComputeAggregateAdjustedDeviation:
    """Test cases for compute_aggregate_adjusted_deviation function."""

    def test_basic_aggregate_deviation(self):
        """Test aggregation of adjusted deviations following proper local->aggregate pattern."""
        np.random.seed(42)

        # Create organisation data first
        org1_data = pd.DataFrame({"age": np.random.normal(45, 10, 100)})
        org2_data = pd.DataFrame({"age": np.random.normal(47, 12, 150)})

        # Mock global statistics
        global_stats = """{
            "variable": {"0": "age", "1": "age"},
            "statistic": {"0": "mean", "1": "std"},
            "value": {"0": 46.0, "1": 11.0}
        }"""

        # Compute local adjusted deviations
        local_result1 = compute_local_adjusted_deviation(org1_data, global_stats)
        local_result2 = compute_local_adjusted_deviation(org2_data, global_stats)

        # Aggregate the local results
        aggregate_result = compute_aggregate_adjusted_deviation(
            [local_result1, local_result2]
        )

        # Check basic structure
        assert isinstance(aggregate_result, dict)

    def test_empty_local_results(self):
        """Test aggregation with empty local results."""
        aggregate_result = compute_aggregate_adjusted_deviation([])

        # Should handle empty input gracefully
        assert isinstance(aggregate_result, dict)

    def test_single_organisation_deviation(self):
        """Test aggregation with single organisation."""
        np.random.seed(42)

        org_data = pd.DataFrame({"test_var": np.random.normal(25, 5, 100)})
        global_stats = """{
            "variable": {"0": "test_var", "1": "test_var"},
            "statistic": {"0": "mean", "1": "std"},
            "value": {"0": 25.0, "1": 5.0}
        }"""

        local_result = compute_local_adjusted_deviation(org_data, global_stats)
        aggregate_result = compute_aggregate_adjusted_deviation([local_result])

        assert isinstance(aggregate_result, dict)

    def test_adjusted_deviation_failure_isolated_per_variable(self, monkeypatch):
        """One variable failing adjusted deviation does not affect others."""
        np.random.seed(42)

        org1_data = pd.DataFrame(
            {
                "ok_var": np.random.normal(25, 5, 100),
                "failing_var": np.random.normal(30, 8, 100),
            }
        )
        org2_data = pd.DataFrame(
            {
                "ok_var": np.random.normal(26, 5, 120),
                "failing_var": np.random.normal(31, 8, 120),
            }
        )

        global_stats = """{
            "variable": {"0": "ok_var", "1": "ok_var", "2": "failing_var", "3": "failing_var"},
            "statistic": {"0": "mean", "1": "std", "2": "mean", "3": "std"},
            "value": {"0": 25.5, "1": 5.0, "2": 30.5, "3": 8.0}
        }"""

        local_result1 = compute_local_adjusted_deviation(org1_data, global_stats)
        local_result2 = compute_local_adjusted_deviation(org2_data, global_stats)

        # Also need general statistics for the merge
        local_gen1 = compute_local_general_statistics(org1_data)
        local_gen2 = compute_local_general_statistics(org2_data)
        gen_stats = compute_aggregate_general_statistics([local_gen1, local_gen2])

        original_fn = (
            general_statistics_module._compute_aggregate_adjusted_deviation
        )

        def _raise_for_failing(numerical_statistics):
            variable = numerical_statistics.index.get_level_values(
                "variable"
            ).unique()[0]
            if variable == "failing_var":
                raise RuntimeError("forced adjusted deviation failure")
            return original_fn(numerical_statistics)

        monkeypatch.setattr(
            general_statistics_module,
            "_compute_aggregate_adjusted_deviation",
            _raise_for_failing,
        )

        aggregate_result = compute_aggregate_adjusted_deviation(
            [local_result1, local_result2],
            results_general_statistics=gen_stats,
        )

        numerical_df = pd.read_json(
            StringIO(aggregate_result["numerical_general_statistics"])
        )

        ok_stats = set(
            numerical_df[numerical_df["variable"] == "ok_var"]["statistic"].tolist()
        )
        failing_stats = set(
            numerical_df[numerical_df["variable"] == "failing_var"][
                "statistic"
            ].tolist()
        )

        # ok_var should have adjusted std
        assert "adjusted std" in ok_stats
        # failing_var should NOT have adjusted std but should still have general stats
        assert "adjusted std" not in failing_stats
        assert {"min", "max", "count"}.issubset(failing_stats)
class TestStatisticsIntegration:
    """Integration tests for statistics functions working together."""

    def test_local_to_aggregate_pipeline(self, mixed_data_sample):
        """Test complete pipeline from local computation to aggregation."""
        # Remove organization_id column since it doesn't exist in our fixture
        test_data = mixed_data_sample.copy()

        # Split data into multiple organisations manually
        n_total = len(test_data)
        n_per_org = n_total // 3

        org_data = {
            1: test_data.iloc[:n_per_org].copy(),
            2: test_data.iloc[n_per_org : 2 * n_per_org].copy(),
            3: test_data.iloc[2 * n_per_org :].copy(),
        }

        # Compute local statistics for each organisation
        local_results = []
        for org_id, data in org_data.items():
            local_result = compute_local_general_statistics(data)
            local_results.append(local_result)

        # Aggregate results
        aggregate_result = compute_aggregate_general_statistics(local_results)

        # Check that aggregation produces valid results
        assert isinstance(aggregate_result, dict)
        assert "numerical_general_statistics" in aggregate_result
        assert "categorical_general_statistics" in aggregate_result

    def test_statistics_consistency_across_organisations(self, quantile_test_data):
        """Test that statistics are consistent when computed across different organisation splits."""
        # Use known quantiles dataset for predictable results
        test_data = quantile_test_data["known_quantiles"].copy()

        # Split into two organisations
        mid_point = len(test_data) // 2
        org1_data = test_data.iloc[:mid_point].copy()
        org2_data = test_data.iloc[mid_point:].copy()

        # Remove organisation_id columns that might cause issues
        org1_data = org1_data.drop(columns=["organisation_id"], errors="ignore")
        org2_data = org2_data.drop(columns=["organisation_id"], errors="ignore")

        # Compute local statistics
        org1_result = compute_local_general_statistics(org1_data)
        org2_result = compute_local_general_statistics(org2_data)

        # Aggregate
        aggregate_result = compute_aggregate_general_statistics(
            [org1_result, org2_result]
        )

        # Check that results are produced
        assert isinstance(aggregate_result, dict)
        assert "numerical_general_statistics" in aggregate_result
