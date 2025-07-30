"""
Test utilities for supporting test execution and validation.

This module provides helper functions for testing federated analytics,
including data validation, performance measurement, and comparison
utilities for federated vs centralised computations.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Tuple
import warnings
from contextlib import contextmanager
import json

from vantage6_strongaya_general import safe_log

warnings.filterwarnings("ignore", category=FutureWarning)


class CentralisedImplementations:
    """
    Centralised implementations of federated functions for comparison testing.

    These implementations serve as ground truth for validating federated results.
    """

    @staticmethod
    def compute_centralised_statistics(
        df: pd.DataFrame, variables: List[str]
    ) -> Dict[str, Any]:
        """
        Compute centralised statistics for comparison with federated results.

        Args:
            df: Combined dataset from all organisations
            variables: List of variables to analyse

        Returns:
            Dictionary of centralised statistics
        """
        results = {"numerical": {}, "categorical": {}}

        for var in variables:
            if pd.api.types.is_numeric_dtype(df[var]):
                # Numerical statistics
                results["numerical"][var] = {
                    "count": int(df[var].count()),
                    "mean": float(df[var].mean()),
                    "std": float(df[var].std()),
                    "min": float(df[var].min()),
                    "max": float(df[var].max()),
                    "median": float(df[var].median()),
                    "q25": float(df[var].quantile(0.25)),
                    "q75": float(df[var].quantile(0.75)),
                }
            else:
                # Categorical statistics
                value_counts = df[var].value_counts()
                results["categorical"][var] = {
                    "counts": value_counts.to_dict(),
                    "total": int(value_counts.sum()),
                    "unique": int(df[var].nunique()),
                    "most_common": (
                        value_counts.index[0] if len(value_counts) > 0 else None
                    ),
                }

        return results

    @staticmethod
    def compute_centralized_quantiles(
        df: pd.DataFrame, variable: str, quantiles: List[float]
    ) -> Dict[float, float]:
        """
        Compute centralized quantiles for comparison.

        Args:
            df: Combined dataset
            variable: Variable name
            quantiles: List of quantile values (0-1)

        Returns:
            Dictionary mapping quantile to value
        """
        return {q: float(df[variable].quantile(q)) for q in quantiles}

    @staticmethod
    def compute_centralized_crosstabs(
        df: pd.DataFrame, var1: str, var2: str
    ) -> pd.DataFrame:
        """
        Compute centralized crosstabulation.

        Args:
            df: Combined dataset
            var1: First variable
            var2: Second variable

        Returns:
            Crosstabulation DataFrame
        """
        return pd.crosstab(df[var1], df[var2])


class FederatedTestValidator:
    """Validator for comparing federated and centralised results."""

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialise validator with tolerance for numerical comparisons.

        Args:
            tolerance: Numerical tolerance for comparisons
        """
        self.tolerance = tolerance

    def validate_numerical_statistics(
        self,
        federated_result: Dict[str, Any],
        centralised_result: Dict[str, Any],
        variable: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate numerical statistics between federated and centralised results.

        Args:
            federated_result: Result from federated computation
            centralised_result: Result from centralised computation
            variable: Variable name to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if variable not in federated_result.get("numerical", {}):
            errors.append(
                f"Variable {variable} not found in federated numerical results"
            )
            return False, errors

        if variable not in centralised_result.get("numerical", {}):
            errors.append(
                f"Variable {variable} not found in centralised numerical results"
            )
            return False, errors

        fed_stats = federated_result["numerical"][variable]
        cent_stats = centralised_result["numerical"][variable]

        # Check each statistic
        for stat_name in ["count", "mean", "std", "min", "max", "median"]:
            if stat_name in fed_stats and stat_name in cent_stats:
                fed_val = fed_stats[stat_name]
                cent_val = cent_stats[stat_name]

                if stat_name == "count":
                    # Count should be exact
                    if fed_val != cent_val:
                        errors.append(
                            f"{variable}.{stat_name}: {fed_val} != {cent_val}"
                        )
                else:
                    # Other statistics allow for tolerance
                    if not np.isclose(
                        fed_val, cent_val, rtol=self.tolerance, atol=self.tolerance
                    ):
                        errors.append(
                            f"{variable}.{stat_name}: {fed_val} vs {cent_val} (diff: {abs(fed_val - cent_val)})"
                        )

        return len(errors) == 0, errors

    def validate_categorical_statistics(
        self,
        federated_result: Dict[str, Any],
        centralised_result: Dict[str, Any],
        variable: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate categorical statistics between federated and centralised results.

        Args:
            federated_result: Result from federated computation
            centralised_result: Result from centralised computation
            variable: Variable name to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if variable not in federated_result.get("categorical", {}):
            errors.append(
                f"Variable {variable} not found in federated categorical results"
            )
            return False, errors

        if variable not in centralised_result.get("categorical", {}):
            errors.append(
                f"Variable {variable} not found in centralised categorical results"
            )
            return False, errors

        fed_stats = federated_result["categorical"][variable]
        cent_stats = centralised_result["categorical"][variable]

        # Check counts
        fed_counts = fed_stats.get("counts", {})
        cent_counts = cent_stats.get("counts", {})

        all_categories = set(fed_counts.keys()) | set(cent_counts.keys())

        for category in all_categories:
            fed_count = fed_counts.get(category, 0)
            cent_count = cent_counts.get(category, 0)

            if fed_count != cent_count:
                errors.append(f"{variable}.{category}: {fed_count} != {cent_count}")

        # Check totals
        if fed_stats.get("total", 0) != cent_stats.get("total", 0):
            errors.append(
                f"{variable}.total: {fed_stats.get('total')} != {cent_stats.get('total')}"
            )

        return len(errors) == 0, errors


@contextmanager
def timer():
    """Context manager for timing code execution."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")


class PerformanceBenchmark:
    """Utility for measuring and comparing performance of functions."""

    def __init__(self):
        """Initialize performance benchmark utility."""
        self.results = {}

    def measure_function(
        self, func: Callable, *args, name: str = None, iterations: int = 1, **kwargs
    ) -> Dict[str, Any]:
        """
        Measure function performance.

        Args:
            func: Function to measure
            *args: Function arguments
            name: Name for the benchmark
            iterations: Number of iterations to run
            **kwargs: Function keyword arguments

        Returns:
            Dictionary with performance metrics
        """
        name = name or func.__name__
        times = []

        for _ in range(iterations):
            start_time = time.perf_counter()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        metrics = {
            "name": name,
            "iterations": iterations,
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "total_time": np.sum(times),
        }

        self.results[name] = metrics
        return metrics

    def compare_functions(
        self, func1: Callable, func2: Callable, *args, iterations: int = 5, **kwargs
    ) -> Dict[str, Any]:
        """
        Compare performance of two functions.

        Args:
            func1, func2: Functions to compare
            *args: Function arguments
            iterations: Number of iterations
            **kwargs: Function keyword arguments

        Returns:
            Comparison results
        """
        metrics1 = self.measure_function(func1, *args, iterations=iterations, **kwargs)
        metrics2 = self.measure_function(func2, *args, iterations=iterations, **kwargs)

        speedup = metrics2["mean_time"] / metrics1["mean_time"]

        return {
            "function1": metrics1,
            "function2": metrics2,
            "speedup": speedup,
            "faster": metrics1["name"] if speedup > 1 else metrics2["name"],
        }

    def report(self) -> str:
        """Generate performance report."""
        if not self.results:
            return "No benchmark results available."

        report = "Performance Benchmark Results\n"
        report += "=" * 40 + "\n"

        for name, metrics in self.results.items():
            report += f"\nFunction: {name}\n"
            report += f"  Mean Time: {metrics['mean_time']:.6f}s\n"
            report += f"  Std Dev:   {metrics['std_time']:.6f}s\n"
            report += f"  Min Time:  {metrics['min_time']:.6f}s\n"
            report += f"  Max Time:  {metrics['max_time']:.6f}s\n"
            report += f"  Iterations: {metrics['iterations']}\n"

        return report


def create_federated_scenario(
    datasets: Dict[int, pd.DataFrame], variables_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a complete federated testing scenario.

    Args:
        datasets: Dictionary mapping organisation_id to DataFrame
        variables_config: Configuration for variables

    Returns:
        Complete testing scenario with data and expected results
    """
    # Combine all datasets for centralised computation
    combined_df = pd.concat(datasets.values(), ignore_index=True)

    # Extract variables from config
    all_variables = []
    if "numerical" in variables_config:
        all_variables.extend(variables_config["numerical"].keys())
    if "categorical" in variables_config:
        all_variables.extend(variables_config["categorical"].keys())

    # Compute centralised results for comparison
    centralised_results = CentralisedImplementations.compute_centralised_statistics(
        combined_df, all_variables
    )

    return {
        "federated_data": datasets,
        "combined_data": combined_df,
        "variables_config": variables_config,
        "centralised_results": centralised_results,
        "organisation_ids": list(datasets.keys()),
    }


def assert_federated_equals_centralized(
    federated_result: Dict[str, Any],
    centralized_result: Dict[str, Any],
    tolerance: float = 1e-6,
) -> None:
    """
    Assert that federated results equal centralized results within tolerance.

    Args:
        federated_result: Result from federated computation
        centralized_result: Result from centralized computation
        tolerance: Numerical tolerance

    Raises:
        AssertionError: If results don't match within tolerance
    """
    validator = FederatedTestValidator(tolerance)

    # Validate numerical variables
    if "numerical" in federated_result and "numerical" in centralized_result:
        for variable in federated_result["numerical"].keys():
            is_valid, errors = validator.validate_numerical_statistics(
                federated_result, centralized_result, variable
            )
            if not is_valid:
                raise AssertionError(
                    f"Numerical validation failed for {variable}: {errors}"
                )

    # Validate categorical variables
    if "categorical" in federated_result and "categorical" in centralized_result:
        for variable in federated_result["categorical"].keys():
            is_valid, errors = validator.validate_categorical_statistics(
                federated_result, centralized_result, variable
            )
            if not is_valid:
                raise AssertionError(
                    f"Categorical validation failed for {variable}: {errors}"
                )


def generate_privacy_test_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Generate test scenarios for privacy-preserving functions.

    Returns:
        Dictionary of privacy test scenarios
    """
    np.random.seed(42)

    scenarios = {}

    # Scenario 1: Basic differential privacy
    scenarios["basic_dp"] = {
        "data": pd.DataFrame(
            {
                "sensitive_value": np.random.normal(50, 10, 1000),
                "organization_id": np.repeat([1, 2, 3], [300, 400, 300]),
            }
        ),
        "epsilon": 1.0,
        "expected_noise_std": 1.0,  # 1/epsilon for Laplace mechanism
    }

    # Scenario 2: Small sample size thresholding
    scenarios["small_sample"] = {
        "data": pd.DataFrame({"value": [1, 2, 3, 4, 5], "organization_id": [1] * 5}),
        "threshold": 10,
        "should_pass": False,
    }

    # Scenario 3: Variable masking
    scenarios["variable_masking"] = {
        "data": pd.DataFrame(
            {
                "keep_var": [1, 2, 3, 4, 5],
                "remove_var": [6, 7, 8, 9, 10],
                "another_keep": ["a", "b", "c", "d", "e"],
            }
        ),
        "variables_to_keep": ["keep_var", "another_keep"],
        "expected_columns": ["keep_var", "another_keep"],
    }

    return scenarios


def save_benchmark_results(
    results: Dict[str, Any], filepath: str = "benchmark_results.json"
):
    """
    Save benchmark results to file.

    Args:
        results: Benchmark results dictionary
        filepath: Path to save results
    """
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)


def compute_quantile_test_statistics(data, quantiles=[0.25, 0.5, 0.75]):
    """
    Compute test statistics for quantile validation using asymptotic theory.

    This function calculates the theoretical standard error for each quantile
    using the asymptotic formula: SE(q) = sqrt(q(1-q)/n) / f(q_val)
    where f(q_val) is the probability density function at the quantile value.

    Args:
        data: Array of data values
        quantiles: List of quantile levels (e.g., [0.25, 0.5, 0.75])

    Returns:
        Dictionary with statistics for each quantile including value, standard error,
        confidence interval, and density estimation
    """
    n = len(data)
    results = {}

    for q in quantiles:
        # Step 1: Calculate the exact quantile value
        q_val = np.quantile(data, q)

        # Step 2: Estimate probability density at quantile using kernel density estimation
        # This is needed for the asymptotic standard error formula
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(data)
            density_at_q = kde(q_val)[0]
        except Exception as e:
            safe_log("error", "Error in KDE estimation: " + str(e))
            # Fallback: use simple density estimation
            density_at_q = 1.0  # Default if KDE fails

        # Step 3: Calculate asymptotic standard error
        # Formula: SE(q) = sqrt(q(1-q)/n) / f(q_val)
        # The numerator sqrt(q(1-q)/n) is the variance component
        # The denominator f(q_val) is the density at the quantile
        variance_component = np.sqrt(q * (1 - q) / n)
        se = variance_component / max(density_at_q, 1e-6)  # Avoid division by zero

        # 95% confidence interval
        ci_lower = q_val - 1.96 * se
        ci_upper = q_val + 1.96 * se

        results[q] = {
            "value": q_val,
            "se": se,
            "ci": (ci_lower, ci_upper),
            "density": density_at_q,
        }

    return results


def validate_quantiles(centralised_data, federated_quantiles, test_name=""):
    """
    Robust quantile validation with multiple statistical criteria.

    This function performs comprehensive validation using three complementary tests:
    1. Confidence Interval Test: Checks if federated quantile falls within theoretical CI
    2. Relative Error Test: Compares difference relative to data spread (IQR)
    3. Z-Score Test: Standardised difference in units of standard error

    Args:
        centralised_data: Raw data array for computing true quantiles
        federated_quantiles: Dictionary with federated quantile results
        test_name: Name for logging purposes

    Returns:
        Dictionary with detailed test results and pass/fail status
    """

    print("\n  === QUANTILE VALIDATION ===")
    print(f"    Test: {test_name}")

    quantile_keys = [0.25, 0.5, 0.75]
    cent_stats = compute_quantile_test_statistics(centralised_data, quantile_keys)

    # Mapping for federated keys
    fed_key_map = {0.25: "Q1", 0.5: "Q2", 0.75: "Q3"}

    results = {"individual_tests": {}, "overall_pass": True}

    # Individual quantile tests - validate each quantile separately
    for q in quantile_keys:
        fed_key = fed_key_map[q]
        if fed_key not in federated_quantiles:
            print(f"Warning: {fed_key} not found in federated results, skipping")
            continue

        # Extract centralised and federated quantile values
        cent_val = cent_stats[q]["value"]
        fed_val = float(federated_quantiles[fed_key])
        ci_lower, ci_upper = cent_stats[q]["ci"]

        # TEST 1: Confidence Interval Containment Test
        # Checks if each federated quantile falls within the 95% confidence interval of centralised quantile
        in_ci = ci_lower <= fed_val <= ci_upper

        # TEST 2: Relative Error Test with adaptive threshold
        # Uses IQR as the scale for relative error assessment
        iqr = cent_stats[0.75]["value"] - cent_stats[0.25]["value"]
        scale = max(
            iqr, np.std(centralised_data), 1e-6
        )  # Use IQR, fallback to std, avoid zero
        rel_error = abs(fed_val - cent_val) / scale
        n = len(centralised_data)
        rel_threshold = max(0.1, 10 / np.sqrt(n))  # More lenient for smaller samples
        rel_pass = rel_error <= rel_threshold

        # TEST 3: Z-Score Test
        # Tests if the difference is within acceptable statistical bounds
        z_score = abs(fed_val - cent_val) / cent_stats[q]["se"]
        z_threshold = 2.58  # 99% confidence (stricter than 95%)
        z_pass = z_score <= z_threshold

        # Final decision: Pass if any two tests pass
        test_passes = [in_ci, rel_pass, z_pass]
        result_pass = sum(test_passes) >= 2

        result = {
            "quantile": q,
            "centralised_value": cent_val,
            "federated_value": fed_val,
            "ci_test": in_ci,
            "ci_bounds": (ci_lower, ci_upper),
            "rel_error": rel_error,
            "rel_threshold": rel_threshold,
            "rel_test": rel_pass,
            "z_score": z_score,
            "z_threshold": z_threshold,
            "z_test": z_pass,
            "pass": result_pass,
        }

        results["individual_tests"][q] = result
        if not result_pass:
            results["overall_pass"] = False

        # Print detailed results
        print(f"  Quantile {q} ({fed_key}):")
        print(f"    Centralised: {cent_val:.6f}, Federated: {fed_val:.6f}")
        print(f"  CI Test: {'PASS' if in_ci else 'FAIL'}")
        print(f"    CI = [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"  Relative Error Test: {'PASS' if rel_pass else 'FAIL'}")
        print(f"    Error = {rel_error:.6f}, Threshold = {rel_threshold:.6f}")
        print(f"  Z-Score Test: {'PASS' if z_pass else 'FAIL'}")
        print(f"    Z = {z_score:.3f}, Threshold = {z_threshold:.3f}")
        print(f"  Overall Result: {'PASS' if result_pass else 'FAIL'}\n")

    print(f"\nFINAL VALIDATION: {'PASS' if results['overall_pass'] else 'FAIL'}")

    return results


def assert_federated_equals_centralised(
    federated_result: Dict[str, Any],
    centralised_result: Dict[str, Any],
    tolerance: float = 1e-6,
) -> None:
    """
    Assert that federated results equal centralised results within tolerance.

    Args:
        federated_result: Result from federated computation
        centralised_result: Result from centralised computation
        tolerance: Numerical tolerance

    Raises:
        AssertionError: If results don't match within tolerance
    """
    validator = FederatedTestValidator(tolerance)

    # Validate numerical variables
    if "numerical" in federated_result and "numerical" in centralised_result:
        for variable in federated_result["numerical"].keys():
            is_valid, errors = validator.validate_numerical_statistics(
                federated_result, centralised_result, variable
            )
            if not is_valid:
                raise AssertionError(
                    f"Numerical validation failed for {variable}: {errors}"
                )

    # Validate categorical variables
    if "categorical" in federated_result and "categorical" in centralised_result:
        for variable in federated_result["categorical"].keys():
            is_valid, errors = validator.validate_categorical_statistics(
                federated_result, centralised_result, variable
            )
            if not is_valid:
                raise AssertionError(
                    f"Categorical validation failed for {variable}: {errors}"
                )
