"""
------------------------------------------------------------------------------
Privacy Protection Utilities

File organisation:
- General data preprocessing utilities (e.g., masking variables, sample size thresholding)
- Privacy budget management
- Differential privacy mechanisms (Laplace and Gaussian)
- Differential privacy application functions
- Main DP function
------------------------------------------------------------------------------
"""

import json
import math
import random
import hashlib

import pandas as pd
import numpy as np

from typing import Any, Dict, List, Literal, Optional, Union, Tuple
from datetime import timedelta

from vantage6.algorithm.tools.util import get_env_var
from vantage6.algorithm.client import AlgorithmClient

# Import safe logging and calculation functions from misc
from .miscellaneous import safe_log, safe_calculate

# Type definitions for improved type checking
DataFrameOrDict = Union[pd.DataFrame, Dict[str, Any]]
Number = Union[int, float]
PrivacyResult = Union[Dict[str, Any], pd.DataFrame]
BinType = Union[List[pd.Timestamp], List[timedelta], List[float]]
NoiseDistribution = Literal["laplace", "gaussian"]


def mask_unnecessary_variables(df: pd.DataFrame, variables_to_use: List[str]) -> pd.DataFrame:
    """
    Mask unnecessary variables in the DataFrame by removing them from the working draft of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be masked.
        variables_to_use (List[str]): The list of variables to keep.

    Returns:
       pd.DataFrame: The processed DataFrame with unnecessary variables removed.
    """
    safe_log("info", "Masking unnecessary variables in the working draft of DataFrame")

    # Drop any irrelevant variables
    for variable in df.columns:
        if variable not in variables_to_use:
            df = df.drop(columns=variable)
            continue

    return df


def apply_sample_size_threshold(client: AlgorithmClient, df: pd.DataFrame,
                                variables_to_check: List[str]) -> DataFrameOrDict:
    """
    This function thresholds the sample size of the input DataFrame.

    Args:
        client (AlgorithmClient): The algorithm client instance.
        df (pd.DataFrame): The DataFrame to check the sample size of.
        variables_to_check (List[str]): The list of variables to check for sample size.

    Returns:
        Union[pd.DataFrame, Dict[str, Any]]: The processed DataFrame or a dict warning message.
    """
    # Retrieve the sample size threshold
    sample_size_threshold = get_env_var("SAMPLE_SIZE_THRESHOLD")
    try:
        sample_size_threshold = int(sample_size_threshold)
    except TypeError:
        sample_size_threshold = 10

    safe_log("info", f"Applying sample size threshold of '{str(sample_size_threshold)}' to working draft of DataFrame")

    # Check if the DataFrame is empty
    if len(df) <= sample_size_threshold:
        safe_log("warn",
                 f"Sub-task was not executed because the number of samples is too small (n <= {sample_size_threshold})")
        return {"N-Threshold not met": client.organization_id}

    # Check if there are enough datapoints
    for variable in variables_to_check:
        if variable in df.columns:
            if df[variable].notnull().sum() <= sample_size_threshold:
                safe_log("warn",
                         f"Variable {variable} was set to completely missing because the number of samples is too small")
                df[variable] = pd.NA
                continue

    return df


class PrivacyBudgetManager:
    """
    Manages the privacy budget to ensure overall privacy guarantees are maintained.

    Tracks consumed privacy budget (epsilon) across multiple queries or operations and
    prevents exceeding the total allocated budget.
    """

    def __init__(self, total_epsilon: float, delta: float):
        """
        Initialise the privacy budget manager.

        Args:
            total_epsilon (float): Total privacy budget available.
            delta (float): Failure probability parameter for differential privacy.
        """
        self.total_epsilon = total_epsilon
        self.delta = delta
        self.used_epsilon = 0.0
        self.query_history: List[Dict[str, Any]] = []

    def check_and_consume(self, requested_epsilon: float, operation_id: Optional[str] = None) -> bool:
        """
        Check if sufficient budget is available and consume it if so.

        Args:
            requested_epsilon (float): The amount of privacy budget to consume.
            operation_id (Optional[str]): Identifier for the operation consuming budget.

        Returns:
            bool: True if budget was successfully consumed, False if budget would be exceeded.
        """
        if self.used_epsilon + requested_epsilon > self.total_epsilon:
            safe_log("warn", f"Privacy budget exceeded. Used: {self.used_epsilon}, Requested: {requested_epsilon}. "
                             f"Skipping this operation")
            return False

        self.used_epsilon += requested_epsilon

        # Record the operation for auditing purposes
        self.query_history.append({
            'operation_id': operation_id or f"query_{len(self.query_history)}",
            'epsilon': requested_epsilon,
            'timestamp': pd.Timestamp.now()
        })

        safe_log("info",
                 f"Privacy budget consumed: {requested_epsilon}. Remaining: {self.total_epsilon - self.used_epsilon}")
        return True

    def get_remaining_budget(self) -> float:
        """
        Get the remaining privacy budget.

        Returns:
            float: The amount of privacy budget remaining.
        """
        return self.total_epsilon - self.used_epsilon

    def get_usage_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of privacy budget usage.

        Returns:
            List[Dict[str, Any]]: List of operations that consumed privacy budget.
        """
        return self.query_history


def _get_laplace_scale(sensitivity: float, epsilon: float) -> float:
    """
    Calculate the scale parameter for the Laplace noise mechanism.

    Args:
        sensitivity (float): The L1-sensitivity of the query.
        epsilon (float): The privacy parameter (smaller means more privacy).

    Returns:
        float: The scale parameter (b) for the Laplace noise.
    """
    return sensitivity / epsilon


def _get_gaussian_noise_scale(sensitivity: float, delta: float, epsilon: float) -> float:
    """
    Calculate the scale parameter for the Gaussian noise mechanism.

    Args:
        sensitivity (float): The sensitivity of the query.
        delta (float): The failure probability parameter.
        epsilon (float): The privacy parameter (smaller means more privacy).

    Returns:
        float: The scale (standard deviation) for the Gaussian noise.
    """
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def _generate_query_seed(
        df_shape: Tuple[int, int],
        variable: str,
        epsilon: float,
        delta: float,
        distribution: str,
        operation: str = None,
        **kwargs
) -> int:
    """
    Generate a deterministic seed for noise generation based on query parameters.

    This function creates a consistent seed value that depends only on the query parameters,
    ensuring that identical queries always produce identical noise values, even across
    different sessions or algorithm runs.

    Args:
        df_shape: Shape of the input DataFrame (rows, columns)
        variable: Variable to apply DP to
        epsilon: Privacy parameter
        delta: Failure probability parameter
        distribution: Noise distribution mechanism
        operation: Statistical operation (for numerical data)
        **kwargs: Additional parameters affecting the result

    Returns:
        int: A deterministic seed value for the random number generator
    """
    # Create a dictionary of all query parameters
    query_dict = {
        "df_shape": df_shape,
        "variable": variable,
        "epsilon": epsilon,
        "delta": delta,
        "distribution": distribution
    }

    # Add operation if provided (for numerical data)
    if operation:
        query_dict["operation"] = operation

    # Add additional parameters that affect query results
    for param in ["top_k", "bin_count"]:
        if param in kwargs:
            query_dict[param] = kwargs[param]

    # Convert to JSON string and hash
    query_json = json.dumps(query_dict, sort_keys=True)
    hash_value = hashlib.sha256(query_json.encode()).hexdigest()

    # Convert hash to integer for seed (modulo to prevent overflow)
    return int(hash_value, 16) % (2 ** 32 - 1)


def _generate_noise(
        sensitivity: float,
        epsilon: float,
        delta: float = 1e-5,
        distribution: NoiseDistribution = "laplace",
        seed: Optional[int] = None
) -> float:
    """
    Generate noise based on the chosen differential privacy mechanism.

    Using a consistent seed ensures the same query always produces the same noise,
    preventing multiple query averaging attacks while maintaining privacy guarantees.

    Args:
        sensitivity (float): The sensitivity of the query.
        epsilon (float): The privacy parameter (smaller means more privacy).
        delta (float): The failure probability parameter (used only for Gaussian).
        distribution (NoiseDistribution): The DP mechanism to use:
            - 'laplace': Provides pure ε-differential privacy
            - 'gaussian': Provides (ε,δ)-differential privacy
        seed (Optional[int]): Seed for the random number generator, used for deterministic noise.
            If provided, the same seed will always produce the same noise value.

    Returns:
        float: A noise value from the appropriate distribution.
    """
    # Create a deterministic RNG if seed is provided
    if seed is not None:
        # Create a separate random number generator that won't affect other code
        rng = np.random.RandomState(seed)
    else:
        # Use the global RNG if no seed (less secure but maintains backwards compatibility)
        rng = np.random

    if distribution == "laplace":
        scale = _get_laplace_scale(sensitivity, epsilon)
        return rng.laplace(0, scale)
    else:  # gaussian
        sigma = _get_gaussian_noise_scale(sensitivity, delta, epsilon)
        return rng.normal(0, sigma)


def _sanitise_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove potentially sensitive information from metadata.

    Args:
        metadata (Dict[str, Any]): Original metadata dictionary

    Returns:
        Dict[str, Any]: Sanitised metadata dictionary
    """
    # Create a copy to avoid modifying the original
    safe_metadata = metadata.copy()

    # Remove or obfuscate sensitive fields
    if "sensitivity" in safe_metadata:
        # Instead of exact sensitivity, use a range indicator
        sensitivity = safe_metadata["sensitivity"]
        if sensitivity < 0.01:
            safe_metadata["sensitivity_range"] = "very low"
        elif sensitivity < 0.1:
            safe_metadata["sensitivity_range"] = "low"
        elif sensitivity < 1.0:
            safe_metadata["sensitivity_range"] = "medium"
        else:
            safe_metadata["sensitivity_range"] = "high"

        del safe_metadata["sensitivity"]

    # Other potentially sensitive fields to remove
    sensitive_fields = ["bin_edges", "categories"]
    for field in sensitive_fields:
        if field in safe_metadata:
            # Instead of values, just store the count
            if isinstance(safe_metadata[field], list):
                safe_metadata[f"{field}_count"] = len(safe_metadata[field])
            del safe_metadata[field]

    return safe_metadata


def _calculate_sensitivity(df: pd.DataFrame,
                           column_name: str,
                           operation: str = 'mean') -> float:
    """
    Calculate the sensitivity of a specific operation on the data.

    The sensitivity represents how much the output of a query could change
    if one record in the dataset changes or is removed.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column to calculate sensitivity for.
        operation (str): The operation being performed ('mean', 'sum', etc).

    Returns:
        float: The calculated sensitivity value.
    """
    n = len(df)
    if n == 0:
        return 0.0

    # For numerical data, get range bounds
    if pd.api.types.is_numeric_dtype(df[column_name]):
        if df[column_name].notnull().sum() == 0:
            return 0.0

        # Remove NaN values for calculations
        values = df[column_name].dropna()
        if len(values) == 0:
            return 0.0

        # Calculate bounds
        min_val = values.min()
        max_val = values.max()
        value_range = max_val - min_val

        # Different operations have different sensitivities
        if operation == 'mean':
            # For mean, sensitivity is range/n
            return value_range / n
        elif operation == 'sum':
            # For sum, sensitivity is just the maximum value
            return max_val
        elif operation == 'count':
            # For count, sensitivity is 1
            return 1.0
        else:
            # Default conservative approach
            return value_range
    else:
        # For non-numeric data, use default sensitivity of 1
        return 1.0


def _apply_dp_to_categorical(df: pd.DataFrame,
                             column_name: str,
                             epsilon: float,
                             delta: float = 1e-5,
                             debug_mode: bool = False,
                             distribution: NoiseDistribution = "laplace") -> Dict[str, Any]:
    """
    Apply differential privacy to categorical data by adding noise to category counts.

    Uses deterministic noise generation to ensure consistent results for identical queries,
    protecting against multiple-query averaging attacks.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column with categorical data.
        epsilon (float): Privacy parameter.
        delta (float): Failure probability parameter.
        debug_mode (bool): Whether to include debug information including true values.
        distribution (NoiseDistribution): The differential privacy mechanism to use.

    Returns:
        Dict[str, Any]: Results including private category counts.
    """
    # Get all possible categories (including those not in the data)
    if hasattr(df[column_name].dtype, 'categories'):
        all_categories = list(df[column_name].dtype.categories)
    else:
        # If categories aren't defined in the dtype, use unique values
        all_categories = df[column_name].dropna().unique().tolist()

    # Get count of each category
    value_counts = df[column_name].value_counts().to_dict()

    # Ensure all categories have a count (even if 0)
    for category in all_categories:
        if category not in value_counts:
            value_counts[category] = 0

    # Calculate sensitivity (always 1 for counts)
    sensitivity = 1.0

    # Get noise scale based on distribution
    if distribution == "laplace":
        noise_scale = _get_laplace_scale(sensitivity, epsilon)
    else:
        noise_scale = _get_gaussian_noise_scale(sensitivity, delta, epsilon)

    # Generate base seed for this query
    base_seed = _generate_query_seed(
        df_shape=df.shape,
        variable=column_name,
        epsilon=epsilon,
        delta=delta,
        distribution=distribution
    )

    # Add noise to each category count
    private_counts: Dict[Any, int] = {}
    for i, (category, count) in enumerate(value_counts.items()):
        # Create a unique seed for each category by combining the base seed with category index
        category_seed = base_seed + i
        noise = _generate_noise(sensitivity, epsilon, delta, distribution, seed=category_seed)
        # Round to the nearest non-negative integer
        private_count = max(0, round(count + noise))
        private_counts[category] = private_count

    result: Dict[str, Any] = {
        "operation": "category_counts",
        "private_counts": private_counts,
        "epsilon_used": epsilon,
        "sample_count": len(df),
        "metadata": {
            "sensitivity": sensitivity,
            "noise_scale": noise_scale,
            "distribution": distribution,
            "categories": all_categories
        }
    }

    # Include true values only in debug mode
    if debug_mode:
        result["true_counts"] = value_counts

    return result


def _apply_dp_to_boolean(df: pd.DataFrame,
                         column_name: str,
                         epsilon: float,
                         delta: float = 1e-5,
                         debug_mode: bool = False,
                         distribution: NoiseDistribution = "laplace") -> Dict[str, Any]:
    """
    Apply differential privacy to boolean data by treating it as a special case of categorical.

    Uses deterministic noise generation to ensure consistent results for identical queries,
    protecting against multiple-query averaging attacks.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column with boolean data.
        epsilon (float): Privacy parameter.
        delta (float): Failure probability parameter.
        debug_mode (bool): Whether to include debug information including true values.
        distribution (NoiseDistribution): The differential privacy mechanism to use.

    Returns:
        Dict[str, Any]: Results including private category counts.
    """
    # Boolean data has True/False categories
    categories = [True, False]

    # Get count of each category
    value_counts = df[column_name].value_counts().to_dict()

    # Ensure both True and False have a count (even if 0)
    for category in categories:
        if category not in value_counts:
            value_counts[category] = 0

    # Calculate sensitivity (always 1 for counts)
    sensitivity = 1.0

    # Get noise scale based on distribution
    if distribution == "laplace":
        noise_scale = _get_laplace_scale(sensitivity, epsilon)
    else:
        noise_scale = _get_gaussian_noise_scale(sensitivity, delta, epsilon)

    # Generate base seed for this query
    base_seed = _generate_query_seed(
        df_shape=df.shape,
        variable=column_name,
        epsilon=epsilon,
        delta=delta,
        distribution=distribution
    )

    # Add noise to each category count
    private_counts: Dict[bool, int] = {}
    for i, (category, count) in enumerate(value_counts.items()):
        # Create a unique seed for each category
        category_seed = base_seed + i
        noise = _generate_noise(sensitivity, epsilon, delta, distribution, seed=category_seed)
        # Round to nearest non-negative integer
        private_count = max(0, round(count + noise))
        private_counts[category] = private_count

    result: Dict[str, Any] = {
        "operation": "boolean_counts",
        "private_counts": private_counts,
        "epsilon_used": epsilon,
        "sample_count": len(df),
        "metadata": {
            "sensitivity": sensitivity,
            "noise_scale": noise_scale,
            "distribution": distribution
        }
    }

    # Include true values only in debug mode
    if debug_mode:
        result["true_counts"] = value_counts

    return result


def _apply_dp_to_string(df: pd.DataFrame,
                        column_name: str,
                        epsilon: float,
                        delta: float = 1e-5,
                        top_k: int = 10,
                        debug_mode: bool = False,
                        distribution: NoiseDistribution = "laplace") -> Dict[str, Any]:
    """
    Apply differential privacy to string data by reporting noisy counts of top values.

    Uses deterministic noise generation to ensure consistent results for identical queries,
    protecting against multiple-query averaging attacks.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column with string data.
        epsilon (float): Privacy parameter.
        delta (float): Failure probability parameter.
        top_k (int): Number of top categories to report.
        debug_mode (bool): Whether to include debug information including true values.
        distribution (NoiseDistribution): The differential privacy mechanism to use.

    Returns:
        Dict[str, Any]: Results including private category counts.
    """
    # Get counts of all values
    value_counts = df[column_name].value_counts().to_dict()

    # Limit to top_k most frequent values to avoid leaking rare values
    top_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_values_dict = dict(top_values)

    # Calculate sensitivity (always 1 for counts)
    sensitivity = 1.0

    # Get noise scale based on distribution
    if distribution == "laplace":
        noise_scale = _get_laplace_scale(sensitivity, epsilon)
    else:
        noise_scale = _get_gaussian_noise_scale(sensitivity, delta, epsilon)

    # Generate base seed for this query
    base_seed = _generate_query_seed(
        df_shape=df.shape,
        variable=column_name,
        epsilon=epsilon,
        delta=delta,
        distribution=distribution,
        top_k=top_k
    )

    # Add noise to each category count
    private_counts: Dict[str, int] = {}
    for i, (category, count) in enumerate(top_values_dict.items()):
        # Create a unique seed for each category
        category_seed = base_seed + i
        noise = _generate_noise(sensitivity, epsilon, delta, distribution, seed=category_seed)
        # Round to nearest non-negative integer
        private_count = max(0, round(count + noise))
        private_counts[category] = private_count

    result: Dict[str, Any] = {
        "operation": "string_top_counts",
        "private_counts": private_counts,
        "epsilon_used": epsilon,
        "sample_count": len(df),
        "metadata": {
            "sensitivity": sensitivity,
            "noise_scale": noise_scale,
            "distribution": distribution,
            "top_k": top_k
        }
    }

    # Include true values only in debug mode
    if debug_mode:
        result["true_counts"] = top_values_dict

    return result


def _apply_dp_to_datetime(df: pd.DataFrame,
                          column_name: str,
                          epsilon: float,
                          delta: float = 1e-5,
                          bins: Optional[BinType] = None,
                          bin_count: int = 10,
                          debug_mode: bool = False,
                          distribution: NoiseDistribution = "laplace") -> Dict[str, Any]:
    """
    Apply differential privacy to datetime data by binning and adding noise to bin counts.

    Uses deterministic noise generation to ensure consistent results for identical queries,
    protecting against multiple-query averaging attacks.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column with datetime data.
        epsilon (float): Privacy parameter.
        delta (float): Failure probability parameter.
        bins (Optional[BinType]): Explicit bin edges for datetime.
        bin_count (int): Number of bins to use if bins not specified.
        debug_mode (bool): Whether to include debug information including true values.
        distribution (NoiseDistribution): The differential privacy mechanism to use.

    Returns:
        Dict[str, Any]: Results including private bin counts.
    """
    # Remove missing values
    valid_data = df[column_name].dropna()

    if len(valid_data) == 0:
        return {
            "operation": "datetime_histogram",
            "private_counts": {},
            "epsilon_used": epsilon,
            "sample_count": 0,
            "error": "No valid datetime data found"
        }

    # Create bins if not provided
    if bins is None:
        min_date = valid_data.min()
        max_date = valid_data.max()

        # Create evenly spaced bins
        date_range = max_date - min_date
        bin_size = date_range / bin_count

        bins = [min_date + i * bin_size for i in range(bin_count + 1)]

    # Create histogram with specified bins
    hist, bin_edges = np.histogram(valid_data, bins=bins)

    # Convert bin edges to strings for dictionary keys
    bin_labels = [f"bin_{i}" for i in range(len(bin_edges) - 1)]

    # Create count dictionary
    count_dict = dict(zip(bin_labels, hist))

    # Calculate sensitivity (always 1 for counts)
    sensitivity = 1.0

    # Get noise scale based on distribution
    if distribution == "laplace":
        noise_scale = _get_laplace_scale(sensitivity, epsilon)
    else:
        noise_scale = _get_gaussian_noise_scale(sensitivity, delta, epsilon)

    # Generate base seed for this query
    base_seed = _generate_query_seed(
        df_shape=df.shape,
        variable=column_name,
        epsilon=epsilon,
        delta=delta,
        distribution=distribution,
        bin_count=bin_count
    )

    # Add noise to each bin count
    private_counts: Dict[str, int] = {}
    for i, (bin_label, count) in enumerate(count_dict.items()):
        # Create a unique seed for each bin
        bin_seed = base_seed + i
        noise = _generate_noise(sensitivity, epsilon, delta, distribution, seed=bin_seed)
        # Round to nearest non-negative integer
        private_count = max(0, round(count + noise))
        private_counts[bin_label] = private_count

    result: Dict[str, Any] = {
        "operation": "datetime_histogram",
        "private_counts": private_counts,
        "epsilon_used": epsilon,
        "sample_count": len(df),
        "metadata": {
            "sensitivity": sensitivity,
            "noise_scale": noise_scale,
            "distribution": distribution,
            "bin_count": len(bins) - 1,
            "bin_edges": bin_edges if debug_mode else [f"bin_edge_{i}" for i in range(len(bin_edges))]
        }
    }

    # Include true values only in debug mode
    if debug_mode:
        result["true_counts"] = count_dict

    return result


def _apply_dp_to_timedelta(df: pd.DataFrame,
                           column_name: str,
                           epsilon: float,
                           delta: float = 1e-5,
                           bins: Optional[BinType] = None,
                           bin_count: int = 10,
                           debug_mode: bool = False,
                           distribution: NoiseDistribution = "laplace") -> Dict[str, Any]:
    """
    Apply differential privacy to timedelta data by binning and adding noise to bin counts.

    Uses deterministic noise generation to ensure consistent results for identical queries,
    protecting against multiple-query averaging attacks.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column with timedelta data.
        epsilon (float): Privacy parameter.
        delta (float): Failure probability parameter.
        bins (Optional[BinType]): Explicit bin edges for timedelta.
        bin_count (int): Number of bins to use if bins not specified.
        debug_mode (bool): Whether to include debug information including true values.
        distribution (NoiseDistribution): The differential privacy mechanism to use.

    Returns:
        Dict[str, Any]: Results including private bin counts.
    """
    # Convert timedeltas to seconds for easier processing
    valid_data = df[column_name].dropna()

    if len(valid_data) == 0:
        return {
            "operation": "timedelta_histogram",
            "private_counts": {},
            "epsilon_used": epsilon,
            "sample_count": 0,
            "error": "No valid timedelta data found"
        }

    # Convert to total seconds for binning
    seconds_data = valid_data.dt.total_seconds()

    # Create bins if not provided
    if bins is None:
        min_seconds = seconds_data.min()
        max_seconds = seconds_data.max()

        # Create evenly spaced bins
        range_seconds = max_seconds - min_seconds
        bin_size = range_seconds / bin_count

        # Create bin edges in seconds
        second_bins = [min_seconds + i * bin_size for i in range(bin_count + 1)]

        # Convert back to timedeltas for labels
        bins = [timedelta(seconds=s) for s in second_bins]
    else:
        # Convert provided bins to seconds for histogram
        second_bins = [b.total_seconds() if isinstance(b, timedelta) else b for b in bins]

    # Create histogram with specified bins
    hist, _ = np.histogram(seconds_data, bins=second_bins)

    # Convert bin edges to strings for dictionary keys - avoid leaking actual time values
    bin_labels = [f"bin_{i}" for i in range(len(bins) - 1)]

    # Create count dictionary
    count_dict = dict(zip(bin_labels, hist))

    # Calculate sensitivity (always 1 for counts)
    sensitivity = 1.0

    # Get noise scale based on distribution
    if distribution == "laplace":
        noise_scale = _get_laplace_scale(sensitivity, epsilon)
    else:
        noise_scale = _get_gaussian_noise_scale(sensitivity, delta, epsilon)

    # Generate base seed for this query
    base_seed = _generate_query_seed(
        df_shape=df.shape,
        variable=column_name,
        epsilon=epsilon,
        delta=delta,
        distribution=distribution,
        bin_count=bin_count
    )

    # Add noise to each bin count
    private_counts: Dict[str, int] = {}
    for i, (bin_label, count) in enumerate(count_dict.items()):
        # Create a unique seed for each bin
        bin_seed = base_seed + i
        noise = _generate_noise(sensitivity, epsilon, delta, distribution, seed=bin_seed)
        # Round to nearest non-negative integer
        private_count = max(0, round(count + noise))
        private_counts[bin_label] = private_count

    result: Dict[str, Any] = {
        "operation": "timedelta_histogram",
        "private_counts": private_counts,
        "epsilon_used": epsilon,
        "sample_count": len(df),
        "metadata": {
            "sensitivity": sensitivity,
            "noise_scale": noise_scale,
            "distribution": distribution,
            "bin_count": len(bins) - 1,
            "bin_edges": bins if debug_mode else [f"bin_edge_{i}" for i in range(len(bins))]
        }
    }

    # Include true values only in debug mode
    if debug_mode:
        result["true_counts"] = count_dict

    return result


def _get_dp_dataframe(df: pd.DataFrame, dp_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate a DataFrame with differentially private values, maintaining the original structure.

    This function creates a copy of the original DataFrame and applies the DP results
    to it, preserving the original structure and data types.

    Args:
        df (pd.DataFrame): The original DataFrame.
        dp_results (Dict[str, Any]): Results from apply_differential_privacy.

    Returns:
        pd.DataFrame: DataFrame with DP-protected values.
    """
    # Create a deep copy of the original DataFrame to maintain structure
    dp_df = df.copy()

    # Store metadata about the DP process (with sanitised information)
    dp_df.attrs['dp_metadata'] = {
        'privacy_budget': dp_results.get('privacy_budget', {}),
        'variables_protected': []
    }

    # Process each variable's DP results
    for var_name, result in dp_results.items():
        # Skip the non-variable entries
        if var_name in ["privacy_budget", "processing_summary", "security_notice", "security_warning"]:
            continue

        # Add to the list of protected variables
        dp_df.attrs['dp_metadata']['variables_protected'].append(var_name)

        # Check the operation type to determine how to process
        operation = result.get("operation", "")

        if operation in ['mean', 'sum', 'count']:
            # For numerical operations with a single result, fill the column with the DP value
            dp_df[var_name] = result['private_result']

        elif operation in ['category_counts', 'boolean_counts', 'string_top_counts']:
            # For categorical data, generate synthetic data based on the private counts
            counts = result['private_counts']
            if sum(counts.values()) > 0:  # Ensure we have some non-zero counts
                # Convert counts to probabilities
                total = sum(counts.values())
                probs = {k: v / total for k, v in counts.items()}

                # Generate synthetic data by sampling from the private distribution
                categories = list(counts.keys())
                probabilities = [probs[cat] for cat in categories]

                # Sample with replacement to create new data
                n_samples = len(df)
                synthetic_data = np.random.choice(categories, size=n_samples, p=probabilities)

                # If it's a pandas categorical type, maintain that
                if pd.api.types.is_categorical_dtype(df[var_name]):
                    dp_df[var_name] = pd.Categorical(synthetic_data, categories=df[var_name].dtype.categories)
                else:
                    dp_df[var_name] = synthetic_data

        elif operation in ['datetime_histogram', 'timedelta_histogram']:
            # For binned data, sample from the bins based on private counts
            if 'private_counts' in result and result['private_counts']:
                bin_edges = result['metadata'].get('bin_edges', [])
                counts = result['private_counts']

                if bin_edges and sum(counts.values()) > 0:
                    # Convert bin labels back to actual edges
                    bin_centers = []
                    bin_probs = []

                    # For each bin with a count, calculate its center
                    for i, (bin_label, count) in enumerate(counts.items()):
                        if i < len(bin_edges) - 1:
                            # Calculate the center of the bin
                            if pd.api.types.is_datetime64_dtype(df[var_name]):
                                # For datetime, the bin center is the midpoint
                                bin_center = bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) / 2
                            else:
                                # For timedelta, convert to seconds, find midpoint, convert back
                                start_seconds = bin_edges[i].total_seconds()
                                end_seconds = bin_edges[i + 1].total_seconds()
                                mid_seconds = (start_seconds + end_seconds) / 2
                                bin_center = timedelta(seconds=mid_seconds)

                            bin_centers.append(bin_center)
                            bin_probs.append(count)

                    # Normalise probabilities
                    total = sum(bin_probs)
                    bin_probs = [p / total for p in bin_probs]

                    # Sample from bin centers with replacement
                    n_samples = len(df)
                    synthetic_data = np.random.choice(bin_centers, size=n_samples, p=bin_probs)

                    # Assign to DataFrame
                    dp_df[var_name] = synthetic_data

    # Store the original column order
    dp_df.attrs['dp_metadata']['original_columns'] = list(df.columns)

    # Store processing summary if available
    if "processing_summary" in dp_results:
        dp_df.attrs['dp_metadata']['processing_summary'] = dp_results["processing_summary"]

    # Store security notice
    dp_df.attrs['dp_metadata']['security_notice'] = dp_results.get("security_notice",
                                                                   "This DataFrame contains privacy-protected data.")

    return dp_df


def apply_differential_privacy(df: pd.DataFrame,
                               variables: Union[str, List[str]],
                               epsilon: float = 0.1,
                               delta: float = 1e-5,
                               distribution: NoiseDistribution = "laplace",
                               operation: str = 'mean',
                               budget_manager: Optional[PrivacyBudgetManager] = None,
                               return_type: Literal['verbose', 'dataframe'] = 'verbose',
                               randomize_order: bool = True,
                               debug_mode: bool = False,
                               **kwargs) -> PrivacyResult:
    """
    Apply differential privacy to one or more variables in a DataFrame.

    This function automatically detects data types and applies the appropriate
    differential privacy mechanism based on the variable's data type:
    - Categorical: Adds noise to category counts
    - Boolean: Treats as special case of categorical with True/False
    - Numerical (int/float): Adds noise to the specified statistical operation
    - String: Reports noisy counts of top-k most frequent values
    - Datetime: Creates histogram with noisy bin counts
    - Timedelta: Creates histogram with noisy bin counts

    If the privacy budget is exceeded during processing, the function will stop applying
    differential privacy to new variables, but will return the results for variables already processed.

     Security Notice:
    ---------------
    This implementation uses deterministic noise generation based on query parameters,
    which ensures that identical queries will always produce identical results,
    even across different sessions. This decreases the risk of multiple-query averaging attacks
    while maintaining differential privacy guarantees.

    For maximum security, consider fixing the epsilon value in your application rather than
    allowing users to specify it directly. Variable epsilon values could enable reconstruction
    attacks over time if an adversary can make multiple similar queries with different
    epsilon values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        variables (Union[str, List[str]]): Column name(s) to apply differential privacy to.
        epsilon (float): The privacy parameter (smaller means more privacy).
        delta (float): The failure probability parameter (used only with Gaussian mechanism).
        distribution (NoiseDistribution): The differential privacy mechanism to use:
            - 'laplace': Provides pure ε-differential privacy (default, better for medical data)
            - 'gaussian': Provides (ε,δ)-differential privacy (better for complex analytics)
        operation (str): For numerical data, the statistical operation to perform:
            - 'mean': Calculate the average (default, appropriate for continuous variables)
            - 'sum': Calculate the sum of values
            - 'count': Count occurrences
        budget_manager (Optional[PrivacyBudgetManager]): A privacy budget manager.
        return_type (Literal['verbose', 'dataframe']): Type of return value:
            - 'verbose': Return detailed results dictionary (default)
            - 'dataframe': Return a DataFrame with differentially private values
        randomize_order (bool): Whether to randomise the order of variables when applying DP.
                               This ensures that if the budget is exceeded, different variables
                               will be skipped each time rather than always the last ones.
        debug_mode (bool): When True, includes true values and additional information for debugging.
                           WARNING: May leak sensitive information. Use only in secure environments.
        **kwargs: Additional arguments for specific data types:
            - top_k (int): Number of top values to report for string variables
            - bin_count (int): Number of bins for datetime/timedelta histograms
            - bins (List): Explicit bin edges for datetime/timedelta

    Returns:
        Union[Dict[str, Any], pd.DataFrame]: Either detailed results or a DataFrame with DP results.

    Mechanism Comparison:
    | Mechanism | Privacy Guarantee | Parameters | Best For | Noise Distribution |
    |-----------|------------------|------------|----------|-------------------|
    | Laplace   | ε-DP (pure)      | ε only     | Medical data, simple queries | Sharper peak, heavier tails |
    | Gaussian  | (ε,δ)-DP (approx)| ε and δ    | Complex analytics, high dimensions | Bell curve, lighter tails |

    Epsilon Value Trade-offs:
    | Epsilon Value | Privacy Protection | Result Accuracy | Typical Use Case |
    |---------------|-------------------|----------------|------------------|
    | 0.01 - 0.1    | Very Strong       | Low            | Highly sensitive data (HIV status, etc.) |
    | 0.1 - 1.0     | Strong            | Moderate       | Medical data, survey responses |
    | 1.0 - 10.0    | Moderate          | High           | Less sensitive aggregate statistics |
    | > 10.0        | Weak              | Very High      | Public or less sensitive data |

    Example:
        # Get detailed results with Laplace mechanism (default)
        results = apply_differential_privacy(df, ['age', 'satisfaction'])

        # Use Gaussian mechanism for high-dimensional data
        results = apply_differential_privacy(df, ['age', 'satisfaction'], distribution='gaussian')

        # Get a DataFrame with DP results
        dp_df = apply_differential_privacy(df, ['age', 'satisfaction'], return_type='dataframe')
    """
    # Log warning if debug mode is enabled
    if debug_mode:
        safe_log("warn",
                 "DEBUG MODE ACTIVE: This operation includes true values and should be used only in secure environments")

    # Log which mechanism is being used
    safe_log("info", f"Using {distribution} mechanism for differential privacy")

    # Handle both single variable and list of variables
    if isinstance(variables, str):
        variables_list = [variables]
    else:
        variables_list = variables

    safe_log("info", f"Applying differential privacy to {len(variables_list)} variables")

    # Create a budget manager if none was provided
    if budget_manager is None:
        budget_manager = PrivacyBudgetManager(epsilon, delta)

    # Distribute epsilon equally among variables
    epsilon_per_variable = epsilon / len(variables_list)

    # Extract additional parameters
    top_k = kwargs.get('top_k', 10)
    bin_count = kwargs.get('bin_count', 10)
    bins = kwargs.get('bins', None)

    # Create a copy of the variables list and potentially randomise order
    variables_to_process = variables_list.copy()
    if randomize_order:
        random.shuffle(variables_to_process)
        safe_log("info", f"Processing variables in randomised order")

    results: Dict[str, Any] = {}
    processed_variables: List[str] = []
    skipped_variables: List[str] = []
    variable_map: Dict[str, bool] = {}  # Map to track which variables were processed

    for variable in variables_to_process:
        if variable not in df.columns:
            safe_log("warn", f"Variable {variable} not found in DataFrame, skipping")
            skipped_variables.append(variable)
            continue

        # Check if we have enough budget remaining
        if not budget_manager.check_and_consume(
                epsilon_per_variable,
                operation_id=f"dp_{variable}"
        ):
            # If not enough budget, add to skipped variables and continue
            skipped_variables.append(variable)
            continue

        # Keep track of processed variables
        processed_variables.append(variable)
        variable_map[variable] = True

        # Safely determine data type and apply appropriate DP mechanism
        try:
            if isinstance(df[variable].dtype, pd.CategoricalDtype) or hasattr(df[variable].dtype, 'categories'):
                # Categorical data
                safe_log("info", f"Applying categorical differential privacy to {variable}")
                results[variable] = _apply_dp_to_categorical(
                    df, variable, epsilon_per_variable, delta, debug_mode, distribution
                )

            elif pd.api.types.is_bool_dtype(df[variable]):
                # Boolean data
                safe_log("info", f"Applying boolean differential privacy to {variable}")
                results[variable] = _apply_dp_to_boolean(
                    df, variable, epsilon_per_variable, delta, debug_mode, distribution
                )

            elif pd.api.types.is_string_dtype(df[variable]):
                # String data
                safe_log("info", f"Applying string differential privacy to {variable}")
                results[variable] = _apply_dp_to_string(
                    df, variable, epsilon_per_variable, delta, top_k, debug_mode, distribution
                )

            elif pd.api.types.is_datetime64_dtype(df[variable]):
                # Datetime data
                safe_log("info", f"Applying datetime differential privacy to {variable}")
                results[variable] = _apply_dp_to_datetime(
                    df, variable, epsilon_per_variable, delta, bins, bin_count, debug_mode, distribution
                )

            elif pd.api.types.is_timedelta64_dtype(df[variable]):
                # Timedelta data
                safe_log("info", f"Applying timedelta differential privacy to {variable}")
                results[variable] = _apply_dp_to_timedelta(
                    df, variable, epsilon_per_variable, delta, bins, bin_count, debug_mode, distribution
                )

            elif pd.api.types.is_numeric_dtype(df[variable]):
                # Numerical data (int or float)
                safe_log("info", f"Applying numerical differential privacy to {variable} with operation '{operation}'")

                # Calculate the true result using safe calculation
                if operation == 'mean':
                    true_result = safe_calculate(lambda: df[variable].mean(), default_value=0.0)
                elif operation == 'sum':
                    true_result = safe_calculate(lambda: df[variable].sum(), default_value=0.0)
                elif operation == 'count':
                    true_result = safe_calculate(lambda: df[variable].count(), default_value=0)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")

                # Calculate sensitivity based on operation and data
                sensitivity = safe_calculate(
                    _calculate_sensitivity,
                    default_value=1.0,  # Conservative default
                    df=df,
                    column_name=variable,
                    operation=operation
                )

                # Generate deterministic seed for this query
                query_seed = _generate_query_seed(
                    df_shape=df.shape,
                    variable=variable,
                    epsilon=epsilon_per_variable,
                    delta=delta,
                    distribution=distribution,
                    operation=operation
                )

                # Generate noise using the selected distribution with deterministic seed
                noise = _generate_noise(
                    sensitivity,
                    epsilon_per_variable,
                    delta,
                    distribution,
                    seed=query_seed
                )
                private_result = true_result + noise

                # Get noise scale for metadata
                if distribution == "laplace":
                    noise_scale = _get_laplace_scale(sensitivity, epsilon_per_variable)
                else:
                    noise_scale = _get_gaussian_noise_scale(sensitivity, delta, epsilon_per_variable)

                result_dict: Dict[str, Any] = {
                    "operation": operation,
                    "private_result": private_result,
                    "epsilon_used": epsilon_per_variable,
                    "sample_count": len(df),
                    "metadata": {
                        "sensitivity": sensitivity,
                        "noise_scale": noise_scale,
                        "distribution": distribution
                    }
                }

                # Include true values only in debug mode
                if debug_mode:
                    result_dict["true_result"] = true_result

                results[variable] = result_dict
            else:
                # Unknown or unsupported data type
                safe_log("warn", f"Unsupported data type {df[variable].dtype} for variable {variable}, skipping")
                skipped_variables.append(variable)
        except Exception as e:
            # Handle any unexpected errors safely
            safe_log("warn", f"Error processing variable {variable}: {type(e).__name__}. Skipping")
            skipped_variables.append(variable)

    # Check which variables from the original list were not processed
    for variable in variables_list:
        if variable not in variable_map and variable not in skipped_variables:
            skipped_variables.append(variable)

    # Add budget usage information
    results["privacy_budget"] = {
        "total": epsilon,
        "used": budget_manager.used_epsilon,
        "remaining": budget_manager.get_remaining_budget()
    }

    # Add variables processing information
    results["processing_summary"] = {
        "processed_variables": processed_variables,
        "skipped_variables": skipped_variables,
        "total_requested": len(variables_list),
        "total_processed": len(processed_variables),
        "total_skipped": len(skipped_variables),
        "randomised_order": randomize_order,
        "distribution": distribution
    }

    # If no variables were processed, warn the user
    if len(processed_variables) == 0:
        safe_log("warn", "No variables were processed. Either privacy budget was too small or no variables were found")

    # If some variables were skipped due to budget, warn the user
    if any(v in variables_list for v in skipped_variables):
        count = len(skipped_variables)
        safe_log("warn",
                 f"Some {'variable was' if count == 1 else 'variables were'} skipped due to privacy budget constraints: "
                 f"{count} {'variable' if count == 1 else 'variables'}")

    # Add security notice
    results["security_notice"] = "This output contains privacy-protected information only."
    if debug_mode:
        results["security_warning"] = "DEBUG MODE ACTIVE: This output contains true values and should not be shared."

    # Sanitise metadata if not in debug mode
    if not debug_mode:
        for var_name, result in results.items():
            if var_name not in ["privacy_budget", "processing_summary", "security_notice", "security_warning"]:
                if "metadata" in result:
                    results[var_name]["metadata"] = _sanitise_metadata(result["metadata"])

    # Return based on specified return_type
    if return_type == 'dataframe':
        return _get_dp_dataframe(df, results)
    else:
        return results
