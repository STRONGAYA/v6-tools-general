"""
------------------------------------------------------------------------------
Statistical Analysis Functions

File organisation:
- Public API functions for computation
- Orchestration functions (_orchestrate_*)
- Basic statistical computation functions (_compute_local_*)
- Aggregate computation functions (_compute_aggregate_*)
------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

from io import StringIO as stringIO
from vantage6.algorithm.tools.exceptions import InputError

# Import safe logging and calculation functions from misc
from .miscellaneous import safe_log, safe_calculate


def compute_aggregate_general_statistics(
    results: List[Dict[str, Any]], return_partials: bool = False
) -> Dict[str, Union[str, None, List[Dict[str, Any]]]]:
    """
    Compute aggregate general statistics from the results of multiple organisations.

    Args:
        results (List[Dict[str, Any]]): List of dictionaries containing the results from each organisation.
        return_partials (bool): Whether to return partial results. Defaults to False.

    Returns:
        Dict[str, Union[str, None, List[Dict[str, Any]]]]:
        A dictionary containing the aggregated general statistics for categorical and numerical variables.
    """
    # Aggregate results
    aggregate_categorical_df = pd.DataFrame(columns=["variable", "value", "count"])
    aggregate_numerical_df = pd.DataFrame(columns=["variable", "statistic", "value"])

    for result in results:
        for result_, variables in result.items():
            if result_ == "categorical_general_partial_statistics":
                categorical_df = pd.read_json(
                    stringIO(result["categorical_general_partial_statistics"])
                )

                # Avoid concatenating empty DataFrames
                if aggregate_categorical_df.empty:
                    aggregate_categorical_df = categorical_df
                else:
                    aggregate_categorical_df = pd.concat(
                        [aggregate_categorical_df, categorical_df]
                    )

            elif result_ == "numerical_general_partial_statistics":
                numerical_df = pd.read_json(
                    stringIO(result["numerical_general_partial_statistics"])
                )

                # Drop the statistics that have to be aggregated separately or should not be aggregated
                numerical_df = numerical_df[
                    ~numerical_df["statistic"].isin(["mean", "std"])
                ]

                # Avoid concatenating empty DataFrames
                if aggregate_numerical_df.empty:
                    aggregate_numerical_df = numerical_df
                else:
                    aggregate_numerical_df = pd.concat(
                        [aggregate_numerical_df, numerical_df], ignore_index=True
                    )

    # Aggregate the categorical results
    aggregate_categorical_df = safe_calculate(
        _orchestrate_aggregate_categorical_statistics,
        pd.DataFrame(columns=["variable", "value", "count"]),
        df=aggregate_categorical_df,
    )

    # Aggregate the numerical results
    aggregate_numerical_df = safe_calculate(
        _orchestrate_aggregate_numerical_statistics,
        pd.DataFrame(columns=["variable", "statistic", "value"]),
        df=aggregate_numerical_df,
    )

    if return_partials:
        safe_log("warn", "Returning partial general statistics")
        return {
            "categorical_general_statistics": aggregate_categorical_df.to_json(),
            "numerical_general_statistics": aggregate_numerical_df.to_json(),
            "partial_results": results,
        }
    else:
        return {
            "categorical_general_statistics": aggregate_categorical_df.to_json(),
            "numerical_general_statistics": aggregate_numerical_df.to_json(),
        }


def compute_aggregate_adjusted_deviation(
    results_adjusted_deviation: List[Dict[str, Any]],
    results_general_statistics: Optional[Dict[str, str]] = None,
    return_partials: bool = False,
) -> Dict[str, Any]:
    """
    Compute aggregate adjusted deviation from the results of multiple organisations.

    Args:
        results_adjusted_deviation (List[Dict[str, Any]]): List of dictionaries containing the
                                                            adjusted deviation results.
        results_general_statistics (Optional[Dict[str, str]]): Dictionary containing the general statistics.
        return_partials (bool): Whether to return partial results. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing the aggregated adjusted deviation results.
    """
    if results_general_statistics is None:
        results_general_statistics = {}

    # Collect the numerical aggregated results - which were already computed
    aggregate_numerical_df = pd.read_json(
        stringIO(results_general_statistics.get("numerical_general_statistics", "{}"))
    )

    aggregate_deviation_df = pd.DataFrame(columns=["variable", "statistic", "value"])

    for result in results_adjusted_deviation:
        for result_, variables in result.items():
            if result_ == "adjusted_deviation":
                deviation_df = pd.read_json(stringIO(result["adjusted_deviation"]))

                # Avoid concatenating empty DataFrames
                if aggregate_deviation_df.empty:
                    aggregate_deviation_df = deviation_df
                else:
                    aggregate_deviation_df = pd.concat(
                        [aggregate_deviation_df, deviation_df], ignore_index=True
                    )

    # Aggregate the adjusted deviation results safely
    aggregate_deviation_df = safe_calculate(
        _orchestrate_aggregate_adjusted_deviation,
        pd.DataFrame(columns=["variable", "statistic", "value"]),
        df=aggregate_deviation_df,
    )

    # Merge the aggregate-adjusted deviation with the general statistics
    aggregate_deviation_df = pd.concat(
        [aggregate_numerical_df, aggregate_deviation_df], ignore_index=True
    ).sort_values("variable")

    # Add the adjusted deviation to the general statistics
    results_general_statistics.update(
        {"numerical_general_statistics": aggregate_deviation_df.to_json()}
    )

    if return_partials:
        safe_log("warn", "Returning partial aggregate-adjusted deviation statistics")
        results_general_statistics.update(
            {"partial_deviation_results": str(results_adjusted_deviation)}
        )
        return results_general_statistics
    else:
        return results_general_statistics


def compute_local_general_statistics(
    df: pd.DataFrame, variable_details: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, str]:
    """
    Compute local general statistics for categorical and numerical variables in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        variable_details (Optional[Dict[str, Dict[str, Any]]]): A dictionary where keys are column names and
                                    values are dictionaries containing inliers.

    Returns:
        Dict[str, str]: A dictionary containing categorical and numerical statistics in JSON format.
    """
    safe_log("info", "Computing local general statistics")

    # Initialise empty DataFrames for categorical and numerical statistics; in case a variable type is not used
    categorical_statistics = pd.DataFrame(columns=["variable", "value", "count"])
    numerical_statistics = pd.DataFrame(columns=["variable", "statistic", "value"])

    # Separate categorical and numerical columns
    categorical_columns = [
        col for col in df.columns if isinstance(df[col].dtype, pd.CategoricalDtype)
    ]
    numerical_columns = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])
    ]

    # Process categorical variables if any exist
    if categorical_columns:
        categorical_statistics = safe_calculate(
            _orchestrate_local_categorical_statistics,
            pd.DataFrame(columns=["variable", "value", "count"]),
            df=df[categorical_columns],
            variable_details=variable_details,
        )

    # Process numerical variables if any exist
    if numerical_columns:
        numerical_statistics = safe_calculate(
            _orchestrate_local_numerical_statistics,
            pd.DataFrame(columns=["variable", "statistic", "value"]),
            df=df[numerical_columns],
            variable_details=variable_details,
        )

    return {
        "categorical_general_partial_statistics": categorical_statistics.to_json(),
        "numerical_general_partial_statistics": numerical_statistics.to_json(),
    }


def compute_local_adjusted_deviation(
    df: pd.DataFrame, numerical_aggregated_results: Optional[str] = None
) -> Dict[str, str]:
    """
    Compute local adjusted deviation for the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        numerical_aggregated_results (Optional[str]): JSON string with the general numerical statistics.

    Returns:
        Dict[str, str]: A dictionary with the local adjusted deviation in JSON format.
    """
    # Collect the general numerical aggregates (safely)
    numerical_results = (
        "{}" if numerical_aggregated_results is None else numerical_aggregated_results
    )
    numerical_df = pd.read_json(stringIO(numerical_results))

    # Collect the variable(s) for which an adjusted deviation can actually be calculated
    variables_to_analyse = [
        column_name
        for column_name in df.columns
        if column_name in numerical_df["variable"].unique()
    ]

    if not variables_to_analyse:
        # Initialise an empty DataFrame to return
        adjusted_deviation = pd.DataFrame(columns=["variable", "statistic", "value"])

        safe_log(
            "warn",
            "No variables to analyse for adjusted deviation due to lacking aggregate numerical statistics",
        )
    else:
        adjusted_deviation = safe_calculate(
            _orchestrate_local_adjusted_deviation,
            pd.DataFrame(columns=["variable", "statistic", "value"]),
            df=df,
            numerical_aggregated_results=numerical_df,
        )

    return {"adjusted_deviation": adjusted_deviation.to_json()}


def _orchestrate_aggregate_categorical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate categorical statistics from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing categorical statistics with columns "variable", "value", and "count".

    Returns:
        pd.DataFrame: Aggregated DataFrame with combined counts for each variable and value.
    """
    aggregated_df = df.groupby(["variable", "value"], as_index=False).sum()
    return aggregated_df


def _orchestrate_aggregate_numerical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate numerical statistics from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with numerical statistics.

    Returns:
        pd.DataFrame: Aggregated statistics DataFrame.
    """
    aggregated_results = []
    has_predetermined_info = hasattr(df, "predetermined_info")

    for variable in df["variable"].unique():
        # Filter the DataFrame for the current variable
        column_statistics = df[df["variable"] == variable]

        # Sort index to prevent performance warning
        column_statistics_series = column_statistics.set_index(
            ["variable", "statistic"]
        )["value"].sort_index()

        # Get all predetermined stats for this column if available
        variable_stats = {}
        if has_predetermined_info:
            try:
                variable_stats = df.predetermined_info.get_column_stats(variable)
            except InputError:
                pass

        # Compute summable statistics if they do not exist yet
        if "summable_statistics" in variable_stats:
            column_statistics_series = pd.Series(variable_stats["summable_statistics"])
        else:
            column_statistics_series = safe_calculate(
                _compute_aggregate_summable_statistics,
                column_statistics_series,
                numerical_statistics=column_statistics_series,
                statistics_to_sum=["sum", "count", "outliers", "na", "sq_dev_sum"],
            )

        # Sort index for performance
        column_statistics_series = column_statistics_series.sort_index()

        # Calculate aggregate statistics safely if they do not exist yet
        if "min_max_values" in variable_stats:
            min_max_values = pd.Series(variable_stats["min_max_values"])
        else:
            min_max_values = safe_calculate(
                _compute_aggregate_minmax,
                {"min": 0.0, "max": 0.0},
                numerical_statistics=column_statistics_series,
            )

        # Calculate federated quantiles safely if they do not exist yet
        if "federated_quantiles" in variable_stats:
            federated_quantiles = variable_stats["federated_quantiles"]
        else:
            federated_quantiles = safe_calculate(
                _compute_aggregate_quantiles,
                {
                    "Q1": 0.0,
                    "Q2": 0.0,
                    "Q3": 0.0,
                    "Q1_std_err": 0.0,
                    "Q2_std_err": 0.0,
                    "Q3_std_err": 0.0,
                },
                numerical_statistics=column_statistics_series,
            )

        # Calculate mean and safely if it does not exist yet
        if "mean" in variable_stats:
            mean = variable_stats["mean"]
        else:
            mean = safe_calculate(
                _compute_aggregate_mean,
                0.0,
                numerical_statistics=column_statistics_series,
            )

        # Calculate standard deviation safely if it does not exist yet
        if "std" in variable_stats:
            std = variable_stats["std"]
        else:
            std = safe_calculate(
                _compute_aggregate_deviation,
                0.0,
                numerical_statistics=column_statistics_series,
            )

        # Create DataFrame with aggregated statistics
        aggregated_stats = pd.DataFrame(
            {
                "variable": [variable] * 10,
                "statistic": [
                    "min",
                    "q1",
                    "median",
                    "q3",
                    "max",
                    "mean",
                    "std",
                    "count",
                    "outliers",
                    "na",
                ],
                "value": [
                    float(min_max_values["min"]),
                    float(federated_quantiles["Q1"]),
                    float(federated_quantiles["Q2"]),
                    float(federated_quantiles["Q3"]),
                    float(min_max_values["max"]),
                    float(mean),
                    float(std),
                    float(column_statistics_series.loc[(variable, "count")].iloc[0]),
                    float(column_statistics_series.loc[(variable, "outliers")].iloc[0]),
                    float(column_statistics_series.loc[(variable, "na")].iloc[0]),
                ],
            }
        )

        aggregated_results.append(aggregated_stats)

    # Concatenate results with explicit dtype specification
    result = pd.concat(aggregated_results, ignore_index=True)
    return result.astype({"variable": str, "statistic": str, "value": float})


def _orchestrate_aggregate_adjusted_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and aggregate adjusted deviations for each variable.

    Args:
        df (pd.DataFrame): DataFrame with columns "variable", "statistic", and "value"

    Returns:
        pd.DataFrame: Aggregated DataFrame with adjusted standard deviations
    """
    adjusted_deviations = []

    for variable in df["variable"].unique():
        # Filter the DataFrame for the current variable
        column_statistics = df[df["variable"] == variable]

        # Convert to Series with MultiIndex for computation
        column_statistics_series = column_statistics.set_index(
            ["variable", "statistic"]
        )["value"]

        # Compute the adjusted sum of squared errors safely
        adjusted_std = safe_calculate(
            _compute_aggregate_adjusted_deviation,
            0.0,
            numerical_statistics=column_statistics_series,
        )

        # Create DataFrame with the result
        aggregated_adjusted_deviations = pd.DataFrame(
            {
                "variable": [variable],
                "statistic": ["adjusted std"],
                "value": [adjusted_std],
            }
        )

        # Add to the list
        adjusted_deviations.append(aggregated_adjusted_deviations)

    # Combine all results
    if adjusted_deviations:
        result = pd.concat(adjusted_deviations, ignore_index=True)
        return result.astype({"variable": str, "statistic": str, "value": float})
    else:
        # Return an empty DataFrame with the correct columns if there are no results
        return pd.DataFrame(columns=["variable", "statistic", "value"])


def _orchestrate_local_categorical_statistics(
    df: pd.DataFrame, variable_details: Optional[Dict[str, Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Retrieve local general statistics for categorical variables in a DataFrame.

    This function processes categorical columns in the provided DataFrame,
    removes outliers based on the provided inliers list, and returns a DataFrame
    with the value counts and outliers for each categorical variable.

    Computation of given statistics can be skipped if they are already present in the predetermined_info
    attribute that can be generated through the miscellaneous module.
    Ensure that statistics that should be skipped can be found through the keys used in this function.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        variable_details (Optional[Dict[str, Dict[str, Any]]]): A dictionary where keys are column names and
                                  values are dictionaries containing inliers.

    Returns:
        pd.DataFrame: A DataFrame with columns "Variable", "Value", and "count"
                      representing the value counts and outliers for each categorical variable.
    """
    categorical_data = []
    has_predetermined_info = hasattr(df, "predetermined_info")

    # Iterate over each column
    for column_name in df.columns:
        safe_log(
            "info", f"General statistics for variable {column_name} are being computed"
        )

        # Get the column values
        column_values = df[column_name]

        # Get all predetermined stats for this column if available
        column_stats = {}
        if has_predetermined_info:
            try:
                column_stats = df.predetermined_info.get_column_stats(column_name)
            except InputError:
                pass

        # Get the value counts for the column safely
        if "value_counts" in column_stats:
            value_counts = pd.Series(column_stats["value_counts"])
        else:
            value_counts = safe_calculate(
                _compute_local_value_counts,
                pd.Series(dtype="float64"),
                column_values=column_values,
            )

        # Get the inliers for the column from the provided dictionary
        if variable_details is not None and column_name in variable_details:
            inliers = variable_details[column_name].get("inliers", None)
            datatype = variable_details[column_name].get("datatype", "categorical")
        else:
            inliers = None
            datatype = "categorical"

        # Get the inliers and outliers safely
        if "inliers_series" in column_stats and "outliers_series" in column_stats:
            inliers_series = pd.Series(column_stats["inliers_series"])
            outliers_series = pd.Series(column_stats["outliers_series"])
        else:
            inliers_series, outliers_series = safe_calculate(
                _compute_local_inliers_and_outliers,
                (pd.Series(dtype="float64"), pd.Series(dtype="float64")),
                column_values=value_counts,
                inliers=inliers,
                datatype=datatype,
            )

        # Get the missing values count and replace with pd.NA safely
        if "na" in column_stats:
            na_count = column_stats["na"]
        else:
            na_count, column_values = safe_calculate(
                _compute_local_missing_values,
                (0, pd.Series(dtype="float64")),
                column_values=column_values,
                replace_with_na=True,
            )

        # Append the value counts to the row's list
        for val, cnt in inliers_series.items():
            categorical_data.append((column_name, val, cnt))

        # Append the outliers count to the row's list
        categorical_data.append((column_name, "outliers", outliers_series.sum()))

        # Append the missing values count to the row's list
        categorical_data.append((column_name, "na", na_count))

    # Return the final DataFrame
    return pd.DataFrame(categorical_data, columns=["variable", "value", "count"])


def _orchestrate_local_numerical_statistics(
    df: pd.DataFrame, variable_details: Optional[Dict[str, Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Retrieve general statistics for numerical variables in a DataFrame.

    Computation of given statistics can be skipped if they are already present in the predetermined_info
    attribute that can be generated through the miscellaneous module.
    Ensure that statistics that should be skipped can be found through the keys used in this function.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        variable_details (Optional[Dict[str, Dict[str, Any]]]): A dictionary where keys are column names
                                      and values are dictionaries containing inliers.

    Returns:
        pd.DataFrame: A DataFrame with columns "variable", "statistic",
                      and "value" representing the statistics for each numerical variable.
    """
    numerical_data = []
    has_predetermined_info = hasattr(df, "predetermined_info")

    # Compute the general statistics for the numerical variables
    for column_name in df.columns:
        safe_log(
            "info", f"General statistics for variable {column_name} are being computed"
        )

        # Get the column values
        column_values = df[column_name]

        # Get all predetermined stats for this column if available
        column_stats = {}
        if has_predetermined_info:
            try:
                column_stats = df.predetermined_info.get_column_stats(column_name)
            except InputError:
                pass

        # Count the occurrences of missing values safely if it does not exist yet
        if "na" in column_stats:
            na_count = column_stats["na"]
        else:
            na_count, column_values = safe_calculate(
                _compute_local_missing_values,
                (0, pd.Series(dtype="float64")),
                column_values=column_values,
                replace_with_na=True,
            )

        # Get the inliers for the column from the provided dictionary
        if variable_details is not None and column_name in variable_details:
            inliers_range = variable_details[column_name].get(
                "inliers", [float("-inf"), float("inf")]
            )
            datatype = variable_details[column_name].get("datatype", "numerical")
        else:
            inliers_range = [float("-inf"), float("inf")]
            datatype = "numerical"

        # Identify outliers by excluding values outside the inliers range safely if they do not exist yet
        if "inlier_series" in column_stats and "outlier_series" in column_stats:
            inliers_series = pd.Series(column_stats["inlier_series"])
            outliers_series = pd.Series(column_stats["outlier_series"])
        else:
            inliers_series, outliers_series = safe_calculate(
                _compute_local_inliers_and_outliers,
                (pd.Series(dtype="float64"), pd.Series(dtype="float64")),
                column_values=column_values,
                inliers=inliers_range,
                datatype=datatype,
            )

        # Compute the mean safely if it does not exist yet
        if "mean" in column_stats:
            mean = column_stats["mean"]
        else:
            mean = safe_calculate(
                _compute_local_mean, 0.0, column_values=inliers_series
            )

        # Compute the minimum and maximum safely if they do not exist yet
        if "min_val" in column_stats and "max_val" in column_stats:
            min_val = column_stats["min_val"]
            max_val = column_stats["max_val"]
        else:
            min_val, max_val = safe_calculate(
                _compute_local_min_max, (0.0, 0.0), column_values=inliers_series
            )

        # Compute the number of rows safely if it does not exist yet
        if "number_of_rows" in column_stats:
            number_of_rows = column_stats["number_of_rows"]
        else:
            number_of_rows = safe_calculate(
                _compute_local_number_of_rows,
                0,
                column_values=inliers_series,
                drop_na=True,
            )

        # Compute quantiles safely if they do not exist yet
        if "quantiles" in column_stats:
            quantiles = column_stats["quantiles"]
        else:
            quantiles = safe_calculate(
                _compute_local_quantiles,
                {
                    "Q1": 0.0,
                    "variance_Q1": 0.0,
                    "Q2": 0.0,
                    "variance_Q2": 0.0,
                    "Q3": 0.0,
                    "variance_Q3": 0.0,
                },
                column_values=inliers_series,
            )

        # Compute the sum of rows safely if it does not exist yet
        if "sum_of_rows" in column_stats:
            sum_of_rows = column_stats["sum_of_rows"]
        else:
            sum_of_rows = safe_calculate(
                _compute_local_sum, 0.0, column_values=inliers_series
            )

        # Compute the sum of squared errors safely if it does not exist yet
        if "sum_errors2" in column_stats:
            sum_errors2 = column_stats["sum_errors2"]
        else:
            sum_errors2 = safe_calculate(
                _compute_local_sum_of_squared_errors, 0.0, column_values=inliers_series
            )

        # Append the statistics to the list
        numerical_data.extend(
            [
                (column_name, "min", min_val),
                (column_name, "Q1", quantiles["Q1"]),
                (column_name, "variance_Q1", quantiles["variance_Q1"]),
                (column_name, "Q2", quantiles["Q2"]),
                (column_name, "variance_Q2", quantiles["variance_Q2"]),
                (column_name, "Q3", quantiles["Q3"]),
                (column_name, "variance_Q3", quantiles["variance_Q3"]),
                (column_name, "max", max_val),
                (column_name, "mean", mean),
                (column_name, "na", na_count),
                (column_name, "sum", sum_of_rows),
                (column_name, "count", number_of_rows),
                (column_name, "sq_dev_sum", sum_errors2),
                (
                    column_name,
                    "std",
                    np.sqrt(sum_errors2 / number_of_rows if number_of_rows > 0 else 1),
                ),
                (column_name, "outliers", int(len(outliers_series))),
            ]
        )

    # Convert the list to a DataFrame
    numerical_df = pd.DataFrame(
        numerical_data, columns=["variable", "statistic", "value"]
    )

    return numerical_df


def _orchestrate_local_adjusted_deviation(
    df: pd.DataFrame,
    numerical_aggregated_results: pd.DataFrame,
    variable_details: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Compute local adjusted deviation for the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        numerical_aggregated_results (pd.DataFrame): DataFrame with general numerical statistics.
        variable_details (Optional[Dict[str, Dict[str, Any]]]): A dictionary where keys are column names
                                 and values are dictionaries containing inliers.

    Returns:
        pd.DataFrame: DataFrame with columns "variable", "statistic", and "value"
                      representing the adjusted deviation for each variable.
    """
    adjusted_deviations = []

    for column_name in df.columns:
        safe_log(
            "info", f"Adjusted deviation for variable {column_name} is being computed"
        )

        # Get the column values
        column_values = df[column_name]

        # Get the aggregated mean safely
        try:
            aggregated_mean = numerical_aggregated_results.loc[
                (numerical_aggregated_results["variable"] == column_name)
                & (numerical_aggregated_results["statistic"] == "mean"),
                "value",
            ].values[0]
        except IndexError:
            safe_log("warn", f"No aggregated mean found for variable {column_name}")
            continue

        # Get the inliers for the column from the provided dictionary
        if variable_details is not None and column_name in variable_details:
            inliers_range = variable_details[column_name].get(
                "inliers", [float("-inf"), float("inf")]
            )
            datatype = variable_details[column_name].get("datatype", "numerical")
        else:
            inliers_range = [float("-inf"), float("inf")]
            datatype = "numerical"

        # Identify outliers by excluding values outside the inliers range safely
        inliers_series, outliers_series = safe_calculate(
            _compute_local_inliers_and_outliers,
            (pd.Series(dtype="float64"), pd.Series(dtype="float64")),
            column_values=column_values,
            inliers=inliers_range,
            datatype=datatype,
        )

        # Compute the adjusted sum of squared errors safely
        adjusted_sum_of_squared_errors, number_of_rows = safe_calculate(
            _compute_local_aggregated_adjusted_deviation,
            (0.0, 0),
            inliers_series=inliers_series,
            aggregated_mean=aggregated_mean,
        )

        # Append the adjusted deviation to the list
        adjusted_deviations.append(
            (
                column_name,
                "adjusted_sum_of_squared_errors",
                adjusted_sum_of_squared_errors,
            )
        )
        adjusted_deviations.append((column_name, "count", number_of_rows))

    # Convert the list to a DataFrame
    adjusted_deviation_df = pd.DataFrame(
        adjusted_deviations, columns=["variable", "statistic", "value"]
    )

    return adjusted_deviation_df


def _compute_local_inliers_and_outliers(
    column_values: pd.Series, inliers: List[Any], datatype: Optional[str] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Identify inliers and outliers based on the provided inliers list or range.

    Args:
        column_values (pd.Series): A Series with the column values to compute the inliers and outliers for.
        inliers (List[Any]): A list of inliers for the categorical variable or
                                a list of inliers range for numerical variables.
        datatype (Optional[str]): The datatype of the variable ("categorical" or "numerical").

    Returns:
        Tuple[pd.Series, pd.Series]: A tuple containing two Series, one for inliers and one for outliers.
    """
    if inliers is None:
        safe_log(
            "warn",
            "No inliers provided, returning all values as inliers and no outliers",
        )
        return column_values, pd.Series(dtype="Float64")

    # Use explicit datatype if provided, otherwise fall back to dtype inspection
    if datatype == "categorical" or (
        datatype is None and isinstance(column_values.dtype, pd.CategoricalDtype)
    ):
        # Categorical variable - inliers is a list of allowed values
        inliers_series = column_values[column_values.isin(inliers)]
        outliers_series = column_values[~column_values.isin(inliers)]
    elif datatype == "numerical" or (
        datatype is None and pd.api.types.is_numeric_dtype(column_values)
    ):
        # Numerical variable - inliers should be a 2-element range [min, max]
        if not isinstance(inliers, list) or len(inliers) != 2:
            safe_log(
                "warn",
                f"For numerical variables, inliers must be a 2-element list [min, max]. Got: {inliers}. "
                "Proceeding without determining outliers",
            )
            inliers_series = column_values
            outliers_series = pd.Series(dtype="Float64")
        elif not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in inliers):
            safe_log(
                "warn",
                f"For numerical variables, inliers must contain numeric values. Got: {inliers}. "
                "Proceeding without determining outliers",
            )
            inliers_series = column_values
            outliers_series = pd.Series(dtype="Float64")
        else:
            inliers_series = column_values[
                (column_values >= inliers[0]) & (column_values <= inliers[1])
            ]
            outliers_series = column_values[
                (column_values < inliers[0]) | (column_values > inliers[1])
            ]
    else:
        safe_log(
            "warn",
            "Expected datatype to be 'categorical' or 'numerical'. Proceeding without determining outliers",
        )
        inliers_series = column_values
        outliers_series = pd.Series(dtype="Float64")

    return inliers_series, outliers_series


def _compute_local_mean(column_values: pd.Series) -> Union[int, float]:
    """
    Compute the mean using the local sum and non-NA rows.

    Args:
        column_values (pd.Series): Series of column values to compute the mean for.

    Returns:
        Union[int, float]: The local mean.
    """
    sum_val = safe_calculate(_compute_local_sum, 0.0, column_values=column_values)
    rows = safe_calculate(
        _compute_local_number_of_rows, 1, column_values=column_values, drop_na=True
    )

    # Avoid division by zero
    return sum_val / rows if rows > 0 else 0.0


def _compute_local_min_max(
    column_values: pd.Series,
) -> Tuple[Union[int, float], Union[int, float]]:
    """
    Compute the local minimum and maximum.

    Args:
        column_values (pd.Series): Series of column values to compute minimum and maximum for.

    Returns:
        Tuple[Union[int, float], Union[int, float]]: Tuple with the local minimum and maximum values.
    """
    # Handle empty series or all NaN values safely
    if column_values.empty or column_values.dropna().empty:
        return 0.0, 0.0

    min_val = column_values.dropna().min()
    max_val = column_values.dropna().max()

    # Convert to primitive types to avoid potential data leakage in numpy/pandas types
    return float(min_val), float(max_val)


def _compute_local_missing_values(
    column_values: pd.Series,
    placeholder: Union[int, str, pd._libs.missing.NAType] = pd.NA,
    replace_with_na: bool = False,
) -> Tuple[int, pd.Series]:
    """
    Count the occurrences of missing values and optionally replace them with pd.NA.

    Args:
        column_values (pd.Series): The input DataFrame containing the data.
        placeholder (Union[int, str, pd._libs.missing.NAType]): The placeholder value to identify missing values.
        replace_with_na (bool): Whether to replace the placeholder with pd.NA.

    Returns:
        Tuple[int, pd.Series]: The count of missing values and the updated column values.
    """
    if isinstance(placeholder, int):
        true_na_count = (column_values == placeholder).sum()
        if replace_with_na:
            column_values = column_values.replace(placeholder, pd.NA)
    elif isinstance(placeholder, str):
        true_na_count = column_values.eq(placeholder).sum()
        if replace_with_na:
            column_values = column_values.replace(placeholder, pd.NA)
    elif isinstance(placeholder, pd._libs.missing.NAType):
        true_na_count = column_values.isna().sum()
        if replace_with_na:
            column_values = column_values.where(~column_values.isna(), pd.NA)
    else:
        safe_log("warn", "Placeholder must be either an integer, a string, or pd.NA")
        return 0, column_values

    return true_na_count, column_values


def _compute_local_number_of_rows(
    column_values: pd.Series, drop_na: bool = True
) -> int:
    """
    Compute the local number of rows.

    Args:
        column_values (pd.Series): Series of column values to compute the number of rows for.
        drop_na (bool): Whether to drop nan rows, defaults to True.

    Returns:
        int: Local number of rows.
    """
    # Handle empty series safely
    if column_values.empty:
        return 0

    number_of_rows = column_values.dropna().size if drop_na else column_values.size
    return number_of_rows


def _compute_local_quantiles(
    column_values: pd.Series, iterations: int = 1000
) -> Dict[str, float]:
    """
    Compute local quantiles and their sampling variances.

    Args:
        column_values (pd.Series): Series of column values to compute quantiles for.
        iterations (int): Number of times to sample, default is 1000.

    Returns:
        Dict[str, float]: Dictionary with local quantiles and their sampling variances.
    """
    quantiles = {1: 0.25, 2: 0.50, 3: 0.75}
    results: Dict[str, float] = {}

    # Handle empty series or all NaN values safely
    if column_values.empty or column_values.dropna().empty:
        for i in quantiles.keys():
            results[f"Q{i}"] = 0.0
            results[f"variance_Q{i}"] = 0.0
        return results

    for i, q in quantiles.items():
        # Calculate quantile safely
        try:
            results[f"Q{i}"] = float(np.quantile(column_values.dropna().values, q))
        except Exception as e:
            safe_log("warn", f"Error computing quantile Q{i}: {type(e).__name__}")
            results[f"Q{i}"] = 0.0

        # Calculate variance safely
        results[f"variance_Q{i}"] = safe_calculate(
            _compute_local_quantile_sampling_variance,
            0.0,
            column_values=column_values,
            quantile=q,
            iterations=iterations,
        )

    return results


def _compute_local_quantile_sampling_variance(
    column_values: pd.Series, quantile: float, iterations: int
) -> float:
    """
    Estimate local sampling variance of the quantile.

    Args:
        column_values (pd.Series): Series of column values to estimate quantile sampling variance for.
        quantile (float): Quantile to estimate local sampling variance.
        iterations (int): Number of times to sample.

    Returns:
        float: Quantile sampling variance.
    """
    # Handle empty series or all NaN values safely
    column_values = column_values.dropna().values
    n = len(column_values)

    if n == 0:
        safe_log(
            "warn",
            "Column contains no actual values, cannot compute quantile sampling variance",
        )
        return 0.0

    np.random.seed(0)  # For reproducibility

    # Generate samples safely
    try:
        quantiles = [
            np.quantile(np.random.choice(column_values, size=n, replace=True), quantile)
            for _ in range(iterations)
        ]
    except Exception as e:
        safe_log("warn", f"Error sampling for quantile variance: {type(e).__name__}")
        return 0.0

    # Calculate variance safely
    quantile_variance = float(np.var(quantiles))

    # If variance is 0, use a small epsilon to avoid division by zero later
    if quantile_variance == 0:
        epsilon = (
            1e-10  # Very small number that will not meaningfully affect calculations
        )
        safe_log(
            "warn",
            f"Quantile sampling variance for {quantile} is 0, using epsilon value {str(epsilon)} instead",
        )
        quantile_variance = epsilon

    return quantile_variance


def _compute_local_sum(column_values: pd.Series) -> Union[int, float]:
    """
    Compute the local sum of the column's values.

    Args:
        column_values (pd.Series): Series of column values to compute the sum for.

    Returns:
        Union[int, float]: Local sum.
    """
    # Handle empty series safely
    if column_values.empty or column_values.dropna().empty:
        return 0.0

    # Use dropna to skip NA values and return a primitive type
    return float(column_values.dropna().sum())


def _compute_local_sum_of_squared_errors(
    column_values: pd.Series, mean: Optional[float] = None
) -> Union[int, float]:
    """
    Compute the local sum of squared errors.

    Args:
        column_values (pd.Series): Series of column values to compute sum of squared errors for
        mean (Optional[float]): Mean value to use for calculation, if None computed locally

    Returns:
        Union[int, float]: Local sum of squared errors
    """
    # Handle empty series safely
    if column_values.empty or column_values.dropna().empty:
        return 0.0

    # Facilitate the adjusted sum of squared errors computation using a provided mean - e.g. the aggregated mean
    if mean is None:
        # If a mean is not provided, compute it locally
        mean = safe_calculate(_compute_local_mean, 0.0, column_values=column_values)
    else:
        # If a mean is provided, ensure it is a float
        mean = float(mean)

    # Calculate safely and return primitive type
    try:
        return float(np.sum((column_values.dropna().values - mean) ** 2))
    except Exception as e:
        safe_log("warn", f"Error computing sum of squared errors: {type(e).__name__}")
        return 0.0


def _compute_local_value_counts(column_values: pd.Series) -> pd.Series:
    """
    Calculate value counts for a given variable.

    Args:
        column_values (pd.Series): The Series containing the data to take the value counts for.

    Returns:
        pd.Series: A Series with value counts for the variable.
    """
    # Handle empty series safely
    if column_values.empty:
        return pd.Series(dtype="float64")

    try:
        value_counts = column_values.value_counts()
        return value_counts
    except Exception as e:
        safe_log("warn", f"Error computing value counts: {type(e).__name__}")
        return pd.Series(dtype="float64")


def _compute_local_aggregated_adjusted_deviation(
    inliers_series: pd.Series, aggregate_mean: float
) -> Tuple[float, int]:
    """
    Compute the adjusted sum of squared errors and the number of rows.

    Args:
        inliers_series (pd.Series): Series of column values to compute the adjusted sum of squared errors for.
        aggregate_mean (float): The mean to use for the adjusted sum of squared errors.

    Returns:
        Tuple[float, int]: A tuple containing the adjusted sum of squared errors and the number of rows.
    """
    # Retrieve the adjusted sum of squared errors safely
    adjusted_sum_of_squared_errors = safe_calculate(
        _compute_local_sum_of_squared_errors,
        0.0,
        column_values=inliers_series,
        mean=aggregate_mean,
    )

    # Retrieve the local number of rows safely
    number_of_rows = safe_calculate(
        _compute_local_number_of_rows, 0, column_values=inliers_series, drop_na=True
    )

    return adjusted_sum_of_squared_errors, number_of_rows


def _compute_aggregate_summable_statistics(
    numerical_statistics: pd.Series, statistics_to_sum: List[str]
) -> pd.Series:
    """
    Filter and sum specific statistics from a series, and place the summed values back into the series.

    Args:
        numerical_statistics (pd.Series): Series containing statistics for a column per participating organisation.
        statistics_to_sum (List[str]): List of statistics to filter and sum.

    Returns:
        pd.Series: Series with the filtered and summed statistics placed back into the series.
    """
    # Handle empty series safely
    if numerical_statistics.empty:
        return numerical_statistics

    try:
        # Filter the Series based on the statistic level in the index
        filtered_stats = numerical_statistics[
            numerical_statistics.index.get_level_values("statistic").isin(
                statistics_to_sum
            )
        ]

        # Group and sum the filtered statistics
        summed_stats = filtered_stats.groupby(level=["variable", "statistic"]).sum()

        # Add the summed values back to the original Series
        # First remove the old values for these statistics
        mask = ~numerical_statistics.index.get_level_values("statistic").isin(
            statistics_to_sum
        )
        result = pd.concat([numerical_statistics[mask], summed_stats])

        return result
    except Exception as e:
        safe_log(
            "warn", f"Error computing aggregate summable statistics: {type(e).__name__}"
        )
        return numerical_statistics


def _compute_aggregate_minmax(numerical_statistics: pd.Series) -> Dict[str, float]:
    """
    Compute federated minimum and maximum values.

    Args:
        numerical_statistics (pd.Series): Series containing minimum and
                                            maximum values for a column per participating organisation.

    Returns:
        Dict[str, float]: Dictionary with federated minimum and maximum values.
    """
    # Handle empty series safely
    if numerical_statistics.empty:
        return {"min": 0.0, "max": 0.0}

    try:
        minimum = numerical_statistics[
            numerical_statistics.index.get_level_values("statistic") == "min"
        ].min()
        maximum = numerical_statistics[
            numerical_statistics.index.get_level_values("statistic") == "max"
        ].max()
        return {"min": float(minimum), "max": float(maximum)}
    except Exception as e:
        safe_log("warn", f"Error computing aggregate min/max: {type(e).__name__}")
        return {"min": 0.0, "max": 0.0}


def _compute_aggregate_mean(numerical_statistics: pd.Series) -> float:
    """
    Compute federated mean.

    Args:
        numerical_statistics (pd.Series): Series containing sums and counts for a column.

    Returns:
        float: Federated mean.
    """
    # Handle empty series safely
    if numerical_statistics.empty:
        return 0.0

    try:
        variable_name = numerical_statistics.index.get_level_values(
            "variable"
        ).unique()[0]
        total_sum = numerical_statistics.loc[(variable_name, "sum")].iloc[0]
        total_count = numerical_statistics.loc[(variable_name, "count")].iloc[0]

        return float(total_sum / total_count) if total_count > 0 else 0.0
    except Exception as e:
        safe_log("warn", f"Error computing aggregate mean: {type(e).__name__}")
        return 0.0


def _compute_aggregate_deviation(numerical_statistics: pd.Series) -> float:
    """
    Compute aggregate standard deviation.

    Args:
        numerical_statistics (pd.Series): Series containing sum of squared deviations and counts.

    Returns:
        float: Federated standard deviation.
    """
    # Handle empty series safely
    if numerical_statistics.empty:
        return 0.0

    try:
        variable_name = numerical_statistics.index.get_level_values(
            "variable"
        ).unique()[0]

        total_sq_dev_sum = numerical_statistics.loc[(variable_name, "sq_dev_sum")].iloc[
            0
        ]
        total_count = numerical_statistics.loc[(variable_name, "count")].iloc[0]

        return (
            float(np.sqrt(total_sq_dev_sum / total_count)) if total_count > 0 else 0.0
        )
    except Exception as e:
        safe_log("warn", f"Error computing aggregate deviation: {type(e).__name__}")
        return 0.0


def _compute_aggregate_quantiles(numerical_statistics: pd.Series) -> Dict[str, float]:
    """
    Compute aggregate quantiles.

    Args:
        numerical_statistics (pd.Series): Series containing local quantiles and their sampling variances.

    Returns:
        Dict[str, float]: Dictionary with aggregate quantiles and their standard errors.
    """
    # Handle empty series safely
    if numerical_statistics.empty:
        return {
            "Q1": 0.0,
            "Q2": 0.0,
            "Q3": 0.0,
            "Q1_std_err": 0.0,
            "Q2_std_err": 0.0,
            "Q3_std_err": 0.0,
        }

    aggregate_quantiles: Dict[str, float] = {}

    try:
        for i in range(1, 4):
            quantiles_i = numerical_statistics[
                numerical_statistics.index.get_level_values("statistic") == f"Q{i}"
            ].values
            variances_i = numerical_statistics[
                numerical_statistics.index.get_level_values("statistic")
                == f"variance_Q{i}"
            ].values

            # Skip if no data
            if len(quantiles_i) == 0 or len(variances_i) == 0:
                aggregate_quantiles[f"Q{i}"] = 0.0
                aggregate_quantiles[f"Q{i}_std_err"] = 0.0
                continue

            # Using DerSimonian and Laird method to estimate tau2
            # Equation 8 in https://doi.org/10.1016/j.cct.2006.04.004
            k = len(quantiles_i)
            omega_i0 = 1.0 / np.power(variances_i, 2)
            quantile_0 = np.sum(omega_i0 * quantiles_i) / np.sum(omega_i0)
            tau2_nom = np.sum(omega_i0 * np.power((quantiles_i - quantile_0), 2)) - (
                k - 1
            )
            tau2_den = np.sum(omega_i0) - np.sum(np.power(omega_i0, 2)) / np.sum(
                omega_i0
            )
            tau2 = np.max([0, tau2_nom / tau2_den])

            # Using approach from McGrath et al. (2019), section 2, see: https://doi.org/10.1002/sim.8013
            omega_i = 1.0 / (variances_i + tau2)
            aggregate_quantile = np.sum(quantiles_i * omega_i) / np.sum(omega_i)
            aggregate_quantile_std_err = np.sqrt(1.0 / np.sum(omega_i))
            aggregate_quantiles[f"Q{i}"] = float(aggregate_quantile)
            aggregate_quantiles[f"Q{i}_std_err"] = float(aggregate_quantile_std_err)

        return aggregate_quantiles
    except Exception as e:
        safe_log("warn", f"Error computing aggregate quantiles: {type(e).__name__}")
        return {
            "Q1": 0.0,
            "Q2": 0.0,
            "Q3": 0.0,
            "Q1_std_err": 0.0,
            "Q2_std_err": 0.0,
            "Q3_std_err": 0.0,
        }


def _compute_aggregate_adjusted_deviation(numerical_statistics: pd.Series) -> float:
    """
    Compute the aggregate-adjusted deviation.

    Args:
        numerical_statistics (pd.Series): local sums of squared errors and number of rows.

    Returns:
        float: aggregate-adjusted standard deviation
    """
    # Handle empty series safely
    if numerical_statistics.empty:
        return 0.0

    try:
        local_adjusted_sum_of_squared_errors = numerical_statistics[
            numerical_statistics.index.get_level_values("statistic")
            == "adjusted_sum_of_squared_errors"
        ].values
        local_number_of_rows = numerical_statistics[
            numerical_statistics.index.get_level_values("statistic") == "count"
        ].values

        if (
            len(local_adjusted_sum_of_squared_errors) == 0
            or len(local_number_of_rows) == 0
            or np.sum(local_number_of_rows) == 0
        ):
            return 0.0

        aggregate_deviation = np.sqrt(
            np.sum(local_adjusted_sum_of_squared_errors) / np.sum(local_number_of_rows)
        )
        return float(aggregate_deviation)
    except Exception as e:
        safe_log(
            "warn", f"Error computing aggregate adjusted deviation: {type(e).__name__}"
        )
        return 0.0
