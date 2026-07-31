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
                    aggregate_categ
orical_df = categorical_df
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


def c
ompute_aggregate_adjusted_deviation(
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
        df=aggrega
te_deviation_df,
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
        adjusted_deviation = pd.DataFrame(columns=["variable", "st
atistic", "value"])

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
                numerical_statistics=column
_statistics_series,
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
        df (pd.DataFrame): DataFrame with columns "variable", "stat
istic", and "value"

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

    Computation of given statistics can be skipped if they are already pres
ent in the predetermined_info
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
            inliers
 = None
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

        # Calculate the missing value count safely
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
    Ensure that stat
istics that should be skipped can be found through the keys used in this function.

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
            quantiles = safe_calc
ulate(
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
            aggregate_mean=aggregated_mean,
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
        datatype (Op
tional[str]): The datatype of the variable ("categorical" or "numerical").

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
        inliers_series = column_values[column_values.index

... [Content truncated]


def _compute_local_missing_values(
    column_values: pd.Series,
    placeholder: Union[int, str, pd._libs.missing.NAType] = pd.NA,
    replace_with_na: bool = False,
) -> Tuple[int, pd.Series]:
    """
    Compute the count of missing values in a column.

    When a placeholder is provided (and not pd.NA), only counts cells matching that placeholder value.
    This prevents double-counting when structural NaN values exist alongside explicit placeholder 
    annotations (e.g., in RDF data with MISSING_DATA_NOTATION).

    Args:
        column_values (pd.Series): The pandas Series to check for missing values
        placeholder (Union[int, str, pd._libs.missing.NAType]): The placeholder value to identify missing values.
            When set to pd.NA (default), counts standard missing values (NaN, None, pd.NA).
            When set to a specific value, counts only cells matching that value.
        replace_with_na (bool): If True, replace the counted values with pd.NA

    Returns:
        Tuple[int, pd.Series]: Tuple of (missing_count, modified_column_values)
    """
    if placeholder is not pd.NA:
        # When a specific placeholder is provided, only count cells matching that placeholder
        # This is for RDF contexts where missing values are explicitly annotated
        missing_mask = column_values == placeholder
        na_count = int(missing_mask.sum())
        
        # Replace placeholder with NA if requested
        if replace_with_na:
            column_values = column_values.replace(placeholder, pd.NA)
    else:
        # When no placeholder or placeholder is pd.NA, count standard missing values (NaN, None, pd.NA)
        missing_mask = column_values.isna()
        na_count = int(missing_mask.sum())

    return (na_count, column_values)
