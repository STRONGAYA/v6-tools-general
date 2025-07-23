"""
------------------------------------------------------------------------------
Miscellaneous Utility Functions

File organisation:
- Core utility functions (safe_log, safe_calculate)
- Data handling utilities (collect_organisation_ids)
- Data transformation functions (apply_data_stratification, set_datatypes)
- DataFrame accessor for storing and retrieving predetermined information and/or statistics (PredeterminedInfoAccessor)
------------------------------------------------------------------------------
"""
import json

import pandas as pd

from typing import Any, Callable, cast, Dict, List, Optional, Union, Tuple, TypedDict, TypeVar

from vantage6.algorithm.tools.exceptions import AlgorithmError, CollectOrganizationError, DataError, InputError
from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.client import AlgorithmClient


# Type definitions for data stratification
class RangeDetails(TypedDict):
    start: Optional[Union[int, float]]
    end: Optional[Union[int, float]]


class StratificationDetails(TypedDict):
    variables_to_stratify: Optional[Dict[str, Union[List[str], RangeDetails]]]


# Type definition for safe calculation
T = TypeVar('T')


# Type definitions for variable details (data type setting)
class CategoricalDetails(TypedDict):
    datatype: str
    inliers: List[str]


class NonCategoricalDetails(TypedDict):
    datatype: str
    inliers: List[int | float]


VariableDetails = Union[CategoricalDetails, NonCategoricalDetails]


def apply_data_stratification(df: pd.DataFrame, variables_to_stratify: Optional[
    StratificationDetails['variables_to_stratify']]) -> pd.DataFrame:
    """
    Stratify the DataFrame based on the specified variables.

    Caution: The flexibility provided by this function may facilitate differencing attacks if not implemented carefully.
             Consider including extra privacy-enhancing mechanisms by applying differential privacy post-stratification
             or restricting the variables on which one can stratify -
             e.g. through environment variables or a dedicated database and/or list.

    Args:
        df (pd.DataFrame): The DataFrame to be stratified.
        variables_to_stratify (StratificationDetails):
            A dictionary where keys are column names and values are lists of acceptable values or range dictionaries.
            Example:
            {
                'gender': ['female'],
                'education': ['primary', 'secondary'],
                'age': {'start': 15, 'end': 39},
                'height': {'end': 196}
            }

    Returns:
        pd.DataFrame: The stratified DataFrame.
    """
    if variables_to_stratify is None:
        return df

    safe_log("info", "Stratifying data based on specified variables")

    # Check if the specified variables exist in the DataFrame
    for variable in variables_to_stratify:
        if variable not in df.columns:
            safe_log("error", f"Variable '{variable}' not found in DataFrame. Data stratification will not be applied")
            return df

    # Build the query string
    query_conditions = []
    for variable, values in variables_to_stratify.items():
        if isinstance(values, dict):
            # Handle range dictionary
            start = values.get('start')
            end = values.get('end')
            if start is not None and end is not None:
                query_conditions.append(f"`{variable}` >= {start} and `{variable}` <= {end}")
            elif start is not None:
                query_conditions.append(f"`{variable}` >= {start}")
            elif end is not None:
                query_conditions.append(f"`{variable}` <= {end}")
        else:
            # Handle the list of values
            query_conditions.append(f"`{variable}` in {values}")

    query_string = " and ".join(query_conditions)

    # Apply the query to filter the DataFrame
    stratified_df = df.query(query_string)

    return stratified_df.reset_index(drop=True)


def collect_organisation_ids(organisation_ids: Optional[List[int]], client: AlgorithmClient) -> List[int]:
    """
    Collect organisation IDs, ensuring they are a list of integers.
    If the input is not a list of integers, attempt to convert it.
    If the input is None, collect all organisation IDs.

    Args:
        organisation_ids (Optional[List[int]]): List of organisation IDs.
        client (AlgorithmClient): The client object to interact with the organisation API.

    Returns:
        List[int]: A list of valid organisation IDs.
    """
    if organisation_ids is None:
        safe_log("info", "No organisation IDs provided. Collecting all organisation IDs")
        organisations = client.organization.list()
        return [organisation.get("id") for organisation in organisations]

    if isinstance(organisation_ids, list):
        try:
            # Attempt to convert all elements to integers
            organisation_ids = [int(i) for i in organisation_ids]
        except ValueError:
            safe_log("error", "Organisation IDs should be a list of integers")
            return []

        # Check if the organisation IDs are valid
        for org_id in organisation_ids:
            if not client.organization.get(org_id):
                raise CollectOrganizationError(f"Organisation ID {org_id} is not valid")

        return organisation_ids
    else:
        safe_log("error", "Organisation IDs should be a list of integers")
        return []


def safe_log(level: str, message: str, variables: Optional[List[str]] = None) -> None:
    """
    Safely log messages without leaking sensitive data.

    Args:
        level (str): Log level ('info', 'warn', 'error')
        message (str): Message to log
        variables (Optional[List[str]]): List of variable names that are safe to include
    """
    # Only allow specific variable names to be logged, not values
    if variables:
        # If more than 5 variables, truncate the list to prevent data mining
        if len(variables) > 5:
            var_str = ", ".join(variables[:5]) + f" and {len(variables) - 5} more"
        else:
            var_str = ", ".join(variables)

        message = message.replace("{variables}", var_str)

    # Ensure that the message ends with a period if proper punctuation is not yet present
    if not message.endswith('.') and not message.endswith('?') and not message.endswith('!'):
        message += '.'

    # Call appropriate log function
    if level == "info":
        info(message)
    elif level == "warn":
        warn(message)
    elif level == "error":
        error(message)


def safe_calculate(calculation_func: Callable[..., T], default_value: T = None, **kwargs: Any) -> T:
    """
    Safely execute a calculation without leaking data in exceptions.

    Args:
        calculation_func (Callable[..., T]): Function to execute
        default_value (T): Value to return if calculation fails
        **kwargs: Arguments to pass to calculation_func

    Returns:
        T: Result of calculation_func or default_value if it fails
    """
    try:
        return calculation_func(**kwargs)
    except Exception as e:
        # Log generic error without data details
        safe_log("warn", f"Calculation error: {type(e).__name__}. Using default value")
        return cast(T, default_value)


def set_datatypes(df: pd.DataFrame, variable_details: Dict[str, VariableDetails]) -> pd.DataFrame:
    """
    Set the datatypes for each variable in the DataFrame based on the provided details.

    Args:
        df (pd.DataFrame): The DataFrame to be modified.
        variable_details (Dict[str, VariableDetails]): A dictionary where keys are column names and
                                values are dictionaries containing datatype information.
            Example:
            {
                'age': {'datatype': 'int'},
                'height': {'datatype': 'float'},
                'isSmoker': {'datatype': 'bool'},
                'gender': {'datatype': 'str'},
                'date_of_birth': {'datatype': 'datetime'},
                'duration': {'datatype': 'timedelta'},
                'education_level': {'datatype': 'categorical', 'inliers': ['primary', 'secondary', 'tertiary']}
            }

    Returns:
        pd.DataFrame: The DataFrame with updated datatypes.
    """
    safe_log("info", "Setting datatypes for variables in the DataFrame")
    for variable, details in variable_details.items():
        if variable in df.columns:
            datatype = details.get("datatype")
            if datatype == "int" or datatype == "integer":
                df[variable] = df[variable].astype('Int64')
            elif datatype == "float" or datatype == "numerical":
                df[variable] = df[variable].astype('Float64')
            elif datatype == "bool" or datatype == "boolean":
                df[variable] = df[variable].astype('boolean')
            elif datatype == "str" or datatype == "string":
                df[variable] = df[variable].astype('String')
            elif datatype == "datetime":
                df[variable] = pd.to_datetime(df[variable], errors='coerce')
            elif datatype == "timedelta":
                df[variable] = pd.to_timedelta(df[variable], errors='coerce')
            elif datatype == "categorical":
                missing_values = df[variable].isnull().sum()
                df[variable] = pd.Categorical(df[variable], categories=details.get('inliers', None))
                if df[variable].isnull().sum() > missing_values:
                    safe_log("warn", f"Unexpected values detected in categorical variable '{variable}',"
                                     f"coercing outliers to missing values")
            else:
                raise InputError(f"Unknown datatype '{datatype}' for variable '{variable}'")
    return df


def sum_variables(df: pd.DataFrame, variables: List[str], new_variable_name: str,
                  check_range: Tuple[float, float] = None, required_items: int = None,
                  strict: bool = True) -> pd.DataFrame:
    """
    Sum the specified variables in the DataFrame and create a new variable with the result.

    Warnings
        Setting the 'strict' parameter to False can use invalid data if adhering to a standardised scoring system
        and consequently skew the output.

    Args:
        df (pd.DataFrame): The DataFrame to be modified.
        variables (List[str]): List of variable names to sum.
        new_variable_name (str): Name of the new variable to store the sum.
        check_range (tuple[float, float], optional): Expected range for individual items (min, max).
        required_items (int, optional): Number of items required for valid score calculation.
        strict (bool): If True, raises errors for invalid data. If False, logs warnings (default: True).

    Returns:
        pd.DataFrame: The DataFrame with the new summed variable.
    """
    # Validate input ranges if specified
    if check_range is not None:
        min_val, max_val = check_range
        for var in variables:
            invalid_values = ~df[var].between(min_val, max_val)
            if invalid_values.any():
                error_msg = (f"Values in {var} outside expected range [{min_val}, {max_val}]. "
                             f"Found invalid values in rows: {df.index[invalid_values].tolist()}")
                if strict:
                    raise DataError(error_msg)
                else:
                    safe_log("warning", error_msg)

    # Check for missing values
    missing_count = df[variables].isnull().sum(axis=1)
    if required_items is not None:
        invalid_rows = (len(variables) - missing_count) < required_items
        if invalid_rows.any():
            error_msg = (f"Insufficient valid responses for {new_variable_name}. "
                         f"Required: {required_items}, Found invalid rows: {df.index[invalid_rows].tolist()}")
            if strict:
                raise DataError(error_msg)
            else:
                safe_log("warning", error_msg)

    # Calculate sum
    safe_log("info", f"Summing variables {variables} into '{new_variable_name}'")
    df[new_variable_name] = df[variables].sum(axis=1)

    return df


@pd.api.extensions.register_dataframe_accessor("predetermined_info")
class PredeterminedInfoAccessor:
    """
    A pandas DataFrame accessor for managing and storing predetermined information and statistics.

    This accessor provides methods to store, retrieve, and update statistical information
    about the DataFrame or its columns. It handles both single values and per-column statistics,
    with built-in safety measures for calculations and logging.

    The accessor stores all information in the DataFrame's attrs, making it persistent
    with the DataFrame. Each statistic includes metadata such as timestamp and user information.

    This can, for example, be used to address issues as one-to-many relationships skewing missing data computation.
    """

    def __init__(self, pandas_obj: pd.DataFrame):
        """
        Initialise the PredeterminedInfoAccessor.

        Args:
            pandas_obj (pd.DataFrame): The DataFrame this accessor is attached to
        """
        self._obj = pandas_obj
        self._initialized = False

    def _check_initialized(self) -> None:
        """
        Initialize the storage if not already done.

        Ensures the DataFrame has the necessary attribute storage structure.
        """
        if not self._initialized:
            if 'stats' not in self._obj.attrs:
                self._obj.attrs['stats'] = {}
            self._initialized = True

    def add_stat(self,
                 stat_name: str,
                 calculation_func: Optional[Callable[..., Any]] = None,
                 value: Any = None,
                 per_column: bool = False,
                 store_output_index: Optional[int] = None,
                 update_with_output_index: Optional[int] = None,
                 **kwargs: Any) -> None:
        """
        Add a custom statistic to the DataFrame.

        Args:
            stat_name (str): Name of the statistic to store
            calculation_func (Optional[Callable[..., Any]]): Function to calculate the statistic
            value (Any, optional): Direct value to store if not using calculation_func
            per_column (bool): If True, calculate/store the stat for each column separately
            store_output_index (Optional[int]): If function returns tuple, index of value to store
            update_with_output_index (Optional[int]): If function returns tuple, index to update DataFrame
            **kwargs: Additional arguments to pass to calculation_func

        Raises:
            ValueError: If the value is not JSON serialisable
        """
        self._check_initialized()
        safe_log("info", f"Adding predetermined information '{stat_name}' to DataFrame attributes")

        if value is None and calculation_func is not None:
            if per_column:
                value = {}
                updated_columns = {}
                for col in self._obj.columns:
                    def calculate_for_column(**kwargs: Any) -> Any:
                        result = calculation_func(self._obj[col], **kwargs)
                        if isinstance(result, tuple):
                            stat_value = result[store_output_index] if store_output_index is not None else result[0]
                            if update_with_output_index is not None:
                                updated_columns[col] = result[update_with_output_index]
                            return stat_value
                        return result

                    value[col] = safe_calculate(
                        calculate_for_column,
                        default_value=None,
                        **kwargs
                    )

                if updated_columns:
                    for col, series in updated_columns.items():
                        self._obj[col] = series
                    safe_log("info", f"Updated {len(updated_columns)} columns with new values")
            else:
                def calculate_for_df(**kwargs: Any) -> Any:
                    result = calculation_func(self._obj, **kwargs)
                    if isinstance(result, tuple):
                        return result[store_output_index] if store_output_index is not None else result[0]
                    return result

                value = safe_calculate(
                    calculate_for_df,
                    default_value=None,
                    **kwargs
                )

        try:
            json.dumps(value)
        except TypeError:
            safe_log("error", f"Value for information '{stat_name}' must be JSON serializable")
            raise AlgorithmError(f"Value for {stat_name} must be JSON serializable")

        self._obj.attrs['stats'][stat_name] = {
            'value': value,
            'per_column': per_column
        }
        safe_log("info", f"Successfully added information on '{stat_name}'")

    def get_stat(self, stat_name: str, column: Optional[str] = None) -> Any:
        """
        Retrieve a stored statistic.

        Args:
            stat_name (str): Name of the statistic to retrieve
            column (Optional[str]): If the stat is per-column, specify which column to get

        Returns:
            Any: The stored statistic value

        Raises:
            KeyError: If statistic not found
        """
        if stat_name not in self._obj.attrs['stats']:
            safe_log("error", f"Statistic '{stat_name}' not found")
            raise InputError(f"Statistic '{stat_name}' not found")

        stat = self._obj.attrs['stats'][stat_name]

        if stat['per_column']:
            if column is None:
                safe_log("info", f"Retrieving predetermined '{stat_name}' for all columns")
                return stat['value']
            safe_log("info", f"Retrieving predetermined '{stat_name}' for column '{column}'")
            return stat['value'].get(column)

        safe_log("info", f"Retrieving predetermined information on '{stat_name}'")
        return stat['value']

    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """
        Retrieve all stored statistics for a specific column.

        Args:
            column (str): Name of the column to get statistics for

        Returns:
            Dict[str, Any]: Dictionary containing all statistics for the column

        Raises:
            KeyError: If column not found in DataFrame
        """
        if column not in self._obj.columns:
            safe_log("error", f"Column '{column}' not found in DataFrame")
            raise InputError(f"Column '{column}' not found in DataFrame")

        safe_log("info", f"Retrieving all predetermined statistics for column '{column}'")

        column_stats = {}
        for stat_name, stat_info in self._obj.attrs.get('stats', {}).items():
            if stat_info['per_column']:
                if column in stat_info['value']:
                    column_stats[stat_name] = stat_info['value'][column]

        stats_count = len(column_stats)
        if stats_count == 0:
            safe_log("info", f"No predetermined statistics found for column '{column}'")
        elif stats_count <= 2:
            safe_log("warn", "For only 1-2 statistics, consider using get_stat() directly")
        else:
            safe_log("info", f"Retrieved {stats_count} statistics efficiently")

        return column_stats

    def list_stats(self) -> Dict[str, Any]:
        """
        List all stored statistics.

        Returns:
            Dict[str, Any]: Dictionary of stored statistics
        """
        safe_log("info", "Listing all available predetermined information")
        return self._obj.attrs.get('stats', {})

    def update_stat(self, stat_name: str, **kwargs: Any) -> None:
        """
        Update a specific statistic.

        Args:
            stat_name (str): Name of the statistic to update
            **kwargs: Arguments to pass to calculation function

        Raises:
            KeyError: If statistic not found
        """
        if stat_name not in self._obj.attrs['stats']:
            safe_log("error", f"Cannot update: predetermined information on '{stat_name}' not found")
            raise InputError(f"Predetermined information '{stat_name}' not found")

        safe_log("info", f"Updating information on '{stat_name}'")
        current_stat = self._obj.attrs['stats'][stat_name]
        self.add_stat(stat_name,
                      calculation_func=current_stat.get('calculation_func'),
                      per_column=current_stat['per_column'],
                      **kwargs)
