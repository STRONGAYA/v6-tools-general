"""
------------------------------------------------------------------------------
Miscellaneous Utility Functions

File organisation:
- Core utility functions (safe_log, safe_calculate)
- Data handling utilities (collect_organisation_ids)
- Data transformation functions (apply_data_stratification, set_datatypes)
------------------------------------------------------------------------------
"""
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, cast

from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.client import AlgorithmClient


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


T = TypeVar('T')  # Define a type variable for the return type


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
                safe_log("error", f"Organisation ID {org_id} is not valid")
                return []

        return organisation_ids
    else:
        safe_log("error", "Organisation IDs should be a list of integers")
        return []


def apply_data_stratification(df: pd.DataFrame,
                              variables_to_stratify: Optional[
                                  Dict[str, Union[List[str], Dict[str, Union[int, float]]]]]) -> pd.DataFrame:
    """
    Stratify the DataFrame based on the specified variables.

    Caution: The flexibility provided by this function may facilitate differencing attacks if not implemented carefully.
             Consider including extra privacy-enhancing mechanisms by applying differential privacy post-stratification
             or restricting the variables on which one can stratify -
             e.g. through environment variables or a dedicated database and/or list.

    Args:
        df (pd.DataFrame): The DataFrame to be stratified.
        variables_to_stratify (Optional[Dict[str, Union[List[str], Dict[str, Union[int, float]]]]]):
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
            # Handle list of values
            query_conditions.append(f"`{variable}` in {values}")

    query_string = " and ".join(query_conditions)

    # Apply the query to filter the DataFrame
    stratified_df = df.query(query_string)

    return stratified_df.reset_index(drop=True)


def set_datatypes(df: pd.DataFrame,
                  variable_details: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Set the datatypes for each variable in the DataFrame based on the provided details.

    Args:
        df (pd.DataFrame): The DataFrame to be modified.
        variable_details (Dict[str, Dict[str, Any]]): A dictionary where keys are column names and
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
                safe_log("error", f"Unknown datatype '{datatype}' for variable '{variable}'")
    return df
