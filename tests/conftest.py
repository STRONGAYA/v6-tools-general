"""
Test configuration and fixtures for vantage6-strongaya-general testing suite.

This module provides common fixtures and utilities used across all test modules.
It includes synthetic data generation, mock clients, and shared test utilities.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from unittest.mock import MagicMock
import warnings

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture
def sample_numerical_data():
    """
    Generate a sample numerical dataset for testing statistical functions.
    
    Returns:
        pd.DataFrame: DataFrame with various numerical variables and edge cases.
    """
    np.random.seed(42)  # For reproducible tests

    n_samples = 1000
    data = {
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'height': np.random.normal(170, 10, n_samples),
        'weight': np.random.normal(70, 15, n_samples),
        'score_1': np.random.uniform(0, 100, n_samples),
        'score_2': np.random.beta(2, 5, n_samples) * 100,
        'outlier_variable': np.concatenate([
            np.random.normal(50, 5, n_samples - 10),
            np.array([1000, -1000, 999, -999, 998, -998, 997, -997, 996, -996])  # Outliers
        ])
    }

    # Add some missing values
    data['height'][np.random.choice(n_samples, 50, replace=False)] = np.nan
    data['weight'][np.random.choice(n_samples, 30, replace=False)] = np.nan

    # Convert all columns to numeric, coercing errors to NaN
    for column in data:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    return pd.DataFrame(data)


@pytest.fixture
def sample_categorical_data():
    """
    Generate a sample categorical dataset for testing.
    
    Returns:
        pd.DataFrame: DataFrame with categorical variables and edge cases.
    """
    np.random.seed(42)

    n_samples = 1000
    data = {
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
        'treatment_group': np.random.choice(['Control', 'Treatment_A', 'Treatment_B'], n_samples, p=[0.3, 0.35, 0.35]),
        'severity': np.random.choice(['Mild', 'Moderate', 'Severe'], n_samples, p=[0.4, 0.4, 0.2]),
        'rare_category': np.concatenate([
            np.random.choice(['Common'], n_samples - 5),
            np.array(['Rare'] * 5)  # Very rare category
        ])
    }

    # Add some missing values
    data['gender'][np.random.choice(n_samples, 20, replace=False)] = None
    data['severity'][np.random.choice(n_samples, 15, replace=False)] = None

    # Convert all columns to categorical
    for column in data:
        data[column] = pd.Categorical(data[column])

    return pd.DataFrame(data)


@pytest.fixture
def mixed_data_sample():
    """
    Generate a mixed dataset with both numerical and categorical variables.
    
    Returns:
        pd.DataFrame: Combined dataset suitable for comprehensive testing.
    """
    np.random.seed(42)

    n_samples = 1000

    # Numerical variables
    numerical_data = {
        'age': np.random.normal(45, 15, n_samples).astype(int),
        'bmi': np.random.normal(25, 5, n_samples),
        'biomarker_1': np.random.exponential(2, n_samples),
        'biomarker_2': np.random.gamma(2, 2, n_samples),
    }

    # Convert numerical columns to numeric, coercing errors to NaN
    for column in numerical_data:
        numerical_data[column] = pd.to_numeric(numerical_data[column], errors='coerce')

    # Categorical variables
    categorical_data = {
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'treatment': np.random.choice(['A', 'B', 'C'], n_samples),
        'response': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
    }

    # Convert categorical variables to categorical types
    for column in categorical_data:
        categorical_data[column] = pd.Categorical(categorical_data[column])

    # Combine all data
    data = {**numerical_data, **categorical_data}

    return pd.DataFrame(data)


@pytest.fixture
def edge_case_data():
    """
    Generate edge case datasets for robust testing.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing various edge case datasets.
    """
    datasets = {}

    # Empty dataset
    datasets['empty'] = pd.DataFrame()

    # Single row dataset
    datasets['single_row'] = pd.DataFrame({
        'value': [42],
        'category': ['A']
    }).astype({'value': 'Int64', 'category': 'category'})

    # Single column dataset
    datasets['single_column'] = pd.DataFrame({
        'value': [1, 2, 3, 4, 5]
    }).astype({'value': 'Int64'})

    # All NaN dataset
    datasets['all_nan'] = pd.DataFrame({
        'col1': [np.nan] * 10,
        'col2': [np.nan] * 10,
        'valid_col': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }).astype({'col1': 'Float64', 'col2': pd.CategoricalDtype(), 'valid_col': 'Int64'})

    # Dataset with extreme values
    datasets['extreme_values'] = pd.DataFrame({
        'tiny': [1e-10, 1e-9, 1e-8],
        'huge': [1e10, 1e11, 1e12],
        'zero': [0, 0, 0],
        'negative': [-1e6, -1e7, -1e8]
    }).astype({'tiny': 'Float64', 'huge': 'Float64', 'zero': 'Int64', 'negative': 'Float64'})

    # Dataset with duplicate values
    datasets['duplicates'] = pd.DataFrame({
        'repeated': [5] * 100
    }).astype({'repeated': 'Int64'})

    return datasets


@pytest.fixture
def mock_algorithm_client():
    """
    Create a mock AlgorithmClient for testing functions that require it.
    
    Returns:
        MagicMock: Mock client with necessary attributes and methods.
    """
    mock_client = MagicMock()

    # Mock organization methods
    mock_client.organization.list.return_value = [
        {'id': 1, 'name': 'Org1'},
        {'id': 2, 'name': 'Org2'},
        {'id': 3, 'name': 'Org3'},
        {'id': 4, 'name': 'Org4'}
    ]

    # Mock task methods
    mock_client.task.create.return_value = {'id': 123}
    mock_client.wait_for_results.return_value = [
        {'organization_id': 1, 'data': 'mock_result_1'},
        {'organization_id': 2, 'data': 'mock_result_2'}
    ]

    return mock_client


@pytest.fixture
def variables_config():
    """
    Provide sample variable configuration for testing.
    
    Returns:
        Dict[str, Dict]: Variable configuration dictionaries.
    """
    return {
        'numerical': {
            'age': {'datatype': 'int', 'inliers': [0, 120]},
            'height': {'datatype': 'float', 'inliers': [100, 250]},
            'weight': {'datatype': 'float', 'inliers': [30, 200]},
            'bmi': {'datatype': 'float', 'inliers': [10, 50]}
        },
        'categorical': {
            'gender': {'datatype': 'str', 'inliers': ['Male', 'Female', 'Other']},
            'treatment': {'datatype': 'str', 'inliers': ['A', 'B', 'C']},
            'response': {'datatype': 'str', 'inliers': ['Yes', 'No']}
        }
    }


@pytest.fixture
def stratification_config():
    """
    Provide sample stratification configuration for testing.
    
    Returns:
        Dict[str, Any]: Stratification configuration.
    """
    return {
        'gender': ['Male', 'Female'],
        'age_group': {'start': 18, 'end': 65},
        'treatment': ['A', 'B', 'C']
    }


@pytest.fixture
def quantile_test_data():
    """
    Generate specific test data for quantile computations.
    
    Returns:
        Dict[str, pd.DataFrame]: Various datasets for testing quantile functions.
    """
    np.random.seed(42)

    datasets = {}

    # Normal distribution
    datasets['normal'] = pd.DataFrame({
        'value': np.random.normal(50, 10, 1000),
        'organization_id': np.random.choice([1, 2, 3], 1000)
    })

    # Skewed distribution
    datasets['skewed'] = pd.DataFrame({
        'value': np.random.exponential(2, 1000),
        'organization_id': np.random.choice([1, 2, 3], 1000)
    })

    # Uniform distribution
    datasets['uniform'] = pd.DataFrame({
        'value': np.random.uniform(0, 100, 1000),
        'organization_id': np.random.choice([1, 2, 3], 1000)
    })

    # Known quantiles (for exact testing)
    datasets['known_quantiles'] = pd.DataFrame({
        'value': list(range(1, 101)),  # 1 to 100
        'organization_id': [1] * 100
    })

    return datasets


@pytest.fixture(scope="session")
def test_performance_data():
    """
    Generate large datasets for performance testing.
    
    Returns:
        Dict[str, pd.DataFrame]: Large datasets for performance benchmarks.
    """
    np.random.seed(42)

    sizes = {'small': 1000, 'medium': 10000, 'large': 100000}
    datasets = {}

    for size_name, n_samples in sizes.items():
        datasets[size_name] = pd.DataFrame({
            'numerical_1': np.random.normal(50, 10, n_samples),
            'numerical_2': np.random.exponential(2, n_samples),
            'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
            'categorical_2': np.random.choice(['X', 'Y'], n_samples),
            'organization_id': np.random.choice([1, 2, 3, 4, 5], n_samples)
        })

    return datasets


def assert_dataframes_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = True, rtol: float = 1e-5):
    """
    Custom assertion for comparing DataFrames with numerical tolerance.
    
    Args:
        df1, df2: DataFrames to compare
        check_dtype: Whether to check data types
        rtol: Relative tolerance for numerical comparisons
    """
    assert df1.shape == df2.shape, f"Shape mismatch: {df1.shape} vs {df2.shape}"
    assert list(df1.columns) == list(df2.columns), f"Column mismatch: {list(df1.columns)} vs {list(df2.columns)}"

    for col in df1.columns:
        if pd.api.types.is_numeric_dtype(df1[col]):
            np.testing.assert_allclose(df1[col].dropna(), df2[col].dropna(), rtol=rtol)
        else:
            pd.testing.assert_series_equal(df1[col], df2[col], check_dtype=check_dtype)


def create_federated_results(local_results: List[Dict], organization_ids: List[int]) -> List[Dict]:
    """
    Create federated-style results structure for testing aggregate functions.
    
    Args:
        local_results: List of local computation results
        organization_ids: List of organization IDs
        
    Returns:
        List of results in federated format
    """
    federated_results = []

    for i, (result, org_id) in enumerate(zip(local_results, organization_ids)):
        federated_results.append({
            'organization_id': org_id,
            'result': result,
            'status': 'success'
        })

    return federated_results
