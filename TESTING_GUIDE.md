# Contributor Testing Guide

This guide helps contributors write effective tests for the v6-tools-general repository, ensuring high-quality, robust federated analytics functions.

## Overview

Our testing framework validates that:
- Individual functions work correctly (unit tests)
- Federated implementations match centralized equivalents (integration tests)  
- Functions handle edge cases gracefully (robustness tests)
- Privacy measures work as expected (privacy tests)
- Performance scales appropriately (performance tests)

## Test Structure and Organization

### Directory Structure
```
tests/
├── conftest.py                 # Shared fixtures and utilities
├── test_data/                  # Synthetic test datasets
├── unit/                       # Unit tests for individual functions
├── integration/               # Tests for federated workflows
└── utils/                     # Test helper functions
```

### Naming Conventions
- Test files: `test_<module_name>.py`
- Test classes: `Test<FunctionName>` or `Test<FeatureName>`
- Test methods: `test_<specific_behavior>`

Example:
```python
class TestComputeLocalGeneralStatistics:
    def test_basic_numerical_statistics(self):
        """Test basic numerical statistics computation."""
        
    def test_handles_missing_values(self):
        """Test behavior with missing values."""
        
    def test_empty_dataframe_input(self):
        """Test edge case with empty DataFrame."""
```

## Writing Effective Unit Tests

### 1. Test Individual Functions in Isolation

```python
def test_mask_unnecessary_variables():
    """Test variable masking functionality."""
    # Arrange
    test_data = pd.DataFrame({
        'keep_me': [1, 2, 3],
        'remove_me': [4, 5, 6],
        'also_keep': ['a', 'b', 'c']
    })
    variables_to_keep = ['keep_me', 'also_keep']
    
    # Act
    result = mask_unnecessary_variables(test_data, variables_to_keep)
    
    # Assert
    assert list(result.columns) == variables_to_keep
    pd.testing.assert_series_equal(result['keep_me'], test_data['keep_me'])
```

### 2. Use Fixtures for Common Test Data

```python
@pytest.fixture
def sample_medical_data():
    """Generate realistic medical test data."""
    return pd.DataFrame({
        'patient_id': range(100),
        'age': np.random.normal(55, 15, 100),
        'treatment': np.random.choice(['A', 'B', 'C'], 100),
        'organization_id': np.random.choice([1, 2, 3], 100)
    })

def test_with_fixture(sample_medical_data):
    result = compute_statistics(sample_medical_data)
    assert isinstance(result, dict)
```

### 3. Test Edge Cases

Always test these scenarios:
- Empty DataFrames
- Single row/column
- All missing values
- Extreme values (very large/small numbers)
- Invalid inputs

```python
def test_edge_cases():
    # Empty DataFrame
    empty_df = pd.DataFrame()
    result = function_under_test(empty_df)
    assert isinstance(result, expected_type)
    
    # Single row
    single_row = pd.DataFrame({'col': [1]})
    result = function_under_test(single_row)
    assert len(result) >= 0
    
    # All NaN
    nan_df = pd.DataFrame({'col': [np.nan, np.nan]})
    result = function_under_test(nan_df)
    # Assert appropriate handling
```

## Writing Integration Tests

### 1. Federated vs Centralized Comparison

The core of our integration testing:

```python
def test_federated_equals_centralized():
    """Test that federated computation matches centralized."""
    # Create test data
    test_data = generate_test_data(n_samples=1000)
    
    # Split by organization for federated computation
    org_data = split_by_organization(test_data)
    
    # Compute federated results
    local_results = []
    for org_id, data in org_data.items():
        local_result = compute_local_statistics(data)
        local_results.append(local_result)
    
    federated_result = aggregate_statistics(local_results)
    
    # Compute centralized result
    centralized_result = compute_centralized_statistics(test_data)
    
    # Validate equivalence
    assert_federated_equals_centralized(
        federated_result, 
        centralized_result,
        tolerance=1e-6
    )
```

### 2. End-to-End Workflow Testing

```python
def test_complete_privacy_workflow():
    """Test complete data processing pipeline."""
    # Start with raw data
    raw_data = load_test_data()
    
    # Apply data preprocessing
    typed_data = set_datatypes(raw_data, type_config)
    stratified_data = apply_stratification(typed_data, strat_config)
    
    # Apply privacy measures
    thresholded_data = apply_sample_threshold(stratified_data)
    masked_data = mask_variables(thresholded_data, keep_vars)
    private_data = apply_differential_privacy(masked_data, epsilon=1.0)
    
    # Compute statistics
    result = compute_statistics(private_data)
    
    # Validate pipeline completed successfully
    assert isinstance(result, dict)
    assert len(private_data) <= len(raw_data)  # Data was filtered
```

## Testing Privacy Measures

### 1. Differential Privacy Testing

```python
def test_differential_privacy_adds_noise():
    """Test that DP actually adds noise to data."""
    original_data = pd.DataFrame({'values': [50] * 100})  # Identical values
    
    # Apply DP
    private_data = apply_differential_privacy(
        original_data, 
        config={'values': {'datatype': 'float'}},
        epsilon=1.0
    )
    
    # Should have variance due to added noise
    assert private_data['values'].var() > 0
    
    # Mean should be approximately preserved
    assert abs(private_data['values'].mean() - 50) < 5  # Allow for noise
```

### 2. Sample Size Thresholding

```python
def test_sample_size_threshold_enforcement():
    """Test that small samples are rejected."""
    small_data = pd.DataFrame({'values': [1, 2, 3]})  # Only 3 samples
    
    with patch('module.get_env_var') as mock_env:
        mock_env.return_value = "10"  # Threshold of 10
        
        # Should raise exception or return empty
        with pytest.raises(PrivacyThresholdViolation):
            apply_sample_size_threshold(small_data)
```

## Performance Testing

### 1. Scalability Testing

```python
def test_function_scales_with_data_size():
    """Test that function performance scales reasonably."""
    import time
    
    sizes = [100, 1000, 10000]
    times = []
    
    for size in sizes:
        data = generate_test_data(size)
        
        start = time.perf_counter()
        result = function_under_test(data)
        end = time.perf_counter()
        
        times.append(end - start)
    
    # Performance should not degrade exponentially
    # (This is a basic check - adjust based on expected complexity)
    assert times[-1] < times[0] * (sizes[-1] / sizes[0]) * 2
```

### 2. Memory Efficiency

```python
def test_memory_usage_reasonable():
    """Test that function doesn't use excessive memory."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Process large dataset
    large_data = generate_test_data(100000)
    result = function_under_test(large_data)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable
    input_size = large_data.memory_usage(deep=True).sum()
    assert memory_increase < input_size * 3  # Allow some overhead
```

## Mocking External Dependencies

### 1. Mock AlgorithmClient

```python
@pytest.fixture
def mock_client():
    """Create mock AlgorithmClient for testing."""
    client = MagicMock()
    client.organization.list.return_value = [
        {'id': 1, 'name': 'Org1'},
        {'id': 2, 'name': 'Org2'}
    ]
    return client

def test_function_with_client(mock_client):
    result = function_that_uses_client(mock_client)
    assert isinstance(result, expected_type)
    mock_client.organization.list.assert_called_once()
```

### 2. Mock Environment Variables

```python
def test_function_with_env_var():
    """Test function that depends on environment variables."""
    with patch('module.get_env_var') as mock_env:
        mock_env.return_value = "test_value"
        
        result = function_that_uses_env_var()
        
        mock_env.assert_called_with("EXPECTED_VAR_NAME")
        assert result is not None
```

## Test Data Guidelines

### 1. Use Realistic Synthetic Data

```python
def generate_patient_data(n_patients=1000):
    """Generate realistic patient data."""
    return pd.DataFrame({
        'age': np.random.normal(55, 15, n_patients).clip(18, 90),
        'bmi': np.random.normal(25, 5, n_patients).clip(15, 50),
        'diagnosis': np.random.choice(['A', 'B', 'C'], n_patients, p=[0.5, 0.3, 0.2]),
        'organization_id': np.random.choice([1, 2, 3, 4], n_patients)
    })
```

### 2. Include Edge Cases in Data

```python
def generate_edge_case_data():
    """Generate data with challenging characteristics."""
    return {
        'empty': pd.DataFrame(),
        'single_row': pd.DataFrame({'col': [1]}),
        'all_missing': pd.DataFrame({'col': [np.nan] * 10}),
        'extreme_values': pd.DataFrame({'col': [1e-10, 1e10, -1e10]}),
        'identical_values': pd.DataFrame({'col': [42] * 100})
    }
```

## Common Patterns and Utilities

### 1. Assertion Helpers

```python
def assert_dataframes_equal(df1, df2, tolerance=1e-6):
    """Assert DataFrames are equal within tolerance."""
    assert df1.shape == df2.shape
    
    for col in df1.columns:
        if pd.api.types.is_numeric_dtype(df1[col]):
            np.testing.assert_allclose(
                df1[col].dropna(), 
                df2[col].dropna(), 
                rtol=tolerance
            )
        else:
            pd.testing.assert_series_equal(df1[col], df2[col])
```

### 2. Statistical Validation

```python
def assert_statistical_properties(data, expected_mean=None, expected_std=None):
    """Validate statistical properties of data."""
    if expected_mean is not None:
        assert abs(data.mean() - expected_mean) < 0.1
    
    if expected_std is not None:
        assert abs(data.std() - expected_std) < 0.1
```

## Best Practices

1. **Test Behavior, Not Implementation**: Focus on what the function should do, not how it does it
2. **Use Descriptive Names**: Test names should explain the scenario being tested
3. **One Assertion Per Concept**: Keep tests focused and easy to debug
4. **Test Edge Cases**: Empty inputs, boundary values, error conditions
5. **Use Fixtures**: Share common test data and setup code
6. **Mock External Dependencies**: Keep tests fast and isolated
7. **Validate Both Structure and Values**: Check return types and actual results
8. **Test Error Conditions**: Ensure functions fail gracefully
9. **Document Complex Tests**: Explain non-obvious test logic
10. **Keep Tests Fast**: Unit tests should run in milliseconds

## Running and Debugging Tests

### Running Specific Tests
```bash
# Run specific test file
pytest tests/unit/test_general_statistics.py

# Run specific test class
pytest tests/unit/test_general_statistics.py::TestComputeLocalStatistics

# Run specific test method
pytest tests/unit/test_general_statistics.py::TestComputeLocalStatistics::test_basic_computation

# Run tests matching pattern
pytest -k "test_privacy"

# Run with verbose output
pytest -v

# Run with debugging output
pytest -s --tb=long
```

### Debugging Failed Tests
```bash
# Drop into debugger on failure
pytest --pdb

# Show local variables in traceback
pytest --tb=long

# Run last failed tests only
pytest --lf

# Run tests with coverage and missing lines
pytest --cov=vantage6_strongaya_general --cov-report=term-missing
```

## Coverage Requirements

- **Minimum Coverage**: 90% line coverage
- **Focus Areas**: All public functions must be tested
- **Edge Cases**: Critical paths and error conditions
- **Integration**: Federated vs centralized equivalence

Check coverage:
```bash
pytest --cov=vantage6_strongaya_general --cov-report=html
open htmlcov/index.html  # View detailed coverage report
```

This guide ensures consistent, high-quality testing practices across the project. When in doubt, look at existing tests for examples and patterns.