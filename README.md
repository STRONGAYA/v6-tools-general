# STRONG AYA's General Vantage6 tools

# Purpose of this repository

This repository contains general functionalities and tools for the STRONG AYA project.
They are designed to be used with the Vantage6 framework for federated analytics and learning
and are intended to facilitate and simplify the development of Vantage6 algorithms.

The code in this repository is available as a Python library here on GitHub or through direct reference with `pip`.

# Structure of the repository

The various functions are organised in different sections, consisting of:

- **General Statistics**: Functions for calculating general statistics such as mean, median, and standard deviation;
- **Privacy Measures**: Functions to enhance privacy protection such as sample size thresholding and differential
  privacy;
- **Miscellaneous**: Functions that do not necessarily fit into the other categories, such as data type setting, and
  data stratification.

# Usage

The library provides functions that can be included in a Vantage6 algorithm as the algorithm developer sees fit.
The functions are designed to be modular and can be used independently or in combination with other functions.

The library can be included in your Vantage6 algorithm by listing it in the `requirements.txt` and `setup.py` file of your
algorithm.

## Including the library in your Vantage6 algorithm

For the `requirements.txt` file, you can add the following line to the file:

```
git+https://github.com/STRONGAYA/v6-tools-general.git@v0.1.2
```

For the `setup.py` file, you can add the following line to the `install_requires` list:

```python
from os import path
from codecs import open
from setuptools import setup, find_packages

# We are using a README.md, if you do not have this in your folder, simply replace this with a string.
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
setup(
    name='v6-not-an-actual-algorithm',
    version="1.0.0",
    description='Fictive Vantage6 algorithm that performs general statistics computation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/STRONGAYA/v6-not-an-actual-algorithm',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'vantage6-algorithm-tools',
        'numpy',
        'pandas',
        "vantage6-strongaya-general @ git+https://github.com/STRONGAYA/v6-tools-general.git@v0.1.2",
    # other dependencies
    ]
)
```

## Central (aggregating) example

Example usage of various functions in a central (aggregating) section of a Vantage6 algorithm:

```python
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient

# General federated algorithm functions
from vantage6_strongaya_general.miscellaneous import collect_organisation_ids, safe_log
from vantage6_strongaya_general.general_statistics import compute_aggregate_general_statistics


@algorithm_client
def central(client: AlgorithmClient, variables_to_analyse: dict, variables_to_stratify: dict = None,
            organisation_ids: list[int] = None) -> dict | None:
    """
    Central function to compute general statistics of specified variables from multiple organisations.

    Args:
        client (AlgorithmClient): The client to communicate with the vantage6 server.
        variables_to_analyse (dict): Dictionary of variables to analyse.
        variables_to_stratify (dict, optional): Dictionary of variables to stratify. Defaults to None.
        organisation_ids (list[int], optional): List of organisation IDs to include. Defaults to None.

    Returns:
        dict|None: A dictionary containing the aggregated EORTC QLQ scoring results,
        or None if the input structure is incorrect.
    """

    # Collect all organisations that participate in this collaboration unless specified
    organisation_ids = collect_organisation_ids(organisation_ids, client)

    # Create the subtask for general statistics
    safe_log("info", "Creating subtask to calculate general statistics.")

    input_ = {"method": "partial",
              "kwargs": {
                  "variables_to_analyse": variables_to_analyse,
                  "variables_to_stratify": variables_to_stratify}
              }

    task_general_statistics = client.task.create(input_, organisation_ids,
                                                 "General Statistics",
                                                 "This subtask determines the general statistics of specified variables.")

    # Wait for the node(s) to return the results of the subtask
    safe_log("info", f"Waiting for results of task {task_general_statistics.get('id')}")
    results_general_statistics = client.wait_for_results(task_general_statistics.get("id"))
    safe_log("info", f"Results of task {task_general_statistics.get('id')} obtained")

    # Aggregate the general statistics
    results_general_statistics = compute_aggregate_general_statistics(results_general_statistics)

    # Return the final results of the algorithm
    return results_general_statistics
```

## Node or local (participating) example

Example usage of various functions in a node (participating) section of a Vantage6 algorithm:

```python
import pandas as pd

from vantage6.algorithm.tools.util import info
from vantage6.algorithm.tools.decorators import data, algorithm_client
from vantage6.algorithm.client import AlgorithmClient

# General federated algorithm functions
from vantage6_strongaya_general.general_statistics import compute_local_general_statistics
from vantage6_strongaya_general.miscellaneous import apply_data_stratification, set_datatypes, safe_log
from vantage6_strongaya_general.privacy_measures import apply_sample_size_threshold, mask_unnecessary_variables, apply_differential_privacy


@data(1)
@algorithm_client
def partial_general_statistics(client: AlgorithmClient, df: pd.DataFrame, variables_to_analyse: dict,
                               variables_to_stratify: dict = None) -> dict:
    """
    Execute the partial algorithm for general statistics computation.

    Args:
        client (AlgorithmClient): The client to communicate with the vantage6 server.
        df (pd.DataFrame): The DataFrame containing the data to be processed.
        variables_to_analyse (dict): Dictionary of variables to analyse.
        variables_to_stratify (dict, optional): Dictionary of variables to stratify. Defaults to None.

    Returns:
        dict: A dictionary containing the computed general statistics.
    """
    safe_log("info", "Executing partial algorithm computation of general statistics.")

    # Set datatypes for each variable
    df = set_datatypes(df, variables_to_analyse)

    # Apply stratification if necessary
    df = apply_data_stratification(df, variables_to_stratify)

    # Ensure that the sample size threshold is met
    df = apply_sample_size_threshold(client, df, variables_to_analyse)

    # Mask unnecessary variables by removal
    df = mask_unnecessary_variables(df, variables_to_analyse)

    # Apply differential privacy (Laplace mechanism as per default)
    df = apply_differential_privacy(df, variables_to_analyse, epsilon=1.0, return_type='dataframe')

    # Compute general statistics
    result = compute_local_general_statistics(df, variables_to_stratify)

    return result
```

The various functions are available through `pip install` for debugging and testing purposes.
The library can be installed as follows:

```bash
pip install git+https://github.com/STRONGAYA/v6-tools-general.git
```

# Contributers

- J. Hogenboom
- V. Gouthamchand
- A. Lobo Gomes

# References

- [STRONG AYA](https://strongaya.eu/)
- [Vantage6](vantage6.ai)