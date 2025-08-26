"""
vantage6-strongaya-general - A library with general utility functions for Vantage6 algorithms
"""

from .miscellaneous import (
    safe_log,
    safe_calculate,
    collect_organisation_ids,
    apply_data_stratification,
    set_datatypes,
)

from .privacy_measures import (
    mask_unnecessary_variables,
    apply_sample_size_threshold,
    apply_differential_privacy,
)

from .general_statistics import (
    compute_aggregate_general_statistics,
    compute_aggregate_adjusted_deviation,
    compute_local_general_statistics,
    compute_local_adjusted_deviation,
)

__all__ = [
    "safe_log",
    "safe_calculate",
    "collect_organisation_ids",
    "apply_data_stratification",
    "set_datatypes",
    "mask_unnecessary_variables",
    "apply_sample_size_threshold",
    "apply_differential_privacy",
    "compute_aggregate_general_statistics",
    "compute_aggregate_adjusted_deviation",
    "compute_local_general_statistics",
    "compute_local_adjusted_deviation",
]
__version__ = "1.0.1"
