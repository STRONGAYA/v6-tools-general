"""
Integration tests for miscellaneous module functions.

This module contains integration tests that verify the interaction between
multiple functions in the miscellaneous module and their behaviour in
federated learning workflows.
"""

import pytest

from vantage6_strongaya_general.miscellaneous import (
    collect_organisation_ids,
    apply_data_stratification,
    check_variable_availability,
    check_partial_result_presence,
)
from vantage6.algorithm.tools.exceptions import UserInputError


class TestVariableAndResultChecksIntegration:
    """Integration tests for variable availability and result checking."""

    def test_full_validation_pipeline(self, mixed_data_sample, mock_algorithm_client):
        """Test complete validation workflow."""
        # Check variables exist
        variables = ["age", "gender"]
        check_variable_availability(mixed_data_sample, variables)

        # Simulate organisation collection and result checking
        org_ids = collect_organisation_ids(None, mock_algorithm_client)
        mock_results = [f"result_{i}" for i in org_ids]
        result = check_partial_result_presence(mock_results, org_ids)

        assert result is True

    def test_variable_check_blocks_execution(self, mixed_data_sample):
        """Test that missing variables prevent further processing."""
        with pytest.raises(UserInputError):
            check_variable_availability(mixed_data_sample, ["nonexistent"])

    def test_integration_with_stratification(self, mixed_data_sample):
        """Test integration with data stratification."""
        # Apply stratification
        stratified = apply_data_stratification(
            mixed_data_sample, {"age": {"start": 30, "end": 60}}
        )

        # Variables should still be available after stratification
        check_variable_availability(stratified, ["age", "gender"])

    def test_federated_workflow_simulation(
        self, mixed_data_sample, mock_algorithm_client
    ):
        """Test simulated federated learning workflow."""
        # Step 1: Validate input variables
        required_vars = ["age", "gender", "treatment"]
        check_variable_availability(mixed_data_sample, required_vars)

        # Step 2: Apply stratification
        stratified_data = apply_data_stratification(
            mixed_data_sample, {"age": {"start": 25, "end": 65}}
        )

        # Step 3: Re-validate variables after stratification
        check_variable_availability(stratified_data, required_vars)

        # Step 4: Collect organisations
        org_ids = collect_organisation_ids(None, mock_algorithm_client)

        # Step 5: Simulate partial results collection
        mock_results = [{"org_id": org_id, "stats": {}} for org_id in org_ids]

        # Step 6: Validate results
        result = check_partial_result_presence(mock_results, org_ids)

        assert result is True
        assert len(stratified_data) <= len(mixed_data_sample)

    def test_error_propagation_in_pipeline(self, mixed_data_sample):
        """Test that errors propagate correctly through the pipeline."""
        # Variable check should fail first
        with pytest.raises(UserInputError):
            check_variable_availability(mixed_data_sample, ["invalid_var"])

        # This code should not execute in a real pipeline
        # but we can test the concept

    def test_stratification_affects_subsequent_validation(self, mixed_data_sample):
        """Test that heavy stratification can affect downstream processes."""
        # Apply very restrictive stratification
        restrictive_stratification = {
            "age": {"start": 40, "end": 45},
            "gender": ["Male"],
        }

        stratified = apply_data_stratification(
            mixed_data_sample, restrictive_stratification
        )

        # Should still be able to validate variables
        check_variable_availability(stratified, ["age", "gender"])

        # But dataset might be very small
        assert len(stratified) <= len(mixed_data_sample)
