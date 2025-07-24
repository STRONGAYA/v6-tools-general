"""
Synthetic test data generators for comprehensive testing.

This module provides utilities to generate realistic synthetic datasets
that cover various scenarios, edge cases, and data distributions for
testing federated analytics functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from faker import Faker
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

fake = Faker()
Faker.seed(42)


class SyntheticDataGenerator:
    """Generator for creating synthetic datasets for testing federated analytics."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            seed: Random seed for reproducible data generation
        """
        self.seed = seed
        np.random.seed(seed)
        Faker.seed(seed)
    
    def generate_patient_data(self, 
                            n_patients: int = 1000,
                            n_organizations: int = 4,
                            missing_rate: float = 0.05) -> pd.DataFrame:
        """
        Generate synthetic patient data for medical research scenarios.
        
        Args:
            n_patients: Number of patients to generate
            n_organizations: Number of participating organizations
            missing_rate: Rate of missing values
            
        Returns:
            pd.DataFrame: Synthetic patient dataset
        """
        data = {}
        
        # Demographics
        data['patient_id'] = range(1, n_patients + 1)
        data['age'] = np.random.normal(55, 15, n_patients).astype(int).clip(18, 90)
        data['gender'] = np.random.choice(['Male', 'Female'], n_patients, p=[0.48, 0.52])
        data['organization_id'] = np.random.choice(range(1, n_organizations + 1), n_patients)
        
        # Clinical measurements
        data['height_cm'] = np.random.normal(170, 10, n_patients).clip(140, 210)
        data['weight_kg'] = np.random.normal(75, 15, n_patients).clip(40, 150)
        data['bmi'] = data['weight_kg'] / (data['height_cm'] / 100) ** 2
        
        # Disease severity (ordinal)
        data['disease_stage'] = np.random.choice(['I', 'II', 'III', 'IV'], n_patients, p=[0.3, 0.35, 0.25, 0.1])
        
        # Treatment information
        data['treatment_type'] = np.random.choice(['Surgery', 'Chemotherapy', 'Radiation', 'Combined'], n_patients)
        data['response'] = np.random.choice(['Complete', 'Partial', 'None'], n_patients, p=[0.3, 0.5, 0.2])
        
        # Biomarkers (with realistic correlations)
        base_biomarker = np.random.normal(50, 15, n_patients)
        data['biomarker_a'] = base_biomarker + np.random.normal(0, 5, n_patients)
        data['biomarker_b'] = base_biomarker * 0.7 + np.random.normal(20, 8, n_patients)
        data['biomarker_c'] = np.random.exponential(2, n_patients)
        
        # Survival/follow-up time (in months)
        data['followup_months'] = np.random.exponential(24, n_patients).clip(1, 120)
        data['event_occurred'] = np.random.choice([0, 1], n_patients, p=[0.6, 0.4])
        
        df = pd.DataFrame(data)
        
        # Introduce missing values
        if missing_rate > 0:
            for col in ['height_cm', 'weight_kg', 'biomarker_a', 'biomarker_b', 'response']:
                n_missing = int(len(df) * missing_rate)
                missing_idx = np.random.choice(len(df), n_missing, replace=False)
                df.loc[missing_idx, col] = np.nan
        
        return df
    
    def generate_quality_of_life_data(self, 
                                    n_patients: int = 800,
                                    n_organizations: int = 3) -> pd.DataFrame:
        """
        Generate synthetic quality of life questionnaire data.
        
        Args:
            n_patients: Number of patients
            n_organizations: Number of organizations
            
        Returns:
            pd.DataFrame: Quality of life dataset
        """
        data = {}
        
        # Patient identifiers
        data['patient_id'] = range(1, n_patients + 1)
        data['organization_id'] = np.random.choice(range(1, n_organizations + 1), n_patients)
        data['assessment_date'] = [fake.date_between(start_date='-2y', end_date='today') for _ in range(n_patients)]
        
        # EORTC QLQ-C30 style questions (scaled 1-4)
        scales = {
            'physical_function': (2.5, 0.8),  # (mean, std)
            'role_function': (2.8, 0.9),
            'emotional_function': (2.4, 1.0),
            'cognitive_function': (3.0, 0.7),
            'social_function': (2.7, 0.9),
            'global_health': (2.6, 0.8)
        }
        
        for scale_name, (mean, std) in scales.items():
            # Generate correlated responses within reasonable bounds
            raw_scores = np.random.normal(mean, std, n_patients)
            data[scale_name] = np.clip(np.round(raw_scores), 1, 4).astype(int)
        
        # Symptom scales (higher = worse, scaled 1-4)
        symptoms = {
            'fatigue': (2.2, 0.8),
            'nausea': (1.8, 0.7),
            'pain': (2.0, 0.9),
            'dyspnea': (1.7, 0.6),
            'insomnia': (2.3, 1.0),
            'appetite_loss': (1.9, 0.8),
            'constipation': (1.6, 0.6),
            'diarrhea': (1.4, 0.5)
        }
        
        for symptom_name, (mean, std) in symptoms.items():
            raw_scores = np.random.normal(mean, std, n_patients)
            data[symptom_name] = np.clip(np.round(raw_scores), 1, 4).astype(int)
        
        return pd.DataFrame(data)
    
    def generate_federated_datasets(self, 
                                  n_organizations: int = 4,
                                  size_distribution: str = 'balanced') -> Dict[int, pd.DataFrame]:
        """
        Generate datasets split across multiple organizations.
        
        Args:
            n_organizations: Number of organizations
            size_distribution: 'balanced', 'unbalanced', or 'extreme'
            
        Returns:
            Dict mapping organization_id to DataFrame
        """
        if size_distribution == 'balanced':
            sizes = [500] * n_organizations
        elif size_distribution == 'unbalanced':
            sizes = [800, 600, 400, 200][:n_organizations]
        elif size_distribution == 'extreme':
            sizes = [1000, 100, 50, 20][:n_organizations]
        else:
            raise ValueError(f"Unknown size distribution: {size_distribution}")
        
        org_datasets = {}
        
        for org_id in range(1, n_organizations + 1):
            n_patients = sizes[org_id - 1]
            
            # Generate base dataset
            df = self.generate_patient_data(n_patients, n_organizations=1, missing_rate=0.08)
            df['organization_id'] = org_id
            
            # Add organization-specific characteristics
            if org_id == 1:  # Academic medical center
                df['age'] = df['age'] + np.random.normal(5, 2, len(df))  # Slightly older patients
                df.loc[:, 'disease_stage'] = np.random.choice(['II', 'III', 'IV'], len(df), p=[0.2, 0.5, 0.3])
            elif org_id == 2:  # Community hospital
                df['age'] = df['age'] + np.random.normal(-3, 2, len(df))  # Slightly younger
                df.loc[:, 'treatment_type'] = np.random.choice(['Surgery', 'Chemotherapy'], len(df), p=[0.6, 0.4])
            elif org_id == 3:  # Specialized cancer center
                df.loc[:, 'disease_stage'] = np.random.choice(['III', 'IV'], len(df), p=[0.6, 0.4])  # More advanced cases
                df['biomarker_a'] = df['biomarker_a'] * 1.2  # Higher biomarker levels
            
            org_datasets[org_id] = df
        
        return org_datasets
    
    def generate_edge_case_scenarios(self) -> Dict[str, pd.DataFrame]:
        """
        Generate specific edge case scenarios for robust testing.
        
        Returns:
            Dict of edge case datasets
        """
        scenarios = {}
        
        # Scenario 1: Very small sample size
        scenarios['tiny_sample'] = pd.DataFrame({
            'value': [1, 2, 3],
            'category': ['A', 'B', 'A'],
            'organization_id': [1, 1, 1]
        })
        
        # Scenario 2: Single organization with one data point
        scenarios['single_point'] = pd.DataFrame({
            'measurement': [42.5],
            'organization_id': [1]
        })
        
        # Scenario 3: All identical values
        scenarios['identical_values'] = pd.DataFrame({
            'constant': [100.0] * 50,
            'organization_id': [1] * 25 + [2] * 25
        })
        
        # Scenario 4: Extreme outliers
        normal_data = np.random.normal(50, 5, 100)
        outliers = [1e6, -1e6, 1e-6, -1e-6]
        scenarios['extreme_outliers'] = pd.DataFrame({
            'value': np.concatenate([normal_data, outliers]),
            'organization_id': [1] * 104
        })
        
        # Scenario 5: High missing rate
        data_with_missing = np.random.normal(50, 10, 100)
        data_with_missing[np.random.choice(100, 80, replace=False)] = np.nan
        scenarios['mostly_missing'] = pd.DataFrame({
            'sparse_data': data_with_missing,
            'organization_id': [1] * 100
        })
        
        # Scenario 6: Mixed data types
        scenarios['mixed_types'] = pd.DataFrame({
            'integer_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'boolean_col': [True, False, True, False, True],
            'organization_id': [1] * 5
        })
        
        # Scenario 7: Categorical with rare categories
        common_cats = ['Common'] * 95
        rare_cats = ['Rare1', 'Rare2', 'Rare3', 'Rare4', 'Rare5']
        scenarios['rare_categories'] = pd.DataFrame({
            'category': common_cats + rare_cats,
            'organization_id': [1] * 100
        })
        
        return scenarios
    
    def generate_quantile_test_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generate specific datasets for testing quantile computations.
        
        Returns:
            Dict of quantile test datasets
        """
        datasets = {}
        
        # Dataset with known quantiles
        datasets['known_quantiles'] = pd.DataFrame({
            'ordered_values': list(range(1, 101)),  # 1 to 100
            'organization_id': [1] * 50 + [2] * 50
        })
        
        # Normal distribution
        np.random.seed(42)
        datasets['normal_dist'] = pd.DataFrame({
            'values': np.random.normal(100, 15, 1000),
            'organization_id': np.repeat([1, 2, 3, 4], 250)
        })
        
        # Highly skewed distribution
        datasets['skewed_dist'] = pd.DataFrame({
            'values': np.random.exponential(2, 1000),
            'organization_id': np.repeat([1, 2], 500)
        })
        
        # Uniform distribution
        datasets['uniform_dist'] = pd.DataFrame({
            'values': np.random.uniform(0, 100, 1000),
            'organization_id': np.repeat([1, 2, 3], [300, 400, 300])
        })
        
        # Distribution with ties
        datasets['with_ties'] = pd.DataFrame({
            'values': np.random.choice([1, 2, 3, 4, 5], 500),
            'organization_id': np.repeat([1, 2], 250)
        })
        
        return datasets
    
    def generate_performance_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Generate datasets of varying sizes for performance testing.
        
        Returns:
            Dict of performance test datasets
        """
        np.random.seed(42)
        sizes = {'small': 1000, 'medium': 10000, 'large': 100000}
        datasets = {}
        
        for size_name, n_samples in sizes.items():
            # Create realistic data with multiple variables
            data = {
                'continuous_1': np.random.normal(50, 15, n_samples),
                'continuous_2': np.random.exponential(2, n_samples),
                'continuous_3': np.random.uniform(0, 100, n_samples),
                'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
                'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples, p=[0.5, 0.3, 0.2]),
                'binary': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                'organization_id': np.random.choice([1, 2, 3, 4, 5], n_samples)
            }
            
            # Add some missing values
            for col in ['continuous_1', 'continuous_2', 'categorical_1']:
                n_missing = int(n_samples * 0.02)  # 2% missing
                missing_idx = np.random.choice(n_samples, n_missing, replace=False)
                if col.startswith('continuous'):
                    data[col][missing_idx] = np.nan
                else:
                    data[col] = [data[col][i] if i not in missing_idx else None for i in range(n_samples)]
            
            datasets[size_name] = pd.DataFrame(data)
        
        return datasets


def save_test_datasets(output_dir: str = "tests/test_data/"):
    """
    Generate and save all test datasets to files.
    
    Args:
        output_dir: Directory to save test datasets
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    generator = SyntheticDataGenerator()
    
    # Generate and save basic datasets
    patient_data = generator.generate_patient_data()
    patient_data.to_csv(f"{output_dir}/patient_data.csv", index=False)
    
    qol_data = generator.generate_quality_of_life_data()
    qol_data.to_csv(f"{output_dir}/quality_of_life_data.csv", index=False)
    
    # Generate and save federated datasets
    federated_data = generator.generate_federated_datasets()
    for org_id, df in federated_data.items():
        df.to_csv(f"{output_dir}/federated_org_{org_id}.csv", index=False)
    
    # Generate and save edge cases
    edge_cases = generator.generate_edge_case_scenarios()
    for scenario_name, df in edge_cases.items():
        df.to_csv(f"{output_dir}/edge_case_{scenario_name}.csv", index=False)
    
    # Generate and save quantile test data
    quantile_data = generator.generate_quantile_test_data()
    for dataset_name, df in quantile_data.items():
        df.to_csv(f"{output_dir}/quantile_{dataset_name}.csv", index=False)
    
    print(f"Test datasets saved to {output_dir}")


if __name__ == "__main__":
    # Generate test datasets when run as script
    save_test_datasets()