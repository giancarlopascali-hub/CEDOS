import unittest

import numpy as np
import pandas as pd
import torch

from src.core.sa import _cross_validated_r2, _deterministic_sample_indices
from src.core.transfer_bo import (
    _reuse_training_data,
    _subsample_training_data,
    preview_transfer_selection,
    resolve_gp_row_capacity,
    resolve_gp_row_limit,
    transfer_exploration_to_beta,
)


class TransferWorkloadTests(unittest.TestCase):
    def test_transfer_exploration_endpoints_have_the_intended_direction(self):
        self.assertEqual(transfer_exploration_to_beta(0), 0.0)
        self.assertEqual(transfer_exploration_to_beta(0.5), 2.0)
        self.assertEqual(transfer_exploration_to_beta(1), 4.0)

    def test_requested_prior_limit_is_bounded_only_by_dataset_rows(self):
        self.assertEqual(resolve_gp_row_limit(200, 1000), 200)
        self.assertEqual(resolve_gp_row_limit(1500, 2000), 1500)
        self.assertEqual(
            resolve_gp_row_limit(3000, 2000, objective_count=3, constraint_count=2),
            2000,
        )

    def test_weighted_subsample_is_reproducible(self):
        rows = 100
        x = torch.arange(rows, dtype=torch.float32).unsqueeze(-1)
        y = x.clone()
        yvar = torch.ones_like(x)
        weights = np.linspace(0.0, 1.0, rows)
        first = _subsample_training_data(x, y, yvar, weights, max_rows=30)[4]
        second = _subsample_training_data(x, y, yvar, weights, max_rows=30)[4]
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 30)
        self.assertGreater(np.mean(weights[first]), np.mean(weights))

    def test_relevance_threshold_filters_before_row_limit(self):
        rows = 10
        x = torch.arange(rows, dtype=torch.float32).unsqueeze(-1)
        weights = np.linspace(0.0, 0.9, rows)
        selected = _subsample_training_data(
            x, x.clone(), torch.ones_like(x), weights,
            max_rows=8, min_relevance=0.5,
        )[4]
        np.testing.assert_array_equal(selected, np.arange(5, 10))

    def test_equal_relevance_ties_still_select_exact_requested_count(self):
        rows = 25
        x = torch.arange(rows, dtype=torch.float32).unsqueeze(-1)
        selected = _subsample_training_data(
            x, x.clone(), torch.ones_like(x), np.ones(rows), max_rows=7,
        )[4]
        self.assertEqual(len(selected), 7)
        self.assertEqual(len(np.unique(selected)), 7)

    def test_exact_training_rows_can_be_reused_by_source_index(self):
        train_x = torch.arange(10, dtype=torch.float32).unsqueeze(-1)
        train_y = train_x.clone()
        train_yvar = torch.ones_like(train_x)
        reused = _reuse_training_data(
            train_x,
            train_y,
            train_yvar,
            np.arange(100, 110),
            [102, 105, 108],
        )
        self.assertIsNotNone(reused)
        reused_x, _, _, positions = reused
        np.testing.assert_array_equal(positions, np.array([2, 5, 8]))
        np.testing.assert_array_equal(reused_x.squeeze(-1).numpy(), np.array([2, 5, 8]))

    def test_reuse_rejects_a_changed_source_dataset(self):
        train_x = torch.arange(5, dtype=torch.float32).unsqueeze(-1)
        self.assertIsNone(_reuse_training_data(
            train_x, train_x.clone(), torch.ones_like(train_x),
            np.arange(5), [0, 2, 99],
        ))

    def test_selection_preview_reports_capacity_eligibility_and_actual_fit(self):
        df = pd.DataFrame({
            'X': np.arange(10, dtype=float),
            'Prior': ['A'] * 6 + ['B'] * 4,
            'Y': np.arange(10, dtype=float),
        })
        preview = preview_transfer_selection(
            df,
            features_config=[{'name': 'X', 'type': 'continuous', 'range': '[0, 9]'}],
            objectives_config=[{'name': 'Y', 'type': 'maximize'}],
            priors_config=[{'name': 'Prior', 'col_type': 'categorical', 'target_value': 'A'}],
            requested_rows=4,
        )
        self.assertEqual(preview['valid_rows'], 10)
        self.assertEqual(preview['eligible_rows'], 6)
        self.assertEqual(preview['rows_to_fit'], 4)
        self.assertEqual(preview['permitted_max_rows'], 10)

    def test_dataset_size_is_the_authoritative_slider_ceiling(self):
        capacity = resolve_gp_row_capacity(
            5000, objective_count=1, constraint_count=1, has_categorical=True
        )
        self.assertEqual(capacity, 5000)
        self.assertEqual(resolve_gp_row_limit(5000, 5000, 1, 1, True), capacity)


class StatisticalAnalysisReliabilityTests(unittest.TestCase):
    def test_deterministic_stratified_sample_is_stable(self):
        target = np.linspace(-3, 4, 1000)
        first = _deterministic_sample_indices(target, 75)
        second = _deterministic_sample_indices(target, 75)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 75)
        self.assertEqual(first[0], 0)
        self.assertEqual(first[-1], 999)

    def test_held_out_r2_is_reproducible(self):
        from sklearn.linear_model import LinearRegression

        x = np.linspace(-2, 2, 120).reshape(-1, 1)
        y = 3.0 * x[:, 0] - 1.0
        first = _cross_validated_r2(LinearRegression, x, y)
        second = _cross_validated_r2(LinearRegression, x, y)
        self.assertEqual(first, second)
        self.assertGreater(first[0], 0.999)


if __name__ == '__main__':
    unittest.main()
