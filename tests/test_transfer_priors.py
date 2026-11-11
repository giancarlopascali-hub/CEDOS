import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app import app
from src.core.bo_policy import BOOptimizationResult
from src.core.prior_utils import (
    compute_combined_prior_relevance,
    compute_transfer_relevance,
)
from src.core.transfer_bo import preview_transfer_selection, run_transfer_bo_optimization


class PriorEligibilityTests(unittest.TestCase):
    def test_numerical_tolerance_has_an_explicit_boundary(self):
        weights, eligible, resolved = compute_transfer_relevance(
            [40, 45, 50, 55, 60, None],
            target_value=50,
            col_type='numerical',
            tolerance=10,
        )

        np.testing.assert_array_equal(
            eligible, np.array([True, True, True, True, True, False])
        )
        self.assertEqual(resolved['tolerance'], 10)
        self.assertEqual(weights[0], 0.0)
        self.assertEqual(weights[2], 1.0)
        self.assertEqual(weights[4], 0.0)
        self.assertGreater(weights[1], weights[0])
        self.assertGreater(weights[3], weights[4])
        self.assertEqual(weights[5], 0.0)

    def test_zero_tolerance_keeps_only_exact_numerical_matches(self):
        weights, eligible, _ = compute_transfer_relevance(
            [4.999, 5.0, 5.001],
            target_value=5,
            col_type='numerical',
            tolerance=0,
        )
        np.testing.assert_array_equal(eligible, np.array([False, True, False]))
        np.testing.assert_array_equal(weights, np.array([0.0, 1.0, 0.0]))

    def test_numerical_default_is_half_the_observed_span(self):
        _, eligible, resolved = compute_transfer_relevance(
            [0, 25, 50, 75, 100],
            target_value=50,
            col_type='numerical',
        )
        self.assertEqual(resolved['tolerance'], 50)
        self.assertTrue(eligible.all())

    def test_smiles_minimum_similarity_is_a_hard_acceptance_rule(self):
        weights, eligible, resolved = compute_transfer_relevance(
            ['CCO', 'CCN', 'c1ccccc1', None],
            target_value='CCO',
            col_type='smiles',
            min_similarity=1.0,
        )
        np.testing.assert_array_equal(
            eligible, np.array([True, False, False, False])
        )
        self.assertEqual(resolved['min_similarity'], 1.0)
        np.testing.assert_array_equal(weights, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_amino_acid_sequences_use_the_similarity_cutoff_and_weights(self):
        weights, eligible, resolved = compute_transfer_relevance(
            ['MKTW', 'mktw', 'MKAW', 'MKT', 'YYYY', 'MK?W', None],
            target_value='MKTW',
            col_type='amino_acid_sequence',
            min_similarity=0.75,
        )
        np.testing.assert_array_equal(
            eligible, [True, True, True, True, False, False, False]
        )
        np.testing.assert_allclose(weights, [1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(resolved['min_similarity'], 0.75)

        closer_weights, closer_eligible, _ = compute_transfer_relevance(
            ['MKTW', 'MKAW', 'YYYY'],
            target_value='MKTW',
            col_type='amino_acid_sequence',
            min_similarity=0.5,
        )
        np.testing.assert_array_equal(closer_eligible, [True, True, False])
        self.assertGreater(closer_weights[0], closer_weights[1])

    def test_nucleotide_sequences_normalize_dna_rna_and_allow_indels(self):
        weights, eligible, _ = compute_transfer_relevance(
            ['ATGC', 'AUGC', 'ATG', 'ATGGGC', 'TTTT', 'AT1C', None],
            target_value='AUGC',
            col_type='nucleotide_sequence',
            min_similarity=0.75,
        )
        np.testing.assert_array_equal(
            eligible, [True, True, True, False, False, False, False]
        )
        np.testing.assert_allclose(weights, [1, 1, 0, 0, 0, 0, 0])

        weights, eligible, _ = compute_transfer_relevance(
            ['AT1C', 'ATGC', 'CCCC'],
            target_value='ATGC',
            col_type='nucleotide_sequence',
            min_similarity=0,
        )
        np.testing.assert_array_equal(eligible, [False, True, True])
        self.assertEqual(weights[0], 0)

        _, eligible, _ = compute_transfer_relevance(
            ['NAT', 'nan', None],
            target_value='NAT',
            col_type='nucleotide_sequence',
            min_similarity=1,
        )
        np.testing.assert_array_equal(eligible, [True, False, False])
        _, eligible, _ = compute_transfer_relevance(
            ['NAN'], 'nan', 'amino_acid_sequence', min_similarity=1
        )
        np.testing.assert_array_equal(eligible, [True])

    def test_sequence_target_and_cutoff_are_validated(self):
        with self.assertRaisesRegex(ValueError, 'invalid characters'):
            compute_transfer_relevance(
                ['ATGC'], 'AT1C', 'nucleotide_sequence', min_similarity=0.5
            )
        with self.assertRaisesRegex(ValueError, 'between 0 and 1'):
            compute_transfer_relevance(
                ['MKTW'], 'MKTW', 'amino_acid_sequence', min_similarity=1.1
            )

    def test_multiply_is_intersection_and_average_is_union(self):
        dataframe = pd.DataFrame({
            'Category': ['A', 'A', 'B', 'B'],
            'Temperature': [0, 5, 0, 5],
        })
        priors = [
            {
                'name': 'Category',
                'col_type': 'categorical',
                'target_value': 'A',
                'prior_weight': 50,
            },
            {
                'name': 'Temperature',
                'col_type': 'numerical',
                'target_value': 0,
                'tolerance': 1,
                'prior_weight': 50,
            },
        ]

        strict = compute_combined_prior_relevance(dataframe, priors, 'multiply')
        flexible = compute_combined_prior_relevance(dataframe, priors, 'average')

        np.testing.assert_array_equal(
            strict['eligible_mask'], np.array([True, False, False, False])
        )
        np.testing.assert_array_equal(
            flexible['eligible_mask'], np.array([True, True, True, False])
        )

    def test_zero_weight_average_prior_does_not_make_rows_eligible(self):
        dataframe = pd.DataFrame({
            'Category': ['A', 'A', 'B', 'B'],
            'Temperature': [0, 5, 0, 5],
        })
        profile = compute_combined_prior_relevance(dataframe, [
            {
                'name': 'Category', 'col_type': 'categorical',
                'target_value': 'A', 'prior_weight': 0,
            },
            {
                'name': 'Temperature', 'col_type': 'numerical',
                'target_value': 0, 'tolerance': 1, 'prior_weight': 100,
            },
        ], 'average')
        np.testing.assert_array_equal(
            profile['eligible_mask'], np.array([True, False, True, False])
        )

    def test_preview_uses_the_same_explicit_prior_rules(self):
        dataframe = pd.DataFrame({
            'X': np.arange(6, dtype=float),
            'Temperature': [45, 49, 50, 51, 55, 70],
            'Y': np.arange(6, dtype=float),
        })
        preview = preview_transfer_selection(
            dataframe,
            features_config=[{'name': 'X', 'type': 'continuous', 'range': '[0, 5]'}],
            objectives_config=[{'name': 'Y', 'type': 'maximize'}],
            priors_config=[{
                'name': 'Temperature',
                'col_type': 'numerical',
                'target_value': 50,
                'tolerance': 2,
            }],
            requested_rows=6,
        )
        self.assertEqual(preview['eligible_rows'], 3)
        self.assertEqual(preview['rows_to_fit'], 3)
        self.assertEqual(preview['prior_stats']['Temperature']['n_eligible'], 3)

    def test_preview_api_rejects_an_invalid_numerical_tolerance(self):
        payload = {
            'data': [[0.0, 50.0, 1.0], [1.0, 51.0, 2.0], [2.0, 52.0, 3.0]],
            'columns': ['X', 'Temperature', 'Y'],
            'features': [{'name': 'X', 'type': 'continuous', 'range': '[0, 2]'}],
            'objectives': [{'name': 'Y', 'type': 'maximize'}],
            'priors': [{
                'name': 'Temperature',
                'col_type': 'numerical',
                'target_value': 50,
                'tolerance': -1,
            }],
            'tweaks': {'max_gp_rows': 3},
        }
        with app.test_client() as client:
            response = client.post('/transfer_selection_preview', json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn('zero or greater', response.get_json()['error'])

    def test_preview_api_accepts_sequence_prior_and_reports_eligible_rows(self):
        payload = {
            'data': [
                [0.0, 'MKTW', 1.0],
                [1.0, 'MKAW', 2.0],
                [2.0, 'MKTW', 3.0],
                [3.0, 'YYYY', 4.0],
            ],
            'columns': ['X', 'Sequence', 'Y'],
            'features': [{'name': 'X', 'type': 'continuous', 'range': '[0, 3]'}],
            'objectives': [{'name': 'Y', 'type': 'maximize'}],
            'priors': [{
                'name': 'Sequence',
                'col_type': 'amino_acid_sequence',
                'target_value': 'MKTW',
                'min_similarity': 0.75,
            }],
            'tweaks': {'max_gp_rows': 4},
        }
        with app.test_client() as client:
            response = client.post('/transfer_selection_preview', json=payload)
            self.assertEqual(response.status_code, 200)
            preview = response.get_json()
            self.assertEqual(preview['eligible_rows'], 3)
            self.assertEqual(preview['prior_stats']['Sequence']['n_eligible'], 3)
            payload['priors'][0]['min_similarity'] = 1.5
            response = client.post('/transfer_selection_preview', json=payload)
            self.assertEqual(response.status_code, 400)
            self.assertIn('between 0 and 100%', response.get_json()['error'])

    def test_upload_preserves_sequence_strings_that_look_like_missing_values(self):
        csv_data = b'Sequence,Y\nNA,1\nNAN,2\nNULL,3\n,4\n'
        with app.test_client() as client:
            response = client.post(
                '/upload',
                data={'file': (io.BytesIO(csv_data), 'sequences.csv')},
                content_type='multipart/form-data',
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()['data'],
            [['NA', 1], ['NAN', 2], ['NULL', 3], [None, 4]],
        )

    def test_no_prior_fallback_delegates_full_data_to_standard_bo(self):
        dataframe = pd.DataFrame({
            'X': np.arange(1000, dtype=float),
            'Y': np.arange(1000, dtype=float),
        })
        standard_result = BOOptimizationResult(
            suggestions=[{'X': 1001.0, 'Y': None}],
            performance={'Y': {'r2': 0.9}},
            constraint_status={'all_likely_feasible': True},
            diagnostics={
                'rows_valid': 1000,
                'data_handling': {
                    'source_rows': 1000,
                    'training_rows': 120,
                    'effective_mode': 'representative',
                    'selection': {'fidelity_score': 88.0},
                },
                'data_warnings': ['Representative-subset fallback used.'],
            },
        )

        with patch(
            'src.core.transfer_bo.run_bo_optimization',
            return_value=standard_result,
        ) as standard_bo:
            suggestions, performance, transfer, constraints = (
                run_transfer_bo_optimization(
                    df=dataframe,
                    features_config=[{
                        'name': 'X', 'type': 'continuous', 'range': '[0, 1100]'
                    }],
                    objectives_config=[{'name': 'Y', 'type': 'maximize'}],
                    priors_config=[],
                    tweaks={'batch_size': 1, 'data_mode': 'auto'},
                    device='cpu',
                    compute_device='cpu',
                    max_gp_rows=650,
                )
            )

        self.assertEqual(len(standard_bo.call_args.kwargs['df']), 1000)
        self.assertEqual(suggestions, standard_result.suggestions)
        self.assertEqual(performance, standard_result.performance)
        self.assertEqual(constraints, standard_result.constraint_status)
        self.assertEqual(transfer['model_rows_used'], 120)
        self.assertEqual(transfer['rows_downsampled'], 880)
        self.assertEqual(
            transfer['data_handling']['effective_mode'], 'representative'
        )


if __name__ == '__main__':
    unittest.main()
