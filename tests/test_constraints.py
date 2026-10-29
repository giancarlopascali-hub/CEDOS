import math
import unittest

import pandas as pd

from src.core.constraints import (
    check_objective_constraints,
    encode_categorical_series,
    encode_categorical_training_series,
    eval_numerical_expression,
    get_categorical_choices,
    validate_feature_constraints,
    validate_feature_ranges,
    validate_objective_constraints,
)


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.features = [
            {'name': 'Temperature (C)', 'type': 'continuous', 'range': '[0, 100]'},
            {'name': 'Flow Rate', 'type': 'continuous', 'range': '[0, 10]'},
        ]

    def test_single_equals_and_punctuation_names(self):
        row = {'Temperature (C)': 50, 'Flow Rate': 5}
        self.assertTrue(
            eval_numerical_expression(
                'Temperature (C) = 10 * Flow Rate', row, self.features
            )
        )

    def test_rejects_code_execution_and_unsupported_power(self):
        with self.assertRaises(ValueError):
            validate_feature_constraints(
                ["__import__('os').system('echo unsafe') == 0"], self.features
            )
        with self.assertRaises(ValueError):
            validate_feature_constraints(['Flow Rate ** 2 <= 100'], self.features)

    def test_missing_or_non_numeric_values_are_not_coerced_to_zero(self):
        with self.assertRaises(ValueError):
            eval_numerical_expression('Flow Rate <= 2', {}, self.features)
        with self.assertRaises(ValueError):
            eval_numerical_expression(
                'Flow Rate <= 2', {'Flow Rate': 'unknown'}, self.features
            )

    def test_feature_range_validation(self):
        validate_feature_ranges([
            {'name': 'A', 'type': 'continuous', 'range': '[0, 1]'},
            {'name': 'B', 'type': 'regular', 'range': '[0, 1]5'},
            {'name': 'C', 'type': 'discrete', 'range': '1, 2, 3'},
            {'name': 'D', 'type': 'categorical', 'range': 'red, blue'},
        ])
        with self.assertRaises(ValueError):
            validate_feature_ranges([
                {'name': 'A', 'type': 'continuous', 'range': '0, 1'}
            ])

    def test_categorical_choices_preserve_embedded_commas(self):
        feature = {
            'name': 'Solvent',
            'type': 'categorical',
            'range': '"1,4-Dioxane", THF',
            'choices': ['1,4-Dioxane', 'THF'],
        }
        encoded, choices = encode_categorical_series(
            pd.Series(['1,4-Dioxane', 'THF']), feature
        )
        self.assertEqual(choices, ['1,4-Dioxane', 'THF'])
        self.assertEqual(encoded.tolist(), [0, 1])
        self.assertEqual(
            get_categorical_choices({
                'name': 'Solvent',
                'type': 'categorical',
                'range': '"1,4-Dioxane", THF',
            }),
            ['1,4-Dioxane', 'THF'],
        )

    def test_unknown_categorical_values_do_not_fall_back_to_first_choice(self):
        feature = {
            'name': 'Solvent',
            'type': 'categorical',
            'range': 'THF',
            'choices': ['THF'],
        }
        with self.assertRaisesRegex(ValueError, '1,4-Dioxane'):
            encode_categorical_series(pd.Series(['1,4-Dioxane']), feature)

    def test_bo_training_keeps_removed_categories_outside_candidate_codes(self):
        feature = {
            'name': 'Solvent',
            'type': 'categorical',
            'range': 'THF, Water',
            'choices': ['THF', 'Water'],
        }
        encoded, model_choices = encode_categorical_training_series(
            pd.Series(['THF', 'DMF', 'Water', 'DMF']), feature
        )
        self.assertEqual(model_choices, ['THF', 'Water', 'DMF'])
        self.assertEqual(encoded.tolist(), [0, 2, 1, 2])

    def test_objective_constraints_are_strictly_validated(self):
        objectives = [{'name': 'Yield'}, {'name': 'Impurity'}]
        normalized = validate_objective_constraints([
            {'objective': 'Yield', 'operator': '≥', 'value': '80'},
            {'objective': 'Impurity', 'operator': '<=', 'value': 2.5},
        ], objectives)
        self.assertEqual(normalized[0]['operator'], '>=')
        with self.assertRaises(ValueError):
            validate_objective_constraints([
                {'objective': 'Missing', 'operator': '>=', 'value': 1}
            ], objectives)
        with self.assertRaises(ValueError):
            validate_objective_constraints([
                {'objective': 'Yield', 'operator': '>', 'value': 1}
            ], objectives)

    def test_objective_boundaries_tolerance_and_nan(self):
        constraints = [
            {'objective': 'Yield', 'operator': '>=', 'value': 80.0},
            {'objective': 'Yield', 'operator': '<=', 'value': 95.0},
        ]
        satisfied, violations = check_objective_constraints(
            {'Yield': 80.0 - 0.5e-7}, constraints
        )
        self.assertTrue(satisfied)
        self.assertEqual(violations, [])
        satisfied, violations = check_objective_constraints(
            {'Yield': math.nan}, constraints
        )
        self.assertFalse(satisfied)
        self.assertEqual(violations[0]['predicted'], None)


if __name__ == '__main__':
    unittest.main()
