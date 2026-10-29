"""Setup CSV round trips and domain validation without experiment-data edits."""

import io
import unittest

from app import app
from src.core.setup_csv import parse_setup_csv, serialize_setup_csv


class SetupCSVTests(unittest.TestCase):
    def test_doe_round_trip_keeps_current_values_and_grouped_constraints(self):
        setup = {
            'features': [
                {'name': 'Solvent', 'type': 'categorical',
                 'range': 'Water, "1,4-Dioxane", New solvent', 'included': True},
                {'name': 'Temperature', 'type': 'continuous', 'range': '[0, 120]',
                 'included': True},
            ],
            'objectives': [
                {'name': 'Yield', 'type': 'target', 'target': '85',
                 'importance': '70', 'included': True},
                {'name': 'Purity', 'type': 'maximize', 'target': '',
                 'importance': '30', 'included': True},
            ],
            'constraints': {
                'numerical': ['Temperature <= 100'],
                'disallowed': [[{'feature': 'Solvent', 'value': '1,4-Dioxane'}]],
            },
            'objective_constraints': [],
        }
        exported = serialize_setup_csv(setup, 'doe')
        parsed = parse_setup_csv(exported, 'doe')
        self.assertEqual(parsed['features'], setup['features'])
        self.assertEqual(parsed['objectives'], setup['objectives'])
        self.assertEqual(parsed['constraints'], setup['constraints'])
        self.assertEqual(parsed['warnings'], [])

    def test_bo_accepts_new_category_and_wider_range_than_dataset(self):
        setup = {
            'features': [
                {'name': 'Solvent', 'type': 'categorical',
                 'range': 'Water, Ethanol, Never tried', 'included': True},
                {'name': 'Temperature', 'type': 'continuous',
                 'range': '[0, 200]', 'included': True},
            ],
            'objectives': [{'name': 'Yield', 'type': 'maximize', 'target': '',
                            'importance': '100', 'included': True}],
            'constraints': {'numerical': ['Temperature <= 180'], 'disallowed': []},
            'objective_constraints': [{'objective': 'Yield', 'operator': '>=', 'value': '75'}],
        }
        imported = parse_setup_csv(serialize_setup_csv(setup, 'bo'), 'bo',
                                   ['Solvent', 'Temperature', 'Yield'])
        self.assertEqual(imported['features'][0]['range'], 'Water, Ethanol, Never tried')
        self.assertEqual(imported['features'][1]['range'], '[0, 200]')
        self.assertEqual(imported['objective_constraints'][0]['value'], '75')

    def test_sa_round_trip_and_cross_module_mapping(self):
        setup = {
            'features': [{'name': 'Temperature', 'type': 'discrete', 'range': '',
                          'included': False}],
            'objectives': [{'name': 'Yield', 'type': 'minimize', 'target': '',
                            'importance': '100', 'included': True}],
            'constraints': {'numerical': [], 'disallowed': []},
            'objective_constraints': [{'objective': 'Yield', 'operator': '<=', 'value': '10'}],
        }
        text = serialize_setup_csv(setup, 'sa')
        imported = parse_setup_csv(text, 'sa', ['Temperature', 'Yield'])
        self.assertEqual(imported['features'], setup['features'])
        self.assertEqual(imported['objective_constraints'], setup['objective_constraints'])
        cross = parse_setup_csv(text, 'bo', ['Temperature', 'Yield'])
        self.assertTrue(any('no feature ranges' in warning for warning in cross['warnings']))

    def test_missing_column_and_invalid_reference_reject_entire_file(self):
        setup = {
            'features': [{'name': 'Solvent', 'type': 'categorical',
                          'range': 'Water, Ethanol', 'included': True}],
            'objectives': [{'name': 'Yield', 'type': 'maximize', 'target': '',
                            'importance': '100', 'included': True}],
            'constraints': {'numerical': [], 'disallowed': [
                [{'feature': 'Solvent', 'value': 'Missing category'}],
            ]},
        }
        with self.assertRaisesRegex(ValueError, 'unknown value'):
            parse_setup_csv(serialize_setup_csv(setup, 'bo'), 'bo', ['Solvent', 'Yield'])
        setup['constraints']['disallowed'] = []
        with self.assertRaisesRegex(ValueError, 'missing from the dataset'):
            parse_setup_csv(serialize_setup_csv(setup, 'bo'), 'bo', ['Yield'])

    def test_http_routes_are_separate_from_dataset_upload(self):
        setup = {
            'features': [{'name': 'A', 'type': 'continuous', 'range': '[0, 2]',
                          'included': True}],
            'objectives': [{'name': 'Y', 'type': 'maximize', 'target': '',
                            'importance': '100', 'included': True}],
            'constraints': {'numerical': [], 'disallowed': []},
        }
        with app.test_client() as client:
            exported = client.post('/setup_csv/export', json={'module': 'bo', 'setup': setup})
            self.assertEqual(exported.status_code, 200)
            self.assertIn(b'CEDOS_BO_setup.csv', exported.headers['Content-Disposition'].encode())
            imported = client.post('/setup_csv/import', data={
                'module': 'bo', 'columns': '["A", "Y"]',
                'file': (io.BytesIO(exported.data), 'setup.csv'),
            })
        self.assertEqual(imported.status_code, 200)
        self.assertEqual(imported.get_json()['features'][0]['range'], '[0, 2]')

    def test_transbo_setup_excludes_prior_targets(self):
        setup = {
            'features': [{'name': 'Temperature', 'type': 'continuous',
                          'range': '[0, 150]', 'included': True}],
            'objectives': [{'name': 'Yield', 'type': 'maximize', 'target': '',
                            'importance': '100', 'included': True}],
            'constraints': {'numerical': ['Temperature <= 125'], 'disallowed': []},
            'objective_constraints': [{'objective': 'Yield', 'operator': '>=', 'value': '60'}],
        }
        imported = parse_setup_csv(serialize_setup_csv(setup, 'tbo'), 'tbo',
                                   ['Temperature', 'Yield', 'Historical solvent'])
        self.assertEqual(imported['features'], setup['features'])
        self.assertEqual(imported['objectives'], setup['objectives'])
        self.assertEqual(imported['constraints'], setup['constraints'])
        self.assertEqual(imported['objective_constraints'], setup['objective_constraints'])
        self.assertNotIn('Historical solvent', str(imported))


if __name__ == '__main__':
    unittest.main()
