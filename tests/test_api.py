"""
tests/test_api.py
==================
Integration tests for the Flask API endpoints.
Uses Flask's built-in test client to verify that each route returns the
expected HTTP status codes and JSON response structures.
"""

import pytest
import json
import io
import sys
import os

# Ensure the app root is on the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app as flask_app
from src.core.bo import exploration_to_beta, validate_exploration


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def client():
    """Create a Flask test client with testing mode enabled."""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture
def sample_doe_payload():
    """A minimal valid DoE request payload."""
    return {
        'features': [
            {'name': 'A', 'type': 'continuous', 'range': '[0, 10]'},
            {'name': 'B', 'type': 'continuous', 'range': '[5, 20]'},
            {'name': 'C', 'type': 'continuous', 'range': '[100, 200]'},
        ],
        'tweaks': {'model': 'bbdesign', 'max_runs': 20}
    }


@pytest.fixture
def sample_csv_data():
    """A small in-memory CSV payload representing a 3-row dataset."""
    return (
        b"A,B,C,Y\n"
        b"1,5,100,0.5\n"
        b"5,12,150,0.8\n"
        b"9,18,190,0.3\n"
    )


@pytest.mark.parametrize(
    ('exploration', 'expected_beta'),
    [(0.0, 0.0), (0.5, 5.0), (1.0, 10.0)],
)
def test_exploration_maps_monotonically_to_standard_bo_beta(exploration, expected_beta):
    assert exploration_to_beta(exploration) == pytest.approx(expected_beta)


@pytest.mark.parametrize('exploration', [-0.01, 1.01, float('nan'), float('inf'), 'invalid'])
def test_exploration_rejects_values_outside_public_scale(exploration):
    with pytest.raises(ValueError, match='between 0 and 1'):
        validate_exploration(exploration)


@pytest.mark.parametrize('model', ['bbdesign', 'ccdesign', 'fracfact', 'fullfact', 'def_screening', 'lhs', 'doptimal'])
def test_all_doe_api_models_accept_constraint_rules(client, model):
    response = client.post('/doe', json={
        'features': [{'name': name, 'type': 'continuous', 'range': '[0, 1]'} for name in 'ABCD'],
        'tweaks': {'model': model, 'max_runs': 8, 'dopt_model_order': 'linear',
                   'constraints': {'numerical': ['A <= 0.5'],
                                   'disallowed': [[{'feature': 'B', 'value': '1'}]]}},
    })
    assert response.status_code == 200
    output = response.get_json()
    assert output['suggestions']
    assert all(row['A'] <= .5 and row['B'] != 1 for row in output['suggestions'])
    if model != 'doptimal':
        assert output['metrics']['constraint_filtered'] is True
        assert 'missing_levels' in output['metrics']


# ==============================================================================
# Test: Static Pages
# ==============================================================================

class TestStaticPages:
    """Tests that the main application pages serve correctly."""

    def test_index_returns_200(self, client):
        """GET / should return HTTP 200."""
        resp = client.get('/')
        assert resp.status_code == 200

    def test_guide_returns_200(self, client):
        """GET /guide should return HTTP 200."""
        resp = client.get('/guide')
        assert resp.status_code == 200


# ==============================================================================
# Test: /upload Endpoint
# ==============================================================================

class TestUploadEndpoint:
    """Tests for the CSV file upload endpoint."""

    def test_upload_valid_csv(self, client, sample_csv_data):
        """POST /upload with a valid CSV should return data and columns."""
        data = {
            'file': (io.BytesIO(sample_csv_data), 'test.csv', 'text/csv')
        }
        resp = client.post('/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 200

        result = resp.get_json()
        assert 'columns' in result
        assert 'data' in result
        assert set(result['columns']) == {'A', 'B', 'C', 'Y'}

    def test_upload_no_file_returns_400(self, client):
        """POST /upload with no file attached should return HTTP 400."""
        resp = client.post('/upload', data={}, content_type='multipart/form-data')
        assert resp.status_code == 400

        result = resp.get_json()
        assert 'error' in result

    def test_upload_empty_filename_returns_400(self, client):
        """POST /upload with an empty filename should return HTTP 400."""
        data = {
            'file': (io.BytesIO(b''), '', 'text/csv')
        }
        resp = client.post('/upload', data=data, content_type='multipart/form-data')
        assert resp.status_code == 400


# ==============================================================================
# Test: /doe Endpoint
# ==============================================================================

class TestDoeEndpoint:
    """Tests for the DoE generation endpoint."""

    @pytest.mark.parametrize('model,runs', [('fullfact', 3), ('fullfact', 8), ('doptimal', 1), ('doptimal', 8)])
    def test_mixed_level_run_counts_and_warning_payload(self, client, model, runs):
        response = client.post('/doe', json={
            'features': [
                {'name': 'Solvent', 'type': 'categorical', 'range': 'Water, Ethanol, DMSO'},
                {'name': 'Dose', 'type': 'discrete', 'range': '1, 2'},
            ],
            'tweaks': {'model': model, 'max_runs': runs, 'use_exact_runs': True},
        })
        assert response.status_code == 200
        result = response.get_json()
        assert 'error' not in result
        assert len(result['suggestions']) == runs
        assert result['metrics']['design_warning']
        assert result['metrics']['added_replicates'] == max(0, runs - 6)
        assert {'orthogonality', 'efficiency', 'resolution', 'curvature'} <= result['metrics'].keys()

    def test_full_factorial_is_exposed_in_selector(self, client):
        response = client.get('/')
        assert b'<option value="fullfact">Full Factorial</option>' in response.data

    def test_doe_returns_suggestions_and_metrics(self, client, sample_doe_payload):
        """POST /doe with a valid payload should return 'suggestions' and 'metrics'."""
        resp = client.post(
            '/doe',
            data=json.dumps(sample_doe_payload),
            content_type='application/json'
        )
        assert resp.status_code == 200

        result = resp.get_json()
        assert 'suggestions' in result
        assert 'metrics' in result
        assert len(result['suggestions']) > 0

    def test_doe_invalid_model_fallback(self, client):
        """POST /doe with only 1 feature should return HTTP 500 with error message."""
        payload = {
            'features': [
                {'name': 'A', 'type': 'continuous', 'range': '[0, 10]'},
            ],
            'tweaks': {'model': 'bbdesign'}
        }
        resp = client.post(
            '/doe',
            data=json.dumps(payload),
            content_type='application/json'
        )
        result = resp.get_json()
        assert 'error' in result


class TestBoSurfaceEndpoint:
    """Tests for the BO surface generation endpoint."""

    def test_bo_surface_success(self, client):
        """POST /bo_surface should return grid data and evaluated experiments."""
        payload = {
            'data': [
                [1.0, 5.0, 10.0],
                [2.0, 6.0, 12.0],
                [3.0, 7.0, 15.0],
                [4.0, 8.0, 18.0],
                [5.0, 9.0, 20.0]
            ],
            'columns': ['X1', 'X2', 'Y1'],
            'features': [
                {'name': 'X1', 'type': 'continuous', 'range': '[1, 5]'},
                {'name': 'X2', 'type': 'continuous', 'range': '[5, 9]'}
            ],
            'objectives': [
                {'name': 'Y1', 'type': 'maximize', 'importance': '100'}
            ],
            'tweaks': {'kernel': 'matern52'},
            'x_feat': 'X1',
            'y_feat': 'X2',
            'z_axis': 'Y1',
            'color_obj': 'Y1',
            'resolution': 10
        }
        resp = client.post('/bo_surface', data=json.dumps(payload), content_type='application/json')
        assert resp.status_code == 200
        result = resp.get_json()
        assert 'x' in result
        assert 'y' in result
        assert 'z' in result
        assert 'color' in result
        assert len(result['x']) == 10
        assert len(result['y']) == 10
        assert len(result['z']) == 10
        assert len(result['experiments']) == 5

    def test_bo_surface_accepts_historical_category_removed_from_search_domain(self, client):
        payload = {
            'data': [
                [1.0, 5.0, 'THF', 10.0],
                [2.0, 6.0, 'DMF', 12.0],
                [3.0, 7.0, 'Water', 15.0],
                [4.0, 8.0, 'DMF', 18.0],
                [5.0, 9.0, 'THF', 20.0],
            ],
            'columns': ['X1', 'X2', 'Solvent', 'Y1'],
            'features': [
                {'name': 'X1', 'type': 'continuous', 'range': '[1, 5]'},
                {'name': 'X2', 'type': 'continuous', 'range': '[5, 9]'},
                {
                    'name': 'Solvent',
                    'type': 'categorical',
                    'range': 'THF, Water',
                    'choices': ['THF', 'Water'],
                },
            ],
            'objectives': [
                {'name': 'Y1', 'type': 'maximize', 'importance': '100'}
            ],
            'tweaks': {'kernel': 'matern52'},
            'x_feat': 'X1',
            'y_feat': 'X2',
            'z_axis': 'Y1',
            'color_obj': 'Y1',
            'fixed_values': {'Solvent': 'THF'},
            'resolution': 5,
        }
        response = client.post('/bo_surface', json=payload)
        assert response.status_code == 200
        result = response.get_json()
        assert 'error' not in result
        assert result['slice_targets']['Solvent'] == 'THF'
        assert len(result['experiments']) == 5


class TestBoCategoricalDomainEdits:
    def test_optimizer_never_proposes_a_removed_historical_category(self, client):
        payload = {
            'data': [
                [1.0, 'THF', 10.0],
                [2.0, 'DMF', 12.0],
                [1.0, 'Water', 15.0],
                [2.0, 'THF', 18.0],
            ],
            'columns': ['Dose', 'Solvent', 'Yield'],
            'features': [
                {'name': 'Dose', 'type': 'discrete', 'range': '1, 2'},
                {
                    'name': 'Solvent',
                    'type': 'categorical',
                    'range': 'THF, Water',
                    'choices': ['THF', 'Water'],
                },
            ],
            'objectives': [
                {'name': 'Yield', 'type': 'maximize', 'importance': '100'}
            ],
            'tweaks': {
                'batch_size': 1,
                'acq_type': 'EI',
                'kernel': 'matern52',
                'exploration': 0.5,
                'optimization_strategy': 'exhaustive_grid',
                'avoid_reval': True,
            },
        }
        response = client.post('/optimize', json=payload)
        assert response.status_code == 200
        result = response.get_json()
        assert 'error' not in result
        assert result['suggestions']
        assert all(
            suggestion['Solvent'] in {'THF', 'Water'}
            for suggestion in result['suggestions']
        )
        assert all(
            suggestion['Solvent'] != 'DMF'
            for suggestion in result['suggestions']
        )

