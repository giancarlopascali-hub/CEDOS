"""CEDOS public acquisition contract around the unchanged transfer engine."""

from unittest.mock import patch

import pytest

from app import app
from src.core.bo_policy import select_bo_policy


def _payload(priors=True, tweaks=None):
    return {
        'data': [[0.0, 'A', 1.0], [1.0, 'A', 2.0], [2.0, 'A', 3.0]],
        'columns': ['x', 'context', 'y'],
        'features': [{'name': 'x', 'type': 'continuous', 'range': '[0, 2]'}],
        'objectives': [{'name': 'y', 'type': 'maximize', 'importance': 100}],
        'priors': ([{'name': 'context', 'col_type': 'categorical',
                    'target_value': 'A'}] if priors else []),
        'tweaks': {'acq_type': 'LCB', 'max_gp_rows': 3,
                   'avoid_reval': False, **(tweaks or {})},
    }


@pytest.mark.parametrize('has_priors, expected_core_acquisition', [
    (True, 'UCB'), (False, 'LCB'),
])
def test_public_lcb_uses_the_correct_core_branch(has_priors, expected_core_acquisition):
    # Transfer models a maximize-oriented utility U. Minimizing the public
    # lower confidence bound of loss L=-U is exactly maximizing UCB of U.
    mean_utility, uncertainty, beta = 2.5, 0.4, 1.2
    assert -mean_utility - beta * uncertainty == -(
        mean_utility + beta * uncertainty
    )
    assert select_bo_policy(1, [100], 'LCB', 4).key == 'qUCB'

    result = ([{'x': 1.5}], {}, {'algorithm_used': 'fixed_noise_gp'}, {})
    with app.test_client() as client, patch(
        'src.api.routes.run_transfer_bo_optimization', return_value=result
    ) as transfer:
        response = client.post('/transfer_optimize', json=_payload(has_priors))
    assert response.status_code == 200, response.get_data(as_text=True)
    assert transfer.call_args.kwargs['tweaks']['acq_type'] == expected_core_acquisition


@pytest.mark.parametrize('unsupported, message', [
    ({'constraints': {'numerical': ['x <= 1']}}, 'feature constraints'),
    ({'constraints': {'disallowed': [[{'feature': 'x', 'value': '0'}]]}},
     'feature constraints'),
    ({'avoid_reval': True}, 'duplicate avoidance'),
])
def test_transfer_rejects_unsupported_settings_before_queuing(unsupported, message):
    with app.test_client() as client:
        response = client.post('/transfer_optimize_submit',
                               json=_payload(tweaks=unsupported))
    assert response.status_code == 400
    assert message in response.get_json()['error']


def test_cedos_exposes_ei_and_lcb_for_both_optimizers():
    with app.test_client() as client:
        html = client.get('/').get_data(as_text=True)
    assert '<select id="tbo-acq-select">' in html
    tbo_options = html.split('<select id="tbo-acq-select">', 1)[1].split('</select>', 1)[0]
    bo_options = html.split('<select id="acq-type-select">', 1)[1].split('</select>', 1)[0]
    for options in (tbo_options, bo_options):
        assert 'value="EI"' in options
        assert 'value="LCB"' in options
        assert 'value="UCB"' not in options
