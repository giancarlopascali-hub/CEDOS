from unittest.mock import patch

import pandas as pd
import torch

from src.core.transfer_bo import (
    run_transfer_bo_estimation,
    run_transfer_bo_optimization,
    run_transfer_bo_surface,
)


FEATURES = [
    {'name': 'X1', 'type': 'continuous', 'range': '[1, 4]'},
    {'name': 'X2', 'type': 'continuous', 'range': '[5, 8]'},
    {
        'name': 'Solvent',
        'type': 'categorical',
        'range': 'THF, Water',
        'choices': ['THF', 'Water'],
    },
]
OBJECTIVES = [{'name': 'Yield', 'type': 'maximize', 'importance': 100}]
PRIORS = [{
    'name': 'Campaign',
    'col_type': 'categorical',
    'target_value': 'A',
    'prior_weight': 100,
}]
TWEAKS = {
    'batch_size': 1,
    'acq_type': 'EI',
    'kernel': 'matern52',
    'exploration': 0.5,
    'max_gp_rows': 4,
}


def _historical_dataframe():
    return pd.DataFrame({
        'X1': [1.0, 2.0, 3.0, 4.0],
        'X2': [5.0, 6.0, 7.0, 8.0],
        'Solvent': ['THF', 'DMF', 'Water', 'THF'],
        'Campaign': ['A', 'A', 'A', 'A'],
        'Yield': [10.0, 12.0, 15.0, 18.0],
    })


def test_transbo_optimizer_accepts_removed_historical_category():
    suggestions, _, diagnostics, _ = run_transfer_bo_optimization(
        df=_historical_dataframe().iloc[:2].copy(),
        features_config=FEATURES,
        objectives_config=OBJECTIVES,
        priors_config=PRIORS,
        tweaks=TWEAKS,
        device=torch.device('cpu'),
        compute_device=torch.device('cpu'),
    )

    assert diagnostics['cold_start_warning'] is True
    assert suggestions
    assert all(item['Solvent'] in {'THF', 'Water'} for item in suggestions)
    assert all(item['Solvent'] != 'DMF' for item in suggestions)


def test_transbo_estimator_accepts_removed_historical_category():
    with patch('src.core.transfer_bo.fit_gpytorch_mll', return_value=None):
        results, _, _, workload = run_transfer_bo_estimation(
            df=_historical_dataframe(),
            input_values={
                'X1': 2.5,
                'X2': 6.5,
                'Solvent': 'THF',
                'Campaign': 'A',
            },
            features_config=FEATURES,
            objectives_config=OBJECTIVES,
            priors_config=PRIORS,
            tweaks=TWEAKS,
            device=torch.device('cpu'),
            compute_device=torch.device('cpu'),
        )

    assert 'Yield' in results
    assert workload['model_rows_used'] == 4


def test_transbo_surface_accepts_removed_historical_category():
    with patch('src.core.transfer_bo.fit_gpytorch_mll', return_value=None):
        result = run_transfer_bo_surface(
            df=_historical_dataframe(),
            features_config=FEATURES,
            objectives_config=OBJECTIVES,
            priors_config=PRIORS,
            tweaks=TWEAKS,
            x_feat='X1',
            y_feat='X2',
            z_axis='Yield',
            color_obj='Yield',
            fixed_values={'Solvent': 'THF'},
            resolution=5,
            device=torch.device('cpu'),
            compute_device=torch.device('cpu'),
        )

    assert result['slice_targets']['Solvent'] == 'THF'
    assert result['workload']['model_rows_used'] == 4
    assert len(result['experiments']) == 4
