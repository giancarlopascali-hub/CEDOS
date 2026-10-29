"""
tests/test_doe.py
==================
Unit tests for the DoE (Design of Experiments) core module.
Tests verify that the generate_doe() function returns valid tables
with the correct structure and design quality metrics.
"""

import itertools
import numpy as np
import pytest
from src.core.doe import generate_doe


# ==============================================================================
# Fixtures — reusable feature/tweaks configurations
# ==============================================================================

@pytest.fixture
def three_cont_features():
    """3 continuous features — minimum for Box-Behnken design."""
    return [
        {'name': 'A', 'type': 'continuous', 'range': '[0, 10]'},
        {'name': 'B', 'type': 'continuous', 'range': '[5, 20]'},
        {'name': 'C', 'type': 'continuous', 'range': '[100, 200]'},
    ]


@pytest.fixture
def two_cont_features():
    """2 continuous features — valid for CCD, LHS but not Box-Behnken."""
    return [
        {'name': 'X', 'type': 'continuous', 'range': '[0, 1]'},
        {'name': 'Y', 'type': 'continuous', 'range': '[0, 1]'},
    ]


# ==============================================================================
# Test: Box-Behnken Design (bbdesign)
# ==============================================================================

class TestBoxBehnkenDesign:
    """Tests for the Box-Behnken design algorithm."""

    def test_returns_non_empty_table(self, three_cont_features):
        """generate_doe should return a non-empty table for 3 continuous features."""
        table, metrics = generate_doe(three_cont_features, {'model': 'bbdesign'})
        assert len(table) > 0

    def test_table_contains_all_features(self, three_cont_features):
        """Every row in the suggestions table should have all feature keys."""
        table, _ = generate_doe(three_cont_features, {'model': 'bbdesign'})
        for row in table:
            assert 'A' in row
            assert 'B' in row
            assert 'C' in row

    def test_metrics_structure(self, three_cont_features):
        """Metrics dictionary should have all required keys."""
        _, metrics = generate_doe(three_cont_features, {'model': 'bbdesign'})
        assert 'orthogonality' in metrics
        assert 'efficiency' in metrics
        assert 'resolution' in metrics
        assert 'curvature' in metrics
        assert 'runs' in metrics
        assert 'model' in metrics

    def test_values_within_bounds(self, three_cont_features):
        """All numerical values should be within the declared feature bounds."""
        table, _ = generate_doe(three_cont_features, {'model': 'bbdesign'})
        for row in table:
            assert 0.0 <= float(row['A']) <= 10.0
            assert 5.0 <= float(row['B']) <= 20.0
            assert 100.0 <= float(row['C']) <= 200.0

    def test_requires_at_least_3_features(self, two_cont_features):
        """Box-Behnken requires ≥3 features; should raise ValueError with 2."""
        with pytest.raises(ValueError, match="Box-Behnken"):
            generate_doe(two_cont_features, {'model': 'bbdesign'})


# ==============================================================================
# Test: Latin Hypercube Sampling (LHS)
# ==============================================================================

class TestLatinHypercubeSampling:
    """Tests for the Latin Hypercube Sampling design."""

    def test_lhs_returns_correct_run_count(self, two_cont_features):
        """LHS should return exactly max_runs rows."""
        table, _ = generate_doe(two_cont_features, {'model': 'lhs', 'max_runs': 15})
        assert len(table) == 15

    def test_lhs_default_run_count(self, two_cont_features):
        """LHS should default to 50 runs when max_runs not specified."""
        table, _ = generate_doe(two_cont_features, {'model': 'lhs'})
        assert len(table) == 50

    def test_lhs_values_within_bounds(self, two_cont_features):
        """All LHS sample values should be within declared feature bounds."""
        table, _ = generate_doe(two_cont_features, {'model': 'lhs', 'max_runs': 10})
        for row in table:
            assert 0.0 <= float(row['X']) <= 1.0
            assert 0.0 <= float(row['Y']) <= 1.0


# ==============================================================================
# Test: Categorical Features
# ==============================================================================

class TestCategoricalFeatures:
    """Tests for designs that include categorical features."""

    def test_categorical_values_are_valid_choices(self):
        """Categorical feature values in the output must only be from the declared choices."""
        features = [
            {'name': 'Solvent', 'type': 'categorical', 'range': 'Water, Ethanol, DMSO'},
            {'name': 'Temp',    'type': 'continuous',  'range': '[20, 80]'},
            {'name': 'pH',      'type': 'continuous',  'range': '[4, 9]'},
        ]
        table, _ = generate_doe(features, {
            'model': 'doptimal', 'dopt_model_order': 'linear', 'max_runs': 8,
        })
        valid_choices = {'Water', 'Ethanol', 'DMSO'}
        for row in table:
            assert row['Solvent'] in valid_choices


# ==============================================================================
# Test: Constraint Filtering
# ==============================================================================

class TestConstraints:
    """Constraints must be applied before D-Optimal selection."""

    def test_numerical_constraint_filters_doptimal_candidates(self):
        """D-Optimal returns the exact count entirely inside the feasible region."""
        features = [
            {'name': 'A', 'type': 'continuous', 'range': '[0, 10]'},
            {'name': 'B', 'type': 'continuous', 'range': '[0, 10]'},
            {'name': 'C', 'type': 'continuous', 'range': '[0, 10]'},
        ]
        tweaks = {
            'model': 'doptimal',
            'dopt_model_order': 'linear',
            'max_runs': 8,
            'constraints': {'numerical': ['A + B <= 10'], 'disallowed': []}
        }
        table, metrics = generate_doe(features, tweaks)
        assert len(table) == 8
        assert metrics['runs'] == 8
        for row in table:
            assert float(row['A']) + float(row['B']) <= 10.0 + 1e-6  # Small tolerance for float rounding

    def test_disallowed_combination_is_excluded_before_doptimal_selection(self):
        """Rows that match a disallowed categorical combination should be removed."""
        features = [
            {'name': 'Solvent', 'type': 'categorical', 'range': 'Water, Ethanol'},
            {'name': 'Catalyst','type': 'categorical', 'range': 'A, B'},
            {'name': 'Temp',    'type': 'continuous',  'range': '[20, 80]'},
        ]
        tweaks = {
            'model': 'doptimal',
            'dopt_model_order': 'linear',
            'max_runs': 6,
            'constraints': {
                'numerical': [],
                'disallowed': [[
                    {'feature': 'Solvent',  'value': 'Water'},
                    {'feature': 'Catalyst', 'value': 'A'}
                ]]
            }
        }
        table, _ = generate_doe(features, tweaks)
        for row in table:
            assert not (row['Solvent'] == 'Water' and row['Catalyst'] == 'A')

    def test_structured_design_reports_filtering(self, three_cont_features):
        table, metrics = generate_doe(three_cont_features, {
            'model': 'bbdesign',
            'constraints': {'numerical': ['A <= 5'], 'disallowed': []},
        })
        assert all(row['A'] <= 5 for row in table)
        assert metrics['constraint_filtered']
        assert metrics['excluded_runs'] > 0


# ==============================================================================
# Test: Error Cases
# ==============================================================================

class TestErrorCases:
    """Tests for invalid input configurations."""

    def test_single_feature_raises_error(self):
        """DoE requires at least 2 features; 1 should raise ValueError."""
        features = [{'name': 'A', 'type': 'continuous', 'range': '[0, 10]'}]
        with pytest.raises(ValueError, match="at least 2 features"):
            generate_doe(features, {'model': 'bbdesign'})

    def test_empty_features_raises_error(self):
        """An empty features list should raise ValueError."""
        with pytest.raises(ValueError):
            generate_doe([], {'model': 'bbdesign'})


# ==============================================================================
# Test: Feature Precision Rules
# ==============================================================================

class TestFeaturePrecision:
    """Tests for decimal precision of generated feature values."""

    def test_lhs_preserves_distinct_strata_for_whole_number_bounds(self):
        """Formatting must not collapse an LHS to a handful of duplicate levels."""
        features = [
            {'name': 'Temp', 'type': 'continuous', 'range': '[10, 50]'},
            {'name': 'Press', 'type': 'continuous', 'range': '[1, 5]'},
        ]
        table, _ = generate_doe(features, {'model': 'lhs', 'max_runs': 50})
        assert len({row['Temp'] for row in table}) == 50
        assert len({row['Press'] for row in table}) == 50

    def test_explicit_decimal_ranges_remain_bounded(self):
        """Continuous LHS values remain inside explicitly formatted bounds."""
        features = [
            {'name': 'Conc', 'type': 'continuous', 'range': '[0.00, 1.00]'},
            {'name': 'Flow', 'type': 'continuous', 'range': '[0.10, 0.50]'},
        ]
        table, _ = generate_doe(features, {'model': 'lhs', 'max_runs': 10})
        for row in table:
            assert 0.0 <= row['Conc'] <= 1.0
            assert 0.10 <= row['Flow'] <= 0.50


# ==============================================================================
# Test: Central Composite Design (ccdesign)
# ==============================================================================

class TestCentralCompositeDesign:
    """Tests for the Central Composite design algorithm."""

    def test_returns_non_empty_table(self, three_cont_features):
        """generate_doe should return a non-empty table for 3 continuous features."""
        table, metrics = generate_doe(three_cont_features, {'model': 'ccdesign'})
        assert len(table) > 0
        assert metrics['model'] == 'ccdesign'
        assert metrics['center_replicates'] > 0

    def test_ccdesign_no_unexpected_duplicates(self, three_cont_features):
        """The only duplicate rows in CCD should be the center-point replicates."""
        table, metrics = generate_doe(three_cont_features, {'model': 'ccdesign'})
        rows = [tuple(r.items()) for r in table]
        unique_rows = set(rows)
        # Total minus unique is the number of excess duplicate rows
        n_dups = len(rows) - len(unique_rows)
        # All duplicates should belong to the single center-point condition
        assert n_dups == metrics['center_replicates'] - 1

    def test_face_centered_ccd_stays_within_user_bounds(self, three_cont_features):
        """A CCD must not silently propose values outside declared hard ranges."""
        table, _ = generate_doe(three_cont_features, {'model': 'ccdesign'})
        for row in table:
            assert 0 <= row['A'] <= 10
            assert 5 <= row['B'] <= 20
            assert 100 <= row['C'] <= 200


# ==============================================================================
# Test: Center-Point Ordering (RSM models)
# ==============================================================================

class TestCenterPointOrdering:
    """Tests verifying that replicate center points appear at the end of the table."""

    def test_bbdesign_center_points_are_last(self, three_cont_features):
        """For Box-Behnken, all center points (midpoint of all features) must be at the end."""
        table, metrics = generate_doe(three_cont_features, {'model': 'bbdesign'})
        n_centers = metrics['center_replicates']
        assert n_centers > 0

        # Midpoint for each feature: A: 5.0, B: 12.5, C: 150.0
        center_row = {'A': 5.0, 'B': 12.5, 'C': 150.0}

        # The last n_centers rows must all be the center point
        for row in table[-n_centers:]:
            for k, v in center_row.items():
                assert abs(row[k] - v) < 1e-3

        # None of the earlier rows should be the center point
        for row in table[:-n_centers]:
            is_center = all(abs(row[k] - center_row[k]) < 1e-3 for k in center_row)
            assert not is_center, "Center point found before the end of the table!"

    def test_ccdesign_center_points_are_last(self, three_cont_features):
        """For CCD, all center points must be at the end of the table (not split mid-design)."""
        table, metrics = generate_doe(three_cont_features, {'model': 'ccdesign'})
        n_centers = metrics['center_replicates']
        assert n_centers > 0

        center_row = {'A': 5.0, 'B': 12.5, 'C': 150.0}

        # The last n_centers rows must all be the center point
        for row in table[-n_centers:]:
            for k, v in center_row.items():
                assert abs(row[k] - v) < 1e-3

        # None of the earlier rows should be the center point
        for row in table[:-n_centers]:
            is_center = all(abs(row[k] - center_row[k]) < 1e-3 for k in center_row)
            assert not is_center, "Center point found before the end of the table!"

    def test_metrics_reports_center_replicates(self, three_cont_features):
        """center_replicates metric should be >0 for RSM models and 0 for others."""
        _, bb_metrics = generate_doe(three_cont_features, {'model': 'bbdesign'})
        assert bb_metrics['center_replicates'] == 3

        _, cc_metrics = generate_doe(three_cont_features, {'model': 'ccdesign'})
        assert cc_metrics['center_replicates'] == 8

        _, lhs_metrics = generate_doe(three_cont_features, {'model': 'lhs', 'max_runs': 10})
        assert lhs_metrics['center_replicates'] == 0

        _, dopt_metrics = generate_doe(three_cont_features, {'model': 'doptimal', 'max_runs': 10})
        assert dopt_metrics['center_replicates'] == 0


# ==============================================================================
# Test: D-Optimal Design Uniqueness
# ==============================================================================

class TestDOptimalUniqueness:
    """Tests verifying that D-Optimal design generates 100% unique experimental runs."""

    def test_doptimal_no_duplicate_runs_linear(self, three_cont_features):
        """Linear model D-optimal must contain zero duplicate rows."""
        table, metrics = generate_doe(three_cont_features, {
            'model': 'doptimal', 'dopt_model_order': 'linear', 'max_runs': 20
        })
        rows = [tuple(r.items()) for r in table]
        assert len(rows) == len(set(rows)), "D-optimal linear produced duplicate rows!"
        assert metrics['center_replicates'] == 0

    def test_doptimal_no_duplicate_runs_interaction(self, three_cont_features):
        """Interaction model D-optimal must contain zero duplicate rows."""
        table, metrics = generate_doe(three_cont_features, {
            'model': 'doptimal', 'dopt_model_order': 'interaction', 'max_runs': 20
        })
        rows = [tuple(r.items()) for r in table]
        assert len(rows) == len(set(rows)), "D-optimal interaction produced duplicate rows!"

    def test_doptimal_no_duplicate_runs_quadratic(self, three_cont_features):
        """Quadratic model D-optimal must contain zero duplicate rows."""
        table, metrics = generate_doe(three_cont_features, {
            'model': 'doptimal', 'dopt_model_order': 'quadratic', 'max_runs': 20
        })
        rows = [tuple(r.items()) for r in table]
        assert len(rows) == len(set(rows)), "D-optimal quadratic produced duplicate rows!"

    def test_doptimal_no_duplicate_runs_large(self, three_cont_features):
        """50-run D-optimal must contain zero duplicate rows."""
        table, _ = generate_doe(three_cont_features, {
            'model': 'doptimal', 'dopt_model_order': 'quadratic', 'max_runs': 50
        })
        rows = [tuple(r.items()) for r in table]
        assert len(rows) == len(set(rows)), f"Expected 50 unique rows, got {len(set(rows))}"


# ==============================================================================
# Test: Exact-N repetition for structured designs
# ==============================================================================

class TestStructuredExactRuns:
    """Exact counts above the natural design add transparent repetitions."""

    def test_fracfact_exact_above_natural_appends_repeats(self, three_cont_features):
        table, metrics = generate_doe(three_cont_features, {
            'model': 'fracfact', 'use_exact_runs': True, 'max_runs': 25
        })
        assert len(table) == 25
        assert metrics['oversampling_warning'] is True
        assert metrics['natural_runs'] == 8
        assert metrics['added_replicates'] == 17
        assert metrics['design_status'] == 'Replicated'
        assert table[8:] == [table[index % 8] for index in range(17)]

    def test_def_screening_exact_above_natural_appends_repeats(self, three_cont_features):
        table, metrics = generate_doe(three_cont_features, {
            'model': 'def_screening', 'use_exact_runs': True, 'max_runs': 15
        })
        assert len(table) == 15
        assert metrics['oversampling_warning'] is True
        assert metrics['natural_runs'] == 13
        assert metrics['added_replicates'] == 2
        assert table[-2:] == table[:2]

    def test_bbdesign_exact_above_natural_reports_both_repeat_types(self, three_cont_features):
        table, metrics = generate_doe(three_cont_features, {
            'model': 'bbdesign', 'use_exact_runs': True, 'max_runs': 25
        })
        assert len(table) == 25
        assert metrics['center_replicates'] == 3
        assert metrics['added_replicates'] == 10
        assert table[-10:] == table[:10]


class TestAlgorithmWiring:
    """Each front-end option must call its named statistical design."""

    def test_definitive_screening_is_three_level_and_not_pb(self):
        features = [
            {'name': f'X{i}', 'type': 'continuous', 'range': '[0, 10]'}
            for i in range(6)
        ]
        table, metrics = generate_doe(features, {'model': 'def_screening'})
        assert len(table) == 13
        assert metrics['model'] == 'def_screening'
        for feature in features:
            assert set(row[feature['name']] for row in table) == {0.0, 5.0, 10.0}

    def test_plackett_burman_request_is_rejected(self, three_cont_features):
        with pytest.raises(ValueError, match="Unsupported DoE model"):
            generate_doe(three_cont_features, {'model': 'pbdesign'})

    def test_fractional_factorial_honours_resolution(self):
        features = [
            {'name': f'X{i}', 'type': 'continuous', 'range': '[-1, 1]'}
            for i in range(5)
        ]
        table, metrics = generate_doe(features, {
            'model': 'fracfact', 'fracfact_resolution': 4,
        })
        assert len(table) == 16
        assert 'IV' in metrics['resolution'] or 'V' in metrics['resolution']

    def test_lhs_metrics_are_not_misrepresented(self, two_cont_features):
        _, metrics = generate_doe(two_cont_features, {'model': 'lhs', 'max_runs': 20})
        assert metrics['orthogonality'] == 'N/A'
        assert metrics['efficiency'] == 'N/A'
        assert metrics['resolution'] == 'N/A'
        assert metrics['curvature'] == 'N/A'


class TestRunCountModes:
    """Max/Exact behavior is defined around each structured design's natural size."""

    def test_max_below_natural_returns_reduced_subset_and_warning(self, three_cont_features):
        table, metrics = generate_doe(three_cont_features, {
            'model': 'bbdesign', 'max_runs': 10, 'use_exact_runs': False,
        })
        assert len(table) == 10
        assert metrics['natural_runs'] == 15
        assert metrics['design_status'] == 'Reduced'
        assert metrics['reduction_warning'] is True
        assert metrics['design_warning']
        assert len({tuple(row.values()) for row in table}) == 10
        assert metrics['rank'] == metrics['parameters']
        assert metrics['efficiency'] > 0

    def test_max_above_natural_does_not_add_runs(self, three_cont_features):
        table, metrics = generate_doe(three_cont_features, {
            'model': 'bbdesign', 'max_runs': 25, 'use_exact_runs': False,
        })
        assert len(table) == 15
        assert metrics['design_status'] == 'Natural'
        assert metrics['added_replicates'] == 0

    def test_string_false_is_not_treated_as_exact(self, three_cont_features):
        table, metrics = generate_doe(three_cont_features, {
            'model': 'bbdesign', 'max_runs': 20, 'use_exact_runs': 'false',
        })
        assert len(table) == 15
        assert metrics['design_status'] == 'Natural'


class TestDOptimalMixedFactors:
    """D-Optimal selection must optimize and preserve all factor types in order."""

    def test_mixed_factor_order_and_variation(self):
        features = [
            {'name': 'Catalyst', 'type': 'categorical', 'range': 'A, B, C'},
            {'name': 'Temperature', 'type': 'continuous', 'range': '[20, 80]'},
            {'name': 'Pressure', 'type': 'continuous', 'range': '[1, 5]'},
        ]
        table, metrics = generate_doe(features, {
            'model': 'doptimal', 'dopt_model_order': 'quadratic', 'max_runs': 18,
        })
        assert len(table) == 18
        assert list(table[0]) == ['Catalyst', 'Temperature', 'Pressure']
        assert {row['Catalyst'] for row in table} == {'A', 'B', 'C'}
        assert len({row['Temperature'] for row in table}) > 1
        assert len({row['Pressure'] for row in table}) > 1
        assert metrics['rank'] == metrics['parameters']


class TestFullFactorial:
    def test_all_mixed_level_combinations_and_feature_order(self):
        features = [
            {'name': 'Solvent', 'type': 'categorical', 'range': 'Water, Ethanol, DMSO'},
            {'name': 'Dose', 'type': 'discrete', 'range': '1, 2, 10'},
            {'name': 'Temperature', 'type': 'continuous', 'range': '[20, 80]'},
            {'name': 'Flow', 'type': 'regular', 'range': '[0, 1, 4]'},
        ]
        table, metrics = generate_doe(features, {'model': 'fullfact', 'max_runs': 100})
        expected = set(itertools.product(['Water', 'Ethanol', 'DMSO'], [1, 2, 10], [20, 80], np.linspace(0, 1, 4)))
        assert set(tuple(row.values()) for row in table) == expected
        assert list(table[0]) == ['Solvent', 'Dose', 'Temperature', 'Flow']
        assert len(table) == metrics['natural_runs'] == 72
        assert metrics['resolution'] == 'Full factorial'
        assert metrics['curvature'] == 'Partial'
        assert metrics['rank'] == metrics['parameters']

    def test_categorical_only_including_quoted_and_repeated_labels(self):
        features = [
            {'name': 'Solvent', 'type': 'categorical', 'range': '"1,4-Dioxane", THF, Water, THF'},
            {'name': 'Catalyst', 'type': 'categorical', 'range': 'A, B, C, D'},
        ]
        table, metrics = generate_doe(features, {'model': 'fullfact'})
        assert len(table) == 12
        assert {row['Solvent'] for row in table} == {'1,4-Dioxane', 'THF', 'Water'}
        assert metrics['curvature'] == 'N/A'
        assert metrics['rank'] == metrics['parameters'] == 12

    def test_two_level_metrics(self, three_cont_features):
        _, metrics = generate_doe(three_cont_features, {'model': 'fullfact'})
        assert metrics['orthogonality'] == metrics['efficiency'] == 100
        assert metrics['rank'] == metrics['parameters'] == 7
        assert metrics['curvature'].startswith('None')

    def test_numeric_three_level_curvature(self):
        features = [{'name': name, 'type': 'discrete', 'range': '0, 1, 2'} for name in ['A', 'B']]
        _, metrics = generate_doe(features, {'model': 'fullfact'})
        assert metrics['curvature'] == 'Excellent'
        assert metrics['rank'] == metrics['parameters'] == 6

    def test_fullfactorial_constraints_filter_combinations(self, three_cont_features):
        table, metrics = generate_doe(three_cont_features, {'model': 'fullfact', 'constraints': {'numerical': ['A < 5']}})
        assert all(row['A'] < 5 for row in table)
        assert metrics['parent_runs'] == 8
        assert metrics['excluded_runs'] == 4

    def test_large_reduced_grid_reports_true_natural_size(self):
        features = [{'name': f'X{i}', 'type': 'categorical', 'range': 'A, B, C'} for i in range(9)]
        table, metrics = generate_doe(features, {'model': 'fullfact', 'max_runs': 5})
        assert len(table) == 5
        assert metrics['natural_runs'] == 3 ** 9
        assert 'sampled candidate pool' in metrics['design_warning']


CLASSICAL_MODELS = ['bbdesign', 'ccdesign', 'fracfact', 'def_screening', 'fullfact']


@pytest.mark.parametrize('model', CLASSICAL_MODELS)
@pytest.mark.parametrize('exact', [False, True])
@pytest.mark.parametrize('runs', [1, 5])
def test_small_classical_budgets_are_unique_and_warn(model, exact, runs, three_cont_features):
    table, metrics = generate_doe(three_cont_features, {'model': model, 'max_runs': runs, 'use_exact_runs': exact})
    assert len(table) == len({tuple(row.values()) for row in table}) == runs
    assert metrics['design_status'] == 'Reduced'
    assert metrics['reduction_warning']
    assert metrics['repeat_runs'] == 0
    if metrics['rank'] < metrics['parameters']:
        assert metrics['efficiency'] == 0
        assert 'not all intended effects' in metrics['design_warning']
    if model in {'fullfact', 'fracfact'}:
        assert metrics['resolution'].startswith('N/A')


@pytest.mark.parametrize('model', CLASSICAL_MODELS)
def test_classical_counts_around_natural_size(model, three_cont_features):
    natural, original = generate_doe(three_cont_features, {'model': model, 'max_runs': 100})
    unique_count = len({tuple(row.values()) for row in natural})
    for exact in [False, True]:
        for count in [unique_count, len(natural) - 1, len(natural), len(natural) + 4]:
            table, metrics = generate_doe(three_cont_features, {'model': model, 'max_runs': count, 'use_exact_runs': exact})
            assert len(table) == (count if exact else min(count, len(natural)))
            keys = [tuple(row.values()) for row in table]
            first_repeat = next((i for i, row in enumerate(keys) if row in keys[:i]), len(keys))
            assert first_repeat == min(len(table), unique_count)
            assert set(keys[first_repeat:]).issubset(keys[:first_repeat])
            assert metrics['repeat_runs'] == len(keys) - first_repeat
            assert metrics['added_replicates'] == (max(0, count - len(natural)) if exact else 0)


@pytest.mark.parametrize('order', ['linear', 'interaction', 'quadratic'])
@pytest.mark.parametrize('runs', [1, 2, 3])
def test_doptimal_accepts_undersized_budgets(order, runs, three_cont_features):
    table, metrics = generate_doe(three_cont_features, {'model': 'doptimal', 'dopt_model_order': order, 'max_runs': runs})
    assert len(table) == len({tuple(row.values()) for row in table}) == runs
    assert metrics['rank'] == runs
    assert metrics['rank'] < metrics['parameters']
    assert metrics['efficiency'] == 0
    assert 'not all intended effects' in metrics['design_warning']


@pytest.mark.parametrize('model', ['lhs', 'doptimal'])
@pytest.mark.parametrize('runs', [1, 5, 6, 10])
def test_exact_designs_exhaust_finite_space_before_repeats(model, runs):
    features = [
        {'name': 'Dose', 'type': 'discrete', 'range': '1, 2, 10'},
        {'name': 'Catalyst', 'type': 'categorical', 'range': 'A, B'},
    ]
    table, metrics = generate_doe(features, {'model': model, 'max_runs': runs})
    keys = [tuple(row.values()) for row in table]
    unique_count = min(runs, 6)
    assert len(table) == runs
    assert len(set(keys[:unique_count])) == unique_count
    assert set(keys[unique_count:]).issubset(keys[:unique_count])
    assert metrics['added_replicates'] == max(0, runs - 6)
    assert metrics['oversampling_warning'] == (runs > 6)


def test_doptimal_exhausts_feasible_combinations_only():
    features = [{'name': name, 'type': 'categorical', 'range': 'A, B, C'} for name in ['X', 'Y']]
    constraints = {'disallowed': [[{'feature': 'X', 'value': 'A'}]]}
    table, metrics = generate_doe(features, {'model': 'doptimal', 'max_runs': 8, 'constraints': constraints})
    assert all(row['X'] != 'A' for row in table)
    assert len({tuple(row.values()) for row in table[:6]}) == 6
    assert table[6:] == table[:2]
    assert metrics['added_replicates'] == 2


def test_doptimal_continuous_grid_is_not_treated_as_exhausted(two_cont_features):
    table, metrics = generate_doe(two_cont_features, {'model': 'doptimal', 'dopt_model_order': 'linear', 'max_runs': 60})
    assert len({tuple(row.values()) for row in table}) == 60
    assert metrics['repeat_runs'] == 0


def test_lhs_single_continuous_run(two_cont_features):
    table, metrics = generate_doe(two_cont_features, {'model': 'lhs', 'max_runs': 1})
    assert len(table) == 1
    assert metrics['efficiency'] == 'N/A'


def test_doptimal_can_find_feasible_points_between_grid_levels(two_cont_features):
    table, _ = generate_doe(two_cont_features, {
        'model': 'doptimal', 'max_runs': 3,
        'constraints': {'numerical': ['X > 0.02', 'X < 0.06']},
    })
    assert len({tuple(row.values()) for row in table}) == 3
    assert all(0.02 < row['X'] < 0.06 for row in table)


def test_doptimal_empty_finite_space_is_not_repeated():
    features = [{'name': name, 'type': 'discrete', 'range': '0, 1'} for name in ['X', 'Y']]
    with pytest.raises(ValueError, match='No candidate design points'):
        generate_doe(features, {'model': 'doptimal', 'max_runs': 2, 'constraints': {'numerical': ['X > 2']}})


ALL_DESIGNS = ['bbdesign', 'ccdesign', 'fracfact', 'fullfact', 'def_screening', 'lhs', 'doptimal']


@pytest.fixture
def constraint_features():
    return [{'name': name, 'type': 'continuous', 'range': '[0, 1]'} for name in 'ABCD']


@pytest.mark.parametrize('model', ALL_DESIGNS)
@pytest.mark.parametrize('rule_kind', ['numerical', 'disallowed', 'both'])
def test_all_designs_enforce_constraints_after_decoding(model, rule_kind, constraint_features):
    from src.core.constraints import check_constraints
    rules = {
        'numerical': ['A <= 0.5'] if rule_kind != 'disallowed' else [],
        'disallowed': [[{'feature': 'B', 'value': '1'}]] if rule_kind != 'numerical' else [],
    }
    table, metrics = generate_doe(constraint_features, {
        'model': model, 'max_runs': 8, 'constraints': rules, 'dopt_model_order': 'linear'
    })
    assert table
    assert all(check_constraints(row, rules['numerical'], rules['disallowed'], constraint_features) for row in table)
    if model != 'doptimal':
        assert metrics['constraint_filtered']
        assert metrics['resolution'].startswith('N/A')
        assert 'original' in metrics['design_warning'].lower()


@pytest.mark.parametrize('model', ['bbdesign', 'ccdesign', 'fracfact', 'fullfact', 'def_screening'])
def test_filtered_exact_count_repeats_only_surviving_parent_rows(model, constraint_features):
    parent, _ = generate_doe(constraint_features, {'model': model, 'max_runs': 100})
    allowed = {tuple(row.values()) for row in parent if row['A'] <= .5}
    table, metrics = generate_doe(constraint_features, {
        'model': model, 'max_runs': 100, 'use_exact_runs': True,
        'constraints': {'numerical': ['A <= 0.5']},
    })
    assert len(table) == 100
    assert {tuple(row.values()) for row in table} <= allowed
    assert metrics['added_replicates'] > 0
    assert metrics['shortfall_runs'] == 0
    assert 'full experimental region' in metrics['design_warning']


@pytest.mark.parametrize('model', ['bbdesign', 'ccdesign', 'fracfact', 'fullfact', 'def_screening'])
def test_filtered_max_count_does_not_top_up(model, constraint_features):
    table, metrics = generate_doe(constraint_features, {
        'model': model, 'max_runs': 100,
        'constraints': {'numerical': ['A <= 0.5']},
    })
    assert len(table) < 100
    assert metrics['added_replicates'] == 0
    assert metrics['shortfall_runs'] == 0


def test_filtering_recomputes_rank_levels_and_center_loss(constraint_features):
    table, metrics = generate_doe(constraint_features, {
        'model': 'bbdesign', 'max_runs': 100,
        'constraints': {'numerical': ['A < 0.5']},
    })
    assert all(row['A'] == 0 for row in table)
    assert metrics['rank'] < metrics['parameters']
    assert metrics['model_estimable'] is False
    assert metrics['efficiency'] == 0
    assert metrics['curvature'] == 'Rank deficient'
    assert metrics['constraint_removed_center_runs'] > 0
    assert metrics['center_replicates'] == 0
    assert '0.5' in metrics['missing_levels']['A']


def test_constraint_is_checked_after_classical_rounding():
    # The centre of [0, .1] is rounded to .1 at the declared one-decimal precision.
    features = [{'name': name, 'type': 'continuous', 'range': '[0, 0.1]'} for name in 'ABC']
    table, _ = generate_doe(features, {
        'model': 'bbdesign', 'constraints': {'numerical': ['A <= 0.05']}
    })
    assert table and all(row['A'] <= .05 for row in table)


@pytest.mark.parametrize('model', ['fullfact', 'fracfact', 'def_screening', 'lhs', 'doptimal'])
def test_mixed_designs_enforce_disallowed_category_pairs(model):
    features = [{'name': name, 'type': 'categorical', 'range': 'A, B'} for name in ['Solvent', 'Catalyst']]
    features += [{'name': name, 'type': 'continuous', 'range': '[0, 1]'} for name in ['X', 'Y']]
    table, _ = generate_doe(features, {
        'model': model, 'max_runs': 8, 'dopt_model_order': 'linear',
        'constraints': {'disallowed': [[{'feature': 'Solvent', 'value': 'A'}, {'feature': 'Catalyst', 'value': 'B'}]]},
    })
    assert table and all(not (row['Solvent'] == 'A' and row['Catalyst'] == 'B') for row in table)


def test_constrained_lhs_finite_shortfall_has_no_repeats():
    features = [{'name': name, 'type': 'discrete', 'range': '0, 1'} for name in 'XY']
    table, metrics = generate_doe(features, {'model': 'lhs', 'max_runs': 8, 'constraints': {'numerical': ['X == 0']}})
    assert len(table) == 2
    assert metrics['repeat_runs'] == 0
    assert metrics['shortfall_runs'] == 6
    assert metrics['candidate_pool_complete'] is True


def test_constrained_lhs_continuous_shortfall_is_explicit(monkeypatch, two_cont_features):
    monkeypatch.setattr('src.core.doe.CONSTRAINED_LHS_BATCH_LIMIT', 1)
    table, metrics = generate_doe(two_cont_features, {'model': 'lhs', 'max_runs': 20, 'constraints': {'numerical': ['X < 0.1']}})
    assert 0 < len(table) < 20
    assert metrics['shortfall_runs'] == 20 - len(table)
    assert metrics['candidate_pool_complete'] is False
    assert metrics['repeat_runs'] == 0


def test_constrained_lhs_is_reproducible(two_cont_features):
    tweaks = {'model': 'lhs', 'max_runs': 12, 'constraints': {'numerical': ['X + Y < 0.7']}}
    first = generate_doe(two_cont_features, tweaks)
    assert first == generate_doe(two_cont_features, tweaks)
    assert len(first[0]) == 12


def test_no_candidates_is_not_claimed_global_infeasibility(monkeypatch, two_cont_features):
    monkeypatch.setattr('src.core.doe.CONSTRAINED_LHS_BATCH_LIMIT', 1)
    with pytest.raises(ValueError, match='not been proven empty'):
        generate_doe(two_cont_features, {'model': 'lhs', 'constraints': {'numerical': ['X == 0.5']}})


def test_empty_parent_has_precise_error(constraint_features):
    with pytest.raises(ValueError, match='parent design'):
        generate_doe(constraint_features, {'model': 'bbdesign', 'constraints': {'numerical': ['A < 0']}})


def test_sampled_factorial_shortfall_does_not_add_repeats(monkeypatch):
    monkeypatch.setattr('src.core.doe.CONSTRAINED_CANDIDATE_LIMIT', 20)
    features = [{'name': name, 'type': 'regular', 'range': '[0, 1]10'} for name in 'XY']
    table, metrics = generate_doe(features, {
        'model': 'fullfact', 'max_runs': 40, 'use_exact_runs': True,
        'constraints': {'numerical': ['X <= 0.5']},
    })
    assert 0 < len(table) < 40
    assert metrics['candidate_pool_complete'] is False
    assert metrics['repeat_runs'] == 0
    assert metrics['shortfall_runs'] == 40 - len(table)


def test_numeric_disallowed_values_match_by_number_not_spelling():
    from src.core.constraints import check_constraints
    features = [{'name': 'X', 'type': 'continuous'}, {'name': 'S', 'type': 'categorical'}]
    assert not check_constraints({'X': 10.0, 'S': '1'}, [], [[{'feature': 'X', 'value': '10'}]], features)
    assert check_constraints({'X': 10.0, 'S': '1'}, [], [[{'feature': 'S', 'value': '1.0'}]], features)


def test_existing_factor_type_restrictions_remain():
    features = [{'name': 'S', 'type': 'categorical', 'range': 'A, B'}]
    features += [{'name': name, 'type': 'continuous', 'range': '[0, 1]'} for name in 'XY']
    with pytest.raises(ValueError, match='numerical factors only'):
        generate_doe(features, {'model': 'bbdesign', 'constraints': {'numerical': ['X < 0.5']}})


def test_metrics_include_the_returned_repeat_rows(three_cont_features):
    _, natural = generate_doe(three_cont_features, {'model': 'fullfact', 'max_runs': 8})
    _, repeated = generate_doe(three_cont_features, {'model': 'fullfact', 'max_runs': 9, 'use_exact_runs': True})
    _, balanced = generate_doe(three_cont_features, {'model': 'fullfact', 'max_runs': 16, 'use_exact_runs': True})
    assert natural['efficiency'] == balanced['efficiency'] == 100
    assert repeated['efficiency'] < natural['efficiency']
    assert repeated['orthogonality'] < natural['orthogonality']
