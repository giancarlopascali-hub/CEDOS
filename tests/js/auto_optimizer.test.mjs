import test from 'node:test';
import assert from 'node:assert/strict';
globalThis.window = {};
const { buildAutoBOPayload, buildAutoTBOPayload } =
    await import('../../static/js/modules/auto_optimizer.js');

const input = (value, extra = {}) => ({ value, ...extra });
const row = fields => ({ querySelector: selector => fields[selector] || null,
    querySelectorAll: selector => fields[selector] || [] });
const doc = (ids, selectors) => ({
    getElementById: id => ids[id] || null,
    querySelector: selector => selectors[selector] || null,
    querySelectorAll: selector => selectors[selector] || []
});

test('Auto BO serializes the manual BO controls, constraints and workload choices', () => {
    const win = {
        currentColumns: ['x', 'y'], currentData: [[1, 5]],
        columnRoles: { x: 'feature', y: 'objective' },
        boFeatureConfigs: { x: { type: 'continuous', range: '0,10', included: true } },
        objectiveConfigs: { y: { type: 'maximize', included: true } },
        collectObjectiveConstraints: module => module === 'bo' ? [{ objective: 'y', operator: '>=', value: 3 }] : []
    };
    const featureRow = row({
        '.bo-f-include': { checked: true },
        '.bo-f-type': input('continuous', { dataset: { col: 'x' } }),
        '.bo-f-range': input('0,10')
    });
    const objectiveRow = row({
        '.bo-o-include': { checked: true },
        '.bo-o-type': input('maximize', { dataset: { col: 'y' } }),
        '.bo-o-target': input(''), '.bo-o-importance': input('100')
    });
    const constraintRow = row({ '.constraint-equation': input('x <= 9') });
    const disallowedRow = row({ '.condition-item': [row({
        '.condition-feature': input('x'), '.condition-value': input('8')
    })] });
    const browser = doc({
        'batch-size': input('2'), 'acq-type-select': input('EI'),
        'bo-mo-method-select': input('auto'), 'bo-data-mode-select': input('representative'),
        'kernel-select': input('matern52'), 'style-slider': input('0.7'),
        'bo-noiseless': { checked: true }, 'bo-avoid-reval': { checked: true }
    }, {
        '#bo-features-list tr': [featureRow], '#bo-objectives-list tr': [objectiveRow],
        '#bo-numerical-constraints-list tr.constraint-row': [constraintRow],
        '#bo-disallowed-constraints-list tr.disallowed-row': [disallowedRow],
        'input[name="bo-strategy"]:checked': input('gradient_based')
    });
    const adapter = buildAutoBOPayload(win, browser);
    assert.equal(adapter.endpoint, '/bo_submit');
    assert.deepEqual(adapter.payload, {
        data: win.currentData, columns: win.currentColumns,
        features: [{ name: 'x', type: 'continuous', range: '0,10' }],
        objectives: [{ name: 'y', type: 'maximize', target: '', importance: '100' }],
        tweaks: {
            batch_size: '2', acq_type: 'EI', multi_objective_method: 'auto',
            data_mode: 'representative', kernel: 'matern52', exploration: '0.7',
            noiseless: true, avoid_reval: true, optimization_strategy: 'gradient_based',
            constraints: {
                numerical: ['x <= 9'],
                disallowed: [[{ feature: 'x', value: '8' }]]
            },
            objective_constraints: [{ objective: 'y', operator: '>=', value: 3 }]
        },
        compute_performance: false
    });
});

test('Auto TransBO mirrors supported manual transfer settings and retains priors', () => {
    const win = {
        currentColumns: ['x', 'source', 'y'], currentData: [[1, 'A', 5]],
        columnRoles: { x: 'feature', source: 'prior', y: 'objective' },
        boFeatureConfigs: { x: { type: 'continuous', range: '0,10', included: true } },
        objectiveConfigs: { y: { type: 'maximize', target: '', importance: 100, included: true } },
        priorConfigs: { source: { col_type: 'categorical', target_value: 'B', prior_weight: 100 } },
        priorCombineRule: 'multiply',
        collectObjectiveConstraints: module => module === 'tbo' ? [] : null
    };
    const browser = doc({
        'tbo-batch-size': input('2'), 'tbo-kernel-select': input('matern52'),
        'tbo-acq-select': input('LCB'), 'tbo-exploration': input('0.5'),
        'tbo-noiseless-toggle': { checked: true },
        'tbo-avoid-reval-toggle': { checked: true },
        'tbo-prior-cap-slider': input('300'),
        'tbo-reduced-row-recovery-toggle': { checked: false }
    }, {
        '#tbo-numerical-constraints-list tr': [],
        '#tbo-disallowed-constraints-list tr': []
    });
    const adapter = buildAutoTBOPayload(win, browser);
    assert.equal(adapter.endpoint, '/transfer_optimize_submit');
    assert.deepEqual(adapter.priorDefaults, { source: 'B' });
    assert.deepEqual(adapter.payload.priors, [{
        name: 'source', col_type: 'categorical', target_value: 'B', prior_weight: 100
    }]);
    assert.deepEqual(adapter.payload.tweaks, {
        batch_size: 2, kernel: 'matern52', acq_type: 'LCB', exploration: 0.5,
        noiseless: true, avoid_reval: false, max_gp_rows: 300,
        allow_reduced_row_recovery: false,
        constraints: { numerical: [], disallowed: [] }, objective_constraints: []
    });
    const constrained = doc({}, {
        '#tbo-numerical-constraints-list tr': [row({ '.constraint-equation': input('x < 5') })],
        '#tbo-disallowed-constraints-list tr': []
    });
    assert.throws(() => buildAutoTBOPayload(win, constrained), /cannot enforce feature constraints/);
});
