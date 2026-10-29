/** CEDOS Auto adapters for the two manual optimizer request contracts. */

import {
    parseCategoricalChoices,
    uniqueCategoricalChoices,
    submitAndPollJob
} from '../utils.js';

const value = (doc, id, fallback) => doc.getElementById(id)?.value ?? fallback;
const checked = (doc, id, fallback = false) => doc.getElementById(id)?.checked ?? fallback;

function selectedColumns(win, role, configs) {
    return (win.currentColumns || []).filter(name =>
        win.columnRoles?.[name] === role && configs?.[name]?.included !== false
    );
}

function requireTrainingRows(win, featureNames, objectiveNames) {
    if (!Array.isArray(win.currentData) || !win.currentData.length) {
        throw new Error('Load measured experimental data before starting Auto optimization.');
    }
    const columns = win.currentColumns || [];
    const required = [...featureNames, ...objectiveNames];
    if (required.some(name => !columns.includes(name))) {
        throw new Error('The selected optimizer setup contains columns missing from the dataset.');
    }
    const indices = required.map(name => columns.indexOf(name));
    const complete = win.currentData.filter(row => indices.every(index =>
        row[index] !== null && row[index] !== undefined && String(row[index]).trim() !== ''
    ));
    if (!complete.length) {
        throw new Error('No complete measured rows are available for the selected optimizer.');
    }
}

function collectNumerical(doc, selector) {
    return Array.from(doc.querySelectorAll(selector))
        .map(row => row.querySelector('.constraint-equation')?.value?.trim())
        .filter(Boolean);
}

function collectDisallowed(doc, selector) {
    return Array.from(doc.querySelectorAll(selector))
        .map(row => Array.from(row.querySelectorAll('.condition-item')).map(item => ({
            feature: item.querySelector('.condition-feature')?.value,
            value: item.querySelector('.condition-value')?.value
        })))
        .filter(conditions => conditions.length > 0);
}

function requireSelections(features, objectives) {
    if (!features.length) throw new Error('Include at least one feature in the active optimizer setup.');
    if (!objectives.length) throw new Error('Include at least one objective in the active optimizer setup.');
    for (const objective of objectives) {
        if (!objective.name) throw new Error('An included objective has no dataset column name.');
        if (objective.type === 'target'
            && (objective.target === '' || objective.target === null
                || objective.target === undefined || !Number.isFinite(Number(objective.target)))) {
            throw new Error(`Set a finite target value for objective ${objective.name}.`);
        }
    }
}

/** Read the same live controls used by manual BO, including Setup CSV changes. */
export function buildAutoBOPayload(win = window, doc = document) {
    let features = Array.from(doc.querySelectorAll('#bo-features-list tr'))
        .filter(row => row.querySelector('.bo-f-include')?.checked)
        .map(row => {
            const typeInput = row.querySelector('.bo-f-type');
            const name = typeInput?.dataset.col;
            const type = typeInput?.value || 'continuous';
            const range = row.querySelector('.bo-f-range')?.value || '';
            const feature = { name, type, range };
            if (type === 'categorical') feature.choices = parseCategoricalChoices(range);
            return feature;
        });
    let objectives = Array.from(doc.querySelectorAll('#bo-objectives-list tr'))
        .filter(row => row.querySelector('.bo-o-include')?.checked)
        .map(row => ({
            name: row.querySelector('.bo-o-type')?.dataset.col,
            type: row.querySelector('.bo-o-type')?.value || 'maximize',
            target: row.querySelector('.bo-o-target')?.value || '',
            importance: row.querySelector('.bo-o-importance')?.value || '1'
        }));

    // The state fallback is needed immediately after a valid Setup CSV import,
    // before a hidden BO panel has been rendered for the first time.
    if (!features.length) features = selectedColumns(win, 'feature', win.boFeatureConfigs).map(name => {
        const cfg = win.boFeatureConfigs[name] || {};
        const type = cfg.type || 'continuous';
        const range = cfg.range || '';
        const feature = { name, type, range };
        if (type === 'categorical') feature.choices = Array.isArray(cfg.choices)
            ? cfg.choices : parseCategoricalChoices(range);
        return feature;
    });
    if (!objectives.length) objectives = selectedColumns(win, 'objective', win.objectiveConfigs).map(name => {
        const cfg = win.objectiveConfigs[name] || {};
        return {
            name, type: cfg.type || 'maximize', target: cfg.target ?? '',
            importance: cfg.importance ?? 1
        };
    });
    requireSelections(features, objectives);
    const featureNames = features.map(item => item.name);
    const objectiveNames = objectives.map(item => item.name);
    requireTrainingRows(win, featureNames, objectiveNames);
    const objectiveConstraints = win.collectObjectiveConstraints?.('bo') || [];
    const tweaks = {
        batch_size: value(doc, 'batch-size', '1'),
        acq_type: value(doc, 'acq-type-select', 'EI'),
        multi_objective_method: objectives.length > 1
            ? value(doc, 'bo-mo-method-select', 'auto') : 'auto',
        data_mode: value(doc, 'bo-data-mode-select', 'auto'),
        kernel: value(doc, 'kernel-select', 'matern52'),
        exploration: value(doc, 'style-slider', '0.5'),
        noiseless: checked(doc, 'bo-noiseless'),
        avoid_reval: checked(doc, 'bo-avoid-reval', true),
        optimization_strategy: doc.querySelector('input[name="bo-strategy"]:checked')?.value
            || 'gradient_based',
        constraints: {
            numerical: collectNumerical(doc, '#bo-numerical-constraints-list tr.constraint-row'),
            disallowed: collectDisallowed(doc, '#bo-disallowed-constraints-list tr.disallowed-row')
        },
        objective_constraints: objectiveConstraints
    };
    return {
        endpoint: '/bo_submit', featureNames, objectiveNames,
        priorDefaults: {}, warnings: [],
        payload: {
            data: win.currentData, columns: win.currentColumns,
            features, objectives, tweaks, compute_performance: false
        }
    };
}

function priorNumericalTolerance(win, name) {
    const index = win.currentColumns.indexOf(name);
    const values = win.currentData.map(row => Number(row[index])).filter(Number.isFinite);
    return values.length ? (Math.max(...values) - Math.min(...values)) / 2 : 0;
}

/** Preserve TransBO's transfer payload and prior targets while using its async endpoint. */
export function buildAutoTBOPayload(win = window, doc = document) {
    const featureNames = (win.currentColumns || []).filter(name =>
        win.columnRoles?.[name] === 'feature' && win.boFeatureConfigs?.[name]?.included);
    const objectiveNames = (win.currentColumns || []).filter(name =>
        win.columnRoles?.[name] === 'objective' && win.objectiveConfigs?.[name]?.included);
    const priorNames = (win.currentColumns || []).filter(name => win.columnRoles?.[name] === 'prior');
    if (!featureNames.length) throw new Error('Include at least one feature in the active optimizer setup.');
    if (!objectiveNames.length) throw new Error('Include at least one objective in the active optimizer setup.');
    requireTrainingRows(win, featureNames, objectiveNames);

    const numerical = collectNumerical(doc, '#tbo-numerical-constraints-list tr');
    const disallowed = collectDisallowed(doc, '#tbo-disallowed-constraints-list tr');
    if (numerical.length || disallowed.length) {
        throw new Error('Auto TransBO cannot enforce feature constraints reliably. Remove them before dispatch.');
    }
    const features = featureNames.map(name => {
        const cfg = win.boFeatureConfigs[name] || {};
        const type = cfg.type || 'continuous';
        const range = cfg.range || '';
        const feature = { name, type, range };
        if (type === 'categorical') feature.choices = Array.isArray(cfg.choices)
            ? uniqueCategoricalChoices(cfg.choices) : parseCategoricalChoices(range);
        return feature;
    });
    const objectives = objectiveNames.map(name => {
        const cfg = win.objectiveConfigs[name] || {};
        return {
            name, type: cfg.type, target: cfg.target,
            importance: cfg.importance
        };
    });
    requireSelections(features, objectives);
    const priors = priorNames.map(name => {
        const cfg = win.priorConfigs?.[name] || {};
        if (cfg.target_value === null || cfg.target_value === undefined
            || String(cfg.target_value).trim() === '') {
            throw new Error(`Set a Prior Target for ${name} before starting Auto TransBO.`);
        }
        const prior = {
            name, col_type: cfg.col_type || 'categorical',
            target_value: cfg.target_value,
            prior_weight: Number(cfg.prior_weight ?? 100)
        };
        if (prior.col_type === 'numerical') {
            const tolerance = cfg.tolerance === null || cfg.tolerance === undefined
                || cfg.tolerance === '' || !Number.isFinite(Number(cfg.tolerance))
                ? priorNumericalTolerance(win, name) : Number(cfg.tolerance);
            prior.tolerance = Math.max(0, tolerance || 0);
        }
        if (prior.col_type === 'smiles') {
            const similarity = cfg.min_similarity === null || cfg.min_similarity === undefined
                || cfg.min_similarity === '' || !Number.isFinite(Number(cfg.min_similarity))
                ? 0.5 : Number(cfg.min_similarity);
            prior.min_similarity = Math.min(1, Math.max(0, similarity));
        }
        return prior;
    });
    const objectiveConstraints = win.collectObjectiveConstraints?.('tbo') || [];
    const tweaks = {
        batch_size: parseInt(value(doc, 'tbo-batch-size', '1'), 10),
        kernel: value(doc, 'tbo-kernel-select', 'matern52'),
        acq_type: value(doc, 'tbo-acq-select', 'EI'),
        exploration: parseFloat(value(doc, 'tbo-exploration', '0.5')),
        noiseless: checked(doc, 'tbo-noiseless-toggle'),
        avoid_reval: false,
        max_gp_rows: parseInt(value(doc, 'tbo-prior-cap-slider', '3'), 10),
        allow_reduced_row_recovery: checked(doc, 'tbo-reduced-row-recovery-toggle'),
        constraints: { numerical: [], disallowed: [] },
        objective_constraints: objectiveConstraints
    };
    const warnings = [
        'TransBO Avoid Re-evaluation is unavailable in CEDOS until the fitted transfer GP path enforces it.'
    ];
    if (!priors.length) warnings.push(
        'No Prior Target is configured. TransBO will use its standard BO fallback and may cap training rows.'
    );
    return {
        endpoint: '/transfer_optimize_submit', featureNames, objectiveNames,
        priorDefaults: Object.fromEntries(priors.map(prior => [prior.name, prior.target_value])),
        warnings,
        payload: {
            data: win.currentData, columns: win.currentColumns,
            features, objectives, priors,
            prior_combine: win.priorCombineRule || 'multiply', tweaks
        }
    };
}

export function buildAutoOptimizerPayload(optimizer, win = window, doc = document) {
    if (optimizer === 'bo') return buildAutoBOPayload(win, doc);
    if (optimizer === 'tbo') return buildAutoTBOPayload(win, doc);
    throw new Error(`Unknown optimizer: ${optimizer}`);
}

export async function submitAutoOptimization(adapter, options) {
    const completed = await submitAndPollJob(adapter.endpoint, adapter.payload, options);
    const result = completed.result || {};
    if (!result.exhausted) {
        if (!Array.isArray(result.suggestions) || !result.suggestions.length) {
            throw new Error('The optimizer returned no suggestions and no exhaustion reason.');
        }
        for (const suggestion of result.suggestions) {
            for (const name of adapter.featureNames) {
                if (suggestion[name] === undefined || suggestion[name] === null
                    || String(suggestion[name]).trim() === '') {
                    throw new Error(`The optimizer omitted feature ${name} from a suggestion.`);
                }
            }
        }
    }
    return result;
}
