// TransEDOS - Transfer Learning Bayesian Optimization (TransBO) Module
// ======================================================================
// Manages the 4-tab interface (Features, Prior Targets, Objectives, Tweaks),
// prior column validation, live eligibility previews, API communication with
// /transfer_optimize and /transfer_surface, and results visualization.

import {
    safeAddListener,
    exportData,
    rebalanceRemaining,
    rebalanceObjectiveConfigs,
    updateRangePlaceholder,
    wireSelectAll,
    submitAndPollJob,
    cancelBackgroundJob,
    escapeHtmlAttribute,
    formatCategoricalChoices,
    parseCategoricalChoices,
    uniqueCategoricalChoices
} from '../utils.js';
import { updateAllConstraintsDropdowns } from './constraints.js';

/**
 * Safely parses a fetch Response as JSON.
 * If the server returns a non-JSON body (e.g. an HTML error page from a 413/500),
 * this returns { error: <status text> } instead of throwing a SyntaxError.
 */
async function safeJson(response) {
    const text = await response.text();
    try {
        return JSON.parse(text);
    } catch (_) {
        // The server returned something non-JSON (e.g. an HTML error page).
        // Surface a readable message based on the HTTP status.
        const status = response.status;
        if (status === 413) {
            return { error: 'The dataset is too large for the server to process in a single request. Try reducing the number of rows or columns.' };
        }
        return { error: `Server error (HTTP ${status}): the server returned an unexpected response. Check the server console for details.` };
    }
}

// Cache of latest results and transfer diagnostics
let latestTBOResults = null;
let activeTBOOptimizationJobId = null;
let activeTBOEstimateJobId = null;
let activeTBOSurfaceJobId = null;
let activeTBOSurfaceCancellation = null;
let tboSelectionPreview = null;
let tboSelectionPreviewTimer = null;
let tboSelectionPreviewController = null;
let reusableTBOTraining = null;

function getPriorColumnValues(column) {
    const index = (window.currentColumns || []).indexOf(column);
    if (index < 0 || !Array.isArray(window.currentData)) return [];
    return window.currentData
        .map(row => row[index])
        .filter(value => value !== null && value !== undefined && String(value).trim() !== '');
}

function numericalPriorScale(column) {
    const rawValues = getPriorColumnValues(column);
    const values = rawValues.map(Number).filter(Number.isFinite);
    const minimum = values.length ? Math.min(...values) : 0;
    const maximum = values.length ? Math.max(...values) : 1;
    const span = Math.max(0, maximum - minimum);
    const precision = rawValues.reduce((highest, value) => {
        const text = String(value);
        const decimal = text.includes('.') ? text.split('.')[1].replace(/0+$/, '').length : 0;
        return Math.max(highest, Math.min(decimal, 8));
    }, 0);
    const precisionStep = 10 ** (-precision);
    const step = precisionStep;
    return { values, minimum, maximum, span, step };
}

function ensurePriorDefaults(column, config) {
    if (
        config.col_type === 'numerical'
        && (config.tolerance === null || config.tolerance === '' || !Number.isFinite(Number(config.tolerance)))
    ) {
        config.tolerance = numericalPriorScale(column).span / 2;
    }
    if (
        config.col_type === 'smiles'
        && (config.min_similarity === null || config.min_similarity === '' || !Number.isFinite(Number(config.min_similarity)))
    ) {
        config.min_similarity = 0.5;
    }
    return config;
}

function serializePriorConfig(column) {
    const config = ensurePriorDefaults(column, window.priorConfigs[column] || {});
    const prior = {
        name: column,
        col_type: config.col_type || 'categorical',
        target_value: config.target_value ?? '',
        prior_weight: Number(config.prior_weight ?? 100.0)
    };
    if (prior.col_type === 'numerical') {
        prior.tolerance = Math.max(0, Number(config.tolerance) || 0);
    } else if (prior.col_type === 'smiles') {
        prior.min_similarity = Math.min(1, Math.max(0, Number(config.min_similarity ?? 0.5)));
    }
    return prior;
}

function buildPriorPayload(columns) {
    return columns.map(serializePriorConfig);
}

function buildTBOFeaturePayload(name, config = {}) {
    const type = config.type || 'continuous';
    const range = config.range || '';
    const feature = { name, type, range };
    if (type === 'categorical') {
        feature.choices = Array.isArray(config.choices)
            ? uniqueCategoricalChoices(config.choices)
            : parseCategoricalChoices(range);
    }
    return feature;
}

function tboTrainingContextKey({ features, objectives, priors, priorCombine, tweaks }) {
    const textValue = value => value === null || value === undefined ? '' : String(value).trim();
    return JSON.stringify({
        features: features.map(({ name, type, range, choices }) => ({
            name, type, range: textValue(range), choices: choices || null
        })),
        objectives: objectives.map(({ name, type, target, importance }) => ({
            name, type, target: textValue(target), importance: Number(importance)
        })),
        priors: priors.map(({ name, col_type, target_value, tolerance, min_similarity, prior_weight }) => ({
            name, col_type, target_value: textValue(target_value),
            tolerance: tolerance === undefined ? null : Number(tolerance),
            min_similarity: min_similarity === undefined ? null : Number(min_similarity),
            prior_weight: Number(prior_weight)
        })),
        priorCombine,
        maxGPRows: tweaks.max_gp_rows,
        objectiveConstraints: tweaks.objective_constraints || []
    });
}

function clearTBOResultsForNewRun() {
    latestTBOResults = null;
    reusableTBOTraining = null;
    window.transferDiagnostics = null;
    document.getElementById('tbo-results-section')?.classList.add('hidden');
    [
        'tbo-results-head',
        'tbo-results-body',
        'tbo-results-legend',
        'tbo-diagnostics-panel',
        'tbo-performance-container',
        'tbo-estimate-results'
    ].forEach(id => {
        const element = document.getElementById(id);
        if (element) element.replaceChildren();
    });
    document.getElementById('tbo-diagnostics-section')?.classList.add('hidden');
    document.getElementById('tbo-cold-start-banner')?.classList.add('hidden');
    document.getElementById('tbo-feasibility-warning')?.remove();
    document.getElementById('tbo-exhaustion-banner')?.remove();
    const commitButton = document.getElementById('tbo-commit-suggestions-btn');
    if (commitButton) {
        commitButton.disabled = true;
        commitButton.style.opacity = '0.45';
    }
}

// Setup edits invalidate suggestions without cancelling the independent,
// delayed prior-selection preview that follows those same edits.
export function invalidateTBOResults() {
    clearTBOResultsForNewRun();
}

export function resetTBOState() {
    clearTBOResultsForNewRun();
    clearTimeout(tboSelectionPreviewTimer);
    tboSelectionPreviewController?.abort();
    tboSelectionPreviewController = null;
    tboSelectionPreview = null;
    cancelBackgroundJob(activeTBOOptimizationJobId);
    cancelBackgroundJob(activeTBOEstimateJobId);
    cancelActiveTBOPlot();
    activeTBOOptimizationJobId = null;
    activeTBOEstimateJobId = null;
}
let activeContourData = null;
let activeSurfaceData = null;

export function renderTBOSetup() {
    renderTBOFeaturesTab();
    renderTBOPriorsTab();
    renderTBOObjectivesTab();
    renderTBOTweaksTab();
    updateTBOExplorationControlAvailability();
    renderTBOEstimator();
    initTBOPriorCapSlider();
    initTBOTabs();
    initTBOButtons();
}

/**
 * Tab 1: Features Setup
 */
export function renderTBOFeaturesTab() {
    const list = document.getElementById('tbo-features-list');
    if (!list) return;

    list.innerHTML = '';
    const featureCols = window.currentColumns.filter(c => window.columnRoles[c] === 'feature');

    if (featureCols.length === 0) {
        list.innerHTML = `
            <tr>
                <td colspan="4" style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                    No feature columns selected. Toggle columns to 'F' in the table above.
                </td>
            </tr>`;
        return;
    }

    featureCols.forEach(col => {
        if (!window.boFeatureConfigs[col]) {
            const idx = window.currentColumns.indexOf(col);
            const vals = window.currentData ? window.currentData.map(r => r[idx]).filter(v => v != null && v !== '') : [];
            const isNum = vals.length > 0 && vals.every(v => !isNaN(parseFloat(v)));
            const min = vals.length ? Math.min(...vals.map(v => parseFloat(v))) : 0;
            const max = vals.length ? Math.max(...vals.map(v => parseFloat(v))) : 1;
            const choices = isNum ? undefined : uniqueCategoricalChoices(vals);
            const range = isNum ? `[${min}, ${max}]` : formatCategoricalChoices(choices);

            window.boFeatureConfigs[col] = {
                type: isNum ? 'continuous' : 'categorical',
                range: range,
                choices,
                included: true
            };
        }

        const cfg = window.boFeatureConfigs[col];
        const row = document.createElement('tr');
        if (!cfg.included) row.classList.add('row-disabled');

        row.innerHTML = `
            <td style="font-weight: 500;">${col}</td>
            <td>
                <select class="tbo-feat-type" data-col="${col}">
                    <option value="continuous" ${cfg.type === 'continuous' ? 'selected' : ''}>Continuous</option>
                    <option value="discrete"   ${cfg.type === 'discrete' ? 'selected' : ''}>Discrete</option>
                    <option value="regular"    ${cfg.type === 'regular' ? 'selected' : ''}>Regular</option>
                    <option value="categorical" ${cfg.type === 'categorical' ? 'selected' : ''}>Categorical</option>
                </select>
            </td>
            <td>
                <input type="text" class="tbo-feat-range" data-col="${col}" value="${escapeHtmlAttribute(cfg.range || '')}">
            </td>
            <td style="text-align: center;">
                <input type="checkbox" class="tbo-feat-inc" data-col="${col}" ${cfg.included ? 'checked' : ''}>
            </td>
        `;

        const rangeInput = row.querySelector('.tbo-feat-range');
        updateRangePlaceholder(cfg.type, rangeInput);

        row.querySelector('.tbo-feat-type').addEventListener('change', (e) => {
            cfg.type = e.target.value;
            if (cfg.type === 'categorical') {
                try { cfg.choices = parseCategoricalChoices(rangeInput.value); }
                catch (_) { delete cfg.choices; }
            }
            updateRangePlaceholder(cfg.type, rangeInput);
            updateAllConstraintsDropdowns();
            scheduleTBOSelectionPreview();
        });
        row.querySelector('.tbo-feat-range').addEventListener('input', (e) => {
            cfg.range = e.target.value;
            if (cfg.type === 'categorical') {
                try { cfg.choices = parseCategoricalChoices(cfg.range); }
                catch (_) { delete cfg.choices; }
            }
        });
        row.querySelector('.tbo-feat-inc').addEventListener('change', (e) => {
            cfg.included = e.target.checked;
            row.classList.toggle('row-disabled', !cfg.included);
            updateAllConstraintsDropdowns();
            renderTBOEstimator();
            scheduleTBOSelectionPreview();
        });

        list.appendChild(row);
    });

    updateAllConstraintsDropdowns();

    wireSelectAll(
        'tbo-select-all-features',
        '#tbo-features-list .tbo-feat-inc',
        (columns, checked) => {
            columns.forEach(col => {
                if (!window.boFeatureConfigs[col]) return;
                window.boFeatureConfigs[col].included = checked;
                document.querySelector(`#tbo-features-list .tbo-feat-inc[data-col="${col}"]`)
                    ?.closest('tr')?.classList.toggle('row-disabled', !checked);
            });
            updateAllConstraintsDropdowns();
            renderTBOEstimator();
            scheduleTBOSelectionPreview();
        }
    );
}

/**
 * Prior Cap Slider & Hardware Time Estimator
 */
export function initTBOPriorCapSlider() {
    const slider = document.getElementById('tbo-prior-cap-slider');
    const valEl = document.getElementById('tbo-prior-cap-val');
    const totalEl = document.getElementById('tbo-prior-cap-total');
    if (!slider || !valEl) return;

    const totalRows = window.currentData?.length || 0;
    if (totalEl) totalEl.textContent = totalRows;

    if (!slider.dataset.initialized) {
        slider.dataset.initialized = 'true';
        slider.addEventListener('input', () => {
            valEl.textContent = slider.value;
            updateTBOSelectionSummary();
            updateTBOTimeEstimate();
        });

        ['tbo-hw-cpu', 'tbo-hw-ram', 'tbo-hw-gpu'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', updateTBOTimeEstimate);
        });
        ['tbo-kernel-select', 'tbo-batch-size', 'tbo-acq-select'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', updateTBOTimeEstimate);
        });
    }

    if (totalRows < 3) {
        tboSelectionPreview = null;
        slider.disabled = true;
        valEl.textContent = '—';
        const summary = document.getElementById('tbo-selection-summary');
        if (summary) summary.textContent = totalRows
            ? 'At least 3 dataset rows are required to fit a Gaussian Process.'
            : 'Load a dataset and assign Feature, Objective, and Prior columns to calculate eligible rows.';
        const badge = document.getElementById('tbo-prior-cap-badge');
        if (badge) badge.textContent = 'Load data to estimate compute time';
        return;
    }

    slider.disabled = false;
    const oldMaximum = Number(slider.max || 0);
    const wasAtMaximum = Number(slider.value) >= oldMaximum;
    slider.min = 3;
    slider.max = totalRows;
    slider.step = 1;
    if (wasAtMaximum || Number(slider.value) > totalRows || Number(slider.value) < 3) {
        slider.value = totalRows;
    }
    valEl.textContent = slider.value;
    const maxLabel = document.getElementById('tbo-prior-cap-max-label');
    if (maxLabel) maxLabel.textContent = `Full dataset (${totalRows.toLocaleString()} rows)`;
    updateTBOTimeEstimate();
    scheduleTBOSelectionPreview();
}

function getTBOPreviewObjectiveConstraints() {
    return Array.from(document.querySelectorAll('#tbo-objective-constraints-list .objective-constraint-row'))
        .map(row => {
            const rawValue = row.querySelector('.objective-constraint-value')?.value;
            return {
                objective: row.querySelector('.objective-constraint-objective')?.value || '',
                operator: row.querySelector('.objective-constraint-operator')?.value || '',
                value: rawValue === '' ? NaN : Number(rawValue)
            };
        })
        .filter(item => item.objective && ['>=', '<='].includes(item.operator) && Number.isFinite(item.value));
}

function buildTBOSelectionPreviewPayload() {
    const featureCols = (window.currentColumns || []).filter(c =>
        window.columnRoles[c] === 'feature' && window.boFeatureConfigs?.[c]?.included !== false
    );
    const objectiveCols = (window.currentColumns || []).filter(c =>
        window.columnRoles[c] === 'objective' && window.objectiveConfigs?.[c]?.included !== false
    );
    const priorCols = (window.currentColumns || []).filter(c => window.columnRoles[c] === 'prior');
    return {
        data: window.currentData || [],
        columns: window.currentColumns || [],
        features: featureCols.map(name => ({ name, ...window.boFeatureConfigs[name] })),
        objectives: objectiveCols.map(name => ({ name, ...window.objectiveConfigs[name] })),
        priors: buildPriorPayload(priorCols),
        prior_combine: window.priorCombineRule || 'multiply',
        tweaks: {
            max_gp_rows: Number(document.getElementById('tbo-prior-cap-slider')?.value || 1),
            allow_reduced_row_recovery: document.getElementById('tbo-reduced-row-recovery-toggle')?.checked || false,
            objective_constraints: getTBOPreviewObjectiveConstraints(),
            constraints: { numerical: [], disallowed: [] }
        }
    };
}

function updateTBOSelectionSummary() {
    const summary = document.getElementById('tbo-selection-summary');
    const slider = document.getElementById('tbo-prior-cap-slider');
    if (!summary || !slider || !tboSelectionPreview) return;
    const requested = Number(slider.value);
    const rowsToFit = Math.min(requested, Number(tboSelectionPreview.eligible_rows || 0));
    const validRows = Number(tboSelectionPreview.valid_rows || 0);
    const availableRows = Number(tboSelectionPreview.available_rows || validRows);
    const eligibleRows = Number(tboSelectionPreview.eligible_rows || 0);
    const invalidNote = availableRows > validRows
        ? ` ${availableRows - validRows} rows are excluded because required feature/objective values are invalid.`
        : '';
    const workloadWarning = rowsToFit >= 1500
        ? `<div style="margin-top:0.55rem;color:#991b1b;font-weight:600;">⚠ High exact-GP workload: this run may take a long time and can exceed the memory or runtime available on a hosted server. The estimate is informative only; this selection is not blocked.</div>`
        : rowsToFit >= 750
            ? `<div style="margin-top:0.55rem;color:#9a6700;">⚠ Large exact-GP workload: allow extra time and ensure the selected hardware has sufficient memory.</div>`
            : '';
    const combineDescription = (window.priorCombineRule || 'multiply') === 'multiply'
        ? 'satisfy all Prior rules using Multiply'
        : 'satisfy at least one active Prior rule using Average';
    if (eligibleRows < 3) {
        summary.innerHTML = `<b>${eligibleRows.toLocaleString()}</b> of <b>${validRows.toLocaleString()}</b> valid rows ${combineDescription}. Too few rows are available for GP fitting; broaden a numerical tolerance or SMILES similarity range, or change the combination rule.${invalidNote}`;
    } else {
        summary.innerHTML = `<b>${eligibleRows.toLocaleString()}</b> of <b>${validRows.toLocaleString()}</b> valid rows ${combineDescription}; the <b>${rowsToFit.toLocaleString()}</b> highest-ranked rows will train the GP.${invalidNote}${workloadWarning}`;
    }
}

function updatePriorEligibilityHints(preview) {
    const stats = preview?.prior_stats || {};
    Object.entries(stats).forEach(([column, values]) => {
        const hint = document.querySelector(`.tbo-live-effective-hint[data-col="${CSS.escape(column)}"]`);
        if (!hint) return;
        hint.innerHTML = `<b>${Number(values.n_eligible || 0).toLocaleString()}</b> of <b>${Number(values.total_rows || 0).toLocaleString()}</b> valid rows satisfy this Prior rule.`;
    });
}

export function scheduleTBOSelectionPreview(delay = 350) {
    clearTimeout(tboSelectionPreviewTimer);
    tboSelectionPreviewTimer = setTimeout(refreshTBOSelectionPreview, delay);
}

async function refreshTBOSelectionPreview() {
    const summary = document.getElementById('tbo-selection-summary');
    const slider = document.getElementById('tbo-prior-cap-slider');
    if (!slider || !window.currentData?.length) {
        if (summary) summary.textContent = 'Load a dataset and assign Feature, Objective, and Prior columns to calculate eligible rows.';
        return;
    }
    const payload = buildTBOSelectionPreviewPayload();
    if (!payload.features.length || !payload.objectives.length) {
        if (summary) summary.textContent = 'Assign at least one included Feature and one included Objective to calculate the GP capacity and eligible rows.';
        return;
    }
    tboSelectionPreviewController?.abort();
    const controller = new AbortController();
    tboSelectionPreviewController = controller;
    if (summary) summary.textContent = 'Calculating valid and relevance-eligible prior rows…';
    try {
        const response = await fetch('/transfer_selection_preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal
        });
        const preview = await safeJson(response);
        if (!response.ok || preview.error) throw new Error(preview.error || 'Capacity preview failed.');
        const totalRows = Math.max(3, Number(preview.available_rows || 0));
        const oldMax = Number(slider.max || 0);
        const wasAtMaximum = Number(slider.value) >= oldMax;
        slider.min = 3;
        slider.max = totalRows;
        slider.step = 1;
        if (wasAtMaximum || Number(slider.value) > totalRows) slider.value = totalRows;
        document.getElementById('tbo-prior-cap-val').textContent = slider.value;
        document.getElementById('tbo-prior-cap-total').textContent = Number(preview.available_rows || 0).toLocaleString();
        const maxLabel = document.getElementById('tbo-prior-cap-max-label');
        if (maxLabel) maxLabel.textContent = `Full dataset (${Number(preview.available_rows || 0).toLocaleString()} rows)`;
        tboSelectionPreview = preview;
        updatePriorEligibilityHints(preview);
        updateTBOSelectionSummary();
        updateTBOTimeEstimate();
    } catch (error) {
        if (error.name !== 'AbortError' && summary) {
            summary.textContent = `Unable to preview training rows: ${error.message}`;
        }
    } finally {
        if (tboSelectionPreviewController === controller) tboSelectionPreviewController = null;
    }
}

window.scheduleTBOSelectionPreview = scheduleTBOSelectionPreview;

export function updateTBOTimeEstimate() {
    const slider = document.getElementById('tbo-prior-cap-slider');
    const badge = document.getElementById('tbo-prior-cap-badge');
    if (!slider || !badge) return;

    const requestedRows = parseInt(slider.value) || 1;
    const featCols = (window.currentColumns || []).filter(c =>
        window.columnRoles[c] === 'feature' && window.boFeatureConfigs?.[c]?.included !== false
    );
    const d = Math.max(2, featCols.length);
    const objectiveCount = Math.max(1, (window.currentColumns || []).filter(c =>
        window.columnRoles[c] === 'objective' && window.objectiveConfigs?.[c]?.included !== false
    ).length);
    const constrainedObjectives = new Set(getTBOPreviewObjectiveConstraints().map(item => item.objective));
    const modelCount = objectiveCount + constrainedObjectives.size;
    const categoricalCount = featCols.filter(c => window.boFeatureConfigs?.[c]?.type === 'categorical').length;
    const eligibleRows = tboSelectionPreview ? Number(tboSelectionPreview.eligible_rows || 0) : requestedRows;
    const fittedRows = Math.max(0, Math.min(requestedRows, eligibleRows));

    const cpuType = document.getElementById('tbo-hw-cpu')?.value || 'mid';
    const ramGb = parseInt(document.getElementById('tbo-hw-ram')?.value || '16');
    const gpuType = document.getElementById('tbo-hw-gpu')?.value || 'none';
    const batchSize = Math.max(1, parseInt(document.getElementById('tbo-batch-size')?.value || '1'));
    const kernel = document.getElementById('tbo-kernel-select')?.value || 'matern52';
    const acquisition = document.getElementById('tbo-acq-select')?.value || 'EI';

    const cpuMultipliers = { entry: 0.65, mid: 1.0, high: 1.6, workstation: 2.4 };
    const cpuMult = cpuMultipliers[cpuType] || 1.0;

    let ramMult = 1.0;
    if (ramGb <= 8) ramMult = 1.3;
    else if (ramGb === 16) ramMult = 1.0;
    else if (ramGb === 32) ramMult = 0.85;
    else ramMult = 0.75;

    const gpuMult = gpuType === 'cuda' ? 0.8 : 1.0;

    // Exact GP fitting is cubic in training rows.  Include additional models,
    // mixed-kernel overhead, and acquisition optimization in a conservative range.
    const dimensionFactor = 1 + Math.max(0, d - 2) * 0.07;
    const mixedFactor = 1 + categoricalCount * 0.18;
    const kernelFactor = { rbf: 0.9, matern32: 1.0, matern52: 1.08 }[kernel] || 1.0;
    const acquisitionFactor = acquisition === 'LCB' ? 0.9 : 1.0;
    const batchFactor = 1 + Math.max(0, batchSize - 1) * 0.35;
    const fittingSeconds = 18 * ((fittedRows / 300) ** 3) * modelCount * dimensionFactor * mixedFactor * kernelFactor;
    const acquisitionSeconds = (8 + 2.5 * d * objectiveCount) * acquisitionFactor * batchFactor;
    const priorCols = (window.currentColumns || []).filter(c => window.columnRoles[c] === 'prior');
    const validRows = Number(tboSelectionPreview?.valid_rows || window.currentData?.length || 0);
    const smilesPriors = priorCols.filter(c => window.priorConfigs?.[c]?.col_type === 'smiles').length;
    const simplePriors = Math.max(0, priorCols.length - smilesPriors);
    const relevanceSeconds = validRows * (simplePriors * 0.00015 + smilesPriors * 0.003);
    const estSec = Math.max(5, ((relevanceSeconds + fittingSeconds + acquisitionSeconds) / cpuMult) * ramMult * gpuMult);
    const lowerSec = Math.max(3, estSec * 0.8);
    const upperSec = estSec * 1.35;
    const estMin = estSec / 60;

    const formatDuration = seconds => {
        if (seconds < 60) return `${Math.max(1, Math.round(seconds))} sec`;
        if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))} min`;
        return `${(seconds / 3600).toFixed(1)} hr`;
    };
    const timeText = `~${formatDuration(lowerSec)}–${formatDuration(upperSec)}`;
    const rowNote = eligibleRows < requestedRows ? `; ${fittedRows}/${requestedRows} rows satisfy Prior rules` : `; ${fittedRows} GP rows`;
    const serverNote = `${modelCount} GP model${modelCount === 1 ? '' : 's'}${rowNote}`;

    if (estMin < 5) {
        badge.style.background = '#dcfce7';
        badge.style.color = '#15803d';
        badge.style.borderColor = '#bbf7d0';
        badge.innerHTML = `⚡ Conservative estimate: <b>${timeText}</b> (${serverNote})`;
    } else if (estMin <= 30) {
        badge.style.background = '#fef9c3';
        badge.style.color = '#854d0e';
        badge.style.borderColor = '#fef08a';
        badge.innerHTML = `⏱️ Conservative estimate: <b>${timeText}</b> (${serverNote})`;
    } else {
        badge.style.background = '#fee2e2';
        badge.style.color = '#991b1b';
        badge.style.borderColor = '#fecaca';
        badge.innerHTML = `⏳ Conservative estimate: <b>${timeText}</b> (${serverNote}; background job)`;
    }
}

/**
 * Tab 2: Prior Targets Setup
 */
function normalizeAveragePriorWeights(priorCols) {
    if (!priorCols.length) return;
    if (priorCols.length === 1) {
        window.priorConfigs[priorCols[0]].prior_weight = 100;
        return;
    }
    const raw = priorCols.map(col => Math.max(0, Number(window.priorConfigs[col]?.prior_weight ?? 0)));
    const rawTotal = raw.reduce((sum, value) => sum + value, 0);
    const denominator = rawTotal > 0 ? rawTotal : priorCols.length;
    const scaled = (rawTotal > 0 ? raw : raw.map(() => 1)).map(value => value / denominator * 1000);
    const units = scaled.map(Math.floor);
    let remainder = 1000 - units.reduce((sum, value) => sum + value, 0);
    const order = scaled
        .map((value, index) => ({ index, fraction: value - units[index] }))
        .sort((a, b) => b.fraction - a.fraction || a.index - b.index);
    for (let i = 0; i < remainder; i++) units[order[i % order.length].index]++;
    priorCols.forEach((col, index) => {
        window.priorConfigs[col].prior_weight = units[index] / 10;
    });
}

export function renderTBOPriorsTab() {
    initTBOPriorCapSlider();
    const container = document.getElementById('tbo-priors-container');
    if (!container) return;

    const priorCols = window.currentColumns.filter(c => window.columnRoles[c] === 'prior');

    if (priorCols.length === 0) {
        container.innerHTML = `
            <div style="background: #faf5ff; border: 1px dashed #d8b4fe; border-radius: 14px; padding: 2.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🧪</div>
                <h3 style="color: #6b21a8; margin-bottom: 0.5rem; font-weight: 600;">No Prior Columns Designated</h3>
                <p style="color: #7e22ce; max-width: 580px; margin: 0 auto 1.25rem; font-size: 0.95rem; line-height: 1.5;">
                    To leverage Transfer Learning, mark one or more columns in the Loaded Data table as 
                    <b style="background: #7c3aed; color: white; padding: 2px 8px; border-radius: 6px;">P (Prior)</b>.
                    You can specify target substrates (SMILES), solvents, temperatures, or catalyst types to weight prior experiments.
                </p>
            </div>
        `;
        return;
    }

    priorCols.forEach(col => {
        if (!window.priorConfigs[col]) {
            window.priorConfigs[col] = {
                name: col,
                col_type: 'categorical',
                target_value: '',
                tolerance: null,
                min_similarity: 0.5,
                prior_weight: 100,
                is_new_level: false
            };
        }
    });
    if (window.priorCombineRule === 'average') normalizeAveragePriorWeights(priorCols);

    let html = `
        <div class="combination-rule-bar" style="display: flex; justify-content: space-between; align-items: center; background: #faf5ff; border: 1px solid #e9d5ff; border-radius: 12px; padding: 0.85rem 1.25rem; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 1.25rem;">
                <span style="font-weight: 600; font-size: 0.95rem; color: #581c87;">Prior Combination Rule:</span>
                <label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer; font-size: 0.9rem;">
                    <input type="radio" name="tbo-prior-combine" value="multiply" ${window.priorCombineRule === 'multiply' ? 'checked' : ''}>
                    <span><b>Multiply</b> (Strict: all priors must align)</span>
                </label>
                <label style="display: flex; align-items: center; gap: 0.4rem; cursor: pointer; font-size: 0.9rem;">
                    <input type="radio" name="tbo-prior-combine" value="average" ${window.priorCombineRule === 'average' ? 'checked' : ''}>
                    <span><b>Average</b> (Flexible: partial matches contribute)</span>
                </label>
            </div>
            <div style="font-size: 0.85rem; color: #6b21a8;">
                ${priorCols.length} prior column${priorCols.length > 1 ? 's' : ''} active
            </div>
        </div>
        <div id="tbo-priors-list"></div>
    `;

    container.innerHTML = html;

    // Listener for combination rule
    container.querySelectorAll('input[name="tbo-prior-combine"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            window.priorCombineRule = e.target.value;
            if (window.priorCombineRule === 'average') normalizeAveragePriorWeights(priorCols);
            renderTBOPriorsTab();
        });
    });

    const listEl = container.querySelector('#tbo-priors-list');

    priorCols.forEach((col, idx) => {
        if (!window.priorConfigs[col]) {
            window.priorConfigs[col] = {
                name: col,
                col_type: 'categorical',
                target_value: '',
                tolerance: null,
                min_similarity: 0.5,
                prior_weight: 100,
                is_new_level: false
            };
        }

        const cfg = ensurePriorDefaults(col, window.priorConfigs[col]);
        const colIdx = window.currentColumns.indexOf(col);
        const colVals = window.currentData ? window.currentData.map(r => r[colIdx]).filter(v => v != null && v !== '') : [];
        const uniqueVals = [...new Set(colVals)];

        const card = document.createElement('div');
        card.className = 'prior-card';
        card.style.cssText = `
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 5px solid #7c3aed;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        `;

        // Type badge label
        let typeBadge = '';
        if (cfg.col_type === 'smiles') {
            typeBadge = `<span style="background: #f5f3ff; color: #7c3aed; border: 1px solid #ddd6fe; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">SMILES Molecule</span>`;
        } else if (cfg.col_type === 'numerical') {
            typeBadge = `<span style="background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">Numerical Distance</span>`;
        } else {
            typeBadge = `<span style="background: #f8fafc; color: #475569; border: 1px solid #cbd5e1; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600;">Categorical</span>`;
        }

        let innerContent = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <h3 style="margin: 0; font-size: 1.05rem; font-weight: 600; color: var(--text-primary);">${col}</h3>
                    ${typeBadge}
                    <select class="tbo-prior-type-override" data-col="${col}" style="font-size: 0.8rem; padding: 2px 6px; border-radius: 6px; border: 1px solid #cbd5e1; color: #64748b;">
                        <option value="categorical" ${cfg.col_type === 'categorical' ? 'selected' : ''}>Categorical</option>
                        <option value="numerical"   ${cfg.col_type === 'numerical' ? 'selected' : ''}>Numerical</option>
                        <option value="smiles"      ${cfg.col_type === 'smiles' ? 'selected' : ''}>SMILES</option>
                    </select>
                </div>
                ${window.priorCombineRule === 'average' ? `
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <label style="font-size: 0.85rem; color: #64748b; font-weight: 500;">Optimizer weight (%):</label>
                        <input type="number" class="tbo-prior-weight" data-col="${col}" value="${Number(cfg.prior_weight ?? 100).toFixed(1)}" min="0" max="100" step="0.1" ${priorCols.length === 1 ? 'disabled' : ''} style="width: 70px; padding: 4px 8px; border-radius: 6px; border: 1px solid #cbd5e1;">
                    </div>
                ` : ''}
            </div>
        `;

        // Render target controls based on detected or selected type
        if (cfg.col_type === 'categorical') {
            const isCustom = cfg.is_new_level || (!uniqueVals.includes(cfg.target_value) && cfg.target_value !== '');
            innerContent += `
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                        <label style="font-weight: 500; font-size: 0.9rem; min-width: 100px;">Target Level:</label>
                        <select class="tbo-prior-cat-select" data-col="${col}" style="min-width: 200px; padding: 6px 12px; border-radius: 8px; border: 1px solid #cbd5e1;">
                            <option value="">-- Select Target Level --</option>
                            ${uniqueVals.map(v => `<option value="${v}" ${v === cfg.target_value && !isCustom ? 'selected' : ''}>${v}</option>`).join('')}
                            <option value="__NEW_LEVEL__" ${isCustom ? 'selected' : ''}>+ New level (unseen)...</option>
                        </select>
                        <input type="text" class="tbo-prior-new-level" data-col="${col}" value="${isCustom ? cfg.target_value : ''}" placeholder="Enter new unseen level name" style="display: ${isCustom ? 'inline-block' : 'none'}; min-width: 200px; padding: 6px 12px; border-radius: 8px; border: 1px solid #cbd5e1;">
                    </div>
                    <div class="warning-badge tbo-new-level-warning" data-col="${col}" style="display: ${isCustom ? 'inline-flex' : 'none'};">
                        ⚠️ <b>Warning:</b> This level is not present in the historical data. The algorithm will cold-start unless other priors provide similarity signal.
                    </div>
                    <div class="tbo-live-effective-hint" data-col="${col}" style="font-size: 0.85rem; color: #475569;"></div>
                </div>
            `;
        } else if (cfg.col_type === 'numerical') {
            const scale = numericalPriorScale(col);
            const target = Number(cfg.target_value);
            const tolerance = Math.max(0, Number(cfg.tolerance) || 0);
            const targetDistance = Number.isFinite(target)
                ? Math.max(Math.abs(target - scale.minimum), Math.abs(scale.maximum - target))
                : scale.span;
            const toleranceMaximum = Math.max(scale.step, scale.span, targetDistance, tolerance);

            innerContent += `
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                        <label style="font-weight: 500; font-size: 0.9rem; min-width: 100px;">Target Value:</label>
                        <input type="number" step="any" class="tbo-prior-num-target" data-col="${col}" value="${cfg.target_value ?? ''}" placeholder="Historical range: [${scale.minimum}, ${scale.maximum}]" style="min-width: 180px; padding: 6px 12px; border-radius: 8px; border: 1px solid #cbd5e1;">
                        <span style="font-size: 0.85rem; color: #64748b;">Observed range in data: <b>[${scale.minimum}, ${scale.maximum}]</b></span>
                    </div>
                    <div style="display:flex;flex-direction:column;gap:0.65rem;background:#f8fafc;padding:0.75rem 1rem;border-radius:8px;border:1px solid #e2e8f0;">
                        <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
                            <label style="font-size:0.9rem;font-weight:500;min-width:170px;">Accepted deviation (±):</label>
                            <input type="range" class="tbo-prior-tolerance-slider" data-col="${col}" min="0" max="${toleranceMaximum}" step="${scale.step}" value="${tolerance}" style="flex:1;min-width:180px;max-width:360px;accent-color:#7c3aed;">
                            <input type="number" class="tbo-prior-tolerance-number" data-col="${col}" min="0" step="${scale.step}" value="${tolerance}" style="width:110px;padding:5px 8px;border-radius:6px;border:1px solid #cbd5e1;">
                        </div>
                        <div class="tbo-numerical-range" data-col="${col}" style="font-size:0.83rem;color:#6b21a8;font-weight:600;">
                            Enter a target value to see the accepted range.
                        </div>
                        <div style="font-size:0.76rem;color:#64748b;">Rows outside this range receive zero relevance. Relevance increases smoothly toward the target.</div>
                    </div>
                    <div class="tbo-live-effective-hint" data-col="${col}" style="font-size: 0.85rem; color: #475569;"></div>
                </div>
            `;
        } else if (cfg.col_type === 'smiles') {
            const minimumSimilarity = Math.min(1, Math.max(0, Number(cfg.min_similarity ?? 0.5)));
            innerContent += `
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                        <label style="font-weight: 500; font-size: 0.9rem; min-width: 100px;">Target SMILES:</label>
                        <input type="text" class="tbo-prior-smiles-target" data-col="${col}" value="${cfg.target_value ?? ''}" placeholder="e.g. c1ccccc1O (Phenol)" style="flex: 1; min-width: 250px; font-family: monospace; padding: 6px 12px; border-radius: 8px; border: 1px solid #cbd5e1;">
                    </div>
                    <div style="display:flex;flex-direction:column;gap:0.65rem;background:#f8fafc;padding:0.75rem 1rem;border-radius:8px;border:1px solid #e2e8f0;">
                        <div style="display:flex;align-items:center;gap:1.25rem;flex-wrap:wrap;">
                            <label style="font-size:0.9rem;font-weight:500;min-width:210px;">
                                Minimum molecular similarity: <span class="tbo-min-similarity-val" data-col="${col}" style="font-weight:700;color:#7c3aed;">${Math.round(minimumSimilarity * 100)}%</span>
                            </label>
                            <input type="range" class="tbo-prior-min-similarity-slider" data-col="${col}" min="0" max="1" step="0.01" value="${minimumSimilarity}" style="flex:1;min-width:200px;max-width:360px;accent-color:#7c3aed;">
                        </div>
                        <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#64748b;max-width:610px;">
                            <span>Broad molecular neighbourhood</span>
                            <span>Exact structure only</span>
                        </div>
                    </div>
                    <div style="font-size: 0.85rem; color: #64748b;">
                        Morgan fingerprint Tanimoto similarity is used. Molecules below the selected minimum receive zero relevance.
                    </div>
                    <div class="tbo-live-effective-hint" data-col="${col}" style="font-size: 0.85rem; color: #475569;"></div>
                </div>
            `;
        }

        card.innerHTML = innerContent;

        // Event listeners for card controls
        const typeSelect = card.querySelector('.tbo-prior-type-override');
        if (typeSelect) {
            typeSelect.addEventListener('change', (e) => {
                cfg.col_type = e.target.value;
                renderTBOPriorsTab();
            });
        }

        const weightInput = card.querySelector('.tbo-prior-weight');
        if (weightInput) {
            weightInput.addEventListener('change', (e) => {
                rebalanceRemaining(e.target, e.target.value, '#tbo-priors-list .tbo-prior-weight');
                document.querySelectorAll('#tbo-priors-list .tbo-prior-weight').forEach(input => {
                    if (window.priorConfigs[input.dataset.col]) {
                        window.priorConfigs[input.dataset.col].prior_weight = Number(input.value);
                    }
                });
                scheduleTBOSelectionPreview();
            });
        }

        // Categorical events
        const catSelect = card.querySelector('.tbo-prior-cat-select');
        const newLevelInput = card.querySelector('.tbo-prior-new-level');
        const newLevelWarn = card.querySelector('.tbo-new-level-warning');

        if (catSelect) {
            catSelect.addEventListener('change', (e) => {
                if (e.target.value === '__NEW_LEVEL__') {
                    newLevelInput.style.display = 'inline-block';
                    newLevelWarn.style.display = 'inline-flex';
                    cfg.is_new_level = true;
                    cfg.target_value = newLevelInput.value;
                } else {
                    newLevelInput.style.display = 'none';
                    newLevelWarn.style.display = 'none';
                    cfg.is_new_level = false;
                    cfg.target_value = e.target.value;
                }
                scheduleTBOSelectionPreview();
            });
        }
        if (newLevelInput) {
            newLevelInput.addEventListener('input', (e) => {
                cfg.target_value = e.target.value;
                scheduleTBOSelectionPreview();
            });
        }

        // Numerical events
        const numTarget = card.querySelector('.tbo-prior-num-target');
        if (numTarget) {
            numTarget.addEventListener('input', (e) => {
                cfg.target_value = e.target.value;
                updateNumericalAcceptedRange(card, cfg);
                scheduleTBOSelectionPreview();
            });
        }

        const toleranceSlider = card.querySelector('.tbo-prior-tolerance-slider');
        const toleranceNumber = card.querySelector('.tbo-prior-tolerance-number');
        const setTolerance = rawValue => {
            const value = Math.max(0, Number(rawValue) || 0);
            cfg.tolerance = value;
            if (toleranceSlider) {
                if (value > Number(toleranceSlider.max)) toleranceSlider.max = String(value);
                toleranceSlider.value = String(value);
            }
            if (toleranceNumber) toleranceNumber.value = String(value);
            updateNumericalAcceptedRange(card, cfg);
            scheduleTBOSelectionPreview();
        };
        toleranceSlider?.addEventListener('input', event => setTolerance(event.target.value));
        toleranceNumber?.addEventListener('input', event => setTolerance(event.target.value));
        if (cfg.col_type === 'numerical') updateNumericalAcceptedRange(card, cfg);

        // SMILES events
        const smilesTarget = card.querySelector('.tbo-prior-smiles-target');
        if (smilesTarget) {
            smilesTarget.addEventListener('input', (e) => {
                cfg.target_value = e.target.value.trim();
                scheduleTBOSelectionPreview(600);
            });
        }

        const similaritySlider = card.querySelector('.tbo-prior-min-similarity-slider');
        const similarityLabel = card.querySelector('.tbo-min-similarity-val');
        similaritySlider?.addEventListener('input', event => {
            const value = Math.min(1, Math.max(0, Number(event.target.value)));
            cfg.min_similarity = value;
            if (similarityLabel) similarityLabel.textContent = `${Math.round(value * 100)}%`;
            scheduleTBOSelectionPreview(600);
        });

        listEl.appendChild(card);
    });
    scheduleTBOSelectionPreview();
}

function formatPriorNumber(value) {
    if (!Number.isFinite(value)) return '—';
    return Number(value.toPrecision(10)).toLocaleString();
}

function updateNumericalAcceptedRange(card, config) {
    const label = card.querySelector('.tbo-numerical-range');
    if (!label) return;
    const target = Number(config.target_value);
    if (!Number.isFinite(target) || String(config.target_value).trim() === '') {
        label.textContent = 'Enter a target value to see the accepted range.';
        return;
    }
    const tolerance = Math.max(0, Number(config.tolerance) || 0);
    label.textContent = `Accepted range: ${formatPriorNumber(target - tolerance)} to ${formatPriorNumber(target + tolerance)} (target ${formatPriorNumber(target)} ± ${formatPriorNumber(tolerance)})`;
}

/**
 * Tab 3: Objectives Setup
 */
export function renderTBOObjectivesTab() {
    const list = document.getElementById('tbo-objectives-list');
    if (!list) return;

    list.innerHTML = '';
    const objCols = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');

    if (objCols.length === 0) {
        list.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                    No objective columns selected. Toggle columns to 'O' in the table above.
                </td>
            </tr>`;
        return;
    }

    objCols.forEach(col => {
        if (!window.objectiveConfigs[col]) {
            window.objectiveConfigs[col] = {
                name: col,
                type: 'maximize',
                target: '',
                importance: Math.round(100 / objCols.length),
                included: true
            };
        }

        const cfg = window.objectiveConfigs[col];
        const row = document.createElement('tr');
        if (!cfg.included) row.classList.add('row-disabled');

        row.innerHTML = `
            <td style="font-weight: 500;">${col}</td>
            <td>
                <select class="tbo-obj-type" data-col="${col}">
                    <option value="maximize" ${cfg.type === 'maximize' ? 'selected' : ''}>Maximize</option>
                    <option value="minimize" ${cfg.type === 'minimize' ? 'selected' : ''}>Minimize</option>
                    <option value="target"   ${cfg.type === 'target' ? 'selected' : ''}>Target</option>
                </select>
            </td>
            <td>
                <input type="number" step="any" class="tbo-obj-target" data-col="${col}" value="${cfg.target ?? ''}" placeholder="Target value" style="display: ${cfg.type === 'target' ? 'block' : 'none'};">
            </td>
            <td>
                <input type="number" class="tbo-obj-importance" data-col="${col}" value="${cfg.importance ?? 100}" min="1" max="100">
            </td>
            <td style="text-align: center;">
                <input type="checkbox" class="tbo-obj-inc" data-col="${col}" ${cfg.included ? 'checked' : ''}>
            </td>
        `;

        row.querySelector('.tbo-obj-type').addEventListener('change', (e) => {
            cfg.type = e.target.value;
            row.querySelector('.tbo-obj-target').style.display = cfg.type === 'target' ? 'block' : 'none';
            scheduleTBOSelectionPreview();
        });
        row.querySelector('.tbo-obj-target').addEventListener('input', (e) => {
            cfg.target = e.target.value;
        });
        row.querySelector('.tbo-obj-importance').addEventListener('input', (e) => {
            const selector = '#tbo-objectives-list .tbo-obj-importance';
            rebalanceRemaining(col, e.target.value, selector);
            document.querySelectorAll(selector).forEach(el => {
                const c = el.dataset.col;
                if (window.objectiveConfigs[c]) {
                    window.objectiveConfigs[c].importance = parseFloat(el.value) || 0;
                }
            });
        });
        row.querySelector('.tbo-obj-inc').addEventListener('change', (e) => {
            cfg.included = e.target.checked;
            row.classList.toggle('row-disabled', !cfg.included);
            rebalanceObjectiveConfigs();
            document.querySelectorAll('#tbo-objectives-list .tbo-obj-importance').forEach(el => {
                const c = el.dataset.col;
                if (window.objectiveConfigs[c]) {
                    el.value = window.objectiveConfigs[c].importance;
                }
            });
            window.refreshObjectiveConstraintRows?.('tbo');
            updateTBOExplorationControlAvailability();
            renderTBOVisualization();
            scheduleTBOSelectionPreview();
        });

        list.appendChild(row);
    });

    wireSelectAll(
        'tbo-select-all-objectives',
        '#tbo-objectives-list .tbo-obj-inc',
        (columns, checked) => {
            columns.forEach(col => {
                if (!window.objectiveConfigs[col]) return;
                window.objectiveConfigs[col].included = checked;
                document.querySelector(`#tbo-objectives-list .tbo-obj-inc[data-col="${col}"]`)
                    ?.closest('tr')?.classList.toggle('row-disabled', !checked);
            });
            rebalanceObjectiveConfigs();
            document.querySelectorAll('#tbo-objectives-list .tbo-obj-importance').forEach(el => {
                const c = el.dataset.col;
                if (window.objectiveConfigs[c]) {
                    el.value = window.objectiveConfigs[c].importance;
                }
            });
            window.refreshObjectiveConstraintRows?.('tbo');
            updateTBOExplorationControlAvailability();
            renderTBOVisualization();
            scheduleTBOSelectionPreview();
        }
    );
    window.refreshObjectiveConstraintRows?.('tbo');
}

/**
 * Tab 4: Tweaks Setup
 */
export function renderTBOTweaksTab() {
    // Tweaks controls are static in the HTML template
}

export function updateTBOExplorationControlAvailability() {
    const acqSelect = document.getElementById('tbo-acq-select');
    const slider = document.getElementById('tbo-exploration');
    const number = document.getElementById('tbo-exploration-number');
    if (!acqSelect || !slider || !number) return;

    const activeObjectives = (window.currentColumns || []).filter(col =>
        window.columnRoles?.[col] === 'objective'
        && window.objectiveConfigs?.[col]?.included !== false
    ).length;
    const isApplicable = activeObjectives === 1 && acqSelect.value === 'LCB';
    const section = slider.closest('.exploration-control') || slider.closest('.tweak-item');

    slider.disabled = !isApplicable;
    number.disabled = !isApplicable;
    section?.classList.toggle('control-disabled', !isApplicable);
    section?.setAttribute('aria-disabled', String(!isApplicable));
    if (section) {
        section.style.opacity = isApplicable ? '1' : '0.45';
        section.title = isApplicable
            ? ''
            : 'Exploration is available only for single-objective LCB optimization.';
    }
}

let tboTabsInitialized = false;
function initTBOTabs() {
    if (tboTabsInitialized) return;
    tboTabsInitialized = true;

    const tabBtns = document.querySelectorAll('#tbo-module .setup-tabs .tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const targetId = btn.dataset.tab;
            document.querySelectorAll('#tbo-module .tab-content').forEach(tc => {
                tc.classList.add('hidden');
            });
            const activeTab = document.getElementById(targetId);
            if (activeTab) activeTab.classList.remove('hidden');
        });
    });
}

let tboButtonsInitialized = false;
export function initTBOButtons() {
    if (tboButtonsInitialized) return;
    tboButtonsInitialized = true;

    safeAddListener('tbo-run-btn', 'click', runTransferBO);
    safeAddListener('tbo-export-btn', 'click', () => {
        exportTBOSuggestions();
    });
    safeAddListener('tbo-estimate-btn', 'click', runTransferBOEstimate);
    safeAddListener('tbo-acq-select', 'change', updateTBOExplorationControlAvailability);
    const numericalConstraintButton = document.getElementById('tbo-add-num-constraint-btn');
    if (numericalConstraintButton) {
        numericalConstraintButton.disabled = true;
        numericalConstraintButton.title = 'The current TransBO engine cannot enforce feature constraints for fitted GP suggestions.';
    }
    const avoidReevaluationToggle = document.getElementById('tbo-avoid-reval-toggle');
    if (avoidReevaluationToggle) {
        avoidReevaluationToggle.checked = false;
        avoidReevaluationToggle.disabled = true;
        avoidReevaluationToggle.title = 'The current TransBO engine does not filter historical duplicates in the fitted GP path.';
    }
    safeAddListener('tbo-commit-suggestions-btn', 'click', (e) => {
        const commitBtn = document.getElementById('tbo-commit-suggestions-btn');
        if (commitBtn && commitBtn.disabled) return;
        const rows = document.querySelectorAll('#tbo-results-body tr');
        if (rows.length === 0) return;

        // Collect and validate objective values from the editable cells
        const objCells = document.querySelectorAll('#tbo-results-body .tbo-obj-cell');
        let allFilled = objCells.length > 0;
        objCells.forEach(td => {
            const val = td.textContent.trim();
            if (val === '' || isNaN(parseFloat(val))) allFilled = false;
        });
        if (!allFilled) {
            alert('Please fill in all objective values before adding experiments to the dataset.');
            return;
        }

        const featureCols   = window.currentColumns.filter(c => window.columnRoles[c] === 'feature' && (window.boFeatureConfigs[c] ? window.boFeatureConfigs[c].included !== false : true));
        const objectiveCols = window.currentColumns.filter(c => window.columnRoles[c] === 'objective' && (window.objectiveConfigs[c] ? window.objectiveConfigs[c].included !== false : true));
        const priorCols     = window.currentColumns.filter(c => window.columnRoles[c] === 'prior');
        const otherCols     = window.currentColumns.filter(c =>
            !featureCols.includes(c) && !objectiveCols.includes(c) && !priorCols.includes(c)
        );
        const tableCols = [...featureCols, ...objectiveCols, ...priorCols, ...otherCols];

        const newRows = [];
        rows.forEach(tr => {
            // First cell is row index # (skip it)
            const cells = Array.from(tr.querySelectorAll('td')).slice(1);
            const newRow = new Array(window.currentColumns.length).fill('');
            tableCols.forEach((col, idx) => {
                const mainColIdx = window.currentColumns.indexOf(col);
                if (mainColIdx !== -1) {
                    const rawVal = cells[idx] ? cells[idx].textContent.trim() : '';
                    newRow[mainColIdx] = (rawVal === '—' || rawVal === '-') ? '' : rawVal;
                }
            });
            newRows.push(newRow);
        });

        if (newRows.length === 0) return;

        // Disable commit button to prevent multiple submissions
        if (commitBtn) {
            commitBtn.disabled = true;
            commitBtn.style.opacity = '0.45';
            commitBtn.style.cursor = 'not-allowed';
        }

        // Clear suggestions table body and header so rows cannot be re-read
        const tboBody = document.getElementById('tbo-results-body');
        if (tboBody) tboBody.innerHTML = '';
        const tboHead = document.getElementById('tbo-results-head');
        if (tboHead) tboHead.innerHTML = '';

        window.currentData = window.currentData.concat(newRows);
        window.cedos?.notifyDatasetMutation('TransBO suggestions committed');
        window.renderMainTable();
        renderTBOVisualization();
        alert(`Successfully added ${newRows.length} experiment(s) to the Main Dataset.`);
        document.getElementById('tbo-results-section')?.classList.add('hidden');
    });

    initTBOVisualizationDropdown();

    const tboSlider = document.getElementById('tbo-exploration');
    const tboNumber = document.getElementById('tbo-exploration-number');
    if (tboSlider && tboNumber) {
        tboSlider.addEventListener('input', (e) => tboNumber.value = e.target.value);
        tboNumber.addEventListener('input', (e) => tboSlider.value = e.target.value);
    }
}

let tboTickerTimer = null;
let tboTickerStartTime = 0;

function startTBOProgressTicker(loadingTextEl, stepDetailEl, progressBarEl, datasetSize) {
    tboTickerStartTime = Date.now();
    if (loadingTextEl) {
        loadingTextEl.textContent = 'Running Transfer Learning Bayesian Optimization...';
    }
    if (stepDetailEl) {
        stepDetailEl.classList.remove('hidden');
        stepDetailEl.textContent = datasetSize > 0 
            ? `Initializing transfer engine with ${datasetSize.toLocaleString()} data points...` 
            : 'Initializing transfer engine...';
    }
    const container = document.getElementById('loading-progress-container');
    if (container) container.classList.remove('hidden');
    if (progressBarEl) progressBarEl.style.width = '5%';

    const stages = [
        { time: 0, text: 'Parsing dataset and calculating prior transfer weights...', progress: 15 },
        { time: 3500, text: 'Constructing heteroscedastic noise GP training matrices...', progress: 35 },
        { time: 8000, text: 'Fitting Gaussian Process surrogate models on weighted data...', progress: 55 },
        { time: 18000, text: 'Optimizing acquisition function over experimental space...', progress: 75 },
        { time: 28000, text: 'Evaluating candidate proposals and validating constraints...', progress: 85 },
        { time: 45000, text: 'Refining suggestions and model diagnostics...', progress: 92 },
        { time: 70000, text: 'Still computing — large datasets require additional steps. Please wait...', progress: 96 }
    ];

    if (tboTickerTimer) clearInterval(tboTickerTimer);

    tboTickerTimer = setInterval(() => {
        const elapsed = Date.now() - tboTickerStartTime;
        let currentStage = stages[0];
        for (const stage of stages) {
            if (elapsed >= stage.time) {
                currentStage = stage;
            }
        }
        if (stepDetailEl) {
            stepDetailEl.textContent = currentStage.text;
        }
        if (progressBarEl) {
            progressBarEl.style.width = `${currentStage.progress}%`;
        }
    }, 500);
}

function stopTBOProgressTicker() {
    if (tboTickerTimer) {
        clearInterval(tboTickerTimer);
        tboTickerTimer = null;
    }
    const stepDetailEl = document.getElementById('loading-step-detail');
    const container = document.getElementById('loading-progress-container');
    const progressBarEl = document.getElementById('loading-progress-bar');
    if (stepDetailEl) {
        stepDetailEl.classList.add('hidden');
        stepDetailEl.textContent = '';
    }
    if (container) {
        container.classList.add('hidden');
    }
    if (progressBarEl) {
        progressBarEl.style.width = '0%';
    }
}

/**
 * Core Execution: runTransferBO()
 */
export async function runTransferBO() {
    if (!window.currentData || window.currentData.length === 0) {
        alert('Please load a CSV dataset first.');
        return;
    }

    const featureCols = window.currentColumns.filter(c => window.columnRoles[c] === 'feature' && window.boFeatureConfigs[c]?.included);
    const objectiveCols = window.currentColumns.filter(c => window.columnRoles[c] === 'objective' && window.objectiveConfigs[c]?.included);
    const priorCols = window.currentColumns.filter(c => window.columnRoles[c] === 'prior');
    if (featureCols.length === 0) {
        alert('Please include at least one Feature column.');
        return;
    }
    if (objectiveCols.length === 0) {
        alert('Please include at least one Objective column.');
        return;
    }
    let objectiveConstraints;
    try {
        objectiveConstraints = window.collectObjectiveConstraints?.('tbo') || [];
    } catch (error) {
        alert(error.message);
        return;
    }

    // The transfer engine only applies feature constraints during cold start.
    // Reject imported/stale rows rather than accepting a rule that may be ignored.
    const hasFeatureConstraint = [...document.querySelectorAll('#tbo-numerical-constraints-list .constraint-equation')]
        .some(input => input.value.trim());
    if (hasFeatureConstraint) {
        alert('TransBO cannot enforce numerical feature constraints for all suggestions. Remove these rules before running TransBO.');
        return;
    }

    const overlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');
    const stepDetail = document.getElementById('loading-step-detail');
    const progressBar = document.getElementById('loading-progress-bar');
    clearTBOResultsForNewRun();
    if (overlay) overlay.classList.remove('hidden');

    const totalRows = window.currentData ? window.currentData.length : 0;
    startTBOProgressTicker(loadingText, stepDetail, progressBar, totalRows);

    const requestController = new AbortController();
    window.currentAbortController = requestController;

    const featuresPayload = featureCols.map(col => (
        buildTBOFeaturePayload(col, window.boFeatureConfigs[col])
    ));

    const objectivesPayload = objectiveCols.map(col => ({
        name: col,
        type: window.objectiveConfigs[col].type,
        target: window.objectiveConfigs[col].target,
        importance: window.objectiveConfigs[col].importance
    }));

    const priorsPayload = buildPriorPayload(priorCols);

    // Numerical and disallowed constraints
    const numConstraints = [];
    document.querySelectorAll('#tbo-numerical-constraints-list tr').forEach(tr => {
        const eqInput = tr.querySelector('.constraint-equation');
        if (eqInput && eqInput.value.trim()) numConstraints.push(eqInput.value.trim());
    });

    const tweaksPayload = {
        batch_size: parseInt(document.getElementById('tbo-batch-size')?.value || '1'),
        kernel: document.getElementById('tbo-kernel-select')?.value || 'matern52',
        acq_type: document.getElementById('tbo-acq-select')?.value || 'EI',
        exploration: parseFloat(document.getElementById('tbo-exploration')?.value || '0.5'),
        noiseless: document.getElementById('tbo-noiseless-toggle')?.checked || false,
        avoid_reval: false,
        max_gp_rows: parseInt(document.getElementById('tbo-prior-cap-slider')?.value || '3'),
        allow_reduced_row_recovery: document.getElementById('tbo-reduced-row-recovery-toggle')?.checked || false,
        constraints: {
            numerical: numConstraints,
            disallowed: []
        },
        objective_constraints: objectiveConstraints
    };

    const payload = {
        data: window.currentData,
        columns: window.currentColumns,
        features: featuresPayload,
        objectives: objectivesPayload,
        priors: priorsPayload,
        prior_combine: window.priorCombineRule || 'multiply',
        tweaks: tweaksPayload
    };
    const trainingContext = tboTrainingContextKey({
        features: featuresPayload,
        objectives: objectivesPayload,
        priors: priorsPayload,
        priorCombine: payload.prior_combine,
        tweaks: tweaksPayload
    });

    try {
        const completed = await submitAndPollJob('/transfer_optimize_submit', payload, {
            signal: requestController.signal,
            onProgress: (status, jobId) => {
                activeTBOOptimizationJobId = jobId;
                // Once the server reports real progress, stop the synthetic
                // timer so it cannot overwrite the authoritative job phase.
                if (status.status === 'running' && tboTickerTimer) {
                    clearInterval(tboTickerTimer);
                    tboTickerTimer = null;
                }
                if (stepDetail && status.phase) stepDetail.textContent = status.phase;
                if (progressBar && Number.isFinite(Number(status.percent))) {
                    progressBar.style.width = `${Math.max(5, Math.min(98, Number(status.percent)))}%`;
                }
            }
        });
        const data = completed.result;
        if (data.exhausted) {
            const section = document.getElementById('tbo-results-section');
            if (section) {
                section.classList.remove('hidden');
                const banner = document.createElement('div');
                banner.id = 'tbo-exhaustion-banner';
                banner.className = 'cold-start-banner';
                banner.textContent = 'TransBO could not suggest another candidate with the current dataset and settings. Review the search space or add measured data before trying again.';
                section.prepend(banner);
            }
            window.transferDiagnostics = data.transfer || null;
            renderTransferDiagnostics(data.transfer);
            renderTBOPerformanceTable(data.performance);
            return;
        }
        latestTBOResults = data;
        window.transferDiagnostics = data.transfer;
        const trainingRows = data.transfer?.training_row_indices;
        reusableTBOTraining = Array.isArray(trainingRows) && trainingRows.length >= 3
            ? { context: trainingContext, sourceRows: trainingRows }
            : null;

        renderTBOResults(data);
        renderTransferDiagnostics(data.transfer);
        renderTBOPerformanceTable(data.performance);
        renderTBOEstimator();

        // Update response surface if active
        renderTBOVisualization();

    } catch (e) {
        if (e.name !== 'AbortError') {
            alert(`Error: ${e.message}`);
        }
    } finally {
        if (requestController.signal.aborted) {
            await cancelBackgroundJob(activeTBOOptimizationJobId);
        }
        activeTBOOptimizationJobId = null;
        stopTBOProgressTicker();
        if (overlay) overlay.classList.add('hidden');
        if (window.currentAbortController === requestController) {
            window.currentAbortController = null;
        }
    }
}

/**
 * Renders Suggestions Table
 * Shows ALL dataset columns:
 *   • Feature columns  → suggested value (highlighted)
 *   • Objective columns → editable input (user fills in after experiment)
 *   • Prior columns    → shown but grayed (context only)
 *   • Other columns    → grayed, empty (user fills manually after commit)
 */
export function renderTBOResults(data) {
    const section = document.getElementById('tbo-results-section');
    if (!section) return;

    section.classList.remove('hidden');

    document.getElementById('tbo-feasibility-warning')?.remove();
    if (data.constraint_status && (
        data.constraint_status.cold_start ||
        data.constraint_status.all_likely_feasible === false
    )) {
        const feasibilityBanner = document.createElement('div');
        feasibilityBanner.id = 'tbo-feasibility-warning';
        feasibilityBanner.style.cssText = 'margin:0.75rem 0;padding:0.8rem 1rem;border-radius:8px;background:#fff7ed;border:1px solid #fdba74;color:#9a3412;font-weight:600;';
        feasibilityBanner.textContent = data.constraint_status.message || '⚠️ The returned candidate has low predicted feasibility for one or more objective constraints.';
        section.prepend(feasibilityBanner);
    }

    const warningBanner = document.getElementById('tbo-cold-start-banner');
    if (data.transfer && data.transfer.cold_start_warning) {
        warningBanner.classList.remove('hidden');
        warningBanner.innerHTML = `
            <div style="display: flex; gap: 0.75rem; align-items: center;">
                <span style="font-size: 1.3rem;">⚠️</span>
                <div>
                    <b>Cold-Start Initialization Active:</b> ${data.transfer.warning_message || 'None of the prior rows matched your targets. Suggestions are exploratory Sobol samples.'}
                </div>
            </div>
        `;
    } else if (warningBanner) {
        warningBanner.classList.add('hidden');
    }

    const head = document.getElementById('tbo-results-head');
    const body = document.getElementById('tbo-results-body');
    if (!head || !body) return;

    // Classify all columns
    const featureCols    = window.currentColumns.filter(c => window.columnRoles[c] === 'feature' && (window.boFeatureConfigs[c] ? window.boFeatureConfigs[c].included !== false : true));
    const objectiveCols  = window.currentColumns.filter(c => window.columnRoles[c] === 'objective' && (window.objectiveConfigs[c] ? window.objectiveConfigs[c].included !== false : true));
    const priorCols      = window.currentColumns.filter(c => window.columnRoles[c] === 'prior');
    const otherCols      = window.currentColumns.filter(c =>
        !featureCols.includes(c) && !objectiveCols.includes(c) && !priorCols.includes(c)
    );

    // Build header
    head.innerHTML = `<tr>
        <th style="width: 50px; text-align: center;">#</th>
        ${featureCols.map(c => `<th style="background:#f5f3ff; color:#6d28d9;">${c} <span style="font-size:0.7rem;font-weight:400;opacity:0.8;">(F)</span></th>`).join('')}
        ${objectiveCols.map(c => `<th style="background:#ecfdf5; color:#065f46;">${c} <span style="font-size:0.7rem;font-weight:400;opacity:0.8;">(O) ✏️</span></th>`).join('')}
        ${priorCols.map(c => `<th style="color:#7c3aed;opacity:0.7;">${c} <span style="font-size:0.7rem;font-weight:400;opacity:0.8;">(P)</span></th>`).join('')}
        ${otherCols.map(c => `<th style="opacity:0.5;">${c}</th>`).join('')}
    </tr>`;

    body.innerHTML = '';
    if (data.suggestions) {
        data.suggestions.forEach((cand, idx) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td style="font-weight: 600; color: #7c3aed; text-align: center;">${idx + 1}</td>
                ${featureCols.map(c => `
                    <td contenteditable="true" class="tbo-feat-cell editable-cell" style="background:#faf5ff; border: 1px dashed #c084fc; padding: 6px 10px; font-weight: 500;">${cand[c] ?? ''}</td>
                `).join('')}
                ${objectiveCols.map(c => `
                    <td contenteditable="true" class="tbo-obj-cell editable-cell" style="background:#f0fdf4; border: 1px dashed #22c55e; padding: 6px 10px; min-width: 80px;" data-col="${c}"></td>
                `).join('')}
                ${priorCols.map(c => `
                    <td contenteditable="true" class="tbo-prior-cell editable-cell" style="background:#fdf4ff; border: 1px dashed #d946ef; padding: 6px 10px; color: #581c87;">${window.priorConfigs[c]?.target_value || ''}</td>
                `).join('')}
                ${otherCols.map(() => `
                    <td contenteditable="true" class="tbo-other-cell editable-cell" style="background:#f8fafc; border: 1px dashed #cbd5e1; padding: 6px 10px; color: #64748b;">—</td>
                `).join('')}
            `;
            body.appendChild(row);
        });
    }

    // Legend
    const legendEl = document.getElementById('tbo-results-legend');
    if (legendEl) {
        legendEl.innerHTML = `
            <span style="display:inline-flex;align-items:center;gap:4px;margin-right:1rem;">
                <span style="width:10px;height:10px;background:#f5f3ff;border:1px solid #7c3aed;border-radius:2px;display:inline-block;"></span> Feature (suggested, editable)
            </span>
            <span style="display:inline-flex;align-items:center;gap:4px;margin-right:1rem;">
                <span style="width:10px;height:10px;background:#f0fdf4;border:1px solid #22c55e;border-radius:2px;display:inline-block;"></span> Objective — <b>enter measured values to enable commit</b>
            </span>
            <span style="display:inline-flex;align-items:center;gap:4px;margin-right:1rem;opacity:0.6;">
                <span style="width:10px;height:10px;background:#f8f8f8;border:1px solid #ccc;border-radius:2px;display:inline-block;"></span> Prior / Other (editable)
            </span>
        `;
    }

    // Global completeness checker
    window._tboCheckCommitReady = function () {
        const objCells = document.querySelectorAll('#tbo-results-body .tbo-obj-cell');
        let allFilled = objCells.length > 0;
        objCells.forEach(td => {
            const val = td.textContent.trim();
            if (val === '' || isNaN(parseFloat(val))) allFilled = false;
        });
        const commitBtn = document.getElementById('tbo-commit-suggestions-btn');
        if (commitBtn) {
            commitBtn.disabled = !allFilled;
            commitBtn.title = allFilled ? '' : 'Fill in all objective values before committing';
            commitBtn.style.opacity = allFilled ? '1' : '0.45';
            commitBtn.style.cursor  = allFilled ? 'pointer' : 'not-allowed';
        }
    };

    // Attach input & blur events to all editable cells
    body.querySelectorAll('.editable-cell').forEach(cell => {
        cell.addEventListener('input', window._tboCheckCommitReady);
        cell.addEventListener('blur', window._tboCheckCommitReady);
    });

    // Check initial commit readiness
    window._tboCheckCommitReady();

    section.scrollIntoView({ behavior: 'smooth' });
}
window.renderTBOResults = renderTBOResults;

/**
 * Exports the TBO suggestions table (all columns) as CSV.
 * Reads the live edited cell contents from the DOM.
 */
function exportTBOSuggestions() {
    const rowsEl = document.querySelectorAll('#tbo-results-body tr');
    if (rowsEl.length === 0) return;

    const featureCols   = window.currentColumns.filter(c => window.columnRoles[c] === 'feature' && (window.boFeatureConfigs[c] ? window.boFeatureConfigs[c].included !== false : true));
    const objectiveCols = window.currentColumns.filter(c => window.columnRoles[c] === 'objective' && (window.objectiveConfigs[c] ? window.objectiveConfigs[c].included !== false : true));
    const priorCols     = window.currentColumns.filter(c => window.columnRoles[c] === 'prior');
    const otherCols     = window.currentColumns.filter(c =>
        !featureCols.includes(c) && !objectiveCols.includes(c) && !priorCols.includes(c)
    );
    const tableCols = [...featureCols, ...objectiveCols, ...priorCols, ...otherCols];

    const escape = (v) => {
        const s = (v === null || v === undefined) ? '' : String(v);
        return s.includes(',') || s.includes('"') || s.includes('\n') ? `"${s.replace(/"/g, '""')}"` : s;
    };

    const outRows = [tableCols.map(escape).join(',')];
    rowsEl.forEach(tr => {
        const cells = Array.from(tr.querySelectorAll('td')).slice(1);
        const rowVals = tableCols.map((col, idx) => {
            const val = cells[idx] ? cells[idx].textContent.trim() : '';
            return (val === '—' || val === '-') ? '' : val;
        });
        outRows.push(rowVals.map(escape).join(','));
    });

    const csv  = outRows.join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = 'transbo_suggestions.csv';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}



/**
 * Renders Transfer Diagnostics & Plotly Weight Bar Chart
 */
export function renderTransferDiagnostics(transfer) {
    const section = document.getElementById('tbo-diagnostics-section');
    if (!section || !transfer) return;

    section.classList.remove('hidden');

    const algoName = transfer.algorithm_used === 'fixed_noise_gp' ? 'Heteroscedastic Fixed-Noise GP' :
                     transfer.algorithm_used === 'multitask_gp' ? 'Multi-Task GP' :
                     transfer.algorithm_used === 'standard_bo' ? 'Standard BO (no Prior Targets)' : 'Cold-Start Sobol';

    let statsHtml = `
        <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem;">
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.75rem 1.25rem; border-radius: 10px; flex: 1; min-width: 200px;">
                <span style="font-size: 0.8rem; color: #64748b; display: block; font-weight: 500;">Surrogate Engine</span>
                <b style="color: #7c3aed; font-size: 1rem;">${algoName}</b>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.75rem 1.25rem; border-radius: 10px; flex: 1; min-width: 160px;">
                <span style="font-size: 0.8rem; color: #64748b; display: block; font-weight: 500;">Eligible Prior Rows</span>
                <b style="color: var(--text-primary); font-size: 1rem;">${transfer.effective_rows ?? 0} / ${transfer.total_rows ?? 0}</b>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.75rem 1.25rem; border-radius: 10px; flex: 1; min-width: 180px;">
                <span style="font-size: 0.8rem; color: #64748b; display: block; font-weight: 500;">Rows Used for Exact GP</span>
                <b style="color: #7c3aed; font-size: 1rem;">${transfer.model_rows_used ?? transfer.total_rows ?? 0}</b>
                <span style="font-size:0.75rem;color:#64748b;display:block;">Requested max: ${transfer.requested_gp_rows ?? 'N/A'}${transfer.eligible_rows !== undefined ? ` · ${transfer.eligible_rows} satisfy the Prior rules` : ''}</span>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.75rem 1.25rem; border-radius: 10px; flex: 1; min-width: 140px;">
                <span style="font-size: 0.8rem; color: #64748b; display: block; font-weight: 500;">RDKit Fingerprints</span>
                <b style="color: #10b981; font-size: 1rem;">✓ Active</b>
            </div>
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 0.75rem 1.25rem; border-radius: 10px; flex: 1; min-width: 140px;">
                <span style="font-size: 0.8rem; color: #64748b; display: block; font-weight: 500;">Combination Mode</span>
                <b style="color: var(--text-primary); font-size: 1rem; text-transform: capitalize;">${transfer.prior_combine_rule || 'Multiply'}</b>
            </div>
        </div>
    `;

    if (transfer.recovery?.used_reduced_rows) {
        statsHtml += `
            <div style="margin:-0.5rem 0 1.25rem;padding:0.8rem 1rem;border-radius:8px;background:#fff7ed;border:1px solid #fdba74;color:#9a3412;font-size:0.88rem;">
                <b>⚠ Reduced-row recovery used.</b> The effective exact-GP fit of ${transfer.recovery.requested_gp_rows} eligible rows did not complete. This result was recovered using ${transfer.model_rows_used} training rows (configured maximum: ${transfer.recovery.configured_max_gp_rows ?? transfer.requested_gp_rows}; retry limit: ${transfer.recovery.retry_max_gp_rows}).
            </div>
        `;
    }

    // Prior columns breakdown table
    if (transfer.weight_stats && Object.keys(transfer.weight_stats).length > 0) {
        statsHtml += `
            <h4 style="margin-bottom: 0.75rem; color: var(--text-primary); font-size: 0.95rem;">Prior Columns Alignment Summary</h4>
            <div class="table-container" style="margin-bottom: 1.5rem;">
                <table class="setup-table">
                    <thead>
                        <tr>
                            <th>Prior Column</th>
                            <th>Type</th>
                            <th>Target Value</th>
                            <th>Acceptance Rule</th>
                            <th>Min Weight</th>
                            <th>Mean Weight</th>
                            <th>Max Weight</th>
                            <th>Eligible Rows</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${Object.entries(transfer.weight_stats).map(([col, s]) => `
                            <tr>
                                <td style="font-weight: 600;">${col}</td>
                                <td style="text-transform: uppercase; font-size: 0.8rem; color: #64748b;">${s.col_type}</td>
                                <td style="font-family: monospace; font-size: 0.85rem;">${s.target_value}</td>
                                <td>${s.col_type === 'numerical' ? `± ${formatPriorNumber(Number(s.tolerance))}` : (s.col_type === 'smiles' ? `≥ ${Math.round(Number(s.min_similarity) * 100)}%` : 'Exact match')}</td>
                                <td>${s.min}</td>
                                <td style="font-weight: 600; color: #7c3aed;">${s.mean}</td>
                                <td>${s.max}</td>
                                <td><b>${s.n_eligible ?? s.n_effective}</b> / ${s.total_rows}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }

    const panel = document.getElementById('tbo-diagnostics-panel');
    if (panel) panel.innerHTML = statsHtml;

    // Render Plotly Bar Chart of combined row weights
    if (transfer.combined_weights && transfer.combined_weights.length > 0 && window.Plotly) {
        const xLabels = transfer.source_labels || transfer.combined_weights.map((_, i) => `Row ${i+1}`);
        const weights = transfer.combined_weights;

        const trace = {
            x: xLabels,
            y: weights,
            type: 'bar',
            marker: {
                color: weights.map(w => {
                    // Smooth gradient from slate grey to vibrant violet
                    if (w < 0.1) return '#94a3b8';
                    if (w < 0.5) return '#a855f7';
                    return '#7c3aed';
                })
            },
            hovertemplate: '<b>%{x}</b><br>Transfer Weight: %{y:.4f}<extra></extra>'
        };

        const layout = {
            title: {
                text: 'Historical Data Weight Distribution (Combined Transfer Weights)',
                font: { size: 14, family: 'Inter, sans-serif' }
            },
            xaxis: { title: 'Dataset Row', tickangle: -45 },
            yaxis: { title: 'Weight in GP [0.0 - 1.0]', range: [0, 1.05] },
            margin: { l: 50, r: 20, t: 40, b: 60 },
            plot_bgcolor: '#f8fafc',
            paper_bgcolor: '#ffffff'
        };

        window.Plotly.newPlot('tbo-weight-chart', [trace], layout, { responsive: true, displayModeBar: false });
    }
}

/**
 * Performance metrics table (R2, MSE)
 */
export function renderTBOPerformanceTable(performance) {
    const container = document.getElementById('tbo-performance-container');
    if (!container || !performance) return;

    let html = `
        <h3 style="margin-bottom: 0.75rem; font-size: 1rem; color: var(--text-primary);">Surrogate Model Fit Quality</h3>
        <p style="margin:0 0 0.75rem;font-size:0.82rem;color:var(--text-secondary);background:#f8fafc;border-left:3px solid #7c3aed;padding:8px 12px;">
            This is an <b>in-sample training-fit R²</b>. It is useful for diagnosing the fitted GP but is normally higher than SA's held-out validation R² and should not be compared directly.
        </p>
        <div class="table-container">
            <table class="setup-table">
                <thead><tr><th>Objective</th><th>Training-fit R²</th><th>Training MSE</th><th>Quality Assessment</th></tr></thead>
                <tbody>
    `;

    Object.entries(performance).forEach(([obj, metrics]) => {
        const r2 = metrics.r2;
        const color = r2 >= 0.7 ? '#10b981' : r2 >= 0.4 ? '#f59e0b' : '#ef4444';
        const rating = r2 >= 0.7 ? 'Strong Fit' : r2 >= 0.4 ? 'Moderate Fit' : 'Exploratory (Sparse Data)';

        html += `
            <tr>
                <td style="font-weight: 500;">${obj}</td>
                <td style="font-weight: 700; color: ${color};">${r2.toFixed(3)}</td>
                <td>${metrics.mse.toFixed(4)}</td>
                <td><span style="color: ${color}; font-weight: 600;">${rating}</span></td>
            </tr>
        `;
    });

    html += `</tbody></table></div>`;
    container.innerHTML = html;
}

/**
 * Point Estimator
 */
export function renderTBOEstimator() {
    const container = document.getElementById('tbo-estimator-inputs');
    if (!container) return;

    // Feature columns: selected as 'feature' and included
    const featureCols = window.currentColumns.filter(c => {
        if (window.columnRoles[c] !== 'feature') return false;
        const incBox = document.querySelector(`#tbo-features-list .tbo-feat-inc[data-col="${c}"]`);
        if (incBox) return incBox.checked;
        return window.boFeatureConfigs[c] ? window.boFeatureConfigs[c].included !== false : true;
    });

    // Prior columns: always available in point estimation
    const priorCols = window.currentColumns.filter(c => window.columnRoles[c] === 'prior');

    container.innerHTML = '';

    const allInputCols = [...featureCols, ...priorCols];
    if (allInputCols.length === 0) {
        container.innerHTML = '<p class="text-secondary" style="grid-column: 1 / -1;"><i>No features or priors selected for optimization.</i></p>';
        return;
    }

    allInputCols.forEach(col => {
        const isPrior = window.columnRoles[col] === 'prior';
        const colIdx  = window.currentColumns.indexOf(col);
        const vals    = window.currentData ? window.currentData.map(r => r[colIdx]).filter(v => v != null && v !== '') : [];
        const isNum   = vals.length > 0 && vals.every(v => !isNaN(parseFloat(v)));

        let fType = 'continuous';
        if (isPrior) {
            const pType = window.priorConfigs[col]?.col_type;
            fType = (pType === 'categorical' || pType === 'smiles' || (!isNum && !pType)) ? 'categorical' : 'continuous';
        } else {
            fType = window.boFeatureConfigs[col]?.type || (isNum ? 'continuous' : 'categorical');
        }

        const item = document.createElement('div');
        item.className = 'tweak-item';

        const labelBadge = isPrior
            ? `<span style="font-size:0.75rem; color:#7c3aed; font-weight:600; margin-left:4px;">(Prior)</span>`
            : `<span style="font-size:0.75rem; color:#2563eb; font-weight:600; margin-left:4px;">(Feature)</span>`;

        if (fType === 'categorical') {
            let uniqueVals = [...new Set(vals.map(String))].sort();
            const targetVal = isPrior ? (window.priorConfigs[col]?.target_value || '') : '';
            if (targetVal && !uniqueVals.includes(targetVal)) {
                uniqueVals.unshift(targetVal);
            }
            if (uniqueVals.length === 0 && window.boFeatureConfigs[col]?.range) {
                uniqueVals = Array.isArray(window.boFeatureConfigs[col].choices)
                    ? window.boFeatureConfigs[col].choices
                    : parseCategoricalChoices(window.boFeatureConfigs[col].range);
            }

            item.innerHTML = `
                <label>${col} ${labelBadge}</label>
                <select class="tbo-est-input" data-col="${col}" style="width: 100%;">
                    ${uniqueVals.map(c => `<option value="${c}" ${c === targetVal ? 'selected' : ''}>${c}</option>`).join('')}
                </select>
            `;
        } else {
            const numVals = vals.map(v => parseFloat(v)).filter(v => !isNaN(v));
            let defVal = numVals.length > 0 ? (numVals.reduce((a, b) => a + b, 0) / numVals.length).toFixed(3) : '';
            if (isPrior && window.priorConfigs[col]?.target_value !== undefined && window.priorConfigs[col]?.target_value !== '') {
                defVal = window.priorConfigs[col].target_value;
            }

            item.innerHTML = `
                <label>${col} ${labelBadge}</label>
                <input type="number" step="any" class="tbo-est-input" data-col="${col}" value="${defVal}" placeholder="Value" style="width: 100%;">
            `;
        }
        container.appendChild(item);
    });
}

export async function runTransferBOEstimate() {
    const inputs = {};
    document.querySelectorAll('#tbo-estimator-inputs .tbo-est-input').forEach(inp => {
        inputs[inp.dataset.col] = inp.value;
    });

    const featureCols = window.currentColumns.filter(c => {
        if (window.columnRoles[c] !== 'feature') return false;
        const incBox = document.querySelector(`#tbo-features-list .tbo-feat-inc[data-col="${c}"]`);
        if (incBox) return incBox.checked;
        return window.boFeatureConfigs[c] ? window.boFeatureConfigs[c].included !== false : true;
    });
    const objectiveCols = window.currentColumns.filter(c => window.columnRoles[c] === 'objective' && window.objectiveConfigs[c]?.included);
    const priorCols = window.currentColumns.filter(c => window.columnRoles[c] === 'prior');
    let objectiveConstraints;
    try {
        objectiveConstraints = window.collectObjectiveConstraints?.('tbo') || [];
    } catch (error) {
        alert(error.message);
        return;
    }

    const featuresPayload = featureCols.map(c => {
        const cfg = window.boFeatureConfigs[c] || {};
        return buildTBOFeaturePayload(c, cfg);
    });
    const objectivesPayload = objectiveCols.map(c => ({
        name: c,
        type: window.objectiveConfigs[c]?.type || 'maximize',
        target: window.objectiveConfigs[c]?.target || '',
        importance: window.objectiveConfigs[c]?.importance || 1
    }));
    const priorsPayload = buildPriorPayload(priorCols).map(prior => ({
        ...prior,
        target_value: inputs[prior.name] !== undefined
            ? inputs[prior.name]
            : prior.target_value
    }));
    const tweaksPayload = {
        kernel: document.getElementById('tbo-kernel-select')?.value || 'matern52',
        max_gp_rows: parseInt(document.getElementById('tbo-prior-cap-slider')?.value || '3'),
        allow_reduced_row_recovery: document.getElementById('tbo-reduced-row-recovery-toggle')?.checked || false,
        objective_constraints: objectiveConstraints
    };
    const priorCombine = window.priorCombineRule || 'multiply';
    const trainingContext = tboTrainingContextKey({
        features: featuresPayload,
        objectives: objectivesPayload,
        priors: priorsPayload,
        priorCombine,
        tweaks: tweaksPayload
    });
    if (reusableTBOTraining?.context === trainingContext) {
        tweaksPayload.reuse_training_row_indices = reusableTBOTraining.sourceRows;
    }

    const payload = {
        data: window.currentData,
        columns: window.currentColumns,
        inputs: inputs,
        features: featuresPayload,
        objectives: objectivesPayload,
        priors: priorsPayload,
        prior_combine: priorCombine,
        tweaks: tweaksPayload
    };

    const estimateBtn = document.getElementById('tbo-estimate-btn');
    const progressContainer = document.getElementById('tbo-estimate-progress-container');
    const statusText = document.getElementById('tbo-estimate-status-text');
    const timerText = document.getElementById('tbo-estimate-timer');
    const progressBar = document.getElementById('tbo-estimate-progress-bar');
    const resultsBox = document.getElementById('tbo-estimate-results');
    const abortButton = document.getElementById('tbo-estimate-abort-btn');

    if (estimateBtn) estimateBtn.disabled = true;
    if (abortButton) abortButton.disabled = false;
    if (progressContainer) progressContainer.classList.remove('hidden');
    if (resultsBox) resultsBox.innerHTML = '';

    let elapsedSec = 0;
    const startTime = Date.now();
    const tickerInterval = setInterval(() => {
        elapsedSec = Math.round((Date.now() - startTime) / 1000);
        if (timerText) timerText.textContent = `${elapsedSec}s`;

        if (elapsedSec < 3) {
            if (statusText) statusText.textContent = 'Parsing dataset & computing transfer weights...';
            if (progressBar) progressBar.style.width = '25%';
        } else if (elapsedSec < 9) {
            if (statusText) statusText.textContent = 'Fitting Gaussian Process surrogate on prior observations...';
            if (progressBar) progressBar.style.width = '60%';
        } else if (elapsedSec < 18) {
            if (statusText) statusText.textContent = 'Evaluating posterior mean and predictive variance...';
            if (progressBar) progressBar.style.width = '85%';
        } else {
            if (statusText) statusText.textContent = 'Computing final confidence bounds (large dataset)...';
            if (progressBar) progressBar.style.width = '95%';
        }
    }, 500);

    const estimateController = new AbortController();
    if (abortButton) abortButton.onclick = () => estimateController.abort();

    try {
        const completed = await submitAndPollJob('/transfer_estimate_submit', payload, {
            signal: estimateController.signal,
            onProgress: (status, jobId) => {
                activeTBOEstimateJobId = jobId;
                if (statusText && status.phase) statusText.textContent = status.phase;
                if (progressBar && Number.isFinite(Number(status.percent))) {
                    progressBar.style.width = `${Math.max(10, Math.min(98, Number(status.percent)))}%`;
                }
            }
        });
        const data = completed.result;
        if (resultsBox && data.results) {
            const violationMap = new Map((data.violations || []).map(item => [item.objective, item]));
            const workload = data.workload;
            const workloadNotice = workload && workload.rows_downsampled > 0
                ? `<div style="margin-bottom:0.8rem;color:#6b21a8;font-size:0.82rem;">Used ${workload.model_rows_used} highest-relevance prior rows (requested maximum ${workload.requested_gp_rows}) for exact GP fitting.</div>`
                : '';
            const recoveryNotice = workload?.recovery?.used_reduced_rows
                ? `<div style="margin-bottom:0.8rem;padding:0.75rem 0.9rem;border-radius:8px;background:#fff7ed;border:1px solid #fdba74;color:#9a3412;font-size:0.85rem;"><b>⚠ Reduced-row recovery used.</b> The effective exact-GP fit of ${workload.recovery.requested_gp_rows} eligible rows did not complete. This estimate uses ${workload.model_rows_used} training rows (configured maximum: ${workload.recovery.configured_max_gp_rows ?? workload.requested_gp_rows}; retry limit: ${workload.recovery.retry_max_gp_rows}).</div>`
                : '';
            const reusedRowsNotice = workload?.reused_training_rows
                ? `<div style="margin-bottom:0.8rem;color:#166534;font-size:0.82rem;">Using the same ${workload.model_rows_used} exact GP training rows as the completed optimization.</div>`
                : '';
            resultsBox.innerHTML = `
                <div style="margin-bottom:0.8rem;font-weight:700;">Estimated Global Success: <span style="color:#7c3aed;">${data.global_success}%</span></div>
                ${workloadNotice}
                ${recoveryNotice}
                ${reusedRowsNotice}
                <div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
                    ${Object.entries(data.results).map(([obj, vals]) => {
                        const violation = violationMap.get(obj);
                        return `
                        <div style="background: white; border: 1px solid #cbd5e1; padding: 1rem 1.5rem; border-radius: 10px; flex: 1; min-width: 180px;">
                            <span style="font-size: 0.85rem; color: #64748b; font-weight: 500;">${obj}</span>
                            <div style="font-size: 1.4rem; font-weight: 700; color: #7c3aed; margin-top: 0.25rem;">
                                ${vals.mean} <span style="font-size: 0.9rem; font-weight: normal; color: #94a3b8;">± ${vals.std}</span>
                            </div>
                            <div style="font-size:0.85rem;margin-top:0.35rem;color:${vals.success > 80 ? '#15803d' : (vals.success > 50 ? '#b45309' : '#b91c1c')};">Success: ${vals.success}%</div>
                            ${violation ? `<div title="Predicted ${violation.predicted ?? 'N/A'} violates ${violation.operator} ${violation.limit}" style="margin-top:0.35rem;color:#b91c1c;font-weight:700;">🚫 Violated: ${violation.operator} ${violation.limit}</div>` : ''}
                        </div>
                    `;}).join('')}
                </div>
            `;
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            if (resultsBox) resultsBox.innerHTML = '<div style="color:#64748b;">Estimation cancelled.</div>';
        } else if (resultsBox) {
            resultsBox.innerHTML = `<div style="background:#fee2e2; border:1px solid #fca5a5; color:#991b1b; padding:0.75rem 1rem; border-radius:8px; font-size:0.9rem;"><b>Estimation error:</b> ${e.message}</div>`;
        } else {
            alert(`Estimation failed: ${e.message}`);
        }
    } finally {
        if (estimateController.signal.aborted) {
            await cancelBackgroundJob(activeTBOEstimateJobId);
        }
        activeTBOEstimateJobId = null;
        clearInterval(tickerInterval);
        if (progressContainer) progressContainer.classList.add('hidden');
        if (estimateBtn) estimateBtn.disabled = false;
        if (abortButton) abortButton.disabled = true;
    }
}

/**
 * Response Surface Visualization for TransBO
 * Supports Objectives Trend, Contour Plot, and 3D Surface Plot.
 * Uses both columns labeled 'F' and columns labeled 'P' as feature candidates.
 */
export function initTBOVisualizationDropdown() {
    const select = document.getElementById('tbo-vis-type-select');
    if (select && !select.dataset.tboInitialized) {
        select.dataset.tboInitialized = 'true';
        select.addEventListener('change', () => {
            renderTBOVisualization();
        });
    }
    const cancelButton = document.getElementById('tbo-plot-cancel-btn');
    if (cancelButton && !cancelButton.dataset.tboInitialized) {
        cancelButton.dataset.tboInitialized = 'true';
        cancelButton.addEventListener('click', () => cancelActiveTBOPlot({ showMessage: true }));
    }
}

function getTBOAllPlotFeatures() {
    return window.currentColumns.filter(c => {
        if (window.columnRoles[c] === 'prior') return true;
        if (window.columnRoles[c] === 'feature') {
            const incBox = document.querySelector(`#tbo-features-list .tbo-feat-inc[data-col="${c}"]`);
            if (incBox) return incBox.checked;
            return window.boFeatureConfigs[c] ? window.boFeatureConfigs[c].included !== false : true;
        }
        return false;
    });
}

function getTBOFeatureType(featName) {
    if (window.columnRoles[featName] === 'prior') {
        const pType = window.priorConfigs[featName]?.col_type;
        if (pType === 'continuous' || pType === 'discrete' || pType === 'numerical') return 'continuous';
        if (pType === 'categorical' || pType === 'smiles') return 'categorical';
        const colIdx = window.currentColumns.indexOf(featName);
        const vals = window.currentData ? window.currentData.map(r => r[colIdx]).filter(v => v != null && v !== '') : [];
        return (vals.length > 0 && vals.every(v => !isNaN(parseFloat(v)))) ? 'continuous' : 'categorical';
    }
    if (window.boFeatureConfigs && window.boFeatureConfigs[featName]) {
        return window.boFeatureConfigs[featName].type || 'continuous';
    }
    const colIdx = window.currentColumns.indexOf(featName);
    const vals = window.currentData ? window.currentData.map(r => r[colIdx]).filter(v => v != null && v !== '') : [];
    return (vals.length > 0 && vals.every(v => !isNaN(parseFloat(v)))) ? 'continuous' : 'categorical';
}

function getTBOFeatureCategoricalValues(featName) {
    const colIdx = window.currentColumns.indexOf(featName);
    if (colIdx < 0) return [];
    const seen = new Set();
    if (window.currentData) {
        window.currentData.forEach(r => {
            if (r[colIdx] != null && r[colIdx] !== '') seen.add(String(r[colIdx]));
        });
    }
    return Array.from(seen).sort();
}

function buildTBOFixedInput(f, cssClass) {
    const colIdx = window.currentColumns.indexOf(f);
    const fType  = getTBOFeatureType(f);
    const vals   = window.currentData ? window.currentData.map(r => r[colIdx]).filter(v => v != null && v !== '') : [];

    if (fType === 'categorical') {
        const options = getTBOFeatureCategoricalValues(f);
        const optHtml = options.map(v => `<option value="${v}">${v}</option>`).join('');
        return `<div class="tweak-item"><label>${f}: <span style="font-size:0.75rem;color:var(--text-secondary);">(categorical)</span></label>
                    <select class="${cssClass}" data-feat="${f}" style="width:100%;">${optHtml}</select></div>`;
    } else {
        const numVals   = vals.map(v => parseFloat(v)).filter(v => !isNaN(v));
        const defaultVal = numVals.length > 0 ? (numVals.reduce((a, b) => a + b, 0) / numVals.length).toFixed(3) : '';
        return `<div class="tweak-item"><label>${f}:</label>
                    <input type="number" class="${cssClass}" data-feat="${f}" value="${defaultVal}" step="any" style="width:100%;"></div>`;
    }
}

export function renderTBOVisualization() {
    const section = document.getElementById('tbo-visualization-section');
    if (!section) return;

    // Any setup/role change invalidates an in-flight surface request.  Expected
    // cancellation is silent; a new valid request will be started below.
    cancelActiveTBOPlot();

    if (!window.currentData || window.currentData.length === 0) {
        try {
            Plotly.purge('tbo-trend-plot');
            const contourEl = document.getElementById('tbo-contour-plot');
            if (contourEl) { Plotly.purge(contourEl); contourEl.innerHTML = ''; }
            const surfEl = document.getElementById('tbo-surface-plot');
            if (surfEl) { Plotly.purge(surfEl); surfEl.innerHTML = ''; }
        } catch(e) {}
        section.classList.add('hidden');
        return;
    }

    section.classList.remove('hidden');
    initTBOVisualizationDropdown();

    const viewType = document.getElementById('tbo-vis-type-select')?.value || 'trend';
    const trendPlotEl   = document.getElementById('tbo-trend-plot');
    const contourPlotEl = document.getElementById('tbo-contour-plot');
    const surfacePlotEl = document.getElementById('tbo-surface-plot');
    const controlsEl    = document.getElementById('tbo-vis-controls');
    const titleEl       = document.getElementById('tbo-vis-title');

    if (viewType === 'trend') {
        if (titleEl) titleEl.textContent = 'TransBO - Objectives Trend';
        if (trendPlotEl) trendPlotEl.style.display = 'block';
        if (contourPlotEl) { contourPlotEl.style.display = 'none'; try { Plotly.purge(contourPlotEl); } catch(e) {} contourPlotEl.innerHTML = ''; }
        if (surfacePlotEl) { surfacePlotEl.style.display = 'none'; try { Plotly.purge(surfacePlotEl); } catch(e) {} surfacePlotEl.innerHTML = ''; }
        if (controlsEl) { controlsEl.style.display = 'none'; controlsEl.innerHTML = ''; }
        renderTBOTrendPlotOnly();
    } else if (viewType === 'contour') {
        if (titleEl) titleEl.textContent = 'TransBO - Contour Plot (Response Surface)';
        if (trendPlotEl) { trendPlotEl.style.display = 'none'; try { Plotly.purge(trendPlotEl); } catch(e) {} }
        if (surfacePlotEl) { surfacePlotEl.style.display = 'none'; try { Plotly.purge(surfacePlotEl); } catch(e) {} surfacePlotEl.innerHTML = ''; }
        if (contourPlotEl) contourPlotEl.style.display = 'block';
        const isValid = renderTBOContourControls();
        if (isValid) {
            updateTBOContourPlot();
        }
    } else if (viewType === 'surface') {
        if (titleEl) titleEl.textContent = 'TransBO - 3D Surface Plot (Response Surface)';
        if (trendPlotEl) { trendPlotEl.style.display = 'none'; try { Plotly.purge(trendPlotEl); } catch(e) {} }
        if (contourPlotEl) { contourPlotEl.style.display = 'none'; try { Plotly.purge(contourPlotEl); } catch(e) {} contourPlotEl.innerHTML = ''; }
        if (surfacePlotEl) surfacePlotEl.style.display = 'block';
        const isValid = renderTBOSurfaceControls();
        if (isValid) {
            updateTBOSurfacePlot();
        }
    }
}

function renderTBOTrendPlotOnly() {
    const trendEl = document.getElementById('tbo-trend-plot');
    if (!trendEl) return;

    if (!window.currentColumns || !window.currentData || window.currentData.length === 0) {
        try { Plotly.purge(trendEl); } catch(e) {}
        trendEl.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-secondary);"><i>No data loaded to display trend.</i></div>';
        return;
    }

    const objCols = window.currentColumns.filter(c =>
        window.columnRoles[c] === 'objective' &&
        (window.objectiveConfigs && window.objectiveConfigs[c] ? window.objectiveConfigs[c].included !== false : true)
    );

    if (objCols.length === 0) {
        try { Plotly.purge(trendEl); } catch(e) {}
        trendEl.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-secondary);"><i>No included objectives to display trend.</i></div>';
        return;
    }

    const traces = objCols.map(col => {
        const colIdx = window.currentColumns.indexOf(col);
        const xDist  = [];
        const yDist  = [];

        window.currentData.forEach((row, i) => {
            const rawVal = row[colIdx];
            const parsed = parseFloat(rawVal);
            if (rawVal !== '' && rawVal !== null && rawVal !== undefined && !isNaN(parsed)) {
                xDist.push(i + 1);
                yDist.push(parsed);
            }
        });

        const isLarge = xDist.length > 1000;
        return {
            x: xDist,
            y: yDist,
            name: col,
            mode: isLarge ? 'lines' : 'lines+markers',
            line: { width: isLarge ? 2 : 3 },
            marker: isLarge ? {} : { size: 6 }
        };
    }).filter(t => t.x.length > 0);

    if (traces.length === 0) {
        try { Plotly.purge(trendEl); } catch(e) {}
        trendEl.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-secondary);"><i>Selected objective column(s) contain no numeric data to display trend.</i></div>';
        return;
    }

    try { Plotly.purge(trendEl); } catch(e) {}
    trendEl.innerHTML = '';

    Plotly.newPlot('tbo-trend-plot', traces, {
        title:    objCols.length === 1 ? `Trend: ${objCols[0]}` : 'TransBO - Objectives Trend',
        autosize: true,
        height:   400,
        margin:   { l: 60, r: 40, t: 80, b: 60 },
        legend:   { orientation: 'h', y: -0.2 }
    }, { responsive: true });

    setTimeout(() => { try { Plotly.Plots.resize('tbo-trend-plot'); } catch(e) {} }, 100);
}

function renderTBOContourControls() {
    const controls = document.getElementById('tbo-vis-controls');
    if (!controls) return false;

    const allFeatCols = getTBOAllPlotFeatures();
    const numFeatCols = allFeatCols.filter(f => getTBOFeatureType(f) !== 'categorical');
    const catFeatCols = allFeatCols.filter(f => getTBOFeatureType(f) === 'categorical');
    const objCols     = window.currentColumns.filter(c => window.columnRoles[c] === 'objective' && (window.objectiveConfigs[c] ? window.objectiveConfigs[c].included !== false : true));

    if (numFeatCols.length < 2 || objCols.length === 0) {
        controls.innerHTML = `<p class="text-secondary" style="grid-column: 1 / -1; margin: 0.5rem 0;"><i>Contour plot requires at least 2 included <b>numerical</b> features/priors and 1 objective.${
            catFeatCols.length > 0 ? ' Categorical features/priors can only be set as fixed values.' : ''}</i></p>`;
        controls.style.display = 'grid';
        const contourPlotEl = document.getElementById('tbo-contour-plot');
        if (contourPlotEl) {
            try { Plotly.purge(contourPlotEl); } catch(e) {}
            contourPlotEl.innerHTML = '';
        }
        return false;
    }

    const currentX   = document.getElementById('tbo-contour-x')?.value;
    const currentY   = document.getElementById('tbo-contour-y')?.value;
    const currentObj = document.getElementById('tbo-contour-obj')?.value;

    const xSelected   = numFeatCols.includes(currentX) ? currentX : numFeatCols[0];
    const ySelected   = numFeatCols.includes(currentY) && currentY !== xSelected ? currentY : (numFeatCols.find(f => f !== xSelected) || numFeatCols[0]);
    const objSelected = objCols.includes(currentObj)   ? currentObj : objCols[0];

    let html = `
        <div class="tweak-item">
            <label>X-Axis Feature (numerical):</label>
            <select id="tbo-contour-x" style="width: 100%;">
                ${numFeatCols.map(f => `<option value="${f}" ${f === xSelected ? 'selected' : ''}>${f}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Y-Axis Feature (numerical):</label>
            <select id="tbo-contour-y" style="width: 100%;">
                ${numFeatCols.map(f => `<option value="${f}" ${f === ySelected ? 'selected' : ''}>${f}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Objective (Color &amp; Contours):</label>
            <select id="tbo-contour-obj" style="width: 100%;">
                ${objCols.map(o => `<option value="${o}" ${o === objSelected ? 'selected' : ''}>${o}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Grid Resolution:</label>
            <select id="tbo-contour-res" style="width: 100%;">
                <option value="15">15 &#215; 15 (Fast)</option>
                <option value="25" selected>25 &#215; 25 (Standard)</option>
                <option value="40">40 &#215; 40 (High)</option>
            </select>
        </div>
    `;

    const fixedFeats = allFeatCols.filter(f => f !== xSelected && f !== ySelected);
    if (fixedFeats.length > 0) {
        html += `<div style="grid-column: 1 / -1; margin-top: 0.5rem;"><b style="font-size: 0.85rem; color: var(--text-secondary);">Fixed values for remaining features/priors:</b></div>`;
        fixedFeats.forEach(f => { html += buildTBOFixedInput(f, 'tbo-contour-fixed'); });
    }

    html += `<div style="grid-column: 1 / -1; display: flex; justify-content: flex-end; margin-top: 0.5rem;">
                <button id="tbo-contour-update-btn" class="btn primary small">Calculate &amp; Update Contour</button>
             </div>`;

    controls.innerHTML = html;
    controls.style.display = 'grid';

    document.getElementById('tbo-contour-update-btn')?.addEventListener('click', updateTBOContourPlot);
    document.getElementById('tbo-contour-x')?.addEventListener('change', () => { renderTBOContourControls(); updateTBOContourPlot(); });
    document.getElementById('tbo-contour-y')?.addEventListener('change', () => { renderTBOContourControls(); updateTBOContourPlot(); });
    document.getElementById('tbo-contour-obj')?.addEventListener('change', updateTBOContourPlot);
    return true;
}

function renderTBOSurfaceControls() {
    const controls = document.getElementById('tbo-vis-controls');
    if (!controls) return false;

    const allFeatCols = getTBOAllPlotFeatures();
    const numFeatCols = allFeatCols.filter(f => getTBOFeatureType(f) !== 'categorical');
    const catFeatCols = allFeatCols.filter(f => getTBOFeatureType(f) === 'categorical');
    const objCols     = window.currentColumns.filter(c => window.columnRoles[c] === 'objective' && (window.objectiveConfigs[c] ? window.objectiveConfigs[c].included !== false : true));

    if (numFeatCols.length < 2 || objCols.length === 0) {
        controls.innerHTML = `<p class="text-secondary" style="grid-column: 1 / -1; margin: 0.5rem 0;"><i>3D surface plot requires at least 2 included <b>numerical</b> features/priors and 1 objective.${
            catFeatCols.length > 0 ? ' Categorical features/priors can only be set as fixed values.' : ''}</i></p>`;
        controls.style.display = 'grid';
        const surfacePlotEl = document.getElementById('tbo-surface-plot');
        if (surfacePlotEl) {
            try { Plotly.purge(surfacePlotEl); } catch(e) {}
            surfacePlotEl.innerHTML = '';
        }
        return false;
    }

    const currentX     = document.getElementById('tbo-surf-x')?.value;
    const currentY     = document.getElementById('tbo-surf-y')?.value;
    const currentZ     = document.getElementById('tbo-surf-z')?.value;
    const currentColor = document.getElementById('tbo-surf-color')?.value;
    const currentRes   = document.getElementById('tbo-surf-res')?.value || '15';

    const xSelected     = numFeatCols.includes(currentX) ? currentX : numFeatCols[0];
    const ySelected     = numFeatCols.includes(currentY) && currentY !== xSelected ? currentY : (numFeatCols.find(f => f !== xSelected) || numFeatCols[0]);
    const zFeatCols     = numFeatCols.filter(f => f !== xSelected && f !== ySelected);
    const validZOptions = [...objCols, ...zFeatCols];
    const zSelected     = validZOptions.includes(currentZ) ? currentZ : objCols[0];
    const colorSelected = objCols.includes(currentColor) ? currentColor : objCols[0];

    let zOptionsHtml = `<optgroup label="Objectives">`;
    objCols.forEach(o => {
        zOptionsHtml += `<option value="${o}" ${o === zSelected ? 'selected' : ''}>${o} (Objective)</option>`;
    });
    zOptionsHtml += `</optgroup>`;
    if (zFeatCols.length > 0) {
        zOptionsHtml += `<optgroup label="Features / Priors">`;
        zFeatCols.forEach(f => {
            zOptionsHtml += `<option value="${f}" ${f === zSelected ? 'selected' : ''}>${f}</option>`;
        });
        zOptionsHtml += `</optgroup>`;
    }

    let html = `
        <div class="tweak-item">
            <label>X-Axis Feature (numerical):</label>
            <select id="tbo-surf-x" style="width: 100%;">
                ${numFeatCols.map(f => `<option value="${f}" ${f === xSelected ? 'selected' : ''}>${f}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Y-Axis Feature (numerical):</label>
            <select id="tbo-surf-y" style="width: 100%;">
                ${numFeatCols.map(f => `<option value="${f}" ${f === ySelected ? 'selected' : ''}>${f}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Z-Axis / Height (Objective or Feature/Prior):</label>
            <select id="tbo-surf-z" style="width: 100%;">
                ${zOptionsHtml}
            </select>
        </div>
        <div class="tweak-item">
            <label>Surface Color (Objective):</label>
            <select id="tbo-surf-color" style="width: 100%;">
                ${objCols.map(o => `<option value="${o}" ${o === colorSelected ? 'selected' : ''}>${o}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Grid Resolution:</label>
            <select id="tbo-surf-res" style="width: 100%;">
                <option value="15" ${currentRes === '15' ? 'selected' : ''}>15 &#215; 15 (Smooth 3D)</option>
                <option value="20" ${currentRes === '20' ? 'selected' : ''}>20 &#215; 20 (Detailed)</option>
                <option value="30" ${currentRes === '30' ? 'selected' : ''}>30 &#215; 30 (Ultra)</option>
            </select>
        </div>
    `;

    const fixedFeats = allFeatCols.filter(f => f !== xSelected && f !== ySelected);
    if (fixedFeats.length > 0) {
        html += `<div style="grid-column: 1 / -1; margin-top: 0.5rem;"><b style="font-size: 0.85rem; color: var(--text-secondary);">Fixed values for remaining features/priors:</b></div>`;
        fixedFeats.forEach(f => { html += buildTBOFixedInput(f, 'tbo-surf-fixed'); });
    }

    html += `<div style="grid-column: 1 / -1; display: flex; justify-content: flex-end; margin-top: 0.5rem;">
                <button id="tbo-surf-update-btn" class="btn primary small">Calculate &amp; Update 3D Surface</button>
             </div>`;

    controls.innerHTML = html;
    controls.style.display = 'grid';

    document.getElementById('tbo-surf-update-btn')?.addEventListener('click', updateTBOSurfacePlot);
    document.getElementById('tbo-surf-x')?.addEventListener('change', () => { renderTBOSurfaceControls(); updateTBOSurfacePlot(); });
    document.getElementById('tbo-surf-y')?.addEventListener('change', () => { renderTBOSurfaceControls(); updateTBOSurfacePlot(); });
    document.getElementById('tbo-surf-z')?.addEventListener('change', updateTBOSurfacePlot);
    document.getElementById('tbo-surf-color')?.addEventListener('change', updateTBOSurfacePlot);
    return true;
}

function setTBOPlotCancelEnabled(enabled) {
    const button = document.getElementById('tbo-plot-cancel-btn');
    if (!button) return;
    button.disabled = !enabled;
    button.classList.toggle('hidden', !enabled);
}

async function cancelActiveTBOPlot({ showMessage = false } = {}) {
    const controller = window.tboSurfaceAbortController;
    const jobId = activeTBOSurfaceJobId;
    window.tboPlotRequestVersion = (window.tboPlotRequestVersion || 0) + 1;
    if (controller) controller.abort();
    window.tboSurfaceAbortController = null;
    setTBOPlotCancelEnabled(false);
    if (showMessage) {
        const plot = document.getElementById('tbo-vis-type-select')?.value === 'surface'
            ? document.getElementById('tbo-surface-plot')
            : document.getElementById('tbo-contour-plot');
        if (plot) plot.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#64748b;font-style:italic;">Stopping plot calculation…</div>';
    }
    if (jobId && !activeTBOSurfaceCancellation) {
        const cancellation = cancelBackgroundJob(jobId, { waitForTerminal: true })
            .finally(() => {
                if (activeTBOSurfaceCancellation === cancellation) {
                    activeTBOSurfaceCancellation = null;
                    if (activeTBOSurfaceJobId === jobId) activeTBOSurfaceJobId = null;
                }
            });
        activeTBOSurfaceCancellation = cancellation;
    }
    if (activeTBOSurfaceCancellation) await activeTBOSurfaceCancellation;
    if (showMessage) {
        const plot = document.getElementById('tbo-vis-type-select')?.value === 'surface'
            ? document.getElementById('tbo-surface-plot')
            : document.getElementById('tbo-contour-plot');
        if (plot) plot.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#64748b;font-style:italic;">Plot calculation cancelled.</div>';
    }
}

function buildTBOSurfaceFeatures() {
    return getTBOAllPlotFeatures().map(c => {
        const colIdx = window.currentColumns.indexOf(c);
        const vals = window.currentData ? window.currentData.map(r => r[colIdx]).filter(v => v != null && v !== '') : [];
        const isNum = vals.length > 0 && vals.every(v => !isNaN(parseFloat(v)));
        const min = vals.length ? Math.min(...vals.map(v => parseFloat(v))) : 0;
        const max = vals.length ? Math.max(...vals.map(v => parseFloat(v))) : 1;
        const choices = isNum ? undefined : uniqueCategoricalChoices(vals);
        const range = isNum ? `[${min}, ${max}]` : formatCategoricalChoices(choices);
        const existing = window.boFeatureConfigs?.[c] || {};
        return buildTBOFeaturePayload(c, {
            type: getTBOFeatureType(c),
            range: existing.range || range,
            choices: existing.choices || choices
        });
    });
}

function buildTBOPriorsPayload() {
    const columns = window.currentColumns.filter(c => window.columnRoles[c] === 'prior');
    return buildPriorPayload(columns);
}

function hasConfiguredTBOPriorTarget() {
    return buildTBOPriorsPayload().some(prior => {
        const target = prior.target_value;
        // A numerical target of zero is valid; an empty target is not.
        return target !== null && target !== undefined && String(target).trim() !== '';
    });
}

function renderTBOPriorPlotPrerequisite(controls, plotContainer, plotName) {
    if (!controls) return;
    controls.innerHTML = `<div style="grid-column:1 / -1;margin:0.5rem 0;padding:1rem;border:1px solid #c4b5fd;border-radius:8px;background:#faf5ff;color:#5b21b6;line-height:1.5;"><b>${plotName} is waiting for transfer conditions.</b> In <b>Prior Targets</b>, designate at least one Prior column and select or enter its target value. TransBO will then fit the plot with rows filtered by those transfer conditions.</div>`;
    controls.style.display = 'grid';
    if (plotContainer) {
        try { Plotly.purge(plotContainer); } catch (_) {}
        plotContainer.innerHTML = '';
    }
}

async function calculateTBOPlot(payload, plotContainer, message) {
    await cancelActiveTBOPlot();
    const controller = new AbortController();
    const version = (window.tboPlotRequestVersion || 0) + 1;
    window.tboPlotRequestVersion = version;
    window.tboSurfaceAbortController = controller;
    let jobId = null;
    setTBOPlotCancelEnabled(false);
    plotContainer.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;min-height:300px;color:#64748b;gap:0.75rem;"><div class="spinner" style="width:24px;height:24px;margin:0;"></div><span>${message}</span></div>`;

    try {
        const completed = await submitAndPollJob('/transfer_surface_submit', payload, {
            signal: controller.signal,
            onProgress: (status, submittedJobId) => {
                jobId = submittedJobId;
                if (version !== window.tboPlotRequestVersion) {
                    cancelBackgroundJob(submittedJobId);
                    return;
                }
                activeTBOSurfaceJobId = submittedJobId;
                setTBOPlotCancelEnabled(true);
                if (status.phase) {
                    plotContainer.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;min-height:300px;color:#64748b;gap:0.75rem;"><div class="spinner" style="width:24px;height:24px;margin:0;"></div><span>${status.phase}…</span></div>`;
                }
            }
        });
        return version === window.tboPlotRequestVersion ? completed.result : null;
    } finally {
        if (controller.signal.aborted) await cancelBackgroundJob(jobId);
        if (window.tboSurfaceAbortController === controller) {
            window.tboSurfaceAbortController = null;
            activeTBOSurfaceJobId = null;
            setTBOPlotCancelEnabled(false);
        }
    }
}

function tboOverlayLabel(label, shown, total) {
    return shown < total ? `${label} (${shown.toLocaleString()} of ${total.toLocaleString()})` : label;
}

function buildTBOExperimentTraces(result, threeDimensional) {
    const experiments = Array.isArray(result.experiments) ? result.experiments : [];
    const inSlice = experiments.filter(experiment => experiment.in_slice);
    const offSlice = experiments.filter(experiment => !experiment.in_slice);
    const overlay = result.overlay || {};
    const buildCustomData = points => points.map(point => [
        point.exp_id, point.color, point.conditions, Number(point.weight || 0).toFixed(3)
    ]);
    const buildCompactCustomData = points => points.map(point => [
        point.exp_id, point.color, Number(point.weight || 0).toFixed(3)
    ]);
    const traces = [];
    const commonHover = `<b>Experiment #%{customdata[0]}</b> [%{customdata[4]}]<br>${result.x_label}: %{x}<br>${result.y_label}: %{y}<br>${result.color_label}: %{customdata[1]}<br>Relevance: %{customdata[3]}<br>%{customdata[2]}<extra></extra>`;

    if (inSlice.length) {
        const trace = {
            x: inSlice.map(point => point.x),
            y: inSlice.map(point => point.y),
            mode: 'markers',
            type: threeDimensional ? 'scatter3d' : 'scatter',
            name: tboOverlayLabel('Transfer-Filtered In-Slice Experiments (Solid)', inSlice.length, overlay.in_slice_total || inSlice.length),
            marker: { size: threeDimensional ? 6 : 9, color: '#ef4444', line: { width: 1.5, color: '#ffffff' } },
            customdata: buildCustomData(inSlice).map(data => [...data, 'In-Slice']),
            hovertemplate: threeDimensional
                ? `<b>Experiment #%{customdata[0]}</b> [In-Slice]<br>${result.x_label}: %{x}<br>${result.y_label}: %{y}<br>${result.z_label}: %{z}<br>${result.color_label}: %{customdata[1]}<br>Relevance: %{customdata[3]}<br>%{customdata[2]}<extra></extra>`
                : commonHover,
        };
        if (threeDimensional) trace.z = inSlice.map(point => point.z);
        traces.push(trace);
    }
    if (offSlice.length) {
        const trace = {
            x: offSlice.map(point => point.x),
            y: offSlice.map(point => point.y),
            mode: 'markers',
            type: threeDimensional ? 'scatter3d' : 'scatter',
            name: tboOverlayLabel('Transfer-Filtered Experiments (Off-Slice)', offSlice.length, overlay.off_slice_total || offSlice.length),
            marker: { size: threeDimensional ? 5 : 8, color: 'rgba(239, 68, 68, 0.15)', line: { width: 1.5, color: '#ef4444' } },
            customdata: buildCompactCustomData(offSlice).map(data => [...data, 'Off-Slice']),
            hovertemplate: threeDimensional
                ? `<b>Experiment #%{customdata[0]}</b> [Off-Slice]<br>${result.x_label}: %{x}<br>${result.y_label}: %{y}<br>${result.z_label}: %{z}<br>${result.color_label}: %{customdata[1]}<br>Relevance: %{customdata[2]}<extra></extra>`
                : `<b>Experiment #%{customdata[0]}</b> [Off-Slice]<br>${result.x_label}: %{x}<br>${result.y_label}: %{y}<br>${result.color_label}: %{customdata[1]}<br>Relevance: %{customdata[2]}<extra></extra>`,
        };
        if (threeDimensional) trace.z = offSlice.map(point => point.z);
        traces.push(trace);
    }
    return traces;
}

async function updateTBOContourPlot() {
    const xFeat = document.getElementById('tbo-contour-x')?.value;
    const yFeat = document.getElementById('tbo-contour-y')?.value;
    const obj = document.getElementById('tbo-contour-obj')?.value;
    const resolution = parseInt(document.getElementById('tbo-contour-res')?.value || '25', 10);
    const plotContainer = document.getElementById('tbo-contour-plot');
    if (!xFeat || !yFeat || !obj || !plotContainer) return;
    if (!hasConfiguredTBOPriorTarget()) {
        renderTBOPriorPlotPrerequisite(
            document.getElementById('tbo-vis-controls'), plotContainer, 'Contour plot'
        );
        return;
    }
    if (xFeat === yFeat) {
        try { Plotly.purge(plotContainer); } catch (_) {}
        plotContainer.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ef4444;font-style:italic;">Please select two distinct features for X and Y axes.</div>';
        return;
    }

    const fixedValues = {};
    document.querySelectorAll('.tbo-contour-fixed').forEach(element => {
        fixedValues[element.dataset.feat] = element.tagName === 'SELECT' ? element.value : parseFloat(element.value);
    });
    const payload = {
        data: window.currentData,
        columns: window.currentColumns,
        features: buildTBOSurfaceFeatures(),
        objectives: [{ name: obj }],
        priors: buildTBOPriorsPayload(),
        prior_combine: window.priorCombineRule || 'multiply',
        x_feat: xFeat,
        y_feat: yFeat,
        color_obj: obj,
        fixed_values: fixedValues,
        resolution,
        tweaks: {
            kernel: document.getElementById('tbo-kernel-select')?.value || 'matern52',
            max_gp_rows: parseInt(document.getElementById('tbo-prior-cap-slider')?.value || '3'),
        }
    };

    try {
        const result = await calculateTBOPlot(payload, plotContainer, 'Computing response surface');
        if (!result) return;
        const contourTrace = {
            x: result.x,
            y: result.y,
            z: result.z,
            type: 'contour',
            colorscale: 'Viridis',
            contours: { coloring: 'heatmap' },
            colorbar: { title: result.color_label, len: 0.9 },
            hovertemplate: `<b>${result.x_label}:</b> %{x}<br><b>${result.y_label}:</b> %{y}<br><b>${result.color_label}:</b> %{z:.3f}<extra></extra>`,
        };
        try { Plotly.purge(plotContainer); } catch (_) {}
        plotContainer.innerHTML = '';
        Plotly.newPlot(plotContainer, [contourTrace, ...buildTBOExperimentTraces(result, false)], {
            title: `TransBO Response Surface: ${obj} vs (${xFeat}, ${yFeat})`,
            xaxis: { title: xFeat },
            yaxis: { title: yFeat },
            margin: { l: 60, r: 40, t: 50, b: 60 },
            legend: { orientation: 'h', y: -0.15 },
        }, { responsive: true });
    } catch (error) {
        if (error.name === 'AbortError') return;
        console.error('Contour plot error:', error);
        try { Plotly.purge(plotContainer); } catch (_) {}
        plotContainer.innerHTML = `<div style="color:red;padding:1rem;">Contour calculation error: ${error.message}</div>`;
    }
}

async function updateTBOSurfacePlot() {
    const xFeat = document.getElementById('tbo-surf-x')?.value;
    const yFeat = document.getElementById('tbo-surf-y')?.value;
    const zObj = document.getElementById('tbo-surf-z')?.value;
    const colorObj = document.getElementById('tbo-surf-color')?.value || zObj;
    const resolution = parseInt(document.getElementById('tbo-surf-res')?.value || '15', 10);
    const plotContainer = document.getElementById('tbo-surface-plot');
    if (!xFeat || !yFeat || !zObj || !plotContainer) return;
    if (!hasConfiguredTBOPriorTarget()) {
        renderTBOPriorPlotPrerequisite(
            document.getElementById('tbo-vis-controls'), plotContainer, '3D surface plot'
        );
        return;
    }
    if (xFeat === yFeat) {
        try { Plotly.purge(plotContainer); } catch (_) {}
        plotContainer.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ef4444;font-style:italic;">Please select two distinct features for X and Y axes.</div>';
        return;
    }

    const fixedValues = {};
    document.querySelectorAll('.tbo-surf-fixed').forEach(element => {
        fixedValues[element.dataset.feat] = element.tagName === 'SELECT' ? element.value : parseFloat(element.value);
    });
    const surfaceObjectives = [...new Set([
        colorObj,
        window.columnRoles[zObj] === 'objective' ? zObj : null,
    ].filter(Boolean))];
    const payload = {
        data: window.currentData,
        columns: window.currentColumns,
        features: buildTBOSurfaceFeatures(),
        objectives: surfaceObjectives.map(name => ({ name })),
        priors: buildTBOPriorsPayload(),
        prior_combine: window.priorCombineRule || 'multiply',
        x_feat: xFeat,
        y_feat: yFeat,
        z_axis: zObj,
        color_obj: colorObj,
        fixed_values: fixedValues,
        resolution,
        tweaks: {
            kernel: document.getElementById('tbo-kernel-select')?.value || 'matern52',
            max_gp_rows: parseInt(document.getElementById('tbo-prior-cap-slider')?.value || '3'),
        }
    };

    try {
        const result = await calculateTBOPlot(payload, plotContainer, 'Generating 3D response surface');
        if (!result) return;
        const surfaceTrace = {
            x: result.x,
            y: result.y,
            z: result.z,
            surfacecolor: result.color || result.z,
            type: 'surface',
            colorscale: 'Viridis',
            colorbar: { title: result.color_label || result.z_label, len: 0.8 },
            hovertemplate: `<b>${result.x_label}:</b> %{x}<br><b>${result.y_label}:</b> %{y}<br><b>${result.z_label}:</b> %{z:.3f}<br><b>${result.color_label || result.z_label}:</b> %{surfacecolor:.3f}<extra></extra>`,
        };
        try { Plotly.purge(plotContainer); } catch (_) {}
        plotContainer.innerHTML = '';
        Plotly.newPlot(plotContainer, [surfaceTrace, ...buildTBOExperimentTraces(result, true)], {
            title: `3D TransBO Surrogate: ${zObj} vs (${xFeat}, ${yFeat})`,
            scene: {
                xaxis: { title: xFeat },
                yaxis: { title: yFeat },
                zaxis: { title: zObj },
            },
            margin: { l: 20, r: 20, t: 40, b: 20 },
            legend: { orientation: 'h', y: -0.1 },
        }, { responsive: true });
    } catch (error) {
        if (error.name === 'AbortError') return;
        console.error('3D surface plot error:', error);
        try { Plotly.purge(plotContainer); } catch (_) {}
        plotContainer.innerHTML = `<div style="color:red;padding:1rem;">3D surface calculation error: ${error.message}</div>`;
    }
}
