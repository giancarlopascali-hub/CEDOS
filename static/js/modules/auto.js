/** CEDOS closed-loop controller. The EDOS and TransBO numerical engines are untouched. */

import { cancelBackgroundJob } from '../utils.js';
import { renderDoEResults } from './doe.js';
import { buildAutoOptimizerPayload, submitAutoOptimization } from './auto_optimizer.js';
import {
    newCampaignId, buildProposalRows, mergeProposalResults,
    removePendingProposalRows, scoreReference, bestMeasuredScore,
    evaluateMeasuredStop
} from './auto_campaign.js';

const POLL_MS = 5000;
const CONNECTION_GRACE_MS = 10 * 60 * 1000;
const SESSION_KEY = 'cedosAutoCampaignId';
const ACCESS_KEY = 'cedosAutoAccessToken';
let campaign = null;
let elapsedTimer = null;
let waitTimer = null;
let initialized = false;
let orphanedCampaignId = null;

const byId = id => document.getElementById(id);
const optimizerName = () => window.cedos?.activeOptimizer === 'tbo' ? 'TransBO' : 'BO';
const activeOptimizer = () => window.cedos?.activeOptimizer || 'bo';
const hostedAuto = () => document.body?.dataset.hostedEnvironment === 'true';

function autoAccessToken() {
    if (!hostedAuto()) return '';
    try { return sessionStorage.getItem(ACCESS_KEY) || ''; }
    catch (_) { return ''; }
}

function autoFetch(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (hostedAuto()) {
        const token = autoAccessToken();
        if (!token) throw new Error('Enter the hosted Auto exchange token before starting a campaign.');
        headers.set('Authorization', `Bearer ${token}`);
    }
    // On a hosted Space the token must only go to this app's own origin.
    const server = hostedAuto() ? '' : (window.autoServerUrl || '');
    return fetch(`${server}${path}`, { ...options, headers });
}

function storedCampaignId() {
    try { return sessionStorage.getItem(SESSION_KEY); }
    catch (_) { return null; }
}

function rememberCampaignId(id) {
    try { sessionStorage.setItem(SESSION_KEY, id); } catch (_) { /* unavailable storage */ }
}

function forgetCampaignId() {
    try { sessionStorage.removeItem(SESSION_KEY); } catch (_) { /* unavailable storage */ }
}

function showRecoveryAction(id) {
    orphanedCampaignId = id;
    byId('auto-recover-btn')?.remove();
    const actions = byId('auto-action-container');
    if (!actions) return;
    const button = document.createElement('button');
    button.id = 'auto-recover-btn';
    button.type = 'button';
    button.className = 'btn secondary small';
    button.textContent = 'Release interrupted Auto campaign';
    button.addEventListener('click', async () => {
        if (!confirm('This browser session lost its dataset. Stop the unfinished bridge proposal so a new campaign can start?')) return;
        try {
            await postJson('/auto/stop', { campaign_id: id });
            orphanedCampaignId = null;
            forgetCampaignId();
            button.remove();
            log('The interrupted bridge proposal was released. Load or restore a dataset before starting again.', 'success');
        } catch (error) {
            if (error.status === 409) {
                orphanedCampaignId = null;
                forgetCampaignId();
                button.remove();
                log('That earlier campaign is no longer active on the bridge.', 'info');
                return;
            }
            log(`Unable to release the interrupted proposal: ${error.message}`, 'error');
        }
    });
    actions.append(button);
}

async function checkInterruptedSession() {
    const id = storedCampaignId();
    if (!id) return;
    if (hostedAuto() && !autoAccessToken()) return;
    orphanedCampaignId = id;
    try {
        const response = await autoFetch(`/auto/status?campaign_id=${encodeURIComponent(id)}`);
        if (storedCampaignId() !== id) return;
        if (response.status === 409) {
            forgetCampaignId();
            orphanedCampaignId = null;
            return;
        }
        const status = await response.json();
        if (!response.ok) throw new Error(status.error || `HTTP ${response.status}`);
        if (['idle', 'merged', 'cancelled'].includes(status.status)) {
            forgetCampaignId();
            orphanedCampaignId = null;
            return;
        }
        log('An earlier Auto proposal is still active on the bridge, but this browser session lost its dataset. Release it before starting again.', 'warning');
        showRecoveryAction(id);
    } catch (error) {
        if (storedCampaignId() !== id) return;
        log(`Could not check the earlier Auto proposal: ${error.message}`, 'warning');
        showRecoveryAction(id);
    }
}

function log(message, type = 'info') {
    const container = byId('auto-activity-log');
    if (!container) return;
    const entry = document.createElement('div');
    const colors = { info: '#64748b', success: '#16a34a', warning: '#d97706', error: '#dc2626' };
    entry.style.cssText = `color:${colors[type] || colors.info};font-size:0.82rem;margin-bottom:2px`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    container.append(entry);
    container.scrollTop = container.scrollHeight;
}

function setBadge(status) {
    const badge = byId('auto-status-badge');
    if (!badge) return;
    const map = {
        idle: ['Idle', '#64748b'], optimizing: ['Optimizing…', '#2563eb'],
        awaiting_hardware: ['Waiting for hardware…', '#d97706'],
        merging: ['Merging results…', '#2563eb'],
        review_optimizer: [`Review ${optimizerName()} setup`, '#a16207'],
        completed: ['Completed', '#16a34a'], cancelled: ['Stopped', '#dc2626'],
        failed: ['Failed', '#dc2626'], exhausted: ['Exhausted', '#92400e']
    };
    const [label, color] = map[status] || map.idle;
    badge.textContent = label;
    badge.style.color = color;
}

function setState(state) {
    if (campaign) campaign.phase = state;
    window.autoModeActive = ['optimizing', 'awaiting_hardware', 'merging'].includes(state);
    window.cedos?.setCampaignState(state);
    setBadge(state);
    const active = window.autoModeActive;
    const doeButton = byId('execute-autodoe-btn');
    const optimizerButton = byId('execute-autobo-btn');
    if (doeButton) doeButton.textContent = active ? 'Stop Auto Loop' : 'Execute AutoDoE';
    if (optimizerButton) optimizerButton.textContent = active
        ? 'Stop Auto Loop' : `Execute Auto${optimizerName()}`;
    const reset = byId('reset-auto-btn');
    if (reset) reset.disabled = active;
    updateButtons();
}

function updateStats() {
    const count = byId('auto-iteration-count');
    if (count) count.textContent = String(campaign?.completedIterations || 0);
    const score = campaign?.scoreDisplay || getBestScoreAndLabel();
    const label = document.querySelector('.auto-stats-row .auto-stat-item:nth-child(2) .auto-stat-label');
    if (label) label.textContent = score.label;
    const value = byId('auto-best-score');
    if (value) value.textContent = score.display;
    const start = campaign?.optimizerStartTime || campaign?.startTime;
    const elapsed = start ? Math.floor((Date.now() - start) / 1000) : 0;
    const elapsedEl = byId('auto-elapsed-time');
    if (elapsedEl) elapsedEl.textContent = `${String(Math.floor(elapsed / 60)).padStart(2, '0')}:${String(elapsed % 60).padStart(2, '0')}`;
}

function getBestScoreAndLabel() {
    if (campaign?.scoreDisplay) return campaign.scoreDisplay;
    const columns = window.currentColumns || [];
    const objectives = columns.filter(name => window.columnRoles?.[name] === 'objective'
        && window.objectiveConfigs?.[name]?.included !== false)
        .map(name => ({ name, ...window.objectiveConfigs[name] }));
    const reference = scoreReference(window.currentData || [], columns, objectives);
    return bestMeasuredScore(window.currentData || [], columns, objectives, reference);
}
window.getBestScoreAndLabel = getBestScoreAndLabel;

function showDataset() {
    window.renderMainTable?.();
    window.renderSetup?.();
    byId('data-section')?.classList.remove('hidden');
    byId('upload-section')?.classList.add('hidden');
}

function readError(response, body) {
    return body?.error || body?.message || `Server responded HTTP ${response.status}.`;
}

async function requestJson(path, options = {}) {
    const response = await autoFetch(path, options);
    let body = {};
    try { body = await response.json(); } catch (_) { /* HTTP errors can be HTML. */ }
    if (!response.ok) {
        const error = new Error(readError(response, body));
        error.status = response.status;
        throw error;
    }
    return { response, body };
}

async function postJson(path, payload) {
    return (await requestJson(path, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })).body;
}

function initializeDoEDataset(features, objectives) {
    if (Array.isArray(window.currentData) && window.currentData.length) {
        throw new Error('Start with DoE only on an empty dataset. Use the active optimizer for existing measurements.');
    }
    window.currentColumns = [...features.map(item => item.name), ...objectives.map(item => item.name)];
    window.currentData = [];
    window.columnRoles ||= {};
    window.boFeatureConfigs ||= {};
    window.saFeatureConfigs ||= {};
    window.objectiveConfigs ||= {};
    features.forEach(feature => {
        window.columnRoles[feature.name] = 'feature';
        window.boFeatureConfigs[feature.name] = {
            type: feature.type, range: feature.range, included: true
        };
        window.saFeatureConfigs[feature.name] = { type: feature.type, included: true };
    });
    objectives.forEach(objective => {
        window.columnRoles[objective.name] = 'objective';
        window.objectiveConfigs[objective.name] ||= {
            name: objective.name, type: 'maximize', target: '',
            importance: 100 / objectives.length, included: true
        };
    });
    window.currentDoEFeatures = features;
    window.currentDoEObjectives = objectives;
    window.cedos?.notifyConfigMutation('Auto DoE created the dataset schema');
}

async function dispatchProposal(current, suggestions, spec) {
    if (campaign !== current || !current.active) return;
    const columns = spec.columns || window.currentColumns;
    const rows = buildProposalRows(columns, spec.featureNames, suggestions, spec.priorDefaults);
    const proposalId = newCampaignId();
    const payload = {
        campaign_id: current.id, proposal_id: proposalId,
        optimizer: spec.optimizer, dataset_rev: window.cedos?.datasetRevision || 0,
        config_rev: (window.cedos?.configRevision || 0) + (spec.initialDoE ? 1 : 0),
        columns, objective_names: spec.objectiveNames,
        features_only: rows.map(row => row.features),
        suggestions, rows: rows.map(row => row.proposalRow)
    };
    const accepted = await postJson('/auto/proposal', payload);
    if (accepted.campaign_id !== current.id || accepted.proposal_id !== proposalId) {
        throw new Error('The bridge acknowledged a different proposal.');
    }
    current.bridgeAccepted = true;
    if (campaign !== current || !current.active) {
        await postJson('/auto/stop', { campaign_id: current.id }).catch(() => {});
        return;
    }
    if (spec.initialDoE) initializeDoEDataset(spec.features, spec.objectives);
    const pending = {
        campaign_id: current.id, proposal_id: proposalId, rows,
        objectiveNames: spec.objectiveNames, priorDefaults: spec.priorDefaults || {},
        optimizer: spec.optimizer, merged: false,
        columns: [...columns], configRevision: window.cedos?.configRevision || 0
    };
    current.pending = pending;
    window.currentData.push(...rows.map(row => row.datasetRow));
    window.cedos?.notifyDatasetMutation('Auto staged a proposal');
    pending.datasetRevision = window.cedos?.datasetRevision || 0;
    showDataset();
    log(`Proposal ${accepted.iteration} accepted: ${rows.length} identified experiment(s).`, 'success');
    setState('awaiting_hardware');
    void pollForResults(current, pending);
}

async function acknowledgeResults(current, pending) {
    let failure;
    for (let attempt = 0; attempt < 3; attempt++) {
        try {
            await postJson(`/auto/results/${encodeURIComponent(pending.proposal_id)}/ack`, {
                campaign_id: current.id
            });
            return;
        } catch (error) {
            failure = error;
            if (attempt < 2) await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
        }
    }
    throw new Error(`Measured rows were saved, but the bridge did not confirm their acknowledgement: ${failure.message}`);
}

async function pollForResults(current, pending) {
    let disconnectedAt = null;
    let lastWaitLog = 0;
    while (campaign === current && current.active && current.pending === pending) {
        try {
            const query = new URLSearchParams({
                campaign_id: current.id, proposal_id: pending.proposal_id
            });
            const response = await autoFetch(`/auto/results?${query}`);
            let result = {};
            try { result = await response.json(); } catch (_) { /* handled below */ }
            if (response.status === 202) {
                if (Date.now() - lastWaitLog > 60000) {
                    log('Waiting for identified hardware results…');
                    lastWaitLog = Date.now();
                }
            } else if (!response.ok) {
                throw new Error(readError(response, result));
            } else {
                if (campaign !== current || !current.active || current.pending !== pending) return;
                if ((window.cedos?.datasetRevision || 0) !== pending.datasetRevision
                    || (window.cedos?.configRevision || 0) !== pending.configRevision
                    || JSON.stringify(window.currentColumns) !== JSON.stringify(pending.columns)) {
                    throw new Error('The dataset or setup changed while hardware was running. Results were not merged.');
                }
                setState('merging');
                const count = mergeProposalResults(result, pending, window.currentData,
                    window.currentColumns, pending.objectiveNames, pending.priorDefaults);
                pending.merged = true;
                window.cedos?.notifyDatasetMutation('Auto merged measured results');
                showDataset();
                await acknowledgeResults(current, pending);
                if (campaign !== current || !current.active) return;
                current.pending = null;
                log(`${count} measured result(s) merged by row ID.`, 'success');
                if (pending.optimizer === 'doe') {
                    enterOptimizerReview(current);
                } else {
                    await afterMeasuredOptimizerBatch(current);
                }
                return;
            }
            disconnectedAt = null;
        } catch (error) {
            if (campaign !== current || !current.active) return;
            // A rejected identity or invalid measurement is terminal. Network
            // interruptions are retried without accepting an unscoped result.
            if (error instanceof TypeError) {
                disconnectedAt ||= Date.now();
                if (Date.now() - disconnectedAt < CONNECTION_GRACE_MS) {
                    log(`Bridge connection interrupted: ${error.message}. Retrying.`, 'warning');
                    await new Promise(resolve => setTimeout(resolve, POLL_MS));
                    continue;
                }
            }
            await finishCampaign(current, 'failed', error.message);
            return;
        }
        await new Promise(resolve => setTimeout(resolve, POLL_MS));
    }
}

function enterOptimizerReview(current) {
    if (campaign !== current || !current.active) return;
    current.active = false;
    window.autoStartedFrom = 'bo';
    byId('execute-autodoe-btn')?.classList.add('hidden');
    const step3 = byId('step-3');
    const ready = byId('check-bo-ready');
    step3?.classList.remove('disabled');
    if (ready) { ready.disabled = false; ready.checked = false; }
    byId('step-4')?.classList.add('disabled');
    log(`DoE is measured. Review ${optimizerName()} features, objectives, constraints and Prior Targets before continuing.`, 'success');
    setState('review_optimizer');
    window.cedos?.switchModuleSafe(current.optimizer);
}

async function afterMeasuredOptimizerBatch(current) {
    const scored = bestMeasuredScore(window.currentData, window.currentColumns,
        current.objectives, current.scoreReference);
    current.scoreDisplay = scored;
    const stop = evaluateMeasuredStop(current, scored.score);
    window.autoCurrentIteration = current.completedIterations;
    window.autoBestScore = current.bestScore;
    window.autoNoImprovementCount = current.noImprovementCount;
    updateStats();
    if (stop) {
        await finishCampaign(current, 'completed',
            `Stopping condition met after ${current.completedIterations} measured optimizer batch(es).`);
        window.cedos?.switchModuleSafe('sa');
        return;
    }
    await runOptimizer(current);
}

async function runOptimizer(current) {
    if (campaign !== current || !current.active) return;
    try {
        if (current.optimizer !== activeOptimizer()
            || current.configRevision !== (window.cedos?.configRevision || 0)) {
            throw new Error('The active optimizer setup changed during the campaign.');
        }
        setState('optimizing');
        const adapter = buildAutoOptimizerPayload(current.optimizer);
        const controller = new AbortController();
        current.controller = controller;
        current.jobId = null;
        let lastPhase = '';
        log(`Running ${optimizerName()} for measured batch ${current.completedIterations + 1}…`);
        const result = await submitAutoOptimization(adapter, {
            signal: controller.signal,
            onSubmitted: jobId => { current.jobId = jobId; },
            onProgress: status => {
                if (status.phase && status.phase !== lastPhase) {
                    lastPhase = status.phase;
                    log(`${optimizerName()}: ${status.phase}`);
                }
            }
        });
        current.jobId = null;
        current.controller = null;
        if (campaign !== current || !current.active) return;
        if (result.exhausted) {
            await finishCampaign(current, 'exhausted',
                `${optimizerName()} found no distinct conditions at the current precision.`);
            return;
        }
        const diagnostics = result.constraint_status;
        if (diagnostics?.message) log(diagnostics.message, 'warning');
        if (result.transfer?.algorithm_used === 'standard_bo') {
            log('TransBO used its standard BO fallback because no Prior Target was selected.', 'warning');
        }
        await dispatchProposal(current, result.suggestions, {
            optimizer: current.optimizer, featureNames: adapter.featureNames,
            objectiveNames: adapter.objectiveNames,
            priorDefaults: adapter.priorDefaults
        });
    } catch (error) {
        current.jobId = null;
        current.controller = null;
        if (campaign !== current || !current.active) return;
        await finishCampaign(current, 'failed', `${optimizerName()} failed: ${error.message}`);
    }
}

function collectDoESetup() {
    const features = Array.from(document.querySelectorAll('#doe-features-list tr')).map(row => ({
        name: row.querySelector('.doe-f-name')?.value?.trim(),
        type: row.querySelector('.doe-f-type')?.value,
        range: row.querySelector('.doe-f-range')?.value
    })).filter(feature => feature.name);
    const objectives = Array.from(document.querySelectorAll('#doe-objectives-list tr'))
        .map(row => ({ name: row.querySelector('.doe-o-name')?.value?.trim() }))
        .filter(objective => objective.name);
    const names = [...features.map(feature => feature.name), ...objectives.map(objective => objective.name)];
    if (!features.length || !objectives.length || new Set(names).size !== names.length) {
        throw new Error('DoE needs distinct feature and objective names.');
    }
    if (Array.isArray(window.currentData) && window.currentData.length) {
        throw new Error('The dataset already has rows. Start Auto from the active optimizer.');
    }
    const numerical = Array.from(document.querySelectorAll('#doe-numerical-constraints-list tr.constraint-row'))
        .map(row => row.querySelector('.constraint-equation')?.value?.trim()).filter(Boolean);
    const disallowed = Array.from(document.querySelectorAll('#doe-disallowed-constraints-list tr.disallowed-row'))
        .map(row => Array.from(row.querySelectorAll('.condition-item')).map(item => ({
            feature: item.querySelector('.condition-feature')?.value,
            value: item.querySelector('.condition-value')?.value
        }))).filter(conditions => conditions.length);
    const tweaks = {
        model: byId('doe-model-select')?.value || 'bbdesign',
        max_runs: parseInt(byId('doe-max-runs')?.value || '50', 10),
        use_exact_runs: byId('doe-exact-runs-toggle')?.checked || false,
        dopt_model_order: byId('doe-dopt-order-select')?.value || 'quadratic',
        fracfact_resolution: parseInt(byId('doe-fracfact-resolution-select')?.value || '4', 10),
        constraints: { numerical, disallowed }
    };
    return { features, objectives, tweaks };
}

async function runDoE(current) {
    try {
        const spec = collectDoESetup();
        setState('optimizing');
        log('Generating the first DoE experiment batch…');
        const controller = new AbortController();
        current.controller = controller;
        const { body } = await requestJson('/doe', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: spec.features, tweaks: spec.tweaks }),
            signal: controller.signal
        });
        current.controller = null;
        if (campaign !== current || !current.active) return;
        if (!Array.isArray(body.suggestions) || !body.suggestions.length) {
            throw new Error('DoE returned no experimental conditions.');
        }
        if (body.metrics?.constraint_filtered) {
            window.suggestionsData = body.suggestions;
            renderDoEResults(body.metrics);
            if (body.metrics.design_warning) log(body.metrics.design_warning, 'warning');
            if (body.metrics.shortfall_runs > 0) {
                throw new Error('The exact DoE run count was not met. Adjust the design before hardware dispatch.');
            }
        }
        const columns = [...spec.features.map(item => item.name), ...spec.objectives.map(item => item.name)];
        await dispatchProposal(current, body.suggestions, {
            optimizer: 'doe', initialDoE: true, ...spec,
            columns, featureNames: spec.features.map(item => item.name),
            objectiveNames: spec.objectives.map(item => item.name), priorDefaults: {}
        });
    } catch (error) {
        if (campaign === current && current.active) await finishCampaign(current, 'failed', `Auto DoE failed: ${error.message}`);
    }
}

async function finishCampaign(current, status, message) {
    if (campaign !== current) return;
    current.active = false;
    current.controller?.abort();
    if (current.jobId) {
        await cancelBackgroundJob(current.jobId, { throwOnError: false }).catch(() => {});
    }
    if (current.pending && !current.pending.merged) {
        window.currentData = removePendingProposalRows(window.currentData, current.pending);
        window.cedos?.notifyDatasetMutation('Auto removed its pending rows');
        showDataset();
    }
    try {
        let shouldStop = current.bridgeAccepted;
        if (!shouldStop) {
            const response = await autoFetch(`/auto/status?campaign_id=${encodeURIComponent(current.id)}`);
            if (response.status === 409) {
                shouldStop = false;
            } else if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            } else {
                const statusBody = await response.json();
                shouldStop = !['idle', 'merged', 'cancelled'].includes(statusBody.status);
            }
        }
        if (shouldStop) await postJson('/auto/stop', { campaign_id: current.id });
        forgetCampaignId();
    } catch (error) {
        if (error.status === 409) {
            forgetCampaignId();
            orphanedCampaignId = null;
        } else {
        log(`Bridge stop could not be confirmed: ${error.message}`, 'warning');
        showRecoveryAction(current.id);
        }
    }
    current.pending = null;
    if (message) log(message, status === 'completed' ? 'success'
        : status === 'failed' ? 'error' : 'warning');
    setState(status);
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
    if (waitTimer) { clearInterval(waitTimer); waitTimer = null; }
    updateStats();
}

function readStopCondition(objectives) {
    const type = byId('auto-stop-type')?.value || '';
    const raw = byId('auto-stop-value')?.value?.trim() || '';
    const number = Number(raw);
    if (!['iterations', 'target', 'improvement', 'time'].includes(type)
        || raw === '' || !Number.isFinite(number) || number <= 0) {
        throw new Error('Choose a positive stopping condition value.');
    }
    if (['iterations', 'improvement'].includes(type) && !Number.isInteger(number)) {
        throw new Error('Iteration limits must be whole numbers.');
    }
    if (type === 'target') {
        if (number > 100 || !objectives.every(objective => objective.type === 'target')) {
            throw new Error('Target match stopping needs target-type objectives and a threshold from 0 to 100%.');
        }
    }
    return { type, value: number };
}

function beginOptimization(current) {
    const previousPhase = current.phase;
    try {
        const adapter = buildAutoOptimizerPayload(current.optimizer);
        const stopCondition = readStopCondition(adapter.payload.objectives);
        if (current.optimizer === 'tbo' && !adapter.payload.priors.length) {
            if (!confirm('No Prior Target is configured. TransBO will use its standard BO fallback and may cap training rows. Continue?')) {
                if (previousPhase !== 'review_optimizer') {
                    current.active = false;
                    campaign = null;
                    forgetCampaignId();
                }
                return;
            }
        }
        adapter.warnings.forEach(warning => log(warning, 'warning'));
        current.active = true;
        current.stopCondition = stopCondition;
        current.objectives = adapter.payload.objectives.map(objective => ({ ...objective }));
        current.scoreReference = scoreReference(window.currentData, window.currentColumns, current.objectives);
        current.scoreDisplay = bestMeasuredScore(window.currentData, window.currentColumns,
            current.objectives, current.scoreReference);
        current.bestScore = current.scoreDisplay.score;
        current.noImprovementCount = 0;
        current.completedIterations = 0;
        current.optimizerStartTime = Date.now();
        current.configRevision = window.cedos?.configRevision || 0;
        window.autoStopCondition = { ...stopCondition };
        window.autoStartTime = current.optimizerStartTime;
        window.autoCurrentIteration = 0;
        updateStats();
        log(`Auto ${optimizerName()} started. Stop: ${stopCondition.type} = ${stopCondition.value}.`);
        setState('optimizing');
        void runOptimizer(current);
    } catch (error) {
        if (previousPhase !== 'review_optimizer') {
            current.active = false;
            campaign = null;
            forgetCampaignId();
        }
        alert(error.message);
        log(error.message, 'error');
    }
}

export function startAutoLoop(fromMethod = 'bo') {
    if (hostedAuto() && !autoAccessToken()) {
        alert('Enter the hosted Auto exchange token before starting a campaign.');
        return;
    }
    if (orphanedCampaignId) {
        alert('Release the interrupted Auto proposal before starting another campaign.');
        return;
    }
    if (!byId('check-imex-started')?.checked) {
        alert('Confirm that the CEDOS IMEX bridge is running before starting Auto.');
        return;
    }
    if (campaign && ['optimizing', 'awaiting_hardware', 'merging'].includes(campaign.phase)) return;
    if (campaign?.phase === 'review_optimizer' && fromMethod !== 'doe') {
        beginOptimization(campaign);
        return;
    }
    const current = {
        id: newCampaignId(), optimizer: activeOptimizer(), startTime: Date.now(),
        active: true, phase: 'idle', pending: null,
        completedIterations: 0, noImprovementCount: 0, bestScore: null
    };
    campaign = current;
    rememberCampaignId(current.id);
    window.autoStartedFrom = fromMethod === 'doe' ? 'doe' : 'bo';
    window.autoMethodSelected = true;
    window.autoStartTime = current.startTime;
    if (!elapsedTimer) elapsedTimer = setInterval(updateStats, 1000);
    if (fromMethod === 'doe') void runDoE(current);
    else beginOptimization(current);
}

export async function stopAutoLoop() {
    if (!campaign || !['optimizing', 'awaiting_hardware', 'merging', 'review_optimizer'].includes(campaign.phase)) return;
    await finishCampaign(campaign, 'cancelled', 'Auto campaign stopped by the user.');
}

function updateButtons() {
    const doe = byId('execute-autodoe-btn');
    const optimizer = byId('execute-autobo-btn');
    const active = window.autoModeActive;
    const selected = window.autoStartedFrom || 'doe';
    if (!window.autoMethodSelected && !active) {
        doe?.classList.add('hidden');
        optimizer?.classList.add('hidden');
        return;
    }
    if (doe) doe.classList.toggle('hidden', selected !== 'doe' && !active);
    if (optimizer) optimizer.classList.toggle('hidden', selected === 'doe'
        || (!active && !(byId('auto-stop-type')?.value && byId('auto-stop-value')?.value?.trim())));
    if (active) {
        doe?.classList.toggle('hidden', selected !== 'doe');
        optimizer?.classList.toggle('hidden', selected === 'doe');
    }
}

function selectStartMethod(method) {
    if (window.cedos?.campaignActive) return;
    window.autoMethodSelected = true;
    window.autoStartedFrom = method;
    byId('auto-select-doe')?.classList.toggle('active', method === 'doe');
    byId('auto-select-bo')?.classList.toggle('active', method !== 'doe');
    byId('step-3')?.classList.toggle('disabled', method === 'doe');
    if (byId('check-bo-ready')) byId('check-bo-ready').disabled = method === 'doe';
    byId('step-4')?.classList.add('disabled');
    updateButtons();
    window.cedos?.switchModuleSafe(method === 'doe' ? 'doe' : activeOptimizer());
}

export function initAutoModule() {
    if (initialized) return;
    initialized = true;
    window.autoModeActive = false;
    window.autoStartedFrom = 'doe';
    window.autoMethodSelected = false;
    window.autoStopCondition = { type: '', value: '' };
    window.autoCurrentIteration = 0;
    window.autoBestScore = null;
    byId('auto-server-url-display')?.replaceChildren(document.createTextNode(window.location.origin));
    if (hostedAuto()) {
        const accessStatus = byId('auto-access-status');
        if (accessStatus && autoAccessToken()) accessStatus.textContent = 'Token saved for this browser tab.';
        byId('auto-access-save')?.addEventListener('click', async () => {
            const field = byId('auto-access-token');
            const token = field?.value.trim() || '';
            if (!token) {
                if (accessStatus) accessStatus.textContent = 'Enter the token supplied by the Space owner.';
                return;
            }
            if (window.autoModeActive) {
                if (accessStatus) accessStatus.textContent = 'Stop the current campaign before changing its token.';
                return;
            }
            try {
                if (accessStatus) accessStatus.textContent = 'Checking token…';
                const response = await fetch('/auto/status', {
                    headers: { Authorization: `Bearer ${token}` }
                });
                if (!response.ok) {
                    const body = await response.json().catch(() => ({}));
                    throw new Error(readError(response, body));
                }
                sessionStorage.setItem(ACCESS_KEY, token);
                field.value = '';
                if (accessStatus) accessStatus.textContent = 'Token saved for this browser tab.';
                void checkInterruptedSession();
            } catch (error) {
                if (accessStatus) accessStatus.textContent = error.message || 'Unable to verify or save the token.';
            }
        });
        byId('auto-access-clear')?.addEventListener('click', () => {
            if (window.autoModeActive) {
                if (accessStatus) accessStatus.textContent = 'Stop the current campaign before clearing its token.';
                return;
            }
            try { sessionStorage.removeItem(ACCESS_KEY); } catch (_) { /* unavailable storage */ }
            if (accessStatus) accessStatus.textContent = 'Token cleared from this browser tab.';
        });
    }
    byId('check-imex-started')?.addEventListener('change', event => {
        byId('step-2')?.classList.toggle('disabled', !event.target.checked);
        if (byId('check-start-method')) byId('check-start-method').disabled = !event.target.checked;
    });
    byId('auto-select-doe')?.addEventListener('click', () => selectStartMethod('doe'));
    byId('auto-select-bo')?.addEventListener('click', () => selectStartMethod('bo'));
    byId('auto-goto-bo')?.addEventListener('click', () => {
        if (byId('check-bo-ready')) byId('check-bo-ready').checked = true;
        byId('step-4')?.classList.remove('disabled');
        if (byId('check-stopping-cond')) byId('check-stopping-cond').disabled = false;
        window.cedos?.switchModuleSafe(activeOptimizer());
    });
    document.addEventListener('click', event => {
        if (!event.target.closest('.back-to-auto-btn')) return;
        window.cedos?.switchModuleSafe('auto');
        if (window.autoStartedFrom !== 'doe') {
            if (byId('check-bo-ready')) byId('check-bo-ready').checked = true;
            byId('step-3')?.classList.remove('disabled');
            byId('step-4')?.classList.remove('disabled');
        }
        updateTargetOptionAvailability();
        updateButtons();
    });
    const stopLabels = {
        iterations: ['Max measured batches', 'e.g. 10'],
        target: ['Minimum target match (%)', 'e.g. 85'],
        improvement: ['Measured batches without improvement', 'e.g. 3'],
        time: ['Time limit (minutes)', 'e.g. 60']
    };
    byId('auto-stop-type')?.addEventListener('change', event => {
        const labels = stopLabels[event.target.value];
        byId('auto-stop-value-container')?.classList.toggle('hidden', !labels);
        if (labels) {
            byId('auto-stop-label').textContent = labels[0];
            byId('auto-stop-value').placeholder = labels[1];
        }
        window.autoStopCondition.type = event.target.value;
        updateButtons();
    });
    byId('auto-stop-value')?.addEventListener('input', event => {
        window.autoStopCondition.value = event.target.value;
        updateButtons();
    });
    byId('execute-autodoe-btn')?.addEventListener('click', () => {
        if (window.autoModeActive) void stopAutoLoop();
        else startAutoLoop('doe');
    });
    byId('execute-autobo-btn')?.addEventListener('click', () => {
        if (window.autoModeActive) void stopAutoLoop();
        else startAutoLoop('bo');
    });
    byId('reset-auto-btn')?.addEventListener('click', async () => {
        if (!confirm('Reset the full app? All loaded data and configurations will be cleared.')) return;
        await stopAutoLoop();
        window.performFullAppReset?.();
        resetAutoUI();
    });
    byId('save-auto-log-btn')?.addEventListener('click', () => {
        const entries = Array.from(byId('auto-activity-log')?.children || []).map(entry => entry.textContent);
        const text = `CEDOS Auto run log\nCampaign: ${campaign?.id || 'none'}\nOptimizer: ${campaign?.optimizer || 'none'}\nState: ${campaign?.phase || 'idle'}\nMeasured batches: ${campaign?.completedIterations || 0}\n\n${entries.join('\n')}`;
        const link = document.createElement('a');
        link.href = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
        link.download = `cedos_auto_${new Date().toISOString().slice(0, 10)}.txt`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    });
    document.addEventListener('change', event => {
        if (event.target.matches('.bo-o-type, .tbo-obj-type')) updateTargetOptionAvailability();
    });
    document.addEventListener('cedos:visibility-changed', () => {
        if (byId('auto-select-bo')) byId('auto-select-bo').textContent = `Start with ${optimizerName()}`;
        if (byId('auto-goto-bo')) byId('auto-goto-bo').textContent = `Check ${optimizerName()} Setup`;
        updateTargetOptionAvailability();
        updateButtons();
    });
    elapsedTimer = setInterval(updateStats, 1000);
    updateTargetOptionAvailability();
    setBadge('idle');
    updateButtons();
    log('Auto is ready. Choose DoE or the active optimizer to begin.');
    void checkInterruptedSession();
}

export function resetAutoUI() {
    if (campaign && ['optimizing', 'awaiting_hardware', 'merging', 'review_optimizer'].includes(campaign.phase)) {
        void stopAutoLoop().then(resetAutoUI);
        return;
    }
    campaign = null;
    ['check-imex-started', 'check-start-method', 'check-bo-ready', 'check-stopping-cond']
        .forEach(id => { if (byId(id)) byId(id).checked = false; });
    ['step-2', 'step-3', 'step-4'].forEach(id => byId(id)?.classList.add('disabled'));
    if (byId('auto-stop-type')) byId('auto-stop-type').value = '';
    if (byId('auto-stop-value')) byId('auto-stop-value').value = '';
    byId('auto-stop-value-container')?.classList.add('hidden');
    window.autoStartedFrom = 'doe';
    window.autoMethodSelected = false;
    window.autoStopCondition = { type: '', value: '' };
    window.autoStartTime = null;
    window.autoCurrentIteration = 0;
    window.autoBestScore = null;
    window.autoModeActive = false;
    if (byId('auto-activity-log')) byId('auto-activity-log').replaceChildren();
    window.cedos?.setCampaignState('idle');
    setBadge('idle');
    updateButtons();
    updateStats();
}
window.resetAutoUI = resetAutoUI;

export function updateTargetOptionAvailability() {
    const option = byId('auto-stop-target-option');
    if (!option) return;
    const optimizer = activeOptimizer();
    const selector = optimizer === 'tbo' ? '#tbo-objectives-list .tbo-obj-type'
        : '#bo-objectives-list .bo-o-type';
    let types = Array.from(document.querySelectorAll(selector)).map(input => input.value);
    if (!types.length) types = (window.currentColumns || [])
        .filter(name => window.columnRoles?.[name] === 'objective'
            && window.objectiveConfigs?.[name]?.included !== false)
        .map(name => window.objectiveConfigs[name]?.type);
    option.disabled = !types.length || !types.every(type => type === 'target');
    if (option.disabled && byId('auto-stop-type')?.value === 'target') {
        byId('auto-stop-type').value = '';
        byId('auto-stop-type').dispatchEvent(new Event('change'));
    }
}
window.updateTargetOptionAvailability = updateTargetOptionAvailability;
window.initAutoModule = initAutoModule;
