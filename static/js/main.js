import './state.js';
import cedos, { initCEDOS } from './cedos.js';
import { safeAddListener, exportData, detectParetoIndices, detectColumnIsNumeric, rebalanceRemaining, rebalanceObjectiveConfigs, updateRangePlaceholder, formatCategoricalChoices, uniqueCategoricalChoices } from './utils.js';
import { runDoE, renderDoEResults, rebalanceDoEDOM, attachDeleteEvents } from './modules/doe.js';
import { abortBO, renderBOSetup, runBO, runBOEstimate, renderTrendPlot, renderParetoPlot, updateAcqSelectForMultiObj } from './modules/bo.js';
import { renderSASetup, runSA, runEstimate, exportSAReport, abortSA } from './modules/sa.js';
import { updateAllConstraintsDropdowns } from './modules/constraints.js';
import { initSetupCSV } from './modules/setup_csv.js';
import { renderTBOSetup, runTransferBO, runTransferBOEstimate, initTBOVisualizationDropdown, renderTBOVisualization, initTBOButtons, resetTBOState, invalidateTBOResults } from './modules/tbo.js';
import { initAutoModule } from './modules/auto.js';

export { detectColumnIsNumeric };


// Bind utilities to window for inline calls
window.rebalanceRemaining = rebalanceRemaining;
window.rebalanceObjectiveConfigs = rebalanceObjectiveConfigs;

// DOMContentLoaded Entrypoint
document.addEventListener('DOMContentLoaded', () => {
    // Inject runtime CSS for row-disabled state
    const style = document.createElement('style');
    style.textContent = `
        .row-disabled { opacity: 0.55; background-color: #f1f5f9; }
        .row-disabled input:not([type="checkbox"]), .row-disabled select { 
            background: #e2e8f0; color: #94a3b8; cursor: not-allowed; pointer-events: none; 
        }
        .row-disabled td { color: #94a3b8; }
    `;
    document.head.appendChild(style);

    initCEDOS(renderModule);
    document.addEventListener('cedos:dataset-changed', () => { resetTBOState(); renderMainTable(); });
    document.addEventListener('cedos:config-changed', event => {
        if (['Active optimizer changed', 'Dataset column role changed',
            'Setup CSV imported', 'Auto DoE created the dataset schema'].includes(event.detail?.reason)) {
            resetTBOState();
        } else {
            invalidateTBOResults();
        }
    });
    document.addEventListener('cedos:campaign-state', () => renderMainTable());
    initModuleSwitcher();
    initFileUpload();
    initSetupCSV();
    initTabs();
    initButtons();
    initSettingsListeners();
    initStrategyToggle();
    initAutoModule();

    // Initialize the correct module view on page load
    switchModule(window.currentModule);

    // Abort button listener
    safeAddListener('abort-calc-btn', 'click', () => { void abortBO(); });
    safeAddListener('sa-abort-btn', 'click', abortSA);

    // Clear Data button listener
    safeAddListener('clear-data-btn', 'click', () => {
        if (!cedos.canEditDataset()) {
            cedos.showNotice('The shared dataset is locked while a job or Auto campaign is active.', true);
            return;
        }
        if (confirm('Are you sure you want to clear all loaded data and reset the module configurations? This cannot be undone.')) {
            performFullAppReset();
        }
    });
});

function performFullAppReset() {
    abortSA();
    window.currentData      = null;
    window.currentColumns   = [];
    window.columnRoles      = {};
    window.objectiveConfigs = {};
    window.boFeatureConfigs = {};
    window.saFeatureConfigs = {};
    window.suggestionsData  = null;
    window.currentDoEFeatures   = [];
    window.currentDoEObjectives = [];

    const resultsTableBody = document.getElementById('results-table-body');
    if (resultsTableBody) resultsTableBody.innerHTML = '';
    const doeResultsBody = document.getElementById('doe-results-body');
    if (doeResultsBody) doeResultsBody.innerHTML = '';

    window.priorConfigs     = {};
    clearDataBackedConstraints();
    ['doe-features-list', 'doe-objectives-list', 'doe-numerical-constraints-list',
        'doe-disallowed-constraints-list'].forEach(id => document.getElementById(id)?.replaceChildren());
    resetTBOState();

    try {
        if (document.getElementById('trend-plot'))  Plotly.purge('trend-plot');
        if (document.getElementById('bo-contour-plot')) Plotly.purge('bo-contour-plot');
        if (document.getElementById('bo-surface-plot')) Plotly.purge('bo-surface-plot');
        if (document.getElementById('pareto-plot')) Plotly.purge('pareto-plot');
        if (document.getElementById('tbo-trend-plot')) Plotly.purge('tbo-trend-plot');
        if (document.getElementById('tbo-contour-plot')) Plotly.purge('tbo-contour-plot');
        if (document.getElementById('tbo-surface-plot')) Plotly.purge('tbo-surface-plot');
        if (document.getElementById('sa-num-corr')) Plotly.purge('sa-num-corr');
        document.querySelectorAll('[id^="sa-parcoords-"]').forEach(el => Plotly.purge(el.id));
        document.querySelectorAll('[id^="sa-bar-"]').forEach(el => Plotly.purge(el.id));
        document.querySelectorAll('[id^="sa-shap-"]').forEach(el => Plotly.purge(el.id));
    } catch(e) { console.error("Error purging plots:", e); }

    renderMainTable();
    renderSetup();
    cedos.syncEditability();

    document.getElementById('data-section').classList.add('hidden');
    document.getElementById('upload-section').classList.remove('hidden');

    document.getElementById('doe-results-section').classList.add('hidden');
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('sa-results-container').classList.add('hidden');
    if (document.getElementById('bo-visualization-section')) document.getElementById('bo-visualization-section').classList.add('hidden');
    if (document.getElementById('bo-visuals')) document.getElementById('bo-visuals').classList.add('hidden');
    if (document.getElementById('pareto-plot-section')) document.getElementById('pareto-plot-section').classList.add('hidden');
    if (document.getElementById('tbo-visualization-section')) document.getElementById('tbo-visualization-section').classList.add('hidden');
    if (document.getElementById('tbo-results-section')) document.getElementById('tbo-results-section').classList.add('hidden');
    cedos.notifyDatasetMutation('Dataset cleared');
}
window.performFullAppReset = performFullAppReset;

function clearDataBackedConstraints() {
    ['bo', 'tbo', 'sa'].forEach(module => {
        ['numerical-constraints-list', 'disallowed-constraints-list', 'objective-constraints-list']
            .forEach(suffix => document.getElementById(`${module}-${suffix}`)?.replaceChildren());
    });
}

function initStrategyToggle() {
    const radios = document.querySelectorAll('input[name="bo-strategy"]');
    radios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            const strategy      = e.target.value;
            const banner        = document.getElementById('bo-warning-banner');
            const warningText   = document.getElementById('bo-warning-text');
            const noiselessCb   = document.getElementById('bo-noiseless');
            const avoidRevalCb  = document.getElementById('bo-avoid-reval');
            let reclassified    = [];

            if (strategy === 'exhaustive_grid') {
                if (noiselessCb)  { noiselessCb.disabled  = true; noiselessCb.closest('.tweak-item').style.opacity  = '0.5'; }
                if (avoidRevalCb) { avoidRevalCb.disabled = true; avoidRevalCb.closest('.tweak-item').style.opacity = '0.5'; }

                document.querySelectorAll('.bo-f-type').forEach(select => {
                    if (select.value === 'continuous') {
                        const col        = select.dataset.col;
                        const tr         = select.closest('tr');
                        const rangeInput = tr.querySelector('.bo-f-range');

                        select.value = 'regular';
                        let val = rangeInput.value.trim();
                        if (val.includes('[') && val.includes(']')) {
                            if (!val.match(/\]\d+/)) rangeInput.value = val + "10";
                        } else {
                            const matches = val.match(/[-+]?\d*\.?\d+/g);
                            if (matches && matches.length >= 2) rangeInput.value = `[${matches[0]}, ${matches[1]}]10`;
                        }

                        window.syncBOFeatureConfig(col, 'type');
                        window.syncBOFeatureConfig(col, 'range');
                        reclassified.push(col);
                    }
                });

                if (reclassified.length > 0) {
                    banner.classList.remove('hidden');
                    warningText.innerHTML = `**Exhaustive Grid selected:** ${reclassified.join(', ')} reclassified as **Regular** (10 steps) to define the search space.`;
                }
            } else {
                if (noiselessCb)  { noiselessCb.disabled  = false; noiselessCb.closest('.tweak-item').style.opacity  = '1'; }
                if (avoidRevalCb) { avoidRevalCb.disabled = false; avoidRevalCb.closest('.tweak-item').style.opacity = '1'; }

                document.querySelectorAll('.bo-f-type').forEach(select => {
                    if (select.value === 'regular') {
                        const col        = select.dataset.col;
                        const tr         = select.closest('tr');
                        const rangeInput = tr.querySelector('.bo-f-range');

                        select.value      = 'continuous';
                        rangeInput.value  = rangeInput.value.replace(/\]\d+$/, ']');

                        window.syncBOFeatureConfig(col, 'type');
                        window.syncBOFeatureConfig(col, 'range');
                        reclassified.push(col);
                    }
                });
                banner.classList.add('hidden');
            }
        });
    });
}

function initModuleSwitcher() {
    document.querySelectorAll('.module-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            switchModule(btn.dataset.module);
        });
    });
}

export function computeColumnNumericRange(vals, isNum) {
    if (!isNum) {
        const choices = uniqueCategoricalChoices(vals);
        return { isNum: false, range: formatCategoricalChoices(choices), choices };
    }
    const numVals = [];
    for (const v of vals) {
        if (v == null) continue;
        const n = Number(String(v).trim());
        if (!isNaN(n) && isFinite(n)) numVals.push(n);
    }
    if (numVals.length === 0) {
        return { isNum: false, range: '' };
    }
    let min = numVals[0], max = numVals[0];
    for (let i = 1; i < numVals.length; i++) {
        if (numVals[i] < min) min = numVals[i];
        if (numVals[i] > max) max = numVals[i];
    }
    return { isNum: true, range: `[${min}, ${max}]` };
}

export function switchModule(module) {
    return cedos.switchModuleSafe(module);
}
window.switchModule = switchModule;

function renderModule(module) {
    console.log(`Switching to module: ${module}`);

    document.querySelectorAll('.module-btn').forEach(b => b.classList.toggle('active', b.dataset.module === module));
    document.querySelectorAll('.module-view').forEach(v => v.classList.toggle('hidden', v.id !== `${module}-module`));

    renderMainTable();
    renderSetup();
    cedos.syncEditability();

    if (module === 'bo' || module === 'sa' || module === 'tbo' || module === 'auto') {
        document.getElementById('upload-section').classList.remove('hidden');
        if (window.currentData) {
            document.getElementById('data-section').classList.remove('hidden');
            if (module === 'bo') renderTrendPlot();
            if (module === 'tbo') renderTBOVisualization();
        }
    } else {
        document.getElementById('upload-section').classList.add('hidden');
        document.getElementById('data-section').classList.add('hidden');
    }
}

function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const parent = btn.closest('section, .module-view');
            parent.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            parent.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.remove('hidden');
        });
    });
}

function initFileUpload() {
    safeAddListener('file-input', 'change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
            e.target.value = '';
        }
    });

    const dropZone = document.getElementById('drop-zone');
    if (dropZone) {
        dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            handleFiles(e.dataTransfer.files);
        });
    }
}

async function handleFiles(files) {
    if (files.length === 0) return;
    if (!cedos.canEditDataset()) {
        cedos.showNotice('The shared dataset is locked while a job or Auto campaign is active.', true);
        return;
    }
    const revision = cedos.captureRevision();
    abortSA();
    const formData = new FormData();
    formData.append('file', files[0]);

    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const result   = await response.json();
        if (result.error) throw new Error(result.error);
        if (!cedos.isRevisionCurrent(revision) || !cedos.canEditDataset()) {
            throw new Error('The app changed while the dataset was uploading. Upload it again.');
        }

        resetTBOState();
        clearDataBackedConstraints();

        window.currentData    = result.data;
        window.currentColumns = result.columns;

        window.columnRoles      = {};
        window.objectiveConfigs = {};
        window.currentColumns.forEach((col, idx) => {
            window.columnRoles[col] = (idx === window.currentColumns.length - 1) ? 'objective' : 'feature';
        });

        const objCols = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');
        objCols.forEach(col => {
            window.objectiveConfigs[col] = {
                name:       col,
                type:       'maximize',
                target:     '',
                importance: (100 / objCols.length).toFixed(1),
                included:   true
            };
        });

        window.boFeatureConfigs = {};
        window.saFeatureConfigs = {};
        window.currentColumns.forEach((col, idx) => {
            if (window.columnRoles[col] === 'feature') {
                const vals  = window.currentData.map(r => r[idx]).filter(v => v != null && v !== '');
                const isNum = detectColumnIsNumeric(vals);
                const { range, choices } = computeColumnNumericRange(vals, isNum);

                window.boFeatureConfigs[col] = { type: isNum ? 'continuous' : 'categorical', range, choices, included: true };
                window.saFeatureConfigs[col] = { type: isNum ? 'continuous' : 'categorical', included: true };
            }
        });

        renderMainTable();
        renderSetup();
        if (window.currentModule === 'bo') renderTrendPlot();
        if (window.currentModule === 'tbo') renderTBOVisualization();

        document.getElementById('data-section').classList.remove('hidden');
        cedos.notifyDatasetMutation('Dataset uploaded');
    } catch (err) {
        alert('Error loading file: ' + err.message);
    }
}

export function renderMainTable() {
    const head = document.getElementById('main-table-head');
    const body = document.getElementById('main-table-body');
    if (!head || !body) return;

    const isTBO = window.currentModule === 'tbo';
    let headHtml = '<tr>';
    window.currentColumns.forEach(col => {
        const role = window.columnRoles[col];
        headHtml += `
            <th>
                <div class="col-header-name">${col}</div>
                <div class="type-toggle" onclick="window.toggleColumnRole('${col}')">
                    <span class="role-f ${role === 'feature'   ? 'active' : ''}">F</span>
                    <span class="role-o ${role === 'objective' ? 'active' : ''}">O</span>
                    ${(isTBO || role === 'prior') ? `<span class="role-p ${role === 'prior' ? 'active' : ''}" title="Prior Target">P</span>` : ''}
                </div>
            </th>`;
    });
    headHtml += '</tr>';
    head.innerHTML = headHtml;

    let bodyHtml = '';
    if (window.currentData) {
        window.currentData.forEach((row, rowIdx) => {
            bodyHtml += '<tr>' +
                row.map((v, colIdx) =>
                    `<td contenteditable="${cedos.canEditDataset()}" onblur="window.updateCellData(${rowIdx}, ${colIdx}, this.textContent)">${v}</td>`
                ).join('') +
                '</tr>';
        });
    }
    body.innerHTML = bodyHtml;
}
window.renderMainTable = renderMainTable;

window.updateCellData = (rowIdx, colIdx, value) => {
    if (!cedos.canEditDataset()) {
        renderMainTable();
        return;
    }
    if (String(window.currentData[rowIdx][colIdx] ?? '') === String(value ?? '')) return;
    window.currentData[rowIdx][colIdx] = value;
    cedos.notifyDatasetMutation('Dataset cell edited');
    if (window.currentModule === 'bo') renderTrendPlot();
    if (window.currentModule === 'tbo') renderTBOVisualization();
};

async function detectPriorType(colName) {
    try {
        const response = await fetch('/detect_prior_types', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                columns: window.currentColumns,
                data: window.currentData,
                prior_cols: [colName]
            })
        });
        const d = await response.json();
        if (d.types && d.types[colName]) {
            if (window.priorConfigs[colName]) {
                window.priorConfigs[colName].col_type = d.types[colName];
            }
            if (window.currentModule === 'tbo') {
                renderTBOSetup();
            }
        }
    } catch (e) {
        console.error('Error detecting prior type:', e);
    }
}
window.detectPriorType = detectPriorType;

window.toggleColumnRole = (colName) => {
    if (!cedos.canEditRoles()) {
        cedos.showNotice('Column roles are locked while a batch is running. Review them in the active optimizer tab after DoE.', true);
        return;
    }
    const oldRole = window.columnRoles[colName];
    const isTBO = window.currentModule === 'tbo';
    let newRole;
    if (oldRole === 'feature') newRole = 'objective';
    else if (oldRole === 'objective') newRole = isTBO ? 'prior' : 'feature';
    else newRole = 'feature'; // prior → feature (TBO only gets here)
    cedos.noteExplicitRoleChange(colName);
    window.columnRoles[colName] = newRole;

    if (newRole === 'feature') {
        const idx   = window.currentColumns.indexOf(colName);
        const vals  = window.currentData ? window.currentData.map(r => r[idx]).filter(v => v != null && v !== '') : [];
        const isNum = detectColumnIsNumeric(vals);
        const { range, choices } = computeColumnNumericRange(vals, isNum);

        if (!window.boFeatureConfigs[colName]) window.boFeatureConfigs[colName] = { type: isNum ? 'continuous' : 'categorical', range, choices, included: true };
        if (!window.saFeatureConfigs[colName]) window.saFeatureConfigs[colName] = { type: isNum ? 'continuous' : 'categorical', included: true };
        delete window.priorConfigs[colName];
    } else if (newRole === 'objective') {
        delete window.boFeatureConfigs[colName];
        delete window.saFeatureConfigs[colName];
        delete window.priorConfigs[colName];
    } else if (newRole === 'prior') {
        delete window.boFeatureConfigs[colName];
        delete window.saFeatureConfigs[colName];
        if (!window.priorConfigs[colName]) {
            window.priorConfigs[colName] = {
                name: colName,
                col_type: 'categorical',
                target_value: '',
                tolerance: null,
                min_similarity: 0.5,
                prior_weight: 100,
                is_new_level: false
            };
        }
        detectPriorType(colName);
    }

    const activeObjs = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');
    activeObjs.forEach(col => {
        if (!window.objectiveConfigs[col]) {
            window.objectiveConfigs[col] = { name: col, type: 'maximize', target: '', importance: 0, included: true };
        } else {
            window.objectiveConfigs[col].included = true;
        }
    });
    Object.keys(window.objectiveConfigs).forEach(col => {
        if (!activeObjs.includes(col)) delete window.objectiveConfigs[col];
    });

    rebalanceObjectiveConfigs();
    renderMainTable();
    renderSetup();
    if (window.currentModule === 'bo') renderTrendPlot();
    if (window.currentModule === 'tbo') renderTBOVisualization();
    cedos.notifyConfigMutation('Dataset column role changed');
};

export function renderSetup() {
    if      (window.currentModule === 'doe') renderDoESetup();
    else if (window.currentModule === 'bo')  renderBOSetup();
    else if (window.currentModule === 'sa')  renderSASetup();
    else if (window.currentModule === 'tbo') renderTBOSetup();
}
window.renderSetup = renderSetup;

function renderDoESetup() {
    // Driven by DOM manually
}

window.updateObjectiveName = (oldName, newName) => {
    if (oldName === newName) return;
    if (window.objectiveConfigs[oldName]) {
        window.objectiveConfigs[newName]      = window.objectiveConfigs[oldName];
        window.objectiveConfigs[newName].name = newName;
        delete window.objectiveConfigs[oldName];
        renderSetup();
    }
};

window.syncObjectiveConfig = (col, sourceModule, field) => {
    if (sourceModule === 'doe') return;

    const rolePrefix = sourceModule === 'bo' ? 'bo-o' : 'sa-o';
    const row        = document.querySelector(`.${rolePrefix}-type[data-col="${col}"]`).closest('tr');

    document.querySelectorAll(`.${rolePrefix}-include`).forEach(cb => {
        const c = cb.dataset.col;
        const r = cb.closest('tr');
        if (window.objectiveConfigs[c]) {
            window.objectiveConfigs[c].type      = r.querySelector(`.${rolePrefix}-type`).value;
            window.objectiveConfigs[c].target    = r.querySelector(`.${rolePrefix}-target`).value;
            if (field === 'importance') window.objectiveConfigs[c].importance = r.querySelector(`.${rolePrefix}-importance`).value;
            window.objectiveConfigs[c].included  = cb.checked;
        }
    });

    if (field === 'importance') {
        const selector = `.${rolePrefix}-importance`;
        rebalanceRemaining(col, row.querySelector(selector).value, selector);
        document.querySelectorAll(selector).forEach(el => {
            if (window.objectiveConfigs[el.dataset.col]) window.objectiveConfigs[el.dataset.col].importance = el.value;
        });
    } else if (field === 'include' || field === 'type') {
        rebalanceObjectiveConfigs();
    }

    renderSetup();
    updateAcqSelectForMultiObj();
    if (window.currentModule === 'bo') renderTrendPlot();
    if (window.currentModule === 'tbo') renderTBOVisualization();
};

function initButtons() {
    safeAddListener('doe-add-feature-btn', 'click', () => {
        const list = document.getElementById('doe-features-list');
        const tr   = document.createElement('tr');
        tr.innerHTML = `
            <td><input type="text" class="doe-f-name" value="Feature_${list.children.length + 1}"></td>
            <td><select class="doe-f-type" onchange="window.updatePlaceholder(this)">
                <option value="continuous">continuous</option>
                <option value="regular">regular</option>
                <option value="discrete">discrete</option>
                <option value="categorical">categorical</option>
            </select></td>
            <td><input type="text" class="doe-f-range" placeholder="[min, max]"></td>
            <td><button class="btn secondary small delete-row">×</button></td>`;
        list.appendChild(tr);
        attachDeleteEvents();
    });

    safeAddListener('doe-add-objective-btn', 'click', () => {
        const list  = document.getElementById('doe-objectives-list');
        const count = list.children.length + 1;
        const tr    = document.createElement('tr');
        tr.innerHTML = `
            <td><input type="text" class="doe-o-name" value="Objective_${count}"></td>
            <td><select class="doe-o-type" onchange="window.toggleRowTarget(this)">
                <option value="maximize">maximize</option>
                <option value="minimize">minimize</option>
                <option value="target">target</option>
            </select></td>
            <td><input type="text" class="doe-o-target hidden" placeholder="Target"></td>
            <td><input type="number" class="doe-o-importance" value="0"></td>
            <td><button class="btn secondary small delete-row">×</button></td>`;
        list.appendChild(tr);
        attachDeleteEvents();
        rebalanceDoEDOM();
    });

    safeAddListener('doe-run-btn',      'click', runDoE);
    safeAddListener('run-btn',          'click', runBO);
    safeAddListener('sa-run-btn',       'click', runSA);
    safeAddListener('sa-abort-btn',     'click', abortSA);
    safeAddListener('sa-estimate-btn',  'click', runEstimate);
    safeAddListener('bo-estimate-btn',  'click', runBOEstimate);
    initTBOButtons();

    safeAddListener('doe-add-num-constraint-btn', 'click', () => window.addNumericalConstraintRow('doe'));
    safeAddListener('doe-add-disallowed-constraint-btn', 'click', () => window.addDisallowedCombinationRow('doe'));
    safeAddListener('bo-add-num-constraint-btn', 'click', () => window.addNumericalConstraintRow('bo'));
    safeAddListener('bo-add-disallowed-constraint-btn', 'click', () => window.addDisallowedCombinationRow('bo'));

    const doeFeatList = document.getElementById('doe-features-list');
    if (doeFeatList) {
        doeFeatList.addEventListener('input', () => updateAllConstraintsDropdowns('doe'));
        doeFeatList.addEventListener('change', () => updateAllConstraintsDropdowns('doe'));
    }

    safeAddListener('shutdown-btn', 'click', async () => {
        if (confirm('Are you sure you want to shut down the CEDOS server?')) {
            try {
                await fetch('/shutdown', { method: 'POST' });
            } catch (_) {}
            document.body.innerHTML = `
                <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;background:#0f172a;color:#e2e8f0;font-family:sans-serif;gap:20px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18.4 6.1a9 9 0 1 1-12.8 0"/><line x1="12" y1="2" x2="12" y2="12"/></svg>
                    <h1 style="font-size:2rem;margin:0;">Server Stopped</h1>
                    <p style="color:#64748b;max-width:400px;text-align:center;">The CEDOS server has been shut down. You can safely close this browser tab.</p>
                </div>`;
        }
    });

    safeAddListener('export-main-btn', 'click', () => exportData(window.currentData, window.currentColumns));

    safeAddListener('commit-suggestions-btn', 'click', (e) => {
        if (e.target.disabled) return;
        if (!cedos.canEditDataset()) return;
        if (!cedos.suggestionSnapshots.bo) return;
        const rows    = document.querySelectorAll('#results-table-body tr');
        const featCols = window.currentColumns.filter(c => window.columnRoles[c] === 'feature');
        const objCols  = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');

        let addedCount = 0;
        rows.forEach(tr => {
            const cells  = tr.querySelectorAll('td');
            const newRow = new Array(window.currentColumns.length).fill('');

            featCols.forEach((col, i) => {
                newRow[window.currentColumns.indexOf(col)] = cells[i].textContent.trim();
            });
            objCols.forEach((col, i) => {
                const val = cells[featCols.length + i].textContent.trim();
                newRow[window.currentColumns.indexOf(col)] = (val === 'N/D') ? '' : val;
            });

            window.currentData.push(newRow);
            addedCount++;
        });

        if (addedCount > 0) {
            cedos.notifyDatasetMutation('BO suggestions committed');
            renderMainTable();
            renderTrendPlot();
            renderParetoPlot();
            alert(`Added ${addedCount} experiments to the main dataset. Plots refreshed.`);
            document.getElementById('results-section').classList.add('hidden');
        }
    });

    safeAddListener('doe-commit-suggestions-btn', 'click', () => {
        if (!cedos.canEditDataset()) return;
        const doeSuggestions = cedos.suggestionSnapshots.doe || window.suggestionsData;
        if (!doeSuggestions || doeSuggestions.length === 0) return;
        const rows = document.querySelectorAll('#doe-results-body tr');
        if (rows.length === 0) return;

        const feats = Object.keys(doeSuggestions[0]);
        const objs  = window.currentDoEObjectives.map(o => o.name);

        if (!window.currentData || window.currentColumns.length === 0) {
            window.currentColumns = [...feats, ...objs];
            window.currentData    = [];
            feats.forEach(col => { window.columnRoles[col] = 'feature'; });
            objs.forEach(col  => { window.columnRoles[col] = 'objective'; });
        }

        window.currentDoEFeatures.forEach(f => {
            if (window.currentColumns.includes(f.name)) {
                window.boFeatureConfigs[f.name] = { type: f.type, range: f.range, choices: f.choices, included: true };
                window.saFeatureConfigs[f.name] = { type: f.type, included: true };
            }
        });
        window.currentDoEObjectives.forEach(o => {
            if (window.currentColumns.includes(o.name) && !window.objectiveConfigs[o.name]) {
                window.objectiveConfigs[o.name] = { name: o.name, type: 'maximize', target: '', importance: (100/objs.length).toFixed(1), included: true };
            }
        });

        const colIndices = {};
        let missing = [];
        [...feats, ...objs].forEach(name => {
            const idx = window.currentColumns.indexOf(name);
            if (idx === -1) missing.push(name);
            else colIndices[name] = idx;
        });

        if (missing.length > 0) {
            alert(`Cannot commit: The current main dataset is missing columns: ${missing.join(', ')}`);
            return;
        }

        let addedCount = 0;
        rows.forEach(tr => {
            const cells  = tr.querySelectorAll('td');
            const newRow = new Array(window.currentColumns.length).fill('');

            feats.forEach((name, i) => { newRow[colIndices[name]] = cells[i].textContent.trim(); });
            objs.forEach((name,  i) => {
                const val = cells[feats.length + i].textContent.trim();
                newRow[colIndices[name]] = (val === 'N/D') ? '' : val;
            });

            window.currentData.push(newRow);
            addedCount++;
        });

        if (addedCount > 0) {
            cedos.notifyDatasetMutation('DoE suggestions committed');
            renderMainTable();
            renderSetup();
            alert(`Added ${addedCount} experiments to the main dataset.`);
            document.getElementById('doe-results-section').classList.add('hidden');
            document.getElementById('data-section').classList.remove('hidden');
            switchModule(cedos.activeOptimizer);
        }
    });

    safeAddListener('doe-export-btn', 'click', () => {
        const doeSuggestions = cedos.suggestionSnapshots.doe || window.suggestionsData;
        if (!doeSuggestions || doeSuggestions.length === 0) return;
        const rows  = document.querySelectorAll('#doe-results-body tr');
        if (rows.length === 0) return;
        const feats = Object.keys(doeSuggestions[0]);
        const cols  = [...feats, ...window.currentDoEObjectives.map(o => o.name)];
        const exportArr = Array.from(rows).map(tr => Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim()));
        exportData(exportArr, cols);
    });

    safeAddListener('export-results-btn', 'click', () => {
        if (!cedos.suggestionSnapshots.bo?.length) return;
        const rows     = document.querySelectorAll('#results-table-body tr');
        if (rows.length === 0) return;
        const featCols = window.currentColumns.filter(c => window.columnRoles[c] === 'feature');
        const objCols  = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');
        const cols     = [...featCols, ...objCols.map(c => `${c} (Result)`)];
        const exportArr = Array.from(rows).map(tr => Array.from(tr.querySelectorAll('td')).map(td => td.textContent.trim()));
        exportData(exportArr, cols);
    });

    safeAddListener('pareto-export-btn', 'click', () => {
        const objCols      = window.currentColumns.filter(c => window.columnRoles[c] === 'objective');
        const paretoIndices = detectParetoIndices(window.currentData, objCols);
        const paretoData   = window.currentData.filter((_, i) => paretoIndices.includes(i));
        exportData(paretoData, window.currentColumns);
    });

    const applyHighlight = (tableId, type) => {
        const table = document.getElementById(tableId);
        if (!table) return;
        const ths = table.querySelectorAll('thead th');
        ths.forEach((th, idx) => {
            let matches = false;
            if (type === 'feature') {
                if (tableId === 'doe-results-table') {
                    const feats = cedos.suggestionSnapshots.doe?.length ? Object.keys(cedos.suggestionSnapshots.doe[0]) : [];
                    matches = idx < feats.length;
                } else {
                    matches = !th.textContent.includes('(Result)');
                }
            } else if (type === 'objective') {
                if (tableId === 'doe-results-table') {
                    const feats = cedos.suggestionSnapshots.doe?.length ? Object.keys(cedos.suggestionSnapshots.doe[0]) : [];
                    matches = idx >= feats.length;
                } else {
                    matches = th.textContent.includes('(Result)');
                }
            }
            if (matches) {
                th.classList.add(type === 'feature' ? 'highlight-feature-col' : 'highlight-objective-col');
                table.querySelectorAll('tbody tr').forEach(tr => {
                    const td = tr.children[idx];
                    if (td) td.classList.add(type === 'feature' ? 'highlight-feature-col' : 'highlight-objective-col');
                });
            }
        });
    };

    const removeHighlight = (tableId) => {
        const table = document.getElementById(tableId);
        if (table) table.querySelectorAll('.highlight-feature-col, .highlight-objective-col').forEach(el => {
            el.classList.remove('highlight-feature-col', 'highlight-objective-col');
        });
    };

    safeAddListener('doe-export-features-btn', 'mouseenter', () => applyHighlight('doe-results-table', 'feature'));
    safeAddListener('doe-export-features-btn', 'mouseleave', () => removeHighlight('doe-results-table'));
    safeAddListener('export-features-btn',     'mouseenter', () => applyHighlight('results-table', 'feature'));
    safeAddListener('export-features-btn',     'mouseleave', () => removeHighlight('results-table'));
    safeAddListener('doe-import-results-btn',  'mouseenter', () => applyHighlight('doe-results-table', 'objective'));
    safeAddListener('doe-import-results-btn',  'mouseleave', () => removeHighlight('doe-results-table'));
    safeAddListener('import-results-btn',      'mouseenter', () => applyHighlight('results-table', 'objective'));
    safeAddListener('import-results-btn',      'mouseleave', () => removeHighlight('results-table'));

    safeAddListener('doe-export-features-btn', 'click', () => {
        if (!cedos.suggestionSnapshots.doe?.length) return;
        const rows  = document.querySelectorAll('#doe-results-body tr');
        if (rows.length === 0) return;
        const feats = Object.keys(cedos.suggestionSnapshots.doe[0]);
        const exportArr = Array.from(rows).map(tr =>
            Array.from(tr.querySelectorAll('td')).slice(0, feats.length).map(td => td.textContent.trim())
        );
        exportData(exportArr, feats);
    });

    safeAddListener('export-features-btn', 'click', () => {
        if (!cedos.suggestionSnapshots.bo?.length) return;
        const rows     = document.querySelectorAll('#results-table-body tr');
        if (rows.length === 0) return;
        const featCols = window.currentColumns.filter(c => window.columnRoles[c] === 'feature');
        const exportArr = Array.from(rows).map(tr =>
            Array.from(tr.querySelectorAll('td')).slice(0, featCols.length).map(td => td.textContent.trim())
        );
        exportData(exportArr, featCols);
    });

    safeAddListener('doe-import-results-btn', 'click', () => document.getElementById('doe-import-results-input').click());
    safeAddListener('doe-import-results-input', 'change', (e) =>
        handleResultsImport(e, 'doe-results-table', window.currentDoEObjectives.map(o => o.name), false)
    );

    safeAddListener('import-results-btn',   'click', () => document.getElementById('import-results-input').click());
    safeAddListener('import-results-input', 'change', (e) =>
        handleResultsImport(e, 'results-table', window.currentColumns.filter(c => window.columnRoles[c] === 'objective'), true)
    );
}

function handleResultsImport(e, tableId, expectedObjCols, autoCommit) {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = '';
    if (!cedos.canEditDataset()) {
        cedos.showNotice('Measured results cannot be imported while an Auto campaign is active.', true);
        return;
    }
    const revision = cedos.captureRevision();
    const reader = new FileReader();
    reader.onload = event => {
        try {
            if (!cedos.isRevisionCurrent(revision) || !cedos.canEditDataset()) {
                throw new Error('Dataset, setup or campaign changed while the results CSV was loading. Upload it again.');
            }
            const records = parseResultsCSV(event.target.result);
            if (records.length < 2) throw new Error('CSV needs a header and at least one result row.');
            const headers = records[0].map(value => value.trim());
            if (new Set(headers).size !== headers.length) throw new Error('CSV headers must be unique.');
            const table = document.getElementById(tableId);
            const tableRows = Array.from(table.querySelectorAll('tbody tr'));
            const dataRows = records.slice(1).filter(row => row.some(value => value.trim() !== ''));
            if (dataRows.length !== tableRows.length) {
                throw new Error(`Expected ${tableRows.length} result rows but found ${dataRows.length}. No results were changed.`);
            }
            const tableHeaders = Array.from(table.querySelectorAll('thead th')).map(cell => cell.textContent.trim().replace(/ \(Result\)$/, ''));
            const featureCount = tableHeaders.length - expectedObjCols.length;
            if (featureCount < 0) throw new Error('The current result table no longer matches its objective setup.');
            const objectiveIndices = expectedObjCols.map(name => {
                const index = headers.findIndex(header => header === name || header === `${name} (Result)`);
                if (index < 0) throw new Error(`Missing objective column '${name}' in the results CSV.`);
                return index;
            });
            const featureIndices = tableHeaders.slice(0, featureCount).map(name => headers.indexOf(name));
            const hasAllFeatures = featureIndices.every(index => index >= 0);
            if (featureIndices.some(index => index >= 0) && !hasAllFeatures) {
                throw new Error('Include all feature columns or none of them so rows can be matched safely.');
            }
            const updates = [];
            dataRows.forEach((dataRow, rowIndex) => {
                if (dataRow.length !== headers.length) throw new Error(`CSV row ${rowIndex + 2} has a different number of columns.`);
                const cells = Array.from(tableRows[rowIndex].querySelectorAll('td'));
                if (cells.length !== tableHeaders.length) throw new Error('The result table changed during import.');
                if (hasAllFeatures) {
                    featureIndices.forEach((csvIndex, featureIndex) => {
                        if (dataRow[csvIndex].trim() !== cells[featureIndex].textContent.trim()) {
                            throw new Error(`Feature values in CSV row ${rowIndex + 2} do not match experiment ${rowIndex + 1}. Check row order.`);
                        }
                    });
                }
                objectiveIndices.forEach((csvIndex, objectiveIndex) => {
                    const value = dataRow[csvIndex].trim();
                    const cell = cells[featureCount + objectiveIndex];
                    if ((value === '' || value === 'N/D') && cell.textContent.trim() === 'N/D') return;
                    if (!value || !Number.isFinite(Number(value))) {
                        throw new Error(`Objective '${expectedObjCols[objectiveIndex]}' in CSV row ${rowIndex + 2} must be a finite number.`);
                    }
                    updates.push([cell, value]);
                });
            });
            if (!hasAllFeatures && !confirm('This CSV has no feature columns. Results will be matched by row order. Apply them?')) return;
            if (!cedos.isRevisionCurrent(revision)) throw new Error('The app changed during validation. Upload the results again.');
            updates.forEach(([cell, value]) => {
                cell.textContent = value;
                cell.dispatchEvent(new Event('input', { bubbles: true }));
            });
            alert(`Imported measured results for ${dataRows.length} experiments.`);
            if (autoCommit) {
                setTimeout(() => {
                    if (!cedos.isRevisionCurrent(revision)) return;
                    const commitBtn = document.getElementById('commit-suggestions-btn');
                    if (commitBtn && !commitBtn.disabled) commitBtn.click();
                }, 100);
            }
        } catch (error) {
            alert(`Could not import measured results: ${error.message}`);
        }
    };
    reader.readAsText(file);
}

export function parseResultsCSV(text) {
    const records = [];
    let row = [], field = '', quoted = false;
    const value = String(text || '').replace(/^\uFEFF/, '');
    for (let i = 0; i < value.length; i++) {
        const char = value[i];
        if (char === '"') {
            if (quoted && value[i + 1] === '"') { field += '"'; i++; }
            else if (quoted || field === '') quoted = !quoted;
            else throw new Error('Unexpected quote in CSV value.');
        } else if (char === ',' && !quoted) {
            row.push(field); field = '';
        } else if ((char === '\n' || char === '\r') && !quoted) {
            if (char === '\r' && value[i + 1] === '\n') i++;
            row.push(field); records.push(row); row = []; field = '';
        } else {
            field += char;
        }
    }
    if (quoted) throw new Error('Unclosed quoted value in CSV.');
    if (row.length || field) { row.push(field); records.push(row); }
    return records;
}

function initSettingsListeners() {
    const saModel = document.getElementById('sa-model-select');
    const updateSAParamsVisibility = () => {
        const val = saModel.value;
        document.querySelectorAll('.sa-rf-param').forEach(el  => el.classList.toggle('hidden', val !== 'random_forest'));
        document.querySelectorAll('.sa-mlp-param').forEach(el => el.classList.toggle('hidden', val !== 'mlp'));
    };
    saModel.addEventListener('change', updateSAParamsVisibility);
    updateSAParamsVisibility();

    safeAddListener('sa-export-report-btn', 'click', exportSAReport);

    const styleSlider = document.getElementById('style-slider');
    const styleNumber = document.getElementById('style-number');
    if (styleSlider && styleNumber) {
        styleSlider.addEventListener('input', (e) => styleNumber.value = e.target.value);
        styleNumber.addEventListener('input', (e) => styleSlider.value = e.target.value);
    }

    const attachNorm = (selector) => {
        document.addEventListener('change', (e) => {
            if (e.target.matches(selector)) {
                let inputs = Array.from(document.querySelectorAll(selector));
                const isDoe = selector.includes('doe');
                if (!isDoe) {
                    inputs = inputs.filter(el => {
                        const col = el.dataset.col;
                        return window.objectiveConfigs[col] && window.objectiveConfigs[col].included;
                    });
                }
                rebalanceRemaining(e.target, e.target.value, selector);

                if (!isDoe) {
                    Array.from(document.querySelectorAll(selector)).forEach(el => {
                        const col = el.dataset.col;
                        if (col && window.objectiveConfigs[col]) window.objectiveConfigs[col].importance = el.value;
                    });
                }

                if (!isDoe && window.currentModule === 'bo') renderTrendPlot();
            }
        });
    };
    attachNorm('.doe-o-importance');
    attachNorm('.bo-o-importance');
    attachNorm('.sa-o-importance');
}

window.toggleRowTarget = (select) => {
    const tr          = select.closest('tr');
    const targetInput = tr.querySelector('.bo-o-target, .doe-o-target, .sa-o-target');
    if (select.value === 'target') {
        targetInput.classList.remove('hidden');
    } else {
        targetInput.classList.add('hidden');
    }
};

window.updatePlaceholder = (select) => {
    const tr    = select.closest('tr');
    const input = tr.querySelector('.doe-f-range');
    updateRangePlaceholder(select.value, input);
};
