import {
    safeJson,
    wireSelectAll,
    rebalanceObjectiveConfigs,
    submitAndPollJob,
    cancelBackgroundJob
} from '../utils.js';

/**
 * Generic CSV downloader.
 * @param {string[][]} rows  - Array of arrays (first row = headers)
 * @param {string}     fname - Filename (without extension)
 */
function _downloadCSV(rows, fname) {
    const escape = v => {
        const s = (v === null || v === undefined) ? '' : String(v);
        return s.includes(',') || s.includes('"') || s.includes('\n')
            ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const csv  = rows.map(r => r.map(escape).join(',')).join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = fname.replace(/[^a-zA-Z0-9_\-]/g, '_') + '.csv';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}


export function renderSASetup() {
    const list    = document.getElementById('sa-features-list');
    const objList = document.getElementById('sa-objectives-list');

    let featHtml = '';
    window.currentColumns.filter(c => window.columnRoles[c] === 'feature').forEach(col => {
        const defaultType = window.boFeatureConfigs?.[col]?.type || 'continuous';
        const cfg = window.saFeatureConfigs[col] || { type: defaultType, included: true };
        featHtml += `<tr>
            <td>${col}</td>
            <td><select class="sa-f-type" data-col="${col}" onchange="window.syncSAFeatureConfig('${col}', 'type')">
                <option value="continuous" ${cfg.type === 'continuous' ? 'selected' : ''}>Numerical</option>
                <option value="discrete"   ${cfg.type === 'discrete'   ? 'selected' : ''}>Discrete</option>
                <option value="categorical"${cfg.type === 'categorical'? 'selected' : ''}>Categorical</option>
            </select></td>
            <td><input type="checkbox" class="sa-f-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="window.syncSAFeatureConfig('${col}', 'include')"> Include</td>
        </tr>`;
    });
    list.innerHTML = featHtml;

    let objHtml    = '';
    const selectedObjs = window.currentColumns.filter(col => window.columnRoles[col] === 'objective');
    selectedObjs.forEach(col => {
        const cfg       = window.objectiveConfigs[col] || { type: 'maximize', target: '', importance: (100/selectedObjs.length).toFixed(1), included: true };
        const isDisabled = !cfg.included;
        objHtml += `<tr class="${isDisabled ? 'row-disabled' : ''}">
            <td>${col}</td>
            <td><select class="sa-o-type" data-col="${col}" onchange="window.syncObjectiveConfig('${col}', 'sa', 'type'); window.toggleRowTarget(this)" ${isDisabled ? 'disabled' : ''}>
                <option value="maximize" ${cfg.type === 'maximize' ? 'selected' : ''}>maximize</option>
                <option value="minimize" ${cfg.type === 'minimize' ? 'selected' : ''}>minimize</option>
                <option value="target"   ${cfg.type === 'target'   ? 'selected' : ''}>target</option>
            </select></td>
            <td><input type="text" class="sa-o-target ${cfg.type === 'target' ? '' : 'hidden'}" data-col="${col}" value="${cfg.target || ''}" placeholder="Target" onchange="window.syncObjectiveConfig('${col}', 'sa', 'target')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="number" class="sa-o-importance" data-col="${col}" value="${cfg.importance}" onchange="window.syncObjectiveConfig('${col}', 'sa', 'importance')" ${isDisabled ? 'disabled' : ''}></td>
            <td><input type="checkbox" class="sa-o-include" data-col="${col}" ${cfg.included ? 'checked' : ''} onchange="window.syncObjectiveConfig('${col}', 'sa', 'include')"> Include</td>
        </tr>`;
    });
    objList.innerHTML = objHtml;

    // --- Select-all / deselect-all for SA Features ---
    wireSelectAll(
        'sa-select-all-features',
        '#sa-features-list .sa-f-include',
        (columns, checked) => {
            columns.forEach(col => {
                if (window.saFeatureConfigs[col]) {
                    window.saFeatureConfigs[col].included = checked;
                }
            });
            renderSASetup();
        }
    );

    // --- Select-all / deselect-all for SA Objectives ---
    wireSelectAll(
        'sa-select-all-objectives',
        '#sa-objectives-list .sa-o-include',
        (columns, checked) => {
            columns.forEach(col => {
                if (window.objectiveConfigs[col]) {
                    window.objectiveConfigs[col].included = checked;
                }
            });
            rebalanceObjectiveConfigs();
            renderSASetup();
        }
    );
    window.refreshObjectiveConstraintRows?.('sa');
}

let activeSAJobId = null;
let saAbortController = null;
let saCancellationRequested = false;
let activeSAEstimateJobId = null;

function waitForPoll(milliseconds, signal) {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, milliseconds);
        signal?.addEventListener('abort', () => {
            clearTimeout(timer);
            reject(new DOMException('Aborted', 'AbortError'));
        }, { once: true });
    });
}

export async function abortSA() {
    if (!activeSAJobId && !saAbortController) return;
    saCancellationRequested = true;
    const abortButton = document.getElementById('sa-abort-btn');
    if (abortButton) abortButton.disabled = true;
    const statusText = document.getElementById('sa-status-text');
    if (statusText) statusText.textContent = 'Requesting cancellation...';
    if (activeSAJobId) {
        try {
            await fetch(`/sa_cancel/${encodeURIComponent(activeSAJobId)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
        } catch (error) {
            console.warn('Unable to send SA cancellation request:', error);
        }
    }
    saAbortController?.abort();
}


export async function runSA() {
    const features = Array.from(document.querySelectorAll('.sa-f-include:checked')).map(cb => {
        const row = cb.closest('tr');
        return {
            name:  cb.dataset.col,
            type:  row.querySelector('.sa-f-type').value
        };
    });

    const objectives = Array.from(document.querySelectorAll('#sa-objectives-list tr'))
        .filter(tr => tr.querySelector('.sa-o-include').checked)
        .map(tr => ({
            name:       tr.querySelector('.sa-o-type').dataset.col,
            type:       tr.querySelector('.sa-o-type').value,
            target:     tr.querySelector('.sa-o-target').value,
            importance: tr.querySelector('.sa-o-importance').value
        }));

    window.saSelectedObjectives = objectives.map(o => o.name);
    window.saSelectedFeatures   = Array.from(document.querySelectorAll('.sa-f-include:checked')).map(cb => cb.dataset.col);

    let objectiveConstraints;
    try {
        objectiveConstraints = window.collectObjectiveConstraints?.('sa') || [];
    } catch (error) {
        alert(error.message);
        return;
    }

    const tweaks = {
        model: document.getElementById('sa-model-select').value,
        params: {
            n_estimators: document.getElementById('sa-rf-trees').value,
            max_depth:    document.getElementById('sa-rf-depth').value,
            mlp_layers:   document.getElementById('sa-mlp-layers').value,
            mlp_iter:     document.getElementById('sa-mlp-iter').value
        },
        objective_constraints: objectiveConstraints
    };

    const runBtn = document.getElementById('sa-run-btn');
    const progressContainer = document.getElementById('sa-progress-container');
    const progressBar = document.getElementById('sa-progress-bar');
    const statusText = document.getElementById('sa-status-text');
    const timerText = document.getElementById('sa-timer-text');
    const abortButton = document.getElementById('sa-abort-btn');

    if (runBtn) runBtn.disabled = true;
    if (abortButton) abortButton.disabled = false;
    if (progressContainer) progressContainer.classList.remove('hidden');
    document.getElementById('sa-results-container')?.classList.add('hidden');
    document.getElementById('sa-estimate-results').innerHTML = '';

    const startTime = Date.now();
    const tickerInterval = setInterval(() => {
        const elapsedSec = Math.round((Date.now() - startTime) / 1000);
        if (timerText) timerText.textContent = `${elapsedSec}s`;
    }, 400);

    saCancellationRequested = false;
    saAbortController = new AbortController();
    window.saAbortController = saAbortController;
    window.currentAbortController = saAbortController;
    try {
        const res = await fetch('/sa_submit', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ data: window.currentData, columns: window.currentColumns, features, objectives, tweaks }),
            signal: saAbortController.signal
        });
        const submitted = await safeJson(res);
        if (!res.ok || submitted.error) throw new Error(submitted.error || 'Unable to start analysis.');
        activeSAJobId = submitted.job_id;

        let result = null;
        while (!result) {
            await waitForPoll(1500, saAbortController.signal);
            const statusResponse = await fetch(
                `/sa_status/${encodeURIComponent(activeSAJobId)}`,
                { signal: saAbortController.signal }
            );
            const status = await safeJson(statusResponse);
            if (!statusResponse.ok || (status.error && status.status !== 'cancelled')) {
                throw new Error(status.error || 'Unable to read analysis status.');
            }
            if (statusText && status.phase) statusText.textContent = status.phase;
            if (progressBar) progressBar.style.width = `${Math.max(0, Math.min(100, status.percent || 0))}%`;
            if (status.status === 'completed') {
                result = status.result;
            } else if (status.status === 'failed') {
                throw new Error(status.error || 'Statistical analysis failed.');
            } else if (status.status === 'cancelled') {
                saCancellationRequested = true;
                throw new DOMException('Analysis aborted by user.', 'AbortError');
            }
        }

        window.lastSAResult = result;
        renderSAResults(result);
    } catch(err) {
        if (err.name === 'AbortError' || saCancellationRequested) {
            if (statusText) statusText.textContent = 'Analysis aborted by user.';
            setTimeout(() => progressContainer?.classList.add('hidden'), 1200);
            console.log('SA calculation aborted by user.');
        } else {
            alert("Statistical Analysis notice: " + err.message);
            console.error(err);
        }
    } finally {
        clearInterval(tickerInterval);
        if (progressContainer && !saCancellationRequested) progressContainer.classList.add('hidden');
        if (runBtn) runBtn.disabled = false;
        if (abortButton) abortButton.disabled = true;
        activeSAJobId = null;
        saAbortController = null;
        window.saAbortController = null;
        window.currentAbortController = null;
    }
}

function purgeAllPlots(container) {
    if (!container) return;
    const plots = container.querySelectorAll('.js-plotly-plot, .plot-container, [id^="sa-parcoords-"], [id^="sa-bar-"], [id^="sa-shap-"]');
    plots.forEach(el => {
        try { Plotly.purge(el); } catch (e) {}
    });
}

export function renderSAResults(result) {
    document.getElementById('sa-results-container').classList.remove('hidden');

    document.getElementById('sa-approximation-notice')?.remove();
    if (result.metadata?.shap_approximated) {
        const notice = document.createElement('div');
        notice.id = 'sa-approximation-notice';
        notice.style.cssText = 'margin:0.75rem 0;padding:0.8rem 1rem;border-radius:8px;background:#eff6ff;border:1px solid #93c5fd;color:#1e40af;';
        notice.textContent = `SHAP values were approximated for this ${result.metadata.original_rows}-row dataset using ${result.metadata.shap_background_rows} background rows and ${result.metadata.shap_nsamples} samples.`;
        document.getElementById('sa-results-container')?.prepend(notice);
    }

    // Purge any existing WebGL & SVG Plotly instances to free GPU WebGL contexts
    purgeAllPlots(document.getElementById('sa-parcoords-container'));
    purgeAllPlots(document.getElementById('sa-impact-container'));
    purgeAllPlots(document.getElementById('sa-cat-interactions-container'));
    try { Plotly.purge('sa-num-corr'); } catch(e) {}

    const numCorr  = result.numerical_correlation;
    const numFeats = result.num_features;
    const objNames = result.extended_obj_names;

    if (numFeats.length > 0) {
        const zNum = objNames.map(obj => numFeats.map(f => (numCorr[f] || {})[obj] ?? 0));
        Plotly.newPlot('sa-num-corr',
            [{ z: zNum, x: numFeats, y: objNames, type: 'heatmap',
               colorscale: 'RdBu', zmin: -1, zmax: 1,
               text: zNum.map(row => row.map(v => v.toFixed(2))),
               hovertemplate: 'Feature: %{x}<br>Objective: %{y}<br>Correlation: %{z:.3f}<extra></extra>',
               texttemplate: '%{text}', showscale: true }],
            {
                xaxis: { title: 'Numerical Features',  automargin: true },
                yaxis: { title: 'Success Targets',      automargin: true },
                margin: { l: 150, b: 80, t: 30, r: 50 },
                height: 400
            });

        // Wire correlation CSV download
        const corrBtn = document.getElementById('sa-corr-csv-btn');
        if (corrBtn) {
            corrBtn.onclick = () => {
                const header = ['Feature / Objective', ...objNames];
                const rows   = [header, ...numFeats.map(f =>
                    [f, ...objNames.map(obj => ((numCorr[f] || {})[obj] ?? 0).toFixed(4))]
                )];
                _downloadCSV(rows, 'SA_Numerical_Correlation');
            };
        }
    } else {
        document.getElementById('sa-num-corr').innerHTML = '<p class="text-secondary"><i>No numerical features available for correlation.</i></p>';
        const corrBtn = document.getElementById('sa-corr-csv-btn');
        if (corrBtn) corrBtn.disabled = true;
    }

    const catContainer = document.getElementById('sa-cat-interactions-container');
    const catControls  = document.getElementById('sa-cat-controls');
    catContainer.innerHTML = '';
    catControls.innerHTML  = '';

    const interactions = result.cat_interactions;
    const pairs        = Object.keys(interactions);

    if (pairs.length > 0) {
        catControls.innerHTML = `<div class="tweak-item">
            <label>Select Pair:</label>
            <select id="sa-cat-pair-select" style="width: 100%;">
                ${pairs.map(p => `<option value="${p}">${p}</option>`).join('')}
            </select>
        </div>
        <div class="tweak-item">
            <label>Select Objective:</label>
            <select id="sa-cat-obj-select" style="width: 100%;">
                ${objNames.map(o => `<option value="${o}">${o}</option>`).join('')}
            </select>
        </div>`;

        const updateInteractionPlot = () => {
            const pair = document.getElementById('sa-cat-pair-select').value;
            const obj  = document.getElementById('sa-cat-obj-select').value;
            const data = (interactions[pair] || {})[obj];
            if (!data) {
                catContainer.innerHTML = '<p class="text-secondary"><i>No data found for this combination.</i></p>';
                return;
            }
            try { Plotly.purge(catContainer); } catch(e) {}
            const trace = {
                z: data.z, x: data.x, y: data.y,
                type: 'heatmap', colorscale: 'RdBu', zmin: 0, zmax: 1,
                hovertemplate: `<b>Best Achievement</b><br>${pair.split(' vs ')[1]}: %{x}<br>${pair.split(' vs ')[0]}: %{y}<br>Success Score: %{z:.3f}<extra></extra>`
            };
            Plotly.newPlot(catContainer, [trace], {
                title: `Optimal Outcome: ${obj} (Max Success)`,
                xaxis: { title: pair.split(' vs ')[1], automargin: true },
                yaxis: { title: pair.split(' vs ')[0], automargin: true },
                margin: { l: 150, b: 80, t: 50, r: 50 },
                height: 500
            });

            // Keep the CSV button in sync with the current view
            const catCsvBtn = document.getElementById('sa-cat-csv-btn');
            if (catCsvBtn) {
                catCsvBtn.onclick = () => {
                    const [rowFeat, colFeat] = pair.split(' vs ');
                    const header = [rowFeat + ' \\ ' + colFeat, ...data.x];
                    const rows   = [header, ...data.y.map((yVal, ri) =>
                        [yVal, ...data.z[ri].map(v => v.toFixed(4))]
                    )];
                    _downloadCSV(rows, `SA_CatInteraction_${pair.replace(/ /g,'_')}_${obj}`);
                };
            }
        };

        document.getElementById('sa-cat-pair-select').addEventListener('change', updateInteractionPlot);
        document.getElementById('sa-cat-obj-select').addEventListener('change',  updateInteractionPlot);
        updateInteractionPlot();
    } else {
        catContainer.innerHTML = '<p class="text-secondary"><i>No categorical interactions possible (requires 2+ categorical features).</i></p>';
        const catCsvBtn = document.getElementById('sa-cat-csv-btn');
        if (catCsvBtn) catCsvBtn.disabled = true;
    }

    const parcoordsContainer = document.getElementById('sa-parcoords-container');
    const parcoordsControls  = document.getElementById('sa-parcoords-controls');
    if (parcoordsControls) parcoordsControls.innerHTML = '';
    purgeAllPlots(parcoordsContainer);
    parcoordsContainer.innerHTML = '';

    const featCols = (window.saSelectedFeatures || window.currentColumns.filter(c => window.columnRoles[c] === 'feature'));
    const objCols  = (window.saSelectedObjectives || window.currentColumns.filter(c => window.columnRoles[c] === 'objective'));

    // Cache parcoords data for all objectives so report generator can embed them
    window.saParcoordsPlotData = {};

    if (featCols.length > 0 && objCols.length > 0 && window.currentData.length > 0) {
        if (parcoordsControls) {
            parcoordsControls.innerHTML = `<div class="tweak-item" style="max-width: 400px;">
                <label>Select Objective to Color By:</label>
                <select id="sa-parcoords-obj-select" style="width: 100%;">
                    ${objCols.map(o => `<option value="${o}">${o}</option>`).join('')}
                </select>
            </div>`;
        }

        // Build data structure for all objectives
        objCols.forEach(objCol => {
            const plotCols = [...featCols, objCol];
            const dimensions = plotCols.map(col => {
                const colIdx  = window.currentColumns.indexOf(col);
                const rawVals = window.currentData.map(r => r[colIdx]);
                // Filter out null/undefined/empty before deciding if column is categorical.
                // parseFloat(null) === NaN, so a numeric column with missing results would
                // otherwise be misclassified as categorical and rendered with string tick-labels.
                const nonEmptyRaw = rawVals.filter(v => v !== null && v !== undefined && v !== '');
                const numVals     = rawVals.map(v => parseFloat(v));
                const isCat       = nonEmptyRaw.some(v => isNaN(parseFloat(v)));

                if (isCat) {
                    const uniqueVals = [...new Set(rawVals.filter(v => v !== '' && v !== null && v !== undefined))];
                    return {
                        label: col,
                        values:   rawVals.map(v => uniqueVals.indexOf(v) >= 0 ? uniqueVals.indexOf(v) : -1),
                        tickvals: uniqueVals.map((_, i) => i),
                        ticktext: uniqueVals,
                        range: [0, Math.max(1, uniqueVals.length - 1)]
                    };
                } else {
                    // Use only the finite (non-null) values for range; map nulls to null so
                    // Plotly skips them rather than snapping them to 0.
                    const cleanVals   = numVals.map(v => isFinite(v) ? v : null);
                    const finiteVals  = cleanVals.filter(v => v !== null);
                    let min = finiteVals.length > 0 ? Math.min(...finiteVals) : 0;
                    let max = finiteVals.length > 0 ? Math.max(...finiteVals) : 1;
                    if (min === max) { min -= 0.5; max += 0.5; }
                    return { label: col, values: cleanVals, range: [min, max] };
                }
            });

            const objIdx    = window.currentColumns.indexOf(objCol);
            const colorVals = window.currentData.map(r => { const v = parseFloat(r[objIdx]); return isFinite(v) ? v : NaN; });
            const finiteColorVals = colorVals.filter(v => isFinite(v));
            let cmin = finiteColorVals.length > 0 ? Math.min(...finiteColorVals) : 0;
            let cmax = finiteColorVals.length > 0 ? Math.max(...finiteColorVals) : 1;
            if (cmin === cmax) { cmin = cmin - 0.5; cmax = cmax + 0.5; }

            window.saParcoordsPlotData[objCol] = {
                plotCols,
                data: [{
                    type: 'parcoords',
                    line: {
                        color: colorVals,
                        colorscale: [
                            [0, '#0d0887'], [0.11, '#46039f'], [0.22, '#7201a8'], [0.33, '#9c179e'],
                            [0.44, '#bd3786'], [0.55, '#d8576b'], [0.66, '#ed7953'], [0.77, '#fb9f3a'],
                            [0.88, '#fdca26'], [1, '#f0f921']
                        ],
                        showscale: true, reversescale: false, opacity: 0.85,
                        cmin, cmax,
                        colorbar: { title: objCol, thickness: 15 }
                    },
                    unselected: { line: { opacity: 0, color: 'rgba(0,0,0,0)' } },
                    dimensions
                }],
                layout: {
                    title: `Parallel Coordinates — Colored by: ${objCol}`,
                    margin: { l: 80, r: 80, t: 80, b: 20 },
                    height: 500
                }
            };
        });

        const renderActiveParcoords = () => {
            const selectEl = document.getElementById('sa-parcoords-obj-select');
            const objCol   = selectEl ? selectEl.value : objCols[0];
            const pObj     = window.saParcoordsPlotData[objCol];
            if (!pObj) return;

            // Wire CSV button
            const pcCsvBtn = document.getElementById('sa-parcoords-csv-btn');
            if (pcCsvBtn) {
                pcCsvBtn.disabled = false;
                pcCsvBtn.onclick = () => {
                    const header = pObj.plotCols;
                    const rows   = [header, ...window.currentData.map(r =>
                        pObj.plotCols.map(col => r[window.currentColumns.indexOf(col)])
                    )];
                    _downloadCSV(rows, `SA_ParCoords_${objCol}`);
                };
            }

            try {
                purgeAllPlots(parcoordsContainer);
                Plotly.newPlot(parcoordsContainer, pObj.data, pObj.layout, { responsive: true });
            } catch (plotErr) {
                console.warn(`Parcoords plot failed for objective "${objCol}":`, plotErr);
                parcoordsContainer.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;font-style:italic;">Could not render parallel coordinates for <b style="margin-left:4px;">${objCol}</b> — insufficient or non-numeric data.</div>`;
            }
        };

        const selectEl = document.getElementById('sa-parcoords-obj-select');
        if (selectEl) selectEl.addEventListener('change', renderActiveParcoords);
        renderActiveParcoords();
    } else {
        parcoordsContainer.innerHTML = '<p class="text-secondary"><i>Parallel coordinates require at least one feature and one objective with experiments loaded.</i></p>';
        const pcCsvBtn = document.getElementById('sa-parcoords-csv-btn');
        if (pcCsvBtn) pcCsvBtn.disabled = true;
    }

    const impactContainer = document.getElementById('sa-impact-container');
    const impactControls  = document.getElementById('sa-impact-controls');
    if (impactControls) impactControls.innerHTML = '';
    purgeAllPlots(impactContainer);
    impactContainer.innerHTML = '';

    const globalCatMap = {};
    objNames.forEach(obj => {
        const res = result.results[obj];
        if (res && res.shap_data && res.shap_data.features) {
            res.shap_data.features.forEach((feat, i) => {
                const fVals = res.shap_data.feature_values[i];
                let isCategorical = fVals.some(v => isNaN(parseFloat(v)));
                if (isCategorical) {
                    if (!globalCatMap[feat]) {
                        globalCatMap[feat] = new Set();
                    }
                    fVals.forEach(v => globalCatMap[feat].add(v));
                }
            });
        }
    });
    Object.keys(globalCatMap).forEach(feat => {
        globalCatMap[feat] = Array.from(globalCatMap[feat]).sort();
    });

    window.saImpactPlotData = {};

    if (objNames.length > 0) {
        if (impactControls) {
            impactControls.innerHTML = `<div class="tweak-item" style="max-width: 400px;">
                <label>Select Objective to Inspect:</label>
                <select id="sa-impact-obj-select" style="width: 100%;">
                    ${objNames.map(o => `<option value="${o}">${o === 'global_success' ? '✨ Global Success Index' : o}</option>`).join('')}
                </select>
            </div>`;
        }

        // Build plot configurations for all objectives
        objNames.forEach(obj => {
            const res = result.results[obj];
            if (!res || !res.importance) return;

            const imp           = res.importance;
            const sortedEntries = Object.entries(imp).sort((a, b) => a[1] - b[1]);
            const yFeats        = sortedEntries.map(e => e[0]);
            const xVals         = sortedEntries.map(e => e[1]);
            const hasShap       = res.shap_data && res.shap_data.features && res.shap_data.features.length > 0;

            const barData = [{
                x: xVals, y: yFeats, type: 'bar', orientation: 'h',
                marker: { color: obj === 'global_success' ? '#FFD700' : 'var(--accent-color, #3b82f6)' }
            }];

            const barLayout = {
                title: 'Feature Importance (%)',
                xaxis: { title: 'Importance (%)', range: [0, 100] },
                yaxis: { automargin: true },
                margin: { l: 120, b: 50, t: 30, r: 20 },
                showlegend: false, height: 400
            };

            let shapTraces = [];
            let shapLayout = null;

            if (hasShap) {
                const sd = res.shap_data;
                sd.features.forEach((feat, i) => {
                    const sVals = sd.shap_values[i];
                    const fVals = sd.feature_values[i];

                    let numericFVals = fVals.map(v => parseFloat(v));
                    let isCategorical = numericFVals.some(isNaN);
                    let colorArray, cmin, cmax, colorscale, reversescale;

                    if (isCategorical) {
                        const uniqueStr = globalCatMap[feat] || [...new Set(fVals)].sort();
                        colorArray   = fVals.map(v => uniqueStr.indexOf(v));
                        colorscale   = 'Portland'; reversescale = false;
                    } else {
                        colorArray  = numericFVals;
                        colorscale  = 'RdBu';
                        cmin        = Math.min(...colorArray);
                        cmax        = Math.max(...colorArray);
                        reversescale = true;
                    }

                    const yBase = yFeats.indexOf(feat);
                    shapTraces.push({
                        x: sVals,
                        y: sVals.map((_, pointIndex) => {
                            const deterministicJitter = ((pointIndex * 0.61803398875) % 1) - 0.5;
                            return yBase + deterministicJitter * 0.4;
                        }),
                        mode: 'markers', type: 'scatter',
                        marker: {
                            size: 7, opacity: 0.6, color: colorArray,
                            colorscale, cmin, cmax, reversescale,
                            line: { width: 0.5, color: '#fff' }
                        },
                        customdata: fVals,
                        hovertemplate: `<b>Value:</b> %{customdata}<br><b>Impact:</b> %{x:.3f}<extra></extra>`
                    });
                });

                const robustLimit = Number(sd.display_limit);
                const xAxis = {
                    title: 'SHAP Value (Impact)',
                    zeroline: true,
                    zerolinecolor: '#94a3b8'
                };
                if (Number.isFinite(robustLimit) && robustLimit > 0) {
                    xAxis.range = [-robustLimit, robustLimit];
                }
                const clippedPoints = Number(sd.display_clipped_points || 0);
                shapLayout = {
                    title: 'SHAP Value Distribution',
                    xaxis: xAxis,
                    yaxis: { showticklabels: false, range: [-0.5, yFeats.length - 0.5] },
                    margin: { l: 20, b: 50, t: 30, r: 20 },
                    showlegend: false,
                    height: 400,
                    annotations: clippedPoints > 0 ? [{
                        text: `${clippedPoints} extreme point${clippedPoints === 1 ? '' : 's'} outside display range; retained in CSV`,
                        xref: 'paper', yref: 'paper', x: 1, y: 1.08,
                        xanchor: 'right', showarrow: false,
                        font: { size: 10, color: '#64748b' }
                    }] : []
                };
            }

            window.saImpactPlotData[obj] = {
                obj,
                objDisplay: obj === 'global_success' ? 'Global Success Index' : obj,
                sortedEntries,
                hasShap,
                rawShapData: res.shap_data,
                barData,
                barLayout,
                shapTraces,
                shapLayout
            };
        });

        const renderActiveImpact = () => {
            const selectEl = document.getElementById('sa-impact-obj-select');
            const obj      = selectEl ? selectEl.value : objNames[0];
            const pObj     = window.saImpactPlotData[obj];
            if (!pObj) return;

            purgeAllPlots(impactContainer);
            impactContainer.innerHTML = '';

            const container = document.createElement('div');
            container.className = 'impact-card';
            container.style.cssText = 'display: flex; flex-direction: column; gap: 15px; padding: 1.25rem; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0;';

            // Top Header Row with Title and Aligned CSV Export Buttons
            const headerRow = document.createElement('div');
            headerRow.style.cssText = 'width: 100%; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; border-bottom: 1px dashed #cbd5e1; padding-bottom: 0.75rem;';

            const titleDiv = document.createElement('h3');
            titleDiv.textContent = `Feature Impact: ${pObj.objDisplay}`;
            titleDiv.style.margin = '0';
            headerRow.appendChild(titleDiv);

            const btnGroup = document.createElement('div');
            btnGroup.style.cssText = 'display: flex; gap: 8px; align-items: center;';

            // Importance CSV Button
            const barCsvBtn = document.createElement('button');
            barCsvBtn.className = 'btn small';
            barCsvBtn.textContent = '⬇ Importance CSV';
            barCsvBtn.onclick = () => {
                const rows = [['Feature', 'Importance (%)'], ...pObj.sortedEntries.map(([f, v]) => [f, v.toFixed(4)])];
                _downloadCSV(rows, `SA_Importance_${obj}`);
            };
            btnGroup.appendChild(barCsvBtn);

            // SHAP CSV Button
            const shapCsvBtn = document.createElement('button');
            shapCsvBtn.className = 'btn small';
            shapCsvBtn.textContent = '⬇ SHAP CSV';
            if (pObj.hasShap) {
                shapCsvBtn.onclick = () => {
                    if (!pObj.rawShapData || !pObj.rawShapData.features) return;
                    const sd = pObj.rawShapData;
                    const csvRows = [['Feature', 'Feature_Value', 'SHAP_Value']];
                    sd.features.forEach((feat, i) => {
                        sd.shap_values[i].forEach((sv, j) => {
                            csvRows.push([feat, sd.feature_values[i][j], sv.toFixed(6)]);
                        });
                    });
                    _downloadCSV(csvRows, `SA_SHAP_${obj}`);
                };
            } else {
                shapCsvBtn.disabled = true;
                shapCsvBtn.style.opacity = '0.5';
            }
            btnGroup.appendChild(shapCsvBtn);

            headerRow.appendChild(btnGroup);
            container.appendChild(headerRow);

            // Symmetrical Plot Row: Both bar and SHAP plots are direct peers
            const plotRow = document.createElement('div');
            plotRow.style.cssText = 'display: flex; flex-wrap: wrap; gap: 15px; width: 100%; align-items: stretch;';

            const barDiv = document.createElement('div');
            barDiv.id = `sa-bar-active`;
            barDiv.style.cssText = 'flex: 1; min-width: 350px; height: 400px;';
            plotRow.appendChild(barDiv);

            const shapDiv = document.createElement('div');
            shapDiv.id = `sa-shap-active`;
            shapDiv.style.cssText = 'flex: 1.3; min-width: 450px; height: 400px;';
            plotRow.appendChild(shapDiv);

            container.appendChild(plotRow);
            impactContainer.appendChild(container);

            Plotly.newPlot(barDiv.id, pObj.barData, pObj.barLayout, { responsive: true });

            if (pObj.hasShap) {
                Plotly.newPlot(shapDiv.id, pObj.shapTraces, pObj.shapLayout, { responsive: true });
            } else {
                shapDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;font-style:italic;">No SHAP data available for this model.</div>';
            }
        };

        const selectEl = document.getElementById('sa-impact-obj-select');
        if (selectEl) selectEl.addEventListener('change', renderActiveImpact);
        renderActiveImpact();
    } else {
        impactContainer.innerHTML = '<p class="text-secondary"><i>No objectives available for feature impact analysis.</i></p>';
    }

    const getFittingSuggestion = (status) => {
        switch(status) {
            case 'Good':    return 'Reliable model for interpolation.';
            case 'Limited': return 'Moderate predictive power; use with caution.';
            case 'Poor':    return 'Low predictive power; consider removing irrelevant features or conducting more experiments to improve data coverage.';
            default:        return '';
        }
    };

    let relHtml = `
        <div style="margin-bottom: 10px; font-weight: 600; color: var(--accent-color);">
            Active Model: ${document.getElementById('sa-model-select').options[document.getElementById('sa-model-select').selectedIndex].text}
        </div>
        <div style="margin-bottom: 12px; font-size: 0.82rem; color: var(--text-secondary); background: var(--bg-secondary, #f8fafc); border-left: 3px solid var(--accent-color); padding: 8px 12px; border-radius: 4px;">
            ℹ️ <b>Note:</b> SA R² is the mean of five deterministic held-out validation splits in normalized success space [0–1].
            It is intentionally stricter than TransBO's in-sample training-fit R², so the values should not be compared directly.
        </div>
        <table class="setup-table" style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="text-align: left; border-bottom: 2px solid var(--border-color);">
                    <th style="padding: 10px;">Objective</th>
                    <th style="padding: 10px;">Held-out R² <span title="Mean and variation across five deterministic held-out validation splits in normalized success space [0–1]." style="cursor:help; color:var(--text-secondary); font-size:0.85em;">ⓘ</span></th>
                    <th style="padding: 10px;">Status</th>
                    <th style="padding: 10px;">Suggestion</th>
                </tr>
            </thead>
            <tbody>`;

    objNames.forEach(obj => {
        const r          = result.results[obj];
        const statusClass = r.fit_flag.toLowerCase();
        const reliabilityDisplay = r.reliability === null || r.reliability === undefined
            ? 'N/A'
            : `${r.reliability}${Number.isFinite(r.reliability_std) ? ` ± ${r.reliability_std}` : ''}`;
        let nameDisplay   = `<b>${obj}</b>`;
        if (obj === 'global_success') nameDisplay = `<span style="color: var(--accent-color);">✨ <b>Global Success Index</b></span>`;
        if (obj === 'pareto_optimal') nameDisplay = `<i>Pareto Optimal Flag</i>`;

        relHtml += `
            <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 10px;">${nameDisplay}</td>
                <td style="padding: 10px;">${reliabilityDisplay}</td>
                <td style="padding: 10px;"><span class="badge ${statusClass}">${r.fit_flag}</span></td>
                <td style="padding: 10px; font-size: 0.9rem; color: var(--text-secondary);">${getFittingSuggestion(r.fit_flag)}</td>
            </tr>`;
    });
    document.getElementById('sa-reliability-info').innerHTML = relHtml;

    document.getElementById('sa-cat-interactions-container').style.minHeight = '500px';

    const estDiv = document.getElementById('sa-estimator-inputs');
    const feats  = window.saSelectedFeatures || window.currentColumns.filter(c => window.columnRoles[c] === 'feature');

    estDiv.innerHTML = feats.map(f => {
        const colIdx     = window.currentColumns.indexOf(f);
        const uniqueVals = [...new Set(window.currentData.map(r => r[colIdx]).filter(v => v !== null && v !== ''))];
        const isNumeric  = uniqueVals.every(v => !isNaN(parseFloat(v)) && isFinite(v));

        if (isNumeric) {
            return `<div class="tweak-item">
                <label>${f}</label>
                <input type="number" class="sa-est-input" data-col="${f}" step="any">
            </div>`;
        } else {
            const options = uniqueVals.map(v => `<option value="${v}">${v}</option>`).join('');
            return `<div class="tweak-item">
                <label>${f}</label>
                <select class="sa-est-input" data-col="${f}">${options}</select>
            </div>`;
        }
    }).join('');
}

export async function runEstimate() {
    const inputs  = {};
    document.querySelectorAll('.sa-est-input').forEach(el => inputs[el.dataset.col] = el.value);

    const featList = (window.saSelectedFeatures || window.currentColumns.filter(c => window.columnRoles[c] === 'feature'))
        .map(c => {
            const defaultType = window.boFeatureConfigs?.[c]?.type || 'continuous';
            const cfg = window.saFeatureConfigs[c] || { type: defaultType };
            return {
                name: c,
                type: cfg.type
            };
        });

    const objList = Array.from(document.querySelectorAll('#sa-objectives-list tr'))
        .filter(tr => tr.querySelector('.sa-o-include').checked)
        .map(tr => ({
            name:       tr.querySelector('.sa-o-type').dataset.col,
            type:       tr.querySelector('.sa-o-type').value,
            target:     tr.querySelector('.sa-o-target').value,
            importance: tr.querySelector('.sa-o-importance').value
        }));

    let objectiveConstraints;
    try {
        objectiveConstraints = window.collectObjectiveConstraints?.('sa') || [];
    } catch (error) {
        document.getElementById('sa-estimate-results').textContent = error.message;
        return;
    }

    const resultsElement = document.getElementById('sa-estimate-results');
    const estimateButton = document.getElementById('sa-estimate-btn');
    const estimateProgress = document.getElementById('sa-estimate-progress');
    const estimateStatus = document.getElementById('sa-estimate-status');
    const estimateBar = document.getElementById('sa-estimate-progress-bar');
    const estimateAbort = document.getElementById('sa-estimate-abort-btn');
    resultsElement.innerHTML = '<div style="color: var(--text-secondary); animation: pulse 1.5s infinite;"><i>Calculating predictions...</i></div>';

    const payload = {
        data: window.currentData,
        columns: window.currentColumns,
        features: featList,
        objectives: objList,
        inputs,
        model: document.getElementById('sa-model-select').value,
        params: {
            n_estimators: document.getElementById('sa-rf-trees').value,
            max_depth: document.getElementById('sa-rf-depth').value,
            mlp_layers: document.getElementById('sa-mlp-layers').value,
            mlp_iter: document.getElementById('sa-mlp-iter').value
        },
        tweaks: { objective_constraints: objectiveConstraints }
    };

    const estimateController = new AbortController();
    if (estimateButton) estimateButton.disabled = true;
    if (estimateProgress) estimateProgress.classList.remove('hidden');
    if (estimateAbort) {
        estimateAbort.disabled = false;
        estimateAbort.onclick = () => estimateController.abort();
    }

    try {
        const completed = await submitAndPollJob('/sa_estimate_submit', payload, {
            signal: estimateController.signal,
            onProgress: (status, jobId) => {
                activeSAEstimateJobId = jobId;
                if (estimateStatus && status.phase) estimateStatus.textContent = status.phase;
                if (estimateBar && Number.isFinite(Number(status.percent))) {
                    estimateBar.style.width = `${Math.max(10, Math.min(98, Number(status.percent)))}%`;
                }
            }
        });
        const result = completed.result;

        let html = '<div style="margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid var(--border-color);">';
        html += `<b>Global Success Score: <span style="font-size: 1.2rem; color: var(--accent-color);">${result.global_success}%</span></b></div>`;
        html += '<b>Predictions:</b><br>';
        const violationMap = new Map((result.violations || []).map(item => [item.objective, item]));
        for (const obj in result.predictions) {
            const val  = result.predictions[obj];
            const err  = result.std_errors ? result.std_errors[obj] : null;
            const succ = result.success_scores[obj];
            const violation = violationMap.get(obj);
            const violationBadge = violation
                ? `<span title="Predicted ${violation.predicted ?? 'N/A'} violates ${violation.operator} ${violation.limit}" style="display:inline-block;margin-left:6px;color:#b91c1c;font-weight:700;">🚫 Violated: ${violation.operator} ${violation.limit}</span>`
                : '';
            const errBadge = (err !== null && err !== undefined) ? `<span style="font-size: 0.85rem; color: var(--text-secondary); margin-left: 4px;">(&plusmn; ${err})</span>` : '';
            html += `<div style="display: flex; justify-content: space-between; align-items: center; margin: 4px 0;">
                        <span><b>${obj}:</b> ${val}${errBadge}${violationBadge}</span>
                        <span style="color: ${succ > 80 ? '#22c55e' : (succ > 50 ? '#f59e0b' : '#ef4444')}">Success: ${succ}%</span>
                     </div>`;
        }
        resultsElement.innerHTML = html;
    } catch(err) {
        if (err.name === 'AbortError') {
            resultsElement.innerHTML = '<span style="color:var(--text-secondary)">Estimation cancelled.</span>';
        } else {
            resultsElement.innerHTML = `<span style="color:red">Request failed: ${err.message}</span>`;
            console.error(err);
        }
    } finally {
        if (estimateController.signal.aborted) {
            await cancelBackgroundJob(activeSAEstimateJobId);
        }
        activeSAEstimateJobId = null;
        if (estimateButton) estimateButton.disabled = false;
        if (estimateProgress) estimateProgress.classList.add('hidden');
        if (estimateAbort) estimateAbort.disabled = true;
    }
}

export async function exportSAReport() {
    const overlay = document.getElementById('loading-overlay');
    document.getElementById('loading-text').textContent = "Generating Dynamic Report...";
    overlay.classList.remove('hidden');

    try {
        const timestamp   = new Date().toLocaleString();
        const activeModel = document.getElementById('sa-model-select').options[document.getElementById('sa-model-select').selectedIndex].text;

        const plotsData  = [];
        const capturePlot = (id, title) => {
            const el = document.getElementById(id);
            if (!el || !el.data) return;
            plotsData.push({ id, title, data: el.data, layout: el.layout });
        };

        capturePlot('sa-num-corr', 'Numerical Correlation Matrix');
        document.querySelectorAll('[id^="sa-parcoords-"]').forEach(el => {
            capturePlot(el.id, el.layout?.title?.text || el.layout?.title || 'Parallel Coordinates');
        });
        document.querySelectorAll('[id^="sa-bar-"], [id^="sa-shap-"]').forEach(el => {
            const h3 = el.parentElement ? el.parentElement.querySelector('h3') : null;
            const title = (h3 ? h3.textContent : null) || el.layout?.title?.text || el.layout?.title || 'Feature Impact Group';
            capturePlot(el.id, title);
        });

        const reliabilityHtml  = document.getElementById('sa-reliability-info').innerHTML;
        const catInteractions  = window.lastSAResult ? window.lastSAResult.cat_interactions : {};
        const objNames         = window.lastSAResult ? window.lastSAResult.extended_obj_names : [];
        const parcoordsData    = window.saParcoordsPlotData || {};
        const impactPlotData   = window.saImpactPlotData || {};

        let reportHtml = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EDOS Analysis Report - ${timestamp}</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"><\/script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; color: #1e293b; background: #f8fafc; max-width: 1200px; margin: 40px auto; line-height: 1.6; padding: 0 20px; }
        .card { background: white; border-radius: 12px; border: 1px solid #e2e8f0; padding: 25px; margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .header { border-bottom: 2px solid #3b82f6; padding-bottom: 20px; margin-bottom: 40px; display: flex; justify-content: space-between; align-items: flex-end; }
        h1 { color: #0f172a; margin: 0; font-size: 2.2rem; }
        .meta { color: #64748b; font-size: 0.95rem; text-align: right; }
        h2 { color: #1e293b; margin-top: 0; border-left: 5px solid #3b82f6; padding-left: 15px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th { background: #f1f5f9; text-align: left; padding: 12px; border: 1px solid #e2e8f0; font-weight: 600; }
        td { padding: 12px; border: 1px solid #e2e8f0; }
        .badge { padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
        .good { background: #dcfce7; color: #166534; }
        .limited { background: #fef9c3; color: #854d0e; }
        .poor { background: #fee2e2; color: #991b1b; }
        .impact-group { display: flex; flex-wrap: wrap; gap: 20px; }
        .plot-box { flex: 1; min-width: 450px; height: 450px; }
        .full-plot { width: 100%; height: 500px; }
        .controls { display: flex; gap: 15px; margin-bottom: 20px; background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; }
        .tweak-item { flex: 1; display: flex; flex-direction: column; gap: 5px; }
        select { padding: 8px; border-radius: 6px; border: 1px solid #cbd5e1; background: white; font-family: inherit; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>EDOS Analysis Report</h1>
            <p style="margin: 5px 0 0 0; color: #64748b;">Statistical Analysis Model Results</p>
        </div>
        <div class="meta">
            <b>Model:</b> ${activeModel}<br>
            <b>Date:</b> ${timestamp}
        </div>
    </div>

    <div class="card">
        <h2>Model Reliability &amp; Validation</h2>
        ${reliabilityHtml}
    </div>

    <div class="card" id="num-corr-section" style="display:none;">
        <h2>Numerical Correlation Matrix</h2>
        <div id="sa-num-corr" class="full-plot"></div>
    </div>

    <div class="card" id="cat-section" style="display:none;">
        <h2>Categorical Interaction Analysis</h2>
        <div class="controls">
            <div class="tweak-item">
                <label>Select Pair:</label>
                <select id="report-cat-pair">
                    ${Object.keys(catInteractions).map(p => `<option value="${p}">${p}</option>`).join('')}
                </select>
            </div>
            <div class="tweak-item">
                <label>Select Objective:</label>
                <select id="report-cat-obj">
                    ${objNames.map(o => `<option value="${o}">${o}</option>`).join('')}
                </select>
            </div>
        </div>
        <div id="sa-cat-interactions-container" class="full-plot"></div>
    </div>

    <div class="card" id="parcoords-section" style="display:none;">
        <h2>Parallel Coordinates Analysis</h2>
        <div class="controls">
            <div class="tweak-item">
                <label>Select Objective to Color By:</label>
                <select id="report-parcoords-obj">
                    ${Object.keys(parcoordsData).map(o => `<option value="${o}">${o}</option>`).join('')}
                </select>
            </div>
        </div>
        <div id="sa-report-parcoords" class="full-plot"></div>
    </div>

    <div id="impact-section"></div>

    <script>
        const plotsData       = ${JSON.stringify(plotsData)};
        const catInteractions = ${JSON.stringify(catInteractions)};
        const parcoordsData   = ${JSON.stringify(parcoordsData)};
        const impactPlotData  = ${JSON.stringify(impactPlotData)};

        const corrPlot = plotsData.find(p => p.id === 'sa-num-corr');
        if (corrPlot) {
            document.getElementById('num-corr-section').style.display = 'block';
            Plotly.newPlot('sa-num-corr', corrPlot.data, corrPlot.layout, { responsive: true });
        }

        if (Object.keys(catInteractions).length > 0) {
            document.getElementById('cat-section').style.display = 'block';
            const updateCatPlot = () => {
                const pair = document.getElementById('report-cat-pair').value;
                const obj  = document.getElementById('report-cat-obj').value;
                const data = (catInteractions[pair] || {})[obj];
                if (!data) return;
                const trace = { z: data.z, x: data.x, y: data.y, type: 'heatmap', colorscale: 'RdBu', zmin: 0, zmax: 1 };
                const layout = {
                    title: \`Optimal Outcome: \${obj}\`,
                    xaxis: { title: pair.split(' vs ')[1], automargin: true },
                    yaxis: { title: pair.split(' vs ')[0], automargin: true },
                    margin: { l: 150, b: 80, t: 50, r: 50 }, height: 500
                };
                Plotly.newPlot('sa-cat-interactions-container', [trace], layout, { responsive: true });
            };
            document.getElementById('report-cat-pair').addEventListener('change', updateCatPlot);
            document.getElementById('report-cat-obj').addEventListener('change',  updateCatPlot);
            updateCatPlot();
        }

        const pcKeys = Object.keys(parcoordsData);
        if (pcKeys.length > 0) {
            document.getElementById('parcoords-section').style.display = 'block';
            const pcContainer = document.getElementById('sa-report-parcoords');
            const updatePCPlot = () => {
                const obj = document.getElementById('report-parcoords-obj').value;
                const pObj = parcoordsData[obj];
                if (!pObj) return;
                try { Plotly.purge(pcContainer); } catch(e) {}
                Plotly.newPlot(pcContainer, pObj.data, pObj.layout, { responsive: true });
            };
            document.getElementById('report-parcoords-obj').addEventListener('change', updatePCPlot);
            updatePCPlot();
        }

        const impKeys = Object.keys(impactPlotData);
        if (impKeys.length > 0) {
            const impactSection = document.getElementById('impact-section');
            const head = document.createElement('h2');
            head.textContent = 'Feature Impact Analysis';
            impactSection.appendChild(head);

            impKeys.forEach((objKey, idx) => {
                const pObj = impactPlotData[objKey];
                const group = document.createElement('div');
                group.className = 'card';
                group.innerHTML = '<h3>Feature Impact: ' + pObj.objDisplay + '</h3><div class="impact-group"></div>';
                const container = group.querySelector('.impact-group');

                const barDiv = document.createElement('div');
                barDiv.id = 'report-bar-' + idx;
                barDiv.className = 'plot-box';
                container.appendChild(barDiv);

                const shapDiv = document.createElement('div');
                shapDiv.id = 'report-shap-' + idx;
                shapDiv.className = 'plot-box';
                container.appendChild(shapDiv);

                impactSection.appendChild(group);
                Plotly.newPlot(barDiv.id, pObj.barData, pObj.barLayout, { responsive: true });
                if (pObj.hasShap) {
                    Plotly.newPlot(shapDiv.id, pObj.shapTraces, pObj.shapLayout, { responsive: true });
                } else {
                    shapDiv.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;font-style:italic;">No SHAP data available for this model.</div>';
                }
            });
        }
    <\/script>
</body>
</html>`;

        const blob = new Blob([reportHtml], { type: 'text/html' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = `EDOS_Report_${new Date().toISOString().slice(0,10)}.html`;
        a.click();

    } catch (err) {
        alert("Dynamic report generation failed: " + err.message);
        console.error(err);
    } finally {
        overlay.classList.add('hidden');
    }
}

window.syncSAFeatureConfig = (col, field) => {
    const tr = document.querySelector(`.sa-f-type[data-col="${col}"]`)?.closest('tr');
    if (!tr || !window.saFeatureConfigs[col]) return;

    if (field === 'type'    || !field) window.saFeatureConfigs[col].type     = tr.querySelector('.sa-f-type').value;
    if (field === 'include' || !field) window.saFeatureConfigs[col].included = tr.querySelector('.sa-f-include').checked;
};

// Bind to window for inline calls
window.syncSAFeatureConfig = window.syncSAFeatureConfig;
