export function safeAddListener(id, event, callback) {
    const el = document.getElementById(id);
    if (!el) {
        console.warn(`Element with ID "${id}" not found. skipping listener.`);
        return;
    }
    if (!el._safeListeners) el._safeListeners = {};
    if (el._safeListeners[event]) {
        el.removeEventListener(event, el._safeListeners[event]);
    }
    el._safeListeners[event] = callback;
    el.addEventListener(event, callback);
}

export function exportData(data, columns) {
    // Build CSV entirely client-side — no server round-trip needed
    const escape = (v) => {
        const s = (v === null || v === undefined) ? '' : String(v);
        return s.includes(',') || s.includes('"') || s.includes('\n')
            ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const rows = [
        columns.map(escape).join(','),
        ...data.map(row => row.map(escape).join(','))
    ];
    const csv  = rows.join('\r\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = 'export.csv'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}

export function detectParetoIndices(data, objCols) {
    const dataPoints = data.map(r => {
        const point = {};
        objCols.forEach(col => point[col] = parseFloat(r[window.currentColumns.indexOf(col)]) || 0);
        return point;
    });

    return dataPoints.map((p1, i) => {
        const isDominated = dataPoints.some((p2, j) => {
            if (i === j) return false;
            let p2AtLeastAsGoodAsP1   = true;
            let p2StrictlyBetterThanP1 = false;

            for (const col of objCols) {
                const config = window.objectiveConfigs[col] || { type: 'maximize', importance: 100 };
                const v1 = p1[col], v2 = p2[col];
                const target = parseFloat(config.target || 0);

                let v1Score, v2Score;
                if      (config.type === 'maximize') { v1Score = v1;               v2Score = v2; }
                else if (config.type === 'minimize') { v1Score = -v1;              v2Score = -v2; }
                else                                 { v1Score = -Math.abs(v1 - target); v2Score = -Math.abs(v2 - target); }

                if (v2Score < v1Score) p2AtLeastAsGoodAsP1   = false;
                if (v2Score > v1Score) p2StrictlyBetterThanP1 = true;
            }
            return p2AtLeastAsGoodAsP1 && p2StrictlyBetterThanP1;
        });
        return !isDominated;
    }).map((val, i) => val ? i : -1).filter(idx => idx !== -1);
}

export function rebalanceRemaining(changedIdOrEl, newValue, selector) {
    const inputs = Array.from(document.querySelectorAll(selector));
    const activeInputs = inputs.filter(el => {
        const col = el.dataset.col;
        return !window.objectiveConfigs || !window.objectiveConfigs[col] || window.objectiveConfigs[col].included !== false;
    });

    if (activeInputs.length <= 1) {
        if (activeInputs.length === 1) activeInputs[0].value = (100).toFixed(1);
        return;
    }

    const changedIdx = activeInputs.findIndex(el => {
        if (changedIdOrEl instanceof HTMLElement) return el === changedIdOrEl;
        return el.dataset.col === changedIdOrEl;
    });
    if (changedIdx === -1) return;

    const total = 1000;

    if (changedIdx < activeInputs.length - 1) {
        let sumAbove = 0;
        for (let i = 0; i < changedIdx; i++) {
            sumAbove += Math.round(parseFloat(activeInputs[i].value) * 10) || 0;
        }
        let changedVal = Math.round(parseFloat(newValue) * 10) || 0;
        if (sumAbove + changedVal > total) {
            changedVal = Math.max(0, total - sumAbove);
        }
        activeInputs[changedIdx].value = (changedVal / 10).toFixed(1);
        const remainingForBelow = total - sumAbove - changedVal;
        const belowRows = activeInputs.slice(changedIdx + 1);
        const share     = Math.floor(remainingForBelow / belowRows.length);
        let currentTargetSum = 0;

        belowRows.forEach((el, idx) => {
            let val = (idx === belowRows.length - 1) ? remainingForBelow - currentTargetSum : share;
            if (val < 0) val = 0;
            el.value = (val / 10).toFixed(1);
            currentTargetSum += val;
        });
    } else {
        let changedVal = Math.round(parseFloat(newValue) * 10) || 0;
        if (changedVal > total) changedVal = total;
        activeInputs[changedIdx].value = (changedVal / 10).toFixed(1);

        const remainderForAbove = total - changedVal;
        const aboveRows = activeInputs.slice(0, changedIdx);
        const share     = Math.floor(remainderForAbove / aboveRows.length);
        let currentTargetSum = 0;

        aboveRows.forEach((el, idx) => {
            let val = (idx === aboveRows.length - 1) ? remainderForAbove - currentTargetSum : share;
            if (val < 0) val = 0;
            el.value = (val / 10).toFixed(1);
            currentTargetSum += val;
        });
    }
}

export function rebalanceObjectiveConfigs() {
    const objCols      = Object.keys(window.objectiveConfigs);
    const includedCols = objCols.filter(c => window.objectiveConfigs[c].included);

    objCols.forEach(col => {
        if (!window.objectiveConfigs[col].included) window.objectiveConfigs[col].importance = (0).toFixed(1);
    });

    if (includedCols.length === 0) return;

    // Keep automatically balanced values numerically identical. Assigning a
    // one-decimal remainder to the last objective (33.3, 33.3, 33.4) makes an
    // equal three-objective setup look like a deliberate priority weighting.
    const equalShare = (100 / includedCols.length).toFixed(4);
    includedCols.forEach(col => {
        window.objectiveConfigs[col].importance = equalShare;
    });
}

/**
 * Safely parses a fetch Response as JSON.
 * If the server returns a non-JSON body (e.g. an HTML error page from a 413/500/504),
 * this returns { error: <message> } instead of throwing a SyntaxError.
 */
export async function safeJson(response) {
    const text = await response.text();
    try {
        return JSON.parse(text);
    } catch (_) {
        const status = response.status;
        if (status === 413) {
            return { error: 'The dataset is too large for the server to process in a single request. Try reducing the number of rows or columns.' };
        }
        if (status === 504 || status === 502) {
            return { error: `Gateway/Timeout error (HTTP ${status}): The calculation timed out on the server.` };
        }
        if (status === 500) {
            return { error: 'Server error (HTTP 500): The calculation crashed or exceeded available memory. Check server console for details.' };
        }
        return { error: `Server response error (HTTP ${status}): The server returned an unexpected response.` };
    }
}

export function waitForAbortableDelay(milliseconds, signal) {
    return new Promise((resolve, reject) => {
        if (signal?.aborted) {
            reject(new DOMException('Aborted', 'AbortError'));
            return;
        }
        const onAbort = () => {
            clearTimeout(timer);
            reject(new DOMException('Aborted', 'AbortError'));
        };
        const timer = setTimeout(() => {
            signal?.removeEventListener('abort', onAbort);
            resolve();
        }, milliseconds);
        signal?.addEventListener('abort', onAbort, { once: true });
    });
}

export async function submitAndPollJob(
    submitUrl,
    payload,
    {
        signal,
        onProgress,
        onSubmitted,
        pollMilliseconds = 1500,
        submitRetryBaseMilliseconds = 1000,
        submitFailureGraceMilliseconds = 2 * 60 * 1000,
        pollFailureGraceMilliseconds = 10 * 60 * 1000
    } = {}
) {
    const requestId = globalThis.crypto?.randomUUID?.()
        || `analysis-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const submitStartedAt = Date.now();
    let submitFailures = 0;
    let submitResponse;
    let submitted;

    while (true) {
        try {
            submitResponse = await fetch(submitUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Analysis-Request-ID': requestId
                },
                body: JSON.stringify(payload),
                signal
            });
            submitted = await safeJson(submitResponse);
        } catch (error) {
            if (error?.name === 'AbortError') throw error;
            submitFailures += 1;
            if (Date.now() - submitStartedAt >= submitFailureGraceMilliseconds) {
                throw new Error(
                    'Unable to confirm whether the server accepted the analysis job.'
                );
            }
            await waitForAbortableDelay(
                Math.min(10000, submitRetryBaseMilliseconds * (2 ** Math.min(submitFailures, 3))),
                signal
            );
            continue;
        }

        if ([502, 503, 504].includes(submitResponse.status)) {
            submitFailures += 1;
            if (Date.now() - submitStartedAt >= submitFailureGraceMilliseconds) {
                throw new Error(
                    submitted.error || 'Unable to confirm whether the server accepted the analysis job.'
                );
            }
            await waitForAbortableDelay(
                Math.min(10000, submitRetryBaseMilliseconds * (2 ** Math.min(submitFailures, 3))),
                signal
            );
            continue;
        }
        if (!submitResponse.ok || submitted.error || !submitted.job_id) {
            throw new Error(submitted.error || 'Unable to start background job.');
        }
        break;
    }

    const jobId = submitted.job_id;
    onSubmitted?.(jobId);
    onProgress?.({ status: submitted.status || 'queued', phase: 'Queued', percent: 0 }, jobId);
    let connectionLostAt = null;
    let consecutivePollFailures = 0;
    let nextPollMilliseconds = pollMilliseconds;
    let lastPercent = 0;

    const retryAfterInterruption = async message => {
        const now = Date.now();
        if (connectionLostAt === null) connectionLostAt = now;
        consecutivePollFailures += 1;
        if (now - connectionLostAt >= pollFailureGraceMilliseconds) {
            await cancelBackgroundJob(jobId);
            const graceMinutes = Math.max(1, Math.ceil(pollFailureGraceMilliseconds / 60000));
            throw new Error(
                `${message} The server could not be reached for ${graceMinutes} minute(s), so cancellation was requested.`
            );
        }
        nextPollMilliseconds = Math.min(
            15000,
            Math.max(pollMilliseconds, pollMilliseconds * (2 ** Math.min(consecutivePollFailures, 4)))
        );
        onProgress?.({
            status: 'reconnecting',
            phase: `${message} Retrying status connection…`,
            percent: lastPercent
        }, jobId);
    };
    while (true) {
        try {
            await waitForAbortableDelay(nextPollMilliseconds, signal);
        } catch (error) {
            if (error?.name === 'AbortError') await cancelBackgroundJob(jobId);
            throw error;
        }

        let statusResponse;
        try {
            statusResponse = await fetch(
                `/job_status/${encodeURIComponent(jobId)}`,
                { signal }
            );
        } catch (error) {
            if (error?.name === 'AbortError') {
                await cancelBackgroundJob(jobId);
                throw error;
            }
            await retryAfterInterruption('Connection to the analysis server was interrupted.');
            continue;
        }

        const status = await safeJson(statusResponse);
        if (statusResponse.status === 404) {
            const error = new Error(
                'The background job was lost or expired, possibly because the server restarted. Please run it again.'
            );
            error.code = 'job_lost';
            throw error;
        }
        if ([502, 503, 504].includes(statusResponse.status)) {
            await retryAfterInterruption(status.error || `Gateway error (HTTP ${statusResponse.status}).`);
            continue;
        }
        if (!statusResponse.ok) {
            const error = new Error(status.error || 'Unable to read background job status.');
            error.transient = statusResponse.status >= 500 || statusResponse.status === 429;
            if (error.transient) {
                await retryAfterInterruption(error.message);
                continue;
            }
            throw error;
        }

        connectionLostAt = null;
        consecutivePollFailures = 0;
        nextPollMilliseconds = pollMilliseconds;
        if (Number.isFinite(Number(status.percent))) lastPercent = Number(status.percent);
        onProgress?.(status, jobId);
        if (status.status === 'completed') return { jobId, result: status.result };
        if (status.status === 'failed') {
            const error = new Error(status.error || 'Background job failed.');
            error.code = status.error_code || 'job_failed';
            error.terminal = true;
            throw error;
        }
        if (status.status === 'cancelled') {
            throw new DOMException(status.error || 'Cancelled', 'AbortError');
        }
    }
}

export async function waitForJobTerminal(
    jobId,
    { timeoutMilliseconds = 12000, pollMilliseconds = 300, onProgress } = {}
) {
    const deadline = Date.now() + timeoutMilliseconds;
    while (Date.now() < deadline) {
        const response = await fetch(`/job_status/${encodeURIComponent(jobId)}`);
        const status = await safeJson(response);
        if (response.status === 404) return { status: 'lost', error: status.error };
        if (!response.ok) throw new Error(status.error || 'Unable to read background job status.');
        onProgress?.(status);
        if (['completed', 'failed', 'cancelled'].includes(status.status)) return status;
        await waitForAbortableDelay(pollMilliseconds);
    }
    return { status: 'cancelling', error: 'The server is still stopping the calculation.' };
}

export function uniqueCategoricalChoices(values) {
    const seen = new Set();
    const choices = [];
    (values || []).forEach(value => {
        if (value === null || value === undefined || String(value) === '') return;
        const text = String(value);
        if (!seen.has(text)) {
            seen.add(text);
            choices.push(text);
        }
    });
    return choices;
}

export function formatCategoricalChoices(values) {
    return uniqueCategoricalChoices(values).map(value => {
        return /[",\r\n]/.test(value) || /^\s|\s$/.test(value)
            ? `"${value.replace(/"/g, '""')}"`
            : value;
    }).join(', ');
}

export function parseCategoricalChoices(value) {
    if (Array.isArray(value)) return uniqueCategoricalChoices(value);
    const text = String(value ?? '');
    const parsed = [];
    let field = '';
    let quoted = false;
    let fieldWasQuoted = false;

    const finishField = () => {
        const normalized = fieldWasQuoted ? field : field.trim();
        if (normalized !== '') parsed.push(normalized);
        field = '';
        fieldWasQuoted = false;
    };

    for (let index = 0; index < text.length; index += 1) {
        const char = text[index];
        if (quoted) {
            if (char === '"') {
                if (text[index + 1] === '"') {
                    field += '"';
                    index += 1;
                } else {
                    quoted = false;
                }
            } else {
                field += char;
            }
        } else if (char === '"' && field.trim() === '') {
            field = '';
            quoted = true;
            fieldWasQuoted = true;
        } else if (char === ',') {
            finishField();
        } else {
            field += char;
        }
    }
    if (quoted) throw new Error('Categorical values contain an unclosed quote.');
    finishField();
    return uniqueCategoricalChoices(parsed);
}

export function escapeHtmlAttribute(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

export async function waitForBackgroundJobTermination(jobId, { pollMilliseconds = 250 } = {}) {
    if (!jobId) return null;
    while (true) {
        const statusResponse = await fetch(
            `/job_status/${encodeURIComponent(jobId)}`
        );
        const status = await safeJson(statusResponse);
        // A pruned job cannot occupy a worker slot, so it is safe to continue.
        if (statusResponse.status === 404) return { status: 'not_found' };
        if (!statusResponse.ok) {
            throw new Error(status.error || 'Unable to read background job status.');
        }
        if (['completed', 'failed', 'cancelled'].includes(status.status)) return status;
        await waitForAbortableDelay(pollMilliseconds);
    }
}

export function detectColumnIsNumeric(values) {
    const nonEmpty = (values || [])
        .filter(value => value != null && String(value).trim() !== '');
    return nonEmpty.length > 0 && nonEmpty.every(value =>
        Number.isFinite(Number(String(value).trim()))
    );
}

export async function cancelBackgroundJob(
    jobId,
    {
        waitForTerminal = false,
        pollMilliseconds = 250,
        timeoutMilliseconds = 12000,
        throwOnError = false,
        onProgress
    } = {}
) {
    if (!jobId) return;
    try {
        const response = await fetch(`/job_cancel/${encodeURIComponent(jobId)}`, {
            method: 'POST'
        });
        if (!response.ok) {
            const result = await safeJson(response);
            throw new Error(result.error || 'Unable to cancel background job.');
        }
        if (!waitForTerminal) return null;
        return await waitForJobTerminal(jobId, {
            timeoutMilliseconds, pollMilliseconds, onProgress
        });
    } catch (error) {
        console.warn('Unable to send background cancellation request:', error);
        if (throwOnError) throw error;
        return null;
    }
}

/**
 * Updates the placeholder of a feature Range/Values input according to its feature type.
 */
export function updateRangePlaceholder(typeValue, rangeInput) {
    if (!rangeInput) return;
    rangeInput.placeholder = rangePlaceholderForType(typeValue);
}

export function rangePlaceholderForType(typeValue) {
    if (typeValue === 'regular') return '[min, max]steps';
    if (typeValue === 'discrete') return 'xx, yy, zz,...';
    if (typeValue === 'categorical') return 'A, B, C,...';
    return '[min, max]';
}
window.updateRangePlaceholder = updateRangePlaceholder;

/**
 * Attaches select-all / deselect-all behavior with indeterminate support.
 */
export function wireSelectAll(headerId, rowSelector, syncFn) {
    const header = document.getElementById(headerId);
    if (!header) return;

    const refreshHeader = () => {
        const rows    = Array.from(document.querySelectorAll(rowSelector));
        const total   = rows.length;
        const checked = rows.filter(cb => cb.checked).length;
        header.checked       = total > 0 && checked === total;
        header.indeterminate = checked > 0 && checked < total;
    };

    header.onchange = () => {
        const rows = Array.from(document.querySelectorAll(rowSelector));
        const changedColumns = [];
        rows.forEach(cb => {
            if (cb.checked !== header.checked) {
                cb.checked = header.checked;
                changedColumns.push(cb.dataset.col);
            }
        });
        if (changedColumns.length > 0) syncFn(changedColumns, header.checked);
        header.indeterminate = false;
    };

    document.querySelectorAll(rowSelector).forEach(cb => {
        cb.removeEventListener('change', refreshHeader);
        cb.addEventListener('change', refreshHeader);
    });

    refreshHeader();
}
window.wireSelectAll = wireSelectAll;
