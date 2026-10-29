/** Identified Auto proposal rows and stable, measured-result scoring. */

export function newCampaignId() {
    return globalThis.crypto?.randomUUID?.()
        || `cedos-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function buildProposalRows(columns, featureNames, suggestions, priorDefaults = {}) {
    if (!Array.isArray(suggestions) || !suggestions.length) {
        throw new Error('No experimental conditions were suggested.');
    }
    if (!Array.isArray(columns) || !columns.length || !featureNames.length) {
        throw new Error('The proposal has no dataset schema or selected features.');
    }
    for (const name of featureNames) {
        if (!columns.includes(name)) throw new Error(`Feature ${name} is absent from the dataset.`);
    }
    for (const name of Object.keys(priorDefaults)) {
        if (!columns.includes(name)) throw new Error(`Prior ${name} is absent from the dataset.`);
    }
    const ids = new Set();
    const rows = suggestions.map(suggestion => {
        if (!suggestion || typeof suggestion !== 'object' || Array.isArray(suggestion)) {
            throw new Error('An optimizer suggestion is invalid.');
        }
        const row_id = newCampaignId();
        if (ids.has(row_id)) throw new Error('Unable to assign distinct proposal row IDs.');
        ids.add(row_id);
        const features = {};
        for (const name of featureNames) {
            const featureValue = suggestion[name];
            if (featureValue === undefined || featureValue === null
                || String(featureValue).trim() === '') {
                throw new Error(`A suggestion has no value for ${name}.`);
            }
            features[name] = featureValue;
        }
        const datasetRow = columns.map(name => {
            if (Object.hasOwn(features, name)) return features[name];
            if (Object.hasOwn(priorDefaults, name)) return priorDefaults[name];
            return '';
        });
        return {
            row_id,
            features,
            datasetRow,
            proposalRow: { row_id, ...Object.fromEntries(columns.map((name, i) => [name, datasetRow[i]])) }
        };
    });
    return rows;
}

export function mergeProposalResults({
    campaign_id, proposal_id, rows: measuredRows
}, pending, dataset, columns, objectiveNames, priorDefaults = {}) {
    if (!pending || campaign_id !== pending.campaign_id || proposal_id !== pending.proposal_id) {
        throw new Error('The hardware results belong to a different campaign or proposal.');
    }
    if (!Array.isArray(measuredRows) || measuredRows.length !== pending.rows.length) {
        throw new Error('The hardware result count does not match the dispatched proposal.');
    }
    const byId = new Map(pending.rows.map(row => [row.row_id, row]));
    const seen = new Set();
    const assignments = [];
    for (const measured of measuredRows) {
        const row = byId.get(measured?.row_id);
        if (!row || seen.has(measured.row_id)) {
            throw new Error('The hardware results contain an unknown or duplicate row ID.');
        }
        if (!dataset.includes(row.datasetRow)) {
            throw new Error('The pending dataset changed before its hardware results arrived.');
        }
        seen.add(measured.row_id);
        for (const [name, expected] of Object.entries(priorDefaults)) {
            if (Object.hasOwn(measured, name) && String(measured[name]) !== String(expected)) {
                throw new Error(`The measured Prior Target for ${name} differs from the dispatched proposal.`);
            }
        }
        for (const name of objectiveNames) {
            const number = Number(measured[name]);
            if (measured[name] === '' || measured[name] === null || measured[name] === undefined
                || typeof measured[name] === 'boolean'
                || !Number.isFinite(number)) {
                throw new Error(`A measured value for objective ${name} is missing or not finite.`);
            }
            const index = columns.indexOf(name);
            if (index < 0) throw new Error(`Objective ${name} is absent from the dataset.`);
            assignments.push([row.datasetRow, index, number]);
        }
    }
    if (seen.size !== byId.size) throw new Error('One or more proposal rows have no measured result.');
    for (const [row, index, number] of assignments) row[index] = number;
    return measuredRows.length;
}

export function removePendingProposalRows(dataset, pending) {
    if (!Array.isArray(dataset) || !pending?.rows) return dataset;
    const ownedRows = new Set(pending.rows.map(row => row.datasetRow));
    return dataset.filter(row => !ownedRows.has(row));
}

export function scoreReference(dataset, columns, objectives) {
    const reference = {};
    for (const objective of objectives) {
        const index = columns.indexOf(objective.name);
        const values = index < 0 ? [] : dataset
            .map(row => row[index])
            .filter(value => value !== '' && value !== null && value !== undefined)
            .map(Number).filter(Number.isFinite);
        const min = values.length ? Math.min(...values) : 0;
        const max = values.length ? Math.max(...values) : 0;
        const target = Number(objective.target);
        const spread = max - min;
        const scale = objective.type === 'target'
            ? Math.max(1, ...values.map(value => Math.abs(value - target)))
            : Math.max(1, spread);
        reference[objective.name] = { min, max, scale };
    }
    return reference;
}

function completeObjectiveValues(row, columns, objectives) {
    const values = [];
    for (const objective of objectives) {
        const raw = row[columns.indexOf(objective.name)];
        if (raw === '' || raw === null || raw === undefined) return null;
        const number = Number(raw);
        if (!Number.isFinite(number)) return null;
        values.push(number);
    }
    return values;
}

export function bestMeasuredScore(dataset, columns, objectives, reference) {
    if (!objectives.length) return { score: null, display: '—', label: 'Best Result' };
    const weights = objectives.map(objective => Math.max(0, Number(objective.importance) || 0));
    const weightTotal = weights.reduce((sum, weight) => sum + weight, 0) || objectives.length;
    let bestScore = -Infinity;
    let bestValues = null;
    for (const row of dataset || []) {
        const values = completeObjectiveValues(row, columns, objectives);
        if (!values) continue;
        let score = 0;
        values.forEach((number, index) => {
            const objective = objectives[index];
            const ref = reference[objective.name] || { min: 0, max: 0, scale: 1 };
            const weight = weights.reduce((sum, value) => sum + value, 0)
                ? weights[index] : 1;
            let utility;
            if (objective.type === 'minimize') utility = (ref.max - number) / ref.scale;
            else if (objective.type === 'target') utility = 1 - Math.abs(number - Number(objective.target)) / ref.scale;
            else utility = (number - ref.min) / ref.scale;
            score += utility * weight / weightTotal;
        });
        if (score > bestScore) {
            bestScore = score;
            bestValues = values;
        }
    }
    if (!bestValues) return { score: null, display: '—', label: 'Best Result' };
    if (objectives.length === 1 && objectives[0].type !== 'target') {
        return { score: bestScore, display: bestValues[0].toFixed(3), label: 'Best Result' };
    }
    if (objectives.every(objective => objective.type === 'target')) {
        return {
            score: bestScore,
            display: `${(bestScore * 100).toFixed(1)}%`,
            label: 'Best target match'
        };
    }
    return { score: bestScore, display: bestScore.toFixed(3), label: 'Best weighted score' };
}

/** Called once after each identified, fully merged optimizer batch. */
export function evaluateMeasuredStop(campaign, measuredScore, now = Date.now()) {
    const condition = campaign.stopCondition;
    campaign.completedIterations += 1;
    const improved = measuredScore !== null &&
        (campaign.bestScore === null || measuredScore > campaign.bestScore + 1e-6);
    if (improved) {
        campaign.bestScore = measuredScore;
        campaign.noImprovementCount = 0;
    } else {
        campaign.noImprovementCount += 1;
    }
    const limit = Number(condition.value);
    if (condition.type === 'iterations') return campaign.completedIterations >= limit;
    if (condition.type === 'improvement') return campaign.noImprovementCount >= limit;
    if (condition.type === 'target') return measuredScore !== null && measuredScore * 100 >= limit;
    if (condition.type === 'time') return (now - campaign.optimizerStartTime) / 60000 >= limit;
    return false;
}
