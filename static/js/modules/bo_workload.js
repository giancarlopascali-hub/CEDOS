const REVIEW_SCORE = 0.55;
const UNSAFE_SCORE = 1.0;
const TARGET_SCORE = 0.95;
const MAX_REPRESENTATIVE_ROWS = 250;
const SAFE_MEMORY_MB = 6144;

export function objectiveWeightsAreEqual(weights) {
    if (!Array.isArray(weights) || weights.length === 0) return true;
    const parsed = weights.map(Number);
    const total = parsed.reduce((sum, value) => sum + value, 0);
    if (!Number.isFinite(total) || total <= 0) return false;
    const normalized = parsed.map(value => value / total);
    return normalized.every(value => Math.abs(value - normalized[0]) <= 1e-6);
}

export function recommendedMultiObjectiveMethod(objectiveCount, weights) {
    if (objectiveCount <= 1) return 'qLogEI';
    if (!objectiveWeightsAreEqual(weights)) return 'scalarized-qLogEI';
    return objectiveCount <= 3 ? 'qLogNEHVI' : 'qLogNParEGO';
}

function methodFactor(method, objectiveCount) {
    if (method === 'qLogEI' || method === 'qUCB') return 0.35;
    if (method === 'scalarized-qLogEI') return 0.65;
    if (method === 'qLogNParEGO') return 1.0;
    if (method === 'qLogNEHVI') {
        return Math.min(32, 1.35 * Math.pow(1.8, Math.max(0, objectiveCount - 3)));
    }
    return 1.0;
}

function riskForRows({ rowCount, featureCount, objectiveCount, modelCount, batchSize, method }) {
    if (rowCount <= 0) return { time: 0, memory: 0, risk: 0 };
    const featureFactor = Math.max(0.5, Math.pow(Math.max(1, featureCount) / 16, 0.6));
    const fit = featureFactor * modelCount * Math.pow(rowCount / 700, 3);
    const modelFactor = Math.max(0.15, Math.sqrt(Math.max(1, modelCount) / 7));
    const acquisition = 0.65
        * (Math.max(1, batchSize) / 10)
        * Math.pow(Math.max(1, rowCount) / 69, 0.2)
        * Math.sqrt(featureFactor)
        * modelFactor
        * methodFactor(method, objectiveCount);
    const time = fit + acquisition;
    const memoryMb = 96 * modelCount * rowCount * rowCount / (1024 * 1024);
    const memory = memoryMb / SAFE_MEMORY_MB;
    return { time, memory, risk: Math.max(time, memory), memoryMb };
}

export function estimateBOWorkload({
    rowCount,
    featureCount,
    objectiveCount,
    batchSize,
    method,
    additionalModelCount = 0
}) {
    const rows = Math.max(0, Math.trunc(Number(rowCount) || 0));
    const features = Math.max(1, Math.trunc(Number(featureCount) || 1));
    const objectives = Math.max(1, Math.trunc(Number(objectiveCount) || 1));
    const batch = Math.max(1, Math.trunc(Number(batchSize) || 1));
    const modelCount = Math.max(objectives, objectives + Math.max(0, Math.trunc(Number(additionalModelCount) || 0)));
    const components = riskForRows({
        rowCount: rows,
        featureCount: features,
        objectiveCount: objectives,
        modelCount,
        batchSize: batch,
        method
    });
    const tier = components.risk >= UNSAFE_SCORE
        ? 'unsafe'
        : (components.risk >= REVIEW_SCORE ? 'review' : 'safe');

    const minimumRows = Math.max(20, features + 2, objectives * 3);
    let recommendedRows = rows;
    let representativeFeasible = true;
    if (components.risk > TARGET_SCORE) {
        const maximumRows = Math.min(rows, MAX_REPRESENTATIVE_ROWS);
        const minimumRisk = riskForRows({
            rowCount: minimumRows,
            featureCount: features,
            objectiveCount: objectives,
            modelCount,
            batchSize: batch,
            method
        }).risk;
        if (maximumRows < minimumRows || minimumRisk > TARGET_SCORE) {
            recommendedRows = Math.min(maximumRows, minimumRows);
            representativeFeasible = false;
        } else {
            let low = minimumRows;
            let high = maximumRows;
            while (low < high) {
                const midpoint = Math.floor((low + high + 1) / 2);
                const candidateRisk = riskForRows({
                    rowCount: midpoint,
                    featureCount: features,
                    objectiveCount: objectives,
                    modelCount,
                    batchSize: batch,
                    method
                }).risk;
                if (candidateRisk <= TARGET_SCORE) low = midpoint;
                else high = midpoint - 1;
            }
            recommendedRows = low;
        }
    }

    return {
        riskScore: components.risk,
        tier,
        controlsRequired: components.risk >= TARGET_SCORE,
        estimatedSeconds: components.time * 300,
        estimatedMemoryMb: components.memoryMb || 0,
        recommendedRows,
        representativeFeasible
    };
}
