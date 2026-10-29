import test from 'node:test';
import assert from 'node:assert/strict';
import {
    buildProposalRows, mergeProposalResults, removePendingProposalRows,
    scoreReference, bestMeasuredScore, evaluateMeasuredStop
} from '../../static/js/modules/auto_campaign.js';

test('identified measurements update only proposal-owned rows and retain Prior Targets', () => {
    const columns = ['temperature', 'source', 'yield'];
    const historicalBlank = [20, 'old', ''];
    const dataset = [historicalBlank, [21, 'old', 4]];
    const rows = buildProposalRows(columns, ['temperature'], [{ temperature: 22 }], { source: 'new' });
    const pending = {
        campaign_id: 'campaign-1', proposal_id: 'proposal-1', rows
    };
    dataset.push(rows[0].datasetRow);
    assert.deepEqual(rows[0].datasetRow, [22, 'new', '']);
    assert.throws(() => mergeProposalResults({
        campaign_id: 'campaign-2', proposal_id: 'proposal-1',
        rows: [{ row_id: rows[0].row_id, yield: 7 }]
    }, pending, dataset, columns, ['yield'], { source: 'new' }), /different campaign/);
    assert.throws(() => mergeProposalResults({
        campaign_id: 'campaign-1', proposal_id: 'proposal-1',
        rows: [{ row_id: rows[0].row_id, source: 'old', yield: 7 }]
    }, pending, dataset, columns, ['yield'], { source: 'new' }), /Prior Target/);
    assert.equal(rows[0].datasetRow[2], '');
    mergeProposalResults({
        campaign_id: 'campaign-1', proposal_id: 'proposal-1',
        rows: [{ row_id: rows[0].row_id, yield: 7 }]
    }, pending, dataset, columns, ['yield'], { source: 'new' });
    assert.equal(rows[0].datasetRow[2], 7);
    assert.equal(historicalBlank[2], '');
});

test('stopping a pending proposal removes its rows without touching historical blanks', () => {
    const oldBlank = [20, ''];
    const dataset = [oldBlank];
    const rows = buildProposalRows(['temperature', 'yield'], ['temperature'], [{ temperature: 22 }]);
    dataset.push(rows[0].datasetRow);
    const kept = removePendingProposalRows(dataset, { rows });
    assert.deepEqual(kept, [oldBlank]);
    assert.equal(kept[0], oldBlank);
});

test('stop scoring uses measured outcomes and a fixed reference', () => {
    const columns = ['yield'];
    const objectives = [{ name: 'yield', type: 'minimize', importance: 100 }];
    const historical = [[10], [8]];
    const reference = scoreReference(historical, columns, objectives);
    const baseline = bestMeasuredScore(historical, columns, objectives, reference);
    assert.equal(baseline.display, '8.000');
    const campaign = {
        completedIterations: 0, bestScore: baseline.score,
        noImprovementCount: 0, optimizerStartTime: 1000,
        stopCondition: { type: 'improvement', value: 2 }
    };
    historical.push([7]);
    assert.equal(evaluateMeasuredStop(campaign,
        bestMeasuredScore(historical, columns, objectives, reference).score, 2000), false);
    assert.equal(campaign.noImprovementCount, 0);
    historical.push([9]);
    assert.equal(evaluateMeasuredStop(campaign,
        bestMeasuredScore(historical, columns, objectives, reference).score, 3000), false);
    assert.equal(campaign.noImprovementCount, 1);
    historical.push([10]);
    assert.equal(evaluateMeasuredStop(campaign,
        bestMeasuredScore(historical, columns, objectives, reference).score, 4000), true);
    assert.equal(campaign.completedIterations, 3);
    assert.deepEqual(reference.yield, { min: 8, max: 10, scale: 2 });
});

test('target match uses the campaign-start scale rather than moving min/max', () => {
    const columns = ['a', 'b'];
    const objectives = [
        { name: 'a', type: 'target', target: 10, importance: 50 },
        { name: 'b', type: 'target', target: 20, importance: 50 }
    ];
    const rows = [[0, 0], [8, 18]];
    const reference = scoreReference(rows, columns, objectives);
    const before = bestMeasuredScore(rows, columns, objectives, reference);
    rows.push([10, 20]);
    const after = bestMeasuredScore(rows, columns, objectives, reference);
    assert.equal(after.score, 1);
    assert.equal(after.display, '100.0%');
    assert.deepEqual(reference, {
        a: { min: 0, max: 8, scale: 10 },
        b: { min: 0, max: 18, scale: 20 }
    });
    assert.ok(after.score > before.score);
});
