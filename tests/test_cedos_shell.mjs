import assert from 'node:assert/strict';

function element() {
    const classes = new Set();
    return {
        classList: {
            toggle(name, force) {
                if (force) classes.add(name);
                else classes.delete(name);
            },
            contains(name) { return classes.has(name); },
        },
        replaceChildren() {},
        dataset: {},
    };
}

const boButton = element();
const tboButton = element();
const autoButton = element();
const tboToggle = element();
const autoToggle = element();
const elements = new Map([
    ['cedos-transbo-toggle', tboToggle], ['cedos-auto-toggle', autoToggle],
    ['cedos-transbo-limit', element()], ['cedos-optimizer-name', element()],
    ['cedos-notice', element()],
]);
globalThis.window = {
    currentModule: 'doe', currentColumns: ['x', 'context', 'yield'],
    currentData: [[1, 'A', 3]],
    columnRoles: { x: 'feature', context: 'prior', yield: 'objective' },
    priorConfigs: { context: { target_value: 'A' } },
    suggestionsData: null,
};
globalThis.CustomEvent = class {
    constructor(type, options) { this.type = type; this.detail = options?.detail; }
};
globalThis.document = {
    body: element(),
    querySelector(selector) {
        return {
            '.module-btn[data-module="bo"]': boButton,
            '.module-btn[data-module="tbo"]': tboButton,
            '.module-btn[data-module="auto"]': autoButton,
        }[selector] || null;
    },
    querySelectorAll() { return []; },
    getElementById(id) { return elements.get(id) || null; },
    createTextNode(text) { return text; },
    dispatchEvent() {},
};

const { default: cedos } = await import('../static/js/cedos.js');
cedos._renderModule = () => {};
assert.equal(cedos.activeOptimizer, 'bo');
assert.equal(tboButton.classList.contains('hidden'), false);
cedos.syncVisibility();
assert.equal(tboButton.classList.contains('hidden'), true);
cedos.switchModuleSafe('sa');
assert.equal(window.columnRoles.context, 'prior');
assert.equal(window.priorConfigs.context.target_value, 'A');

cedos.setTransBOEnabled(true);
assert.equal(cedos.activeOptimizer, 'tbo');
assert.equal(boButton.classList.contains('hidden'), true);
assert.equal(tboButton.classList.contains('hidden'), false);
cedos.switchModuleSafe('bo');
assert.equal(window.currentModule, 'tbo');
assert.equal(window.columnRoles.context, 'prior');

const before = cedos.captureRevision();
cedos.setCampaignState('optimizing');
assert.equal(cedos.canImportSetup('tbo'), false);
assert.equal(cedos.setTransBOEnabled(false), false);
assert.equal(cedos.activeOptimizer, 'tbo');
cedos.setCampaignState('review_optimizer');
assert.equal(cedos.canImportSetup('tbo'), true);
assert.equal(cedos.canImportSetup('sa'), false);
assert.equal(cedos.canEditDataset(), false);
assert.equal(cedos.canEditRoles(), true);
cedos.switchModuleSafe('sa');
assert.equal(cedos.canEditRoles(), false);
cedos.switchModuleSafe('tbo');
cedos.notifyDatasetMutation('merged measurement');
assert.equal(cedos.isRevisionCurrent(before), false);
assert.equal(window.columnRoles.context, 'prior');

cedos.setCampaignState('idle');
elements.set('loading-overlay', { classList: { contains: () => false } });
assert.equal(cedos.setTransBOEnabled(false), false);
assert.equal(cedos.activeOptimizer, 'tbo');
elements.delete('loading-overlay');
cedos.setAutoEnabled(true);
assert.equal(autoButton.classList.contains('hidden'), false);
cedos.switchModuleSafe('auto');
assert.equal(window.currentModule, 'auto');
cedos.setAutoEnabled(false);
assert.equal(window.currentModule, 'doe');

const dataBeforeRoleSwitch = JSON.stringify(window.currentData);
const objectiveBeforeRoleSwitch = JSON.stringify(window.objectiveConfigs || {});
const revisionBeforeRoleSwitch = cedos.configRevision;
cedos.setTransBOEnabled(false);
assert.equal(cedos.activeOptimizer, 'bo');
assert.equal(window.columnRoles.context, 'feature');
assert.deepEqual(window.boFeatureConfigs.context, {
    type: 'categorical', range: 'A', choices: ['A'], included: true,
});
assert.deepEqual(window.saFeatureConfigs.context, { type: 'categorical', included: true });
assert.equal(window.priorConfigs.context, undefined);
assert.equal(JSON.stringify(window.currentData), dataBeforeRoleSwitch);
assert.equal(JSON.stringify(window.objectiveConfigs || {}), objectiveBeforeRoleSwitch);
assert.ok(cedos.configRevision > revisionBeforeRoleSwitch);

cedos.setTransBOEnabled(true);
assert.equal(window.columnRoles.context, 'prior');
assert.equal(window.priorConfigs.context.target_value, 'A');
assert.equal(window.boFeatureConfigs.context, undefined);
assert.equal(window.saFeatureConfigs.context, undefined);

// An explicit BO feature edit must not be overwritten by a later toggle.
cedos.setTransBOEnabled(false);
window.boFeatureConfigs.context.range = 'A, B';
cedos.setTransBOEnabled(true);
assert.equal(window.columnRoles.context, 'feature');
assert.equal(window.boFeatureConfigs.context.range, 'A, B');

// An explicit role change must not be undone even if feature settings match.
window.columnRoles.context = 'prior';
window.priorConfigs.context = { target_value: 'B', col_type: 'categorical' };
cedos.setTransBOEnabled(false);
cedos.noteExplicitRoleChange('context');
window.columnRoles.context = 'objective';
cedos.setTransBOEnabled(true);
assert.equal(window.columnRoles.context, 'objective');

// Importing a replacement dataset removes suspended roles from the old schema.
window.columnRoles.context = 'prior';
window.priorConfigs.context = { target_value: 'A', col_type: 'categorical' };
cedos.setTransBOEnabled(false);
window.currentData = [[2, 'B', 4]];
window.columnRoles.context = 'feature';
window.boFeatureConfigs.context.range = 'B';
cedos.notifyDatasetMutation('Dataset uploaded');
cedos.setTransBOEnabled(true);
assert.equal(window.columnRoles.context, 'feature');

console.log('CEDOS shell state flow passed');
