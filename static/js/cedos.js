// CEDOS owns navigation and the shared browser state around the unchanged engines.
import { detectColumnIsNumeric, formatCategoricalChoices, uniqueCategoricalChoices } from './utils.js';

const TERMINAL_STATES = new Set(['idle', 'completed', 'cancelled', 'failed', 'exhausted']);
const MODULES = new Set(['doe', 'bo', 'tbo', 'sa', 'auto']);

function featureConfigFromPrior(column, priorConfig) {
    const index = (window.currentColumns || []).indexOf(column);
    const values = index < 0 ? [] : (window.currentData || [])
        .map(row => row[index]).filter(value => value !== null && value !== undefined && String(value).trim() !== '');
    const declaredCategorical = ['categorical', 'smiles'].includes(priorConfig?.col_type);
    const numeric = !declaredCategorical &&
        (priorConfig?.col_type === 'numerical' || detectColumnIsNumeric(values));
    if (numeric) {
        let minimum = Infinity;
        let maximum = -Infinity;
        for (const value of values) {
            const number = Number(String(value).trim());
            if (!Number.isFinite(number)) continue;
            minimum = Math.min(minimum, number);
            maximum = Math.max(maximum, number);
        }
        const range = Number.isFinite(minimum) ? `[${minimum}, ${maximum}]` : '';
        return { type: 'continuous', range, included: true };
    }
    const choices = uniqueCategoricalChoices(values);
    return { type: 'categorical', range: formatCategoricalChoices(choices), choices, included: true };
}

function sameConfig(left, right) {
    return JSON.stringify(left || null) === JSON.stringify(right || null);
}

const cedos = {
    activeOptimizer: 'bo',
    autoEnabled: false,
    transBOEnabled: false,
    campaignState: 'idle',
    campaignRevision: 0,
    datasetRevision: 0,
    configRevision: 0,
    suggestionSnapshots: { doe: null, bo: null, tbo: null },
    _suspendedPriorRoles: Object.create(null),
    _renderModule: null,
    _trackingPaused: 0,

    get campaignActive() {
        return !TERMINAL_STATES.has(this.campaignState);
    },

    get manualJobActive() {
        const overlay = document.getElementById('loading-overlay');
        return Boolean(overlay && !overlay.classList.contains('hidden'));
    },

    canEditSetup(module = window.currentModule) {
        if (this.manualJobActive) return false;
        if (!this.campaignActive) return true;
        return this.campaignState === 'review_optimizer' && module === this.activeOptimizer;
    },

    canImportSetup(module = window.currentModule) {
        return this.canEditSetup(module);
    },

    canEditDataset() {
        return !this.campaignActive && !this.manualJobActive;
    },

    canEditRoles() {
        return this.canEditDataset()
            || (!this.manualJobActive && this.campaignState === 'review_optimizer'
                && window.currentModule === this.activeOptimizer);
    },

    showNotice(message, error = false) {
        const notice = document.getElementById('cedos-notice');
        if (!notice) return;
        notice.textContent = message;
        notice.classList.toggle('hidden', !message);
        notice.classList.toggle('cedos-notice-error', error);
    },

    captureRevision() {
        return {
            dataset: this.datasetRevision,
            config: this.configRevision,
            campaign: this.campaignRevision,
            optimizer: this.activeOptimizer,
            module: window.currentModule,
            columns: JSON.stringify(window.currentColumns || []),
            roles: JSON.stringify(window.columnRoles || {}),
        };
    },

    isRevisionCurrent(token) {
        return token.dataset === this.datasetRevision
            && token.config === this.configRevision
            && token.campaign === this.campaignRevision
            && token.optimizer === this.activeOptimizer
            && token.module === window.currentModule
            && token.columns === JSON.stringify(window.currentColumns || [])
            && token.roles === JSON.stringify(window.columnRoles || {});
    },

    pauseTracking(callback) {
        this._trackingPaused++;
        try { return callback(); }
        finally { this._trackingPaused--; }
    },

    invalidateResults() {
        this.suggestionSnapshots = { doe: null, bo: null, tbo: null };
        window.suggestionsData = null;
        window.transferDiagnostics = null;
        window.lastSAResult = null;
        window.saSelectedFeatures = [];
        window.saSelectedObjectives = [];
        ['doe-results-section', 'results-section', 'tbo-results-section',
            'sa-results-container', 'bo-visualization-section',
            'tbo-visualization-section', 'pareto-plot-section'].forEach(id => {
            document.getElementById(id)?.classList.add('hidden');
        });
        document.dispatchEvent(new CustomEvent('cedos:invalidated'));
    },

    notifyDatasetMutation(reason = 'Dataset changed') {
        if (reason === 'Dataset uploaded' || reason === 'Dataset cleared') {
            this._suspendedPriorRoles = Object.create(null);
        }
        this.datasetRevision++;
        this.invalidateResults();
        document.dispatchEvent(new CustomEvent('cedos:dataset-changed', { detail: { reason } }));
    },

    notifyConfigMutation(reason = 'Setup changed') {
        if (reason === 'Auto DoE created the dataset schema') {
            this._suspendedPriorRoles = Object.create(null);
        }
        this.configRevision++;
        this.invalidateResults();
        document.dispatchEvent(new CustomEvent('cedos:config-changed', { detail: { reason } }));
    },

    setCampaignState(state) {
        if (typeof state !== 'string' || !state) throw new Error('A campaign state is required.');
        if (this.campaignState === state) return;
        this.campaignState = state;
        this.campaignRevision++;
        this.syncVisibility();
        document.dispatchEvent(new CustomEvent('cedos:campaign-state', { detail: { state } }));
    },

    syncEditability() {
        document.querySelectorAll('.module-view:not(#auto-module) input, .module-view:not(#auto-module) select, .module-view:not(#auto-module) textarea, #upload-section input').forEach(control => {
            const module = control.closest('.module-view');
            const editable = module ? this.canEditSetup(module.id.replace('-module', '')) : this.canEditDataset();
            if (!editable && control.dataset.cedosLocked === undefined) {
                control.dataset.cedosLocked = control.disabled ? 'disabled' : 'enabled';
                control.disabled = true;
            } else if (editable && control.dataset.cedosLocked !== undefined) {
                control.disabled = control.dataset.cedosLocked === 'disabled';
                delete control.dataset.cedosLocked;
            }
        });
    },

    saveSuggestionSnapshot(module = window.currentModule) {
        if ((module === 'doe' || module === 'bo') && Array.isArray(window.suggestionsData)) {
            this.suggestionSnapshots[module] = window.suggestionsData;
        }
    },

    restoreSuggestionSnapshot(module) {
        window.suggestionsData = this.suggestionSnapshots[module] || null;
    },

    noteExplicitRoleChange(column) {
        delete this._suspendedPriorRoles[column];
    },

    _suspendPriorRolesForBO() {
        const converted = [];
        window.columnRoles ||= {};
        window.priorConfigs ||= {};
        window.boFeatureConfigs ||= {};
        window.saFeatureConfigs ||= {};
        for (const column of window.currentColumns || []) {
            if (window.columnRoles[column] !== 'prior') continue;
            const prior = window.priorConfigs[column];
            const boFeature = featureConfigFromPrior(column, prior);
            const saFeature = {
                type: boFeature.type === 'categorical' ? 'categorical' : 'continuous',
                included: true,
            };
            this._suspendedPriorRoles[column] = {
                schema: JSON.stringify(window.currentColumns),
                prior: prior ? structuredClone(prior) : null,
                boFeature: structuredClone(boFeature),
                saFeature: structuredClone(saFeature),
            };
            window.columnRoles[column] = 'feature';
            window.boFeatureConfigs[column] = boFeature;
            window.saFeatureConfigs[column] = saFeature;
            delete window.priorConfigs[column];
            converted.push(column);
        }
        return converted;
    },

    _restorePriorRolesForTransBO() {
        const restored = [];
        const retained = [];
        window.priorConfigs ||= {};
        for (const [column, saved] of Object.entries(this._suspendedPriorRoles)) {
            const unchanged = (window.currentColumns || []).includes(column)
                && saved.schema === JSON.stringify(window.currentColumns)
                && window.columnRoles?.[column] === 'feature'
                && sameConfig(window.boFeatureConfigs?.[column], saved.boFeature)
                && sameConfig(window.saFeatureConfigs?.[column], saved.saFeature);
            if (unchanged) {
                window.columnRoles[column] = 'prior';
                delete window.boFeatureConfigs[column];
                delete window.saFeatureConfigs[column];
                if (saved.prior) window.priorConfigs[column] = structuredClone(saved.prior);
                restored.push(column);
            } else {
                retained.push(column);
            }
        }
        this._suspendedPriorRoles = Object.create(null);
        return { restored, retained };
    },

    switchModuleSafe(module) {
        if (!MODULES.has(module)) return false;
        if (module === 'bo' || module === 'tbo') module = this.activeOptimizer;
        if (module === 'auto' && !this.autoEnabled) return false;
        if (window.currentModule === module && this._renderModule) {
            this._renderModule(module);
            return true;
        }
        this.saveSuggestionSnapshot(window.currentModule);
        window.currentModule = module;
        this.restoreSuggestionSnapshot(module);
        this._renderModule?.(module);
        return true;
    },

    setTransBOEnabled(enabled) {
        if (this.campaignActive || this.manualJobActive) {
            this.showNotice('Finish the current job or Auto campaign before changing the optimizer.', true);
            this.syncVisibility();
            return false;
        }
        if (this.transBOEnabled === Boolean(enabled)) return true;
        const roleChange = enabled
            ? this._restorePriorRolesForTransBO()
            : { converted: this._suspendPriorRolesForBO() };
        this.transBOEnabled = Boolean(enabled);
        this.activeOptimizer = this.transBOEnabled ? 'tbo' : 'bo';
        this.notifyConfigMutation('Active optimizer changed');
        this.syncVisibility();
        if (window.currentModule === 'bo' || window.currentModule === 'tbo') {
            this.switchModuleSafe(this.activeOptimizer);
        } else {
            window.renderMainTable?.();
            window.renderSetup?.();
        }
        this.showNotice('');
        if (roleChange.converted?.length) {
            this.showNotice(`${roleChange.converted.join(', ')} changed from Prior to Feature for BO. Their Prior Targets will return when TransBO is enabled again if those columns are not edited.`);
        } else if (roleChange.restored?.length) {
            this.showNotice(`${roleChange.restored.join(', ')} restored as Prior columns for TransBO.`);
        }
        if (roleChange.retained?.length) {
            this.showNotice(`${roleChange.retained.join(', ')} were not restored as Prior columns because their BO roles or settings changed. Mark them as Prior in TransBO if needed.`);
        }
        return true;
    },

    setAutoEnabled(enabled) {
        if (this.campaignActive || this.manualJobActive) {
            this.showNotice('Finish the current job or Auto campaign before changing Auto visibility.', true);
            this.syncVisibility();
            return false;
        }
        this.autoEnabled = Boolean(enabled);
        this.syncVisibility();
        if (!this.autoEnabled && window.currentModule === 'auto') this.switchModuleSafe('doe');
        return true;
    },

    syncVisibility() {
        const bo = document.querySelector('.module-btn[data-module="bo"]');
        const tbo = document.querySelector('.module-btn[data-module="tbo"]');
        const auto = document.querySelector('.module-btn[data-module="auto"]');
        bo?.classList.toggle('hidden', this.transBOEnabled);
        tbo?.classList.toggle('hidden', !this.transBOEnabled);
        auto?.classList.toggle('hidden', !this.autoEnabled);
        const tboToggle = document.getElementById('cedos-transbo-toggle');
        const autoToggle = document.getElementById('cedos-auto-toggle');
        if (tboToggle) { tboToggle.checked = this.transBOEnabled; tboToggle.disabled = this.campaignActive; }
        if (autoToggle) { autoToggle.checked = this.autoEnabled; autoToggle.disabled = this.campaignActive; }
        document.body.classList.toggle('cedos-campaign-active', this.campaignActive);
        document.body.classList.toggle('cedos-auto-enabled', this.autoEnabled);
        document.body.classList.toggle('cedos-review-optimizer', this.campaignState === 'review_optimizer');
        document.getElementById('cedos-transbo-limit')?.classList.toggle('hidden', !this.transBOEnabled);
        document.getElementById('cedos-optimizer-name')?.replaceChildren(document.createTextNode(this.transBOEnabled ? 'TransBO' : 'BO'));
        this.syncEditability();
        document.dispatchEvent(new CustomEvent('cedos:visibility-changed', {
            detail: { activeOptimizer: this.activeOptimizer, autoEnabled: this.autoEnabled },
        }));
    },
};

export function initCEDOS(renderModule) {
    cedos._renderModule = renderModule;
    document.getElementById('cedos-transbo-toggle')?.addEventListener('change', event => {
        cedos.setTransBOEnabled(event.target.checked);
    });
    document.getElementById('cedos-auto-toggle')?.addEventListener('change', event => {
        cedos.setAutoEnabled(event.target.checked);
    });
    // Edits can originate in the manual modules' inline handlers. Capture them
    // before those handlers run so a dispatched batch remains immutable.
    document.addEventListener('click', event => {
        const target = event.target.closest('button, .type-toggle, label[for="file-input"]');
        if (!target || target.closest('#auto-module, #cedos-extras')) return;
        if (target.classList.contains('module-btn') || target.classList.contains('tab-btn')
            || target.classList.contains('back-to-auto-btn') || target.id === 'export-main-btn'
            || target.id?.endsWith('-setup-download-btn')) return;
        if (cedos.campaignActive && /^(doe-run-btn|run-btn|tbo-run-btn|sa-run-btn|commit-suggestions-btn|doe-commit-suggestions-btn|tbo-commit-suggestions-btn|import-results-btn|doe-import-results-btn|tbo-import-results-btn)$/.test(target.id)) {
            event.preventDefault(); event.stopImmediatePropagation();
            cedos.showNotice('Manual runs and measured-result imports are unavailable during an Auto campaign.', true);
            return;
        }
        const manual = target.closest('.module-view');
        if (manual && !cedos.canEditSetup(manual.id.replace('-module', ''))) {
            event.preventDefault(); event.stopImmediatePropagation();
            cedos.showNotice('This setup is locked while the Auto batch is active.', true);
        } else if (target.closest('#data-section, #upload-section')
            && !(target.classList.contains('type-toggle') && cedos.canEditRoles())
            && !cedos.canEditDataset()) {
            event.preventDefault(); event.stopImmediatePropagation();
            cedos.showNotice('The shared dataset is locked while the Auto campaign is active.', true);
        }
    }, true);
    document.addEventListener('beforeinput', event => {
        const dataset = event.target.closest?.('#data-section');
        const manual = event.target.closest?.('.module-view');
        if ((dataset && !cedos.canEditDataset())
            || (manual && manual.id !== 'auto-module' && !cedos.canEditSetup(manual.id.replace('-module', '')))) {
            event.preventDefault();
            cedos.showNotice('Editing is locked while the Auto batch is active.', true);
        }
    }, true);
    for (const type of ['input', 'change']) {
        document.addEventListener(type, event => {
            if (cedos._trackingPaused) return;
            const manual = event.target.closest?.('.module-view');
            if (manual && manual.id !== 'auto-module'
                && !event.target.closest('.setup-csv-actions, #doe-results-section, #results-section, #tbo-results-section, #sa-results-container, #bo-visualization-section, #tbo-visualization-section')) {
                // Many existing inline handlers repaint the control before bubbling.
                // Capture its origin now; invalidate only after the handler finishes.
                queueMicrotask(() => cedos.notifyConfigMutation('Manual setup changed'));
            }
        }, true);
    }
    document.addEventListener('click', event => {
        if (cedos._trackingPaused) return;
        const button = event.target.closest?.('button');
        if (!button || !button.closest('.module-view:not(#auto-module)')) return;
        if (button.classList.contains('delete-row') || /^.+-add-(feature|objective|num-constraint|disallowed-constraint)/.test(button.id)
            || button.id?.includes('add-objective-constraint')) {
            queueMicrotask(() => cedos.notifyConfigMutation('Manual setup rows changed'));
        }
    });
    // The two result tables identify which engine wrote the shared suggestions slot.
    for (const module of ['doe', 'bo']) {
        const id = module === 'doe' ? 'doe-results-body' : 'results-table-body';
        const body = document.getElementById(id);
        if (body) new MutationObserver(() => cedos.saveSuggestionSnapshot(module))
            .observe(body, { childList: true, subtree: true });
    }
    cedos.syncVisibility();
}

window.cedos = cedos;
export default cedos;
