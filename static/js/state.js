// TransEDOS Global State Management
// ===================================
// Extends EDOS state with transfer learning prior targets and diagnostics.

window.currentModule = 'doe';          // Active module: 'doe' | 'bo' | 'tbo' | 'sa'
window.currentData = null;             // Full dataset: Array of Arrays (row-major)
window.currentColumns = [];            // Column header names: Array of Strings
window.columnRoles = {};               // Map of column name → 'feature' | 'objective' | 'prior'
window.objectiveConfigs = {};          // Objective settings: {name, type, target, importance, included}
window.boFeatureConfigs = {};          // BO feature settings: {type, range, included}
window.saFeatureConfigs = {};          // SA feature settings: {type, included}
window.priorConfigs = {};              // Prior settings: {col_type, target_value, tolerance, min_similarity, prior_weight, is_new_level}
window.priorCombineRule = 'multiply';  // Multi-prior combination: 'multiply' | 'average'
window.transferDiagnostics = null;     // Cached diagnostics from latest /transfer_optimize call
window.suggestionsData = null;         // Latest DoE / BO / TransBO suggestions
window.currentDoEFeatures = [];       // Snapshot of DoE feature configs
window.currentDoEObjectives = [];     // Snapshot of DoE objective configs
window.currentAbortController = null;  // AbortController for in-flight requests
window.lastSAResult = null;           // Cached last SA result
window.saSelectedObjectives = [];     // Cached selected objectives for SA
window.saSelectedFeatures = [];       // Cached selected features for SA
