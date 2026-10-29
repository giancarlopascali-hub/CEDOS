# CEDOS Implementation Plan — v2

**CEDOS** = Complete Experimental Design, Optimization and Statistics.  
**Plan date:** 27 October 2026.  
**Status:** Updated after hands-on code inspection. Supersedes `CEDOS_IMPLEMENTATION_PLAN.md`.

---

## What changed from v1

v1 was written from static analysis of file paths and line references. v2 is grounded in reading the actual source. Key updates:

- **Dependencies are already identical** in EDOS and TransEDOS (`requirements.txt` is the same base). TransEDOS adds only `rdkit==2026.3.6` and `python-dotenv>=1.0.0`. No Dockerfile conflict to resolve for the Python layer; the Docker step is about labelling only.
- **The EDOS canonical commit is confirmed** as `68a7d586e77dc893176bfface7ba38653d0a1f15` (from `alignment/foundation.json`). The shared-file list is already machine-readable in `alignment/shared_files.json` (19 files, covering all `src/core/`, `src/api/jobs.py`, core JS modules and tests). Use this JSON as the ground truth for Phase A alignment verification.
- **EDOS `routes.py` still attempts torch-directml** (lines 37-42), which TransEDOS intentionally removed after finding that DirectML is incompatible with BoTorch quasirandom sampling. CEDOS must carry the TransEDOS routes, not the EDOS routes.
- **AutoEDOS `auto.js` `buildBOPayload`** places `objective_constraints` at the top-level payload correctly (confirmed). However, `tweaks.constraints` is not populated from the UI constraint table - it omits `numerical` and `disallowed` constraints that manual BO does send. This is still a live defect to fix in Phase C.
- **Auto's "global success" stop-check uses model R2**: the bug is specifically in `continueAutoLoop()`, where `window._lastAutoGlobalScore` is set to `result.performance?.global_success?.r2 ?? 0` from the BO API response. That *is* the model R2, and *this* is then fed into `shouldStopAutoLoop()`. The per-row score computed by `getBestScoreAndLabel()` uses actual observed values, but it does not gate the loop. Only R2 does.
- **IMEX transport stubs confirmed**: `check_serial_input()`, `check_network_input()`, `check_usb_input()` are `pass` stubs. Only `check_local_exports()` (folder mode) is functional. This must be documented and folder-only mode clearly presented as the only working closed-loop path at release.
- **`switchModule()` prior-to-feature conversion** is confirmed present in `main.js`. It iterates `window.currentColumns` and resets every `prior` role to `feature` with a fresh `boFeatureConfigs` entry, unconditionally for any destination that is not `tbo`. This is the exact mutation that must be removed.
- **TransEDOS `app.py`** adds `app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024`. EDOS `app.py` does not. CEDOS must carry this setting.
- **Port differences**: EDOS defaults to `5000` (frozen) / `7860` (dev with env). TransEDOS defaults to `5001` (dev) / `7860` (HF). AutoEDOS hardcodes `5000` as the IMEX-compatible port. CEDOS must use `5000` for the standalone/IMEX case. The `_available_local_port()` helper from EDOS `app.py` is worth keeping.
- **Auto module selector is BO-only**: `auto-select-bo` always calls `window.switchModule('bo')`. There is no `tbo` path in the wizard today. The entire Phase C adapter work starts from a zero-baseline for TransBO.
- **`tbo.js` is 2,445 lines** with full commit, export, visualization and polling logic. It must be preserved verbatim in CEDOS; only the `switchModule` interaction and the Auto adapter layer around it changes.

---

## 1. Goal and non-negotiable boundaries

*(Unchanged from v1.)*

Build one application with EDOS's DoE, BO and Statistical Analysis (SA), plus optional TransBO and Auto modules. The user can show or hide TransBO and Auto. When TransBO is shown, it replaces BO in the navigation and in Auto's optimization cycles. EDOS computational behavior and TransBO's transfer-learning method remain intact. Integration work belongs in the application shell, shared browser state, Auto orchestration, API composition, and focused corrections to prevent silent failures.

The empty `EDOS complete app` folder is the destination. Source trees remain untouched.

| Source | Role in CEDOS | Confirmed commit / revision |
|---|---|---|
| `../Optimized EDOS` | Canonical EDOS core - do not modify | `68a7d586e77dc893176bfface7ba38653d0a1f15` |
| `../TransEDOS` | Application shell base | working tree (aligned to EDOS foundation) |
| `../autoEDOS_combined/AutoEDOS opt` | Auto UI, loop, bridge routes | working tree |
| `../autoEDOS_combined/autoEDOS-IMEX` | IMEX companion client | working tree |

---

## 2. Confirmed findings from code inspection

### 2.1 Foundation and routing

- **Route divergence**: EDOS `routes.py` retains DirectML detection (lines 37-42 of the EDOS file). TransEDOS `routes.py` removed it after confirming DirectML incompatibility with BoTorch. **CEDOS must use the TransEDOS routes file** as the starting point for the standard BO / SA / DoE / upload routes. Do not carry the DirectML block forward.
- **Blueprint registration**: EDOS registers `api_bp` only. AutoEDOS registers `api_bp` + `auto_bp`. TransEDOS registers `api_bp` only. CEDOS `app.py` must register `api_bp` (from TransEDOS) + `auto_bp` (from AutoEDOS). Do not concatenate the routes files; add only the blueprint `import` and `register_blueprint` call.
- **MAX_CONTENT_LENGTH**: TransEDOS `app.py` sets 200 MB; EDOS does not. CEDOS must carry the TransEDOS setting.
- **Port/IMEX URL**: standardize on `5000` for standalone. The IMEX URL is currently a GUI field in the IMEX client; it must remain configurable and the CEDOS health/version endpoint must echo the actual origin URL.
- **Job registry**: `src/api/jobs.py` is already process-local, single-slot by default, configurable via `MAX_CONCURRENT_ANALYSIS_JOBS`. The registry is identical between EDOS and TransEDOS (confirmed by `alignment/shared_files.json`). CEDOS keeps this file verbatim.

### 2.2 Navigation and shared data

- **`switchModule()` prior-erasure bug** *(confirmed)*: any call to `switchModule` with a destination other than `tbo` resets every `prior`-role column to `feature` in `boFeatureConfigs` and `saFeatureConfigs`. This is the dominant state-corruption risk for SA and Auto.
- **Upload resets `boFeatureConfigs` and `saFeatureConfigs`** but leaves `priorConfigs` and `priorCombineRule` in place (confirmed in `handleFiles`). A subsequent upload with matching column names will inherit stale prior config.
- **`suggestionsData` is shared**: DoE (`doe.js:259`) and BO (`bo.js:262`) write to the same `window.suggestionsData`. TransBO uses module-local `latestTBOResults` inside `tbo.js`. Commit buttons in each module read from different sources. Auto reads from `window.suggestionsData` for BO and must not silently fall through to a stale DoE result.
- **SA uses `saSelectedFeatures` / `saSelectedObjectives`**: SA reads these as cached globals, not live from `columnRoles`. If column roles change after a run, the cached result remains until next SA run - must be invalidated by the dataset revision counter.

### 2.3 Auto-specific confirmed findings

- **Auto's `buildBOPayload` omits numerical and disallowed constraints** from `tweaks.constraints`. The backend `_validated_tweaks()` reads them from inside `tweaks.constraints.numerical` and `tweaks.constraints.disallowed`. Manual BO JS does pass these. This is a **live silent defect** - Auto runs BO without user-configured constraints.
- **Auto stop-check uses model R2**: `window._lastAutoGlobalScore = result.performance?.global_success?.r2 ?? 0` (line ~563 of `auto.js`). The `shouldStopAutoLoop()` `target` branch compares `(globalSuccess * 100) >= parseFloat(value)`, so the target-percentage stop is gated on R2, not on observed experimental results. The iterations and time-limit stops are unaffected.
- **`autoBestScore` mutation during render**: `getBestScoreAndLabel()` writes `window.autoBestScore` as a side-effect. `updateAutoIterationDisplay()` calls `getBestScoreAndLabel()` every second via `setTimeout`. The no-improvement stop also reads and writes `window.autoBestScore` in `shouldStopAutoLoop()`. The display tick can update the best score between the loop's comparison and its update - independent bookkeeping is required.
- **Row-appending and matching**: confirmed that blank-objective rows are appended in `pushProposalToServer()` then matched positionally by iteration. `stopAutoLoop()` removes all `currentData` rows with blank objective values, not only the campaign-owned rows.
- **Auto wizard is BO-only**: the "select BO" path in `initAutoWizard()` calls `window.switchModule('bo')`. There is no `tbo` path anywhere in `auto.js`. The adapter work in Phase C is a net-new addition.
- **DoE-start review gate**: confirmed present at lines 449-478. After DoE results, Auto sets `window.autoStartedFrom = 'doe'` and the flow pauses for the user to switch to BO and fill in setup before returning. This review gate must be preserved and extended to route to the active optimizer (BO or TransBO).

### 2.4 TransBO confirmed risks

- **`avoid_reval` unimplemented branch** *(confirmed in plan)*: flag it as a known gap; disable the `avoid_reval` checkbox for TransBO until fixed.
- **Constraint enforcement in fitted-GP path** *(confirmed in plan)*: numerical/disallowed constraints are applied in the cold-start path; the fitted-GP path needs an enforcement step.
- **`exhausted` response not handled** in `tbo.js` polling: the manual TBO result handler (near line 1368-1386) does not branch on `result.exhausted`. The backend route *does* return `exhausted` (TransEDOS `routes.py:558-561`). Add a visible exhaustion message and disable the Run button on receipt.
- **TransBO fallback disclosure**: when no configured prior means `algorithm_used: standard_bo`, the UI must label this clearly. The returned object includes `algorithm_used`; surface this in the UI.

### 2.5 IMEX transport reality

- **Folder mode only is functional**: `check_local_exports()` reads CSV files from a watched folder. Serial, network and USB receive paths are `pass` stubs.
- **Proposal consumption**: GET `/auto/proposal` sets `AUTO_STATE['proposal'] = None` before returning. If IMEX client crashes after the GET but before hardware dispatch, the proposal is lost with no recovery path.
- **No campaign identity**: `AUTO_STATE` is a flat global dict with one proposal slot and one results slot. Iteration counter increments on each POST. There is no `campaign_id`, no `proposal_id`, no row IDs. Merge is purely positional.

---

## 3. Product rules *(unchanged from v1)*

1. **Initial view**: DoE, BO, SA visible; TransBO and Auto hidden. Add module toggles.
2. **Optimizer selection**: `activeOptimizer` is `bo` or `tbo`. TransBO on hides BO everywhere including Auto wording.
3. **Safe toggling**: freeze optimizer/Auto changes during active campaigns.
4. **Canonical dataset**: one dataset and schema for all modules. Column roles do not change on tab switch.
5. **Auto entry**: DoE or active optimizer. Freeze optimizer for campaign duration.
6. **No-prior TransBO**: require visible acknowledgement in Auto before dispatch.
7. **Stop semantics**: count measured batches; gate on observed objectives, not R2. Distinct terminal states.

---

## 4. Implementation sequence and file ownership

### Phase A - Compose a clean, traceable CEDOS foundation

**Files to copy verbatim from TransEDOS (canonical app shell):**
- `app.py` - keep `MAX_CONTENT_LENGTH`, add `_available_local_port` helper (port 5000 default), add `auto_bp` import/register; rename application labels to CEDOS.
- `src/api/routes.py` (TransEDOS version, not EDOS - retains transfer routes, excludes DirectML block)
- `src/core/transfer_bo.py`, `src/core/prior_utils.py` (TransBO-only files)
- `static/js/modules/tbo.js`, `static/js/state.js` (TransBO and extended state)
- `templates/index.html` (TransEDOS version - has TBO section)
- `static/style.css`, `static/banner.png`, `static/favicon.ico`

**Files to copy verbatim from canonical EDOS (locked at `68a7d586`):**
- `src/core/bo.py`, `src/core/doe.py`, `src/core/sa.py`, `src/core/constraints.py`, `src/core/feasibility.py`
- `src/api/jobs.py`
- `static/js/modules/bo.js`, `static/js/modules/doe.js`, `static/js/modules/sa.js`, `static/js/modules/constraints.js`
- `static/js/utils.js`
- All 19 files listed in `alignment/shared_files.json`

**Files to copy from AutoEDOS and add (do not replace TransEDOS files):**
- `src/api/auto_routes.py` - add blueprint registration in CEDOS `app.py` only
- `static/js/modules/auto.js` - merge Auto tab HTML into TransEDOS `templates/index.html`; do not replace the full template
- `templates/auto_guide.html` - add route for the guide page

**Files to merge carefully (TransEDOS base + Auto additions):**
- `static/js/main.js` - TransEDOS base; add Auto module init call; do not use AutoEDOS's `main.js` wholesale
- `static/js/state.js` - TransEDOS base; add Auto state variables (`autoModeActive`, `autoStopCondition`, etc.)

**Phase A dependency set:**

Base (from EDOS `requirements.txt`): Flask==3.1.3, pandas==2.2.3, numpy==2.2.6, torch==2.4.1, botorch==0.17.2, gpytorch==1.15.2, pydoe==1.5.0, scikit-learn==1.6.1, matplotlib==3.9.4, scipy==1.15.3, werkzeug==3.1.6, gunicorn==23.0.0, shap>=0.40.0

Add (TransEDOS extras): rdkit==2026.3.6, python-dotenv>=1.0.0

No other dependency changes needed.

**Phase A metadata:**
- Create `alignment/foundation.json` in CEDOS pointing at `68a7d586e77dc893176bfface7ba38653d0a1f15` plus TransEDOS and AutoEDOS working-tree descriptions.
- Create `alignment/shared_files.json` mirroring the EDOS canonical list for the 19 core files.
- Create `build-info.json` with CEDOS version and source commit references.
- Write `scripts/check_alignment.py` that reads `shared_files.json` and verifies LF-normalized SHA-256 of the 19 canonical files against EDOS source.

**Phase A gate**: run EDOS test suite (35 tests) + TransEDOS audited selection (81 tests) + AutoEDOS Python/Node tests (15+4). All must pass before Phase B begins.

---

### Phase B - CEDOS shell controller

**New file: `static/js/cedos.js`**

This module is the single authority for:

- `enabledModules` : Set of `'doe'|'bo'|'tbo'|'sa'|'auto'`
- `activeOptimizer` : `'bo'` or `'tbo'`
- `currentModule` : string
- `datasetRevision` : integer (increment on any dataset/role/config mutation)
- `activeCampaignId` : string or null
- `activeJobId` : string or null
- `moduleResults` : `{ doe: null, bo: null, tbo: null, sa: null }`

**Key behaviors to implement in `cedos.js`:**

1. **`switchModuleSafe(target)`** - replaces the raw `switchModule()`. Before switching, checks `activeCampaignId` and `activeJobId`; if active, shows an explanatory message instead of switching. Never mutates column roles.

2. **`setActiveOptimizer(opt)`** - toggles `enabledModules` to include exactly one of `bo`/`tbo`. Fires only if no active campaign. Updates navigation labels throughout the DOM (all elements with `data-optimizer-label` attribute). If Auto was configured for the other optimizer, resets Auto's optimizer selection state.

3. **`toggleModule(mod, enabled)`** - shows or hides TransBO/Auto tabs and navigation items. While `activeCampaignId` is set, freezes this call with an explanation.

4. **`incrementRevision(reason)`** - call on upload, role change, cell edit, import, objective/feature change. Clears `moduleResults[module]` for the affected module(s). Cancels the active abort controller if present. Does not change `activeCampaignId` (that requires explicit Stop).

5. **`validateModuleResult(result, expectedRevision, expectedModule, expectedCampaignId)`** - gates all async result handlers. Returns `false` with a logged warning if any field has drifted; the result is then discarded silently (not applied).

**Remove from `main.js`**: the prior-to-feature conversion inside `switchModule()` (lines 199-227 of the TransEDOS `main.js`). Replace calls to `switchModule` everywhere with `cedos.switchModuleSafe()`.

**`templates/index.html` changes:**
- Add nav items for Auto and TransBO, initially hidden via `display:none`.
- Add a "Modules" preferences toggle (gear icon or dedicated button) with two checkboxes: "Enable TransBO" and "Enable Auto". Wire to `cedos.toggleModule()`.
- Add `data-optimizer-label` attributes to all UI elements that switch wording between "BO" and "TransBO".
- Merge Auto wizard HTML from AutoEDOS `templates/index.html` into the CEDOS template, using prefixed IDs (`auto-*`) and a new `id="auto-module"` section. Audit for duplicate IDs against existing CEDOS template IDs before merge.
- Add a "Return to Auto" button in the TransBO view (`id="tbo-return-to-auto-btn"`) - visible only when `activeCampaignId` is set and `activeOptimizer === 'tbo'`.

**Per-module result snapshots:**
- `cedos.moduleResults.doe` = snapshot of last DoE suggestion set (written by `doe.js`, read only by DoE commit).
- `cedos.moduleResults.bo` = snapshot of last BO suggestion set (written by `bo.js`).
- `cedos.moduleResults.tbo` = snapshot of last TransBO result (from `latestTBOResults` in `tbo.js`; move storage here).
- `cedos.moduleResults.sa` = snapshot of last SA report.
- The DoE, BO and TransBO commit/export buttons read from their respective snapshot, not from `window.suggestionsData` or `window.latestTBOResults`.
- Clear `moduleResults[mod]` when `incrementRevision()` is called for that module.

**DoE commit routing:** After DoE commit, check `cedos.activeOptimizer`. If `bo`, call `cedos.switchModuleSafe('bo')`. If `tbo`, call `cedos.switchModuleSafe('tbo')`. If Auto campaign is active, call the Auto review-gate instead.

**Constraints UI prefix audit:** Audit all constraint form elements for `doe-*`, `bo-*`, `tbo-*`, `sa-*` prefixes. Where a single constraints panel is reused, duplicate it or add a `data-module` attribute to route the payload correctly.

**Phase B gate:** all Phase A tests pass. Add DOM tests for: module toggle states (all four combinations), role preservation across `switchModuleSafe`, and per-module result isolation.

---

### Phase C - Optimizer adapters for Auto

Create `static/js/cedos_adapters.js` with two exported adapter objects.

**Common adapter interface:**

```
validateSetup(dataset)                              -> { ok, errors }
buildRequest(dataset)                               -> payload | null
async generateSuggestions({ signal, onProgress })  -> AdapterResult
normalizeResult(response)                           -> AdapterResult
cancel()
```

AdapterResult contains: `suggestions`, `featureNames`, `objectiveNames`, `constraintWarning`, `feasibilityWarning`, `exhausted`, `algorithmUsed`, `diagnostics`.

**BO adapter (`cedosBoAdapter`):**
- `buildRequest()` uses the same serialization as manual `bo.js`: reads `boFeatureConfigs`, `objectiveConfigs`, `window.collectObjectiveConstraints('bo')`, and - critically - `tweaks.constraints.numerical` and `tweaks.constraints.disallowed` from the BO constraints UI. This fixes the existing Auto defect where numerical/disallowed constraints were dropped.
- POSTs to `/optimize`. Validates non-empty `suggestions` array and schema-correct feature names before returning.

**TransBO adapter (`cedosTboAdapter`):**
- `buildRequest()` uses the same serialization as manual `tbo.js`: reads `priorConfigs`, `priorCombineRule`, `boFeatureConfigs` (for row-cap), `objectiveConfigs`, numerical constraints, selected features.
- Before building the request, checks that at least one `prior` column is configured. If none: surfaces a modal asking the user to acknowledge the no-prior fallback (`algorithm_used: standard_bo`) before dispatching. Without acknowledgement, returns `null` from `buildRequest()`.
- POSTs to `/transfer_optimize_submit`, polls `/job_status/<id>` until terminal, cancels via `/job_cancel/<id>` on Stop.
- Maps poll states: `queued`, `running`, `cancelling` to `waiting`; `completed` to parse result; `failed` to error; `cancelled` to cancelled.
- Inspects `result.transfer.algorithm_used` and sets `algorithmUsed` on the returned `AdapterResult`. If `standard_bo`, logs a labeled disclosure in the Auto activity log.
- Surfaces `exhausted` response explicitly.
- Does not change the transfer model or weighting math inside `transfer_bo.py`.

**Auto loop rewiring in `auto.js`:**
- Replace the hardcoded `fetch('/optimize', ...)` in `continueAutoLoop()` with a dispatch through the active adapter.
- Replace `window._lastAutoGlobalScore = result.performance?.global_success?.r2 ?? 0` with a campaign-local score variable derived from the observed objectives in the result, not from the model R2. For `target` stop: compare observed best objective value (using the same direction-aware logic as `getBestScoreAndLabel()`) against the configured threshold. Keep R2 available as a diagnostic log item only.
- Move `window.autoBestScore` to a campaign-local `let` variable inside the campaign object so the display ticker cannot mutate it. `updateAutoIterationDisplay()` reads a separate display copy.
- Give every Auto generation and polling cycle a `campaignToken` (AbortController wrapping `cedos.activeCampaignId`). After Stop, invalidate the token before returning. Delayed fetch responses that arrive post-Stop must be discarded by `validateModuleResult()`.
- Auto wizard "select optimizer" step: the wizard label shows the active optimizer name. The "Check optimizer setup" button in Step 3 switches to whichever optimizer is active, not always BO.
- DoE review gate: after DoE dispatch/results, route to `cedos.activeOptimizer` tab (not always BO). TransBO prior prerequisite/fallback check fires at this gate if no prior columns exist in the DoE-created dataset.

**Phase C gate:** Add end-to-end simulated-bridge tests for Auto to BO and Auto to TransBO full cycles. Tests must cover: correct constraint serialization, stop on iterations, stop on target (using observed objectives), exhaustion handling, and Stop-mid-cycle (no stale result applied).

---

### Phase D - Make Auto/IMEX exchange transactional

**`src/api/auto_routes.py` refactoring:**

Replace `AUTO_STATE` flat dict with a `Campaign` structure containing: `campaign_id` (UUID), `optimizer` (`bo` or `tbo`), `proposal_id` (UUID per batch), `dataset_rev`, `feature_names`, `proposal`, `results`, `status` (`idle|running|waiting|completed|cancelled|failed`), `iteration`.

**Dispatch protocol changes:**
- POST `/auto/proposal` requires `campaign_id` and `proposal_id` in the payload. Server validates they match the active campaign. Returns HTTP 409 if a different campaign is active.
- GET `/auto/proposal` returns the proposal with `campaign_id`, `proposal_id`, and `dataset_rev`. Does **not** consume/null the proposal on GET. Instead, sets `status = 'dispatched'`. Only cleared by the next POST to `/auto/proposal` or a DELETE `/auto/proposal/<proposal_id>` from IMEX on acknowledgement.
- POST `/auto/results` requires `campaign_id` and `proposal_id`. Server validates both match. Returns HTTP 409 for mismatches; HTTP 200 + `status: merged` on success.
- Stop campaign endpoint: POST `/auto/stop` sets status to `cancelled`, clears proposal and results.

**Row identity:**
- Each suggestion row in the proposal carries a `row_id` (UUID). IMEX includes `row_id` in result rows. Merge uses `row_id` matching, not positional.
- Pending rows added to `currentData` carry `row_id` in a synthetic column (or in a side map). Stop removes only rows whose `row_id` is in the active campaign's pending set. Unrelated blank-objective rows are not touched.

**IMEX transport documentation:**
- Add a `TRANSPORT_STATUS.md` and comment block to `auto_routes.py` and the guide template explicitly listing:
  - **Folder mode**: functional, tested.
  - **Serial / Network / USB**: stub only, not implemented, not tested. Do not present as available options in CEDOS release UI. Hide or disable those radio buttons in IMEX until implemented.

**Recovery:**
- Add GET `/auto/campaign` returning the full campaign record. The Auto JS polls `/auto/status` after reconnect to recover the displayed state.
- If CEDOS restarts mid-campaign, the pending rows remain in the browser dataset (unsaved). The server campaign is lost. The UI must detect the mismatch on the next poll (`campaign_id` no longer matches) and offer to cancel or resume manually.

**Phase D gate:** Simulated-bridge tests for: proposal acknowledgement and idempotent retry, wrong-campaign rejection, row-ID-based merge, Stop without touching unrelated rows, mismatched `proposal_id` rejection.

---

### Phase E - Score, status and TransBO bug corrections

**Stop-check corrections:**
- `shouldStopAutoLoop()` is rewritten as a pure function (no side effects on `autoBestScore`).
- Stop types and their observed-outcome sources:
  - `iterations`: count completed `push_results` cycles. Not affected by score issues.
  - `target`: compare against the best observed objective value (direction-aware, from `getBestScoreAndLabel().rawVal`), not R2. For multi-objective: use the weighted % Global Success value.
  - `improvement`: compare the new best against the previous-cycle best. Update the previous-cycle best only after a successful merge, not during render.
  - `time`: unchanged - time from campaign start.
- Validate stop value inputs: positive integer for `iterations`/`improvement`, finite positive number for `target`/`time`.

**State machine for Auto (explicit transitions):**

```
idle
  -> generating_doe       (Start from DoE)
  -> generating_optimizer (Start from optimizer directly)
generating_doe
  -> awaiting_dispatch    (DoE results ready, proposal sent)
  -> failed               (DoE error)
awaiting_dispatch
  -> awaiting_results     (IMEX acknowledges proposal)
  -> failed               (timeout / dispatch error)
awaiting_results
  -> review_optimizer     (DoE path: first results received, user must review optimizer setup)
  -> evaluating_stop      (Optimizer path: results received)
  -> failed               (timeout / result error)
review_optimizer
  -> generating_optimizer (User approves and returns to Auto)
evaluating_stop
  -> generating_optimizer (Stop not met; continue)
  -> completed            (Stop condition met)
  -> exhausted            (Optimizer returned exhausted)
  -> cancelled            (User clicked Stop)
generating_optimizer
  -> awaiting_dispatch    (Suggestions generated, proposal sent)
  -> failed               (Optimizer error)
completed / cancelled / failed / exhausted  -> idle (on new campaign start)
```

Terminal states are idempotent: calling Stop on an already-completed campaign is a no-op with a visible notice.

**TransBO bug patches:**

1. **`exhausted` not rendered in manual TBO**: Add a branch in the TBO result handler (near `tbo.js` line 1368) to detect `result.exhausted === true`, show a styled message in the results panel, and disable the Run button until the dataset changes. Regression test: confirm the backend can return `exhausted` and the UI shows it.

2. **`avoid_reval` disabled for TransBO**: Disable the TransBO `avoid_reval` checkbox with a tooltip: "Avoid re-evaluation is not yet implemented for Transfer BO." Gate re-enabling on a verified fix to `transfer_bo.py:867-870`.

3. **Numerical constraints in fitted-GP path**: Before enabling numerical/disallowed constraints for TransBO in the UI, write a targeted test confirming the fitted-GP path enforces them. If the test fails, disable those constraint inputs for TransBO with a visible explanation. Do not silently accept an unsupported constraint.

**Phase E gate:**
- Regression test for `exhausted` in both manual TBO and Auto to TransBO.
- Regression test for `avoid_reval` in BO (confirming it still works) and confirmation it is disabled for TBO.
- Score/stop unit tests using a deterministic simulated dataset: verify that a `target` stop fires on the correct observed objective iteration, not on R2.

---

## 5. Workflow verification matrix

*(Updated to reflect the IMEX folder-only reality.)*

| Flow | Expected result / failure check |
|---|---|
| Extra modules off: DoE to BO to SA; upload to BO to SA | EDOS behavior unchanged; DoE commit routes to BO; objectives and dataset consistent. No DirectML-related errors. |
| TransBO on: DoE to TransBO; upload to TransBO to SA | BO hidden; prior roles survive viewing Auto or SA (no prior-to-feature erasure); SA excludes prior columns; DoE commit routes to TransBO. |
| Manual TransBO: optimize, estimate, visualization, commit/export | `algorithm_used` disclosed; no-prior fallback labeled; `exhausted` renders; `avoid_reval` disabled with tooltip; prior cells in result table populated with target values. |
| Auto direct BO | Correct constraints serialized (including numerical/disallowed); measured-objective stop check (not R2); one proposal per batch; `row_id` merge. |
| Auto direct TransBO | Prior prerequisite acknowledged before dispatch; TransBO adapter used; fallback disclosed; exhaustion handled; `row_id` merge. |
| Auto DoE to BO and DoE to TransBO | DoE dispatch and results work; review gate points to active optimizer; post-review cycle works with correct optimizer. |
| Stop, timeout, exhaustion, restart | In-flight work discarded after Stop; only campaign-owned rows removed; state machine reaches correct terminal state; repeated Stop is a no-op. |
| CSV upload/import and malformed data | Quoted commas, reordered rows, missing objectives, non-numeric results, extra columns, duplicate rows produce visible validation, not silent positional merge. |
| Concurrent SA/surface/TransBO, multiple browser sessions | Busy job reported; campaigns from separate sessions cannot cross. |
| Deployment / IMEX folder transport | Port 5000, IMEX URL configurable, folder receive acknowledged, result retry works. Serial/network/USB modes hidden or labeled "not implemented". |

---

## 6. Required test gates

1. **EDOS core alignment**: 19 canonical files pass LF-normalized SHA-256 against EDOS `68a7d586`. Run as a CI gate on every commit to CEDOS.
2. **EDOS API/DoE/BO/SA tests**: 35 tests (carried from `alignment/shared_files.json` test entries). All must pass.
3. **TransEDOS audited tests**: 81 tests. All must pass. Add regressions for: `exhausted` response handling (manual TBO), `avoid_reval` disabled for TBO, numerical constraint forwarding.
4. **AutoEDOS bridge tests**: 15 Python + 4 Node tests. Add: simulated Auto to BO full cycle, simulated Auto to TransBO full cycle, row-ID merge, Stop mid-cycle, wrong-campaign rejection.
5. **CEDOS shell tests**: Add DOM/navigation tests for all four module-toggle states; role preservation across `switchModuleSafe`; per-module result snapshot isolation; duplicate DOM ID detection; revision-counter invalidation.
6. **Score/stop unit tests**: Confirm target stop uses observed objectives. Confirm no-improvement stop uses campaign-local bookkeeping (not the display ticker). Confirm time stop is unaffected.
7. **Build gate**: fail on any duplicate DOM IDs, stale response application, or canonical hash drift.

---

## 7. Completion criteria

CEDOS is ready when:

- Canonical EDOS files pass alignment at `68a7d586`.
- TransEDOS `routes.py` (no DirectML), `MAX_CONTENT_LENGTH`, and `_available_local_port` are all present.
- All module-toggle combinations behave predictably without role mutation.
- Auto can start from DoE or either optimizer (BO or TransBO).
- TransBO manual algorithm is reproducible: diagnostics, prior cells, `algorithm_used` disclosure all present.
- BO and TransBO each receive the intended constraint payload.
- Auto stop checks use observed experimental outcomes for `target` and `improvement` stops.
- Each hardware result (folder mode) is linked to exactly one proposed row via `row_id`.
- `exhausted`, `cancelled`, `completed`, `failed` and `waiting` states are all visible in the UI and distinct from each other.
- `avoid_reval` for TransBO and any unverified constraint path are disabled with visible explanations, not silently ignored.
- IMEX serial/network/USB modes are hidden or clearly marked as not implemented.
- All test gates in Section 6 pass in a single-process packaged build.
- Source repositories (`Optimized EDOS`, `TransEDOS`, `autoEDOS_combined`) are untouched.
- A `CEDOS_ALIGNMENT.md` documents every deliberate departure from source repositories.
