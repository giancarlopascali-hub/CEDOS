# CEDOS implementation plan

**CEDOS** = Complete Experimental Design, Optimization and Statistics.  
**Plan date:** 27 October 2026.  
**Status:** Code review and implementation plan only. No CEDOS application code has been created yet.

## 1. Goal and non-negotiable boundaries

Build one application with EDOS's DoE, BO and Statistical Analysis (SA), plus optional TransBO and Auto modules. The user can show or hide TransBO and Auto. When TransBO is shown, it replaces BO in the navigation and in Auto's optimization cycles. EDOS computational behavior and TransBO's transfer-learning method should remain intact. Integration work belongs in the application shell, shared browser state, Auto orchestration, API composition, and focused corrections to prevent silent failures.

The empty `EDOS complete app` folder is the destination. The reviewed source trees are sibling folders:

| Source | Use in CEDOS | Baseline |
|---|---|---|
| `../Optimized EDOS` | Canonical EDOS behavior and unchanged computational core | Git `ec44ab41571bb45bffaaa375c68c7337442dde12` |
| `../TransEDOS` | Starting application shell and transfer-specific UI, routes and engine | Git `c0bc364f72dba7dcfb37389c70752fe017689855` |
| `../autoEDOS_combined/AutoEDOS opt` | Auto UI, loop, bridge routes and guide to adapt | Repository Git `e24adff81b8de1eee9c526fdea4efa1d5800aaf9` |
| `../autoEDOS_combined/autoEDOS-IMEX` | Companion hardware bridge protocol and client changes | Review current working tree |

The source repositories contain unrelated papers, benchmarks and local changes. Copy the app and test files deliberately; do not overwrite or clean the source trees. Both downstream alignment checks currently pass against EDOS's locked revision. The apparent raw SHA-256 differences on Windows are line-ending differences; use the existing LF-normalizing alignment checker, not raw file hashes, for the no-core-change gate. "EDOS core untouched" means the canonical numerical engines and their existing API behavior remain identical; the browser shell and integration extensions need reviewed changes.

## 2. What the code review found

### Foundation and routing

- EDOS, TransEDOS and AutoEDOS use a Flask API plus a browser single-page interface. Their shared `src/core/{bo,doe,sa,constraints,feasibility}.py` files and job registry are aligned. TransEDOS already contains both `/optimize` and transfer routes, so it is the lowest-risk composition base. Auto adds a separate `auto_bp` (`../autoEDOS_combined/AutoEDOS opt/app.py:44-52`).
- TransBO's manual path submits `/transfer_optimize_submit`, receives a job ID, then polls `/job_status/<id>`; ordinary BO calls `/optimize` directly (`../TransEDOS/static/js/modules/tbo.js:1260-1398`; `../TransEDOS/src/api/routes.py:897-929,1243-1287`). The routes are distinct. Their main conflict is browser state and dispatch, not a duplicate optimization endpoint.
- The shared background job registry is process-local and admits one job by default (`../Optimized EDOS/src/api/jobs.py:1-5,21-53`). Auto→TransBO must handle a busy job explicitly. A multi-worker deployment would also split bridge and job state between processes unless that state is externalized.

### Navigation, roles and shared data

- The browser owns one row-major dataset (`currentData`, `currentColumns`), one column-role map, shared objective settings and BO feature settings, separate SA feature settings, one suggestions cache and one abort controller (`../TransEDOS/static/js/state.js:5-21`). There is no dataset persistence across page reloads. TransEDOS upload resets roles and feature settings, but leaves `priorConfigs`/`priorCombineRule` behind; matching column names in a later file could inherit stale prior settings (`../TransEDOS/static/js/main.js:281-325`).
- TransEDOS's `switchModule()` changes **every** `prior` role back to an included `feature` whenever the destination is not TransBO. Adding Auto to that switch unchanged would erase prior intent just by opening Auto or SA (`../TransEDOS/static/js/main.js:199-227`). SA must use selected feature/objective roles without silently turning prior context into an explanatory feature.
- DoE commit always switches to BO (`../TransEDOS/static/js/main.js:623-685`). TransBO-on must route that handoff to TransBO. Its result tables and commit logic are separate from BO's (`../TransEDOS/static/js/modules/tbo.js:1117-1177,1409-1528`).
- DoE and BO overwrite the same `suggestionsData`; TransBO stores results in module-local `latestTBOResults` and `transferDiagnostics` instead (`../Optimized EDOS/static/js/modules/doe.js:259`; `../Optimized EDOS/static/js/modules/bo.js:262`; `../TransEDOS/static/js/modules/tbo.js:1368-1379`). Auto cannot treat these caches as an optimizer-neutral result. Main-table cell edits also leave transfer previews and reusable training-row snapshots intact (`../TransEDOS/static/js/main.js:374-378`; `../TransEDOS/static/js/modules/tbo.js:126-150`).
- The same main dataset feeds BO/TransBO and SA, but each engine can clean or impute different effective training rows. Surface the usable-row count and missing-value issues before running, and invalidate SA results, plots and suggestion snapshots after any dataset/role/configuration revision (`../Optimized EDOS/src/core/bo.py:114-150`; `../Optimized EDOS/src/core/sa.py:82-142`).

### Auto-specific failure paths already present

- Auto can start with DoE or an uploaded dataset. The DoE phase deliberately pauses after measured results and sends the user to BO setup; it is not one uninterrupted DoE→BO run (`../autoEDOS_combined/AutoEDOS opt/static/js/modules/auto.js:449-478`). Preserve that review gate, but send the user to whichever optimizer is active.
- Auto hardcodes BO navigation, DOM selectors and `/optimize` (`auto.js:474-476,517-521,831-842,892-906`). Its BO request places `objective_constraints` outside `tweaks` and omits numerical/disallowed constraints; the backend reads all of them inside `tweaks` (`auto.js:580-633`; `../Optimized EDOS/src/api/routes.py:58-70`; manual BO payload in `../Optimized EDOS/static/js/modules/bo.js:203-239`). This can silently run a different optimization from the one configured.
- Auto's stop conditions use model cross-validation R² as though it were observed experimental success. Its display also overwrites the same `autoBestScore` used by the no-improvement stop check every second; a single-objective run may default the R² input to zero (`auto.js:98-255,563,746-757`). Stop criteria need defined experimental-outcome units and independent bookkeeping.
- Auto appends blank-objective rows before a bridge POST succeeds, then matches incoming results by position to **all** blank-objective rows in the dataset. Stop removes all such rows, including unrelated user rows (`auto.js:260-360,412-440,760-779`). Its feature-only hardware payload can also keep using an earlier DoE feature snapshot after optimizer setup changes (`auto.js:321-338`).
- The bridge has a single process-global proposal/results slot, no campaign or proposal identity, and consumes a proposal before IMEX confirms hardware dispatch (`../autoEDOS_combined/AutoEDOS opt/src/api/auto_routes.py:30-91`). In-flight optimization/polling can still update the UI after Stop. IMEX's folder result path is implemented; serial, network and USB result receive paths are stubs (`../autoEDOS_combined/autoEDOS-IMEX/auto_imex_v2.0.py:322-400`).

### TransBO behavior to preserve and make explicit

- Transfer optimization uses the shared EDOS BO engine as its *existing* no-prior fallback; it reports `algorithm_used: standard_bo` and may cap the training rows using TransBO's `max_gp_rows` (`../TransEDOS/src/core/transfer_bo.py:318-340`). This can differ from a full-data BO run. CEDOS should call the TransBO route when TransBO is active, and clearly tell the user when no configured prior means no transfer weighting is applied. Do not silently label fallback as transfer learning.
- The manual TransBO result table prepopulates prior cells with configured target values and commits all dataset columns (`../TransEDOS/static/js/modules/tbo.js:1117-1177,1447-1483`). Auto→TransBO must preserve that context in pending/measured rows, rather than appending blank prior columns.
- Existing TransBO code has three acceptance risks to assess before release: `avoid_reval` has an unimplemented duplicate-filter branch (`../TransEDOS/src/core/transfer_bo.py:867-870`); numerical/disallowed feature constraints are read but applied in the cold-start path, while the fitted-GP path needs enforcement and cold start can fall back to unconstrained Sobol points (`transfer_bo.py:495-539`); and the TransBO UI does not handle an `exhausted` response even though its route can return one (`../TransEDOS/src/api/routes.py:558-561`; `../TransEDOS/static/js/modules/tbo.js:1368-1386`). Preserve the transfer weighting/GP method. Correct a verified defect narrowly, or block an unsupported option with a visible explanation; never silently accept it.

## 3. Product rules to implement

1. **Initial view:** DoE, BO and SA are visible; TransBO and Auto are hidden. Add an obvious Modules/Preferences control with independent TransBO and Auto toggles. Store visibility preferences per browser if desired, but never persist or discard a dataset merely because a tab is hidden.
2. **Optimizer selection:** `activeOptimizer` is exactly `bo` or `tbo`. TransBO on means show TransBO and hide BO, including navigation, setup links and Auto wording. TransBO off restores BO. The backend may retain both endpoints, but only the selected optimizer is dispatched by the UI and Auto. If a hidden module is currently selected, switch to its replacement immediately.
3. **Safe toggling:** While an Auto campaign, optimization job, or uncommitted hardware batch is active, freeze optimizer/Auto visibility changes and explain how to Stop or finish first. A tab switch for inspection may remain possible without changing roles or optimizer. A stale asynchronous response must not restore a hidden tab or apply results after a toggle or dataset change.
4. **Canonical dataset:** Keep one dataset and schema for all modules. A column can have the explicit role `feature`, `objective`, `prior` or unselected; changing tabs does not change roles. Prior columns remain context and are excluded from BO/SA features unless the user explicitly reclassifies them. Shared objective definitions stay synchronized; module-specific tweaks and results remain scoped to their module.
5. **Auto entry:** Auto can begin with DoE or the currently selected optimizer. A DoE result batch pauses for review of BO **or** TransBO setup before optimization starts. A direct optimizer start requires a validated existing dataset. Freeze the selected optimizer for the campaign; switching it requires stopping and starting a new campaign.
6. **No-prior TransBO:** Manual TransBO retains its existing fallback. For Auto→TransBO, require the user to acknowledge a visible no-prior fallback before dispatch; otherwise prompt for prior targets. Validate prior columns and target values before each transfer batch. DoE-created datasets without prior columns must make this choice visible at the review gate.
7. **Stop semantics:** Count completed measured batches as iterations. Evaluate target/no-improvement against observed objectives, never model-fit R². Set time limits from the optimization phase start, and display DoE time separately. Define direction-aware single-objective thresholds and a fixed, documented multiobjective target rule before enabling percentage-based target stops. Preserve distinct `completed`, `exhausted`, `cancelled`, `failed` and `waiting` states in UI and log.

## 4. Implementation sequence and file ownership

### Phase A — Compose a clean, traceable CEDOS foundation

- Bring the application portions of TransEDOS into this folder as the shell. Keep EDOS-owned `src/core/{bo,doe,sa,constraints,feasibility}.py` and `src/api/jobs.py` unchanged at the aligned revision; preserve the canonical DoE JS and EDOS behavior. Record the EDOS commit and TransEDOS/Auto source commits in CEDOS alignment metadata; add a normalized-hash verification task for canonical files.
- Review every TransEDOS file marked `shared_with_extensions` against its EDOS counterpart before adding Auto. In particular, `src/api/routes.py`, `static/js/{main,state,utils}.js`, `static/js/modules/{bo,sa,constraints}.js`, `templates/index.html` and `static/style.css` already differ. Keep necessary TransBO compatibility and reliability fixes; reject accidental drift in ordinary EDOS workflows. Merge Auto's shell additions into those reviewed files, rather than replacing TransEDOS's complete file with the AutoEDOS version.
- Add Auto's `src/api/auto_routes.py`, `static/js/modules/auto.js` and `templates/auto_guide.html`; register its blueprint and guide route in CEDOS `app.py`/`src/api/routes.py`. Check every URL and HTML ID in the merged template (each source template currently has no duplicate IDs). Keep transfer routes in the TransEDOS API shell; do not concatenate whole route files.
- Use one tested dependency set: EDOS pins plus TransBO's RDKit and `python-dotenv` as needed. Resolve the TransEDOS Dockerfile's separate `rdkit-pypi` install versus its pinned `rdkit` requirement. Choose local port 5000 or make the IMEX URL configurable and display the actual CEDOS origin; ensure the deployed port and bridge configuration agree. Update health/version labels, README, guide links and build metadata to CEDOS.
- At this gate, run foundation alignment and the existing EDOS, TransEDOS and Auto suites. Any failing EDOS core hash or regression blocks feature work.

### Phase B — Add a small CEDOS shell controller

- Add one owner for `enabledModules`, `activeOptimizer`, `currentModule`, dataset revision, active request/campaign IDs and per-module result snapshots (for example `static/js/cedos.js` plus a small state extension). Put tab switching and DoE commit routing here. Remove TransEDOS's implicit prior→feature conversion from `main.js:199-227`; change only shell behavior, not EDOS calculation modules.
- Merge the Auto tab and module controls into `templates/index.html`, with separate `bo-*`, `tbo-*` and `auto-*` IDs. Add a return-to-Auto control in the TransBO view and replace BO-specific labels in the Auto wizard with the active optimizer's name.
- On upload, clear, role edit, cell edit, import or objective/feature change, increment dataset revision; invalidate plots, SA report cache, optimizer results and request tokens. Cancel an active request/job where possible. Before rendering any asynchronous result, confirm that its revision, module and campaign still match.
- Keep BO and TransBO setup separate where their parameters differ. Shared column-role/objective edits may update both, but hidden BO strategy controls must not silently rewrite TransBO feature types. Audit constraints UI for `doe`, `bo`, `tbo` and `sa` prefixes and ensure each request reads the active setup.
- Give each module its own suggestion/result snapshot so a DoE or BO run cannot make another module's Commit/Export button act on the wrong rows. Validate exact schema and selected features for imports and commits; use a real CSV parser and explicit row matching for user-imported measured results instead of simple comma splitting/positional fallback (`../Optimized EDOS/static/js/main.js:692-770`).

### Phase C — Define optimizer adapters for Auto

Create a narrow interface in the CEDOS Auto layer: `validateSetup(dataset)`, `buildRequest(dataset)`, `generateSuggestions({signal, onProgress})`, `normalizeResult(response)`, `cancel()`. Both adapters return a common result containing suggestions, selected feature/objective names, constraint/feasibility warnings, exhaustion status and diagnostics. The Auto loop must not scrape BO table rows or rely on `window.suggestionsData`/`latestTBOResults`.

- **BO adapter:** use the same selected feature/objective and `tweaks.constraints`/`tweaks.objective_constraints` serialization as manual BO. POST `/optimize`; validate nonempty, schema-correct suggestions.
- **TransBO adapter:** use the same prior serialization, row-cap, objective constraints, numerical constraints and selected features as manual TransBO. POST `/transfer_optimize_submit`, poll `/job_status/<id>`, cancel `/job_cancel/<id>` on Stop, and surface queued/running/recovery/cold-start/429/exhausted states. Do not change the transfer model or weighting math. Inspect `transfer.algorithm_used` to disclose the native no-prior fallback.
- Do not keep `window.currentAbortController` as the sole lifecycle guard. Give every Auto generation and polling cycle a campaign-scoped cancellation token; Stop also invalidates the token, so delayed responses cannot push proposals or merge results.

### Phase D — Make Auto/IMEX exchange transactional

- Replace the single unqualified bridge slot with campaign-scoped proposal records. Generate a `campaign_id`, unique `proposal_id`, dataset/schema revision and row IDs for every batch; bind the campaign to one optimizer. Use a pairing token so a connected IMEX instance reads and posts only its campaign. Keep the server single-worker until bridge/job state is made shared or durable; use a small persistent store for recoverable pending batches if hosting or restart recovery is required.
- Validate the complete suggestion set *before* dispatch: exact selected feature names, no duplicate row IDs, finite values and required prior context. Create only proposal-owned pending rows after the server accepts a proposal; keep a rollback record. For TransBO, fill configured prior target values in each pending row and allow measured prior values from IMEX to replace them if the result schema includes them.
- Make dispatch acknowledgeable and idempotent: IMEX receives an identified proposal, acknowledges hardware acceptance, and can safely retry. A GET must not permanently consume work before dispatch succeeds. Include proposal/row IDs in folder filenames or a manifest and in result uploads. Reject stale, duplicate, reordered, missing-column or wrong-campaign results visibly; merge only into the matching pending rows. Do not remove unrelated blank-objective rows on Stop.
- Provide an actionable wait timeout, reconnection/retry and error state. Clearly mark which IMEX transports actually receive results; do not present stub serial/network/USB receive modes as working closed-loop paths. Add a way to recover or cancel a pending batch without silently losing it.
- Validate that changes to dataset, column roles, selected objectives, prior targets or optimizer during a campaign either are blocked or explicitly restart the campaign. SA may inspect a snapshot while Auto runs, but competing analysis/transfer jobs must queue or show the shared one-job-at-a-time limit.

### Phase E — Score and status corrections

- Separate **model diagnostics** (R², uncertainty, transfer diagnostics) from **measured experimental outcomes**. Use one pure scoring function for the displayed result and stop checks; it must not mutate the stored best score during a render tick. Compare improvement only after a valid newly measured batch. Validate stop values as positive integers/finite thresholds and reject unknown types.
- Keep DoE-start and optimizer-start transitions explicit: `idle → generating DoE/optimizer → awaiting dispatch/results → review optimizer (DoE path only) → optimizing → awaiting results → terminal`. Terminal transitions are idempotent. A failed POST/job/merge keeps the last committed dataset and shows a recovery action.
- Test BO and TransBO exhaustion separately. For TransBO, render the backend's `exhausted` sentinel in manual and Auto flows. For the preexisting TransBO `avoid_reval`/constraint gaps, write a reproduction and either apply a narrow, separately reviewed correctness fix or visibly disable the unsupported setting. The release gate is that a selected constraint or avoidance setting cannot be ignored silently.

## 5. Workflow verification matrix

Use a simulated IMEX and small deterministic datasets in automated browser/API tests; reserve real hardware validation for a final smoke run. Test all four toggle combinations and repeat the important paths after page reload and Stop/restart.

| Flow | Expected result / failure check |
|---|---|
| Extra modules off: DoE→BO→SA; upload→BO→SA | EDOS behavior unchanged; DoE commit lands on BO; shared objectives and main dataset consistent. |
| TransBO on: DoE→TransBO; upload→TransBO→SA; TransBO off→BO | BO is hidden; no hidden BO dispatch; prior roles/targets survive viewing Auto or SA; SA excludes prior context unless reclassified. |
| Manual TransBO optimize, estimate, visualization, commit/export | Native transfer diagnostics and target prior cells retained; no-prior fallback labeled; exhausted, cancelled, 429 and constraint cases visible. |
| Auto direct BO and direct TransBO | Selected optimizer's exact request shape, selected features, constraints and results used; one proposal per batch; measured rows merged once. |
| Auto DoE→BO and DoE→TransBO | Initial DoE dispatch/results work; review gate points to active optimizer; TransBO prior prerequisite/fallback is explicit; post-review cycle works. |
| Stop, timeout, exhaustion, restart | In-flight work cannot apply after Stop; only campaign-owned pending rows are cleaned up; status and time/iteration counters are correct. |
| CSV upload/import and malformed data | Quoted commas, reordered rows, missing objectives, nonnumeric results, extra columns and duplicate rows produce a visible validation result rather than a silent positional merge. |
| Concurrent SA/surface/TransBO work, multiple browser sessions | Busy jobs are serialized or reported; datasets/proposals/results from separate campaigns cannot cross. |
| Deployment/IMEX folder transport | CEDOS URL/port pairing, RDKit availability, single/shared worker assumptions, dispatch acknowledgment and result retry work in the packaged app. |

### Required test gates

1. Preserve and run existing EDOS API/DoE/BO/SA tests and core alignment checks. Existing focused EDOS API/SA/hosted UI run: **35 passed**.
2. Preserve and run TransEDOS prior, API, surface, reliability, background and hosted tests. Existing focused audited selection: **81 passed**. Add regressions for exhaustion, duplicate avoidance and constraints before claiming those options are safe.
3. Preserve and run Auto bridge/DoE tests. Existing targeted runs: **15 Python tests** plus **4 Node tests** passed. Add end-to-end AutoBO and AutoTransBO simulated bridge tests; the existing suite does not cover the combined cycle.
4. Add DOM/navigation tests for all toggle states, schema/role preservation and per-module result isolation. Add API contract tests for the two optimizer adapters and campaign/result IDs. Fail the build on duplicate DOM IDs, stale response application or shared-core hash drift.

## 6. Completion criteria

CEDOS is ready to implement/release when: unchanged EDOS core files pass alignment; all toggle combinations behave predictably; Auto can start from DoE or either selected optimizer; TransBO's manual algorithm remains reproducible; BO/TransBO/SA see the intended dataset roles; each hardware result is linked to exactly one proposed row; constraints, stopping rules and exhaustion are honored or visibly rejected; and the full workflow matrix passes in a single-process packaged build. Keep source app repositories untouched during the merge and document any deliberate departure from them in CEDOS's README and alignment manifest.
