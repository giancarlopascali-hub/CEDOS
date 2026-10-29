# CEDOS implementation plan — v3

**CEDOS:** Complete Experimental Design, Optimization and Statistics  
**Date:** 27 October 2026  
**Status:** Local CEDOS build implemented for user testing. Hugging Face Space and standalone packaging remain future phases. See `CEDOS_LOCAL_STATUS.md` for the verified local result. This document supersedes `CEDOS_IMPLEMENTATION_PLAN_v2.md`.

## 1. What v3 changes

V2 described the older sibling checkouts. The current feature code is in three newer, committed worktrees under `../_codex_worktrees/`. Their alignment checks pass against a newer EDOS foundation. V3 also treats the new **Setup CSV** exchange as its own workflow, separate from experimental dataset and measured-results CSV files.

| V2 assumption | Verified current state and v3 decision |
|---|---|
| Foundation `68a7d586`, 19 exact shared files | EDOS Setup CSV foundation is `6af1c9e5ea48ac4ca08799c83d5ab72c5748a7b7`. EDOS `alignment/shared_files.json` now lists 30 canonical paths; the downstream exact/extension sets differ. Verify the current manifests and hashes instead of using a fixed count. |
| Copy EDOS `utils.js`, `bo.js`, `sa.js`, `constraints.js` verbatim into TransEDOS | TransEDOS extends those files for prior roles, transfer constraints and navigation. Replacing them with EDOS versions would break TransBO. Use the current TransEDOS shell and review its extension diffs. Preserve only the truly exact shared files byte-for-byte after line-ending normalization. |
| AutoBO calls synchronous `/optimize` | Current manual BO and AutoBO submit guarded background jobs through `/bo_submit`, then poll/cancel. CEDOS's BO adapter must preserve that contract, workload policy, progress and timeout behavior. `/optimize` remains a compatibility endpoint. |
| Auto target stop reads model R² | Current Auto no longer assigns R² there; it reads an observed-data score **before** the next measurement. Stop evaluation, display and best-score bookkeeping still need repair. Evaluate only after an identified IMEX result batch is merged. |
| Auto places objective constraints correctly at the request top level | Current `auto.js` puts objective constraints under `tweaks`, as the API expects. It still omits BO numerical and disallowed feature constraints and newer BO workload choices. |
| Block every tab switch during an Auto campaign | The DoE-start review gate requires navigation to the active optimizer. Permit viewing and review-stage setup changes; lock optimizer toggles and unsafe edits while a job or hardware batch is active. |
| One generic CSV import scenario | Setup CSV exchanges feature/objective configuration and constraints. Dataset CSV and measured-results CSV are separate code paths with different validation and safety rules. |

### Current integration inputs

| Product | Source to use | Current revision / ownership |
|---|---|---|
| EDOS | `../_codex_worktrees/setup-csv-edos` | HEAD `d481dbbf2c6a0f82b7d6aec59db336667b3b6aef` publishes foundation `6af1c9e5ea48ac4ca08799c83d5ab72c5748a7b7`. |
| TransEDOS | `../_codex_worktrees/setup-csv-transedos` | HEAD `0af60b40885e3ad78d87c8157625677e8762a5ba`; aligned to `6af1c9e`. Use as the CEDOS application shell. |
| AutoEDOS | `../_codex_worktrees/setup-csv-autoedos/AutoEDOS opt` | Repository HEAD `4aabb13bbbc474092d0774e07f660667334f4ec1`; aligned to `6af1c9e`. Port its Auto-owned layer. |
| IMEX | `../_codex_worktrees/setup-csv-autoedos/autoEDOS-IMEX` | Companion client; review and version its bridge protocol separately. |

The older `../Optimized EDOS`, `../TransEDOS` and `../autoEDOS_combined` sibling checkouts remain at October 8 revisions and lack Setup CSV. If these worktree commits are later integrated into their main checkouts, reverify the exact commits and alignment before copying. Use committed source content; do not copy unrelated uncommitted favicon changes present in the TransEDOS and AutoEDOS worktrees.

**Verification already performed:** both downstream `alignment/check_alignment.py` checks pass against EDOS `6af1c9e` (TransEDOS: 21 exact paths; AutoEDOS: 32). Focused Setup CSV tests pass (EDOS 5, AutoEDOS 5, TransEDOS 6). The shared Chrome smoke script passed in all three products; it covered DoE/BO/SA setup download/upload, plus TransBO in TransEDOS, constraint restoration, unchanged dataset rows and preserved TransBO Prior Targets. These are feature checks, not CEDOS end-to-end tests. Re-run the full current suites after composition; the v2 test counts came from an older baseline.

## 2. Verified Setup CSV contract and limits

`src/core/setup_csv.py` is identical in the three new source trees. Its version 1 CSV has a UTF-8 BOM and 13 named columns (`setup_csv.py:17-23,48-84`). Records contain feature name/type/range or categorical choices/inclusion; objective name/direction/target/importance/inclusion; numerical and disallowed feature constraints; and objective constraints. The products expose `/setup_csv/export` and `/setup_csv/import` alongside, but distinct from, `/upload` and measured-results imports (TransEDOS `src/api/routes.py:1022-1056`). Import accepts at most 2 MB, validates the schema and declared domains, and returns parsed JSON plus warnings without editing experiment rows (`setup_csv.py:87-246`).

Each product's `static/js/modules/setup_csv.js` captures the **live Setup controls** and applies the parsed response to browser state. EDOS and AutoEDOS expose DoE, BO and SA; TransEDOS also exposes TransBO. The TransEDOS browser file is the correct CEDOS starting point because it has `tbo` collectors, render hooks and prior protection (`../_codex_worktrees/setup-csv-transedos/static/js/modules/setup_csv.js:1-294`). The common Python parser should remain exact; CEDOS should register one route set, not concatenate EDOS, TransEDOS and AutoEDOS copies.

**What the format does not save:** experimental data rows; DoE design-model/budget tweaks; BO/TransBO acquisition, kernel, batch or workload choices; SA model tweaks; TransBO prior columns, Prior Targets, prior combination rule or row-cap configuration. The TransEDOS README explicitly describes its TransBO controls as **Features & Objectives** only (`../_codex_worktrees/setup-csv-transedos/README.md:33-37`). A setup file is therefore a portable partial configuration, not a complete campaign restore. Keep schema version 1 compatible; if complete transfer/campaign export is later requested, add a separately versioned extension rather than silently changing this file.

**Cross-module mapping:** imports may target another module. The parser requires matching dataset column names for BO, SA and TransBO, but deliberately permits configured ranges and categories outside observed historical values (`setup_csv.py:194-202`). DoE drops excluded rows and objective constraints; SA drops ranges/feature constraints and converts `regular` to `discrete`; TransBO drops disallowed-combination controls. These changes generate warnings (`setup_csv.py:204-236`). CEDOS must show a clear preview of dropped or transformed settings before Apply, and keep those warnings visible after Apply.

**Current integration gaps:**

- Upload checks that `currentColumns` still matches after the server replies, but does not check dataset contents, role/configuration revision, active module, or Auto campaign state (`setup_csv.js:238-266`). A delayed import can overwrite a newer setup or an active campaign.
- Applying a data-backed setup rewrites shared `columnRoles`, `objectiveConfigs` and BO/SA feature configs (`setup_csv.js:130-175`). An import into BO, SA or TransBO can therefore change another module's interpretation of the same dataset. The current prior-role guard runs only for a TransBO destination (`setup_csv.js:255-260`); BO or SA imports can still reclassify a prior column.
- DoE import replaces its Setup rows. Data-backed imports update listed names but retain omitted existing roles/configurations (`setup_csv.js:105-175`). The status message does not explain that overlay behavior. CEDOS must state which settings were applied and which were retained.
- The JavaScript rollback only attempts to restore captured config objects and reconstructed Setup controls if `applySetup()` throws; it is not a complete campaign/cache rollback. Setup import needs one transaction boundary and downstream invalidation.
- TransBO Prior Targets are intentionally preserved by the happy-path TransBO import, but the imported file cannot make a TransBO/AutoTransBO run transfer-ready. Prior targets and the selected optimizer must be validated separately before dispatch.

## 3. CEDOS behavior and state rules

1. **EDOS foundation:** preserve the aligned EDOS numerical engines and new BO workload components. The new `setup_csv.py` is additive and exact shared code; do not edit existing EDOS computational modules to implement CEDOS navigation or Auto.
2. **Visibility:** initial navigation shows DoE, BO and SA. Independent toggles reveal Auto and/or TransBO. TransBO on hides BO and becomes the only visible optimizer; turning it off restores BO. Hiding a module does not erase dataset rows or its saved setup. Use one `activeOptimizer: 'bo'|'tbo'` decision for navigation, DoE handoff and Auto dispatch.
3. **Shared data:** one row-major dataset and schema serve BO/TransBO and SA. Roles remain stable across ordinary tab switches. On the user's later correction, turning TransBO off explicitly converts `prior` columns to BO `feature` columns and seeds BO/SA feature settings; re-enabling TransBO restores their Prior roles and targets only if those BO roles/settings stayed untouched. Objective settings are shared; BO and TransBO currently share `boFeatureConfigs`, while SA has separate feature configs. Make every cross-module side effect of a Setup CSV import visible.
4. **Setup CSV lifecycle:** export does not change state. Import parses and previews before mutation. For DoE, it replaces the DoE Setup rows as today. For BO/SA/TransBO, treat imported names as an **overlay** by default; explicitly show omitted existing rows that will remain. Reject a file that would turn a `prior` column into a feature/objective until the user deliberately changes that role in the main dataset UI. Never silently discard `priorConfigs`.
5. **Import timing:** permit Setup CSV import in idle/preflight and in the post-DoE `review_optimizer` state for the selected optimizer. During a queued/running BO or TransBO job, pending hardware proposal, or result merge, block import with an explanation. Permit navigation for inspection and review; freeze the active optimizer toggle and edits that would alter a dispatched batch.
6. **Revision gate:** capture dataset/schema, role/configuration and campaign revisions when import starts. Recheck all of them before Apply. After a successful import, increment the configuration revision, invalidate affected DoE/BO/TransBO suggestions, transfer previews/training snapshots, SA reports/plots and Auto preflight. Re-render linked Setup panels. A failed import leaves both data and configuration unchanged.
7. **Auto start:** permit a DoE start or a start from the active optimizer with an existing validated dataset. Preserve Auto's post-DoE optimizer review gate. A Setup CSV import at that gate may set features/objectives/constraints, but Auto must build its next proposal from the **current selected optimizer setup**, not a stale `currentDoEFeatures` snapshot. Prior targets, workload choices and stopping rules still require explicit review.
8. **Scoring/status:** evaluate stop conditions only after a valid measured batch is merged. Keep model diagnostics separate from experimental objective outcomes. Track no-improvement state independently of the display renderer. Count completed batches and distinguish completed, exhausted, cancelled and failed states.

## 4. Implementation phases and file ownership

### Phase A — Freeze the new baseline and compose the app

- Start from committed TransEDOS `0af60b4` in a new CEDOS repository/working tree. Record EDOS foundation `6af1c9e`, TransEDOS `0af60b4` and AutoEDOS `4aabb13` in CEDOS alignment/build metadata. Keep its `app.py`, `src/api/routes.py`, template and frontend extensions as the merge base; modify them deliberately for CEDOS. Do **not** describe a file as copied verbatim if CEDOS must edit it.
- Use the current `alignment/alignment.yml` classifications. Keep TransEDOS's exact shared numerical core (`bo.py`, `bo_policy.py`, `bo_sampling.py`, `doe.py`, `sa.py`, `constraints.py`, `feasibility.py`), `src/core/setup_csv.py`, `src/api/{bo_worker,jobs}.py`, `static/js/modules/{doe,bo_workload}.js` and their exact tests aligned. Preserve TransEDOS's extended `utils.js`, `bo.js`, `sa.js`, `constraints.js`, `main.js`, `state.js` and `setup_csv.js` where its manifest calls them extensions. EDOS's 30-path list is a foundation reference, not a command to replace every extended TransEDOS file.
- Add Auto-owned `src/api/auto_routes.py`, `static/js/modules/auto.js`, `templates/auto_guide.html` and focused tests. Register one `auto_bp` alongside the TransEDOS `api_bp` and add the Auto guide route. Merge Auto's tab and controls into the TransEDOS template without duplicate IDs or duplicate `/setup_csv/*` routes. Use the TransEDOS Setup CSV controls for all four manual modules; change only the download filename prefix/labels to CEDOS.
- Preserve the hardened BO worker, workload selector and tests introduced after v2 (`src/core/bo_policy.py`, `bo_sampling.py`, `src/api/bo_worker.py`, `static/js/modules/bo_workload.js`). Manual BO uses `/bo_submit`; `/optimize` is a compatibility endpoint that may refuse hosted workloads with `async_required` (`../_codex_worktrees/setup-csv-edos/src/api/routes.py:464-580`). AutoBO must use the same guarded submit/status/cancel workflow.
- Keep TransEDOS's CPU/CUDA route policy (it excludes DirectML) and its 200 MB Flask request limit. Use a local port compatible with the configurable IMEX URL, and display the actual CEDOS origin. Resolve the TransEDOS Dockerfile's duplicate RDKit installation (`requirements.txt` pins `rdkit==2026.3.6` and Dockerfile also installs `rdkit-pypi`); test one supported package path. Keep a single server worker until background jobs and bridge state are shared across processes.
- Phase A gate: current alignment checks, current app test suites, startup, route registration and duplicate-ID scan. Record actual passing totals from this baseline; do not reuse v2's historical counts as a release claim.

### Phase B — Add a small CEDOS shell and dataset/configuration revision model

- Add one shell owner for visibility flags, `activeOptimizer`, `currentModule`, dataset revision, configuration revision, module-specific result snapshots, active job and active campaign. Route every tab and DoE handoff through it. Remove TransEDOS's `switchModule()` prior→feature mutation (`../_codex_worktrees/setup-csv-transedos/static/js/main.js:194-242`); perform the later user-requested P→F conversion only when the TransBO toggle is turned off, preserving explicit BO edits when it is turned on again.
- Allow tab navigation while a campaign is active, because Auto's DoE review gate needs it. Lock TransBO/Auto toggles throughout a campaign. During a dispatched/in-flight batch, make setup/data controls read-only or reject their action with a visible reason; restore editing only in the review state or after Stop.
- Keep DoE, BO and TransBO suggestion snapshots separate. `window.suggestionsData` is shared by DoE and BO while TransBO keeps a module-local result; never let a hidden module's stale result feed Commit, Export or Auto. Invalidate relevant snapshots and asynchronous response tokens after edits, imports or optimizer changes.
- Treat BO and SA effective training rows as potentially different; provide a readiness check for missing values/types. Keep experimental data CSV/result CSV import validation separate from Setup CSV validation. The old measured-results importer still uses simplistic comma splitting/positional mapping and needs its own correction before automated use.

### Phase C — Integrate Setup CSV as a transaction

- Preserve `src/core/setup_csv.py` schema version 1 and the `/setup_csv/export`/`/setup_csv/import` API. Adapt TransEDOS `static/js/modules/setup_csv.js` with CEDOS shell hooks, rather than creating another configuration format or a second handler. Keep Setup CSV controls in DoE, BO, TransBO Features & Objectives, and SA.
- Before Apply, show source/destination module, imported and retained names, transformed/dropped settings, prior-role conflicts, and the fact that dataset rows, Prior Targets and optimizer tweaks will not be changed. Require explicit Apply for cross-module or role-changing imports; reject invalid/unsupported imports without partial state changes.
- Use the shell's full revision/campaign token around the asynchronous server parse. Stage new DOM/config state, validate role and constraint references, then apply atomically. At the review gate, revalidate Auto's active optimizer and rebuild its future feature/objective selection. Invalidate caches and plots affected by shared objective/role changes. Preserve current dataset rows and columns exactly.
- Add regression tests for each module's own-file round trip, BO↔SA and BO↔TransBO cross-module imports, quoted categorical values, widened/unobserved ranges, unsupported-setting warnings, omitted-row overlay semantics, duplicate names, stale replies, prior-role protection for **every** destination, import during Auto review, and blocked import during an in-flight Auto batch. Verify exported/imported constraints match the active manual optimizer payload.

### Phase D — Adapt Auto to the selected optimizer

Define a small optimizer adapter contract that validates the active Setup and data, builds the exact request, submits/polls/cancels, and returns normalized suggestions, selected feature/objective names, feasibility/transfer diagnostics and exhaustion state. It must read the current Setup after an allowed Setup CSV import; it must not read a stale `suggestionsData` or DoE snapshot.

- **BO adapter:** serialize the same feature/objective/constraint/workload choices as current manual BO (`static/js/modules/bo.js:354-465` in the EDOS worktree). Include `tweaks.constraints.{numerical,disallowed}` and `tweaks.objective_constraints`. Submit `/bo_submit`, use job progress and cancellation, and keep Auto's `compute_performance: false` behavior unless a separate measured-result decision requires diagnostics. Do not fall back silently to synchronous `/optimize` on 429/422.
- **TransBO adapter:** use TransEDOS's transfer payload and `/transfer_optimize_submit` job flow, including prior configuration, row cap, objective constraints, progress and cancellation. Keep the transfer weighting/GP algorithm intact. If no prior is configured, ask for acknowledgement and disclose the existing `algorithm_used: standard_bo` fallback (which may use capped rows). Preserve target prior values in pending/measured rows.
- Freeze the selected optimizer for a campaign. A DoE start dispatches and merges its first measured batch, pauses at `review_optimizer`, then sends the user to BO or TransBO. Setup CSV import is allowed there, followed by revalidation. A direct optimizer start requires a data-backed Setup and any separate Prior Target/workload choices.
- Compute stop decisions from newly merged measured objectives. Current Auto has switched from R² to `getBestScoreAndLabel().rawVal`, but stores that score before the next IMEX result and reuses `autoBestScore` in a display side effect (`../_codex_worktrees/setup-csv-autoedos/AutoEDOS opt/static/js/modules/auto.js:98-255,490-580`). Define direction-aware single-objective thresholds; for multiple objectives use a fixed, documented target/reference instead of a percentage whose min/max shifts with each dataset update. Reset timing/iteration bookkeeping at the intended campaign or optimization phase boundary and preserve distinct terminal badges.
- Phase D gate: simulated DoE→BO, DoE→TransBO, direct BO and direct TransBO cycles; Setup CSV import at review; identical manual-versus-Auto payload checks for constraints; stop-on-measured-result checks; 429, cancel, timeout and exhaustion handling.

### Phase E — Make the IMEX exchange identifiable and recoverable

- Give each campaign, proposal and suggested row its own ID, schema/config revision and selected optimizer. Stage only the proposal-owned pending rows after the server accepts a validated batch. Merge returned measurements by row ID and objective name after validating campaign/proposal IDs, counts, finite values and prior context. Stop removes only that campaign's pending rows, not every blank-objective row.
- Change proposal delivery to a lease/acknowledgment or other idempotent protocol. A non-consuming GET alone would let repeated IMEX polls redispatch the same experiment; consumption-before-dispatch loses it on failure. Have IMEX deduplicate proposal IDs and acknowledge hardware acceptance, then support safe retry. Carry IDs through folder filenames/manifests and result CSV uploads. Scope status/recovery endpoints to the paired campaign; do not expose a full global campaign record to another browser or IMEX client.
- Folder result transport works today; serial/network/USB receive paths are stubs. Present only working closed-loop modes, or mark unsupported modes unavailable. Handle server restart, stale result, duplicate result, missing result and Stop/retry with a visible recovery action. Keep one worker until bridge/job state is made durable/shared.
- Phase E gate: simulated IMEX tests for acknowledgement, deduplication, wrong-campaign rejection, row-ID merge, restart/reconnect, safe Stop, and unchanged unrelated dataset rows.

### Phase F — Close TransBO's known silent-failure gates

- Manual and Auto TransBO must render the backend's `exhausted` sentinel instead of an empty suggestions table. Surface cold-start, reduced-row recovery and fallback diagnostics.
- The existing `avoid_reval` GP branch contains a `pass` (`../_codex_worktrees/setup-csv-transedos/src/core/transfer_bo.py:867-870`). Disable that control with a visible explanation until a focused fix and regression prove it works.
- Numerical/disallowed feature constraints need a fitted-GP regression; the reviewed transfer implementation applies them in cold start and can fall back to unconstrained Sobol points (`transfer_bo.py:495-539`). If the selected constraint cannot be enforced without changing transfer math, block that combination visibly. Do not allow a Setup CSV import to make an unsupported constraint silently active.
- Keep any necessary transfer correction separately reviewed and tested; do not alter the EDOS core or quietly change TransBO's transfer-learning method.

## 5. Workflow and test matrix

| Scenario | Required observation |
|---|---|
| Extras off: DoE→BO→SA; upload→BO→SA | Existing EDOS calculations and UI behavior preserved; DoE hands off to BO; hardened `/bo_submit` works. |
| TransBO on/off with SA | BO hidden/restored as selected; P stays P across ordinary tabs while TransBO is active. Turning TransBO off converts P to F for BO/SA; turning it on restores untouched Prior roles and targets. |
| Setup CSV in DoE, BO, SA and TransBO | Version 1 files round-trip features/objectives/constraints; TransBO prior targets and optimizer tweaks remain untouched and visibly outside file scope; data rows/columns unchanged. |
| Cross-module Setup CSV import | Unsupported records are previewed/warned; unobserved ranges/categories remain valid; omitted rows follow the documented overlay rule; no silent prior conversion. |
| Setup CSV while Auto is idle, reviewing, or running | Idle/review import validates and invalidates results; in-flight import is blocked; stale server replies do not overwrite newer configuration. |
| Manual BO and TransBO | BO uses guarded background workload policy; TransBO uses its own job path and shows fallback/exhaustion/constraint limitations. |
| Auto direct BO/TransBO and DoE→BO/TransBO | The active optimizer alone runs; latest imported Setup is used; IMEX rows merge once by ID; observed-result stop logic works. |
| Dataset/result CSV import | Quoted values, reordered/missing rows, invalid objectives and stale batches produce visible errors; never confuse measured results with Setup CSV. |
| Multi-session/busy/restart | One-job limit is reported or queued; campaigns cannot exchange data; restart and retries have a clear recovery path. |

**Test gates:**

1. Run the current EDOS, TransEDOS and AutoEDOS suites and both alignment checks at their new revisions. Add an exact/shared ownership gate that includes `setup_csv.py` and the new BO hardening files but respects TransEDOS's extended frontend files.
2. Keep the existing Setup CSV Python tests and browser smoke in CI. Add browser tests for partial/cross-module imports, rollback, role/config revision races and active Auto states; the current tests cover happy paths only.
3. Add payload contract tests that compare manual BO/TransBO requests with their Auto adapters after a Setup CSV import. Include numerical, disallowed and objective constraints plus BO workload settings.
4. Add integrated simulated-IMEX tests for all four module-toggle combinations, both Auto starts, job cancellation, distinct terminal states and setup/data separation. Test the packaged single-worker build with the supported folder transport.

## 6. Completion criteria

CEDOS is ready when the aligned EDOS core and Setup CSV parser remain unchanged; the new BO background workflow is preserved; each manual module can exchange Setup CSV without altering experimental rows; unsupported cross-module conversions and prior conflicts are explicit; imports cannot mutate an active batch or apply stale responses; Auto uses the selected optimizer and current setup; BO/TransBO/SA share the intended dataset roles; every hardware result matches exactly one proposal row; and all gates above pass. Record deliberate deviations from the three source products in `CEDOS_ALIGNMENT.md` and document the precise scope of Setup CSV in the CEDOS guide.
