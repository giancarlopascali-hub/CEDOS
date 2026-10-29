# CEDOS local build status

**Date:** 27 October 2026  
**Stage:** Running locally for user testing at `http://127.0.0.1:5002`. Hugging Face Space deployment and standalone packaging are pending.

## Implemented

- One CEDOS app in this project folder, based on the committed TransEDOS shell and the aligned EDOS numerical foundation. The original EDOS, TransEDOS and AutoEDOS apps were not edited.
- DoE, BO and SA appear initially. The TransBO toggle replaces BO as the active optimizer; the Auto toggle reveals Auto. Turning TransBO off converts P columns to F with BO/SA feature settings; turning it back on restores untouched Prior roles and targets. The main dataset is shared with SA.
- Setup CSV download/import for DoE, BO, TransBO Features & Objectives, and SA. The browser previews cross-module changes, protects Prior Targets, checks revision/campaign state and does not edit experimental rows on import.
- Auto starts from DoE or from the active BO/TransBO optimizer. Its proposal and result flow uses campaign, proposal and row IDs, a folder IMEX companion, measured-result stopping, cancellation and reload recovery action.
- TransBO keeps its source transfer-learning engine. CEDOS shows the no-Prior standard-BO fallback and exhausted state, while blocking settings that the current TransBO GP path cannot enforce reliably.
- The Quick Guide covers the CEDOS TransBO toggle and workflow. Auto-only navigation and guide controls are visible only when Auto is enabled.
- BO and TransBO show EI/LCB acquisition. BO contour and 3D surface plots clear their completed progress wheel and keep their controls visible.

## Verification performed

| Check | Result |
|---|---|
| Full Python suite | 307 passed, 18 dependency/runtime warnings |
| Auto JavaScript contract tests | 6 passed |
| CEDOS shell state test | Passed |
| Source-content alignment | 9 numerical/parser files match committed source after line-ending normalization |
| Manual browser DoE → BO → SA | Four DoE suggestions; measured values committed into shared dataset; BO handoff and SA view passed |
| Real manual BO | Background job returned a suggestion without browser errors |
| Real manual TransBO | GP job returned a suggestion using 3 eligible Prior rows from a 6-row dataset, without browser errors |
| Setup CSV browser exchange | BO export to SA import showed preview and preserved all dataset rows; attempted BO import that would change a Prior column was rejected visibly |
| Auto browser cycles | Direct BO, direct TransBO, DoE → TransBO review/optimization, and reload recovery passed with mocked optimizer jobs and the real bridge |
| Auto-only controls | Hidden when Auto is off, visible when on, hidden again when off |
| Reported role regression | Browser test passed P→F on TransBO off, reversible restoration when untouched, and retention of explicit BO edits |
| Reported acquisition regression | Focused API tests passed for LCB transfer mapping and no-Prior BO fallback; BO/TransBO menus show EI/LCB |
| Real TransBO LCB run | A six-row dataset with three eligible Prior rows returned a fixed-noise GP suggestion; no browser alert or script error |
| Reported surface regression | Browser test passed for contour and 3D plots: completed spinner removed and controls remain usable |
| Local server | `/health`, `/version`, `/guide` and app page respond on `127.0.0.1:5002` |

The browser Auto tests mock the optimizer job responses to focus on orchestration and the identified bridge. Real BO and TransBO manual jobs were also run separately against the local server. The folder client's proposal/result/acknowledgment path has targeted Python coverage; no physical hardware is connected in this stage.

## Known local limits and next release phases

- Auto bridge state is in memory in one process. A server restart clears the campaign; the Auto panel can release an interrupted proposal after a browser reload, but a server restart requires a fresh campaign.
- The CEDOS IMEX companion supports folder exchange. Serial, network and USB hardware receive modes are not implemented here. The older AutoEDOS companion uses a different proposal/result protocol.
- TransBO numerical feature constraints and fitted-GP duplicate avoidance are disabled because the source engine cannot enforce them throughout that path. Objective constraints remain supported.
- HF Space deployment, clean Docker dependency review and a standalone executable build are intentionally deferred until local testing is accepted.

Use `run_cedos.bat` to start the app if it is not already running, `stop_cedos.bat` to stop it, and `README.md` for the local user workflow.
