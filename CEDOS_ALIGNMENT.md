# CEDOS local source alignment

CEDOS is a new app in this project folder. Its base was copied from committed TransEDOS revision `0af60b40885e3ad78d87c8157625677e8762a5ba`, aligned to EDOS foundation `6af1c9e5ea48ac4ca08799c83d5ab72c5748a7b7`. The Auto-owned layer comes from AutoEDOS revision `4aabb13bbbc474092d0774e07f660667334f4ec1`. No original EDOS-family checkout is modified by this local build.

The numerical engines for DoE, BO, SA and TransBO, plus the version 1 Setup CSV parser, retain the source logic. `alignment/cedos_core.sha256` records normalized source-content hashes. Run `python alignment/check_cedos_core.py` to verify those nine files. The normalized hashes match the committed TransEDOS worktree; normalization only removes Windows/Linux line-ending differences.

## CEDOS-owned changes

| Area | Local change |
|---|---|
| Application entry | Register one Auto bridge beside the TransEDOS API; use CEDOS identity and a local port. |
| Module shell | Add Auto and TransBO visibility toggles, one active optimizer, and revision-based invalidation of stale suggestions. Turning TransBO off converts P columns to F for BO; turning it back on restores untouched Prior roles and targets. |
| Setup CSV browser flow | Keep the common parser and schema; add preview, cross-module/prior guards, campaign timing guards, and cache invalidation. |
| Auto orchestration | Support DoE start or direct active-optimizer start; dispatch BO/TransBO through their existing background-job APIs; merge identified measured results. |
| IMEX exchange | Add a CEDOS-only folder client with proposal acknowledgment, row IDs, result retry and deduplication. The original AutoEDOS companion is not changed. |
| BO and TransBO UI | Offer EI/LCB consistently. The CEDOS transfer route translates LCB to the equivalent source UCB utility path when Priors are active; no-Prior standard BO keeps its LCB policy. Disclose fallback/exhaustion; reject unsupported TransBO feature constraints and duplicate-avoidance requests. Numerical core files remain unchanged. |
| BO visualization | Clear completed progress graphics before rendering contour and 3D plots, and keep visualization selections out of setup invalidation. |
| Local operation | Add a Python environment, launcher, source metadata and CEDOS guide. HF and standalone packaging follow local acceptance. |

`TRANSEDOS_SOURCE_README.md` preserves the original base app's README for reference. It describes TransEDOS, while `README.md` describes this CEDOS build.
