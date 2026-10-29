---
title: CEDOS
emoji: 🧪
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# CEDOS — Complete Experimental Design, Optimization and Statistics

CEDOS combines Design of Experiments (DoE), Bayesian Optimization (BO), Transfer Bayesian Optimization (TransBO), Statistical Analysis (SA), and an optional Auto campaign. This is a **new app** built from the committed EDOS-family sources listed in [`build-info.json`](build-info.json). The original app folders have not been edited.

## Hosted Space

The public app runs at [Hugging Face Spaces](https://huggingface.co/spaces/realgcp/CEDOS). DoE, BO, TransBO and SA use the browser dataset and are available without an access code. The Auto exchange requires an owner-provided access code. It is stored only for the current browser tab and is never included in this repository.

Auto still uses the CEDOS folder-exchange companion on your own computer. In Auto Step 1, enter the code and choose **Use token for this tab**. Save the same code in the private, ignored file `.local/cedos_auto_token.txt`, then start the companion with:

```powershell
.\.venv\Scripts\python.exe imex\cedos_imex.py --server https://realgcp-cedos.hf.space --token-file .local\cedos_auto_token.txt
```

The code must match the Space's `CEDOS_AUTO_TOKEN` secret. A Space restart clears active Auto campaigns and background jobs, so start a fresh campaign after a restart.

Every push to the GitHub repository's `main` branch runs the automated checks, publishes a clean runtime bundle to this Space, and verifies its `/health` and `/version` endpoints. The GitHub Actions `HF_TOKEN` secret authorizes this sync; the Space's `CEDOS_AUTO_TOKEN` secret is only for the Auto exchange. Neither secret belongs in source code.

## Start CEDOS locally

On this computer, double-click `run_cedos.bat`. It uses the prepared `.venv` environment and opens `http://127.0.0.1:5002`. Port 5000 is used by another local program on this computer, so the launcher selects 5002. Keep the server window open while testing; closing it stops the app. `stop_cedos.bat` also stops the local server, including the preview instance started for you now.

From PowerShell, the equivalent is:

```powershell
$env:CEDOS_PORT = '5002'
.\.venv\Scripts\python.exe app.py
```

To rebuild the local Python environment if needed:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_local.ps1
```

The setup script reuses scientific packages already installed on this computer and installs the pinned `requirements.txt`. Add `-CleanEnvironment` to create a fully isolated environment when starting from an empty `.venv`; that option downloads the large PyTorch stack. Python 3.11 is the tested interpreter.

The server exposes `/health` and `/version` for a quick startup check. The local launcher binds to localhost.

## Manual workflow

DoE, BO and SA appear initially. The TransBO toggle reveals TransBO and hides BO; turning it off restores BO. The Auto toggle reveals the campaign panel. DoE passes proposed conditions to the currently selected optimizer. BO or TransBO and SA work from the same main dataset. When TransBO is turned off, its `Prior` (`P`) columns become usable BO `Feature` (`F`) columns. Turning TransBO on again restores their Prior roles and targets if those columns' BO roles and feature settings were not edited.

Each of DoE, BO, TransBO Features & Objectives, and SA can download and upload a **Setup CSV**. This file stores feature/objective settings and applicable constraints. It does not contain experimental dataset rows, measured results, TransBO Prior Targets, or optimizer tuning controls. Upload shows a preview and asks you to apply it. Cross-module conversions and retained settings are shown there. Dataset CSV and measured-results CSV are separate workflows.

BO and TransBO offer EI or LCB acquisition. TransBO keeps the source transfer-learning calculation; CEDOS translates its public LCB choice to the equivalent internal confidence bound on maximize-oriented utility. Its fitted GP path does not enforce numerical feature constraints or historical duplicate filtering. CEDOS disables those controls and rejects such TransBO requests instead of silently ignoring a rule. Standard BO supports both controls. A TransBO run with no Prior Target uses the source standard-BO fallback and says so in diagnostics. The app also reports an exhausted search instead of displaying an empty suggestion table.

## Auto workflow with the local folder exchange

Enable Auto, then start from DoE or from the active optimizer (BO or TransBO) with a loaded dataset and valid Setup. A DoE start pauses for optimizer review after its first measured batch. During an active batch, CEDOS protects the dispatched setup and data; Stop cancels the campaign. The optimizer choice is fixed for that campaign.

Auto proposals and results carry campaign, proposal and row IDs. For a local closed-loop test, start the folder exchange in a second terminal **after** starting the CEDOS server:

```powershell
.\.venv\Scripts\python.exe imex\cedos_imex.py --server http://127.0.0.1:5002
```

It creates `imex/proposals` and `imex/results`. When Auto proposes a batch, the exchange writes `Proposal_<campaign>_<proposal>_<hash>.csv` and a matching JSON manifest. A matching `Results_<same-stem>.csv.example` appears in `imex/results`. Fill the objective values in that example and save a copy without `.example`, as `Results_<same-stem>.csv`. The exchange validates and uploads it; an accepted file is renamed with a `SENT_` prefix. Keep the `row_id` values and proposal identity intact. The example is a template for **measured** results, not a Setup CSV.

The exchange supports folders only. The older AutoEDOS IMEX program uses a different, unidentified protocol and cannot safely serve this CEDOS Auto loop. A server restart loses the in-memory campaign; start a fresh Auto campaign after restarting. CEDOS runs one server process.

## Tests and source ownership

Run the Python checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The EDOS numerical engines and shared Setup CSV parser are copied without logic changes. CEDOS changes the application shell, navigation, Auto orchestration and exchange, Setup CSV browser behavior, and TransBO UI guards. See [`CEDOS_ALIGNMENT.md`](CEDOS_ALIGNMENT.md) for the source revisions and exact ownership notes. The implementation decisions and remaining deployment work are in [`CEDOS_IMPLEMENTATION_PLAN_v3.md`](CEDOS_IMPLEMENTATION_PLAN_v3.md).
