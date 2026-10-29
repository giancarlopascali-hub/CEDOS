---
title: Experimental Design, Optimization and Statistics, with Transfer Learning
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
tags:
  - bayesian-optimization
  - transfer-learning
  - chemistry
  - experimental-design
---

## EDOS foundation alignment

TransEDOS remains an independent application repository. Its standard EDOS
foundation is pinned in `alignment/edos.lock`; exact shared files are verified
by `alignment/check_alignment.py`, while transfer-learning modules remain
TransEDOS-owned.

Every push to `main` must pass both shared and transfer-specific tests before a
clean bundle is deployed to
[realgcp/TransEDOS](https://huggingface.co/spaces/realgcp/TransEDOS). The live
application exposes `/health` and `/version` for source-revision traceability.

# Experimental Design, Optimization and Statistics, with Transfer Learning

TransEDOS extends the EDOS Bayesian Optimization platform with **Transfer Learning**, enabling experimental scientists and synthetic chemists to transfer knowledge across reaction campaigns, substrates, solvents, and condition sets.

## Setup CSV files

DoE, BO, and SA each have **Download setup CSV** and **Upload setup CSV** controls. TransBO has equivalent controls for **Features & Objectives**; its **Prior Targets** tab is outside this exchange. Each download saves the current controls and applicable constraints in one module-specific file. All four files share a versioned CSV schema and can be uploaded to another module; unsupported settings are reported and skipped.

Uploading a setup CSV changes Setup controls only. It does not add, remove, or edit rows or columns in the experimental dataset. BO, SA, and TransBO require a loaded dataset with matching feature and objective column names. Configured ranges and categorical choices may extend beyond values already observed in that dataset.

---

## Key Features

1. **Prior Column Role (`P`):**
   - Main data table features a 3-state column role toggle: **Feature (`F`, Blue)** ➔ **Objective (`O`, Amber)** ➔ **Prior (`P`, Violet)**.
   - Designate historical context variables (e.g. Substrate SMILES, Solvent Dielectric, Temperature) as priors without forcing them into the search space.

2. **Automated Prior Type Detection & Metrics:**
   - **SMILES (Molecules):** Uses 2048-bit Morgan circular fingerprints (radius 2) and a user-selected minimum Tanimoto similarity. Molecules below the cutoff are excluded and accepted analogues receive smoothly increasing relevance toward an exact match.
   - **Numerical:** Uses a user-selected tolerance in the column's native units. Rows outside `target ± tolerance` are excluded, while accepted rows receive smoothly increasing relevance toward the target.
   - **Categorical:** Exact match or unseen level detection with cold-start alerting.

3. **Multi-Prior Combination Modes:**
   - **Multiply (Strict):** All prior targets must simultaneously match for high weight.
   - **Average (Flexible):** Partial matches contribute with customizable per-prior importance weights ($1-100\%$).

4. **Heteroscedastic Noise GP Modeling:**
   - BoTorch `SingleTaskGP` with observation noise variance inversely proportional to combined prior weight:
     $$\sigma_{obs}^2 = \frac{\sigma_{base}^2}{w_i + \epsilon}$$
   - Dissimilar historical points receive high noise variance, while matching points guide the surrogate posterior.

5. **Transfer Diagnostics & Visualization:**
   - Summary diagnostics displaying Prior-eligible rows, each Prior's acceptance rule, surrogate engine status, and RDKit availability.
   - Interactive Plotly distribution bar chart of historical row weights.
   - Transfer-conditioned 2D contour and 3D response surfaces.

6. **All Core EDOS Modules Intact:**
   - Full support for Design of Experiments (**DoE**), Standard Bayesian Optimization (**BO**), and Statistical Analysis (**SA**).
   - Standard BO follows the EDOS reference implementation: observable process-isolated jobs, bounded double-precision GP fitting, selectable multi-objective methods, and representative-row protection for large datasets. The server seeks the configured fidelity target and reports any best-safe-subset fallback explicitly.

---

## Local Development Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: `rdkit-pypi` is included in `requirements.txt`)*

### 2. Run the Server
```bash
python app.py
```
- By default, TransEDOS starts on **http://127.0.0.1:5001** and automatically opens your web browser.

---

## Hugging Face Spaces Deployment

TransEDOS is 100% pre-configured for Docker deployment on **Hugging Face Spaces**:

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose **Docker** as the Space SDK (Blank template).
3. Push or upload the contents of the `TransEDOS` folder:
   - `Dockerfile`
   - `requirements.txt`
   - `app.py`
   - `src/`
   - `static/`
   - `templates/`
4. The application automatically detects the Hugging Face environment (`$SPACE_ID` / `$PORT=7860`), binds to `0.0.0.0:7860`, and serves via `gunicorn`.

---

## Architecture Overview

```
TransEDOS/
├── app.py                     # Entry point (port 5001 dev / 7860 HF Spaces)
├── Dockerfile                 # Production container spec
├── requirements.txt           # Python dependencies + rdkit-pypi
├── src/
│   ├── api/
│   │   ├── routes.py          # Flask REST API endpoints
│   │   ├── jobs.py            # Shared background-job state and capacity
│   │   └── bo_worker.py       # Isolated BO process supervision and timeout
│   └── core/
│       ├── prior_utils.py     # SMILES fingerprints & weight computation
│       ├── transfer_bo.py     # Transfer Learning BO engine
│       ├── bo.py              # Standard BO engine
│       ├── bo_policy.py       # BO method and resource policy
│       ├── bo_sampling.py     # Representative-row selection and fidelity
│       ├── doe.py             # Design of Experiments engine
│       ├── sa.py              # Statistical Analysis engine
│       └── constraints.py     # Numerical & categorical constraints
├── static/
│   ├── css/style.css          # Design system + TransBO violet accents
│   └── js/
│       ├── state.js           # Global reactive state
│       ├── main.js            # App controller & 3-role column toggling
│       └── modules/
│           ├── tbo.js         # TransBO 4-tab UI & diagnostics
│           ├── bo.js          # Standard BO module
│           ├── bo_workload.js # Browser-side workload preview
│           ├── doe.js         # DoE module
│           └── sa.js          # Statistical Analysis module
└── templates/
    ├── index.html             # Single-page application
    └── guide.html             # User manual including TransBO guide
```

## Licence and citation

TransEDOS is licensed under the [Apache License 2.0](LICENSE). Copyright and
attribution information is provided in [NOTICE](NOTICE), and third-party
components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

If you use TransEDOS in research, please see [CITATION.md](CITATION.md).
Definitive publication and DOI details are coming soon.
