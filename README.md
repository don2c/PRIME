# PRIME-Ring Artifact (USENIX Security submission)

This repository contains the artifacts for **PRIME-Ring** (Policy-hidden Ring with Identifiable Members and Epoch security).
It reproduces the paper’s tables from the released datasets, generators, and evaluation scripts.

## Start Here (For Evaluators)

### Option A: Docker (most consistent)
```bash
docker build -t primering-ae .
docker run --rm -it primering-ae bash -lc "bash scripts/run_all.sh"
```

### Option B: Native Python
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
bash scripts/run_all.sh
```

### What gets reproduced
The one-command script regenerates all paper tables into `./results/`:

- **Table 2–3 (AOV-hiding evaluation):** `results/aov_hiding/`  
  Source bundle: `artifacts/prime_ring_results/`

- **Table 4–5 (Certified release / inference-bounded anonymity):** `results/cert_release/`  
  Source bundle: `artifacts/prime_ring_cert_release_artifact/`

- **Table 6–7 (Ring synthesis under sparsity and churn):** `results/sparsity_churn/`  
  Source bundle: `artifacts/prime_ring_sparsity_churn_artifact/`

- **Table 8 (Constant-shape enforcement):** `results/shape/`  
  Source bundle: `artifacts/prime_ring_shape_artifact/`

- **Table 9 (Comparison matrix):** `results/comparison_matrix/`  
  Source bundle: `artifacts/prime_ring_comparison_artifact_final/`

- **Table 10 (Overhead and scalability):** `results/overhead_by_scheme/`  
  Source bundle: `artifacts/prime_ring_overhead_scheme_artifact_v4/`

- **Table 11 (Governed opening and auditability):** `results/governed_opening/`  
  Source bundle: `artifacts/prime_ring_governed_opening_artifact/`

- **Table 12 (Epoch freshness and forward security):** `results/epoch_fs/`  
  Source bundle: `artifacts/prime_ring_epoch_fs_artifact/`


## What to download

For Phase-2 discussion you may use GitHub, but for Phase-1/Phase-2 final submission you must provide a permanent archive (e.g., Zenodo).
- `PRIME-Ring` source code and evaluation scripts
- Small CSV datasets used in the paper (`./data/`)
- Per-table, self-contained bundles under `./artifacts/`

## Repository layout

- `data/`  
  CSV inputs used across evaluations (e-health access log; OSN Twitter/Facebook action logs, user tables, and edge lists).

- `scripts/run_all.sh`  
  One-command regeneration of all paper tables (CSV + LaTeX).

- `artifacts/`  
  Extracted, self-contained bundles (each has its own `README.md`):
  - `prime_ring_cert_release_artifact/` (Certified release / inference-bounded anonymity)
  - `prime_ring_sparsity_churn_artifact/` (Ring synthesis under sparsity and churn)
  - `prime_ring_shape_artifact/` (Constant-shape enforcement and side-channel suppression)
  - `prime_ring_governed_opening_artifact/` (Governed opening and auditability)
  - `prime_ring_epoch_fs_artifact/` (Epoch freshness and forward security)
  - `prime_ring_overhead_scheme_artifact_v4/` (Overhead + scaling vs existing schemes)
  - `prime_ring_comparison_artifact_final/` (Comparison matrix table)

## System requirements

Tested with Python 3.10+ on Linux/macOS.
No GPU is required.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Quick start (reproduce all tables)

```bash
bash scripts/run_all.sh
```

Outputs are written to `./results/` as:
- CSV summaries used to populate tables
- LaTeX tables (`.tex`) matching the paper format

## Reproducing a single table

Each table bundle under `./artifacts/` can be run independently. Example:

```bash
python artifacts/prime_ring_epoch_fs_artifact/prime_ring_epoch_fs_artifact/prime_ring_eval_epoch_forward_security.py \
  --data_dir ./data \
  --out_dir ./results/epoch_fs \
  --seed 123
```

## Datasets

The included CSV files are small, review-friendly slices meant to support table regeneration and sanity checks.
They are not intended to be a full public release of any private operational data. The e-health log is de-identified.

## Determinism

All evaluation scripts accept `--seed`. Table values in the paper are computed as mean ± std over repeated runs; the default scripts regenerate the same aggregation protocol used for the LaTeX tables.

## Expected outputs

After running `scripts/run_all.sh`, you should see:

- `results/aov_hiding/` (AOV hiding tables)
- `results/cert_release/` (certificate and tightening outputs)
- `results/sparsity_churn/` (sparsity/churn curves + tables)
- `results/shape/` (length/time envelope checks)
- `results/governed_opening/` (open success/failure + audit completeness)
- `results/epoch_fs/` (fresh accept, stale reject, past-epoch tests)
- `results/overhead_by_scheme/` (overhead + scaling tables)
- `results/comparison_matrix/` (comparison table)

## License and third-party material

This artifact includes only the code and datasets needed to reproduce paper results.
If you add larger upstream datasets (e.g., SNAP traces), follow their licenses and citation requirements.

## Citation

If you use this artifact, cite the PRIME-Ring paper (camera-ready bib entry to be added).

## Artifact appendix

A USENIX-style artifact appendix template is provided in `artifact_appendix/`. Compile `artifact_appendix/artifact_appendix.tex` and upload the resulting PDF in HotCRP for Phase-2.


## AEC checklist

See `AEC_CHECKLIST.md` for a reviewer-style Phase-1/Phase-2 checklist.
