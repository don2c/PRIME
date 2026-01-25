# PRIME-Ring constant-shape evaluation artifact

This artifact reproduces the evaluation for **constant-shape enforcement and side-channel suppression**.

It simulates the verifier-side enforcement described by the PRIME-Ring writeup:
- A fixed plan `(L, Θ)` determines a transcript byte-length envelope and a target verification-time envelope.
- The planner computes padding bytes `b = max(0, L - L_S')` and schedules dummy steps `d = max(0, Θ - θ_S')`.
- Verification rejects if `|L_S - L| > δL` or `|θ_S - Θ| > δΘ`.

## What is measured
For each dataset and churn level, the harness records per-run:
- `pad_bytes` (b)
- `drift_ms = |θ_final - Θ|` (timing drift)
- rejection rate and rejection reason (`length` or `time`)

## Data
The `data/` folder contains the CSV files you provided:
- `prime_ring_ehealth_access_log.csv`
- `prime_ring_osn_twitter_users.csv`
- `prime_ring_osn_facebook_users.csv`

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 -m prime_ring_shape.run \
  --data-dir data \
  --out-dir results \
  --seed 7 \
  --runs 400 \
  --tau 0.02 \
  --Hstar 4.0 \
  --k-support 16 \
  --delta-n-max 64 \
  --L-bytes 4096 \
  --Theta-ms 8.0 \
  --delta-L 8 \
  --delta-Theta-ms 0.2
```

Outputs:
- `results/shape_runs.csv` (all runs)
- `results/shape_summary.csv` (aggregated table)
- `results/SUMMARY.md`
- `results/metadata.json`

## Generate the LaTeX table

After running the experiment, generate a drop-in LaTeX table:

```bash
python3 -m prime_ring_shape.render_latex \
  --summary-csv results/shape_summary.csv \
  --metadata-json results/metadata.json \
  --out-tex results/shape_summary_table.tex \
  --label tab:prime_ring_shape
```

This writes `results/shape_summary_table.tex`.
