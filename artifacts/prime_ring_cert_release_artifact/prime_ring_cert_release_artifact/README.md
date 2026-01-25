# PRIME-Ring: Certified release and inference-bounded anonymity (Evaluation 2)

This artifact reproduces the **certified release** experiment for PRIME-Ring.
It measures whether attacker posteriors under the declared feature set and attack class stay below the release certificate, and reports:

- \(\hat{\tau}\): base certificate (product-prior posterior max)
- \(\hat{\tau}_{\mathrm{LLM}}\): tightened certificate (max over declared attack class)
- Violation rate: fraction of runs where an attacker exceeds the tightened bound
- Refusal rate: fraction of runs where release is infeasible under targets

## Inputs
The script expects these CSV files to exist in the same workspace:

- `/mnt/data/prime_ring_ehealth_access_log.csv`
- `/mnt/data/prime_ring_ehealth_staff_graph_edges.csv`
- `/mnt/data/abhrs_ehealth_nhdh_with_roles.csv`

- `/mnt/data/prime_ring_osn_twitter_actions.csv`
- `/mnt/data/prime_ring_osn_twitter_users.csv`
- `/mnt/data/prime_ring_osn_twitter_edges.csv`

- `/mnt/data/prime_ring_osn_facebook_actions.csv`
- `/mnt/data/prime_ring_osn_facebook_users.csv`
- `/mnt/data/prime_ring_osn_facebook_edges.csv`

## How to run

```bash
python prime_ring_eval_certified_release.py \
  --out_dir prime_ring_cert_release_results \
  --seed 0 \
  --runs_per_domain 80 \
  --max_osn_users 4000 \
  --tau_target 0.15 \
  --hmin_target 3.0 \
  --max_iters 3 \
  --n0 12 \
  --n_step 6 \
  --n_max 60
```

## Outputs
The output directory contains:

- `results_cert_release.csv` (per-run results)
- `table_cert_release_summary.tex` (main table for the paper)
- `table_cert_release_attacks.tex` (attack-class breakdown)
- `config.json` (all parameters)

## Notes on the declared feature set and attack class
- Feature set: ring membership \(S'\), attribute bucket signatures, and graph distance (truncated BFS). Plan/epoch fields are treated as fixed-shape metadata.
- Attack class \(\mathcal{H}\): a small family of priors (product prior + ablations) plus a lightweight action-propensity prior.
- Tightened certificate \(\hat{\tau}_{\mathrm{LLM}}\) is computed as the max posterior mass over \(\mathcal{H}\).

## Requirements
- Python 3.9+
- `numpy`, `pandas`

