# PRIME-Ring: Epoch Freshness and Forward Security (Artifact)

This artifact reproduces the evaluation **Epoch freshness and forward security** on the PRIME-Ring synthetic datasets (e-health + OSN).

## What it checks

1. **Epoch freshness (stale replay rejection)**
   - Computes an epoch->statusID map (mode per epoch) as the "current snapshot".
   - Reports acceptance on:
     - valid (t, statusID)
     - stale statusID for the same epoch
     - stale epoch replay at epoch t+1 (no wrap-around)

2. **Forward security under simulated compromise**
   - Uses a one-way per-epoch key update and a lightweight signing model (HMAC-like).
   - Compromise happens at the final epoch; the attacker tries to:
     - forge past-epoch signatures
     - link two past-epoch transcripts to the same signer

## Inputs

The script reads the CSVs generated earlier in this workspace:

- `/mnt/data/prime_ring_ehealth_access_log.csv`
- `/mnt/data/prime_ring_osn_twitter_actions.csv`
- `/mnt/data/prime_ring_osn_twitter_users.csv`
- `/mnt/data/prime_ring_osn_facebook_actions.csv`
- `/mnt/data/prime_ring_osn_facebook_users.csv`

## Run

From the artifact directory:

```bash
python prime_ring_eval_epoch_forward_security.py --out_dir results --seed 0 --n_forge 2000 --n_pairs 5000
```

## Outputs

- `results/results.csv` (raw numbers)
- `results/table_epoch_freshness_fs.tex` (LaTeX table for the paper)

