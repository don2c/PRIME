# PRIME-Ring governed opening and auditability (artifact)

This artifact evaluates PRIME-Ring accountability on the synthetic e-health and OSN transcripts in this workspace.

## Inputs (already present)
- `/mnt/data/prime_ring_ehealth_access_log.csv`
- `/mnt/data/prime_ring_osn_twitter_actions.csv`
- `/mnt/data/prime_ring_osn_twitter_users.csv`
- `/mnt/data/prime_ring_osn_facebook_actions.csv`
- `/mnt/data/prime_ring_osn_facebook_users.csv`

## Run
```bash
python prime_ring_eval_governed_opening.py --out_dir prime_ring_governed_opening_results --seed 0
```

Optional flags:
- `--max_rows` (default 50000)
- `--open_frac` (default 0.10)
- `--corrupt_frac` (default 0.05)

## Outputs
- `prime_ring_governed_opening_results/table_governed_opening.tex`
- `prime_ring_governed_opening_results/results.csv`
- `prime_ring_governed_opening_results/audit_log.csv`
- `prime_ring_governed_opening_results/transcripts_sample.csv` (open-payload sample)
