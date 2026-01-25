# PRIME-Ring Artifact: Ring Synthesis under Sparsity and Churn

This artifact reproduces the **ring synthesis under sparsity and churn** evaluation for PRIME-Ring.

## Inputs
The script reads the already-prepared PRIME-Ring datasets in this workspace:
- `prime_ring_ehealth_access_log.csv`
- `prime_ring_ehealth_staff_graph_edges.csv`
- `prime_ring_osn_twitter_users.csv`, `prime_ring_osn_twitter_edges.csv`, `prime_ring_osn_twitter_actions.csv`
- `prime_ring_osn_facebook_users.csv`, `prime_ring_osn_facebook_edges.csv`, `prime_ring_osn_facebook_actions.csv`

## Run
From the artifact root:

```bash
python code/prime_ring_eval_sparsity_churn.py --out_dir outputs --seed 0 --runs_per_setting 5
```

## Outputs
- `outputs/results_raw.csv`: per-run metrics for each (domain, tau, H*, d0, churn)
- `outputs/results_agg.csv`: aggregated metrics per setting
- `outputs/table_sparsity_churn_summary.tex`: main LaTeX table
- `outputs/table_sparsity_churn_hardcase.tex`: hard-case / failure-mode LaTeX table

## Notes
- The evaluation reports: $H_{\min}$, $\hat{\tau}$, success rate (SR), $|S'|$, and failure modes under empty pools.
- Randomness is controlled via `--seed`.
