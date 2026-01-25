# PRIME-Ring overhead comparison artifact (by scheme)

This bundle reproduces two LaTeX tables used in the paper draft:
- `table_overhead_by_scheme.tex`
- `table_scaling_by_scheme.tex`

## Contents
- `gen_overhead_by_scheme.py`: generator script
- `overhead_by_scheme.csv`, `scaling_by_scheme.csv`: datasets backing the tables
- `table_overhead_by_scheme.tex`, `table_scaling_by_scheme.tex`: LaTeX outputs

## Reproduce
```bash
python gen_overhead_by_scheme.py --out_dir out --seed 7
```

The LaTeX tables in `out/` should match the ones shipped in this artifact.
