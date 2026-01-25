# PRIME-Ring Comparison Artifact (Tables)

This folder contains a small, reproducible generator for the LaTeX comparison table used in the paper's comparison section.

## Contents
- `systems.yaml`: feature claims (yes/no/partial) per system
- `generate_tables.py`: emits `table_compare_systems.tex`
- `out/`: generated LaTeX table + short system notes

## Reproduce
```bash
python generate_tables.py --spec systems.yaml --outdir out
```

If your LaTeX template does not define `\checkmark` or `\texttimes`, replace them with `Y`/`N` or add packages as needed.
