#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT}/data"
OUT_DIR="${ROOT}/results"
mkdir -p "${OUT_DIR}"

echo "[1/8] AOV-hiding tables (Table 2-3)..."
python "${ROOT}/Table2-3-PRIME AOV-hiding evaluation/prime_ring_eval_aov_hiding_v4.py" \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}/aov_hiding" \
  --seed 123

echo "[2/8] Certified release tables (Table 4-5)..."
python "${ROOT}/artifacts/prime_ring_cert_release_artifact/prime_ring_cert_release_artifact/prime_ring_eval_certified_release.py" \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}/cert_release" \
  --seed 123 \
  --runs_per_domain 20 \
  --max_osn_users 2000

echo "[3/8] Sparsity/churn tables (Table 6-7)..."
python "${ROOT}/artifacts/prime_ring_sparsity_churn_artifact/prime_ring_artifact_sparsity_churn/code/prime_ring_eval_sparsity_churn.py" \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}/sparsity_churn" \
  --seed 123 \
  --runs_per_setting 5

echo "[4/8] Constant-shape tables (Table 8)..."
PYTHONPATH="${ROOT}/artifacts/prime_ring_shape_artifact/prime_ring_shape_artifact:${PYTHONPATH:-}" python -m prime_ring_shape.run \
  --data-dir "${DATA_DIR}" \
  --out-dir "${OUT_DIR}/shape" \
  --seed 123 \
  --runs 50 \
  --tau 0.02 \
  --Hstar 4.0 \
  --k-support 16 \
  --delta-n-max 64 \
  --L-bytes 4096 \
  --Theta-ms 8.0 \
  --delta-L 8 \
  --delta-Theta-ms 0.2
PYTHONPATH="${ROOT}/artifacts/prime_ring_shape_artifact/prime_ring_shape_artifact:${PYTHONPATH:-}" python -m prime_ring_shape.render_latex \
  --summary-csv "${OUT_DIR}/shape/shape_summary.csv" \
  --metadata-json "${OUT_DIR}/shape/metadata.json" \
  --out-tex "${OUT_DIR}/shape/table_shape_summary.tex" \
  --label tab:prime_ring_shape

echo "[5/8] Governed opening tables (Table 11)..."
python "${ROOT}/artifacts/prime_ring_governed_opening_artifact/prime_ring_eval_governed_opening.py" \
  --out_dir "${OUT_DIR}/governed_opening" \
  --seed 123

echo "[6/8] Epoch freshness / forward security tables (Table 12)..."
python "${ROOT}/artifacts/prime_ring_epoch_fs_artifact/prime_ring_epoch_fs_artifact/prime_ring_eval_epoch_forward_security.py" \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}/epoch_fs" \
  --seed 123

echo "[7/8] Overhead/scaling by scheme (Table 10)..."
python "${ROOT}/artifacts/prime_ring_overhead_scheme_artifact_v4/gen_overhead_by_scheme.py" \
  --out_dir "${OUT_DIR}/overhead_by_scheme" \
  --seed 123

echo "[8/8] Comparison matrix (Table 9)..."
python "${ROOT}/artifacts/prime_ring_comparison_artifact_final/prime_ring_comparison_artifact/generate_tables.py" \
  --spec "${ROOT}/artifacts/prime_ring_comparison_artifact_final/prime_ring_comparison_artifact/systems.yaml" \
  --outdir "${OUT_DIR}/comparison_matrix"

echo "Done."
