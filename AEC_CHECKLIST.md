# AEC Checklist (USENIX Security '26)

This checklist is written for quick, reviewer-style verification.

## Phase-1: Artifacts Available (mandatory)

- [ ] Permanent archive link prepared (e.g., Zenodo/FigShare/Dryad/Software Heritage) and points to a specific version.
- [x] Repository contains a top-level README with clear usage instructions.
- [x] Repository contains a license (`LICENSE`).
- [x] Repository contains citation metadata (`CITATION.cff`).
- [x] Artifact does not require privileged access, special hardware, or network interaction.
- [x] Data included is documented (`README.md` + per-table bundle READMEs).

**Risk if unchecked:** Phase-1 failure if the permanent archive link is missing at submission time.

## Phase-2: Artifacts Functional / Results Reproducible (optional)

### Packaging and entry point
- [x] Single entry point exists: `bash scripts/run_all.sh`.
- [x] A docker-based entry point exists (recommended): see README “Start Here (For Evaluators)”.
- [x] Outputs are written under `./results/` with stable file names.

### Artifact appendix
- [x] Artifact appendix PDF exists: `ARTIFACT_APPENDIX.pdf` (≤3 pages).
- [x] Appendix includes hardware/software requirements.
- [x] Appendix includes claim → command → output mapping (Table 2–12).
- [x] Appendix explains how to compare regenerated outputs with the paper.

### Determinism and comparability
- [x] Scripts accept `--seed` and use fixed seeds by default.
- [x] Appendix notes acceptable minor numerical drift if dependency versions change.

### Data and scripts
- [x] All scripts run without absolute paths (no `/mnt/...` hard-coding).
- [x] Dataset slices needed for table regeneration exist under `./data/`.
- [x] Per-table bundles under `./artifacts/` are runnable independently.

## Pass/Fail risk assessment (excluding permanent archive)

**Low risk** for Phase-2 functionality if evaluators use Docker, and **low-to-moderate risk** for native Python due to host differences.
The remaining high-stakes item is the Phase-1 permanent archive requirement.
