# Riot broad-cohort modeling — completed 2026-10-28

Both full models, validation and final-test evaluation, curves and paired recommendations are complete. The requested pause/resume history is preserved as `pause_handoff.md`, `paused_state.json` and `paused_executed.ipynb` in the output directory.

- Notebook: `hlr_model/broad_rank_modeling.ipynb` (fully executed; full profile enabled).
- Output: `hlr_model/data/analyses/broad_rank_mastery_v1/20261028T193830350096Z_5a469939/`.
- Report: notebook section 11 and versioned `execution_summary.json`.
- Both models have four completed chains and passed the unchanged full diagnostic gate. Original settings/specifications were retained; exact Pyro 1.9.2 was used.
- Usable train/validation/test rows: 3,907 / 852 / 928.
- Validation comparison/decision was recorded before the final test. Shared remained preferred; grouped did not improve held-out log loss or Brier. Personalized paired preferences were mostly uncertain; sparse support is flagged.
- No fitting, evaluation or reporting work is pending. The modeling tests passed all 18 tests. EDA and original outputs are preserved.

To reproduce the executed notebook from the repository root with the existing project environment:

```sh
./.venv-modeling/bin/python -u hlr_model/data/analyses/broad_rank_mastery_v1/20261028T193830350096Z_5a469939/execute_notebook.py
```

The runner needs local Jupyter loopback sockets. Keep inputs/helper/configuration unchanged to reuse preparation and all completed chain caches. Completed validation/test predictions and metrics are checksummed and reused, so reruns do not rescore test. The written validation decision is frozen. See `requirements-modeling.txt` and the versioned environment/configuration manifests for dependencies and full sampling settings. The earlier `pending_reporting_cell.py` was incorporated into notebook section 9 and executed; its filename is historical.
