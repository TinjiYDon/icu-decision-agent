# Independent decision experiment snapshot

This directory belongs only to `icu-decision-agent` and runs without the sibling repository.
The original shared workspace experiment sources have been copied locally, not linked.

## Scope and status

Archived November B0–B7 / LARE comparisons (decision) or eight-policy simulator (scheduling), as applicable to this repository only. These are existing experimental algorithms, not newly completed V3 work.
Decision results retain historical 70/10/20 splits and charttime-based availability; they are NOT causally sealed V3 60/20/20 results. Scheduling archives may contain multiple method subsets: use the methods and n_runs recorded in each JSON, not a blanket eight-method claim.

## Reproduction

Use this repository's Python environment from the repository root. Set MIMIC_DSN and the project-specific ICU_DECISION_DSN or ICU_SCHEDULING_DSN. Set MIMIC_CSV_ROOT to the licensed MIMIC CSV directory for ETL. No credentials are committed.

Data preparation: `python experiments/etl/build_tables.py --only all`. ETL is a local historical snapshot; read its database/table assumptions before running. It does not import either sibling project.

Run in order: `build_base.py`, `build_features.py`, then `train_eval.py --suite full` under `experiments/decision/src/`. See RUNLOG.md for supplemental feature and finalization steps. `verify.py` requires local prediction parquet files, intentionally excluded from Git.
The separately named lare_controlled_validation.py was not found in the workspace and is not claimed delivered.

## Published material

Only source, historical aggregate tables, reports and figures are included. Patient-level data, predictions, split IDs and trajectories are excluded. Original RUNLOG/SECTION text is retained as historical evidence; this README controls current release claims.
