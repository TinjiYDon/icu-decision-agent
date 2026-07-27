"""Optional local MLflow tracking (file store). No-op if mlflow not installed."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def tracking_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    db = (_root() / "mlflow.db").resolve()
    return f"sqlite:///{db.as_posix()}"


def log_run(experiment: str, run_name: str, params: dict[str, Any], metrics: dict[str, Any]) -> str | None:
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        flat_p = {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v) for k, v in params.items()}
        mlflow.log_params({k: flat_p[k] for k in list(flat_p)[:50]})
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v == v:
                mlflow.log_metric(k, float(v))
        return run.info.run_id
