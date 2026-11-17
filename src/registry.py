"""Thin helpers around the MLflow Model Registry (aliases @champion / @challenger)."""

from __future__ import annotations

import os
from pathlib import Path

CHAMPION = "champion"
CHALLENGER = "challenger"

# Quiet, offline-friendly MLflow client defaults (no telemetry pings, no progress bars).
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")


def client(tracking_uri: str):
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def version_by_alias(tracking_uri: str, model_name: str, alias: str):
    """Return the ModelVersion behind ``alias`` or ``None`` if it does not exist."""
    from mlflow.exceptions import MlflowException

    c = client(tracking_uri)
    try:
        return c.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def download_checkpoint(tracking_uri: str, model_name: str, alias: str, dst: str | Path) -> Path:
    """Download the registered model behind ``alias`` and return its checkpoint path.

    The pyfunc model stores the checkpoint under ``artifacts/<file>.pt``.
    """
    import mlflow

    client(tracking_uri)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}@{alias}", dst_path=str(dst)
    )
    matches = sorted(Path(local).glob("artifacts/*.pt")) or sorted(Path(local).rglob("*.pt"))
    if not matches:
        raise FileNotFoundError(f"No .pt checkpoint inside models:/{model_name}@{alias}")
    return matches[0]


def download_bundle(tracking_uri: str, model_name: str, alias: str, dst: str | Path) -> Path:
    """Download the registered bundle behind ``alias``; returns the folder with bundle.json."""
    import mlflow

    from src.bundle import find_bundle_dir

    client(tracking_uri)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}@{alias}", dst_path=str(dst)
    )
    return find_bundle_dir(local)


def set_alias(tracking_uri: str, model_name: str, alias: str, version: str | int) -> None:
    client(tracking_uri).set_registered_model_alias(model_name, alias, str(version))
