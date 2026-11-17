"""Central configuration loader.

Every module reads its numbers from ``params.yaml`` through this helper so the
training pipeline, the API and the tests can never silently disagree about
sample rate, window size or the alert threshold.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_params(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Return the parsed params.yaml (cached). ``$PARAMS_PATH`` overrides the default."""
    resolved = Path(path or os.environ.get("PARAMS_PATH", DEFAULT_PARAMS_PATH)).resolve()
    return _load(str(resolved))


def tracking_uri(params: dict[str, Any] | None = None) -> str:
    """MLflow tracking URI: environment variable wins over params.yaml."""
    params = params or load_params()
    return os.environ.get("MLFLOW_TRACKING_URI", params["mlflow"]["tracking_uri"])
