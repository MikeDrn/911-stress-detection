"""911 Under-Threat Voice Stress Detection System."""

import os as _os

__version__ = "1.0.0"

# Quiet, offline-friendly MLflow defaults for every entry point (no telemetry
# pings to the internet, no progress bars in logs, no assistant hints).
_os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
_os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
_os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
