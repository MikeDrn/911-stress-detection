"""End-to-end model smoke test (marked slow): the whole DVC pipeline in a sandbox.

Copies the code into a temporary project, generates a tiny synthetic dataset
and runs prepare -> validate -> featurize -> train (1 epoch) -> quality gate,
all offline (``--no-mlflow``). Proves the stages wire together without
touching the developer's real data/, models/ or MLflow server.

    uv run pytest -m slow
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_full_pipeline_offline(tmp_path):
    proj = tmp_path / "proj"
    shutil.copytree(ROOT / "src", proj / "src")
    shutil.copytree(ROOT / "scripts", proj / "scripts")
    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    params["synthetic"]["clips_per_class"] = 10
    params["train"]["batch_size"] = 16
    params["quality_gate"]["latency_trials"] = 10
    params["quality_gate"]["min_stressed_recall"] = 0.0  # 1 epoch on 20 clips: not a quality test
    params["quality_gate"]["max_p95_latency_ms"] = 10_000
    (proj / "params.yaml").write_text(yaml.safe_dump(params))

    env = {**os.environ, "PYTHONPATH": str(proj), "PARAMS_PATH": str(proj / "params.yaml")}

    def run(*args):
        res = subprocess.run(
            [sys.executable, "-m", *args],
            cwd=proj,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert res.returncode == 0, f"{args}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        return res.stdout

    run("scripts.generate_synthetic_data")
    run("src.data_contracts", "prepare")
    assert "all data contracts satisfied" in run("src.data_contracts", "validate")
    run("src.features")
    run("src.train", "--no-mlflow", "--epochs", "1")
    run("src.quality_gate", "--no-mlflow")

    gate = json.loads((proj / "reports" / "quality_gate.json").read_text())
    assert gate["promoted"] is True  # bootstrap promotion with relaxed floors
    assert (proj / "models" / "champion.pt").exists()
    assert gate["challenger"]["p95_latency_ms"] > 0
