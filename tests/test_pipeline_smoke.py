"""End-to-end smoke test (marked slow): the whole DVC pipeline in a sandbox.

Copies the code into a temporary project and runs every stage offline
(``--no-mlflow``): prepare -> validate -> featurize -> train_acoustic (1 epoch) ->
make_transcripts -> validate_text -> train_text -> package -> quality_gate.

The audio here is a tiny synthetic FIXTURE (CI must not depend on downloading the
real dataset): "CI checks that training works, not that it is good" (Course 7).

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
    params["text"]["per_family"] = 12
    gate = params["quality_gate"]
    gate.update(
        latency_trials=10,
        max_p95_latency_ms=10_000,
        min_covert_recall=0.0,
        max_genuine_false_alarm=1.0,
        min_voice_accuracy=0.0,
        auto_promote=False,
    )
    gate["behavioural"].update(
        min_functionality_accuracy=0.0,
        invariance_min_agreement=0.0,
        directional_max_genuine_alert=1.0,
        directional_min_covert_alert=0.0,
        clips_per_slice=1,
        slice_min_accuracy=0.0,
        slice_min_recall_stressed=0.0,
    )
    (proj / "params.yaml").write_text(yaml.safe_dump(params))

    env = {**os.environ, "PYTHONPATH": str(proj), "PARAMS_PATH": str(proj / "params.yaml")}

    def run(*args):
        res = subprocess.run(
            [sys.executable, "-m", *args],
            cwd=proj,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert res.returncode == 0, f"{args}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        return res.stdout

    run("scripts.generate_synthetic_data")  # fixture audio, not training data
    run("src.data_contracts", "prepare")
    assert "all data contracts satisfied" in run("src.data_contracts", "validate")
    run("src.features")
    run("src.train", "--epochs", "1")
    run("src.transcripts")
    assert "all text data contracts satisfied" in run("src.text_contracts")
    assert "TEST (unseen families)" in run("src.text_model")
    run("src.package", "--no-mlflow")
    run("src.quality_gate", "--no-mlflow")

    report = json.loads((proj / "reports" / "quality_gate.json").read_text())
    assert report["promoted"] is True and report["bootstrap"] is True
    assert (proj / "models" / "champion_bundle" / "bundle.json").exists()
    checks = report["decision"]["checks"]
    assert "behavioural_content_min_functionality" in checks
    assert "genuine_emergency_false_alarm_ceiling" in checks
    assert report["challenger"]["p95_voice_latency_ms"] > 0
    assert report["sanity_checks"]["shuffled_labels_text_accuracy"] is not None
