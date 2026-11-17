"""Bundle = voice model + transcript model + fusion rule, saved and loaded as one unit."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.bundle import BUNDLE_FILE, find_bundle_dir, load_bundle, save_bundle  # noqa: E402
from src.model import create_model, save_checkpoint  # noqa: E402
from src.text_model import fit  # noqa: E402
from src.transcripts import generate  # noqa: E402

FUSION = {
    "covert_base": 0.85,
    "covert_stress_weight": 0.25,
    "incongruence_weight": 0.4,
    "alert_threshold": 0.5,
    "review_threshold": 0.3,
}


def test_bundle_roundtrip(tmp_path):
    ckpt = save_checkpoint(
        create_model("resnet18_2d"),
        tmp_path / "a.pt",
        {"architecture": "resnet18_2d", "num_classes": 2},
    )
    rows = [r for r in generate(10, 1) if r.split == "train"]
    text = fit([r.text for r in rows], [r.label for r in rows], {"min_df": 1})
    text_path = text.save(tmp_path / "t.json")
    out = save_bundle(tmp_path / "bundle", ckpt, text_path, FUSION, {"trained_channels": ["clean"]})
    meta = json.loads((out / BUNDLE_FILE).read_text())
    assert meta["fusion"] == FUSION and meta["trained_channels"] == ["clean"]

    nested = tmp_path / "mlflow" / "artifacts"
    nested.mkdir(parents=True)
    (out).rename(nested / "bundle")
    assert find_bundle_dir(tmp_path / "mlflow") == nested / "bundle"

    model = load_bundle(tmp_path / "mlflow", version="v9")
    scores = model.predict_proba({"logmel": np.zeros((3, 1, 64, 201), np.float32)})
    assert scores.shape == (3,)
    d = model.decide(scores, "hi i'd like to order a large pizza with pepperoni")
    assert set(d) >= {"threat_score", "threat_level", "reason", "stress_score", "content"}


def test_old_acoustic_only_model_gives_a_clear_error(tmp_path):
    (tmp_path / "artifacts").mkdir()
    with pytest.raises(FileNotFoundError, match="acoustic-only"):
        find_bundle_dir(tmp_path)
