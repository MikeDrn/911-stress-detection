"""The registered unit: voice-stress model + transcript model + fusion rule.

A *bundle* is a folder::

    bundle.json        metadata (versions, fusion parameters, metrics)
    acoustic.pt        ResNet-18 checkpoint (weights only)
    text_model.json    TF-IDF + logistic regression, plain JSON

Champion / challenger, canary, A/B and rollback all operate on whole bundles,
so a release always pairs a voice model and a content model that were
evaluated *together* by the quality gate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.fusion import fuse

BUNDLE_FILE = "bundle.json"


class CallModel:
    """Analyse one call: per-window voice stress + transcript class + fused threat."""

    def __init__(self, acoustic, text, fusion_cfg: dict[str, Any], version: str = "local"):
        self.acoustic = acoustic
        self.text = text
        self.fusion_cfg = dict(fusion_cfg)
        self.version = version
        self.input_key = getattr(acoustic, "input_key", "logmel")

    def predict_proba(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Per-window P(stressed voice) - kept so window-level tooling works unchanged."""
        return self.acoustic.predict_proba(features)

    def analyse_text(self, transcript: str | None) -> dict[str, Any] | None:
        if not transcript or not transcript.strip():
            return None
        return self.text.predict(transcript)

    def decide(self, window_scores: np.ndarray, transcript: str | None) -> dict[str, Any]:
        stress = float(np.mean(window_scores)) if len(window_scores) else 0.0
        content = self.analyse_text(transcript)
        return {
            "stress_score": round(stress, 4),
            "content": content,
            **fuse(content, stress, self.fusion_cfg),
        }


def save_bundle(
    out_dir: str | Path,
    acoustic_ckpt: str | Path,
    text_model: str | Path,
    fusion_cfg: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copyfile(acoustic_ckpt, out / "acoustic.pt")
    shutil.copyfile(text_model, out / "text_model.json")
    (out / BUNDLE_FILE).write_text(
        json.dumps({"format": 1, "fusion": dict(fusion_cfg), **(meta or {})}, indent=2)
    )
    return out


def find_bundle_dir(root: str | Path) -> Path:
    root = Path(root)
    if (root / BUNDLE_FILE).exists():
        return root
    hits = sorted(root.rglob(BUNDLE_FILE))
    if not hits:
        raise FileNotFoundError(
            f"no {BUNDLE_FILE} under {root} - is this an old "
            "acoustic-only model? Register a bundle with `dvc repro`."
        )
    return hits[0].parent


def load_bundle(
    path: str | Path, version: str = "local", fusion_override: dict[str, Any] | None = None
) -> CallModel:
    from src.model import TorchStressPredictor
    from src.text_model import TextModel

    d = find_bundle_dir(path)
    meta = json.loads((d / BUNDLE_FILE).read_text())
    acoustic = TorchStressPredictor.from_checkpoint(d / "acoustic.pt", version=version)
    text = TextModel.load(d / "text_model.json", version=version)
    return CallModel(acoustic, text, fusion_override or meta["fusion"], version)
