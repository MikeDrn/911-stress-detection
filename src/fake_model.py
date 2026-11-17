"""Lightweight stand-ins for the neural network and the transcript model.

``FakeModel`` implements the ``predict_proba`` interface of the voice model and
``FakeTextModel`` the ``predict`` interface of the transcript model. Neither
imports PyTorch or loads weights, so CI exercises the API contract, health
probes, fusion, release modes and telemetry in seconds on a CPU-only runner.

They are *test doubles*, not models: their outputs carry no meaning.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

LABELS = ("covert", "genuine_emergency", "non_emergency")


class FakeModel:
    architecture = "fake"
    input_key = "logmel"

    def __init__(self, scores: Sequence[float] | None = None, version: str = "fake-0"):
        """``scores``: fixed per-window stress scores, repeated/truncated to n_windows.

        Without ``scores`` every window returns 0.10 (a calm voice).
        """
        self.scores = None if scores is None else np.asarray(scores, dtype=np.float32)
        self.version = version
        self.calls = 0

    def predict_proba(self, features: dict[str, np.ndarray]) -> np.ndarray:
        self.calls += 1
        n = len(features[self.input_key])
        if self.scores is None:
            return np.full(n, 0.10, dtype=np.float32)
        reps = int(np.ceil(n / max(1, len(self.scores))))
        return np.tile(self.scores, reps)[:n].astype(np.float32)


class FakeTextModel:
    """Keyword stand-in: 'pizza/pepperoni/order' -> covert, 'dead/breath/fire' -> genuine."""

    classes = list(LABELS)

    def __init__(self, fixed: dict[str, float] | None = None, version: str = "fake-text"):
        self.fixed = fixed
        self.version = version

    def predict(self, text: str) -> dict:
        if self.fixed is not None:
            probs = dict(self.fixed)
        else:
            t = text.lower()
            if any(k in t for k in ("pizza", "pepperoni", "order")):
                probs = {"covert": 0.9, "genuine_emergency": 0.05, "non_emergency": 0.05}
            elif any(k in t for k in ("dead", "breath", "fire", "blood")):
                probs = {"covert": 0.03, "genuine_emergency": 0.94, "non_emergency": 0.03}
            else:
                probs = {"covert": 0.05, "genuine_emergency": 0.05, "non_emergency": 0.9}
        label = max(probs, key=probs.get)
        evidence = [w for w in text.lower().split() if w in ("pizza", "pepperoni", "dead")]
        return {"label": label, "probs": probs, "evidence": evidence, "known_terms": 1}


def fake_call_model(scores=None, text_probs=None, version: str = "fake-0", fusion=None):
    from src.bundle import CallModel
    from src.config import load_params

    return CallModel(
        FakeModel(scores, version),
        FakeTextModel(text_probs, version),
        fusion or load_params()["fusion"],
        version,
    )
