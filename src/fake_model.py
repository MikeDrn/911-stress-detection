"""Lightweight stand-in for the neural network.

``FakeModel`` implements the exact ``predict_proba`` interface of
``TorchStressPredictor`` but imports no PyTorch and loads no weights. CI uses it
to exercise the API contract, the health probes, the alert rule and the
telemetry writer in well under a second on a CPU-only runner.

It is a *test double*, not a model: its scores carry no acoustic meaning.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class FakeModel:
    architecture = "fake"
    input_key = "logmel"

    def __init__(self, scores: Sequence[float] | None = None, version: str = "fake-0"):
        """``scores``: optional fixed score sequence, repeated/truncated to n_windows.

        Without ``scores`` every window returns 0.10 (a calm, below-threshold call).
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
