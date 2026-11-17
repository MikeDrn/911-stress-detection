"""Behavioural model tests (Course 7, slide 12 - inspired by CheckList).

One global accuracy hides failures, so the quality gate also checks *what the
model must do*, on freshly synthesised clips with controlled properties:

* Minimum functionality - obvious, clean-line calls must be classified correctly.
* Invariance            - changes that must not change the answer: louder /
                          quieter voice (x0.5, x1.5 gain) and light background
                          noise (30 dB SNR).
* Slices                - accuracy and stressed-recall floors for every slice
                          (speaker sex x telephone channel). A model can gain a
                          point overall while one slice collapses.

Floors are *enforced* only for channels the training data covers. A slice the
dataset does not cover yet (e.g. phone lines before any phone data exists) is
still measured and reported as a coverage-gap WARNING: the gate cannot demand
what the data never showed the model, but the team must see the risk.

Decisions are made per 2-second window at p >= 0.5, like the offline gate.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from src.preprocessing import AudioSpec, FeatureExtractor, sliding_windows
from src.synthetic import synth_utterance

SLICES = list(product(("female", "male"), ("clean", "phone")))


def make_slice_set(clips_per_slice: int, seed: int = 4242, sr: int = 16000) -> list[dict]:
    rng = np.random.default_rng(seed)
    clips = []
    for (sex, channel), label in product(SLICES, ("calm", "stressed")):
        for _ in range(clips_per_slice):
            y = synth_utterance(
                3.0, label == "stressed", rng, sr, female=sex == "female", channel=channel
            )
            clips.append({"y": y, "label": label, "sex": sex, "channel": channel})
    return clips


def _decisions(predictor, extractor: FeatureExtractor, spec: AudioSpec, y: np.ndarray):
    windows, _ = sliding_windows(y.astype(np.float32), spec)
    return predictor.predict_proba(extractor.transform(windows)) >= 0.5


def run_behavioural_tests(
    predictor,
    cfg: dict[str, Any],
    spec: AudioSpec | None = None,
    seed: int = 4242,
    enforced_channels: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    spec = spec or AudioSpec()
    extractor = FeatureExtractor(spec)
    clips = make_slice_set(int(cfg["clips_per_slice"]), seed, spec.sample_rate)
    rng = np.random.default_rng(seed + 1)
    for c in clips:
        c["pred"] = _decisions(predictor, extractor, spec, c["y"])
        c["truth"] = c["label"] == "stressed"

    checks: dict[str, dict[str, Any]] = {}

    clean = [c for c in clips if c["channel"] == "clean"]
    acc = float(np.mean(np.concatenate([c["pred"] == c["truth"] for c in clean])))
    checks["min_functionality_clean_calls"] = {
        "passed": acc >= float(cfg["min_functionality_accuracy"]),
        "accuracy": acc,
        "required": float(cfg["min_functionality_accuracy"]),
    }

    agree = []
    for c in clean:
        noise_pow = np.mean(c["y"] ** 2) / 10 ** (30 / 10)
        for variant in (
            c["y"] * 0.5,
            np.clip(c["y"] * 1.5, -1, 1),
            c["y"] + rng.normal(0, np.sqrt(noise_pow), len(c["y"])),
        ):
            agree.append(_decisions(predictor, extractor, spec, variant) == c["pred"])
    agreement = float(np.mean(np.concatenate(agree)))
    checks["invariance_gain_and_noise"] = {
        "passed": agreement >= float(cfg["invariance_min_agreement"]),
        "agreement": agreement,
        "required": float(cfg["invariance_min_agreement"]),
    }

    enforced_channels = set(enforced_channels or {ch for _, ch in SLICES})
    per_slice, ok, gaps = {}, True, []
    for sex, channel in SLICES:
        rows = [c for c in clips if c["sex"] == sex and c["channel"] == channel]
        preds = np.concatenate([c["pred"] for c in rows])
        truth = np.concatenate([np.full(len(c["pred"]), c["truth"]) for c in rows])
        s_acc = float(np.mean(preds == truth))
        s_rec = float(np.mean(preds[truth])) if truth.any() else 0.0
        meets = s_acc >= float(cfg["slice_min_accuracy"]) and s_rec >= float(
            cfg["slice_min_recall_stressed"]
        )
        enforced = channel in enforced_channels
        if enforced:
            ok &= meets
        elif not meets:
            gaps.append(f"{sex}/{channel}")
        per_slice[f"{sex}/{channel}"] = {
            "accuracy": s_acc,
            "recall_stressed": s_rec,
            "passed": meets,
            "enforced": enforced,
        }
    checks["slice_floors_sex_x_channel"] = {
        "passed": ok,
        "slices": per_slice,
        "enforced_channels": sorted(enforced_channels),
        "coverage_gap_warnings": gaps,
        "required_accuracy": float(cfg["slice_min_accuracy"]),
        "required_recall_stressed": float(cfg["slice_min_recall_stressed"]),
    }
    return checks
