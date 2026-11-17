"""Behavioural model tests (Course 7 slide 12) exercised with FakeModels."""

from __future__ import annotations

import numpy as np

from src.behavioural import SLICES, make_slice_set, run_behavioural_tests
from src.fake_model import FakeModel

CFG = {
    "clips_per_slice": 2,
    "min_functionality_accuracy": 0.9,
    "invariance_min_agreement": 0.95,
    "slice_min_recall_stressed": 0.85,
    "slice_min_accuracy": 0.8,
}


def test_slice_set_covers_every_slice_and_class():
    clips = make_slice_set(2)
    assert len(clips) == len(SLICES) * 2 * 2
    assert {(c["sex"], c["channel"]) for c in clips} == set(SLICES)
    assert all(c["y"].dtype == np.float32 and len(c["y"]) == 48000 for c in clips)


def test_always_stressed_model_fails_functionality_and_slices():
    checks = run_behavioural_tests(FakeModel([0.99]), CFG)
    assert checks["invariance_gain_and_noise"]["passed"]  # constant output is invariant ...
    assert not checks["min_functionality_clean_calls"]["passed"]  # ... but useless
    slices = checks["slice_floors_sex_x_channel"]["slices"]
    assert all(s["recall_stressed"] == 1.0 and s["accuracy"] == 0.5 for s in slices.values())
    assert not checks["slice_floors_sex_x_channel"]["passed"]


def test_uncovered_channels_are_warnings_not_failures():
    checks = run_behavioural_tests(FakeModel([0.99]), CFG, enforced_channels={"clean"})
    sl = checks["slice_floors_sex_x_channel"]
    assert not sl["passed"]  # clean slices are enforced and fail
    assert sl["coverage_gap_warnings"] == ["female/phone", "male/phone"]
    assert sl["slices"]["male/phone"]["enforced"] is False
