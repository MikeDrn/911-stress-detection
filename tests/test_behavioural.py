"""Behavioural tests (Course 7 slide 12) exercised with stand-in models."""

from __future__ import annotations

import random

from src.behavioural import CANONICAL, content_checks, directional_checks, perturb
from src.fake_model import FakeTextModel
from src.transcripts import generate

CFG = {
    "min_functionality_accuracy": 0.9,
    "invariance_min_agreement": 0.9,
    "directional_max_genuine_alert": 0.02,
    "directional_min_covert_alert": 0.85,
}
FUSION = {
    "covert_base": 0.85,
    "covert_stress_weight": 0.25,
    "incongruence_weight": 0.4,
    "alert_threshold": 0.5,
    "review_threshold": 0.3,
}


def test_canonical_set_contains_the_traps():
    labels = dict(CANONICAL)
    assert labels["my son is choking on a piece of pizza he can't breathe"] == "genuine_emergency"
    assert labels["sorry wrong number i was trying to call the pizza place bye"] == "non_emergency"


def test_perturb_swaps_slots_but_keeps_meaning():
    rng = random.Random(0)
    out = perturb("order a large pizza to 61 elm street", rng)
    assert "61 elm street" not in out and out != ""


def test_keyword_model_fails_min_functionality():
    """A 'pizza => covert' rule must NOT pass: the traps catch it."""
    rows = [r.__dict__ for r in generate(10, 1) if r.split == "test"]
    checks = content_checks(FakeTextModel(), rows, CFG)
    missed = checks["content_min_functionality"]["missed"]
    assert not checks["content_min_functionality"]["passed"]
    assert any("choking" in m for m in missed)


def test_directional_rules():
    rows = [r.__dict__ for r in generate(10, 1) if r.split == "test"]
    always_covert = FakeTextModel(
        {"covert": 0.95, "genuine_emergency": 0.03, "non_emergency": 0.02}
    )
    d = directional_checks(always_covert, rows, FUSION, CFG)
    assert d["directional_covert_calm_voice_still_alerts"]["passed"]
    assert not d["directional_genuine_emergency_max_stress_no_alert"]["passed"]
