"""Statistics behind Course 4, checked against the lecture's own worked examples."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src import rollout as ro

CANARY_CFG = {
    "steps": [10, 25, 50, 100],
    "min_challenger_calls_per_step": 20,
    "max_latency_ratio": 1.10,
    "latency_slack_ms": 3.0,
    "error_z_critical": 1.645,
    "collapse_share": 0.98,
    "psi_warn": 0.10,
    "psi_major": 0.25,
}


def arm(calls=40, errors=0, p95=10.0, share=0.5):
    return {
        "calls": calls,
        "errors": errors,
        "p95_ms_per_window": p95,
        "alert_rate": share,
        "stressed_window_share": share,
        "labelled": calls,
        "correct": calls,
        "accuracy": 1.0,
    }


# ---------------------------------------------------------------- routing
def test_sticky_bucket_is_deterministic_and_uniform():
    assert ro.sticky_bucket("call-42") == ro.sticky_bucket("call-42")
    buckets = [ro.sticky_bucket(f"call-{i}") for i in range(20_000)]
    share = np.mean(np.array(buckets) < 10)
    assert 0.09 < share < 0.11  # ~10% land in a 10% canary


def test_route_split_propensity():
    assert ro.route_split("x", 0) == ("champion", 1.0)
    assert ro.route_split("x", 100) == ("challenger", 1.0)
    arm_, p = ro.route_split("x", 25)
    assert p == (0.25 if arm_ == "challenger" else 0.75)


# ------------------------------------------------- Course 4 worked examples
def test_canary_z_test_matches_appendix_a5():
    # v1: 500 errors / 100,000 ; v2: 16 / 2,000  ->  z ~ 1.87  -> roll back
    r = ro.z_test_pooled_one_sided(500, 100_000, 16, 2_000)
    assert r["se"] == pytest.approx(0.00160, abs=2e-5)
    assert r["z"] == pytest.approx(1.87, abs=0.02)
    # same rates on only 500 canary requests -> z ~ 0.95, no alarm (low power)
    assert ro.z_test_pooled_one_sided(500, 100_000, 4, 500)["z"] == pytest.approx(0.95, abs=0.03)


def test_ab_z_test_matches_appendix_a6():
    # 10.0% vs 10.9% conversion with 15,000 users each -> z ~ 2.55, CI [0.21, 1.59] pts
    r = ro.z_test_unpooled_two_sided(1500, 15_000, 1635, 15_000)
    assert r["z"] == pytest.approx(2.55, abs=0.02)
    assert r["p_value"] == pytest.approx(0.011, abs=0.001)
    assert 100 * r["ci_low"] == pytest.approx(0.21, abs=0.02)
    assert 100 * r["ci_high"] == pytest.approx(1.59, abs=0.02)


def test_sample_size_formula():
    n = ro.ab_sample_size(0.85, 0.10, alpha=0.05, power=0.80)
    expected = (1.96 + 0.8416) ** 2 * (0.85 * 0.15 + 0.95 * 0.05) / 0.10**2
    assert n == math.ceil(expected)
    assert ro.ab_sample_size(0.85, 0.05) > n  # smaller effects need more users


def test_srm_detects_broken_split():
    assert ro.srm_p_value(500, 500) == pytest.approx(1.0)
    assert ro.srm_p_value(600, 400) < 0.001


def test_psi_and_ks():
    rng = np.random.default_rng(0)
    a = rng.beta(2, 5, 5000)
    assert ro.psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert ro.psi(a, rng.beta(2, 5, 5000)) < 0.1
    assert ro.psi(a, rng.beta(5, 2, 5000)) > 0.25
    assert ro.ks_statistic(a, a) == 0.0
    assert ro.psi_band(0.05, 0.1, 0.25) == "stable"
    assert ro.psi_band(0.3, 0.1, 0.25) == "major shift"


# ------------------------------------------------------- canary decisions
def test_canary_waits_for_enough_traffic():
    stats = {"champion": arm(200), "challenger": arm(5)}
    assert ro.canary_decision(stats, [0.5], [0.5], CANARY_CFG, 0)["decision"] == "wait"


def test_canary_waits_for_a_champion_baseline():
    stats = {"champion": arm(2), "challenger": arm(40, share=1.0)}
    d = ro.canary_decision(stats, [0.5], [0.99], CANARY_CFG, 0)
    assert d["decision"] == "wait" and "champion" in d["reason"]


def test_canary_advances_then_completes():
    stats = {"champion": arm(200), "challenger": arm(25)}
    scores = list(np.linspace(0, 1, 50))
    d = ro.canary_decision(stats, scores, scores, CANARY_CFG, 0)
    assert d["decision"] == "advance" and "25%" in d["reason"]
    assert ro.canary_decision(stats, scores, scores, CANARY_CFG, 3)["decision"] == "complete"


def test_canary_rolls_back_on_class_collapse():
    stats = {"champion": arm(200, share=0.6), "challenger": arm(25, share=1.0)}
    d = ro.canary_decision(stats, [0.2, 0.8], [0.99, 0.99], CANARY_CFG, 0)
    assert d["decision"] == "rollback" and "no_class_collapse" in d["reason"]


def test_canary_rolls_back_on_latency():
    stats = {"champion": arm(200, p95=10.0), "challenger": arm(25, p95=30.0)}
    d = ro.canary_decision(stats, [0.5], [0.5], CANARY_CFG, 0)
    assert d["decision"] == "rollback" and "p95_latency" in d["reason"]


def test_canary_rolls_back_on_error_rate():
    stats = {"champion": arm(1000, errors=1), "challenger": arm(40, errors=6)}
    d = ro.canary_decision(stats, [0.5], [0.5], CANARY_CFG, 0)
    assert d["decision"] == "rollback" and d["checks"]["error_rate_z_test"]["z"] > 1.645


def test_psi_alone_never_rolls_back():
    stats = {"champion": arm(200), "challenger": arm(25)}
    d = ro.canary_decision(stats, [0.05] * 100, [0.95] * 100, CANARY_CFG, 0)
    assert d["checks"]["score_psi"]["band"] == "major shift" and d["decision"] == "advance"


# ------------------------------------------------------------ reports
def test_shadow_report_recomputes_correctness_from_labels():
    calls = [
        {
            "alert": 1,
            "shadow_alert": 0,
            "true_label": "calm",
            "model_ms_per_window": 10,
            "shadow_ms_per_window": 11,
            "shadow_error": 0,
        },
        {
            "alert": 1,
            "shadow_alert": 1,
            "true_label": "stressed",
            "model_ms_per_window": 10,
            "shadow_ms_per_window": 11,
            "shadow_error": 0,
        },
    ]
    r = ro.shadow_report(calls)
    assert r["alert_agreement"] == 0.5
    assert r["champion_accuracy"] == 0.5 and r["challenger_accuracy"] == 1.0


def test_ab_report_no_peeking_and_verdict():
    a = {**arm(100), "labelled": 100, "correct": 85, "calls": 100}
    b = {**arm(100), "labelled": 100, "correct": 98, "calls": 100}
    early = ro.ab_report({"champion": a, "challenger": b}, planned_n_per_arm=150)
    assert early["verdict"].startswith("RUNNING")
    final = ro.ab_report({"champion": a, "challenger": b}, planned_n_per_arm=100)
    assert final["verdict"].startswith("B BETTER") and final["ci_low"] > 0
