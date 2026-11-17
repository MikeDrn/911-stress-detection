"""Online evaluation logic (Course 4): shadow, canary, A/B test, bandits.

Everything here is a pure function of logged calls, so it is unit-tested
without a server. ``serve.py`` routes traffic and stores rows; this module
turns those rows into decisions.

Vocabulary (slide 5): champion = control = arm A = arm 0;
challenger = treatment = arm B = arm 1.

Why each strategy is used the way it is in a 911 setting
---------------------------------------------------------
* Shadow and canary are the real release path: shadow never shows the
  challenger to a dispatcher; canary limits the blast radius and rolls back
  automatically on guardrails.
* A/B tests and bandits deliberately expose some calls to the weaker model to
  learn which is better. For emergency calls that is only acceptable on
  replayed or simulated traffic, so the console labels those modes as
  simulation-only.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

MODES = ("champion_only", "shadow", "canary", "ab", "bandit")
ARMS = ("champion", "challenger")


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def sticky_bucket(call_id: str) -> int:
    """Deterministic bucket 0..99: the same call id always gets the same model (slide 6)."""
    return int(hashlib.sha256(call_id.encode()).hexdigest()[:8], 16) % 100


def route_split(call_id: str, challenger_percent: float) -> tuple[str, float]:
    """Return (arm, propensity) for a fixed percentage split."""
    w = min(max(challenger_percent, 0.0), 100.0) / 100.0
    if sticky_bucket(call_id) < challenger_percent:
        return "challenger", w
    return "champion", 1.0 - w


# --------------------------------------------------------------------------- #
# Statistics (Appendix A of Course 4)
# --------------------------------------------------------------------------- #
def normal_sf(z: float) -> float:
    """P(Z > z) for a standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def z_test_pooled_one_sided(x1: int, n1: int, x2: int, n2: int) -> dict[str, float]:
    """Canary test (3.3): is arm 2's event rate (errors) *higher* than arm 1's?"""
    if n1 == 0 or n2 == 0:
        return {"z": 0.0, "p_value": 1.0, "rate_1": 0.0, "rate_2": 0.0, "se": 0.0}
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = 0.0 if se == 0 else (p2 - p1) / se
    return {"z": z, "p_value": normal_sf(z), "rate_1": p1, "rate_2": p2, "se": se}


def z_test_unpooled_two_sided(x_a: int, n_a: int, x_b: int, n_b: int) -> dict[str, float]:
    """A/B test (4.2): difference of proportions, unpooled SE, 95% CI, two-sided p."""
    if n_a == 0 or n_b == 0:
        return {
            "p_a": 0.0,
            "p_b": 0.0,
            "diff": 0.0,
            "se": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }
    p_a, p_b = x_a / n_a, x_b / n_b
    se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    diff = p_b - p_a
    z = 0.0 if se == 0 else diff / se
    return {
        "p_a": p_a,
        "p_b": p_b,
        "diff": diff,
        "se": se,
        "z": z,
        "p_value": 2 * normal_sf(abs(z)) if se else 1.0,
        "ci_low": diff - 1.96 * se,
        "ci_high": diff + 1.96 * se,
    }


Z_TABLE = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}  # two-sided critical values
POWER_Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}


def ab_sample_size(p_baseline: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Users per arm to detect an absolute lift ``mde`` (decided *before* the test, 4.1)."""
    p2 = min(p_baseline + mde, 0.999)
    z_a, z_b = Z_TABLE.get(alpha, 1.96), POWER_Z.get(power, 0.8416)
    var = p_baseline * (1 - p_baseline) + p2 * (1 - p2)
    return int(math.ceil((z_a + z_b) ** 2 * var / mde**2))


def srm_p_value(n_a: int, n_b: int, expected_share_b: float = 0.5) -> float:
    """Sample-ratio-mismatch chi-square test (1 dof). Tiny p => broken split."""
    n = n_a + n_b
    if n == 0:
        return 1.0
    exp_b = n * expected_share_b
    exp_a = n - exp_b
    chi2 = (n_a - exp_a) ** 2 / exp_a + (n_b - exp_b) ** 2 / exp_b
    return math.erfc(math.sqrt(chi2 / 2.0))


def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    """Population Stability Index on scores in [0, 1] (slide 11). 0 = identical."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    if len(e) == 0 or len(a) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    e_share = np.histogram(np.clip(e, 0, 1), edges)[0] / len(e)
    a_share = np.histogram(np.clip(a, 0, 1), edges)[0] / len(a)
    e_share, a_share = np.clip(e_share, 1e-4, None), np.clip(a_share, 1e-4, None)
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if len(a) == 0 or len(b) == 0:
        return 0.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def p95(values: Sequence[float]) -> float:
    return float(np.percentile(values, 95)) if len(values) else 0.0


def psi_band(value: float, warn: float, major: float) -> str:
    return "major shift" if value >= major else "moderate shift" if value >= warn else "stable"


# --------------------------------------------------------------------------- #
# Arm statistics from logged calls
# --------------------------------------------------------------------------- #
def arm_stats(calls: list[dict[str, Any]], window_scores: dict[str, list[float]]) -> dict:
    """Summarise calls served by each arm. ``window_scores[arm]`` = flat window scores."""
    out = {}
    for arm in ARMS:
        rows = [c for c in calls if c["arm"] == arm]
        labelled = [c for c in rows if c.get("correct") is not None]
        scores = window_scores.get(arm, [])
        out[arm] = {
            "calls": len(rows),
            "errors": sum(int(c.get("error") or 0) for c in rows),
            "p95_ms_per_window": p95(
                [c["model_ms_per_window"] for c in rows if c.get("model_ms_per_window") is not None]
            ),
            "alert_rate": float(np.mean([c["alert"] for c in rows])) if rows else 0.0,
            "stressed_window_share": float(np.mean(np.asarray(scores) > 0.5)) if scores else 0.0,
            "labelled": len(labelled),
            "correct": sum(int(c["correct"]) for c in labelled),
            "accuracy": float(np.mean([c["correct"] for c in labelled])) if labelled else None,
        }
    return out


# --------------------------------------------------------------------------- #
# Canary (section 3): guardrails -> wait / advance / rollback / complete
# --------------------------------------------------------------------------- #
def canary_decision(
    stats: dict,
    champion_scores: list[float],
    challenger_scores: list[float],
    cfg: dict,
    step: int,
) -> dict[str, Any]:
    ch, ca = stats["champion"], stats["challenger"]
    steps = list(cfg["steps"])
    checks: dict[str, dict[str, Any]] = {}

    min_calls = int(cfg["min_challenger_calls_per_step"])
    if ch["calls"] < min_calls:  # guardrails compare v2 with v1: no baseline, no verdict
        return {
            "decision": "wait",
            "checks": checks,
            "reason": f"{ch['calls']}/{min_calls} champion calls for the baseline",
        }
    if ca["calls"] < min_calls:
        return {
            "decision": "wait",
            "checks": checks,
            "reason": f"{ca['calls']}/{cfg['min_challenger_calls_per_step']} challenger "
            "calls in this step",
        }

    err = z_test_pooled_one_sided(ch["errors"], max(ch["calls"], 1), ca["errors"], ca["calls"])
    checks["error_rate_z_test"] = {
        "passed": err["z"] <= float(cfg["error_z_critical"]),
        "champion_rate": err["rate_1"],
        "challenger_rate": err["rate_2"],
        "z": err["z"],
        "critical": float(cfg["error_z_critical"]),
    }

    budget = float(cfg["max_latency_ratio"]) * ch["p95_ms_per_window"] + float(
        cfg["latency_slack_ms"]
    )
    checks["p95_latency"] = {
        "passed": ch["p95_ms_per_window"] == 0 or ca["p95_ms_per_window"] <= budget,
        "champion_ms": ch["p95_ms_per_window"],
        "challenger_ms": ca["p95_ms_per_window"],
        "budget_ms": budget,
    }

    share_c, share_ch = ca["stressed_window_share"], ch["stressed_window_share"]
    hi = float(cfg["collapse_share"])
    collapsed = (share_c >= hi or share_c <= 1 - hi) and (1 - hi < share_ch < hi)
    checks["no_class_collapse"] = {
        "passed": not collapsed,
        "challenger_stressed_share": share_c,
        "champion_stressed_share": share_ch,
    }

    value = psi(champion_scores, challenger_scores)
    checks["score_psi"] = {
        "passed": True,  # rule of thumb, not a test: reported, never auto-rollback
        "psi": value,
        "band": psi_band(value, cfg["psi_warn"], cfg["psi_major"]),
        "ks": ks_statistic(champion_scores, challenger_scores),
    }

    if not all(c["passed"] for c in checks.values()):
        failed = [k for k, c in checks.items() if not c["passed"]]
        return {"decision": "rollback", "checks": checks, "reason": "failed: " + ", ".join(failed)}
    if step >= len(steps) - 1:
        return {
            "decision": "complete",
            "checks": checks,
            "reason": "100% reached with healthy guardrails - awaiting human promotion",
        }
    return {
        "decision": "advance",
        "checks": checks,
        "reason": f"healthy - advance to {steps[step + 1]}%",
    }


# --------------------------------------------------------------------------- #
# Shadow (slide 4) and A/B (section 4) reports
# --------------------------------------------------------------------------- #
def shadow_report(calls: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [c for c in calls if c.get("shadow_alert") is not None]
    if not rows:
        return {"calls": 0}
    agree = [int(c["alert"]) == int(c["shadow_alert"]) for c in rows]
    # Correctness is recomputed from the stored label: feedback can arrive before the
    # background shadow scoring has finished, so a stored shadow_correct may be missing.
    labelled = [c for c in rows if c.get("true_label") is not None]

    def correct(alert_key: str) -> float | None:
        if not labelled:
            return None
        return float(
            np.mean([bool(c[alert_key]) == (c["true_label"] == "stressed") for c in labelled])
        )

    return {
        "calls": len(rows),
        "alert_agreement": float(np.mean(agree)),
        "champion_alert_rate": float(np.mean([c["alert"] for c in rows])),
        "challenger_alert_rate": float(np.mean([c["shadow_alert"] for c in rows])),
        "champion_p95_ms_per_window": p95([c["model_ms_per_window"] for c in rows]),
        "challenger_p95_ms_per_window": p95([c["shadow_ms_per_window"] for c in rows]),
        "challenger_errors": sum(int(c.get("shadow_error") or 0) for c in rows),
        "labelled": len(labelled),
        "champion_accuracy": correct("alert"),
        "challenger_accuracy": correct("shadow_alert"),
    }


def ab_report(stats: dict, planned_n_per_arm: int, alpha: float = 0.05) -> dict[str, Any]:
    a, b = stats["champion"], stats["challenger"]
    test = z_test_unpooled_two_sided(a["correct"], a["labelled"], b["correct"], b["labelled"])
    srm = srm_p_value(a["calls"], b["calls"])
    reached = min(a["labelled"], b["labelled"]) >= planned_n_per_arm
    if srm < 0.001:
        verdict = "INVALID - sample ratio mismatch: the split is broken, do not trust results"
    elif not reached:
        verdict = (
            f"RUNNING - {min(a['labelled'], b['labelled'])}/{planned_n_per_arm} labelled "
            "calls per arm; no decision before the planned sample (no peeking)"
        )
    elif test["p_value"] < alpha and test["diff"] > 0:
        verdict = "B BETTER - significant improvement; promote after human approval"
    elif test["p_value"] < alpha:
        verdict = "B WORSE - significant degradation; keep the champion"
    else:
        verdict = "NO EVIDENCE - difference within noise; keep the champion"
    return {
        **test,
        "planned_n_per_arm": planned_n_per_arm,
        "sample_reached": reached,
        "srm_p_value": srm,
        "verdict": verdict,
    }
