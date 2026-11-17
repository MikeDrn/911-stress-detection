"""Combine WHAT is said (content) with HOW it is said (voice stress).

The rule encodes the project's key insight: on a 911 line stress is normal.

    threat = p_covert * (base + stress_weight * stress)
           + incongruence_weight * p_non_emergency * stress

* genuine emergency + very stressed voice  -> ~0      (stress is expected)
* covert-looking content + calm voice      -> high    ("pizza order" on 911)
* covert-looking content + tense voice     -> higher
* harmless question + terrified voice      -> raised  (content/voice mismatch)

It is a transparent, configurable rule rather than a learned model on purpose:
no labelled dataset of real calls with both audio and outcome exists, and a
dispatcher must be able to read *why* a call was flagged.
"""

from __future__ import annotations

from typing import Any


def fuse(text: dict[str, Any] | None, stress: float, cfg: dict[str, Any]) -> dict[str, Any]:
    stress = float(min(max(stress, 0.0), 1.0))
    if not text:
        return {
            "threat_score": 0.0,
            "threat_level": "normal",
            "reason": "No transcript available - content could not be assessed; "
            f"voice stress {stress:.2f} shown for information only.",
        }
    p = text["probs"]
    covert = p.get("covert", 0.0) * (
        float(cfg["covert_base"]) + float(cfg["covert_stress_weight"]) * stress
    )
    incongruent = float(cfg["incongruence_weight"]) * p.get("non_emergency", 0.0) * stress
    threat = min(1.0, covert + incongruent)
    level = (
        "alert"
        if threat >= float(cfg["alert_threshold"])
        else "review"
        if threat >= float(cfg["review_threshold"])
        else "normal"
    )
    return {
        "threat_score": round(threat, 4),
        "threat_level": level,
        "reason": explain(text, stress, level, covert, incongruent),
    }


def explain(
    text: dict[str, Any], stress: float, level: str, covert: float, incongruent: float
) -> str:
    p, label = text["probs"], text["label"]
    words = ", ".join(f"'{w}'" for w in text.get("evidence", [])[:4])
    voice = "calm" if stress < 0.35 else "tense" if stress < 0.65 else "highly stressed"
    if level == "normal" and label == "genuine_emergency":
        return (
            f"Genuine emergency content ({p['genuine_emergency']:.0%}) with a {voice} "
            f"voice ({stress:.2f}). Stress is expected in a real emergency."
        )
    if covert >= incongruent and p.get("covert", 0) >= 0.3:
        return (
            f"Content resembles a covert call ({p['covert']:.0%}; cues: {words}) "
            f"with a {voice} voice ({stress:.2f})."
        )
    if incongruent > 0.1 and level != "normal":
        return (
            f"Harmless-sounding content ({p.get('non_emergency', 0):.0%} non-emergency) "
            f"but a {voice} voice ({stress:.2f}): content and voice do not match."
        )
    return (
        f"Most likely {label.replace('_', ' ')} ({p[label]:.0%}) with a {voice} voice "
        f"({stress:.2f})."
    )
