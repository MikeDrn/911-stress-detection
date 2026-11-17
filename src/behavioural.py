"""Behavioural tests for the call model (Course 7, slide 12 - inspired by CheckList).

One global accuracy hides failures, so the gate also checks *what the system must do*:

Content (transcript model)
  min_functionality  obvious canonical calls must be classified correctly, including
                     the traps: "my son is choking on pizza" is a genuine emergency,
                     "sorry, wrong number, I wanted the pizza place" is not a threat.
  invariance         changing the address, a name, the food or the topping, or adding
                     speech-to-text noise, must not change the class.

Fusion (the 911 rule)
  directional        a genuine emergency must NOT alert even with a maximally stressed
                     voice; a covert-looking call must alert even with a calm voice.

Voice (acoustic model) - slices
  accuracy / stressed-recall floors per speaker sex x telephone channel, on real
  held-out speakers. Floors are enforced only for channels the training data covers;
  an uncovered channel is measured and reported as a coverage-gap WARNING.
"""

from __future__ import annotations

import csv
import random
import re
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from src.fusion import fuse
from src.transcripts import SLOTS, asr_noise

CANONICAL = [
    ("hi i'd like to order a large pizza with pepperoni to 61 elm street please hurry", "covert"),
    ("yes yes with pepperoni please hurry", "covert"),
    ("hi mom yeah everything is fine can you come over to 88 oak avenue right now", "covert"),
    ("my husband collapsed and he is not breathing please send an ambulance", "genuine_emergency"),
    ("someone is dead in front of me please come quickly", "genuine_emergency"),
    ("my son is choking on a piece of pizza he can't breathe", "genuine_emergency"),
    ("there's a fire in my kitchen and the smoke is everywhere", "genuine_emergency"),
    ("sorry wrong number i was trying to call the pizza place bye", "non_emergency"),
    ("what time does the post office open on saturday", "non_emergency"),
    ("my neighbors are playing loud music and i can't sleep", "non_emergency"),
]

_SWAP_KEYS = ("addr", "name", "food", "topping", "road", "office", "business", "color")


def perturb(text: str, rng: random.Random) -> str:
    """Swap every known slot value for another one, then add speech-to-text noise."""
    out = text
    for key in _SWAP_KEYS:
        for value in sorted(SLOTS[key], key=len, reverse=True):
            if value in out:
                out = out.replace(value, rng.choice([v for v in SLOTS[key] if v != value]))
    return asr_noise(out, rng, strength=0.5)


def content_checks(text_model, test_rows: list[dict], cfg: dict, seed: int = 7) -> dict:
    checks: dict[str, dict[str, Any]] = {}
    preds = [text_model.predict(t)["label"] for t, _ in CANONICAL]
    misses = [t for (t, y), p in zip(CANONICAL, preds, strict=True) if p != y]
    acc = 1 - len(misses) / len(CANONICAL)
    checks["content_min_functionality"] = {
        "passed": acc >= float(cfg["min_functionality_accuracy"]),
        "accuracy": acc,
        "required": float(cfg["min_functionality_accuracy"]),
        "missed": misses,
    }
    rng = random.Random(seed)
    sample = rng.sample(test_rows, min(150, len(test_rows)))
    agree = [
        text_model.predict(r["text"])["label"]
        == text_model.predict(perturb(r["text"], rng))["label"]
        for r in sample
    ]
    checks["content_invariance_slots_and_asr_noise"] = {
        "passed": float(np.mean(agree)) >= float(cfg["invariance_min_agreement"]),
        "agreement": float(np.mean(agree)),
        "required": float(cfg["invariance_min_agreement"]),
    }
    return checks


def directional_checks(text_model, test_rows: list[dict], fusion_cfg: dict, cfg: dict) -> dict:
    genuine = [r for r in test_rows if r["label"] == "genuine_emergency"]
    covert = [r for r in test_rows if r["label"] == "covert"]
    g_alert = (
        np.mean(
            [
                fuse(text_model.predict(r["text"]), 1.0, fusion_cfg)["threat_level"] == "alert"
                for r in genuine
            ]
        )
        if genuine
        else 0.0
    )
    c_alert = (
        np.mean(
            [
                fuse(text_model.predict(r["text"]), 0.0, fusion_cfg)["threat_level"] == "alert"
                for r in covert
            ]
        )
        if covert
        else 1.0
    )
    return {
        "directional_genuine_emergency_max_stress_no_alert": {
            "passed": float(g_alert) <= float(cfg["directional_max_genuine_alert"]),
            "alert_rate": float(g_alert),
            "maximum": float(cfg["directional_max_genuine_alert"]),
        },
        "directional_covert_calm_voice_still_alerts": {
            "passed": float(c_alert) >= float(cfg["directional_min_covert_alert"]),
            "alert_rate": float(c_alert),
            "minimum": float(cfg["directional_min_covert_alert"]),
        },
    }


def read_speakers(raw_dir: Path) -> dict[str, str]:
    path = raw_dir / "speakers.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["speaker"]: r["sex"] for r in csv.DictReader(fh)}


def voice_slice_checks(
    acoustic,
    manifest_rows: list[dict],
    root: Path,
    raw_dir: Path,
    cfg: dict,
    enforced_channels: set[str],
    seed: int = 11,
) -> dict:
    """Window accuracy / stressed recall per (sex x channel) on held-out test speakers."""
    import soundfile as sf

    from src.augment import phone_line, seed_for
    from src.preprocessing import AudioSpec, FeatureExtractor, sliding_windows

    spec = AudioSpec()
    fx = FeatureExtractor(spec)
    sexes = read_speakers(raw_dir)
    rows = [
        r
        for r in manifest_rows
        if r["split"] == "test"
        and r.get("speaker")
        and not Path(r["path"]).name.startswith("phone_")
        and r["speaker"] in sexes
    ]
    rng = random.Random(seed)
    per_slice, ok, gaps = {}, True, []
    for sex, channel in product(("female", "male"), ("clean", "phone")):
        pool = [r for r in rows if sexes[r["speaker"]] == sex]
        pick = rng.sample(pool, min(int(cfg["clips_per_slice"]), len(pool)))
        if not pick:
            continue
        preds, truth = [], []
        for r in pick:
            y, _ = sf.read(root / r["path"], dtype="float32")
            if channel == "phone":
                y = phone_line(y, np.random.default_rng(seed_for(Path(r["path"]).name)))
            w, _ = sliding_windows(y, spec)
            p = acoustic.predict_proba(fx.transform(w)) >= 0.5
            preds.append(p)
            truth.append(np.full(len(p), r["label"] == "stressed"))
        p, t = np.concatenate(preds), np.concatenate(truth)
        acc = float(np.mean(p == t))
        rec = float(np.mean(p[t])) if t.any() else 0.0
        meets = acc >= float(cfg["slice_min_accuracy"]) and rec >= float(
            cfg["slice_min_recall_stressed"]
        )
        enforced = channel in enforced_channels
        if enforced:
            ok &= meets
        elif not meets:
            gaps.append(f"{sex}/{channel}")
        per_slice[f"{sex}/{channel}"] = {
            "accuracy": acc,
            "recall_stressed": rec,
            "passed": meets,
            "enforced": enforced,
            "clips": len(pick),
        }
    return {
        "voice_slice_floors_sex_x_channel": {
            "passed": ok,
            "slices": per_slice,
            "enforced_channels": sorted(enforced_channels),
            "coverage_gap_warnings": gaps,
            "required_accuracy": float(cfg["slice_min_accuracy"]),
            "required_recall_stressed": float(cfg["slice_min_recall_stressed"]),
        }
    }


def label_of(text: str) -> str:  # tiny helper for tests
    return re.sub(r"\s+", " ", text).strip()
