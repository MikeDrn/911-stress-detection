"""DVC ``quality_gate`` stage: champion vs challenger *bundles* and the promotion decision.

What is evaluated
-----------------
Both bundles are scored on the same, current held-out data:

* voice model      - window accuracy / stressed recall on held-out SPEAKERS (CREMA-D)
* content model    - transcripts from held-out scenario FAMILIES
* the full system  - simulated test calls pairing each held-out transcript with a
                     held-out speaker's real recording. Stress profile assumption
                     (configurable): genuine emergencies are mostly frightened voices,
                     covert callers often tense, non-emergency callers mostly calm.
                     The pairing is seeded and identical for both bundles.

Promotion rules (all must pass)
-------------------------------
1. threat-decision accuracy >= champion + delta          (delta = 0.015)
2. covert-call recall      >= 0.90                        (missed danger)
3. false alarms on GENUINE emergencies <= 5%              (stress alone is not a threat)
4. voice-model window accuracy >= floor                   (the voice model learned something)
5. p95 single-window voice latency <= 150 ms
6. behavioural tests: content min-functionality and invariance, the two directional
   911 rules, voice slice floors (sex x channel, enforced for trained channels)

The report also states the standard error of the accuracy (Course 7 slide 13) and
the sanity checks (keyword rule, shuffled labels) that make the numbers believable.

Continuous delivery: with ``auto_promote: false`` a passing bundle is tagged
``awaiting_approval``; a human runs ``--promote-challenger``; ``--rollback`` undoes it.
The very first bundle is promoted automatically (bootstrap: nothing to protect yet).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.bundle import CallModel, load_bundle
from src.config import PROJECT_ROOT, load_params, tracking_uri
from src.data_contracts import read_manifest
from src.features import load_split
from src.fusion import fuse
from src.transcripts import read_csv

CHALLENGER_DIR = PROJECT_ROOT / "models" / "challenger_bundle"
CHAMPION_DIR = PROJECT_ROOT / "models" / "champion_bundle"
REPORT_PATH = PROJECT_ROOT / "reports" / "quality_gate.json"


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def pair_calls(
    text_rows: list[dict], clip_labels: dict[int, str], profile: dict, seed: int = 2026
) -> list[tuple[dict, int]]:
    """Give every held-out transcript a held-out recording (seeded, model-independent)."""
    rng = random.Random(seed)
    stressed = sorted(c for c, lab in clip_labels.items() if lab == "stressed")
    calm = sorted(c for c, lab in clip_labels.items() if lab == "calm")
    pairs = []
    for row in text_rows:
        pool = stressed if rng.random() < float(profile[row["label"]]) else calm
        pairs.append((row, rng.choice(pool)))
    return pairs


def p95_ms(fn, trials: int) -> float:
    for _ in range(3):
        fn()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 95))


def evaluate_bundle(
    model: CallModel, test: dict, pairs: list, fusion_cfg: dict, trials: int
) -> dict[str, Any]:
    probs = model.predict_proba({model.input_key: test[model.input_key]})
    win_pred, y = probs >= 0.5, test["y"].astype(bool)
    stress_by_clip: dict[int, float] = {}
    for clip in np.unique(test["clip"]):
        stress_by_clip[int(clip)] = float(probs[test["clip"] == clip].mean())

    text_cache: dict[str, dict] = {}
    decisions = []
    for row, clip in pairs:
        if row["text"] not in text_cache:
            text_cache[row["text"]] = model.text.predict(row["text"])
        fused = fuse(text_cache[row["text"]], stress_by_clip[clip], fusion_cfg)
        decisions.append((row["label"], fused["threat_level"]))

    def rate(label: str, level: str = "alert") -> float:
        rows = [lv for lab, lv in decisions if lab == label]
        return float(np.mean([lv == level for lv in rows])) if rows else 0.0

    truth = np.array([lab == "covert" for lab, _ in decisions])
    alert = np.array([lv == "alert" for _, lv in decisions])
    tp = int((alert & truth).sum())
    one = {model.input_key: test[model.input_key][:1]}
    sample_text = pairs[0][0]["text"] if pairs else "hello"
    return {
        "threat_accuracy": float(np.mean(alert == truth)),
        "covert_recall": rate("covert"),
        "covert_review_or_alert": rate("covert") + rate("covert", "review"),
        "alert_precision": tp / max(1, int(alert.sum())),
        "genuine_emergency_false_alarm_rate": rate("genuine_emergency"),
        "non_emergency_false_alarm_rate": rate("non_emergency"),
        "voice_window_accuracy": float(np.mean(win_pred == y)),
        "voice_recall_stressed": float(np.mean(win_pred[y])) if y.any() else 0.0,
        "voice_recall_calm": float(np.mean(~win_pred[~y])) if (~y).any() else 0.0,
        "text_accuracy": float(
            np.mean([text_cache[r["text"]]["label"] == r["label"] for r, _ in pairs])
        ),
        "simulated_calls": len(pairs),
        "p95_voice_latency_ms": p95_ms(lambda: model.predict_proba(one), trials),
        "p95_text_latency_ms": p95_ms(lambda: model.text.predict(sample_text), trials),
    }


def decide(ch: dict, champ: dict | None, rules: dict, behavioural: dict) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    if champ is None:
        checks["accuracy_margin"] = {"passed": True, "note": "no champion yet (bootstrap)"}
    else:
        need = champ["threat_accuracy"] + float(rules["min_accuracy_margin"])
        checks["accuracy_margin"] = {
            "passed": ch["threat_accuracy"] >= need,
            "challenger": ch["threat_accuracy"],
            "required": need,
        }
    checks["covert_recall_floor"] = {
        "passed": ch["covert_recall"] >= float(rules["min_covert_recall"]),
        "challenger": ch["covert_recall"],
        "required": float(rules["min_covert_recall"]),
    }
    checks["genuine_emergency_false_alarm_ceiling"] = {
        "passed": ch["genuine_emergency_false_alarm_rate"]
        <= float(rules["max_genuine_false_alarm"]),
        "challenger": ch["genuine_emergency_false_alarm_rate"],
        "maximum": float(rules["max_genuine_false_alarm"]),
    }
    checks["voice_model_accuracy_floor"] = {
        "passed": ch["voice_window_accuracy"] >= float(rules["min_voice_accuracy"]),
        "challenger": ch["voice_window_accuracy"],
        "required": float(rules["min_voice_accuracy"]),
    }
    checks["p95_latency"] = {
        "passed": ch["p95_voice_latency_ms"] <= float(rules["max_p95_latency_ms"]),
        "challenger_ms": ch["p95_voice_latency_ms"],
        "budget_ms": float(rules["max_p95_latency_ms"]),
    }
    for name, check in behavioural.items():
        checks[f"behavioural_{name}"] = check
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def measurement_noise(accuracy: float, n: int, delta: float) -> dict[str, Any]:
    """Standard error of the threat accuracy (Course 7 slide 13): is delta above the noise?"""
    se = (accuracy * (1 - accuracy) / max(n, 1)) ** 0.5
    return {
        "accuracy": accuracy,
        "n_simulated_calls": n,
        "standard_error": se,
        "two_standard_errors": 2 * se,
        "delta": delta,
        "delta_exceeds_2se": delta >= 2 * se,
    }


# --------------------------------------------------------------------------- #
# Champion lookup / promotion / rollback
# --------------------------------------------------------------------------- #
def _load_champion(params: dict, use_mlflow: bool, tmp: Path):
    if use_mlflow:
        from src import registry

        uri, name = tracking_uri(params), params["mlflow"]["registered_model_name"]
        mv = registry.version_by_alias(uri, name, registry.CHAMPION)
        if mv is not None:
            path = registry.download_bundle(uri, name, registry.CHAMPION, tmp / "champion")
            return (
                load_bundle(path, f"v{mv.version}", params["fusion"]),
                f"mlflow:{name}@champion(v{mv.version})",
                mv,
            )
        return None, "none (no @champion alias yet)", None
    if CHAMPION_DIR.exists():
        return load_bundle(CHAMPION_DIR, "local", params["fusion"]), "file", None
    return None, "none (no local champion bundle)", None


def promote(params: dict, version: int | str | None, use_mlflow: bool) -> None:
    """Point @champion at ``version`` and refresh the local fallback bundle."""
    if use_mlflow and version is not None:
        from src import registry

        uri, name = tracking_uri(params), params["mlflow"]["registered_model_name"]
        c = registry.client(uri)
        current = registry.version_by_alias(uri, name, registry.CHAMPION)
        if current is not None and str(current.version) == str(version):
            print(f"[gate] v{version} is already @champion")
            return
        if current is not None:  # remembered for a one-command rollback
            c.set_registered_model_tag(name, "previous_champion", str(current.version))
        registry.set_alias(uri, name, registry.CHAMPION, version)
        c.set_model_version_tag(
            name, str(version), "promoted_at", time.strftime("%Y-%m-%dT%H:%M:%S")
        )
        c.set_model_version_tag(name, str(version), "approval", "approved")
        with tempfile.TemporaryDirectory() as td:
            src = registry.download_bundle(uri, name, registry.CHAMPION, td)
            if CHAMPION_DIR.exists():
                shutil.rmtree(CHAMPION_DIR)
            shutil.copytree(src, CHAMPION_DIR)
        print(
            f"[gate] {name} v{version} is now @champion"
            + (f" (previous: v{current.version})" if current is not None else "")
        )
    else:
        if CHAMPION_DIR.exists():
            shutil.rmtree(CHAMPION_DIR)
        shutil.copytree(CHALLENGER_DIR, CHAMPION_DIR)
        print(f"[gate] copied challenger -> {CHAMPION_DIR.relative_to(PROJECT_ROOT)}")


def rollback(params: dict) -> None:
    """Course 7 slide 24: rollback = move @champion back to the previous version."""
    from src import registry

    uri, name = tracking_uri(params), params["mlflow"]["registered_model_name"]
    c = registry.client(uri)
    prev = c.get_registered_model(name).tags.get("previous_champion")
    if not prev:
        raise SystemExit("[gate] no previous_champion recorded - nothing to roll back to")
    current = registry.version_by_alias(uri, name, registry.CHAMPION)
    promote(params, prev, use_mlflow=True)
    if current is not None:
        c.set_model_version_tag(name, str(current.version), "approval", "rolled_back")
    print(f"[gate] ROLLBACK: @champion -> v{prev}")


def promote_challenger(params: dict) -> None:
    """Human approval step: promote whatever version currently holds @challenger."""
    from src import registry

    uri, name = tracking_uri(params), params["mlflow"]["registered_model_name"]
    mv = registry.version_by_alias(uri, name, registry.CHALLENGER)
    if mv is None:
        raise SystemExit("[gate] no @challenger alias")
    promote(params, mv.version, use_mlflow=True)


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def run_gate(params: dict, use_mlflow: bool = True) -> dict[str, Any]:
    from src.behavioural import content_checks, directional_checks, voice_slice_checks
    from src.package import training_channels

    torch.manual_seed(0)
    rules = params["quality_gate"]
    trials = int(rules["latency_trials"])
    fusion_cfg = params["fusion"]
    test = load_split("test", params)
    manifest = read_manifest(PROJECT_ROOT / params["data"]["manifest"])
    clip_labels = {i: r["label"] for i, r in enumerate(manifest) if r["split"] == "test"}
    text_rows = [
        r for r in read_csv(PROJECT_ROOT / params["text"]["transcripts"]) if r["split"] == "test"
    ]
    pairs = pair_calls(text_rows, clip_labels, rules["stress_profile"])

    run_info = {}
    run_file = PROJECT_ROOT / "reports" / "train_run.json"
    if run_file.exists():
        run_info = json.loads(run_file.read_text())
    use_mlflow = use_mlflow and run_info.get("mlflow", False)

    challenger = load_bundle(CHALLENGER_DIR, "challenger", fusion_cfg)
    ch_metrics = evaluate_bundle(challenger, test, pairs, fusion_cfg, trials)
    b = rules["behavioural"]
    behavioural = {
        **content_checks(challenger.text, text_rows, b),
        **directional_checks(challenger.text, text_rows, fusion_cfg, b),
        **voice_slice_checks(
            challenger.acoustic,
            manifest,
            PROJECT_ROOT,
            PROJECT_ROOT / params["data"]["raw_dir"],
            b,
            set(training_channels(params)),
        ),
    }

    with tempfile.TemporaryDirectory() as td:
        champ, champ_source, champ_mv = _load_champion(params, use_mlflow, Path(td))
        same_as_champion = champ_mv is not None and str(champ_mv.version) == str(
            run_info.get("version")
        )
        if same_as_champion:
            print(f"[gate] v{champ_mv.version} is already @champion - re-checking it (report only)")
            champ = None
        champ_metrics = evaluate_bundle(champ, test, pairs, fusion_cfg, trials) if champ else None

    decision = decide(ch_metrics, champ_metrics, rules, behavioural)
    if same_as_champion:
        decision["checks"]["accuracy_margin"]["note"] = "this version is the current champion"
    auto = bool(rules["auto_promote"])
    bootstrap = champ_metrics is None and champ_source.startswith("none")
    promoted = decision["passed"] and (auto or bootstrap)
    text_metrics_path = PROJECT_ROOT / "reports" / "text_metrics.json"
    text_metrics = json.loads(text_metrics_path.read_text()) if text_metrics_path.exists() else {}
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "challenger": {"version": run_info.get("version"), **ch_metrics},
        "champion": {"source": champ_source, **(champ_metrics or {})},
        "decision": decision,
        "measurement_noise": measurement_noise(
            (champ_metrics or ch_metrics)["threat_accuracy"],
            len(pairs),
            float(rules["min_accuracy_margin"]),
        ),
        "sanity_checks": {
            "keyword_rule_test": text_metrics.get("keyword_baseline_test"),
            "shuffled_labels_text_accuracy": text_metrics.get(
                "shuffled_label_control_test_accuracy"
            ),
        },
        "stress_profile_assumption": rules["stress_profile"],
        "auto_promote": auto,
        "bootstrap": bootstrap,
        "promoted": promoted,
    }

    if use_mlflow:
        import mlflow

        from src import registry

        registry.client(tracking_uri(params))
        with mlflow.start_run(run_id=run_info["run_id"]):
            mlflow.log_metrics(
                {f"gate_{k}": v for k, v in ch_metrics.items() if isinstance(v, int | float)}
            )
            mlflow.set_tag("quality_gate_passed", str(decision["passed"]))
            mlflow.log_dict(
                json.loads(json.dumps(report, default=str)), "gate/quality_gate_report.json"
            )
        c = registry.client(tracking_uri(params))
        c.set_model_version_tag(
            run_info["model_name"],
            str(run_info["version"]),
            "quality_gate",
            "passed" if decision["passed"] else "failed",
        )
        if decision["passed"] and not auto and not same_as_champion and not promoted:
            c.set_model_version_tag(
                run_info["model_name"], str(run_info["version"]), "approval", "awaiting_approval"
            )

    if promoted:
        if bootstrap and not auto:
            print("[gate] bootstrap: no champion in production yet -> promoting the first bundle")
        promote(params, run_info.get("version"), use_mlflow)
    elif same_as_champion:
        print("[gate] report only - this version is already in production")
    elif decision["passed"]:
        print("[gate] PASSED but auto_promote=false -> awaiting human approval")
    else:
        print("[gate] FAILED -> challenger stays @challenger; champion unchanged")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    _print_report(report, champ_metrics)
    return report


def _print_report(report: dict, champ: dict | None) -> None:
    ch = report["challenger"]
    print(
        f"[gate] challenger: threat acc={ch['threat_accuracy']:.3f}  covert recall="
        f"{ch['covert_recall']:.3f}  false alarms: genuine="
        f"{ch['genuine_emergency_false_alarm_rate']:.3f} non-emergency="
        f"{ch['non_emergency_false_alarm_rate']:.3f}  voice acc="
        f"{ch['voice_window_accuracy']:.3f}  text acc={ch['text_accuracy']:.3f}"
    )
    if champ:
        print(
            f"[gate] champion:   threat acc={champ['threat_accuracy']:.3f}  covert recall="
            f"{champ['covert_recall']:.3f}  voice acc={champ['voice_window_accuracy']:.3f}"
        )
    for name, c in report["decision"]["checks"].items():
        details = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in c.items()
            if k != "passed" and not isinstance(v, dict | list)
        )
        print(f"[gate] {'PASS' if c['passed'] else 'FAIL'}  {name}: {details}")
        for sl, v in c.get("slices", {}).items():
            status = "ok" if v["passed"] else "BELOW FLOOR"
            if not v["passed"] and not v.get("enforced", True):
                status = "WARNING: coverage gap (no training data for this channel yet)"
            print(
                f"[gate]        slice {sl:13s} acc={v['accuracy']:.3f} "
                f"recall_stressed={v['recall_stressed']:.3f} {status}"
            )
        for miss in c.get("missed", []):
            print(f"[gate]        missed: {miss!r}")
    n = report["measurement_noise"]
    print(
        f"[gate] noise: SE={n['standard_error']:.4f}  2*SE={n['two_standard_errors']:.4f}  "
        f"delta={n['delta']:.3f}"
    )
    s = report["sanity_checks"]
    if s.get("keyword_rule_test"):
        print(
            f"[gate] sanity: keyword rule covert recall="
            f"{s['keyword_rule_test']['covert_recall']:.3f} | shuffled-label text acc="
            f"{s['shuffled_labels_text_accuracy']:.3f} (chance 0.333)"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Champion/challenger quality gate")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 when the gate fails")
    ap.add_argument("--promote-version", help="manual (human-approved) promotion of version N")
    ap.add_argument(
        "--promote-challenger",
        action="store_true",
        help="human approval: promote the current @challenger",
    )
    ap.add_argument(
        "--rollback",
        action="store_true",
        help="move @champion back to the previously promoted version",
    )
    args = ap.parse_args(argv)
    params = load_params()
    if args.rollback:
        rollback(params)
        return 0
    if args.promote_challenger:
        promote_challenger(params)
        return 0
    if args.promote_version:
        promote(params, args.promote_version, use_mlflow=not args.no_mlflow)
        return 0
    report = run_gate(params, use_mlflow=not args.no_mlflow)
    return 1 if args.strict and not report["decision"]["passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
