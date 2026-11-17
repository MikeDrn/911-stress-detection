"""DVC ``quality_gate`` stage: champion vs challenger promotion decision.

Both models are scored on the *same, current* held-out test split (comparing
against a number logged months ago on different data would be meaningless).

Promote the challenger to ``@champion`` only if ALL rules hold:

1. accuracy_challenger >= accuracy_champion + delta         (delta = 0.015)
2. recall(stressed) of the challenger >= 0.92               (missed-danger floor)
3. p95 single-window CPU inference latency <= 150 ms

With no existing champion (first run) rule 1 is skipped (bootstrap), rules 2-3
still apply. If ``quality_gate.auto_promote`` is false the decision is recorded
but a human must promote (``python -m src.quality_gate --promote-version N``).

Source of the champion, in order: MLflow alias ``@champion`` -> local
``models/champion.pt`` (offline fallback, also used when ``--no-mlflow``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import PROJECT_ROOT, load_params, tracking_uri
from src.features import load_split
from src.model import TorchStressPredictor

CHALLENGER_PATH = PROJECT_ROOT / "models" / "challenger.pt"
CHAMPION_PATH = PROJECT_ROOT / "models" / "champion.pt"
REPORT_PATH = PROJECT_ROOT / "reports" / "quality_gate.json"


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = (proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    recall_s = tp / (tp + fn) if tp + fn else 0.0
    precision_s = tp / (tp + fp) if tp + fp else 0.0
    return {
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "recall_stressed": recall_s,
        "recall_calm": tn / (tn + fp) if tn + fp else 0.0,
        "precision_stressed": precision_s,
        "f1_stressed": 2 * precision_s * recall_s / (precision_s + recall_s)
        if precision_s + recall_s
        else 0.0,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def p95_latency_ms(predictor: TorchStressPredictor, sample: np.ndarray, trials: int) -> float:
    """Wall-clock latency of one forward pass on a single window (batch size 1)."""
    x = {predictor.input_key: sample[:1]}
    for _ in range(5):  # warm-up: first calls allocate buffers
        predictor.predict_proba(x)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        predictor.predict_proba(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 95))


def evaluate_predictor(pred: TorchStressPredictor, test: dict, trials: int) -> dict[str, Any]:
    proba = pred.predict_proba({pred.input_key: test[pred.input_key]})
    m = classification_metrics(test["y"], proba)
    m["p95_latency_ms"] = p95_latency_ms(pred, test[pred.input_key], trials)
    return m


def decide(challenger: dict, champion: dict | None, rules: dict) -> dict[str, Any]:
    checks = {}
    if champion is None:
        checks["accuracy_margin"] = {"passed": True, "note": "no champion yet (bootstrap)"}
    else:
        need = champion["accuracy"] + float(rules["min_accuracy_margin"])
        checks["accuracy_margin"] = {
            "passed": challenger["accuracy"] >= need,
            "challenger": challenger["accuracy"],
            "required": need,
        }
    checks["stressed_recall_floor"] = {
        "passed": challenger["recall_stressed"] >= float(rules["min_stressed_recall"]),
        "challenger": challenger["recall_stressed"],
        "required": float(rules["min_stressed_recall"]),
    }
    checks["p95_latency"] = {
        "passed": challenger["p95_latency_ms"] <= float(rules["max_p95_latency_ms"]),
        "challenger_ms": challenger["p95_latency_ms"],
        "budget_ms": float(rules["max_p95_latency_ms"]),
    }
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def _load_champion(params: dict, use_mlflow: bool, tmp: Path):
    """Returns (predictor | None, source description, ModelVersion | None)."""
    if use_mlflow:
        from src import registry

        uri, name = tracking_uri(params), params["mlflow"]["registered_model_name"]
        mv = registry.version_by_alias(uri, name, registry.CHAMPION)
        if mv is not None:
            path = registry.download_checkpoint(uri, name, registry.CHAMPION, tmp / "champion")
            return (
                TorchStressPredictor.from_checkpoint(path, f"v{mv.version}"),
                f"mlflow:{name}@champion(v{mv.version})",
                mv,
            )
        return None, "none (no @champion alias yet)", None
    if CHAMPION_PATH.exists():
        return TorchStressPredictor.from_checkpoint(CHAMPION_PATH, "local"), "file", None
    return None, "none (no local champion.pt)", None


def promote(params: dict, version: int | str | None, use_mlflow: bool) -> None:
    """Point @champion at ``version`` and refresh the local fallback checkpoint."""
    if use_mlflow and version is not None:
        from src import registry

        uri, name = tracking_uri(params), params["mlflow"]["registered_model_name"]
        registry.set_alias(uri, name, registry.CHAMPION, version)
        registry.client(uri).set_model_version_tag(
            name, str(version), "promoted_at", time.strftime("%Y-%m-%dT%H:%M:%S")
        )
        with tempfile.TemporaryDirectory() as td:
            path = registry.download_checkpoint(uri, name, registry.CHAMPION, td)
            shutil.copyfile(path, CHAMPION_PATH)
        print(f"[gate] {name} v{version} is now @champion")
    else:
        shutil.copyfile(CHALLENGER_PATH, CHAMPION_PATH)
        print(f"[gate] copied challenger -> {CHAMPION_PATH.relative_to(PROJECT_ROOT)}")


def run_gate(params: dict, use_mlflow: bool = True) -> dict[str, Any]:
    torch.manual_seed(0)
    rules = params["quality_gate"]
    trials = int(rules["latency_trials"])
    test = load_split("test", params)

    run_info = {}
    run_file = PROJECT_ROOT / "reports" / "train_run.json"
    if run_file.exists():
        run_info = json.loads(run_file.read_text())
    use_mlflow = use_mlflow and run_info.get("mlflow", False)

    challenger = TorchStressPredictor.from_checkpoint(CHALLENGER_PATH, "challenger")
    ch_metrics = evaluate_predictor(challenger, test, trials)

    with tempfile.TemporaryDirectory() as td:
        champ, champ_source, champ_mv = _load_champion(params, use_mlflow, Path(td))
        if champ_mv is not None and str(champ_mv.version) == str(run_info.get("version")):
            print(f"[gate] v{champ_mv.version} is already @champion - nothing to compare")
            champ = None
            champ_source = f"{champ_source} (same version as challenger)"
        champ_metrics = evaluate_predictor(champ, test, trials) if champ else None

    decision = decide(ch_metrics, champ_metrics, rules)
    auto = bool(rules["auto_promote"])
    promoted = decision["passed"] and auto
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_windows": int(len(test["y"])),
        "challenger": {"version": run_info.get("version"), **ch_metrics},
        "champion": {"source": champ_source, **(champ_metrics or {})},
        "decision": decision,
        "auto_promote": auto,
        "promoted": promoted,
    }

    if use_mlflow:
        import mlflow

        from src import registry

        registry.client(tracking_uri(params))
        with mlflow.start_run(run_id=run_info["run_id"]):
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in ch_metrics.items() if isinstance(v, int | float)}
            )
            mlflow.set_tag("quality_gate_passed", str(decision["passed"]))
        registry.client(tracking_uri(params)).set_model_version_tag(
            run_info["model_name"],
            str(run_info["version"]),
            "quality_gate",
            "passed" if decision["passed"] else "failed",
        )

    if promoted:
        promote(params, run_info.get("version"), use_mlflow)
    elif decision["passed"]:
        print("[gate] PASSED but auto_promote=false -> awaiting human approval")
    else:
        print("[gate] FAILED -> challenger stays @challenger; champion unchanged")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    for name, c in decision["checks"].items():
        print(
            f"[gate] {'PASS' if c['passed'] else 'FAIL'}  {name}: "
            + ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in c.items()
                if k != "passed"
            )
        )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Champion/challenger quality gate")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 when the gate fails")
    ap.add_argument("--promote-version", help="manual (human-approved) promotion of version N")
    args = ap.parse_args(argv)
    params = load_params()
    if args.promote_version:
        promote(params, args.promote_version, use_mlflow=not args.no_mlflow)
        return 0
    report = run_gate(params, use_mlflow=not args.no_mlflow)
    return 1 if args.strict and not report["decision"]["passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
