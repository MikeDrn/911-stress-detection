"""DVC ``package`` stage: bundle voice + content models, log ONE MLflow run, register.

Every bundle version in the registry records, in a single run:

* params   - audio/feature/voice-training params, text-model params, fusion rule
* metrics  - per-epoch voice curves, voice validation metrics, transcript metrics
             on unseen families, and the sanity checks (keyword baseline,
             shuffled-label control)
* tags     - git commit, DVC data hash, which telephone channels were trained on
* model    - the bundle (pyfunc) registered as a new version with alias @challenger

Promotion is never done here; that is the quality gate's (and a human's) job.

    python -m src.package               # log to $MLFLOW_TRACKING_URI
    python -m src.package --no-mlflow   # offline: only writes models/challenger_bundle/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.bundle import save_bundle
from src.config import PROJECT_ROOT, load_params, tracking_uri
from src.train import dvc_data_hash, flatten, git_commit

BUNDLE_DIR = PROJECT_ROOT / "models" / "challenger_bundle"
RUN_FILE = PROJECT_ROOT / "reports" / "train_run.json"


def training_channels(params: dict) -> list[str]:
    from src.data_contracts import read_manifest

    manifest = PROJECT_ROOT / params["data"]["manifest"]
    if not manifest.exists():
        return ["clean"]
    return sorted(
        {
            "phone" if Path(r["path"]).name.startswith("phone_") else "clean"
            for r in read_manifest(manifest)
            if r["split"] == "train"
        }
    )


def package(params: dict[str, Any], use_mlflow: bool = True) -> dict[str, Any]:
    reports = PROJECT_ROOT / "reports"
    acoustic = json.loads((reports / "acoustic_metrics.json").read_text())
    history = json.loads((reports / "acoustic_history.json").read_text())
    text = json.loads((reports / "text_metrics.json").read_text())
    channels = training_channels(params)
    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trained_channels": channels,
        "acoustic_architecture": params["train"]["architecture"],
        "git_commit": git_commit(),
        "dvc_data_md5": dvc_data_hash(),
    }
    save_bundle(
        BUNDLE_DIR,
        PROJECT_ROOT / "models" / "acoustic.pt",
        PROJECT_ROOT / "models" / "text_model.json",
        params["fusion"],
        meta,
    )
    print(f"[package] bundle -> {BUNDLE_DIR.relative_to(PROJECT_ROOT)} (channels: {channels})")

    info: dict[str, Any] = {"mlflow": False}
    if use_mlflow:
        info = _log_and_register(params, acoustic, history, text, meta)
    RUN_FILE.parent.mkdir(exist_ok=True)
    RUN_FILE.write_text(json.dumps(info, indent=2))
    return info


def _log_and_register(params, acoustic, history, text, meta) -> dict[str, Any]:
    import mlflow

    from src import registry
    from src.mlflow_model import CallPyfunc

    mlflow.set_tracking_uri(tracking_uri(params))
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    name = params["mlflow"]["registered_model_name"]
    with mlflow.start_run(run_name=f"bundle-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.log_params(flatten({k: params[k] for k in ("audio", "features", "train")}))
        mlflow.log_params(flatten({"text": params["text"]["model"], "fusion": params["fusion"]}))
        mlflow.set_tags(
            {
                "git_commit": meta["git_commit"],
                "dvc_data_md5": meta["dvc_data_md5"],
                "trained_channels": ",".join(meta["trained_channels"]),
                "acoustic_architecture": meta["acoustic_architecture"],
            }
        )
        for epoch, row in enumerate(history, 1):
            mlflow.log_metrics({f"voice_{k}": v for k, v in row.items()}, step=epoch)
        mlflow.log_metrics(
            {f"voice_{k}": v for k, v in acoustic.items() if isinstance(v, int | float)}
        )
        t = text["test_heldout_families"]
        mlflow.log_metrics(
            {
                "text_test_accuracy": t["accuracy"],
                **{f"text_test_recall_{c}": v for c, v in t["recall"].items() if v is not None},
                "text_val_accuracy": text["val"]["accuracy"],
                "sanity_keyword_rule_accuracy": text["keyword_baseline_test"]["accuracy"],
                "sanity_keyword_rule_covert_recall": text["keyword_baseline_test"]["covert_recall"],
                "sanity_shuffled_labels_accuracy": text["shuffled_label_control_test_accuracy"],
            }
        )
        mlflow.log_artifact(str(PROJECT_ROOT / "params.yaml"))
        manifest = PROJECT_ROOT / params["data"]["manifest"]
        if manifest.exists():
            mlflow.log_artifact(str(manifest), artifact_path="data")
        mlflow.pyfunc.log_model(
            name="model",
            python_model=CallPyfunc(),
            artifacts={"bundle": str(BUNDLE_DIR)},
            code_paths=[str(PROJECT_ROOT / "src")],
            pip_requirements=[
                "torch",
                "torchvision",
                "numpy",
                "librosa",
                "soundfile",
                "pyyaml",
                "pandas",
            ],
            registered_model_name=name,
        )
        run_id = run.info.run_id
    c = registry.client(mlflow.get_tracking_uri())
    versions = c.search_model_versions(f"name='{name}' and run_id='{run_id}'")
    version = max(int(v.version) for v in versions)
    registry.set_alias(mlflow.get_tracking_uri(), name, registry.CHALLENGER, version)
    print(f"[package] registered {name} v{version} -> alias @{registry.CHALLENGER}")
    return {"mlflow": True, "run_id": run_id, "model_name": name, "version": version}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bundle and register the call model")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)
    package(load_params(), use_mlflow=not args.no_mlflow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
