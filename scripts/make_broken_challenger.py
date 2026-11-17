"""Register a deliberately broken @challenger to demo an automatic canary rollback.

Copies the current @champion bundle and sabotages its VOICE model's output layer so
every window is scored "stressed" (a silent failure: HTTP 200, confident, wrong -
Course 7 slide 7). The bundle is registered as a new version with alias @challenger.
A canary then catches it with the class-collapse guardrail and rolls back.

    uv run python -m scripts.make_broken_challenger            # create + alias it
    uv run python -m scripts.make_broken_challenger --restore  # alias back to the real one

Never point @champion at this version.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch

from src import registry
from src.bundle import BUNDLE_FILE
from src.config import PROJECT_ROOT, load_params, tracking_uri
from src.model import load_checkpoint, save_checkpoint


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--restore", action="store_true", help="point @challenger back to the version it had before"
    )
    args = ap.parse_args(argv)
    params = load_params()
    uri, name = tracking_uri(params), params["mlflow"]["registered_model_name"]
    client = registry.client(uri)

    if args.restore:
        prev = client.get_registered_model(name).tags.get("challenger_before_sabotage")
        if not prev:
            raise SystemExit("nothing to restore")
        registry.set_alias(uri, name, registry.CHALLENGER, prev)
        print(f"[broken] @challenger restored to v{prev}")
        return 0

    import mlflow

    from src.mlflow_model import CallPyfunc

    current = registry.version_by_alias(uri, name, registry.CHALLENGER)
    if current is not None:
        client.set_registered_model_tag(name, "challenger_before_sabotage", current.version)
    with tempfile.TemporaryDirectory() as td:
        bundle = registry.download_bundle(uri, name, registry.CHAMPION, Path(td) / "src")
        out = Path(td) / "broken_bundle"
        shutil.copytree(bundle, out)
        model, cfg = load_checkpoint(out / "acoustic.pt")
        with torch.no_grad():
            linear = model.fc[-1]
            linear.weight.mul_(0.0)
            linear.bias.copy_(torch.tensor([-6.0, 6.0]))  # P(stressed) ~ 1.0 for every window
        save_checkpoint(model, out / "acoustic.pt", cfg)
        meta = json.loads((out / BUNDLE_FILE).read_text())
        meta["sabotaged"] = True
        (out / BUNDLE_FILE).write_text(json.dumps(meta, indent=2))

        mlflow.set_experiment(params["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="SABOTAGED-canary-demo") as run:
            mlflow.set_tags(
                {"sabotaged": "true", "purpose": "demo of automatic canary rollback (Course 4)"}
            )
            mlflow.pyfunc.log_model(
                name="model",
                python_model=CallPyfunc(),
                artifacts={"bundle": str(out)},
                code_paths=[str(PROJECT_ROOT / "src")],
                pip_requirements=["torch", "torchvision", "numpy", "librosa", "soundfile"],
                registered_model_name=name,
            )
    versions = client.search_model_versions(f"name='{name}' and run_id='{run.info.run_id}'")
    version = max(int(v.version) for v in versions)
    client.set_model_version_tag(name, str(version), "sabotaged", "true")
    registry.set_alias(uri, name, registry.CHALLENGER, version)
    print(
        f"[broken] registered SABOTAGED v{version} as @challenger "
        f"(previous challenger: v{current.version if current else '-'})"
    )
    print("[broken] the API picks it up within 30 s (or click 'Hot-reload models').")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
