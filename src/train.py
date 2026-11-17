"""DVC ``train_acoustic`` stage: fit the ResNet-18 voice-stress model.

Reads the windowed Log-Mel features, trains on real speech (CREMA-D: neutral vs
fear), keeps the best epoch on the speaker-disjoint validation set and writes

* models/acoustic.pt               - weights + config (no pickled code)
* reports/acoustic_metrics.json    - best-epoch validation metrics (DVC metrics)
* reports/acoustic_history.json    - per-epoch curves (logged to MLflow by ``package``)

Registration happens in the ``package`` stage, which ships this model together
with the transcript model as one bundle.

    python -m src.train                # params.yaml epochs
    python -m src.train --epochs 1     # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import PROJECT_ROOT, load_params
from src.features import load_split
from src.model import INPUT_KEY, count_parameters, create_model, save_checkpoint

ACOUSTIC_PATH = PROJECT_ROOT / "models" / "acoustic.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def dvc_data_hash() -> str:
    """md5 recorded by `dvc add data/raw` - ties the run to an exact dataset version."""
    dvc_file = PROJECT_ROOT / "data" / "raw.dvc"
    if not dvc_file.exists():
        return "untracked"
    import yaml

    meta = yaml.safe_load(dvc_file.read_text())
    return str(meta.get("outs", [{}])[0].get("md5", "unknown"))


def evaluate(model: nn.Module, loader: DataLoader, loss_fn) -> dict[str, float]:
    model.eval()
    losses, preds, targets = [], [], []
    with torch.inference_mode():
        for xb, yb in loader:
            logits = model(xb)
            losses.append(loss_fn(logits, yb).item() * len(yb))
            preds.append(logits.argmax(1))
            targets.append(yb)
    p, t = torch.cat(preds), torch.cat(targets)
    stressed = t == 1
    return {
        "loss": float(sum(losses) / len(t)),
        "accuracy": float((p == t).float().mean()),
        "recall_stressed": float((p[stressed] == 1).float().mean()) if stressed.any() else 0.0,
    }


def train(params: dict[str, Any], epochs: int | None = None) -> dict:
    tp = params["train"]
    arch = tp["architecture"]
    key = INPUT_KEY[arch]
    epochs = int(epochs if epochs is not None else tp["epochs"])
    set_seed(int(tp["seed"]))
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    tr, va = load_split("train", params), load_split("val", params)
    train_ds = TensorDataset(torch.from_numpy(tr[key]), torch.from_numpy(tr["y"]))
    val_ds = TensorDataset(torch.from_numpy(va[key]), torch.from_numpy(va["y"]))
    g = torch.Generator().manual_seed(int(tp["seed"]))
    train_dl = DataLoader(train_ds, batch_size=int(tp["batch_size"]), shuffle=True, generator=g)
    val_dl = DataLoader(val_ds, batch_size=128)

    model = create_model(
        arch, num_classes=len(params["data"]["classes"]), n_mfcc=int(params["features"]["n_mfcc"])
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(tp["learning_rate"]), weight_decay=float(tp["weight_decay"])
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.CrossEntropyLoss()

    ckpt_config = {
        "architecture": arch,
        "num_classes": len(params["data"]["classes"]),
        "classes": params["data"]["classes"],
        "n_mels": int(params["features"]["n_mels"]),
        "n_mfcc": int(params["features"]["n_mfcc"]),
        "sample_rate": int(params["audio"]["sample_rate"]),
        "window_seconds": float(params["audio"]["window_seconds"]),
        "hop_seconds": float(params["audio"]["hop_seconds"]),
    }

    best = {"val_accuracy": -1.0, "val_loss": float("inf")}
    best_state = None
    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        seen, loss_sum, correct = 0, 0.0, 0
        for xb, yb in train_dl:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            seen += len(yb)
            loss_sum += loss.item() * len(yb)
            correct += int((logits.argmax(1) == yb).sum())
        sched.step()
        val = evaluate(model, val_dl, loss_fn)
        row = {
            "train_loss": loss_sum / seen,
            "train_accuracy": correct / seen,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_recall_stressed": val["recall_stressed"],
            "lr": sched.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"[train] epoch {epoch:02d}/{epochs} "
            + " ".join(f"{k}={v:.4f}" for k, v in row.items())
        )
        better = val["accuracy"] > best["val_accuracy"] or (
            val["accuracy"] == best["val_accuracy"] and val["loss"] < best["val_loss"]
        )
        if better:
            best = {
                "val_accuracy": val["accuracy"],
                "val_loss": val["loss"],
                "epoch": epoch,
                "val_recall_stressed": val["recall_stressed"],
            }
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    save_checkpoint(model, ACOUSTIC_PATH, ckpt_config)
    result: dict[str, Any] = {
        "best_epoch": best["epoch"],
        "best_val_accuracy": best["val_accuracy"],
        "best_val_loss": best["val_loss"],
        "best_val_recall_stressed": best["val_recall_stressed"],
        "train_seconds": round(time.time() - t0, 2),
        "n_parameters": count_parameters(model),
    }

    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (PROJECT_ROOT / "reports" / "acoustic_metrics.json").write_text(json.dumps(result, indent=2))
    (PROJECT_ROOT / "reports" / "acoustic_history.json").write_text(json.dumps(history, indent=2))
    print(
        f"[train] best epoch {best['epoch']} val_acc={best['val_accuracy']:.4f} "
        f"val_recall_stressed={best['val_recall_stressed']:.4f} -> "
        f"{ACOUSTIC_PATH.relative_to(PROJECT_ROOT)}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the voice-stress ResNet-18")
    ap.add_argument("--epochs", type=int, default=None, help="override params.yaml train.epochs")
    args = ap.parse_args(argv)
    train(load_params(), epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
