"""Sanity report: are the accuracy numbers believable? (one-off, ~2 minutes)

A high accuracy only means something if it survives three checks:

1. Shuffled-label control - the SAME model trained on randomly shuffled labels must
   fall to chance. If it does not, test information leaks into training.
2. Simple baseline - a much simpler model (linear model on the average spectrum;
   a keyword rule for transcripts). The real model must clearly beat it, otherwise
   the task is too easy or the model learned a shortcut.
3. Held-out groups - test speakers / scenario families never seen in training
   (enforced by the data contracts; repeated here for the record).

    uv run python -m scripts.sanity_report      # writes reports/sanity_report.json
"""

from __future__ import annotations

import json

import numpy as np
import torch

from src.config import PROJECT_ROOT, load_params
from src.data_contracts import check_speaker_leakage, read_manifest
from src.features import load_split
from src.model import TorchStressPredictor, create_model


def shuffled_voice_control(tr: dict, te: dict, max_windows: int = 900, epochs: int = 3) -> float:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(tr["y"]), min(max_windows, len(tr["y"])), replace=False)
    X = torch.from_numpy(tr["logmel"][idx])
    Y = torch.from_numpy(rng.permutation(tr["y"][idx]))
    torch.manual_seed(0)
    model = create_model("resnet18_2d")
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            b = perm[i : i + 32]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(X[b]), Y[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(te["logmel"])).argmax(1).numpy()
    return float(np.mean(pred == te["y"]))


def main() -> int:
    from sklearn.linear_model import LogisticRegression

    params = load_params()
    tr, te = load_split("train", params), load_split("test", params)
    real = TorchStressPredictor.from_checkpoint(PROJECT_ROOT / "models" / "acoustic.pt")
    real_acc = float(np.mean((real.predict_proba({"logmel": te["logmel"]}) >= 0.5) == te["y"]))
    lin = LogisticRegression(max_iter=3000).fit(tr["logmel"][:, 0].mean(axis=2), tr["y"])
    lin_acc = float(lin.score(te["logmel"][:, 0].mean(axis=2), te["y"]))
    print("[sanity] training the voice model on SHUFFLED labels (about 1-2 minutes) ...")
    shuf = shuffled_voice_control(tr, te)
    manifest = read_manifest(PROJECT_ROOT / params["data"]["manifest"])
    speakers = {
        s: {r["split"] for r in manifest if r["speaker"] == s}
        for s in {r["speaker"] for r in manifest if r["speaker"]}
    }
    text = json.loads((PROJECT_ROOT / "reports" / "text_metrics.json").read_text())

    report = {
        "voice": {
            "real_model_test_accuracy": real_acc,
            "linear_baseline_average_spectrum": lin_acc,
            "shuffled_label_control": shuf,
            "chance": float(max(np.mean(te["y"]), 1 - np.mean(te["y"]))),
            "test_speakers": sorted(s for s, sp in speakers.items() if sp == {"test"}),
            "speaker_leakage": check_speaker_leakage(manifest),
        },
        "content": {
            "real_model_test_accuracy_unseen_families": text["test_heldout_families"]["accuracy"],
            "keyword_rule": text["keyword_baseline_test"],
            "shuffled_label_control": text["shuffled_label_control_test_accuracy"],
            "chance": 1 / 3,
        },
    }
    out = PROJECT_ROOT / "reports" / "sanity_report.json"
    out.write_text(json.dumps(report, indent=2))
    v, c = report["voice"], report["content"]
    leak = v["speaker_leakage"] or "none"
    kw = c["keyword_rule"]
    print(
        f"[sanity] VOICE   real={v['real_model_test_accuracy']:.3f}  linear baseline="
        f"{v['linear_baseline_average_spectrum']:.3f}  shuffled labels="
        f"{v['shuffled_label_control']:.3f}  (majority class {v['chance']:.3f}) | "
        f"{len(v['test_speakers'])} unseen test speakers, leakage={leak}"
    )
    real_c = c["real_model_test_accuracy_unseen_families"]
    print(
        f"[sanity] CONTENT real={real_c:.3f}  keyword rule="
        f"{kw['accuracy']:.3f} (covert recall {kw['covert_recall']:.3f})"
        f"  shuffled labels={c['shuffled_label_control']:.3f}  (chance 0.333)"
    )
    print(f"[sanity] -> {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
