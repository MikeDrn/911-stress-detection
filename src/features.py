"""DVC ``featurize`` stage: manifest -> windowed Log-Mel / MFCC tensors.

Outputs ``data/features/{train,val,test}.npz`` with arrays

* ``logmel``  float32 (n_windows, 1, n_mels, frames)
* ``mfcc``    float32 (n_windows, n_mfcc, frames)
* ``y``       int64   (n_windows,)  window label inherited from its clip
* ``clip``    int64   (n_windows,)  index of the source row in the manifest

plus ``summary.json``. Splitting happened at clip level *before* windowing, so
overlapping windows of one recording never leak across splits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, load_params
from src.data_contracts import SPLITS, read_manifest
from src.preprocessing import AudioSpec, FeatureExtractor, load_audio, sliding_windows


def featurize(params: dict | None = None) -> dict:
    params = params or load_params()
    audio_spec = AudioSpec.from_params(params)
    extractor = FeatureExtractor(audio_spec)
    rows = read_manifest(PROJECT_ROOT / params["data"]["manifest"])
    out_dir = PROJECT_ROOT / params["data"]["features_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"n_frames": extractor.n_frames, "splits": {}}
    for split in SPLITS:
        logmels, mfccs, ys, clips = [], [], [], []
        for clip_idx, row in enumerate(rows):
            if row["split"] != split:
                continue
            y = load_audio(PROJECT_ROOT / row["path"], audio_spec)
            windows, _ = sliding_windows(y, audio_spec)
            feats = extractor.transform(windows)
            logmels.append(feats["logmel"])
            mfccs.append(feats["mfcc"])
            ys.append(np.full(len(windows), int(row["label_idx"]), dtype=np.int64))
            clips.append(np.full(len(windows), clip_idx, dtype=np.int64))
        if not ys:
            raise SystemExit(f"[featurize] split '{split}' is empty - run the validate stage")
        arrays = {
            "logmel": np.concatenate(logmels),
            "mfcc": np.concatenate(mfccs),
            "y": np.concatenate(ys),
            "clip": np.concatenate(clips),
        }
        np.savez_compressed(out_dir / f"{split}.npz", **arrays)
        counts = np.bincount(arrays["y"], minlength=len(params["data"]["classes"]))
        summary["splits"][split] = {
            "windows": int(len(arrays["y"])),
            "clips": int(len(np.unique(arrays["clip"]))),
            "windows_per_class": dict(zip(params["data"]["classes"], counts.tolist(), strict=True)),
            "logmel_shape": list(arrays["logmel"].shape[1:]),
            "mfcc_shape": list(arrays["mfcc"].shape[1:]),
        }
        print(f"[featurize] {split:5s} {summary['splits'][split]}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_split(split: str, params: dict | None = None) -> dict[str, np.ndarray]:
    params = params or load_params()
    path = Path(PROJECT_ROOT / params["data"]["features_dir"] / f"{split}.npz")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


if __name__ == "__main__":
    featurize()
