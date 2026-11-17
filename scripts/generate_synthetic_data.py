"""Generate the synthetic calm / stressed training corpus into data/raw/<class>/.

    uv run python -m scripts.generate_synthetic_data                 # params.yaml sizes
    uv run python -m scripts.generate_synthetic_data --clips-per-class 10   # quick CI run
    uv run python -m scripts.generate_synthetic_data --channel phone        # dataset v2 add-on

``--channel phone`` simulates the data-shift scenario used in the Chapter 4/7
demo: new calls arrive over narrow-band, noisy mobile lines. The clips are
written next to the originals as ``phone_<class>_NNNN.wav`` with their own seed,
so the original clean files are never modified.

Deterministic: the same seed always produces byte-identical WAV files, so DVC
hashes stay stable across machines.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from src.config import PROJECT_ROOT, load_params
from src.synthetic import synth_utterance


def main(argv: list[str] | None = None) -> int:
    params = load_params()
    syn, audio = params["synthetic"], params["audio"]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=params["data"]["raw_dir"])
    ap.add_argument("--clips-per-class", type=int, default=syn["clips_per_class"])
    ap.add_argument("--seed", type=int, default=None, help="default: params seed (+1 for phone)")
    ap.add_argument("--channel", choices=["clean", "phone"], default="clean")
    ap.add_argument("--clean", action="store_true", help="delete the output dir first")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    if args.clean and out.exists():
        shutil.rmtree(out)
    sr = int(audio["sample_rate"])
    seed = args.seed if args.seed is not None else syn["seed"] + (args.channel == "phone")
    prefix = "" if args.channel == "clean" else f"{args.channel}_"
    rng = np.random.default_rng(seed)
    for label in params["data"]["classes"]:
        (out / label).mkdir(parents=True, exist_ok=True)
        for i in range(args.clips_per_class):
            dur = float(rng.uniform(syn["min_clip_seconds"], syn["max_clip_seconds"]))
            y = synth_utterance(
                dur, stressed=(label == "stressed"), rng=rng, sr=sr, channel=args.channel
            )
            sf.write(out / label / f"{prefix}{label}_{i:04d}.wav", y, sr, subtype="PCM_16")
        print(f"[synth] {args.clips_per_class} {args.channel} '{label}' clips -> {out / label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
