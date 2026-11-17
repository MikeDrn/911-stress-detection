"""Create phone-line versions of the CREMA-D clips (dataset v2 for the CT demo).

For every data/raw/<class>/cremad_*.wav this writes data/raw/<class>/phone_cremad_*.wav.
The speaker id is preserved in the name, so an original and its phone copy always
land in the same split (no speaker leakage). Deterministic: re-running produces
byte-identical files, so DVC hashes are stable.

    uv run python -m scripts.augment_phone
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

from src.augment import phone_line, seed_for
from src.config import PROJECT_ROOT, load_params


def main() -> int:
    params = load_params()
    raw = PROJECT_ROOT / params["data"]["raw_dir"]
    made = 0
    for label in params["data"]["classes"]:
        for src in sorted((raw / label).glob("cremad_*.wav")):
            dst = src.with_name(f"phone_{src.name}")
            if dst.exists():
                continue
            y, sr = sf.read(src, dtype="float32")
            out = phone_line(y, np.random.default_rng(seed_for(src.name)), sr)
            sf.write(dst, out, sr, subtype="PCM_16")
            made += 1
    print(f"[phone] wrote {made} phone-line clips next to the originals in {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
