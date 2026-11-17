"""Create demo 911 calls for the live presentation in data/demo/.

* call_01_calm.wav            15 s of calm speech            -> green, no alert
* call_02_under_duress.wav    calm 5 s -> stressed 9 s -> calm 4 s -> red alert segment
* call_03_brief_spike.wav     calm with one ~0.9 s stressed burst -> typically 1-2 amber
                              windows but *no* alert
                              (shows the consecutive-window rule suppressing blips)
* call_04_contract_violation_44k_stereo.wav  44.1 kHz stereo -> HTTP 422 data contract

    uv run python -m scripts.make_demo_calls
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from src.config import PROJECT_ROOT
from src.synthetic import crossfade_concat, synth_utterance

SR = 16000
SPIKE_S = float(os.environ.get("DEMO_SPIKE_SECONDS", "0.9"))


def main() -> int:
    out = PROJECT_ROOT / "data" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)

    calm = synth_utterance(15.0, False, rng, SR)
    sf.write(out / "call_01_calm.wav", calm, SR, subtype="PCM_16")

    duress = crossfade_concat(
        [
            synth_utterance(5.0, False, rng, SR),
            synth_utterance(9.0, True, rng, SR),
            synth_utterance(4.0, False, rng, SR),
        ],
        SR,
    )
    sf.write(out / "call_02_under_duress.wav", duress, SR, subtype="PCM_16")

    spike = crossfade_concat(
        [
            synth_utterance(6.0, False, rng, SR),
            synth_utterance(SPIKE_S, True, rng, SR),
            synth_utterance(6.0, False, rng, SR),
        ],
        SR,
    )
    sf.write(out / "call_03_brief_spike.wav", spike, SR, subtype="PCM_16")

    t = np.arange(int(4 * 44100)) / 44100
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32)
    sf.write(out / "call_04_contract_violation_44k_stereo.wav", stereo, 44100)

    for p in sorted(out.glob("*.wav")):
        print(f"[demo] {p.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
