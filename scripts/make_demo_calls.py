"""Create demo calls in data/demo/ from REAL held-out speakers (CREMA-D test split).

* call_01_real_voice_calm.wav       one held-out actor, neutral sentences joined (~10 s)
* call_02_real_voice_frightened.wav the SAME actor, fearful sentences joined (~10 s)
* call_03_contract_violation_44k_stereo.wav  44.1 kHz stereo -> HTTP 422 data contract

The actors read fixed sentences ("It's eleven o'clock"...), so for the content part
type a transcript in the console's "what the caller said" box. The key demo:

    call_02 + "someone is dead in front of me, please come"   -> NO alert (stress expected)
    call_02 + "hi, a large pizza with pepperoni to 61 elm st"  -> ALERT  (covert call)
    call_01 + "hi, a large pizza with pepperoni to 61 elm st"  -> ALERT  (calm covert caller)

For a fully realistic demo, record the scripts in docs/demo_call_scripts.md yourself
(the API's speech-to-text then writes the transcript) and convert them with
``uv run python -m scripts.convert_audio``.

    uv run python -m scripts.make_demo_calls
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.config import PROJECT_ROOT, load_params
from src.data_contracts import read_manifest

SR = 16000


def join(paths: list[Path], gap_s: float = 0.35) -> np.ndarray:
    gap = np.zeros(int(gap_s * SR), dtype=np.float32)
    parts = []
    for p in paths:
        y, _ = sf.read(p, dtype="float32")
        parts += [y / (np.abs(y).max() + 1e-9) * 0.7, gap]
    return np.concatenate(parts)


def main() -> int:
    params = load_params()
    out = PROJECT_ROOT / "data" / "demo"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = PROJECT_ROOT / params["data"]["manifest"]
    if manifest_path.exists():
        rows = [
            r
            for r in read_manifest(manifest_path)
            if r["split"] == "test"
            and r.get("speaker")
            and not Path(r["path"]).name.startswith("phone_")
        ]
        for spk in sorted({r["speaker"] for r in rows}):  # first actor with both kinds
            calm = sorted(
                PROJECT_ROOT / r["path"]
                for r in rows
                if r["speaker"] == spk and r["label"] == "calm"
            )[:4]
            fear = sorted(
                PROJECT_ROOT / r["path"]
                for r in rows
                if r["speaker"] == spk and r["label"] == "stressed"
            )[:4]
            if len(calm) >= 3 and len(fear) >= 3:
                sf.write(out / "call_01_real_voice_calm.wav", join(calm), SR, subtype="PCM_16")
                sf.write(
                    out / "call_02_real_voice_frightened.wav", join(fear), SR, subtype="PCM_16"
                )
                print(f"[demo] real-voice calls from held-out speaker {spk}")
                break
    else:
        print("[demo] no manifest yet - run `uv run dvc repro` first for the real-voice calls")

    t = np.arange(int(4 * 44100)) / 44100
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32)
    sf.write(out / "call_03_contract_violation_44k_stereo.wav", stereo, 44100)

    for p in sorted(out.glob("*.wav")):
        print(f"[demo] {p.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
