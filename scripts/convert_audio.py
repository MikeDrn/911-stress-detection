"""Convert any recording (m4a, mp3, wav, ogg, flac, ...) to the 16 kHz mono PCM WAV contract.

    uv run python -m scripts.convert_audio my_voice.m4a data/demo/my_voice.wav

Phone recordings (m4a/aac/mp3) are decoded with PyAV, which is installed with the
speech-to-text package, so no separate FFmpeg installation is needed.
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf

from src.config import load_params


def decode(path: str, sr: int) -> np.ndarray:
    try:
        from faster_whisper.audio import decode_audio  # PyAV: m4a, mp3, aac, ogg, wav ...

        return decode_audio(path, sampling_rate=sr)
    except ImportError:
        import librosa  # fallback: wav/flac/ogg without FFmpeg

        y, _ = librosa.load(path, sr=sr, mono=True)
        return y


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resample + downmix audio to the data contract")
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args(argv)
    sr = int(load_params()["audio"]["sample_rate"])
    y = decode(args.src, sr)
    y = y / (np.abs(y).max() + 1e-9) * 0.8
    sf.write(args.dst, y.astype(np.float32), sr, subtype="PCM_16")
    print(f"[convert] {args.src} -> {args.dst} ({len(y) / sr:.1f}s, {sr} Hz mono PCM_16)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
