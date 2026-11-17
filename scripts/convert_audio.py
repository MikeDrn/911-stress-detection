"""Convert any recording to the 16 kHz mono PCM WAV ingestion contract.

    uv run python -m scripts.convert_audio my_voice.m4a data/demo/my_voice.wav

WAV/FLAC/OGG work out of the box; MP3/M4A need FFmpeg installed on the host.
"""

from __future__ import annotations

import argparse

import librosa
import soundfile as sf

from src.config import load_params


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resample + downmix audio to the data contract")
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args(argv)
    sr = int(load_params()["audio"]["sample_rate"])
    y, _ = librosa.load(args.src, sr=sr, mono=True)
    sf.write(args.dst, y, sr, subtype="PCM_16")
    print(f"[convert] {args.src} -> {args.dst} ({len(y) / sr:.1f}s, {sr} Hz mono PCM_16)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
