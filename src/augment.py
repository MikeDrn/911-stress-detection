"""Telephone-channel simulation applied to REAL recordings (the data-shift scenario).

New calls arrive over narrow-band, noisy mobile lines: 300-3400 Hz band-pass,
codec soft-clipping and background noise at 4-12 dB SNR. Applying this to the
CREMA-D clips creates the "phone-line" version of the dataset used in the
continuous-training demo (Course 4 distribution shift, Course 7 CT trigger).
"""

from __future__ import annotations

import zlib

import numpy as np
from scipy.signal import butter, sosfilt


def phone_line(y: np.ndarray, rng: np.random.Generator, sr: int = 16000) -> np.ndarray:
    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    band = sosfilt(sos, y.astype(np.float64))
    drive = rng.uniform(2.0, 4.0)
    peak = np.abs(band).max() + 1e-9
    coded = np.tanh(drive * band / peak) / np.tanh(drive)
    snr_db = rng.uniform(4.0, 12.0)
    noise = rng.standard_normal(len(coded)) * np.sqrt(np.mean(coded**2) / 10 ** (snr_db / 10))
    out = coded + noise
    return (out / (np.abs(out).max() + 1e-9) * rng.uniform(0.3, 0.8)).astype(np.float32)


def seed_for(name: str) -> int:
    """Deterministic per-file seed: the same input always gives byte-identical output."""
    return zlib.crc32(name.encode())
