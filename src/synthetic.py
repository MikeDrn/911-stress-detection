"""Synthetic speech-like audio for demonstrations and CI.

Real duress recordings are sensitive and not freely shareable, so the project
ships a *parametric voice synthesiser* that produces vowel-like utterances
with the acoustic correlates the literature associates with acute stress:

* raised fundamental frequency (F0) and wider intonation range,
* 8-12 Hz physiological micro-tremor modulating F0,
* higher cycle-to-cycle jitter and amplitude shimmer,
* faster syllable rate, flatter spectral tilt (pressed phonation),
* more aspiration / breath noise.

Both classes share overlapping speaker pitch ranges and random background
noise so the task is not trivially solvable by loudness alone. The data is a
stand-in that lets the whole MLOps lifecycle run end to end; a model trained on
it says nothing about real callers. Swap in a licensed corpus (see README) for
any real experiment.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

VOWEL_FORMANTS = {  # (F1, F2, F3) in Hz, rough averages
    "a": (730, 1090, 2440),
    "e": (530, 1840, 2480),
    "i": (390, 1990, 2550),
    "o": (570, 840, 2410),
    "u": (440, 1020, 2240),
}


def _smooth_noise(n: int, rng: np.random.Generator, cutoff_hz: float, sr: int) -> np.ndarray:
    """Low-pass filtered Gaussian noise normalised to unit std."""
    alpha = np.exp(-2 * np.pi * cutoff_hz / sr)
    x = lfilter([1 - alpha], [1, -alpha], rng.standard_normal(n))
    x = lfilter([1 - alpha], [1, -alpha], x)
    return x / (x.std() + 1e-9)


def _resonator(x: np.ndarray, freq: float, bw: float, sr: int) -> np.ndarray:
    r = np.exp(-np.pi * bw / sr)
    theta = 2 * np.pi * freq / sr
    a = [1.0, -2 * r * np.cos(theta), r * r]
    b = [1 - r]
    return lfilter(b, a, x)


def synth_utterance(
    duration_s: float,
    stressed: bool,
    rng: np.random.Generator,
    sr: int = 16000,
) -> np.ndarray:
    n = int(duration_s * sr)
    t = np.arange(n) / sr

    # --- speaker + state dependent prosody -------------------------------
    female = rng.random() < 0.5
    f0_base = rng.uniform(165, 235) if female else rng.uniform(90, 145)
    if stressed:
        f0_base *= rng.uniform(1.18, 1.45)
    contour_depth = rng.uniform(0.08, 0.16) if stressed else rng.uniform(0.03, 0.07)
    tremor_depth = rng.uniform(0.025, 0.06) if stressed else rng.uniform(0.0, 0.008)
    tremor_rate = rng.uniform(8.0, 12.0)
    jitter_depth = rng.uniform(0.015, 0.035) if stressed else rng.uniform(0.003, 0.01)
    syllable_rate = rng.uniform(5.0, 7.0) if stressed else rng.uniform(2.8, 4.2)
    tilt = rng.uniform(0.75, 1.0) if stressed else rng.uniform(1.15, 1.45)
    breath_level = rng.uniform(0.08, 0.18) if stressed else rng.uniform(0.01, 0.05)
    shimmer = rng.uniform(0.15, 0.3) if stressed else rng.uniform(0.03, 0.1)

    f0 = f0_base * (
        1
        + contour_depth * _smooth_noise(n, rng, 1.5, sr)
        + tremor_depth * np.sin(2 * np.pi * tremor_rate * t + rng.uniform(0, 2 * np.pi))
        + jitter_depth * _smooth_noise(n, rng, 60.0, sr)
    )
    f0 = np.clip(f0, 60, 500)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    # --- glottal source: harmonic series with spectral tilt --------------
    n_harm = int(4000 // f0_base)
    source = np.zeros(n)
    for h in range(1, n_harm + 1):
        source += np.sin(h * phase) / (h**tilt)

    # --- syllabic envelope with pauses and shimmer -----------------------
    syl_phase = 2 * np.pi * np.cumsum(syllable_rate * (1 + 0.2 * _smooth_noise(n, rng, 0.5, sr)))
    env = np.clip(np.sin(syl_phase / sr), 0, None) ** 0.7
    env *= 1 + shimmer * _smooth_noise(n, rng, 30.0, sr)
    pause_mask = (_smooth_noise(n, rng, 0.4, sr) > -0.9).astype(float)
    env = np.clip(env * lfilter([0.002], [1, -0.998], pause_mask) / 1.0, 0, None)

    # --- vocal tract: cascade of formant resonators -----------------------
    vowel = VOWEL_FORMANTS[rng.choice(list(VOWEL_FORMANTS))]
    shift = 1.08 if female else 1.0
    voiced = source
    for f, bw in zip(vowel, (80, 100, 140), strict=True):
        voiced = _resonator(voiced, f * shift * rng.uniform(0.95, 1.05), bw, sr)

    breath = _resonator(rng.standard_normal(n), 2500, 1500, sr) * breath_level
    speech = (voiced / (np.abs(voiced).max() + 1e-9) + breath) * env

    # --- channel: background noise at a random SNR ------------------------
    snr_db = rng.uniform(12, 30)
    sig_pow = np.mean(speech**2) + 1e-12
    noise = rng.standard_normal(n) * np.sqrt(sig_pow / (10 ** (snr_db / 10)))
    y = speech + noise
    y = y / (np.abs(y).max() + 1e-9) * rng.uniform(0.3, 0.9)
    return y.astype(np.float32)


def crossfade_concat(parts: list[np.ndarray], sr: int = 16000, fade_s: float = 0.25) -> np.ndarray:
    """Concatenate segments with a short linear crossfade (used for demo calls)."""
    fade = int(fade_s * sr)
    out = parts[0].astype(np.float32)
    for p in parts[1:]:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        overlap = out[-fade:] * (1 - ramp) + p[:fade] * ramp
        out = np.concatenate([out[:-fade], overlap, p[fade:]])
    return out
