"""Audio ingestion, data-contract enforcement and acoustic feature extraction.

Pipeline for one call:

    raw bytes / WAV file
      -> load_audio()          (16 kHz mono contract, duration bounds, finite samples)
      -> sliding_windows()     (2.0 s windows, 0.5 s hop)
      -> FeatureExtractor      (Log-Mel spectrogram [CNN input] + MFCC [1-D input])

The *same* functions are used offline (DVC ``featurize`` stage) and online
(FastAPI ``/predict``). Sharing one implementation removes training/serving
skew, one of the most common silent failure modes in production ML.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, BinaryIO

import numpy as np
import soundfile as sf

from src.config import load_params

ALLOWED_FORMATS = {"WAV"}
ALLOWED_SUBTYPES = {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}


class AudioContractError(ValueError):
    """Raised when audio violates the ingestion data contract."""


@dataclass(frozen=True)
class AudioSpec:
    sample_rate: int = 16000
    channels: int = 1
    window_seconds: float = 2.0
    hop_seconds: float = 0.5
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 600.0

    @classmethod
    def from_params(cls, params: dict[str, Any] | None = None) -> AudioSpec:
        a = (params or load_params())["audio"]
        return cls(
            sample_rate=int(a["sample_rate"]),
            channels=int(a["channels"]),
            window_seconds=float(a["window_seconds"]),
            hop_seconds=float(a["hop_seconds"]),
            min_duration_seconds=float(a["min_duration_seconds"]),
            max_duration_seconds=float(a["max_duration_seconds"]),
        )

    @property
    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.sample_rate))

    @property
    def hop_samples(self) -> int:
        return int(round(self.hop_seconds * self.sample_rate))


# --------------------------------------------------------------------------- #
# Loading + contract validation
# --------------------------------------------------------------------------- #
AudioSource = str | os.PathLike | bytes | BinaryIO


def _open(source: AudioSource):
    if isinstance(source, bytes | bytearray):
        return io.BytesIO(source)
    return source


def inspect_audio(source: AudioSource) -> sf._SoundFileInfo:
    """Read only the header; raises AudioContractError on corrupt/unsupported files."""
    try:
        info = sf.info(_open(source))
    except Exception as exc:  # soundfile raises RuntimeError/LibsndfileError
        raise AudioContractError(f"Unreadable or corrupt audio header: {exc}") from exc
    return info


def validate_header(info: Any, spec: AudioSpec) -> None:
    """Enforce the format half of the data contract using header metadata only."""
    if info.format not in ALLOWED_FORMATS:
        raise AudioContractError(f"Unsupported container '{info.format}'; expected WAV.")
    if info.subtype not in ALLOWED_SUBTYPES:
        raise AudioContractError(f"Unsupported codec '{info.subtype}'; expected PCM or float WAV.")
    if info.samplerate != spec.sample_rate:
        raise AudioContractError(
            f"Sample rate {info.samplerate} Hz violates contract; expected {spec.sample_rate} Hz."
        )
    if info.channels != spec.channels:
        raise AudioContractError(
            f"{info.channels} channels violates contract; expected mono ({spec.channels})."
        )
    duration = info.frames / float(info.samplerate)
    if duration < spec.min_duration_seconds:
        raise AudioContractError(
            f"Audio is {duration:.2f}s; at least {spec.min_duration_seconds:.1f}s is required."
        )
    if duration > spec.max_duration_seconds:
        raise AudioContractError(
            f"Audio is {duration:.0f}s; maximum is {spec.max_duration_seconds:.0f}s."
        )


def load_audio(source: AudioSource, spec: AudioSpec | None = None) -> np.ndarray:
    """Load a WAV file/bytes as float32 mono after validating the data contract."""
    spec = spec or AudioSpec.from_params()
    if isinstance(source, bytes | bytearray):
        source = bytes(source)
    info = inspect_audio(source)
    validate_header(info, spec)
    try:
        y, _ = sf.read(_open(source), dtype="float32", always_2d=False)
    except Exception as exc:
        raise AudioContractError(f"Failed to decode audio samples: {exc}") from exc
    if y.ndim != 1:
        raise AudioContractError("Decoded audio is not mono.")
    if not np.all(np.isfinite(y)):
        raise AudioContractError("Audio contains NaN or infinite samples.")
    return y


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def sliding_windows(y: np.ndarray, spec: AudioSpec) -> tuple[np.ndarray, np.ndarray]:
    """Split ``y`` into overlapping windows.

    Returns ``(windows, start_times)`` where ``windows`` has shape
    ``(n_windows, window_samples)`` and ``start_times`` is in seconds. Trailing
    audio shorter than one hop is dropped (it is still covered by the previous
    overlapping window). A clip shorter than one window (the contract allows
    >= ``min_duration_seconds``) is zero-padded to exactly one window.
    """
    win, hop = spec.window_samples, spec.hop_samples
    if len(y) == 0:
        raise AudioContractError("Signal is empty.")
    if len(y) < win:
        y = np.pad(y, (0, win - len(y)))
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    windows = y[idx].astype(np.float32, copy=False)
    starts = (hop * np.arange(n)) / float(spec.sample_rate)
    return windows, starts


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSpec:
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    n_mfcc: int = 20
    fmin: float = 50.0
    fmax: float = 8000.0

    @classmethod
    def from_params(cls, params: dict[str, Any] | None = None) -> FeatureSpec:
        f = (params or load_params())["features"]
        return cls(
            n_mels=int(f["n_mels"]),
            n_fft=int(f["n_fft"]),
            hop_length=int(f["hop_length"]),
            n_mfcc=int(f["n_mfcc"]),
            fmin=float(f["fmin"]),
            fmax=float(f["fmax"]),
        )


def _standardize(x: np.ndarray) -> np.ndarray:
    """Per-example zero-mean / unit-variance normalisation (gain invariant)."""
    mean = x.mean(axis=(-2, -1), keepdims=True)
    std = x.std(axis=(-2, -1), keepdims=True)
    return ((x - mean) / (std + 1e-6)).astype(np.float32)


class FeatureExtractor:
    """Converts windows into Log-Mel spectrograms and MFCC matrices."""

    def __init__(self, audio_spec: AudioSpec | None = None, feat_spec: FeatureSpec | None = None):
        self.audio = audio_spec or AudioSpec.from_params()
        self.feat = feat_spec or FeatureSpec.from_params()

    @property
    def n_frames(self) -> int:
        return 1 + self.audio.window_samples // self.feat.hop_length

    def log_mel(self, windows: np.ndarray) -> np.ndarray:
        """``(n, samples)`` -> ``(n, 1, n_mels, frames)`` standardized Log-Mel tensor."""
        import librosa

        mel = librosa.feature.melspectrogram(
            y=windows,
            sr=self.audio.sample_rate,
            n_fft=self.feat.n_fft,
            hop_length=self.feat.hop_length,
            n_mels=self.feat.n_mels,
            fmin=self.feat.fmin,
            fmax=self.feat.fmax,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
        return _standardize(log_mel)[:, None, :, :]

    def mfcc(self, windows: np.ndarray) -> np.ndarray:
        """``(n, samples)`` -> ``(n, n_mfcc, frames)`` standardized MFCC tensor."""
        import librosa

        m = librosa.feature.mfcc(
            y=windows,
            sr=self.audio.sample_rate,
            n_mfcc=self.feat.n_mfcc,
            n_fft=self.feat.n_fft,
            hop_length=self.feat.hop_length,
            n_mels=self.feat.n_mels,
            fmin=self.feat.fmin,
            fmax=self.feat.fmax,
        )
        return _standardize(m)

    def transform(self, windows: np.ndarray) -> dict[str, np.ndarray]:
        return {"logmel": self.log_mel(windows), "mfcc": self.mfcc(windows)}


def preprocess_call(
    source: AudioSource,
    audio_spec: AudioSpec | None = None,
    extractor: FeatureExtractor | None = None,
) -> dict[str, Any]:
    """Full online path used by the API: validate -> window -> features."""
    audio_spec = audio_spec or AudioSpec.from_params()
    extractor = extractor or FeatureExtractor(audio_spec)
    y = load_audio(source, audio_spec)
    windows, starts = sliding_windows(y, audio_spec)
    feats = extractor.transform(windows)
    return {
        "duration_s": len(y) / float(audio_spec.sample_rate),
        "starts": starts,
        "ends": starts + audio_spec.window_seconds,
        "audio": y,  # raw samples, for speech-to-text
        **feats,
    }
