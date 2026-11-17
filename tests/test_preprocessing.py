"""Unit tests for audio loading, the ingestion contract, windowing and features."""

from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing import (
    AudioContractError,
    AudioSpec,
    FeatureExtractor,
    load_audio,
    preprocess_call,
    sliding_windows,
)

SPEC = AudioSpec()  # 16 kHz, mono, 2.0 s windows, 0.5 s hop


def test_spec_matches_params_yaml():
    assert AudioSpec.from_params() == SPEC
    assert SPEC.window_samples == 32000 and SPEC.hop_samples == 8000


def test_valid_wav_loads_as_float32_mono(make_wav):
    y = load_audio(make_wav(seconds=3.0), SPEC)
    assert y.dtype == np.float32 and y.ndim == 1 and len(y) == 48000


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sr": 44100}, "Sample rate 44100"),
        ({"sr": 8000}, "Sample rate 8000"),
        ({"channels": 2}, "2 channels"),
        ({"seconds": 1.0}, "at least 2.0s"),
        ({"fmt": "FLAC", "subtype": "PCM_16"}, "expected WAV"),
    ],
)
def test_contract_violations_are_rejected(make_wav, kwargs, message):
    with pytest.raises(AudioContractError, match=message):
        load_audio(make_wav(**kwargs), SPEC)


def test_corrupt_header_rejected():
    with pytest.raises(AudioContractError, match="Unreadable or corrupt"):
        load_audio(b"RIFF\x00\x00garbage-not-a-wav", SPEC)


def test_float_wav_is_accepted(make_wav):
    assert len(load_audio(make_wav(subtype="FLOAT"), SPEC)) == 64000


@pytest.mark.parametrize("seconds, expected", [(2.0, 1), (2.4, 1), (2.5, 2), (4.0, 5), (10.0, 17)])
def test_sliding_window_count(seconds, expected):
    y = np.zeros(int(seconds * 16000), dtype=np.float32)
    windows, starts = sliding_windows(y, SPEC)
    assert windows.shape == (expected, 32000)
    np.testing.assert_allclose(starts, np.arange(expected) * 0.5)


def test_windows_overlap_exactly_by_hop():
    y = np.arange(16000 * 3, dtype=np.float32)
    w, _ = sliding_windows(y, SPEC)
    np.testing.assert_array_equal(w[1][:24000], w[0][8000:])


def test_feature_shapes_and_normalisation(make_wav):
    fx = FeatureExtractor(SPEC)
    y = load_audio(make_wav(seconds=3.0), SPEC)
    windows, _ = sliding_windows(y, SPEC)
    feats = fx.transform(windows)
    assert feats["logmel"].shape == (3, 1, 64, fx.n_frames) and fx.n_frames == 201
    assert feats["mfcc"].shape == (3, 20, 201)
    for x in feats.values():
        assert x.dtype == np.float32 and np.isfinite(x).all()
        assert abs(float(x[0].mean())) < 1e-3 and abs(float(x[0].std()) - 1) < 1e-2


def test_features_are_gain_invariant(make_wav):
    fx = FeatureExtractor(SPEC)
    y = load_audio(make_wav(seconds=2.0), SPEC)
    w, _ = sliding_windows(y, SPEC)
    a, b = fx.log_mel(w), fx.log_mel(w * 0.5)
    np.testing.assert_allclose(a, b, atol=0.05)


def test_preprocess_call_end_to_end(make_wav):
    out = preprocess_call(make_wav(seconds=5.0))
    assert out["duration_s"] == pytest.approx(5.0)
    assert len(out["starts"]) == len(out["ends"]) == out["logmel"].shape[0] == 7
    np.testing.assert_allclose(out["ends"] - out["starts"], 2.0)
