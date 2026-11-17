"""Shared pytest fixtures: in-memory WAV factory and a FakeModel-backed API client."""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.fake_model import FakeModel
from src.serve import Settings, create_app


def wav_bytes(
    seconds: float = 4.0,
    sr: int = 16000,
    channels: int = 1,
    freq: float = 220.0,
    subtype: str = "PCM_16",
    fmt: str = "WAV",
) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    if channels > 1:
        y = np.stack([y] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


@pytest.fixture
def make_wav() -> Callable[..., bytes]:
    return wav_bytes


@pytest.fixture
def write_wav(tmp_path):
    def _write(rel: str, **kw) -> str:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(wav_bytes(**kw))
        return str(p)

    return _write


def wait_ready(client: TestClient, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get("/health")
        if r.status_code == 200:
            return r.json()
        time.sleep(0.02)
    raise AssertionError(f"API not ready after {timeout}s: {r.json()}")


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(telemetry_db=tmp_path / "telemetry.db", threshold=0.82, consecutive_windows=3)


@pytest.fixture
def api_factory(settings):
    """Build a TestClient around create_app() with an injected FakeModel loader."""
    clients: list[TestClient] = []

    def _make(scores=None, gate: threading.Event | None = None, fail: bool = False):
        def loader():
            if gate is not None:
                gate.wait(timeout=10)
            if fail:
                raise RuntimeError("weights corrupted")
            return FakeModel(scores, version="fake-test"), "fake", "fake-test"

        client = TestClient(create_app(loader=loader, settings=settings))
        client.__enter__()  # runs the lifespan (telemetry init + background load)
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)
