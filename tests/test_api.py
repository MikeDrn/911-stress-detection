"""FastAPI integration tests driven by FakeModel (no GPU, no weights, < 2 s)."""

from __future__ import annotations

import threading
import time

from tests.conftest import wait_ready


def post(client, wav: bytes, name="call.wav", **form):
    return client.post("/predict", files={"file": (name, wav, "audio/wav")}, data=form)


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def test_liveness_is_always_200(api_factory):
    gate = threading.Event()
    client = api_factory(gate=gate)
    assert client.get("/health/live").status_code == 200
    gate.set()


def test_readiness_503_while_loading_then_200(api_factory, make_wav):
    gate = threading.Event()
    client = api_factory(gate=gate)

    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "loading"
    blocked = post(client, make_wav())
    assert blocked.status_code == 503 and blocked.headers["Retry-After"] == "5"

    gate.set()  # weights "finish loading"
    body = wait_ready(client)
    assert body["status"] == "ready" and body["model_version"] == "fake-test"
    assert body["threshold"] == 0.82 and body["consecutive_windows"] == 3
    assert post(client, make_wav()).status_code == 200


def test_failed_load_reports_503_failed(api_factory, make_wav):
    client = api_factory(fail=True)
    deadline = time.time() + 5
    while client.get("/health").json()["status"] == "loading" and time.time() < deadline:
        time.sleep(0.02)
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "failed"
    assert "weights corrupted" in r.json()["error"]
    assert post(client, make_wav()).status_code == 503


# --------------------------------------------------------------------------- #
# /predict contract
# --------------------------------------------------------------------------- #
def test_predict_schema_for_calm_call(api_factory, make_wav):
    client = api_factory()  # default FakeModel: every window 0.10
    wait_ready(client)
    r = post(client, make_wav(seconds=4.0), call_id="unit-1")
    assert r.status_code == 200
    body = r.json()
    assert body["call_id"] == "unit-1" and body["filename"] == "call.wav"
    assert body["n_windows"] == 5 and len(body["windows"]) == 5
    assert body["alert"] is False and body["alert_segments"] == []
    assert body["max_score"] == 0.1 and body["model_source"] == "fake"
    w0 = body["windows"][0]
    assert w0 == {
        "index": 0,
        "start_s": 0.0,
        "end_s": 2.0,
        "duress_score": 0.1,
        "above_threshold": False,
        "in_alert": False,
    }


def test_sustained_high_scores_raise_alert(api_factory, make_wav):
    client = api_factory(scores=[0.1, 0.9, 0.95, 0.93, 0.2, 0.99, 0.1])
    wait_ready(client)
    body = post(client, make_wav(seconds=5.0)).json()  # 7 windows
    assert body["alert"] is True
    (seg,) = body["alert_segments"]
    assert (seg["start_window"], seg["end_window"]) == (1, 3)
    assert seg["start_s"] == 0.5 and seg["end_s"] == 3.5 and seg["max_score"] == 0.95
    flags = [(w["above_threshold"], w["in_alert"]) for w in body["windows"]]
    # window 5 crosses theta but is isolated -> above_threshold without alert
    assert flags[5] == (True, False) and flags[2] == (True, True)


def test_two_isolated_spikes_do_not_alert(api_factory, make_wav):
    client = api_factory(scores=[0.95, 0.1])
    wait_ready(client)
    body = post(client, make_wav(seconds=5.0)).json()
    assert body["alert"] is False
    assert sum(w["above_threshold"] for w in body["windows"]) == 4


def test_contract_violations_return_422(api_factory, make_wav):
    client = api_factory()
    wait_ready(client)
    for wav, msg in [
        (make_wav(sr=44100), "Sample rate"),
        (make_wav(channels=2), "channels"),
        (make_wav(seconds=1.0), "at least"),
        (b"definitely not audio", "corrupt"),
    ]:
        r = post(client, wav)
        assert r.status_code == 422, msg
        assert r.json()["detail"]["error"] == "audio_contract_violation"
        assert msg in r.json()["detail"]["message"]


def test_missing_file_field_is_422(api_factory):
    client = api_factory()
    wait_ready(client)
    assert client.post("/predict").status_code == 422


# --------------------------------------------------------------------------- #
# Telemetry + hot reload
# --------------------------------------------------------------------------- #
def test_predictions_are_audited_in_sqlite(api_factory, make_wav, settings):
    client = api_factory(scores=[0.9, 0.9, 0.9])
    wait_ready(client)
    post(client, make_wav(seconds=3.0), call_id="audit-1")
    post(client, make_wav(seconds=3.0), call_id="audit-2")

    calls = client.get("/telemetry/calls").json()
    assert {c["call_id"] for c in calls} == {"audit-1", "audit-2"}
    detail = client.get("/telemetry/calls/audit-1").json()
    assert detail["alert"] == 1 and detail["model_version"] == "fake-test"
    assert [w["window_idx"] for w in detail["windows"]] == [0, 1, 2]
    stats = client.get("/telemetry/stats").json()
    assert stats["total_calls"] == 2 and stats["alert_calls"] == 2
    assert client.get("/telemetry/calls/nope").status_code == 404
    assert settings.telemetry_db.exists()


def test_hot_reload_keeps_serving(api_factory, make_wav):
    client = api_factory()
    wait_ready(client)
    r = client.post("/admin/reload")
    assert r.status_code == 202
    assert post(client, make_wav()).status_code == 200  # never 503 during reload
    deadline = time.time() + 5
    while client.get("/health").json()["reloading"] and time.time() < deadline:
        time.sleep(0.02)
    assert client.get("/health").status_code == 200


def test_fake_model_needs_no_torch():
    import subprocess
    import sys

    code = (
        "import sys; from src.fake_model import FakeModel; import numpy as np;"
        "FakeModel().predict_proba({'logmel': np.zeros((2,1,64,201))});"
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
