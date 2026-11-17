"""FastAPI integration tests driven by FakeModels (no GPU, no weights, no Whisper)."""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from src.asr import StaticTranscriber
from src.serve import create_app
from tests.conftest import wait_ready

PIZZA = "hi i'd like to order a large pizza with pepperoni to 61 elm street please hurry"
DEAD = "someone is dead in front of me please come the blood is everywhere"


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
    assert body["asr"] == "disabled"
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
# /predict: content + voice -> combined decision
# --------------------------------------------------------------------------- #
def test_without_transcript_content_is_unavailable_and_no_alert(api_factory, make_wav):
    client = api_factory(scores=[0.95])  # very stressed voice, nothing said
    wait_ready(client)
    body = post(client, make_wav(seconds=4.0), call_id="unit-1").json()
    assert body["call_id"] == "unit-1" and body["n_windows"] == 5
    assert body["transcript_source"] == "unavailable" and body["content"] is None
    assert body["alert"] is False and body["threat_level"] == "normal"
    assert body["stress_score"] == 0.95  # stress is reported, but stress alone never alerts
    assert "No transcript" in body["reason"]


def test_covert_content_alerts_even_with_calm_voice(api_factory, make_wav):
    client = api_factory(scores=[0.1])
    wait_ready(client)
    body = post(client, make_wav(seconds=4.0), transcript=PIZZA).json()
    assert body["transcript_source"] == "provided" and body["content"]["label"] == "covert"
    assert body["alert"] is True and body["threat_level"] == "alert"
    assert body["threat_score"] >= 0.5 and "covert" in body["reason"]


def test_genuine_emergency_with_stressed_voice_does_not_alert(api_factory, make_wav):
    client = api_factory(scores=[0.99])
    wait_ready(client)
    body = post(client, make_wav(seconds=4.0), transcript=DEAD).json()
    assert body["content"]["label"] == "genuine_emergency"
    assert body["stress_score"] == 0.99
    assert body["alert"] is False and body["threat_level"] == "normal"
    assert "expected" in body["reason"]


def test_harmless_content_with_terrified_voice_is_flagged_for_review(api_factory, make_wav):
    client = api_factory(scores=[0.99])
    wait_ready(client)
    body = post(client, make_wav(seconds=4.0), transcript="what time does the library open").json()
    assert body["content"]["label"] == "non_emergency"
    assert body["threat_level"] == "review" and body["alert"] is False


def test_speech_to_text_is_used_when_no_transcript_is_given(settings, make_wav):
    app = create_app(
        loader=lambda: (
            __import__("src.fake_model").fake_model.FakeModel([0.2], "fake"),
            "fake",
            "fake",
        ),
        settings=settings,
        transcriber=StaticTranscriber(PIZZA, name="static-test"),
    )
    with TestClient(app) as client:
        wait_ready(client)
        assert "static-test" in client.get("/health").json()["asr"]
        body = post(client, make_wav(seconds=3.0)).json()
        assert body["transcript_source"] == "asr" and body["transcript"] == PIZZA
        assert body["alert"] is True
        typed = post(client, make_wav(seconds=3.0), transcript=DEAD).json()
        assert typed["transcript_source"] == "provided" and typed["alert"] is False


def test_sustained_voice_stress_runs_are_still_reported(api_factory, make_wav):
    client = api_factory(scores=[0.1, 0.9, 0.95, 0.93, 0.2, 0.99, 0.1])
    wait_ready(client)
    body = post(client, make_wav(seconds=5.0)).json()  # 7 windows
    (seg,) = body["alert_segments"]
    assert (seg["start_window"], seg["end_window"]) == (1, 3)
    flags = [(w["above_threshold"], w["in_alert"]) for w in body["windows"]]
    assert flags[5] == (True, False) and flags[2] == (True, True)


def test_contract_violations_return_422(api_factory, make_wav):
    client = api_factory()
    wait_ready(client)
    for wav, msg in [
        (make_wav(sr=44100), "Sample rate"),
        (make_wav(channels=2), "channels"),
        (make_wav(seconds=0.5), "at least"),
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
# Telemetry + feedback + hot reload
# --------------------------------------------------------------------------- #
def test_predictions_are_audited_in_sqlite(api_factory, make_wav, settings):
    client = api_factory(scores=[0.3])
    wait_ready(client)
    post(client, make_wav(seconds=3.0), call_id="audit-1", transcript=PIZZA)
    post(client, make_wav(seconds=3.0), call_id="audit-2", transcript=DEAD)

    calls = client.get("/telemetry/calls").json()
    assert {c["call_id"] for c in calls} == {"audit-1", "audit-2"}
    detail = client.get("/telemetry/calls/audit-1").json()
    assert detail["alert"] == 1 and detail["threat_level"] == "alert"
    assert detail["content_label"] == "covert" and detail["transcript"] == PIZZA
    assert [w["window_idx"] for w in detail["windows"]] == [0, 1, 2]
    stats = client.get("/telemetry/stats").json()
    assert stats["total_calls"] == 2 and stats["alert_calls"] == 1
    assert client.get("/telemetry/calls/nope").status_code == 404
    assert settings.telemetry_db.exists()


def test_feedback_marks_decisions_correct_or_wrong(api_factory, make_wav):
    client = api_factory(scores=[0.3])
    wait_ready(client)
    post(client, make_wav(seconds=3.0), call_id="fb-1", transcript=PIZZA)
    ok = client.post("/feedback", json={"call_id": "fb-1", "true_label": "threat"}).json()
    assert ok["correct"] == 1
    wrong = client.post("/feedback", json={"call_id": "fb-1", "true_label": "no_threat"}).json()
    assert wrong["correct"] == 0
    assert (
        client.post("/feedback", json={"call_id": "fb-1", "true_label": "calm"}).status_code == 422
    )


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
        "import sys; from src.fake_model import FakeModel, FakeTextModel; import numpy as np;"
        "FakeModel().predict_proba({'logmel': np.zeros((2,1,64,201))});"
        "FakeTextModel().predict('pizza');"
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
