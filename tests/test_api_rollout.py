"""Release modes end to end through the API, with two FakeModels (no torch, no MLflow)."""

from __future__ import annotations

import asyncio
import copy
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from src.config import load_params
from src.fake_model import FakeModel
from src.serve import create_app
from src.telemetry import CALL_COLUMNS, TelemetryStore
from tests.conftest import wait_ready

MIXED = [0.1, 0.9, 0.95, 0.92, 0.1]  # some windows stressed, some calm
COLLAPSED = [0.99]  # silent failure: everything "stressed"


@pytest.fixture
def params():
    p = copy.deepcopy(load_params())
    p["rollout"]["canary"].update(steps=[50, 100], min_challenger_calls_per_step=3)
    return p


@pytest.fixture
def make_client(settings, params):
    clients = []

    def _make(champion=MIXED, challenger=MIXED):
        client = TestClient(
            create_app(
                loader=lambda: (FakeModel(champion, "fake-v1"), "fake", "fake-v1"),
                challenger_loader=(
                    None
                    if challenger is None
                    else lambda: (FakeModel(challenger, "fake-v2"), "fake", "fake-v2")
                ),
                settings=settings,
                params=params,
            )
        )
        client.__enter__()
        clients.append(client)
        wait_ready(client)
        deadline = time.time() + 5
        while challenger is not None and not client.get("/health").json()["challenger_version"]:
            assert time.time() < deadline
            time.sleep(0.02)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def predict(client, wav, call_id):
    r = client.post(
        "/predict", files={"file": ("c.wav", wav, "audio/wav")}, data={"call_id": call_id}
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_experiments_need_a_challenger(settings, params, make_client):
    client = make_client(challenger=None)
    assert client.post("/rollout", json={"mode": "canary"}).status_code == 409
    assert client.post("/rollout", json={"mode": "champion_only"}).status_code == 200


def test_shadow_scores_challenger_silently_and_feedback_labels_both(make_client, make_wav):
    client = make_client(challenger=[0.1])  # challenger never alerts
    assert client.post("/rollout", json={"mode": "shadow"}).json()["mode"] == "shadow"
    body = predict(client, make_wav(seconds=4.0), "s-1")
    assert body["served_by"] == "champion" and body["alert"] is True

    call = client.get("/telemetry/calls/s-1").json()
    assert call["shadow_alert"] == 0 and call["shadow_version"] == "fake-v2"
    fb = client.post("/feedback", json={"call_id": "s-1", "true_label": "calm"}).json()
    assert fb["correct"] == 0 and fb["shadow_correct"] == 1  # challenger was right

    rep = client.get("/rollout/report").json()
    assert rep["shadow"]["alert_agreement"] == 0.0
    assert rep["shadow"]["challenger_accuracy"] == 1.0


def test_canary_rolls_back_a_collapsed_challenger(make_client, make_wav):
    client = make_client(challenger=COLLAPSED)
    client.post("/rollout", json={"mode": "canary"})
    wav = make_wav(seconds=4.0)
    for i in range(60):
        predict(client, wav, f"c-{i}")
        if client.get("/rollout").json()["status"] == "rolled_back":
            break
    ro = client.get("/rollout").json()
    assert ro["status"] == "rolled_back" and ro["challenger_percent"] == 0.0
    assert "no_class_collapse" in ro["last_decision"]["reason"]
    events = [e["event"] for e in client.get("/rollout/events").json()]
    assert "canary_rolled_back" in events
    # after rollback every call goes back to the champion
    assert all(predict(client, wav, f"after-{i}")["served_by"] == "champion" for i in range(10))


def test_healthy_canary_ramps_to_completion(make_client, make_wav):
    client = make_client(challenger=MIXED)
    client.post("/rollout", json={"mode": "canary"})
    wav = make_wav(seconds=4.0)
    for i in range(80):
        predict(client, wav, f"h-{i}")
        if client.get("/rollout").json()["status"] == "complete":
            break
    ro = client.get("/rollout").json()
    assert ro["status"] == "complete" and ro["challenger_percent"] == 100.0
    events = [e["event"] for e in client.get("/rollout/events").json()]
    assert "canary_advanced" in events and "canary_complete" in events


def test_ab_test_reports_planned_sample_and_verdict(make_client, make_wav):
    client = make_client(challenger=[0.1])
    started = client.post("/rollout", json={"mode": "ab", "baseline_rate": 0.5, "mde": 0.3}).json()
    assert started["challenger_percent"] == 50.0 and started["planned_n_per_arm"] > 0
    wav = make_wav(seconds=4.0)
    arms = set()
    for i in range(40):
        body = predict(client, wav, f"ab-{i}")
        arms.add(body["served_by"])
        client.post("/feedback", json={"call_id": f"ab-{i}", "true_label": "calm"})
    assert arms == {"champion", "challenger"}
    ab = client.get("/rollout/report").json()["ab"]
    assert ab["p_b"] == 1.0 and ab["p_a"] == 0.0  # only the challenger stays quiet on calm calls
    assert ab["verdict"].startswith(("B BETTER", "RUNNING"))


def test_bandit_logs_propensities_and_shifts_traffic(make_client, make_wav):
    client = make_client(challenger=[0.1])
    client.post("/rollout", json={"mode": "bandit", "policy": "thompson"})
    wav = make_wav(seconds=4.0)
    served = []
    for i in range(60):
        served.append(predict(client, wav, f"b-{i}")["served_by"])
        client.post("/feedback", json={"call_id": f"b-{i}", "true_label": "calm"})
    calls = client.get("/telemetry/calls?limit=60").json()
    assert all(0.05 - 1e-9 <= c["propensity"] <= 0.95 + 1e-9 for c in calls)
    assert served[-20:].count("challenger") > 14  # it learned the challenger is better here
    assert client.get("/rollout/report").json()["bandit"]["estimated_regret"] >= 0


def test_failing_challenger_falls_back_to_champion(make_client, make_wav):
    class Broken(FakeModel):
        def predict_proba(self, features):
            raise RuntimeError("boom")

    client = make_client(challenger=MIXED)
    client.app.state.model.challenger = Broken(version="broken")
    client.post("/rollout", json={"mode": "ab", "baseline_rate": 0.5, "mde": 0.3})
    wav = make_wav(seconds=4.0)
    for i in range(30):
        predict(client, wav, f"f-{i}")  # never fails for the caller
    calls = client.get("/telemetry/calls?limit=30").json()
    assert any(c["arm"] == "challenger" and c["error"] == 1 for c in calls)


def test_old_audit_database_is_migrated(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE calls (call_id TEXT PRIMARY KEY, received_at TEXT NOT NULL, "
        "alert INTEGER NOT NULL, max_score REAL)"
    )
    con.execute("INSERT INTO calls VALUES ('old-1', '2026-01-01', 1, 0.9)")
    con.commit()
    con.close()
    asyncio.run(TelemetryStore(db).init())
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(calls)")}
    assert set(CALL_COLUMNS) <= cols
    assert con.execute("SELECT max_score FROM calls WHERE call_id='old-1'").fetchone()[0] == 0.9
