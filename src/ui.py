"""Streamlit dispatcher console (``dispatcher-ui``).

The UI is intentionally a *thin client*: it holds no model and no database
handle. Everything goes through the ``stress-api`` REST contract over the
private Docker network (``http://stress-api:8000``), so the console can be
redeployed, restyled or replaced without touching inference or audit logging.

Run locally:  streamlit run src/ui.py
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
DEMO_DIR = Path(os.environ.get("DEMO_DIR", Path(__file__).resolve().parent.parent / "data/demo"))
RED, AMBER, GREY, GREEN = "#d62828", "#f4a261", "#8d99ae", "#2a9d8f"

st.set_page_config(
    page_title="911 Dispatcher - Voice Stress Monitor", page_icon="🚨", layout="wide"
)

st.markdown(
    f"""
    <style>
    .alert-banner {{
        background:{RED}; color:white; padding:18px 22px; border-radius:10px;
        font-size:1.25rem; font-weight:700; animation: pulse 1.2s ease-in-out infinite;
        border:3px solid #7f1010;
    }}
    .ok-banner {{
        background:{GREEN}; color:white; padding:14px 22px; border-radius:10px;
        font-size:1.1rem; font-weight:600;
    }}
    .wait-banner {{
        background:#264653; color:white; padding:14px 22px; border-radius:10px;
        font-size:1.05rem;
    }}
    @keyframes pulse {{ 0%{{box-shadow:0 0 0 0 rgba(214,40,40,.7)}}
                        70%{{box-shadow:0 0 0 14px rgba(214,40,40,0)}}
                        100%{{box-shadow:0 0 0 0 rgba(214,40,40,0)}} }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def api_health(base: str) -> tuple[int, dict]:
    try:
        r = requests.get(f"{base}/health", timeout=3)
        return r.status_code, r.json()
    except requests.RequestException as exc:
        return 0, {"status": "unreachable", "error": str(exc)}


def api_predict(base: str, name: str, data: bytes) -> tuple[int, dict]:
    try:
        r = requests.post(f"{base}/predict", files={"file": (name, data, "audio/wav")}, timeout=120)
        return r.status_code, r.json()
    except requests.RequestException as exc:
        return 0, {"detail": str(exc)}


def api_get(base: str, path: str):
    try:
        r = requests.get(f"{base}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def windows_frame(windows: list[dict], threshold: float) -> pd.DataFrame:
    df = pd.DataFrame(windows)
    if df.empty:
        return df
    df["center_s"] = (df["start_s"] + df["end_s"]) / 2
    df["level"] = df.apply(
        lambda r: (
            "ALERT" if r["in_alert"] else ("above θ" if r["duress_score"] > threshold else "normal")
        ),
        axis=1,
    )
    return df


def score_chart(df: pd.DataFrame, threshold: float, segments: list[dict], duration: float):
    base = alt.Chart(df).encode(
        x=alt.X("center_s:Q", title="Call time (s)", scale=alt.Scale(domain=[0, duration])),
    )
    bars = base.mark_bar(size=14).encode(
        y=alt.Y("duress_score:Q", title="Duress confidence", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "level:N",
            scale=alt.Scale(domain=["normal", "above θ", "ALERT"], range=[GREY, AMBER, RED]),
            legend=alt.Legend(title=None, orient="top"),
        ),
        tooltip=["index", "start_s", "end_s", alt.Tooltip("duress_score:Q", format=".3f"), "level"],
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(color=RED, strokeDash=[6, 4], size=2)
        .encode(y="y:Q")
    )
    layers = [bars, rule]
    if segments:
        seg_df = pd.DataFrame(segments)
        layers.insert(
            0,
            alt.Chart(seg_df)
            .mark_rect(color=RED, opacity=0.15)
            .encode(x="start_s:Q", x2="end_s:Q"),
        )
    return alt.layer(*layers).properties(height=320)


def style_windows(df: pd.DataFrame, threshold: float):
    cols = ["index", "start_s", "end_s", "duress_score", "level"]

    def paint(row):
        if row["level"] == "ALERT":
            return [f"background-color:{RED}; color:white; font-weight:700"] * len(row)
        if row["duress_score"] > threshold:
            return [f"background-color:{AMBER}; color:black"] * len(row)
        return [""] * len(row)

    return (
        df[cols]
        .style.apply(paint, axis=1)
        .format({"duress_score": "{:.3f}", "start_s": "{:.1f}", "end_s": "{:.1f}"})
    )


def banner(result: dict, upto_window: int | None = None) -> str:
    segs = [
        s
        for s in result["alert_segments"]
        if upto_window is None
        or s["start_window"] + result["consecutive_windows"] - 1 <= upto_window
    ]
    if segs:
        spans = ", ".join(
            f"{s['start_s']:.1f}s–{s['end_s']:.1f}s (peak {s['max_score']:.2f})" for s in segs
        )
        return (
            f'<div class="alert-banner">🚨 PRIORITY ALERT — sustained duress indicators '
            f"in segment(s): {spans}. Review call immediately.</div>"
        )
    if upto_window is not None and upto_window < result["n_windows"] - 1:
        return '<div class="wait-banner">📡 Monitoring live call…</div>'
    return (
        '<div class="ok-banner">✅ No sustained stress pattern above '
        f"θ = {result['threshold']:.2f} detected.</div>"
    )


def render_result(result: dict, audio: bytes, simulate: bool, speed: float) -> None:
    threshold = result["threshold"]
    df = windows_frame(result["windows"], threshold)
    banner_ph, metrics_ph = st.empty(), st.container()
    chart_ph, table_ph = st.empty(), st.empty()

    with metrics_ph:
        c = st.columns(5)
        c[0].metric("Call duration", f"{result['duration_s']:.1f} s")
        c[1].metric("Windows scored", result["n_windows"])
        c[2].metric("Peak duress", f"{result['max_score']:.3f}")
        c[3].metric("Server latency", f"{result['latency_ms']:.0f} ms")
        c[4].metric("Alert rule", f"θ>{threshold:.2f} × {result['consecutive_windows']}")
        st.caption(
            f"Model `{result['model_version']}` · source `{result['model_source']}` · "
            f"call id `{result['call_id']}`"
        )
    st.audio(io.BytesIO(audio), format="audio/wav")

    if simulate and len(df) > 1:
        hop = float(df["start_s"].iloc[1] - df["start_s"].iloc[0])
        for i in range(len(df)):
            part = df.iloc[: i + 1]
            visible = [s for s in result["alert_segments"] if s["end_window"] <= i]
            banner_ph.markdown(banner(result, upto_window=i), unsafe_allow_html=True)
            chart_ph.altair_chart(
                score_chart(part, threshold, visible, result["duration_s"]), width="stretch"
            )
            time.sleep(hop / speed)
    banner_ph.markdown(banner(result), unsafe_allow_html=True)
    chart_ph.altair_chart(
        score_chart(df, threshold, result["alert_segments"], result["duration_s"]), width="stretch"
    )
    with table_ph.container():
        st.markdown("**Window-level scores** (red = part of an alert run, amber = above θ)")
        st.dataframe(
            style_windows(df, threshold),
            hide_index=True,
            width="stretch",
            height=min(420, 38 + 35 * len(df)),
        )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Console")
    api = st.text_input("stress-api URL", API_URL)
    code, health = api_health(api)
    if code == 200:
        st.success(f"API READY · {health.get('model_version')}")
    elif code == 503:
        st.warning(
            f"API {health.get('status', 'loading').upper()} (HTTP 503) — "
            f"{health.get('error') or 'model weights loading'}"
        )
    else:
        st.error("API unreachable")
    if st.button("🔄 Refresh status", width="stretch"):
        st.rerun()
    if st.button("⬆️ Hot-reload @champion", width="stretch", disabled=code == 0):
        try:
            r = requests.post(f"{api}/admin/reload", timeout=5)
            st.info(r.json())
        except requests.RequestException as exc:
            st.error(str(exc))
    st.divider()
    simulate = st.toggle(
        "Simulate live stream",
        value=True,
        help="Reveal windows one hop at a time, like a live call.",
    )
    speed = st.slider("Playback speed", 1.0, 8.0, 4.0, 0.5, disabled=not simulate)
    st.divider()
    st.caption(
        "Decision support only. A high score means the acoustic pattern resembles the "
        "training data's stressed class — it is **not** evidence of a threat. "
        "The dispatcher always decides."
    )

st.title("🚨 911 Dispatcher Console — Voice Stress Monitor")
tab_call, tab_audit, tab_status = st.tabs(["📞 Incoming call", "🗂️ Audit log", "🩺 System"])

with tab_call:
    left, right = st.columns([2, 1])
    with left:
        uploaded = st.file_uploader("Upload a call recording (16 kHz mono WAV)", type=["wav"])
    with right:
        demos = sorted(DEMO_DIR.glob("*.wav")) if DEMO_DIR.is_dir() else []
        demo_choice = st.selectbox("…or pick a demo call", ["—"] + [p.name for p in demos])

    audio_bytes, name = None, None
    if uploaded is not None:
        audio_bytes, name = uploaded.getvalue(), uploaded.name
    elif demo_choice != "—":
        audio_bytes, name = (DEMO_DIR / demo_choice).read_bytes(), demo_choice

    if st.button("▶️ Analyze call", type="primary", disabled=audio_bytes is None):
        with st.spinner("Scoring call windows…"):
            status, body = api_predict(api, name, audio_bytes)
        if status == 200:
            render_result(body, audio_bytes, simulate, speed)
        elif status == 422:
            detail = body.get("detail", {})
            st.error(f"⛔ Data contract violation: {detail.get('message', detail)}")
            st.caption("Convert with `uv run python scripts/convert_audio.py in.ext out.wav`.")
        elif status == 503:
            st.warning("Model is still loading (HTTP 503). Retry in a few seconds.")
        else:
            st.error(f"Request failed (HTTP {status}): {body}")

with tab_audit:
    stats = api_get(api, "/telemetry/stats") or {}
    c = st.columns(3)
    c[0].metric("Calls audited", stats.get("total_calls", "—"))
    c[1].metric("Calls with alert", stats.get("alert_calls", "—"))
    c[2].metric("Avg latency", f"{stats.get('avg_latency_ms') or 0:.0f} ms")
    calls = api_get(api, "/telemetry/calls?limit=100") or []
    if calls:
        cdf = pd.DataFrame(calls)[
            [
                "received_at",
                "call_id",
                "filename",
                "duration_s",
                "max_score",
                "alert",
                "model_version",
                "latency_ms",
            ]
        ]
        cdf["alert"] = cdf["alert"].astype(bool)
        st.dataframe(
            cdf.style.apply(
                lambda r: (
                    [f"background-color:{RED}; color:white"] * len(r)
                    if r["alert"]
                    else [""] * len(r)
                ),
                axis=1,
            ).format({"max_score": "{:.3f}", "duration_s": "{:.1f}", "latency_ms": "{:.0f}"}),
            hide_index=True,
            width="stretch",
        )
        pick = st.selectbox("Inspect call", cdf["call_id"].tolist())
        detail = api_get(api, f"/telemetry/calls/{pick}")
        if detail and detail.get("windows"):
            ddf = pd.DataFrame(detail["windows"]).rename(columns={"window_idx": "index"})
            ddf = windows_frame(ddf.to_dict("records"), detail["threshold"])
            st.altair_chart(
                score_chart(ddf, detail["threshold"], [], detail["duration_s"]), width="stretch"
            )
    else:
        st.info("No calls audited yet. Analyze a call first.")

with tab_status:
    st.subheader("Readiness probe — GET /health")
    st.code(f"HTTP {code}\n{health}", language="json")
    st.markdown(
        "- **503** while the model weights are loading or if loading failed\n"
        "- **200** once the @champion model is in memory and `/predict` is safe to call\n"
        "- Docker's healthcheck polls this endpoint; `dispatcher-ui` only starts after "
        "`stress-api` reports healthy."
    )
