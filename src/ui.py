"""Streamlit dispatcher console (``dispatcher-ui``).

The UI is intentionally a *thin client*: it holds no model and no database
handle. Everything goes through the ``stress-api`` REST contract over the
private Docker network (``http://stress-api:8000``), so the console can be
redeployed, restyled or replaced without touching inference or audit logging.

Tabs
----
Incoming call  analyse a call, see red alert segments, give ground-truth feedback
Release        Course 4: shadow / canary / A/B / bandit experiments, live
Audit log      every scored call from SQLite
System         readiness probe

Run locally:  streamlit run src/ui.py
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bandits  # noqa: E402  (pure-NumPy module shipped next to this file)

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
DEMO_DIR = Path(os.environ.get("DEMO_DIR", Path(__file__).resolve().parent.parent / "data/demo"))
RED, AMBER, GREY, GREEN, BLUE = "#d62828", "#f4a261", "#8d99ae", "#2a9d8f", "#3a86ff"
ARM_COLORS = alt.Scale(domain=["champion", "challenger"], range=[BLUE, AMBER])

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
    .decision {{ padding:14px 20px; border-radius:10px; color:white; font-weight:700;
                 font-size:1.1rem; }}
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
        r = requests.get(f"{base}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def api_post(base: str, path: str, payload: dict) -> tuple[int, dict]:
    try:
        r = requests.post(f"{base}{path}", json=payload, timeout=10)
        return r.status_code, r.json()
    except requests.RequestException as exc:
        return 0, {"detail": str(exc)}


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


# --------------------------------------------------------------------------- #
# Incoming-call rendering
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
        s for s in result["alert_segments"] if upto_window is None or s["end_window"] <= upto_window
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


def render_result(result: dict, audio: bytes, animate: bool, speed: float) -> None:
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
            f"Model `{result['model_version']}` · served by **{result.get('served_by')}** "
            f"· release mode `{result.get('rollout_mode')}` · call id `{result['call_id']}`"
        )
    st.audio(io.BytesIO(audio), format="audio/wav")

    if animate and len(df) > 1:
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
        score_chart(df, threshold, result["alert_segments"], result["duration_s"]),
        width="stretch",
    )
    with table_ph.container():
        st.markdown("**Window-level scores** (red = part of an alert run, amber = above θ)")
        st.dataframe(
            style_windows(df, threshold),
            hide_index=True,
            width="stretch",
            height=min(420, 38 + 35 * len(df)),
        )


def feedback_box(api: str, result: dict) -> None:
    st.markdown("##### Dispatcher feedback (ground truth)")
    st.caption(
        "After the call is resolved, record what really happened. This delayed label is "
        "the reward that A/B tests and bandits learn from (Course 4)."
    )
    done = st.session_state.get("feedback_done", {}).get(result["call_id"])
    c1, c2, _ = st.columns([1, 1, 2])
    for col, label, text in (
        (c1, "stressed", "🆘 Caller WAS under duress"),
        (c2, "calm", "🙂 Caller was NOT under duress"),
    ):
        if col.button(text, key=f"fb-{label}-{result['call_id']}", disabled=bool(done)):
            code, body = api_post(
                api, "/feedback", {"call_id": result["call_id"], "true_label": label}
            )
            if code == 200:
                st.session_state.setdefault("feedback_done", {})[result["call_id"]] = body
                st.rerun()
            else:
                st.error(f"Feedback failed (HTTP {code}): {body}")
    if done:
        verdict = (
            "✅ the alert decision was correct"
            if done["correct"]
            else "❌ the alert decision was wrong"
        )
        st.success(f"Recorded true label **{done['true_label']}** — {verdict}.")


# --------------------------------------------------------------------------- #
# Release tab (Course 4)
# --------------------------------------------------------------------------- #
MODE_HELP = {
    "champion_only": "Every call is scored by @champion. No experiment.",
    "shadow": "Is it correct on real inputs? @challenger scores every call silently after "
    "the response; dispatchers only ever see @champion.",
    "canary": "Is it safe? A sticky slice of calls goes to @challenger: 10% → 25% → 50% → "
    "100%. Guardrails are checked after every call; failure = automatic rollback.",
    "ab": "Is it better, with statistical confidence? Fixed 50/50 split, sample size decided "
    "in advance, two-sided z-test at the end. SIMULATED TRAFFIC ONLY in a 911 setting.",
    "bandit": "Which model should get traffic while we learn? Adaptive split that moves "
    "towards the better model (5% floor). SIMULATED TRAFFIC ONLY in a 911 setting.",
}


def decision_box(decision: dict | None) -> None:
    if not decision:
        st.info("No canary decision yet — waiting for traffic.")
        return
    color = {"advance": GREEN, "rollback": RED, "complete": BLUE, "wait": "#6c757d"}[
        decision["decision"]
    ]
    st.markdown(
        f'<div class="decision" style="background:{color}">Last decision (step '
        f"{decision['step']}): {decision['decision'].upper()} — {decision['reason']}</div>",
        unsafe_allow_html=True,
    )


def guardrail_table(checks: dict) -> pd.DataFrame:
    rows = []
    for name, c in checks.items():
        if name == "error_rate_z_test":
            value = (
                f"champion {pct(c['champion_rate'])} vs challenger "
                f"{pct(c['challenger_rate'])}, z = {c['z']:.2f}"
            )
            rule = f"rollback if z > {c['critical']}"
        elif name == "p95_latency":
            value = f"champion {c['champion_ms']:.1f} ms vs challenger {c['challenger_ms']:.1f} ms"
            rule = f"challenger ≤ {c['budget_ms']:.1f} ms (1.1 × champion + slack)"
        elif name == "no_class_collapse":
            value = (
                f"stressed windows: challenger {pct(c['challenger_stressed_share'])}, "
                f"champion {pct(c['champion_stressed_share'])}"
            )
            rule = "challenger not stuck on one class while champion is mixed"
        else:
            value = f"PSI = {c['psi']:.3f} ({c['band']}), KS = {c['ks']:.3f}"
            rule = "reported only: < 0.10 stable · 0.10–0.25 moderate · > 0.25 major"
        rows.append(
            {
                "guardrail": name,
                "observed": value,
                "rule": rule,
                "result": "PASS" if c["passed"] else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def paint_pass(row):
    color = GREEN if row["result"] == "PASS" else RED
    return [""] * (len(row) - 1) + [f"background-color:{color}; color:white; font-weight:700"]


def timeline_chart(timeline: list[dict], title: str):
    df = pd.DataFrame(timeline)
    if df.empty:
        return None
    df["is_challenger"] = (df["arm"] == "challenger").astype(float)
    df["challenger_share"] = df["is_challenger"].expanding().mean()
    df["rolling_share"] = df["is_challenger"].rolling(25, min_periods=1).mean()
    long = df.melt(
        id_vars=["i"],
        value_vars=["rolling_share", "challenger_share"],
        var_name="series",
        value_name="share",
    )
    long["series"] = long["series"].map(
        {"rolling_share": "last 25 calls", "challenger_share": "cumulative"}
    )
    st.markdown(f"**{title}**")
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("i:Q", title="Call #"),
            y=alt.Y(
                "share:Q",
                title="Challenger share",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("series:N", legend=alt.Legend(orient="bottom", title=None)),
        )
        .properties(height=280)
    )


def arms_table(arms: dict) -> pd.DataFrame:
    rows = []
    for arm, s in arms.items():
        rows.append(
            {
                "arm": arm,
                "calls served": s["calls"],
                "errors": s["errors"],
                "labelled": s["labelled"],
                "correct": s["correct"],
                "correct-decision rate": pct(s["accuracy"]),
                "alert rate": pct(s["alert_rate"]),
                "p95 ms / window": round(s["p95_ms_per_window"], 2),
            }
        )
    return pd.DataFrame(rows)


def render_shadow(rep: dict) -> None:
    sh = rep.get("shadow") or {}
    if not sh.get("calls"):
        st.info("No shadow-scored calls yet. Send traffic (simulator or Incoming call tab).")
        return
    c = st.columns(5)
    c[0].metric("Mirrored calls", sh["calls"])
    c[1].metric("Alert agreement", pct(sh["alert_agreement"]))
    c[2].metric("Champion correct", pct(sh["champion_accuracy"]))
    c[3].metric(
        "Challenger correct",
        pct(sh["challenger_accuracy"]),
        delta=None
        if sh["challenger_accuracy"] is None
        else f"{100 * (sh['challenger_accuracy'] - sh['champion_accuracy']):+.1f} pts",
    )
    c[4].metric("Challenger errors", sh["challenger_errors"])
    lat = pd.DataFrame(
        {
            "model": ["champion", "challenger"],
            "p95 ms per window": [
                sh["champion_p95_ms_per_window"],
                sh["challenger_p95_ms_per_window"],
            ],
            "alert rate": [sh["champion_alert_rate"], sh["challenger_alert_rate"]],
        }
    )
    left, right = st.columns(2)
    left.altair_chart(
        alt.Chart(lat)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title=None),
            y="p95 ms per window:Q",
            color=alt.Color("model:N", scale=ARM_COLORS, legend=None),
        )
        .properties(height=220, title="Latency (p95, per 2 s window)"),
        width="stretch",
    )
    if sh["champion_accuracy"] is not None:
        acc = pd.DataFrame(
            {
                "model": ["champion", "challenger"],
                "correct": [sh["champion_accuracy"], sh["challenger_accuracy"]],
            }
        )
        right.altair_chart(
            alt.Chart(acc)
            .mark_bar()
            .encode(
                x=alt.X("model:N", title=None),
                y=alt.Y(
                    "correct:Q",
                    title="Correct alert decisions",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color("model:N", scale=ARM_COLORS, legend=None),
            )
            .properties(height=220, title="Same calls, both models (labelled)"),
            width="stretch",
        )
    st.caption(
        "Shadow answers *does it run correctly on real inputs?* The challenger never "
        "influenced a dispatcher; its predictions were only logged."
    )


def render_canary(rep: dict) -> None:
    can = rep.get("canary") or {}
    steps, status, cur = can.get("steps", []), can.get("status"), can.get("current_step", 0)
    cols = st.columns(len(steps) or 1)
    for i, s in enumerate(steps):
        if status == "complete" or i < cur:
            mark = "🟢"
        elif i == cur:
            mark = "🔴" if status == "rolled_back" else "🟡"
        else:
            mark = "⚪"
        cols[i].markdown(f"### {mark} {s}%")
    decision_box((rep.get("canary") or {}).get("last_decision"))
    last = (rep.get("canary") or {}).get("last_decision") or {}
    if last.get("checks"):
        st.markdown("**Guardrails at the last decision** (same time window, v2 vs v1)")
        st.dataframe(
            guardrail_table(last["checks"]).style.apply(paint_pass, axis=1),
            hide_index=True,
            width="stretch",
        )
    chart = timeline_chart(rep.get("timeline", []), "Traffic ramp: share served by @challenger")
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    st.caption(
        "PSI compares the score distributions of v2 and v1. After a data shift a large PSI "
        "is expected (v1 is the one that is wrong on phone-line calls), which is why the "
        "course treats PSI as a rule of thumb to investigate, not an automatic rollback rule."
    )
    per_step = (rep.get("canary") or {}).get("per_step", [])
    if per_step:
        st.dataframe(pd.DataFrame(per_step), hide_index=True)


def render_ab(rep: dict) -> None:
    ab = rep.get("ab") or {}
    if not ab:
        return
    verdict = ab["verdict"]
    color = (
        GREEN
        if verdict.startswith("B BETTER")
        else RED
        if verdict.startswith(("B WORSE", "INVALID"))
        else "#6c757d"
    )
    st.markdown(
        f'<div class="decision" style="background:{color}">{verdict}</div>', unsafe_allow_html=True
    )
    arms = rep["arms"]
    done = min(arms["champion"]["labelled"], arms["challenger"]["labelled"])
    st.progress(
        min(1.0, done / max(1, ab["planned_n_per_arm"])),
        text=f"{done} / {ab['planned_n_per_arm']} labelled calls per arm (planned "
        "before the test — no peeking)",
    )
    c = st.columns(6)
    c[0].metric("A (champion) correct", pct(ab["p_a"]))
    c[1].metric("B (challenger) correct", pct(ab["p_b"]))
    c[2].metric("Difference B − A", f"{100 * ab['diff']:+.1f} pts")
    c[3].metric("z", f"{ab['z']:.2f}")
    c[4].metric("p-value", f"{ab['p_value']:.4f}")
    c[5].metric(
        "SRM p-value",
        f"{ab['srm_p_value']:.3f}",
        help="Sample-ratio mismatch: a tiny value means the 50/50 split is broken",
    )
    ci = pd.DataFrame(
        {
            "label": ["B − A"],
            "diff": [100 * ab["diff"]],
            "lo": [100 * ab["ci_low"]],
            "hi": [100 * ab["ci_high"]],
        }
    )
    base = alt.Chart(ci).encode(y=alt.Y("label:N", title=None))
    chart = (
        base.mark_rule(strokeWidth=4, color=BLUE).encode(
            x=alt.X(
                "lo:Q", title="Difference in correct-decision rate (percentage points), 95% CI"
            ),
            x2="hi:Q",
        )
        + base.mark_point(size=200, filled=True, color=BLUE).encode(x="diff:Q")
        + alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=RED, strokeDash=[5, 4])
        .encode(x="x:Q")
    ).properties(height=110)
    st.altair_chart(chart, width="stretch")
    st.caption(
        f"95% CI = diff ± 1.96 × SE = [{100 * ab['ci_low']:.1f}, "
        f"{100 * ab['ci_high']:.1f}] pts (unpooled SE = {100 * ab['se']:.2f} pts). "
        "If the interval excludes 0 the difference is significant at the 5% level."
    )


def render_bandit(rep: dict) -> None:
    b = rep.get("bandit")
    chart = timeline_chart(
        rep.get("timeline", []), f"Adaptive allocation ({(rep.get('rollout') or {}).get('policy')})"
    )
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    if b:
        c = st.columns(3)
        c[0].metric("Estimated regret (lost correct decisions)", f"{b['estimated_regret']:.1f}")
        c[1].metric("Same traffic as a 50/50 A/B test", f"{b['ab_test_equivalent_regret']:.1f}")
        c[2].metric(
            "Calls saved from the weaker model",
            f"{b['ab_test_equivalent_regret'] - b['estimated_regret']:.1f}",
        )
    tl = pd.DataFrame(rep.get("timeline", []))
    if not tl.empty:
        st.altair_chart(
            alt.Chart(tl)
            .mark_circle(size=30, opacity=0.6)
            .encode(
                x=alt.X("i:Q", title="Call #"),
                y=alt.Y("propensity:Q", title="P(chosen arm)", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("arm:N", scale=ARM_COLORS),
            )
            .properties(height=200, title="Logged propensities (enable off-policy evaluation)"),
            width="stretch",
        )


def render_offline_regret(rep: dict | None) -> None:
    with st.expander("📉 Offline regret simulation: ε-greedy vs UCB1 vs Thompson vs A/B"):
        arms = (rep or {}).get("arms") or {}
        acc_a = (arms.get("champion") or {}).get("accuracy") or 0.85
        acc_b = (arms.get("challenger") or {}).get("accuracy") or 0.95
        c = st.columns(3)
        mu_a = c[0].slider("True reward of champion", 0.5, 1.0, float(round(acc_a, 2)), 0.01)
        mu_b = c[1].slider("True reward of challenger", 0.5, 1.0, float(round(acc_b, 2)), 0.01)
        rounds = c[2].select_slider("Rounds", [200, 500, 1000, 2000], value=1000)
        if st.button("Run simulation (20 seeds)"):
            with st.spinner("Simulating…"):
                curves = bandits.simulate_regret([mu_a, mu_b], rounds=rounds, seeds=20)
            df = pd.DataFrame({k: v for k, v in curves.items()})
            df["round"] = np.arange(1, rounds + 1)
            long = df.melt("round", var_name="policy", value_name="regret")
            st.altair_chart(
                alt.Chart(long)
                .mark_line(strokeWidth=2.5)
                .encode(
                    x="round:Q",
                    y=alt.Y("regret:Q", title="Cumulative regret"),
                    color=alt.Color("policy:N", legend=alt.Legend(orient="top", title=None)),
                )
                .properties(height=300),
                width="stretch",
            )
            final = {k: round(float(v[-1]), 1) for k, v in curves.items()}
            st.caption(
                f"Final regret after {rounds} rounds: {final}. A 50/50 A/B test has "
                "linear regret (Δ × T / 2); good bandits grow logarithmically."
            )


def release_tab(api: str, code: int, health: dict) -> None:
    ro = api_get(api, "/rollout") or {}
    c = st.columns(4)
    c[0].metric("Release mode", ro.get("mode", "—"))
    c[1].metric("Status", ro.get("status", "—"))
    c[2].metric("@champion", (ro.get("champion_version") or "—").split("@")[-1])
    c[3].metric("@challenger", (ro.get("challenger_version") or "none").split("@")[-1])
    if ro.get("experiment_id"):
        st.caption(f"Experiment `{ro['experiment_id']}` started {ro.get('started_at')}")

    with st.expander("🎛️ Start a new experiment", expanded=not ro.get("experiment_id")):
        mode = st.radio(
            "Strategy",
            list(MODE_HELP),
            horizontal=True,
            format_func=lambda m: {
                "champion_only": "Champion only",
                "shadow": "Shadow",
                "canary": "Canary",
                "ab": "A/B test",
                "bandit": "Bandit",
            }[m],
        )
        st.info(MODE_HELP[mode])
        payload: dict = {"mode": mode}
        if mode == "bandit":
            payload["policy"] = st.selectbox("Policy", ["thompson", "ucb1", "epsilon_greedy"])
        if mode == "ab":
            a1, a2 = st.columns(2)
            payload["baseline_rate"] = a1.number_input(
                "Expected champion correct rate", 0.5, 0.99, 0.85, 0.01
            )
            payload["mde"] = a2.number_input(
                "Minimum detectable effect (absolute)", 0.02, 0.3, 0.10, 0.01
            )
        if st.button("▶️ Start", type="primary", disabled=code != 200):
            status, body = api_post(api, "/rollout", payload)
            if status == 200:
                st.success(f"Started `{body.get('experiment_id') or 'champion_only'}`")
                st.rerun()
            else:
                st.error(f"HTTP {status}: {body.get('detail', body)}")

    exps = api_get(api, "/rollout/experiments") or []
    cur = ro.get("experiment_id")
    options = ([cur] if cur else []) + [
        e["experiment_id"] for e in exps if e["experiment_id"] != cur
    ]
    left, right = st.columns([3, 1])
    chosen = left.selectbox(
        "Experiment to display",
        options or ["—"],
        format_func=lambda e: f"{e}  (current)" if e == cur else e,
    )
    live = right.toggle("Auto-refresh every 3 s", value=False)

    @st.fragment(run_every=3 if live else None)
    def live_report():
        query = f"?experiment_id={chosen}" if chosen and chosen != "—" else ""
        rep = api_get(api, f"/rollout/report{query}") or {}
        if rep.get("experiment_id"):
            label = "Live report" if rep.get("is_current") else "Finished experiment"
            st.markdown(f"#### {label} — `{rep['experiment_id']}` · {rep['n_calls']} calls")
            mode = rep.get("mode")
            if mode != "shadow":  # in shadow mode the challenger serves nobody
                st.dataframe(arms_table(rep["arms"]), hide_index=True, width="stretch")
            if mode == "shadow":
                render_shadow(rep)
            elif mode == "canary":
                render_canary(rep)
            elif mode == "ab":
                render_ab(rep)
            elif mode == "bandit":
                render_bandit(rep)
        else:
            st.info(
                "No experiment running. Start one above, then send traffic with "
                "`uv run python -m scripts.simulate_traffic`."
            )
        render_offline_regret(rep)
        events = api_get(api, "/rollout/events?limit=50") or []
        if events:
            st.markdown("#### Release audit trail")
            ev = pd.DataFrame(events)[["at", "experiment_id", "event"]]
            st.dataframe(
                ev.style.apply(
                    lambda r: (
                        [f"background-color:{RED}; color:white"] * len(r)
                        if "rolled_back" in r["event"]
                        else [""] * len(r)
                    ),
                    axis=1,
                ),
                hide_index=True,
                width="stretch",
                height=240,
            )

    live_report()


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Console")
    api = st.text_input("stress-api URL", API_URL)
    code, health = api_health(api)
    if code == 200:
        st.success(f"API READY · {health.get('model_version')}")
        if health.get("challenger_version"):
            st.caption(f"challenger loaded: `{health['challenger_version']}`")
        st.caption(f"release mode: `{health.get('rollout_mode')}`")
    elif code == 503:
        st.warning(
            f"API {health.get('status', 'loading').upper()} (HTTP 503) — "
            f"{health.get('error') or 'model weights loading'}"
        )
    else:
        st.error("API unreachable")
    if st.button("🔄 Refresh status", width="stretch"):
        st.rerun()
    if st.button("⬆️ Hot-reload models", width="stretch", disabled=code == 0):
        try:
            r = requests.post(f"{api}/admin/reload", timeout=5)
            st.info(r.json())
        except requests.RequestException as exc:
            st.error(str(exc))
    st.divider()
    simulate = st.toggle(
        "Simulate live stream",
        value=True,
        help="Replay the scored windows one hop at a time, like a live call.",
    )
    speed = st.slider("Playback speed", 1.0, 8.0, 4.0, 0.5, disabled=not simulate)
    st.divider()
    st.caption(
        "Decision support only. A high score means the acoustic pattern resembles the "
        "training data's stressed class — it is **not** evidence of a threat. "
        "The dispatcher always decides."
    )

st.title("🚨 911 Dispatcher Console — Voice Stress Monitor")
tab_call, tab_release, tab_audit, tab_status = st.tabs(
    ["📞 Incoming call", "🚦 Release (canary · A/B · bandit)", "🗂️ Audit log", "🩺 System"]
)

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
            st.session_state["last"] = {"result": body, "audio": audio_bytes, "fresh": True}
        elif status == 422:
            st.session_state.pop("last", None)
            detail = body.get("detail", {})
            st.error(f"⛔ Data contract violation: {detail.get('message', detail)}")
            st.caption("Convert with `uv run python -m scripts.convert_audio in.ext out.wav`.")
        elif status == 503:
            st.warning("Model is still loading (HTTP 503). Retry in a few seconds.")
        else:
            st.error(f"Request failed (HTTP {status}): {body}")

    last = st.session_state.get("last")
    if last:
        render_result(last["result"], last["audio"], simulate and last["fresh"], speed)
        last["fresh"] = False
        feedback_box(api, last["result"])

with tab_release:
    release_tab(api, code, health)

with tab_audit:
    stats = api_get(api, "/telemetry/stats") or {}
    c = st.columns(4)
    c[0].metric("Calls audited", stats.get("total_calls", "—"))
    c[1].metric("Calls with alert", stats.get("alert_calls", "—"))
    c[2].metric("Calls with feedback", stats.get("labelled_calls", "—"))
    c[3].metric("Avg latency", f"{stats.get('avg_latency_ms') or 0:.0f} ms")
    calls = api_get(api, "/telemetry/calls?limit=200") or []
    if calls:
        cols = [
            "received_at",
            "call_id",
            "filename",
            "mode",
            "arm",
            "max_score",
            "alert",
            "true_label",
            "correct",
            "model_version",
            "latency_ms",
        ]
        cdf = pd.DataFrame(calls)
        cdf = cdf[[c for c in cols if c in cdf.columns]]
        cdf["alert"] = cdf["alert"].astype(bool)
        st.dataframe(
            cdf.style.apply(
                lambda r: (
                    [f"background-color:{RED}; color:white"] * len(r)
                    if r["alert"]
                    else [""] * len(r)
                ),
                axis=1,
            ).format({"max_score": "{:.3f}", "latency_ms": "{:.0f}"}),
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
        "`stress-api` reports healthy.\n"
        "- The API also polls the MLflow registry every 30 s and reloads when the "
        "`@champion` / `@challenger` aliases move (no restart needed)."
    )
