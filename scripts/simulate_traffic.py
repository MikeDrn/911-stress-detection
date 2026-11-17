"""Replay labelled synthetic 911 calls through the running API (Course 4 demos).

Each simulated call is POSTed to /predict, then the true label is POSTed to
/feedback, playing the role of the dispatcher who learns afterwards whether the
caller really was under duress. That delayed label is the reward used by the
A/B test and the bandit, and the accuracy shown in shadow mode.

    uv run python -m scripts.simulate_traffic                       # 200 phone-line calls
    uv run python -m scripts.simulate_traffic --calls 400 --until-decision
    uv run python -m scripts.simulate_traffic --channel mix --stressed-share 0.3

``--channel phone`` (default) reproduces the data-shift scenario: new calls
arrive over narrow-band, noisy mobile lines.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
import soundfile as sf

from scripts.smoke_test import _json, get, post_wav_bytes
from src.synthetic import synth_utterance

SR = 16000


def post_json(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, _json(e.read())


def make_pool(n: int, channel: str, seed: int) -> list[tuple[bytes, str]]:
    """Pre-render a pool of labelled calls (half calm, half stressed)."""
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(n):
        label = "stressed" if i % 2 else "calm"
        ch = channel if channel != "mix" else ("phone" if rng.random() < 0.5 else "clean")
        y = synth_utterance(float(rng.uniform(4.0, 6.0)), label == "stressed", rng, SR, channel=ch)
        buf = io.BytesIO()
        sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
        pool.append((buf.getvalue(), label))
    return pool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--channel", choices=["phone", "clean", "mix"], default="phone")
    ap.add_argument("--stressed-share", type=float, default=0.5)
    ap.add_argument("--pool", type=int, default=120, help="distinct synthetic calls to cycle")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-feedback", action="store_true", help="do not send ground truth")
    ap.add_argument(
        "--until-decision",
        action="store_true",
        help="stop as soon as a canary completes / rolls back or an A/B test "
        "reaches its planned sample",
    )
    args = ap.parse_args(argv)

    code, health = get(f"{args.url}/health")
    if code != 200:
        print(f"[sim] API not ready (HTTP {code}): {health}", file=sys.stderr)
        return 1
    _, ro = get(f"{args.url}/rollout")
    print(
        f"[sim] mode={ro.get('mode')} status={ro.get('status')} "
        f"champion={ro.get('champion_version')} challenger={ro.get('challenger_version')}"
    )
    print(f"[sim] rendering {args.pool} {args.channel}-line calls ...")
    pool = make_pool(args.pool, args.channel, args.seed)
    calm = [p for p in pool if p[1] == "calm"]
    stressed = [p for p in pool if p[1] == "stressed"]

    rng = np.random.default_rng(args.seed + 1)
    run = uuid.uuid4().hex[:6]
    served = {"champion": 0, "challenger": 0}
    correct = {"champion": [0, 0], "challenger": [0, 0]}
    t0 = time.time()
    for i in range(1, args.calls + 1):
        source = stressed if rng.random() < args.stressed_share else calm
        wav, label = source[int(rng.integers(len(source)))]
        call_id = f"sim-{run}-{i:05d}"
        code, body = post_wav_bytes(f"{args.url}/predict", wav, f"{call_id}.wav", call_id)
        if code != 200:
            print(f"[sim] call {i}: HTTP {code} {body.get('detail')}")
            continue
        arm = body.get("served_by", "champion")
        served[arm] += 1
        if not args.no_feedback:
            fc, fb = post_json(f"{args.url}/feedback", {"call_id": call_id, "true_label": label})
            if fc == 200:
                correct[arm][0] += int(fb["correct"])
                correct[arm][1] += 1
        if i % 20 == 0 or i == args.calls:
            _, ro = get(f"{args.url}/rollout")
            extra = ""
            if ro.get("mode") == "canary":
                extra = f" canary={ro.get('challenger_percent')}% status={ro.get('status')}"
            acc = {a: (f"{c / n:.0%}" if n else "-") for a, (c, n) in correct.items()}
            print(
                f"[sim] {i:4d} calls | served {served} | correct {acc}{extra} "
                f"| {i / (time.time() - t0):.1f} calls/s"
            )
            if args.until_decision:
                if ro.get("mode") == "canary" and ro.get("status") in ("complete", "rolled_back"):
                    print(f"[sim] canary finished: {ro.get('status')}")
                    break
                if ro.get("mode") == "ab":
                    _, rep = get(f"{args.url}/rollout/report")
                    if (rep.get("ab") or {}).get("sample_reached"):
                        print("[sim] A/B planned sample reached")
                        break

    _, rep = get(f"{args.url}/rollout/report")
    if rep.get("experiment_id"):
        print(f"[sim] experiment {rep['experiment_id']} ({rep['mode']}), {rep['n_calls']} calls")
        for arm, s in rep["arms"].items():
            print(
                f"[sim]   {arm:10s} served={s['calls']:4d} errors={s['errors']} "
                f"correct={s['accuracy'] if s['accuracy'] is None else round(s['accuracy'], 3)}"
                f" p95_ms/window={s['p95_ms_per_window']:.1f}"
            )
        for key in ("shadow", "ab", "bandit"):
            if rep.get(key):
                print(
                    f"[sim]   {key}: "
                    + json.dumps(
                        {
                            k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in rep[key].items()
                            if not isinstance(v, dict)
                        }
                    )
                )
        if rep.get("canary"):
            d = rep["canary"].get("last_decision") or {}
            print(f"[sim]   canary last decision: {d.get('decision')} - {d.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
