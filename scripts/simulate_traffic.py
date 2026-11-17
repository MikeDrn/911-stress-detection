"""Replay labelled 911 calls through the running API (Course 4 demos).

Each simulated call combines a HELD-OUT transcript (unseen scenario families) with a
HELD-OUT speaker's real CREMA-D recording, chosen with the same stress profile the
quality gate uses (genuine emergencies mostly frightened voices, covert callers often
tense, non-emergency callers mostly calm). The transcript is sent with the audio
because CREMA-D actors read fixed sentences; with your own recorded calls the API's
speech-to-text produces the transcript instead.

After every call the true outcome is POSTed to /feedback ("threat" for covert calls,
"no_threat" otherwise), playing the dispatcher who learns afterwards what really
happened. That delayed label is the reward of the A/B test and the bandit.

    uv run python -m scripts.simulate_traffic                         # 200 calls
    uv run python -m scripts.simulate_traffic --calls 600 --until-decision
    uv run python -m scripts.simulate_traffic --channel phone         # data-shift traffic
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.smoke_test import _json, get, post_wav_bytes
from src.augment import phone_line, seed_for
from src.config import PROJECT_ROOT, load_params
from src.data_contracts import read_manifest
from src.transcripts import read_csv


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


def wav_bytes(path: Path, channel: str) -> bytes:
    y, sr = sf.read(path, dtype="float32")
    if channel == "phone":
        y = phone_line(y, np.random.default_rng(seed_for(path.name)), sr)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--calls", type=int, default=200)
    ap.add_argument("--channel", choices=["clean", "phone", "mix"], default="clean")
    ap.add_argument(
        "--covert-share",
        type=float,
        default=0.35,
        help="share of covert calls in the traffic (rest split genuine/non-emergency)",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--mode",
        choices=["champion_only", "shadow", "canary", "ab", "bandit"],
        help="start this rollout mode on the API before sending traffic "
        "(the API stays in champion_only until a mode is started)",
    )
    ap.add_argument("--policy", choices=["thompson", "ucb1", "epsilon_greedy"])
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument(
        "--until-decision",
        action="store_true",
        help="stop when a canary completes / rolls back or an A/B test reaches its planned sample",
    )
    args = ap.parse_args(argv)

    params = load_params()
    manifest = read_manifest(PROJECT_ROOT / params["data"]["manifest"])
    clips = {
        lab: [
            PROJECT_ROOT / r["path"]
            for r in manifest
            if r["split"] == "test"
            and r["label"] == lab
            and not Path(r["path"]).name.startswith("phone_")
        ]
        for lab in ("calm", "stressed")
    }
    texts = {}
    for r in read_csv(PROJECT_ROOT / params["text"]["transcripts"]):
        if r["split"] == "test":
            texts.setdefault(r["label"], []).append(r["text"])
    profile = params["quality_gate"]["stress_profile"]
    if not clips["calm"] or not texts:
        print("[sim] no held-out data - run `uv run dvc repro` first", file=sys.stderr)
        return 1

    code, health = get(f"{args.url}/health")
    if code != 200:
        print(f"[sim] API not ready (HTTP {code}): {health}", file=sys.stderr)
        return 1
    if args.mode:
        if args.mode != "champion_only" and not health.get("challenger_version"):
            print("[sim] no @challenger loaded - register V2 first", file=sys.stderr)
            return 1
        body = {"mode": args.mode, **({"policy": args.policy} if args.policy else {})}
        code, started = post_json(f"{args.url}/rollout", body)
        if code != 200:
            print(f"[sim] could not start {args.mode} (HTTP {code}): {started}", file=sys.stderr)
            return 1
    _, ro = get(f"{args.url}/rollout")
    print(
        f"[sim] mode={ro.get('mode')} status={ro.get('status')} "
        f"champion={ro.get('champion_version')} challenger={ro.get('challenger_version')}"
    )

    rng = random.Random(args.seed)
    run = uuid.uuid4().hex[:6]
    served = {"champion": 0, "challenger": 0}
    correct = {"champion": [0, 0], "challenger": [0, 0]}
    levels = {"alert": 0, "review": 0, "normal": 0}
    t0 = time.time()
    for i in range(1, args.calls + 1):
        r = rng.random()
        scenario = (
            "covert"
            if r < args.covert_share
            else "genuine_emergency"
            if r < args.covert_share + (1 - args.covert_share) / 2
            else "non_emergency"
        )
        voice = "stressed" if rng.random() < float(profile[scenario]) else "calm"
        channel = args.channel if args.channel != "mix" else rng.choice(["clean", "phone"])
        wav = wav_bytes(rng.choice(clips[voice]), channel)
        transcript = rng.choice(texts[scenario])
        call_id = f"sim-{run}-{i:05d}"
        code, body = post_wav_bytes(
            f"{args.url}/predict", wav, f"{call_id}.wav", call_id, transcript=transcript
        )
        if code != 200:
            print(f"[sim] call {i}: HTTP {code} {body.get('detail')}")
            continue
        arm = body.get("served_by", "champion")
        served[arm] += 1
        levels[body.get("threat_level", "normal")] += 1
        if not args.no_feedback:
            truth = "threat" if scenario == "covert" else "no_threat"
            fc, fb = post_json(f"{args.url}/feedback", {"call_id": call_id, "true_label": truth})
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
                f"[sim] {i:4d} calls | served {served} | correct {acc} | levels {levels}"
                f"{extra} | {i / (time.time() - t0):.1f} calls/s"
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
            acc = s["accuracy"] if s["accuracy"] is None else round(s["accuracy"], 3)
            print(
                f"[sim]   {arm:10s} served={s['calls']:4d} errors={s['errors']} "
                f"correct={acc} p95_ms/window={s['p95_ms_per_window']:.1f}"
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