"""Post-deployment smoke test against a running stress-api.

Polls the readiness probe (printing every 503 -> 200 transition), then sends
each demo call to /predict and prints the verdict.

    uv run python -m scripts.smoke_test                       # http://localhost:8000
    uv run python -m scripts.smoke_test --url http://localhost:8000 --timeout 90
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from src.config import PROJECT_ROOT


def _json(raw: bytes) -> dict:
    try:
        return json.loads(raw or b"{}")
    except json.JSONDecodeError:
        return {"detail": raw.decode(errors="replace")[:300]}


def get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, _json(e.read())
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        return 0, {"status": f"unreachable ({e})"}


def post_wav_bytes(
    url: str, data: bytes, filename: str, call_id: str | None = None
) -> tuple[int, dict]:
    boundary = uuid.uuid4().hex
    parts = [
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'
        ).encode()
        + data
        + b"\r\n"
    ]
    if call_id:
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="call_id"\r\n\r\n'
                f"{call_id}\r\n"
            ).encode()
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, _json(e.read())


def post_wav(url: str, path: Path) -> tuple[int, dict]:
    return post_wav_bytes(url, path.read_bytes(), path.name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args(argv)

    deadline, last = time.time() + args.timeout, None
    while True:
        code, body = get(f"{args.url}/health")
        if code != last:
            print(f"[smoke] GET /health -> {code} {body.get('status')}")
            last = code
        if code == 200:
            print(f"[smoke] model: {body.get('model_version')} ({body.get('model_source')})")
            break
        if time.time() > deadline:
            print("[smoke] API never became ready", file=sys.stderr)
            return 1
        time.sleep(1)

    demo = sorted((PROJECT_ROOT / "data" / "demo").glob("*.wav"))
    if not demo:
        print("[smoke] no demo calls; run: uv run python -m scripts.make_demo_calls")
        return 1
    for p in demo:
        code, body = post_wav(f"{args.url}/predict", p)
        if code == 200:
            verdict = "ALERT" if body["alert"] else "no alert"
            print(
                f"[smoke] {p.name:42s} HTTP 200  max={body['max_score']:.3f}  "
                f"windows={body['n_windows']:3d}  {verdict}  ({body['latency_ms']:.0f} ms)"
            )
        else:
            print(f"[smoke] {p.name:42s} HTTP {code}  {body.get('detail')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
