"""Download the real-voice training data: CREMA-D (neutral vs fear) into data/raw/.

CREMA-D (Cao et al., 2014) contains 7,442 clips of 91 actors (48 male, 43 female,
ages 20-74, several ethnicities) reading 12 sentences in six emotions. Licence:
Open Database License (ODbL) / Database Contents License - free for academic use.
Source: https://github.com/CheyneyComputerScience/CREMA-D

Mapping used by this project (an *acted* proxy for real duress - say so on a slide):
    NEU (neutral) -> data/raw/calm/cremad_<file>.wav
    FEA (fear)    -> data/raw/stressed/cremad_<file>.wav
Anger is excluded on purpose: an angry caller is not the same as a frightened one,
and including it would also unbalance the classes.

Files are already 16 kHz mono PCM WAV, so they satisfy the ingestion contract
unchanged. Speaker demographics go to data/raw/speakers.csv (used for the
speaker-level split and the sex slice tests).

    uv run python -m scripts.download_cremad                # ~2,350 files, ~200 MB
    uv run python -m scripts.download_cremad --actors 20    # quick subset for testing
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import PROJECT_ROOT, load_params

MEDIA = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV"
RAW = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master"
SENTENCES = ["IEO", "TIE", "IOM", "IWW", "TAI", "MTI", "IWL", "ITH", "DFA", "ITS", "TSI", "WSI"]
EMOTIONS = {"NEU": "calm", "FEA": "stressed"}


def candidates(actors: list[int]) -> list[tuple[str, str]]:
    """All possible file names; the 'IEO' sentence also exists at three intensity levels."""
    out = []
    for actor in actors:
        for emo, label in EMOTIONS.items():
            for sent in SENTENCES:
                for level in ["XX", "LO", "MD", "HI"] if sent == "IEO" else ["XX"]:
                    out.append((f"{actor}_{sent}_{emo}_{level}.wav", label))
    return out


def fetch(url: str, retries: int = 4) -> bytes | None:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None  # this emotion/level combination does not exist
            time.sleep(1 + attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(1 + attempt)
    raise RuntimeError(f"could not download {url}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=load_params()["data"]["raw_dir"])
    ap.add_argument("--actors", type=int, default=91, help="first N actors (1001...)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--clean", action="store_true", help="delete data/raw first (removes old synthetic files)"
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out = out if out.is_absolute() else PROJECT_ROOT / out
    if args.clean and out.exists():
        shutil.rmtree(out)
    for label in set(EMOTIONS.values()):
        (out / label).mkdir(parents=True, exist_ok=True)

    demo = fetch(f"{RAW}/VideoDemographics.csv")
    rows = list(csv.DictReader(io.StringIO(demo.decode())))
    actors = sorted(int(r["ActorID"]) for r in rows)[: args.actors]
    with open(out / "speakers.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["speaker", "sex", "age"])
        for r in rows:
            if int(r["ActorID"]) in actors:
                w.writerow([f"cremad{r['ActorID']}", r["Sex"].lower(), r["Age"]])

    todo = [(n, lab) for n, lab in candidates(actors) if not (out / lab / f"cremad_{n}").exists()]
    print(f"[cremad] {len(actors)} actors, checking {len(todo)} candidate files ...")
    got = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, f"{MEDIA}/{n}"): (n, lab) for n, lab in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            name, label = futures[fut]
            data = fut.result()
            if data:
                (out / label / f"cremad_{name}").write_bytes(data)
                got += 1
            if i % 250 == 0 or i == len(todo):
                print(f"[cremad] {i}/{len(todo)} checked, {got} downloaded")
    for label in sorted(set(EMOTIONS.values())):
        n = len(list((out / label).glob("cremad_*.wav")))
        print(f"[cremad] {label:9s} {n} files in {out / label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
