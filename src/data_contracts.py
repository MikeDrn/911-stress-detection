"""Dataset manifest builder and data-contract validator.

Two DVC stages use this module:

``prepare``  (``python -m src.data_contracts prepare``)
    Scans ``data/raw/<class>/*.wav``, hashes every file with SHA-256 and writes
    a stratified train/val/test manifest. Byte-identical files are grouped by
    hash *before* splitting so a duplicate can never straddle two splits.

``validate`` (``python -m src.data_contracts validate``)
    Re-checks the manifest against the contract and fails the pipeline
    (non-zero exit) on any violation:

    1. Format      - every file is a readable 16 kHz mono PCM/float WAV with a
                     duration inside the configured bounds.
    2. Integrity   - the SHA-256 recorded in the manifest still matches the file.
    3. Leakage     - no SHA-256 appears in more than one split.
    4. Distribution floor - each class holds >= ``min_class_ratio`` of clips,
                     overall and within every split.

A JSON report is written to ``reports/data_contract.json`` either way.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT, load_params
from src.preprocessing import AudioContractError, AudioSpec, inspect_audio, validate_header

MANIFEST_FIELDS = [
    "path",
    "label",
    "label_idx",
    "split",
    "sha256",
    "duration_s",
    "sample_rate",
    "channels",
    "speaker",
]
SPLITS = ("train", "val", "test")
_SPEAKER_RE = re.compile(r"^(?:phone_)?cremad_(\d{4})_")


def speaker_of(path: str | Path) -> str:
    """Speaker id from the file name (CREMA-D actor), or '' when unknown.

    Phone-line augmented copies (``phone_cremad_1001_...``) map to the same actor,
    so an original and its augmented copy always land in the same split.
    """
    m = _SPEAKER_RE.match(Path(path).name)
    return f"cremad{m.group(1)}" if m else ""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Manifest construction
# --------------------------------------------------------------------------- #
def scan_raw(raw_dir: Path, classes: list[str]) -> list[dict[str, Any]]:
    rows = []
    for idx, label in enumerate(classes):
        class_dir = raw_dir / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for wav in sorted(class_dir.glob("*.wav")):
            rows.append(
                {
                    "path": wav.relative_to(PROJECT_ROOT).as_posix()
                    if wav.resolve().is_relative_to(PROJECT_ROOT)
                    else wav.as_posix(),
                    "abs_path": wav,
                    "label": label,
                    "label_idx": idx,
                    "sha256": sha256_file(wav),
                    "speaker": speaker_of(wav),
                }
            )
    return rows


def assign_splits(
    rows: list[dict[str, Any]], test_fraction: float, val_fraction: float, seed: int
) -> None:
    """Group-level split (in place).

    * Rows with a known speaker are split **by speaker**: every clip of an actor
      lands in the same split, so the model can never recognise a voice it was
      trained on (speaker leakage - invisible to a file-hash check).
    * Rows without a speaker are split by content hash, stratified by label, so
      byte-identical duplicates can never straddle two splits.
    """
    rng = np.random.default_rng(seed)

    def cut(keys: list[str]) -> dict[str, str]:
        keys = sorted(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction))) if n >= 3 else 0
        return {
            k: "test" if i < n_test else "val" if i < n_test + n_val else "train"
            for i, k in enumerate(keys)
        }

    speakers = sorted({r["speaker"] for r in rows if r.get("speaker")})
    split_of_speaker = cut(speakers) if speakers else {}

    by_label: dict[str, list[str]] = defaultdict(list)
    seen: set[str] = set()
    for r in rows:
        if not r.get("speaker") and r["sha256"] not in seen:
            seen.add(r["sha256"])
            by_label[r["label"]].append(r["sha256"])
    split_of_hash: dict[str, str] = {}
    for _, hashes in sorted(by_label.items()):
        split_of_hash.update(cut(hashes))

    for r in rows:
        r["split"] = (
            split_of_speaker[r["speaker"]] if r.get("speaker") else split_of_hash[r["sha256"]]
        )


def check_speaker_leakage(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return {speaker: [splits...]} for every speaker present in more than one split."""
    splits_by_speaker: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("speaker"):
            splits_by_speaker[r["speaker"]].add(r["split"])
    return {s: sorted(v) for s, v in splits_by_speaker.items() if len(v) > 1}


def build_manifest(params: dict[str, Any] | None = None) -> Path:
    params = params or load_params()
    d = params["data"]
    raw_dir = PROJECT_ROOT / d["raw_dir"]
    rows = scan_raw(raw_dir, d["classes"])
    if not rows:
        raise SystemExit(f"No WAV files found under {raw_dir}/<class>/")
    assign_splits(rows, d["test_fraction"], d["val_fraction"], d["seed"])

    for r in rows:
        try:
            info = inspect_audio(r["abs_path"])
            r.update(
                duration_s=round(info.frames / info.samplerate, 4),
                sample_rate=info.samplerate,
                channels=info.channels,
            )
        except AudioContractError:
            r.update(duration_s=0.0, sample_rate=0, channels=0)  # validate stage will reject

    out = PROJECT_ROOT / d["manifest"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["path"]))
    counts = Counter((r["split"], r["label"]) for r in rows)
    print(f"[prepare] wrote {len(rows)} rows -> {out}")
    for (split, label), c in sorted(counts.items()):
        print(f"           {split:5s} {label:9s} {c}")
    return out


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# Contract checks (pure functions -> easy to unit test)
# --------------------------------------------------------------------------- #
@dataclass
class ContractReport:
    passed: bool = True
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self, name: str, ok: bool, **details: Any) -> None:
        self.checks[name] = {"passed": bool(ok), **details}
        self.passed = self.passed and bool(ok)


def check_format(rows: list[dict[str, Any]], spec: AudioSpec, root: Path) -> list[dict[str, str]]:
    violations = []
    for r in rows:
        p = Path(r["path"])
        p = p if p.is_absolute() else root / p
        try:
            validate_header(inspect_audio(p), spec)
        except (AudioContractError, FileNotFoundError) as exc:
            violations.append({"path": r["path"], "error": str(exc)})
    return violations


def check_integrity(rows: list[dict[str, Any]], root: Path) -> list[str]:
    bad = []
    for r in rows:
        p = Path(r["path"])
        p = p if p.is_absolute() else root / p
        if not p.exists() or sha256_file(p) != r["sha256"]:
            bad.append(r["path"])
    return bad


def check_leakage(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return {sha256: [splits...]} for every hash present in >1 split."""
    splits_by_hash: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        splits_by_hash[r["sha256"]].add(r["split"])
    return {h: sorted(s) for h, s in splits_by_hash.items() if len(s) > 1}


def check_label_conflicts(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Identical audio carrying different labels is a labelling bug."""
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        labels_by_hash[r["sha256"]].add(r["label"])
    return {h: sorted(s) for h, s in labels_by_hash.items() if len(s) > 1}


def class_ratios(rows: list[dict[str, Any]], classes: list[str]) -> dict[str, float]:
    counts = Counter(r["label"] for r in rows)
    total = sum(counts.values()) or 1
    return {c: counts.get(c, 0) / total for c in classes}


def validate_manifest(
    manifest_path: str | Path,
    params: dict[str, Any] | None = None,
    root: Path = PROJECT_ROOT,
) -> ContractReport:
    params = params or load_params()
    d = params["data"]
    spec = AudioSpec.from_params(params)
    rows = read_manifest(manifest_path)
    report = ContractReport()

    report.record("non_empty", len(rows) > 0, n_rows=len(rows))

    fmt = check_format(rows, spec, root)
    report.record("audio_format_16k_mono_wav", not fmt, violations=fmt[:50], n_violations=len(fmt))

    integ = check_integrity(rows, root)
    report.record("sha256_integrity", not integ, mismatched=integ[:50])

    leaks = check_leakage(rows)
    report.record("no_train_test_leakage_sha256", not leaks, leaked_hashes=leaks)

    spk = check_speaker_leakage(rows)
    n_speakers = len({r.get("speaker") for r in rows if r.get("speaker")})
    report.record("no_speaker_leakage", not spk, n_speakers=n_speakers, leaked_speakers=spk)

    conflicts = check_label_conflicts(rows)
    report.record("no_label_conflicts", not conflicts, conflicts=conflicts)

    floor = float(d["min_class_ratio"])
    overall = class_ratios(rows, d["classes"])
    per_split = {
        s: class_ratios([r for r in rows if r["split"] == s], d["classes"]) for s in SPLITS
    }
    ok_floor = all(v >= floor for v in overall.values()) and all(
        v >= floor for ratios in per_split.values() for v in ratios.values()
    )
    report.record(
        "class_distribution_floor", ok_floor, floor=floor, overall=overall, per_split=per_split
    )

    present = {r["split"] for r in rows}
    report.record("all_splits_present", set(SPLITS) <= present, present=sorted(present))
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="scan data/raw and write the split manifest")
    v = sub.add_parser("validate", help="enforce data contracts on the manifest")
    v.add_argument("--report", default="reports/data_contract.json")
    args = parser.parse_args(argv)

    params = load_params()
    if args.cmd == "prepare":
        build_manifest(params)
        return 0

    manifest = PROJECT_ROOT / params["data"]["manifest"]
    report = validate_manifest(manifest, params)
    out = PROJECT_ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": report.passed, "checks": report.checks}, indent=2))
    for name, res in report.checks.items():
        print(f"[validate] {'PASS' if res['passed'] else 'FAIL'}  {name}")
    if not report.passed:
        print(f"[validate] data contract violated - see {out}", file=sys.stderr)
        return 1
    print("[validate] all data contracts satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
