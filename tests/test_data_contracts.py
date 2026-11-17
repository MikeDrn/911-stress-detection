"""Data-contract tests: format checks, SHA-256 leakage detection, distribution floors."""

from __future__ import annotations

import copy
import csv
import hashlib

import pytest

from src.config import load_params
from src.data_contracts import (
    MANIFEST_FIELDS,
    assign_splits,
    check_label_conflicts,
    check_leakage,
    class_ratios,
    sha256_file,
    validate_manifest,
)


@pytest.fixture
def params():
    return copy.deepcopy(load_params())


def write_manifest(tmp_path, rows):
    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def make_dataset(tmp_path, write_wav, per_split=None):
    """Two classes x three splits, each clip a distinct tone (distinct SHA-256)."""
    per_split = per_split or {"train": 3, "val": 1, "test": 1}
    rows, k = [], 0
    for label_idx, label in enumerate(["calm", "stressed"]):
        for split, n in per_split.items():
            for _ in range(n):
                k += 1
                rel = f"{label}/{label}_{k}.wav"
                p = write_wav(rel, seconds=2.5, freq=100 + 13 * k)
                rows.append(
                    {
                        "path": rel,
                        "label": label,
                        "label_idx": label_idx,
                        "split": split,
                        "sha256": sha256_file(p),
                    }
                )
    return rows


def test_sha256_matches_hashlib(write_wav):
    p = write_wav("a.wav")
    assert sha256_file(p) == hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_clean_dataset_passes_all_contracts(tmp_path, write_wav, params):
    rows = make_dataset(tmp_path, write_wav)
    report = validate_manifest(write_manifest(tmp_path, rows), params, root=tmp_path)
    failed = [k for k, v in report.checks.items() if not v["passed"]]
    assert report.passed, failed


def test_leakage_duplicate_audio_in_train_and_test_is_detected(tmp_path, write_wav, params):
    rows = make_dataset(tmp_path, write_wav)
    # Copy a training clip byte-for-byte into the test split under a new name.
    src = next(r for r in rows if r["split"] == "train")
    dup = tmp_path / "stressed" / "copied_clip.wav"
    dup.write_bytes((tmp_path / src["path"]).read_bytes())
    rows.append({**src, "path": "stressed/copied_clip.wav", "split": "test"})

    leaks = check_leakage(rows)
    assert src["sha256"] in leaks and leaks[src["sha256"]] == ["test", "train"]
    report = validate_manifest(write_manifest(tmp_path, rows), params, root=tmp_path)
    assert not report.passed
    assert not report.checks["no_train_test_leakage_sha256"]["passed"]


def test_same_audio_with_two_labels_is_a_conflict():
    rows = [{"sha256": "abc", "label": "calm"}, {"sha256": "abc", "label": "stressed"}]
    assert check_label_conflicts(rows) == {"abc": ["calm", "stressed"]}


def test_format_violation_44k_stereo_is_detected(tmp_path, write_wav, params):
    rows = make_dataset(tmp_path, write_wav)
    p = write_wav("calm/bad_44k.wav", sr=44100, channels=2, seconds=3)
    rows.append(
        {
            "path": "calm/bad_44k.wav",
            "label": "calm",
            "label_idx": 0,
            "split": "train",
            "sha256": sha256_file(p),
        }
    )
    report = validate_manifest(write_manifest(tmp_path, rows), params, root=tmp_path)
    check = report.checks["audio_format_16k_mono_wav"]
    assert not check["passed"] and check["n_violations"] == 1


def test_tampered_file_fails_integrity(tmp_path, write_wav, params):
    rows = make_dataset(tmp_path, write_wav)
    (tmp_path / rows[0]["path"]).write_bytes(
        (tmp_path / rows[1]["path"]).read_bytes()
    )  # content changed after hashing
    report = validate_manifest(write_manifest(tmp_path, rows), params, root=tmp_path)
    assert not report.checks["sha256_integrity"]["passed"]


def test_class_imbalance_violates_distribution_floor(tmp_path, write_wav, params):
    rows = make_dataset(tmp_path, write_wav)  # train: 3 calm / 3 stressed
    stressed_train = [r for r in rows if r["label"] == "stressed" and r["split"] == "train"]
    rows = [r for r in rows if r not in stressed_train[1:]]  # train now 3 calm / 1 stressed
    report = validate_manifest(write_manifest(tmp_path, rows), params, root=tmp_path)
    assert report.checks["class_distribution_floor"]["per_split"]["train"]["stressed"] == 0.25
    assert not report.checks["class_distribution_floor"]["passed"]


def test_class_ratios():
    rows = [{"label": "calm"}] * 3 + [{"label": "stressed"}]
    assert class_ratios(rows, ["calm", "stressed"]) == {"calm": 0.75, "stressed": 0.25}


def test_assign_splits_keeps_duplicates_together_and_is_deterministic():
    rows = [{"sha256": f"h{i}", "label": "calm" if i % 2 else "stressed"} for i in range(40)]
    rows += [{"sha256": "h3", "label": "calm"}, {"sha256": "h3", "label": "calm"}]
    a = copy.deepcopy(rows)
    b = copy.deepcopy(rows)
    assign_splits(a, 0.2, 0.1, seed=42)
    assign_splits(b, 0.2, 0.1, seed=42)
    assert [r["split"] for r in a] == [r["split"] for r in b]
    assert len({r["split"] for r in a if r["sha256"] == "h3"}) == 1
    assert check_leakage(a) == {}
    assert {r["split"] for r in a} == {"train", "val", "test"}


# ------------------------------------------------------------ speaker leakage
def test_speaker_of_parses_cremad_and_phone_copies():
    from src.data_contracts import speaker_of

    assert speaker_of("data/raw/calm/cremad_1001_DFA_NEU_XX.wav") == "cremad1001"
    assert speaker_of("data/raw/calm/phone_cremad_1001_DFA_NEU_XX.wav") == "cremad1001"
    assert speaker_of("data/raw/calm/calm_0001.wav") == ""


def test_split_is_by_speaker_and_leakage_is_detected():
    from src.data_contracts import check_speaker_leakage

    rows = []
    for spk in range(30):
        for i, label in enumerate(["calm", "stressed", "calm", "stressed"]):
            rows.append({"sha256": f"h{spk}-{i}", "label": label, "speaker": f"cremad{1000 + spk}"})
    assign_splits(rows, 0.2, 0.1, seed=1)
    assert check_speaker_leakage(rows) == {}
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    rows[0]["split"] = "test" if rows[0]["split"] != "test" else "train"
    assert rows[0]["speaker"] in check_speaker_leakage(rows)
