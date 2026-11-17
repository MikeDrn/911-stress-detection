"""Content branch: transcript dataset, explainable text model, fusion rule, text contracts."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src import text_model as tm
from src.fusion import fuse
from src.text_contracts import validate_rows
from src.transcripts import FAMILIES, LABELS, asr_noise, generate, render

FUSION = {
    "covert_base": 0.85,
    "covert_stress_weight": 0.25,
    "incongruence_weight": 0.4,
    "alert_threshold": 0.5,
    "review_threshold": 0.3,
}


@pytest.fixture(scope="module")
def rows():
    return [r.__dict__ for r in generate(per_family=25, seed=1)]


@pytest.fixture(scope="module")
def model(rows):
    train = [r for r in rows if r["split"] == "train"]
    return tm.fit(
        [r["text"] for r in train],
        [r["label"] for r in train],
        {"ngram_max": 2, "min_df": 2, "C": 2.0},
    )


# ----------------------------------------------------------------- dataset
def test_every_family_has_one_split_and_all_classes_in_every_split(rows):
    fam_split = {}
    for r in rows:
        fam_split.setdefault(r["family"], set()).add(r["split"])
    assert all(len(v) == 1 for v in fam_split.values())
    for split in ("train", "val", "test"):
        assert {r["label"] for r in rows if r["split"] == split} == set(LABELS)


def test_generation_is_deterministic():
    a = [r.text for r in generate(5, seed=3)]
    b = [r.text for r in generate(5, seed=3)]
    assert a == b and len(set(a)) == len(a)


def test_pronouns_match_the_relative():
    rng = random.Random(0)
    for _ in range(200):
        t = render("collapse", rng, noise=0)
        if "my father" in t or "my husband" in t:
            assert " she " not in f" {t} "
        if "my mother" in t or "my wife" in t:
            assert " he " not in f" {t} "


def test_asr_noise_lowercases_and_strips_punctuation():
    out = asr_noise("Hello, My Father COLLAPSED!", random.Random(0), strength=0)
    assert out == out.lower() and "," not in out and "!" not in out


def test_hard_negatives_exist():
    assert FAMILIES["choking_food"][0] == "genuine_emergency"
    assert FAMILIES["honest_misdial_food"][0] == "non_emergency"
    assert FAMILIES["food_order_other"][0] == "covert"


def test_text_contract_detects_family_and_text_leakage(rows):
    assert validate_rows(rows, 0.2).passed
    leaked = [dict(r) for r in rows]
    leaked[0]["split"] = "test" if leaked[0]["split"] != "test" else "train"
    report = validate_rows(leaked, 0.2)
    assert not report.checks["no_family_leakage"]["passed"]


# ----------------------------------------------------------------- model
def test_model_generalises_to_unseen_families_and_beats_keyword_rule(rows, model):
    test = [r for r in rows if r["split"] == "test"]
    ev = tm.evaluate(model, test)
    kw = tm.keyword_baseline(test)
    assert ev["accuracy"] > 0.85
    assert ev["recall"]["covert"] > kw["covert_recall"] + 0.25


def test_traps_are_handled(model):
    assert (
        model.predict("my son is choking on a piece of pizza he can't breathe")["label"]
        == "genuine_emergency"
    )
    assert (
        model.predict("sorry wrong number i was trying to call the pizza place bye")["label"]
        == "non_emergency"
    )
    assert model.predict("yes yes with pepperoni please hurry")["label"] == "covert"


def test_numpy_inference_matches_sklearn(rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    train = [r for r in rows if r["split"] == "train"]
    texts, labels = [r["text"] for r in train], [r["label"] for r in train]
    m = tm.fit(texts, labels, {"ngram_max": 2, "min_df": 2, "C": 2.0})
    vec = TfidfVectorizer(
        tokenizer=lambda t: tm.ngrams(tm.tokenize(t), 2),
        token_pattern=None,
        lowercase=False,
        sublinear_tf=True,
        min_df=2,
        norm="l2",
    )
    X = vec.fit_transform(texts)
    clf = LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced").fit(X, labels)
    sample = [r["text"] for r in rows if r["split"] == "test"][:30]
    expected = clf.predict_proba(vec.transform(sample))
    np.testing.assert_allclose(m.predict_many(sample), expected, atol=1e-6)


def test_json_roundtrip_and_evidence(tmp_path, model):
    path = model.save(tmp_path / "m.json")
    again = tm.TextModel.load(path)
    t = "hi i'd like to order a large pizza with pepperoni to 61 elm street"
    assert again.predict(t)["probs"] == pytest.approx(model.predict(t)["probs"])
    ev = again.predict(t)["evidence"]
    assert ev and all(isinstance(w, str) for w in ev)


def test_shuffled_labels_fall_to_chance(rows):
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    acc = tm.shuffled_label_control(train, test, {"ngram_max": 2, "min_df": 2, "C": 2.0})
    assert acc < 0.6


# ----------------------------------------------------------------- fusion
def probs(covert=0.0, genuine=0.0, non=0.0):
    return {
        "label": max(
            {"covert": covert, "genuine_emergency": genuine, "non_emergency": non}.items(),
            key=lambda kv: kv[1],
        )[0],
        "probs": {"covert": covert, "genuine_emergency": genuine, "non_emergency": non},
        "evidence": ["pepperoni"],
    }


def test_stressed_genuine_emergency_never_alerts():
    out = fuse(probs(genuine=0.95, covert=0.03, non=0.02), stress=1.0, cfg=FUSION)
    assert out["threat_level"] == "normal" and "expected" in out["reason"]


def test_covert_content_alerts_with_calm_voice_and_more_with_stress():
    calm = fuse(probs(covert=0.75, non=0.2, genuine=0.05), 0.1, FUSION)
    tense = fuse(probs(covert=0.75, non=0.2, genuine=0.05), 0.9, FUSION)
    assert calm["threat_level"] == "alert" and tense["threat_score"] > calm["threat_score"]


def test_voice_content_mismatch_raises_review():
    assert fuse(probs(non=0.95, covert=0.03, genuine=0.02), 1.0, FUSION)["threat_level"] == "review"
    assert fuse(probs(non=0.95, covert=0.03, genuine=0.02), 0.1, FUSION)["threat_level"] == "normal"


def test_no_transcript_is_never_an_alert():
    out = fuse(None, 1.0, FUSION)
    assert out["threat_level"] == "normal" and "No transcript" in out["reason"]
