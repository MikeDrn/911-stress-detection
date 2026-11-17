"""Explainable transcript classifier: TF-IDF (1-2 grams) + multinomial logistic regression.

Training uses scikit-learn; the fitted model is then exported to plain JSON
(vocabulary, idf weights, coefficients). Serving re-implements the forward pass in
NumPy, so:

* loading a model never unpickles code (safe to pull from a registry),
* the API needs no scikit-learn,
* every prediction can be *explained*: each word's contribution to the winning
  class is idf x tf x coefficient, and the top words are highlighted for the
  dispatcher ("why did it say covert?").

``python -m src.text_model`` is the DVC ``train_text`` stage.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

TOKEN_RE = re.compile(r"(?u)\b\w[\w']*\b")

# Words that carry no meaning on their own. They stay in the MODEL (word order and
# phrasing matter), but are not shown to the dispatcher as "evidence".
# "yes" / "no" are deliberately kept: answering yes/no in code is a covert-call cue.
FILLER = set(
    """
a an the and or but so to of in on at for from with by as is are was were be been am
i i'm i'd i've i'll me my mine we our you your he she it it's they them his her their this
that there here what who how do does did can could would will just okay ok hi hello hey
please uh um hmm like know get got go going one
""".split()
)


def informative(term: str) -> bool:
    words = term.split()
    return not all(w in FILLER for w in words)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n_max: int = 2) -> list[str]:
    out = list(tokens)
    for n in range(2, n_max + 1):
        out += [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return out


class TextModel:
    """NumPy inference for an exported TF-IDF + logistic-regression model."""

    def __init__(self, spec: dict[str, Any], version: str = "local"):
        self.spec = spec
        self.classes: list[str] = spec["classes"]
        self.vocab: dict[str, int] = spec["vocabulary"]
        self.idf = np.asarray(spec["idf"], dtype=np.float64)
        self.coef = np.asarray(spec["coef"], dtype=np.float64)
        self.intercept = np.asarray(spec["intercept"], dtype=np.float64)
        self.n_max = int(spec.get("ngram_max", 2))
        self.version = version
        self._terms = {i: t for t, i in self.vocab.items()}

    @classmethod
    def load(cls, path: str | Path, version: str = "local") -> TextModel:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), version)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.spec), encoding="utf-8")
        return path

    def _vector(self, text: str) -> dict[int, float]:
        counts = Counter(t for t in ngrams(tokenize(text), self.n_max) if t in self.vocab)
        vec = {
            self.vocab[t]: (1.0 + math.log(c)) * self.idf[self.vocab[t]] for t, c in counts.items()
        }
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {i: v / norm for i, v in vec.items()}

    def predict(self, text: str, top_k: int = 6) -> dict[str, Any]:
        vec = self._vector(text or "")
        logits = self.intercept.copy()
        for i, v in vec.items():
            logits += v * self.coef[:, i]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        best = int(np.argmax(p))
        contrib = sorted(
            ((v * self.coef[best, i], self._terms[i]) for i, v in vec.items()), reverse=True
        )
        return {
            "label": self.classes[best],
            "probs": {c: float(x) for c, x in zip(self.classes, p, strict=True)},
            "evidence": [t for w, t in contrib if w > 0 and informative(t)][:top_k],
            "known_terms": len(vec),
        }

    def predict_many(self, texts: list[str]) -> np.ndarray:
        return np.array([[self.predict(t)["probs"][c] for c in self.classes] for t in texts])


def fit(texts: list[str], labels: list[str], params: dict[str, Any]) -> TextModel:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    n_max = int(params.get("ngram_max", 2))
    vec = TfidfVectorizer(
        tokenizer=lambda t: ngrams(tokenize(t), n_max),
        token_pattern=None,
        lowercase=False,
        sublinear_tf=True,
        min_df=int(params.get("min_df", 2)),
        norm="l2",
    )
    X = vec.fit_transform(texts)
    clf = LogisticRegression(
        C=float(params.get("C", 2.0)), max_iter=4000, class_weight="balanced"
    ).fit(X, labels)
    spec = {
        "classes": [str(c) for c in clf.classes_],
        "vocabulary": {t: int(i) for t, i in vec.vocabulary_.items()},
        "idf": vec.idf_.tolist(),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
        "ngram_max": n_max,
    }
    return TextModel(spec)


def evaluate(model: TextModel, rows: list[dict[str, str]]) -> dict[str, Any]:
    preds = [model.predict(r["text"])["label"] for r in rows]
    truth = [r["label"] for r in rows]
    acc = float(np.mean([p == t for p, t in zip(preds, truth, strict=True)]))
    per_class = {}
    for c in model.classes:
        idx = [i for i, t in enumerate(truth) if t == c]
        per_class[c] = float(np.mean([preds[i] == c for i in idx])) if idx else None
    confusion = Counter(zip(truth, preds, strict=True))
    return {
        "accuracy": acc,
        "recall": per_class,
        "confusion": {f"{t}->{p}": n for (t, p), n in sorted(confusion.items())},
        "n": len(rows),
    }


KEYWORDS = ("pizza", "order", "deliver", "topping", "pepperoni")


def keyword_baseline(rows: list[dict[str, str]]) -> dict[str, float]:
    """Sanity check: a naive rule 'food/order words => covert'. The model must beat it."""
    preds = ["covert" if any(k in r["text"] for k in KEYWORDS) else "other" for r in rows]
    truth = ["covert" if r["label"] == "covert" else "other" for r in rows]
    tp = sum(p == t == "covert" for p, t in zip(preds, truth, strict=True))
    fp = sum(p == "covert" and t == "other" for p, t in zip(preds, truth, strict=True))
    pos = sum(t == "covert" for t in truth)
    return {
        "accuracy": float(np.mean([p == t for p, t in zip(preds, truth, strict=True)])),
        "covert_recall": tp / pos if pos else 0.0,
        "false_alarms": fp,
    }


def shuffled_label_control(train: list[dict], test: list[dict], params: dict, seed: int = 0):
    """Sanity check: same model on shuffled labels must fall to chance (~1/3)."""
    rng = np.random.default_rng(seed)
    labels = [r["label"] for r in train]
    shuffled = list(rng.permutation(labels))
    model = fit([r["text"] for r in train], shuffled, params)
    return evaluate(model, test)["accuracy"]


def main() -> int:
    from src.config import PROJECT_ROOT, load_params
    from src.transcripts import read_csv

    params = load_params()
    p = params["text"]
    rows = read_csv(PROJECT_ROOT / p["transcripts"])
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    model = fit([r["text"] for r in train], [r["label"] for r in train], p["model"])
    out = PROJECT_ROOT / "models" / "text_model.json"
    model.save(out)
    metrics = {
        "train": evaluate(model, train),
        "val": evaluate(model, val),
        "test_heldout_families": evaluate(model, test),
        "keyword_baseline_test": keyword_baseline(test),
        "shuffled_label_control_test_accuracy": shuffled_label_control(train, test, p["model"]),
        "vocabulary_size": len(model.vocab),
    }
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (PROJECT_ROOT / "reports" / "text_metrics.json").write_text(json.dumps(metrics, indent=2))
    t = metrics["test_heldout_families"]
    print(
        f"[text] vocab={len(model.vocab)}  train acc={metrics['train']['accuracy']:.3f}  "
        f"val acc={metrics['val']['accuracy']:.3f}  TEST (unseen families) acc="
        f"{t['accuracy']:.3f}"
    )
    print(f"[text] test recall per class: {t['recall']}")
    kb = metrics["keyword_baseline_test"]
    print(
        f"[text] sanity: keyword rule acc={kb['accuracy']:.3f} covert_recall="
        f"{kb['covert_recall']:.3f} | shuffled labels acc="
        f"{metrics['shuffled_label_control_test_accuracy']:.3f} (chance = 0.333)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
