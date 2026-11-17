"""DVC ``validate_text`` stage: data contract for the transcript dataset.

Fails the pipeline (non-zero exit) on any violation:

1. Schema          - every row has non-empty text and a known label.
2. Family leakage  - each scenario family lives in exactly one split, so the test
                     set measures generalisation to unseen phrasings.
3. Text leakage    - no identical transcript appears in two splits.
4. Coverage        - every split contains every class; each class >= min_class_ratio.

Report: reports/text_contract.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

from src.config import PROJECT_ROOT, load_params
from src.data_contracts import SPLITS, ContractReport
from src.transcripts import LABELS, read_csv


def validate_rows(rows: list[dict], min_ratio: float) -> ContractReport:
    report = ContractReport()
    bad = [i for i, r in enumerate(rows) if not r["text"].strip() or r["label"] not in LABELS]
    report.record("schema_text_and_label", not bad, n_rows=len(rows), bad_rows=bad[:20])

    fam_splits: dict[str, set[str]] = defaultdict(set)
    text_splits: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        fam_splits[r["family"]].add(r["split"])
        text_splits[r["text"]].add(r["split"])
    fam_leaks = {f: sorted(s) for f, s in fam_splits.items() if len(s) > 1}
    report.record("no_family_leakage", not fam_leaks, leaked_families=fam_leaks)
    txt_leaks = [t for t, s in text_splits.items() if len(s) > 1]
    report.record("no_duplicate_text_across_splits", not txt_leaks, examples=txt_leaks[:5])

    counts = Counter(r["label"] for r in rows)
    total = sum(counts.values()) or 1
    ratios = {c: counts.get(c, 0) / total for c in LABELS}
    per_split = {s: sorted({r["label"] for r in rows if r["split"] == s}) for s in SPLITS}
    ok = all(v >= min_ratio for v in ratios.values()) and all(
        set(v) == set(LABELS) for v in per_split.values()
    )
    report.record(
        "class_coverage_and_floor", ok, floor=min_ratio, overall=ratios, classes_per_split=per_split
    )
    return report


def main() -> int:
    params = load_params()["text"]
    rows = read_csv(PROJECT_ROOT / params["transcripts"])
    report = validate_rows(rows, float(params["min_class_ratio"]))
    out = PROJECT_ROOT / "reports" / "text_contract.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"passed": report.passed, "checks": report.checks}, indent=2))
    for name, res in report.checks.items():
        print(f"[validate_text] {'PASS' if res['passed'] else 'FAIL'}  {name}")
    if not report.passed:
        print(f"[validate_text] text data contract violated - see {out}", file=sys.stderr)
        return 1
    print("[validate_text] all text data contracts satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
