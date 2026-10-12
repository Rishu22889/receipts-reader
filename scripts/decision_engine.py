"""Run every claim through the frozen decision rules -> predictions.csv, reports/decisions.csv.

Usage: python scripts/decision_engine.py
Needs data/processed/receipt_fields.csv (scripts/normalize_fields.py) and the image-audit reports.
Rules, evidence and evaluation: notebooks/04_decision_engine.ipynb and docs/POLICY_RULES.md.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tarang.decision_batch import run  # noqa: E402


def main():
    pred, summary, version = run(ROOT)
    print(f"rules version {version}")
    print(f"predictions.csv: {len(pred)} claims, approve {(pred.decision == 'approve').sum()}, review {(pred.decision == 'review').sum()}")
    for r in summary.itertuples():
        print(f"  {r.metric:55s} {r.value}")


if __name__ == "__main__":
    main()
