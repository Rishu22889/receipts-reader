"""Score every reading method on the chronological holdout -> reports/ocr_benchmark*.csv, reports/holdout_adjudication.csv.

Usage: python scripts/run_benchmark.py
Needs data/interim/ocr (scripts/ocr_extract.py) and data/interim/vlm (Kaggle, see kaggle/README.md).
Method, rules and findings: notebooks/03_ocr_benchmark.ipynb and docs/OCR_BENCHMARK.md.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tarang.benchmark import METHODS, run  # noqa: E402


def main():
    results, err = run(ROOT)
    adj = results[results.reference == "adjudicated"].pivot(index="method", columns="field", values="accuracy").loc[METHODS]
    cols = ["vendor", "receipt_date", "receipt_number", "gstin", "total_amount", "complete_record"]
    print("Holdout accuracy (n=60, adjudicated labels):")
    print((adj[cols] * 100).round(1).to_string())
    flagged = err[err.kind == "wrong value"].flagged
    print(f"\ncombined: {len(err)} wrong fields of 300 ({(err.kind == 'missed (blank)').sum()} blank, "
          f"{len(flagged)} wrong values, {int(flagged.sum())} of them flagged by its own checks)")


if __name__ == "__main__":
    main()
