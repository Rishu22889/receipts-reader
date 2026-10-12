"""Audit every receipt photo: data/raw/images/ + data/processed/ -> reports/.

Usage: python scripts/audit_images.py      (needs the `tesseract` binary on PATH; ~3 minutes)
Writes reports/missing_images.csv, duplicate_images.csv, image_audit.csv and reports/contact_sheets/.
reports/visual_review.csv is written by notebooks/02_image_audit.ipynb (labels checked by eye) and is not touched here.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tarang.images import run  # noqa: E402


def main():
    if shutil.which("tesseract") is None:
        sys.exit("tesseract not found. Install it first (macOS: brew install tesseract; Ubuntu: apt install tesseract-ocr).")
    audit = run(ROOT / "data" / "raw" / "images", ROOT / "data" / "processed", ROOT / "reports")
    issues = audit.issues.str.split("; ").explode().loc[lambda s: s != ""]
    counts = issues.str.replace(r"^same photo as .*", "same photo as another claim", regex=True).value_counts()
    print(f"Audited {len(audit)} claims -> reports/")
    for issue, n in counts.items():
        print(f"  {issue:28s} {n}")


if __name__ == "__main__":
    main()
