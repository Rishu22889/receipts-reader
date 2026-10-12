"""Clean the raw claims export: data/raw/ -> data/processed/.

Usage: python scripts/clean_data.py
See notebooks/01_data_cleaning.ipynb and docs/DATA_NOTES.md for what is cleaned and why.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tarang.cleaning import run  # noqa: E402


def main():
    train, test = run(ROOT / "data" / "raw", ROOT / "data" / "processed")
    print(f"Cleaned {len(train)} train + {len(test)} test claims -> data/processed/")
    print(f"  photo missing (train / test): {(~train.photo_available).sum()} / {(~test.photo_available).sum()}")
    print(f"  clerk GSTINs failing checksum: {(train.clerk_gstin_valid == False).sum()}")  # noqa: E712


if __name__ == "__main__":
    main()
