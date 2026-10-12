"""Run the two Tesseract baselines on every receipt photo -> data/interim/ocr/tesseract_{raw,pre}.jsonl.

Usage: python scripts/ocr_extract.py          (needs the `tesseract` binary; ~3 minutes)
Raw = Tesseract defaults on the photo as uploaded. Pre = rotated upright (image audit), upscaled, shading removed, thresholded.
"""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tarang.ocr import load_jsonl, run_tesseract  # noqa: E402


def main():
    if shutil.which("tesseract") is None:
        sys.exit("tesseract not found. Install it first (macOS: brew install tesseract; Ubuntu: apt install tesseract-ocr).")
    audit = pd.read_csv(ROOT / "reports" / "image_audit.csv", dtype=str, keep_default_na=False)
    rotation = {r.image_file: int(float(r.rotation)) for r in audit.itertuples() if r.rotation not in ("", "nan")}
    out = ROOT / "data" / "interim" / "ocr"
    run_tesseract(ROOT / "data" / "raw" / "images", out, rotation)
    for name in ["tesseract_raw", "tesseract_pre"]:
        d = load_jsonl(out / f"{name}.jsonl")
        print(f"{name}: {len(d)} photos, median {d.seconds.median():.2f}s/photo -> data/interim/ocr/{name}.jsonl")


if __name__ == "__main__":
    main()
