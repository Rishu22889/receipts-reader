"""Check and merge the PaddleOCR-VL output produced on Kaggle -> data/interim/vlm/paddleocr_vl_merged.jsonl.

Usage: python scripts/vlm_extract.py
The model runs on a Kaggle T4 (kaggle/paddleocr_vl.ipynb, then kaggle/paddleocr_vl_retry.ipynb for looped photos); this laptop
(8 GB RAM) cannot hold it. See kaggle/README.md. This script verifies every photo was read, merges the retry for looped photos,
and reports what is left to fix.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tarang.ocr import VLM_TOKEN_CAP, load_jsonl, merge_retry  # noqa: E402

VLM = ROOT / "data" / "interim" / "vlm"


def main():
    first_path, retry_path = VLM / "paddleocr_vl_outputs.jsonl", VLM / "paddleocr_vl_retry.jsonl"
    if not first_path.exists():
        sys.exit(f"{first_path.relative_to(ROOT)} missing: run kaggle/paddleocr_vl.ipynb on Kaggle and download its output (kaggle/README.md).")
    first = load_jsonl(first_path)
    photos = {p.name for p in (ROOT / "data" / "raw" / "images").glob("*.jpg")}
    missing, errors = photos - set(first.index), first[first.error != ""]
    looped = first[first.new_tokens >= VLM_TOKEN_CAP]
    print(f"first run: {len(first)} photos, {len(errors)} errors, {len(missing)} not read, {len(looped)} hit the {VLM_TOKEN_CAP}-token cap")

    merged = first
    if retry_path.exists():
        retry = load_jsonl(retry_path)
        merged = merge_retry(first, retry)
        print(f"retry: {len(retry)} photos re-read, {int((retry.new_tokens >= VLM_TOKEN_CAP).sum())} still hit the cap")
    elif len(looped):
        print("retry missing: run kaggle/paddleocr_vl_retry.ipynb for the looped photos")

    out = VLM / "paddleocr_vl_merged.jsonl"
    merged.reset_index().to_json(out, orient="records", lines=True, force_ascii=False)
    still = int((merged.new_tokens >= VLM_TOKEN_CAP).sum())
    print(f"-> {out.relative_to(ROOT)}: {len(merged)} photos, {still} still looped")


if __name__ == "__main__":
    main()
