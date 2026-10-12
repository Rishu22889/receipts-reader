"""Reading receipt photos: the two Tesseract baselines (run locally) and loading PaddleOCR-VL output (run on Kaggle).

Packaged from notebooks/03_ocr_benchmark.ipynb. Every reader returns one record per photo:
{file, backend, text, rotation_applied, seconds, error}, stored as JSON lines.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile
import time

import pandas as pd
from PIL import Image, ImageOps

from tarang.images import preprocess

VLM_TOKEN_CAP = 768


def tesseract_raw(path):
    """Baseline 1: Tesseract with default settings on the photo exactly as uploaded."""
    t0 = time.perf_counter()
    text = subprocess.run(["tesseract", str(path), "-"], capture_output=True, text=True).stdout
    return {"file": Path(path).name, "backend": "tesseract-raw", "text": text, "rotation_applied": 0,
            "seconds": round(time.perf_counter() - t0, 3), "error": ""}


def tesseract_pre(path, angle=0):
    """Baseline 2: rotate upright, 3x upscale, remove shading, threshold, then Tesseract as one text block."""
    t0 = time.perf_counter()
    clean = preprocess(ImageOps.grayscale(Image.open(path))).rotate(angle, expand=True, fillcolor=255)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        clean.save(f.name, dpi=(300, 300))
        text = subprocess.run(["tesseract", f.name, "-", "--psm", "6"], capture_output=True, text=True).stdout
    return {"file": Path(path).name, "backend": "tesseract-pre", "text": text, "rotation_applied": angle,
            "seconds": round(time.perf_counter() - t0, 3), "error": ""}


def write_jsonl(rows, path):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def load_jsonl(path):
    return pd.read_json(path, lines=True, dtype=False).set_index("file")


def run_tesseract(images_dir, out_dir, rotation, workers=6):
    """Run both baselines on every photo -> out_dir/tesseract_{raw,pre}.jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    photos = sorted(Path(images_dir).glob("*.jpg"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        write_jsonl(list(pool.map(tesseract_raw, photos)), out_dir / "tesseract_raw.jsonl")
        write_jsonl(list(pool.map(lambda p: tesseract_pre(p, rotation.get(p.name, 0)), photos)), out_dir / "tesseract_pre.jsonl")


def merge_retry(first, retry):
    """First VLM run with the looped photos replaced by their retry (an automatic rule: re-read when the token cap was hit)."""
    merged = first.copy()
    merged.loc[retry.index, ["text", "new_tokens", "backend"]] = retry[["text", "new_tokens", "backend"]].values
    return merged


def load_outputs(interim):
    """All reading methods' text, keyed by method name."""
    interim = Path(interim)
    outputs = {
        "tesseract_raw": load_jsonl(interim / "ocr" / "tesseract_raw.jsonl"),
        "tesseract_pre": load_jsonl(interim / "ocr" / "tesseract_pre.jsonl"),
        "paddleocr_vl": load_jsonl(interim / "vlm" / "paddleocr_vl_outputs.jsonl"),
    }
    outputs["paddleocr_vl_retry"] = merge_retry(outputs["paddleocr_vl"], load_jsonl(interim / "vlm" / "paddleocr_vl_retry.jsonl"))
    return outputs
