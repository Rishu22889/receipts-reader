"""Turn every method's receipt text into normalised fields for every claim -> data/processed/receipt_fields.csv.

Usage: python scripts/normalize_fields.py
One row per (claim, method). Each field has its own status (ok / verified / ok_normalised / format_mismatch / uncertain /
invalid / absent ...); a missing field stays blank and an invalid GSTIN is never corrected into a valid-looking one.
Claims whose photo never arrived get status "no_photo". The vendor list here is learned from all transcribed train claims
(the benchmark, by contrast, used dev claims only).
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from tarang.fields import build_vendor_master, canonical_vendors, combine, parse_receipt  # noqa: E402
from tarang.ocr import load_outputs  # noqa: E402

STATUS_COLUMNS = ["vendor_status", "date_status", "number_status", "gstin_status", "total_status"]


def main():
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    train = read(ROOT / "data" / "processed" / "claims_train_clean.csv")
    test = read(ROOT / "data" / "processed" / "claims_test_clean.csv")
    claims = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)

    outputs = load_outputs(ROOT / "data" / "interim")
    labelled = train[train.clerk_bill_date != ""]
    canonical = canonical_vendors(labelled.clerk_vendor.tolist())
    master = build_vendor_master(labelled.assign(canonical_vendor=labelled.clerk_vendor.map(canonical),
                                                 text=labelled.image_file.map(outputs["paddleocr_vl"].text)))

    rows = []
    for _, c in claims.iterrows():
        if c.photo_available != "True":
            blank = {"vendor": "", "receipt_date": "", "receipt_number": "", "gstin": "", "gstin_read": "", "total_amount": "",
                     "looped": False, **{s: "no_photo" for s in STATUS_COLUMNS}}
            rows += [{"claim_id": c.claim_id, "split": c.split, "method": m, **blank} for m in [*outputs, "combined"]]
            continue
        parsed = {m: parse_receipt(o.loc[c.image_file].text, master, o.loc[c.image_file].get("new_tokens")) for m, o in outputs.items()}
        parsed["combined"] = combine(parsed["paddleocr_vl_retry"], parsed["tesseract_pre"])
        rows += [{"claim_id": c.claim_id, "split": c.split, "method": m, **p} for m, p in parsed.items()]

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "processed" / "receipt_fields.csv", index=False)
    best = out[out.method == "combined"]
    print(f"{len(claims)} claims x {out.method.nunique()} methods -> data/processed/receipt_fields.csv")
    print("combined, share of claims per status:")
    print(pd.concat({s: best[s].value_counts(normalize=True) for s in STATUS_COLUMNS}, axis=1).fillna(0).round(3).to_string())


if __name__ == "__main__":
    main()
