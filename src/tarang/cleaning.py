"""Cleaning for the Tarang Pharma expense-claims export.

Packaged from notebooks/01_data_cleaning.ipynb, which explains every decision made here.
Raw files are never modified; cleaned copies are written to data/processed/.
"""
from pathlib import Path
import re

import pandas as pd

# Every date format reps/clerks use. All are day-first (proved in the notebook, step 3).
DATE_FORMATS = {
    r"\d{2}-\d{2}-\d{4}": "%d-%m-%Y",
    r"\d{4}-\d{2}-\d{2}": "%Y-%m-%d",
    r"\d{2}/\d{2}/\d{2}": "%d/%m/%y",
}

GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_PATTERN = r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]"


def read_raw(path):
    """Read a CSV with every column as text, so blanks stay '' and bill numbers keep leading zeros."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def parse_date(value):
    """Parse a rep- or clerk-typed date into a Timestamp. Blank -> NaT. Unknown format -> ValueError."""
    value = value.strip()
    if not value:
        return pd.NaT
    for pattern, fmt in DATE_FORMATS.items():
        if re.fullmatch(pattern, value):
            return pd.to_datetime(value, format=fmt)
    raise ValueError(f"Unrecognised date format: {value!r}")


def parse_amount(value):
    """'Rs. 1,234.5' -> 1234.5. Blank -> None. Anything else non-numeric -> ValueError."""
    value = value.strip()
    if not value:
        return None
    cleaned = re.sub(r"^(Rs\.?|INR|₹)\s*", "", value, flags=re.IGNORECASE).replace(",", "")
    if not re.fullmatch(r"\d+(\.\d+)?", cleaned):
        raise ValueError(f"Unrecognised amount: {value!r}")
    return float(cleaned)


def gstin_is_valid(gstin):
    """True if the GSTIN has the right structure and its check character matches."""
    gstin = gstin.strip().upper()
    if not re.fullmatch(GSTIN_PATTERN, gstin):
        return False
    total = 0
    for i, ch in enumerate(gstin[:14]):
        product = GSTIN_CHARS.index(ch) * (2 if i % 2 else 1)
        total += product // 36 + product % 36
    return GSTIN_CHARS[(36 - total % 36) % 36] == gstin[14]


def image_index(images_dir):
    """Map lower-case file stem -> file name for every photo on disk."""
    names = sorted(p.name for p in Path(images_dir).glob("*.jpg"))
    index = {Path(name).stem.lower(): name for name in names}
    if len(index) != len(names):
        raise ValueError("two photos share a stem; image matching would be ambiguous")
    return index


def resolve_image(value, index):
    """Return the photo's file name on disk, or '' if the upload never arrived.

    Handles full phone paths (/storage/emulated/0/DCIM/...) and .jpeg vs .jpg by matching on the stem.
    """
    stem = Path(value.strip().replace("\\", "/")).stem.lower()
    return index.get(stem, "")


def _iso(series):
    return series.dt.strftime("%Y-%m-%d").fillna("")


def _money(series):
    return series.map(lambda v: "" if pd.isna(v) else f"{v:.2f}")


def clean_claims(df, index):
    """Clean one claims table (train or test). Returns a new DataFrame; the input is not modified."""
    with_clerk = "clerk_bill_date" in df.columns
    expense_date = df.expense_date.map(parse_date)
    amount = df.amount_claimed.map(parse_amount)
    image = df.image_file.map(lambda v: resolve_image(v, index))

    out = pd.DataFrame({
        "claim_id": df.claim_id,
        "emp_id": df.emp_id,
        "submitted_on": df.submitted_on,
        "expense_date": _iso(expense_date),
        "category": df.category,
        "vendor": df.vendor,
        "amount_claimed": _money(amount),
        "rep_comment": df.rep_comment,
        "image_file": image,
        "photo_available": image != "",
    })
    if with_clerk:
        out = out.assign(
            clerk_vendor=df.clerk_vendor,
            clerk_bill_date=_iso(df.clerk_bill_date.map(parse_date)),
            clerk_bill_no=df.clerk_bill_no,
            clerk_bill_amount=_money(df.clerk_bill_amount.map(parse_amount)),
            clerk_gstin=df.clerk_gstin,
            # '' when no GSTIN was typed; True/False otherwise
            clerk_gstin_valid=df.clerk_gstin.map(lambda g: gstin_is_valid(g) if g.strip() else ""),
            status=df.status,
            clerk_note=df.clerk_note,
        )

    raw = {"expense_date_raw": df.expense_date, "amount_claimed_raw": df.amount_claimed, "image_file_raw": df.image_file}
    if with_clerk:
        raw["clerk_bill_date_raw"] = df.clerk_bill_date
    return out.assign(**raw).reset_index(drop=True)


def run(raw_dir, out_dir):
    """Clean claims_train.csv and claims_test.csv from raw_dir into out_dir. Returns (train, test)."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = image_index(raw_dir / "images")

    train = clean_claims(read_raw(raw_dir / "claims_train.csv"), index)
    test = clean_claims(read_raw(raw_dir / "claims_test.csv"), index)

    train.to_csv(out_dir / "claims_train_clean.csv", index=False)
    test.to_csv(out_dir / "claims_test_clean.csv", index=False)
    return train, test
