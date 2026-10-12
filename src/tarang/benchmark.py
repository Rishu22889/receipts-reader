"""OCR benchmark: chronological holdout, adjudicated reference labels, per-field scoring.

Packaged from notebooks/03_ocr_benchmark.ipynb, which explains every choice and shows the results.
"""
from math import sqrt
from pathlib import Path

import pandas as pd

from tarang.fields import build_vendor_master, canonical_vendors, combine, norm, parse_receipt
from tarang.ocr import load_outputs

METHODS = ["tesseract_raw", "tesseract_pre", "paddleocr_vl", "paddleocr_vl_retry", "combined"]
HANDWRITTEN = {"Highway King Dhaba", "Hotel Rajdhani Dhaba", "Punjabi Tadka Dhaba", "Sai Ram Tea & Snacks",
               "Maa Durga Cab Service", "Shree Ganesh Tours & Travels", "Royal Travels"}   # from the contact sheets

# Holdout fields where a method and the clerk gave different values, checked on the photo by the author (notebook 03, section 6).
ADJUDICATION = [
    # claim, field, clerk typed, printed on the receipt, who was right, note
    ("CLM-30807", "receipt_number", "598", "5968", "model", "Bill No: 5968; clerk dropped a digit"),
    ("CLM-30814", "total_amount", "279.00", "270.00", "model", "handwritten 150 + 120 = 270"),
    ("CLM-30820", "total_amount", "259.00", "250.00", "model", "Amount (Rs) 250.00 and TOTAL 250.00"),
    ("CLM-30824", "gstin", "27DPAFU412UX1ZM", "27DPAFU4120X1ZM", "model", "clerk value fails checksum (notebook 01)"),
    ("CLM-30846", "receipt_number", "6185", "6185", "clerk", "VLM text missed the bill number"),
    ("CLM-30868", "total_amount", "1130.00", "1130.00", "clerk", "handwritten 1130/-, model read 11307"),
    ("CLM-30883", "receipt_date", "2026-07-03", "2026-07-02", "model", "Date: 2026-07-02 20:20"),
    ("CLM-30919", "total_amount", "800.00", "300.00", "model", "TOTAL's first digit overwritten in pen (300 -> 800); "
                                                              "Amount (Rs) 300.00 and 104.64 x 2.87 = 300; receipt also carries the 'approve it' stamp"),
    ("CLM-30920", "receipt_number", "00638", "001638", "model", "Receipt No: 001638; clerk dropped a digit"),
    ("CLM-30923", "receipt_number", "007012", "007012", "clerk", "model read 067012 (flagged format_mismatch)"),
    ("CLM-30925", "receipt_number", "008732", "008732", "clerk", "blurred photo; model read 668732 (flagged format_mismatch)"),
    ("CLM-30940", "gstin", "", "27GFZCR8659B2Z4", "model", "GSTIN printed in header; clerk left it blank"),
    ("CLM-30950", "receipt_date", "2026-05-07", "2026-07-05", "model", "Date: 05-07-2026; clerk swapped day and month"),
    ("CLM-30978", "receipt_number", "006239", "006239", "clerk", "Tesseract read 906239 (flagged format_mismatch)"),
    ("CLM-30982", "gstin", "", "27OILFO1994Z2ZV", "model", "GSTIN printed; clerk left it blank"),
    ("CLM-30983", "gstin", "", "27YUFFZ9311K2ZV", "model", "GSTIN printed; clerk left it blank"),
    ("CLM-31009", "total_amount", "1536.01", "536.01", "model", "blue pen '1' added in front of TOTAL 536.01; Amount (Rs) 536.01 = "
                                                              "104.69 x 5.12; clerk rejected it as 'bill looks overwritten'"),
    ("CLM-31015", "receipt_date", "2026-07-09", "2026-07-09", "clerk", "Invoice Date 09 Jul 2026; model took check-in 07 Jul"),
    ("CLM-31034", "gstin", "27ODUFJ1789K3ZU", "27ODUFJ1789K3ZU", "clerk", "agrees with combined method"),
]
ADJUDICATION_COLUMNS = ["claim_id", "field", "clerk_value", "printed_value", "correct", "note"]
FIELDS_SCORED = ["vendor", "receipt_date", "receipt_number", "gstin", "gstin_absent", "total_amount", "total_within_10", "complete_record"]


def select_holdout(train, audit, canonical, dup, n=60, pool_frac=0.4, seed=42):
    """Chronological holdout: from the latest pool_frac of transcribed train claims, keep every hard case, fill to n by category.

    Same-photo duplicate groups never straddle dev/holdout: a group touching dev is kept entirely in dev.
    Returns (holdout, dev, pool).
    """
    t = train[train.clerk_bill_date != ""].sort_values(["submitted_on", "claim_id"]).reset_index(drop=True)
    cut = int(len(t) * (1 - pool_frac))
    pool = t.iloc[cut:].copy()
    dev_ids = set(t.claim_id.iloc[:cut])
    for _, g in dup.groupby("duplicate_group"):
        if set(g.claim_id) & dev_ids:
            pool = pool[~pool.claim_id.isin(g.claim_id)]
    issues = pool.claim_id.map(audit.set_index("claim_id").issues)
    pool["hard"] = issues.str.contains("rotated|low OCR|instruction")
    pool["handwritten"] = pool.clerk_vendor.map(canonical).isin(HANDWRITTEN)
    hard = pool[pool.hard]
    rest = pool[~pool.hard]
    k = n - len(hard)
    share = (rest.category.value_counts(normalize=True) * k).round().astype(int)
    while share.sum() > k: share[share.idxmax()] -= 1
    while share.sum() < k: share[share.idxmin()] += 1
    fill = pd.concat([rest[rest.category == c].sample(m, random_state=seed) for c, m in share.items() if m > 0])
    holdout = pd.concat([hard, fill]).sort_values(["submitted_on", "claim_id"])
    return holdout, t[~t.claim_id.isin(holdout.claim_id) & ~t.claim_id.isin(pool.claim_id)], pool


def wilson(k, n, z=1.96):
    """Wilson 95% interval for k correct out of n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def references(holdout, canonical, adj, kind):
    """Per-claim reference values: the clerk's, or the clerk's with adjudicated corrections."""
    ref = holdout.set_index("claim_id")
    ref = pd.DataFrame({
        "image_file": ref.image_file, "vendor": ref.clerk_vendor.map(canonical), "receipt_date": ref.clerk_bill_date,
        "receipt_number": ref.clerk_bill_no, "gstin": ref.clerk_gstin, "gstin_valid": ref.clerk_gstin_valid,
        "total_amount": ref.clerk_bill_amount})
    if kind == "adjudicated":
        for r in adj.itertuples():
            ref.loc[r.claim_id, r.field] = r.printed_value
            if r.field == "gstin":
                ref.loc[r.claim_id, "gstin_valid"] = "True" if r.printed_value else ""
    return ref


def score(parsed, ref):
    """One row per claim: did the method get each field right? GSTIN refs failing the checksum are excluded (None)."""
    rows = []
    for cid, r in ref.iterrows():
        p = parsed[r.image_file]
        row = {"claim_id": cid}
        row["vendor"] = norm(p["vendor"]) == norm(r.vendor)
        row["receipt_date"] = p["receipt_date"] == r.receipt_date
        row["receipt_number"] = p["receipt_number"].strip().upper() == r.receipt_number.strip().upper()
        if r.gstin and r.gstin_valid == "False":
            row["gstin"] = None                                   # unreliable reference: excluded, counted
        elif r.gstin:
            row["gstin"] = p["gstin"] == r.gstin
        else:
            row["gstin_absent"] = p["gstin"] == ""
        ref_amt = float(r.total_amount) if r.total_amount else None
        got = float(p["total_amount"]) if p["total_amount"] else None
        row["total_amount"] = got is not None and ref_amt is not None and abs(got - ref_amt) < 0.005
        row["total_within_10"] = got is not None and ref_amt is not None and abs(got - ref_amt) <= 10
        for f in ["vendor", "receipt_date", "receipt_number", "total_amount"]:
            row[f + "_answered"] = bool(p[f])
        row["gstin_answered"] = bool(p["gstin"])
        row["complete_record"] = all([row["vendor"], row["receipt_date"], row["receipt_number"], row["total_amount"],
                                      row.get("gstin", row.get("gstin_absent")) is not False])
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(method, kind, scored):
    s = scored
    out = []
    for f in FIELDS_SCORED:
        col = s[f].dropna() if f in s else pd.Series(dtype=bool)
        k, n = int(col.sum()), len(col)
        lo, hi = wilson(k, n)
        answered = s.get(f + "_answered")
        cov = answered.mean() if answered is not None else float("nan")
        prec = s.loc[answered & s[f].notna(), f].astype(bool).mean() if answered is not None and answered.any() else float("nan")
        out.append({"method": method, "reference": kind, "field": f, "n": n, "correct": k, "accuracy": k / n if n else float("nan"),
                    "ci95_low": lo, "ci95_high": hi, "coverage": cov, "precision_when_answered": prec,
                    "excluded_refs": int(s["gstin"].isna().sum() - s["gstin_absent"].notna().sum()) if f == "gstin" else 0})
    return out


def errors(parsed, ref, audit):
    """Every wrong field of one method, with whether its own checks flagged it."""
    rows = []
    issues = audit.set_index("claim_id").issues
    for cid, r in ref.iterrows():
        p = parsed[r.image_file]
        for f, st in [("vendor", "vendor_status"), ("receipt_date", "date_status"), ("receipt_number", "number_status"),
                      ("gstin", "gstin_status"), ("total_amount", "total_status")]:
            want, got = r[f], p[f]
            ok = norm(got) == norm(want) if f == "vendor" else (got.strip().upper() == want.strip().upper())
            if f == "total_amount" and got and want:
                ok = abs(float(got) - float(want)) < 0.005
            if not ok:
                rows.append({"claim_id": cid, "field": f, "reference": want, "predicted": got, "status": p[st],
                             "kind": "missed (blank)" if not got else "wrong value",
                             "flagged": p[st] not in ("ok", "verified", "ok_normalised") or p["looped"],
                             "issues": issues.get(cid, "")})
    return pd.DataFrame(rows)


def parse_all(outputs, master, files):
    """Fields from every method's text for the given photos, plus the combined method."""
    parsed = {m: {f: parse_receipt(o.loc[f].text, master, o.loc[f].get("new_tokens")) for f in files} for m, o in outputs.items()}
    parsed["combined"] = {f: combine(parsed["paddleocr_vl_retry"][f], parsed["tesseract_pre"][f]) for f in files}
    return parsed


def setup(root):
    """Load everything the benchmark needs. Returns a dict (train, audit, canonical, holdout, dev, pool, outputs, master)."""
    root = Path(root)
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    train = read(root / "data" / "processed" / "claims_train_clean.csv")
    audit = read(root / "reports" / "image_audit.csv")
    dup = read(root / "reports" / "duplicate_images.csv")
    canonical = canonical_vendors(train[train.clerk_vendor != ""].clerk_vendor.tolist())
    holdout, dev, pool = select_holdout(train, audit, canonical, dup)
    outputs = load_outputs(root / "data" / "interim")
    dev_text = dev.assign(canonical_vendor=dev.clerk_vendor.map(canonical), text=dev.image_file.map(outputs["paddleocr_vl"].text))
    master = build_vendor_master(dev_text)
    return {"train": train, "audit": audit, "canonical": canonical, "holdout": holdout, "dev": dev, "pool": pool,
            "outputs": outputs, "master": master}


def run(root):
    """Write reports/ocr_benchmark.csv, ocr_benchmark_rows.csv, ocr_benchmark_errors.csv, holdout_adjudication.csv."""
    root = Path(root)
    ctx = setup(root)
    holdout, canonical = ctx["holdout"], ctx["canonical"]
    adj = pd.DataFrame(ADJUDICATION, columns=ADJUDICATION_COLUMNS)
    parsed = parse_all(ctx["outputs"], ctx["master"], holdout.image_file.tolist())

    results = pd.DataFrame([r for kind in ["clerk", "adjudicated"] for m in METHODS
                            for r in summarise(m, kind, score(parsed[m], references(holdout, canonical, adj, kind)))])
    ref = references(holdout, canonical, adj, "adjudicated")
    per_claim = [{"method": m, "claim_id": cid, **parsed[m][r.image_file]} for m in METHODS for cid, r in ref.iterrows()]
    err = errors(parsed["combined"], ref, ctx["audit"])

    reports = root / "reports"
    adj.to_csv(reports / "holdout_adjudication.csv", index=False)
    results.to_csv(reports / "ocr_benchmark.csv", index=False)
    pd.DataFrame(per_claim).to_csv(reports / "ocr_benchmark_rows.csv", index=False)
    err.to_csv(reports / "ocr_benchmark_errors.csv", index=False)
    return results, err
