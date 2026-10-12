"""Batch run of the decision engine: every claim, both evaluations, predictions.csv.

Packaged from notebooks/04_decision_engine.ipynb (sections 4-12), which explains each step.
"""
import hashlib
import inspect
from pathlib import Path
import time

import pandas as pd

from tarang import decision_engine as de
from tarang.fields import canonical_vendors, norm
from tarang.ocr import load_outputs

# Evidence checked on the photo (notebooks 03-04), used only to build the policy reference, never by the engine.
ALTERED_ON_PHOTO = {"CLM-30919", "CLM-31009", "CLM-30568", "CLM-30764"}
COLLEAGUE_VEHICLE_ON_PHOTO = {"CLM-30954", "CLM-30978", "CLM-31023"}
PHOTO_CHECKED = [
    ("CLM-30421", "clerk_bill_amount", "250.00", "TOTAL 250.00 (circled); clerk typed 240"),
    ("CLM-30543", "clerk_bill_date", "2026-06-09", "ticket 09/06/26; clerk typed 2026-09-06"),
    ("CLM-30436", "clerk_bill_no", "A/6178", "Bill No: A/6178; clerk typed A/618"),
    ("CLM-30519", "clerk_bill_date", "2026-06-14", "Date: 14.06.2026; clerk typed the 13th"),
    ("CLM-30600", "clerk_bill_date", "2026-06-19", "Date: 19/06/2026; clerk typed the 18th"),
    ("CLM-30717", "clerk_bill_no", "004023", "Receipt No: 004023; clerk typed 00402"),
    ("CLM-30568", "clerk_bill_amount", "514.00", "printed 514.00, blue pen '1' added in front; clerk typed 1514"),
    ("CLM-30764", "clerk_bill_amount", "268.00", "printed 268.00, blue pen '1' added in front; clerk typed 1268"),
]
ADJ_COLUMNS = {"total_amount": "clerk_bill_amount", "receipt_number": "clerk_bill_no", "receipt_date": "clerk_bill_date", "gstin": "clerk_gstin"}


def rules_version():
    """Fingerprint of the frozen rules, printed into every report."""
    source = "\n".join(inspect.getsource(f) for f in [de.parse_details, de.validate_fields, de.history_checks, de.decide_claim, de.rounding_ok])
    return hashlib.sha256((source + repr(de.REASONS) + repr(de.LODGING_LIMIT) + str(de.APPROVAL_LIMIT)).encode()).hexdigest()[:12]


def load(root):
    root = Path(root)
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    train = read(root / "data" / "processed" / "claims_train_clean.csv").assign(split="train")
    test = read(root / "data" / "processed" / "claims_test_clean.csv").assign(split="test")
    fields_all = read(root / "data" / "processed" / "receipt_fields.csv")
    return {
        "train": train, "test": test, "claims": pd.concat([train, test], ignore_index=True).fillna(""),
        "employees": read(root / "data" / "raw" / "employees.csv").set_index("emp_id"),
        "fields": fields_all[fields_all.method == "combined"].set_index("claim_id"),
        "audit": read(root / "reports" / "image_audit.csv").set_index("claim_id"),
        "dup_photos": read(root / "reports" / "duplicate_images.csv").set_index("claim_id"),
        "visual": read(root / "reports" / "visual_review.csv"),
        "adj": read(root / "reports" / "holdout_adjudication.csv"),
        "sample": read(root / "data" / "raw" / "sample_submission.csv"),
        "vlm": load_outputs(root / "data" / "interim")["paddleocr_vl_retry"],
    }


def build_history(claims, fields):
    return pd.DataFrame({
        "claim_id": claims.claim_id, "split": claims.split, "status": claims.status, "photo_group": claims.photo_group,
        "vendor_key": claims.claim_id.map(fields.vendor).map(norm),
        "bill_key": [de.bill_key(fields.loc[c, "vendor"], fields.loc[c, "receipt_number"]) for c in claims.claim_id],
        "receipt_date": claims.claim_id.map(fields.receipt_date), "total_amount": claims.claim_id.map(fields.total_amount),
        "order": list(zip(claims.submitted_on, claims.claim_id)),
    })


def run_all(ctx):
    """Every claim through the frozen rules. Train: history = earlier claims. Test: history = train + the whole batch."""
    claims, fields, employees, vlm = ctx["claims"], ctx["fields"], ctx["employees"], ctx["vlm"]
    claims["photo_group"] = claims.claim_id.map(ctx["dup_photos"].duplicate_group).fillna("")
    claims["instruction_stamp"] = claims.claim_id.map(ctx["audit"].instruction_text).fillna("")
    history = build_history(claims, fields)
    rows = []
    for _, c in claims.iterrows():
        t0 = time.perf_counter()
        f = fields.loc[c.claim_id].to_dict()
        details = de.parse_details(vlm.loc[c.image_file].text) if c.image_file in vlm.index else {}
        employee = employees.loc[c.emp_id].to_dict()
        validation = de.validate_fields(f, c.to_dict(), employee, details, employees)
        hist = history[(history.split == "train") & (history.order < (c.submitted_on, c.claim_id))] if c.split == "train" else history
        d = de.decide_claim(f, validation, c.to_dict(), employee, hist)
        rows.append({"claim_id": c.claim_id, "split": c.split, "decision": d.decision, "reason_codes": "; ".join(d.reason_codes),
                     "notes": "; ".join(d.notes), "reason": d.reason, "decision_ms": (time.perf_counter() - t0) * 1000})
    return pd.DataFrame(rows).set_index("claim_id")


def policy_reference(ctx):
    """Reference outcome per train claim from evidence the engine does not produce (clerk fields + photo checks)."""
    train, dup_photos = ctx["train"], ctx["dup_photos"]
    canon = canonical_vendors(train[train.clerk_vendor != ""].clerk_vendor.tolist())
    unreadable = set(ctx["visual"][ctx["visual"].label == "unreadable"].claim_id)
    stamped = set(ctx["visual"][ctx["visual"].label == "instruction_text"].claim_id)
    ref = train.set_index("claim_id").copy()
    for cid, col, value, _ in PHOTO_CHECKED:
        ref.loc[cid, col] = value
    for r in ctx["adj"].itertuples():
        col = ADJ_COLUMNS.get(r.field)
        if col and r.claim_id in ref.index:
            ref.loc[r.claim_id, col] = r.printed_value
    ref["ref_key"] = [f"{norm(canon.get(v, v))}|{n.upper()}" if v and n else "" for v, n in zip(ref.clerk_vendor, ref.clerk_bill_no)]
    ref = ref.sort_values(["submitted_on"]).assign(order=lambda d: list(zip(d.submitted_on, d.index)))

    def reasons(cid, r):
        out = []
        if r.photo_available != "True":
            return ["MISSING_IMAGE"]
        if cid in unreadable or not r.clerk_bill_amount:
            return ["UNREADABLE_IMAGE"]
        claimed, bill = float(r.amount_claimed), float(r.clerk_bill_amount)
        if max(claimed, bill) >= de.APPROVAL_LIMIT: out.append("AMOUNT_OVER_LIMIT")
        if not de.rounding_ok(claimed, bill): out.append("AMOUNT_MISMATCH")
        earlier = ref[ref.order < r.order]
        if r.ref_key and (earlier.ref_key == r.ref_key).any(): out.append("DUPLICATE_BILL")
        if r.name in dup_photos.index:
            twins = dup_photos[(dup_photos.duplicate_group == dup_photos.loc[r.name, "duplicate_group"])].index
            if any(t in earlier.index for t in twins): out.append("DUPLICATE_PHOTO")
        if cid in ALTERED_ON_PHOTO: out.append("ALTERED_BILL")
        if cid in COLLEAGUE_VEHICLE_ON_PHOTO: out.append("FUEL_VEHICLE_OTHER_EMPLOYEE")
        if cid in stamped: out.append("INSTRUCTION_TEXT")
        if r.clerk_bill_date and abs((pd.Timestamp(r.expense_date) - pd.Timestamp(r.clerk_bill_date)).days) > 1 and r.category != "Lodging":
            out.append("DATE_MISMATCH")
        return out

    ref["ref_reasons"] = [reasons(cid, r) for cid, r in ref.iterrows()]
    ref["ref_decision"] = ref.ref_reasons.map(lambda x: "review" if x else "approve")
    return ref


def short_reason(d):
    if d.decision == "approve":
        return "Approved: amount, bill number, date and vendor read and checked; no duplicate or policy exception" + (
            f" ({d.notes})" if d.notes else "")
    return "Review: " + d.reason


def make_predictions(ctx, decisions):
    """predictions.csv in the exact sample_submission.csv format. Missing photos: receipt fields stay blank."""
    sample, fields, test = ctx["sample"], ctx["fields"], ctx["test"]
    f = fields.loc[sample.claim_id]
    pred = pd.DataFrame({
        "claim_id": sample.claim_id.values, "vendor": f.vendor.values, "receipt_date": f.receipt_date.values,
        "receipt_number": f.receipt_number.values, "gstin": f.gstin.values, "total_amount": f.total_amount.values,
        "decision": [decisions.loc[c, "decision"] for c in sample.claim_id],
        "reason": [short_reason(decisions.loc[c]) for c in sample.claim_id],
    })
    assert list(pred.columns) == list(sample.columns) and len(pred) == len(sample)
    assert pred.claim_id.is_unique and set(pred.decision) <= {"approve", "review"}
    missing = test[test.photo_available != "True"].claim_id
    assert (pred.set_index("claim_id").loc[missing, ["vendor", "receipt_date", "receipt_number", "gstin", "total_amount"]] == "").all().all()
    return pred


def evaluate(ctx, decisions, ref):
    tr = decisions[decisions.split == "train"].join(ctx["train"].set_index("claim_id")[["status", "amount_claimed", "clerk_note"]])
    tr["clerk"] = tr.status.map({"approved": "approve", "queried": "review", "rejected": "review"})
    ev = tr.join(ref[["ref_decision", "ref_reasons"]])
    te = decisions[decisions.split == "test"]
    unsafe = ev[(ev.decision == "approve") & (ev.ref_decision == "review")]
    extra = ev[(ev.decision == "review") & (ev.ref_decision == "approve")]
    return pd.DataFrame([
        {"metric": "auto-approval coverage (train)", "value": f"{(tr.decision == 'approve').mean():.1%}"},
        {"metric": "auto-approval coverage (test)", "value": f"{(te.decision == 'approve').mean():.1%}"},
        {"metric": "unsafe auto-approvals (train, vs policy reference)", "value": f"{len(unsafe)} of {(ev.ref_decision == 'review').sum()} review-worthy"},
        {"metric": "extra reviews (train)", "value": f"{len(extra)} of {len(ev)}"},
        {"metric": "agreement with clerk decisions (train)", "value": f"{(tr.decision == tr.clerk).mean():.1%}"},
    ]), ev


def run(root):
    """Writes predictions.csv, reports/decisions.csv and reports/decision_summary.csv."""
    root = Path(root)
    ctx = load(root)
    decisions = run_all(ctx)
    ref = policy_reference(ctx)
    version = rules_version()
    summary, ev = evaluate(ctx, decisions, ref)
    pred = make_predictions(ctx, decisions)
    pred.to_csv(root / "predictions.csv", index=False)
    out = decisions.join(ctx["claims"].set_index("claim_id")[["category", "amount_claimed", "emp_id"]])
    out = out.join(ref[["ref_decision", "ref_reasons"]].assign(ref_reasons=lambda d: d.ref_reasons.map("; ".join)))
    out["rules_version"] = version
    out.drop(columns="decision_ms").reset_index().to_csv(root / "reports" / "decisions.csv", index=False)   # timings vary per run
    return pred, summary, version
