"""Decision engine: receipt fields + claim + employee + claim history -> approve | review, with reason codes.

The OCR/VLM only reads; every decision below is a plain Python rule traceable to expense-policy.pdf (v3.2).
Section references (P2, P3, ...) point to the policy's numbered sections. Packaged from notebooks/04_decision_engine.ipynb,
which shows the evidence behind each rule and both evaluations (clerk agreement, policy correctness).

    fields = extract_receipt(photo_path, master)              # read: Tesseract (+ VLM text if available)
    validation = validate_fields(fields, claim, employee, details, employees)
    result = decide_claim(fields, validation, claim, employee, history)
"""
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re

import pandas as pd

from tarang.fields import clean_text, find_dates, norm, similarity
from tarang.images import injection_hits

APPROVAL_LIMIT = 2000          # P7: clerks check claims of Rs 2,000 and above
LODGING_LIMIT = {"M1": 3000, "M2": 3000, "M3": 4500}   # P2: per night before GST
DATE_TOLERANCE_DAYS = 1
CITIES = ["Nagpur", "Wardha", "Amravati", "Yavatmal", "Chandrapur"]

# ---------------------------------------------------------------------------------------- reason codes
# code: (meaning, blocks auto-approval?)
REASONS = {
    "MISSING_IMAGE": ("No photo reached the server", True),
    "UNREADABLE_IMAGE": ("Photo present but neither amount nor bill number could be read", True),
    "AMOUNT_OVER_LIMIT": ("Claim or bill is Rs 2,000 or more: clerk checks it (P7)", True),
    "AMOUNT_UNVERIFIED": ("Bill total not read, or read but contradicted by the receipt itself", True),
    "AMOUNT_MISMATCH": ("Claimed amount differs from the bill beyond rounding to the nearest Rs 10 (P3)", True),
    "ALTERED_BILL": ("Bill total disagrees with the bill's own amount line / arithmetic: possible alteration (P3, P8)", True),
    "DUPLICATE_BILL": ("Same vendor and bill number already claimed (P3, P8)", True),
    "DUPLICATE_PHOTO": ("Same photo used on another claim (P3, P8)", True),
    "POSSIBLE_DUPLICATE": ("Bill number unreadable; same vendor, date and amount already claimed", True),
    "INSTRUCTION_TEXT": ("Text on the bill addresses automated processing; treated as data, sent to a person", True),
    "FIELDS_INCOMPLETE": ("Vendor, bill date or bill number not read", True),
    "NUMBER_FORMAT_MISMATCH": ("Bill number does not fit the vendor's numbering: likely misread", True),
    "DATE_MISMATCH": ("Bill date differs from the claimed expense date by more than a day, or is after submission", True),
    "VENDOR_MISMATCH": ("Vendor on the bill differs from the vendor the rep typed", True),
    "FUEL_VEHICLE_MISSING": ("Fuel bill does not show a vehicle number (P2)", True),
    "FUEL_VEHICLE_OTHER_EMPLOYEE": ("Fuel bill shows another employee's vehicle: colleague's bill (P2, P8)", True),
    "FUEL_VEHICLE_MISMATCH": ("Fuel bill vehicle number is not the claimant's (P2)", True),
    "MEMO_NAME_MISMATCH": ("Travel memo is made out to someone else: shared bill claimed by a non-payer? (P3)", True),
    "LODGING_NOT_OUTSTATION": ("Hotel is in the employee's headquarter city (P2)", True),
    "LODGING_OVER_LIMIT": ("Room rate per night above the grade limit before GST (P2)", True),
    "LODGING_NAME_MISMATCH": ("Hotel invoice is not in the employee's name (P2)", True),
    "LODGING_NOT_GST_INVOICE": ("Hotel bill has no valid GSTIN: not a GST tax invoice (P2)", True),
    # informational, never block on their own
    "GSTIN_ABSENT": ("No GSTIN printed: fine for most bills", False),
    "GSTIN_UNREADABLE": ("A GSTIN is printed but could not be read with a valid checksum", False),
    "ROUNDED_TO_10": ("Claim is the bill rounded to the nearest Rs 10: accepted (P3)", False),
}

# Policy matrix: every rule, where it comes from, and the regression test that proves it.
POLICY_MATRIX = [
    ("P3", "No readable bill photo", "review", "MISSING_IMAGE / UNREADABLE_IMAGE", "test_missing_photo_goes_to_review"),
    ("P7", "Claim or bill >= Rs 2,000", "review", "AMOUNT_OVER_LIMIT", "test_over_limit_goes_to_review"),
    ("P3", "Bill total unread or self-contradicted", "review", "AMOUNT_UNVERIFIED", "test_unverified_amount_goes_to_review"),
    ("P3", "Claimed != bill, beyond nearest-Rs-10 rounding", "review", "AMOUNT_MISMATCH", "test_amount_mismatch_and_rounding"),
    ("P3/P8", "Bill total contradicts its own amount line or arithmetic", "review", "ALTERED_BILL", "test_altered_fuel_total"),
    ("P3/P8", "Same vendor + bill number claimed before (any employee)", "review", "DUPLICATE_BILL", "test_duplicate_bill_*"),
    ("P3/P8", "Same photo on another claim", "review", "DUPLICATE_PHOTO", "test_duplicate_photo_pairs"),
    ("P3/P8", "Bill no. unreadable, same vendor + date + amount claimed before", "review", "POSSIBLE_DUPLICATE", "test_possible_duplicate"),
    ("-", "Instruction-like text on the bill", "review", "INSTRUCTION_TEXT", "test_instruction_stamp"),
    ("P4", "Vendor / date / bill number not read", "review", "FIELDS_INCOMPLETE", "test_incomplete_fields"),
    ("P4", "Bill number outside the vendor's format", "review", "NUMBER_FORMAT_MISMATCH", "test_number_format"),
    ("P4", "Bill date != expense date (>1 day) or after submission", "review", "DATE_MISMATCH", "test_date_mismatch"),
    ("P4", "Bill vendor != vendor typed by the rep", "review", "VENDOR_MISMATCH", "test_vendor_mismatch"),
    ("P2", "Fuel bill without vehicle number", "review", "FUEL_VEHICLE_MISSING", "test_fuel_vehicle_rules"),
    ("P2/P8", "Fuel bill with a colleague's vehicle", "review", "FUEL_VEHICLE_OTHER_EMPLOYEE", "test_fuel_vehicle_rules"),
    ("P2", "Fuel bill with an unknown vehicle", "review", "FUEL_VEHICLE_MISMATCH", "test_fuel_vehicle_rules"),
    ("P3", "Travel memo in another person's name", "review", "MEMO_NAME_MISMATCH", "test_memo_name"),
    ("P2", "Lodging in headquarter city", "review", "LODGING_NOT_OUTSTATION", "test_lodging_rules"),
    ("P2", "Lodging rate above grade limit (before GST)", "review", "LODGING_OVER_LIMIT", "test_lodging_rules"),
    ("P2", "Hotel invoice not in employee's name", "review", "LODGING_NAME_MISMATCH", "test_lodging_rules"),
    ("P2", "Hotel bill without valid GSTIN", "review", "LODGING_NOT_GST_INVOICE", "test_lodging_rules"),
    ("P3", "Pen notes (doctor, pax, 'paid'), daily Rs 30 parking", "no effect", "-", "test_allowed_notes_and_parking"),
    ("-", "GSTIN absent / unreadable on a non-hotel bill", "no effect (noted)", "GSTIN_ABSENT / GSTIN_UNREADABLE", "test_gstin_states"),
    ("P7", "Everything above passes, claim < Rs 2,000", "approve", "-", "test_clean_claim_is_approved"),
]


# ---------------------------------------------------------------------------------------- extra receipt details
VEHICLE = re.compile(r"veh(?:icle)?\s*no\.?\s*[:.]?\s*([A-Z]{2}\s?\d{1,2}\s?[A-Z]{0,3}\s?\d{3,4})", re.I)
NUM = r"(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"


def _after(label, text, same_line_only=False):
    """Number after a label, on the same line or (VLM layout) the next line."""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if re.search(label, l, re.I):
            m = re.search(label + r"[^\d]*" + NUM, l, re.I)
            if m:
                return float(m.group(m.lastindex).replace(",", ""))
            if not same_line_only and i + 1 < len(lines):
                n = re.fullmatch(r"\s*(?:rs\.?)?\s*" + NUM + r"\s*", lines[i + 1], re.I)
                if n:
                    return float(n.group(1).replace(",", ""))
    return None


def parse_details(text):
    """Policy-relevant details beyond the five fields: vehicle, fuel arithmetic, hotel stay, memo name, instruction text."""
    t = clean_text(text)
    d = {}
    m = VEHICLE.search(t)
    d["vehicle_number"] = re.sub(r"\s", "", m.group(1).upper()) if m else ""
    d["fuel_rate"] = _after(r"rate\s*\(rs/l\)", t)
    d["fuel_volume"] = _after(r"volume\s*\(l\)", t)
    d["fuel_amount_line"] = _after(r"amount\s*\(rs\)", t)
    d["printed_total"] = _after(r"\btotal\s*rs\.?", t)
    g = re.search(r"guest\s*:?\s*(?:mr\.?\s*/?\s*ms\.?\s*)?([A-Za-z][A-Za-z .]+)", t, re.I)
    d["guest_name"] = g.group(1).strip() if g else ""
    d["hotel_city"] = next((c for l in t.splitlines()[:4] for c in CITIES if c.lower() in l.lower()), "")
    d["nights"] = _after(r"\bnights\b", t, same_line_only=True)
    room = re.search(r"room\s*tariff.*?\n?.*?([\d,]+\.\d\d)", t, re.I | re.S)
    d["room_rate"] = float(room.group(1).replace(",", "")) if room else None
    nm = re.search(r"\bname\s*:?\s*([A-Za-zÀ-ſ]+)", t, re.I)
    d["memo_name"] = nm.group(1) if nm else ""
    d["instruction_hits"] = "; ".join(injection_hits(t))
    return d


# ---------------------------------------------------------------------------------------- validation
@dataclass
class Check:
    name: str
    passed: bool
    reason: str = ""          # reason code when the check fails (or an informational code)
    detail: str = ""


def _amount(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def rounding_ok(claimed, bill):
    """P3: the claim is the bill amount, or the bill rounded to the nearest Rs 10 (rounding to the rupee is gentler, also fine)."""
    if claimed is None or bill is None:
        return False
    allowed = {round(bill, 2), float(int(bill + 0.5)), float(int(bill / 10 + 0.5) * 10)}
    return any(abs(claimed - a) < 0.005 for a in allowed)


def plate_distance(a, b):
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def validate_fields(fields, claim, employee, details=None, employees=None, vendor_shapes=None):
    """Field-level and policy checks for one claim. Returns a list of Check (no history needed here)."""
    details = details or {}
    checks = []
    if claim.get("photo_available") not in (True, "True"):
        return [Check("photo", False, "MISSING_IMAGE", "photo never reached the server")]

    claimed = _amount(claim.get("amount_claimed"))
    bill = _amount(fields.get("total_amount"))
    readable = bool(fields.get("total_amount") or fields.get("receipt_number"))
    checks.append(Check("readable", readable, "UNREADABLE_IMAGE", "no amount and no bill number read"))
    if not readable:
        return checks

    limit_hit = max(x for x in (claimed, bill, 0) if x is not None) >= APPROVAL_LIMIT
    checks.append(Check("under_limit", not limit_hit, "AMOUNT_OVER_LIMIT", f"claimed {claimed}, bill {bill}"))

    total_status = fields.get("total_status", "").replace("_fallback", "")
    # altered bill: the printed total contradicts the bill's own arithmetic (fuel rate x volume and amount line, or
    # sub total + taxes); fields.parse_total then reports the self-consistent amount
    altered = total_status == "inconsistent"
    checks.append(Check("not_altered", not altered, "ALTERED_BILL", f"printed total contradicts the bill's own arithmetic; consistent amount {bill}"))
    amount_ok = total_status in ("verified", "ok", "words_only", "inconsistent") and not _truthy(fields.get("looped"))
    checks.append(Check("amount_read", amount_ok and bill is not None, "AMOUNT_UNVERIFIED", f"status {fields.get('total_status')}"))

    if bill is not None and claimed is not None:
        match = rounding_ok(claimed, bill)
        checks.append(Check("amount_matches_claim", match, "AMOUNT_MISMATCH", f"claimed {claimed} vs bill {bill}"))
        if match and abs(claimed - bill) >= 0.005:
            checks.append(Check("rounding", True, "ROUNDED_TO_10", f"{bill} -> {claimed}"))

    missing = [f for f, st in (("vendor", "vendor_status"), ("receipt_date", "date_status"), ("receipt_number", "number_status"))
               if not fields.get(f) or fields.get(st) in ("absent", "unknown")]
    checks.append(Check("fields_complete", not missing, "FIELDS_INCOMPLETE", "missing: " + ", ".join(missing)))
    checks.append(Check("number_format", fields.get("number_status") != "format_mismatch", "NUMBER_FORMAT_MISMATCH", fields.get("receipt_number", "")))

    if fields.get("receipt_date"):
        bd = pd.Timestamp(fields["receipt_date"])
        ed = pd.Timestamp(claim["expense_date"]) if claim.get("expense_date") else None
        late = claim.get("submitted_on") and bd > pd.Timestamp(claim["submitted_on"])
        far = ed is not None and abs((ed - bd).days) > DATE_TOLERANCE_DAYS
        if claim.get("category") == "Lodging" and details.get("nights"):
            far = ed is not None and not (bd - pd.Timedelta(days=int(details["nights"])) <= ed <= bd)   # any night of the stay
        checks.append(Check("date_matches_claim", not (late or far), "DATE_MISMATCH", f"bill {fields['receipt_date']} vs expense {claim.get('expense_date')}"))

    if fields.get("vendor") and claim.get("vendor"):
        checks.append(Check("vendor_matches_claim", similarity(fields["vendor"], claim["vendor"]) >= 0.6, "VENDOR_MISMATCH",
                            f"bill {fields['vendor']} vs typed {claim['vendor']}"))

    gst_state = fields.get("gstin_status", "absent").replace("_fallback", "")
    if gst_state == "absent":
        checks.append(Check("gstin", True, "GSTIN_ABSENT"))
    elif gst_state == "invalid":
        checks.append(Check("gstin", True, "GSTIN_UNREADABLE", fields.get("gstin_read", "")))

    if details.get("instruction_hits") or _truthy(claim.get("instruction_stamp")):
        checks.append(Check("no_instruction_text", False, "INSTRUCTION_TEXT", details.get("instruction_hits") or "violet stamp (image audit)"))

    category = claim.get("category")
    own = re.sub(r"\s", "", str(employee.get("vehicle_no", ""))).upper()
    if category == "Fuel":
        plate = details.get("vehicle_number", "")
        others = {} if employees is None else {re.sub(r"\s", "", v).upper(): e for e, v in employees.vehicle_no.items() if e != claim.get("emp_id")}
        if not plate:
            checks.append(Check("fuel_vehicle", False, "FUEL_VEHICLE_MISSING"))
        elif plate in others:
            checks.append(Check("fuel_vehicle", False, "FUEL_VEHICLE_OTHER_EMPLOYEE", f"{plate} belongs to {others[plate]}"))
        elif plate_distance(plate, own) > 1:
            checks.append(Check("fuel_vehicle", False, "FUEL_VEHICLE_MISMATCH", f"{plate} vs own {own}"))
        else:
            checks.append(Check("fuel_vehicle", True, "", f"{plate} ~ own {own}"))

    if category == "Local travel" and details.get("memo_name"):
        first = str(employee.get("name", "")).split()[0] if employee.get("name") else ""
        ok = SequenceMatcher(None, details["memo_name"].lower(), first.lower()).ratio() >= 0.6
        checks.append(Check("memo_name", ok, "MEMO_NAME_MISMATCH", f"memo {details['memo_name']} vs {first}"))

    if category == "Lodging":
        hq = employee.get("headquarter", "")
        city = details.get("hotel_city", "")
        checks.append(Check("lodging_outstation", not city or city != hq, "LODGING_NOT_OUTSTATION", f"hotel {city or '?'} vs HQ {hq}"))
        rate, limit = details.get("room_rate"), LODGING_LIMIT.get(employee.get("grade", ""), 3000)
        checks.append(Check("lodging_rate", rate is None or rate <= limit, "LODGING_OVER_LIMIT", f"{rate} per night vs {limit}"))
        guest = details.get("guest_name", "")
        checks.append(Check("lodging_name", not guest or similarity(guest, employee.get("name", "")) >= 0.7, "LODGING_NAME_MISMATCH",
                            f"guest {guest} vs {employee.get('name')}"))
        checks.append(Check("lodging_gst_invoice", gst_state.startswith("ok"), "LODGING_NOT_GST_INVOICE", gst_state))
    return checks


def _truthy(v):
    return v in (True, "True", "true", 1)


# ---------------------------------------------------------------------------------------- history + decision
def bill_key(vendor, number):
    return f"{norm(vendor)}|{number.strip().upper()}" if vendor and number else ""


def history_checks(claim_id, fields, claim, history):
    """Duplicate checks against every other claim (past and same batch).

    history: DataFrame of other claims with claim_id, vendor (as read), receipt_number, receipt_date, total_amount,
    photo_group (same-photo group id or ''), status ('approved' is used as a proxy for 'paid'; there is no payment record).
    """
    others = history[history.claim_id != claim_id]
    checks = []
    key = bill_key(fields.get("vendor", ""), fields.get("receipt_number", ""))
    same_bill = others[(others.bill_key == key)] if key else others.iloc[0:0]
    if len(same_bill):
        paid = same_bill[same_bill.status == "approved"]
        checks.append(Check("not_duplicate_bill", False, "DUPLICATE_BILL",
                            f"same bill as {', '.join(same_bill.claim_id)}" + (" (approved earlier: likely already paid)" if len(paid) else "")))
    group = claim.get("photo_group", "")
    if group:
        twins = others[others.photo_group == group]
        if len(twins):
            checks.append(Check("not_duplicate_photo", False, "DUPLICATE_PHOTO", f"same photo as {', '.join(twins.claim_id)}"))
    if not key and fields.get("vendor") and fields.get("receipt_date") and fields.get("total_amount"):
        twins = others[(others.vendor_key == norm(fields["vendor"])) & (others.receipt_date == fields["receipt_date"])
                       & (others.total_amount == fields["total_amount"])]
        if len(twins):
            checks.append(Check("not_possible_duplicate", False, "POSSIBLE_DUPLICATE", f"same vendor/date/amount as {', '.join(twins.claim_id)}"))
    return checks


@dataclass
class Decision:
    decision: str
    reason_codes: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    reason: str = ""
    checks: list = field(default_factory=list)


def decide_claim(fields, validation, claim, employee=None, history=None):
    """approve only if every blocking check passes; otherwise review, with every failing reason code."""
    checks = list(validation) + (history_checks(claim["claim_id"], fields, claim, history) if history is not None else [])
    blocking = [c for c in checks if not c.passed and REASONS.get(c.reason, ("", True))[1]]
    notes = sorted({c.reason for c in checks if c.reason and not REASONS.get(c.reason, ("", True))[1]})   # informational codes only
    codes = list(dict.fromkeys(c.reason for c in blocking))
    if blocking:
        reason = "; ".join(f"{c.reason}: {c.detail}" if c.detail else c.reason for c in blocking)
        return Decision("review", codes, notes, reason, checks)
    return Decision("approve", [], notes, "all checks passed" + (f" ({', '.join(notes)})" if notes else ""), checks)


# ---------------------------------------------------------------------------------------- extraction for one photo
def extract_receipt(image_path, master, vlm_text=None, vlm_new_tokens=None):
    """Read one receipt photo: fields (with statuses), policy details, and image signals.

    Tesseract with preprocessing always runs locally. If PaddleOCR-VL text is supplied (batch run on a GPU, or a hosted
    endpoint), the VLM is primary and Tesseract fills only checked gaps, as in the benchmark's "combined" method.
    """
    from tarang.fields import combine, parse_receipt
    from tarang.images import dhash, orientation, quality
    from tarang.ocr import tesseract_pre

    angle = orientation(image_path)["rotation"]
    tess = tesseract_pre(image_path, angle)
    fields = parse_receipt(tess["text"], master)
    text = tess["text"]
    if vlm_text:
        fields = combine(parse_receipt(vlm_text, master, vlm_new_tokens), fields)
        text = vlm_text
    q = quality(image_path)
    details = parse_details(text)
    details.update({"rotation": angle, "purple_pct": q["purple_pct"], "dhash": dhash(image_path),
                    "instruction_stamp": bool(details["instruction_hits"]) or q["purple_pct"] > 0.5})
    return {"fields": fields, "details": details, "reader": "paddleocr_vl+tesseract" if vlm_text else "tesseract_pre"}
