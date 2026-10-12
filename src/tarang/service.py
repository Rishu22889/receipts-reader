"""Receipt-check service: one endpoint (photo + claim as the rep entered it -> fields, checks, approve/review) and one screen.

Run:  uvicorn tarang.service:app --app-dir src        (then open http://127.0.0.1:8000)

Design, in order of importance:
* It never approves on a failure. No photo, a photo that won't open, or no working reader -> review, with the reason.
* Nothing heavy loads at startup. Data, history and readers load on the first request; /api/health reports what is available.
* The decision rules are the frozen ones (tarang.decision_engine, rules version in every response). Service-level problems
  get their own codes (SERVICE_REASONS) outside those rules.
* Readers, best first: PaddleOCR-VL text for a photo the GPU batch already read (matched by image hash, labelled "cached");
  optional live PaddleOCR-VL (TARANG_VLM=live, needs torch + transformers + a GPU); Tesseract with preprocessing, live on CPU.
  No paid API is called, so no key is needed.
"""
from datetime import date
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import traceback

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from tarang import decision_engine as de
from tarang.decision_batch import rules_version
from tarang.fields import build_vendor_master, canonical_vendors, combine, norm, parse_receipt
from tarang.images import SAME_PHOTO_MAX_DISTANCE, dhash, orientation, quality
from tarang.ocr import load_jsonl, merge_retry, tesseract_pre

ROOT = Path(os.environ.get("TARANG_ROOT", Path(__file__).resolve().parents[2]))
VLM_MODE = os.environ.get("TARANG_VLM", "cache").lower()        # off | cache | live
MAX_UPLOAD_BYTES = int(float(os.environ.get("TARANG_MAX_UPLOAD_MB", "10")) * 1024 * 1024)
CATEGORIES = ["Fuel", "Food", "Lodging", "Local travel", "Courier", "Parking"]
SERVICE_REASONS = {
    "EXTRACTOR_UNAVAILABLE": "No receipt reader is available on this server (Tesseract missing and no VLM): a person must read it",
    "IMAGE_NOT_OPENABLE": "The uploaded file is not a readable image",
    "EXTRACTION_FAILED": "The reader crashed on this photo: a person must read it",
}

app = FastAPI(title="Tarang receipt check", version="1.0")
_lock = threading.Lock()
_state = {"loaded": False, "load_error": None, "session": [], "vlm_model": None, "vlm_error": None}


# ---------------------------------------------------------------------------------------- lazy resources
def resources():
    """Load reference data once, on first use. A failure is remembered and reported, never raised at startup."""
    with _lock:
        if _state["loaded"]:
            return _state
        try:
            read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
            train = read(ROOT / "data" / "processed" / "claims_train_clean.csv").assign(split="train")
            test = read(ROOT / "data" / "processed" / "claims_test_clean.csv").assign(split="test")
            claims = pd.concat([train, test], ignore_index=True).fillna("")
            fx = read(ROOT / "data" / "processed" / "receipt_fields.csv")
            fx = fx[fx.method == "combined"].set_index("claim_id")
            labelled = train[train.clerk_bill_date != ""]
            canon = canonical_vendors(labelled.clerk_vendor.tolist())
            vlm_first = ROOT / "data" / "interim" / "vlm" / "paddleocr_vl_outputs.jsonl"
            vlm = None
            if vlm_first.exists():
                vlm = load_jsonl(vlm_first)
                retry = ROOT / "data" / "interim" / "vlm" / "paddleocr_vl_retry.jsonl"
                if retry.exists():
                    vlm = merge_retry(vlm, load_jsonl(retry))
            text_for_master = labelled.image_file.map(vlm.text) if vlm is not None else ""
            master = build_vendor_master(labelled.assign(canonical_vendor=labelled.clerk_vendor.map(canon), text=text_for_master))
            images = ROOT / "data" / "raw" / "images"
            hashes = {p.name: dhash(p) for p in sorted(images.glob("*.jpg"))} if images.exists() else {}
            photo_of = dict(zip(claims.claim_id, claims.image_file))
            history = pd.DataFrame({
                "claim_id": claims.claim_id, "status": claims.status,
                "vendor_key": claims.claim_id.map(fx.vendor).fillna("").map(norm),
                "bill_key": [de.bill_key(fx.vendor.get(c, ""), fx.receipt_number.get(c, "")) for c in claims.claim_id],
                "receipt_date": claims.claim_id.map(fx.receipt_date).fillna(""),
                "total_amount": claims.claim_id.map(fx.total_amount).fillna(""),
                "photo_hash": [hashes.get(photo_of[c], "") for c in claims.claim_id],
            })
            employees = read(ROOT / "data" / "raw" / "employees.csv").set_index("emp_id")
            _state.update(loaded=True, load_error=None, employees=employees, master=master, history=history,
                          vlm_text=vlm, photo_hashes=hashes, rules_version=rules_version(),
                          claims=claims.set_index("claim_id"))
        except Exception as e:   # reported by /api/health and every response; requests fail safe
            _state.update(load_error=f"{type(e).__name__}: {e}")
            raise
        return _state


def tesseract_available():
    return shutil.which("tesseract") is not None


def live_vlm():
    """Lazy-load PaddleOCR-VL (native transformers) only when TARANG_VLM=live. Any failure becomes a warning, not a crash."""
    if VLM_MODE != "live":
        return None, None
    with _lock:
        if _state["vlm_model"] is not None or _state["vlm_error"]:
            return _state["vlm_model"], _state["vlm_error"]
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            if not torch.cuda.is_available() and os.environ.get("TARANG_VLM_ALLOW_CPU") != "1":
                raise RuntimeError("no CUDA GPU (set TARANG_VLM_ALLOW_CPU=1 to try on CPU; needs >8 GB RAM and minutes per photo)")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability(0) >= (8, 0) else torch.float32
            model = AutoModelForImageTextToText.from_pretrained("PaddlePaddle/PaddleOCR-VL", dtype=dtype).to(device).eval()
            model.generation_config.use_cache = True     # the published config disables the KV cache
            _state["vlm_model"] = (model, AutoProcessor.from_pretrained("PaddlePaddle/PaddleOCR-VL"), device)
        except Exception as e:
            _state["vlm_error"] = f"live VLM unavailable ({type(e).__name__}: {e}); using Tesseract"
        return _state["vlm_model"], _state["vlm_error"]


def read_with_live_vlm(path, angle):
    model, processor, device = _state["vlm_model"]
    import torch
    image = Image.open(path).convert("RGB").rotate(angle, expand=True)
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "OCR:"}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=768, do_sample=False, use_cache=True, repetition_penalty=1.05)
    new = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True)[0], int(new.shape[1])


# ---------------------------------------------------------------------------------------- reading one photo
def read_receipt(path, st):
    """-> (fields, details, reader, warnings). Raises RuntimeError('EXTRACTOR_UNAVAILABLE') if nothing can read."""
    warnings = []
    h = dhash(path)
    cached_file = None
    if VLM_MODE in ("cache", "live") and st["vlm_text"] is not None:
        near = [(sum(a != b for a, b in zip(h, ph)), f) for f, ph in st["photo_hashes"].items() if ph]
        near = [x for x in near if x[0] <= SAME_PHOTO_MAX_DISTANCE]
        if near:
            cached_file = min(near)[1]
    if not tesseract_available() and cached_file is None and VLM_MODE != "live":
        raise RuntimeError("EXTRACTOR_UNAVAILABLE")

    angle, tess_fields, tess_text = 0, None, ""
    if tesseract_available():
        angle = orientation(path)["rotation"]
        tess_text = tesseract_pre(path, angle)["text"]
        tess_fields = parse_receipt(tess_text, st["master"])
    else:
        warnings.append("Tesseract is not installed: no local reading")

    vlm_text, vlm_tokens, reader = None, None, []
    if cached_file is not None:
        rec = st["vlm_text"].loc[cached_file]
        vlm_text, vlm_tokens = rec.text, rec.get("new_tokens")
        reader.append(f"PaddleOCR-VL (cached GPU batch output for this photo: {cached_file})")
    elif VLM_MODE == "live":
        model, err = live_vlm()
        if model is not None:
            vlm_text, vlm_tokens = read_with_live_vlm(path, angle)
            reader.append("PaddleOCR-VL (live)")
        else:
            warnings.append(err)
    if tess_fields is not None:
        reader.append("Tesseract (live, preprocessed)")

    if vlm_text is None and tess_fields is None:
        raise RuntimeError("EXTRACTOR_UNAVAILABLE")
    if vlm_text is not None:
        vlm_fields = parse_receipt(vlm_text, st["master"], vlm_tokens)
        fields = combine(vlm_fields, tess_fields) if tess_fields is not None else vlm_fields
        text = vlm_text
    else:
        fields, text = tess_fields, tess_text
        warnings.append("Only Tesseract read this photo: it misreads small digits often, so most claims will go to review")
    q = quality(path)
    details = de.parse_details(text)
    details.update(rotation=angle, purple_pct=q["purple_pct"])
    return fields, details, h, " + ".join(reader), warnings


# ---------------------------------------------------------------------------------------- endpoints
@app.get("/")
def screen():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health():
    """What this server can do right now. Never loads the VLM."""
    out = {"status": "ok", "tesseract": tesseract_available(), "vlm_mode": VLM_MODE, "vlm_error": _state["vlm_error"],
           "paid_api": "none used"}
    try:
        st = resources()
        out.update(rules_version=st["rules_version"], claims_in_history=len(st["history"]),
                   cached_vlm_photos=0 if st["vlm_text"] is None else len(st["vlm_text"]))
    except Exception:
        out["status"] = "degraded"
    out.update(data_loaded=_state["loaded"], load_error=_state["load_error"])
    cached = VLM_MODE in ("cache", "live") and bool(out.get("cached_vlm_photos"))
    out["can_read_receipts"] = out["tesseract"] or VLM_MODE == "live" or cached
    out["can_read_new_photos"] = out["tesseract"] or VLM_MODE == "live"   # cached text only covers the data pack's photos
    return out


@app.get("/api/employees")
def list_employees():
    st = resources()
    return [{"emp_id": e, "name": r["name"], "designation": r["designation"], "headquarter": r["headquarter"], "grade": r["grade"]}
            for e, r in st["employees"].iterrows()]


@app.get("/api/reasons")
def reasons():
    return {**{c: {"meaning": m, "blocks_approval": b} for c, (m, b) in de.REASONS.items()},
            **{c: {"meaning": m, "blocks_approval": True} for c, m in SERVICE_REASONS.items()}}


EXAMPLES = [   # real claims from the data pack, for the demo screen (expected outcome in brackets)
    ("CLM-31454", "Clear fuel receipt (approve)"),
    ("CLM-31475", "Re-claim of a bill already approved (review: duplicate)"),
    ("CLM-31299", "Total altered by pen 472 -> 1472 (review: altered bill)"),
    ("CLM-31190", "Photo never uploaded (review: missing image)"),
    ("CLM-31401", "'Approve it' stamp on the bill (review: instruction text)"),
    ("CLM-31120", "Colleague's vehicle on a fuel bill (review)"),
]


@app.get("/api/examples")
def examples():
    st = resources()
    out = []
    for cid, label in EXAMPLES:
        if cid not in st["claims"].index:
            continue
        c = st["claims"].loc[cid]
        out.append({"claim_id": cid, "label": label, "emp_id": c.emp_id, "category": c.category, "vendor": c.vendor,
                    "amount_claimed": c.amount_claimed, "expense_date": c.expense_date, "submitted_on": c.submitted_on,
                    "has_photo": c.photo_available == "True"})
    return out


@app.get("/api/examples/{claim_id}/photo")
def example_photo(claim_id: str):
    """Photo of an example claim, looked up by claim id (never by a client-supplied path)."""
    st = resources()
    if claim_id not in dict(EXAMPLES) or claim_id not in st["claims"].index or st["claims"].loc[claim_id, "photo_available"] != "True":
        raise HTTPException(404, "no photo for this example")
    return FileResponse(ROOT / "data" / "raw" / "images" / st["claims"].loc[claim_id, "image_file"], media_type="image/jpeg")


@app.post("/api/claims/check")
async def check_claim(
    emp_id: str = Form(...), category: str = Form(...), vendor: str = Form(""), amount_claimed: str = Form(...),
    expense_date: str = Form(...), submitted_on: str = Form(""), claim_id: str = Form(""),
    photo: UploadFile | None = File(None),
):
    """Check one claim. claim_id is optional: give an existing id to re-check that claim (it is left out of its own history)."""
    t0 = time.perf_counter()
    try:
        st = resources()
    except Exception:
        raise HTTPException(503, f"reference data not loaded ({_state['load_error']}); run the setup in README.md. Nothing was approved.")
    if emp_id not in st["employees"].index:
        raise HTTPException(422, f"unknown employee {emp_id!r}")
    if category not in CATEGORIES:
        raise HTTPException(422, f"category must be one of {CATEGORIES}")
    try:
        amount = float(amount_claimed.replace(",", "").replace("Rs", "").strip())
        pd.Timestamp(expense_date)
        submitted_on = submitted_on or date.today().isoformat()
        pd.Timestamp(submitted_on)
    except (ValueError, TypeError):
        raise HTTPException(422, "amount_claimed must be a number and dates YYYY-MM-DD")
    claim_id = claim_id.strip() or f"NEW-{len(st['session']) + 1:04d}"
    employee = st["employees"].loc[emp_id].to_dict()
    claim = {"claim_id": claim_id, "emp_id": emp_id, "category": category, "vendor": vendor, "amount_claimed": f"{amount:.2f}",
             "expense_date": expense_date, "submitted_on": submitted_on, "photo_available": "False", "photo_group": "",
             "instruction_stamp": ""}
    blank = {"vendor": "", "vendor_status": "absent", "receipt_date": "", "date_status": "absent", "receipt_number": "",
             "number_status": "absent", "gstin": "", "gstin_read": "", "gstin_status": "absent", "total_amount": "",
             "total_status": "absent", "looped": False}
    fields, details, reader, warnings, service_codes, photo_hash = blank, {}, "", [], [], ""

    if photo is not None and photo.filename:
        data = await photo.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"photo larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.jpg"
            path.write_bytes(data)
            try:
                with Image.open(path) as im:
                    im.verify()
                    fmt = im.format
                if fmt != "JPEG":   # convert PNG etc.; a JPEG is kept byte-for-byte so it reads exactly as in the batch run
                    Image.open(path).convert("RGB").save(path, "JPEG", quality=95)
                claim["photo_available"] = "True"
            except Exception:
                service_codes.append("IMAGE_NOT_OPENABLE")
            if claim["photo_available"] == "True":
                try:
                    fields, details, photo_hash, reader, w = read_receipt(path, st)
                    warnings += w
                except RuntimeError as e:
                    if str(e) != "EXTRACTOR_UNAVAILABLE":
                        raise
                    service_codes.append("EXTRACTOR_UNAVAILABLE")
                except Exception:
                    service_codes.append("EXTRACTION_FAILED")
                    warnings.append(traceback.format_exc(limit=1))

    # history: every dataset claim and every claim checked in this session, except this claim itself
    session = pd.DataFrame(st["session"], columns=st["history"].columns)
    hist = pd.concat([st["history"], session], ignore_index=True).drop_duplicates("claim_id", keep="last")   # a re-check replaces the stored row
    hist = hist[hist.claim_id != claim_id].copy()
    if photo_hash:
        same = hist.photo_hash.map(lambda p: bool(p) and sum(a != b for a, b in zip(p, photo_hash)) <= SAME_PHOTO_MAX_DISTANCE)
        hist["photo_group"] = same.map({True: "same", False: ""})
        claim["photo_group"] = "same" if same.any() else ""
    else:
        hist["photo_group"] = ""
    claim["instruction_stamp"] = "True" if details.get("purple_pct", 0) > 0.5 else ""

    validation = de.validate_fields(fields, claim, employee, details, st["employees"]) if not service_codes else []
    decision = de.decide_claim(fields, validation, claim, employee, hist if not service_codes else None)
    codes, reason_text = list(decision.reason_codes), decision.reason
    if service_codes:   # never approve when the photo could not be read
        codes = service_codes + [c for c in codes if c not in service_codes]
        reason_text = "; ".join(f"{c}: {SERVICE_REASONS[c]}" for c in service_codes)
    final = "review" if service_codes or decision.decision == "review" else "approve"

    st["session"].append({"claim_id": claim_id, "status": "approved" if final == "approve" else "", "vendor_key": norm(fields["vendor"]),
                          "bill_key": de.bill_key(fields["vendor"], fields["receipt_number"]), "receipt_date": fields["receipt_date"],
                          "total_amount": fields["total_amount"], "photo_hash": photo_hash})
    return {
        "claim_id": claim_id, "decision": final, "reason_codes": codes, "notes": decision.notes, "reason": reason_text,
        "fields": {k: fields[k] for k in ["vendor", "receipt_date", "receipt_number", "gstin", "total_amount"]},
        "field_status": {k: fields[k] for k in ["vendor_status", "date_status", "number_status", "gstin_status", "total_status"]},
        "checks": [{"check": c.name, "passed": c.passed, "code": c.reason, "detail": c.detail} for c in decision.checks],
        "details": {k: v for k, v in details.items() if k not in ("purple_pct",)},
        "reader": reader or "none", "warnings": [w for w in warnings if w],
        "rules_version": st["rules_version"], "seconds": round(time.perf_counter() - t0, 2),
    }
