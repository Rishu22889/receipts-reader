"""Service tests: the endpoint, the screen, and every way it can fail. A failure must never produce an approval."""
from pathlib import Path
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from tarang import service

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "processed" / "receipt_fields.csv").exists() or shutil.which("tesseract") is None,
                                reason="needs the processed data (README setup) and tesseract")
client = TestClient(service.app)
TEST = pd.read_csv(ROOT / "data" / "processed" / "claims_test_clean.csv", dtype=str, keep_default_na=False).set_index("claim_id") \
    if (ROOT / "data" / "processed" / "claims_test_clean.csv").exists() else None


def form_for(cid, **over):
    c = TEST.loc[cid]
    return {"claim_id": cid, "emp_id": c.emp_id, "category": c.category, "vendor": c.vendor, "amount_claimed": c.amount_claimed,
            "expense_date": c.expense_date, "submitted_on": c.submitted_on, **over}


def photo_of(cid):
    return {"photo": ("receipt.jpg", (ROOT / "data" / "raw" / "images" / TEST.loc[cid, "image_file"]).read_bytes(), "image/jpeg")}


def test_screen_and_health():
    assert "Receipt check" in client.get("/").text
    h = client.get("/api/health").json()
    assert h["rules_version"] == "95bef2d00f35" and h["paid_api"] == "none used" and h["can_read_receipts"]


def test_clear_receipt_is_approved_and_matches_the_batch():
    j = client.post("/api/claims/check", data=form_for("CLM-31454"), files=photo_of("CLM-31454")).json()
    pred = pd.read_csv(ROOT / "predictions.csv", dtype=str, keep_default_na=False).set_index("claim_id").loc["CLM-31454"]
    assert j["decision"] == "approve" == pred.decision
    assert j["fields"]["total_amount"] == pred.total_amount and j["fields"]["receipt_number"] == pred.receipt_number


def test_missing_photo_goes_to_review():
    j = client.post("/api/claims/check", data=form_for("CLM-31190")).json()
    assert j["decision"] == "review" and j["reason_codes"] == ["MISSING_IMAGE"]
    assert all(v == "" for v in j["fields"].values())


def test_duplicate_same_photo_under_a_new_claim_id():
    client.post("/api/claims/check", data=form_for("CLM-31454"), files=photo_of("CLM-31454"))
    j = client.post("/api/claims/check", data=form_for("CLM-31454", claim_id=""), files=photo_of("CLM-31454")).json()
    assert j["decision"] == "review" and {"DUPLICATE_BILL", "DUPLICATE_PHOTO"} <= set(j["reason_codes"])


def test_altered_bill_goes_to_review():
    j = client.post("/api/claims/check", data=form_for("CLM-31299"), files=photo_of("CLM-31299")).json()
    assert j["decision"] == "review" and "ALTERED_BILL" in j["reason_codes"] and j["fields"]["total_amount"] == "472.00"


def test_instruction_stamp_is_data_not_an_instruction():
    j = client.post("/api/claims/check", data=form_for("CLM-31401"), files=photo_of("CLM-31401")).json()
    assert j["decision"] == "review" and "INSTRUCTION_TEXT" in j["reason_codes"]
    assert "Reconciled by Tarang" not in j["reason"]


def test_file_that_is_not_an_image_goes_to_review():
    j = client.post("/api/claims/check", data=form_for("CLM-31454"), files={"photo": ("x.jpg", b"not an image", "image/jpeg")}).json()
    assert j["decision"] == "review" and j["reason_codes"][0] == "IMAGE_NOT_OPENABLE"


def test_no_reader_available_goes_to_review(monkeypatch):
    monkeypatch.setattr(service, "tesseract_available", lambda: False)
    monkeypatch.setattr(service, "VLM_MODE", "off")
    j = client.post("/api/claims/check", data=form_for("CLM-31454"), files=photo_of("CLM-31454")).json()
    assert j["decision"] == "review" and j["reason_codes"][0] == "EXTRACTOR_UNAVAILABLE"


def test_live_vlm_misconfigured_falls_back_with_a_warning(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "VLM_MODE", "live")
    monkeypatch.setitem(service._state, "vlm_model", None)
    monkeypatch.setitem(service._state, "vlm_error", None)
    im = Image.new("RGB", (900, 500), "white")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=40)
    for i, line in enumerate(["SPICE ROUTE KITCHEN", "Bill No: 9999", "Date: 01-08-2026", "GRAND TOTAL Rs 250.00"]):
        d.text((40, 40 + 100 * i), line, fill="black", font=font)
    path = tmp_path / "new.jpg"
    im.save(path)
    j = client.post("/api/claims/check", data=form_for("CLM-31454", claim_id="NEW-LIVE"),
                    files={"photo": ("new.jpg", path.read_bytes(), "image/jpeg")}).json()
    assert any("live VLM unavailable" in w for w in j["warnings"])     # no GPU / torch here: warned, not crashed
    assert "Tesseract" in j["reader"] and j["decision"] in ("approve", "review")


def test_bad_inputs_are_rejected_not_decided():
    assert client.post("/api/claims/check", data=form_for("CLM-31454", emp_id="TPD-9999")).status_code == 422
    assert client.post("/api/claims/check", data=form_for("CLM-31454", category="Gifts")).status_code == 422
    assert client.post("/api/claims/check", data=form_for("CLM-31454", amount_claimed="lots")).status_code == 422


def test_missing_reference_data_is_a_503_not_an_approval(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "ROOT", tmp_path)
    monkeypatch.setitem(service._state, "loaded", False)
    r = client.post("/api/claims/check", data=form_for("CLM-31454"))
    assert r.status_code == 503 and "Nothing was approved" in r.json()["detail"]
    monkeypatch.setitem(service._state, "load_error", None)
