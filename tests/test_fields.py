from pathlib import Path

import pandas as pd
import pytest

from tarang.benchmark import ADJUDICATION, ADJUDICATION_COLUMNS, select_holdout, wilson
from tarang.fields import (build_vendor_master, canonical_vendors, clean_text, combine, gstin_is_valid, looped, number_shape,
                           parse_date, parse_gstin, parse_receipt, parse_receipt_number, parse_total, parse_vendor, words_to_number)

ROOT = Path(__file__).resolve().parent.parent


# --- text -------------------------------------------------------------------------------------------------------------------

def test_clean_text_strips_table_markup_and_splits_run_together_labels():
    raw = "<fcel>HPCL<lcel><nl><fcel>Receipt No : 002147Date: 13/06/26Amount (Rs): 300.00Vehicle No : MH31DX0981<nl>"
    lines = clean_text(raw).splitlines()
    assert "Receipt No : 002147" in lines and "Amount (Rs): 300.00" in lines
    assert not any("<" in l for l in lines)


def test_looped_output_is_detected():
    assert looped("No. 3678\n" * 20)
    assert looped("short", new_tokens=768)
    assert not looped("BPCL\nTOTAL Rs. 250.00", new_tokens=120)


# --- vendor names -------------------------------------------------------------------------------------------------------------

def test_clerk_vendor_variants_map_to_one_shop():
    labels = ["Laxmi Auto Fuels"] * 3 + ["Auto Laxmi Fuels", "Laxmi Atuo Fuels", "Laxmi Auto Fuels Nagpur",
                                          "Om Sai Fuel Point", "Om Sai Fuel Point", "Om SaiF uel Point", "Sitabuldi M.L. Parking"]
    m = canonical_vendors(labels)
    assert {m[l] for l in labels[:6]} == {"Laxmi Auto Fuels"}
    assert m["Om SaiF uel Point"] == "Om Sai Fuel Point"
    assert m["Sitabuldi M.L. Parking"] == "Sitabuldi Multilevel Parking"   # hand alias, checked on the photo


# --- GSTIN --------------------------------------------------------------------------------------------------------------------

def test_valid_gstin_is_read_as_is():
    assert parse_gstin("GSTIN: 27DPAFU4120X1ZM") == ("27DPAFU4120X1ZM", "ok")


def test_ocr_letter_digit_confusion_is_fixed_only_when_checksum_passes():
    # '270DUFJ1789K3ZU': third character is a letter slot, so 0 -> O, and the checksum then passes
    assert parse_gstin("GSTIN: 270DUFJ1789K3ZU") == ("27ODUFJ1789K3ZU", "ok_normalised")
    assert gstin_is_valid("27ODUFJ1789K3ZU")


def test_invalid_gstin_is_never_corrected_into_a_valid_looking_one():
    value, status = parse_gstin("GSTIN: 27HJXCW997531ZJ")   # the clerk's typo of 27HJXCW9975D1ZJ
    assert status == "invalid" and value == "27HJXCW997531ZJ"
    assert parse_receipt("GSTIN: 27HJXCW997531ZJ\nTOTAL Rs. 10.00", {"names": [], "gstin": {}, "prefix": {}, "lines": {}, "shapes": {}})["gstin"] == ""


def test_receipt_without_gstin_is_absent():
    assert parse_gstin("HIGHWAY KING DHABA\nTotal Rs. 330") == ("", "absent")


# --- dates --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("line, iso", [
    ("Date: 08-06-2026", "2026-06-08"), ("Date: 09/06/26", "2026-06-09"), ("Date: 2026-06-13 Time: 18:14", "2026-06-13"),
    ("Date: 13-Jun-26", "2026-06-13"), ("Invoice Date: 08 Jun 2026", "2026-06-08"), ("Date: 15.06.2026", "2026-06-15"),
    ("Date: 9/7/2026", "2026-07-09"),
])
def test_date_formats_are_day_first(line, iso):
    assert parse_date(line)[0] == iso


def test_invoice_date_wins_over_check_in_and_out():
    text = "Check-in: 07 Jul 2026 13:30\nInvoice Date: 09 Jul 2026\nCheck-out: 09 Jul 2026 10:15"
    assert parse_date(text) == ("2026-07-09", "ok")


def test_missing_date_stays_blank():
    assert parse_date("Bill No. 1223\nTotal Rs. 330") == ("", "absent")


# --- receipt number -----------------------------------------------------------------------------------------------------------

def test_receipt_number_keeps_leading_zeros_and_ignores_vehicle_number():
    text = "Receipt No : 003203\nVehicle No : MH32BB1205"
    assert parse_receipt_number(text) == ("003203", "ok")


def test_receipt_number_stops_at_a_run_together_label():
    assert parse_receipt_number("Receipt No : 002147Date: 13/06/26")[0] == "002147"


def test_number_shape_learns_shop_formats():
    assert number_shape("006239") == "00dddd"
    assert number_shape("HCP/26-27/4594") == "HCP/dd-dd/dddd"


# --- amounts ------------------------------------------------------------------------------------------------------------------

def test_restaurant_total_is_verified_by_its_own_arithmetic():
    text = "Sub Total\n350.00\nCGST @2.5%\n8.75\nSGST @2.5%\n8.75\nRound Off\n+0.50\nGRAND TOTAL Rs\n368.00"
    assert parse_total(text) == ("368.00", "verified")


def test_total_printed_before_its_label_is_found():
    text = "Sub Total\n870.00\nCGST @2.5%\n21.75\nSGST @2.5%\n21.75\nRound Off\n+0.50\n914.00\nGRAND TOTAL Rs"
    assert parse_total(text) == ("914.00", "verified")


def test_amount_in_words_fixes_slash_dash_misread():
    assert words_to_number("One Thousand Two Hundred Thirty only") == 1230
    text = "Amount Rs. 12307-\nRupees (in words): One Thousand Two Hundred Thirty only"
    assert parse_total(text) == ("1230.00", "verified")


def test_handwritten_total_equals_sum_of_items():
    text = "Particulars\n1 Samosa\n2 Dal Fry\n1 Thali\nTotal Rs.\n20\n240\n150\n410"
    assert parse_total(text) == ("410.00", "verified")


def test_digits_contradicting_words_are_uncertain():
    text = "Amount Rs. 1940/-\nRupees (in words): Nine Hundred Forty only"
    assert parse_total(text)[1] == "uncertain"


def test_looped_output_never_has_a_trusted_amount():
    master = {"names": [], "gstin": {}, "prefix": {}, "lines": {}, "shapes": {}}
    p = parse_receipt("GRAND TOTAL Rs\n304.00\n" + "Revenue: 3,997,249\n" * 20, master)
    assert p["looped"] and p["total_status"] == "uncertain"


# --- vendor matching ----------------------------------------------------------------------------------------------------------

@pytest.fixture
def master():
    dev = pd.DataFrame({
        "canonical_vendor": ["Sitabuldi Multilevel Parking"] * 2 + ["Speedlink Couriers"] * 2 + ["Shree Sai Petroleum"] * 2,
        "clerk_gstin": ["", "", "27HJXCW9975D1ZJ", "27HJXCW9975D1ZJ", "", ""],
        "clerk_gstin_valid": ["", "", "True", "True", "", ""],
        "clerk_bill_no": ["108873", "110241", "SL92216340", "SL51865365", "006093", "006790"],
        "text": ["SITABULDI MULTILEVEL PARKING\nNagpur Municipal Corporation\nVeh No : MH31DX0981",
                 "Nagpur Municipal Corporation\nVeh No : MH31DX0981",
                 "Gandhibagh, Nagpur\nAWB No : SL92216340", "Gandhibagh, Nagpur\nAWB No : SL51865365",
                 "IndianOil\nSHREE SAI PETROLEUM\nReceipt No : 006093", "SHREE SAI PETROLEUM\nReceipt No : 006790"],
    })
    return build_vendor_master(dev)


def test_unseen_shop_is_not_forced_onto_a_known_one_via_vehicle_number(master):
    text = "MEDICAL SQUARE PAY & PARK\nMedical Square, Nagpur\nVeh No : MH31DX0981\nAmount Rs 20.00"
    assert parse_vendor(text, master) == ("Medical Square Pay & Park", "unknown")


def test_vendor_inferred_from_gstin_when_header_is_missing(master):
    assert parse_vendor("BOOKING RECEIPT\nAWB No : SL1", master, gstin="27HJXCW9975D1ZJ") == ("Speedlink Couriers", "inferred_gstin")


def test_vendor_inferred_from_signature_line(master):
    assert parse_vendor("Gandhibagh, Nagpur\nTOTAL Rs 339.84", master)[0] == "Speedlink Couriers"


def test_number_outside_the_shops_format_is_flagged(master):
    p = parse_receipt("SHREE SAI PETROLEUM\nReceipt No : 067012\nTOTAL Rs. 2949.80", master)
    assert p["number_status"] == "format_mismatch"


# --- combining methods --------------------------------------------------------------------------------------------------------

def _fields(**over):
    base = {"vendor": "", "vendor_status": "absent", "receipt_date": "", "date_status": "absent", "receipt_number": "",
            "number_status": "absent", "gstin": "", "gstin_read": "", "gstin_status": "absent", "total_amount": "",
            "total_status": "absent", "looped": False}
    return {**base, **over}


def test_fallback_fills_gaps_only_with_checked_values():
    vlm = _fields(total_amount="250.00", total_status="verified")
    tess = _fields(vendor="Gajanan Petroleum", vendor_status="ok", gstin_read="27ABCDE1234F1Z0", gstin_status="invalid",
                   total_amount="0.00", total_status="ok")
    out = combine(vlm, tess)
    assert out["vendor"] == "Gajanan Petroleum" and out["vendor_status"] == "ok_fallback"
    assert out["gstin"] == "" and out["gstin_status"] == "absent"          # invalid fallback GSTIN not taken
    assert out["total_amount"] == "250.00"                                  # primary kept


# --- benchmark ----------------------------------------------------------------------------------------------------------------

def test_wilson_interval():
    lo, hi = wilson(57, 60)
    assert 0.85 < lo < 0.87 and 0.98 < hi < 0.99


def test_adjudication_rows_are_consistent():
    adj = pd.DataFrame(ADJUDICATION, columns=ADJUDICATION_COLUMNS)
    assert not adj.duplicated(["claim_id", "field"]).any()
    assert ((adj.correct == "model") == (adj.clerk_value != adj.printed_value)).all()


@pytest.mark.skipif(not (ROOT / "reports" / "duplicate_images.csv").exists(), reason="audit not run")
def test_holdout_is_later_than_dev_and_never_splits_a_reused_photo():
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    train = read(ROOT / "data" / "processed" / "claims_train_clean.csv")
    audit, dup = read(ROOT / "reports" / "image_audit.csv"), read(ROOT / "reports" / "duplicate_images.csv")
    canon = canonical_vendors(train[train.clerk_vendor != ""].clerk_vendor.tolist())
    holdout, dev, _ = select_holdout(train, audit, canon, dup)
    assert len(holdout) == 60 and not set(holdout.claim_id) & set(dev.claim_id)
    for _, g in dup.groupby("duplicate_group"):
        ids = set(g.claim_id)
        assert not (ids & set(holdout.claim_id) and ids & set(dev.claim_id))
    assert holdout.submitted_on.min() >= dev[~dev.claim_id.isin(dup.claim_id)].submitted_on.max()


def test_pen_added_digit_in_front_of_total_is_detected():
    text = ("Sub Total\n490.00\nCGST @2.5%\n12.25\nSGST @2.5%\n12.25\nRound Off\n-0.50\nGRAND TOTAL Rs\n1514.00")
    assert parse_total(text) == ("514.00", "inconsistent")
    fuel = "Rate (Rs/L): 104.69\nVolume (L): 5.12\nAmount (Rs): 536.01\nTOTAL Rs. 1536.01"
    assert parse_total(fuel) == ("536.01", "inconsistent")


def test_round_off_out_of_place_does_not_fake_an_alteration():
    text = "Sub Total\n820.00\nCGST @2.5%\n20.50\nSGST @2.5%\n20.50\n+0.00\nRound Off\n861.00\nGRAND TOTAL Rs"
    assert parse_total(text) == ("861.00", "verified")


def test_tax_rate_misread_as_02_5_is_not_taken_as_tax():
    text = "Sub Total\n780.00\nCGST 02.5%\n19.50\nSGST 02.5%\n19.50\nRound Off\n+0.00\nGRAND TOTAL Rs\n819.00"
    assert parse_total(text) == ("819.00", "verified")
