from pathlib import Path

import pandas as pd
import pytest

from tarang.cleaning import gstin_is_valid, parse_amount, parse_date, resolve_image, run

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


@pytest.mark.parametrize("value, expected", [
    ("08-06-2026", "2026-06-08"),
    ("2026-06-13", "2026-06-13"),
    ("09/06/26", "2026-06-09"),
    (" 14-06-2026 ", "2026-06-14"),
])
def test_parse_date_formats(value, expected):
    assert parse_date(value) == pd.Timestamp(expected)


def test_parse_date_blank_and_unknown():
    assert pd.isna(parse_date(""))
    with pytest.raises(ValueError):
        parse_date("June 8th")


@pytest.mark.parametrize("value, expected", [
    ("330", 330.0), ("250.00", 250.0), ("Rs 2352", 2352.0),
    ("Rs. 1,234", 1234.0), ("3,539.00", 3539.0), ("330.4", 330.4),
])
def test_parse_amount(value, expected):
    assert parse_amount(value) == expected


def test_parse_amount_blank_and_bad():
    assert parse_amount("") is None
    with pytest.raises(ValueError):
        parse_amount("three hundred")


def test_gstin_checksum():
    assert gstin_is_valid("27HJXCW9975D1ZJ")      # as printed on the Speedlink receipt
    assert not gstin_is_valid("27HJXCW997531ZJ")  # as typed by the clerk (5D -> 53)
    assert not gstin_is_valid("27ODUFJ1Q89K3ZU")  # wrong structure
    assert not gstin_is_valid("")


def test_resolve_image():
    index = {"img_20260614_185526": "IMG_20260614_185526.jpg"}
    assert resolve_image("IMG_20260614_185526.jpeg", index) == "IMG_20260614_185526.jpg"
    assert resolve_image("/storage/emulated/0/DCIM/Camera/IMG_20260614_185526.jpg", index) == "IMG_20260614_185526.jpg"
    assert resolve_image("IMG_20990101_000000.jpg", index) == ""


@pytest.mark.skipif(not (RAW / "images").exists(), reason="photos not downloaded")
def test_full_run(tmp_path):
    train, test = run(RAW, tmp_path)
    assert (len(train), len(test)) == (283, 190)
    assert (~train.photo_available).sum() == 3 and (~test.photo_available).sum() == 3
    assert (train.clerk_gstin_valid == False).sum() == 7  # noqa: E712
    assert train.expense_date.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    assert train.clerk_bill_no.str.startswith("0").any()  # leading zeros kept
