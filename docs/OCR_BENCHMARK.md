# OCR benchmark: Tesseract vs PaddleOCR-VL

Full method, code and results: [`notebooks/03_ocr_benchmark.ipynb`](../notebooks/03_ocr_benchmark.ipynb). Reproduce with `python scripts/run_benchmark.py`.

## Setup

- **Held-out set:** 60 training claims submitted 3–18 July, the latest part of the data. It includes all 16 hard cases from that period (8 rotated, 6 low OCR confidence, 2 stamped), plus 14 handwritten bills and 17 with pen marks. The parser rules and the vendor list were built from the 168 earlier ("dev") claims only. A photo reused on two claims never sits on both sides.
- **Same parser for every method.** The OCR or VLM only reads the text. One Python parser turns that text into fields, and each field gets its own status.
- **Reference labels:** where a method disagreed with the clerk, the photo was checked ([`reports/holdout_adjudication.csv`](../reports/holdout_adjudication.csv), 18 fields). The clerk was wrong on **12 of the 300 held-out fields**: dropped digits, a swapped day and month, GSTINs left blank, wrong amounts, and two totals altered in pen and typed as altered (one approved, one rejected). For an altered bill, the reference is the original printed amount. Scores are reported against both the raw clerk labels and the adjudicated ones. The table below uses the adjudicated labels.

## Results (n = 60, adjudicated labels, exact match)

| Method | Vendor | Date | Bill no. | GSTIN (n = 26) | Amount | All 5 right | Median time per photo |
|---|---|---|---|---|---|---|---|
| Tesseract, raw | 60.0% | 16.7% | 25.0% | 19.2% | 16.7% | 1.7% | 0.2 s (M1 CPU) |
| Tesseract + preprocessing | 91.7% | 31.7% | 61.7% | 23.1% | 60.0% | 10.0% | 1.0 s (M1 CPU) |
| PaddleOCR-VL 0.9B | 78.3% | 85.0% | 93.3% | 76.9% | 93.3% | 60.0% | 2.8 s (T4 GPU) |
| PaddleOCR-VL + retry for loops | 78.3% | 85.0% | 93.3% | 76.9% | 95.0% | 61.7% | 2.8 s (T4 GPU) |
| **Combined** (VLM, gaps filled by checked Tesseract values) | **98.3%** | **85.0%** | **95.0%** | **80.8%** | **95.0%** | **71.7%** | both |

- "Within ±₹10" is reported separately in `reports/ocr_benchmark.csv` and is never counted as correct.
- For GSTIN, all 34 bills with no GSTIN were correctly left blank by every method.
- With n = 60, the 95% intervals are wide: combined amount 86–98%, bill number 86–98%.
- **The amount figure is somewhat optimistic.** The amount parser was revised during the decision-engine work, and two of the bugs fixed were found through held-out errors (CLM-30829, CLM-31010). The first, untouched run scored 93.3%.

## What it means

- **Errors are mostly silence, not wrong answers.** Of the combined method's 21 wrong fields, 17 are blanks. Of the 4 wrong values, 3 are caught by its own checks: the bill number doesn't fit the shop's format, or the amount doesn't match the receipt's arithmetic. Only **one wrong value passed every check**: a hotel's check-in date taken in place of its invoice date.
- **Coverage and precision are separate.** The combined method filled vendor on 59/60 receipts, date 52/60, bill number 59/60, amount 58/60 and GSTIN 21/60 (all 34 without one correctly blank). Of the fields it filled, it was right on 59/59 vendors, 51/52 dates, 57/59 bill numbers, 57/58 amounts and 21/21 GSTINs. Those counts include every filled value, whether or not its own check passed; 3 of the 4 wrong values carried a failing status. So the plan is to auto-approve only claims whose fields are read *and* checked, and send everything else to a clerk.
- **The 95% target is not met for every field on every receipt**, mostly because of blanks, and n = 60 can't prove 95% for any field. The clerks themselves scored about 96% on the same fields.

## Failure modes found

| Problem | Effect | What handles it |
|---|---|---|
| The VLM skips the big header line | Vendor missing | Shop recognised from its GSTIN, bill-number prefix or address/phone line, or from Tesseract, which reads headers well |
| The VLM loops: repeats a line until the 768-token cap | 11 of 467 photos; one invented `₹3,997,249` | Automatic retry with `repetition_penalty=1.05` fixed all 11; a looped amount is never trusted |
| `1130/-` read as `11307` | Amount 10× too high | Checked against the amount in words |
| O/0 and I/1 confusion in GSTINs | Invalid GSTIN | Corrected by character position, accepted only if the checksum passes |
| Handwritten dates on dhaba bills | Date missing | Left blank, so the claim goes to review |
| Thermal-print digits for Tesseract | `300.00` read as `360.60` | Tesseract is never trusted for amounts unless verified |
| A digit added in front of the total by pen (`472` → `1472`) | Inflated total | Total compared with the bill's own arithmetic (sub total + taxes, or fuel rate × volume); a mismatch is marked `inconsistent` and sent to review |

## Running the VLM

PaddleOCR-VL needs a GPU: on the 8 GB laptop, loading it pushed memory to 9.5 GB of swap. It ran on a free Kaggle T4; see [`kaggle/README.md`](../kaggle/README.md). Two problems found and fixed along the way:
- **No KV cache:** the model's published `generation_config.json` sets `"use_cache": false`, which made it ~180 s per receipt.
- **Emulated bf16:** the T4 reports bf16 support but only emulates it.

After both fixes it ran at 2.8 s per photo.
