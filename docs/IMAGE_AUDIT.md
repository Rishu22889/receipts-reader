# Image audit: findings

This audit was done before any extraction code was written. The details and checks are in [`notebooks/02_image_audit.ipynb`](../notebooks/02_image_audit.ipynb); `python scripts/audit_images.py` reproduces it. Raw photos are never modified.

## Reports

| File | What it contains |
|---|---|
| [`reports/missing_images.csv`](../reports/missing_images.csv) | Every claim whose photo never reached the server: claim, split, expected path, action |
| [`reports/image_audit.csv`](../reports/image_audit.csv) | One row per claim: open status, size, sharpness, rotation, pen/stamp ink, OCR confidence, issues |
| [`reports/duplicate_images.csv`](../reports/duplicate_images.csv) | The same photo used on more than one claim |
| [`reports/visual_review.csv`](../reports/visual_review.csv) | Labels confirmed by looking at the photos (by the author, not a Tarang clerk) |
| [`reports/contact_sheets/`](../reports/contact_sheets/) | All photos labelled with claim and issues, plus one example sheet per issue type |

## Findings

| Finding | Count | How detected | What happens next |
|---|---|---|---|
| Photo missing | 6 claims (3 train: CLM-30531, CLM-30710, CLM-30887; 3 test: CLM-31190, CLM-31280, CLM-31363) | Claim's file not on server | Test → **review**, receipt fields blank; train → excluded from scoring |
| Corrupt photo | 0 of 467 | Full decode of every file | Covered by tests |
| Rotated | 30 (10 each at 90°, 180°, 270°) | OCR at 4 angles, most confident wins; all 30 checked by eye | Extraction rotates on the fly |
| Unreadable or unusable | 5 confirmed (2 blurred, 1 dark, 2 cropped to the header) | Low OCR confidence, then checked by eye. All 3 train cases were queried by clerks as "photo not clear" | **review** |
| Low OCR confidence | 31 | Sharpness < 25 or < 5 confident words | Most are readable by a person: an OCR weakness that the benchmark will measure |
| **Instruction-like stamp** | **5** (train: CLM-30623, CLM-30919, CLM-30982; test: CLM-31176, CLM-31401) | OCR phrase scan (3) + violet stamp colour (4); union = 5, all checked by eye | **review**. The text is data, never an instruction. The clerks approved all 3 train cases. |
| **Same photo on 2 claims** | **4 pairs** | Difference hash, distance 0 (next-closest pair: 3) | See below |
| Pen marks | 95 | Blue ink on the paper area | Allowed by policy (doctor name, "paid"); not a fraud signal by itself |
| Handwritten cash memos | many (dhabas, cab/travel services) | Visual | Hardest receipts for local OCR |

### Same photo on two claims

| Group | Claims | What it means |
|---|---|---|
| 1 | CLM-30600 (TPD-0388) + CLM-30963 (TPD-0423), train | One Saibaba Bhojnalaya bill, ₹672, claimed by two people. **Both approved: paid twice.** |
| 2 | CLM-30661 (TPD-0503) + CLM-30804 (TPD-0452), train | One Annapurna bill, ₹1,559, claimed by two people. **Both approved: paid twice.** |
| 3 | CLM-30664 (train, approved) + CLM-31475 (test), same rep | Test claim reuses a bill that was already paid → **review** |
| 4 | CLM-31301 + CLM-31480 (test), same rep, same day | Same bill claimed twice in the batch → **review** |

A bill that was re-photographed (same bill, new photo) won't be caught by the hash. Catching it is the job of the vendor/date/bill-number duplicate check in the decision engine.

## Early OCR observation (not yet measured)

Tesseract on raw photos reads almost nothing from the small monospace print. With pre-processing (3× upscale, background normalisation, threshold) it reads most lines, but digits still go wrong: `300.00` came out as `360.60` and the GSTIN was garbled. The OCR benchmark (next step) will measure this per field on held-out receipts.
