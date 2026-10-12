# Data notes: what we cleaned and why

The brief says "where something is unclear, decide, and write down what you decided and why." This file is that record for the data.
The worked analysis, with outputs, is in [`notebooks/01_data_cleaning.ipynb`](../notebooks/01_data_cleaning.ipynb); the same logic runs as `python scripts/clean_data.py`.

**Rule:** `data/raw/` is kept exactly as received (checked with checksums). Cleaned copies go to `data/processed/`, and every changed column keeps its original value in a `*_raw` column.

## Fixed: mechanical, no judgement needed

| Field | Problem | Fix | Rows |
|---|---|---|---|
| `expense_date` | Three formats: `DD-MM-YYYY`, `YYYY-MM-DD`, `DD/MM/YY` | Parsed to ISO `YYYY-MM-DD`. All formats are day-first: 197 values have a day > 12 and none has a month > 12. An unknown format raises an error rather than being guessed. | 320 |
| `amount_claimed` | `Rs `, `Rs. ` prefixes and thousands commas (17 different shapes) | Converted to a number with 2 decimals | 168 |
| `image_file` | Full phone path (`/storage/emulated/0/DCIM/...`) or `.jpeg` instead of `.jpg` | Matched to the file on disk by stem, case-insensitive (stems are unique) | 33 |
| `clerk_bill_date` | `DD-MM-YYYY` | Converted to ISO, to match the submission format | 277 |

Sanity checks that passed with no fix needed: claim IDs are unique, every `emp_id` is in `employees.csv`, there is no leading or trailing whitespace, and `submitted_on` is always ISO. No expense date falls after its submission date.

## Flagged: kept as-is, marked for later steps

| Flag | What | Decision |
|---|---|---|
| `photo_available = False` | 6 claims whose photo never reached the server. **Train (3): CLM-30531, CLM-30710, CLM-30887.** **Test (3): CLM-31190, CLM-31280, CLM-31363.** All 3 train cases were queried as "bill not attached", which confirms the photos are missing and not a matching error. Full list with expected paths: [`reports/missing_images.csv`](../reports/missing_images.csv). | Keep the rows. The 3 test claims get decision **review**. Their receipt fields stay **blank**: nothing is invented, and the rep's typed vendor/amount are *not* copied in as if they were read from a receipt. |
| Blank clerk fields | 6 train rows: the 3 above plus 3 "photo not clear" (CLM-30856, CLM-30886, CLM-30979) | Blank means "not transcribed", not zero. **Exclude from field-accuracy scoring.** |
| `clerk_gstin_valid = False` | 7 of 112 clerk-typed GSTINs fail the official GSTIN checksum. Example: CLM-30437's receipt prints `27HJXCW9975D1ZJ`, but the clerk typed `27HJXCW997531ZJ`. Some of these were marked "verified". | **Not corrected** (correcting them would mean guessing). The training labels are not perfect ground truth, so exclude these rows when scoring GSTIN accuracy. This also gives a measured clerk error rate to put in the memo. |

### GSTIN reference coverage (so the accuracy figure can't look better than it is)

| Train claims | Count |
|---|---|
| All train claims | 283 |
| Transcribed by a clerk (others have no readable photo) | 277 |
| GSTIN reference blank (bill should have no GSTIN) | 165 |
| GSTIN reference present, checksum valid | 105 |
| GSTIN reference present, checksum **fails**: **excluded** | 7 |
| **Scoreable GSTIN references** | **270 of 277 (97.5%)**; 105 of 112 present GSTINs (93.8%) |

Every GSTIN accuracy figure we report states the excluded count next to it. A blank reference can't be checksum-verified: if a clerk skipped a GSTIN that is printed on the bill, a model that reads it correctly is scored "wrong". The image audit and error review look for these cases instead of assuming blank means no GSTIN.

`clerk_bill_no` is kept as text: values like `003203` and `HGE/26-27/7471` are identifiers, and their leading zeros matter.

## Left for later steps (not data cleaning)

- **Photos** (rotation, blur, pen marks, instruction-like stamps, the same photo on two claims): see [IMAGE_AUDIT.md](IMAGE_AUDIT.md).
- **Duplicate bills that are not the same photo** (e.g. CLM-31079 "same bill as CLM-30679", and Priya's Hotel Shubham case) and the **daily Rs 30 parking** pattern: these belong to the decision logic.
- **Late claims** (up to 58 days after the expense): a policy check, not a data error.
