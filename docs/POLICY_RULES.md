# Decision engine: policy rules and results

Code: [`src/tarang/decision_engine.py`](../src/tarang/decision_engine.py). Evidence and evaluation: [`notebooks/04_decision_engine.ipynb`](../notebooks/04_decision_engine.ipynb). Reproduce with `python scripts/decision_engine.py`. Rules version: `95bef2d00f35` (a fingerprint of the frozen rules, printed into every report).

**How it decides:** the OCR/VLM only reads the receipt; plain Python rules decide. A claim is **approved only if every blocking check passes**. Otherwise it goes to **review**, with every failing reason code. "P2", "P3" and so on refer to sections of `expense-policy.pdf` v3.2.

## Policy matrix

| Policy | Condition | Action | Reason code |
|---|---|---|---|
| P3 | No photo, or nothing readable on it | review | `MISSING_IMAGE`, `UNREADABLE_IMAGE` |
| P7 | Claim or bill is ₹2,000 or more | review | `AMOUNT_OVER_LIMIT` |
| P3 | Bill total not read, or contradicted by the receipt itself | review | `AMOUNT_UNVERIFIED` |
| P3 | Claimed amount ≠ bill, beyond rounding (to the rupee or the nearest ₹10) | review | `AMOUNT_MISMATCH` |
| P3, P8 | Printed total contradicts the bill's own arithmetic (fuel rate × volume, or sub total + taxes) | review | `ALTERED_BILL` |
| P3, P8 | Same vendor and bill number claimed before, by anyone, even from a new photo | review | `DUPLICATE_BILL` |
| P3, P8 | Same photo used on another claim | review | `DUPLICATE_PHOTO` |
| P3, P8 | Bill number unreadable, but the same vendor, date and amount were claimed before | review | `POSSIBLE_DUPLICATE` |
| — | Text on the bill addressed to "automated processing" | review | `INSTRUCTION_TEXT` |
| P4 | Vendor, bill date or bill number not read | review | `FIELDS_INCOMPLETE` |
| P4 | Bill number doesn't fit the shop's numbering format | review | `NUMBER_FORMAT_MISMATCH` |
| P4 | Bill date more than a day from the claimed expense date, or later than the submission date | review | `DATE_MISMATCH` |
| P4 | Vendor on the bill isn't the one the rep typed | review | `VENDOR_MISMATCH` |
| P2 | Fuel bill without a vehicle number | review | `FUEL_VEHICLE_MISSING` |
| P2, P8 | Fuel bill showing another employee's vehicle | review | `FUEL_VEHICLE_OTHER_EMPLOYEE` |
| P2 | Fuel bill showing an unknown vehicle (one OCR character of difference is tolerated) | review | `FUEL_VEHICLE_MISMATCH` |
| P3 | Cab memo made out to someone else | review | `MEMO_NAME_MISMATCH` |
| P2 | Lodging in the employee's headquarter city | review | `LODGING_NOT_OUTSTATION` |
| P2 | Room rate per night above the grade limit before GST (M1/M2 ₹3,000; M3 ₹4,500) | review | `LODGING_OVER_LIMIT` |
| P2 | Hotel invoice not in the employee's name | review | `LODGING_NAME_MISMATCH` |
| P2 | Hotel bill without a valid GSTIN (not a GST tax invoice) | review | `LODGING_NOT_GST_INVOICE` |
| P3 | Pen notes (doctor's name, number of people, "paid"); daily ₹30 parking | no effect | — |
| — | GSTIN absent or unreadable on a non-hotel bill | noted, never blocks | `GSTIN_ABSENT`, `GSTIN_UNREADABLE` |
| P7 | Everything passes and the claim is under ₹2,000 | **approve** | — |

Each row has a matching test in [`tests/test_decision_engine.py`](../tests/test_decision_engine.py). A meta-test fails if any row lacks one.

**A missing GSTIN never causes a review on its own.** Dhabas, cab memos and parking tickets don't print one. GSTIN has three separate states: *absent*, *unreadable/invalid* (noted, never blocks), and *valid* (required only for lodging).

## What the data showed (all checked on the photos)

- **Re-claimed bills:** 28 bills appear on more than one claim, mostly re-photographed, so image hashing finds only 4 of them. The clerks approved most of them twice and caught one.
- **Pen-altered totals:** 6 bills had a digit added in front of the printed total. In train: CLM-30919 (₹300 → ₹800), CLM-31009 (₹536.01 → ₹1,536.01), CLM-30568 (₹514 → ₹1,514), CLM-30764 (₹268 → ₹1,268); in test: CLM-31299 and CLM-31335. **Three of the four in train were approved at the altered amount.**
- **Colleague's vehicle on fuel bills:** CLM-30954, CLM-30978, CLM-31023 (all approved) and CLM-31120 (test).
- **A shared cab memo claimed by the non-payer:** CLM-31497 (test).
- **The rep's date as a signal:** almost every claim dated more than a day from its bill is a re-claim of an old bill.

## Results

### Evaluation A: agreement with the clerks (train, 283 claims)

The engine agrees with the clerks on **67.5%** of claims. Of the 8 claims the clerks queried or rejected, none is approved. Most of the disagreement is by design: claims of ₹2,000 or more always go to a person, and the engine flags re-claims and over-claims that the clerks approved. This is a workflow comparison, not a measure of correctness.

### Evaluation B: policy correctness (train, 283 claims)

The reference outcome comes from evidence the engine doesn't produce: the clerk-typed fields corrected where checked on the photo, the duplicates implied by those clerk fields, and the alterations, vehicles, unreadable photos and stamps checked by eye.

| Metric | Result |
|---|---|
| Auto-approval coverage | 64.7% of train claims; **67.9% of test claims** (129 of 190) |
| **Unsafe auto-approvals** (engine approves, reference says review) | **0 of 60** review-worthy claims |
| …against the clerk labels as typed | 4. All four are clerk typing errors, confirmed on the photo |
| Extra reviews (reference clean, engine reviews) | 40 of 283 (14%), mostly fields the reader couldn't confirm |
| Exceptions caught | missing photo 3/3, unreadable 3/3, re-claimed bill 16/16, same photo 2/2, altered 4/4, amount mismatch 14/14, colleague's vehicle 3/3, stamp 3/3, date mismatch 8/8 |
| Known cases sent to review | 26 of 26 |
| Decision time | 0.7 ms per claim, plus ~3 s per photo to read it on a T4 GPU |

### Limitations

- **The rules were written knowing these training cases.** Evaluation B therefore mostly tests whether reading errors cause unsafe approvals. It doesn't independently test the rules themselves. The rules are frozen (`95bef2d00f35`) and the 190 test claims were run once.
- **"Approved" stands in for "paid".** The data has no payment record.
- **The reference corrections are my own reading of the photos**, not Finance's. They are listed in the notebook and in `reports/holdout_adjudication.csv`.
- **The 60-claim OCR holdout was enriched with hard cases** and partly investigated by hand. It is reported only as a subset.

## Money

- **Leakage the engine flags.** In two months the Nagpur clerks approved **25 claims, ₹14,283**, that the policy says should have been stopped, each counted once: 16 bills claimed a second time (₹9,963; three of them also show a colleague's vehicle), 3 pen-altered totals (₹2,500 excess) and 6 over-claims (₹1,820 excess). The engine flags all 25 for review before payment: **about ₹7,100 a month of potentially preventable leakage in this division**, estimated from the historical sample. Actual avoided payments need payment records and human review to confirm, since approved isn't proof of payment.
- **Clerk time.** At the 68% of test claims classified as eligible for auto-approval, 3.5 minutes per claim, and an *assumed* 160 paid hours per clerk per month (₹76,000 / 320 h = ₹237.50 per hour):

  | Scope | Claims auto-approved per month | Clerk hours freed per month | Capacity value |
  |---|---|---|---|
  | Nagpur division (~240 claims/month) | ~163 | 9.5 | ~₹2,260/month |
  | Company (~2,400 claims/month) | ~1,630 | 95 | ~₹22,600/month |

  These figures are capacity, not cash. The P7 draft's 90% would mean 126 h and ₹29,925 company-wide, but it would also auto-pay every small re-claim and altered bill.
- **Running cost (estimate):** ~3 s per receipt on a T4 GPU, about 2 GPU-hours of inference for 2,400 receipts a month. Retries, idle time and deployment overhead aren't included.

**Provisional interpretations, pending Finance's ruling:** rounding to the rupee is accepted as well as to ₹10; a bill dated more than a day from the claimed expense date goes to review (any night of a hotel stay is accepted); a hotel invoice must name the employee. These are applied as reasonable readings of the policy, not confirmed company policy. The four-week shadow pilot is a precondition for automatic payment, because the 0-of-60 result comes from the same claims the rules were written on.
