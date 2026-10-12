# Memo: reading receipt photos and auto-approving claims

**To:** Nilesh Agrawal, Finance Controller · **Evidence:** two months of Nagpur claims (283 checked by clerks, 190 new)

## Recommendation

**Auto-approve a claim only when the receipt is readable and every blocking check passes. Do not automatically approve every claim under ₹2,000.**

On the 190 new claims, the tool classified **129 (68%) as eligible for auto-approval** and sent **61 (32%) to review**, each with a reason. These are proposed decisions, not payments made.

**Run a four-week shadow pilot in Nagpur first.** The tool makes its decisions while clerks keep checking every claim. We compare its decisions with independently verified receipt evidence and your policy rulings before switching on automatic payment.

## Why not "everything under ₹2,000"

Among the claims your clerks already processed, I found **25 approved claims, ₹14,283, that the policy says should have been stopped** (each counted once):

| What happened | Claims | ₹ |
|---|---|---|
| Bill claimed a second time, mostly from a new photo (includes Priya's Hotel Shubham case; 3 also show a colleague's vehicle) | 16 | 9,963 |
| Total altered in pen: a "1" added in front, e.g. ₹514 → ₹1,514 (excess) | 3 | 2,500 |
| Claimed more than the bill (excess) | 6 | 1,820 |

**23 of the 25 were under ₹2,000, so the draft rule would have paid them automatically.** The engine flags all 25 for review before payment. That's about **₹7,100 a month of potentially preventable leakage in Nagpur**, based on the two-month sample. Actual avoided payments need confirming from payment records and the outcome of human review; "approved" isn't proof of payment.

## What it saves (estimates)

- **Clerk time:**
  - 2,400 claims × 68% = 1,632 claims a month.
  - × 3.5 minutes = 95 hours.
  - × ₹237.50 an hour (₹76,000 / 320 h, *assuming 160 paid hours per clerk; please confirm*) = **~₹22,600 a month of capacity value, not cash savings.**
- **Leakage:** ~₹7,100 a month potentially preventable in Nagpur. Company-wide would be about ten times that *only if* other divisions look like Nagpur.
- **Running cost:** no paid API. About 2 GPU-hours of inference a month (~3 s per photo). Retries, idle time and deployment overhead aren't included.

## Accuracy against your 95% target

Measured on 60 later claims, as exact match against the printed value (photo-checked where the reader and clerk disagreed):

| Vendor | Bill no. | Amount | Date | GSTIN |
|---|---|---|---|---|
| 98.3% (n = 60) | 95.0% (n = 60) | 95.0%* (n = 60) | 85.0% (n = 60) | 80.8% (n = 26 that print one) |

*Amount scored **93.3% on the first run**. The fixes came after seeing errors on these same claims, so treat ~93% as the untouched estimate.

- **Dates and GSTINs miss the target.** Most misses are blanks (a skipped handwritten date or GSTIN line). Those claims go to a clerk; the tool doesn't guess.
- **Filled fields were right:** 59/59 vendors, 51/52 dates, 57/59 bill numbers, 57/58 amounts, 21/21 GSTINs. All 34 bills without a GSTIN were correctly left blank.
- **Your clerks' typing was 96% accurate** on the same claims (12 errors in 300 fields).

**Not a safety guarantee:** the tool auto-approved none of the 60 past claims that needed a person, but its rules were written while studying those claims. Hence the pilot is required before automatic payment.

## What I need from you

1. **Payment records** for the 25 flagged claims.
2. **Rulings**, applied provisionally until you decide:
   - Is rounding to the rupee acceptable, as well as to ₹10?
   - May a rep claim on a date other than the bill date?
   - Must a hotel invoice name the employee, or is the company enough?
3. **A GPU for the reader.** Without one, new photos are read less reliably and more claims go to review.
4. **Approval for the four-week shadow pilot.**
