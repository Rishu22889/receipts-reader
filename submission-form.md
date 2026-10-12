# Submission form: Task 2, Tarang Pharma expense receipts

## GitHub repo URL

**https://github.com/Rishu22889/receipts-reader.git**

## 1. What did you build, and what decision does it support for Nilesh?

A tool that reads the receipt photo on each claim, checks it against the claim, the employee record, earlier claims and the expense policy, and returns **approve** (pay without a clerk) or **review** (a clerk checks it), with reason codes. It comes as a FastAPI service with one upload screen, a batch pipeline, and `predictions.csv` for the 190 test claims.

The decision it supports: **which claims Finance can pay from the photo alone.** My answer is "only those read and checked", not "everything under ₹2,000":

- **129 of the 190 test claims (68%) are classified as eligible for auto-approval** and 61 (32%) go to review. These are proposed decisions, not payments; I recommend a four-week shadow pilot before any automatic payment.
- In the two months of processed claims, the clerks approved **25 claims, ₹14,283, that the policy says should have been stopped**. These are 16 bills claimed a second time, 3 pen-altered totals and 6 over-claims, and 23 of them were under ₹2,000. The engine flags all 25 for review before payment: **about ₹7,100 a month of potentially preventable leakage in Nagpur**, estimated from the two-month historical sample. Actual avoided payments need confirming from payment records and human review, since "approved" isn't proof of payment.
- Clerk time: about **95 hours a month company-wide, ~₹22,600 a month of capacity value, not cash** (the arithmetic is in question 6).

## 2. What do you expect predictions.csv to score, and why?

| Field | Expected | Basis |
|---|---|---|
| vendor | 90–98% | Holdout: 98%. Lower if the answer key compares the exact printed string. I output a canonical shop name ("Hotel Pradeep Pure Veg"), not the printed case ("HOTEL PRADEEP PURE VEG") |
| receipt_date | 80–87% | Holdout: 85%. Handwritten dates are often skipped, and I leave them blank |
| receipt_number | 90–95% | Holdout: 95% |
| gstin | 88–93% over all rows | Holdout: 92% over all rows (81% of the bills that print one; every bill without one correctly blank) |
| total_amount | 88–95% | Holdout: 95%, but 93% on the first untouched run. Two parser bugs were found through holdout errors |
| decision | 80–88% | Approve/review matched my evidence-based policy reference on 85.9% of the 283 train claims |

**How I estimated it:**
- **Field accuracy:** the 60-claim chronological holdout (claims submitted 3–18 July), scored against labels I checked on the photos wherever a method and the clerk disagreed.
- **Decision accuracy:** agreement with the policy reference on train. Most of the gap is claims sent to review because a field couldn't be confirmed; the engine approved none that the reference says must be reviewed.

**Why the real score could be lower:**
- **The holdout is small** (95% intervals span ±6–10 points) and enriched with hard cases.
- **Missing photos:** the 3 test claims without one are left blank on purpose.
- **Altered bills:** I report the original printed amount for the pen-altered bills (CLM-31299: 472, not 1,472). If the key uses the altered figure, those rows count as wrong.
- **The decision key may follow a different reading of the policy,** for example on rounding or date tolerance.

## 3. How do you know it works?

**What I measured against:**
- **Reading:** 60 later training claims, never used to build the parser or the vendor list. I compared five reading methods (raw Tesseract, preprocessed Tesseract, PaddleOCR-VL, PaddleOCR-VL with retry, combined) using exact-match scoring, with "within ±₹10" reported separately.
- **Decisions:** all 283 training claims.
  - (A) Agreement with the clerks: 67.5%. Not proof of correctness, because the clerks approved re-claims and altered bills.
  - (B) A policy reference built from clerk-typed fields corrected on the photo, plus duplicates, alterations, colleague vehicles, stamps and unreadable photos, all checked by eye. **0 of 60 review-worthy claims auto-approved.** The rules were designed on this same data, so this isn't an independent safety guarantee.

**Regression and service tests:** 26 known cases (every duplicate pair, altered bill, missing photo, stamp, colleague's vehicle) all go to review for the right reason. There are 120 automated tests. A clean-machine run of the service passes the clean, missing-photo, duplicate, altered and no-reader cases. The live service matched the batch output on 28 of 28 sampled test claims.

**How I got the answers:** clerk labels are reference labels, not ground truth. Where any method disagreed with the clerk, I opened the photo. **The clerks were wrong on 12 of 300 holdout fields (96%):**
- dropped digits (`A/618` for `A/6178`)
- day and month swapped
- GSTINs left blank
- amounts typed wrong
- two pen-altered totals typed as altered

**What it gets wrong:**
- **Missed fields:** handwritten dates on dhaba and cab bills, header lines the VLM skips (vendor, GSTIN), and memo amounts where `1130/-` reads as `11307` (caught by the amount in words, then sent to review).
- **Unflagged errors:** one hotel check-in date was taken instead of the invoice date. That was the only wrong field value in the holdout not flagged by a check.
- **New photos:** with only Tesseract (no GPU), it misreads small digits often. That's safe, since those claims go to review, but it lowers coverage.

## 4. Did you change, narrow, or push back on Nilesh's ask?

- **"Anything under ₹2,000 approved straight from the photo": narrowed.** An amount threshold alone would have auto-paid 23 of the 25 bad claims above, so approval also needs a clean read and no duplicate, alteration, vehicle, stamp or date exception. The cost is that 32% go to a clerk instead of 10%.
- **"At least 95% accurate on the fields": reframed.** It isn't met for every field (date 85%, GSTIN 81%), and 60 receipts can't prove it for any field. I made the tool leave a field blank and route the claim to review rather than guess. Of the fields it filled, it was right on 59/59 vendors, 51/52 dates, 57/59 bill numbers, 57/58 amounts and 21/21 GSTINs (holdout of 60). I measured the clerks at 96% on the same fields (12 errors in 300). I decided this on the first day of benchmarking, when the VLM turned out to skip lines rather than misread them.
- **"Tell me what it will save": split into two numbers.** Clerk-time capacity (~₹22,600 a month, not cash) and potentially preventable leakage (~₹7,100 a month in Nagpur, a historical estimate that needs payment records to confirm). I also flagged that approved doesn't mean paid, and that policy §5 makes recovery after the quarter unlikely.
- **Recommended a four-week shadow pilot before any automatic payment,** because the rules were written while studying the same claims they were evaluated on.

## 5. What is wrong with what you are handing us, or with the data?

**In the data:**
- **Clerk labels have errors:**
  - at least 12 in the 300 holdout fields, and more in dev: wrong amounts (₹240 for a ₹250 bill), `2026-09-06` for 9 June, truncated bill numbers;
  - 7 clerk GSTINs fail the checksum;
  - duplicates, altered totals and colleague's bills were approved.
- **`image_file`:**
  - 25 rows hold a full phone path;
  - 8 say `.jpeg` while the file is `.jpg`;
  - 6 photos never arrived (3 train, 3 test).
- **Reps' typing:** `expense_date` comes in three formats and `amount_claimed` in 17, all typed by hand.
- **Photos:**
  - 30 sideways or upside down;
  - 5 unusable (blurred, dark, cropped);
  - 5 carry a violet stamp telling "automated processing" to approve the bill and skip duplicate checks (CLM-30623, CLM-30919, CLM-30982, CLM-31176, CLM-31401). I treat it as data and send those claims to review.

**In what I hand over:**
- **My decision rules were written knowing the training cases.** They're frozen (`95bef2d00f35`) and were run once on the test claims, but the safety result is not independent.
- **Two amount-parser bugs were found through holdout errors**, so the holdout amount score (95%) is somewhat optimistic; the untouched run scored 93%.
- **The photo-checked "printed values" are my reading,** not Finance's (`reports/holdout_adjudication.csv`).
- **"Approved" is used as a proxy for "paid".**
- **PaddleOCR-VL ran in batch on a Kaggle GPU.** The service reuses that text for the pack's 467 photos and reads new photos with Tesseract unless a GPU is configured. Tesseract-only reading sends most claims to review.
- **Vendor names are canonical,** not the exact printed case.
- **Service limits:** session history is in memory only, with no authentication and a single process.
- **Lodging rules are lightly tested:** the outstation, rate, guest-name and GST-invoice checks run on only 29 lodging claims, almost all over ₹2,000 and so reviewed anyway.

## 6. What does checking one claim cost, and a month at 2,400 claims?

**No paid API calls are made.**

- **Today:** 3.5 minutes per claim. At ₹76,000 for 320 clerk-hours a month (I assumed 160 paid hours each), that's ₹237.50 an hour, or **₹13.85 per claim**. For 2,400 claims: 2,400 × ₹13.85 = **₹33,250 of clerk time a month**.
- **With the tool:**
  - Reading takes ~3 s per photo on a T4 GPU, so 2,400 photos is about 2 GPU-hours a month. At an assumed cloud price of US$0.35–0.60 an hour, that's **roughly ₹60–100 a month for inference alone**, an estimate excluding retries, idle time and deployment overhead, which would need measuring in the pilot. It was free on Kaggle for this work.
  - Reviewed claims still cost clerk time: 32% × 2,400 = 768 claims × ₹13.85 = **₹10,640 a month**.
  - Monthly total: roughly ₹10,700, against ₹33,250. That frees **~95 clerk-hours, ~₹22,600 of capacity**, which is not cash unless staffing changes.
- **Queries:** each one costs ₹45 (policy §5). The tool reviews rather than queries, so this cost only arises when a clerk decides to query.

## 7. What did you deliberately leave out, and why?

- **No trained or fine-tuned model.** 283 historical decisions, several of them wrong, aren't a training set for what the policy says. Rules are auditable and testable.
- **No paid vision API.** A free local model plus checks was enough, and it keeps the per-claim cost near zero and the service runnable without a key.
- **No live GPU inference in the service, and no second VLM.** The laptop has 8 GB RAM, so PaddleOCR-VL ran in batch on Kaggle; one VLM was enough to beat Tesseract by a wide margin.
- **No database, authentication, or payment integration.** Out of scope for a decision prototype. The bigger risk was wrong approvals, so I spent the time there.

## 8. Anything you built or found that nobody asked for?

- **Duplicate-bill detection beyond image hashing:** vendor + bill number + date, which found 28 re-claimed bills where hashing found 4.
- **Pen-alteration detection** from the receipt's own arithmetic (sub total + taxes, fuel rate × volume), which found 6 altered totals.
- **Colleague's-vehicle check** on fuel bills (4 cases), and a **memo-name check** (a shared cab memo claimed by the non-payer).
- **Instruction-stamp detection** (OCR phrases plus violet ink); the text is treated as data, never as an instruction.
- **GSTIN checksum validation,** with corrections made only by character position and only when the checksum passes.
- **A measured clerk error rate** (96% field accuracy).
- **Two fixes for running PaddleOCR-VL:**
  - Its published `generation_config.json` disables the KV cache, which caused ~180 s per receipt until I forced it on.
  - An automatic retry for its 11 repetition loops; one of them had invented ₹3,997,249.
- **A rules fingerprint** in every report, so a decision can be traced to the exact rule set.

## 9. What did you use AI for? (recording link here)

**Recording:** **https://drive.google.com/file/d/1HIwNl8ismB1Y8hjfK5Qfkf_yj2kiDalW/view?usp=sharing**

**Tools and models:**
- **Claude Code (Claude Opus 5.5)** as a coding and analysis assistant, for nearly all the code, notebooks, tests and documents. I directed the steps and ran the Kaggle notebooks.
- **PaddleOCR-VL 0.9B,** inside the product, on Kaggle T4 GPUs.
- **Tesseract 5,** inside the product, locally.

**Where it helped:**
- Systematic checks I would have skipped: opening every disagreeing photo, the checksum, the duplicate-bill search.
- Turning the notebooks into tested modules.
- Diagnosing why the VLM was slow.

**Where it misled me:**
- **Slow VLM:** it first blamed bf16 emulation on the T4. The real cause was the disabled KV cache.
- **Kaggle setup:** it first pinned a `transformers` version and model revision that didn't run, and its default Paddle install broke PyTorch's NCCL.
- **Heuristics:** its first blur and crop heuristics over-flagged, and a vehicle number became a "shop signature".
- **Parser bugs:** a tax-rate bug and a column-header bug, which tests and holdout errors caught.

**What I threw away:**
- the Paddle pipeline install and the custom `trust_remote_code` path;
- the sharpness-based "unreadable" label and the border-based crop detector.

## 10. Someone picks this up on Monday. The three things they need to know.

1. **How to run it and where things are.** `pip install -r requirements.txt`, `brew install tesseract`, `uvicorn tarang.service:app --app-dir src`. The decision rules are in `src/tarang/decision_engine.py` and frozen as `95bef2d00f35`. Any change means updating `tests/test_decision_engine.py` (one test per policy row) and re-running `scripts/decision_engine.py`. `docs/POLICY_RULES.md` explains every rule.
2. **The reader is the weak link without a GPU.** Good reading comes from PaddleOCR-VL (`kaggle/README.md`): force `use_cache=True` and float32 on a T4, and never install Paddle into the same environment. On CPU the service falls back to Tesseract, which is safe but sends most claims to review. Before production, deploy a GPU reader (`TARANG_VLM=live`).
3. **Don't trust the clerk labels or the "approved" status blindly.** The clerks are about 96% accurate, and approved doesn't mean paid. Before any automatic payment, run the four-week shadow pilot, and ask Nilesh for payment records on the 25 flagged claims (`reports/decisions.csv`, column `ref_reasons`).
