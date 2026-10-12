# Tarang Pharma: expense receipt reader

This project reads the receipt photo on each of Tarang Pharma's field expense claims, checks it against the claim and the expense policy, and decides whether the claim can be **approved without a clerk opening it** or must be **reviewed**. Every decision comes with reason codes. The brief is in [`docs/problem_statement.txt`](docs/problem_statement.txt). The recommendation to Finance is in [`MEMO.md`](MEMO.md).

## Start the service (clean machine, ~5 minutes)

Requirements: Python 3.11 or newer, and the Tesseract binary.

```bash
git clone <this repo> && cd <repo>
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract                       # Ubuntu: sudo apt install tesseract-ocr
uvicorn tarang.service:app --app-dir src     # then open http://127.0.0.1:8000
```

The screen has buttons that load real claims from the data: a clean receipt, a duplicate, an altered bill, a missing photo, an "approve it" stamp, and a colleague's vehicle. You can also upload any receipt photo with the claim details. The API is `POST /api/claims/check` (multipart: `emp_id`, `category`, `vendor`, `amount_claimed`, `expense_date`, `submitted_on`, optional `claim_id`, optional `photo`). `GET /api/health` reports what the server can do.

- **No API key or paid service is needed.** Nothing paid is called.
- **Nothing heavy loads at startup.** If a reader or the data is missing, the server still starts. Claims then go to **review** (or the request gets a 503), never to approve.
- **Readers:**
  - **Cached PaddleOCR-VL text.** For any photo the GPU batch already read (all 467 in the pack), the service reuses that text, matched by image hash.
  - **Tesseract, live.** Reads every upload on the CPU.
  - **Live PaddleOCR-VL (optional).** `pip install -r requirements-vlm.txt`, then `TARANG_VLM=live`. Needs a CUDA GPU.
- **New photos on a laptop are read by Tesseract alone.** It misreads small digits, so most such claims go to review. Set `TARANG_VLM=off` to see that behaviour on the pack's photos too.

## Reproduce everything

The repository already contains every intermediate output, so this is only needed to rebuild from the raw data pack:

```bash
python scripts/clean_data.py        # data/raw/ -> data/processed/ (cleaned claims)
python scripts/audit_images.py      # photo audit -> reports/ (~5 min)
python scripts/ocr_extract.py       # Tesseract baselines -> data/interim/ocr/ (~3 min)
python scripts/vlm_extract.py       # check + merge PaddleOCR-VL output from Kaggle (kaggle/README.md)
python scripts/normalize_fields.py  # receipt text -> fields with per-field status -> data/processed/receipt_fields.csv
python scripts/run_benchmark.py     # OCR benchmark on the chronological holdout -> reports/ocr_benchmark*.csv
python scripts/decision_engine.py   # frozen rules on every claim -> predictions.csv, reports/decisions.csv
pytest                              # 120+ tests: parser, policy rules, known cases, service failure modes
```

- **Photos:** the receipt photos are committed in `data/raw/images/`. `scripts/download_images.py` re-fetches them; it reads your tracking link from `TARANG_TRACK_URL`.
- **PaddleOCR-VL runs on a Kaggle T4.** This laptop has 8 GB RAM; see `kaggle/README.md`.
- **Notebooks:** each pipeline step started as a notebook. The scripts reproduce the notebooks' outputs exactly.

## What is where

```
predictions.csv            190 test claims: fields read from the photo, approve/review, reason
submission-form.md         Answers to the submission form
MEMO.md                    One page for Nilesh: recommendation, rupees, what we need from him
docs/
  DATA_NOTES.md            What was cleaned or flagged in the data, and why
  IMAGE_AUDIT.md           Photo audit: rotation, blur, stamps, the same photo on two claims
  OCR_BENCHMARK.md         Tesseract vs PaddleOCR-VL on a chronological holdout
  POLICY_RULES.md          Policy matrix, reason codes, decision results, savings
notebooks/                 01 cleaning, 02 image audit, 03 OCR benchmark, 04 decision engine
src/tarang/                cleaning, images, ocr, fields, benchmark, decision_engine, decision_batch, service (+ static/index.html)
scripts/                   one command per pipeline step
tests/                     pytest suite
reports/                   audit, benchmark and decision outputs (CSV), contact sheets
kaggle/                    GPU notebooks for PaddleOCR-VL (full run + retry) and how to run them
data/raw/                  data pack as received (never modified)
data/processed/, data/interim/   cleaned claims, OCR/VLM text, extracted fields
```

## Key numbers

| | |
|---|---|
| **OCR benchmark** (60 held-out claims, exact match) | combined reader: vendor 98%, bill no. 95%, amount 95%, date 85%, GSTIN 81% ([details](docs/OCR_BENCHMARK.md)) |
| **Decision engine** | classifies 129 of 190 test claims (68%) as eligible for auto-approval (proposed decisions; a shadow pilot comes before any automatic payment). On train: 0 of 60 review-worthy claims auto-approved. *The rules were designed on that same data, so this is not an independent safety guarantee* ([details](docs/POLICY_RULES.md)) |
| **Leakage found** | in two months the clerks approved 25 claims (₹14,283) that the policy says should have been stopped: re-claimed bills (3 also a colleague's fuel bill), pen-altered totals, over-claims. The engine flags all 25 for review: ~₹7,100/month potentially preventable in Nagpur, a historical estimate (approved ≠ paid) |
| **Rules version** | `95bef2d00f35`. Frozen before the 190 test claims were run |
