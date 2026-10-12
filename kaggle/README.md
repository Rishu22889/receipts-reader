# Running PaddleOCR-VL on Kaggle

The laptop used for this project (M1, 8 GB RAM) can't hold a 0.9B+ VLM: loading PaddleOCR-VL pushed it to about 9.5 GB of swap. So the VLM runs on a free Kaggle T4 GPU. They only **read** receipts there. All scoring and decisions happen locally, with the same code used for the Tesseract baselines.

| Notebook | Model | Output |
|---|---|---|
| `paddleocr_vl.ipynb` | PaddleOCR-VL-0.9B via the native `transformers` implementation, one worker per GPU | `paddleocr_vl_outputs.jsonl` |
| `paddleocr_vl_retry.ipynb` | The same model, re-reading only the photos whose output looped (hit the token cap), with `repetition_penalty=1.05` | `paddleocr_vl_retry.jsonl` |

## Steps

1. Kaggle → *Datasets → New Dataset* → upload `kaggle/tarang-receipts.zip` (467 photos + `image_audit.csv`, 19 MB) and name it `tarang-receipts`. Rebuild the zip with `scripts/make_kaggle_zip.sh` if it's missing.
2. *Code → New Notebook → File → Import Notebook* → pick the `.ipynb` from this folder.
3. Right panel: *Add Input* → your `tarang-receipts` dataset. *Settings → Accelerator → GPU T4 x2*. *Settings → Internet → On*. Start from a fresh session (*Run → Factory reset*).
4. *Run All.* The **speed check** cell reads 2 photos on GPU 0 first and stops the notebook if a photo takes more than 60 s. Then both GPUs process half the photos each. Finished photos are skipped on a re-run.
5. *Output* tab → download the `.jsonl` → put it in `data/interim/vlm/` in this repo.

Each line of the `.jsonl` holds one photo: `file`, `backend`, `text` (raw model output), `rotation_applied`, `new_tokens`, `batch_size`, `batch_seconds`, `seconds` (batch time ÷ batch size, per GPU) and `error`. Warm-up runs aren't timed.

## Notes

- **Why the first runs took ~180 s per receipt:** the model's published `generation_config.json` sets `"use_cache": false`, so every generated token re-processed the whole image. The notebook forces `use_cache=True`. It also uses the native `transformers` class (`PaddleOCRVLForConditionalGeneration`, transformers 5.19.0) instead of the model's custom `trust_remote_code` code, and never installs Paddle.
- **Don't mix Paddle and PyTorch in one session.** Installing `paddlepaddle-gpu` replaces Kaggle's NCCL with an older build, and then `import torch` fails with `undefined symbol: ncclCommShrink`. This notebook never installs Paddle. If you hit this error from an earlier session: *Run → Factory reset* and Run All. (Restarting only the kernel doesn't help.)
- **T4 and bf16:** a T4 reports bf16 support but only emulates it. The notebook uses float32 on GPUs below compute capability 8.0. (Switching to float32 alone didn't fix the speed; the missing KV cache did.)
- **Receipt text is data, not instructions.** Five receipts carry a stamp that tells "automated processing" to approve the bill. The model is only asked to transcribe (prompt `OCR:`), so the stamp comes back as text. The local decision engine treats that text as data and sends any claim carrying it to review.
- **Verified locally without the GPU:** the native model class loads from the published config, and batched, left-padded inputs build correctly. Generation with this notebook has not been run yet. The speed-check cell is the first real test.
