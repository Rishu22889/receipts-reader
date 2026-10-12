"""Download the receipt photos from the data-pack page -> data/raw/images/.

Usage: TARANG_TRACK_URL="https://atslite.interviewgod.ai/track/<your token>" python scripts/download_images.py
The photos are already committed in data/raw/images/, so this is only needed to re-fetch them. The tracking link is personal,
so it is read from the environment and never stored in the repository.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

# The track page is a JavaScript app, so the attachment list comes from its JSON API,
# not from <a href> links in the HTML.
API = "https://atslite.interviewgod.ai/api/track"
track_url = os.environ.get("TARANG_TRACK_URL") or (sys.argv[1] if len(sys.argv) > 1 else "")
if "/track/" not in track_url:
    sys.exit("Set TARANG_TRACK_URL (or pass the tracking link as an argument): https://atslite.interviewgod.ai/track/<token>")
TOKEN = track_url.rstrip("/").split("/track/")[-1]

# Save into data/raw/images/ (relative to the project root)
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "images")
os.makedirs(save_dir, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

print("Fetching attachment list...")
data = requests.get(f"{API}/{TOKEN}", headers=headers, timeout=30).json()
attachments = [a for a in data["current"]["task"]["attachments"] if a["label"].lower().endswith((".jpg", ".jpeg", ".png"))]
print(f"Found {len(attachments)} images to download.\n")


def download(att):
    # Labels look like "images/IMG-20260610-WA0001.jpg"; keep only the file name
    filename = os.path.basename(att["label"])
    file_path = os.path.join(save_dir, filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        return filename, "skipped (exists)"
    try:
        # Each key resolves to a short-lived signed S3 URL
        res = requests.get(f"{API}/{TOKEN}/uploads/resolve", params={"key": att["key"]}, headers=headers, timeout=30)
        res.raise_for_status()
        img = requests.get(res.json()["url"], timeout=60)
        img.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(img.content)
        return filename, "ok"
    except Exception as e:
        return filename, f"error: {e}"


failed = 0
with ThreadPoolExecutor(max_workers=8) as pool:
    for idx, (filename, status) in enumerate(pool.map(download, attachments), 1):
        if status.startswith("error"):
            failed += 1
        print(f"[{idx}/{len(attachments)}] {filename}: {status}")

print(f"\nDone. {len(attachments) - failed} saved, {failed} failed.")
sys.exit(1 if failed else 0)
