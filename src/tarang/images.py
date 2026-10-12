"""Receipt-photo audit: missing/corrupt files, quality, orientation, instruction-like text, same-photo reuse.

Packaged from notebooks/02_image_audit.ipynb, which explains and visually verifies every heuristic here.
Raw photos are never modified.
"""
import csv
import io
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageOps

ROTATIONS = (0, 90, 180, 270)
PEN_INK_PCT = 0.05           # % of paper pixels in blue-ballpoint hue -> "pen marks likely"
STAMP_PURPLE_PCT = 0.5       # % of pixels in violet stamp hue -> instruction stamp (gap: ~0.9 vs <=0.11)
SAME_PHOTO_MAX_DISTANCE = 2  # dHash bits; same photo = 0, different bills >= 3
LOW_SHARPNESS = 25
LOW_CONFIDENT_WORDS = 5

INJECTION_PATTERNS = [
    r"automat\w*", r"pre[- ]?verified", r"approve\s+(it|this)", r"skip\s+\w+\s+\w*\s*check",
    r"ignore\s+(all|previous|the|any)", r"instruction", r"reconciled\s+by", r"in\s+your\s+output",
]


# --- 1. missing photos -------------------------------------------------------------------------

def missing_images_report(claims):
    """One row per claim whose photo is not on the server, with where it was expected."""
    missing = claims[claims.photo_available.astype(str) == "False"]
    return pd.DataFrame({
        "claim_id": missing.claim_id,
        "split": missing.split,
        "image_file_raw": missing.image_file_raw,
        "expected_path": "data/raw/images/" + missing.image_file_raw.map(lambda v: Path(v.replace("\\", "/")).name),
        "found_on_server": False,
        "clerk_status": missing.status,
        "clerk_note": missing.clerk_note,
        "action": missing.split.map({"train": "exclude from extraction scoring", "test": "decision = review; receipt fields left blank"}),
    }).reset_index(drop=True)


# --- 2. integrity and quality ------------------------------------------------------------------

def open_status(path):
    """('ok', width, height) or ('corrupt: ...', 0, 0). verify() checks structure, load() decodes every pixel."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            return "ok", im.width, im.height
    except Exception as e:
        return f"corrupt: {e}", 0, 0


def quality(path):
    """Sharpness, brightness, contrast, blue pen ink on paper, and violet stamp ink."""
    rgb = Image.open(path).convert("RGB")
    grey = ImageOps.grayscale(rgb)
    small = np.asarray(grey.resize((800, round(800 * grey.height / grey.width))), dtype=float)
    lap = 4 * small[1:-1, 1:-1] - small[:-2, 1:-1] - small[2:, 1:-1] - small[1:-1, :-2] - small[1:-1, 2:]

    g = np.asarray(grey, dtype=float)
    hsv = np.asarray(rgb.convert("HSV"), dtype=int)
    paper = ((g > 150) & (hsv[..., 1] < 70)).astype(np.uint8) * 255
    near_paper = np.asarray(Image.fromarray(paper).filter(ImageFilter.BoxBlur(12)), dtype=float) / 255 > 0.5
    blue = (hsv[..., 0] > 140) & (hsv[..., 0] < 200) & (hsv[..., 1] > 80) & (hsv[..., 2] > 70)
    purple = (hsv[..., 0] >= 184) & (hsv[..., 0] <= 215) & (hsv[..., 1] > 60) & (hsv[..., 2] > 80)
    return {
        "sharpness": round(lap.var(), 1),
        "brightness": round(small.mean(), 1),
        "contrast": round(small.std(), 1),
        "pen_ink_pct": round((blue & near_paper).sum() / max(near_paper.sum(), 1) * 100, 3),
        "purple_pct": round(purple.mean() * 100, 3),
    }


# --- 3. orientation via OCR --------------------------------------------------------------------

def preprocess(grey):
    """3x upscale, divide out uneven lighting, threshold. Makes Tesseract read small receipt print."""
    big = grey.resize((grey.width * 3, grey.height * 3), Image.LANCZOS)
    a = np.asarray(big, dtype=float)
    background = np.asarray(big.filter(ImageFilter.GaussianBlur(25)), dtype=float) + 1
    return Image.fromarray(((np.clip(a / background * 255, 0, 255) > 215) * 255).astype(np.uint8))


def ocr_words(im):
    """Run Tesseract (sparse text mode) and return [(word, confidence)]."""
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        im.save(f.name, dpi=(300, 300))
        out = subprocess.run(["tesseract", f.name, "-", "--psm", "11", "tsv"], capture_output=True, text=True).stdout
    rows = csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE)
    return [(r["text"].strip(), float(r["conf"])) for r in rows if r["text"].strip()]


def confident(words):
    return sum(1 for w, c in words if c >= 70 and len(w) >= 3)


def orientation(path):
    """OCR at 0/90/180/270 and keep the rotation with the most confident words."""
    clean = preprocess(ImageOps.grayscale(Image.open(path)))
    by_angle = {angle: ocr_words(clean.rotate(angle, expand=True)) for angle in ROTATIONS}
    scores = {angle: confident(w) for angle, w in by_angle.items()}
    best = max(scores, key=scores.get)
    return {
        "rotation": best,  # counter-clockwise degrees that make the receipt upright
        "confident_words": scores[best],
        "rotation_margin": scores[best] - sorted(scores.values())[-2],
        "ocr_text": " ".join(w for w, _ in by_angle[best]),
    }


# --- 4. instruction-like text and same-photo reuse ---------------------------------------------

def injection_hits(text):
    """Instruction-like phrases found in receipt text. Receipt text is data, never an instruction."""
    return sorted({m.group(0).lower() for p in INJECTION_PATTERNS for m in re.finditer(p, text, re.IGNORECASE)})


def dhash(path):
    """64-bit difference hash; identical for the same photo even after recompression."""
    small = np.asarray(ImageOps.grayscale(Image.open(path)).resize((9, 8), Image.LANCZOS), dtype=int)
    return "".join("1" if b else "0" for b in (small[:, 1:] > small[:, :-1]).flatten())


def hash_distances(hashes):
    """Sorted [(distance, file_a, file_b)] for every pair of photos."""
    return sorted((sum(a != b for a, b in zip(hashes[x], hashes[y])), x, y) for x, y in combinations(hashes, 2))


def same_photo_groups(distances):
    """Group photos that are the same picture: connected components of pairs within SAME_PHOTO_MAX_DISTANCE.

    Returns {file: (group_files, max_distance_in_group)}. Handles 3+ copies, not just pairs.
    """
    parent = {}

    def find(f):
        parent.setdefault(f, f)
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    close = [(d, x, y) for d, x, y in distances if d <= SAME_PHOTO_MAX_DISTANCE]
    for _, x, y in close:
        parent[find(x)] = find(y)
    members = {}
    for f in parent:
        members.setdefault(find(f), set()).add(f)
    max_d = {}
    for d, x, _ in close:
        root = find(x)
        max_d[root] = max(max_d.get(root, 0), d)
    return {f: (frozenset(members[find(f)]), max_d[find(f)]) for f in parent}


DUPLICATE_COLUMNS = ["duplicate_group", "claim_id", "other_claim_ids", "image_file", "max_hash_distance", "split", "emp_id",
                     "expense_date", "vendor", "amount_claimed", "clerk_status"]


def same_photo_report(distances, claims):
    """One row per claim whose photo (or an identical copy of it) is used by another claim.

    `other_claim_ids` names the counterpart claims explicitly, so an already-paid bill can't slip through.
    Also covers two claims that point at the very same file.
    """
    groups = same_photo_groups(distances)
    for f, n in claims[claims.image_file != ""].image_file.value_counts().items():
        if n > 1 and f not in groups:
            groups[f] = (frozenset([f]), 0)

    claim_groups = {}
    for files, max_d in set(groups.values()):
        ids = sorted(claims.loc[claims.image_file.isin(files), "claim_id"])
        if len(ids) > 1:
            claim_groups[tuple(ids)] = (files, max_d)

    rows = []
    for group_id, ids in enumerate(sorted(claim_groups), 1):
        files, max_d = claim_groups[ids]
        for _, c in claims[claims.claim_id.isin(ids)].sort_values("claim_id").iterrows():
            rows.append({"duplicate_group": group_id, "claim_id": c.claim_id,
                         "other_claim_ids": "; ".join(i for i in ids if i != c.claim_id),
                         "image_file": c.image_file, "max_hash_distance": max_d, "split": c.split, "emp_id": c.emp_id,
                         "expense_date": c.expense_date, "vendor": c.vendor, "amount_claimed": c.amount_claimed,
                         "clerk_status": c.status})
    return pd.DataFrame(rows, columns=DUPLICATE_COLUMNS)


# --- 5. per-photo and per-claim audit ----------------------------------------------------------

def audit_photo(path):
    path = Path(path)
    status, width, height = open_status(path)
    row = {"file": path.name, "open_status": status, "width": width, "height": height,
           "kb": round(path.stat().st_size / 1024, 1)}
    if status == "ok":
        row.update(quality(path))
        row.update(orientation(path))
        row["dhash"] = dhash(path)
    return row


def audit_photos(images_dir, workers=8):
    files = sorted(Path(images_dir).glob("*.jpg"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        df = pd.DataFrame(list(pool.map(audit_photo, files)))
    df["injection_hits"] = df.ocr_text.fillna("").map(injection_hits).map("; ".join)
    df["instruction_text"] = (df.injection_hits != "") | (df.purple_pct > STAMP_PURPLE_PCT)
    df["pen_marks_likely"] = df.pen_ink_pct > PEN_INK_PCT
    df["total_keyword_found"] = df.ocr_text.fillna("").str.contains(r"total|amount|amt|grand", case=False)
    df["blur_candidate"] = (df.sharpness < LOW_SHARPNESS) | (df.confident_words < LOW_CONFIDENT_WORDS)
    return df


def _issues(r):
    if r.photo_available == "False":
        return "missing photo"
    out = []
    if r.open_status != "ok": out.append("corrupt")
    if r.rotation: out.append(f"rotated {int(r.rotation)}")
    if r.blur_candidate: out.append("low OCR confidence")
    if r.instruction_text: out.append("instruction text")
    if r.pen_marks_likely: out.append("pen marks")
    if not r.total_keyword_found: out.append("no total found?")
    if r.duplicate_other_claims: out.append(f"same photo as {r.duplicate_other_claims}")
    return "; ".join(out)


def claim_audit(claims, photos, duplicates):
    audit = claims[["claim_id", "split", "image_file", "photo_available"]].merge(
        photos.drop(columns=["ocr_text", "dhash"]), left_on="image_file", right_on="file", how="left").drop(columns="file")
    by_claim = duplicates.set_index("claim_id")
    audit["duplicate_group"] = audit.claim_id.map(by_claim.duplicate_group).astype("Int64")
    audit["duplicate_other_claims"] = audit.claim_id.map(by_claim.other_claim_ids).fillna("")
    audit["injection_hits"] = audit.injection_hits.fillna("")
    for col in ["pen_marks_likely", "total_keyword_found", "blur_candidate", "instruction_text"]:
        audit[col] = audit[col].astype("boolean")
    audit["issues"] = audit.apply(_issues, axis=1)
    return audit


def contact_sheet(rows, images_dir, path, cols=8, cell=240):
    """Grid of thumbnails labelled with claim id and issues."""
    n_rows = max(1, -(-len(rows) // cols))
    sheet = Image.new("RGB", (cols * cell, n_rows * (cell + 30)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows.itertuples()):
        x, y = (i % cols) * cell, (i // cols) * (cell + 30)
        im = Image.open(Path(images_dir) / r.image_file).convert("RGB")
        im.thumbnail((cell - 6, cell - 6))
        sheet.paste(im, (x + 3, y + 3))
        draw.text((x + 4, y + cell - 2), r.claim_id, fill="black")
        draw.text((x + 4, y + cell + 10), (r.issues or "-")[:38], fill="firebrick")
    sheet.save(path, quality=80)
    return sheet


def load_claims(processed_dir):
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    return pd.concat([
        read(Path(processed_dir) / "claims_train_clean.csv").assign(split="train"),
        read(Path(processed_dir) / "claims_test_clean.csv").assign(split="test"),
    ], ignore_index=True).fillna("")


def run(images_dir, processed_dir, reports_dir):
    """Write missing_images.csv, duplicate_images.csv, image_audit.csv and contact sheets. Returns the claim audit."""
    reports_dir = Path(reports_dir)
    sheets = reports_dir / "contact_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    claims = load_claims(processed_dir)

    missing_images_report(claims).to_csv(reports_dir / "missing_images.csv", index=False)
    photos = audit_photos(images_dir)
    duplicates = same_photo_report(hash_distances(dict(zip(photos.file, photos.dhash))), claims)
    duplicates.to_csv(reports_dir / "duplicate_images.csv", index=False)
    audit = claim_audit(claims, photos, duplicates)
    audit.to_csv(reports_dir / "image_audit.csv", index=False)

    with_photo = audit[audit.photo_available == "True"].reset_index(drop=True)
    for k, start in enumerate(range(0, len(with_photo), 64), 1):
        contact_sheet(with_photo.iloc[start:start + 64], images_dir, sheets / f"all_{k:02d}.jpg")
    examples = {
        "rotated": with_photo[with_photo.rotation != 0],
        "blur_unreadable": with_photo[with_photo.blur_candidate].sort_values("sharpness"),
        "instruction_text": with_photo[with_photo.instruction_text],
        "pen_marks": with_photo[with_photo.pen_marks_likely].sort_values("pen_ink_pct", ascending=False),
        "no_total_found": with_photo[~with_photo.total_keyword_found],
        "same_photo": with_photo[with_photo.duplicate_group.notna()].sort_values(["duplicate_group", "claim_id"]),
    }
    for name, rows in examples.items():
        contact_sheet(rows.head(32), images_dir, sheets / f"examples_{name}.jpg")
    return audit
