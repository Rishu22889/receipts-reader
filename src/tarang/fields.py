"""Receipt text -> the five bill fields, each with its own status. Nothing is guessed: a missing field stays blank.

Packaged from notebooks/03_ocr_benchmark.ipynb, which explains and evaluates every rule here. The same parser is used for every
reading method (Tesseract, PaddleOCR-VL), so benchmark differences come from the reading, not the post-processing.
"""
import re
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd

# ---------------------------------------------------------------------------------------- text
TABLE_TAGS = re.compile(r"<(fcel|lcel|ecel|ucel|xcel|nl)>")


FIELD_LABELS = r"(Receipt No|Bill No|Invoice No|AWB No|Ticket No|Date|Time|Pump|Nozzle|Product|Rate|Volume|Amount|Vehicle No|Mode|Thank|Save Fuel|GSTIN)"


def clean_text(text):
    """Strip the VLM's table markup (<fcel>, <nl>, ...) into plain lines; split labels that ran together ("300.00Vehicle No")."""
    text = text.replace("<nl>", "\n")
    text = TABLE_TAGS.sub("\n", text)
    text = re.sub(r"(?<=[0-9a-z.)])(?=" + FIELD_LABELS + r"\b)", "\n", text)
    return "\n".join(l.strip(" |") for l in text.splitlines())


def looped(text, new_tokens=None, cap=768):
    """True when generation ran away: hit the token cap or one line repeats many times."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    most = Counter(lines).most_common(1)[0][1] if lines else 0
    return bool((new_tokens is not None and new_tokens >= cap) or most >= 8)


# ---------------------------------------------------------------------------------------- vendor names
CITY_SUFFIX = {"ngp", "nagpur", "wardha", "amravati", "yavatmal", "chandrapur"}
GENERIC = {"hotel"}
# Clerk spellings that no string measure can safely join; each checked against the name printed on the photo.
VENDOR_ALIASES = {
    "sitabuldi m l parking": "Sitabuldi Multilevel Parking",   # receipt prints SITABULDI MULTILEVEL PARKING
    "shree shirdi sai petro": "Shree Sai Petroleum",           # receipt prints SHREE SAI PETROLEUM (IndianOil, Khamla)
}


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower().replace("&", " ")).strip()


def tokens_key(s):
    return " ".join(sorted(t for t in norm(s).split() if t not in CITY_SUFFIX | GENERIC))


def similarity(a, b):
    """Best of: word-order-free match, and spacing-free match (catches 'Om SaiF uel Point')."""
    ka, kb = tokens_key(a), tokens_key(b)
    sa, sb = "".join(t for t in norm(a).split() if t not in CITY_SUFFIX | GENERIC), "".join(t for t in norm(b).split() if t not in CITY_SUFFIX | GENERIC)
    return max(SequenceMatcher(None, ka, kb).ratio(), SequenceMatcher(None, sa, sb).ratio())


def canonical_vendors(labels, min_count=2, threshold=0.8):
    """Map noisy clerk vendor labels onto canonical shop names. Returns {label: canonical name}.

    Centres are names typed >= min_count times; each label joins its most similar centre if similarity >= threshold.
    The display name is the centre's most common spelling, in title case.
    """
    keyed = Counter(norm(l) for l in labels)
    spellings = {}
    for l in labels:
        spellings.setdefault(norm(l), Counter())[l] += 1
    centres = sorted((k for k, n in keyed.items() if n >= min_count), key=lambda k: -keyed[k])
    mapping = {}
    for l in set(labels):
        if norm(l) in VENDOR_ALIASES:
            mapping[l] = VENDOR_ALIASES[norm(l)]
            continue
        best = max(centres, key=lambda c: (similarity(l, c), keyed[c]))
        key = best if similarity(l, best) >= threshold else norm(l)
        mapping[l] = spellings[key].most_common(1)[0][0].title()
    return mapping


# ---------------------------------------------------------------------------------------- GSTIN
GSTIN_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_PATTERN = r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]"
TO_DIGIT = str.maketrans({"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"})
TO_LETTER = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A", "7": "T"})


def gstin_is_valid(g):
    if not re.fullmatch(GSTIN_PATTERN, g):
        return False
    total = 0
    for i, ch in enumerate(g[:14]):
        p = GSTIN_CHARS.index(ch) * (2 if i % 2 else 1)
        total += p // 36 + p % 36
    return GSTIN_CHARS[(36 - total % 36) % 36] == g[14]


def gstin_by_position(raw):
    """Fix OCR letter/digit confusions using the GSTIN's fixed layout (2 digits, 5 letters, 4 digits, letter, entity, Z, check).

    Only the character class is forced by position; the result is accepted only if the official checksum passes.
    """
    g = list(raw)
    for i in range(15):
        if i in (0, 1, 7, 8, 9, 10):
            g[i] = g[i].translate(TO_DIGIT)
        elif i in (2, 3, 4, 5, 6, 11):
            g[i] = g[i].translate(TO_LETTER)
        elif i == 13:
            g[i] = "Z" if g[i] in "Z2" else g[i]
    return "".join(g)


def parse_gstin(text):
    """-> (value, status). status: ok | ok_normalised | invalid | absent. Never invents a GSTIN that fails the checksum."""
    for line in text.splitlines():
        if not re.search(r"G\s*[S5$]\s*T\s*[I1lL|]\s*N", line, re.IGNORECASE):
            continue
        after = re.split(r"G\s*[S5$]\s*T\s*[I1lL|]\s*N\s*[:.;]?", line, maxsplit=1, flags=re.IGNORECASE)[-1]
        compact = re.sub(r"[^0-9A-Za-z]", "", after.upper())
        if len(compact) < 15:
            continue
        raw = compact[:15]
        if gstin_is_valid(raw):
            return raw, "ok"
        fixed = gstin_by_position(raw)
        if gstin_is_valid(fixed):
            return fixed, "ok_normalised"
        return raw, "invalid"
    return "", "absent"


# ---------------------------------------------------------------------------------------- dates
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
DATE_RES = [
    (re.compile(r"\b(20\d\d)[-/.](\d{1,2})[-/.](\d{1,2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d\d|\d\d)\b"), "dmy"),
    (re.compile(r"\b(\d{1,2})[-\s]([A-Za-z]{3})[a-z]*[-\s,]+(20\d\d|\d\d)\b"), "dMy"),
]
DATE_LABEL = re.compile(r"(invoice\s*date|bill\s*date|date|dt\b|\bin\s*:)", re.IGNORECASE)


def _to_date(m, kind):
    try:
        if kind == "ymd":
            y, mo, d = int(m[1]), int(m[2]), int(m[3])
        elif kind == "dmy":
            d, mo, y = int(m[1]), int(m[2]), int(m[3])
        else:
            d, mo, y = int(m[1]), MONTHS.get(m[2][:3].lower()), int(m[3])
        y = y + 2000 if y < 100 else y
        return pd.Timestamp(year=y, month=mo, day=d)
    except (TypeError, ValueError):
        return None


def find_dates(line):
    out = []
    for rx, kind in DATE_RES:
        for m in rx.finditer(line):
            ts = _to_date(m, kind)
            if ts is not None and 2025 <= ts.year <= 2027:
                out.append((m.start(), ts))
    return [ts for _, ts in sorted(out, key=lambda x: x[0])]


def parse_date(text):
    """-> (YYYY-MM-DD, status). Prefers the bill/invoice date line over check-in/check-out and other dates."""
    lines = text.splitlines()
    ranked = []
    for i, line in enumerate(lines):
        dates = find_dates(line)
        if not dates:
            continue
        low = line.lower()
        if "invoice date" in low or "bill date" in low:
            rank = 0
        elif re.search(r"check[\s-]*(in|out)", low):
            rank = 3
        elif DATE_LABEL.search(line):
            rank = 1
        else:
            rank = 2
        ranked.append((rank, i, dates[0]))
    if not ranked:
        return "", "absent"
    ranked.sort(key=lambda x: (x[0], x[1]))
    best = ranked[0][2].strftime("%Y-%m-%d")
    others = {r[2].strftime("%Y-%m-%d") for r in ranked if r[0] == ranked[0][0]}
    return best, "ok" if len(others) == 1 else "uncertain"


# ---------------------------------------------------------------------------------------- receipt number
NUMBER_LABEL = re.compile(
    r"(invoice\s*no|bill\s*no|receipt\s*no|awb\s*no|ticket\s*no|\bno)\s*[.:;|]*\s*([A-Za-z0-9][A-Za-z0-9/\-]*)", re.IGNORECASE)


def parse_receipt_number(text):
    """-> (number, status). Kept as text, leading zeros preserved."""
    priority = ["invoice", "bill", "receipt", "awb", "ticket", "no"]
    found = []
    for line in text.splitlines():
        for m in NUMBER_LABEL.finditer(line):
            label, value = m.group(1).lower(), m.group(2).strip("-/").upper()
            value = re.match(r"[A-Z0-9/\-]*\d", value).group(0) if re.search(r"\d", value) else value
            if not re.search(r"\d", value) or label.startswith("no") and len(value) < 3:
                continue
            if "vehicle" in line.lower()[:m.start() + 1] or "room" in line.lower()[:m.start() + 1]:
                continue
            kind = next(p for p in priority if label.startswith(p) or p in label)
            found.append((priority.index(kind), value))
    if not found:
        return "", "absent"
    found.sort(key=lambda x: x[0])
    return found[0][1], "ok"


# ---------------------------------------------------------------------------------------- amounts
NUM = r"(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)"
UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def words_to_number(phrase):
    """'One Thousand Two Hundred Thirty only' -> 1230. None if it can't be read."""
    total, current, seen = 0, 0, False
    for w in re.findall(r"[a-z]+", phrase.lower()):
        if w in UNITS:
            current += UNITS[w]; seen = True
        elif w == "hundred":
            current = max(current, 1) * 100; seen = True
        elif w == "thousand":
            total += max(current, 1) * 1000; current = 0; seen = True
        elif w == "lakh":
            total += max(current, 1) * 100000; current = 0; seen = True
        elif w in ("rupees", "rs", "and", "only", "in", "words"):
            continue
        elif seen:
            break
    return total + current if seen else None


def to_amount(s):
    return float(s.replace(",", ""))


TOTAL_LABELS = [  # (pattern, rank): lower rank wins
    (re.compile(r"grand\s*total", re.I), 0),
    (re.compile(r"total\s*\(?\s*inr", re.I), 0),
    (re.compile(r"\btotal\b", re.I), 1),
    (re.compile(r"amount\s*(?:rs\.?|\(rs\)|:)(?!\s*in\s*words)", re.I), 2),   # "Amount Rs", "Amount (Rs):"; not the "Amount" column header
]


def _standalone_numbers(lines):
    out = []
    for i, l in enumerate(lines):
        m = re.fullmatch(r"\s*(?:rs\.?|₹)?\s*" + NUM + r"\s*(/-|7-)?\s*", l, re.I)
        if m:
            out.append((i, m.group(1), m.group(2) or ""))
    return out


def _fix_slash_dash(val, suffix, words_value):
    """'1230/-' is often read as '12307-' or '11207'. Drop the stray 7 only if the amount in words confirms it."""
    amount = to_amount(val)
    if val.endswith("7") and len(val) > 2 and (suffix == "7-" or words_value is not None):
        shorter = to_amount(val[:-1])
        if words_value is None or abs(shorter - words_value) < 0.01:
            return shorter
    return amount


SPACED = re.compile(r"\b(\d) (\d{3}(?:\.\d{1,2})?)\b")   # "1 472.00": a digit added in front, often by pen


def _fuel_amount(lines):
    """Fuel slips print rate, volume and amount: (amount line, True if rate x volume agrees within Rs 1)."""
    get = lambda label: next((float(m.group(1).replace(",", "")) for l in lines for m in [re.search(label + r"[^\d]*" + NUM, l, re.I)] if m), None)
    rate, vol, amt = get(r"rate\s*\(rs/l\)"), get(r"volume\s*\(l\)"), get(r"amount\s*\(rs\)")
    return amt, bool(rate and vol and amt and abs(rate * vol - amt) <= 1.0)


def _digits_match_words(val, words_value):
    """'11.30/-' or '12307-' next to 'One Thousand One Hundred Thirty': same digits, misplaced punctuation."""
    w = str(int(words_value))
    return re.sub(r"\D", "", val) in {w, w + "0", w + "00", w + "7"}


def parse_total(text):
    """-> (amount 'NNNN.NN', status). status: verified | words_only | ok | inconsistent | uncertain | absent.

    verified = a second source on the receipt agrees (amount in words, sub total + taxes, items sum, fuel rate x volume).
    inconsistent = the printed total contradicts the receipt's own arithmetic (a misread or an alteration); the amount
    returned is the self-consistent one.
    """
    lines = [SPACED.sub(r"\1\2", l) for l in text.splitlines()]
    words_value = None
    for i, l in enumerate(lines):
        if re.search(r"(in\s*words|rupees)", l, re.I):
            phrase = re.split(r"words\)?\s*:?", l, flags=re.I)[-1] + " " + (lines[i + 1] if i + 1 < len(lines) else "")
            words_value = words_to_number(phrase) or words_value   # the words may continue on the next line
    expected = _items_expected(lines)
    fuel_amount, fuel_ok = _fuel_amount(lines)
    candidates = []
    for i, l in enumerate(lines):
        for rx, rank in TOTAL_LABELS:
            if not rx.search(l) or re.search(r"sub\s*total|in\s*words|rate|volume", l, re.I):
                continue
            tail = rx.split(l, maxsplit=1)[-1] if rank < 2 else l
            same = re.findall(NUM + r"\s*(/-|7-)?", tail)
            if same:
                candidates.append((rank, i, *same[-1]))
            else:   # value on a nearby line: after the label, or just before a GRAND TOTAL ("914.00 / GRAND TOTAL")
                nearby = list(range(i + 1, min(i + 4, len(lines)))) + ([i - 1] if rank == 0 else [])
                for j in nearby:
                    if 0 <= j < len(lines):
                        m = re.fullmatch(r"\s*(?:rs\.?|₹)?\s*" + NUM + r"\s*(/-|7-)?\s*", lines[j], re.I)
                        if m:
                            candidates.append((rank + 0.5, i, m.group(1), m.group(2) or ""))
                            break
            break
    candidates.sort(key=lambda c: (c[0], -c[1]))
    printed = to_amount(candidates[0][2]) if candidates else None

    # fuel: the amount line is confirmed by rate x volume; a TOTAL that disagrees with it is inconsistent
    if fuel_ok:
        if printed is None or abs(printed - fuel_amount) < 0.01 or candidates[0][0] >= 2:
            return f"{fuel_amount:.2f}", "verified"
        if _extra_leading_digits(printed, fuel_amount):
            return f"{_strip_leading_digits(printed, fuel_amount):.2f}", "inconsistent"   # 536.01 -> 1536.01: digit added in front
        return f"{printed:.2f}", "inconsistent"
    # restaurants / hotels: sub total + taxes + round off
    if expected is not None:
        if printed is None or abs(printed - expected) < 0.01:
            return f"{expected:.2f}", "verified" if printed is not None or expected in _numbers(lines) else "ok"
        if _extra_leading_digits(printed, expected):
            return f"{_strip_leading_digits(printed, expected):.2f}", "inconsistent"   # 472.00 printed, "1" added by pen -> 1472.00
        return f"{printed:.2f}", "inconsistent"         # receipt contradicts itself: keep what is printed, flag it
    if words_value is not None and round(words_value, 2) in _numbers(lines) | {round(to_amount(c[2]), 2) for c in candidates}:
        return f"{words_value:.2f}", "verified"
    if words_value is not None:
        if printed is None:
            return f"{words_value:.2f}", "words_only"
        val, suffix = candidates[0][2], candidates[0][3]
        if abs(printed - words_value) < 0.01 or _digits_match_words(val, words_value):
            return f"{words_value:.2f}", "verified"
        return f"{_fix_slash_dash(val, suffix, words_value):.2f}", "uncertain"
    # handwritten bills: the largest standalone number is the sum of the others
    nums = [to_amount(v) for _, v, _ in _standalone_numbers(lines)]
    if len(nums) >= 3 and abs(max(nums) * 2 - sum(nums)) < 0.01:
        if printed is None or printed in nums:   # the label may sit next to an item price in a one-column layout
            return f"{max(nums):.2f}", "verified"
    if printed is None:
        return "", "absent"
    others = [to_amount(c[2]) for c in candidates[1:]]
    return f"{printed:.2f}", "verified" if printed in others else "ok"


def _extra_leading_digits(printed, expected):
    """True when the printed total is the expected total with digits added in front (472.00 -> 1472.00).

    Compared on whole rupees, because printed round-off can leave the computed total a few paise away (268.01 vs 268.00).
    """
    p, e = str(int(printed + 0.5)), str(int(expected + 0.5))
    return len(p) > len(e) and p.endswith(e)


def _strip_leading_digits(printed, expected):
    """1268.00 with an expected ~268 -> 268.00: the printed total without the digits added in front."""
    return round(printed % 10 ** len(str(int(expected + 0.5))), 2)


def _numbers(lines):
    return {round(to_amount(v), 2) for _, v, _ in _standalone_numbers(lines)}


def _items_expected(lines):
    """Sub total (or taxable value) + CGST + SGST + round off, as printed. None if there's no sub total."""
    def grab(label):
        for i, l in enumerate(lines):
            if re.search(label, l, re.I):
                rest = re.sub(r"[@0]?\s*[\d.]+\s*%", "", re.split(label, l, flags=re.I)[-1])   # "CGST @2.5%" / OCR "02.5%": the rate, not the tax
                same = re.findall(r"[+-]?" + NUM, rest)
                if same:
                    return float(same[-1].replace(",", ""))
                if i + 1 < len(lines):
                    nxt = re.fullmatch(r"\s*([+-]?" + NUM + r")\s*", lines[i + 1])
                    if nxt:
                        return float(nxt.group(1).replace(",", ""))
        return None
    sub = grab(r"sub\s*total|taxable\s*value")
    if sub is None:
        return None
    cgst, sgst = grab(r"cgst"), grab(r"sgst")
    if cgst is None or sgst is None or abs(cgst - sgst) > 0.01:
        return None   # CGST and SGST are always equal on these bills; unequal means the text lost a line
    round_off = grab(r"round\s*off") or 0.0
    if abs(round_off) >= 1:
        round_off = 0.0   # a real round-off is under Rs 1; a bigger value is a neighbouring line read out of place
    return round(sub + cgst + sgst + round_off, 2)


# ---------------------------------------------------------------------------------------- vendor master + matching
BRANDS = {"bpcl", "hpcl", "indianoil", "indian oil", "tax invoice", "booking receipt"}


VARIABLE_LINE = re.compile(r"\b(date|bill|receipt|invoice|awb|ticket|total|amount|rs|qty|item|steward|table|time|pump|nozzle|"
                           r"rate|volume|vehicle|veh|type|in|out|mode|name|from|to|guest|check|room|pcs|origin|dest|consignee|freight|sub|cgst|sgst)\b"
                           r"|mh\s?\d\d")   # vehicle registrations identify the rep, not the shop


def _signature_candidate(nl):
    """Lines that identify a shop (address, phone, GSTIN), not per-bill content (dates, amounts, field values)."""
    return (len(nl) >= 8 and sum(c.isalpha() for c in nl) >= 4 and nl not in BRANDS
            and not VARIABLE_LINE.search(nl) and not find_dates(nl.replace(" ", "/")))


def number_shape(no):
    """'006239' -> '00dddd', 'HCP/26-27/4594' -> 'HCP/dd-dd/dddd', 'A/6178' -> 'A/dddd'. Leading zeros are part of the shape."""
    no = no.upper()
    lead = re.match(r"0*", no).group(0)
    return lead + re.sub(r"\d", "d", no[len(lead):])


def build_vendor_master(dev):
    """Learn each shop's identifiers from DEV claims only: canonical name, GSTINs, bill-number prefixes, signature lines.

    dev: DataFrame with canonical_vendor, clerk_gstin, clerk_gstin_valid, clerk_bill_no, text (reference OCR text).
    Signature lines (address, phone) appear on >= 2 of a shop's receipts and on no other shop's.
    """
    master = {"names": sorted(dev.canonical_vendor.unique()), "gstin": {}, "prefix": {}, "lines": {}, "shapes": {}}
    for v, g in dev.groupby("canonical_vendor"):
        master["shapes"][v] = sorted({number_shape(n) for n in g.clerk_bill_no if n} | {number_shape(n)[1:] for n in g.clerk_bill_no if n.startswith("00")})
    for v, g in dev.groupby("canonical_vendor"):
        for gst in g.loc[g.clerk_gstin_valid == "True", "clerk_gstin"]:
            master["gstin"][gst] = v
    prefixes = {}
    for v, g in dev.groupby("canonical_vendor"):
        for no in g.clerk_bill_no:
            m = re.match(r"([A-Z]+/?\d*-?\d*/?|[A-Z]{2})", no.upper())
            if m and re.search(r"[A-Z]", m.group(1)):
                prefixes.setdefault(re.sub(r"\d+$", "", m.group(1)) if "/" not in m.group(1) else m.group(1).split("/")[0] + "/", set()).add(v)
    master["prefix"] = {p: next(iter(vs)) for p, vs in prefixes.items() if len(vs) == 1}
    line_vendors, line_count = {}, Counter()
    for v, g in dev.groupby("canonical_vendor"):
        seen = Counter()
        for t in g.text:
            for l in {norm(x) for x in clean_text(t).splitlines()}:
                if _signature_candidate(l):
                    seen[l] += 1
        for l, n in seen.items():
            line_vendors.setdefault(l, set()).add(v)
            line_count[(l, v)] = n
    master["lines"] = {l: next(iter(vs)) for l, vs in line_vendors.items() if len(vs) == 1 and line_count[(l, next(iter(vs)))] >= 2}
    return master


def parse_vendor(text, master, gstin=""):
    """-> (canonical vendor, status). status: ok (name read) | inferred_gstin | inferred_line | inferred_prefix | unknown | absent."""
    lines = [l for l in clean_text(text).splitlines() if l.strip()]
    best, score = None, 0.0
    for l in lines[:12]:
        for name in master["names"]:
            s = similarity(l, name)
            if s > score:
                best, score = name, s
    if score >= 0.85:
        return best, "ok"
    if gstin and gstin in master["gstin"]:
        return master["gstin"][gstin], "inferred_gstin"
    for l in lines:
        nl = norm(l)
        if nl in master["lines"]:
            return master["lines"][nl], "inferred_line"
    for l in lines:
        nl = norm(l)
        if _signature_candidate(nl):
            for sig, v in master["lines"].items():
                if SequenceMatcher(None, nl, sig).ratio() >= 0.9:
                    return v, "inferred_line"
    number, _ = parse_receipt_number(text)
    for p, v in master["prefix"].items():
        if number.startswith(p):
            return v, "inferred_prefix"
    for l in lines[:4]:
        if re.fullmatch(r"[A-Za-z&.' ]{6,}", l.strip()) and norm(l) not in BRANDS:
            return l.strip().title(), "unknown"
    return "", "absent"


def parse_receipt(text, master, new_tokens=None):
    """All five fields from one OCR/VLM text. Every field gets its own status; nothing is guessed."""
    text = clean_text(text)
    gstin, gstin_status = parse_gstin(text)
    vendor, vendor_status = parse_vendor(text, master, gstin if gstin_status.startswith("ok") else "")
    date, date_status = parse_date(text)
    number, number_status = parse_receipt_number(text)
    shapes = master["shapes"].get(vendor)
    if number and shapes:   # the shop's own numbering format, learned from dev (clerk truncations allowed one leading zero less)
        number_status = "ok" if number_shape(number) in shapes else "format_mismatch"
    total, total_status = parse_total(text)
    if looped(text, new_tokens):
        total_status = "uncertain" if total else total_status   # runaway generation can invent amounts
    return {"vendor": vendor, "vendor_status": vendor_status, "receipt_date": date, "date_status": date_status,
            "receipt_number": number, "number_status": number_status, "gstin": gstin if gstin_status != "invalid" else "",
            "gstin_read": gstin, "gstin_status": gstin_status, "total_amount": total, "total_status": total_status,
            "looped": looped(text, new_tokens)}


FIELDS = [("vendor", "vendor_status"), ("receipt_date", "date_status"), ("receipt_number", "number_status"),
          ("gstin_read", "gstin_status"), ("total_amount", "total_status")]
TRUSTED = {"vendor_status": {"ok"}, "date_status": {"ok"}, "number_status": {"ok"},  # number "ok" = fits the shop's format
           "gstin_status": {"ok", "ok_normalised"}, "total_status": {"verified"}}


def combine(primary, fallback):
    """Primary (VLM) fields, with gaps filled from the fallback (Tesseract) only where the fallback value passes its own check."""
    out = dict(primary)
    for field, status in FIELDS:
        if primary[status] in ("absent", "uncertain", "unknown", "invalid", "format_mismatch") and fallback[status] in TRUSTED[status]:
            out[field], out[status] = fallback[field], fallback[status] + "_fallback"
    out["gstin"] = out["gstin_read"] if out["gstin_status"].startswith("ok") else ""
    return out
