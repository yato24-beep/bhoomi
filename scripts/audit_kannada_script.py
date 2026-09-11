import json
import os
import sys
import re
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VOWELS = [
    "ಅ", "ಆ", "ಇ", "ಈ", "ಉ", "ಊ", "ಋ", "ಎ", "ಏ", "ಐ", "ಒ", "ಓ", "ಔ"
]

CONSONANTS = [
    "ಕ", "ಖ", "ಗ", "ಘ", "ಙ",
    "ಚ", "ಛ", "ಜ", "ಝ", "ಞ",
    "ಟ", "ಠ", "ಡ", "ಢ", "ಣ",
    "ತ", "ಥ", "ದ", "ಧ", "ನ",
    "ಪ", "ಫ", "ಬ", "ಭ", "ಮ",
    "ಯ", "ರ", "ಲ", "ವ",
    "ಶ", "ಷ", "ಸ", "ಹ",
    "ಳ"
]

MATRAS = {
    "\u0CBE": "ಾ (aa)",
    "\u0CBF": "ಿ (i)",
    "\u0CC0": "ೀ (ii)",
    "\u0CC1": "ು (u)",
    "\u0CC2": "ೂ (uu)",
    "\u0CC3": "ೃ (ru)",
    "\u0CC4": "ೄ (rru)",
    "\u0CC6": "ೆ (e)",
    "\u0CC7": "ೇ (ee)",
    "\u0CC8": "ೈ (ai)",
    "\u0CCA": "ೊ (o)",
    "\u0CCB": "ೋ (oo)",
    "\u0CCC": "ೌ (au)",
}

MODIFIERS = {
    "\u0C82": "ಂ (anusvara)",
    "\u0C83": "ಃ (visarga)",
    "\u0CCD": "್ (virama)",
}

NUMERALS = {
    "೦": "0", "೧": "1", "೨": "2", "೩": "3", "೪": "4",
    "೫": "5", "೬": "6", "೭": "7", "೮": "8", "೯": "9"
}

def audit_dataset(manifest_paths):
    char_counts = Counter()
    matra_counts = Counter()
    modifier_counts = Counter()
    numeral_counts = Counter()
    conjunct_counts = Counter()
    syllable_counts = Counter()
    total_words = 0
    total_chars = 0

    # Regex for Kannada conjuncts: Consonant + Virama + Consonant (+ Virama + Consonant)*
    kannada_cons_pattern = r"[\u0C95-\u0CB9\u0CB3]"
    conjunct_regex = re.compile(f"({kannada_cons_pattern}\u0CCD{kannada_cons_pattern}(?:\u0CCD{kannada_cons_pattern})*)")

    for path_name, manifest_path in manifest_paths:
        if not os.path.exists(manifest_path):
            continue
        print(f"Auditing: {path_name} ({manifest_path})...", flush=True)
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                txt = item.get("text", "") or item.get("label", "")
                if not txt:
                    continue
                total_words += 1
                total_chars += len(txt)

                # Count raw characters
                for ch in txt:
                    char_counts[ch] += 1
                    if ch in MATRAS:
                        matra_counts[ch] += 1
                    if ch in MODIFIERS:
                        modifier_counts[ch] += 1
                    if ch in NUMERALS:
                        numeral_counts[ch] += 1

                # Extract conjuncts
                for c_match in conjunct_regex.findall(txt):
                    conjunct_counts[c_match] += 1

    return {
        "total_words": total_words,
        "total_chars": total_chars,
        "char_counts": char_counts,
        "matra_counts": matra_counts,
        "modifier_counts": modifier_counts,
        "numeral_counts": numeral_counts,
        "conjunct_counts": conjunct_counts,
    }

def print_audit_report(data):
    total_w = data["total_words"]
    total_c = data["total_chars"]
    c_counts = data["char_counts"]
    m_counts = data["matra_counts"]
    mod_counts = data["modifier_counts"]
    num_counts = data["numeral_counts"]
    conj_counts = data["conjunct_counts"]

    print("=" * 80)
    print(f"KANNADA DATASET FULL SCRIPT AUDIT REPORT (Total Words: {total_w:,}, Total Chars: {total_c:,})")
    print("=" * 80)

    print("\n[1] INDEPENDENT VOWELS FREQUENCY:")
    print("-" * 60)
    for v in VOWELS:
        cnt = c_counts[v]
        pct = (cnt / total_c) * 100 if total_c else 0
        print(f"  {v} (U+{ord(v):04X}) : {cnt:8,d} occurrences ({pct:6.3f}%)")

    print("\n[2] CONSONANTS FREQUENCY:")
    print("-" * 60)
    for c in CONSONANTS:
        cnt = c_counts[c]
        pct = (cnt / total_c) * 100 if total_c else 0
        print(f"  {c} (U+{ord(c):04X}) : {cnt:8,d} occurrences ({pct:6.3f}%)")

    print("\n[3] VOWEL SIGNS / MATRAS FREQUENCY:")
    print("-" * 60)
    for m_hex, m_name in MATRAS.items():
        cnt = m_counts[m_hex]
        pct = (cnt / total_c) * 100 if total_c else 0
        print(f"  {m_hex} {m_name:16s} (U+{ord(m_hex):04X}) : {cnt:8,d} occurrences ({pct:6.3f}%)")

    print("\n[4] MODIFIERS & SPECIAL SIGNS:")
    print("-" * 60)
    for mod_hex, mod_name in MODIFIERS.items():
        cnt = mod_counts[mod_hex]
        pct = (cnt / total_c) * 100 if total_c else 0
        print(f"  {mod_hex} {mod_name:16s} (U+{ord(mod_hex):04X}) : {cnt:8,d} occurrences ({pct:6.3f}%)")

    print("\n[5] KANNADA NUMERALS FREQUENCY:")
    print("-" * 60)
    for kn_num, digit in NUMERALS.items():
        cnt = num_counts[kn_num]
        pct = (cnt / total_c) * 100 if total_c else 0
        print(f"  {kn_num} (Digit {digit}) : {cnt:8,d} occurrences ({pct:6.3f}%)")

    print("\n[6] TOP 25 CONJUNCT / OTTAKSHARA PATTERNS:")
    print("-" * 60)
    for cj, cnt in conj_counts.most_common(25):
        pct = (cnt / total_c) * 100 if total_c else 0
        print(f"  {cj:10s} : {cnt:8,d} occurrences ({pct:6.3f}%)")

    print("\n[7] RAREST CHARACTERS & LOW REPRESENTATION (<0.1% frequency):")
    print("-" * 60)
    rare_items = []
    for ch in VOWELS + CONSONANTS:
        cnt = c_counts[ch]
        pct = (cnt / total_c) * 100 if total_c else 0
        if pct < 0.15:
            rare_items.append((ch, cnt, pct))
    rare_items.sort(key=lambda x: x[1])
    for ch, cnt, pct in rare_items:
        print(f"  {ch} : {cnt:6,d} occurrences ({pct:6.4f}%) - LOW REPRESENTATION")

    print("\n[8] VISUALLY CONFUSABLE CHARACTER GROUPS DISCOVERED:")
    print("-" * 60)
    groups = [
        ("Group 1 (i-matra glyph confusion)", ["ತಿ", "ಥಿ", "ದಿ", "ನಿ", "ಲಿ", "ರಿ", "ಟಿ", "ಡಿ", "ಠಿ", "ಢಿ"]),
        ("Group 2 (Top loop / crest curvature)", ["ತ", "ದ", "ನ"]),
        ("Group 3 (Right vertical / tail curve)", ["ಲ", "ಳ", "ರ"]),
        ("Group 4 (Aspirated / Notch distinctions)", ["ಕ", "ಖ", "ಗ", "ಘ"]),
        ("Group 5 (Labial stroke closure)", ["ಪ", "ಫ", "ಬ", "ಭ", "ಮ"]),
        ("Group 6 (Retroflex loop closure)", ["ಟ", "ಠ", "ಡ", "ಢ"]),
        ("Group 7 (Sibilant loop vs notch)", ["ಶ", "ಷ", "ಸ", "ಹ"]),
        ("Group 8 (Nasal / Labial loop)", ["ಮ", "ನ"]),
    ]
    for g_title, g_chars in groups:
        counts_str = ", ".join([f"{ch}: {c_counts.get(ch, 0):,}" for ch in g_chars])
        print(f"  * {g_title:40s} -> {counts_str}")

if __name__ == "__main__":
    paths = [
        ("IIIT Train", r"c:\Land Record\training\datasets\iiit_kannada_train.jsonl"),
        ("IIIT Val", r"c:\Land Record\training\datasets\iiit_kannada_val.jsonl"),
        ("IIIT Test", r"c:\Land Record\training\datasets\iiit_kannada_test.jsonl"),
    ]
    report_data = audit_dataset(paths)
    print_audit_report(report_data)
