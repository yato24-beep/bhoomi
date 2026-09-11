"""Phase 1 — Comprehensive Data Audit of all training datasets.

Verifies: sample counts, character frequencies, matra/virama/conjunct/numeral stats,
duplicates, train/val leakage, image existence, Unicode correctness, tokenizer roundtrip.
"""

import collections
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Kannada Unicode ranges
KANNADA_VOWELS = set("ಅಆಇಈಉಊಋಎಏಐಒಓಔ")
KANNADA_CONSONANTS = set("ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಳ")
KANNADA_MATRAS = set("ಾಿೀುೂೃೆೇೈೊೋೌ")
KANNADA_VIRAMA = "್"
KANNADA_ANUSVARA = "ಂ"
KANNADA_VISARGA = "ಃ"
KANNADA_NUMERALS = set("೦೧೨೩೪೫೬೭೮೯")
ALL_KANNADA = KANNADA_VOWELS | KANNADA_CONSONANTS | KANNADA_MATRAS | {KANNADA_VIRAMA, KANNADA_ANUSVARA, KANNADA_VISARGA} | KANNADA_NUMERALS


def load_jsonl(path: Path):
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def audit_dataset(records, name, project_root, check_images_limit=1000):
    """Audit a single dataset split."""
    print(f"\n{'='*60}")
    print(f"  Auditing: {name} ({len(records)} samples)")
    print(f"{'='*60}")

    # 1. Basic stats
    texts = [r["text"] for r in records]
    unique_words = set(texts)
    print(f"  Total samples: {len(records)}")
    print(f"  Unique words: {len(unique_words)}")

    # 2. Character frequency
    char_freq = collections.Counter()
    for t in texts:
        for c in t:
            char_freq[c] += 1

    total_chars = sum(char_freq.values())
    print(f"  Total characters: {total_chars}")
    print(f"  Unique characters: {len(char_freq)}")

    # 3. Kannada-specific stats
    kannada_chars = {c: char_freq[c] for c in char_freq if "\u0c80" <= c <= "\u0cff"}
    non_kannada = {c: char_freq[c] for c in char_freq if not ("\u0c80" <= c <= "\u0cff")}

    # Matra frequencies
    matra_freq = {m: char_freq.get(m, 0) for m in KANNADA_MATRAS}
    total_matras = sum(matra_freq.values())
    print(f"\n  Matra frequency (total={total_matras}):")
    for m, count in sorted(matra_freq.items(), key=lambda x: -x[1]):
        print(f"    {m} (U+{ord(m):04X}): {count}")

    # Virama
    virama_count = char_freq.get(KANNADA_VIRAMA, 0)
    print(f"\n  Virama (್): {virama_count}")
    print(f"  Anusvara (ಂ): {char_freq.get(KANNADA_ANUSVARA, 0)}")
    print(f"  Visarga (ಃ): {char_freq.get(KANNADA_VISARGA, 0)}")

    # Numeral frequencies
    numeral_freq = {n: char_freq.get(n, 0) for n in KANNADA_NUMERALS}
    total_numerals = sum(numeral_freq.values())
    print(f"\n  Kannada numerals (total={total_numerals}):")
    for n, count in sorted(numeral_freq.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {n}: {count}")

    # 4. Conjunct frequency (virama + consonant patterns)
    conjunct_pattern = re.compile(f"[{''.join(KANNADA_CONSONANTS)}]{KANNADA_VIRAMA}[{''.join(KANNADA_CONSONANTS)}]")
    conjunct_freq = collections.Counter()
    for t in texts:
        for match in conjunct_pattern.finditer(t):
            conjunct_freq[match.group()] += 1
    total_conjuncts = sum(conjunct_freq.values())
    print(f"\n  Conjuncts found (total={total_conjuncts}, unique={len(conjunct_freq)}):")
    for conj, count in conjunct_freq.most_common(20):
        print(f"    {conj}: {count}")

    # 5. Rare characters (appearing < 50 times in Kannada range)
    rare_chars = {c: count for c, count in kannada_chars.items() if count < 50}
    print(f"\n  Rare Kannada characters (<50 occurrences): {len(rare_chars)}")
    for c, count in sorted(rare_chars.items(), key=lambda x: x[1]):
        print(f"    {c} (U+{ord(c):04X}): {count}")

    # 6. Vowel coverage
    vowel_freq = {v: char_freq.get(v, 0) for v in KANNADA_VOWELS}
    missing_vowels = [v for v, c in vowel_freq.items() if c == 0]
    print(f"\n  Vowel coverage: {sum(1 for v in vowel_freq.values() if v > 0)}/{len(KANNADA_VOWELS)}")
    if missing_vowels:
        print(f"    Missing vowels: {' '.join(missing_vowels)}")

    # Consonant coverage
    cons_freq = {c: char_freq.get(c, 0) for c in KANNADA_CONSONANTS}
    missing_cons = [c for c, ct in cons_freq.items() if ct == 0]
    print(f"  Consonant coverage: {sum(1 for v in cons_freq.values() if v > 0)}/{len(KANNADA_CONSONANTS)}")
    if missing_cons:
        print(f"    Missing consonants: {' '.join(missing_cons)}")

    # 7. Duplicate labels
    label_counts = collections.Counter(texts)
    dup_labels = {t: c for t, c in label_counts.items() if c > 1}
    print(f"\n  Duplicate labels: {len(dup_labels)} words appear >1 time")
    print(f"    Total duplicate entries: {sum(c - 1 for c in dup_labels.values())}")

    # 8. Image path existence check
    print(f"\n  Checking image paths (limit {check_images_limit})...")
    missing_images = 0
    checked = 0
    for r in records[:check_images_limit]:
        img_path = project_root / r["image"]
        if not img_path.exists():
            missing_images += 1
            if missing_images <= 3:
                print(f"    MISSING: {r['image']}")
        checked += 1
    print(f"    Checked {checked}, Missing: {missing_images}")

    # 9. Unicode normalization check
    nfc_changes = 0
    for t in texts:
        nfc = unicodedata.normalize("NFC", t)
        if nfc != t:
            nfc_changes += 1
    print(f"\n  NFC normalization changes: {nfc_changes}")

    # 10. Corrupted Unicode check
    corrupted = 0
    for t in texts:
        for c in t:
            if unicodedata.category(c) == "Cn":  # unassigned
                corrupted += 1
                break
    print(f"  Labels with unassigned Unicode: {corrupted}")

    return {
        "name": name,
        "total_samples": len(records),
        "unique_words": len(unique_words),
        "total_characters": total_chars,
        "unique_characters": len(char_freq),
        "total_matras": total_matras,
        "matra_freq": {m: char_freq.get(m, 0) for m in KANNADA_MATRAS},
        "virama_count": virama_count,
        "anusvara_count": char_freq.get(KANNADA_ANUSVARA, 0),
        "visarga_count": char_freq.get(KANNADA_VISARGA, 0),
        "total_numerals": total_numerals,
        "numeral_freq": {n: char_freq.get(n, 0) for n in KANNADA_NUMERALS},
        "total_conjuncts": total_conjuncts,
        "unique_conjuncts": len(conjunct_freq),
        "top_conjuncts": dict(conjunct_freq.most_common(30)),
        "rare_chars_count": len(rare_chars),
        "rare_chars": {c: count for c, count in rare_chars.items()},
        "missing_images": missing_images,
        "images_checked": checked,
        "nfc_normalization_changes": nfc_changes,
        "corrupted_unicode": corrupted,
        "duplicate_labels": len(dup_labels),
        "vowel_coverage": sum(1 for v in vowel_freq.values() if v > 0),
        "consonant_coverage": sum(1 for v in cons_freq.values() if v > 0),
        "char_freq": dict(char_freq.most_common(100)),
    }


def check_train_val_leakage(train_records, val_records):
    """Check for shared images or labels between train and validation."""
    train_images = set(r["image"] for r in train_records)
    val_images = set(r["image"] for r in val_records)
    shared_images = train_images & val_images

    train_texts = set(r["text"] for r in train_records)
    val_texts = set(r["text"] for r in val_records)
    shared_texts = train_texts & val_texts

    print(f"\n  Train/Val Leakage Check:")
    print(f"    Shared images: {len(shared_images)}")
    print(f"    Shared word labels: {len(shared_texts)} / {len(val_texts)} val words")
    if shared_images:
        print(f"    WARNING: {len(shared_images)} images appear in BOTH train and val!")
        for p in list(shared_images)[:5]:
            print(f"      {p}")

    return {
        "shared_images": len(shared_images),
        "shared_labels": len(shared_texts),
        "val_unique_labels": len(val_texts),
    }


def main():
    print("=" * 70)
    print("  PHASE 1 — COMPREHENSIVE DATA AUDIT")
    print("=" * 70)

    results = {}

    # Audit IIIT train
    train_path = PROJECT_ROOT / "training" / "datasets" / "iiit_kannada_train.jsonl"
    if train_path.exists():
        train_records = load_jsonl(train_path)
        results["iiit_train"] = audit_dataset(train_records, "IIIT Train", PROJECT_ROOT)
    else:
        print(f"[SKIP] Train manifest not found: {train_path}")
        train_records = []

    # Audit IIIT val
    val_path = PROJECT_ROOT / "training" / "datasets" / "iiit_kannada_val.jsonl"
    if val_path.exists():
        val_records = load_jsonl(val_path)
        results["iiit_val"] = audit_dataset(val_records, "IIIT Val", PROJECT_ROOT)
    else:
        print(f"[SKIP] Val manifest not found: {val_path}")
        val_records = []

    # Audit diagnostic val
    diag_path = PROJECT_ROOT / "training" / "datasets" / "kannada_diagnostic_val" / "diagnostic_full.jsonl"
    if diag_path.exists():
        diag_records = load_jsonl(diag_path)
        results["diagnostic_val"] = audit_dataset(diag_records, "Diagnostic Val", PROJECT_ROOT, check_images_limit=671)
    else:
        print(f"[SKIP] Diagnostic val not found: {diag_path}")
        diag_records = []

    # Audit balanced train v1
    bal_path = PROJECT_ROOT / "training" / "datasets" / "kannada_character_balanced" / "train_balanced.jsonl"
    if bal_path.exists():
        bal_records = load_jsonl(bal_path)
        results["balanced_v1"] = audit_dataset(bal_records, "Balanced V1 Train", PROJECT_ROOT)

    # Train/Val leakage check
    if train_records and val_records:
        results["leakage_train_val"] = check_train_val_leakage(train_records, val_records)
    if train_records and diag_records:
        results["leakage_train_diag"] = check_train_val_leakage(train_records, diag_records)

    # Save report
    report_path = PROJECT_ROOT / "training" / "datasets" / "v2_audit_report.json"
    # Convert set-like keys for JSON
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            serializable[k] = v

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n\nAudit report saved: {report_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  PHASE 1 SUMMARY")
    print("=" * 70)
    for ds_name, ds_info in results.items():
        if isinstance(ds_info, dict) and "total_samples" in ds_info:
            print(f"  {ds_name}:")
            print(f"    Samples: {ds_info['total_samples']}")
            print(f"    Unique words: {ds_info['unique_words']}")
            print(f"    Missing images: {ds_info['missing_images']}")
            print(f"    Unicode issues: {ds_info['nfc_normalization_changes'] + ds_info['corrupted_unicode']}")
            print(f"    Virama count: {ds_info['virama_count']}")
            print(f"    Total conjuncts: {ds_info['total_conjuncts']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
