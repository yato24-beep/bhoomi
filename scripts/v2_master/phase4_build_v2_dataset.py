"""Phase 4 — Build V2 Balanced Training Dataset from full IIIT pool.

Creates training/datasets/kannada_character_balanced_v2/ with:
- train_v2.jsonl (10K-15K samples, character-balanced)
- validation_v2.jsonl (separate from diagnostic benchmark)
- audit_report.json
"""

import collections
import json
import random
import re
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Weak characters from V1 analysis (Phase 3 targets)
WEAK_CHARS = set("ತಿಲಿರಿದಿನಿಟಿಡಿರಕತದವಸನಗಮಪಫಬಭವಟಡಠಢ")
WEAK_MATRAS = set("್ಂೂಿೇೈೆುೂ್")
WEAK_CONJUNCTS = ["ಣ್ಣ", "ದ್ದ", "ಷ್ಟ", "ಕ್ಷ", "ಣ್ಣು", "ಕ್ತ", "ತ್ರ", "ದ್ಧ", "ಶ್ರ"]
KANNADA_CONSONANTS = set("ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಳ")
KANNADA_VIRAMA = "್"
KANNADA_NUMERALS = set("೦೧೨೩೪೫೬೭೮೯")

# Target minimum samples per rare character
MIN_SAMPLES_PER_RARE = 80
# Maximum oversampling ratio (don't duplicate more than 3x)
MAX_OVERSAMPLE_RATIO = 3
# Target total dataset size
TARGET_DATASET_SIZE = 12000
# Validation holdout fraction
VAL_FRACTION = 0.10


def load_jsonl(path: Path):
    """Load all records from a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def char_set(text):
    """Get the set of Kannada characters in a text."""
    return {c for c in text if "\u0c80" <= c <= "\u0cff"}


def build_char_index(records):
    """Build index: character → list of record indices containing that character."""
    index = collections.defaultdict(list)
    for idx, r in enumerate(records):
        for c in char_set(r["text"]):
            index[c].append(idx)
    return index


def build_conjunct_index(records):
    """Build index: conjunct → list of record indices containing that conjunct."""
    pattern = re.compile(f"[{''.join(KANNADA_CONSONANTS)}]{KANNADA_VIRAMA}[{''.join(KANNADA_CONSONANTS)}]")
    index = collections.defaultdict(list)
    for idx, r in enumerate(records):
        for match in pattern.finditer(r["text"]):
            index[match.group()].append(idx)
    return index


def main():
    print("=" * 70)
    print("  PHASE 4 — BUILD V2 BALANCED TRAINING DATASET")
    print("=" * 70)

    # Load full IIIT training pool
    train_path = PROJECT_ROOT / "training" / "datasets" / "iiit_kannada_train.jsonl"
    all_records = load_jsonl(train_path)
    print(f"Full IIIT training pool: {len(all_records)} samples")

    # Load diagnostic val to exclude (prevent leakage)
    diag_path = PROJECT_ROOT / "training" / "datasets" / "kannada_diagnostic_val" / "diagnostic_full.jsonl"
    diag_records = load_jsonl(diag_path) if diag_path.exists() else []
    diag_images = {r["image"] for r in diag_records}
    print(f"Diagnostic val images to exclude: {len(diag_images)}")

    # Filter out any records that are in the diagnostic benchmark
    available = [r for r in all_records if r["image"] not in diag_images]
    print(f"Available for V2 training (after exclusion): {len(available)}")

    # Build character index
    char_idx = build_char_index(available)
    conjunct_idx = build_conjunct_index(available)

    # Step 1: Character coverage analysis
    print(f"\n  Character coverage analysis:")
    char_counts = {}
    for c in sorted(char_idx.keys()):
        char_counts[c] = len(char_idx[c])
    
    # Find underrepresented characters
    rare_chars = {c: count for c, count in char_counts.items() 
                  if count < MIN_SAMPLES_PER_RARE and "\u0c80" <= c <= "\u0cff"}
    print(f"  Characters with < {MIN_SAMPLES_PER_RARE} samples: {len(rare_chars)}")
    for c, count in sorted(rare_chars.items(), key=lambda x: x[1]):
        print(f"    {c} (U+{ord(c):04X}): {count} samples")

    # Step 2: Build base selection — stratified sample from full pool
    random.seed(42)
    selected_indices = set()

    # 2a: First, ensure ALL characters are covered
    for c in sorted(char_idx.keys()):
        if "\u0c80" <= c <= "\u0cff":
            samples_for_char = char_idx[c]
            # Take up to MIN_SAMPLES_PER_RARE
            needed = min(MIN_SAMPLES_PER_RARE, len(samples_for_char))
            chosen = random.sample(samples_for_char, needed)
            selected_indices.update(chosen)

    print(f"\n  After character coverage pass: {len(selected_indices)} samples")

    # 2b: Ensure conjuncts are well-represented
    for conj in WEAK_CONJUNCTS:
        if conj in conjunct_idx:
            samples_for_conj = conjunct_idx[conj]
            # Take up to 60 per weak conjunct
            needed = min(60, len(samples_for_conj))
            chosen = random.sample(samples_for_conj, needed)
            selected_indices.update(chosen)

    print(f"  After conjunct coverage pass: {len(selected_indices)} samples")

    # 2c: Ensure numerals are well-represented
    for n in KANNADA_NUMERALS:
        if n in char_idx:
            samples_for_num = char_idx[n]
            needed = min(80, len(samples_for_num))
            chosen = random.sample(samples_for_num, needed)
            selected_indices.update(chosen)

    print(f"  After numeral coverage pass: {len(selected_indices)} samples")

    # 2d: Fill up to target with uniform random sampling
    remaining_indices = [i for i in range(len(available)) if i not in selected_indices]
    fill_needed = max(0, TARGET_DATASET_SIZE - len(selected_indices))
    if fill_needed > 0 and remaining_indices:
        fill_sample = random.sample(remaining_indices, min(fill_needed, len(remaining_indices)))
        selected_indices.update(fill_sample)

    print(f"  After fill pass: {len(selected_indices)} samples")

    # 2e: Conservative oversampling of rare characters (no more than 3x)
    oversample_records = []
    for c, count in rare_chars.items():
        if count < MIN_SAMPLES_PER_RARE and c in char_idx:
            available_for_char = [i for i in char_idx[c] if i in selected_indices]
            if not available_for_char:
                available_for_char = char_idx[c][:MIN_SAMPLES_PER_RARE]
            
            deficit = MIN_SAMPLES_PER_RARE - len(available_for_char)
            if deficit > 0:
                # Duplicate existing samples up to MAX_OVERSAMPLE_RATIO
                max_duplicates = min(deficit, len(available_for_char) * (MAX_OVERSAMPLE_RATIO - 1))
                if max_duplicates > 0:
                    dup_indices = random.choices(available_for_char, k=max_duplicates)
                    for di in dup_indices:
                        rec = dict(available[di])
                        rec["metadata"] = dict(rec.get("metadata", {}))
                        rec["metadata"]["oversampled_for"] = c
                        oversample_records.append(rec)

    print(f"  Oversampled records added: {len(oversample_records)}")

    # Build final dataset
    selected_records = [available[i] for i in sorted(selected_indices)]
    selected_records.extend(oversample_records)
    random.shuffle(selected_records)

    print(f"\n  Total V2 dataset: {len(selected_records)} samples")

    # Step 3: Split into train and validation
    val_size = max(100, int(len(selected_records) * VAL_FRACTION))
    val_records = selected_records[:val_size]
    train_records = selected_records[val_size:]

    print(f"  V2 Train: {len(train_records)} samples")
    print(f"  V2 Val: {len(val_records)} samples")

    # Step 4: Save
    output_dir = PROJECT_ROOT / "training" / "datasets" / "kannada_character_balanced_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path_out = output_dir / "train_v2.jsonl"
    val_path_out = output_dir / "validation_v2.jsonl"

    with open(train_path_out, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path_out, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  Saved: {train_path_out}")
    print(f"  Saved: {val_path_out}")

    # Step 5: Audit report
    train_chars = collections.Counter()
    for r in train_records:
        for c in r["text"]:
            if "\u0c80" <= c <= "\u0cff":
                train_chars[c] += 1

    audit = {
        "total_train": len(train_records),
        "total_val": len(val_records),
        "unique_train_words": len(set(r["text"] for r in train_records)),
        "unique_val_words": len(set(r["text"] for r in val_records)),
        "oversampled_records": len(oversample_records),
        "kannada_char_coverage": {c: train_chars.get(c, 0) for c in sorted(train_chars.keys())},
        "min_char_count": min(train_chars.values()) if train_chars else 0,
        "max_char_count": max(train_chars.values()) if train_chars else 0,
    }

    audit_path = output_dir / "audit_report.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print(f"  Audit: {audit_path}")

    # Character coverage summary
    print(f"\n  Character Coverage Summary (V2 Train):")
    print(f"    Unique Kannada chars: {len(train_chars)}")
    print(f"    Min char count: {audit['min_char_count']}")
    print(f"    Max char count: {audit['max_char_count']}")

    # Check coverage of critical characters
    print(f"\n  Critical Character Counts:")
    critical = "ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಲವಶಷಸಹಳ"
    for c in critical:
        print(f"    {c}: {train_chars.get(c, 0)}")

    print(f"\n{'='*70}")
    print(f"  PHASE 4 COMPLETE — V2 Dataset Built")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
