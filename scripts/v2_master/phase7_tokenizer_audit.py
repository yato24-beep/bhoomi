"""Phase 7 — Tokenizer Audit.

Verifies encode→decode roundtrip for every Kannada character, matra, virama, numeral, 
and representative conjuncts/words. Must find ZERO corruption.
"""

import json
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

V1_CHECKPOINT = PROJECT_ROOT / "models" / "trocr" / "kannada_generalized_checkpoints" / "best_checkpoint"

# All Kannada characters to test
TEST_CHARS = {
    "Independent Vowels": list("ಅಆಇಈಉಊಋಎಏಐಒಓಔ"),
    "Consonants Row 1 (ka)": list("ಕಖಗಘಙ"),
    "Consonants Row 2 (ca)": list("ಚಛಜಝಞ"),
    "Consonants Row 3 (Ta)": list("ಟಠಡಢಣ"),
    "Consonants Row 4 (ta)": list("ತಥದಧನ"),
    "Consonants Row 5 (pa)": list("ಪಫಬಭಮ"),
    "Consonants (ya-va)": list("ಯರಲವ"),
    "Consonants (sha-La)": list("ಶಷಸಹಳ"),
    "Matras": list("ಾಿೀುೂೃೆೇೈೊೋೌ"),
    "Virama": ["್"],
    "Anusvara/Visarga": ["ಂ", "ಃ"],
    "Kannada Numerals": list("೦೧೨೩೪೫೬೭೮೯"),
}

TEST_WORDS = [
    "ಕೋತಿ",
    "ಮರ",
    "ಹಣ್ಣು",
    "ಣ್ಣ",
    "ಷ್ಟ",
    "ಕ್ಷ",
    "ಣ್ನ",
    "ದ್ದ",
    "ಕ್ತ",
    "ತ್ರ",
    "ದ್ಧ",
    "ಶ್ರ",
    "ನಾಜೂಕಾಗಿರುವುದರಿಂದ",
    "ಗುರುತಿಸಿಕೊಂಡಮೇಲೆ",
    "ಸರ್ವೇ ನಂಬರ್",
    "ಗ್ರಾಮ ದಾಖಲೆ",
    "ಪೂಜೆಯಲ್ಲಿ",
    "ಮೊರೆಹೋಗಿದ್ದಾರೆ",
]


def main():
    print("=" * 70)
    print("  PHASE 7 — TOKENIZER AUDIT")
    print("=" * 70)

    # Load tokenizer from V1 checkpoint
    from transformers import AutoTokenizer

    print(f"Loading tokenizer from: {V1_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(str(V1_CHECKPOINT))
    print(f"  Tokenizer type: {type(tokenizer).__name__}")
    print(f"  Vocab size: {len(tokenizer)}")

    total_tests = 0
    total_failures = 0
    failures = []

    # Test individual characters
    print(f"\n{'='*60}")
    print(f"  Testing Individual Characters")
    print(f"{'='*60}")

    for category, chars in TEST_CHARS.items():
        cat_fail = 0
        for ch in chars:
            total_tests += 1
            encoded = tokenizer.encode(ch, add_special_tokens=False)
            decoded = tokenizer.decode(encoded, skip_special_tokens=True).strip()

            if decoded != ch:
                cat_fail += 1
                total_failures += 1
                failures.append({
                    "type": "character",
                    "category": category,
                    "input": ch,
                    "input_codepoint": f"U+{ord(ch):04X}",
                    "decoded": decoded,
                    "token_ids": encoded,
                })

        status = "PASS" if cat_fail == 0 else f"FAIL ({cat_fail})"
        print(f"  {category}: {status}")

    # Test words and conjuncts
    print(f"\n{'='*60}")
    print(f"  Testing Words & Conjuncts")
    print(f"{'='*60}")

    for word in TEST_WORDS:
        total_tests += 1
        encoded = tokenizer.encode(word, add_special_tokens=False)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True).strip()

        if decoded != word:
            total_failures += 1
            failures.append({
                "type": "word",
                "input": word,
                "decoded": decoded,
                "token_ids": encoded,
            })
            print(f"  FAIL: '{word}' → encode → decode → '{decoded}'")
        else:
            print(f"  PASS: '{word}' ({len(encoded)} tokens)")

    # Summary
    print(f"\n{'='*70}")
    print(f"  TOKENIZER AUDIT SUMMARY")
    print(f"{'='*70}")
    print(f"  Total tests: {total_tests}")
    print(f"  Passed: {total_tests - total_failures}")
    print(f"  Failed: {total_failures}")

    if failures:
        print(f"\n  FAILURES:")
        for f in failures:
            print(f"    {f['type']}: '{f['input']}' → '{f['decoded']}'")
    else:
        print(f"\n  ALL TESTS PASSED — Zero tokenizer corruption.")

    # Save report
    report_path = PROJECT_ROOT / "evaluation" / "v2_tokenizer_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total_tests": total_tests,
        "passed": total_tests - total_failures,
        "failed": total_failures,
        "failures": failures,
        "tokenizer_type": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "checkpoint": str(V1_CHECKPOINT),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
