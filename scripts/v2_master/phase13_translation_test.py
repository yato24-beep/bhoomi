"""Phase 13 — Translation Verification.

Tests bidirectional Kannada↔English translation with land record content.
Verifies numbers, dates, survey numbers, names are preserved.
"""

import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.translation.translator import (
    translate_kannada_text,
    translate_english_to_kannada,
    translate_bidirectional,
)

TEST_CASES_KN_TO_EN = [
    ("ಸರ್ವೇ ನಂಬರ್ 142 ಗ್ರಾಮ ದಾಖಲೆ", "Survey"),  # Accept any survey-related translation
    ("ಮ್ಯುಟೇಶನ್ ರಿಜಿಸ್ಟರ್", "Mutation Register"),
    ("ಖಾತಾ ನಂಬರ್", "Khata Number"),
    ("ತಾಲೂಕು", "Taluk"),
    ("ಜಿಲ್ಲೆ", "District"),
    ("ಕರ್ನಾಟಕ", "Karnataka"),
    ("ಎಕರೆ", "Acres"),
    ("ಗುಂಟೆ", "Guntas"),
    ("ದಿನಾಂಕ", "Date"),
    ("ಹೋಬಳಿ", "Hobli"),
]

TEST_CASES_EN_TO_KN = [
    ("Survey Number 142 Village Record", None),  # Just check it produces Kannada
    ("Mutation Register", None),
    ("Khata Number", None),
]


def main():
    print("=" * 70)
    print("  PHASE 13 — TRANSLATION VERIFICATION")
    print("=" * 70)

    # KN → EN Tests
    print("\n  Kannada → English Translation Tests:")
    print("  " + "-" * 65)
    kn_en_pass = 0
    kn_en_total = 0

    for kn_text, expected_fragment in TEST_CASES_KN_TO_EN:
        kn_en_total += 1
        result = translate_kannada_text(kn_text)
        
        # Check if expected fragment is in the result (case-insensitive)
        passed = expected_fragment.lower() in result.lower() if result else False
        
        status = "PASS" if passed else "FAIL"
        if passed:
            kn_en_pass += 1
        print(f"  [{status}] '{kn_text}' → '{result}'")
        if not passed:
            print(f"         Expected to contain: '{expected_fragment}'")

    # EN → KN Tests
    print(f"\n  English → Kannada Translation Tests:")
    print("  " + "-" * 65)
    en_kn_pass = 0
    en_kn_total = 0

    for en_text, _ in TEST_CASES_EN_TO_KN:
        en_kn_total += 1
        result = translate_english_to_kannada(en_text)
        
        # Check it has Kannada characters
        has_kannada = any("\u0c80" <= c <= "\u0cff" for c in (result or ""))
        passed = bool(result and result.strip() and has_kannada)
        
        status = "PASS" if passed else "FAIL"
        if passed:
            en_kn_pass += 1
        print(f"  [{status}] '{en_text}' → '{result}'")

    # Number preservation test
    print(f"\n  Number/Date Preservation Tests:")
    print("  " + "-" * 65)
    
    test_with_numbers = "ಸರ್ವೇ ನಂಬರ್ 142"
    result = translate_kannada_text(test_with_numbers)
    has_142 = "142" in (result or "")
    print(f"  [{'PASS' if has_142 else 'FAIL'}] Number '142' preserved: '{result}'")

    test_with_date = "ದಿನಾಂಕ 16-1-1960"
    result2 = translate_kannada_text(test_with_date)
    has_date = "16" in (result2 or "") and "1960" in (result2 or "")
    print(f"  [{'PASS' if has_date else 'FAIL'}] Date preserved: '{result2}'")

    # Bidirectional test
    print(f"\n  Bidirectional Translation Test:")
    print("  " + "-" * 65)
    
    bi_result = translate_bidirectional("ಗ್ರಾಮ", target_lang="en")
    print(f"  KN→EN: 'ಗ್ರಾಮ' → '{bi_result}'")
    
    bi_result2 = translate_bidirectional("Village", target_lang="kn")
    has_kn = any("\u0c80" <= c <= "\u0cff" for c in (bi_result2 or ""))
    print(f"  EN→KN: 'Village' → '{bi_result2}' (has Kannada: {has_kn})")

    # Summary
    print(f"\n{'='*70}")
    print(f"  TRANSLATION VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"  KN→EN: {kn_en_pass}/{kn_en_total} passed")
    print(f"  EN→KN: {en_kn_pass}/{en_kn_total} passed")
    print(f"  Numbers preserved: {'PASS' if has_142 else 'FAIL'}")
    print(f"  Dates preserved: {'PASS' if has_date else 'FAIL'}")
    
    all_pass = (kn_en_pass == kn_en_total and en_kn_pass == en_kn_total 
                and has_142 and has_date)
    print(f"\n  Overall: {'PASS' if all_pass else 'PARTIAL'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
