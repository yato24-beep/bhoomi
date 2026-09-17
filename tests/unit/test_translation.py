"""Unit tests for Land Record Translation Quality Upgrade.

Validates:
1. Step A (Contextual Text Normalization): repair of broken character sequences,
   merged words, missing spaces, and stray punctuation using semantic evidence or standalone fallback.
2. Step B (Meaning-Preserving Translation): translation for meaning (not transliteration),
   with verbatim preservation of names, survey numbers, khata numbers, dates, addresses, and official IDs (BBMP).
3. Step C (Hard Language-Purity Validation): flagging of stray English fragments in Kannada
   and untranslated Kannada in English, with zero silent masking.
4. Distinct 3-Representation Integrity: original_ocr, kannada_translation, english_translation.
5. Regression testing on clean and noisy Kannada land records.
"""

import pytest

from src.translation.translator import (
    LanguagePurityReport,
    TranslationResult,
    clean_kannada_translation,
    normalize_ocr_text,
    protect_verbatim_entities,
    restore_verbatim_entities,
    translate_document_text,
    translate_kannada_text,
    translate_normalized_kannada_to_english,
    validate_language_purity,
)


def test_step_a_noisy_kannada_ocr_normalization():
    """Validates Step A: contextual cleanup pass repairing broken Kannada character sequences,
    merged words, missing spaces, and stray OCR noise.
    """
    noisy_ocr_snippet = (
        "ಮೃಹಮೃ ಬೆಂಗಳೂದು ನೇಬಾತ್ ಬೃಹತ್ಬೆಂಗಳೂದುಮಹಾನಗರಪಾ\n"
        "ಖಾತಾದಾಖಲೆಒದಗಿಸುವಬಗ್ಗೆ\n"
        "ದನಾ0ಕ:20-09-2024\n"
        "ಸೈಬ್ವೇ-ನಂಬರ್-.12/3- ನಿವೇಶನ\n"
        "ಬಾತಾಸಂಭೈ:68-76-470/a\n"
        "ವಾರ್ಡ್ಸಂಖ್ಯೆ:148-Ejipura\n"
        "ನಿವೇಶನದವಿಸ್ತೀರ್ಣ:2200.00 Sq Ft\n"
        "ಕಟಡದ ಎನೀಣ೯:500.00 Sq Ft\n"
        "ಸಹಾಯಕಕಂದಾಯಿಅಧಕಾರ\n"
        "Gy. Ayo? Wmt?\n"
    )

    clean_kannada, token_map, is_fallback = normalize_ocr_text(
        raw_ocr_text=noisy_ocr_snippet,
        semantic_fields={
            "survey_number": {"raw_value": "12/3"},
            "khata_number": {"raw_value": "68-76-470/a"},
            "date": {"raw_value": "20-09-2024"},
        },
    )

    # Fallback should be False since semantic fields were provided
    assert is_fallback is False

    # 1. Broken sequences repaired
    assert "ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ" in clean_kannada
    assert "ಖಾತಾ ದಾಖಲೆ ಒದಗಿಸುವ ಬಗ್ಗೆ" in clean_kannada
    assert "ದಿನಾಂಕ: 20-09-2024" in clean_kannada
    assert "ಸರ್ವೆ ನಂಬರ್" in clean_kannada
    assert "ಖಾತಾ ಸಂಖ್ಯೆ: 68-76-470/a" in clean_kannada or "68-76-470/a" in clean_kannada
    assert "ವಾರ್ಡ್ ಸಂಖ್ಯೆ: 148-Ejipura" in clean_kannada or "148-Ejipura" in clean_kannada
    assert "ನಿವೇಶನದ ವಿಸ್ತೀರ್ಣ: 2200.00 Sq Ft" in clean_kannada or "2200.00 Sq Ft" in clean_kannada
    assert "ಕಟ್ಟಡದ ವಿಸ್ತೀರ್ಣ: 500.00 Sq Ft" in clean_kannada or "500.00 Sq Ft" in clean_kannada
    assert "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿ" in clean_kannada

    # 2. Stray OCR noise tokens stripped
    assert "Gy." not in clean_kannada
    assert "Ayo?" not in clean_kannada
    assert "Wmt?" not in clean_kannada

    # 3. Missing spaces between colons and tokens fixed
    assert "ದನಾ0ಕ:20-09-2024" not in clean_kannada
    assert "ದಿನಾಂಕ: 20-09-2024" in clean_kannada


def test_step_a_standalone_normalization_fallback():
    """Validates that when semantic evidence is not provided, normalization falls back gracefully
    and correctly flags fallback_normalization = True.
    """
    unannotated_text = "ಬೃಹತ್ಬೆಂಗಳೂರುಮಹಾನಗರಪಾಲಿಕೆ ದಿನಾಂಕ:15-08-2023 ಸರ್ವೆನಂಬರ್:45"

    clean_kannada, _, is_fallback = normalize_ocr_text(unannotated_text)

    assert is_fallback is True
    assert "ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ" in clean_kannada
    assert "ದಿನಾಂಕ: 15-08-2023" in clean_kannada
    assert "ಸರ್ವೆ ನಂಬರ್: 45" in clean_kannada


def test_step_b_verbatim_entity_preservation():
    """Validates Step B: meaning-preserving translation with strict verbatim preservation
    of names, survey numbers, khata numbers, dates, addresses, and official identifiers.
    """
    mixed_source = (
        "ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ\n"
        "ಖಾತಾ ಪ್ರಮಾಣ ಪತ್ರ\n"
        "ದಿನಾಂಕ: 20-09-2024\n"
        "Mrs. Dorothy Charles ರವರ ಹೆಸರಿನಲ್ಲಿ ದಾಖಲಾಗಿರುತ್ತದೆ\n"
        "ಆಸ್ತಿ ಸಂಖ್ಯೆ: 68-76-470/a\n"
        "ಸರ್ವೆ ನಂಬರ್: 12/3\n"
        "ವಾರ್ಡ್ ಸಂಖ್ಯೆ: 148-Ejipura\n"
        "ನಿವೇಶನದ ವಿಸ್ತೀರ್ಣ: 2200.00 Sq Ft\n"
        "ಕಟ್ಟಡದ ವಿಸ್ತೀರ್ಣ: 500.00 Sq Ft\n"
        "ಸ್ಥಳ: S.T. Bed, Koramangala, Bengaluru\n"
        "ಇದನ್ನು ಇತರ ಕಾನೂನು ಉದ್ದೇಶಗಳಿಗೆ ಬಳಸಿಕೊಳ್ಳಬಹುದು.\n"
        "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿ"
    )

    res = translate_document_text(
        text=mixed_source,
        semantic_fields={
            "owner_name": {"raw_value": "Mrs. Dorothy Charles"},
            "khata_number": {"raw_value": "68-76-470/a"},
            "survey_number": {"raw_value": "12/3"},
            "date": {"raw_value": "20-09-2024"},
            "ward": {"raw_value": "148-Ejipura"},
        },
    )

    en = res.english_translation
    kn = res.kannada_translation

    # 1. Owner Name preserved verbatim in both English and Kannada
    assert "Mrs. Dorothy Charles" in en
    assert "Mrs. Dorothy Charles" in kn

    # 2. Alphanumeric identifiers preserved verbatim
    assert "68-76-470/a" in en
    assert "68-76-470/a" in kn
    assert "12/3" in en
    assert "12/3" in kn

    # 3. Dates preserved verbatim
    assert "20-09-2024" in en
    assert "20-09-2024" in kn

    # 4. Official acronyms & Ward preserved verbatim
    assert "BBMP" in en or "Bruhat Bengaluru Mahanagara Palike" in en
    assert "148-Ejipura" in en
    assert "148-Ejipura" in kn

    # 5. Measurements preserved verbatim
    assert "2200.00 Sq Ft" in en or "2200.00" in en
    assert "500.00 Sq Ft" in en or "500.00" in en

    # 6. Meaning-preserving translation (not transliteration)
    assert "Khata Certificate" in en
    assert "Property Number" in en
    assert "Survey Number" in en
    assert "Ward Number" in en
    assert "Built-up Area" in en
    assert "Assistant Revenue Officer" in en
    assert "This can be used for other legal purposes." in en


def test_step_c_language_purity_validation_passes_clean_document():
    """Validates that a clean, properly translated document passes language purity checks."""
    kannada_text = (
        "ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ\n"
        "ಖಾತಾ ಪ್ರಮಾಣ ಪತ್ರ\n"
        "ದಿನಾಂಕ: 20-09-2024\n"
        "Mrs. Dorothy Charles ರವರ ಹೆಸರಿನಲ್ಲಿ ದಾಖಲಾಗಿರುತ್ತದೆ\n"
        "ಆಸ್ತಿ ಸಂಖ್ಯೆ: 68-76-470/a\n"
        "ಸರ್ವೆ ನಂಬರ್: 12/3\n"
        "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿ"
    )

    english_text = (
        "Bruhat Bengaluru Mahanagara Palike (BBMP)\n"
        "Khata Certificate\n"
        "Date: 20-09-2024\n"
        "is registered in the name of Mrs. Dorothy Charles\n"
        "Property Number: 68-76-470/a\n"
        "Survey Number: 12/3\n"
        "Assistant Revenue Officer"
    )

    purity = validate_language_purity(
        kannada_text=kannada_text,
        english_text=english_text,
        allowed_verbatim_tokens=["Mrs. Dorothy Charles", "68-76-470/a", "12/3", "20-09-2024"],
    )

    assert purity.is_kannada_pure is True
    assert purity.is_english_pure is True
    assert purity.quality == "high"
    assert purity.requires_review is False
    assert len(purity.warnings) == 0


def test_step_c_language_purity_validation_flags_stray_english_in_kannada():
    """Validates that stray English fragments in Kannada output trigger quality warning
    and human review flag.
    """
    kannada_with_stray_english = (
        "ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ\n"
        "ಖಾತಾ ಪ್ರಮಾಣ ಪತ್ರ\n"
        "strayfragment unverifiedwords ದಿನಾಂಕ: 20-09-2024"
    )
    english_text = (
        "Bruhat Bengaluru Mahanagara Palike\n"
        "Khata Certificate\n"
        "Date: 20-09-2024"
    )

    purity = validate_language_purity(
        kannada_text=kannada_with_stray_english,
        english_text=english_text,
        allowed_verbatim_tokens=["20-09-2024"],
    )

    assert purity.is_kannada_pure is False
    assert purity.quality == "low / needs review"
    assert purity.requires_review is True
    assert any("stray English fragments" in w for w in purity.warnings)


def test_step_c_language_purity_validation_flags_untranslated_kannada_in_english():
    """Validates that untranslated Kannada characters remaining in English output
    are NOT silently deleted, but are explicitly flagged as low quality / needs review.
    """
    kannada_text = "ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ ದಿನಾಂಕ: 20-09-2024"
    # Defective English output with untranslated Kannada characters
    defective_english = "Bruhat Bengaluru Corporation ದಿನಾಂಕ: 20-09-2024"

    purity = validate_language_purity(
        kannada_text=kannada_text,
        english_text=defective_english,
        allowed_verbatim_tokens=["20-09-2024"],
    )

    assert purity.is_english_pure is False
    assert purity.quality == "low / needs review"
    assert purity.requires_review is True
    assert any("Contains untranslated Kannada" in w for w in purity.warnings)


def test_three_distinct_representations_never_collapse():
    """Validates that original_ocr, kannada_translation, and english_translation
    remain three distinct representations and never collapse into one another.
    """
    raw_ocr = (
        "ಬೃಹತ್ಬೆಂಗಳೂದುಮಹಾನಗರಪಾ\n"
        "ದನಾ0ಕ:20-09-2024\n"
        "ಸೈಬ್ವೇ-ನಂಬರ್-.12/3- ನಿವೇಶನ\n"
        "Mrs. Dorothy Charles"
    )

    result = translate_document_text(text=raw_ocr)

    # 1. Verify original_ocr is preserved verbatim
    assert result.original_ocr == raw_ocr
    assert "ಬೃಹತ್ಬೆಂಗಳೂದುಮಹಾನಗರಪಾ" in result.original_ocr
    assert "ದನಾ0ಕ:20-09-2024" in result.original_ocr

    # 2. Verify kannada_translation is cleaned and normalized
    assert "ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ" in result.kannada_translation
    assert "ದಿನಾಂಕ: 20-09-2024" in result.kannada_translation
    assert "ಸರ್ವೆ ನಂಬರ್" in result.kannada_translation

    # 3. Verify english_translation is clean, meaning-preserving English
    assert "Bruhat Bengaluru Mahanagara Palike" in result.english_translation or "BBMP" in result.english_translation
    assert "Date: 20-09-2024" in result.english_translation
    assert "Survey Number" in result.english_translation
    assert "Mrs. Dorothy Charles" in result.english_translation

    # 4. Strict inequality check: None of the three should equal each other
    assert result.original_ocr != result.kannada_translation
    assert result.kannada_translation != result.english_translation
    assert result.original_ocr != result.english_translation
