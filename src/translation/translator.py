"""Kannada to English translation pipeline for land records and government documents.

Preserves names, survey numbers, dates, measurements, and legal terms while providing
high-quality English translations using a domain glossary + neural online translation fallback.
"""

import logging
import re
import urllib.parse
import urllib.request
import json
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Comprehensive Karnataka Revenue / Land Records Bilingual Glossary
LAND_RECORD_GLOSSARY: Dict[str, str] = {
    # Document Types
    "ಮ್ಯುಟೇಶನ್ ರಿಜಿಸ್ಟರ್": "Mutation Register",
    "ಮ್ಯುಟೇಶನ್": "Mutation",
    "ರಿಜಿಸ್ಟರ್": "Register",
    "ಪಹಣಿ": "Pahani (RTC)",
    "ಖಾತಾ": "Khata",
    "ಖಾತೆ": "Khata",
    "ಆರ್ ಟಿ ಸಿ": "RTC (Record of Rights)",
    "ಟಿಪ್ಪಣಿ": "Tippani",
    "ಆಕಾರಬಂಧ": "Akarband",
    "ವಂಶವೃಕ್ಷ": "Family Tree (Genealogy)",
    "ಕ್ರಯಪತ್ರ": "Sale Deed",
    "ದಾನಪತ್ರ": "Gift Deed",
    "ಭಾಗಪತ್ರ": "Partition Deed",

    # Land & Cadastral Terms
    "ಸರ್ವೆ ನಂಬರ್": "Survey Number",
    "ಸರ್ವೆ ನಂ": "Survey No.",
    "ಸರ್ವೆ": "Survey",
    "ಹಿಸ್ಸಾ": "Hissa",
    "ಹಿಸ್ಸಾ ನಂ": "Hissa No.",
    "ಖಾತಾ ನಂಬರ್": "Khata Number",
    "ಖಾತಾ ನಂ": "Khata No.",
    "ವಿಸ್ತೀರ್ಣ": "Extent / Area",
    "ಕ್ಷೇತ್ರ": "Area",
    "ಎಕರೆ": "Acres",
    "ಗುಂಟೆ": "Guntas",
    "ಆಣೆ": "Anas",
    "ಖುಷ್ಕಿ": "Dry Land",
    "ತರಿ": "Wet Land",
    "ಬಾಗಾಯ್ತು": "Garden Land",
    "ಪೋಡು": "Podu",
    "ಖರಾಬು": "Kharab (Uncultivable)",
    "ಪೊಟ ಖರಾಬು": "Pota Kharab",

    # Administrative Divisions
    "ಗ್ರಾಮ": "Village",
    "ಹಳ್ಳಿ": "Village",
    "ಹೋಬಳಿ": "Hobli",
    "ತಾಲೂಕು": "Taluk",
    "ತಾಲ್ಲೂಕು": "Taluk",
    "ಜಿಲ್ಲೆ": "District",
    "ರಾಜ್ಯ": "State",
    "ಕರ್ನಾಟಕ": "Karnataka",

    # Parties & Ownership
    "ಖಾತೆದಾರರ ಹೆಸರು": "Khata Holder Name",
    "ಮಾಲೀಕರ ಹೆಸರು": "Owner Name",
    "ಹೆಸರು": "Name",
    "ತಂದೆ": "Father",
    "ತಂದೆಯ ಹೆಸರು": "Father's Name",
    "ಗಂಡ": "Husband",
    "ಗಂಡನ ಹೆಸರು": "Husband's Name",
    "ಅನುಭವದಾರರು": "Occupant / Possessor",
    "ಸ್ವಾಧೀನದಾರರು": "Possessor",
    "ಅರ್ಜಿದಾರ": "Applicant",
    "ಕಂದಾಯ": "Revenue / Tax",
    "ಆಕಾರ": "Assessment",

    # Legal & Actions
    "ದಿನಾಂಕ": "Date",
    "ಆದೇಶ": "Order",
    "ಆದೇಶ ಸಂಖ್ಯೆ": "Order Number",
    "ಮಂಜೂರಾತಿ": "Sanction / Approval",
    "ದೃಢೀಕರಣ": "Certification",
    "ಷರಾ": "Remarks",
    "ಷರಾ ಕಾಲಂ": "Remarks Column",
    "ಹಕ್ಕು": "Right / Title",
    "ಸ್ವತ್ತು": "Property",
    "ಸಹಿ": "Signature",
    "ತಹಶೀಲ್ದಾರ್": "Tahsildar",
    "ಗ್ರಾಮ ಲೆಕ್ಕಿಗ": "Village Accountant",
    "ಕಂದಾಯ ನಿರೀಕ್ಷಕ": "Revenue Inspector",
}


def _protect_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    """Protects numeric values, survey numbers, dates, and English words from translation distortion."""
    token_map: Dict[str, str] = {}
    counter = 0

    # Pattern for dates, survey numbers (e.g., 125, 0-15, 16-1-1960, S.No.125)
    def repl_num(match):
        nonlocal counter
        placeholder = f"__TOKEN_{counter}__"
        token_map[placeholder] = match.group(0)
        counter += 1
        return placeholder

    # Protect dates like DD-MM-YYYY or DD/MM/YYYY
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', repl_num, text)
    # Protect compound extent/survey formats like 125/1, 0-15, 2-10
    text = re.sub(r'\b\d+[-/]\d+\b', repl_num, text)

    return text, token_map


def _restore_tokens(text: str, token_map: Dict[str, str]) -> str:
    """Restores protected tokens back into the translated text."""
    for placeholder, original in token_map.items():
        # Match placeholder even if spaces were injected by MT engine
        regex = re.escape(placeholder).replace(r"\_", r"[\s_]*")
        text = re.sub(regex, original, text, flags=re.IGNORECASE)
    return text


def translate_kannada_to_english_online(text: str, timeout: int = 5) -> Optional[str]:
    """Translates Kannada text to English using Google's public translation endpoint."""
    if not text or not text.strip():
        return ""

    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=kn&tl=en&dt=t&q=" + urllib.parse.quote(text)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated_chunks = [item[0] for item in data[0] if item and item[0]]
            return "".join(translated_chunks)
    except Exception as exc:
        logger.debug(f"Online translation failed: {exc}")
        return None


def translate_kannada_text(text: str) -> str:
    """End-to-end robust Kannada to English translation for land records.

    Pipeline:
    1. Check if text is already English / numeric.
    2. Protect survey numbers, dates, and identifiers.
    3. Apply dictionary replacement for official land record phrases.
    4. Call online neural translation for remaining natural Kannada text.
    5. Fall back gracefully to dictionary terms if offline.
    6. Restore protected tokens.
    """
    if not text or not text.strip():
        return ""

    # If text has no Kannada unicode characters (U+0C80 to U+0CFF), return directly
    has_kannada = any('\u0c80' <= c <= '\u0cff' for c in text)
    if not has_kannada:
        return text

    # Step 1: Protect dates and numbers
    protected_text, token_map = _protect_tokens(text)

    # Step 2: Apply Domain Glossary substitutions on exact matches
    glossary_replaced = protected_text
    for kn_term, en_term in sorted(LAND_RECORD_GLOSSARY.items(), key=lambda x: -len(x[0])):
        if kn_term in glossary_replaced:
            glossary_replaced = glossary_replaced.replace(kn_term, en_term)

    # If all Kannada characters were successfully translated via domain glossary
    still_has_kannada = any('\u0c80' <= c <= '\u0cff' for c in glossary_replaced)
    if not still_has_kannada:
        return _restore_tokens(glossary_replaced, token_map)

    # Step 3: Try online translation for fluent sentence translation
    online_res = translate_kannada_to_english_online(protected_text)
    if online_res and online_res.strip():
        final_trans = _restore_tokens(online_res, token_map)
        return final_trans

    # Step 4: Fallback to glossary-replaced text
    return _restore_tokens(glossary_replaced, token_map)


def translate_english_to_kannada_online(text: str, timeout: int = 5) -> Optional[str]:
    """Translates English text to Kannada using Google's public translation endpoint."""
    if not text or not text.strip():
        return ""

    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=kn&dt=t&q=" + urllib.parse.quote(text)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            translated_chunks = [item[0] for item in data[0] if item and item[0]]
            return "".join(translated_chunks)
    except Exception as exc:
        logger.debug(f"Online English to Kannada translation failed: {exc}")
        return None


def translate_english_to_kannada(text: str) -> str:
    """Translates English land record text to Kannada."""
    if not text or not text.strip():
        return ""

    protected_text, token_map = _protect_tokens(text)
    online_res = translate_english_to_kannada_online(protected_text)
    if online_res and online_res.strip():
        return _restore_tokens(online_res, token_map)

    glossary_replaced = protected_text
    for kn_term, en_term in sorted(LAND_RECORD_GLOSSARY.items(), key=lambda x: -len(x[1])):
        if en_term.lower() in glossary_replaced.lower():
            pattern = re.compile(re.escape(en_term), re.IGNORECASE)
            glossary_replaced = pattern.sub(kn_term, glossary_replaced)

    return _restore_tokens(glossary_replaced, token_map)


def translate_bidirectional(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """General bidirectional translation helper between Kannada and English."""
    if not text or not text.strip():
        return ""
    
    has_kannada = any('\u0c80' <= c <= '\u0cff' for c in text)
    if target_lang.lower().startswith("kn") or (source_lang == "en" and not has_kannada):
        return translate_english_to_kannada(text)
    else:
        return translate_kannada_text(text)

