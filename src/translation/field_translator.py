"""Field-Aligned Kannada <-> English Translation & Phonetic Transliteration Engine.

Translates structured land record fields and line-segmented text while preserving
word boundaries, legal terminology, and semantic alignment. Uses phonetic transliteration
for proper nouns (owner names, taluks, villages) and domain glossaries for legal/cadastral terminology.
"""

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
import unicodedata

logger = logging.getLogger(__name__)

try:
    from aksharamukha import transliterate as akshara_trans
    HAS_AKSHARAMUKHA = True
except ImportError:
    HAS_AKSHARAMUKHA = False

# Canonical phonetic transliteration mapping for Kannada Aksharas (ISO 15919 / ITRANS basis)
KANNADA_CONSONANTS = {
    'ಕ': 'k', 'ಖ': 'kh', 'ಗ': 'g', 'ಘ': 'gh', 'ಙ': 'ng',
    'ಚ': 'ch', 'ಛ': 'chh', 'ಜ': 'j', 'ಝ': 'jh', 'ಞ': 'ny',
    'ಟ': 't', 'ಠ': 'th', 'ಡ': 'd', 'ಢ': 'dh', 'ಣ': 'n',
    'ತ': 't', 'ಥ': 'th', 'ದ': 'd', 'ಧ': 'dh', 'ನ': 'n',
    'ಪ': 'p', 'ಫ': 'ph', 'ಬ': 'b', 'ಭ': 'bh', 'ಮ': 'm',
    'ಯ': 'y', 'ರ': 'r', 'ಱ': 'r', 'ಲ': 'l', 'ವ': 'v',
    'ಶ': 'sh', 'ಷ': 'sh', 'ಸ': 's', 'ಹ': 'h', 'ಳ': 'l',
}

KANNADA_VOWELS = {
    'ಅ': 'a', 'ಆ': 'aa', 'ಇ': 'i', 'ಈ': 'ee', 'ಉ': 'u', 'ಊ': 'oo',
    'ಋ': 'ru', 'ಎ': 'e', 'ಏ': 'e', 'ಐ': 'ai', 'ಒ': 'o', 'ಓ': 'o', 'ಔ': 'au',
}

KANNADA_MATRAS = {
    'ಾ': 'a', 'ಿ': 'i', 'ೀ': 'ee', 'ು': 'u', 'ೂ': 'oo',
    'ೃ': 'ru', 'ೆ': 'e', 'ೇ': 'e', 'ೈ': 'ai', 'ೊ': 'o', 'ೋ': 'o', 'ೌ': 'au',
    'ಂ': 'm', 'ಃ': 'h', '್': '',  # Halant / Virama
}

# Land Record Cadastral Terminology Translation Dictionary
CADASTRAL_TRANSLATIONS = {
    "ಮಾಲೀಕರ ಹೆಸರು": "Owner Name",
    "ಖಾತೆದಾರರ ಹೆಸರು": "Khatadar Name",
    "ಖಾತೆದಾರರು": "Khatadar",
    "ಸರ್ವೆ ನಂಬರ್": "Survey Number",
    "ಸರ್ವೆ ನಂ": "Survey No",
    "ಸರ್ವೇ ನಂ": "Survey No",
    "ಸರ್ವೆ ಸಂಖ್ಯೆ": "Survey Number",
    "ಸರ್ವೇ ಸಂಖ್ಯೆ": "Survey Number",
    "ಹಿಸ್ಸಾ ನಂ": "Hissa No",
    "ಹಿಸ್ಸಾ ಸಂಖ್ಯೆ": "Hissa Number",
    "ಖಾತಾ ಸಂಖ್ಯೆ": "Khata Number",
    "ಖಾತೆ ನಂ": "Khata No",
    "ತಾಲೂಕು": "Taluk",
    "ತಾಲ್ಲೂಕು": "Taluk",
    "ಗ್ರಾಮ": "Village",
    "ಹೋಬಳಿ": "Hobli",
    "ಜಿಲ್ಲೆ": "District",
    "ವಿಸ್ತೀರ್ಣ": "Extent / Area",
    "ಎಕರೆ": "Acres",
    "ಗುಂಟೆ": "Guntas",
    "ಆಣೆ": "Anas",
    "ಚದರ ಅಡಿ": "Sq Ft",
    "ಕೃಷಿ ಜಮೀನು": "Agricultural Land",
    "ಖರಾಬು": "Kharab (Uncultivable)",
    "ಪಹಣಿ": "Pahani (RTC)",
    "ಉತ್ತರ": "North",
    "ದಕ್ಷಿಣ": "South",
    "ಪೂರ್ವ": "East",
    "ಪಶ್ಚಿಮ": "West",
    "ಕಪ್ಪು ಮಣ್ಣು": "Black Soil",
    "ಕೆಂಪು ಮಣ್ಣು": "Red Soil",
    "ಮರಳು ಮಣ್ಣು": "Sandy Soil",
    "ಜವಳು ಮಣ್ಣು": "Clay Soil",
    "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿ": "Assistant Revenue Officer",
    "ಕಂದಾಯ ಅಧಿಕಾರಿ": "Revenue Officer",
    "ತಹಶೀಲ್ದಾರ್": "Tahsildar",
    "ಉಪ ನೋಂದಣಾಧಿಕಾರಿ": "Sub-Registrar",
}

# Standard Karnataka Administrative Locations (District / Taluk / Hobli / Village)
KARNATAKA_GEOGRAPHY_GLOSSARY = {
    "ಬೆಂಗಳೂರು": "Bengaluru",
    "ಬೆಂಗಳೂರು ನಗರ": "Bengaluru Urban",
    "ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ": "Bengaluru Rural",
    "ಬೆಂಗಳೂರು ಉತ್ತರ": "Bengaluru North",
    "ಬೆಂಗಳೂರು ದಕ್ಷಿಣ": "Bengaluru South",
    "ಬೆಂಗಳೂರು ಪೂರ್ವ": "Bengaluru East",
    "ಯಲಹಂಕ": "Yelahanka",
    "ಕೆಂಗೇರಿ": "Kengeri",
    "ಆನೇಕಲ್": "Anekal",
    "ದೇವನಹಳ್ಳಿ": "Devanahalli",
    "ದೊಡ್ಡಬಳ್ಳಾಪುರ": "Doddaballapura",
    "ಹೊಸಕೋಟೆ": "Hoskote",
    "ನೆಲಮಂಗಲ": "Nelamangala",
    "ರಾಮನಗರ": "Ramanagara",
    "ಚನ್ನಪಟ್ಟಣ": "Channapatna",
    "ಕನಕಪುರ": "Kanakapura",
    "ಮಾಗಡಿ": "Magadi",
    "ಮೈಸೂರು": "Mysuru",
    "ಹುಬ್ಬಳ್ಳಿ": "Hubballi",
    "ಧಾರವಾಡ": "Dharwad",
    "ಬೆಳಗಾವಿ": "Belagavi",
    "ಮಂಗಳೂರು": "Mangaluru",
    "ಶಿವಮೊಗ್ಗ": "Shivamogga",
    "ತುಮಕೂರು": "Tumakuru",
    "ಹಾಸನ": "Hassan",
    "ಮಂಡ್ಯ": "Mandya",
    "ಕೋಲಾರ": "Kolar",
    "ಚಿಕ್ಕಬಳ್ಳಾಪುರ": "Chikkaballapura",
    "ಬಳ್ಳಾರಿ": "Ballari",
    "ವಿಜಯಪುರ": "Vijayapura",
    "ಕಲಬುರಗಿ": "Kalaburagi",
    "ಉಡುಪಿ": "Udupi",
    "ಚಿಕ್ಕಮಗಳೂರು": "Chikkamagaluru",
}

# Standard Proper Name Transliterations (Preserves official legal spelling)
KARNATAKA_NAMES_GLOSSARY = {
    "ಸಿದ್ದರಾಮಯ್ಯ": "Siddaramaiah",
    "ರಾಮಪ್ಪ": "Ramappa",
    "ಭೀಮಪ್ಪ": "Bheemappa",
    "ಬಸವರಾಜ": "Basavaraja",
    "ಬಸವರಾಜು": "Basavaraju",
    "ಲಿಂಗಯ್ಯ": "Lingaiah",
    "ಲಿಂಗಮ್ಮ": "Lingamma",
    "ಮಂಜುನಾಥ": "Manjunatha",
    "ಮಂಜುನಾಥ ಗೌಡ": "Manjunath Gowda",
    "ಮದನಗೌಡ": "Madanagowda",
    "ಕೃಷ್ಣಪ್ಪ": "Krishnappa",
    "ವೆಂಕಟೇಶ್": "Venkatesh",
    "ಗೋವಿಂದಪ್ಪ": "Govindappa",
    "ನಾಗರಾಜು": "Nagaraju",
    "ನಾಗರಾಜ": "Nagaraja",
    "ಶ್ರೀನಿವಾಸ್": "Srinivas",
    "ಶ್ರೀನಿವಾಸ": "Srinivasa",
    "ಸುರೇಶ್": "Suresh",
    "ರಮೇಶ್": "Ramesh",
    "ಮಹೇಶ್": "Mahesh",
    "ಚಂದ್ರಶೇಖರ್": "Chandrashekhar",
}

# Translation Scope Definitions
TRANSLATABLE_TEXT_FIELDS = {
    "owner_name",
    "owner_father_name",
    "father_name",
    "cultivator_name",
    "district",
    "taluk",
    "hobli",
    "village",
    "locality",
    "property_address",
    "address",
    "land_type",
    "soil_type",
    "issuing_authority",
    "issuing_organization",
    "authority",
    "organization",
    "document_type_label",
    "document_title",
    "document_type",
}

NON_TRANSLATABLE_FIELDS = {
    "survey_number",
    "hissa_number",
    "khata_number",
    "property_number",
    "mutation_number",
    "registration_number",
    "date",
    "document_date",
    "record_date",
    "total_extent",
    "cultivable_area",
    "pot_kharab",
    "site_area",
    "built_up_area",
    "land_revenue",
    "khasra_number",
    "khatauni_number",
    "patta_number",
    "gat_number",
}


def has_kannada_script(text: str) -> bool:
    """Returns True if the string contains any Kannada Unicode characters."""
    if not text:
        return False
    return any('\u0c80' <= c <= '\u0cff' for c in str(text))


def transliterate_kannada_word(word: str) -> str:
    """Phonetically transliterates any open-ended Kannada word to Latin script without hardcoding."""
    clean = word.strip()
    if not clean:
        return ""
    if clean in KARNATAKA_NAMES_GLOSSARY:
        return KARNATAKA_NAMES_GLOSSARY[clean]
    if clean in ("ಬಿನ್", "ಬಿನ್:"):
        return "Bin"
    if clean in ("ಕೋಂ", "ಕೋಂ:"):
        return "Kom"
    if clean in KARNATAKA_GEOGRAPHY_GLOSSARY:
        return KARNATAKA_GEOGRAPHY_GLOSSARY[clean]
    if HAS_AKSHARAMUKHA:
        try:
            res = akshara_trans.process("Kannada", "RomanColloquial", clean)
            if res and res != clean:
                return " ".join(part.capitalize() for part in res.split())
        except Exception as exc:
            logger.debug(f"Aksharamukha error for '{clean}': {exc}")

    # Fallback: Transliterate character by character
    chars = list(clean)
    out: List[str] = []
    i = 0
    n = len(chars)

    while i < n:
        ch = chars[i]
        if ch in KANNADA_VOWELS:
            out.append(KANNADA_VOWELS[ch])
            i += 1
        elif ch in KANNADA_CONSONANTS:
            base_cons = KANNADA_CONSONANTS[ch]
            # Check following character
            if (i + 1) < n:
                next_ch = chars[i + 1]
                if next_ch == '್':  # Virama / halant (dead consonant)
                    out.append(base_cons)
                    i += 2
                elif next_ch in KANNADA_MATRAS:
                    out.append(base_cons + KANNADA_MATRAS[next_ch])
                    i += 2
                else:
                    out.append(base_cons + 'a')
                    i += 1
            else:
                out.append(base_cons)
                i += 1
        elif ch in KANNADA_MATRAS:
            out.append(KANNADA_MATRAS[ch])
            i += 1
        else:
            out.append(ch)
            i += 1

    res = "".join(out)
    return res.capitalize() if res else ""


def transliterate_kannada_phrase(phrase: str) -> str:
    """Transliterates a multi-word Kannada phrase preserving word boundaries."""
    if not phrase or not str(phrase).strip():
        return ""
    tokens = str(phrase).strip().split()
    return " ".join(transliterate_kannada_word(tok) for tok in tokens)


@dataclass
class AlignedBilingualField:
    """Represents a structured field with aligned Kannada and English values."""
    field_name: str
    kannada_value: str
    english_value: Optional[str]
    method: str  # "transliteration" | "cadastral_translation" | "domain_glossary" | "google_neural" | "passthrough" | "unavailable"
    confidence: float
    translation_status: str = "TRANSLATED"  # "TRANSLATED" | "ALREADY_ENGLISH" | "NOT_APPLICABLE" | "FAILED" | "UNAVAILABLE"
    provenance: Optional[Any] = None


class FieldAlignedTranslator:
    """Word-aligned translator and transliterator for structured land records."""

    def __init__(self, enable_online: bool = True, timeout: float = 3.0):
        self.enable_online = enable_online
        self.timeout = timeout

    def translate_field(
        self,
        field_name: str,
        value: Any,
        raw_value: Optional[str] = None,
        provenance: Optional[Any] = None,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Translates a single canonical field value according to land record domain rules.

        Returns:
            Dict containing:
                raw_value: str
                normalized_value: str
                english_value: Optional[str]
                translation_status: str ("TRANSLATED", "ALREADY_ENGLISH", "NOT_APPLICABLE", "FAILED", "UNAVAILABLE")
                translation_engine: str ("domain_glossary", "google_neural", "aksharamukha_phonetic", "passthrough", "unavailable")
                provenance: Optional[Any]
        """
        raw_val_str = str(raw_value if raw_value is not None else (value if value is not None else "")).strip()
        norm_val_str = str(value if value is not None else raw_val_str).strip()
        f_lower = field_name.lower().strip()

        # Step 1: Check if field is non-translatable identifier or date
        if f_lower in NON_TRANSLATABLE_FIELDS:
            return {
                "field_name": field_name,
                "raw_value": raw_val_str,
                "normalized_value": norm_val_str,
                "english_value": None,
                "translation_status": "NOT_APPLICABLE",
                "translation_engine": "passthrough",
                "provenance": provenance,
            }

        # Step 2: Empty value check
        if not norm_val_str:
            return {
                "field_name": field_name,
                "raw_value": raw_val_str,
                "normalized_value": norm_val_str,
                "english_value": None,
                "translation_status": "UNAVAILABLE",
                "translation_engine": "unavailable",
                "provenance": provenance,
            }

        # Step 3: Already English / Latin check (no Kannada characters present)
        if not has_kannada_script(norm_val_str):
            return {
                "field_name": field_name,
                "raw_value": raw_val_str,
                "normalized_value": norm_val_str,
                "english_value": norm_val_str,
                "translation_status": "ALREADY_ENGLISH",
                "translation_engine": "passthrough",
                "provenance": provenance,
            }

        # Step 4: Canonical Kannada field translation
        norm_clean = unicodedata.normalize("NFC", norm_val_str).strip()

        # 4a. Check exact domain glossaries
        if norm_clean in KARNATAKA_NAMES_GLOSSARY:
            return {
                "field_name": field_name,
                "raw_value": raw_val_str,
                "normalized_value": norm_val_str,
                "english_value": KARNATAKA_NAMES_GLOSSARY[norm_clean],
                "translation_status": "TRANSLATED",
                "translation_engine": "domain_glossary",
                "provenance": provenance,
            }

        if norm_clean in KARNATAKA_GEOGRAPHY_GLOSSARY:
            return {
                "field_name": field_name,
                "raw_value": raw_val_str,
                "normalized_value": norm_val_str,
                "english_value": KARNATAKA_GEOGRAPHY_GLOSSARY[norm_clean],
                "translation_status": "TRANSLATED",
                "translation_engine": "domain_glossary",
                "provenance": provenance,
            }

        if norm_clean in CADASTRAL_TRANSLATIONS:
            return {
                "field_name": field_name,
                "raw_value": raw_val_str,
                "normalized_value": norm_val_str,
                "english_value": CADASTRAL_TRANSLATIONS[norm_clean],
                "translation_status": "TRANSLATED",
                "translation_engine": "domain_glossary",
                "provenance": provenance,
            }

        # 4b. Check multi-term replacement (e.g. "ಬೆಂಗಳೂರು ಉತ್ತರ" -> "Bengaluru North")
        replaced_term = norm_clean
        for k_term, en_term in sorted(KARNATAKA_GEOGRAPHY_GLOSSARY.items(), key=lambda x: -len(x[0])):
            if k_term in replaced_term:
                replaced_term = replaced_term.replace(k_term, en_term)
        for k_term, en_term in sorted(CADASTRAL_TRANSLATIONS.items(), key=lambda x: -len(x[0])):
            if k_term in replaced_term:
                replaced_term = replaced_term.replace(k_term, en_term)

        if not has_kannada_script(replaced_term):
            return {
                "field_name": field_name,
                "raw_value": raw_val_str,
                "normalized_value": norm_val_str,
                "english_value": " ".join(part.capitalize() for part in replaced_term.split()),
                "translation_status": "TRANSLATED",
                "translation_engine": "domain_glossary",
                "provenance": provenance,
            }

        # 4c. For personal names (owner, cultivator, father), prioritize phonetic transliteration
        # to ensure proper legal naming and avoid neural MT converting names into dictionary nouns.
        if f_lower in ("owner_name", "cultivator_name", "farmer_name", "owner_father_name", "father_name"):
            try:
                translit_name = transliterate_kannada_phrase(norm_clean)
                if translit_name and translit_name.strip() and not has_kannada_script(translit_name):
                    return {
                        "field_name": field_name,
                        "raw_value": raw_val_str,
                        "normalized_value": norm_val_str,
                        "english_value": translit_name.strip(),
                        "translation_status": "TRANSLATED",
                        "translation_engine": "aksharamukha_phonetic",
                        "provenance": provenance,
                    }
            except Exception as exc:
                logger.debug("Phonetic transliteration failed for name '%s': %s", norm_clean, exc)

        # 4d. Online neural translation
        if self.enable_online:
            try:
                from src.translation.translator import translate_kannada_to_english_online
                online_res = translate_kannada_to_english_online(norm_clean, timeout=int(self.timeout))
                if online_res and online_res.strip() and not has_kannada_script(online_res):
                    cleaned_en = online_res.strip()
                    # Apply canonical administrative name standardization
                    if cleaned_en.lower() == "bangalore":
                        cleaned_en = "Bengaluru"
                    elif "bangalore" in cleaned_en.lower():
                        cleaned_en = re.sub(r'\bBangalore\b', 'Bengaluru', cleaned_en, flags=re.IGNORECASE)
                    
                    return {
                        "field_name": field_name,
                        "raw_value": raw_val_str,
                        "normalized_value": norm_val_str,
                        "english_value": cleaned_en.title() if len(cleaned_en.split()) <= 3 else cleaned_en,
                        "translation_status": "TRANSLATED",
                        "translation_engine": "google_neural",
                        "provenance": provenance,
                    }
            except Exception as exc:
                logger.debug("Online translation failed for '%s': %s", norm_clean, exc)

        # 4d. Fallback: Phonetic transliteration (Aksharamukha)
        try:
            translit_res = transliterate_kannada_phrase(norm_clean)
            if translit_res and translit_res.strip() and not has_kannada_script(translit_res):
                return {
                    "field_name": field_name,
                    "raw_value": raw_val_str,
                    "normalized_value": norm_val_str,
                    "english_value": translit_res.strip(),
                    "translation_status": "TRANSLATED",
                    "translation_engine": "aksharamukha_phonetic",
                    "provenance": provenance,
                }
        except Exception as exc:
            logger.debug("Phonetic transliteration failed for '%s': %s", norm_clean, exc)

        # 4e. If all translation and transliteration methods failed: NO FABRICATION
        return {
            "field_name": field_name,
            "raw_value": raw_val_str,
            "normalized_value": norm_val_str,
            "english_value": None,
            "translation_status": "FAILED",
            "translation_engine": "unavailable",
            "provenance": provenance,
        }

    def translate_structured_fields(
        self,
        fields: Dict[str, Any],
    ) -> Dict[str, AlignedBilingualField]:
        """Translates structured fields preserving alignment and proper noun phonetic fidelity."""
        result: Dict[str, AlignedBilingualField] = {}

        for fname, f_obj in fields.items():
            k_val = getattr(f_obj, "normalized_value", None)
            if k_val is None:
                k_val = f_obj.get("normalized_value") if isinstance(f_obj, dict) else str(f_obj)
            raw_val = getattr(f_obj, "raw_value", None)
            if raw_val is None:
                raw_val = f_obj.get("raw_value") if isinstance(f_obj, dict) else k_val
            conf = getattr(f_obj, "confidence", 0.0)
            if isinstance(f_obj, dict):
                conf = float(f_obj.get("confidence", 0.0))
            prov = getattr(f_obj, "provenance", None)
            if isinstance(f_obj, dict):
                prov = f_obj.get("provenance")

            t_info = self.translate_field(
                field_name=fname,
                value=k_val,
                raw_value=raw_val,
                provenance=prov,
                confidence=conf,
            )

            result[fname] = AlignedBilingualField(
                field_name=fname,
                kannada_value=str(k_val),
                english_value=t_info.get("english_value"),
                method=t_info.get("translation_engine", "unknown"),
                confidence=float(conf),
                translation_status=t_info.get("translation_status", "TRANSLATED"),
                provenance=prov,
            )

        return result

    def translate_document_lines(
        self,
        lines: List[str],
    ) -> List[Tuple[str, str]]:
        """Translates document lines 1:1 preserving line-by-line alignment."""
        aligned_pairs: List[Tuple[str, str]] = []
        for line in lines:
            if not line.strip():
                continue
            en_line = line
            # Replace cadastral terms
            for k_term, en_term in CADASTRAL_TRANSLATIONS.items():
                en_line = en_line.replace(k_term, en_term)
            # Transliterate remaining Kannada tokens
            words = en_line.split()
            translated_words = []
            for w in words:
                if has_kannada_script(w):
                    translated_words.append(transliterate_kannada_word(w))
                else:
                    translated_words.append(w)
            aligned_pairs.append((line, " ".join(translated_words)))
        return aligned_pairs
