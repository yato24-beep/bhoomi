"""Kannada to English translation pipeline for land records and government documents.

Preserves names, survey numbers, dates, measurements, and legal terms while providing
high-quality English translations using a domain glossary + neural online translation fallback.
Enforces contextual OCR normalization (Step A), meaning-preserving translation (Step B),
and hard post-translation language-purity validation (Step C).
"""

import json
import logging
import re
import unicodedata
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Strict Pydantic Data Contracts for Translation System
# ============================================================================

class LanguagePurityReport(BaseModel):
    """Post-translation language purity validation audit report."""
    is_kannada_pure: bool = True
    is_english_pure: bool = True
    kannada_warnings: List[str] = Field(default_factory=list)
    english_warnings: List[str] = Field(default_factory=list)
    quality: str = "high"  # "high" | "low / needs review"
    requires_review: bool = False
    warnings: List[str] = Field(default_factory=list)


class TranslationResult(BaseModel):
    """Comprehensive output capturing the 3 distinct representations and validation metadata."""
    original_ocr: str = Field(..., description="1. Original raw recognized OCR text")
    kannada_translation: str = Field(..., description="2. Clean, normalized, natural Kannada text")
    english_translation: str = Field(..., description="3. Clean, natural, meaning-preserving English text")
    purity_report: LanguagePurityReport = Field(default_factory=LanguagePurityReport)
    fallback_normalization: bool = Field(False, description="True if fallback standalone pass was used without semantic evidence")
    protected_tokens: Dict[str, str] = Field(default_factory=dict, description="Verbatim protected entities")

    @property
    def clean_kannada_text(self) -> str:
        return self.kannada_translation

    @property
    def translated_text(self) -> str:
        return self.english_translation


# ============================================================================
# Comprehensive Karnataka Revenue / Land Records & Municipal Bilingual Glossary
# ============================================================================

LAND_RECORD_GLOSSARY: Dict[str, str] = {
    # Full Administrative Clauses & Phrases
    "ಖಾತಾದಾಖಲೆ ಒದಗಿಸುವ ಬಗ್ಗೆ": "Regarding provision of Khata document",
    "ಖಾತಾ ದಾಖಲೆ ಒದಗಿಸುವ ಬಗ್ಗೆ": "Regarding provision of Khata document",
    "ಖಾತಾ ದಾಖಲೆ ಒದಗಿಸುವಬಗ್ಗೆ": "Regarding provision of Khata document",
    "ಇದನ್ನು ಇತರ ಕಾನೂನು ಉದ್ದೇಶಗಳಿಗೆ ಬಳಸಿಕೊಳ್ಳಬಹುದು": "This can be used for other legal purposes.",
    "ಇದನ್ನು ಇತರ ಕಾನೂನು ಉದ್ದೇಶಗಳಿಗೆ ಬಳಸಿಕೊಳ್ಳಬಹುದು.": "This can be used for other legal purposes.",
    "ರವರ ಹೆಸರಿನಲ್ಲಿ ದಾಖಲಾಗಿರುತ್ತದೆ": "is registered in the name of",
    "ಹೆಸರಿನಲ್ಲಿ ದಾಖಲಾಗಿರುತ್ತದೆ": "is registered in the name of",
    "ರವರ ಹೆಸರಿನಲ್ಲಿ": "in the name of",
    "ದಾಖಲಾಗಿರುತ್ತದೆ": "is registered",
    "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿಗಳ ಕಚೇರಿ": "Office of the Assistant Revenue Officer",
    "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿ": "Assistant Revenue Officer",
    "ಕಂದಾಯ ಅಧಿಕಾರಿಗಳ ಕಚೇರಿ": "Office of the Revenue Officer",
    "ಕಂದಾಯ ಅಧಿಕಾರಿ": "Revenue Officer",
    "ಬೆಂಗಳೂರು ದಕ್ಷಿಣ ವಿಭಾಗ": "Bengaluru South Division",
    "ಜಯನಗರ ಉಪವಿಭಾಗ": "Jayanagar Sub-Division",
    "ಕೋರಮಂಗಲ ಉಪವಿಭಾಗ": "Koramangala Sub-Division",
    "ಉಪವಿಭಾಗ": "Sub-Division",
    "ಉಪ-ವಿಭಾಗ": "Sub-Division",
    "ದಕ್ಷಿಣ ವಿಭಾಗ": "South Division",
    "ವಿಭಾಗ": "Division",

    # BBMP & Municipal Terms
    "ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ": "Bruhat Bengaluru Mahanagara Palike (BBMP)",
    "ಬೃಹತ್ ಬೆಂಗಳೂರು": "Bruhat Bengaluru",
    "ಮಹಾನಗರ ಪಾಲಿಕೆ": "City Municipal Corporation",
    "ಮಹಾನಗರ": "Metropolitan",
    "ಪಾಲಿಕೆ": "Corporation / Municipal Body",
    "ದೃಢೀಕರಣ ಪತ್ರ": "Certificate / Attestation",
    "ಪ್ರಮಾಣ ಪತ್ರ": "Certificate",
    "ಖಾತಾ ಪ್ರಮಾಣ ಪತ್ರ": "Khata Certificate",
    "ಖಾತಾ ಪ್ರಮಾಣಪತ್ರ": "Khata Certificate",
    "ಖಾತಾ ದೃಢೀಕರಣ ಪತ್ರ": "Khata Certificate",
    "ಆಸ್ತಿ ಸಂಖ್ಯೆ": "Property Number",
    "ಸ್ವತ್ತಿನ ಸಂಖ್ಯೆ": "Property Number",
    "ವಾರ್ಡ್ ಸಂಖ್ಯೆ": "Ward Number",
    "ವಾರ್ಡ್ ಹೆಸರು": "Ward Name",
    "ಕಟ್ಟಡದ ವಿಸ್ತೀರ್ಣ": "Built-up Area",
    "ಕಟ್ಟಡ ವಿಸ್ತೀರ್ಣ": "Built-up Area",
    "ನಿವೇಶನದ ವಿಸ್ತೀರ್ಣ": "Site / Plot Area",
    "ನಿವೇಶನ ವಿಸ್ತೀರ್ಣ": "Site / Plot Area",
    "ಕರ ವಸೂಲಿ": "Tax Collection",
    "ಆಸ್ತಿ ತೆರಿಗೆ": "Property Tax",
    "ತೆರಿಗೆ": "Tax",
    "ದೃಢೀಕರಿಸಲಾಗಿದೆ": "Certified / Attested",
    "ವಲಯ": "Zone",

    # Document Types
    "ಮ್ಯುಟೇಶನ್ ರಿಜಿಸ್ಟರ್": "Mutation Register",
    "ಮ್ಯುಟೇಶನ್": "Mutation",
    "ರಿಜಿಸ್ಟರ್": "Register",
    "ಪಹಣಿ": "Pahani (RTC)",
    "ಖಾತಾ": "Khata",
    "ಖಾತೆ": "Khata",
    "ಆರ್.ಟಿ.ಸಿ.": "RTC (Record of Rights)",
    "ಆರ್ ಟಿ ಸಿ": "RTC (Record of Rights)",
    "ಆರ್ಟಿಸಿ": "RTC (Record of Rights)",
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
    "ಹಿಸ್ಸಾ ನಂಬರ್": "Hissa Number",
    "ಹಿಸ್ಸಾ ನಂ": "Hissa No.",
    "ಹಿಸ್ಸಾ": "Hissa",
    "ಖಾತಾ ನಂಬರ್": "Khata Number",
    "ಖಾತಾ ನಂ": "Khata No.",
    "ಖಾತಾ ಸಂಖ್ಯೆ": "Khata Number",
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
    "ಚದರ ಅಡಿ": "Sq Ft",
    "ಚದರ ಮೀಟರ್": "Sq Meter",
    "ನಿವೇಶನ": "Site / Plot",
    "ಕಟ್ಟಡ": "Building",
    "ಪ್ಲಾಟ್": "Plot",
    "ಸೈಟ್": "Site",
    "ರಲ್ಲಿರುವ": "situated in / located at",

    # Administrative Divisions
    "ಗ್ರಾಮ": "Village",
    "ಹಳ್ಳಿ": "Village",
    "ಹೋಬಳಿ": "Hobli",
    "ತಾಲೂಕು": "Taluk",
    "ತಾಲ್ಲೂಕು": "Taluk",
    "ಜಿಲ್ಲೆ": "District",
    "ರಾಜ್ಯ": "State",
    "ಕರ್ನಾಟಕ": "Karnataka",
    "ಬೆಂಗಳೂರು ನಗರ": "Bengaluru Urban",
    "ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ": "Bengaluru Rural",
    "ಬೆಂಗಳೂರು ದಕ್ಷಿಣ": "Bengaluru South",
    "ಬೆಂಗಳೂರು ಪೂರ್ವ": "Bengaluru East",
    "ಬೆಂಗಳೂರು ಉತ್ತರ": "Bengaluru North",
    "ಬೆಂಗಳೂರು": "Bengaluru",
    "ಸ್ಥಳ": "Location",

    # Parties & Ownership
    "ಖಾತೆದಾರರ ಹೆಸರು": "Khata Holder Name",
    "ಖಾತೆದಾರರ": "Khata Holder",
    "ಮಾಲೀಕರ ಹೆಸರು": "Owner Name",
    "ಮಾಲೀಕರು": "Owner",
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
    "ವಿಷಯ": "Subject",
    "ದೂರವಾಣಿ": "Telephone",
    "ಆದೇಶ": "Order",
    "ಆದೇಶ ಸಂಖ್ಯೆ": "Order Number",
    "ಮಂಜೂರಾತಿ": "Sanction / Approval",
    "ದೃಢೀಕರಣ": "Certification",
    "ಷರಾ": "Remarks",
    "ಷರಾ ಕಾಲಂ": "Remarks Column",
    "ಹಕ್ಕು": "Right / Title",
    "ಸ್ವತ್ತು": "Property",
    "ಸಹಿ": "Signature",
    "ಅಧಿಕಾರಿಯ ಸಹಿ": "Officer's Signature",
    "ತಹಶೀಲ್ದಾರ್": "Tahsildar",
    "ಗ್ರಾಮ ಲೆಕ್ಕಿಗ": "Village Accountant",
    "ಕಂದಾಯ ನಿರೀಕ್ಷಕ": "Revenue Inspector",
    "ಉಪ ನೋಂದಣಾಧಿಕಾರಿ": "Sub-Registrar",
}


# ============================================================================
# Step A: Contextual Text Normalization & Verbatim Protection
# ============================================================================

def protect_verbatim_entities(
    text: str,
    semantic_entities: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, str]]:
    """Protects numeric values, survey numbers, khata numbers, dates, measurements,
    official abbreviations (BBMP, RTC, PID), and proper names with collision-free tokens.

    Preserves exact verbatim content without reformatting.
    """
    token_map: Dict[str, str] = {}
    counter = 0

    def repl_token(val: str) -> str:
        nonlocal counter
        placeholder = f"__VERBATIM_TOKEN_{counter}__"
        token_map[placeholder] = val
        counter += 1
        return placeholder

    protected_text = text

    # 1. First priority: Pre-protect known semantic entities from extraction evidence
    if semantic_entities:
        # Sort descending by length to avoid partial substring collisions
        clean_entities = sorted(
            {str(e).strip() for e in semantic_entities if e and len(str(e).strip()) >= 2},
            key=lambda x: -len(x)
        )
        for ent in clean_entities:
            # Only match exact entity boundaries
            if ent in protected_text:
                pattern = re.compile(re.escape(ent))
                matches = list(pattern.finditer(protected_text))
                for m in reversed(matches):
                    placeholder = repl_token(m.group(0))
                    start, end = m.span()
                    protected_text = protected_text[:start] + placeholder + protected_text[end:]

    # 2. Standalone measurements like 2200.00 Sq Ft, 500.00 Sq Ft, 1200 Sq Ft, 900 Sq M
    def match_repl(m):
        return repl_token(m.group(0))

    protected_text = re.sub(
        r'\b\d+(?:\.\d+)?\s*(?:Sq\.?\s*Ft\.?|Sq\.?\s*M\.?|sqft|sqm|acres?|guntas?)\b',
        match_repl,
        protected_text,
        flags=re.IGNORECASE,
    )

    # 3. Dates like DD-MM-YYYY, DD/MM/YYYY, DD--MM--YYYY, YYYY-MM-DD
    protected_text = re.sub(
        r'\b\d{1,2}(?:--|-|/|\.)\d{1,2}(?:--|-|/|\.)\d{2,4}\b|\b\d{4}(?:--|-|/|\.)\d{1,2}(?:--|-|/|\.)\d{1,2}\b',
        match_repl,
        protected_text,
    )

    # 4. Alphanumeric property, survey, and khata numbers like 68-76-470/a, 470/A, 12/3, 213/2022-23, PID-1234
    protected_text = re.sub(
        r'\b\d+/\d+\b|\b\d+/[A-Za-z]\b|\b[A-Za-z0-9]+(?:[/-][A-Za-z0-9]+)+\b|\bPID[\s\-_:]*[A-Za-z0-9]+\b',
        match_repl,
        protected_text,
        flags=re.IGNORECASE,
    )

    # 5. English person names with honorifics (e.g. Mrs. Dorothy Charles, Sri. Ramesh B N)
    protected_text = re.sub(
        r'\b(?:Mrs?|Miss|Dr|Sri|Smt|Shri)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        match_repl,
        protected_text,
    )

    # 6. Multi-word proper names (e.g. Dorothy Charles)
    protected_text = re.sub(
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        match_repl,
        protected_text,
    )

    # 7. English address and location lines (e.g. S.T. Bed, Koramangala, Bengaluru)
    def repl_address(m):
        leading = m.group(1)
        prefix = m.group(2)
        addr = m.group(3).strip()
        return f"{leading}{prefix} {repl_token(addr)}"

    protected_text = re.sub(
        r'(^|\n|[ \t]+)(ಸ್ಥಳ:|ಸ್ಥಳ\s*:|ವಿಳಾಸ:|ವಿಳಾಸ\s*:|Location:|Address:)\s*([A-Za-z0-9\s.,\-\&]+)(?=\n|$)',
        repl_address,
        protected_text,
    )

    # 8. Official municipal and administrative acronyms
    protected_text = re.sub(
        r'\b(?:BBMP|RTC|PID|ARO|BDA|BMRDA|KHB|UIDAI)\b',
        match_repl,
        protected_text,
    )

    # 9. Emails and official domains
    protected_text = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|\b[a-zA-Z0-9.-]+\.gov\.in\b',
        match_repl,
        protected_text,
    )

    return protected_text, token_map


def restore_verbatim_entities(text: str, token_map: Dict[str, str]) -> str:
    """Restores protected verbatim entities exactly into the text."""
    if not text or not token_map:
        return text or ""

    restored = text
    # Sort descending by placeholder length
    for placeholder, original in sorted(token_map.items(), key=lambda x: -len(x[0])):
        # Match standard placeholder with or without leading/trailing underscores or symbols
        num_m = re.search(r'\d+', placeholder)
        if num_m:
            token_num = num_m.group(0)
            pat = re.compile(rf'(?:<<<|__|_)?VERBATIM_TOKEN_{token_num}(?:>>>|__|_)?', re.IGNORECASE)
            restored = pat.sub(original, restored)
        else:
            regex = re.escape(placeholder).replace(r"\_", r"[\s_]*")
            restored = re.sub(regex, original, restored, flags=re.IGNORECASE)
    return restored


def normalize_ocr_text(
    raw_ocr_text: str,
    semantic_fields: Optional[Any] = None,
    semantic_evidence: Optional[Any] = None,
    regions: Optional[Sequence[Any]] = None,
) -> Tuple[str, Dict[str, str], bool]:
    """Step A (Text Normalization): Performs contextual cleanup pass on raw OCR text before translation.

    - Reuses semantic evidence / context-window output from extraction pipeline if available.
    - If semantic evidence is not available, falls back to standalone normalization with fallback notice.
    - Fixes missing spaces around punctuation and between Kannada/English/number tokens.
    - Repairs merged words and broken Kannada character sequences.
    - Strips OCR noise artifacts while preserving verbatim entities.

    Returns:
        Tuple of (clean_normalized_kannada_text, token_map, fallback_normalization_flag)
    """
    if not raw_ocr_text or not str(raw_ocr_text).strip():
        return "", {}, False

    # Extract semantic entities if available
    semantic_entities: List[str] = []
    fallback_normalization = True

    if semantic_fields:
        fallback_normalization = False
        if isinstance(semantic_fields, dict):
            for field_key, field_val in semantic_fields.items():
                if isinstance(field_val, dict):
                    raw_val = field_val.get("raw_value") or field_val.get("extracted_value")
                    norm_val = field_val.get("normalized_value")
                    span = field_val.get("evidence_span")
                    for v in [raw_val, norm_val, span]:
                        if v and str(v).strip() and len(str(v).strip()) >= 2:
                            semantic_entities.append(str(v).strip())
                elif hasattr(field_val, "raw_value") or hasattr(field_val, "extracted_value"):
                    raw_val = getattr(field_val, "raw_value", None) or getattr(field_val, "extracted_value", None)
                    norm_val = getattr(field_val, "normalized_value", None)
                    span = getattr(field_val, "evidence_span", None)
                    for v in [raw_val, norm_val, span]:
                        if v and str(v).strip() and len(str(v).strip()) >= 2:
                            semantic_entities.append(str(v).strip())
        elif isinstance(semantic_fields, list):
            for field_item in semantic_fields:
                val = getattr(field_item, "raw_value", None) or getattr(field_item, "original_value", None)
                if val and str(val).strip():
                    semantic_entities.append(str(val).strip())

    if semantic_evidence:
        fallback_normalization = False
        if isinstance(semantic_evidence, list):
            for ev in semantic_evidence:
                span = getattr(ev, "evidence_span", None) or (ev.get("evidence_span") if isinstance(ev, dict) else None)
                if span and str(span).strip():
                    semantic_entities.append(str(span).strip())

    if fallback_normalization:
        logger.info("Normalization Pass: Standalone fallback normalization used (no semantic extraction evidence provided).")
    else:
        logger.info(f"Normalization Pass: Contextual normalization active with {len(semantic_entities)} semantic anchors.")

    # 1. Protect verbatim entities before applying regex transformations
    raw_normalized = unicodedata.normalize("NFC", raw_ocr_text)
    protected_text, token_map = protect_verbatim_entities(raw_normalized, semantic_entities=semantic_entities)

    lines = [l.strip() for l in protected_text.split("\n") if l.strip()]
    cleaned_lines = []

    # Comprehensive Kannada Land Record OCR Corrections & Merge Repairs
    CORRECTIONS: List[Tuple[str, str]] = [
        # Administrative & Municipal Bodies
        (r"(?:ಸಹಾಯಕ|ಸಹ)?\s*ಕಂದಾಯ\s*ಅಧ?ಿ?ಕಾರ?ಿ?ಗಳ\s*ಕಛೀ?ೇ?ರಿ", "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿಗಳ ಕಚೇರಿ"),
        (r"ಸಹಕಂದಾಯಅಧಕಾಂಗಳಕಛೀರ|ಸಹಾಯಕಕಂದಾಯಅಧಿಕಾರಿಗಳಕಚೇರಿ", "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿಗಳ ಕಚೇರಿ"),
        (r"ಸಹಾಯಕಕಂದಾಯಿಅಧಕಾರ|ಸಹಾಯಕಕಂದಾಯಅಧಿಕಾರಿ", "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿ"),
        (r"ಕಂದಾಯಅಧಿಕಾರಿ|ಕಂದಾಯಿಅಧಕಾರ", "ಕಂದಾಯ ಅಧಿಕಾರಿ"),
        (r"ಬೃಹತ್\s*ಬೆಂಗಳೂ[ದುರು]\s*ಮಹಾನಗರ\s*ಪಾ[ಲಿಕೆ]*|ಬೃಹತ್ಬೆಂಗಳೂದುಮಹಾನಗರಪಾ|ಬೃಹತ್ಬೆಂಗಳೂರುಮಹಾನಗರಪಾಲಿಕೆ|ಮೃಹಮೃ\s*ಬಂಗಳೂದು\s*ನೇಬಾತ್", "ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ"),
        (r"ಬೆಂಗಳೂರುದಿಣಏಭಾಗ|ಬೆಂಗಳೂರುದಕ್ಷಿಣವಿಭಾಗ|ಬೆಂಗಳೂರು\s*ದಕ್ಷಿಣ\s*ವಿಭಾಗ", "ಬೆಂಗಳೂರು ದಕ್ಷಿಣ ವಿಭಾಗ"),
        (r"ಜಯನಗರಿಉಪಏಭಾಗ|ಜಯನಗರಉಪಏಭಾಗ|ಜಯನಗರಉಪವಿಭಾಗ|ಜಯನಗರ\s*ಉಪ\-?ವಿಭಾಗ", "ಜಯನಗರ ಉಪವಿಭಾಗ"),
        (r"ಕೋರಮಂಗಲಉಪವಿಭಾಗ", "ಕೋರಮಂಗಲ ಉಪವಿಭಾಗ"),
        (r"ಉಪಏಭಾಗ|ಉಪ\-ವಿಭಾಗ", "ಉಪವಿಭಾಗ"),

        # Document & Record Terms
        (r"ಖಾತಾದಾಖಲೆಒದಗಿಸುವಬಗ್ಗೆ|ಬಾತಾದಾಭಲಿಒದಂಸುವಬಗೆ|ಖಾತಾದಾಖಲೆಒದಗಿಸುವಬಗೆ", "ಖಾತಾ ದಾಖಲೆ ಒದಗಿಸುವ ಬಗ್ಗೆ"),
        (r"ಖಾತಾಪ್ರಮಾಣಪತ್ರ|ಖಾತಾದೃಢೀಕರಣಪತ್ರ", "ಖಾತಾ ಪ್ರಮಾಣ ಪತ್ರ"),
        (r"ಪ್ರಮಾಣಪತ್ರ", "ಪ್ರಮಾಣ ಪತ್ರ"),
        (r"ದೃಢೀಕರಣಪತ್ರ", "ದೃಢೀಕರಣ ಪತ್ರ"),
        (r"ಸರ್ವೆನಂಬರ್|ಸೈಬ್ವೇ[\-\s]*ನಂಬರ್[\-\s.]*|ಸವee\s*ಸoui", "ಸರ್ವೆ ನಂಬರ್"),
        (r"ಖಾತಾನಂಬರ್|ಬಾತಾಸಂಭೈ|ಬಾತಾಸಂಖ್ಯೆ", "ಖಾತಾ ಸಂಖ್ಯೆ"),
        (r"ಆಸ್ತಿಸಂಖ್ಯೆ|ಸ್ವತ್ತಿನಸಂಖ್ಯೆ", "ಆಸ್ತಿ ಸಂಖ್ಯೆ"),
        (r"ವಾರ್ಡ್ಸಂಖ್ಯೆ", "ವಾರ್ಡ್ ಸಂಖ್ಯೆ"),
        (r"ವಾರ್ಡ್ಹೆಸರು", "ವಾರ್ಡ್ ಹೆಸರು"),
        (r"ಕಟ್ಟಡದವಿಸ್ತೀರ್ಣ|ಕಟಡದ\s*ಎನೀಣ೯", "ಕಟ್ಟಡದ ವಿಸ್ತೀರ್ಣ"),
        (r"ನಿವೇಶನದವಿಸ್ತೀರ್ಣ|ನವೇಶನದ\s*ಏ೯ೀಣ೯|ನವೇಶನದ\s*ವಿಸ್ತೀರ್ಣ", "ನಿವೇಶನದ ವಿಸ್ತೀರ್ಣ"),
        (r"ಕರವಸೂಲಿ", "ಕರ ವಸೂಲಿ"),
        (r"ಆಸ್ತಿತೆರಿಗೆ", "ಆಸ್ತಿ ತೆರಿಗೆ"),

        # Broken Character Sequences
        (r"ಎನಾ೦ಕ|ಎನಾಂಕ|ದನಾ0ಕ|bನಾoಕ", "ದಿನಾಂಕ"),
        (r"ಏಷಯಃ?|ವಿಷಯಃ", "ವಿಷಯ:"),
        (r"ರಲ[\-_]ಇದುವ|ರಲಇದುವ", "ರಲ್ಲಿರುವ"),
        (r"ನವೇಶನಫ|ನವೇಶನ", "ನಿವೇಶನ"),
        (r"ಬೆಂಗಳೂದು", "ಬೆಂಗಳೂರು"),
        (r"ದೂರವಾಣ:?", "ದೂರವಾಣಿ:"),
        (r"ಇದನು[\.]?ಇತರಕಾನೂನುಉದೀತಗಳಿಗೆಿ?ಬಳಸಕೊಳೃಬಹುದು|ಇದನ್ನು\s*ಇತರ\s*ಕಾನೂನು\s*ಉದ್ದೇಶಗಳಿಗೆ\s*ಬಳಸಿಕೊಳ್ಳಬಹುದು", "ಇದನ್ನು ಇತರ ಕಾನೂನು ಉದ್ದೇಶಗಳಿಗೆ ಬಳಸಿಕೊಳ್ಳಬಹುದು."),
        (r"ಹೆಸರಿನಲ್ಲಿದಾಖಲಾಗಿರುತ್ತದೆ", "ಹೆಸರಿನಲ್ಲಿ ದಾಖಲಾಗಿರುತ್ತದೆ"),
        (r"ರವರಹೆಸರಿನಲ್ಲಿ", "ರವರ ಹೆಸರಿನಲ್ಲಿ"),
        (r"ಆರ್\s*ಟಿ\s*ಸಿ", "ಆರ್.ಟಿ.ಸಿ."),
    ]

    for line in lines:
        cur_line = line

        # Apply specific OCR merge & corruption corrections
        for pat, repl in CORRECTIONS:
            cur_line = re.sub(pat, repl, cur_line, flags=re.UNICODE)

        # Collapse duplicate title fragments
        cur_line = re.sub(r'ಮೃಹಮೃ\s*ಬೆಂಗಳೂ[ದುರು]\s*ನೇಬಾತ್\s*', '', cur_line)
        cur_line = re.sub(r'(ಬೃಹತ್\s*ಬೆಂಗಳೂರು\s*ಮಹಾನಗರ\s*ಪಾಲಿಕೆ\s*){2,}', 'ಬೃಹತ್ ಬೆಂಗಳೂರು ಮಹಾನಗರ ಪಾಲಿಕೆ ', cur_line)

        # Fix spacing around punctuation: colon, comma, semicolon, danda
        cur_line = re.sub(r'(\S)([:;,|])', r'\1\2 ', cur_line)
        cur_line = re.sub(r'([:;,|\u0964])([^\s])', r'\1 \2', cur_line)

        # Fix spacing between Kannada characters and verbatim tokens or Latin words
        cur_line = re.sub(r'([\u0C80-\u0CFF])(_*VERBATIM_TOKEN_\d+_*|__VERBATIM_TOKEN_\d+__)', r'\1 \2', cur_line)
        cur_line = re.sub(r'(_*VERBATIM_TOKEN_\d+_*|__VERBATIM_TOKEN_\d+__)([\u0C80-\u0CFF])', r'\1 \2', cur_line)
        cur_line = re.sub(r'([\u0C80-\u0CFF])([A-Za-z0-9])', r'\1 \2', cur_line)
        cur_line = re.sub(r'([A-Za-z0-9])([\u0C80-\u0CFF])', r'\1 \2', cur_line)

        # Remove isolated OCR noise tokens while keeping legitimate content
        cur_line = re.sub(
            r'(?:^|\s)(?:Hi\.|Gy\.|Aವee್|Ayo\?|Wmt\?|Holli!|\*\*\*|\-\-\-|===)(?=\s|$)',
            ' ',
            cur_line,
        )
        cur_line = re.sub(r'\b(?:Hi|Gy|Aವee್|Ayo|Wmt|Holli)\b[?.!]?', ' ', cur_line)
        cur_line = re.sub(r'[\*\-=]{2,}', ' ', cur_line)

        # Clean multiple spaces and whitespace
        cur_line = re.sub(r'[ \t]+', ' ', cur_line).strip()

        if cur_line:
            cleaned_lines.append(cur_line)

    normalized_protected = "\n".join(cleaned_lines)
    # The clean Kannada translation is the normalized Kannada with tokens restored
    clean_kannada = restore_verbatim_entities(normalized_protected, token_map)

    return clean_kannada, token_map, fallback_normalization


# ============================================================================
# Step B: Meaning-Preserving Translation
# ============================================================================

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


def _clean_english_postprocess(text: str) -> str:
    """Refines machine-translated English output to standard Indian administrative & land terminology.

    Never deletes Kannada characters silently; preserves them for Step C purity validation.
    """
    if not text:
        return ""

    # Map erroneous generic translations to standard Karnataka revenue terminology
    text = re.sub(r'\bBarangay\b', 'Layout', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcertificate of account\b|\baccount certificate\b', 'Khata Certificate', text, flags=re.IGNORECASE)
    text = re.sub(r'\baccount document\b|\baccount record\b', 'Khata Document', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAccount of\b', 'Khata of', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAccount No\.?\b', 'Khata No.', text, flags=re.IGNORECASE)
    text = re.sub(r'\bBuilding\s+area\b', 'Built-up Area', text, flags=re.IGNORECASE)
    text = re.sub(r'\bSite/Plot area\b|\bPlot area\b|\bSite\s+area\b', 'Site Area', text, flags=re.IGNORECASE)
    text = re.sub(r'\bExtent / Area\b', 'Extent', text, flags=re.IGNORECASE)

    # Standardize Bruhat Bengaluru Mahanagara Palike (BBMP)
    text = re.sub(
        r'\bGreater\s+Bangalore\s+Metropolitan\s+Corporation\b|\bGreater\s+Bengaluru\s+Municipal\s+Corporation\b|\bBangalore\s+Municipal\s+Corporation\b|\bBangalore\s+Metropolitan\s+Corporation\b',
        'Bruhat Bengaluru Mahanagara Palike (BBMP)',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'\bGreater\s+Bangalore\b|\bGreater\s+Bengaluru\b',
        'Bruhat Bengaluru',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r'\bIt\s+can\s+be\s+used\s+for\s+other\s+legal\s+purposes\b',
        'This can be used for other legal purposes',
        text,
        flags=re.IGNORECASE,
    )

    # Clean double punctuation or double spaces
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'[ \t]+', ' ', text).strip()
    return text


def translate_normalized_kannada_to_english(
    normalized_kannada_text: str,
    token_map: Optional[Dict[str, str]] = None,
) -> str:
    """Step B (Meaning-Preserving Translation): Translates normalized Kannada text into clean, fluent English.

    - Operates ONLY on normalized text from Step A, never raw OCR.
    - Translates for semantic meaning rather than transliteration.
    - Preserves verbatim entities (names, survey numbers, dates, addresses, BBMP, etc.).
    - Does NOT silently delete untranslated Kannada script.
    """
    if not normalized_kannada_text or not normalized_kannada_text.strip():
        return ""

    # If tokens are not yet protected, protect them
    active_token_map = token_map or {}
    if not active_token_map or not any(k in normalized_kannada_text for k in active_token_map.keys()):
        protected_text, new_tokens = protect_verbatim_entities(normalized_kannada_text)
        active_token_map = {**active_token_map, **new_tokens}
    else:
        protected_text = normalized_kannada_text

    lines = [l.strip() for l in protected_text.split("\n") if l.strip()]
    translated_lines: List[str] = []

    for line in lines:
        # If line contains no Kannada characters at all, it's pure English/numbers
        if not any('\u0c80' <= c <= '\u0cff' for c in line):
            translated_lines.append(line)
            continue

        # 1. Attempt online neural translation
        online_line = translate_kannada_to_english_online(line)
        if online_line and online_line.strip():
            # Check if online translation succeeded in translating Kannada
            refined_online = _clean_english_postprocess(online_line.strip())
            translated_lines.append(refined_online)
            continue

        # 2. High-precision offline glossary translation (matching longest phrases first)
        line_replaced = line
        for kn_term, en_term in sorted(LAND_RECORD_GLOSSARY.items(), key=lambda x: -len(x[0])):
            if kn_term in line_replaced:
                line_replaced = line_replaced.replace(kn_term, en_term)

        # 3. Post-process terminology refinement
        clean_en_line = _clean_english_postprocess(line_replaced)
        translated_lines.append(clean_en_line)

    translated_text = "\n".join(translated_lines)
    # Restore verbatim entities back into English translation
    restored_english = restore_verbatim_entities(translated_text, active_token_map)
    final_english = _clean_english_postprocess(restored_english)
    return final_english


# ============================================================================
# Step C: Hard Post-Translation Language-Purity Validation
# ============================================================================

def validate_language_purity(
    kannada_text: str,
    english_text: str,
    allowed_verbatim_tokens: Optional[List[str]] = None,
) -> LanguagePurityReport:
    """Step C (Language Purity Validation): Validates language integrity of translations.

    - Kannada output must not contain stray English fragments (except allowed verbatim entities).
    - English output must not contain untranslated Kannada script.
    - If either fails, flags the translation quality as "low / needs review".
    """
    report = LanguagePurityReport()

    # Build set of allowed Latin words / phrases
    allowed_words: Set[str] = {
        "bbmp", "rtc", "pid", "aro", "bda", "bmrda", "khb", "uidai",
        "sq", "ft", "sqft", "m", "sqm", "no", "dr", "mr", "mrs", "miss", "smt", "shri",
        # Common Karnataka administrative, geographical, and address tokens
        "bengaluru", "bangalore", "koramangala", "jayanagar", "ejipura", "whitefield",
        "mysuru", "mysore", "mangaluru", "mangalore", "belagavi", "belgaum", "yelahanka",
        "rajajinagar", "mahadevapura", "indiranagar", "basavanagudi", "kengeri",
        "bed", "layout", "road", "cross", "main", "block", "stage", "sector", "post", "nagar",
    }
    if allowed_verbatim_tokens:
        for tok in allowed_verbatim_tokens:
            if tok and isinstance(tok, str):
                for word in re.findall(r'[A-Za-z0-9]+', tok):
                    allowed_words.add(word.lower())

    # 1. Validate Kannada text purity
    stray_english_fragments: List[str] = []
    if kannada_text:
        # Extract all Latin words
        latin_words = re.findall(r'[A-Za-z]+', kannada_text)
        for w in latin_words:
            if w.lower() not in allowed_words and len(w) > 1:
                stray_english_fragments.append(w)

    if stray_english_fragments:
        report.is_kannada_pure = False
        unique_frags = list(dict.fromkeys(stray_english_fragments))[:5]
        frag_str = ", ".join(f"'{f}'" for f in unique_frags)
        msg = f"Translation quality: low / needs review: Contains stray English fragments: {frag_str}"
        report.kannada_warnings.append(msg)
        report.warnings.append(msg)

    # 2. Validate English text purity
    untranslated_kannada_chars: List[str] = []
    if english_text:
        kannada_spans = re.findall(r'[\u0C80-\u0CFF]+', english_text)
        if kannada_spans:
            report.is_english_pure = False
            unique_kannada = list(dict.fromkeys(kannada_spans))[:5]
            kn_str = ", ".join(f"'{k}'" for k in unique_kannada)
            msg = f"Translation quality: low / needs review: Contains untranslated Kannada: {kn_str}"
            report.english_warnings.append(msg)
            report.warnings.append(msg)

    # Overall decision
    if not report.is_kannada_pure or not report.is_english_pure:
        report.quality = "low / needs review"
        report.requires_review = True
    else:
        report.quality = "high"
        report.requires_review = False

    return report


# ============================================================================
# Unified Translation Entry Points
# ============================================================================

def translate_document_text(
    text: str,
    semantic_fields: Optional[Any] = None,
    semantic_evidence: Optional[Any] = None,
    regions: Optional[Sequence[Any]] = None,
) -> TranslationResult:
    """Executes the complete translation quality pipeline:
    1. Step A: Contextual text normalization (with semantic evidence reuse).
    2. Step B: Meaning-preserving translation of normalized text.
    3. Step C: Hard post-translation language-purity validation.
    4. Distinct preservation: original_ocr, kannada_translation, english_translation.
    """
    if not text or not text.strip():
        empty_report = LanguagePurityReport(quality="high", requires_review=False)
        return TranslationResult(
            original_ocr="",
            kannada_translation="",
            english_translation="",
            purity_report=empty_report,
            fallback_normalization=False,
            protected_tokens={},
        )

    # Step A: Contextual text normalization
    clean_kannada, token_map, fallback_norm = normalize_ocr_text(
        raw_ocr_text=text,
        semantic_fields=semantic_fields,
        semantic_evidence=semantic_evidence,
        regions=regions,
    )

    # Step B: Meaning-preserving translation of normalized text
    clean_english = translate_normalized_kannada_to_english(
        normalized_kannada_text=clean_kannada,
        token_map=token_map,
    )

    # Step C: Language purity validation
    allowed_tokens = list(token_map.values())
    purity_report = validate_language_purity(
        kannada_text=clean_kannada,
        english_text=clean_english,
        allowed_verbatim_tokens=allowed_tokens,
    )

    return TranslationResult(
        original_ocr=text,
        kannada_translation=clean_kannada,
        english_translation=clean_english,
        purity_report=purity_report,
        fallback_normalization=fallback_norm,
        protected_tokens=token_map,
    )


def clean_kannada_translation(
    text: str,
    semantic_fields: Optional[Any] = None,
    semantic_evidence: Optional[Any] = None,
) -> str:
    """Produces clean, natural Kannada text using Step A contextual normalization."""
    if not text or not text.strip():
        return ""
    clean_kannada, _, _ = normalize_ocr_text(
        raw_ocr_text=text,
        semantic_fields=semantic_fields,
        semantic_evidence=semantic_evidence,
    )
    return clean_kannada


def translate_kannada_text(
    text: str,
    semantic_fields: Optional[Any] = None,
    semantic_evidence: Optional[Any] = None,
) -> str:
    """Produces clean, natural English translation using Step A -> Step B pipeline."""
    if not text or not text.strip():
        return ""
    result = translate_document_text(
        text=text,
        semantic_fields=semantic_fields,
        semantic_evidence=semantic_evidence,
    )
    return result.english_translation


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

    protected_text, token_map = protect_verbatim_entities(text)
    online_res = translate_english_to_kannada_online(protected_text)
    if online_res and online_res.strip():
        return restore_verbatim_entities(online_res, token_map)

    glossary_replaced = protected_text
    for kn_term, en_term in sorted(LAND_RECORD_GLOSSARY.items(), key=lambda x: -len(x[1])):
        if en_term.lower() in glossary_replaced.lower():
            pattern = re.compile(re.escape(en_term), re.IGNORECASE)
            glossary_replaced = pattern.sub(kn_term, glossary_replaced)

    return restore_verbatim_entities(glossary_replaced, token_map)


def translate_bidirectional(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """General bidirectional translation helper between Kannada and English."""
    if not text or not text.strip():
        return ""

    has_kannada = any('\u0c80' <= c <= '\u0cff' for c in text)
    if target_lang.lower().startswith("kn") or (source_lang == "en" and not has_kannada):
        return translate_english_to_kannada(text)
    else:
        return translate_kannada_text(text)
