"""Content-based Land Record Classifier.

Classifies incoming documents into:
- Land record
- RTC/Bhoomi record
- Khata/property certificate
- Mutation/property document
- Other government property document
- Not a land record

Determines whether the document is an authentic land/property record based on
actual OCR and visual content, without relying on filenames.
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple
from schemas import DocumentType

logger = logging.getLogger(__name__)

# Specific domain keywords for each document category
RTC_KEYWORDS = [
    "ಪಹಣಿ", "pahani", "ಭೂಮಿ", "bhoomi", "rtc", "ಹಕ್ಕು ದಾಖಲೆ", "ಖಾತೆದಾರರ ವಿವರ",
    "ಸರ್ವೆ ನಂ", "ಸರ್ವೇ ನಂ", "survey no", "ಕಂದಾಯ", "ಹಿಸ್ಸಾ", "ಆರ್ ಟಿ ಸಿ", "ಹಿಡುವಳಿ",
    "ಸರ್ವೆ", "ಸರ್ವೇ", "ವಿಸ್ತೀರ್ಣ", "ಸರ್ವೆ ಸಂಖ್ಯೆ", "ಸರ್ವೇ ಸಂಖ್ಯೆ", "survey number"
]

KHATA_KEYWORDS = [
    "ಖಾತಾ ಪ್ರಮಾಣ", "ಖಾತಾ ದೃಢೀಕರಣ", "khata certificate", "khata extract", "ಖಾತೆ",
    "ದೃಢೀಕರಣ ಪತ್ರ", "ಬೃಹತ್ ಬೆಂಗಳೂರು", "bbmp", "bruhat bangalore", "ಆಸ್ತಿ ಸಂಖ್ಯೆ",
    "ಸ್ವತ್ತಿನ ಸಂಖ್ಯೆ", "ನಿವೇಶನದ ವಿಸ್ತೀರ್ಣ", "ಕಟ್ಟಡದ ವಿಸ್ತೀರ್ಣ", "pid", "ಅಸೆಸ್ಮೆಂಟ್",
    "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿ", "assistant revenue officer", "ಬಾತಾದಾಖಲೆ", "ಖಾತಾ"
]

MUTATION_KEYWORDS = [
    "ಮ್ಯುಟೇಶನ್", "mutation", "ನಮೂನೆ ೧೨", "ನಮೂನೆ 12", "ಜಂಟಿ ಹಕ್ಕು", "ವಾರಸುದಾರರು",
    "ಹಕ್ಕು ಬದಲಾವಣೆ", "mutation register", "ಹಿಸ್ಸಾ ನಂ"
]

OTHER_GOV_PROPERTY_KEYWORDS = [
    "ಕ್ರಯಪತ್ರ", "sale deed", "ದಾನಪತ್ರ", "gift deed", "ಭಾಗಪತ್ರ", "partition deed",
    "ನೋಂದಣಿ ಇಲಾಖೆ", "sub-registrar", "encumbrance certificate", "ಋಣಭಾರ ಪ್ರಮಾಣ",
    "tahsildar", "ತಹಶೀಲ್ದಾರ್", "order", "ಆದೇಶ", "ಸ್ವಾಧೀನ ಪತ್ರ"
]

GENERAL_LAND_KEYWORDS = [
    "ಸರ್ವೆ ನಂಬರ್", "ಸರ್ವೇ ನಂಬರ್", "survey number", "khasra", "khatauni", "patta",
    "chitta", "satbara", "7/12", "jamabandi", "ವಿಸ್ತೀರ್ಣ", "extent", "ಚದರ ಅಡಿ",
    "sq ft", "sq. ft.", "acres", "guntas", "ಗುಂಟೆ", "village", "ಗ್ರಾಮ", "taluk",
    "ತಾಲೂಕು", "hobli", "ಹೋಬಳಿ", "district", "ಜಿಲ್ಲೆ", "land", "plot", "site",
    "revenue department", "ಕಂದಾಯ ಇಲಾಖೆ", "ಕಂದಾಯ"
]

NON_LAND_EXCLUSIVE_KEYWORDS = [
    "cash receipt", "tax invoice", "bill of supply", "table no", "server:", "cashier",
    "prescription", "patient name", "dr.", "curriculum vitae", "resume", "experience",
    "boarding pass", "flight ticket", "hotel booking", "restaurant menu"
]


def classify_land_document(text: str) -> Dict[str, Any]:
    """Classifies document content into one of the designated land record types or not_land_record."""
    cleaned = (text or "").lower()

    # Check for explicit non-land indicators
    non_land_hits = sum(1 for kw in NON_LAND_EXCLUSIVE_KEYWORDS if kw in cleaned)
    
    # Check category hits
    rtc_hits = sum(1 for kw in RTC_KEYWORDS if kw in cleaned)
    khata_hits = sum(1 for kw in KHATA_KEYWORDS if kw in cleaned)
    mutation_hits = sum(1 for kw in MUTATION_KEYWORDS if kw in cleaned)
    other_gov_hits = sum(1 for kw in OTHER_GOV_PROPERTY_KEYWORDS if kw in cleaned)
    general_land_hits = sum(1 for kw in GENERAL_LAND_KEYWORDS if kw in cleaned)

    total_land_hits = rtc_hits + khata_hits + mutation_hits + other_gov_hits + general_land_hits

    # 1. If non-land markers dominate or zero land markers exist
    if total_land_hits == 0 or (non_land_hits >= 2 and total_land_hits <= 1):
        return {
            "document_type": DocumentType.NOT_LAND_RECORD,
            "document_type_label": "Not a land record",
            "is_land_record": False,
            "confidence": 0.95,
            "notice": "This document does not appear to be a land record.",
            "reasoning": "Document text lacks cadastral, revenue, Bhoomi, or property certification markers.",
        }

    # 2. RTC / Bhoomi record
    if rtc_hits >= 2 or ("bhoomi" in cleaned and ("rtc" in cleaned or "ಪಹಣಿ" in cleaned)):
        return {
            "document_type": DocumentType.BHOOMI_RTC,
            "document_type_label": "RTC/Bhoomi record",
            "is_land_record": True,
            "confidence": 0.95,
            "notice": None,
            "reasoning": "Karnataka Bhoomi Record of Rights, Tenancy and Crops (RTC) identified.",
        }

    # 3. Khata / property certificate
    if khata_hits >= 2 or ("bbmp" in cleaned and "ಖಾತಾ" in cleaned) or "khata certificate" in cleaned:
        return {
            "document_type": DocumentType.KHATA_CERTIFICATE,
            "document_type_label": "Khata/property certificate",
            "is_land_record": True,
            "confidence": 0.95,
            "notice": None,
            "reasoning": "Municipal Property Khata Certificate identified from official civic markings.",
        }

    # 4. Mutation / property document
    if mutation_hits >= 2 or "mutation register" in cleaned:
        return {
            "document_type": DocumentType.MUTATION_REGISTER,
            "document_type_label": "Mutation/property document",
            "is_land_record": True,
            "confidence": 0.92,
            "notice": None,
            "reasoning": "Mutation Register document identified from succession/transfer markings.",
        }

    # 5. Other government property document
    if other_gov_hits >= 2 or "sale deed" in cleaned or "sub-registrar" in cleaned:
        return {
            "document_type": DocumentType.OTHER_GOV_PROPERTY,
            "document_type_label": "Other government property document",
            "is_land_record": True,
            "confidence": 0.90,
            "notice": None,
            "reasoning": "Registered property conveyance or encumbrance document identified.",
        }

    # 6. Generic Land record
    if general_land_hits >= 2:
        return {
            "document_type": DocumentType.LAND_RECORD,
            "document_type_label": "Land record",
            "is_land_record": True,
            "confidence": 0.88,
            "notice": None,
            "reasoning": "Cadastral survey markers and land revenue measurements identified.",
        }

    # Default fallback if insufficient confidence
    return {
        "document_type": DocumentType.NOT_LAND_RECORD,
        "document_type_label": "Not a land record",
        "is_land_record": False,
        "confidence": 0.80,
        "notice": "This document does not appear to be a land record.",
        "reasoning": "Insufficient cadastral or property evidence found in document content.",
    }
