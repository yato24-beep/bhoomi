"""
src/validation/normalization.py
Standardizes extracted names, dates, land units, and Khasra identifiers.
Always preserves raw_value and original extracted units for auditability.
"""

import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

from src.extraction.extractor import convert_devanagari_numerals


class FieldNormalizer:
    """
    Normalizes extracted fields into canonical representations according to state-specific rules.
    """

    def __init__(self, state_config: Dict[str, Any]):
        self.config = state_config
        self.unit_conversions = state_config.get("unit_conversions_to_hectares", {
            "hectare": 1.0,
            "हेक्टेयर": 1.0,
            "हे": 1.0,
            "acre": 0.404686,
            "एकड़": 0.404686,
            "bigha": 0.2529,
            "बीघा": 0.2529,
            "biswa": 0.012645,
            "बिसवा": 0.012645,
            "guntha": 0.010117,
            "गुंठा": 0.010117,
            "sq_meter": 0.0001,
            "वर्ग मीटर": 0.0001,
        })

    def normalize_name(self, raw_name: str) -> str:
        """
        Cleans person/owner names by removing honorific titles, extraneous prefixes, and normalizing whitespace.
        """
        if not raw_name:
            return ""

        name = raw_name.strip()
        # Remove common honorific titles in Hindi / Kannada / Tamil / English
        titles_pattern = r"^(?:श्रीमान|श्रीमती|सुश्री|श्री|ಶ್ರೀ|ಶ್ರೀಮತಿ|ತಿರು|திருமதி|स्व०|स्वर्गवासी|मर्हूम|जनाब|shri|smt|mr|mrs|miss|late|dr)\.?\s+"
        name = re.sub(titles_pattern, "", name, flags=re.IGNORECASE | re.UNICODE)

        # Remove circled numbers (①, ②, ③)
        name = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩\(\)]", "", name)

        # Remove trailing relationship / locational / register boilerplate
        boilerplate_pattern = r"(?:\s*(?:ರವರ\s*ಹೆಸರಿನಲ್ಲಿ|ರವರ|ರವರಿಗೆ|ಇವರ|ದಾಖಲಾಗಿರುತ್ತದೆ|in\s*the\s*register|Bangalore|Bengaluru))+$"
        name = re.sub(boilerplate_pattern, "", name, flags=re.IGNORECASE | re.UNICODE)

        # Remove trailing slash or symbols
        name = re.sub(r"[\/\-\:\,\.]+$", "", name).strip()
        # Collapse whitespace
        name = re.sub(r"\s+", " ", name).strip()
        return name


    KNOWN_LOCALITIES = {
        "ಜಯನಗರ": "Jayanagar",
        "ಕೋರಮಂಗಲ": "Koramangala",
        "ನಾರಾಯಣಘಟ್ಟ": "Narayanaghatta",
        "ಕೆಂಗೇರಿ": "Kengeri",
        "ಇಜಿಪುರ": "Ejipura",
        "ಮಲ್ಲೇಶ್ವರಂ": "Malleshwaram",
        "ಇಂದಿರಾನಗರ": "Indiranagar",
        "ಬಸವನಗುಡಿ": "Basavanagudi",
        "ಯಲಹಂಕ": "Yelahanka",
        "ರಾಜಾಜಿನಗರ": "Rajajinagar",
        "ಮಹಾದೇವಪುರ": "Mahadevapura",
    }

    KNOWN_TALUKS = {
        "ಜಯನಗರಿಉಪಏಭಾಗ": "Jayanagar Sub-division",
        "ಜಯನಗರ ಉಪವಿಭಾಗ": "Jayanagar Sub-division",
        "ಜಯನಗರ ಉಪ-ವಿಭಾಗ": "Jayanagar Sub-division",
        "ಕೋರಮಂಗಲ ಉಪವಿಭಾಗ": "Koramangala Sub-division",
        "ಬೆಂಗಳೂರು ದಕ್ಷಿಣ": "Bangalore South",
    }

    def normalize_locality(self, raw_loc: str) -> str:
        """Cleans administrative locality / village strings by stripping field labels and mapping known Kannada entities."""
        if not raw_loc:
            return ""
        clean = re.sub(r"^(?:Locality|ಸಳಸಳೀಯಹೆಸರು|ಸ್ಥಳೀಯ\s*ಹೆಸರು|ಬಡಾವಣೆ|Village|ಗ್ರಾಮ)[\s:.\-_|I]*", "", raw_loc.strip(), flags=re.IGNORECASE | re.UNICODE)
        clean = clean.strip(" \t\n,.-:|I")
        return self.KNOWN_LOCALITIES.get(clean, clean)

    def normalize_taluk(self, raw_taluk: str) -> str:
        """Cleans sub-division / taluk strings by stripping field labels."""
        if not raw_taluk:
            return ""
        clean = re.sub(r"^(?:Taluk|ತಾಲೂಕು|ತಾಲ್ಲೂಕು|ತಾಲೂಕ್|Sub\s*Division|ಉಪ\s*ವಿಭಾಗ|ಉಪಏಭಾಗ)[\s:.\-_|I]*", "", raw_taluk.strip(), flags=re.IGNORECASE | re.UNICODE)
        clean = clean.strip(" \t\n,.-:|I")
        return self.KNOWN_TALUKS.get(clean, clean)

    def normalize_district(self, raw_dist: str) -> str:
        """Cleans district strings by stripping field labels."""
        if not raw_dist:
            return ""
        clean = re.sub(r"^(?:District|ಜಿಲ್ಲೆ|ಬಲೈ)[\s:.\-_|I]*", "", raw_dist.strip(), flags=re.IGNORECASE | re.UNICODE)
        clean = clean.strip(" \t\n,.-:|I")
        if clean in ("ಬೆಂಗಳೂದು", "ಬೆಂಗಳೂರು"):
            return "Bengaluru"
        return clean

    def normalize_khasra(self, raw_khasra: str) -> str:
        """
        Normalizes Khasra / Plot / Survey / Ward numbers:
        Converts Indic numerals, eliminates spaces around slashes, standardizes delimiters,
        and drops a bare single-letter sub-division suffix (e.g. '89/1/A' -> '89/1') while
        preserving richer alphanumeric identifiers unchanged (e.g. BBMP '708/417/F-104').
        Example: '१४२ / १' -> '142/1', 'W.NO. 84' -> 'W.NO. 84'
        """
        if not raw_khasra:
            return ""

        # Remove label prefixes like "Survey/Number/", "Survey Number ", "Sy. No. ", "ಸರ್ವೆ ನಂ"
        raw_khasra = re.sub(r"^(?:Survey[\s\-_/]*(?:Number|No|Num)?|Sy\.?[\s\-_/]*No\.?|ಸರ್ವೆ[\s\-_/]*(?:ನಂ|ನಂಬರ್)?|ಸವee[\s\-_/]*ಸoui)[\s:.\-_|I]*", "", raw_khasra.strip(), flags=re.IGNORECASE | re.UNICODE)

        converted = convert_devanagari_numerals(raw_khasra)
        if "W.NO" in converted.upper() or "WARD" in converted.upper():
            return re.sub(r"\s+", " ", converted).strip()

        stripped = converted.strip()

        if not re.search(r"[0-9]", stripped):
            # No numeric content at all - nothing safe to extract, keep as-is.
            return stripped

        # Remove leading non-numeric noise (e.g. "No.", "Khasra:") up to the first digit.
        stripped = re.sub(r"^[^\d]+(?=[0-9])", "", stripped)

        match = re.match(r"^([0-9]+(?:[\/\-][0-9]+)*)", stripped)
        if match:
            core = match.group(1)
            remainder = stripped[len(core):].strip()
            if remainder and not re.match(r"^[\/\-][A-Za-z]$", remainder):
                # Richer identifier (e.g. '/F-104') - keep in full, only tidy slash spacing.
                return re.sub(r"\s*/\s*", "/", stripped)
            # Bare or no suffix - normalize the numeric core (hyphens -> slashes).
            return re.sub(r"\s*[\/\-]\s*", "/", core)

        # Digits present but no clean leading numeric run - keep as-is, tidy slash spacing.
        return re.sub(r"\s*/\s*", "/", stripped)

    def normalize_land_area(
        self, raw_area: str, raw_unit: Optional[str] = None
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """
        Normalizes land area string into a float value in HECTARES.
        Returns (normalized_value_in_hectares, raw_unit_detected, 'hectare').
        """
        if not raw_area:
            return 0.0, raw_unit, "hectare"

        converted = convert_devanagari_numerals(str(raw_area))
        
        # Extract numeric component
        num_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", converted)
        if not num_match:
            return 0.0, raw_unit, "hectare"

        numeric_val = float(num_match.group(1))

        # Detect unit if not already provided
        detected_unit = raw_unit
        if not detected_unit:
            for unit_key in self.unit_conversions.keys():
                if unit_key.lower() in converted.lower():
                    detected_unit = unit_key
                    break

        # Calculate conversion factor
        factor = 1.0
        if detected_unit:
            unit_norm = detected_unit.lower().strip()
            factor = self.unit_conversions.get(unit_norm, 1.0)

        hectares_val = round(numeric_val * factor, 6)
        return hectares_val, detected_unit, "hectare"

    def normalize_date(self, raw_date: str) -> Optional[str]:
        """
        Converts various date formats (DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD) to ISO format (YYYY-MM-DD).
        """
        if not raw_date:
            return None

        converted = convert_devanagari_numerals(raw_date).strip()
        date_formats = [
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
            "%Y-%m-%d", "%Y/%m/%d",
            "%d %b %Y", "%d %B %Y"
        ]

        for fmt in date_formats:
            try:
                dt = datetime.strptime(converted, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return converted

    def normalize_fasli_year(self, raw_fasli: str) -> Tuple[str, Optional[int]]:
        """
        Standardizes Fasli year string and calculates corresponding Gregorian starting year.
        Fasli Year + 592/593 = Gregorian Year.
        Example: '1428-1433' -> ('1428-1433', 2020)
        """
        if not raw_fasli:
            return "", None

        converted = convert_devanagari_numerals(raw_fasli).strip()
        match = re.search(r"([0-9]{4})", converted)
        gregorian_start = None
        if match:
            start_fasli = int(match.group(1))
            # Fasli era standard conversion offset
            gregorian_start = start_fasli + 592

        return converted, gregorian_start
