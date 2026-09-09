"""Conservative, Non-Destructive Kannada Text Normalizer and Domain Lexicon Suggester.

Applies deterministic Unicode NFC normalization and whitespace sanitation without
fabricating or silently altering ambiguous or low-confidence recognition outputs.
Provides domain-specific land record dictionary suggestions for Person C validation.
"""

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Set, Tuple


# Core Karnataka Land Record & Revenue Terminology Dictionary
KANNADA_LAND_RECORD_LEXICON: List[str] = [
    "ಖಾತೆ",                # Khata (Account / Ledger)
    "ಖಾತೆದಾರರು",          # Khatadar (Account Holder)
    "ಸರ್ವೆ ನಂಬರ್",         # Survey Number
    "ಹಿಸ್ಸಾ",              # Hissa (Subdivision)
    "ಪಹಣಿ",                # Pahani (RTC - Record of Rights, Tenancy and Crops)
    "ಗ್ರಾಮ",               # Grama (Village)
    "ಹೋಬಳಿ",              # Hobli (Revenue Circle)
    "ತಾಲೂಕು",             # Taluk (Sub-district)
    "ಜಿಲ್ಲೆ",              # Jille (District)
    "ವಿಸ್ತೀರ್ಣ",           # Visteerna (Area / Extent)
    "ಎಕರೆ",               # Acre
    "ಗುಂಟೆ",              # Gunte
    "ಆಣೆ",                # Aane
    "ಖರಾಬು",              # Kharabu (Uncultivable land)
    "ಆಕಾರಬಂಧು",           # Aakarbandhu (Land survey record)
    "ಮ್ಯುಟೇಶನ್",           # Mutation
    "ಸ್ವಾಧೀನದಾರರು",       # Occupant / Possessor
    "ಮಾಲೀಕರ ಹೆಸರು",       # Owner Name
    "ಚಕ್ಕುಬಂದಿ",           # Chakkubandi (Boundaries)
    "ಪೂರ್ವ",              # East
    "ಪಶ್ಚಿಮ",             # West
    "ಉತ್ತರ",              # North
    "ದಕ್ಷಿಣ",              # South
    "ಕಂದಾಯ",              # Revenue / Tax
    "ದಸ್ತಾವೇಜು",           # Dastaveju (Document / Deed)
    "ಕ್ರಯಪತ್ರ",            # Sale Deed
    "ದಾನಪತ್ರ",            # Gift Deed
    "ಭಾಗಪತ್ರ",            # Partition Deed
]


@dataclass
class NormalizationResult:
    """Detailed result of conservative text normalization."""
    raw_text: str
    normalized_text: str
    is_modified: bool
    candidate_suggestions: List[str] = field(default_factory=list)
    applied_rules: List[str] = field(default_factory=list)


def _compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Computes exact character-level Levenshtein edit distance."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0 = list(v1)

    return v0[len(s2)]


class KannadaNormalizer:
    """Non-destructive, conservative text normalizer tailored for Kannada land records."""

    def __init__(
        self,
        lexicon: Optional[Sequence[str]] = None,
        max_suggestion_distance: int = 2,
    ):
        """Initializes the normalizer with optional custom terminology lexicon."""
        self.lexicon = list(lexicon) if lexicon is not None else list(KANNADA_LAND_RECORD_LEXICON)
        self.max_suggestion_distance = max_suggestion_distance

    def normalize(self, text: str) -> NormalizationResult:
        """Applies conservative, non-destructive normalization to recognized Kannada text.

        Preserves raw model output and strictly avoids replacing uncertain words.
        """
        if text is None:
            return NormalizationResult(
                raw_text="",
                normalized_text="",
                is_modified=False,
                candidate_suggestions=[],
                applied_rules=[],
            )

        raw = str(text)
        current = raw
        rules: List[str] = []

        # 1. Unicode NFC Normalization (canonical composition of base glyphs & matras)
        nfc_text = unicodedata.normalize("NFC", current)
        if nfc_text != current:
            rules.append("unicode_nfc_composition")
            current = nfc_text

        # 2. Clean extraneous Zero-Width Space / Joiner artifacts if isolated
        # Keep ZWJ (\u200D) and ZWNJ (\u200C) only when surrounded by Kannada characters
        cleaned_zw = re.sub(r"(?<![\u0C80-\u0CFF])[\u200C\u200D\u200B]+|[\u200C\u200D\u200B]+(?![\u0C80-\u0CFF])", "", current)
        if cleaned_zw != current:
            rules.append("clean_isolated_zero_width_chars")
            current = cleaned_zw

        # 3. Standardize whitespace (tabs, multiple spaces, non-breaking spaces \u00A0)
        norm_space = re.sub(r"[\t\r\f\v\u00A0\u2000-\u200A]+", " ", current)
        norm_space = re.sub(r" +", " ", norm_space).strip()
        if norm_space != current:
            rules.append("whitespace_sanitization")
            current = norm_space

        # 4. Find candidate terminology suggestions without overwriting the text
        suggestions = self.find_candidate_suggestions(current)

        return NormalizationResult(
            raw_text=raw,
            normalized_text=current,
            is_modified=(raw != current),
            candidate_suggestions=suggestions,
            applied_rules=rules,
        )

    def find_candidate_suggestions(self, text: str) -> List[str]:
        """Identifies domain-specific terminology matches or close candidates."""
        if not text or len(text.strip()) == 0:
            return []

        cleaned = text.strip()
        suggestions: List[str] = []
        seen: Set[str] = set()

        for term in self.lexicon:
            # Exact match
            if term in cleaned:
                if term not in seen:
                    suggestions.append(term)
                    seen.add(term)
                continue

            # Fuzzy match on individual words or short phrases
            dist = _compute_levenshtein_distance(cleaned, term)
            # Allow distance 1 for short words (<=4 chars), distance 2 for longer words
            threshold = 1 if len(term) <= 4 else self.max_suggestion_distance
            if dist <= threshold:
                if term not in seen:
                    suggestions.append(term)
                    seen.add(term)

        return suggestions
