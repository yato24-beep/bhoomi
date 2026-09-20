"""Regional Alias Mapping for Cross-State Land Record Terminology.

Establishes Karnataka Land Administration terminology as the primary canonical schema
and provides explicit, deterministic resolution for Maharashtra (7/12), Tamil Nadu
(Patta Chitta), and Northern/Central Indian cadastral nomenclature.
"""

from typing import Dict, Optional, Tuple

# Karnataka Canonical Cadastral Fields
KARNATAKA_CANONICAL_FIELDS = {
    "document_type",
    "district",          # ಜಿಲ್ಲೆ
    "taluk",             # ತಾಲೂಕು
    "hobli",             # ಹೋಬಳಿ
    "village",           # ಗ್ರಾಮ
    "survey_number",     # ಸರ್ವೆ ನಂ
    "hissa_number",      # ಹಿಸ್ಸಾ ನಂ
    "khata_number",      # ಖಾತಾ ಸಂಖ್ಯೆ
    "owner_name",        # ಖಾತೆದಾರರ ಹೆಸರು / ಮಾಲೀಕರ ಹೆಸರು
    "cultivator_name",   # ಅನುಭೋಗದಾರರ ಹೆಸರು / ಗೇಣಿದಾರರ ಹೆಸರು
    "extent",            # ವಿಸ್ತೀರ್ಣ
    "land_type",         # ಜಮೀನಿನ ವಿವರ (ಖುಷ್ಕಿ/ತರಿ/ಬಾಗಾಯ್ತು)
    "assessment",        # ಕಂದಾಯ / ಆಕಾರಬಂಧು
    "mutation_number",   # ಮ್ಯುಟೇಶನ್ ಸಂಖ್ಯೆ / MR No
    "registration_number",# ನೋಂದಣಿ ಸಂಖ್ಯೆ
    "record_date",       # ದಾಖಲೆ ದಿನಾಂಕ
    "remarks",           # ಷರಾ / ಇತರ ವಿವರ
}

# Regional Alias -> (Canonical Karnataka Field, State / Tradition)
REGIONAL_ALIASES: Dict[str, Tuple[str, str]] = {
    # -------------------------------------------------------------------------
    # Maharashtra (7/12 Satbara & Mutation)
    # -------------------------------------------------------------------------
    "gat_number": ("survey_number", "Maharashtra"),
    "gat_no": ("survey_number", "Maharashtra"),
    "gat": ("survey_number", "Maharashtra"),
    "pot_hissa": ("hissa_number", "Maharashtra"),
    "pothissa": ("hissa_number", "Maharashtra"),
    "khatedar": ("owner_name", "Maharashtra"),
    "khatedar_name": ("owner_name", "Maharashtra"),
    "bhogwathdar": ("owner_name", "Maharashtra"),
    "occupant": ("owner_name", "Maharashtra"),
    "occupant_khatedar": ("owner_name", "Maharashtra"),
    "kshetra": ("extent", "Maharashtra"),
    "area_hec": ("extent", "Maharashtra"),
    "aakar": ("assessment", "Maharashtra"),
    "assessment_rs": ("assessment", "Maharashtra"),
    "taluka": ("taluk", "Maharashtra"),
    "jilha": ("district", "Maharashtra"),
    "gav": ("village", "Maharashtra"),
    "ferfar_number": ("mutation_number", "Maharashtra"),
    "ferfar_no": ("mutation_number", "Maharashtra"),

    # -------------------------------------------------------------------------
    # Tamil Nadu (Patta Chitta & A-Register)
    # -------------------------------------------------------------------------
    "patta_number": ("khata_number", "Tamil Nadu"),
    "patta_no": ("khata_number", "Tamil Nadu"),
    "patta": ("khata_number", "Tamil Nadu"),
    "pula_en": ("survey_number", "Tamil Nadu"),
    "pula_number": ("survey_number", "Tamil Nadu"),
    "sub_division": ("hissa_number", "Tamil Nadu"),
    "utpirivu": ("hissa_number", "Tamil Nadu"),
    "pattadharar": ("owner_name", "Tamil Nadu"),
    "pattadar_name": ("owner_name", "Tamil Nadu"),
    "parappalavu": ("extent", "Tamil Nadu"),
    "vattam": ("taluk", "Tamil Nadu"),
    "maavattam": ("district", "Tamil Nadu"),
    "kiramam": ("village", "Tamil Nadu"),

    # -------------------------------------------------------------------------
    # Northern / Central India (Khasra / Khatauni / Jamabandi)
    # -------------------------------------------------------------------------
    "khasra_number": ("survey_number", "North India"),
    "khasra_no": ("survey_number", "North India"),
    "khasra": ("survey_number", "North India"),
    "khatauni_number": ("khata_number", "North India"),
    "khatauni_no": ("khata_number", "North India"),
    "khatauni": ("khata_number", "North India"),
    "bhumidhar": ("owner_name", "North India"),
    "kastakar": ("cultivator_name", "North India"),
    "tehsil": ("taluk", "North India"),
    "mauza": ("village", "North India"),
    "rakba": ("extent", "North India"),
    "lagan": ("assessment", "North India"),
    "dakhil_kharij": ("mutation_number", "North India"),
}


def resolve_field_alias(field_name: str) -> Tuple[str, Optional[str]]:
    """Resolves an extracted field name to the primary Karnataka canonical schema.

    Args:
        field_name: Extracted field name (e.g. 'gat_number', 'patta_number', 'survey_number').

    Returns:
        Tuple of (canonical_field_name, source_state_or_tradition_if_aliased).
        If already canonical or unaliased, returns (canonical_field_name, None).
    """
    clean_name = field_name.strip().lower().replace(" ", "_")
    
    if clean_name in KARNATAKA_CANONICAL_FIELDS:
        return clean_name, None

    if clean_name in REGIONAL_ALIASES:
        canonical, region = REGIONAL_ALIASES[clean_name]
        return canonical, region

    return clean_name, None
