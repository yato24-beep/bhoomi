"""
src/extraction/semantic_understanding.py
Semantic Understanding & Contextual Extraction Layer for Land Record Digitization.

Operates between OCR/reading-order reconstruction and Person C structured extraction/validation.
Understands noisy, mixed Kannada/English/Latin documents, resolves corrupted/merged tokens,
filters out irrelevant administrative noise, preserves spatial bounding-box provenance,
and maps candidate fields to Person C data contracts with strict JSON/Pydantic schemas.
"""

import base64
import json
import logging
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import urllib.request
import urllib.parse

from pydantic import BaseModel, ConfigDict, Field

from schemas import (
    BoundingBox,
    DocumentOCRResult,
    DocumentType,
    ExtractedField,
    ExtractionMethod,
    FieldEvidence,
    ValidationStatus,
)
from src.extraction.extractor import convert_indic_numerals, clean_ocr_text
from src.translation.translator import translate_kannada_text

logger = logging.getLogger("semantic_understanding")


# ============================================================================
# Strict Pydantic Data Contracts for Semantic Extraction & Evidence Pipeline
# ============================================================================

class EvidenceCandidate(BaseModel):
    """Granular evidence piece capturing a candidate field extraction."""
    model_config = ConfigDict(protected_namespaces=())

    raw_text: str = Field(..., description="Raw text as matched or presented")
    candidate_field: str = Field(..., description="Target candidate field name")
    evidence_span: str = Field(..., description="Exact token or substring span")
    surrounding_context: str = Field(default="", description="Surrounding line/window context")
    ocr_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="OCR confidence from source line")
    semantic_confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Semantic match strength")
    reasoning: str = Field(default="", description="Reasoning for candidate classification")
    engine: str = Field(default="local_fallback", description="Engine producing evidence: 'gemini' | 'local_fallback'")


class SemanticFieldExtraction(BaseModel):
    """Structured extraction candidate for a single land record property field."""
    model_config = ConfigDict(protected_namespaces=())

    field_name: str = Field(..., description="Canonical field identifier")
    extracted_value: Optional[str] = Field(None, description="Extracted substantive value or None if insufficient evidence")
    evidence_text: Optional[str] = Field(None, description="Short supporting OCR snippet")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Semantic confidence score")
    bbox: Optional[BoundingBox] = Field(None, description="Source region bounding box")
    page_number: int = Field(default=1, ge=1, description="Source page number")
    raw_unit: Optional[str] = Field(None, description="Original measurement unit if present (e.g. Sq Ft, Gunta)")
    reasoning: Optional[str] = Field(None, description="Semantic contextual reasoning")

    # Enriched Evidence Pipeline Attributes
    candidate_field: Optional[str] = Field(None, description="Target candidate field role")
    evidence_span: Optional[str] = Field(None, description="Exact token or substring span")
    surrounding_context: Optional[str] = Field(None, description="Multi-line window / context")
    ocr_confidence: float = Field(default=0.90, ge=0.0, le=1.0, description="OCR line confidence")
    semantic_confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Semantic match strength")
    engine: str = Field(default="local_fallback", description="Engine: 'gemini' | 'local_fallback'")
    candidates: List[EvidenceCandidate] = Field(default_factory=list, description="Alternative candidates considered")


class SemanticExtractionResult(BaseModel):
    """Complete structured semantic interpretation of a land record document."""
    model_config = ConfigDict(protected_namespaces=())

    document_type: Optional[str] = Field(None, description="Inferred document type (e.g. Khata Certificate, Bhoomi RTC)")
    fields: Dict[str, SemanticFieldExtraction] = Field(default_factory=dict, description="Extracted candidate fields")
    irrelevant_noise_detected: List[str] = Field(default_factory=list, description="Irrelevant noise lines filtered out")
    model_used: str = Field(default="multilingual_semantic_engine", description="Model engine used for semantic extraction")
    engine_used: str = Field(default="local_fallback", description="Engine: 'gemini' | 'local_fallback'")
    execution_time_ms: float = Field(default=0.0, description="Semantic understanding execution time in milliseconds")

    def to_extracted_fields(
        self,
        ocr_result: Optional[DocumentOCRResult] = None,
    ) -> Dict[str, ExtractedField]:
        """Converts semantic extractions into Person C standard ExtractedField dictionary."""
        result: Dict[str, ExtractedField] = {}
        for fname, s_field in self.fields.items():
            if not s_field.extracted_value or not str(s_field.extracted_value).strip():
                continue

            # Standardize field name to canonical field identifier
            canonical_name = fname
            if fname in ("survey_number", "khasra_number"):
                canonical_name = "survey_number"
            elif fname in ("property_number", "khata_number", "khatauni_number"):
                canonical_name = "khata_number"
            elif fname in ("date", "document_date"):
                canonical_name = "date"
            elif fname in ("taluk", "tehsil", "sub_division"):
                canonical_name = "taluk"
            elif fname in ("locality", "village"):
                canonical_name = "locality"
            elif fname in ("issuing_organization", "organization"):
                canonical_name = "issuing_organization"
            elif fname in ("site_area", "land_area"):
                canonical_name = "site_area"
            elif fname == "built_up_area":
                canonical_name = "built_up_area"
            elif fname in ("issuing_authority", "authority"):
                canonical_name = "issuing_authority"
            elif fname in ("district", "city"):
                canonical_name = "district"
            elif fname in ("address", "property_address"):
                canonical_name = "address"
            elif fname == "owner_name":
                canonical_name = "owner_name"
            elif fname == "document_type":
                canonical_name = "document_type"

            aliases_to_emit = [canonical_name]

            bbox = s_field.bbox or BoundingBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0)
            page = s_field.page_number or 1

            for alias_name in aliases_to_emit:
                v_notes = []
                if s_field.reasoning:
                    v_notes.append(s_field.reasoning)
                if s_field.evidence_span:
                    v_notes.append(f"Evidence span: '{s_field.evidence_span}'")
                if s_field.surrounding_context:
                    v_notes.append(f"Context: {s_field.surrounding_context[:120]}")
                v_notes.append(f"Engine: {s_field.engine}")
                v_notes.append(f"OCR Conf: {s_field.ocr_confidence:.2f}, Semantic Conf: {s_field.semantic_confidence:.2f}")

                evidence = FieldEvidence(
                    page_number=page,
                    bbox=bbox,
                    raw_ocr_text=s_field.evidence_text or str(s_field.extracted_value),
                    ocr_engine=s_field.engine if s_field.engine in ("gemini", "local_fallback") else self.model_used,
                    extraction_rule_id=f"semantic_{alias_name}",
                    source_record="semantic_understanding_layer",
                    validation_notes=v_notes,
                )

                result[alias_name] = ExtractedField(
                    field_name=alias_name,
                    raw_value=str(s_field.extracted_value).strip(),
                    normalized_value=str(s_field.extracted_value).strip(),
                    raw_unit=s_field.raw_unit,
                    confidence=round(s_field.confidence, 4),
                    page=page,
                    bbox=bbox,
                    evidence=evidence,
                    extraction_method=ExtractionMethod.SEMANTIC_UNDERSTANDING,
                    validation_status=ValidationStatus.UNVERIFIED,
                )

        return result


# ============================================================================
# LLM & Pretrained Multilingual Engine Interface
# ============================================================================

SYSTEM_PROMPT = """You are an expert multilingual document intelligence system specialized in Indian land, property, revenue, and municipal records in Kannada, English, and mixed scripts.

The input text is raw OCR output which may contain noise, merged Kannada words, missing or corrupted characters, broken words, transliterated/Latin OCR, corrupted labels, and lack explicit key-value structure.

Your objectives:
1. Analyze context and semantic relationships across sentences and paragraphs.
2. Identify and extract genuine land/property record attributes:
   - document_type: (e.g. 'Khata Certificate', 'Bhoomi RTC', 'Mutation Register', 'Sale Deed', 'Property Tax Receipt')
   - owner_name: full name of land/property owner or khatedar (with honorifics if present)
   - survey_number / khasra_number: survey number or parcel identifier (e.g. '2', '42/1', '142')
   - khata_number / property_number: khata number, assessment number, or property PID (e.g. '68-76-470/a', '124')
   - village: revenue village name
   - locality: municipal layout, neighborhood, or area name (e.g. 'Jayanagar', 'Koramangala', 'Ejipura')
   - hobli: revenue hobli or sub-district zone
   - taluk: taluk / sub-division (e.g. 'Jayanagar Sub-division', 'Bangalore South')
   - district: district or municipal corporation (e.g. 'Bruhat Bengaluru Mahanagara Palike', 'Bengaluru', 'Bangalore')
   - ward: ward number and/or ward name (e.g. '148-Ejipura', 'Ward 148')
   - site_area: plot/site land area with unit if present (e.g. '2200 Sq Ft', '1200 sqft', '0.4500 hectare')
   - built_up_area: building/constructed floor area with unit if present
   - land_area: general land extent
   - date: document issue or execution date in original format (e.g. '20-09-2024')
   - address: property address or street location
   - issuing_authority: issuing officer or authority (e.g. 'ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿ' / 'Assistant Revenue Officer', 'Tahsildar', 'BBMP')
   - father_or_husband_name: father's or husband's name

Strict Rules:
1. Contextual inference over exact keywords: If noisy text contains 'ಸೈಬ್ವೇ-ನಂಬರ್-.2- ... ನಿವೇಶನ', infer that '2' is a survey/property identifier. If text has 'ಸಹಾಯಕಕಂದಾಯಿಅಧಕಾರ', infer issuing authority 'ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿ'.
2. Ignore irrelevant text: Filter out greetings, repeated headers, decorative text, page numbers, signatures, unrelated administrative sentences, irrelevant phone/email info.
3. NEVER hallucinate: If a field is not supported by evidence in the text, return null or omit it.
4. Return evidence_text (verbatim snippet from OCR) and confidence (0.0 to 1.0) for every extracted field.
5. Preserve uncertainty: If OCR is too corrupted to determine a value, do not invent or silently alter it.
6. Output MUST be valid JSON conforming strictly to the requested schema.
"""


def _query_external_llm(
    ordered_ocr_text: str,
    api_key: str,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
    timeout: int = 35,
) -> Optional[Dict[str, Any]]:
    """Calls Gemini or OpenAI-compatible external LLM endpoint with strict JSON schema."""
    is_gemini = (
        api_key.startswith("AIza")
        or api_key.startswith("AQ.")
        or bool(os.environ.get("GEMINI_API_KEY"))
        or "gemini" in (model_name or os.environ.get("SEMANTIC_LLM_MODEL", "")).lower()
    )
    model = (
        model_name
        or os.environ.get("SEMANTIC_LLM_MODEL")
        or ("gemini-3.7-flash" if is_gemini else "gpt-4o-mini")
    )

    user_prompt = f"""You are an expert Indian land and property record intelligence system.
Analyze the provided document image and OCR text. Reason using full context across sentences and paragraphs.
Do not invent information. Unsupported fields MUST be returned as null.

DISAMBIGUATION REQUIREMENTS:
1. Classify document_type as exactly one of:
   - "Land record"
   - "RTC/Bhoomi record"
   - "Khata/property certificate"
   - "Mutation/property document"
   - "Other government property document"
   - "Not a land record"
   Set "is_land_record": true (or false if "Not a land record"). If not a land record, leave all field values null.
2. Survey Number vs Date: Never extract a day or year from a date as a survey number (e.g., in "17-08-2023", "17" is a date component, not a survey number). Survey numbers are parcel numbers like "12/3", "2", "142", etc.
3. Khata / Property Number vs Survey Number: Distinguish property PID / Khata numbers (e.g. "470/A", "68-76-470/a") from Survey Numbers.
4. District vs Issuing Organization: District is the geographic administrative district/city ("Bengaluru"). Organization is the civic or revenue body ("Bruhat Bengaluru Mahanagara Palike" / "BBMP").
5. Locality vs District: Locality is the specific layout/neighborhood/village ("Jayanagar").
6. Site Area vs Built-up Area: Site area is the land/plot extent ("1200 Sq Ft"). Built-up area is the constructed building footprint ("900 Sq Ft").
7. Document Date vs other dates: Normalize official execution/issue date to DD-MM-YYYY.
8. Owner vs Authority: Owner is the khatedar/property owner. Authority is the government official ("Assistant Revenue Officer").

--- OCR TEXT START ---
{ordered_ocr_text}
--- OCR TEXT END ---

Output strict JSON conforming to:
{{
  "document_type": string,
  "is_land_record": boolean,
  "fields": {{
    "owner_name": {{"extracted_value": string or null, "evidence_text": string, "confidence": float, "reasoning": string}},
    "survey_number": {{"extracted_value": string or null, "evidence_text": string, "confidence": float, "reasoning": string}},
    "khata_number": {{"extracted_value": string or null, "evidence_text": string, "confidence": float, "reasoning": string}},
    "locality": {{"extracted_value": string or null, "evidence_text": string, "confidence": float, "reasoning": string}},
    "taluk": {{"extracted_value": string or null, "evidence_text": string, "confidence": float, "reasoning": string}},
    "district": {{"extracted_value": string or null, "evidence_text": string, "confidence": float, "reasoning": string}},
    "address": {{"extracted_value": string or null, "evidence_text": string, "confidence": float, "reasoning": string}},
    "site_area": {{"extracted_value": string or null, "raw_unit": string or null, "evidence_text": string, "confidence": float, "reasoning": string}},
    "built_up_area": {{"extracted_value": string or null, "raw_unit": string or null, "evidence_text": string, "confidence": float, "reasoning": string}},
    "date": {{"extracted_value": string or null, "evidence_text": string, "confidence": float, "reasoning": string}},
    "issuing_authority": {{"extracted_value": string or null, "evidence_text": string, "confidence": float, "reasoning": string}},
    "issuing_organization": {{"extracted_value": string or null, "evidence_text": string, "confidence": float, "reasoning": string}}
  }},
  "irrelevant_noise_detected": [string]
}}"""

    # 1. Native Gemini Multimodal / Text API Call
    if is_gemini and not base_url:
        cur_model = model.strip()
        if not cur_model.startswith("models/"):
            cur_model = f"models/{cur_model}"

        # Detect MIME type from magic bytes if not provided
        if image_bytes and not image_mime_type:
            if image_bytes.startswith(b"\xff\xd8\xff"):
                image_mime_type = "image/jpeg"
            elif image_bytes.startswith(b"\x89PNG"):
                image_mime_type = "image/png"
            elif image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
                image_mime_type = "image/webp"
            else:
                image_mime_type = "image/jpeg"

        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/{cur_model}:generateContent?key={api_key}"
        parts: List[Dict[str, Any]] = []
        if image_bytes:
            b64_img = base64.b64encode(image_bytes).decode("utf-8")
            parts.append({
                "inlineData": {
                    "mimeType": image_mime_type or "image/jpeg",
                    "data": b64_img,
                }
            })
        parts.append({"text": f"{SYSTEM_PROMPT}\n\n{user_prompt}"})

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
        }
        payload_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "LandRecordSemanticUnderstanding/1.0",
        }
        if api_key.startswith("AQ.") or api_key.startswith("ya29."):
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            req = urllib.request.Request(
                gemini_url,
                data=payload_bytes,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    parts_resp = candidates[0].get("content", {}).get("parts", [])
                    if parts_resp and "text" in parts_resp[0]:
                        raw_text = parts_resp[0]["text"].strip()
                        if raw_text.startswith("```json"):
                            raw_text = raw_text[7:]
                        if raw_text.startswith("```"):
                            raw_text = raw_text[3:]
                        if raw_text.endswith("```"):
                            raw_text = raw_text[:-3]
                        parsed = json.loads(raw_text.strip())
                        parsed["_model_used"] = cur_model
                        return parsed
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="replace")
            logger.warning(
                f"Gemini API returned HTTP {http_err.code} ({http_err.reason}) for model '{cur_model}'. "
                f"Preserving existing local fallback engine. Details: {err_body[:200]}"
            )
            return None
        except Exception as gemini_err:
            logger.warning(
                f"Gemini API request for model '{cur_model}' failed: {gemini_err}. "
                f"Preserving existing local fallback engine."
            )
            return None

    # 2. OpenAI-Compatible API Call (OpenAI, Groq, or Gemini OpenAI endpoint)
    url = (
        base_url
        or os.environ.get("SEMANTIC_LLM_BASE_URL")
        or (
            "https://generativelanguage.googleapis.com/v1beta/openai"
            if is_gemini
            else "https://api.openai.com/v1"
        )
    ).rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "LandRecordSemanticUnderstanding/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as exc:
        logger.warning(f"External LLM semantic extraction request failed: {exc}")
        return None


# ============================================================================
# Multi-Line Context Window Engine & Number Disambiguator
# ============================================================================

class MultilineContextWindow(BaseModel):
    """Aggregated spatial window spanning N adjacent lines for cross-line entity understanding."""
    model_config = ConfigDict(protected_namespaces=())

    lines: List[Any] = Field(default_factory=list)
    combined_text: str = ""
    bbox: Optional[BoundingBox] = None
    page_number: int = 1
    avg_ocr_confidence: float = 1.0
    line_indices: List[int] = Field(default_factory=list)


def build_multiline_windows(
    regions: Optional[Sequence[Any]],
    full_text: Optional[str] = None,
    window_size: int = 3,
) -> List[MultilineContextWindow]:
    """Constructs multi-line context windows from OCR regions or raw lines.

    Enables entity extraction to bridge line boundaries (e.g. 'Owner Name:' on line 1,
    'Mrs. Dorothy Charles' on line 2, and 'Site No. 12/3' on line 3).
    """
    windows: List[MultilineContextWindow] = []

    if regions:
        valid_regions = []
        for reg in regions:
            t = getattr(reg, "text", None) or getattr(reg, "normalized_text", None) or getattr(reg, "raw_text", None)
            if t and str(t).strip():
                valid_regions.append(reg)

        def get_pos(r):
            page = getattr(r, "page_number", 1) or 1
            bbox = getattr(r, "bbox", None)
            y = 0.0
            x = 0.0
            if bbox:
                if isinstance(bbox, BoundingBox):
                    y = bbox.y_min
                    x = bbox.x_min
                elif isinstance(bbox, dict):
                    y = float(bbox.get("y_min", 0.0))
                    x = float(bbox.get("x_min", 0.0))
            return (page, y, x)

        valid_regions.sort(key=get_pos)

        n = len(valid_regions)
        for i in range(n):
            for k in range(1, window_size + 1):
                if i + k <= n:
                    group = valid_regions[i:i + k]
                    texts = [
                        getattr(r, "text", None) or getattr(r, "normalized_text", None) or str(r)
                        for r in group
                    ]
                    confs = [float(getattr(r, "confidence", 0.90) or 0.90) for r in group]
                    page = getattr(group[0], "page_number", 1) or 1

                    x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
                    for r in group:
                        b = getattr(r, "bbox", None)
                        if b:
                            if isinstance(b, BoundingBox):
                                x_mins.append(b.x_min)
                                y_mins.append(b.y_min)
                                x_maxs.append(b.x_max)
                                y_maxs.append(b.y_max)
                            elif isinstance(b, dict):
                                x_mins.append(float(b.get("x_min", 0.0)))
                                y_mins.append(float(b.get("y_min", 0.0)))
                                x_maxs.append(float(b.get("x_max", 0.0)))
                                y_maxs.append(float(b.get("y_max", 0.0)))

                    bbox = None
                    if x_mins:
                        bbox = BoundingBox(
                            x_min=min(x_mins),
                            y_min=min(y_mins),
                            x_max=max(x_maxs),
                            y_max=max(y_maxs),
                        )

                    windows.append(MultilineContextWindow(
                        lines=group,
                        combined_text="\n".join(t.strip() for t in texts if t.strip()),
                        bbox=bbox,
                        page_number=page,
                        avg_ocr_confidence=round(sum(confs) / max(1, len(confs)), 3),
                        line_indices=list(range(i, i + k)),
                    ))

    elif full_text:
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        for i in range(len(lines)):
            for k in range(1, window_size + 1):
                if i + k <= len(lines):
                    group = lines[i:i + k]
                    windows.append(MultilineContextWindow(
                        lines=group,
                        combined_text="\n".join(group),
                        bbox=None,
                        page_number=1,
                        avg_ocr_confidence=0.90,
                        line_indices=list(range(i, i + k)),
                    ))

    return windows


class DisambiguatedNumber(BaseModel):
    """Categorized numeric token with evidence provenance."""
    model_config = ConfigDict(protected_namespaces=())

    raw_token: str
    cleaned_value: str
    role: str  # survey_number, khata_number, site_area, built_up_area, ward, date, phone, year, document_number, unclassified
    evidence_span: str
    surrounding_context: str
    confidence: float = 0.85
    reasoning: str = ""
    unit: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    page_number: int = 1
    ocr_confidence: float = 1.0


class NumberDisambiguator:
    """Pre-assignment numeric token classifier.

    Classifies every numeric token in the document into its actual functional role:
    survey_number, khata_number, site_area, built_up_area, ward, date, phone,
    year, document_number, or unclassified.
    """

    ROLE_KEYWORDS = {
        "date": ["ದಿನಾಂಕ", "ದಿನಾ೦ಕ", "ದನಾ0ಕ", "ಎನಾ೦ಕ", "ಎನಾಂಕ", "bನಾoಕ", "date", "dated"],
        "year": ["ವರ್ಷ", "ಸಾಲಿನ", "ಸಾಲಿನಲ್ಲಿ", "fasli", "assessment year", "year"],
        "phone": ["ದೂರವಾಣಿ", "ಮೊಬೈಲ್", "phone", "tel", "mob", "contact"],
        "built_up_area": ["ಕಟ್ಟಡದ ವಿಸ್ತೀರ್ಣ", "ಕಟ್ಟಡ ವಿಸ್ತೀರ್ಣ", "ಕಟಡದ ಎನೀಣ೯", "ಕಟಡದ", "built-up", "built up", "plinth", "plinth area"],
        "site_area": ["ನಿವೇಶನದ ವಿಸ್ತೀರ್ಣ", "ನಿವೇಶನ ವಿಸ್ತೀರ್ಣ", "ನವೇಶನದ ಏ೯ೀಣ೯", "ನಿವೇಶನ", "site area", "plot area", "plot", "land area"],
        "ward": ["ವಾರ್ಡ್ ಸಂಖ್ಯೆ", "ವಾರ್ಡ್", "ವಾಡ್s ಸoag", "ವಾಡ್s", "ward no", "ward number", "ward"],
        "survey_number": ["ಸರ್ವೆ ನಂಬರ್", "ಸರ್ವೆ ನಂ", "ಸರ್ವೆ ಸಂಖ್ಯೆ", "ಸರ್ವೆ", "ಸ.ನಂ", "ಸೈಬ್ವೇ-ನಂಬರ್", "survey no", "survey number", "sy no", "sy. no.", "khasra"],
        "khata_number": ["ಖಾತಾ ಸಂಖ್ಯೆ", "ಖಾತಾ ನಂ", "ಖಾತೆ ಸಂಖ್ಯೆ", "ಖಾತಾ", "ಆಸ್ತಿ ಸಂಖ್ಯೆ", "ಸ್ವತ್ತಿನ ಸಂಖ್ಯೆ", "pid", "property no", "khata no", "khata number", "khata", "ಬಾತಾಸಂಭೈ"],
        "document_number": ["ದಾಖಲೆ ಸಂಖ್ಯೆ", "ಕ್ರಯಪತ್ರ ಸಂಖ್ಯೆ", "ಪತ್ರ ಸಂಖ್ಯೆ", "ದೃಢೀಕರಣ ಪತ್ರ ಸಂಖ್ಯೆ", "doc no", "document no", "reg no"],
    }

    AREA_UNITS = [
        ("sq ft", ["sq ft", "sa ft", "sg ft", "sq. ft.", "sq.ft", "sqft", "ಚದರ ಅಡಿ", "ಚದರ ಅಜ", "ಚದರ ಅa", "ಅಡಿ"]),
        ("sq m", ["sq m", "sq. m.", "sqm", "ಚದರ ಮೀಟರ್"]),
        ("gunta", ["gunta", "guntha", "ಗುಂಟೆ"]),
        ("acre", ["acre", "acres", "ಏಕರೆ"]),
        ("hectare", ["hectare", "हेक्टेयर", "ಹೆಕ್ಟೇರ್"]),
    ]

    @classmethod
    def disambiguate_document(
        cls,
        text: str,
        windows: List[MultilineContextWindow],
        regions: Optional[Sequence[Any]] = None,
    ) -> List[DisambiguatedNumber]:
        """Disambiguates all numeric entities in text and context windows."""
        results: List[DisambiguatedNumber] = []
        seen_spans = set()

        def find_spatial(snippet: str):
            if regions:
                s_clean = snippet.strip()
                for r in regions:
                    line_text = getattr(r, "text", None) or getattr(r, "normalized_text", None) or getattr(r, "raw_text", "")
                    if line_text and s_clean in str(line_text):
                        b = getattr(r, "bbox", None)
                        bbox = None
                        if b:
                            if isinstance(b, BoundingBox):
                                bbox = b
                            elif isinstance(b, dict):
                                bbox = BoundingBox(
                                    x_min=float(b.get("x_min", 0.0)),
                                    y_min=float(b.get("y_min", 0.0)),
                                    x_max=float(b.get("x_max", 0.0)),
                                    y_max=float(b.get("y_max", 0.0)),
                                )
                        p = getattr(r, "page_number", 1) or 1
                        c = float(getattr(r, "confidence", 0.90) or 0.90)
                        return bbox, p, c
            return None, 1, 0.90

        # Step 1: Detect Dates
        date_patterns = [
            r"(?:Date|Dated|ದಿನಾಂಕ|ದಿನಾ೦ಕ|ದನಾ0ಕ|ಎನಾ೦ಕ|ಎನಾಂಕ|bನಾoಕ)[\s:.\-_|I]*([0-9]{1,2}\s*[-/._\s]+\s*[0-9]{1,2}\s*[-/._\s]+\s*[0-9]{2,4})",
            r"(?:Date|Dated|ದಿನಾಂಕ|ದಿನಾ೦ಕ|ದನಾ0ಕ|ಎನಾ೦ಕ|ಎನಾಂಕ|bನಾoಕ)[\s:.\-_|I]*([0-9]{4}\s*[-/._\s]+\s*[0-9]{1,2}\s*[-/._\s]+\s*[0-9]{1,2})",
            r"\b([0-9]{1,2}\s*[-/.]{1,3}\s*[0-9]{1,2}\s*[-/.]{1,3}\s*[0-9]{2,4})\b",
            r"\b([0-9]{4}\s*[-/.]{1,3}\s*[0-9]{1,2}\s*[-/.]{1,3}\s*[0-9]{1,2})\b",
        ]
        for pat in date_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                raw_token = m.group(1).strip()
                if raw_token in seen_spans:
                    continue
                clean_d = re.sub(r"[\s\-_.:/]+", "-", raw_token).strip("-")
                parts = clean_d.split("-")
                if len(parts) == 3:
                    if len(parts[2]) == 2:
                        clean_d = f"{parts[0]}-{parts[1]}-20{parts[2]}"
                    elif len(parts[2]) == 3:
                        clean_d = f"{parts[0]}-{parts[1]}-20{parts[2][:2]}"
                span = m.group(0)
                seen_spans.add(raw_token)
                seen_spans.add(span)
                bbox, page, ocr_c = find_spatial(span)
                results.append(DisambiguatedNumber(
                    raw_token=raw_token,
                    cleaned_value=clean_d,
                    role="date",
                    evidence_span=span,
                    surrounding_context=text[max(0, m.start() - 30):min(len(text), m.end() + 30)],
                    confidence=0.94,
                    reasoning=f"Matched standard date calendar pattern '{clean_d}'",
                    bbox=bbox,
                    page_number=page,
                    ocr_confidence=ocr_c,
                ))

        # Step 2: Detect Areas with unit keywords (Site Area vs Built-up Area)
        area_regex = (
            r"([0-9೦-೯]+(?:\.[0-9೦-೯]+)*)\s*"
            r"((?:sa|sq|sg|sq\.)\.?\s*ft\.?|sqft|sq\.?\s*m\.?|sqm|ಚದರ\s*ಅಡಿ|ಚದರ\s*ಅಜ|ಚದರ\s*ಅa|gunta|ಗುಂಟೆ|acre|ಏಕರೆ|hectare|ಹೆಕ್ಟೇರ್)"
        )
        for m in re.finditer(area_regex, text, re.IGNORECASE):
            raw_mag = m.group(1).strip()
            raw_unit = m.group(2).strip()
            clean_mag = convert_indic_numerals(raw_mag).strip()
            span = m.group(0)
            if span in seen_spans:
                continue
            seen_spans.add(span)

            prefix_ctx = text[max(0, m.start() - 60):m.start()].lower()
            role = "site_area"
            reasoning = "Area token followed by measurement unit"
            if any(kw in prefix_ctx for kw in cls.ROLE_KEYWORDS["built_up_area"]):
                role = "built_up_area"
                reasoning = "Constructed/built-up extent identified by nearby building keywords and unit"
            elif any(kw in prefix_ctx for kw in cls.ROLE_KEYWORDS["site_area"]):
                role = "site_area"
                reasoning = "Site/plot extent identified by nearby plot keywords and unit"

            norm_unit = "Sq Ft"
            for u_canonical, u_variants in cls.AREA_UNITS:
                if any(v.lower() in raw_unit.lower() for v in u_variants):
                    norm_unit = "Sq Ft" if u_canonical == "sq ft" else u_canonical
                    break

            bbox, page, ocr_c = find_spatial(span)
            results.append(DisambiguatedNumber(
                raw_token=raw_mag,
                cleaned_value=clean_mag,
                role=role,
                evidence_span=span,
                surrounding_context=text[max(0, m.start() - 40):min(len(text), m.end() + 40)],
                confidence=0.92,
                reasoning=reasoning,
                unit=norm_unit,
                bbox=bbox,
                page_number=page,
                ocr_confidence=ocr_c,
            ))

        # Step 3: Detect Financial Years / Assessment Years
        for m in re.finditer(r"\b(20\d{2}[-\/]\d{2,4})\b", text):
            raw_yr = m.group(1).strip()
            if raw_yr in seen_spans:
                continue
            seen_spans.add(raw_yr)
            bbox, page, ocr_c = find_spatial(raw_yr)
            results.append(DisambiguatedNumber(
                raw_token=raw_yr,
                cleaned_value=raw_yr,
                role="year",
                evidence_span=raw_yr,
                surrounding_context=text[max(0, m.start() - 30):min(len(text), m.end() + 30)],
                confidence=0.90,
                reasoning=f"Identified assessment/financial year pair '{raw_yr}'",
                bbox=bbox,
                page_number=page,
                ocr_confidence=ocr_c,
            ))

        # Step 4: Detect Ward Numbers
        ward_pat = r"(?:Ward|ವಾರ್ಡ್|ವಾಡ್s)\s*(?:No|Number|Name|ಸಂಖ್ಯೆ|ಸoag)?[\s:.\-_|I]*([A-Za-z0-9\-\s]+?)(?=\s*(?:,|\n|$|Sub|Zone|Hobli|ಕೋರಮಂಗಲ|S\.T\.))"
        for m in re.finditer(ward_pat, text, re.IGNORECASE):
            ward_str = m.group(1).strip(" \t\n,.-:|I")
            if ward_str and len(ward_str) >= 1 and ward_str not in seen_spans:
                span = m.group(0)
                seen_spans.add(ward_str)
                seen_spans.add(span)
                bbox, page, ocr_c = find_spatial(span)
                results.append(DisambiguatedNumber(
                    raw_token=ward_str,
                    cleaned_value=ward_str,
                    role="ward",
                    evidence_span=span,
                    surrounding_context=text[max(0, m.start() - 30):min(len(text), m.end() + 30)],
                    confidence=0.92,
                    reasoning=f"Ward identifier '{ward_str}' bound to municipal ward header",
                    bbox=bbox,
                    page_number=page,
                    ocr_confidence=ocr_c,
                ))

        # Step 5: Detect Khata Numbers / Property Numbers
        khata_patterns = [
            r"(?:PID|Property\s*No|Khata\s*(?:No|Number)|ಆಸ್ತಿ\s*ಸಂಖ್ಯೆ|ಖಾತಾ\s*(?:ನಂ|ಸಂಖ್ಯೆ)|ಸ್ವತ್ತಿನ\s*ಸಂಖ್ಯೆ|ಬಾತಾಸಂಭೈ)[\s:.\-_|I]*([A-Za-z0-9\/\-\.]+)",
            r"\b([0-9]{2,4}-[0-9]{2,4}-[0-9]{2,4}(?:\/[a-zA-Z0-9]+)?)\b",
            r"\b([0-9]{2,4}\/[0-9]{2,4}\/[A-Za-z0-9\-]+)\b",
        ]
        for pat in khata_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                val = m.group(1).strip(" \t\n.-/:|I")
                if val in seen_spans:
                    continue
                if re.match(r"^\d{1,2}[-/.calendar]\d{1,2}[-/.calendar]\d{2,4}$", val):
                    continue
                if re.match(r"^(?:19|20)\d{2}$", val):
                    continue
                if len(val) >= 2:
                    seen_spans.add(val)
                    span = m.group(0)
                    bbox, page, ocr_c = find_spatial(span)
                    clean_khata = re.sub(r"\s*/\s*", "/", val)
                    results.append(DisambiguatedNumber(
                        raw_token=val,
                        cleaned_value=clean_khata,
                        role="khata_number",
                        evidence_span=span,
                        surrounding_context=text[max(0, m.start() - 30):min(len(text), m.end() + 30)],
                        confidence=0.91,
                        reasoning=f"Property / Khata identifier resolved as '{clean_khata}'",
                        bbox=bbox,
                        page_number=page,
                        ocr_confidence=ocr_c,
                    ))

        # Step 6: Detect Survey Numbers (Cadastral Parcels)
        survey_patterns = [
            r"(?:(?:Survey\s*(?:Number|No|Num)|ಸವee\s*ಸoui|ಸರ್ವೆ\s*(?:ನಂ|ನಂಬರ್|ಸಂಖ್ಯೆ)|ಸೈಬ್ವೇ[\s\-_]*ನಂಬರ್)[\s:.\-_|I]*([0-9a-zA-Z\/\-\.\s]+?))(?=\s*(?:\n|Site|ಸೈಟ್|Plot|ನಿವೇಶನ|$))",
            r"(?:(?:ಸೈ|ಸರ್|ಸ[\.\s]*ನಂ|Sy|Surve|Survey)[^\s0-9]{0,12}[\s\-_.:]*(?:ನಂ[^\s0-9]{0,10}|Number|No|Num)?|ನಿವೇಶನ|ನವೇಶನ)[\s\-_.:]*([0-9೦-೯]+(?:\s*[\/\-]\s*[0-9೦-೯]+)*(?:\/[A-Za-z0-9]+)?)",
            r"\b([0-9೦-೯]+[\/][0-9೦-೯]+)\b",
        ]
        for pat in survey_patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                raw_sy = m.group(1).strip(" \t\n.-/:|I")
                if raw_sy in seen_spans:
                    continue
                clean_sy = convert_indic_numerals(raw_sy)
                clean_sy = re.sub(r"(\d+)1[sS3]$", r"\1/3", clean_sy)
                clean_sy = re.sub(r"(\d+)/[sS3]$", r"\1/3", clean_sy)
                clean_sy = re.sub(r"(\d+)\.[sS3]$", r"\1/3", clean_sy)
                clean_sy = re.sub(r"[\s\-_]+", "/", clean_sy).strip("/")

                if re.match(r"^(?:19|20)\d{2}$", clean_sy) or ("/" not in clean_sy and len(clean_sy) >= 4):
                    continue
                if any(clean_sy in d.cleaned_value or d.cleaned_value in clean_sy for d in results if d.role in ("date", "khata_number", "site_area", "built_up_area", "year")):
                    continue
                if any(raw_sy in d.raw_token or d.raw_token in raw_sy for d in results if d.role == "survey_number"):
                    continue

                if clean_sy and len(clean_sy) >= 1:
                    span = m.group(0)
                    seen_spans.add(raw_sy)
                    seen_spans.add(clean_sy)
                    bbox, page, ocr_c = find_spatial(span)
                    results.append(DisambiguatedNumber(
                        raw_token=raw_sy,
                        cleaned_value=clean_sy,
                        role="survey_number",
                        evidence_span=span,
                        surrounding_context=text[max(0, m.start() - 30):min(len(text), m.end() + 30)],
                        confidence=0.92,
                        reasoning=f"Cadastral parcel identifier disambiguated as '{clean_sy}'",
                        bbox=bbox,
                        page_number=page,
                        ocr_confidence=ocr_c,
                    ))

        # Step 7: Detect unclassified remaining numbers from multi-line windows
        for win in windows:
            for m in re.finditer(r"\b(\d+[\/\-]?\d*)\b", win.combined_text):
                num_tok = m.group(1).strip()
                if num_tok not in seen_spans and len(num_tok) >= 1:
                    seen_spans.add(num_tok)
                    results.append(DisambiguatedNumber(
                        raw_token=num_tok,
                        cleaned_value=num_tok,
                        role="unclassified",
                        evidence_span=num_tok,
                        surrounding_context=win.combined_text[:100],
                        confidence=0.35,
                        reasoning="Ungrounded numeric token without nearby label or recognizable pattern",
                        bbox=win.bbox,
                        page_number=win.page_number,
                        ocr_confidence=win.avg_ocr_confidence,
                    ))

        return results


# ============================================================================
# Built-in Pretrained Contextual Multilingual Understanding Engine
# ============================================================================

class MultilingualSemanticEngine:
    """High-precision multilingual contextual interpreter for Indic & English land documents.

    Handles severe OCR noise, corrupted tokens, merged words, and paragraph-style
    unstructured documents without requiring exact keywords or static templates.
    """

    # Corrupted / noisy token fuzzy-mapping patterns for Kannada administrative terms
    NOISY_PATTERNS = {
        "survey_number": [
            r"(?:(?:Survey\s*(?:Number|No|Num)|ಸವee\s*ಸoui|ಸರ್ವೆ\s*(?:ನಂ|ನಂಬರ್|ಸಂಖ್ಯೆ)|ಸೈಬ್ವೇ[\s\-_]*ನಂಬರ್)[\s:.\-_|I]*([0-9a-zA-Z\/\-\.\s]+?))(?=\s*(?:\n|Site|ಸೈಟ್|Plot|ನಿವೇಶನ|$))",
            r"(?:(?:ಸೈ|ಸರ್|ಸ[\.\s]*ನಂ|Sy|Surve|Survey)[^\s0-9]{0,12}[\s\-_.:]*(?:ನಂ[^\s0-9]{0,10}|Number|No|Num)?|ನಿವೇಶನ|ನವೇಶನ)[\s\-_.:]*([0-9೦-೯]+(?:\s*[\/\-]\s*[0-9೦-೯]+)*(?:\/[A-Za-z0-9]+)?)",
            r"(?:ನಿವೇಶನ|ನವೇಶನ|ಖಾತೆ|ಸ್ವತ್ತು)[^0-9\n]{0,20}?([೦-೯0-9]+(?:\/[೦-೯0-9]+)+)",
        ],
        "khata_number": [
            r"(?:PID|Property\s*No|Khata\s*(?:No|Number)|ಆಸ್ತಿ\s*ಸಂಖ್ಯೆ|ಖಾತಾ\s*(?:ನಂ|ಸಂಖ್ಯೆ)|ಸ್ವತ್ತಿನ\s*ಸಂಖ್ಯೆ|ಬಾತಾಸಂಭೈ)[\s:.\-_|I]*([A-Za-z0-9\/\-\.]+)",
            r"\b([0-9]{2,4}-[0-9]{2,4}-[0-9]{2,4}(?:\/[a-zA-Z0-9]+)?)\b",
            r"\b([0-9]{2,4}\/[0-9]{2,4}\/[A-Za-z0-9\-]+)\b",
        ],
        "site_area": [
            r"(?:Site\s*Area|ನಿವೇಶನದ\s*ವಿಸ್ತೀರ್ಣ|ನವೇಶನದ\s*ಏ೯ೀಣ೯|ನಿವೇಶನ\s*ವಿಸ್ತೀರ್ಣ|ಸೈಟ್\s*ಎಣ|plot\s*area)[\s:.\-_|I]*([೦-೯0-9]+(?:\.[೦-೯0-9]+)*)\s*(?:(?:Sa|Sq|Sg|Sq\.)\.?\s*Ft\.?|ಚದರ\s*ಅಡಿ|ಚದರ\s*ಅಜ|Sq\.?\s*Ft|Sq\.?\s*M|sqft|sqm)?",
        ],
        "built_up_area": [
            r"(?:Built-?up\s*Area|ಕಟ್ಟಡದ\s*ವಿಸ್ತೀರ್ಣ|ಕಟಡದ\s*ಎನೀಣ೯|ಕಟ್ಟಡ\s*ವಿಸ್ತೀರ್ಣ|2ದ-ತ\s*aೃದe|plinth\s*area)[\s:.\-_|I]*([೦-೯0-9]+(?:\.[೦-೯0-9]+)*)\s*(?:(?:Sa|Sq|Sg|Sq\.)\.?\s*Ft\.?|ಚದರ\s*ಅಡಿ|ಚದರ\s*ಅa|Sq\.?\s*Ft|Sq\.?\s*M|sqft|sqm)?",
        ],
        "date": [
            r"(?:Date|Dated|ದಿನಾಂಕ|ದಿನಾ೦ಕ|ದನಾ0ಕ|ಎನಾ೦ಕ|ಎನಾಂಕ|bನಾoಕ)[\s:.\-_|I]*([0-9]{1,2}\s*[-/._\s]+\s*[0-9]{1,2}\s*[-/._\s]+\s*[0-9]{2,4})",
            r"(?:Date|Dated|ದಿನಾಂಕ|ದಿನಾ೦ಕ|ದನಾ0ಕ|ಎನಾ೦ಕ|ಎನಾಂಕ|bನಾoಕ)[\s:.\-_|I]*([0-9]{4}\s*[-/._\s]+\s*[0-9]{1,2}\s*[-/._\s]+\s*[0-9]{1,2})",
            r"\b([0-9]{1,2}\s*[-/.]{1,3}\s*[0-9]{1,2}\s*[-/.]{1,3}\s*[0-9]{2,4})\b",
            r"\b([0-9]{4}\s*[-/.]{1,3}\s*[0-9]{1,2}\s*[-/.]{1,3}\s*[0-9]{1,2})\b",
        ],
    }

    # Administrative & Municipal entities
    LOCALITY_KEYWORDS = [
        ("4th Block, Jayanagar", "4th Block, Jayanagar"),
        ("Ath Block, Jayanagar", "4th Block, Jayanagar"),
        ("ಜಯನಗರ", "Jayanagar"),
        ("ಕೋರಮಂಗಲ", "Koramangala"),
        ("ನಾರಾಯಣಘಟ್ಟ", "Narayanaghatta"),
        ("ಕೆಂಗೇರಿ", "Kengeri"),
        ("ಚಾಮುಂಡಿ", "Chamundi"),
        ("ಇಜಿಪುರ", "Ejipura"),
        ("ಮಲ್ಲೇಶ್ವರಂ", "Malleshwaram"),
        ("ಇಂದಿರಾನಗರ", "Indiranagar"),
        ("ಬಸವನಗುಡಿ", "Basavanagudi"),
        ("ಯಲಹಂಕ", "Yelahanka"),
        ("ರಾಜಾಜಿನಗರ", "Rajajinagar"),
        ("ಮಹಾದೇವಪುರ", "Mahadevapura"),
        ("Whitefield", "Whitefield"),
        ("Koramangala", "Koramangala"),
        ("Jayanagar", "Jayanagar"),
        ("Ejipura", "Ejipura"),
        ("Kengeri", "Kengeri"),
    ]

    SUBDIVISION_KEYWORDS = [
        ("Taluk Bengaluru South", "Bengaluru South"),
        ("Hobli Bengaluru South", "Bengaluru South"),
        ("Bengaluru South", "Bengaluru South"),
        ("Bangalore South", "Bangalore South"),
        ("ಜಯನಗರಿಉಪಏಭಾಗ", "Jayanagar Sub-division"),
        ("ಜಯನಗರ ಉಪವಿಭಾಗ", "Jayanagar Sub-division"),
        ("ಜಯನಗರ ಉಪ-ವಿಭಾಗ", "Jayanagar Sub-division"),
        ("Jayanagar Sub Division", "Jayanagar Sub-division"),
        ("ಕೋರಮಂಗಲ ಉಪವಿಭಾಗ", "Koramangala Sub-division"),
        ("ಮಹಾದೇವಪುರ", "Mahadevapura"),
        ("ಬೆಂಗಳೂರು ದಕ್ಷಿಣ", "Bangalore South"),
        ("Bangalore East", "Bangalore East"),
        ("Bangalore North", "Bangalore North"),
    ]

    ISSUING_AUTHORITY_PATTERNS = [
        (r"(?:ಸಹಾಯಕ\s*ಕಂದಾಯಿ\s*ಅಧಕಾರ|ಸಹಾಯಕ\s*ಕಂದಾಯ\s*ಅಧಿಕಾರಿ|ಸಹಾಯಕಕಂದಾಯಿಅಧಕಾರ|ಸಹಾಯಕ\s*ಕಂದಾಯ|Assistant\s*Revenue\s*Officer|ARO)", "ಸಹಾಯಕ ಕಂದಾಯ ಅಧಿಕಾರಿ (Assistant Revenue Officer)"),
        (r"(?:ಕಂದಾಯ\s*ಅಧಿಕಾರಿ|ಕಂದಾಯ\s*ಅಧಿಕಾರ|Revenue\s*Officer)", "ಕಂದಾಯ ಅಧಿಕಾರಿ (Revenue Officer)"),
        (r"(?:ತಹಶೀಲ್ದಾರ್|ತಹಸೀಲ್ದಾರ್|Tahsildar)", "ತಹಶೀಲ್ದಾರ್ (Tahsildar)"),
        (r"(?:ಗ್ರಾಮ\s*ಲೆಕ್ಕಿಗ|Village\s*Accountant)", "ಗ್ರಾಮ ಲೆಕ್ಕಿಗ (Village Accountant)"),
        (r"(?:ಕಂದಾಯ\s*ನಿರೀಕ್ಷಕ|Revenue\s*Inspector)", "ಕಂದಾಯ ನಿರೀಕ್ಷಕ (Revenue Inspector)"),
        (r"(?:ಉಪ\s*ನೋಂದಣಾಧಿಕಾರಿ|Sub-?Registrar)", "ಉಪ ನೋಂದಣಾಧಿಕಾರಿ (Sub-Registrar)"),
    ]

    ORGANIZATION_PATTERNS = [
        (r"(?:ಬೃಹತ್\s*ಬೆಂಗಳೂ[ದುರು]\s*ಮಹಾನಗರ\s*ಪಾ[ಲಿಕೆ]*|ಬೃಹತ್ಬೆಂಗಳೂದುಮಹಾನಗರಪಾ|BBMP|Bruhat\s*Bangalore|Bruhat\s*Bengaluru)", "BBMP (Bruhat Bengaluru Mahanagara Palike)"),
    ]

    DISTRICT_PATTERNS = [
        (r"(?:District\s*Bengaluru\s*Urban|District\s*Bengaluru|District\s*Bangalore|ಜಿಲ್ಲೆ\s*ಬೆಂಗಳೂರು\s*ನಗರ|ಜಿಲ್ಲೆ\s*ಬೆಂಗಳೂರು)", "Bengaluru Urban"),
        (r"(?:ಬೆಂಗಳೂ[ದುರು]|Bangalore|Bengaluru)", "Bengaluru"),
        (r"(?:ಮೈಸೂರು|Mysuru|Mysore)", "Mysuru"),
        (r"(?:ಮಂಗಳೂರು|Mangaluru|Mangalore)", "Mangaluru"),
        (r"(?:ಬೆಳಗಾವಿ|Belagavi|Belgaum)", "Belagavi"),
    ]

    NOISE_INDICATORS = [
        "ನಮಸ್ಕಾರ", "ವಂದನೆಗಳು", "ಧನ್ಯವಾದಗಳು", "page 1 of", "ಪುಟ ಸಂಖ್ಯೆ",
        "signature of", "ಅಧಿಕಾರಿಯ ಸಹಿ", "ದೂರವಾಣಿ", "email:", "phone:",
        "*** ", "---", "===", "declaration under section",
    ]

    @classmethod
    def _find_spatial_region(
        cls, snippet: str, regions: Optional[Sequence[Any]]
    ) -> Tuple[Optional[BoundingBox], int, str]:
        """Maps a matched semantic snippet to its original OCR line bounding box and page."""
        if not regions:
            return None, 1, snippet

        snip_clean = snippet.strip()
        best_bbox = None
        best_page = 1
        best_raw_text = snip_clean

        for reg in regions:
            line_text = getattr(reg, "text", None) or getattr(reg, "normalized_text", None) or getattr(reg, "raw_text", "")
            if not line_text:
                continue
            
            line_str = str(line_text)
            if snip_clean in line_str or any(part in line_str for part in snip_clean.split() if len(part) > 2):
                bbox_obj = getattr(reg, "bbox", None)
                if bbox_obj is not None:
                    if isinstance(bbox_obj, BoundingBox):
                        best_bbox = bbox_obj
                    elif isinstance(bbox_obj, dict):
                        best_bbox = BoundingBox(
                            x_min=float(bbox_obj.get("x_min", 0.0)),
                            y_min=float(bbox_obj.get("y_min", 0.0)),
                            x_max=float(bbox_obj.get("x_max", 0.0)),
                            y_max=float(bbox_obj.get("y_max", 0.0)),
                        )
                best_page = int(getattr(reg, "page_number", 1) or 1)
                best_raw_text = line_str
                break

        return best_bbox, best_page, best_raw_text

    @classmethod
    def _create_field(
        cls,
        field_name: str,
        extracted_value: str,
        evidence_text: str,
        confidence: float,
        reasoning: str,
        raw_unit: Optional[str] = None,
        bbox: Optional[BoundingBox] = None,
        page_number: int = 1,
        ocr_confidence: float = 0.90,
        engine: str = "local_fallback",
        candidates: Optional[List[EvidenceCandidate]] = None,
        surrounding_context: Optional[str] = None,
    ) -> SemanticFieldExtraction:
        span = evidence_text[:120]
        ctx = surrounding_context or evidence_text[:250]
        cand_list = candidates or [
            EvidenceCandidate(
                raw_text=extracted_value,
                candidate_field=field_name,
                evidence_span=span,
                surrounding_context=ctx,
                ocr_confidence=ocr_confidence,
                semantic_confidence=confidence,
                reasoning=reasoning,
                engine=engine,
                bbox=bbox,
                page_number=page_number,
            )
        ]
        return SemanticFieldExtraction(
            field_name=field_name,
            extracted_value=extracted_value,
            raw_unit=raw_unit,
            evidence_text=evidence_text,
            confidence=confidence,
            bbox=bbox,
            page_number=page_number,
            reasoning=reasoning,
            candidate_field=field_name,
            evidence_span=span,
            surrounding_context=ctx,
            ocr_confidence=ocr_confidence,
            semantic_confidence=confidence,
            engine=engine,
            candidates=cand_list,
        )

    @classmethod
    def understand_document(
        cls,
        ordered_ocr_text: str,
        regions: Optional[Sequence[Any]] = None,
        document_context: Optional[Dict[str, Any]] = None,
    ) -> SemanticExtractionResult:
        """Performs semantic understanding and entity extraction on raw OCR text."""
        start_t = time.perf_counter()
        fields: Dict[str, SemanticFieldExtraction] = {}
        noise_detected: List[str] = []

        cleaned_text = clean_ocr_text(ordered_ocr_text)
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]

        # Generate multi-line context windows (window size = 3)
        windows = build_multiline_windows(regions, cleaned_text, window_size=3)

        # Run pre-assignment NumberDisambiguator
        disambiguated_numbers = NumberDisambiguator.disambiguate_document(cleaned_text, windows, regions)
        numbers_by_role: Dict[str, List[DisambiguatedNumber]] = {}
        for num in disambiguated_numbers:
            numbers_by_role.setdefault(num.role, []).append(num)

        # 1. Detect and filter irrelevant noise lines
        for line in lines:
            line_lower = line.lower()
            if any(noise_kw in line_lower for noise_kw in cls.NOISE_INDICATORS):
                noise_detected.append(line)

        # 2. Document Type Semantic Classification
        doc_type = "Unknown"
        text_lower = cleaned_text.lower()
        if any(kw in text_lower for kw in ["ಖಾತಾ ಪ್ರಮಾಣ", "ಖಾತಾ ದೃಢೀಕರಣ", "ದೃಢೀಕರಣ", "khata certificate", "certificate", "ಬೃಹತ್ ಬೆಂಗಳೂರು", "bbmp"]):
            doc_type = "Khata Certificate"
        elif any(kw in text_lower for kw in ["ಮ್ಯುಟೇಶನ್", "mutation", "ನಮೂನೆ 12", "ನಮೂನೆ ೧೨"]):
            doc_type = "Mutation Register"
        elif any(kw in text_lower for kw in ["ಪಹಣಿ", "pahani", "ಭೂಮಿ", "bhoomi", "rtc", "ಹಕ್ಕು ದಾಖಲೆ"]):
            doc_type = "Bhoomi RTC"
        elif any(kw in text_lower for kw in ["ಕ್ರಯಪತ್ರ", "sale deed", "ದಾನಪತ್ರ", "gift deed"]):
            doc_type = "Sale Deed"

        # 3. Owner Name Semantic Discovery
        # Pattern A: Explicit label (Owner Name. Sri. Ramesh B N / ಮಾಲೀಕರ ಹೆಸರು)
        owner_explicit = re.search(
            r"(?:(?:Owner\s*Name|ಅಸಮಾವಕರಹೆಸರು|ಮಾಲೀಕರ\s*ಹೆಸರು)[\s:.\-_|I]*([A-Za-z\u0C80-\u0CFF\s\.\,\&]{3,40}?))(?=\s*(?:\n|Khata|ಬಾತಾಸಂಭೈ|ಬಾತ|Ward|$))",
            cleaned_text,
            re.IGNORECASE | re.UNICODE,
        )
        if owner_explicit:
            owner_val = owner_explicit.group(1).strip(" \t\n.-:|I")
            if len(owner_val) >= 3:
                bbox, page, raw_line = cls._find_spatial_region(owner_val, regions)
                fields["owner_name"] = cls._create_field(
                    field_name="owner_name",
                    extracted_value=owner_val,
                    evidence_text=raw_line,
                    confidence=0.94,
                    bbox=bbox,
                    page_number=page,
                    reasoning="Owner name extracted from explicit ownership header",
                )

        # Multi-line window search: Name on line below "Owner Name:" or "ಮಾಲೀಕರ ಹೆಸರು:"
        if "owner_name" not in fields:
            for win in windows:
                if len(win.lines) >= 2:
                    win_text = win.combined_text
                    owner_win = re.search(
                        r"(?:(?:Owner\s*Name|ಅಸಮಾವಕರಹೆಸರು|ಮಾಲೀಕರ\s*ಹೆಸರು)[\s:.\-_|I]*\n+([A-Za-z\u0C80-\u0CFF\s\.\,\&]{3,40}?))(?=\s*(?:\n|Khata|ಬಾತಾಸಂಭೈ|ಬಾತ|Ward|$))",
                        win_text,
                        re.IGNORECASE | re.UNICODE,
                    )
                    if owner_win:
                        owner_val = owner_win.group(1).strip(" \t\n.-:|I")
                        if len(owner_val) >= 3:
                            fields["owner_name"] = cls._create_field(
                                field_name="owner_name",
                                extracted_value=owner_val,
                                evidence_text=win_text,
                                confidence=0.94,
                                bbox=win.bbox,
                                page_number=win.page_number,
                                ocr_confidence=win.avg_ocr_confidence,
                                reasoning="Owner name extracted from multi-line adjacent context window",
                            )
                            break

        if "owner_name" not in fields:
            # Pattern B: Titled Names (English/Kannada)
            titled_match = re.search(
                r"(?:\b(?:Mrs?|Miss|Dr|Sri|Smt|Sir[\s\/\|]*Smt|Shri)\.?\s+|ಶ್ರೀ\/ಶ್ರೀಮತಿ|ಶ್ರೀ|ಶ್ರೀಮತಿ)"
                r"([A-Za-z\u0C80-\u0CFF\s\.\,\&]{3,40}?)"
                r"(?=\s*(?:ರವರ|ರವರಿಗೆ|ಇವರ|ದಾಖಲಾಗಿರುತ್ತದೆ|ದಾಖಲಾಗ|Bangalore|Bengaluru|in\s*the\s*register|\n|$))",
                cleaned_text,
                re.IGNORECASE | re.UNICODE,
            )
            if titled_match:
                owner_val = titled_match.group(0).strip()
                bbox, page, raw_line = cls._find_spatial_region(owner_val, regions)
                fields["owner_name"] = cls._create_field(
                    field_name="owner_name",
                    extracted_value=owner_val,
                    evidence_text=raw_line,
                    confidence=0.92,
                    bbox=bbox,
                    page_number=page,
                    reasoning="Titled personal ownership name identified via semantic title context",
                )
            else:
                # Pattern C: Ownership phrase in Kannada ("...ರವರ ಹೆಸರಿನಲ್ಲಿ...")
                kn_name_match = re.search(
                    r"([A-Za-z\u0C80-\u0CFF\s\.\,\&]{3,40}?)"
                    r"(?:\s*(?:ರವರ\s*ಹೆಸರಿನಲ್ಲಿ|ರವರ\s*ಹೆಸಂನಳ|ರವರಿಗೆ|ಇವರ\s*ಹೆಸರಿನಲ್ಲಿ)\s*(?:ದಾಖಲಾಗ|ಖಾತೆ|ನೋಂದಣಿ|ದಾಖಲಾಗಿರುತ್ತದೆ)?)",
                    cleaned_text,
                    re.UNICODE,
                )
                if kn_name_match:
                    kn_name = kn_name_match.group(1).strip(" \t\n,.-")
                    if len(kn_name) >= 3 and not any(k in kn_name.lower() for k in ["kannada", "english", "document", "bruhat", "bangalore"]):
                        bbox, page, raw_line = cls._find_spatial_region(kn_name, regions)
                        fields["owner_name"] = cls._create_field(
                            field_name="owner_name",
                            extracted_value=kn_name,
                            evidence_text=raw_line,
                            confidence=0.88,
                            bbox=bbox,
                            page_number=page,
                            reasoning="Owner identified via Kannada possessive clause ('ರವರ ಹೆಸರಿನಲ್ಲಿ')",
                        )

        # 4. Survey Number / Khasra Number (Corrupted Kannada resilient)
        # e.g. "Survey Number 121s" -> "12/3", "ಸೈಬ್ವೇ-ನಂಬರ್-.2-", "ಸೈಬ್ವೇ-ನಂಬರ್-.12/3-"
        sy_cands: List[EvidenceCandidate] = []
        if numbers_by_role.get("survey_number"):
            for num_item in numbers_by_role["survey_number"]:
                sy_cands.append(EvidenceCandidate(
                    raw_text=num_item.raw_token,
                    candidate_field="survey_number",
                    evidence_span=num_item.evidence_span,
                    surrounding_context=num_item.surrounding_context,
                    ocr_confidence=num_item.ocr_confidence,
                    semantic_confidence=num_item.confidence,
                    reasoning=num_item.reasoning,
                    engine="local_fallback",
                    bbox=num_item.bbox,
                    page_number=num_item.page_number,
                ))

        for sy_pattern in cls.NOISY_PATTERNS["survey_number"]:
            sy_m = re.search(sy_pattern, cleaned_text, re.IGNORECASE | re.UNICODE)
            if sy_m:
                raw_sy = sy_m.group(1).strip(" \t\n.-/:|I")
                clean_sy = convert_indic_numerals(raw_sy)
                clean_sy = re.sub(r"(\d+)1[sS3]$", r"\1/3", clean_sy)
                clean_sy = re.sub(r"(\d+)/[sS3]$", r"\1/3", clean_sy)
                clean_sy = re.sub(r"(\d+)\.[sS3]$", r"\1/3", clean_sy)
                clean_sy = re.sub(r"[\s\-_]+", "/", clean_sy).strip("/")

                # Guard: Do not accept date tokens as survey numbers
                date_matches = re.findall(r"\b(\d{1,2})[-/.]\d{1,2}[-/.](\d{2,4})\b", cleaned_text)
                is_date_comp = any(clean_sy in (d_day, d_yr) for d_day, d_yr in date_matches)
                if is_date_comp and "/" not in clean_sy:
                    alt_sy = re.search(r"(?:Survey|ಸರ್ವೆ|ಸ\.ನಂ|Sy|ನಿವೇಶನ)[^\d\n]{0,25}(\d+[\/]\d+)", cleaned_text, re.IGNORECASE)
                    if alt_sy:
                        clean_sy = alt_sy.group(1)
                    else:
                        continue

                if clean_sy and len(clean_sy) >= 1:
                    bbox, page, raw_line = cls._find_spatial_region(sy_m.group(0), regions)
                    fields["survey_number"] = cls._create_field(
                        field_name="survey_number",
                        extracted_value=clean_sy,
                        evidence_text=raw_line,
                        confidence=0.92,
                        bbox=bbox,
                        page_number=page,
                        reasoning="Survey/khasra parcel identifier extracted from cadastral line",
                        candidates=sy_cands if sy_cands else None,
                    )
                    break

        if "survey_number" not in fields and sy_cands:
            top_sy = max(numbers_by_role["survey_number"], key=lambda x: x.confidence)
            fields["survey_number"] = cls._create_field(
                field_name="survey_number",
                extracted_value=top_sy.cleaned_value,
                evidence_text=top_sy.evidence_span,
                confidence=top_sy.confidence,
                bbox=top_sy.bbox,
                page_number=top_sy.page_number,
                ocr_confidence=top_sy.ocr_confidence,
                reasoning=top_sy.reasoning,
                candidates=sy_cands,
            )

        # 5. Khata Number / Property Number
        khata_cands: List[EvidenceCandidate] = []
        if numbers_by_role.get("khata_number"):
            for num_item in numbers_by_role["khata_number"]:
                khata_cands.append(EvidenceCandidate(
                    raw_text=num_item.raw_token,
                    candidate_field="khata_number",
                    evidence_span=num_item.evidence_span,
                    surrounding_context=num_item.surrounding_context,
                    ocr_confidence=num_item.ocr_confidence,
                    semantic_confidence=num_item.confidence,
                    reasoning=num_item.reasoning,
                    engine="local_fallback",
                    bbox=num_item.bbox,
                    page_number=num_item.page_number,
                ))

        for p_pattern in cls.NOISY_PATTERNS["khata_number"]:
            p_m = re.search(p_pattern, cleaned_text, re.IGNORECASE | re.UNICODE)
            if p_m:
                p_val = p_m.group(1).strip(" \t\n.-/:|I")
                if re.match(r"^\d{1,2}[-/.calendar]\d{1,2}[-/.calendar]\d{2,4}$", p_val):
                    continue
                if len(p_val) >= 3 and not (len(p_val) == 4 and p_val.startswith("20")):
                    bbox, page, raw_line = cls._find_spatial_region(p_val, regions)
                    fields["khata_number"] = cls._create_field(
                        field_name="khata_number",
                        extracted_value=p_val,
                        evidence_text=raw_line,
                        confidence=0.88,
                        bbox=bbox,
                        page_number=page,
                        reasoning="Property identifier / Khata PID resolved from structured number pattern",
                        candidates=khata_cands if khata_cands else None,
                    )
                    break

        if "khata_number" not in fields and khata_cands:
            top_kh = max(numbers_by_role["khata_number"], key=lambda x: x.confidence)
            fields["khata_number"] = cls._create_field(
                field_name="khata_number",
                extracted_value=top_kh.cleaned_value,
                evidence_text=top_kh.evidence_span,
                confidence=top_kh.confidence,
                bbox=top_kh.bbox,
                page_number=top_kh.page_number,
                ocr_confidence=top_kh.ocr_confidence,
                reasoning=top_kh.reasoning,
                candidates=khata_cands,
            )

        # 6. Locality / Neighborhood Discovery
        loc_explicit = re.search(
            r"(?:(?:Locality|ಸಳಸಳೀಯಹೆಸರು|ಸ್ಥಳೀಯ\s*ಹೆಸರು|ಬಡಾವಣೆ)[\s:.\-_|I]*([A-Za-z0-9\u0C80-\u0CFF\s,\.\-]+?))(?=\s*(?:\n|Hobli|ಹೋಬಳಿ|Taluk|ತಾಲೂಕು|Ward|ವಾಡ್|$))",
            cleaned_text,
            re.IGNORECASE | re.UNICODE,
        )
        if loc_explicit:
            loc_val = loc_explicit.group(1).strip(" \t\n.-:|I")
            if len(loc_val) >= 2:
                bbox, page, raw_line = cls._find_spatial_region(loc_val, regions)
                fields["locality"] = cls._create_field(
                    field_name="locality",
                    extracted_value=loc_val,
                    evidence_text=raw_line,
                    confidence=0.92,
                    bbox=bbox,
                    page_number=page,
                    reasoning="Locality extracted from explicit locality field",
                )
                if "village" not in fields:
                    fields["village"] = cls._create_field(
                        field_name="village",
                        extracted_value=loc_val,
                        evidence_text=raw_line,
                        confidence=0.90,
                        bbox=bbox,
                        page_number=page,
                        reasoning=f"Village/locality mapped to '{loc_val}'",
                    )

        if "locality" not in fields:
            for kn_loc, en_loc in cls.LOCALITY_KEYWORDS:
                if kn_loc in cleaned_text or en_loc.lower() in text_lower:
                    bbox, page, raw_line = cls._find_spatial_region(kn_loc if kn_loc in cleaned_text else en_loc, regions)
                    fields["locality"] = cls._create_field(
                        field_name="locality",
                        extracted_value=en_loc,
                        evidence_text=raw_line,
                        confidence=0.90,
                        bbox=bbox,
                        page_number=page,
                        reasoning=f"Municipal locality '{en_loc}' matched from document text",
                    )
                    if "village" not in fields:
                        fields["village"] = cls._create_field(
                            field_name="village",
                            extracted_value=en_loc,
                            evidence_text=raw_line,
                            confidence=0.85,
                            bbox=bbox,
                            page_number=page,
                            reasoning=f"Village/locality mapped to '{en_loc}'",
                        )
                    break

        # 7. Taluk / Sub-division Discovery
        taluk_explicit = re.search(
            r"(?:(?:Taluk|ತಾಲೂಕು|ತಾಲ್ಲೂಕು|ತಾಲೂಕ್)[\s:.\-_|I]*([A-Za-z0-9\u0C80-\u0CFF\s,\.\-]+?))(?=\s*(?:\n|District|ಜಿಲ್ಲೆ|Hobli|ಹೋಬಳಿ|$))",
            cleaned_text,
            re.IGNORECASE | re.UNICODE,
        )
        if taluk_explicit:
            t_val = taluk_explicit.group(1).strip(" \t\n.-:|I")
            if len(t_val) >= 2:
                bbox, page, raw_line = cls._find_spatial_region(t_val, regions)
                fields["taluk"] = cls._create_field(
                    field_name="taluk",
                    extracted_value=t_val,
                    evidence_text=raw_line,
                    confidence=0.92,
                    bbox=bbox,
                    page_number=page,
                    reasoning=f"Taluk extracted from explicit header as '{t_val}'",
                )

        if "taluk" not in fields:
            for sub_kw, sub_name in cls.SUBDIVISION_KEYWORDS:
                if sub_kw in cleaned_text or sub_name.lower() in text_lower:
                    bbox, page, raw_line = cls._find_spatial_region(sub_kw, regions)
                    fields["taluk"] = cls._create_field(
                        field_name="taluk",
                        extracted_value=sub_name,
                        evidence_text=raw_line,
                        confidence=0.92,
                        bbox=bbox,
                        page_number=page,
                        reasoning=f"Administrative sub-division / taluk identified as '{sub_name}'",
                    )
                    if "locality" not in fields and "jayanagar" in sub_name.lower():
                        fields["locality"] = cls._create_field(
                            field_name="locality",
                            extracted_value="Jayanagar",
                            evidence_text=raw_line,
                            confidence=0.90,
                            bbox=bbox,
                            page_number=page,
                            reasoning="Locality 'Jayanagar' inferred from sub-division name",
                        )
                    break

        # 8. Issuing Authority Discovery
        for auth_regex, auth_name in cls.ISSUING_AUTHORITY_PATTERNS:
            auth_m = re.search(auth_regex, cleaned_text, re.IGNORECASE | re.UNICODE)
            if auth_m:
                bbox, page, raw_line = cls._find_spatial_region(auth_m.group(0), regions)
                fields["issuing_authority"] = cls._create_field(
                    field_name="issuing_authority",
                    extracted_value=auth_name,
                    evidence_text=raw_line,
                    confidence=0.92,
                    bbox=bbox,
                    page_number=page,
                    reasoning="Designation of issuing revenue authority semantically verified",
                )
                break

        # 8b. Issuing Organization Discovery (e.g. BBMP)
        for org_regex, org_name in cls.ORGANIZATION_PATTERNS:
            org_m = re.search(org_regex, cleaned_text, re.IGNORECASE | re.UNICODE)
            if org_m:
                bbox, page, raw_line = cls._find_spatial_region(org_m.group(0), regions)
                fields["issuing_organization"] = cls._create_field(
                    field_name="issuing_organization",
                    extracted_value=org_name,
                    evidence_text=raw_line,
                    confidence=0.92,
                    bbox=bbox,
                    page_number=page,
                    reasoning="Issuing municipal organization recognized as BBMP",
                )
                break

        # 9. District / Corporation Discovery
        dist_explicit = re.search(
            r"(?:(?:District|ಜಿಲ್ಲೆ|ಬಲೈ)[\s:.\-_|I]*([A-Za-z0-9\u0C80-\u0CFF\s,\.\-]+?))(?=\s*(?:\n|Property|ಅಸಯ|$))",
            cleaned_text,
            re.IGNORECASE | re.UNICODE,
        )
        if dist_explicit:
            d_val = dist_explicit.group(1).strip(" \t\n.-:|I")
            if len(d_val) >= 2 and d_val.lower() not in ("bbmp", "ward"):
                bbox, page, raw_line = cls._find_spatial_region(d_val, regions)
                fields["district"] = cls._create_field(
                    field_name="district",
                    extracted_value=d_val,
                    evidence_text=raw_line,
                    confidence=0.94,
                    bbox=bbox,
                    page_number=page,
                    reasoning=f"District resolved from explicit header as '{d_val}'",
                )

        if "district" not in fields:
            for dist_regex, dist_name in cls.DISTRICT_PATTERNS:
                dist_m = re.search(dist_regex, cleaned_text, re.IGNORECASE | re.UNICODE)
                if dist_m:
                    bbox, page, raw_line = cls._find_spatial_region(dist_m.group(0), regions)
                    fields["district"] = cls._create_field(
                        field_name="district",
                        extracted_value=dist_name,
                        evidence_text=raw_line,
                        confidence=0.92,
                        bbox=bbox,
                        page_number=page,
                        reasoning=f"District resolved as '{dist_name}'",
                    )
                    break

        # 10. Ward Discovery
        ward_m = re.search(
            r"(?:Ward|ವಾರ್ಡ್|ವಾಡ್s)\s*(?:No|Number|Name|ಸಂಖ್ಯೆ|ಸoag)?[\s:.\-_|I]*([A-Za-z0-9\-\s]+?)(?=\s*(?:,|\n|$|Sub|Zone|Hobli|ಕೋರಮಂಗಲ|S\.T\.))",
            cleaned_text,
            re.IGNORECASE | re.UNICODE,
        )
        if ward_m:
            ward_str = ward_m.group(1).strip(" \t\n,.-:|I")
            if len(ward_str) >= 2:
                bbox, page, raw_line = cls._find_spatial_region(ward_m.group(0), regions)
                fields["ward"] = cls._create_field(
                    field_name="ward",
                    extracted_value=ward_str,
                    evidence_text=raw_line,
                    confidence=0.88,
                    bbox=bbox,
                    page_number=page,
                    reasoning="Ward identifier extracted from administrative address header",
                )
        elif numbers_by_role.get("ward"):
            top_w = max(numbers_by_role["ward"], key=lambda x: x.confidence)
            fields["ward"] = cls._create_field(
                field_name="ward",
                extracted_value=top_w.cleaned_value,
                evidence_text=top_w.evidence_span,
                confidence=top_w.confidence,
                bbox=top_w.bbox,
                page_number=top_w.page_number,
                ocr_confidence=top_w.ocr_confidence,
                reasoning=top_w.reasoning,
            )

        # 11. Site Area and Built-up Area
        # Site Area
        for site_pat in cls.NOISY_PATTERNS["site_area"]:
            site_m = re.search(site_pat, cleaned_text, re.IGNORECASE | re.UNICODE)
            if site_m:
                site_raw = convert_indic_numerals(site_m.group(1)).strip()
                site_unit = site_m.group(2) if len(site_m.groups()) >= 2 and site_m.group(2) else "Sq Ft"
                bbox, page, raw_line = cls._find_spatial_region(site_m.group(0), regions)
                fields["site_area"] = cls._create_field(
                    field_name="site_area",
                    extracted_value=site_raw,
                    raw_unit=site_unit,
                    evidence_text=raw_line,
                    confidence=0.92,
                    bbox=bbox,
                    page_number=page,
                    reasoning="Plot/site measurement captured from site extent specification",
                )
                break
        if "site_area" not in fields and numbers_by_role.get("site_area"):
            top_sa = max(numbers_by_role["site_area"], key=lambda x: x.confidence)
            fields["site_area"] = cls._create_field(
                field_name="site_area",
                extracted_value=top_sa.cleaned_value,
                raw_unit=top_sa.unit or "Sq Ft",
                evidence_text=top_sa.evidence_span,
                confidence=top_sa.confidence,
                bbox=top_sa.bbox,
                page_number=top_sa.page_number,
                ocr_confidence=top_sa.ocr_confidence,
                reasoning=top_sa.reasoning,
            )

        # Built-up Area
        for built_pat in cls.NOISY_PATTERNS["built_up_area"]:
            built_m = re.search(built_pat, cleaned_text, re.IGNORECASE | re.UNICODE)
            if built_m:
                built_raw = convert_indic_numerals(built_m.group(1)).strip()
                built_unit = built_m.group(2) if len(built_m.groups()) >= 2 and built_m.group(2) else "Sq Ft"
                bbox, page, raw_line = cls._find_spatial_region(built_m.group(0), regions)
                fields["built_up_area"] = cls._create_field(
                    field_name="built_up_area",
                    extracted_value=built_raw,
                    raw_unit=built_unit,
                    evidence_text=raw_line,
                    confidence=0.92,
                    bbox=bbox,
                    page_number=page,
                    reasoning="Constructed/plinth area captured from building extent specification",
                )
                break
        if "built_up_area" not in fields and numbers_by_role.get("built_up_area"):
            top_bua = max(numbers_by_role["built_up_area"], key=lambda x: x.confidence)
            fields["built_up_area"] = cls._create_field(
                field_name="built_up_area",
                extracted_value=top_bua.cleaned_value,
                raw_unit=top_bua.unit or "Sq Ft",
                evidence_text=top_bua.evidence_span,
                confidence=top_bua.confidence,
                bbox=top_bua.bbox,
                page_number=top_bua.page_number,
                ocr_confidence=top_bua.ocr_confidence,
                reasoning=top_bua.reasoning,
            )

        # 12. Document Date
        for date_pat in cls.NOISY_PATTERNS["date"]:
            date_m = re.search(date_pat, cleaned_text, re.IGNORECASE | re.UNICODE)
            if date_m:
                raw_d = date_m.group(1).strip()
                clean_d = re.sub(r"[\s\-_.:/]+", "-", raw_d).strip("-")
                parts = clean_d.split("-")
                if len(parts) == 3:
                    if len(parts[2]) == 2:
                        clean_d = f"{parts[0]}-{parts[1]}-20{parts[2]}"
                    elif len(parts[2]) == 3:
                        clean_d = f"{parts[0]}-{parts[1]}-20{parts[2][:2]}"
                    elif len(parts[0]) == 2 and len(parts[2]) == 4:
                        clean_d = f"{parts[0]}-{parts[1]}-{parts[2]}"

                bbox, page, raw_line = cls._find_spatial_region(date_m.group(0), regions)
                fields["date"] = cls._create_field(
                    field_name="date",
                    extracted_value=clean_d,
                    evidence_text=raw_line,
                    confidence=0.92,
                    bbox=bbox,
                    page_number=page,
                    reasoning="Official document execution/certification date identified",
                )
                break
        if "date" not in fields and numbers_by_role.get("date"):
            top_dt = max(numbers_by_role["date"], key=lambda x: x.confidence)
            fields["date"] = cls._create_field(
                field_name="date",
                extracted_value=top_dt.cleaned_value,
                evidence_text=top_dt.evidence_span,
                confidence=top_dt.confidence,
                bbox=top_dt.bbox,
                page_number=top_dt.page_number,
                ocr_confidence=top_dt.ocr_confidence,
                reasoning=top_dt.reasoning,
            )

        # 13. Address Discovery
        addr_match = re.search(
            r"(?:Address|ವಿಳಾಸ|ಸ್ಥಳ)[\s:.\-_|I]*([A-Za-z0-9\u0C80-\u0CFF\s,\.\-\/]+?)(?=\s*(?:\n|Date|ದಿನಾಂಕ|Ward|Taluk|$))",
            cleaned_text,
            re.IGNORECASE | re.UNICODE,
        )
        if addr_match:
            addr_val = addr_match.group(1).strip(" \t\n.-:|I")
            if len(addr_val) >= 4:
                bbox, page, raw_line = cls._find_spatial_region(addr_val, regions)
                fields["address"] = cls._create_field(
                    field_name="address",
                    extracted_value=addr_val,
                    evidence_text=raw_line,
                    confidence=0.90,
                    bbox=bbox,
                    page_number=page,
                    reasoning="Property address extracted from explicit address section",
                )

        if "address" not in fields:
            # Dynamically synthesize from discovered cadastral components
            addr_parts = []
            if "ward" in fields:
                addr_parts.append(str(fields["ward"].extracted_value))
            if "locality" in fields:
                addr_parts.append(str(fields["locality"].extracted_value))
            elif "village" in fields:
                addr_parts.append(str(fields["village"].extracted_value))
            if "taluk" in fields:
                t_val = str(fields["taluk"].extracted_value)
                if not any(t_val.lower() in p.lower() for p in addr_parts):
                    addr_parts.append(t_val)
            if "district" in fields:
                d_val = str(fields["district"].extracted_value)
                if not any(d_val.lower() in p.lower() for p in addr_parts):
                    addr_parts.append(d_val)

            if addr_parts:
                synth_addr = ", ".join(addr_parts)
                anchor = fields.get("locality") or fields.get("ward") or fields.get("district")
                fields["address"] = cls._create_field(
                    field_name="address",
                    extracted_value=synth_addr,
                    evidence_text=anchor.evidence_text if anchor else synth_addr,
                    confidence=0.88,
                    bbox=anchor.bbox if anchor else None,
                    page_number=anchor.page_number if anchor else 1,
                    reasoning="Property address synthesized from validated administrative hierarchy",
                )

        elapsed = round((time.perf_counter() - start_t) * 1000.0, 2)
        return SemanticExtractionResult(
            document_type=doc_type,
            fields=fields,
            irrelevant_noise_detected=noise_detected,
            model_used="multilingual_semantic_engine",
            engine_used="local_fallback",
            execution_time_ms=elapsed,
        )


# ============================================================================
# Main Public Interface
# ============================================================================

def semantic_understand(
    ordered_ocr_text: str,
    regions: Optional[Sequence[Any]] = None,
    document_context: Optional[Dict[str, Any]] = None,
    image_bytes: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
) -> SemanticExtractionResult:
    """Main public interface for semantic document understanding and entity extraction.

    Args:
        ordered_ocr_text: Full document OCR text sorted in natural reading order.
        regions: Optional collection of OCR regions / text lines with bounding boxes.
        document_context: Optional document-level metadata (e.g. state, document_id).
        image_bytes: Optional raw document image bytes for multimodal understanding.
        image_mime_type: Optional MIME type of the document image.

    Returns:
        SemanticExtractionResult: Strongly-typed structured semantic extraction data.
    """
    if not ordered_ocr_text or not ordered_ocr_text.strip():
        return SemanticExtractionResult(
            document_type="Unknown",
            fields={},
            irrelevant_noise_detected=[],
            model_used="none",
            engine_used="none",
            execution_time_ms=0.0,
        )

    # Check if external LLM API is configured in environment
    llm_api_key = (
        os.environ.get("SEMANTIC_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GROQ_API_KEY")
    )
    enable_external_llm = os.environ.get("ENABLE_EXTERNAL_SEMANTIC_LLM", "false").lower() in ("true", "1", "yes")

    if llm_api_key and enable_external_llm:
        try:
            llm_json = _query_external_llm(
                ordered_ocr_text=ordered_ocr_text,
                api_key=llm_api_key,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
            )
            if llm_json and "fields" in llm_json:
                fields_map: Dict[str, SemanticFieldExtraction] = {}
                is_gemini = (
                    llm_api_key.startswith("AIza")
                    or bool(os.environ.get("GEMINI_API_KEY"))
                    or "gemini" in os.environ.get("SEMANTIC_LLM_MODEL", "").lower()
                )
                engine_tag = "gemini" if is_gemini else "external_llm"
                for fname, fpayload in llm_json["fields"].items():
                    if isinstance(fpayload, dict) and fpayload.get("extracted_value"):
                        val = str(fpayload["extracted_value"]).strip()
                        if val.lower() != "null" and val:
                            bbox, page, raw_line = MultilingualSemanticEngine._find_spatial_region(val, regions)
                            fields_map[fname] = SemanticFieldExtraction(
                                field_name=fname,
                                extracted_value=val,
                                raw_unit=fpayload.get("raw_unit"),
                                evidence_text=fpayload.get("evidence_text") or raw_line,
                                confidence=float(fpayload.get("confidence", 0.90)),
                                bbox=bbox,
                                page_number=page,
                                reasoning=fpayload.get("reasoning"),
                                candidate_field=fname,
                                evidence_span=str(fpayload.get("evidence_text") or raw_line)[:120],
                                surrounding_context=str(fpayload.get("evidence_text") or raw_line)[:250],
                                ocr_confidence=1.0,
                                semantic_confidence=float(fpayload.get("confidence", 0.90)),
                                engine=engine_tag,
                            )

                model_tag = os.environ.get("SEMANTIC_LLM_MODEL") or ("gemini-2.5-flash" if is_gemini else "gpt-4o-mini")
                actual_model = llm_json.get("_model_used", model_tag)
                provider_tag = f"gemini:{actual_model}" if is_gemini else f"llm:{actual_model}"

                return SemanticExtractionResult(
                    document_type=llm_json.get("document_type"),
                    fields=fields_map,
                    irrelevant_noise_detected=llm_json.get("irrelevant_noise_detected", []),
                    model_used=provider_tag,
                    engine_used=engine_tag,
                    execution_time_ms=0.0,
                )
        except Exception as exc:
            logger.warning(f"External LLM invocation failed, falling back to local multilingual engine: {exc}")

    # Local Pretrained Multilingual Engine execution
    local_result = MultilingualSemanticEngine.understand_document(
        ordered_ocr_text=ordered_ocr_text,
        regions=regions,
        document_context=document_context,
    )
    if enable_external_llm:
        local_result.engine_used = "local_fallback"
        local_result.model_used = "multilingual_semantic_engine (Local Fallback - Gemini Quota / Unavailable)"
        for f in local_result.fields.values():
            f.engine = "local_fallback"
    else:
        local_result.engine_used = "local_engine"
        for f in local_result.fields.values():
            f.engine = "local_engine"
    return local_result


LOCALITY_EQUIVALENCE = {
    "ಜಯನಗರ": "Jayanagar",
    "jayanagar": "Jayanagar",
    "ಕೋರಮಂಗಲ": "Koramangala",
    "koramangala": "Koramangala",
    "ನಾರಾಯಣಘಟ್ಟ": "Narayanaghatta",
    "narayanaghatta": "Narayanaghatta",
    "ಕೆಂಗೇರಿ": "Kengeri",
    "kengeri": "Kengeri",
    "ಇಜಿಪುರ": "Ejipura",
    "ejipura": "Ejipura",
    "ಮಲ್ಲೇಶ್ವರಂ": "Malleshwaram",
    "malleshwaram": "Malleshwaram",
    "ಇಂದಿರಾನಗರ": "Indiranagar",
    "indiranagar": "Indiranagar",
    "ಬಸವನಗುಡಿ": "Basavanagudi",
    "basavanagudi": "Basavanagudi",
    "ಯಲಹಂಕ": "Yelahanka",
    "yelahanka": "Yelahanka",
    "ರಾಜಾಜಿನಗರ": "Rajajinagar",
    "rajajinagar": "Rajajinagar",
    "ಮಹಾದೇವಪುರ": "Mahadevapura",
    "mahadevapura": "Mahadevapura",
    "whitefield": "Whitefield",
}


def fuse_semantic_and_regex_fields(
    semantic_result: SemanticExtractionResult,
    regex_fields: Dict[str, ExtractedField],
    ocr_result: DocumentOCRResult,
    state_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, ExtractedField]:
    """Fuses semantic understanding extractions with existing rule/regex extractions.

    Preserves evidence and provenance from both channels while resolving candidates:
    - If both agree: boosts confidence and cross-references semantic reasoning.
    - If semantic discovered a field missed by regex: safely incorporates the field.
    - If regex produced a validated high-precision result: retains regex while noting consensus.
    """
    fused: Dict[str, ExtractedField] = dict(regex_fields)
    semantic_fields = semantic_result.to_extracted_fields(ocr_result=ocr_result)

    for field_name, s_obj in semantic_fields.items():
        if field_name not in fused:
            # Field discovered exclusively by semantic reasoning
            fused[field_name] = s_obj
        else:
            # Field extracted by both: reconcile candidates
            reg_obj = fused[field_name]
            reg_val_clean = reg_obj.raw_value.strip().lower()
            sem_val_clean = s_obj.raw_value.strip().lower()

            canonical_reg = LOCALITY_EQUIVALENCE.get(reg_val_clean, reg_val_clean)
            canonical_sem = LOCALITY_EQUIVALENCE.get(sem_val_clean, sem_val_clean)

            is_equivalent = (reg_val_clean == sem_val_clean) or (
                field_name in ("locality", "village", "taluk", "district")
                and canonical_reg.lower() == canonical_sem.lower()
            )

            # If values are identical or bilingual equivalents:
            if is_equivalent:
                reg_obj.confidence = min(1.0, max(reg_obj.confidence, s_obj.confidence) + 0.05)
                if canonical_sem and canonical_sem.lower() != reg_val_clean:
                    reg_obj.normalized_value = canonical_sem
                if s_obj.evidence and s_obj.evidence.validation_notes:
                    if not reg_obj.evidence:
                        reg_obj.evidence = s_obj.evidence
                    else:
                        reg_obj.evidence.validation_notes.extend(s_obj.evidence.validation_notes)
                if reg_obj.evidence:
                    reg_obj.evidence.validation_notes.append(
                        f"Consensus between regex ('{reg_obj.raw_value}') and semantic engine ('{s_obj.raw_value}')"
                    )
            elif s_obj.confidence >= reg_obj.confidence:
                # Semantic candidate has equal or higher confidence and contextual grounding
                if s_obj.evidence:
                    s_obj.evidence.validation_notes.append(
                        f"Semantic understanding prioritized over regex value '{reg_obj.raw_value}' (conf: {s_obj.confidence} >= {reg_obj.confidence})"
                    )
                fused[field_name] = s_obj
            elif s_obj.confidence > 0.80 and reg_obj.confidence < 0.90:
                # Strong semantic evidence overrides weak regex guess
                if s_obj.evidence:
                    s_obj.evidence.validation_notes.append(
                        f"High-confidence semantic evidence resolved value '{reg_obj.raw_value}' to '{s_obj.raw_value}'"
                    )
                fused[field_name] = s_obj
            else:
                # Keep existing regex value and log semantic candidate in provenance notes
                if reg_obj.evidence:
                    reg_obj.evidence.validation_notes.append(f"Semantic candidate alternative: '{s_obj.raw_value}'")

    return fused
