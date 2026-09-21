"""Model-Agnostic Semantic Reasoning Engine Interface and Implementations.

Provides abstractions for AI-driven semantic understanding of Karnataka land records:
- BaseSemanticEngine: Abstract base contract
- GeminiSemanticEngine: Google Gemini reasoning engine via official google-genai SDK
- LocalSemanticEngine: Model-agnostic stub for local Qwen-family multilingual models
- RuleSemanticEngine: Deterministic spatial/anchor rule-based fallback
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try importing official google-genai SDK
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError
    HAS_GOOGLE_GENAI = True
except ImportError:
    genai = None
    types = None
    APIError = Exception
    HAS_GOOGLE_GENAI = False


@dataclass
class SemanticEvidence:
    """Compact, structured multimodal evidence representation passed to semantic engines."""
    document_id: Optional[str] = None
    page_number: int = 1
    document_type: str = "Unknown / Not classified"
    raw_text: str = ""
    regions: List[Any] = field(default_factory=list)
    tables: List[Any] = field(default_factory=list)
    ner_candidates: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_compact_prompt_payload(self) -> str:
        """Formats a token-efficient semantic representation of OCR layout evidence."""
        lines = []
        lines.append(f"DOCUMENT ID: {self.document_id or 'unknown'}")
        lines.append(f"DOCUMENT TYPE: {self.document_type}")
        lines.append(f"PAGE NUMBER: {self.page_number}")
        lines.append("")

        # 1. Reconstructed Tables (Highest structural fidelity)
        if self.tables:
            lines.append("=== DETECTED TABLES ===")
            for t in self.tables:
                t_id = getattr(t, "table_id", "table")
                headers = getattr(t, "headers", [])
                if headers:
                    lines.append(f"Table [{t_id}] Headers: {' | '.join(headers)}")
                cells = getattr(t, "cells", [])
                if cells:
                    lines.append(f"Table [{t_id}] Cells:")
                    for c in cells:
                        r_idx = getattr(c, "row_index", 0)
                        c_idx = getattr(c, "col_index", 0)
                        txt = getattr(c, "text", "")
                        src_id = getattr(c, "source_region_id", "")
                        lines.append(f"  Row {r_idx}, Col {c_idx} [ID:{src_id}]: {txt}")
                elif getattr(t, "rows", []):
                    for r_idx, row in enumerate(getattr(t, "rows", [])):
                        lines.append(f"  Row {r_idx}: {' | '.join(row)}")
            lines.append("")

        # 2. OCR Regions in Reading Order with Geometry
        lines.append("=== OCR REGIONS (READING ORDER) ===")
        for r in self.regions:
            reg_id = getattr(r, "region_id", "unknown")
            text = getattr(r, "raw_text", getattr(r, "text", ""))
            bbox = getattr(r, "bbox", None)
            conf = getattr(r, "confidence", None)
            box_str = ""
            if bbox:
                x1, y1 = round(float(getattr(bbox, "x_min", 0.0)), 1), round(float(getattr(bbox, "y_min", 0.0)), 1)
                x2, y2 = round(float(getattr(bbox, "x_max", 0.0)), 1), round(float(getattr(bbox, "y_max", 0.0)), 1)
                box_str = f" bbox=({x1},{y1},{x2},{y2})"
            conf_str = f" conf={conf:.2f}" if conf is not None else ""
            lines.append(f"[ID: {reg_id}{box_str}{conf_str}]: {text}")
        lines.append("")

        # 3. Upstream NER / Gazetteer hints if available
        if self.ner_candidates:
            lines.append("=== UPSTREAM NER CANDIDATES ===")
            for fname, val in self.ner_candidates.items():
                if isinstance(val, dict):
                    v_str = val.get("normalized_value") or val.get("raw_value") or ""
                else:
                    v_str = str(val)
                if v_str:
                    lines.append(f"{fname}: {v_str}")
            lines.append("")

        return "\n".join(lines)


class SemanticExtractedField(BaseModel):
    """Raw semantic field extraction output from an AI engine."""
    field_name: str
    raw_value: Optional[str] = None
    normalized_value: Optional[str] = None
    source_region_id: Optional[str] = None
    model_confidence: Optional[float] = None
    conflict_candidates: List[str] = Field(default_factory=list)


class SemanticEngineOutputSchema(BaseModel):
    """Schema-constrained generation contract for LLM structured output."""
    document_type: str = "Unknown / Not classified"
    district: Optional[SemanticExtractedField] = None
    taluk: Optional[SemanticExtractedField] = None
    hobli: Optional[SemanticExtractedField] = None
    village: Optional[SemanticExtractedField] = None
    survey_number: Optional[SemanticExtractedField] = None
    hissa_number: Optional[SemanticExtractedField] = None
    khata_number: Optional[SemanticExtractedField] = None
    owner_name: Optional[SemanticExtractedField] = None
    cultivator_name: Optional[SemanticExtractedField] = None
    extent: Optional[SemanticExtractedField] = None
    land_type: Optional[SemanticExtractedField] = None
    mutation_number: Optional[SemanticExtractedField] = None
    registration_number: Optional[SemanticExtractedField] = None
    record_date: Optional[SemanticExtractedField] = None
    extra_fields: List[SemanticExtractedField] = Field(default_factory=list)


@dataclass
class SemanticEngineResult:
    """Unified result container returned by any SemanticEngine implementation."""
    fields: Dict[str, SemanticExtractedField] = field(default_factory=dict)
    document_type: str = "Unknown / Not classified"
    status: str = "success"  # "success", "partial", "error"
    error: Optional[str] = None
    raw_model_response: Optional[str] = None


class BaseSemanticEngine(ABC):
    """Abstract base interface for land-record semantic reasoning engines."""

    @abstractmethod
    def extract(self, evidence: SemanticEvidence) -> SemanticEngineResult:
        """Infers structured semantic fields from layout, reading order, and OCR evidence."""
        pass


class GeminiSemanticEngine(BaseSemanticEngine):
    """Production semantic reasoning engine powered by Google Gemini API using google-genai SDK."""

    SYSTEM_PROMPT = """You are an expert Karnataka Land Record Semantic AI Reasoning Engine.
Your task is to infer structured cadastral and revenue fields from OCR layout evidence.

CRITICAL INSTRUCTIONS:
1. UNLABELED OCR CONTENT:
   Land record OCR lines may have NO explicit labels like "Name:" or "Survey:".
   For example, an OCR line might read:
   "ರಾಮಪ್ಪ 493/2 2-15 ದೊಡ್ಡಬಳ್ಳಾಪುರ"
   You must infer:
   - "ರಾಮಪ್ಪ" -> owner_name
   - "493/2" -> survey_number
   - "2-15" -> extent (2 acres 15 guntas)
   - "ದೊಡ್ಡಬಳ್ಳಾಪುರ" -> taluk or village
   Use layout position, reading order, table cells, and neighboring context to infer meanings.

2. NEVER INVENT OR HALLUCINATE VALUES:
   - If a field cannot be determined from the evidence, set it to null.
   - Do NOT fabricate names, survey numbers, dates, or extents.
   - Every populated field MUST specify its `source_region_id` from the provided OCR regions.
   - If a value cannot be tied to a source region ID, return null.

3. KANNADA TEXT PRESERVATION:
   - In `raw_value`, preserve the verbatim Kannada/English text exactly as recognized.
   - In `normalized_value`, provide clean standard format (e.g. converting Kannada numerals ೧೨೫ to 125).

4. CONFLICTS:
   - If multiple contradictory candidates exist for the same field (e.g. two conflicting survey numbers or multiple distinct owners in a single-owner slot), populate `conflict_candidates`.

5. FIELDS TO EXTRACT:
   - document_type (e.g. "Bhoomi RTC", "Pahani", "Mutation Register", "Sale Deed", "Unknown")
   - district (ಜಿಲ್ಲೆ)
   - taluk (ತಾಲೂಕು)
   - hobli (ಹೋಬಳಿ)
   - village (ಗ್ರಾಮ)
   - survey_number (ಸರ್ವೆ ನಂ)
   - hissa_number (ಹಿಸ್ಸಾ ನಂ)
   - khata_number (ಖಾತಾ ಸಂಖ್ಯೆ)
   - owner_name (ಖಾತೆದಾರರ ಹೆಸರು / ಮಾಲೀಕರ ಹೆಸರು)
   - cultivator_name (ಅನುಭೋಗದಾರರ ಹೆಸರು / ಗೇಣಿದಾರರ ಹೆಸರು)
   - extent (ವಿಸ್ತೀರ್ಣ)
   - land_type (ಜಮೀನಿನ ವಿವರ: ಖುಷ್ಕಿ/ತರಿ/ಬಾಗಾಯ್ತು/Dry/Wet)
   - mutation_number (ಮ್ಯುಟೇಶನ್ ಸಂಖ್ಯೆ / MR No)
   - registration_number (ನೋಂದಣಿ ಸಂಖ್ಯೆ / Document No)
   - record_date (ದಿನಾಂಕ)
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: int = 30,
        temperature: float = 0.0,
    ):
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
                except Exception:
                    pass
            if not api_key:
                try:
                    from backend.app.config import settings
                    api_key = (getattr(settings, "GEMINI_API_KEY", "") or "").strip()
                except Exception:
                    pass
        self.api_key = api_key
        self.model_name = (
            model_name
            or os.environ.get("SEMANTIC_MODEL_NAME", "gemini-3.1-flash-lite").strip()
        )

        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self._client = None

        if HAS_GOOGLE_GENAI and self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini Client: {e}")
                self._client = None

    @property
    def is_available(self) -> bool:
        """Returns True if the Google GenAI SDK and API key are configured."""
        return HAS_GOOGLE_GENAI and bool(self.api_key)

    def extract(self, evidence: SemanticEvidence) -> SemanticEngineResult:
        """Executes AI semantic reasoning on OCR evidence using structured JSON output."""
        if not self.is_available:
            err_msg = "Gemini API client not configured or GEMINI_API_KEY missing"
            logger.info(f"GeminiSemanticEngine unavailable: {err_msg}")
            return SemanticEngineResult(
                status="error",
                error=err_msg,
                document_type=evidence.document_type,
            )

        prompt_payload = evidence.to_compact_prompt_payload()

        try:
            config = types.GenerateContentConfig(
                system_instruction=self.SYSTEM_PROMPT,
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=SemanticEngineOutputSchema,
            )

            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt_payload,
                config=config,
            )

            raw_text = response.text or "{}"
            return self._parse_response(raw_text, evidence.document_type)

        except Exception as exc:
            logger.error(f"Gemini API inference failure: {exc}", exc_info=True)
            return SemanticEngineResult(
                status="error",
                error=f"Gemini API execution error: {exc}",
                document_type=evidence.document_type,
            )

    def _parse_response(self, raw_json_str: str, default_doc_type: str) -> SemanticEngineResult:
        """Parses model response JSON safely into SemanticEngineResult."""
        try:
            # Strip markdown formatting fences if present
            cleaned = raw_json_str.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            doc_type = parsed.get("document_type") or default_doc_type

            fields: Dict[str, SemanticExtractedField] = {}
            for k in (
                "district", "taluk", "hobli", "village", "survey_number",
                "hissa_number", "khata_number", "owner_name", "cultivator_name",
                "extent", "land_type", "mutation_number", "registration_number",
                "record_date",
            ):
                f_data = parsed.get(k)
                if f_data and isinstance(f_data, dict):
                    raw_v = f_data.get("raw_value")
                    if raw_v and str(raw_v).strip():
                        fields[k] = SemanticExtractedField(
                            field_name=k,
                            raw_value=str(raw_v).strip(),
                            normalized_value=str(f_data.get("normalized_value") or raw_v).strip(),
                            source_region_id=f_data.get("source_region_id"),
                            model_confidence=f_data.get("model_confidence"),
                            conflict_candidates=f_data.get("conflict_candidates") or [],
                        )

            # Include any extra fields (handles both list and dict schemas)
            extras = parsed.get("extra_fields") or []
            if isinstance(extras, list):
                for f_data in extras:
                    if isinstance(f_data, dict):
                        f_name = f_data.get("field_name")
                        raw_v = f_data.get("raw_value")
                        if f_name and raw_v and str(raw_v).strip():
                            fields[f_name] = SemanticExtractedField(
                                field_name=f_name,
                                raw_value=str(raw_v).strip(),
                                normalized_value=str(f_data.get("normalized_value") or raw_v).strip(),
                                source_region_id=f_data.get("source_region_id"),
                                model_confidence=f_data.get("model_confidence"),
                                conflict_candidates=f_data.get("conflict_candidates") or [],
                            )
            elif isinstance(extras, dict):
                for k, f_data in extras.items():
                    if f_data and isinstance(f_data, dict):
                        raw_v = f_data.get("raw_value")
                        if raw_v and str(raw_v).strip():
                            fields[k] = SemanticExtractedField(
                                field_name=k,
                                raw_value=str(raw_v).strip(),
                                normalized_value=str(f_data.get("normalized_value") or raw_v).strip(),
                                source_region_id=f_data.get("source_region_id"),
                                model_confidence=f_data.get("model_confidence"),
                                conflict_candidates=f_data.get("conflict_candidates") or [],
                            )


            return SemanticEngineResult(
                fields=fields,
                document_type=doc_type,
                status="success",
                raw_model_response=raw_json_str,
            )

        except Exception as e:
            logger.warning(f"Failed to parse Gemini structured output: {e}. Raw response: {raw_json_str[:200]}")
            return SemanticEngineResult(
                status="error",
                error=f"Malformed model response JSON: {e}",
                document_type=default_doc_type,
                raw_model_response=raw_json_str,
            )


class LocalSemanticEngine(BaseSemanticEngine):
    """Model-agnostic placeholder for future fine-tuned local models (e.g. Qwen2.5-VL / Multilingual LLM)."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.environ.get("LOCAL_SEMANTIC_MODEL_PATH", "")

    def extract(self, evidence: SemanticEvidence) -> SemanticEngineResult:
        logger.info("LocalSemanticEngine invoked (offline stub).")
        return SemanticEngineResult(
            status="error",
            error="Local fine-tuned semantic model is not currently loaded",
            document_type=evidence.document_type,
        )


class RuleSemanticEngine(BaseSemanticEngine):
    """Deterministic spatial and label-anchor extraction engine (offline fallback)."""

    def __init__(self):
        from src.semantic.field_detector import FieldDetector
        from src.semantic.value_associator import ValueAssociator
        self.detector = FieldDetector()
        self.associator = ValueAssociator()

    def extract(self, evidence: SemanticEvidence) -> SemanticEngineResult:
        anchors = self.detector.detect_anchors(evidence.regions)
        associated = self.associator.associate_fields(
            anchors=anchors,
            regions=evidence.regions,
            document_id=evidence.document_id,
            page_number=evidence.page_number,
        )
        fields: Dict[str, SemanticExtractedField] = {}
        for fname, item in associated.items():
            fields[fname] = SemanticExtractedField(
                field_name=fname,
                raw_value=item.raw_value,
                normalized_value=item.value,
                source_region_id=item.source_region_ids[0] if item.source_region_ids else None,
                model_confidence=1.0,
            )
        # Also incorporate any upstream NER / spatial candidate extractions not yet covered
        if evidence.ner_candidates:
            for fname, cand in evidence.ner_candidates.items():
                if fname not in fields:
                    raw_val = None
                    norm_val = None
                    if isinstance(cand, dict):
                        raw_val = cand.get("raw_value") or cand.get("value_text") or cand.get("text")
                        norm_val = cand.get("normalized_value") or raw_val
                    elif hasattr(cand, "raw_value"):
                        raw_val = cand.raw_value
                        norm_val = getattr(cand, "normalized_value", raw_val)
                    elif isinstance(cand, str):
                        raw_val = cand
                        norm_val = cand
                    if raw_val and str(raw_val).strip():
                        fields[fname] = SemanticExtractedField(
                            field_name=fname,
                            raw_value=str(raw_val).strip(),
                            normalized_value=str(norm_val).strip() if norm_val else str(raw_val).strip(),
                            source_region_id=getattr(evidence.regions[0], "region_id", None) if evidence.regions else None,
                            model_confidence=0.90,
                        )
        return SemanticEngineResult(
            fields=fields,
            document_type=evidence.document_type,
            status="success",
        )
