"""Configurable Language and Script Router for Indic Regional OCR & Handwriting Recognition.

Provides multimodal routing to dispatch image crops to appropriate engines based on:
1. Language/Script (Kannada, Telugu, Tamil, Hindi, English, etc.)
2. Text Type (Handwritten vs. Printed)
3. Explicit fallback/review routing for unsupported handwritten languages without fabrication.
"""

from typing import Any, Dict, List, Optional, Set, Union

from schemas import BoundingBox, OCRResult
from src.handwriting.confidence import build_confidence_audit_trail
from src.handwriting.paddle_recognizer import PaddleKannadaRecognizer
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.handwriting.trocr_recognizer import (
    DEFAULT_ENGLISH_MODEL_PATH,
    DEFAULT_KANNADA_MODEL_PATH,
    TrocrHandwritingRecognizer,
    get_english_handwriting_recognizer,
    get_kannada_handwriting_recognizer,
)

# Standard language code normalization mapping
LANGUAGE_ALIASES: Dict[str, str] = {
    "kn": "kannada",
    "kannada": "kannada",
    "kan": "kannada",
    "te": "telugu",
    "telugu": "telugu",
    "tel": "telugu",
    "ta": "tamil",
    "tamil": "tamil",
    "tam": "tamil",
    "hi": "hindi",
    "hindi": "hindi",
    "devanagari": "hindi",
    "ml": "malayalam",
    "malayalam": "malayalam",
    "en": "english",
    "english": "english",
    "eng": "english",
}

# Known Indic languages supported in the overall system taxonomy
KNOWN_INDIC_LANGUAGES: Set[str] = {"kannada", "telugu", "tamil", "hindi", "malayalam"}


class LanguageScriptRouter:
    """Routes image crops to appropriate language and script OCR recognizers.

    Pre-configures:
    - Handwritten Kannada -> Trained Kannada TrOCR checkpoint
    - Printed Kannada -> PaddleOCR Kannada backend
    - Handwritten English -> Pretrained TrOCR baseline
    - Printed English -> PaddleOCR English baseline
    - Unsupported handwritten languages -> Auditable review fallback without text fabrication
    """

    def __init__(
        self,
        default_language: str = "kannada",
        auto_register_kannada: bool = True,
        kannada_recognizer: Optional[BaseHandwritingRecognizer] = None,
        kannada_handwriting_recognizer: Optional[BaseHandwritingRecognizer] = None,
        auto_register_defaults: bool = False,
    ):
        """Initializes the script router.

        Args:
            default_language: Fallback language code when not explicitly passed.
            auto_register_kannada: Whether to auto-register Kannada recognizers.
            kannada_recognizer: Optional pre-configured printed Kannada recognizer.
            kannada_handwriting_recognizer: Optional pre-configured handwritten Kannada recognizer.
            auto_register_defaults: Whether to register additional default handwriting and printed engines (e.g. English).
        """
        self._registry: Dict[str, BaseHandwritingRecognizer] = {}
        self._handwritten_registry: Dict[str, BaseHandwritingRecognizer] = {}
        self._printed_registry: Dict[str, BaseHandwritingRecognizer] = {}
        self._default_language = self._normalize_language_code(default_language)

        if auto_register_kannada:
            # 1. Printed Kannada
            printed_kn = (
                kannada_recognizer
                if kannada_recognizer is not None
                else PaddleKannadaRecognizer(lang="kannada")
            )
            self.register_recognizer("kannada", printed_kn, is_handwritten=False, set_as_default=True)

            # 2. Handwritten Kannada (Trained TrOCR Checkpoint)
            hw_kn = (
                kannada_handwriting_recognizer
                if kannada_handwriting_recognizer is not None
                else get_kannada_handwriting_recognizer(auto_load=False)
            )
            self.register_recognizer("kannada", hw_kn, is_handwritten=True)

        if auto_register_defaults:
            # 3. Handwritten English (Pretrained TrOCR)
            hw_en = get_english_handwriting_recognizer(auto_load=False)
            self.register_recognizer("english", hw_en, is_handwritten=True)

            # 4. Printed English (PaddleOCR)
            printed_en = PaddleKannadaRecognizer(
                model_name="paddleocr-english",
                model_version="ppocr_v4_en",
                lang="english",
            )
            self.register_recognizer("english", printed_en, is_handwritten=False)

    @staticmethod
    def _normalize_language_code(lang_code: Optional[str]) -> str:
        """Normalizes language code/string to standard lower-case identifier."""
        if not lang_code:
            return "kannada"
        cleaned = str(lang_code).strip().lower()
        return LANGUAGE_ALIASES.get(cleaned, cleaned)

    def register_recognizer(
        self,
        language: str,
        recognizer: BaseHandwritingRecognizer,
        is_handwritten: Optional[bool] = None,
        set_as_default: bool = False,
    ) -> None:
        """Registers a recognizer backend for a specified language and optional text type.

        Args:
            language: Language name or ISO code (e.g. 'kannada', 'kn', 'telugu', 'te').
            recognizer: An instance implementing BaseHandwritingRecognizer.
            is_handwritten: If True, registers for handwritten text. If False, for printed.
                If None, registers in the general registry and printed/handwritten fallbacks.
            set_as_default: If True, sets this recognizer as default fallback.
        """
        if not isinstance(recognizer, BaseHandwritingRecognizer):
            raise TypeError(
                f"Expected BaseHandwritingRecognizer instance, got {type(recognizer).__name__}"
            )

        norm_lang = self._normalize_language_code(language)

        if is_handwritten is True:
            self._handwritten_registry[norm_lang] = recognizer
        elif is_handwritten is False:
            self._printed_registry[norm_lang] = recognizer
            self._registry[norm_lang] = recognizer
        else:
            # General registration
            self._registry[norm_lang] = recognizer
            if isinstance(recognizer, TrocrHandwritingRecognizer):
                self._handwritten_registry[norm_lang] = recognizer
            else:
                self._printed_registry[norm_lang] = recognizer

        if set_as_default:
            self._default_language = norm_lang

    def get_recognizer(
        self,
        language: Optional[str] = None,
        is_handwritten: Optional[bool] = None,
    ) -> BaseHandwritingRecognizer:
        """Retrieves the recognizer for a given language and text type.

        Args:
            language: Target language code, or None to use router's default.
            is_handwritten: Whether looking for handwritten or printed recognizer.

        Returns:
            Registered BaseHandwritingRecognizer instance.

        Raises:
            KeyError: If no recognizer is registered for the requested criteria.
        """
        target_lang = self._normalize_language_code(language) if language else self._default_language

        if is_handwritten is True:
            if target_lang in self._handwritten_registry:
                return self._handwritten_registry[target_lang]
            if target_lang == "english":
                self._handwritten_registry["english"] = get_english_handwriting_recognizer(auto_load=False)
                return self._handwritten_registry["english"]
            if self._default_language in self._handwritten_registry:
                return self._handwritten_registry[self._default_language]

        if is_handwritten is False:
            if target_lang in self._printed_registry:
                return self._printed_registry[target_lang]
            if target_lang in KNOWN_INDIC_LANGUAGES or target_lang == "english":
                self._printed_registry[target_lang] = PaddleKannadaRecognizer(
                    model_name=f"paddleocr-{target_lang}",
                    model_version=f"ppocr_v4_{target_lang}",
                    lang=target_lang,
                )
                return self._printed_registry[target_lang]
            if self._default_language in self._printed_registry:
                return self._printed_registry[self._default_language]

        # General lookup
        if target_lang in self._registry:
            return self._registry[target_lang]
        if target_lang in self._printed_registry:
            return self._printed_registry[target_lang]
        if target_lang in self._handwritten_registry:
            return self._handwritten_registry[target_lang]

        if self._default_language in self._registry:
            return self._registry[self._default_language]

        raise KeyError(
            f"No recognizer registered for language '{target_lang}' (is_handwritten={is_handwritten}). "
            f"Supported languages: {self.list_supported_languages()}"
        )

    def list_supported_languages(self) -> List[str]:
        """Returns a list of all currently registered primary language names."""
        all_langs = set(self._registry.keys()) | set(self._handwritten_registry.keys()) | set(self._printed_registry.keys())
        return sorted(list(all_langs))

    def list_supported_handwritten_languages(self) -> List[str]:
        """Returns a list of languages with trained handwritten recognizers registered."""
        return sorted(list(self._handwritten_registry.keys()))

    def list_supported_printed_languages(self) -> List[str]:
        """Returns a list of languages with printed recognizers registered."""
        return sorted(list(self._printed_registry.keys()))

    def route_and_recognize(
        self,
        image: ImageInput,
        language: Optional[str] = None,
        is_handwritten: Optional[bool] = None,
        bbox: Optional[BoundingBox] = None,
        page_number: Optional[int] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> OCRResult:
        """Dispatches an image crop to the appropriate language and text type recognizer.

        Handles unsupported handwriting languages by generating an explicit review result
        with transparent audit trail rather than fabricating unrecognized characters.

        Args:
            image: Image crop input.
            language: Target language/script code.
            is_handwritten: Whether the crop is identified as handwritten or printed.
            bbox: Optional bounding box.
            page_number: Optional page number.
            preprocessing_info: Optional preprocessing parameters.
            **kwargs: Implementation-specific OCR kwargs.

        Returns:
            OCRResult produced by the routed engine or explicit unsupported review flag.
        """
        target_lang = self._normalize_language_code(language) if language else self._default_language

        # Check for unsupported handwritten Indic language (e.g. Telugu, Tamil, Hindi, Malayalam)
        if is_handwritten is True and target_lang not in self._handwritten_registry:
            if target_lang == "english":
                self._handwritten_registry["english"] = get_english_handwriting_recognizer(auto_load=False)
            else:
                audit = build_confidence_audit_trail(
                    raw_token_probabilities=None,
                    calculation_method="unsupported_handwriting_language",
                    custom_metadata={
                        "engine_status": "unsupported_language_model",
                        "language": target_lang,
                        "is_handwritten": True,
                        "reason": f"Handwritten model for '{target_lang}' is not yet trained/available. Routed to manual human review.",
                        "requires_human_review": True,
                        "preprocessing": preprocessing_info or {},
                    },
                )
                return OCRResult(
                    text="",
                    confidence=None,
                    bbox=bbox,
                    is_handwritten=True,
                    page_number=page_number,
                    model_name="none",
                    model_version="unsupported",
                    metadata=audit,
                )

        recognizer = self.get_recognizer(language=target_lang, is_handwritten=is_handwritten)
        return recognizer.recognize_handwriting(
            image=image,
            bbox=bbox,
            page_number=page_number,
            preprocessing_info=preprocessing_info,
            is_handwritten=is_handwritten,
            **kwargs,
        )
