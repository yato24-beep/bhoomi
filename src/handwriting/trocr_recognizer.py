"""Pretrained & Fine-Tuned TrOCR (Vision-Encoder-Decoder) handwriting recognition backend.

Implements BaseHandwritingRecognizer using Hugging Face Transformers and PyTorch,
providing genuine handwriting recognition with transparent logit-derived confidence scoring,
lazy loading, global model caching, FP16 CUDA acceleration, and configurable checkpoint loading.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image

from schemas import BoundingBox, OCRResult
from src.handwriting.confidence import (
    build_confidence_audit_trail,
    compute_token_geometric_mean_confidence,
    compute_token_mean_confidence,
)
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.preprocessing.image_enhancement import (
    load_image_as_pil,
    preprocess_document_image,
)

# Project root path resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default model paths configurable via environment variables
DEFAULT_KANNADA_MODEL_PATH = os.environ.get(
    "KANNADA_HANDWRITING_MODEL_PATH",
    "models/trocr/kannada_generalized_checkpoints/best_checkpoint",
)
DEFAULT_ENGLISH_MODEL_PATH = os.environ.get(
    "TROCR_PRETRAINED_MODEL_PATH",
    "microsoft/trocr-small-handwritten",
)

# Global in-memory cache for loaded (processor, model) pairs to avoid reloading 600+ MB weights
_GLOBAL_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}

# Soft import PyTorch and Transformers to support environments where packages are optional
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        RobertaTokenizer,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
        XLMRobertaTokenizer,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    AutoImageProcessor = None
    AutoTokenizer = None
    RobertaTokenizer = None
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None
    XLMRobertaTokenizer = None
    HAS_TRANSFORMERS = False


def resolve_model_path(model_path_or_name: str) -> str:
    """Resolves a model name or relative path against PROJECT_ROOT if exists locally."""
    if not model_path_or_name:
        return model_path_or_name
    p = Path(model_path_or_name)
    if p.is_absolute() and p.exists():
        return str(p)
    rel_p = PROJECT_ROOT / p
    if rel_p.exists():
        return str(rel_p)
    return str(model_path_or_name)


class TrocrHandwritingRecognizer(BaseHandwritingRecognizer):
    """TrOCR handwriting recognizer powered by Hugging Face VisionEncoderDecoder.

    Features:
    - Lazy loading: Weights are loaded only on demand or when load_model() is called.
    - Model caching: Cached globally by checkpoint and device to prevent redundant 600MB reloads.
    - Non-fabricated confidence: Derived directly from generation output step logits/softmax.
    - Hardware agnostic: Uses CUDA/GPU when available with FP16 support and seamless CPU fallback.
    - Dependency injection: Accepts pre-instantiated processor/model for fast mock unit tests.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        model_version: Optional[str] = None,
        processor: Optional[Any] = None,
        model: Optional[Any] = None,
        device: Optional[str] = None,
        auto_load: bool = False,
        preprocess_input: bool = True,
        max_new_tokens: int = 64,
        language: str = "kannada",
        script: Optional[str] = None,
        fp16: Optional[bool] = None,
        confidence_threshold: float = 0.60,
    ):
        """Initializes the TrOCR recognizer configuration without immediate heavy loading.

        Args:
            model_name_or_path: Hugging Face hub repository ID or local checkpoint path.
                Defaults to KANNADA_HANDWRITING_MODEL_PATH env var / default Kannada checkpoint.
            model_version: Descriptive version tag. If None, inspected from checkpoint metadata.
            processor: Optional pre-loaded TrOCRProcessor instance (for injection/testing).
            model: Optional pre-loaded VisionEncoderDecoderModel instance (for injection/testing).
            device: Target device ("cuda", "cpu", or None for auto-detection).
            auto_load: If True, loads weights immediately during initialization. Default: False.
            preprocess_input: Whether to apply non-destructive document enhancement before inference.
            max_new_tokens: Maximum number of tokens generated per crop line.
            language: Intended script/language code (e.g. 'kannada', 'english').
            script: Script name (e.g. 'Kannada', 'Latin'). If None, inferred from language.
            fp16: Whether to use FP16 precision on CUDA. Defaults to True if CUDA is active.
            confidence_threshold: Threshold below which recognition is flagged for human review.
        """
        if model_name_or_path is not None:
            raw_path = model_name_or_path
        elif language in ("kannada", "kn"):
            raw_path = os.getenv("KANNADA_HANDWRITING_MODEL_PATH", DEFAULT_KANNADA_MODEL_PATH)
        else:
            raw_path = DEFAULT_ENGLISH_MODEL_PATH

        resolved_path = resolve_model_path(raw_path)

        # Infer version and language metadata if local checkpoint has training_metadata.json
        inferred_version = model_version
        inferred_language = language
        metadata_file = Path(resolved_path) / "training_metadata.json"
        if metadata_file.exists() and not model_version:
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    inferred_version = meta.get("model_version") or meta.get("checkpoint_id")
                    if "languages" in meta and meta["languages"]:
                        inferred_language = meta["languages"][0]
            except Exception:
                pass

        final_version = inferred_version or ("trocr_kannada_v1" if "kannada" in resolved_path else "trocr_small_v1")
        super().__init__(model_name=resolved_path, model_version=final_version)

        self.model_name_or_path = resolved_path
        self.preprocess_input = preprocess_input
        self.max_new_tokens = max_new_tokens
        self.language = inferred_language
        self.script = script or ("Kannada" if self.language in ("kannada", "kn") else "Latin")
        self.confidence_threshold = confidence_threshold

        # Device selection
        if device is not None:
            self._device_str = device
        elif HAS_TORCH and torch is not None and torch.cuda.is_available():
            self._device_str = "cuda"
        else:
            self._device_str = "cpu"

        # FP16 selection
        if fp16 is not None:
            self._use_fp16 = bool(fp16 and self._device_str == "cuda")
        else:
            self._use_fp16 = bool(self._device_str == "cuda")

        self._processor: Optional[Any] = processor
        self._model: Optional[Any] = model
        self._load_error: Optional[str] = None

        if processor is not None and model is not None:
            self._is_loaded = True
            if HAS_TORCH and torch is not None and isinstance(model, torch.nn.Module):
                try:
                    self._model = model.to(self._device_str)
                except Exception:
                    pass
        else:
            self._is_loaded = False
            if auto_load:
                self.load_model()

    @property
    def is_available(self) -> bool:
        """Returns True if the required libraries are installed and no fatal init error occurred."""
        return HAS_TORCH and HAS_TRANSFORMERS and (self._load_error is None)

    @property
    def is_loaded(self) -> bool:
        """Returns True if model and processor are currently loaded in memory."""
        return self._is_loaded and self._model is not None and self._processor is not None

    @property
    def device(self) -> str:
        """Returns the active execution device ('cuda' or 'cpu')."""
        return self._device_str

    @property
    def is_fp16(self) -> bool:
        """Returns True if running in FP16 mixed precision on CUDA."""
        return self._use_fp16

    def load_model(self) -> bool:
        """Explicitly loads the TrOCR processor and model weights onto the target device.

        Uses global model caching so that 600MB weights are not reloaded on every instance creation.

        Returns:
            bool: True if loading succeeded, False otherwise.
        """
        if self._is_loaded and self._model is not None and self._processor is not None:
            return True

        if not HAS_TORCH or torch is None:
            self._load_error = "PyTorch is not installed in the environment."
            return False

        if not HAS_TRANSFORMERS or VisionEncoderDecoderModel is None:
            self._load_error = "transformers library is not installed in the environment."
            return False

        cache_key = f"{self.model_name_or_path}::{self._device_str}::{self._use_fp16}"

        # Check global memory cache
        if cache_key in _GLOBAL_MODEL_CACHE:
            cached_proc, cached_mod = _GLOBAL_MODEL_CACHE[cache_key]
            self._processor = cached_proc
            self._model = cached_mod
            self._is_loaded = True
            self._load_error = None
            return True

        try:
            is_local = Path(self.model_name_or_path).is_dir()
            kwargs = {"local_files_only": True} if is_local else {}

            # 1. Load processor / tokenizer
            if self._processor is None:
                try:
                    tok = XLMRobertaTokenizer.from_pretrained(self.model_name_or_path, **kwargs)
                    img_proc = AutoImageProcessor.from_pretrained(self.model_name_or_path, **kwargs)
                    self._processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tok)
                except Exception:
                    try:
                        self._processor = TrOCRProcessor.from_pretrained(self.model_name_or_path, **kwargs)
                    except Exception:
                        img_proc = AutoImageProcessor.from_pretrained(self.model_name_or_path, **kwargs)
                        try:
                            tok = RobertaTokenizer.from_pretrained(self.model_name_or_path, **kwargs)
                        except Exception:
                            tok = AutoTokenizer.from_pretrained(self.model_name_or_path, **kwargs)
                        self._processor = TrOCRProcessor(image_processor=img_proc, tokenizer=tok)

            # 2. Load model
            if self._model is None:
                self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name_or_path, **kwargs)
                if HAS_TORCH and torch is not None and isinstance(self._model, torch.nn.Module):
                    self._model.to(self._device_str)
                self._model.eval()

            # Store in global cache for subsequent requests
            _GLOBAL_MODEL_CACHE[cache_key] = (self._processor, self._model)

            self._is_loaded = True
            self._load_error = None
            return True
        except Exception as exc:
            self._is_loaded = False
            self._load_error = f"Failed to load TrOCR model '{self.model_name_or_path}': {str(exc)}"
            return False

    def _extract_token_probabilities(
        self,
        scores: Sequence[Any],
        sequences: Any,
    ) -> List[float]:
        """Extracts genuine per-token generation probabilities from output scores.

        Uses softmax over the step logits to determine the true posterior probability
        assigned to each generated token ID. Zero fabrication.

        Args:
            scores: Sequence of step tensors of shape (batch_size, vocab_size).
            sequences: Generated token IDs tensor of shape (batch_size, seq_len).

        Returns:
            List of float token probabilities in [0.0, 1.0].
        """
        if not scores or not HAS_TORCH or torch is None:
            return []

        token_probs: List[float] = []
        try:
            seq_tokens = sequences[0] if sequences.ndim > 1 else sequences
            start_offset = len(seq_tokens) - len(scores)

            for step_idx, step_scores in enumerate(scores):
                probs = torch.softmax(step_scores[0], dim=-1)
                token_idx = start_offset + step_idx
                if 0 <= token_idx < len(seq_tokens):
                    target_token_id = int(seq_tokens[token_idx].item())
                    prob_val = float(probs[target_token_id].item())
                    token_probs.append(round(prob_val, 6))
        except Exception:
            return []

        return token_probs

    def recognize_handwriting(
        self,
        image: ImageInput,
        bbox: Optional[BoundingBox] = None,
        page_number: Optional[int] = None,
        preprocessing_info: Optional[Dict[str, Any]] = None,
        is_handwritten: Optional[bool] = True,
        **kwargs: Any,
    ) -> OCRResult:
        """Performs handwriting recognition on a single crop image using TrOCR.

        Args:
            image: Image crop input (file path, PIL Image, or NumPy array).
            bbox: Optional caller-provided bounding box.
            page_number: Optional 1-indexed document page number.
            preprocessing_info: Optional upstream preprocessing metadata.
            is_handwritten: Boolean flag (defaults to True for TrOCR).
            **kwargs: Extra arguments passed to model.generate().

        Returns:
            OCRResult with recognized text, logit-derived confidence, timing, and full audit trail.
        """
        start_time = time.perf_counter()

        # Step 1: Preprocess image if configured
        prep_metadata: Dict[str, Any] = {}
        if preprocessing_info:
            prep_metadata.update(preprocessing_info)

        pil_image: Image.Image
        if self.preprocess_input:
            prep_res = preprocess_document_image(
                image_input=image,
                apply_thresholding=False,
            )
            pil_image = prep_res.image
            prep_metadata["preprocessing_pipeline"] = prep_res.audit_metadata
        else:
            pil_image = load_image_as_pil(image)

        # Convert to RGB mode for VisionEncoderDecoder
        rgb_image = pil_image.convert("RGB")

        # Step 2: Ensure model is loaded lazily
        if not self._is_loaded:
            success = self.load_model()
            if not success:
                elapsed = time.perf_counter() - start_time
                audit = build_confidence_audit_trail(
                    raw_token_probabilities=None,
                    calculation_method="unavailable",
                    custom_metadata={
                        "engine_status": "unavailable",
                        "reason": self._load_error or "TrOCR engine not loaded",
                        "model_name": self.model_name,
                        "checkpoint": self.model_name_or_path,
                        "device": self._device_str,
                        "language": self.language,
                        "script": self.script,
                        "inference_time_sec": round(elapsed, 4),
                        "inference_time_ms": round(elapsed * 1000.0, 2),
                        "raw_text": "",
                        "normalized_text": "",
                        "requires_human_review": True,
                        "preprocessing": prep_metadata,
                    },
                )
                return OCRResult(
                    text="",
                    confidence=None,
                    bbox=bbox,
                    is_handwritten=is_handwritten,
                    page_number=page_number,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    metadata=audit,
                )

        # Step 3: Run genuine model inference
        try:
            pixel_values = self._processor(images=rgb_image, return_tensors="pt").pixel_values
            if HAS_TORCH and torch is not None:
                pixel_values = pixel_values.to(self._device_str)

            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if hasattr(self._model, "config"):
                if getattr(self._model.config, "decoder_start_token_id", None) is not None:
                    gen_kwargs["decoder_start_token_id"] = self._model.config.decoder_start_token_id
                if getattr(self._model.config, "pad_token_id", None) is not None:
                    gen_kwargs["pad_token_id"] = self._model.config.pad_token_id
                if getattr(self._model.config, "eos_token_id", None) is not None:
                    gen_kwargs["eos_token_id"] = self._model.config.eos_token_id

            ignored_kwargs = {"preprocess_input", "enable_multipass", "override_preprocess", "preprocess"}
            for k, v in kwargs.items():
                if k not in gen_kwargs and k not in ignored_kwargs:
                    gen_kwargs[k] = v

            # Use autocast for FP16 on CUDA if enabled
            if HAS_TORCH and torch is not None:
                autocast_ctx = (
                    torch.amp.autocast("cuda", enabled=True)
                    if (self._device_str == "cuda" and self._use_fp16 and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"))
                    else _DummyContext()
                )
                with torch.no_grad(), autocast_ctx:
                    outputs = self._model.generate(pixel_values, **gen_kwargs)
            else:
                outputs = self._model.generate(pixel_values, **gen_kwargs)

            # Decode predicted token IDs into text
            generated_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs
            if hasattr(self._processor, "batch_decode"):
                raw_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elif hasattr(self._processor, "tokenizer") and hasattr(self._processor.tokenizer, "batch_decode"):
                raw_text = self._processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            else:
                raw_text = ""
            recognized_text = raw_text.strip()

            # Step 4: Extract step-by-step token generation probabilities
            token_scores = outputs.scores if hasattr(outputs, "scores") else None
            token_probs = self._extract_token_probabilities(token_scores, generated_ids) if token_scores else []

            # Step 5: Compute defensible confidence metrics
            mean_conf = compute_token_mean_confidence(token_probs) if token_probs else None
            geo_conf = compute_token_geometric_mean_confidence(token_probs) if token_probs else None

            # Phase 7: Multi-Pass Recognition for challenging / low-confidence handwriting crops
            best_variant_used = "variant_a_enhanced"
            enable_multipass = kwargs.get("enable_multipass", False)
            if enable_multipass and (mean_conf is None or mean_conf < 0.65):
                from PIL import ImageEnhance, ImageFilter
                variants = [
                    ("variant_b_upscaled", rgb_image.resize((int(rgb_image.width * 1.5), int(rgb_image.height * 1.5)), Image.BICUBIC)),
                    ("variant_c_contrast", ImageEnhance.Contrast(rgb_image).enhance(1.30)),
                    ("variant_d_denoised", rgb_image.filter(ImageFilter.MedianFilter(size=3))),
                ]
                for var_name, var_img in variants:
                    try:
                        var_pixels = self._processor(images=var_img, return_tensors="pt").pixel_values
                        if HAS_TORCH and torch is not None:
                            var_pixels = var_pixels.to(self._device_str)
                        with torch.no_grad(), autocast_ctx:
                            var_outputs = self._model.generate(var_pixels, **gen_kwargs)
                        var_ids = var_outputs.sequences if hasattr(var_outputs, "sequences") else var_outputs
                        if hasattr(self._processor, "batch_decode"):
                            var_raw = self._processor.batch_decode(var_ids, skip_special_tokens=True)[0]
                        else:
                            var_raw = ""
                        var_text = var_raw.strip()
                        var_scores = var_outputs.scores if hasattr(var_outputs, "scores") else None
                        var_probs = self._extract_token_probabilities(var_scores, var_ids) if var_scores else []
                        var_conf = compute_token_mean_confidence(var_probs) if var_probs else None

                        if var_conf is not None and (mean_conf is None or var_conf > mean_conf):
                            recognized_text = var_text
                            raw_text = var_raw
                            mean_conf = var_conf
                            geo_conf = compute_token_geometric_mean_confidence(var_probs) if var_probs else geo_conf
                            token_probs = var_probs
                            best_variant_used = var_name
                    except Exception:
                        pass

            elapsed = time.perf_counter() - start_time
            requires_review = (
                mean_conf is None
                or mean_conf < self.confidence_threshold
                or len(recognized_text) == 0
            )

            custom_meta = {
                "engine_status": "active",
                "device": self._device_str,
                "is_fp16": self._use_fp16,
                "geometric_mean_confidence": geo_conf,
                "token_count": len(token_probs),
                "model_type": "vision_encoder_decoder",
                "language": self.language,
                "script": self.script,
                "checkpoint": self.model_name_or_path,
                "inference_time_sec": round(elapsed, 4),
                "inference_time_ms": round(elapsed * 1000.0, 2),
                "raw_text": raw_text,
                "normalized_text": recognized_text,
                "requires_human_review": requires_review,
                "best_variant_used": best_variant_used,
                "preprocessing": prep_metadata,
            }
            if self.language != "kannada" or "microsoft/trocr" in str(self.model_name_or_path):
                custom_meta["language_limitation_notice"] = (
                    "Model is base English TrOCR; Indic handwriting recognition requires fine-tuned checkpoint."
                )

            audit_trail = build_confidence_audit_trail(
                raw_token_probabilities=token_probs if token_probs else None,
                calculation_method="trocr_step_softmax_mean" if token_probs else "unavailable",
                custom_metadata=custom_meta,
            )

            return OCRResult(
                text=recognized_text,
                confidence=mean_conf,
                bbox=bbox,
                is_handwritten=is_handwritten,
                page_number=page_number,
                model_name=self.model_name,
                model_version=self.model_version,
                metadata=audit_trail,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            audit = build_confidence_audit_trail(
                raw_token_probabilities=None,
                calculation_method="inference_error",
                custom_metadata={
                    "engine_status": "error",
                    "error": str(exc),
                    "model_name": self.model_name,
                    "checkpoint": self.model_name_or_path,
                    "device": self._device_str,
                    "language": self.language,
                    "script": self.script,
                    "inference_time_sec": round(elapsed, 4),
                    "inference_time_ms": round(elapsed * 1000.0, 2),
                    "raw_text": "",
                    "normalized_text": "",
                    "requires_human_review": True,
                    "preprocessing": prep_metadata,
                },
            )
            return OCRResult(
                text="",
                confidence=None,
                bbox=bbox,
                is_handwritten=is_handwritten,
                page_number=page_number,
                model_name=self.model_name,
                model_version=self.model_version,
                metadata=audit,
            )

    def recognize_batch(
        self,
        images: Sequence[ImageInput],
        bboxes: Optional[Sequence[Optional[BoundingBox]]] = None,
        page_numbers: Optional[Sequence[Optional[int]]] = None,
        preprocessing_info_list: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        **kwargs: Any,
    ) -> List[OCRResult]:
        """Performs batched handwriting recognition on a sequence of crops."""
        results: List[OCRResult] = []
        for i, img_item in enumerate(images):
            bbox = bboxes[i] if bboxes and i < len(bboxes) else None
            page_num = page_numbers[i] if page_numbers and i < len(page_numbers) else None
            prep_info = (
                preprocessing_info_list[i]
                if preprocessing_info_list and i < len(preprocessing_info_list)
                else None
            )
            res = self.recognize_handwriting(
                image=img_item,
                bbox=bbox,
                page_number=page_num,
                preprocessing_info=prep_info,
                **kwargs,
            )
            results.append(res)
        return results


def get_kannada_handwriting_recognizer(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    auto_load: bool = False,
    **kwargs: Any,
) -> TrocrHandwritingRecognizer:
    """Factory helper to obtain a configured Kannada TrOCR recognizer."""
    path = model_path or DEFAULT_KANNADA_MODEL_PATH
    return TrocrHandwritingRecognizer(
        model_name_or_path=path,
        language="kannada",
        script="Kannada",
        device=device,
        auto_load=auto_load,
        **kwargs,
    )


def get_english_handwriting_recognizer(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    auto_load: bool = False,
    **kwargs: Any,
) -> TrocrHandwritingRecognizer:
    """Factory helper to obtain a configured English TrOCR recognizer."""
    path = model_path or DEFAULT_ENGLISH_MODEL_PATH
    return TrocrHandwritingRecognizer(
        model_name_or_path=path,
        language="english",
        script="Latin",
        device=device,
        auto_load=auto_load,
        **kwargs,
    )


class _DummyContext:
    """Fallback dummy context manager for environments without PyTorch."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
