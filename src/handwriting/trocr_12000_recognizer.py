"""Fine-Tuned Indic-TrOCR Checkpoint-12000 Kannada Handwriting Recognizer.

========================================================================================
MODEL PROVENANCE & EMPIRICAL BENCHMARK PERFORMANCE (DO NOT OMIT OR SOFTEN)
========================================================================================
Base Architecture:
- IIT Bombay Indic-TrOCR v0.0.2 (ViT-base 224px vision encoder + 6-layer RoBERTa decoder)
- Tokenizer: Chakita/KannadaBERT (100,000 vocab, genuine Kannada Unicode coverage, 0 <unk>)
- Model Weights: Checkpoint-12000 fine-tuned on IIIT-INDIC-HW-WORDS Kannada dataset

VERIFIED, HONEST BENCHMARK RESULTS:
1. In-Distribution Performance (Pilot set: 400 IIIT isolated Kannada dictionary words):
   - Character Error Rate (CER): ~4.86%
   - Word Error Rate (WER): ~16.50%
   - Exact Match Accuracy: 83.50%

2. Out-of-Distribution Performance (Locked benchmark: 13 real archival land-record crops):
   - Character Error Rate (CER): ~93.06% - 93.73%
   - Word Error Rate (WER): ~96.08% - 96.92%
   - Exact Match Accuracy: 15.38% (2 / 13 crops — isolated word crops match; multi-word lines fail)

KNOWN FAILURE MODE (STRONG LANGUAGE-MODEL PRIOR):
- The model was trained exclusively on isolated Kannada dictionary words (IIIT-INDIC-HW-WORDS).
- It was NEVER trained on real multi-word archival cursive lines.
- When presented with multi-word cursive archival lines, the decoder outputs plausible-sounding
  hallucinated dictionary words (e.g. "ಸ್ಪರ್ಧಿಸಿಕೊಂಡಿದ್ದು", "ಘಟ್ನಿಸಿಕೊಳ್ಳುವುದಕ್ಕೂ", "ತಪ್ಪಿಸಿಕೊಳ್ಳುವುದಕ್ಕೂ")
  driven by RoBERTa's language model prior, rather than reading the actual physical ink strokes.

PRACTICAL PRODUCTION IMPLICATION:
- RELIABLE for: Short, isolated field values (e.g. a single isolated name crop or isolated numeral).
- NOT RELIABLE for: Full unsegmented handwritten lines, cursive sentences, or archival paragraphs.

FUTURE ROADMAP (NOT YET DONE):
- Retraining on genuine line-level cursive land records using ICDAR 2025 IHDR Task B
  (page/line recognition) or segmented Bhoomi/Satbara cursive crops.
========================================================================================
"""

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import unicodedata

from PIL import Image

from schemas import BoundingBox, OCRResult
from src.handwriting.confidence import (
    build_confidence_audit_trail,
    compute_token_geometric_mean_confidence,
    compute_token_mean_confidence,
)
from src.handwriting.recognizer import BaseHandwritingRecognizer, ImageInput
from src.preprocessing.image_enhancement import load_image_as_pil

logger = logging.getLogger("trocr_12000_recognizer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Checkpoint paths
DEFAULT_CHECKPOINT_12000_PATH = str(PROJECT_ROOT / "models" / "trocr" / "checkpoint-12000")
DEFAULT_BASE_PROCESSOR_PATH = str(PROJECT_ROOT / "models" / "trocr" / "experimental" / "iitb_kannada_v002")
TRACKED_BASE_PROCESSOR_PATH = str(PROJECT_ROOT / "src" / "handwriting" / "configs" / "iitb_kannada_v002")
DEFAULT_TOKENIZER_NAME = "Chakita/KannadaBERT"

# Global process-level cache for loaded (processor, tokenizer, model)
_CHECKPOINT_12000_CACHE: Optional["TrOCR12000KannadaRecognizer"] = None


def _has_weights(path_str: str) -> bool:
    """Check if directory exists and contains PyTorch or Safetensors model weights."""
    p = Path(path_str)
    if not p.is_dir():
        return False
    return (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()


def resolve_or_download_trocr_checkpoint(model_path: Optional[str] = None) -> str:
    """Resolves local checkpoint-12000 path or downloads from Hugging Face if absent.
    
    Priority:
    1. Explicit model_path if valid weights exist
    2. TROCR_CHECKPOINT_12000_PATH / KANNADA_HANDWRITING_MODEL_PATH
    3. Container standard path: /data/models/trocr/checkpoint-12000
    4. Local project path: models/trocr/checkpoint-12000
    5. If absent and MODEL_REPO_ID is set: download via huggingface_hub into /data/models/trocr/checkpoint-12000
    """
    candidates = []
    if model_path:
        candidates.append(Path(model_path))
    if os.environ.get("TROCR_CHECKPOINT_12000_PATH"):
        candidates.append(Path(os.environ["TROCR_CHECKPOINT_12000_PATH"]))
    if os.environ.get("KANNADA_HANDWRITING_MODEL_PATH"):
        candidates.append(Path(os.environ["KANNADA_HANDWRITING_MODEL_PATH"]))
    candidates.append(Path("/data/models/trocr/checkpoint-12000"))
    candidates.append(PROJECT_ROOT / "models" / "trocr" / "checkpoint-12000")

    for cand in candidates:
        if _has_weights(str(cand)):
            logger.info(f"Using local TrOCR checkpoint-12000 at: {cand}")
            return str(cand.resolve())

    # If absent locally, check MODEL_REPO_ID (only download if explicitly authorized to protect 512MB RAM containers)
    model_repo_id = os.environ.get("MODEL_REPO_ID", "").strip()
    enable_server_dl = os.environ.get("ENABLE_SERVER_TROCR_DOWNLOAD", "false").lower() in ("true", "1", "yes")
    if model_repo_id and enable_server_dl:
        target_dir = Path("/data/models/trocr/checkpoint-12000")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            target_dir = PROJECT_ROOT / "models" / "trocr" / "checkpoint-12000"
            target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[HUGGINGFACE] Checkpoint-12000 absent locally. "
            f"Downloading from Hugging Face repository '{model_repo_id}' into {target_dir}..."
        )
        try:
            from huggingface_hub import snapshot_download
            token = os.environ.get("HF_TOKEN") or None
            snapshot_download(
                repo_id=model_repo_id,
                local_dir=str(target_dir),
                token=token,
            )
            logger.info(f"[HUGGINGFACE] Successfully downloaded TrOCR checkpoint into {target_dir}")
            return str(target_dir.resolve())
        except Exception as dl_err:
            logger.error(f"[HUGGINGFACE] Failed downloading from '{model_repo_id}': {dl_err}")
            return str(target_dir.resolve())
    elif model_repo_id and not enable_server_dl:
        logger.info(
            "[HUGGINGFACE] Server-side model download disabled (ENABLE_SERVER_TROCR_DOWNLOAD=false). "
            "Client-side browser execution active."
        )

    # Fallback to preferred path
    default_cand = candidates[0] if candidates else Path("/data/models/trocr/checkpoint-12000")
    return str(default_cand)


def resolve_base_processor_path(proc_path: Optional[str] = None) -> str:
    """Resolves IIT Bombay v0.0.2 base processor path with fallback to tracked config."""
    candidates = []
    if proc_path:
        candidates.append(Path(proc_path))
    if os.environ.get("TROCR_BASE_PROCESSOR_PATH"):
        candidates.append(Path(os.environ["TROCR_BASE_PROCESSOR_PATH"]))
    candidates.append(Path(DEFAULT_BASE_PROCESSOR_PATH))
    candidates.append(Path(TRACKED_BASE_PROCESSOR_PATH))

    for cand in candidates:
        if (cand / "preprocessor_config.json").exists():
            return str(cand.resolve())

    return TRACKED_BASE_PROCESSOR_PATH


class TrOCR12000KannadaRecognizer(BaseHandwritingRecognizer):
    """Production handwriting recognizer using fine-tuned TrOCR checkpoint-12000.
    
    Loads:
    - Weights from checkpoint-12000 (configurable via MODEL_REPO_ID)
    - Image processor from IITB v0.0.2 base checkpoint
    - Tokenizer from Chakita/KannadaBERT
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        base_processor_path: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = None,
        auto_load: bool = False,
        confidence_threshold: float = 0.30,
        num_beams: int = 4,
        max_length: int = 64,
        length_penalty: float = 2.0,
        **kwargs: Any,
    ):
        """Initializes the Checkpoint-12000 recognizer."""
        resolved_model_path = resolve_or_download_trocr_checkpoint(model_path)
        resolved_proc_path = resolve_base_processor_path(base_processor_path)
        resolved_tok_name = tokenizer_name or os.environ.get(
            "TROCR_TOKENIZER_NAME",
            DEFAULT_TOKENIZER_NAME,
        )

        super().__init__(
            model_name="indic-trocr-checkpoint-12000",
            model_version="checkpoint-12000",
        )
        self.confidence_threshold = confidence_threshold

        self.model_path = resolved_model_path
        self.base_processor_path = resolved_proc_path
        self.tokenizer_name = resolved_tok_name

        self.num_beams = num_beams
        self.max_length = max_length
        self.length_penalty = length_penalty

        # Device determination
        import torch
        if device is not None:
            self._device_str = device
        else:
            self._device_str = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = None
        self._processor = None
        self._tokenizer = None
        self._is_loaded = False
        self._load_error: Optional[str] = None

        if auto_load:
            self.load_model()

    @property
    def is_available(self) -> bool:
        """Returns whether the model is loaded and ready for inference."""
        return self._is_loaded and self._model is not None

    def load_model(self) -> bool:
        """Loads model weights, image processor, and Chakita/KannadaBERT tokenizer."""
        if self._is_loaded and self._model is not None:
            return True

        try:
            import torch
            from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

            logger.info(f"Loading TrOCR Checkpoint-12000 from: {self.model_path}")
            logger.info(f"Loading base image processor from: {self.base_processor_path}")
            logger.info(f"Loading tokenizer from: {self.tokenizer_name}")

            self._processor = AutoImageProcessor.from_pretrained(self.base_processor_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_path)

            self._model.to(self._device_str)
            self._model.eval()

            self._is_loaded = True
            self._load_error = None
            logger.info(f"Checkpoint-12000 loaded successfully on {self._device_str}.")
            return True
        except Exception as exc:
            self._is_loaded = False
            self._load_error = f"Failed to load checkpoint-12000: {exc}"
            logger.error(self._load_error, exc_info=True)
            return False

    def _extract_token_probabilities(
        self,
        scores: Sequence[Any],
        sequences: Any,
    ) -> List[float]:
        """Extracts genuine softmax probabilities for generated tokens."""
        import torch
        if not scores or sequences is None:
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
        except Exception as err:
            logger.debug(f"Could not compute token probabilities: {err}")
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
        """Runs handwriting inference on an image crop using Checkpoint-12000.
        
        Applies decode logic proven in training:
        - Strips only pad_token_id, bos_token_id, and eos_token_id
        - Does NOT use skip_special_tokens=True in a way that hides <unk>
        - Surfaces <unk> in output as a visible low-confidence marker
        """
        import torch

        start_time = time.perf_counter()

        # Step 1: Ensure image is PIL Image
        pil_image = load_image_as_pil(image).convert("RGB")

        # Step 2: Ensure model is loaded
        if not self._is_loaded:
            success = self.load_model()
            if not success:
                elapsed = time.perf_counter() - start_time
                audit = build_confidence_audit_trail(
                    raw_token_probabilities=None,
                    calculation_method="unavailable",
                    custom_metadata={
                        "engine_status": "unavailable",
                        "reason": self._load_error or "Checkpoint-12000 model not loaded",
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "requires_human_review": True,
                    },
                )
                return OCRResult(
                    text="",
                    confidence=None,
                    bbox=bbox,
                    is_handwritten=True,
                    page_number=page_number,
                    model_name=self.model_name,
                    model_version=self.model_version,
                    metadata=audit,
                )

        # Step 3: Preprocess image using ViTImageProcessor
        pixel_values = self._processor(pil_image, return_tensors="pt").pixel_values.to(self._device_str)

        # Step 4: Generation settings (Beam Search matching training configuration)
        num_beams = kwargs.get("num_beams", self.num_beams)
        max_length = kwargs.get("max_length", self.max_length)
        length_penalty = kwargs.get("length_penalty", self.length_penalty)

        gen_kwargs = {
            "max_length": max_length,
            "num_beams": num_beams,
            "length_penalty": length_penalty,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
            "decoder_start_token_id": (
                getattr(self._model.config, "decoder_start_token_id", None)
                or self._tokenizer.bos_token_id
            ),
            "return_dict_in_generate": True,
            "output_scores": True,
        }

        with torch.no_grad():
            outputs = self._model.generate(pixel_values, **gen_kwargs)

        generated_ids = outputs.sequences if hasattr(outputs, "sequences") else outputs

        # Step 5: Decode logic proven in training:
        # Strip ONLY pad, bos, and eos token ids; do NOT use skip_special_tokens=True to hide <unk>.
        pad_id = self._tokenizer.pad_token_id
        bos_id = self._tokenizer.bos_token_id
        eos_id = self._tokenizer.eos_token_id

        raw_ids = generated_ids[0].tolist()
        filtered_ids = [t for t in raw_ids if t not in (pad_id, bos_id, eos_id)]

        # Decode with skip_special_tokens=False to surface <unk> visibly
        raw_text = self._tokenizer.decode(filtered_ids, skip_special_tokens=False).strip()
        recognized_text = unicodedata.normalize("NFC", raw_text)

        # Step 6: Confidence calculation
        token_scores = outputs.scores if hasattr(outputs, "scores") else None
        token_probs = self._extract_token_probabilities(token_scores, generated_ids) if token_scores else []

        mean_conf = compute_token_mean_confidence(token_probs) if token_probs else None
        geo_conf = compute_token_geometric_mean_confidence(token_probs) if token_probs else None

        # If <unk> is present, penalize confidence and flag for review
        has_unk = "<unk>" in recognized_text
        if has_unk and mean_conf is not None:
            mean_conf = round(mean_conf * 0.5, 4)

        elapsed = time.perf_counter() - start_time
        requires_review = (
            has_unk
            or mean_conf is None
            or mean_conf < self.confidence_threshold
            or len(recognized_text) == 0
        )

        audit = build_confidence_audit_trail(
            raw_token_probabilities=token_probs,
            calculation_method="trocr_checkpoint_12000_logits",
            custom_metadata={
                "engine_status": "active",
                "device": self._device_str,
                "inference_time_ms": round(elapsed * 1000.0, 2),
                "num_beams": num_beams,
                "length_penalty": length_penalty,
                "has_unk": has_unk,
                "requires_human_review": requires_review,
                "geometric_mean_confidence": geo_conf,
                "benchmark_note": "Trained on isolated Kannada words (~4.86% CER in-domain; ~93% CER out-of-domain archival lines)",
                "preprocessing": preprocessing_info or {},
            },
        )

        return OCRResult(
            text=recognized_text,
            confidence=mean_conf,
            bbox=bbox,
            is_handwritten=True,
            page_number=page_number,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata=audit,
        )

    def recognize_batch(
        self,
        images: Sequence[ImageInput],
        bboxes: Optional[Sequence[Optional[BoundingBox]]] = None,
        **kwargs: Any,
    ) -> List[OCRResult]:
        """Recognizes a batch of image crops."""
        results: List[OCRResult] = []
        for idx, img in enumerate(images):
            box = bboxes[idx] if bboxes and idx < len(bboxes) else None
            res = self.recognize_handwriting(img, bbox=box, **kwargs)
            results.append(res)
        return results


def get_checkpoint_12000_recognizer(
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    auto_load: bool = False,
    **kwargs: Any,
) -> TrOCR12000KannadaRecognizer:
    """Factory helper returning the process-level singleton Checkpoint-12000 recognizer."""
    global _CHECKPOINT_12000_CACHE
    if _CHECKPOINT_12000_CACHE is None:
        _CHECKPOINT_12000_CACHE = TrOCR12000KannadaRecognizer(
            model_path=model_path,
            device=device,
            auto_load=auto_load,
            **kwargs,
        )
    elif auto_load and not _CHECKPOINT_12000_CACHE.is_available:
        _CHECKPOINT_12000_CACHE.load_model()

    return _CHECKPOINT_12000_CACHE


# Canonical aliases
get_trocr_12000_recognizer = get_checkpoint_12000_recognizer
