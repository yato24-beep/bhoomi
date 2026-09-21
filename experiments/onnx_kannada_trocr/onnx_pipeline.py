"""End-to-End TrOCR Kannada ONNX Runtime Pipeline.

Provides complete autoregressive decoding using ONNX Runtime sessions:
- Vision Encoder: pixel_values -> last_hidden_state
- Text Decoder: (input_ids, encoder_hidden_states) -> logits
- Autoregressive generation: Greedy & Beam Search (num_beams=4)
- Exact Unicode reconstruction and <unk> monitoring
"""

import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = Path(__file__).resolve().parent / "models"
PROCESSOR_DIR = PROJECT_ROOT / "src" / "handwriting" / "configs" / "iitb_kannada_v002"
TOKENIZER_DIR = PROJECT_ROOT / "models" / "trocr" / "experimental" / "iitb_kannada_v002"


class TrOCRONNXPipeline:
    """End-to-end OCR recognizer using ONNX Runtime models."""

    def __init__(
        self,
        encoder_path: Optional[Path] = None,
        decoder_path: Optional[Path] = None,
        processor_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
        intra_op_num_threads: int = 4,
    ):
        self.encoder_path = encoder_path or (MODELS_DIR / "encoder.onnx")
        self.decoder_path = decoder_path or (MODELS_DIR / "decoder.onnx")
        self.processor_path = processor_path or PROCESSOR_DIR
        self.tokenizer_path = tokenizer_path or TOKENIZER_DIR

        # ONNX Runtime Session Options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = intra_op_num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        print(f"[ONNX Pipeline] Loading Encoder Session: {self.encoder_path.name}", flush=True)
        self.encoder_session = ort.InferenceSession(
            str(self.encoder_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        print(f"[ONNX Pipeline] Loading Decoder Session: {self.decoder_path.name}", flush=True)
        self.decoder_session = ort.InferenceSession(
            str(self.decoder_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.processor = AutoImageProcessor.from_pretrained(str(self.processor_path), local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path), local_files_only=True)

        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 0
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Runs ViT ONNX encoder to produce last_hidden_state."""
        pixel_values = self.processor(image.convert("RGB"), return_tensors="np").pixel_values
        ort_inputs = {"pixel_values": pixel_values.astype(np.float32)}
        last_hidden_state = self.encoder_session.run(None, ort_inputs)[0]
        return last_hidden_state

    def generate_greedy(
        self,
        image: Image.Image,
        max_length: int = 64,
    ) -> Tuple[str, List[int], float]:
        """Greedy autoregressive decoding using ONNX models."""
        t0 = time.perf_counter()
        encoder_hidden_states = self.encode_image(image)

        input_ids = np.array([[self.bos_token_id]], dtype=np.int64)

        for _ in range(max_length):
            ort_inputs = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_hidden_states,
            }
            logits = self.decoder_session.run(None, ort_inputs)[0]
            # Next token logits at last position: shape (1, 100000)
            next_token_logits = logits[0, -1, :]
            next_token_id = int(np.argmax(next_token_logits))

            if next_token_id == self.eos_token_id:
                break

            input_ids = np.append(input_ids, [[next_token_id]], axis=1)

        elapsed = time.perf_counter() - t0
        token_list = input_ids[0].tolist()
        raw_text = self.tokenizer.decode(token_list, skip_special_tokens=False)
        cleaned_text = (
            raw_text.replace(self.tokenizer.bos_token or "<s>", "")
            .replace(self.tokenizer.eos_token or "</s>", "")
            .replace(self.tokenizer.pad_token or "<pad>", "")
            .strip()
        )
        return cleaned_text, token_list, elapsed

    def generate_beam_search(
        self,
        image: Image.Image,
        num_beams: int = 4,
        max_length: int = 64,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
    ) -> Tuple[str, List[int], float]:
        """Autoregressive Beam Search decoding matching PyTorch configuration."""
        t0 = time.perf_counter()
        encoder_hidden_states = self.encode_image(image)

        # Each beam: (cumulative_score, token_sequence)
        beams = [(0.0, [self.bos_token_id])]
        completed_beams = []

        for step in range(max_length):
            all_candidates = []
            for score, tokens in beams:
                if tokens[-1] == self.eos_token_id and step > 0:
                    completed_beams.append((score, tokens))
                    continue

                input_ids = np.array([tokens], dtype=np.int64)
                ort_inputs = {
                    "input_ids": input_ids,
                    "encoder_hidden_states": encoder_hidden_states,
                }
                logits = self.decoder_session.run(None, ort_inputs)[0]
                next_logits = logits[0, -1, :].copy()

                # N-gram repetition penalty
                if no_repeat_ngram_size > 0 and len(tokens) >= no_repeat_ngram_size:
                    ngram = tuple(tokens[-(no_repeat_ngram_size - 1):])
                    for i in range(len(tokens) - no_repeat_ngram_size + 1):
                        if tuple(tokens[i : i + no_repeat_ngram_size - 1]) == ngram:
                            forbidden_tok = tokens[i + no_repeat_ngram_size - 1]
                            next_logits[forbidden_tok] = -1e9

                # Softmax log probabilities
                exp_logits = np.exp(next_logits - np.max(next_logits))
                log_probs = np.log(exp_logits / np.sum(exp_logits) + 1e-12)

                # Top-K candidates for this beam
                top_indices = np.argpartition(log_probs, -num_beams)[-num_beams:]
                top_indices = top_indices[np.argsort(-log_probs[top_indices])]

                for tok_id in top_indices:
                    tok_id = int(tok_id)
                    all_candidates.append((score + log_probs[tok_id], tokens + [tok_id]))

            if not all_candidates:
                break

            # Length penalty normalization
            def beam_score(cand):
                s, tok = cand
                # Standard HF length penalty
                lp = ((5.0 + len(tok)) / 6.0) ** length_penalty
                return s / lp

            # Select top num_beams
            all_candidates.sort(key=beam_score, reverse=True)
            beams = all_candidates[:num_beams]

            # Early stopping check
            if len(completed_beams) >= num_beams:
                break

        # Pick best beam
        all_final = completed_beams + beams
        def final_score(cand):
            s, tok = cand
            lp = ((5.0 + len(tok)) / 6.0) ** length_penalty
            return s / lp

        all_final.sort(key=final_score, reverse=True)
        best_score, best_tokens = all_final[0]

        elapsed = time.perf_counter() - t0
        raw_text = self.tokenizer.decode(best_tokens, skip_special_tokens=False)
        cleaned_text = (
            raw_text.replace(self.tokenizer.bos_token or "<s>", "")
            .replace(self.tokenizer.eos_token or "</s>", "")
            .replace(self.tokenizer.pad_token or "<pad>", "")
            .strip()
        )
        return cleaned_text, best_tokens, elapsed
