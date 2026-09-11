"""Generalization Evaluation Workflow for Kannada Handwriting OCR.

Evaluates unseen handwritten Kannada images across:
1. Base/Original Pretrained Kannada TrOCR Checkpoint
2. Personal Fine-Tuned Checkpoint

Runs:
Image -> Orientation Correction -> Denoise/Deskew -> Word/Line Crop -> Kannada TrOCR -> Confidence -> English Translation
Computes Character Error Rate (CER), Levenshtein distance, and Exact Match without retraining or polluting datasets.
"""

import argparse
import io
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import cv2
import torch
import numpy as np
from PIL import Image, ImageOps

# Ensure UTF-8 output encoding for Indic characters
sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import src

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer

from src.preprocessing.image_enhancement import (
    detect_and_correct_coarse_orientation,
    preprocess_document_image,
)
from src.training.evaluate import compute_cer, levenshtein_distance
from src.translation.translator import translate_kannada_text

from src.postprocessing.beam_rescorer import KannadaBeamRescorer, RescoringResult

BASE_CKPT = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
TRIAL_CKPT = r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"


class GeneralizationEvaluator:
    """Evaluates generalization performance on unseen handwriting samples with beam search & candidate rescoring."""

    def __init__(
        self,
        base_checkpoint: str = BASE_CKPT,
        finetuned_checkpoint: str = TRIAL_CKPT,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_checkpoint = base_checkpoint
        self.finetuned_checkpoint = finetuned_checkpoint
        self.rescorer = KannadaBeamRescorer()

        print(f"Loading Base TrOCR Model from: {self.base_checkpoint}...")
        self.tokenizer_base = XLMRobertaTokenizer.from_pretrained(self.base_checkpoint)
        self.processor_base = TrOCRProcessor.from_pretrained(self.base_checkpoint)
        self.model_base = VisionEncoderDecoderModel.from_pretrained(self.base_checkpoint).to(self.device)
        self.model_base.eval()

        print(f"Loading Fine-Tuned Model from: {self.finetuned_checkpoint}...")
        self.tokenizer_ft = XLMRobertaTokenizer.from_pretrained(self.finetuned_checkpoint)
        self.processor_ft = TrOCRProcessor.from_pretrained(self.finetuned_checkpoint)
        self.model_ft = VisionEncoderDecoderModel.from_pretrained(self.finetuned_checkpoint).to(self.device)
        self.model_ft.eval()
        print(f"[✓] Both models loaded onto {self.device} successfully.\n")

    def preprocess_and_crop(self, image_input: Any) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
        """Applies orientation correction, contrast enhancement, and extracts tight handwritten word crop."""
        if isinstance(image_input, (str, Path)):
            raw_img = Image.open(image_input)
        else:
            raw_img = image_input

        raw_img = ImageOps.exif_transpose(raw_img).convert("RGB")
        w, h = raw_img.size

        # 1. Orientation check: if portrait (height > width), rotate 90 degrees CCW to align horizontally
        if h > w:
            rotated = raw_img.rotate(90, expand=True)
            rot_applied = 90
        else:
            rotated = raw_img
            rot_applied = 0

        # 2. Extract ink bounding box on the oriented image
        gray = cv2.cvtColor(np.array(rotated), cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Suppress horizontal ruling / notebook lines (width 45, height 1)
        line_k = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
        lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, line_k)
        ink_no_lines = cv2.subtract(thresh, lines)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ink_no_lines)
        valid_boxes = []
        for i in range(1, num_labels):
            bx, by, bw, bh, barea = stats[i]
            # Keep text strokes, ignoring tiny dust (< 50 area) or oversized page borders
            if 50 < barea < (rotated.width * rotated.height * 0.5) and bw < rotated.width * 0.85 and bh < rotated.height * 0.85:
                valid_boxes.append((bx, by, bw, bh))

        pad = 20
        if valid_boxes:
            min_x = max(0, min(b[0] for b in valid_boxes) - pad)
            min_y = max(0, min(b[1] for b in valid_boxes) - pad)
            max_x = min(rotated.width, max(b[0] + b[2] for b in valid_boxes) + pad)
            max_y = min(rotated.height, max(b[1] + b[3] for b in valid_boxes) + pad)
        else:
            # Fallback: ink intensity percentile box
            ys, xs = np.where(thresh > 0)
            if len(xs) > 10 and len(ys) > 10:
                min_x = max(0, int(np.percentile(xs, 1)) - pad)
                min_y = max(0, int(np.percentile(ys, 1)) - pad)
                max_x = min(rotated.width, int(np.percentile(xs, 99)) + pad)
                max_y = min(rotated.height, int(np.percentile(ys, 99)) + pad)
            else:
                # Central crop rather than full oversized canvas
                cx, cy = rotated.width // 2, rotated.height // 2
                cw, ch = min(rotated.width, 400) // 2, min(rotated.height, 200) // 2
                min_x, min_y, max_x, max_y = cx - cw, cy - ch, cx + cw, cy + ch

        final_crop_box = (int(min_x), int(min_y), int(max_x), int(max_y))

        # 3. Enhanced image
        prep_res = preprocess_document_image(rotated, apply_thresholding=False)
        enhanced_img = prep_res.image

        # 4. Crop the tight word region from the rotated image
        word_crop = rotated.crop(final_crop_box)

        meta = {
            "rotation_degrees": int(rot_applied),
            "original_size": (int(w), int(h)),
            "oriented_size": (int(rotated.width), int(rotated.height)),
            "crop_box": final_crop_box,
            "crop_size": (int(word_crop.width), int(word_crop.height)),
            "aspect_ratio": round(float(word_crop.width) / max(1, float(word_crop.height)), 2),
        }
        return enhanced_img, word_crop, meta

    def _infer_beam(self, model: Any, proc: Any, tok: Any, crop_img: Image.Image, num_beams: int = 5) -> RescoringResult:
        """Runs forward inference using beam search (num_beams=5) with multi-signal candidate rescoring."""
        pixel_values = proc(crop_img.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                max_new_tokens=32,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
            )

        beam_candidates = []
        seq_scores = getattr(outputs, "sequences_scores", None)
        for idx, seq in enumerate(outputs.sequences):
            cand_text = tok.decode(seq, skip_special_tokens=True).strip()
            cand_score = float(seq_scores[idx].item()) if seq_scores is not None else float(-idx)
            beam_candidates.append({
                "text": cand_text,
                "token_ids": seq.tolist(),
                "score": cand_score,
            })

        res = self.rescorer.rescore(beam_candidates)
        return res

    def evaluate_sample(
        self,
        image_path: str,
        ground_truth: Optional[str] = None,
        save_crop_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluates a single unseen handwritten sample across base and fine-tuned models."""
        _, word_crop, meta = self.preprocess_and_crop(image_path)
        if save_crop_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_crop_path)), exist_ok=True)
            word_crop.save(save_crop_path)

        # 1. Base Model Inference (Beam 5 + Rescoring)
        res_base = self._infer_beam(self.model_base, self.processor_base, self.tokenizer_base, word_crop)
        pred_base = res_base.final_prediction
        conf_base = res_base.confidence

        # 2. Fine-Tuned Model Inference (Beam 5 + Rescoring)
        res_ft = self._infer_beam(self.model_ft, self.processor_ft, self.tokenizer_ft, word_crop)
        pred_ft = res_ft.final_prediction
        conf_ft = res_ft.confidence

        # 3. Translation
        trans_base = translate_kannada_text(pred_base) if pred_base else ""
        trans_ft = translate_kannada_text(pred_ft) if pred_ft else ""

        # 4. Metrics
        gt = (ground_truth or "").strip()
        exact_base = bool(gt and pred_base == gt)
        exact_ft = bool(gt and pred_ft == gt)
        cer_base = round(compute_cer(gt, pred_base), 4) if gt else None
        cer_ft = round(compute_cer(gt, pred_ft), 4) if gt else None

        res = {
            "image": os.path.basename(image_path),
            "image_path": str(image_path),
            "ground_truth": gt,
            "base_prediction": pred_base,
            "finetuned_prediction": pred_ft,
            "base_confidence": conf_base,
            "finetuned_confidence": conf_ft,
            "base_exact_match": exact_base,
            "finetuned_exact_match": exact_ft,
            "base_cer": cer_base,
            "finetuned_cer": cer_ft,
            "base_translation": trans_base,
            "finetuned_translation": trans_ft,
            "requires_human_review": res_ft.requires_human_review,
            "rescoring_details": {
                "original_top": res_ft.original_top_candidate,
                "rescored_candidate": res_ft.rescored_candidate,
                "is_switched": res_ft.is_switched,
                "all_beam_candidates": [
                    {
                        "rank": c.rank,
                        "text": c.text,
                        "model_prob": c.model_probability,
                        "lexicon_score": c.lexicon_score,
                        "ortho_score": c.orthographic_score,
                        "composite_score": c.composite_score,
                    }
                    for c in res_ft.all_candidates
                ],
            },
            "metadata": meta,
        }
        return res


def print_comparison_table(results: List[Dict[str, Any]]):
    """Prints a beautiful markdown comparison table."""
    print("=" * 110)
    print("  KANNADA TrOCR GENERALIZATION EVALUATION REPORT (UNSEEN HANDWRITING)")
    print("=" * 110)
    print(
        f"{'Image':<12} | {'Ground Truth':<14} | {'Base Prediction':<16} | {'Fine-Tuned Pred':<16} | {'Base Conf':<10} | {'FT Conf':<10} | {'Exact Match'}"
    )
    print("-" * 110)
    for r in results:
        img_name = r["image"]
        gt = r["ground_truth"] or "N/A"
        b_pred = r["base_prediction"] or "(empty)"
        ft_pred = r["finetuned_prediction"] or "(empty)"
        b_conf = f"{r['base_confidence']*100:.1f}%"
        ft_conf = f"{r['finetuned_confidence']*100:.1f}%"
        match_str = "✓ Match" if r["finetuned_exact_match"] else ("~ Improved" if r.get("finetuned_cer", 1) < r.get("base_cer", 1) else "✗ Differ")
        print(
            f"{img_name:<12} | {gt:<14} | {b_pred:<16} | {ft_pred:<16} | {b_conf:<10} | {ft_conf:<10} | {match_str}"
        )
    print("-" * 110)
    print("\nEnglish Translations of Model Outputs:")
    for r in results:
        print(f"  [{r['image']}] Fine-Tuned Output: '{r['finetuned_prediction']}' -> English: '{r['finetuned_translation']}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Kannada TrOCR Generalization on Unseen Images.")
    parser.add_argument("--image", type=str, help="Path to a single unseen test image")
    parser.add_argument("--label", type=str, default="", help="Expected ground-truth Kannada label")
    parser.add_argument("--manifest", type=str, default=None, help="Optional JSONL manifest of test samples")
    args = parser.parse_args()

    evaluator = GeneralizationEvaluator()

    test_queue = []
    if args.image:
        test_queue.append((args.image, args.label))
    elif args.manifest and os.path.exists(args.manifest):
        with open(args.manifest, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    test_queue.append((item["image"], item.get("text", "")))
    else:
        # Default: evaluate sample image i.jpeg (unseen during training)
        candidate = r"c:\Users\achyu\Downloads\i.jpeg"
        if os.path.exists(candidate):
            test_queue.append((candidate, "ಕೋತಿ"))  # i.jpeg is monkey sample
        else:
            print("Usage: python scripts/evaluate_generalization.py --image path/to/image.jpeg --label <ground_truth>")
            sys.exit(0)

    results = []
    for img_p, gt_lbl in test_queue:
        crop_save = os.path.join(r"c:\Land Record\scratch\generalization_crops", f"eval_{os.path.basename(img_p)}.png")
        res = evaluator.evaluate_sample(img_p, ground_truth=gt_lbl, save_crop_path=crop_save)
        results.append(res)

    print_comparison_table(results)

    # Save artifact
    out_json = r"c:\Land Record\scratch\generalization_eval_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
            default=lambda o: int(o) if isinstance(o, (np.integer, np.int32, np.int64)) else float(o) if isinstance(o, (np.floating, np.float32, np.float64)) else str(o),
        )
    print(f"\n[EVALUATION COMPLETE] Results saved to {out_json}")
