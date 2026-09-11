import os
import sys
sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer

from scripts.evaluate_generalization import GeneralizationEvaluator
from src.postprocessing.multi_variant_ocr import (
    MultiVariantBeamEvaluator,
    generate_crop_variants,
)

BASE_CKPT = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
TRIAL_CKPT = r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"
OUT_DIR = r"c:\Land Record\scratch\kothi_debug"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("MULTI-VARIANT VISUAL INFERENCE & CONSENSUS AGGREGATION TEST")
print("=" * 80)

# 1. Load fine-tuned model
print("Loading fine-tuned model and processor...")
tokenizer_ft = XLMRobertaTokenizer.from_pretrained(TRIAL_CKPT)
processor_ft = TrOCRProcessor.from_pretrained(TRIAL_CKPT)
model_ft = VisionEncoderDecoderModel.from_pretrained(TRIAL_CKPT).cuda().eval()

multi_evaluator = MultiVariantBeamEvaluator()
gen_evaluator = GeneralizationEvaluator(base_checkpoint=BASE_CKPT, finetuned_checkpoint=TRIAL_CKPT)

def evaluate_and_report(sample_name: str, crop_img: Image.Image, ground_truth: str):
    print("\n" + "=" * 80)
    print(f"SAMPLE: {sample_name} | Ground Truth: '{ground_truth}' | Input Crop Size: {crop_img.size}")
    print("=" * 80)

    # Save visual variants
    variants = generate_crop_variants(crop_img)
    print(f"Generated {len(variants)} aspect-ratio-preserving variants:")
    for v_name, v_img in variants.items():
        v_path = os.path.join(OUT_DIR, f"{sample_name}_{v_name}.png")
        v_img.save(v_path)
        print(f"  - {v_name:22s} (Size: {v_img.size[0]}x{v_img.size[1]}) -> Saved to: {v_path}")

    # Run multi-variant visual beam search
    res = multi_evaluator.run_multi_variant_inference(
        model=model_ft,
        processor=processor_ft,
        tokenizer=tokenizer_ft,
        crop_img=crop_img,
        device="cuda",
        num_beams=5,
    )

    # 1. Print every variant's top-5 candidates
    print("\n[A] PER-VARIANT TOP-5 BEAM CANDIDATES:")
    print("-" * 75)
    for v_name, cand_list in res.variant_breakdown.items():
        c_str = ", ".join([f"#{c['rank']} '{c['text']}' ({c['prob']*100:.1f}%)" for c in cand_list])
        print(f"  {v_name:22s} : {c_str}")

    # 2. Print cross-variant consensus ranking
    print("\n[B] CROSS-VARIANT CONSENSUS RANKING:")
    print("-" * 75)
    print(f"  {'Rank':<4} | {'Candidate':<10} | {'Variants':<8} | {'Agreement':<10} | {'MeanProb':<9} | {'LexScore':<9} | {'Composite':<9}")
    print("-" * 75)
    for idx, c in enumerate(res.all_aggregated_candidates[:8], 1):
        print(f"  {idx:<4} | '{c.text:<8}' | {f'{c.variant_count}/5':<8} | {c.agreement_ratio*100:6.1f}%   | {c.mean_model_prob*100:6.1f}%  | {c.lexicon_score:6.2f}    | {c.composite_score:7.4f}")

    # 3. Print final decision
    print("\n[C] FINAL MULTI-VARIANT DECISION:")
    print("-" * 75)
    print(f"  Primary Variant Top:     '{res.primary_variant_top}' (Prob: {res.primary_variant_confidence*100:.1f}%)")
    print(f"  Aggregated Top:          '{res.aggregated_top}'")
    print(f"  Candidate Switched:      {res.is_switched}")
    print(f"  Final Prediction:        '{res.final_prediction}'")
    print(f"  Final Confidence:        {res.confidence*100:.1f}%")
    print(f"  Requires Human Review:   {res.requires_human_review}")
    print(f"  Exact Match with GT:     {res.final_prediction == ground_truth}")

# Test 1: new_kothi.jpeg tight crop
new_kothi_path = r"C:\Users\achyu\Downloads\new_kothi.jpeg"
_, new_kothi_crop, _ = gen_evaluator.preprocess_and_crop(new_kothi_path)
evaluate_and_report("new_kothi", new_kothi_crop, "ಕೋತಿ")

# Test 2: Training sample crop_o_kothi.png
training_crop_path = r"c:\Land Record\training\datasets\personal_trial\crops\crop_o_kothi.png"
train_crop = Image.open(training_crop_path).convert("RGB")
evaluate_and_report("training_kothi", train_crop, "ಕೋತಿ")

print("\n" + "=" * 80)
print("[✓] Multi-Variant Visual Inference Test Complete.")
print("=" * 80)
