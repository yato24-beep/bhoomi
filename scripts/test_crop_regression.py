import os
import sys
sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, XLMRobertaTokenizer

from scripts.evaluate_generalization import GeneralizationEvaluator

BASE_CKPT = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
TRIAL_CKPT = r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"
OUT_DIR = r"c:\Land Record\scratch\kothi_debug"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 80)
print("RUNNING REGRESSION TEST: KANNADA TrOCR HANDWRITING CROP & INFERENCE")
print("=" * 80)

evaluator = GeneralizationEvaluator(base_checkpoint=BASE_CKPT, finetuned_checkpoint=TRIAL_CKPT)

# 1. Test Unseen Handwriting Image: new_kothi.jpeg
new_kothi_path = r"C:\Users\achyu\Downloads\new_kothi.jpeg"
print(f"\n[1] Evaluating Unseen Sample: {new_kothi_path}...")
res_new_kothi = evaluator.evaluate_sample(
    image_path=new_kothi_path,
    ground_truth="ಕೋತಿ",
    save_crop_path=os.path.join(OUT_DIR, "after_tight_crop.png")
)

# 2. Test Existing Training Sample: crop_o_kothi.png
training_kothi_path = r"c:\Land Record\training\datasets\personal_trial\crops\crop_o_kothi.png"
print(f"\n[2] Evaluating Known Training Sample: {training_kothi_path}...")
res_train_kothi = evaluator.evaluate_sample(
    image_path=training_kothi_path,
    ground_truth="ಕೋತಿ",
    save_crop_path=os.path.join(OUT_DIR, "training_kothi_verified_crop.png")
)

# 3. Save Before Crop (the oversized 1400x731 fallback)
raw_img = Image.open(new_kothi_path)
raw_img.save(os.path.join(OUT_DIR, "before_crop_raw.png"))

# 4. Regression Assertions
old_dim = (1400, 731)
new_dim = res_new_kothi["metadata"]["crop_size"]

is_substantially_tighter = (new_dim[0] < old_dim[0] * 0.7) and (new_dim[1] < old_dim[1] * 0.7)
ink_occupancy_ratio = (new_dim[0] * new_dim[1]) / (old_dim[0] * old_dim[1]) * 100

print("\n" + "=" * 80)
print("REGRESSION TEST RESULTS SUMMARY")
print("=" * 80)
print(f"Old Fallback Crop Dimensions:  {old_dim[0]} x {old_dim[1]} (Area: {old_dim[0]*old_dim[1]} px)")
print(f"New Tight Crop Dimensions:      {new_dim[0]} x {new_dim[1]} (Area: {new_dim[0]*new_dim[1]} px)")
print(f"Crop Area Reduction:            {100 - ink_occupancy_ratio:.1f}% reduction (Crop is {is_substantially_tighter})")
print(f"New Base Prediction:            '{res_new_kothi['base_prediction']}' (Conf: {res_new_kothi['base_confidence']*100:.1f}%)")
print(f"New Fine-Tuned Prediction:      '{res_new_kothi['finetuned_prediction']}' (Conf: {res_new_kothi['finetuned_confidence']*100:.1f}%)")
print(f"Training Sample Prediction:     '{res_train_kothi['finetuned_prediction']}' (Conf: {res_train_kothi['finetuned_confidence']*100:.1f}%, Exact Match: {res_train_kothi['finetuned_exact_match']})")
print(f"Zero Numeral Output:            {'PASS (No digits generated)' if not any(c in '೦೧೨೩೪೫೬೭೮೯0123456789' for c in res_new_kothi['finetuned_prediction']) else 'FAIL'}")
print(f"Regression Test Status:         {'[PASS]' if is_substantially_tighter and not any(c in '೦೧೨೩೪೫೬೭೮೯0123456789' for c in res_new_kothi['finetuned_prediction']) else '[FAIL]'}")
print("=" * 80)
