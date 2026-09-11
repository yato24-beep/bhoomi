import os
import sys
sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
from scripts.evaluate_generalization import GeneralizationEvaluator

BASE_CKPT = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
TRIAL_CKPT = r"c:\Land Record\models\trocr\personal_trial_checkpoints\best_checkpoint"

print("=" * 80)
print("TESTING BEAM SEARCH (num_beams=5) + MULTI-SIGNAL RESCORING")
print("=" * 80)

evaluator = GeneralizationEvaluator(base_checkpoint=BASE_CKPT, finetuned_checkpoint=TRIAL_CKPT)

# Test 1: Unseen new_kothi.jpeg
new_kothi_path = r"C:\Users\achyu\Downloads\new_kothi.jpeg"
print(f"\n[1] Evaluating Unseen Sample: {new_kothi_path} (GT: 'ಕೋತಿ')...")
res_new_kothi = evaluator.evaluate_sample(
    image_path=new_kothi_path,
    ground_truth="ಕೋತಿ",
)

print("\n--- NEW KOTHI BEAM CANDIDATES & RESCORING ---")
print(f"Original Top Model Candidate: '{res_new_kothi['rescoring_details']['original_top']}'")
print(f"Rescored Top Candidate:       '{res_new_kothi['rescoring_details']['rescored_candidate']}'")
print(f"Final Prediction:             '{res_new_kothi['finetuned_prediction']}' (Confidence: {res_new_kothi['finetuned_confidence']*100:.1f}%)")
print(f"Candidate Switched:           {res_new_kothi['rescoring_details']['is_switched']}")
print(f"Requires Human Review:        {res_new_kothi['requires_human_review']}")
print("\nAll Top-5 Beam Candidates:")
for cand in res_new_kothi['rescoring_details']['all_beam_candidates']:
    print(f"  Rank {cand['rank']}: '{cand['text']:8s}' | ModelProb: {cand['model_prob']*100:5.1f}% | LexScore: {cand['lexicon_score']:.2f} | OrthoScore: {cand['ortho_score']:.2f} | Composite: {cand['composite_score']:.4f}")

# Test 2: Training sample crop_o_kothi.png
training_kothi_path = r"c:\Land Record\training\datasets\personal_trial\crops\crop_o_kothi.png"
print(f"\n[2] Evaluating Training Sample: {training_kothi_path} (GT: 'ಕೋತಿ')...")
res_train_kothi = evaluator.evaluate_sample(
    image_path=training_kothi_path,
    ground_truth="ಕೋತಿ",
)

print("\n--- TRAINING SAMPLE BEAM CANDIDATES & RESCORING ---")
print(f"Original Top Model Candidate: '{res_train_kothi['rescoring_details']['original_top']}'")
print(f"Rescored Top Candidate:       '{res_train_kothi['rescoring_details']['rescored_candidate']}'")
print(f"Final Prediction:             '{res_train_kothi['finetuned_prediction']}' (Confidence: {res_train_kothi['finetuned_confidence']*100:.1f}%)")
print(f"Candidate Switched:           {res_train_kothi['rescoring_details']['is_switched']}")
print(f"Requires Human Review:        {res_train_kothi['requires_human_review']}")
print("\nAll Top-5 Beam Candidates:")
for cand in res_train_kothi['rescoring_details']['all_beam_candidates']:
    print(f"  Rank {cand['rank']}: '{cand['text']:8s}' | ModelProb: {cand['model_prob']*100:5.1f}% | LexScore: {cand['lexicon_score']:.2f} | OrthoScore: {cand['ortho_score']:.2f} | Composite: {cand['composite_score']:.4f}")

print("\n" + "=" * 80)
print("[✓] Rescoring Test Complete.")
print("=" * 80)
