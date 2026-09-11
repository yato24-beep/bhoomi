"""
Phase 8: Hard Case Evaluation Script
Compares Base Checkpoint vs Generalized Candidate Checkpoint using the GeneralizationEvaluator pipeline.
"""
import os
import sys
import json
import torch
from PIL import Image

sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.evaluate_generalization import GeneralizationEvaluator

BASE_CKPT = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
GEN_CKPT = r"c:\Land Record\models\trocr\kannada_generalized_checkpoints\best_checkpoint"

def evaluate_all_hard_cases():
    print("=" * 80)
    print("PHASE 8: HARD CASE EVALUATION (BASE VS GENERALIZED CANDIDATE)")
    print("=" * 80)
    
    evaluator = GeneralizationEvaluator(
        base_checkpoint=BASE_CKPT,
        finetuned_checkpoint=GEN_CKPT
    )
    
    test_cases = [
        {
            "name": "Unseen Personal: new_kothi.jpeg",
            "path": r"C:\Users\achyu\Downloads\new_kothi.jpeg",
            "gt": "ಕೋತಿ"
        },
        {
            "name": "Personal Crop: crop_o_kothi.png",
            "path": r"c:\Land Record\training\datasets\personal_trial\crops\crop_o_kothi.png",
            "gt": "ಕೋತಿ"
        },
        {
            "name": "Personal Crop: crop_a_mara.png",
            "path": r"c:\Land Record\training\datasets\personal_trial\crops\crop_a_mara.png",
            "gt": "ಮರ"
        },
        {
            "name": "Personal Crop: crop_p_hannu.png",
            "path": r"c:\Land Record\training\datasets\personal_trial\crops\crop_p_hannu.png",
            "gt": "ಹಣ್ಣು"
        }
    ]
    
    results = []
    
    for case in test_cases:
        if not os.path.exists(case["path"]):
            print(f"Skipping {case['name']} (not found: {case['path']})")
            continue
            
        print(f"\n--- Testing: {case['name']} ---")
        res = evaluator.evaluate_sample(
            image_path=case["path"],
            ground_truth=case["gt"],
            save_crop_path=f"scratch/{os.path.basename(case['path'])}_crop.png"
        )
        
        row = {
            "name": case["name"],
            "gt": case["gt"],
            "base_pred": res["base_prediction"],
            "base_conf": res["base_confidence"] * 100,
            "base_exact": res["base_exact_match"],
            "gen_pred": res["finetuned_prediction"],
            "gen_conf": res["finetuned_confidence"] * 100,
            "gen_exact": res["finetuned_exact_match"],
            "gen_rescored": res.get("rescored_prediction", res["finetuned_prediction"]),
            "gen_rescored_conf": res.get("rescored_confidence", res["finetuned_confidence"]) * 100
        }
        results.append(row)
        
        print(f"  Ground Truth:          '{case['gt']}'")
        print(f"  Base Model Prediction: '{row['base_pred']}' ({row['base_conf']:.1f}%) | Exact: {row['base_exact']}")
        print(f"  Gen Candidate Pred:    '{row['gen_pred']}' ({row['gen_conf']:.1f}%) | Exact: {row['gen_exact']}")
        print(f"  Gen Rescored Pred:     '{row['gen_rescored']}' ({row['gen_rescored_conf']:.1f}%)")
        
    print("\n" + "=" * 80)
    print("PHASE 8 HARD CASE SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Test Case':<32} | {'GT':<8} | {'Base Pred':<12} | {'Gen Pred':<12} | {'Base Match':<10} | {'Gen Match':<10}")
    print("-" * 92)
    for r in results:
        print(f"{r['name']:<32} | {r['gt']:<8} | {r['base_pred']:<12} | {r['gen_pred']:<12} | {str(r['base_exact']):<10} | {str(r['gen_exact']):<10}")
        
    with open("hard_case_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    evaluate_all_hard_cases()
