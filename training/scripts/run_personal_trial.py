import os
import sys
import json
import time
import torch
from PIL import Image

sys.path.insert(0, r"c:\Land Record")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import src

from src.training.config import TrainingConfig
from src.training.dataset import HandwritingDataset
from src.training.trainer import HandwritingTrainer
from src.training.checkpointing import load_training_checkpoint, save_training_checkpoint, CheckpointMetadata
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, XLMRobertaTokenizer

CONFIG_PATH = r"c:\Land Record\training\configs\personal_trial.yaml"
BASE_CKPT = r"c:\Land Record\models\trocr\kannada_full_checkpoints\best_checkpoint"
TRIAL_OUTPUT_DIR = r"c:\Land Record\models\trocr\personal_trial_checkpoints"

os.makedirs(TRIAL_OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("  PERSONAL HANDWRITING FINE-TUNING TRIAL (3 SAMPLES)")
print("=" * 70)

# Samples for evaluation
test_samples = [
    {
        "name": "a.jpeg",
        "crop_path": r"c:\Land Record\training\datasets\personal_trial\crops\crop_a_mara.png",
        "ground_truth": "ಮರ",
        "english": "TREE",
    },
    {
        "name": "o.jpeg",
        "crop_path": r"c:\Land Record\training\datasets\personal_trial\crops\crop_o_kothi.png",
        "ground_truth": "ಕೋತಿ",
        "english": "MONKEY",
    },
    {
        "name": "p.jpeg",
        "crop_path": r"c:\Land Record\training\datasets\personal_trial\crops\crop_p_hannu.png",
        "ground_truth": "ಹಣ್ಣು",
        "english": "FRUIT",
    },
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# -------------------------------------------------------------
# STEP 1: Evaluate BEFORE Fine-Tuning (Baseline Checkpoint)
# -------------------------------------------------------------
print("\n--- [STEP 1] Evaluating Original Checkpoint (Before Fine-Tuning) ---")
tokenizer = XLMRobertaTokenizer.from_pretrained(BASE_CKPT)
processor = TrOCRProcessor.from_pretrained(BASE_CKPT)
model_orig = VisionEncoderDecoderModel.from_pretrained(BASE_CKPT).to(device)
model_orig.eval()

def run_sample_inference(model, proc, tok, s_info):
    img = Image.open(s_info["crop_path"]).convert("RGB")
    pixel_values = proc(img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_new_tokens=32,
            return_dict_in_generate=True,
            output_scores=True,
        )
    gen_ids = outputs.sequences[0]
    pred_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    
    # Compute confidence from softmax logits
    if outputs.scores:
        token_probs = []
        for step_idx, step_logits in enumerate(outputs.scores):
            tok_id = gen_ids[step_idx + 1] if step_idx + 1 < len(gen_ids) else None
            if tok_id is not None:
                probs = torch.softmax(step_logits[0], dim=-1)
                token_probs.append(probs[tok_id].item())
        conf = float(sum(token_probs) / len(token_probs)) if token_probs else 0.50
    else:
        conf = 0.50
    return pred_text, round(conf, 4)

before_results = []
for s in test_samples:
    pred, conf = run_sample_inference(model_orig, processor, tokenizer, s)
    before_results.append({
        "sample": s["name"],
        "ground_truth": s["ground_truth"],
        "english": s["english"],
        "orig_prediction": pred,
        "orig_confidence": conf,
    })
    print(f"  {s['name']} -> GT: '{s['ground_truth']}' | Orig Pred: '{pred}' (conf={conf:.4f})")

# -------------------------------------------------------------
# STEP 2: Fine-Tune on the 3 Personal Samples for 3 Epochs
# -------------------------------------------------------------
print("\n--- [STEP 2] Executing Fine-Tuning Trial (3 Epochs) ---")
config = TrainingConfig.from_yaml(CONFIG_PATH)

train_dataset = HandwritingDataset(
    manifest_path=config.dataset.train_manifest,
    processor=processor,
    max_target_length=32,
    root_dir=r"c:\Land Record",
)
val_dataset = HandwritingDataset(
    manifest_path=config.dataset.val_manifest,
    processor=processor,
    max_target_length=32,
    root_dir=r"c:\Land Record",
)

trainer = HandwritingTrainer(
    config=config,
    model=model_orig,
    processor=processor,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

train_summary = trainer.train(num_epochs=3)
print("\nFine-tuning completed successfully!")
print(f"Total steps: {train_summary['total_steps']}, Best CER: {train_summary['best_cer']:.4f}")

# -------------------------------------------------------------
# STEP 3: Evaluate AFTER Fine-Tuning (Fine-Tuned Checkpoint)
# -------------------------------------------------------------
print("\n--- [STEP 3] Evaluating Fine-Tuned Checkpoint (After Fine-Tuning) ---")
best_trial_ckpt = os.path.join(TRIAL_OUTPUT_DIR, "best_checkpoint")
if not os.path.exists(best_trial_ckpt):
    best_trial_ckpt = os.path.join(TRIAL_OUTPUT_DIR, f"checkpoint-step-{train_summary['total_steps']}")

model_finetuned = VisionEncoderDecoderModel.from_pretrained(best_trial_ckpt).to(device)
model_finetuned.eval()

after_results = []
for idx, s in enumerate(test_samples):
    pred, conf = run_sample_inference(model_finetuned, processor, tokenizer, s)
    b_res = before_results[idx]
    after_results.append({
        "image": s["name"],
        "ground_truth": s["ground_truth"],
        "english": s["english"],
        "orig_prediction": b_res["orig_prediction"],
        "orig_confidence": b_res["orig_confidence"],
        "finetuned_prediction": pred,
        "finetuned_confidence": conf,
        "exact_match_before": bool(b_res["orig_prediction"] == s["ground_truth"]),
        "exact_match_after": bool(pred == s["ground_truth"]),
    })
    print(f"  {s['name']} -> GT: '{s['ground_truth']}' | Before: '{b_res['orig_prediction']}' | After: '{pred}' (conf={conf:.4f})")

# -------------------------------------------------------------
# STEP 4: Save Structured Trial Report
# -------------------------------------------------------------
trial_report = {
    "trial_name": "personal_handwritten_kannada_fine_tune_trial",
    "base_checkpoint": BASE_CKPT,
    "trial_checkpoint": best_trial_ckpt,
    "device": device,
    "num_epochs": 3,
    "num_samples": len(test_samples),
    "results": after_results,
    "training_history": train_summary["history"],
}

out_report_path = r"c:\Land Record\scratch\personal_trial_report.json"
with open(out_report_path, "w", encoding="utf-8") as f:
    json.dump(trial_report, f, indent=2, ensure_ascii=False)

print(f"\n[TRIAL REPORT SAVED] {out_report_path}")
