import json
import os
import sys
import random
import re
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

random.seed(42)

TRAIN_MANIFEST = r"c:\Land Record\training\datasets\iiit_kannada_train.jsonl"
OUT_DIR = r"c:\Land Record\training\datasets\kannada_character_balanced"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "train_balanced.jsonl")

# Priority categories based on Phase 1 Audit and Phase 3 Weakness Analysis
RARE_CHARS = ["ಋ", "ಙ", "ಔ", "ಝ", "ಊ", "ಢ", "ಞ", "ಛ", "ಓ", "ಐ", "ಏ", "ಠ", "ಘ", "ಫ", "ಖ", "ಣ"]
CONFUSING_SYLLABLES = [
    "ತಿ", "ಥಿ", "ದಿ", "ನಿ", "ಲಿ", "ರಿ", "ಟಿ", "ಡಿ", "ಠಿ", "ಢಿ",
    "ತ", "ದ", "ನ", "ಲ", "ಳ", "ರ", "ಮ", "ಶ", "ಷ", "ಸ", "ಹ", "ಬ", "ಭ", "ಪ"
]
RARE_MATRAS = ["ೂ", "ೈ", "ೌ", "ೃ", "ೀ", "ೊ", "ೋ"]
NUMERALS = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"]
PUNCTUATIONS = ["-", ".", ",", "/", "(", ")", "?"]

def build_balanced_dataset():
    print(f"Reading training pool from: {TRAIN_MANIFEST}...", flush=True)
    valid_train_samples = []
    with open(TRAIN_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            rel_img = item.get("image", "")
            img_p = os.path.join(r"c:\Land Record", rel_img)
            if os.path.exists(img_p):
                valid_train_samples.append(item)

    print(f"Total valid training images: {len(valid_train_samples):,}", flush=True)

    pools = defaultdict(list)
    short_words_pool = []
    long_words_pool = []
    conjunct_pool = []
    numeral_pool = []
    punct_pool = []

    kannada_cons_pattern = r"[\u0C95-\u0CB9\u0CB3]"
    conjunct_regex = re.compile(f"{kannada_cons_pattern}\u0CCD{kannada_cons_pattern}")

    for s in valid_train_samples:
        txt = s.get("text", "")
        if not txt:
            continue

        # Check rare chars
        for rc in RARE_CHARS:
            if rc in txt:
                pools[f"rare_{rc}"].append(s)

        # Check confusing syllables
        for cs in CONFUSING_SYLLABLES:
            if cs in txt:
                pools[f"confuse_{cs}"].append(s)

        # Check rare matras
        for rm in RARE_MATRAS:
            if rm in txt:
                pools[f"matra_{rm}"].append(s)

        # Check numerals
        for num in NUMERALS:
            if num in txt:
                pools[f"num_{num}"].append(s)
                numeral_pool.append(s)

        # Check punctuation
        for p in PUNCTUATIONS:
            if p in txt:
                pools[f"punct_{p}"].append(s)
                punct_pool.append(s)

        # Conjuncts
        if conjunct_regex.search(txt):
            conjunct_pool.append(s)

        # Short words vs Long words
        if len(txt) <= 4:
            short_words_pool.append(s)
        elif len(txt) >= 8:
            long_words_pool.append(s)

    selected_samples = {}

    def add_from_pool(pool, max_k):
        if not pool:
            return
        sample_k = random.sample(pool, min(max_k, len(pool)))
        for s in sample_k:
            selected_samples[s["image"]] = s

    # 1. Heavily harvest all rare characters
    for rc in RARE_CHARS:
        add_from_pool(pools[f"rare_{rc}"], 60)

    # 2. Add confusing syllables (up to 40 per category)
    for cs in CONFUSING_SYLLABLES:
        add_from_pool(pools[f"confuse_{cs}"], 40)

    # 3. Add rare matras (up to 50 per category)
    for rm in RARE_MATRAS:
        add_from_pool(pools[f"matra_{rm}"], 50)

    # 4. Add all numeral samples (harvest up to 100 per numeral)
    for num in NUMERALS:
        add_from_pool(pools[f"num_{num}"], 100)

    # 5. Add conjunct samples
    add_from_pool(conjunct_pool, 300)

    # 6. Add short words & long words
    add_from_pool(short_words_pool, 300)
    add_from_pool(long_words_pool, 300)
    add_from_pool(punct_pool, 150)

    # 7. Add general diverse anchor samples
    anchor_candidates = [s for s in valid_train_samples if s["image"] not in selected_samples]
    add_from_pool(anchor_candidates, 800)

    # 8. Add personal trial samples
    personal_samples = [
        {"image": "training/datasets/personal_trial/crops/crop_o_kothi.png", "text": "ಕೋತಿ", "language": "kannada", "script": "Kannada"},
        {"image": "training/datasets/personal_trial/crops/crop_a_mara.png", "text": "ಮರ", "language": "kannada", "script": "Kannada"},
        {"image": "training/datasets/personal_trial/crops/crop_p_hannu.png", "text": "ಹಣ್ಣು", "language": "kannada", "script": "Kannada"},
    ]
    for _ in range(5):
        for ps in personal_samples:
            selected_samples[f"{ps['image']}_{_}"] = ps

    final_dataset = list(selected_samples.values())
    random.shuffle(final_dataset)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[✓] Balanced Generalized Training Set Built: {len(final_dataset):,} samples saved to {OUT_PATH}", flush=True)

if __name__ == "__main__":
    build_balanced_dataset()
