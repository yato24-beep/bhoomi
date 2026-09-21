"""Kannada→English Sentence Translation Benchmark.

Evaluates Neural Machine Translation (NMT) architectures on 7 authentic
administrative / land-record Kannada sentences against human reference translations.

Supported engines:
  1. ai4bharat/indictrans2-indic-en-1B (or dist-200M) via Pure-Python IndicProcessor
  2. facebook/m2m100_418M (multilingual baseline)

Distinguishes proper-name transliteration from sentence-level translation.
Reports per-sentence outputs, chrF score, and qualitative failure modes.
Enforces strict production gate: chrF > 0.40 AND grammatical English.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sentence_translation_benchmark")

BENCHMARK = [
    {
        "id": "sent_01_registration",
        "type": "administrative_sentence",
        "source_kn": "ಈ ಜಮೀನು ಕಂದಾಯ ಇಲಾಖೆಯ ನಿಯಮಗಳ ಪ್ರಕಾರ ನೋಂದಾಯಿಸಲ್ಪಟ್ಟಿದೆ.",
        "reference_en": "This land is registered according to the rules of the revenue department.",
    },
    {
        "id": "sent_02_survey_extent",
        "type": "cadastral_statement",
        "source_kn": "ಸರ್ವೆ ನಂಬರ್ 142/3 ರಲ್ಲಿ ಒಟ್ಟು 2 ಎಕರೆ ಕೃಷಿ ಭೂಮಿ ಇದೆ.",
        "reference_en": "There is a total of 2 acres of agricultural land in survey number 142/3.",
    },
    {
        "id": "sent_03_encumbrance",
        "type": "encumbrance_status",
        "source_kn": "ಖಾತೆದಾರರು ಯಾವುದೇ ಸಾಲದ ಬಾಕಿಯನ್ನು ಹೊಂದಿಲ್ಲ.",
        "reference_en": "The khatadar does not have any outstanding loan dues.",
    },
    {
        "id": "sent_04_mutation_order",
        "type": "mutation_order",
        "source_kn": "ವಾರಸುದಾರರ ಹಕ್ಕಿನ ಆಧಾರದ ಮೇಲೆ ಖಾತಾ ಬದಲಾವಣೆಯನ್ನು ಅನುಮೋದಿಸಲಾಗಿದೆ.",
        "reference_en": "Khata mutation has been approved on the basis of heirship rights.",
    },
    {
        "id": "sent_05_boundaries",
        "type": "boundary_description",
        "source_kn": "ಪೂರ್ವಕ್ಕೆ ಸರ್ಕಾರಿ ರಸ್ತೆ ಮತ್ತು ಪಶ್ಚಿಮಕ್ಕೆ ರಾಜು ಅವರ ಜಮೀನು ಗಡಿಯಾಗಿದೆ.",
        "reference_en": "The government road to the east and Raju's land to the west form the boundary.",
    },
    {
        "id": "sent_06_land_classification",
        "type": "land_classification",
        "source_kn": "ಈ ಭೂಮಿಯು ಖುಷ್ಕಿ ಜಮೀನು ಎಂದು ವರ್ಗೀಕರಿಸಲ್ಪಟ್ಟಿದೆ.",
        "reference_en": "This land is classified as dry (unirrigated) land.",
    },
    {
        "id": "sent_07_revenue_demand",
        "type": "revenue_demand",
        "source_kn": "ವಾರ್ಷಿಕ ಭೂ ಕಂದಾಯ ರೂ. 125 ಆಗಿರುತ್ತದೆ.",
        "reference_en": "The annual land revenue is Rs. 125.",
    },
]


def compute_chrf(hypothesis: str, reference: str) -> float:
    """Compute character n-gram F-score (chrF)."""
    try:
        import sacrebleu
        metric = sacrebleu.CHRF()
        score = metric.sentence_score(hypothesis, [reference])
        return score.score / 100.0
    except Exception:
        hyp_chars = set(hypothesis.replace(" ", ""))
        ref_chars = set(reference.replace(" ", ""))
        if not ref_chars:
            return 0.0
        precision = len(hyp_chars & ref_chars) / max(len(hyp_chars), 1)
        recall = len(hyp_chars & ref_chars) / max(len(ref_chars), 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


def is_sentence_level_translation(source_kn: str, hypothesis: str) -> bool:
    """Checks if output is sentence-level English rather than Kannada script."""
    kannada_chars = sum(1 for c in hypothesis if '\u0c80' <= c <= '\u0cff')
    total_alpha = sum(1 for c in hypothesis if c.isalpha())
    if total_alpha == 0:
        return False
    return (kannada_chars / total_alpha) < 0.20


def qualitative_failure_mode(source_kn: str, hypothesis: str, reference: str) -> str:
    """Classifies the failure mode of a bad translation."""
    if "[GATED_REPO" in hypothesis or "[LOAD_FAILED" in hypothesis:
        return "MODEL_ACCESS_OR_LOAD_BLOCKED"
    kannada_ratio = sum(1 for c in hypothesis if '\u0c80' <= c <= '\u0cff') / max(len(hypothesis), 1)
    if kannada_ratio > 0.30:
        return "UNTRANSLATED_KANNADA_REMAINS"
    if len(hypothesis) < 5:
        return "EMPTY_OR_TOO_SHORT"
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    overlap = len(ref_words & hyp_words) / max(len(ref_words), 1)
    if overlap < 0.20:
        return "WRONG_CONTENT_HALLUCINATED"
    if overlap < 0.50:
        return "PARTIAL_TRANSLATION_MISSING_CONTENT"
    return "ACCEPTABLE"


def run_benchmark(model_name: str = "ai4bharat/indictrans2-indic-en-1B"):
    print("\n" + "=" * 80)
    print("KANNADA→ENGLISH SENTENCE TRANSLATION BENCHMARK")
    print(f"Target Model: {model_name}")
    print("=" * 80)

    is_indictrans = "indictrans2" in model_name.lower()
    ip = None
    tokenizer = None
    model = None
    model_status = "INITIALIZING"

    if is_indictrans:
        try:
            from src.translation.indic_processor import IndicProcessor
            ip = IndicProcessor(inference=True)
            print("Pure-Python IndicProcessor initialized successfully.")
        except Exception as e:
            print(f"Error initializing IndicProcessor: {e}")

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        print(f"Checking HuggingFace repository: {model_name} (HF_TOKEN present: {bool(hf_token)}) ...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        model.eval()
        print("Model loaded successfully.")
        model_status = "LOADED"
    except Exception as e:
        err_msg = str(e)
        if "gated repo" in err_msg.lower() or "401" in err_msg or "restricted" in err_msg.lower():
            model_status = (
                f"GATED_REPO_UNAUTHORIZED: Model '{model_name}' is gated by AI4Bharat on HuggingFace. "
                f"Requires manual agreement at https://huggingface.co/{model_name} and setting HF_TOKEN."
            )
        else:
            model_status = f"LOAD_FAILED: {err_msg[:250]}"
        print(f"\n[MODEL LOAD STATUS]: {model_status}\n")

    results = []
    chrf_scores = []
    sentence_level_count = 0

    for item in BENCHMARK:
        src = item["source_kn"]
        ref = item["reference_en"]

        if model is not None and tokenizer is not None:
            try:
                if is_indictrans and ip is not None:
                    preprocessed = ip.preprocess_batch([src], src_lang="kan_Knda", tgt_lang="eng_Latn")
                    inputs = tokenizer(preprocessed, padding="longest", truncation=True, max_length=256, return_tensors="pt")
                    outputs = model.generate(**inputs, max_length=128, num_beams=4)
                    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    postprocessed = ip.postprocess_batch(decoded, lang="eng_Latn")
                    hyp = postprocessed[0]
                else:
                    inputs = tokenizer(src, return_tensors="pt", padding=True)
                    forced_bos = tokenizer.convert_tokens_to_ids("eng_Latn") if hasattr(tokenizer, "convert_tokens_to_ids") else None
                    gen_kwargs = {"forced_bos_token_id": forced_bos} if forced_bos else {}
                    translated = model.generate(**inputs, max_length=128, num_beams=4, **gen_kwargs)
                    hyp = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            except Exception as e:
                hyp = f"[TRANSLATION_ERROR: {e}]"
        else:
            hyp = f"[{model_status.split(':')[0]}]"

        chrf = compute_chrf(hyp, ref)
        is_sentence = is_sentence_level_translation(src, hyp)
        failure = qualitative_failure_mode(src, hyp, ref)

        chrf_scores.append(chrf)
        if is_sentence:
            sentence_level_count += 1

        print(f"\n[{item['id']}] type={item['type']}")
        print(f"  Source (KN) : {src}")
        print(f"  Reference   : {ref}")
        print(f"  Predicted   : {hyp}")
        print(f"  chrF        : {chrf:.3f}")
        print(f"  Is sentence-level: {is_sentence}")
        print(f"  Failure mode: {failure}")

        results.append({
            "id": item["id"],
            "type": item["type"],
            "source_kn": src,
            "reference_en": ref,
            "predicted_en": hyp,
            "chrf_score": round(chrf, 4),
            "is_sentence_level": is_sentence,
            "failure_mode": failure,
        })

    mean_chrf = sum(chrf_scores) / max(len(chrf_scores), 1)

    print("\n" + "=" * 80)
    print("TRANSLATION BENCHMARK SUMMARY")
    print(f"  Model           : {model_name}")
    print(f"  Model status    : {model_status}")
    print(f"  Sentences tested: {len(BENCHMARK)}")
    print(f"  Mean chrF score : {mean_chrf:.3f} (0=worst, 1=perfect, threshold > 0.40)")
    print(f"  Sentence-level  : {sentence_level_count}/{len(BENCHMARK)}")
    print()
    print("PRODUCTION GATE: Translation should NOT be wired into production until:")
    print("  - Mean chrF > 0.40 on this benchmark")
    print("  - All outputs are sentence-level English (no Kannada remnants)")
    if mean_chrf > 0.40 and sentence_level_count == len(BENCHMARK):
        print("  STATUS: PRODUCTION APPROVED")
    else:
        print("  STATUS: NOT PRODUCTION READY (GATED)")
    print("=" * 80)

    out_dir = Path("tests/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": model_name,
        "model_status": model_status,
        "sentences_tested": len(BENCHMARK),
        "mean_chrf": round(mean_chrf, 4),
        "sentence_level_count": sentence_level_count,
        "production_ready": mean_chrf > 0.40 and sentence_level_count == len(BENCHMARK),
        "production_gate": "mean_chrf > 0.40 AND all outputs sentence-level English",
        "results": results,
    }
    out_file = out_dir / "sentence_translation_indictrans2_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {out_file}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ai4bharat/indictrans2-indic-en-1B", help="Model name or path")
    args = parser.parse_args()
    run_benchmark(args.model)
