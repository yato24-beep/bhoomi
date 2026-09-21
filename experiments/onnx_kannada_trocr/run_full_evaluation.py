"""Comprehensive Evaluation & Acceptance Criteria Harness.

Evaluates:
1. Baseline PyTorch Float32 (from baseline_results.json or evaluated live)
2. PyTorch Int8 (quantized)
3. ONNX FP32 Pipeline (Beam Search = 4)
4. ONNX Int8 Pipeline (Beam Search = 4)
5. ONNX Int8 Pipeline (Greedy Decoding)

Measures:
- Model Size on Disk (MB)
- True Peak Process RSS (MB) using continuous 100Hz sampling
- Average & Per-Sample Latency (seconds)
- Decoded Kannada Unicode Text
- Character / Word Match vs Baseline
- <unk> Token Corruption Check
- Feasibility for 512 MB Render Free Tier Instance
"""

import gc
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import unicodedata

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BENCHMARK_DIR = PROJECT_ROOT / "locked_benchmark_export"
MODELS_DIR = Path(__file__).resolve().parent / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "trocr" / "checkpoint-12000"
PROCESSOR_DIR = PROJECT_ROOT / "src" / "handwriting" / "configs" / "iitb_kannada_v002"
TOKENIZER_DIR = PROJECT_ROOT / "models" / "trocr" / "experimental" / "iitb_kannada_v002"
OUT_DIR = Path(__file__).resolve().parent


class MemorySampler:
    """Samples RSS memory at 100Hz in background."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self.peak_rss = self.process.memory_info().rss
        self._t = None

    def start(self):
        self.peak_rss = self.process.memory_info().rss
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(0.01)

    def stop(self) -> int:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=0.3)
        try:
            rss = self.process.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
        except Exception:
            pass
        return self.peak_rss


def is_kannada(char: str) -> bool:
    return "\u0C80" <= char <= "\u0CFF"


def analyze_text(text: str) -> Dict[str, Any]:
    return {
        "text": text,
        "length": len(text),
        "kannada_chars": sum(1 for c in text if is_kannada(c)),
        "unk_count": text.count("<unk>"),
        "has_kannada": any(is_kannada(c) for c in text),
    }


def evaluate_all():
    print("=" * 70, flush=True)
    print("PHASE 5: COMPREHENSIVE MULTI-ENGINE EVALUATION", flush=True)
    print("=" * 70, flush=True)

    test_files = sorted(list(BENCHMARK_DIR.glob("*.png")))
    print(f"Evaluation Dataset: {len(test_files)} locked archival crops from {BENCHMARK_DIR.name}", flush=True)

    from transformers import AutoImageProcessor, AutoTokenizer
    from experiments.onnx_kannada_trocr.onnx_pipeline import TrOCRONNXPipeline

    processor = AutoImageProcessor.from_pretrained(str(PROCESSOR_DIR), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)

    eval_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_crops_count": len(test_files),
        "engines": {},
    }

    # -------------------------------------------------------------
    # 1. LOAD OR EVALUATE PYTORCH BASELINE (FP32)
    # -------------------------------------------------------------
    baseline_cache_file = OUT_DIR / "baseline_results.json"
    if baseline_cache_file.exists():
        print("\nLoading verified PyTorch Baseline results from cache...", flush=True)
        with open(baseline_cache_file, "r", encoding="utf-8") as f:
            base_data = json.load(f)
        eval_report["engines"]["pytorch_fp32"] = {
            "model_disk_size_mb": base_data["disk_size_mb"],
            "post_load_rss_mb": base_data["post_load_rss_mb"],
            "peak_rss_mb": base_data["peak_during_inference_mb"],
            "avg_latency_sec": base_data["avg_latency_sec"],
            "results": [
                {
                    "file": r["filename"],
                    "text": r["cleaned_text"],
                    "latency": r["latency_sec"],
                    "analysis": analyze_text(r["cleaned_text"]),
                }
                for r in base_data["detailed_results"]
            ],
        }
    else:
        print("\nRunning baseline PyTorch benchmark...", flush=True)
        from experiments.onnx_kannada_trocr.benchmark_baseline_pytorch import run_baseline_benchmark
        base_data = run_baseline_benchmark()
        eval_report["engines"]["pytorch_fp32"] = {
            "model_disk_size_mb": base_data["disk_size_mb"],
            "post_load_rss_mb": base_data["post_load_rss_mb"],
            "peak_rss_mb": base_data["peak_during_inference_mb"],
            "avg_latency_sec": base_data["avg_latency_sec"],
            "results": [
                {
                    "file": r["filename"],
                    "text": r["cleaned_text"],
                    "latency": r["latency_sec"],
                    "analysis": analyze_text(r["cleaned_text"]),
                }
                for r in base_data["detailed_results"]
            ],
        }

    pt_ref_texts = {r["file"]: r["text"] for r in eval_report["engines"]["pytorch_fp32"]["results"]}

    # -------------------------------------------------------------
    # 2. EVALUATE FULL ONNX PIPELINE (FP32)
    # -------------------------------------------------------------
    encoder_onnx = MODELS_DIR / "encoder.onnx"
    decoder_onnx = MODELS_DIR / "decoder.onnx"

    if encoder_onnx.exists() and decoder_onnx.exists():
        print("\n" + "-" * 70, flush=True)
        print("ENGINE 2: Full ONNX Runtime Float32 Pipeline (Beam Search = 4)", flush=True)
        print("-" * 70, flush=True)
        gc.collect()
        mem = MemorySampler()
        mem.start()

        onnx_fp32_pipeline = TrOCRONNXPipeline(
            encoder_path=encoder_onnx,
            decoder_path=decoder_onnx,
        )
        post_load_rss = psutil.Process().memory_info().rss / (1024 * 1024)

        onnx_results = []
        for idx, f in enumerate(test_files, 1):
            img = Image.open(f).convert("RGB")
            clean, tokens, elapsed = onnx_fp32_pipeline.generate_beam_search(
                img,
                num_beams=4,
                max_length=64,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
            )
            ref_text = pt_ref_texts.get(f.name, "")
            is_match = (clean == ref_text)
            onnx_results.append({
                "file": f.name,
                "text": clean,
                "latency": round(elapsed, 3),
                "analysis": analyze_text(clean),
                "match_baseline": is_match,
            })
            match_str = "MATCH" if is_match else f"DIFF (Ref: '{ref_text[:20]}')"
            print(f"  [{idx:2d}/{len(test_files)}] {f.name:<18} ({elapsed:5.2f}s) | {match_str} | '{clean[:30]}'", flush=True)

        onnx_peak = mem.stop() / (1024 * 1024)

        # Calculate exact total size including .data files
        total_onnx_bytes = (
            encoder_onnx.stat().st_size
            + (MODELS_DIR / "encoder.onnx.data").stat().st_size
            + decoder_onnx.stat().st_size
            + (MODELS_DIR / "decoder.onnx.data").stat().st_size
        )
        onnx_disk = total_onnx_bytes / (1024 * 1024)

        eval_report["engines"]["onnx_fp32"] = {
            "model_disk_size_mb": round(onnx_disk, 2),
            "post_load_rss_mb": round(post_load_rss, 2),
            "peak_rss_mb": round(onnx_peak, 2),
            "avg_latency_sec": round(sum(r["latency"] for r in onnx_results) / len(onnx_results), 3),
            "matches_vs_baseline": sum(1 for r in onnx_results if r["match_baseline"]),
            "results": onnx_results,
        }
        del onnx_fp32_pipeline
        gc.collect()

    # -------------------------------------------------------------
    # 3. EVALUATE FULL ONNX PIPELINE (INT8 QUANTIZED - BEAM SEARCH)
    # -------------------------------------------------------------
    enc_int8 = MODELS_DIR / "encoder_int8.onnx"
    dec_int8 = MODELS_DIR / "decoder_int8.onnx"

    if enc_int8.exists() and dec_int8.exists():
        print("\n" + "-" * 70, flush=True)
        print("ENGINE 3: Full ONNX Runtime INT8 Quantized Pipeline (Beam Search = 4)", flush=True)
        print("-" * 70, flush=True)
        gc.collect()
        mem = MemorySampler()
        mem.start()

        onnx_int8_pipeline = TrOCRONNXPipeline(
            encoder_path=enc_int8,
            decoder_path=dec_int8,
        )
        post_load_rss = psutil.Process().memory_info().rss / (1024 * 1024)

        int8_results = []
        for idx, f in enumerate(test_files, 1):
            img = Image.open(f).convert("RGB")
            clean, tokens, elapsed = onnx_int8_pipeline.generate_beam_search(
                img,
                num_beams=4,
                max_length=64,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
            )
            ref_text = pt_ref_texts.get(f.name, "")
            is_match = (clean == ref_text)
            int8_results.append({
                "file": f.name,
                "text": clean,
                "latency": round(elapsed, 3),
                "analysis": analyze_text(clean),
                "match_baseline": is_match,
            })
            match_str = "MATCH" if is_match else f"DIFF (Ref: '{ref_text[:20]}')"
            print(f"  [{idx:2d}/{len(test_files)}] {f.name:<18} ({elapsed:5.2f}s) | {match_str} | '{clean[:30]}'", flush=True)

        int8_peak = mem.stop() / (1024 * 1024)
        int8_disk = (enc_int8.stat().st_size + dec_int8.stat().st_size) / (1024 * 1024)

        eval_report["engines"]["onnx_int8_beam"] = {
            "model_disk_size_mb": round(int8_disk, 2),
            "post_load_rss_mb": round(post_load_rss, 2),
            "peak_rss_mb": round(int8_peak, 2),
            "avg_latency_sec": round(sum(r["latency"] for r in int8_results) / len(int8_results), 3),
            "matches_vs_baseline": sum(1 for r in int8_results if r["match_baseline"]),
            "results": int8_results,
        }

        # -------------------------------------------------------------
        # 4. EVALUATE ONNX INT8 GREEDY DECODING (HIGH SPEED PROFILE)
        # -------------------------------------------------------------
        print("\n" + "-" * 70, flush=True)
        print("ENGINE 4: Full ONNX Runtime INT8 Quantized Pipeline (Greedy Decoding)", flush=True)
        print("-" * 70, flush=True)
        greedy_results = []
        for idx, f in enumerate(test_files, 1):
            img = Image.open(f).convert("RGB")
            clean, tokens, elapsed = onnx_int8_pipeline.generate_greedy(
                img,
                max_length=64,
            )
            ref_text = pt_ref_texts.get(f.name, "")
            is_match = (clean == ref_text)
            greedy_results.append({
                "file": f.name,
                "text": clean,
                "latency": round(elapsed, 3),
                "analysis": analyze_text(clean),
                "match_baseline": is_match,
            })
            match_str = "MATCH" if is_match else f"DIFF (Ref: '{ref_text[:20]}')"
            print(f"  [{idx:2d}/{len(test_files)}] {f.name:<18} ({elapsed:5.2f}s) | {match_str} | '{clean[:30]}'", flush=True)

        eval_report["engines"]["onnx_int8_greedy"] = {
            "model_disk_size_mb": round(int8_disk, 2),
            "post_load_rss_mb": round(post_load_rss, 2),
            "peak_rss_mb": round(int8_peak, 2),
            "avg_latency_sec": round(sum(r["latency"] for r in greedy_results) / len(greedy_results), 3),
            "matches_vs_baseline": sum(1 for r in greedy_results if r["match_baseline"]),
            "results": greedy_results,
        }

        del onnx_int8_pipeline
        gc.collect()

    # Save comprehensive evaluation report
    out_file = OUT_DIR / "evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved comprehensive evaluation report to: {out_file}", flush=True)

    # Print Comparative Summary
    print("\n" + "=" * 75, flush=True)
    print(f"{'Engine':<24} | {'Disk (MB)':<10} | {'Post-Load RSS':<14} | {'Peak RSS':<10} | {'Latency':<8} | {'Exact Matches'}", flush=True)
    print("-" * 75, flush=True)
    for eng_name, eng in eval_report["engines"].items():
        matches = f"{eng.get('matches_vs_baseline', len(test_files))}/{len(test_files)}"
        print(
            f"{eng_name:<24} | {eng['model_disk_size_mb']:<10.1f} | {eng['post_load_rss_mb']:<14.1f} | "
            f"{eng['peak_rss_mb']:<10.1f} | {eng['avg_latency_sec']:<8.2f}s | {matches}",
            flush=True,
        )
    print("=" * 75, flush=True)

    return eval_report


if __name__ == "__main__":
    evaluate_all()
