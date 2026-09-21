"""Baseline PyTorch TrOCR Checkpoint-12000 Profiler & Benchmark Harness.

Measures:
- Model size on disk
- Process RSS memory: initial, post-load, and true peak during inference (via high-frequency psutil sampling)
- Latency per sample
- Complete autoregressive decoding output
- Kannada Unicode preservation and <unk> token check
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

from PIL import Image
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "trocr" / "checkpoint-12000"
PROCESSOR_DIR = PROJECT_ROOT / "src" / "handwriting" / "configs" / "iitb_kannada_v002"
TOKENIZER_DIR = PROJECT_ROOT / "models" / "trocr" / "experimental" / "iitb_kannada_v002"
BENCHMARK_DIR = PROJECT_ROOT / "locked_benchmark_export"
OUTPUT_DIR = Path(__file__).resolve().parent


class PeakMemoryTracker:
    """Tracks peak process RSS memory using a high-frequency polling thread."""

    def __init__(self, interval_sec: float = 0.01):
        self.interval = interval_sec
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread = None
        self.peak_rss = 0

    def start(self):
        self.peak_rss = self.process.memory_info().rss
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while not self._stop_event.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self.peak_rss:
                    self.peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> int:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        # Final check
        try:
            rss = self.process.memory_info().rss
            if rss > self.peak_rss:
                self.peak_rss = rss
        except Exception:
            pass
        return self.peak_rss


def get_model_disk_size_mb(path: Path) -> float:
    total_bytes = 0
    if path.is_file():
        total_bytes = path.stat().st_size
    elif path.is_dir():
        for p in path.rglob("*"):
            if p.is_file():
                total_bytes += p.stat().st_size
    return total_bytes / (1024 * 1024)


def is_kannada_char(char: str) -> bool:
    return "\u0C80" <= char <= "\u0CFF"


def analyze_kannada_text(text: str) -> Dict[str, Any]:
    total_chars = len(text)
    kannada_chars = sum(1 for c in text if is_kannada_char(c))
    unk_count = text.count("<unk>")
    return {
        "text": text,
        "total_chars": total_chars,
        "kannada_chars": kannada_chars,
        "unk_count": unk_count,
        "has_kannada": kannada_chars > 0,
        "has_unk": unk_count > 0,
    }


def run_baseline_benchmark() -> Dict[str, Any]:
    print("=" * 70)
    print("PHASE 1: BASELINE PYTORCH TrOCR CHECKPOINT-12000 PROFILING")
    print("=" * 70)

    process = psutil.Process(os.getpid())
    gc.collect()
    initial_rss_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Initial Process RSS: {initial_rss_mb:.2f} MB")

    disk_size_mb = get_model_disk_size_mb(CHECKPOINT_DIR)
    print(f"Model Checkpoint Disk Size: {disk_size_mb:.2f} MB")

    import torch
    from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

    tracker = PeakMemoryTracker(interval_sec=0.01)
    tracker.start()

    load_start = time.perf_counter()
    print(f"\nLoading processor from: {PROCESSOR_DIR}", flush=True)
    processor = AutoImageProcessor.from_pretrained(str(PROCESSOR_DIR), local_files_only=True)

    print(f"Loading tokenizer from: {TOKENIZER_DIR}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True)

    print(f"Loading VisionEncoderDecoderModel from: {CHECKPOINT_DIR} on CPU...", flush=True)
    model = VisionEncoderDecoderModel.from_pretrained(str(CHECKPOINT_DIR), local_files_only=True)
    model.eval()
    load_time_sec = time.perf_counter() - load_start

    load_peak_rss_mb = tracker.stop() / (1024 * 1024)
    post_load_rss_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Model loaded in {load_time_sec:.2f}s")
    print(f"Post-Load RSS: {post_load_rss_mb:.2f} MB")
    print(f"Peak RSS during load: {load_peak_rss_mb:.2f} MB (Delta: +{load_peak_rss_mb - initial_rss_mb:.2f} MB)")

    # Find test crops
    test_files = sorted(list(BENCHMARK_DIR.glob("*.png")))
    print(f"\nFound {len(test_files)} locked benchmark test crops in {BENCHMARK_DIR.name}/")

    benchmark_results = []
    overall_peak_tracker = PeakMemoryTracker(interval_sec=0.005)
    overall_peak_tracker.start()

    for idx, img_path in enumerate(test_files, 1):
        crop_tracker = PeakMemoryTracker(interval_sec=0.005)
        crop_tracker.start()

        img = Image.open(img_path).convert("RGB")
        pixel_values = processor(img, return_tensors="pt").pixel_values

        t0 = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=64,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                decoder_start_token_id=model.config.decoder_start_token_id or tokenizer.bos_token_id,
            )
        elapsed = time.perf_counter() - t0
        crop_peak_mb = crop_tracker.stop() / (1024 * 1024)

        raw_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        # Standard cleaning: strip bos/eos/pad
        cleaned_text = raw_text.replace(tokenizer.bos_token or "<s>", "")
        cleaned_text = cleaned_text.replace(tokenizer.eos_token or "</s>", "")
        cleaned_text = cleaned_text.replace(tokenizer.pad_token or "<pad>", "").strip()

        analysis = analyze_kannada_text(cleaned_text)
        analysis.update({
            "filename": img_path.name,
            "latency_sec": round(elapsed, 4),
            "peak_rss_mb": round(crop_peak_mb, 2),
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
        })
        benchmark_results.append(analysis)

        print(
            f"[{idx:2d}/{len(test_files)}] {img_path.name:<18} | "
            f"Latency: {elapsed:5.2f}s | Peak RSS: {crop_peak_mb:6.1f} MB | "
            f"Output: '{cleaned_text[:35]}'",
            flush=True,
        )

    overall_peak_mb = overall_peak_tracker.stop() / (1024 * 1024)
    final_rss_mb = process.memory_info().rss / (1024 * 1024)

    summary = {
        "framework": "PyTorch (CPU)",
        "model_path": str(CHECKPOINT_DIR),
        "disk_size_mb": round(disk_size_mb, 2),
        "initial_rss_mb": round(initial_rss_mb, 2),
        "post_load_rss_mb": round(post_load_rss_mb, 2),
        "peak_during_load_mb": round(load_peak_rss_mb, 2),
        "peak_during_inference_mb": round(overall_peak_mb, 2),
        "final_rss_mb": round(final_rss_mb, 2),
        "avg_latency_sec": round(sum(r["latency_sec"] for r in benchmark_results) / len(benchmark_results), 3),
        "samples_tested": len(benchmark_results),
        "detailed_results": benchmark_results,
    }

    out_file = OUTPUT_DIR / "baseline_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved baseline summary to {out_file}")

    print("=" * 70)
    print(f"SUMMARY (PyTorch CPU):")
    print(f"  Model Size on Disk:      {summary['disk_size_mb']} MB")
    print(f"  Post-Load RSS:           {summary['post_load_rss_mb']} MB")
    print(f"  Peak RSS (Inference):    {summary['peak_during_inference_mb']} MB")
    print(f"  Average Latency:         {summary['avg_latency_sec']}s")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    run_baseline_benchmark()
