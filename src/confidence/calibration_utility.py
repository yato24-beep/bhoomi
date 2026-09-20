"""Confidence Calibration Evaluation and Fitting Utility.

Provides a reproducible, production-grade utility for calibrating OCR recognizer confidence:
- Validates authentic held-out labeled datasets (zero fabrication of confidence data).
- Enforces strict sample requirements (minimum N >= 50 valid samples per engine).
- If dataset is insufficient or labels are missing, strictly keeps calibrated confidence = null (UNCALIBRATED).
- Fits Platt Scaling (logistic regression) or Temperature Scaling.
- Evaluates bin-level calibration: Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
- Serializes fitted calibrators into MultiEngineCalibrator JSON models for runtime consumption.

Data Requirements to Activate Calibration:
Each labeled sample must provide:
1. `engine_name` (str): Identifier of the OCR engine (e.g. "easyocr_kannada", "iitb_kannada_v002", "ppocr_v4_en").
2. `raw_score` or `confidence` (float): Raw posterior/model confidence in range [0.0, 1.0].
3. `label` or `is_correct` (int or bool): Binary indicator (1 for correct transcription, 0 for error).
Minimum sample requirement: >= 50 valid samples per engine.
"""

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from src.confidence.calibration import (
    CalibrationMethod,
    EngineCalibrator,
    MultiEngineCalibrator,
    ReliabilityDiagramData,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_SAMPLES = 50


@dataclass
class DatasetValidationReport:
    """Report on dataset validity and data hygiene."""
    total_records: int = 0
    valid_records: int = 0
    missing_label_count: int = 0
    invalid_score_count: int = 0
    missing_engine_count: int = 0
    per_engine_counts: Dict[str, int] = field(default_factory=dict)
    per_engine_valid_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_calibration_dataset(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], DatasetValidationReport]:
    """Inspects and validates a labeled calibration dataset.

    Filters out:
    - Missing or unparseable labels
    - Confidence values outside [0.0, 1.0] or non-numeric (NaN, None)
    - Missing engine names

    Returns:
        (clean_records, validation_report)
    """
    clean_records: List[Dict[str, Any]] = []
    report = DatasetValidationReport(total_records=len(records))

    for idx, raw_rec in enumerate(records):
        if not isinstance(raw_rec, dict):
            report.missing_engine_count += 1
            continue

        # 1. Engine name
        engine_name = raw_rec.get("engine_name") or raw_rec.get("engine") or raw_rec.get("recognizer")
        if not engine_name or not str(engine_name).strip():
            report.missing_engine_count += 1
            continue
        canon_engine = str(engine_name).lower().strip()
        report.per_engine_counts[canon_engine] = report.per_engine_counts.get(canon_engine, 0) + 1

        # 2. Raw score / confidence
        score_val = raw_rec.get("raw_score", raw_rec.get("confidence", raw_rec.get("score")))
        if score_val is None:
            report.invalid_score_count += 1
            continue
        try:
            score = float(score_val)
            if math.isnan(score) or math.isinf(score) or score < 0.0 or score > 1.0:
                report.invalid_score_count += 1
                continue
        except (ValueError, TypeError):
            report.invalid_score_count += 1
            continue

        # 3. Label / Ground truth match
        raw_label = raw_rec.get("label", raw_rec.get("is_correct", raw_rec.get("correct")))
        if raw_label is None:
            # Check if predicted_text and ground_truth are provided for string comparison
            pred = raw_rec.get("predicted_text", raw_rec.get("pred"))
            gt = raw_rec.get("ground_truth", raw_rec.get("gt"))
            if pred is not None and gt is not None:
                label = 1 if str(pred).strip() == str(gt).strip() else 0
            else:
                report.missing_label_count += 1
                continue
        else:
            if isinstance(raw_label, bool):
                label = 1 if raw_label else 0
            elif isinstance(raw_label, (int, float)):
                label = 1 if int(raw_label) == 1 else 0
            elif str(raw_label).strip().lower() in ("1", "true", "yes", "correct"):
                label = 1
            elif str(raw_label).strip().lower() in ("0", "false", "no", "incorrect"):
                label = 0
            else:
                report.missing_label_count += 1
                continue

        clean_rec = {
            "engine_name": canon_engine,
            "raw_score": score,
            "label": label,
        }
        clean_records.append(clean_rec)
        report.per_engine_valid_counts[canon_engine] = report.per_engine_valid_counts.get(canon_engine, 0) + 1

    report.valid_records = len(clean_records)
    return clean_records, report


def fit_engine_calibration(
    records: Sequence[Dict[str, Any]],
    engine_name: str,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    method: CalibrationMethod = CalibrationMethod.PLATTS_SCALING,
) -> Tuple[Optional[EngineCalibrator], Dict[str, Any]]:
    """Fits an EngineCalibrator if sample requirements are satisfied.

    If sample count is below min_samples, returns (None, metadata) preserving
    calibrated_confidence = None without fabricating values.
    """
    canon = engine_name.lower().strip()
    engine_records = [r for r in records if r["engine_name"] == canon]
    sample_count = len(engine_records)

    if sample_count < min_samples:
        logger.info(
            "Calibration skipped for '%s': %d samples available, %d required.",
            canon, sample_count, min_samples
        )
        return None, {
            "engine_name": canon,
            "status": "SKIPPED_INSUFFICIENT_DATA",
            "sample_count": sample_count,
            "min_required": min_samples,
            "is_calibrated": False,
            "calibrated_confidence": None,
            "reason": f"Insufficient authentic samples ({sample_count} < {min_samples})",
        }

    scores = [r["raw_score"] for r in engine_records]
    labels = [r["label"] for r in engine_records]

    calibrator = EngineCalibrator(engine_name=canon, method=method)
    try:
        metrics = calibrator.fit(scores, labels)
        logger.info(
            "Calibration fitted for '%s': baseline ECE=%.4f, calibrated ECE=%.4f",
            canon, metrics.get("baseline_ece", 0.0), metrics.get("calibrated_ece", 0.0)
        )
        return calibrator, {
            "engine_name": canon,
            "status": "FITTED",
            "sample_count": sample_count,
            "min_required": min_samples,
            "is_calibrated": True,
            "method": method.value,
            "metrics": metrics,
            "parameters": {
                "a": calibrator.a,
                "b": calibrator.b,
                "temperature": calibrator.temperature,
            },
        }
    except Exception as e:
        logger.warning("Fitting error for engine '%s': %s", canon, e)
        return None, {
            "engine_name": canon,
            "status": "ERROR",
            "sample_count": sample_count,
            "is_calibrated": False,
            "calibrated_confidence": None,
            "error": str(e),
        }


def build_multi_engine_calibration(
    records: Sequence[Dict[str, Any]],
    min_samples: int = DEFAULT_MIN_SAMPLES,
    method: CalibrationMethod = CalibrationMethod.PLATTS_SCALING,
) -> Tuple[MultiEngineCalibrator, Dict[str, Any]]:
    """Builds a MultiEngineCalibrator across all engines present in the dataset."""
    clean_records, val_report = validate_calibration_dataset(records)
    multi_cal = MultiEngineCalibrator()
    summary: Dict[str, Any] = {
        "dataset_validation": val_report.to_dict(),
        "engines": {},
        "calibrated_engine_count": 0,
        "uncalibrated_engine_count": 0,
    }

    unique_engines = sorted(val_report.per_engine_valid_counts.keys())
    for engine in unique_engines:
        cal, eng_summary = fit_engine_calibration(
            records=clean_records,
            engine_name=engine,
            min_samples=min_samples,
            method=method,
        )
        summary["engines"][engine] = eng_summary
        if cal is not None and cal.is_fitted:
            multi_cal.calibrators[engine] = cal
            summary["calibrated_engine_count"] += 1
        else:
            summary["uncalibrated_engine_count"] += 1

    return multi_cal, summary


def run_calibration_from_file(
    data_file_path: Union[str, Path],
    output_file_path: Optional[Union[str, Path]] = None,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    method: CalibrationMethod = CalibrationMethod.PLATTS_SCALING,
) -> Tuple[MultiEngineCalibrator, Dict[str, Any]]:
    """Loads a labeled dataset from JSON, fits calibration, and optionally saves the model."""
    p = Path(data_file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Calibration data file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data if isinstance(data, list) else data.get("samples", data.get("records", []))
    multi_cal, summary = build_multi_engine_calibration(
        records=records,
        min_samples=min_samples,
        method=method,
    )

    if output_file_path:
        out_p = Path(output_file_path)
        multi_cal.save(out_p)
        summary["saved_to"] = str(out_p)
        logger.info("Saved MultiEngineCalibrator model to %s", out_p)

    return multi_cal, summary


def main():
    """CLI entrypoint for offline calibration fitting and validation."""
    parser = argparse.ArgumentParser(
        description="Fit and evaluate OCR confidence calibrators from held-out labeled data."
    )
    parser.add_argument("--data", "-d", required=True, help="Path to held-out labeled dataset JSON file")
    parser.add_argument("--output", "-o", default="models/calibration/multi_engine_calibrator.json", help="Path to save calibrator JSON")
    parser.add_argument("--min-samples", "-m", type=int, default=DEFAULT_MIN_SAMPLES, help="Minimum authentic samples required per engine (default: 50)")
    parser.add_argument("--method", choices=["platts_scaling", "temperature_scaling"], default="platts_scaling", help="Calibration method")
    args = parser.parse_args()

    cal_method = CalibrationMethod(args.method)
    multi_cal, summary = run_calibration_from_file(
        data_file_path=args.data,
        output_file_path=args.output,
        min_samples=args.min_samples,
        method=cal_method,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
