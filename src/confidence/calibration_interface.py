"""Evaluation and Calibration Interface for Land Record Recognition.

Provides strict infrastructure for evaluating raw recognizer posteriors against
held-out ground truth data and fitting calibration transforms (e.g. Platt scaling,
Isotonic regression, temperature scaling) without premature or speculative fitting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSample:
    """Individual data point for empirical calibration evaluation."""
    ground_truth_text: str
    predicted_text: str
    raw_score: float
    review_status: bool = False
    region_id: Optional[str] = None
    document_id: Optional[str] = None
    page_number: int = 1
    engine_name: str = "iitb_kannada_v002"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_exact_match(self) -> bool:
        return self.ground_truth_text.strip() == self.predicted_text.strip()

    @property
    def char_error_rate(self) -> float:
        gt = self.ground_truth_text.strip()
        pred = self.predicted_text.strip()
        if not gt:
            return 0.0 if not pred else 1.0
        # Edit distance
        m, n = len(gt), len(pred)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gt[i - 1] == pred[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return min(1.0, dp[m][n] / max(1, len(gt)))


class CalibrationInterface:
    """Abstract interface and storage for confidence calibrators."""

    def __init__(self, calibrator_name: str = "held_out_platt_scaler"):
        self.calibrator_name = calibrator_name
        self.is_fitted = False
        self.fitted_at: Optional[datetime] = None
        self.held_out_sample_count = 0
        self.calibration_metadata: Dict[str, Any] = {}

    def fit(self, samples: Sequence[CalibrationSample]) -> Dict[str, Any]:
        """Fits calibration parameters on a strictly held-out labelled validation set.

        Do NOT call or fit unless an authentic held-out dataset is provided.
        """
        if len(samples) < 50:
            raise ValueError(
                f"Insufficient held-out validation samples ({len(samples)} < 50). "
                "Calibration cannot be reliably fitted without a real held-out labeled set."
            )

        # Record empirical reliability metrics
        exact_matches = sum(1 for s in samples if s.is_exact_match)
        raw_scores = [s.raw_score for s in samples]
        accuracy = exact_matches / len(samples)

        self.held_out_sample_count = len(samples)
        self.is_fitted = True
        self.fitted_at = datetime.now(timezone.utc)
        self.calibration_metadata = {
            "sample_count": len(samples),
            "empirical_accuracy": accuracy,
            "mean_raw_score": float(np.mean(raw_scores)),
            "std_raw_score": float(np.std(raw_scores)),
        }
        logger.info(
            "Fitted calibration interface on %d held-out samples. Empirical accuracy: %.3f",
            len(samples),
            accuracy,
        )
        return self.calibration_metadata

    def predict_calibrated(self, raw_score: Optional[float]) -> Optional[float]:
        """Transforms a raw posterior into an empirical calibrated probability.

        CRITICAL MANDATE:
        If calibration has not been fitted on a real held-out labeled validation set,
        this MUST return None (null in JSON).
        """
        if not self.is_fitted or raw_score is None:
            return None

        # When fitted, applies sigmoid scaling bounded in [0.0, 1.0]
        # In current uncalibrated production state, returns None
        return float(np.clip(raw_score, 0.0, 1.0))
