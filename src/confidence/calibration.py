"""Confidence Calibration System with Platt and Temperature Scaling.

Implements honest confidence calibration for OCR engines (EasyOCR and IITB TrOCR):
- Platt's scaling (logistic regression on raw posteriors)
- Temperature scaling on logit representations
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) binning
- Reliability diagram generation
- Multi-engine isolation: separate calibrator per engine
- Strict fallback to UNCALIBRATED when no calibrator is fitted
- Zero fabrication of confidence values
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class CalibrationMethod(str, Enum):
    PLATTS_SCALING = "platts_scaling"
    TEMPERATURE_SCALING = "temperature_scaling"


@dataclass
class ReliabilityDiagramData:
    """Binned empirical accuracy vs predicted confidence data for reliability diagrams."""
    bin_edges: List[float]
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    total_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EngineCalibrator:
    """Individual calibrator fitted to a specific OCR recognizer engine."""

    def __init__(
        self,
        engine_name: str,
        method: CalibrationMethod = CalibrationMethod.PLATTS_SCALING,
        num_bins: int = 10,
    ):
        self.engine_name = engine_name
        self.method = method
        self.num_bins = num_bins
        self.is_fitted = False
        # Platt parameters: sigmoid(a * score + b)
        self.a: float = 1.0
        self.b: float = 0.0
        # Temperature parameter: sigmoid(logit / T)
        self.temperature: float = 1.0
        self.calibration_metrics: Dict[str, Any] = {}
        self.reliability_diagram: Optional[ReliabilityDiagramData] = None

    def fit(
        self,
        raw_scores: Sequence[float],
        labels: Sequence[int],  # 1 for correct, 0 for incorrect
    ) -> Dict[str, Any]:
        """Fits calibration parameters using gradient-free or logistic regression.

        Args:
            raw_scores: Sequence of raw recognizer confidence scores in [0.0, 1.0]
            labels: 1 if prediction matches ground truth, 0 otherwise

        Returns:
            Dictionary of calibration metrics including ECE before and after.
        """
        scores = np.asarray(raw_scores, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.float64)

        if len(scores) < 50:
            raise ValueError(
                f"Insufficient samples ({len(scores)} < 50) to reliably fit calibrator for engine {self.engine_name}. "
                "Confidence calibration requires at least 50 authentic held-out samples."
            )
        if len(scores) != len(targets):
            raise ValueError("Scores and targets must have identical length.")

        # Clip scores to avoid numerical overflow in logits
        eps = 1e-6
        clipped_scores = np.clip(scores, eps, 1.0 - eps)

        # Baseline uncalibrated ECE
        base_reliability = self.compute_reliability_diagram(clipped_scores, targets, num_bins=self.num_bins)
        base_ece = base_reliability.ece

        if self.method == CalibrationMethod.PLATTS_SCALING:
            # Fit sigmoid(a * score + b) via mini Nelder-Mead or closed-form / Scipy if available
            self._fit_platts(clipped_scores, targets)
        elif self.method == CalibrationMethod.TEMPERATURE_SCALING:
            self._fit_temperature(clipped_scores, targets)

        self.is_fitted = True

        # Post-calibration evaluation
        calibrated_scores = np.array([self.predict(s) for s in scores])
        cal_reliability = self.compute_reliability_diagram(calibrated_scores, targets, num_bins=self.num_bins)
        self.reliability_diagram = cal_reliability

        self.calibration_metrics = {
            "engine_name": self.engine_name,
            "method": self.method.value,
            "sample_count": len(scores),
            "uncalibrated_ece": float(base_ece),
            "calibrated_ece": float(cal_reliability.ece),
            "calibrated_mce": float(cal_reliability.mce),
            "empirical_accuracy": float(np.mean(targets)),
            "parameters": (
                {"a": self.a, "b": self.b}
                if self.method == CalibrationMethod.PLATTS_SCALING
                else {"temperature": self.temperature}
            ),
        }
        logger.info(
            "Engine %s calibrated with %s. ECE reduced from %.4f to %.4f",
            self.engine_name,
            self.method.value,
            base_ece,
            cal_reliability.ece,
        )
        return self.calibration_metrics

    def _fit_platts(self, scores: np.ndarray, targets: np.ndarray) -> None:
        """Fits Platt scaling parameters a and b using logistic log-likelihood."""
        try:
            from scipy.optimize import minimize
            def loss(params):
                a_val, b_val = params
                p = 1.0 / (1.0 + np.exp(-(a_val * scores + b_val)))
                eps = 1e-9
                p = np.clip(p, eps, 1.0 - eps)
                return -np.sum(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p))

            res = minimize(loss, [1.0, 0.0], method="L-BFGS-B")
            self.a = float(res.x[0])
            self.b = float(res.x[1])
        except Exception:
            # Fallback simple iterative optimization
            a_cur, b_cur = 1.0, 0.0
            lr = 0.05
            for _ in range(200):
                p = 1.0 / (1.0 + np.exp(-(a_cur * scores + b_cur)))
                diff = p - targets
                grad_a = np.mean(diff * scores)
                grad_b = np.mean(diff)
                a_cur -= lr * grad_a
                b_cur -= lr * grad_b
            self.a = float(a_cur)
            self.b = float(b_cur)

    def _fit_temperature(self, scores: np.ndarray, targets: np.ndarray) -> None:
        """Fits temperature T > 0 on logit transformed probabilities."""
        logits = np.log(scores / (1.0 - scores))
        try:
            from scipy.optimize import minimize_scalar
            def loss(t_val):
                if t_val <= 0.01:
                    return 1e9
                p = 1.0 / (1.0 + np.exp(-logits / t_val))
                eps = 1e-9
                p = np.clip(p, eps, 1.0 - eps)
                return -np.sum(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p))

            res = minimize_scalar(loss, bounds=(0.05, 10.0), method="bounded")
            self.temperature = float(res.x)
        except Exception:
            # Simple grid search
            best_t = 1.0
            best_loss = float("inf")
            for t_val in np.linspace(0.1, 5.0, 50):
                p = 1.0 / (1.0 + np.exp(-logits / t_val))
                eps = 1e-9
                p = np.clip(p, eps, 1.0 - eps)
                l = -np.sum(targets * np.log(p) + (1.0 - targets) * np.log(1.0 - p))
                if l < best_loss:
                    best_loss = l
                    best_t = t_val
            self.temperature = float(best_t)

    def predict(self, raw_score: Optional[float]) -> Optional[float]:
        """Transforms raw score to calibrated posterior probability.

        Returns None if calibrator is not fitted or score is None.
        """
        if not self.is_fitted or raw_score is None:
            return None

        s = float(raw_score)
        if self.method == CalibrationMethod.PLATTS_SCALING:
            val = 1.0 / (1.0 + math.exp(-(self.a * s + self.b)))
        elif self.method == CalibrationMethod.TEMPERATURE_SCALING:
            eps = 1e-6
            s_clamped = max(eps, min(1.0 - eps, s))
            logit = math.log(s_clamped / (1.0 - s_clamped))
            val = 1.0 / (1.0 + math.exp(-logit / max(0.01, self.temperature)))
        else:
            val = s

        return round(float(np.clip(val, 0.0, 1.0)), 4)

    @staticmethod
    def compute_reliability_diagram(
        confs: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 10,
    ) -> ReliabilityDiagramData:
        """Computes bin metrics, ECE, and MCE."""
        bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        ece = 0.0
        mce = 0.0
        total_n = len(confs)

        for i in range(num_bins):
            b_low = bin_edges[i]
            b_high = bin_edges[i + 1]
            if i == num_bins - 1:
                mask = (confs >= b_low) & (confs <= b_high)
            else:
                mask = (confs >= b_low) & (confs < b_high)

            count = int(np.sum(mask))
            bin_counts.append(count)
            if count > 0:
                acc = float(np.mean(labels[mask]))
                conf = float(np.mean(confs[mask]))
                bin_accuracies.append(round(acc, 4))
                bin_confidences.append(round(conf, 4))
                err = abs(acc - conf)
                ece += (count / total_n) * err
                mce = max(mce, err)
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(round((b_low + b_high) / 2.0, 4))

        return ReliabilityDiagramData(
            bin_edges=[round(float(x), 2) for x in bin_edges],
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
            ece=round(float(ece), 4),
            mce=round(float(mce), 4),
            total_samples=total_n,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "method": self.method.value,
            "num_bins": self.num_bins,
            "is_fitted": self.is_fitted,
            "a": self.a,
            "b": self.b,
            "temperature": self.temperature,
            "calibration_metrics": self.calibration_metrics,
            "reliability_diagram": self.reliability_diagram.to_dict() if self.reliability_diagram else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineCalibrator":
        cal = cls(
            engine_name=data["engine_name"],
            method=CalibrationMethod(data.get("method", "platts_scaling")),
            num_bins=data.get("num_bins", 10),
        )
        cal.is_fitted = data.get("is_fitted", False)
        cal.a = data.get("a", 1.0)
        cal.b = data.get("b", 0.0)
        cal.temperature = data.get("temperature", 1.0)
        cal.calibration_metrics = data.get("calibration_metrics", {})
        if data.get("reliability_diagram"):
            cal.reliability_diagram = ReliabilityDiagramData(**data["reliability_diagram"])
        return cal


class MultiEngineCalibrator:
    """Manages independent calibrators for all integrated OCR recognition engines."""

    def __init__(self):
        self.calibrators: Dict[str, EngineCalibrator] = {}

    def get_or_create(
        self,
        engine_name: str,
        method: CalibrationMethod = CalibrationMethod.PLATTS_SCALING,
    ) -> EngineCalibrator:
        """Retrieves or registers an engine calibrator."""
        canon = engine_name.lower().strip()
        if canon not in self.calibrators:
            self.calibrators[canon] = EngineCalibrator(engine_name=canon, method=method)
        return self.calibrators[canon]

    def calibrate(
        self,
        engine_name: Optional[str],
        raw_score: Optional[float],
    ) -> Tuple[Optional[float], str]:
        """Calibrates a raw engine score.

        Returns:
            Tuple of (calibrated_score, confidence_state_string)
            where confidence_state is 'CALIBRATED' if fitted, else 'UNCALIBRATED'.
        """
        if not engine_name or raw_score is None:
            return None, "UNCALIBRATED"

        canon = engine_name.lower().strip()
        cal = self.calibrators.get(canon)
        if cal is None or not cal.is_fitted:
            return None, "UNCALIBRATED"

        calibrated = cal.predict(raw_score)
        return calibrated, "CALIBRATED"

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves all engine calibrators to JSON."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self.calibrators.items()}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "MultiEngineCalibrator":
        """Loads engine calibrators from JSON."""
        p = Path(file_path)
        if not p.is_file():
            logger.warning("Calibration file %s not found. Initializing uncalibrated interface.", p)
            return cls()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        mec = cls()
        for k, v in data.items():
            mec.calibrators[k] = EngineCalibrator.from_dict(v)
        return mec
