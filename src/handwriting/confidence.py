"""Confidence calculation and audit utilities for handwriting recognition.

Provides transparent, non-fabricated confidence evaluation based strictly on
model-derived token probabilities, logits, or explicit absence thereof.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


def compute_token_mean_confidence(token_probabilities: Optional[List[float]]) -> Optional[float]:
    """Computes arithmetic mean over token-level probabilities.

    Args:
        token_probabilities: List of token probabilities in [0.0, 1.0].

    Returns:
        Mean probability rounded to 4 decimals, or None if probabilities are
        missing, empty, or invalid.
    """
    if not token_probabilities:
        return None
    
    valid_probs = [p for p in token_probabilities if isinstance(p, (int, float)) and 0.0 <= p <= 1.0]
    if not valid_probs or len(valid_probs) != len(token_probabilities):
        return None
        
    return round(sum(valid_probs) / len(valid_probs), 4)


def compute_token_geometric_mean_confidence(token_probabilities: Optional[List[float]]) -> Optional[float]:
    """Computes geometric mean over token probabilities using log-space summation.

    This penalizes tokens with very low probabilities more heavily than arithmetic mean.

    Args:
        token_probabilities: List of token probabilities in (0.0, 1.0].

    Returns:
        Geometric mean score in [0.0, 1.0], or None if unavailable or contains <= 0 values.
    """
    if not token_probabilities:
        return None

    # Filter/validate strictly positive probabilities
    for p in token_probabilities:
        if not isinstance(p, (int, float)) or p <= 0.0 or p > 1.0:
            return None

    log_sum = sum(math.log(p) for p in token_probabilities)
    geo_mean = math.exp(log_sum / len(token_probabilities))
    return round(geo_mean, 4)


def compute_min_token_confidence(token_probabilities: Optional[List[float]]) -> Optional[float]:
    """Returns the minimum confidence score among generated tokens.

    Useful as a worst-case risk metric for sensitive fields.

    Args:
        token_probabilities: List of token probabilities.

    Returns:
        Minimum token confidence or None if empty.
    """
    if not token_probabilities:
        return None
    valid_probs = [p for p in token_probabilities if isinstance(p, (int, float)) and 0.0 <= p <= 1.0]
    if not valid_probs or len(valid_probs) != len(token_probabilities):
        return None
    return round(min(valid_probs), 4)


def classify_confidence_tier(
    confidence: Optional[float],
    high_threshold: float = 0.85,
    low_threshold: float = 0.60,
) -> str:
    """Classifies a numeric confidence score into standard validation tiers.

    Args:
        confidence: Normalized score in [0.0, 1.0], or None if unavailable.
        high_threshold: Threshold above which confidence is considered 'HIGH'.
        low_threshold: Threshold below which confidence is considered 'LOW'.

    Returns:
        One of 'HIGH', 'MEDIUM', 'LOW', or 'UNKNOWN'.
    """
    if confidence is None or not (0.0 <= confidence <= 1.0):
        return "UNKNOWN"
    if confidence >= high_threshold:
        return "HIGH"
    if confidence >= low_threshold:
        return "MEDIUM"
    return "LOW"


def build_confidence_audit_trail(
    raw_token_probabilities: Optional[List[float]] = None,
    calculation_method: str = "token_mean",
    custom_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Constructs a structured audit dictionary capturing the exact confidence derivation.

    Args:
        raw_token_probabilities: Raw token probabilities from model logits if available.
        calculation_method: Identifier of method used (e.g. 'token_mean', 'geometric_mean', 'direct').
        custom_metadata: Optional auxiliary info (e.g. temperature, beam_size).

    Returns:
        Structured audit dictionary preserving evidence without inventing data.
    """
    audit: Dict[str, Any] = {
        "calculation_method": calculation_method,
        "token_count": len(raw_token_probabilities) if raw_token_probabilities is not None else None,
        "has_token_probabilities": bool(raw_token_probabilities),
    }

    if raw_token_probabilities:
        audit["raw_token_probabilities"] = raw_token_probabilities
        audit["min_token_confidence"] = compute_min_token_confidence(raw_token_probabilities)
        audit["mean_token_confidence"] = compute_token_mean_confidence(raw_token_probabilities)

    if custom_metadata:
        audit["metadata"] = custom_metadata
        for k, v in custom_metadata.items():
            if k not in audit:
                audit[k] = v

    return audit
