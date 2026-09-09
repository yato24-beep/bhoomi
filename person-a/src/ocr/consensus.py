"""person-a/src/ocr/consensus.py
Multi-candidate OCR Consensus & Disagreement Engine.
Compares OCR predictions across multiple engines / preprocessing paths using normalized Levenshtein distance.
Preserves candidate provenance and penalizes confidence upon conflict without discarding alternatives.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def normalized_levenshtein_distance(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein edit distance in range [0.0, 1.0]."""
    if s1 == s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 1.0

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    max_len = max(len1, len2)
    return dp[len1][len2] / max_len


class OCRConsensusCandidate(BaseModel):
    """A single OCR observation candidate with engine and preprocessing provenance."""
    engine_name: str
    raw_text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    preprocessing_path: str = "standard"


class OCRConsensusResult(BaseModel):
    """Consolidated consensus outcome preserving all candidate readings."""
    selected_text: str
    selected_confidence: float
    disagreement_detected: bool
    disagreement_distance: float = 0.0
    candidates_count: int
    all_candidates: List[OCRConsensusCandidate]
    resolution_method: str = "highest_confidence"


class OCRConsensusEngine:
    """Evaluates multi-pass / multi-engine OCR predictions."""

    def __init__(self, disagreement_threshold: float = 0.20, penalty_factor: float = 0.15):
        self.disagreement_threshold = disagreement_threshold
        self.penalty_factor = penalty_factor

    def evaluate_consensus(self, candidates: List[OCRConsensusCandidate]) -> OCRConsensusResult:
        if not candidates:
            return OCRConsensusResult(
                selected_text="",
                selected_confidence=0.0,
                disagreement_detected=False,
                candidates_count=0,
                all_candidates=[],
            )

        if len(candidates) == 1:
            c = candidates[0]
            return OCRConsensusResult(
                selected_text=c.raw_text,
                selected_confidence=c.confidence,
                disagreement_detected=False,
                disagreement_distance=0.0,
                candidates_count=1,
                all_candidates=candidates,
                resolution_method="single_candidate",
            )

        # Find pair with highest distance
        max_dist = 0.0
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                dist = normalized_levenshtein_distance(candidates[i].raw_text, candidates[j].raw_text)
                if dist > max_dist:
                    max_dist = dist

        disagreement = max_dist >= self.disagreement_threshold

        # Sort by confidence descending
        sorted_candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
        top = sorted_candidates[0]

        final_conf = top.confidence
        if disagreement:
            # Penalize confidence when engines significantly disagree
            final_conf = max(0.05, round(top.confidence * (1.0 - self.penalty_factor * max_dist), 4))

        return OCRConsensusResult(
            selected_text=top.raw_text,
            selected_confidence=final_conf,
            disagreement_detected=disagreement,
            disagreement_distance=round(max_dist, 4),
            candidates_count=len(candidates),
            all_candidates=candidates,
            resolution_method="confidence_weighted_with_penalty" if disagreement else "highest_confidence",
        )
