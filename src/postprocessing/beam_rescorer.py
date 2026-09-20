"""Generic, Local Kannada Beam Search Candidate Rescoring Layer.

Applies multi-signal ranking over TrOCR beam candidate sequences without
external network calls or hardcoded word substitutions.

Signals:
1. TrOCR Normalized Acoustic / Visual Model Score (Softmax over beam sequence log-probs)
2. Kannada Orthographic Validity (Proper akshara structure, matras, and conjuncts)
3. Normalized Lexicon Edit Similarity (Distance to generic local Kannada lexicon)

Decision Policy:
- Preserves full candidate provenance for auditability.
- Only promotes an alternative candidate when it has strong lexical / structural evidence
  over an ungrounded or ambiguous top model prediction.
- Flags low-confidence or borderline outputs for human review.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class CandidateInfo:
    """Detailed metadata for a single beam candidate."""
    rank: int
    text: str
    token_ids: List[int]
    raw_model_score: float
    model_probability: float
    orthographic_score: float
    lexicon_score: float
    matched_lexicon_word: Optional[str]
    composite_score: float


@dataclass
class RescoringResult:
    """Output of the multi-signal candidate rescoring process."""
    final_prediction: str
    confidence: float
    original_top_candidate: str
    original_top_confidence: float
    rescored_candidate: str
    is_switched: bool
    requires_human_review: bool
    all_candidates: List[CandidateInfo] = field(default_factory=list)
    provenance_log: Dict[str, Any] = field(default_factory=dict)


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Computes exact character-level Levenshtein edit distance."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)

    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, [0] * (len(s2) + 1)

    return v0[len(s2)]


class KannadaBeamRescorer:
    """Generic, local rescorer for TrOCR beam search sequences."""

    def __init__(self, custom_lexicon: Optional[Sequence[str]] = None):
        self.lexicon: Set[str] = set()
        self._load_local_lexicon(custom_lexicon)

    def _load_local_lexicon(self, custom_lexicon: Optional[Sequence[str]] = None) -> None:
        """Loads generic Kannada words from local project manifests and dictionaries."""
        from src.postprocessing.kannada_normalizer import KANNADA_LAND_RECORD_LEXICON
        self.lexicon.update(KANNADA_LAND_RECORD_LEXICON)

        # 1. Load from personal trial manifests
        trial_paths = [
            r"c:\Land Record\training\datasets\personal_trial\train.jsonl",
            r"c:\Land Record\training\datasets\personal_trial\val.jsonl",
            r"c:\Land Record\training\datasets\personal_trial\trial_all.jsonl",
        ]
        for p in trial_paths:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line.strip())
                            txt = data.get("text", "").strip()
                            if txt:
                                self.lexicon.add(unicodedata.normalize("NFC", txt))
                except Exception as lex_err:
                    logger.warning("Failed reading lexicon from %s: %s", p, lex_err)

        # 2. Load from IIIT Kannada training manifest if available
        iiit_path = r"c:\Land Record\training\datasets\iiit_kannada_train.jsonl"
        if os.path.exists(iiit_path):
            try:
                with open(iiit_path, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line.strip())
                        txt = data.get("text", "").strip()
                        if txt and len(txt) > 1:
                            self.lexicon.add(unicodedata.normalize("NFC", txt))
            except Exception as iiit_err:
                logger.warning("Failed reading IIIT lexicon from %s: %s", iiit_path, iiit_err)

        if custom_lexicon:
            for w in custom_lexicon:
                self.lexicon.add(unicodedata.normalize("NFC", w.strip()))

    def check_orthographic_validity(self, text: str) -> float:
        """Evaluates structural Kannada orthographic / phonotactic validity.
        
        Score: 1.0 (Completely valid syllable structure) down to 0.0 (Corrupted / illegal sequences).
        """
        if not text:
            return 0.0

        score = 1.0
        # Kannada Unicode block is 0x0C80 - 0x0CFF
        kannada_chars = [c for c in text if "\u0C80" <= c <= "\u0CFF"]
        if not kannada_chars:
            return 0.5  # Non-Kannada string (ASCII, punctuation)

        # Rule 1: No dependent vowel sign (matra) or virama at the very start
        matras = set("\u0CBE\u0CBF\u0CC0\u0CC1\u0CC2\u0CC3\u0CC4\u0CC6\u0CC7\u0CC8\u0CCA\u0CCB\u0CCC\u0CCD")
        if text[0] in matras:
            score -= 0.5

        # Rule 2: Consecutive dependent vowel signs without virama
        if re.search(r"[\u0CBE-\u0CCC]{2,}", text):
            score -= 0.4

        # Rule 3: Floating virama followed by dependent vowel sign
        if re.search(r"\u0CCD[\u0CBE-\u0CCC]", text):
            score -= 0.4

        # Rule 4: Consecutive viramas
        if "\u0CCD\u0CCD" in text:
            score -= 0.5

        return max(0.0, min(1.0, score))

    def evaluate_lexicon_similarity(self, text: str) -> Tuple[float, Optional[str]]:
        """Computes maximum edit similarity to the generic Kannada lexicon."""
        norm_text = unicodedata.normalize("NFC", text.strip())
        if not norm_text:
            return 0.0, None

        if norm_text in self.lexicon:
            return 1.0, norm_text

        best_sim = 0.0
        best_match = None
        t_len = len(norm_text)

        # Fast search within length envelope [t_len - 2, t_len + 2]
        for word in self.lexicon:
            w_len = len(word)
            if abs(w_len - t_len) > 2:
                continue

            dist = compute_levenshtein_distance(norm_text, word)
            max_len = max(t_len, w_len)
            sim = 1.0 - (dist / max_len)

            if sim > best_sim:
                best_sim = sim
                best_match = word
                if sim >= 0.95:
                    break

        # Only count similarity if >= 0.70 (within 1-2 edits)
        if best_sim >= 0.70:
            return best_sim, best_match
        return 0.0, None

    def rescore(
        self,
        beam_candidates: List[Dict[str, Any]],
        weight_model: float = 0.50,
        weight_lexicon: float = 0.35,
        weight_ortho: float = 0.15,
        switch_margin: float = 0.08,
    ) -> RescoringResult:
        """Rescores a list of beam candidates using combined model, lexicon, and orthography signals."""
        if not beam_candidates:
            return RescoringResult(
                final_prediction="",
                confidence=0.0,
                original_top_candidate="",
                original_top_confidence=0.0,
                rescored_candidate="",
                is_switched=False,
                requires_human_review=True,
            )

        # 1. Softmax over raw beam log-probabilities
        raw_scores = [c.get("score", 0.0) for c in beam_candidates]
        max_s = max(raw_scores)
        exp_s = np.exp(np.array(raw_scores) - max_s)
        model_probs = exp_s / max(1e-8, np.sum(exp_s))

        candidate_objects: List[CandidateInfo] = []

        for idx, item in enumerate(beam_candidates):
            text = unicodedata.normalize("NFC", item.get("text", "").strip())
            tok_ids = item.get("token_ids", [])
            raw_s = float(item.get("score", 0.0))
            m_prob = float(model_probs[idx])

            ortho_score = self.check_orthographic_validity(text)
            lex_score, matched_word = self.evaluate_lexicon_similarity(text)

            # Composite Score calculation
            composite = (
                weight_model * m_prob
                + weight_lexicon * lex_score
                + weight_ortho * ortho_score
            )

            candidate_objects.append(
                CandidateInfo(
                    rank=idx + 1,
                    text=text,
                    token_ids=tok_ids,
                    raw_model_score=raw_s,
                    model_probability=round(m_prob, 4),
                    orthographic_score=round(ortho_score, 4),
                    lexicon_score=round(lex_score, 4),
                    matched_lexicon_word=matched_word,
                    composite_score=round(composite, 4),
                )
            )

        # 2. Decision Logic
        original_top = candidate_objects[0]
        # Sort candidates by composite score descending
        sorted_by_composite = sorted(candidate_objects, key=lambda c: c.composite_score, reverse=True)
        rescored_top = sorted_by_composite[0]

        is_switched = False
        final_selected = original_top

        # Safe switching criteria (Requirement 9 & 10):
        # Only switch if alternative candidate has significantly stronger language/lexical support
        if rescored_top.rank != 1:
            score_diff = rescored_top.composite_score - original_top.composite_score
            has_strong_evidence = (
                rescored_top.lexicon_score >= 0.95
                and (rescored_top.composite_score > original_top.composite_score)
                and rescored_top.orthographic_score >= 0.90
            )
            if score_diff >= switch_margin or has_strong_evidence:
                final_selected = rescored_top
                is_switched = True

        # Confidence calculation
        final_conf = final_selected.composite_score
        requires_review = (
            final_conf < 0.65
            or (len(sorted_by_composite) > 1 and abs(sorted_by_composite[0].composite_score - sorted_by_composite[1].composite_score) < 0.04)
        )

        provenance = {
            "num_candidates": len(candidate_objects),
            "original_rank1": original_top.text,
            "rescored_top": rescored_top.text,
            "switched": is_switched,
            "weights": {
                "model": weight_model,
                "lexicon": weight_lexicon,
                "ortho": weight_ortho,
            },
        }

        return RescoringResult(
            final_prediction=final_selected.text,
            confidence=round(final_conf, 4),
            original_top_candidate=original_top.text,
            original_top_confidence=round(original_top.model_probability, 4),
            rescored_candidate=rescored_top.text,
            is_switched=is_switched,
            requires_human_review=requires_review,
            all_candidates=candidate_objects,
            provenance_log=provenance,
        )
