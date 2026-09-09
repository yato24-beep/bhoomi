"""person-a/src/classification/classifier.py
State-Configuration Driven Rule-Based Land Record Document Classifier.
Evaluates OCR text, mandatory/optional keywords, negative keywords, and layout signals (tables, aspect ratio).
Provides full signal provenance and explainability.
"""

from typing import Dict, List, Optional, Tuple
from ..config.models import DocumentTypeRule, StateConfiguration
from ..schemas import ClassificationResult, OCROutput, OCRPageResult


class DocumentClassifier:
    """Classifies land-record documents using evidence gathered during OCR and layout analysis."""

    def __init__(self, state_config: StateConfiguration):
        self.state_config = state_config

    def classify(
        self,
        full_text: str,
        pages: Optional[List[OCRPageResult]] = None,
    ) -> ClassificationResult:
        """Score all configured document types and return top classification result."""
        if not full_text or len(full_text.strip()) < 5:
            return ClassificationResult(
                predicted_type="unknown",
                confidence=0.0,
                state=self.state_config.state_code,
                explanation="Insufficient OCR text for document classification",
            )

        lower_text = full_text.lower()
        has_table = any(len(p.tables) > 0 for p in (pages or []))
        aspect_ratio = 1.0
        if pages and pages[0].height > 0:
            aspect_ratio = pages[0].width / pages[0].height

        scores: Dict[str, float] = {}
        signals_map: Dict[str, List[str]] = {}

        for rule in self.state_config.document_types:
            score, signals = self._evaluate_rule(rule, lower_text, has_table, aspect_ratio, pages=pages)
            if score > 0:
                scores[rule.document_type] = score
                signals_map[rule.document_type] = signals

        if not scores:
            return ClassificationResult(
                predicted_type="unknown",
                confidence=0.10,
                state=self.state_config.state_code,
                explanation=f"No matching document type rules satisfied for state {self.state_config.state_code}",
            )

        top_type = max(scores, key=lambda k: scores[k])
        top_score = scores[top_type]
        top_signals = signals_map.get(top_type, [])

        return ClassificationResult(
            predicted_type=top_type,
            confidence=round(min(top_score, 1.0), 4),
            state=self.state_config.state_code,
            matched_signals=top_signals,
            all_scores={k: round(v, 4) for k, v in scores.items()},
            explanation=(
                f"Classified as '{top_type}' with confidence {top_score:.2f}. "
                f"Matched {len(top_signals)} evidence signal(s): {', '.join(top_signals)}"
            ),
        )

    def _evaluate_rule(
        self,
        rule: DocumentTypeRule,
        lower_text: str,
        has_table: bool,
        aspect_ratio: float,
        pages: Optional[List[OCRPageResult]] = None,
    ) -> Tuple[float, List[str]]:
        matched_signals: List[str] = []

        # 1. Negative keywords disqualification
        for neg_kw in rule.negative_keywords:
            if neg_kw.lower() in lower_text:
                return 0.0, []

        # 2. Mandatory keywords
        for m_kw in rule.mandatory_keywords:
            if m_kw.lower() in lower_text:
                matched_signals.append(f"mandatory_keyword:{m_kw}")
            else:
                return 0.0, []

        # 3. Optional keywords
        opt_matches = 0
        for opt_kw in rule.optional_keywords:
            if opt_kw.lower() in lower_text:
                matched_signals.append(f"keyword:{opt_kw}")
                opt_matches += 1

        if rule.min_keyword_matches > 0 and (len(rule.mandatory_keywords) + opt_matches) < rule.min_keyword_matches:
            return 0.0, []

        # 4. Layout signals
        if "table_required" in rule.layout_signals:
            if has_table:
                matched_signals.append("layout:table_matched")
            elif pages is not None and len(pages) > 0:
                # If layout was analyzed and explicitly found no tables
                return 0.0, []

        # 5. Aspect ratio
        if rule.aspect_ratio_range and len(rule.aspect_ratio_range) == 2:
            min_ar, max_ar = rule.aspect_ratio_range
            if min_ar <= aspect_ratio <= max_ar:
                matched_signals.append(f"layout:aspect_ratio_{aspect_ratio:.2f}_in_range")

        # Compute confidence score
        base_score = 0.50 + 0.10 * len(matched_signals)
        final_score = base_score * rule.confidence_weight
        return min(final_score, 0.99), matched_signals
