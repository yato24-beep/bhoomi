"""Handwriting OCR Evaluation and Error Rate Metrics.

Implements exact, non-fabricated metrics for handwriting transcription:
- Character Error Rate (CER) computed via Unicode code-point Levenshtein distance
- Word Error Rate (WER) computed via whitespace-tokenized Levenshtein distance
- Global and per-language metric aggregations for multilingual evaluation
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple


def levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """Computes standard Wagner-Fischer Levenshtein distance between two sequences.

    Args:
        seq1: First sequence (e.g. reference characters or words).
        seq2: Second sequence (e.g. hypothesis characters or words).

    Returns:
        int: Minimum edit operations (insertions, deletions, substitutions).
    """
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    # Dynamic programming matrix (2 rows optimization)
    dp_prev = list(range(m + 1))
    dp_curr = [0] * (m + 1)

    for i in range(1, n + 1):
        dp_curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp_curr[j] = min(
                dp_prev[j] + 1,        # Deletion
                dp_curr[j - 1] + 1,    # Insertion
                dp_prev[j - 1] + cost, # Substitution
            )
        dp_prev, dp_curr = dp_curr, [0] * (m + 1)

    return dp_prev[m]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Computes Character Error Rate (CER) between a reference and hypothesis text.

    $$\text{CER} = \frac{\text{Levenshtein}(ref, hyp)}{\text{len}(ref)}$$

    Args:
        reference: Ground truth reference text string.
        hypothesis: Predicted hypothesis text string.

    Returns:
        float: Character Error Rate (0.0 for perfect match, >= 1.0 for severe mismatch).
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    edit_dist = levenshtein_distance(ref_chars, hyp_chars)
    return float(edit_dist) / float(len(ref_chars))


def compute_wer(reference: str, hypothesis: str) -> float:
    r"""Computes Word Error Rate (WER) between a reference and hypothesis text.

    $$\text{WER} = \frac{\text{Levenshtein}(ref\_words, hyp\_words)}{\text{len}(ref\_words)}$$

    Args:
        reference: Ground truth reference text string.
        hypothesis: Predicted hypothesis text string.

    Returns:
        float: Word Error Rate (0.0 for perfect match, >= 1.0 for severe mismatch).
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    edit_dist = levenshtein_distance(ref_words, hyp_words)
    return float(edit_dist) / float(len(ref_words))


@dataclass
class LanguageMetrics:
    """Error metrics for a specific language or script subset."""
    language: str
    sample_count: int = 0
    total_ref_chars: int = 0
    total_char_edits: int = 0
    total_ref_words: int = 0
    total_word_edits: int = 0
    cer: float = 0.0
    wer: float = 0.0

    def compute(self) -> None:
        """Calculates CER and WER from aggregated edits."""
        self.cer = (
            float(self.total_char_edits) / float(self.total_ref_chars)
            if self.total_ref_chars > 0
            else (0.0 if self.total_char_edits == 0 else 1.0)
        )
        self.wer = (
            float(self.total_word_edits) / float(self.total_ref_words)
            if self.total_ref_words > 0
            else (0.0 if self.total_word_edits == 0 else 1.0)
        )


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for handwriting recognition."""
    overall_cer: float = 0.0
    overall_wer: float = 0.0
    total_samples: int = 0
    total_reference_characters: int = 0
    total_reference_words: int = 0
    per_language: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes report into JSON-compatible dictionary."""
        return asdict(self)


def evaluate_predictions(
    references: Sequence[str],
    hypotheses: Sequence[str],
    languages: Optional[Sequence[str]] = None,
) -> EvaluationReport:
    """Evaluates a batch of predicted text against ground truth references.

    Args:
        references: Sequence of ground truth reference strings.
        hypotheses: Sequence of predicted text strings.
        languages: Optional sequence of language identifiers matching each sample.

    Returns:
        EvaluationReport: Aggregated global and per-language error rates.

    Raises:
        ValueError: If references and hypotheses lengths mismatch.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Length mismatch: {len(references)} references vs {len(hypotheses)} hypotheses."
        )

    if languages is not None and len(languages) != len(references):
        raise ValueError(
            f"Length mismatch: {len(references)} references vs {len(languages)} language tags."
        )

    total_samples = len(references)
    if total_samples == 0:
        return EvaluationReport()

    total_ref_chars = 0
    total_char_edits = 0
    total_ref_words = 0
    total_word_edits = 0

    lang_buckets: Dict[str, LanguageMetrics] = {}

    for i in range(total_samples):
        ref = references[i]
        hyp = hypotheses[i]
        lang = str(languages[i]).lower().strip() if languages else "all"

        if lang not in lang_buckets:
            lang_buckets[lang] = LanguageMetrics(language=lang)

        bucket = lang_buckets[lang]
        bucket.sample_count += 1

        # Character edits
        ref_c = list(ref)
        hyp_c = list(hyp)
        c_edits = levenshtein_distance(ref_c, hyp_c)
        total_ref_chars += len(ref_c)
        total_char_edits += c_edits
        bucket.total_ref_chars += len(ref_c)
        bucket.total_char_edits += c_edits

        # Word edits
        ref_w = ref.strip().split()
        hyp_w = hyp.strip().split()
        w_edits = levenshtein_distance(ref_w, hyp_w)
        total_ref_words += len(ref_w)
        total_word_edits += w_edits
        bucket.total_ref_words += len(ref_w)
        bucket.total_word_edits += w_edits

    # Compute overall rates
    overall_cer = (
        float(total_char_edits) / float(total_ref_chars)
        if total_ref_chars > 0
        else (0.0 if total_char_edits == 0 else 1.0)
    )
    overall_wer = (
        float(total_word_edits) / float(total_ref_words)
        if total_ref_words > 0
        else (0.0 if total_word_edits == 0 else 1.0)
    )

    per_lang_dict: Dict[str, Dict[str, Any]] = {}
    for lang_name, bucket in lang_buckets.items():
        bucket.compute()
        per_lang_dict[lang_name] = {
            "cer": round(bucket.cer, 4),
            "wer": round(bucket.wer, 4),
            "sample_count": bucket.sample_count,
            "total_characters": bucket.total_ref_chars,
            "total_words": bucket.total_ref_words,
        }

    return EvaluationReport(
        overall_cer=round(overall_cer, 4),
        overall_wer=round(overall_wer, 4),
        total_samples=total_samples,
        total_reference_characters=total_ref_chars,
        total_reference_words=total_ref_words,
        per_language=per_lang_dict,
    )
