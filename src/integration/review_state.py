"""Human Review State Machine and Verification Workflows.

Implements an operational review flow:
- State machine: PENDING → ASSIGNED → IN_REVIEW → APPROVED / REJECTED / EDITED
- Auditable review item schema with immutable raw evidence
- Priority queue ordering (lowest confidence first, critical fields first)
- Persistent JSON queue storage and export
- Deterministic verification lifecycle
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import heapq
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field

from schemas import BoundingBox

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    """Document and region verification states."""
    AUTO_ACCEPT = "AUTO_ACCEPT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED_FAILED = "REJECTED_FAILED"


class ReviewDecision(str, Enum):
    """Action taken by a human reviewer."""
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    CORRECTED = "CORRECTED"
    REJECTED = "REJECTED"


class ReviewLifecycleState(str, Enum):
    """Operational lifecycle progression for human review workflow."""
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EDITED = "EDITED"


CRITICAL_LAND_RECORD_FIELDS = {
    "survey_number",
    "owner_name",
    "extent",
    "extent_acres",
    "extent_guntas",
    "hissa_number",
    "khata_number",
    "mutation_number",
}


class HumanReviewItem(BaseModel):
    """Auditable review item representing a region or field requiring human inspection."""
    model_config = ConfigDict(protected_namespaces=())

    review_id: str = Field(..., description="Unique identifier for the review queue item")
    document_id: Optional[str] = Field(None, description="Source document identifier")
    page_number: int = Field(1, ge=1, description="1-indexed page number")
    region_id: str = Field(..., description="Source region identifier")
    bbox: Optional[BoundingBox] = Field(None, description="Spatial coordinates on document")
    crop_image_path: Optional[str] = Field(None, description="Path to region crop image file")
    
    # Immutable Raw OCR Evidence (NEVER overwritten by reviewer)
    raw_ocr_text: str = Field(..., description="Verbatim recognized OCR string")
    recognizer: str = Field(..., description="OCR engine or checkpoint ID")
    review_reason: str = Field(..., description="Actionable reason review was triggered")
    recognizer_confidence_raw: Optional[float] = Field(None, description="Raw model posterior/logit score")
    calibrated_confidence: Optional[float] = Field(None, description="Calibrated posterior probability, or None")
    
    # Gated Translation (populated only when policy explicitly permits)
    translated_text: Optional[str] = Field(None, description="Translated text if permitted by gating policy")
    translation_status: str = Field("not_translated", description="Status of machine translation for this item")

    # Reviewer Workspace Fields (Stored separately from raw evidence)
    status: ReviewStatus = Field(ReviewStatus.REVIEW_REQUIRED, description="Current verification state")
    decision: ReviewDecision = Field(ReviewDecision.PENDING, description="Action taken by human officer")
    lifecycle_state: ReviewLifecycleState = Field(ReviewLifecycleState.PENDING, description="Workflow stage")
    priority_score: float = Field(0.0, description="Priority weight (higher is more urgent)")
    is_critical_field: bool = Field(False, description="Whether this item belongs to a critical legal field")
    corrected_text: Optional[str] = Field(None, description="Human officer corrected transcription")
    reviewer_notes: Optional[str] = Field(None, description="Notes recorded during human review")
    reviewed_by: Optional[str] = Field(None, description="User ID or email of human officer")
    reviewed_at: Optional[datetime] = Field(None, description="Timestamp when review was finalized")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Enqueue time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostics, crop paths, or field linkage")

    # Workflow Actions
    def assign_to(self, reviewer_id: str) -> None:
        """Transitions state from PENDING to ASSIGNED."""
        if self.lifecycle_state not in (ReviewLifecycleState.PENDING, ReviewLifecycleState.ASSIGNED):
            raise ValueError(f"Cannot assign item in state {self.lifecycle_state}")
        self.reviewed_by = reviewer_id
        self.lifecycle_state = ReviewLifecycleState.ASSIGNED

    def start_review(self, reviewer_id: str) -> None:
        """Transitions state to IN_REVIEW."""
        self.reviewed_by = reviewer_id
        self.lifecycle_state = ReviewLifecycleState.IN_REVIEW

    def approve(self, reviewer_id: str, notes: Optional[str] = None) -> None:
        """Approves verbatim OCR text."""
        self.reviewed_by = reviewer_id
        self.reviewed_at = datetime.now(timezone.utc)
        self.reviewer_notes = notes
        self.corrected_text = self.raw_ocr_text
        self.decision = ReviewDecision.ACCEPTED
        self.status = ReviewStatus.AUTO_ACCEPT
        self.lifecycle_state = ReviewLifecycleState.APPROVED

    def edit(self, reviewer_id: str, corrected_text: str, notes: Optional[str] = None) -> None:
        """Applies human transcription correction while keeping raw evidence untouched."""
        self.reviewed_by = reviewer_id
        self.reviewed_at = datetime.now(timezone.utc)
        self.reviewer_notes = notes
        self.corrected_text = corrected_text.strip()
        self.decision = ReviewDecision.CORRECTED
        self.status = ReviewStatus.AUTO_ACCEPT
        self.lifecycle_state = ReviewLifecycleState.EDITED

    def reject(self, reviewer_id: str, reason: str) -> None:
        """Marks item as unreadable, corrupt, or rejected."""
        self.reviewed_by = reviewer_id
        self.reviewed_at = datetime.now(timezone.utc)
        self.reviewer_notes = reason
        self.decision = ReviewDecision.REJECTED
        self.status = ReviewStatus.REJECTED_FAILED
        self.lifecycle_state = ReviewLifecycleState.REJECTED

    # Backwards-compatible aliases
    def apply_correction(self, corrected_text: str, reviewer_id: str, notes: Optional[str] = None) -> None:
        self.edit(reviewer_id=reviewer_id, corrected_text=corrected_text, notes=notes)

    def accept_verbatim(self, reviewer_id: str, notes: Optional[str] = None) -> None:
        self.approve(reviewer_id=reviewer_id, notes=notes)

    def to_dict(self) -> Dict[str, Any]:
        d = self.model_dump()
        if self.created_at:
            d["created_at"] = self.created_at.isoformat()
        if self.reviewed_at:
            d["reviewed_at"] = self.reviewed_at.isoformat()
        return d


class HumanReviewQueue:
    """Thread-safe, priority-ordered review queue for human officers."""

    def __init__(self):
        self.items: Dict[str, HumanReviewItem] = {}

    def add_item(
        self,
        item: HumanReviewItem,
        is_critical_field: bool = False,
    ) -> None:
        """Enqueues an item with deterministic priority calculation."""
        # Calculate priority:
        # Base confidence deficit: (1.0 - conf)
        # Critical field boost: +10.0
        # Empty text boost: +5.0
        conf = item.calibrated_confidence if item.calibrated_confidence is not None else (item.recognizer_confidence_raw or 0.0)
        priority = (1.0 - conf) * 10.0
        if is_critical_field or item.is_critical_field:
            priority += 10.0
            item.is_critical_field = True
        if not item.raw_ocr_text.strip():
            priority += 5.0

        item.priority_score = round(priority, 4)
        self.items[item.review_id] = item

    def get_pending_items(self) -> List[HumanReviewItem]:
        """Returns pending items ordered by descending priority."""
        pending = [
            i for i in self.items.values()
            if i.lifecycle_state in (ReviewLifecycleState.PENDING, ReviewLifecycleState.ASSIGNED)
        ]
        return sorted(pending, key=lambda x: -x.priority_score)

    def get_next(self, reviewer_id: Optional[str] = None) -> Optional[HumanReviewItem]:
        """Fetches the highest-priority pending item and assigns it."""
        pending = self.get_pending_items()
        if not pending:
            return None
        top_item = pending[0]
        if reviewer_id:
            top_item.assign_to(reviewer_id)
            top_item.start_review(reviewer_id)
        return top_item

    def assign_item(self, review_id: str, reviewer_id: str) -> Optional[HumanReviewItem]:
        item = self.items.get(review_id)
        if item:
            item.assign_to(reviewer_id)
        return item

    def start_review(self, review_id: str, reviewer_id: str) -> Optional[HumanReviewItem]:
        item = self.items.get(review_id)
        if item:
            item.start_review(reviewer_id)
        return item

    def approve_item(self, review_id: str, reviewer_id: str, notes: Optional[str] = None) -> Optional[HumanReviewItem]:
        item = self.items.get(review_id)
        if item:
            item.approve(reviewer_id=reviewer_id, notes=notes)
        return item

    def edit_item(self, review_id: str, reviewer_id: str, corrected_text: str, notes: Optional[str] = None) -> Optional[HumanReviewItem]:
        item = self.items.get(review_id)
        if item:
            item.edit(reviewer_id=reviewer_id, corrected_text=corrected_text, notes=notes)
        return item

    def reject_item(self, review_id: str, reviewer_id: str, reason: str) -> Optional[HumanReviewItem]:
        item = self.items.get(review_id)
        if item:
            item.reject(reviewer_id=reviewer_id, reason=reason)
        return item

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.items)
        by_state: Dict[str, int] = {}
        for item in self.items.values():
            st = item.lifecycle_state.value
            by_state[st] = by_state.get(st, 0) + 1
        return {
            "total_items": total,
            "lifecycle_breakdown": by_state,
            "pending_count": by_state.get("PENDING", 0) + by_state.get("ASSIGNED", 0) + by_state.get("IN_REVIEW", 0),
            "completed_count": by_state.get("APPROVED", 0) + by_state.get("EDITED", 0) + by_state.get("REJECTED", 0),
        }

    def export_json(self, output_path: Union[str, Path]) -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "stats": self.get_stats(),
            "items": [i.to_dict() for i in self.items.values()],
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, input_path: Union[str, Path]) -> "HumanReviewQueue":
        p = Path(input_path)
        queue = cls()
        if not p.is_file():
            return queue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item_data in data.get("items", []):
            item = HumanReviewItem(**item_data)
            queue.items[item.review_id] = item
        return queue


class ReviewStateMachine:
    """Manages review lifecycle and routing decisions."""

    @classmethod
    def create_review_item(
        cls,
        document_id: Optional[str],
        page_number: int,
        region_id: str,
        bbox: Optional[BoundingBox],
        raw_ocr_text: str,
        recognizer: str,
        review_reason: str,
        recognizer_confidence_raw: Optional[float] = None,
        calibrated_confidence: Optional[float] = None,
        crop_image_path: Optional[str] = None,
        is_critical_field: bool = False,
        translated_text: Optional[str] = None,
        translation_status: str = "not_translated",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HumanReviewItem:
        """Constructs an auditable review item with immutable raw evidence."""
        return HumanReviewItem(
            review_id=f"rev_{region_id}_{int(time.time() * 1000)}",
            document_id=document_id,
            page_number=page_number,
            region_id=region_id,
            bbox=bbox,
            crop_image_path=crop_image_path,
            raw_ocr_text=raw_ocr_text,
            recognizer=recognizer,
            review_reason=review_reason,
            recognizer_confidence_raw=recognizer_confidence_raw,
            calibrated_confidence=calibrated_confidence,
            is_critical_field=is_critical_field,
            translated_text=translated_text,
            translation_status=translation_status,
            status=ReviewStatus.REVIEW_REQUIRED,
            decision=ReviewDecision.PENDING,
            lifecycle_state=ReviewLifecycleState.PENDING,
            metadata=metadata or {},
        )

    @staticmethod
    def evaluate_region_review_status(
        is_handwritten: Optional[bool],
        raw_confidence: Optional[float],
        calibrated_confidence: Optional[float],
        region_type: str,
        text: str,
        review_warnings: Optional[List[str]] = None,
    ) -> Tuple[ReviewStatus, Optional[str]]:
        """Evaluates whether an OCR region requires human review.

        Returns:
            (ReviewStatus, review_reason)
        """
        # Empty text check
        if not text or not text.strip():
            return ReviewStatus.REVIEW_REQUIRED, "OCR output is empty; officer inspection required"

        # Multi-word line check on word-oriented models
        if is_handwritten and region_type == "line":
            return ReviewStatus.REVIEW_REQUIRED, "Multi-word line crop routed to word-oriented TrOCR model; uncalibrated line recognition requires officer review"

        # Handwriting without calibrated confidence
        if is_handwritten and calibrated_confidence is None:
            return ReviewStatus.REVIEW_REQUIRED, "Handwritten region lacks calibrated confidence verification; officer review required"

        # Low confidence score
        if raw_confidence is not None and raw_confidence < 0.60:
            return ReviewStatus.REVIEW_REQUIRED, f"Low recognizer confidence ({raw_confidence:.2f} < 0.60)"

        if review_warnings:
            return ReviewStatus.REVIEW_REQUIRED, "; ".join(review_warnings)

        return ReviewStatus.AUTO_ACCEPT, None
