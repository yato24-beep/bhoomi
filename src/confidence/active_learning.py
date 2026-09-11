"""
src/confidence/active_learning.py
Active Learning & Continuous Feedback Loop Engine.
Logs human corrections alongside original predictions, compiles versioned DVC dataset batches,
and gates model promotion based on MLflow evaluation criteria.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from sqlalchemy.orm import Session

from schemas import BoundingBox, ExtractedField
from src.database.db_session import SessionLocal
from src.database.repository import DocumentRepository


class ActiveLearningService:
    """
    Manages the human-in-the-loop active learning cycle:
    Prediction -> Human Correction -> Versioned Dataset Batch -> Retraining -> Promotion Gate.
    """

    def __init__(self, data_dir: str = "data/corrections"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def log_human_correction(
        self,
        document_id: str,
        field_name: str,
        original_prediction: str,
        corrected_value: str,
        page_number: int = 1,
        bbox: Optional[BoundingBox] = None,
        model_version: str = "v1.0.0",
        corrected_by: str = "human_reviewer",
        db_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Stores an explicit correction without overwriting the original prediction.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "document_id": document_id,
            "field_name": field_name,
            "original_prediction": original_prediction,
            "corrected_value": corrected_value,
            "page_number": page_number,
            "bbox": bbox.model_dump() if bbox else None,
            "model_version": model_version,
            "timestamp": timestamp,
            "corrected_by": corrected_by,
            "dataset_batch_id": f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        }

        # 1. Append to JSONL audit log
        log_file = self.data_dir / "corrections_log.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 2. Persist to Database if session provided
        if db_session:
            repo = DocumentRepository(db_session)
            repo.save_correction(
                document_id=document_id,
                field_name=field_name,
                original_prediction=original_prediction,
                corrected_value=corrected_value,
                page_number=page_number,
                bbox_json=record["bbox"],
                model_version=model_version,
                dataset_batch_id=record["dataset_batch_id"],
                corrected_by=corrected_by,
            )

        logger.info(f"[ACTIVE_LEARNING] Stored correction for doc '{document_id}' field '{field_name}': '{original_prediction}' -> '{corrected_value}'")
        return record

    def export_dataset_batch(self, batch_id: Optional[str] = None) -> Path:
        """
        Compiles all unpromoted corrections into a versioned training dataset artifact for DVC.
        """
        bid = batch_id or f"active_batch_{int(time.time())}"
        batch_file = self.data_dir / f"{bid}.json"
        
        log_file = self.data_dir / "corrections_log.jsonl"
        records = []
        if log_file.is_file():
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))

        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump({
                "batch_id": bid,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "sample_count": len(records),
                "samples": records,
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"[ACTIVE_LEARNING] Exported dataset batch with {len(records)} samples to {batch_file}")
        return batch_file

    def evaluate_promotion_gate(
        self,
        baseline_cer: float,
        new_model_cer: float,
        baseline_field_acc: float,
        new_model_field_acc: float,
        min_acc_improvement: float = 0.01,
    ) -> Tuple[bool, str]:
        """
        Strict promotion gate: Only promotes new models if CER decreased and Field Accuracy improved.
        Never promotes without verifiable metric improvements.
        """
        cer_improved = new_model_cer < baseline_cer
        acc_improved = (new_model_field_acc - baseline_field_acc) >= min_acc_improvement

        if cer_improved and acc_improved:
            msg = f"PROMOTION APPROVED: CER decreased ({baseline_cer:.4f} -> {new_model_cer:.4f}) and Field Accuracy improved ({baseline_field_acc:.2f}% -> {new_model_field_acc:.2f}%)"
            logger.info(f"[PROMOTION_GATE] {msg}")
            return True, msg
        else:
            msg = f"PROMOTION REJECTED: Insufficient improvement (CER: {baseline_cer:.4f} -> {new_model_cer:.4f}, Acc: {baseline_field_acc:.2f}% -> {new_model_field_acc:.2f}%)"
            logger.warning(f"[PROMOTION_GATE] {msg}")
            return False, msg
