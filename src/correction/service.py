"""Human Correction & Active Learning Log Management Service.

Provides a non-retraining active learning persistence layer to:
1. Safely record verified human ground-truth corrections with full audit lineage.
2. Prevent malformed, empty, or duplicate entries.
3. Export active learning datasets into training-pipeline-compatible JSONL manifests.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import uuid

from src.correction.schemas import HumanCorrectionRecord

DEFAULT_CORRECTIONS_PATH = Path("data/corrections/human_corrections.jsonl")


class CorrectionService:
    """Manages recording, validation, and dataset manifest export of human corrections."""

    def __init__(self, storage_path: Union[str, Path] = DEFAULT_CORRECTIONS_PATH):
        """Initializes the correction service with a persistent JSONL file path."""
        self.storage_path = Path(storage_path)
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """Ensures the parent directory for correction logs exists."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def record_correction(
        self,
        record: Union[HumanCorrectionRecord, Dict[str, Any]],
    ) -> HumanCorrectionRecord:
        """Validates and persists a human correction record into the audit log.

        Args:
            record: A HumanCorrectionRecord instance or dictionary of correction fields.

        Returns:
            The validated HumanCorrectionRecord.

        Raises:
            ValueError: If correction data is invalid or empty.
        """
        if isinstance(record, dict):
            # Auto-assign UUID if correction_id is missing
            rec_dict = dict(record)
            if "correction_id" not in rec_dict or not rec_dict["correction_id"]:
                rec_dict["correction_id"] = f"corr_{uuid.uuid4().hex[:12]}"
            correction_obj = HumanCorrectionRecord(**rec_dict)
        elif isinstance(record, HumanCorrectionRecord):
            correction_obj = record
        else:
            raise TypeError(f"Expected HumanCorrectionRecord or dict, got {type(record).__name__}")

        self._ensure_storage_dir()

        # Append to JSONL storage
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(correction_obj.model_dump(), ensure_ascii=False) + "\n")

        return correction_obj

    def list_corrections(
        self,
        document_id: Optional[str] = None,
        language: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[HumanCorrectionRecord]:
        """Loads and filters persisted human correction records."""
        if not self.storage_path.exists():
            return []

        results: List[HumanCorrectionRecord] = []
        with open(self.storage_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rec = HumanCorrectionRecord(**data)
                    if document_id and rec.document_id != document_id:
                        continue
                    if language and rec.language.lower() != language.lower():
                        continue
                    results.append(rec)
                    if limit and len(results) >= limit:
                        break
                except Exception:
                    continue

        return results

    def export_training_manifest(
        self,
        output_path: Union[str, Path],
        language: Optional[str] = None,
        is_handwritten_only: bool = True,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
    ) -> int:
        """Exports recorded human corrections into a standard JSONL training manifest.

        Matches the schema:
        {"image_path": "...", "text": "...", "language": "...", "script": "...", "is_handwritten": true}

        Note: This does NOT initiate model retraining; it only generates dataset artifacts.
        """
        corrections = self.list_corrections(language=language)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        exported_count = 0
        with open(out_p, "w", encoding="utf-8") as f:
            for rec in corrections:
                if is_handwritten_only and not rec.is_handwritten:
                    continue
                if min_confidence is not None and rec.ai_confidence is not None and rec.ai_confidence < min_confidence:
                    continue
                if max_confidence is not None and rec.ai_confidence is not None and rec.ai_confidence > max_confidence:
                    continue

                manifest_entry = {
                    "image_path": rec.image_path,
                    "text": rec.corrected_text,
                    "language": rec.language,
                    "script": rec.script,
                    "is_handwritten": rec.is_handwritten,
                    "metadata": {
                        "source": "human_active_learning",
                        "correction_id": rec.correction_id,
                        "document_id": rec.document_id,
                        "region_id": rec.region_id,
                        "original_raw_prediction": rec.raw_prediction,
                        "original_ai_confidence": rec.ai_confidence,
                        "model_version": rec.model_version,
                        "timestamp": rec.timestamp,
                    },
                }
                f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
                exported_count += 1

        return exported_count
