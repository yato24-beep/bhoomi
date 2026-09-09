"""person-a/src/utils/logging.py
Structured logging helper for Person A pipeline stages.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("person_a")


def log_stage_event(
    stage_name: str,
    doc_id: str,
    page_num: Optional[int] = None,
    duration_ms: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a structured log message for a pipeline stage."""
    fields = [f"doc_id={doc_id}", f"stage={stage_name}"]
    if page_num is not None:
        fields.append(f"page={page_num}")
    if duration_ms is not None:
        fields.append(f"duration_ms={duration_ms:.2f}")
    if extra:
        for k, v in extra.items():
            fields.append(f"{k}={v}")
    logger.info(" | ".join(fields))
