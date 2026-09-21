"""Demo mode service stub.

Production demo mode is not yet implemented.
All functions return defaults that disable demo behaviour.
"""

from typing import Any, Dict, List


# Stub demo fields — these would be pre-populated synthetic field values shown
# when running in demonstration / read-only mode without real document ingestion.
DEMO_FIELDS: List[Dict[str, Any]] = []


def is_demo_mode() -> bool:
    """Returns True if the application is running in demo mode."""
    return False


def apply_demo_results(document_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    """Applies demo-mode synthetic results to an extraction response.
    In production this is a no-op; in demo mode it would inject sample data.
    """
    return results
