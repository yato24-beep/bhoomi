"""Demo mode service stub.

Production demo mode is not yet implemented.
All functions return defaults that disable demo behaviour.
"""

from typing import Any, Dict, List

DEMO_FIELDS: List[Dict[str, Any]] = []


def is_demo_mode() -> bool:
    """Returns True if the application is running in demo mode."""
    return False


def apply_demo_results(
    document_id: str,
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply demo-mode results; production mode is a no-op."""
    return results
