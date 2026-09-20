"""Visualization module for confidence uncertainty, overlays, and OCR diagnostics."""

from src.visualization.uncertainty_overlay import (
    generate_uncertainty_report,
    render_side_by_side_comparison,
    render_uncertainty_overlay,
)

__all__ = [
    "generate_uncertainty_report",
    "render_side_by_side_comparison",
    "render_uncertainty_overlay",
]
