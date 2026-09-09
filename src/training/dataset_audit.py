"""Dataset Readiness Audit Utilities for Land Record Digitization.

Scans manifests and physical directories to generate an honest, comprehensive audit:
- Inventory of physical images and manifest records.
- Validation of broken vs. valid paths.
- Categorization into real handwriting, printed text, and synthetic data via metadata.
- Honest assessment of dataset sufficiency for smoke-testing, small fine-tuning, and production training.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Ensure UTF-8 output encoding for terminal display
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.training.data_preparation import validate_manifest


@dataclass
class DatasetSufficiency:
    """Sufficiency evaluation for a specific training stage."""
    target_stage: str
    is_sufficient: bool
    current_count: int
    minimum_required: int
    rationale: str


@dataclass
class DatasetAuditReport:
    """Comprehensive readiness audit of local datasets and manifests."""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    scanned_directories: List[str] = field(default_factory=list)
    total_physical_images: int = 0
    physical_images: List[str] = field(default_factory=list)
    total_manifest_records: int = 0
    valid_manifest_records: int = 0
    missing_manifest_records: int = 0
    samples_by_language: Dict[str, int] = field(default_factory=dict)
    samples_by_type: Dict[str, int] = field(default_factory=dict)  # "real_handwriting", "printed", "synthetic", "unknown"
    manifest_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sufficiency: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Renders the audit report as GitHub-flavored markdown."""
        lines = [
            "# Dataset Readiness Audit Report",
            f"**Timestamp:** `{self.timestamp}`",
            "",
            "## 1. Physical Image Inventory",
            f"- **Total Image Files Found:** {self.total_physical_images}",
        ]
        for img in self.physical_images:
            lines.append(f"  - `{img}`")

        lines.extend([
            "",
            "## 2. Manifest Inventory & Language Breakdown",
            f"- **Total Manifest Records:** {self.total_manifest_records}",
            f"- **Valid Accessible Records:** {self.valid_manifest_records}",
            f"- **Broken / Missing Records:** {self.missing_manifest_records}",
            "",
            "### Usable Samples by Language:",
        ])
        for lang, count in sorted(self.samples_by_language.items()):
            lines.append(f"- **{lang.capitalize()}:** {count} samples")

        lines.extend([
            "",
            "### Usable Samples by Text Type:",
            f"- **Real Handwriting:** {self.samples_by_type.get('real_handwriting', 0)}",
            f"- **Printed Text:** {self.samples_by_type.get('printed', 0)}",
            f"- **Synthetic Handwriting:** {self.samples_by_type.get('synthetic', 0)}",
            f"- **Unknown / Unspecified:** {self.samples_by_type.get('unknown', 0)}",
            "",
            "## 3. Dataset Sufficiency Evaluation",
        ])
        for stage_name, suff in self.sufficiency.items():
            status_tag = "✅ SUFFICIENT" if suff.get("is_sufficient") else "❌ INSUFFICIENT"
            lines.append(f"### {stage_name.replace('_', ' ').title()} — {status_tag}")
            lines.append(f"- **Current Samples:** {suff.get('current_count')} (Min Required: {suff.get('minimum_required')})")
            lines.append(f"- **Rationale:** {suff.get('rationale')}")
            lines.append("")

        lines.extend([
            "## 4. Key Recommendations",
        ])
        for rec in self.recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)


def classify_sample_text_type(metadata: Dict[str, Any], path_str: str) -> str:
    """Infers whether a sample represents real handwriting, printed text, synthetic data, or unknown."""
    meta_type = str(metadata.get("source_type") or metadata.get("text_type", "")).lower()
    if meta_type in ("handwriting", "handwritten", "real_handwriting"):
        return "real_handwriting"
    if meta_type in ("printed", "print"):
        return "printed"
    if meta_type in ("synthetic", "synthetic_handwriting"):
        return "synthetic"
    if meta_type in ("unknown",):
        return "unknown"

    # Infer from filename or path conventions
    p_lower = path_str.lower()
    if "synthetic" in p_lower:
        return "synthetic"
    if "handwritten" in p_lower or "cursive" in p_lower:
        return "synthetic" if "sample" in p_lower else "real_handwriting"
    if "kannada" in p_lower or "print" in p_lower or "document" in p_lower:
        return "printed"

    return "unknown"


def run_dataset_audit(
    root_dir: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    manifest_dir: Optional[Union[str, Path]] = None,
) -> DatasetAuditReport:
    """Executes a full dataset readiness audit over project directories and manifests.

    Args:
        root_dir: Project root directory. Defaults to current working directory or Land Record.
        data_dir: Directory storing image files (defaults to root_dir / 'data').
        manifest_dir: Directory storing manifests (defaults to root_dir / 'training' / 'datasets').

    Returns:
        DatasetAuditReport: Structured audit report.
    """
    base_root = Path(root_dir) if root_dir else Path.cwd()
    data_path = Path(data_dir) if data_dir else base_root / "data"
    manifest_path = Path(manifest_dir) if manifest_dir else base_root / "training" / "datasets"

    scanned_dirs = [
        str(data_path.relative_to(base_root)) if data_path.is_relative_to(base_root) else str(data_path),
        str(manifest_path.relative_to(base_root)) if manifest_path.is_relative_to(base_root) else str(manifest_path),
    ]

    # 1. Scan physical images
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    found_images: List[str] = []

    if data_path.exists():
        for root, _, files in os.walk(data_path):
            for file in sorted(files):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    full_p = Path(root) / file
                    rel_p = full_p.relative_to(base_root) if full_p.is_relative_to(base_root) else full_p
                    found_images.append(str(rel_p).replace("\\", "/"))

    # 2. Scan and validate JSONL manifests
    manifest_summaries: Dict[str, Dict[str, Any]] = {}
    total_records = 0
    valid_records = 0
    missing_records = 0
    samples_by_lang: Dict[str, int] = {}
    samples_by_type: Dict[str, int] = {
        "real_handwriting": 0,
        "printed": 0,
        "synthetic": 0,
        "unknown": 0,
    }

    if manifest_path.exists():
        for manifest_file in sorted(manifest_path.glob("*.jsonl")):
            rel_man = str(manifest_file.relative_to(base_root)).replace("\\", "/")
            valid_samples, val_report = validate_manifest(
                manifest_path=manifest_file,
                root_dir=base_root,
                verify_images=True,
            )

            manifest_summaries[rel_man] = {
                "total_records": val_report.total_records,
                "valid_records": val_report.valid_count,
                "error_records": val_report.error_count,
                "warning_records": val_report.warning_count,
                "duplicate_records": val_report.duplicate_count,
                "is_valid": val_report.is_valid,
            }

            total_records += val_report.total_records
            valid_records += val_report.valid_count
            missing_records += val_report.error_count

            for s in valid_samples:
                samples_by_lang[s.language] = samples_by_lang.get(s.language, 0) + 1
                t_type = classify_sample_text_type(s.metadata, str(s.image_path))
                samples_by_type[t_type] = samples_by_type.get(t_type, 0) + 1

    # 3. Evaluate Sufficiency
    kannada_count = samples_by_lang.get("kannada", 0)
    real_hw_count = samples_by_type.get("real_handwriting", 0)

    sufficiency = {
        "smoke_testing": {
            "target_stage": "Pipeline Dry-Runs & Smoke Testing",
            "is_sufficient": valid_records >= 1,
            "current_count": valid_records,
            "minimum_required": 1,
            "rationale": "Sufficient to validate batch collation, optimizer loops, metric computation, and checkpoint serialization.",
        },
        "small_fine_tuning": {
            "target_stage": "Small Kannada Fine-Tuning",
            "is_sufficient": kannada_count >= 50 and real_hw_count >= 20,
            "current_count": kannada_count,
            "minimum_required": 50,
            "rationale": "Insufficient. Neural VisionEncoderDecoder models require at least 50-200 distinct line crops with varying handwriting strokes to learn vocabulary without collapsing.",
        },
        "production_training": {
            "target_stage": "Production Multilingual Training",
            "is_sufficient": valid_records >= 1000 and real_hw_count >= 500,
            "current_count": valid_records,
            "minimum_required": 1000,
            "rationale": "Insufficient. Production-grade Indic handwriting recognition requires thousands of diverse samples spanning multiple scribes, degraded paper, and complex conjuncts.",
        },
    }

    # 4. Generate Recommendations
    recommendations = []
    if kannada_count < 50:
        recommendations.append(
            "Place 50–200 authentic Kannada land record line/word crops into `data/raw/kannada/`."
        )
    if real_hw_count == 0:
        recommendations.append(
            "Annotate authentic cursive/pen-drawn Kannada text lines rather than relying exclusively on printed typography."
        )
    if not (manifest_path / "test_kannada.jsonl").exists():
        recommendations.append(
            "Use `src/training/splitter.py` to create clean `train_kannada.jsonl`, `val_kannada.jsonl`, and `test_kannada.jsonl` splits."
        )

    return DatasetAuditReport(
        scanned_directories=scanned_dirs,
        total_physical_images=len(found_images),
        physical_images=found_images,
        total_manifest_records=total_records,
        valid_manifest_records=valid_records,
        missing_manifest_records=missing_records,
        samples_by_language=samples_by_lang,
        samples_by_type=samples_by_type,
        manifest_summaries=manifest_summaries,
        sufficiency=sufficiency,
        recommendations=recommendations,
    )


def main():
    """CLI runner for dataset audit."""
    print("=" * 70)
    print("  LAND RECORD DIGITIZATION - DATASET READINESS AUDIT")
    print("=" * 70)

    report = run_dataset_audit()
    print(report.to_markdown())
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
