"""Dataset Preparation and Validation Utilities for Multilingual Handwriting OCR.

Provides robust validation, cleaning, and normalization of raw manifests:
- Validates image existence, accessibility, and image file integrity using PIL.
- Validates non-empty Unicode transcriptions with normalization.
- Detects duplicate image references and path collisions.
- Generates detailed, structured validation reports with error and warning categorization.
"""

import json
import os
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from PIL import Image, UnidentifiedImageError

from src.training.dataset import MultilingualHandwritingSample, parse_manifest_record


@dataclass
class ValidationIssue:
    """Represents a specific validation error or warning for a dataset record."""
    line_number: Optional[int]
    image_path: Optional[str]
    issue_type: str  # "missing_image", "corrupt_image", "empty_text", "duplicate_path", "invalid_json", "missing_field"
    message: str
    severity: str = "error"  # "error" or "warning"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Aggregated validation report summarizing dataset manifest health."""
    total_records: int = 0
    valid_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    duplicate_count: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Returns True if there are zero blocking errors."""
        return self.error_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_count": self.valid_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "duplicate_count": self.duplicate_count,
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_image_file(image_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int]]]:
    """Verifies that an image file exists, is non-empty, and can be decoded by PIL.

    Args:
        image_path: Path to image file.

    Returns:
        Tuple[bool, Optional[str], Optional[Tuple[int, int]]]: (is_valid, error_message, (width, height)).
    """
    if not image_path.exists():
        return False, f"Image file not found on disk: '{image_path}'", None

    if not image_path.is_file():
        return False, f"Path is not a regular file: '{image_path}'", None

    if image_path.stat().st_size == 0:
        return False, f"Image file is empty (0 bytes): '{image_path}'", None

    try:
        with Image.open(image_path) as img:
            size = img.size
            if size[0] <= 0 or size[1] <= 0:
                return False, f"Invalid image dimensions {size}: '{image_path}'", None
            img.verify()
            return True, None, size
    except (UnidentifiedImageError, OSError, Exception) as exc:
        return False, f"Cannot decode image file '{image_path}': {str(exc)}", None



def normalize_unicode_text(text: str) -> str:
    """Normalizes Unicode text using NFC canonical composition."""
    if not text:
        return ""
    # NFC normalization ensures Indic conjuncts and vowels compose properly
    return unicodedata.normalize("NFC", text).strip()


def validate_manifest(
    manifest_path: Union[str, Path],
    root_dir: Optional[Union[str, Path]] = None,
    verify_images: bool = True,
    allow_duplicates: bool = False,
) -> Tuple[List[MultilingualHandwritingSample], ValidationReport]:
    """Validates a JSONL manifest file and returns clean samples with a detailed report.

    Args:
        manifest_path: Path to the JSONL manifest file.
        root_dir: Optional base directory to resolve relative image paths.
        verify_images: If True, checks file existence and decodability.
        allow_duplicates: If False, duplicate image paths trigger validation errors.

    Returns:
        Tuple[List[MultilingualHandwritingSample], ValidationReport]: Clean valid samples and report.
    """
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        report = ValidationReport(
            error_count=1,
            issues=[
                ValidationIssue(
                    line_number=None,
                    image_path=None,
                    issue_type="missing_manifest",
                    message=f"Manifest file does not exist: '{manifest_file}'",
                    severity="error",
                )
            ],
        )
        return [], report

    base_dir = Path(root_dir) if root_dir else manifest_file.parent

    valid_samples: List[MultilingualHandwritingSample] = []
    issues: List[ValidationIssue] = []
    seen_image_paths: Set[str] = set()

    total_records = 0
    duplicate_count = 0

    with open(manifest_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            clean_line = line.strip()
            if not clean_line or clean_line.startswith("#"):
                continue

            total_records += 1

            # 1. Parse JSON structure
            try:
                raw_dict = json.loads(clean_line)
            except json.JSONDecodeError as json_err:
                issues.append(
                    ValidationIssue(
                        line_number=line_idx,
                        image_path=None,
                        issue_type="invalid_json",
                        message=f"Malformed JSON on line {line_idx}: {str(json_err)}",
                        severity="error",
                    )
                )
                continue

            # 2. Check required fields
            img_val = raw_dict.get("image") or raw_dict.get("image_path")
            txt_val = raw_dict.get("text") if raw_dict.get("text") is not None else raw_dict.get("ground_truth")

            if not img_val:
                issues.append(
                    ValidationIssue(
                        line_number=line_idx,
                        image_path=None,
                        issue_type="missing_field",
                        message=f"Missing 'image' or 'image_path' field on line {line_idx}.",
                        severity="error",
                    )
                )
                continue

            if txt_val is None:
                issues.append(
                    ValidationIssue(
                        line_number=line_idx,
                        image_path=str(img_val),
                        issue_type="missing_field",
                        message=f"Missing 'text' or 'ground_truth' field on line {line_idx}.",
                        severity="error",
                    )
                )
                continue

            # 3. Validate text content
            norm_text = normalize_unicode_text(str(txt_val))
            if not norm_text:
                issues.append(
                    ValidationIssue(
                        line_number=line_idx,
                        image_path=str(img_val),
                        issue_type="empty_text",
                        message=f"Transcription is empty or whitespace-only on line {line_idx}.",
                        severity="error",
                    )
                )
                continue

            # 4. Resolve image path
            img_path = Path(img_val)
            if not img_path.is_absolute():
                img_path = base_dir / img_path

            norm_path_str = str(img_path.resolve()) if img_path.exists() else str(img_path)

            # 5. Check duplicate image paths
            if norm_path_str in seen_image_paths:
                duplicate_count += 1
                if not allow_duplicates:
                    issues.append(
                        ValidationIssue(
                            line_number=line_idx,
                            image_path=str(img_val),
                            issue_type="duplicate_path",
                            message=f"Duplicate image path detected on line {line_idx}: '{img_val}'.",
                            severity="error",
                        )
                    )
                    continue
                else:
                    issues.append(
                        ValidationIssue(
                            line_number=line_idx,
                            image_path=str(img_val),
                            issue_type="duplicate_path",
                            message=f"Duplicate image path on line {line_idx}: '{img_val}'.",
                            severity="warning",
                        )
                    )
            seen_image_paths.add(norm_path_str)

            # 6. Verify image existence and decodability
            if verify_images:
                is_img_valid, img_err, dims = validate_image_file(img_path)
                if not is_img_valid:
                    issues.append(
                        ValidationIssue(
                            line_number=line_idx,
                            image_path=str(img_val),
                            issue_type="missing_image" if "not found" in (img_err or "") else "corrupt_image",
                            message=f"Line {line_idx}: {img_err}",
                            severity="error",
                        )
                    )
                    continue

            # 7. Construct clean sample
            lang_val = str(raw_dict.get("language", "kannada")).lower().strip()
            script_val = str(raw_dict.get("script", lang_val.capitalize()))
            meta_val = raw_dict.get("metadata", {})
            if not isinstance(meta_val, dict):
                meta_val = {"raw_metadata": meta_val}

            sample = MultilingualHandwritingSample(
                image_path=img_path,
                text=norm_text,
                language=lang_val,
                script=script_val,
                metadata=meta_val,
            )
            valid_samples.append(sample)

    error_count = sum(1 for iss in issues if iss.severity == "error")
    warning_count = sum(1 for iss in issues if iss.severity == "warning")

    report = ValidationReport(
        total_records=total_records,
        valid_count=len(valid_samples),
        error_count=error_count,
        warning_count=warning_count,
        duplicate_count=duplicate_count,
        issues=issues,
    )

    return valid_samples, report


def clean_and_save_manifest(
    input_manifest: Union[str, Path],
    output_manifest: Union[str, Path],
    root_dir: Optional[Union[str, Path]] = None,
    verify_images: bool = True,
    allow_duplicates: bool = False,
) -> Tuple[int, ValidationReport]:
    """Validates an input manifest, removes invalid/corrupt records, and saves the cleaned manifest.

    Args:
        input_manifest: Source JSONL manifest path.
        output_manifest: Target JSONL manifest path for clean records.
        root_dir: Base directory for resolving relative paths.
        verify_images: Whether to verify images on disk.
        allow_duplicates: Whether to allow duplicate image references.

    Returns:
        Tuple[int, ValidationReport]: Number of clean records saved and the validation report.
    """
    valid_samples, report = validate_manifest(
        manifest_path=input_manifest,
        root_dir=root_dir,
        verify_images=verify_images,
        allow_duplicates=allow_duplicates,
    )

    out_path = Path(output_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    return len(valid_samples), report
