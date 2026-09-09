"""Reusable IIIT Indic Handwriting Dataset Importer.

Imports official IIIT-INDIC-HW-WORDS and similar IIIT Indic handwriting datasets
(Kannada, Telugu, Tamil, Hindi/Devanagari, Malayalam, Bengali, Gujarati, etc.),
preserving Unicode transcriptions, resolving image paths, validating image existence,
and generating standard JSONL manifests for training/evaluation pipelines.
"""

import json
import os
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.training.bootstrap import LANGUAGE_SCRIPT_MAP
from src.training.dataset import MultilingualHandwritingSample


@dataclass
class ImportIssue:
    """Represents an issue encountered during dataset import."""
    line_number: Optional[int]
    image_path: Optional[str]
    issue_type: str  # "missing_image", "malformed_line", "empty_text", "duplicate_path", "unicode_error", "missing_split"
    message: str
    raw_line: Optional[str] = None
    severity: str = "error"  # "error" or "warning"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImportReport:
    """Aggregated import report for a single split or full dataset."""
    split_name: str
    total_lines: int = 0
    imported_count: int = 0
    missing_images_count: int = 0
    malformed_lines_count: int = 0
    empty_text_count: int = 0
    duplicate_paths_count: int = 0
    unicode_issues_count: int = 0
    issues: List[ImportIssue] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Returns True if there are zero errors or rejected records."""
        return (
            self.missing_images_count == 0
            and self.malformed_lines_count == 0
            and self.empty_text_count == 0
            and self.duplicate_paths_count == 0
            and self.unicode_issues_count == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split_name": self.split_name,
            "total_lines": self.total_lines,
            "imported_count": self.imported_count,
            "missing_images_count": self.missing_images_count,
            "malformed_lines_count": self.malformed_lines_count,
            "empty_text_count": self.empty_text_count,
            "duplicate_paths_count": self.duplicate_paths_count,
            "unicode_issues_count": self.unicode_issues_count,
            "is_clean": self.is_clean,
            "issues_sample": [issue.to_dict() for issue in self.issues[:20]],
            "total_issues_logged": len(self.issues),
        }


class IIITIndicHWImporter:
    """Reusable importer for official IIIT Indic Handwriting datasets."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        language: str = "kannada",
        script: Optional[str] = None,
        project_root: Optional[Union[str, Path]] = None,
        normalize_unicode: bool = True,
    ):
        """Initializes the IIIT Indic Handwriting dataset importer.

        Args:
            dataset_root: Root path to the extracted IIIT dataset (contains train/, val/, test/).
            language: Target language name (e.g. "kannada", "telugu", "tamil", "hindi").
            script: Target script name (e.g. "Kannada", "Telugu", "Tamil", "Devanagari").
                    If omitted, inferred from LANGUAGE_SCRIPT_MAP or capitalized language.
            project_root: Base project root directory to resolve relative paths for manifests.
            normalize_unicode: If True, applies NFC normalization to text while storing raw_text in metadata.
        """
        self.dataset_root = Path(dataset_root).resolve()
        self.language = language.lower().strip()
        self.script = script or LANGUAGE_SCRIPT_MAP.get(self.language, self.language.capitalize())
        self.project_root = Path(project_root).resolve() if project_root else self._find_project_root()
        self.normalize_unicode = normalize_unicode

    def _find_project_root(self) -> Path:
        """Determines the workspace/project root directory."""
        cwd = Path.cwd().resolve()
        if (cwd / "src").exists() or (cwd / "training").exists():
            return cwd
        return cwd

    def _to_manifest_path(self, file_path: Path) -> Path:
        """Converts an absolute file path to a project-relative Path with POSIX formatting."""
        resolved = file_path.resolve()
        try:
            rel = resolved.relative_to(self.project_root)
            return rel
        except ValueError:
            return resolved

    def parse_ground_truth_line(
        self,
        line: str,
        line_number: int,
        split_dir: Path,
        split_name: str,
        validate_images: bool = True,
    ) -> Tuple[Optional[MultilingualHandwritingSample], Optional[ImportIssue]]:
        """Parses a single ground-truth line.

        Line format: <relative_image_path><whitespace or tab><transcription>
        Split strictly once to preserve potential spaces inside Unicode transcription.

        Args:
            line: Raw line string from ground truth file.
            line_number: 1-indexed line number for error reporting.
            split_dir: Directory of the current split (e.g. dataset_root/train).
            split_name: Name of current split ("train", "val", "test").
            validate_images: Whether to verify that the image exists on disk.

        Returns:
            Tuple[Optional[MultilingualHandwritingSample], Optional[ImportIssue]]
        """
        raw_line = line.rstrip("\r\n")
        stripped = raw_line.strip()

        if not stripped:
            return None, ImportIssue(
                line_number=line_number,
                image_path=None,
                issue_type="malformed_line",
                message=f"Empty line encountered at line {line_number}",
                raw_line=raw_line,
            )

        # Split strictly once by first whitespace/tab delimiter
        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            return None, ImportIssue(
                line_number=line_number,
                image_path=parts[0] if parts else None,
                issue_type="malformed_line",
                message=f"Missing transcription or image path at line {line_number}: '{raw_line}'",
                raw_line=raw_line,
            )

        rel_img_str, raw_transcription = parts[0].strip(), parts[1].strip()

        if not rel_img_str:
            return None, ImportIssue(
                line_number=line_number,
                image_path=None,
                issue_type="malformed_line",
                message=f"Empty image path at line {line_number}",
                raw_line=raw_line,
            )

        if not raw_transcription:
            return None, ImportIssue(
                line_number=line_number,
                image_path=rel_img_str,
                issue_type="empty_text",
                message=f"Empty transcription at line {line_number} for image '{rel_img_str}'",
                raw_line=raw_line,
            )

        # Resolve image file location
        clean_rel_path = Path(rel_img_str.replace("\\", "/"))
        image_file = (split_dir / clean_rel_path).resolve()

        if validate_images and not image_file.exists():
            return None, ImportIssue(
                line_number=line_number,
                image_path=str(image_file),
                issue_type="missing_image",
                message=f"Referenced image file not found: '{image_file}' (referenced as '{rel_img_str}')",
                raw_line=raw_line,
            )

        # Process Unicode text
        final_text = unicodedata.normalize("NFC", raw_transcription) if self.normalize_unicode else raw_transcription

        if not final_text:
            return None, ImportIssue(
                line_number=line_number,
                image_path=str(image_file),
                issue_type="empty_text",
                message=f"Transcription became empty after normalization at line {line_number}",
                raw_line=raw_line,
            )

        manifest_path = self._to_manifest_path(image_file)

        metadata: Dict[str, Any] = {
            "source": "IIIT-INDIC-HW-WORDS",
            "source_type": "real_handwriting",
            "original_split": split_name,
            "raw_text": raw_transcription,
        }

        sample = MultilingualHandwritingSample(
            image_path=manifest_path,
            text=final_text,
            language=self.language,
            script=self.script,
            metadata=metadata,
        )

        return sample, None

    def find_ground_truth_file(self, split_dir: Path, split_name: str) -> Path:
        """Discovers the ground truth file for a given split."""
        candidates = [
            split_dir / f"{split_name}_gt.txt",
            split_dir / "gt.txt",
            split_dir / f"{self.language}_{split_name}_gt.txt",
            split_dir / f"{split_name}.txt",
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        # Search for any *_gt.txt file in split_dir
        gt_files = list(split_dir.glob("*_gt.txt")) + list(split_dir.glob("*.txt"))
        gt_files = [f for f in gt_files if f.name != "vocabulary.txt"]
        if gt_files:
            return gt_files[0]

        raise FileNotFoundError(
            f"Ground truth file not found in split directory '{split_dir}'. Checked: {[str(c) for c in candidates]}"
        )

    def import_split(
        self,
        split_name: str,
        gt_filename: Optional[str] = None,
        validate_images: bool = True,
    ) -> Tuple[List[MultilingualHandwritingSample], ImportReport]:
        """Imports an individual dataset split (e.g. 'train', 'val', 'test').

        Args:
            split_name: Name of the split subdirectory ('train', 'val', 'test').
            gt_filename: Optional specific filename for the ground-truth file.
            validate_images: If True, checks that each referenced image exists on disk.

        Returns:
            Tuple[List[MultilingualHandwritingSample], ImportReport]: Imported samples and report.
        """
        split_dir = (self.dataset_root / split_name).resolve()
        report = ImportReport(split_name=split_name)

        if not split_dir.exists() or not split_dir.is_dir():
            issue = ImportIssue(
                line_number=None,
                image_path=None,
                issue_type="missing_split",
                message=f"Split directory not found: '{split_dir}'",
            )
            report.issues.append(issue)
            return [], report

        # Find ground truth file
        if gt_filename:
            gt_file = split_dir / gt_filename
            if not gt_file.exists():
                issue = ImportIssue(
                    line_number=None,
                    image_path=None,
                    issue_type="missing_split",
                    message=f"Specified ground truth file '{gt_file}' does not exist",
                )
                report.issues.append(issue)
                return [], report
        else:
            try:
                gt_file = self.find_ground_truth_file(split_dir, split_name)
            except FileNotFoundError as e:
                report.issues.append(
                    ImportIssue(
                        line_number=None,
                        image_path=None,
                        issue_type="missing_split",
                        message=str(e),
                    )
                )
                return [], report

        samples: List[MultilingualHandwritingSample] = []
        seen_image_keys: Set[str] = set()

        with open(gt_file, "r", encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f, start=1):
                report.total_lines += 1

                sample, issue = self.parse_ground_truth_line(
                    line=line,
                    line_number=line_idx,
                    split_dir=split_dir,
                    split_name=split_name,
                    validate_images=validate_images,
                )

                if issue is not None:
                    report.issues.append(issue)
                    if issue.issue_type == "missing_image":
                        report.missing_images_count += 1
                    elif issue.issue_type == "empty_text":
                        report.empty_text_count += 1
                    elif issue.issue_type == "malformed_line":
                        report.malformed_lines_count += 1
                    elif issue.issue_type == "unicode_error":
                        report.unicode_issues_count += 1
                    continue

                if sample is not None:
                    norm_path_key = str(sample.image_path).replace("\\", "/").lower()
                    if norm_path_key in seen_image_keys:
                        report.duplicate_paths_count += 1
                        dup_issue = ImportIssue(
                            line_number=line_idx,
                            image_path=str(sample.image_path),
                            issue_type="duplicate_path",
                            message=f"Duplicate image path in split '{split_name}': {sample.image_path}",
                            raw_line=line.strip(),
                            severity="warning",
                        )
                        report.issues.append(dup_issue)
                    else:
                        seen_image_keys.add(norm_path_key)

                    samples.append(sample)
                    report.imported_count += 1

        return samples, report

    def import_all_splits(
        self,
        splits: Sequence[str] = ("train", "val", "test"),
        validate_images: bool = True,
    ) -> Dict[str, Tuple[List[MultilingualHandwritingSample], ImportReport]]:
        """Imports all standard splits independently and preserves original partitioning."""
        results: Dict[str, Tuple[List[MultilingualHandwritingSample], ImportReport]] = {}
        for split in splits:
            results[split] = self.import_split(split, validate_images=validate_images)
        return results

    def save_manifest(
        self,
        samples: Sequence[MultilingualHandwritingSample],
        output_path: Union[str, Path],
    ) -> Path:
        """Serializes samples to a clean UTF-8 JSONL manifest."""
        dest = Path(output_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "w", encoding="utf-8") as f:
            for s in samples:
                record = s.to_dict()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return dest

    def verify_dataset_integrity(
        self,
        split_results: Dict[str, Tuple[List[MultilingualHandwritingSample], ImportReport]],
    ) -> Dict[str, Any]:
        """Performs dataset integrity checks across all imported splits."""
        integrity: Dict[str, Any] = {
            "all_images_exist": True,
            "all_transcriptions_non_empty": True,
            "zero_split_overlap": True,
            "sample_1_verified": {},
            "split_counts": {},
            "overlap_details": {},
            "total_imported": 0,
        }

        split_paths: Dict[str, Set[str]] = {}

        for split_name, (samples, report) in split_results.items():
            integrity["split_counts"][split_name] = len(samples)
            integrity["total_imported"] += len(samples)
            paths_for_split: Set[str] = set()

            if samples:
                first_sample = samples[0]
                integrity["sample_1_verified"][split_name] = {
                    "image": str(first_sample.image_path).replace("\\", "/"),
                    "text": first_sample.text,
                    "language": first_sample.language,
                }

            for sample in samples:
                if not sample.text or not sample.text.strip():
                    integrity["all_transcriptions_non_empty"] = False

                full_path = (self.project_root / sample.image_path).resolve()
                if not full_path.exists():
                    integrity["all_images_exist"] = False

                path_str = str(full_path).replace("\\", "/").lower()
                paths_for_split.add(path_str)

            split_paths[split_name] = paths_for_split

        split_names = list(split_paths.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                intersection = split_paths[s1].intersection(split_paths[s2])
                if intersection:
                    integrity["zero_split_overlap"] = False
                    integrity["overlap_details"][f"{s1}_vs_{s2}"] = len(intersection)

        return integrity
