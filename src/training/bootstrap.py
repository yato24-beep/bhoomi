"""Dataset Bootstrap and Multilingual Manifest Utilities.

Provides structured scanning, discovery, template generation, and manifest
bootstrapping for regional Indic languages without fabricating ground-truth labels.
"""

import json
import os
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from src.training.data_preparation import normalize_unicode_text, validate_image_file
from src.training.dataset import MultilingualHandwritingSample

# Canonical language-to-script mapping
LANGUAGE_SCRIPT_MAP: Dict[str, str] = {
    "kannada": "Kannada",
    "hindi": "Devanagari",
    "devanagari": "Devanagari",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "malayalam": "Malayalam",
    "bengali": "Bengali",
    "gujarati": "Gujarati",
    "marathi": "Devanagari",
    "english": "Latin",
}

SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
}


@dataclass
class LanguageScanResult:
    """Discovery and annotation state for a specific language directory."""
    language: str
    script: str
    directory_path: Path
    total_images_found: int = 0
    annotated_count: int = 0
    unannotated_count: int = 0
    annotated_samples: List[MultilingualHandwritingSample] = field(default_factory=list)
    unannotated_images: List[Path] = field(default_factory=list)
    source_type_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "script": self.script,
            "directory": str(self.directory_path).replace("\\", "/"),
            "total_images_found": self.total_images_found,
            "annotated_count": self.annotated_count,
            "unannotated_count": self.unannotated_count,
            "source_type_counts": self.source_type_counts,
            "annotated_samples": [s.to_dict() for s in self.annotated_samples],
            "unannotated_images": [str(p).replace("\\", "/") for p in self.unannotated_images],
        }


@dataclass
class BootstrapSummary:
    """Aggregated summary of dataset bootstrap execution."""
    data_root: Path
    languages_scanned: List[str] = field(default_factory=list)
    created_directories: List[str] = field(default_factory=list)
    results_by_language: Dict[str, LanguageScanResult] = field(default_factory=dict)
    templates_generated: List[str] = field(default_factory=list)
    manifests_generated: List[str] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return sum(res.total_images_found for res in self.results_by_language.values())

    @property
    def total_annotated(self) -> int:
        return sum(res.annotated_count for res in self.results_by_language.values())

    @property
    def total_unannotated(self) -> int:
        return sum(res.unannotated_count for res in self.results_by_language.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_root": str(self.data_root).replace("\\", "/"),
            "languages_scanned": self.languages_scanned,
            "created_directories": self.created_directories,
            "total_images": self.total_images,
            "total_annotated": self.total_annotated,
            "total_unannotated": self.total_unannotated,
            "templates_generated": self.templates_generated,
            "manifests_generated": self.manifests_generated,
            "results": {
                lang: res.to_dict() for lang, res in self.results_by_language.items()
            },
        }


def get_canonical_script(language: str) -> str:
    """Returns the canonical script name for a supported language."""
    clean_lang = language.lower().strip()
    return LANGUAGE_SCRIPT_MAP.get(clean_lang, clean_lang.capitalize())


def ensure_dataset_directories(
    data_root: Optional[Union[str, Path]] = None,
    languages: Sequence[str] = ("kannada", "hindi", "tamil", "telugu", "malayalam"),
) -> Tuple[Path, List[Path]]:
    """Ensures standard dataset directory structure exists.

    Args:
        data_root: Base project or data root directory.
        languages: Language folders to create under data/raw/.

    Returns:
        Tuple[Path, List[Path]]: (data_dir_path, list_of_created_or_verified_dirs).
    """
    root = Path(data_root) if data_root else Path.cwd()
    if root.name != "data" and (root / "data").exists():
        data_base = root / "data"
    elif root.name == "data":
        data_base = root
    else:
        data_base = root / "data"

    standard_subdirs = [
        data_base / "raw",
        data_base / "annotations",
        data_base / "processed",
        data_base / "synthetic",
        data_base / "samples",
    ]

    for lang in languages:
        standard_subdirs.append(data_base / "raw" / lang.lower().strip())

    created = []
    for d in standard_subdirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d)
        # Ensure a .gitkeep exists if empty
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists() and not any(d.iterdir()):
            gitkeep.touch()

    return data_base, standard_subdirs


def find_existing_annotations(
    annotations_dir: Path,
    manifests_dir: Optional[Path] = None,
    language: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Scans annotations directory and manifest files to map known image paths to ground truth.

    Args:
        annotations_dir: Path to data/annotations directory.
        manifests_dir: Optional path to training/datasets directory.
        language: Optional language filter.

    Returns:
        Dict[str, Dict[str, Any]]: Map of normalized absolute image path -> annotation dict.
    """
    known_annotations: Dict[str, Dict[str, Any]] = {}
    search_files: List[Path] = []

    if annotations_dir.exists():
        search_files.extend(annotations_dir.glob("*.jsonl"))
        search_files.extend(annotations_dir.glob("*.json"))

    if manifests_dir and manifests_dir.exists():
        search_files.extend(manifests_dir.glob("*.jsonl"))

    for f_path in search_files:
        # Ignore template files
        if "template" in f_path.name.lower():
            continue
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_clean = line.strip()
                    if not line_clean or line_clean.startswith("#"):
                        continue
                    try:
                        rec = json.loads(line_clean)
                        img_val = rec.get("image") or rec.get("image_path")
                        txt_val = rec.get("text") if rec.get("text") is not None else rec.get("ground_truth")
                        if not img_val or txt_val is None:
                            continue

                        norm_txt = normalize_unicode_text(str(txt_val))
                        if not norm_txt:
                            continue  # Ignore placeholder/empty entries

                        # Normalize path
                        p_obj = Path(img_val)
                        if not p_obj.is_absolute():
                            # Resolve relative to file parent or root
                            candidate = f_path.parent / p_obj
                            if candidate.exists():
                                p_obj = candidate

                        norm_key = str(p_obj.resolve()) if p_obj.exists() else str(p_obj).replace("\\", "/")
                        rec_lang = str(rec.get("language", "kannada")).lower().strip()

                        if language and rec_lang != language.lower().strip():
                            continue

                        known_annotations[norm_key] = {
                            "text": norm_txt,
                            "language": rec_lang,
                            "script": rec.get("script", get_canonical_script(rec_lang)),
                            "metadata": rec.get("metadata", {}),
                        }
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    return known_annotations


def scan_language_directory(
    data_root: Union[str, Path],
    language: str,
    annotations_dir: Optional[Union[str, Path]] = None,
    manifests_dir: Optional[Union[str, Path]] = None,
) -> LanguageScanResult:
    """Scans raw language folder and samples for images and categorizes annotation status.

    Args:
        data_root: Root data directory (e.g. Land Record/data).
        language: Language to scan (e.g. 'kannada', 'hindi', 'tamil').
        annotations_dir: Optional custom annotations path.
        manifests_dir: Optional custom manifests path.

    Returns:
        LanguageScanResult: Detailed breakdown of discovered images and annotations.
    """
    root_p = Path(data_root)
    clean_lang = language.lower().strip()
    script = get_canonical_script(clean_lang)

    raw_lang_dir = root_p / "raw" / clean_lang if (root_p / "raw").exists() else root_p / clean_lang
    ann_dir = Path(annotations_dir) if annotations_dir else root_p / "annotations"
    man_dir = Path(manifests_dir) if manifests_dir else root_p.parent / "training" / "datasets"

    known_ann = find_existing_annotations(ann_dir, man_dir, clean_lang)

    found_images: List[Path] = []
    if raw_lang_dir.exists():
        for f in sorted(raw_lang_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                found_images.append(f)

    # Also check data/samples/ if matching language keyword or baseline sample
    samples_dir = root_p / "samples"
    if samples_dir.exists():
        for f in sorted(samples_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                if clean_lang in f.name.lower() or (clean_lang == "english" and "handwritten" in f.name.lower()):
                    found_images.append(f)

    annotated_samples: List[MultilingualHandwritingSample] = []
    unannotated_images: List[Path] = []
    type_counts: Dict[str, int] = {"real_handwriting": 0, "printed": 0, "synthetic": 0, "unknown": 0}

    for img_p in found_images:
        norm_key = str(img_p.resolve())
        # Check if known in annotations
        ann = known_ann.get(norm_key)
        if not ann:
            # Check relative key match
            rel_str = str(img_p).replace("\\", "/")
            ann = known_ann.get(rel_str)

        if ann and ann.get("text"):
            meta = dict(ann.get("metadata", {}))
            stype = str(meta.get("source_type") or meta.get("text_type") or "").lower()
            if not stype:
                if "kannada" in img_p.name.lower() or "print" in img_p.name.lower() or "document" in img_p.name.lower():
                    stype = "printed"
                elif "handwritten" in img_p.name.lower() or "cursive" in img_p.name.lower():
                    stype = "synthetic" if "sample" in img_p.name.lower() else "real_handwriting"
                else:
                    stype = "unknown"

            type_counts[stype] = type_counts.get(stype, 0) + 1

            sample = MultilingualHandwritingSample(
                image_path=img_p,
                text=ann["text"],
                language=clean_lang,
                script=script,
                metadata={"source_type": stype, **meta},
            )
            annotated_samples.append(sample)
        else:
            unannotated_images.append(img_p)

    return LanguageScanResult(
        language=clean_lang,
        script=script,
        directory_path=raw_lang_dir,
        total_images_found=len(found_images),
        annotated_count=len(annotated_samples),
        unannotated_count=len(unannotated_images),
        annotated_samples=annotated_samples,
        unannotated_images=unannotated_images,
        source_type_counts=type_counts,
    )


def generate_annotation_template(
    unannotated_paths: Sequence[Union[str, Path]],
    language: str,
    output_file: Union[str, Path],
    source_type: str = "real_handwriting",
    root_dir: Optional[Union[str, Path]] = None,
) -> int:
    """Generates an empty annotation template JSONL for human labeling.

    Args:
        unannotated_paths: List of image paths requiring ground-truth transcription.
        language: Language identifier.
        output_file: Target JSONL template file.
        source_type: Default source type tag ('real_handwriting', 'printed', 'synthetic').
        root_dir: Optional base directory to express image paths as relative paths.

    Returns:
        int: Number of template entries generated.
    """
    clean_lang = language.lower().strip()
    script = get_canonical_script(clean_lang)
    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    base_p = Path(root_dir) if root_dir else None

    count = 0
    with open(out_p, "w", encoding="utf-8") as f:
        for p in unannotated_paths:
            img_path = Path(p)
            if base_p and img_path.is_relative_to(base_p):
                disp_path = str(img_path.relative_to(base_p)).replace("\\", "/")
            else:
                disp_path = str(img_path).replace("\\", "/")

            record = {
                "image": disp_path,
                "text": "",  # Empty placeholder for human annotator to fill
                "language": clean_lang,
                "script": script,
                "metadata": {
                    "source_type": source_type,
                    "status": "unannotated",
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def bootstrap_multilingual_datasets(
    project_root: Optional[Union[str, Path]] = None,
    languages: Sequence[str] = ("kannada", "hindi", "tamil", "telugu", "malayalam"),
    generate_templates: bool = True,
    output_manifest_dir: Optional[Union[str, Path]] = None,
) -> BootstrapSummary:
    """Orchestrates directory creation, scan, template creation, and manifest compilation.

    Args:
        project_root: Base project root.
        languages: Languages to scan and bootstrap.
        generate_templates: Whether to write annotation templates for unannotated images.
        output_manifest_dir: Destination directory for compiled manifests.

    Returns:
        BootstrapSummary: Full execution summary.
    """
    root = Path(project_root) if project_root else Path.cwd()
    data_dir, created_dirs = ensure_dataset_directories(root, languages)

    ann_dir = data_dir / "annotations"
    man_dir = Path(output_manifest_dir) if output_manifest_dir else root / "training" / "datasets"
    man_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, LanguageScanResult] = {}
    templates_gen: List[str] = []
    manifests_gen: List[str] = []

    for lang in languages:
        scan_res = scan_language_directory(
            data_root=data_dir,
            language=lang,
            annotations_dir=ann_dir,
            manifests_dir=man_dir,
        )
        results[lang] = scan_res

        # 1. Generate template if unannotated images exist
        if generate_templates and scan_res.unannotated_images:
            tmpl_file = ann_dir / f"{lang}_template.jsonl"
            generate_annotation_template(
                unannotated_paths=scan_res.unannotated_images,
                language=lang,
                output_file=tmpl_file,
                root_dir=root,
            )
            templates_gen.append(str(tmpl_file.relative_to(root)).replace("\\", "/"))

        # 2. Write valid annotated samples to language manifest if any exist
        if scan_res.annotated_samples:
            man_file = man_dir / f"{lang}_all.jsonl"
            with open(man_file, "w", encoding="utf-8") as f:
                for s in scan_res.annotated_samples:
                    f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
            manifests_gen.append(str(man_file.relative_to(root)).replace("\\", "/"))

    return BootstrapSummary(
        data_root=data_dir,
        languages_scanned=list(languages),
        created_directories=[str(d.relative_to(root)).replace("\\", "/") for d in created_dirs],
        results_by_language=results,
        templates_generated=templates_gen,
        manifests_generated=manifests_gen,
    )
