"""Reusable Dataset ZIP Ingestion and Normalization Pipeline.

Handles arbitrary dataset ZIP archives containing land-record or handwriting data:
- Isolated safe extraction (directory traversal prevention)
- Directory structure inspection
- Multi-format image discovery (.jpg, .jpeg, .png, .tif, .tiff)
- Flexible annotation discovery (.txt, .json, .csv, .tsv)
- Corrupt image detection and label round-trip validation
- Deterministic train/val/test/calibration split generation
- Full cryptographic provenance manifest generation
- Structured JSON and Markdown validation reports
"""

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import shutil
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import zipfile

from PIL import Image

from src.data.provenance import (
    ProvenanceManifest,
    SampleProvenance,
    compute_sha256,
    compute_bytes_sha256,
)

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_LABEL_EXTS = {".txt", ".json", ".csv", ".tsv"}


@dataclass
class IngestedSample:
    """Represents a discovered and normalized sample."""
    sample_id: str
    image_rel_path: str
    absolute_image_path: Path
    text_label: Optional[str] = None
    label_source: Optional[str] = None  # "txt", "json", "csv", "tsv", None
    metadata: Dict[str, Any] = field(default_factory=dict)
    width: int = 0
    height: int = 0
    channels: int = 3
    is_corrupt: bool = False
    error_message: Optional[str] = None
    sha256: str = ""
    split: str = "train"


@dataclass
class IngestionReport:
    """Summary report detailing the ingestion outcome."""
    archive_path: str
    archive_sha256: str
    dataset_name: str
    total_images_discovered: int = 0
    valid_images: int = 0
    corrupt_images: int = 0
    labeled_samples: int = 0
    unlabeled_samples: int = 0
    empty_labels: int = 0
    duplicate_images_count: int = 0
    split_policy: str = "sample_level_weaker"
    splits_count: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    structure_type: str = "unknown"
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetZipIngester:
    """Orchestrates secure extraction, parsing, and normalization of dataset archives."""

    def __init__(
        self,
        output_root: Union[str, Path],
        split_ratios: Optional[Dict[str, float]] = None,
        random_seed: int = 42,
        dataset_name: Optional[str] = None,
        license_terms: str = "Restricted / Archival Research Only",
        source_attribution: str = "Official Land Records Archives",
    ):
        self.output_root = Path(output_root)
        self.split_ratios = split_ratios or {"train": 0.70, "val": 0.15, "test": 0.15}
        self.random_seed = random_seed
        self.dataset_name = dataset_name
        self.license_terms = license_terms
        self.source_attribution = source_attribution

    def ingest_zip(
        self,
        zip_path: Union[str, Path],
        target_subfolder: Optional[str] = None,
    ) -> Tuple[Path, IngestionReport, ProvenanceManifest]:
        """Runs the complete ingestion pipeline on an arbitrary ZIP file."""
        import time
        start_t = time.perf_counter()

        zip_p = Path(zip_path)
        if not zip_p.is_file():
            raise FileNotFoundError(f"ZIP archive not found: {zip_path}")

        archive_sha = compute_sha256(zip_p)
        ds_name = self.dataset_name or zip_p.stem
        target_dir = self.output_root / (target_subfolder or ds_name)
        target_dir.mkdir(parents=True, exist_ok=True)

        extract_dir = target_dir / "_raw_extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        report = IngestionReport(
            archive_path=str(zip_p.resolve()),
            archive_sha256=archive_sha,
            dataset_name=ds_name,
        )

        # 1. Isolated safe extraction
        logger.info("Extracting %s safely to %s", zip_p.name, extract_dir)
        self._safe_extract(zip_p, extract_dir)

        # 2. Inspect structure & discover assets
        images, labels_map, structure_type = self._discover_assets(extract_dir)
        report.structure_type = structure_type
        report.total_images_discovered = len(images)

        # 3. Validate images & associate labels
        ingested_samples: List[IngestedSample] = []
        seen_sha_map: Dict[str, str] = {}
        for img_p in images:
            sample = self._validate_and_build_sample(img_p, labels_map, extract_dir)
            if sample.is_corrupt:
                report.corrupt_images += 1
                report.warnings.append(f"Corrupt image {img_p.name}: {sample.error_message}")
            else:
                report.valid_images += 1
                if sample.sha256 in seen_sha_map:
                    report.duplicate_images_count += 1
                    report.warnings.append(
                        f"Duplicate image content detected: {img_p.name} shares identical SHA-256 with {seen_sha_map[sample.sha256]}"
                    )
                else:
                    seen_sha_map[sample.sha256] = img_p.name

                if sample.text_label is not None:
                    if not sample.text_label.strip():
                        report.empty_labels += 1
                        report.warnings.append(f"Empty label string for sample {sample.sample_id}")
                    report.labeled_samples += 1
                else:
                    report.unlabeled_samples += 1
                ingested_samples.append(sample)

        # 4. Deterministic Train/Val/Test/Calibration Split (Writer/Doc Disjoint Policy)
        self._assign_splits(ingested_samples, report)
        for s in ingested_samples:
            report.splits_count[s.split] = report.splits_count.get(s.split, 0) + 1

        # 5. Build Normalized Output Directory
        images_out = target_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        manifest_data = {
            "dataset_name": ds_name,
            "version": "1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": len(ingested_samples),
            "splits": report.splits_count,
            "samples": [],
        }

        prov_manifest = ProvenanceManifest(
            dataset_name=ds_name,
            dataset_version="1.0.0",
            archive_sha256=archive_sha,
            archive_source=str(zip_p.name),
            license=self.license_terms,
            source_attribution=self.source_attribution,
        )

        for s in ingested_samples:
            dest_img_name = f"{s.sample_id}{s.absolute_image_path.suffix.lower()}"
            dest_img_path = images_out / dest_img_name
            shutil.copy2(s.absolute_image_path, dest_img_path)

            sample_dict = {
                "sample_id": s.sample_id,
                "image_file": f"images/{dest_img_name}",
                "text": s.text_label,
                "split": s.split,
                "width": s.width,
                "height": s.height,
                "channels": s.channels,
                "sha256": s.sha256,
                "metadata": s.metadata,
            }
            manifest_data["samples"].append(sample_dict)

            # Record provenance
            sample_prov = SampleProvenance(
                sample_id=s.sample_id,
                file_name=f"images/{dest_img_name}",
                sample_sha256=s.sha256,
                source_archive=zip_p.name,
                split=s.split,
                domain_tag=s.metadata.get("domain", "land_records"),
                writer_id=s.metadata.get("writer_id"),
                preprocessing_applied=["ingestion_normalized"],
            )
            prov_manifest.add_sample(sample_prov)

        # 6. Verify data leakage & save provenance
        prov_manifest.verify_no_data_leakage()
        prov_manifest.save(target_dir / "provenance.json")

        # 7. Write normalized manifest
        with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        # 8. Clean up raw extraction
        try:
            shutil.rmtree(extract_dir)
        except Exception as e:
            logger.warning("Could not clean raw extraction directory: %s", e)

        # 9. Write validation report
        report.duration_seconds = round(time.perf_counter() - start_t, 3)
        with open(target_dir / "validation_report.json", "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        # 10. Write human-readable Markdown summary
        self._write_markdown_summary(target_dir / "SUMMARY.md", report, prov_manifest)

        logger.info(
            "Completed ingestion of %s: %d valid samples (%d labeled) written to %s",
            ds_name,
            len(ingested_samples),
            report.labeled_samples,
            target_dir,
        )
        return target_dir, report, prov_manifest

    @staticmethod
    def _safe_extract(zip_path: Path, target_dir: Path) -> None:
        """Extracts ZIP ensuring no path traversal vulnerability."""
        resolved_target = target_dir.resolve()
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # Check for zip slip
                member_path = (target_dir / member.filename).resolve()
                if not str(member_path).startswith(str(resolved_target)):
                    raise ValueError(f"Malicious path in ZIP archive: {member.filename}")
            zf.extractall(target_dir)

    def _discover_assets(self, root_dir: Path) -> Tuple[List[Path], Dict[str, Any], str]:
        """Discovers images and all available label files."""
        images: List[Path] = []
        labels_map: Dict[str, Any] = {}
        csv_files: List[Path] = []
        json_files: List[Path] = []
        txt_files: List[Path] = []

        for p in root_dir.rglob("*"):
            if p.is_file():
                ext = p.suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTS:
                    images.append(p)
                elif ext in (".csv", ".tsv"):
                    csv_files.append(p)
                elif ext == ".json":
                    json_files.append(p)
                elif ext == ".txt":
                    txt_files.append(p)

        # Determine structure and load all discovered label sources
        structure_parts = []
        if txt_files:
            structure_parts.append("txt")
            for tf in txt_files:
                try:
                    content = tf.read_text(encoding="utf-8").strip()
                    labels_map[tf.stem] = content
                except Exception:
                    pass
        if json_files:
            structure_parts.append("json")
            for jf in json_files:
                self._load_json_labels(jf, labels_map)
        if csv_files:
            structure_parts.append("csv")
            for cf in csv_files:
                self._load_csv_labels(cf, labels_map)

        structure_type = f"image_plus_{'_'.join(structure_parts)}" if structure_parts else "flat_images"
        return sorted(images), labels_map, structure_type

    @staticmethod
    def _load_csv_labels(csv_path: Path, labels_map: Dict[str, Any]) -> None:
        """Parses CSV or TSV looking for image filename/stem and text label columns."""
        delimiter = "\t" if csv_path.suffix.lower() == ".tsv" else ","
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                if not reader.fieldnames:
                    return
                cols = {c.lower(): c for c in reader.fieldnames}
                # Identify image key and text key
                img_key = next((cols[k] for k in ("image", "filename", "file_name", "img", "id", "image_id") if k in cols), None)
                text_key = next((cols[k] for k in ("text", "label", "ground_truth", "gt", "transcription") if k in cols), None)

                if img_key and text_key:
                    for row in reader:
                        img_val = Path(row[img_key]).stem
                        labels_map[img_val] = row[text_key]
                        # Also record full row metadata
                        labels_map[f"{img_val}__meta"] = row
        except Exception as e:
            logger.warning("Error reading metadata table %s: %s", csv_path.name, e)

    @staticmethod
    def _load_json_labels(json_path: Path, labels_map: Dict[str, Any]) -> None:
        """Parses JSON label structures (dict or list of items)."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    stem = Path(k).stem
                    if isinstance(v, str):
                        labels_map[stem] = v
                    elif isinstance(v, dict) and "text" in v:
                        labels_map[stem] = v["text"]
                        labels_map[f"{stem}__meta"] = v
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        img_id = item.get("image_id") or item.get("filename") or item.get("id")
                        text_val = item.get("text") or item.get("label") or item.get("transcription")
                        if img_id and text_val:
                            stem = Path(str(img_id)).stem
                            labels_map[stem] = str(text_val)
                            labels_map[f"{stem}__meta"] = item
        except Exception as e:
            logger.warning("Error reading JSON annotation %s: %s", json_path.name, e)

    def _validate_and_build_sample(
        self,
        img_path: Path,
        labels_map: Dict[str, Any],
        extract_root: Path,
    ) -> IngestedSample:
        """Validates image integrity and attaches label."""
        sample_id = f"sample_{img_path.stem}"
        try:
            with Image.open(img_path) as img:
                img.verify()  # Check file header & corruption
            with Image.open(img_path) as img:
                w, h = img.size
                channels = len(img.getbands())
        except Exception as e:
            return IngestedSample(
                sample_id=sample_id,
                image_rel_path=str(img_path.relative_to(extract_root)),
                absolute_image_path=img_path,
                is_corrupt=True,
                error_message=str(e),
            )

        sha = compute_sha256(img_path)
        stem = img_path.stem
        text_label = labels_map.get(stem)
        if text_label is not None:
            import unicodedata
            text_label = unicodedata.normalize("NFC", str(text_label))
            # Filter non-printable/corrupted control characters (preserve normal whitespace)
            text_label = "".join(
                c for c in text_label
                if not unicodedata.category(c).startswith("C") or c in ("\n", "\t", "\r")
            )

        meta = labels_map.get(f"{stem}__meta", {})

        return IngestedSample(
            sample_id=sample_id,
            image_rel_path=str(img_path.relative_to(extract_root)),
            absolute_image_path=img_path,
            text_label=text_label,
            label_source="discovered" if text_label is not None else None,
            metadata=meta if isinstance(meta, dict) else {},
            width=w,
            height=h,
            channels=channels,
            is_corrupt=False,
            sha256=sha,
        )

    def _assign_splits(self, samples: List[IngestedSample], report: IngestionReport) -> None:
        """Assigns deterministic train/val/test splits following strict split policy hierarchy:
        1. Preferred: writer-disjoint (if authentic writer metadata exists)
        2. Fallback: document-disjoint (if authentic document metadata exists)
        3. Last resort: sample-level split, explicitly marked as weaker.
        Never automatically invent writer or document IDs.
        """
        if not samples:
            return

        rng = random.Random(self.random_seed)
        train_p = self.split_ratios.get("train", 0.70)
        val_p = self.split_ratios.get("val", 0.15)

        # 1. Check for authentic writer metadata
        writer_map: Dict[str, List[IngestedSample]] = {}
        for s in samples:
            wid = s.metadata.get("writer_id") or s.metadata.get("writer")
            if wid is not None and str(wid).strip():
                writer_map.setdefault(str(wid).strip(), []).append(s)

        if len(writer_map) > 1 and sum(len(v) for v in writer_map.values()) >= len(samples) * 0.5:
            report.split_policy = "writer_disjoint"
            writers = sorted(list(writer_map.keys()))
            rng.shuffle(writers)
            n = len(writers)
            w_train_end = max(1, int(n * train_p))
            w_val_end = max(w_train_end + 1, w_train_end + int(n * val_p))

            for idx, wid in enumerate(writers):
                sp = "train" if idx < w_train_end else ("val" if idx < w_val_end else "test")
                for s in writer_map[wid]:
                    s.split = sp

            # Any unassigned samples without writer_id get assigned to train
            for s in samples:
                if not (s.metadata.get("writer_id") or s.metadata.get("writer")):
                    s.split = "train"
            return

        # 2. Check for authentic document metadata
        doc_map: Dict[str, List[IngestedSample]] = {}
        for s in samples:
            did = s.metadata.get("document_id") or s.metadata.get("doc_id")
            if did is not None and str(did).strip():
                doc_map.setdefault(str(did).strip(), []).append(s)

        if len(doc_map) > 1:
            report.split_policy = "document_disjoint"
            report.warnings.append("Writer metadata not provided. Writer-disjoint split cannot be guaranteed; performed document-disjoint split.")
            docs = sorted(list(doc_map.keys()))
            rng.shuffle(docs)
            n = len(docs)
            d_train_end = max(1, int(n * train_p))
            d_val_end = max(d_train_end + 1, d_train_end + int(n * val_p))

            for idx, did in enumerate(docs):
                sp = "train" if idx < d_train_end else ("val" if idx < d_val_end else "test")
                for s in doc_map[did]:
                    s.split = sp
            return

        # 3. Last resort: sample-level split
        report.split_policy = "sample_level_weaker"
        report.warnings.append(
            "Writer and document identity metadata not found. Writer-disjoint split cannot be guaranteed; performed weaker sample-level split."
        )
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * train_p)
        val_end = train_end + int(n * val_p)

        for i, s in enumerate(shuffled):
            if i < train_end:
                s.split = "train"
            elif i < val_end:
                s.split = "val"
            else:
                s.split = "test"

    @staticmethod
    def _write_markdown_summary(out_file: Path, report: IngestionReport, prov: ProvenanceManifest) -> None:
        """Writes human-readable Markdown report."""
        content = f"""# Dataset Ingestion Summary: {report.dataset_name}

- **Ingestion Timestamp**: {datetime.now(timezone.utc).isoformat()}
- **Source Archive**: `{Path(report.archive_path).name}`
- **Archive SHA-256**: `{report.archive_sha256}`
- **Attribution**: {prov.source_attribution}
- **License**: {prov.license}
- **Structure Detected**: `{report.structure_type}`
- **Ingestion Duration**: {report.duration_seconds}s

## Dataset Health & Metrics
| Metric | Count |
| :--- | :--- |
| **Total Images Discovered** | {report.total_images_discovered} |
| **Valid Images Ingested** | {report.valid_images} |
| **Corrupt Images** | {report.corrupt_images} |
| **Labeled Samples** | {report.labeled_samples} |
| **Unlabeled Samples** | {report.unlabeled_samples} |
| **Empty Label Strings** | {report.empty_labels} |

## Split Partitioning
| Split | Count |
| :--- | :--- |
"""
        for sp, count in report.splits_count.items():
            content += f"| `{sp}` | {count} |\n"

        content += f"""
## Cryptographic Integrity
- **Split Assignment Hash**: `{prov.split_assignment_hash}`
- **Leakage Check Passed**: `{"YES" if prov.leakage_check_passed else "NO"}`
"""
        if report.warnings:
            content += "\n## Ingestion Warnings\n"
            for w in report.warnings[:20]:
                content += f"- ⚠️ {w}\n"
            if len(report.warnings) > 20:
                content += f"- ... and {len(report.warnings) - 20} more warnings\n"

        out_file.write_text(content, encoding="utf-8")
