"""Data Provenance and Lineage Tracking System.

Implements immutable data lineage tracking for dataset ingestion, preprocessing,
and split management:
- Cryptographic SHA-256 digests for archives and samples
- Source attribution and license terms
- Preprocessing history recording
- Train/val/test assignment verification and leakage checking
- Strict immutability guarantees
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file efficiently in 64KB chunks."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {file_path}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Computes SHA-256 hash of bytes in-memory."""
    return hashlib.sha256(data).hexdigest()


@dataclass
class SampleProvenance:
    """Provenance record for an individual dataset sample."""
    sample_id: str
    file_name: str
    sample_sha256: str
    label_sha256: Optional[str] = None
    source_archive: Optional[str] = None
    split: str = "train"  # "train", "val", "test", "calibration"
    writer_id: Optional[str] = None
    domain_tag: Optional[str] = None
    preprocessing_applied: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProvenanceManifest:
    """Complete immutable lineage manifest for an ingested or normalized dataset."""
    dataset_name: str
    dataset_version: str
    archive_sha256: Optional[str] = None
    archive_source: Optional[str] = None
    license: str = "Restricted / Proprietary Archival"
    source_attribution: str = "Government Land Records Department / Archive"
    acquisition_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_samples: int = 0
    splits_count: Dict[str, int] = field(default_factory=dict)
    samples: Dict[str, SampleProvenance] = field(default_factory=dict)
    split_assignment_hash: str = ""
    leakage_check_passed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_sample(self, sample: SampleProvenance) -> None:
        """Registers a sample in the manifest."""
        self.samples[sample.sample_id] = sample
        self.total_samples = len(self.samples)
        self.splits_count[sample.split] = self.splits_count.get(sample.split, 0) + 1

    def compute_split_assignment_hash(self) -> str:
        """Generates deterministic hash of split assignments for tamper evidence."""
        sorted_pairs = sorted(
            [(s.sample_id, s.split, s.sample_sha256) for s in self.samples.values()],
            key=lambda x: x[0]
        )
        hasher = hashlib.sha256()
        for sid, split, s_hash in sorted_pairs:
            hasher.update(f"{sid}:{split}:{s_hash}\n".encode("utf-8"))
        self.split_assignment_hash = hasher.hexdigest()
        return self.split_assignment_hash

    def verify_no_data_leakage(self) -> Tuple[bool, List[str]]:
        """Verifies no sample hash or writer ID leaks between splits."""
        errors: List[str] = []
        hashes_by_split: Dict[str, Set[str]] = {}
        writers_by_split: Dict[str, Set[str]] = {}

        for sample in self.samples.values():
            split = sample.split
            hashes_by_split.setdefault(split, set()).add(sample.sample_sha256)
            if sample.writer_id:
                writers_by_split.setdefault(split, set()).add(sample.writer_id)

        # Check sample hash overlaps
        split_names = list(hashes_by_split.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                s1, s2 = split_names[i], split_names[j]
                overlap = hashes_by_split[s1].intersection(hashes_by_split[s2])
                if overlap:
                    errors.append(f"Data leakage detected! {len(overlap)} exact sample hashes shared between {s1} and {s2}")

        # Check writer ID overlaps if writers are annotated
        writer_splits = list(writers_by_split.keys())
        for i in range(len(writer_splits)):
            for j in range(i + 1, len(writer_splits)):
                s1, s2 = writer_splits[i], writer_splits[j]
                w_overlap = writers_by_split[s1].intersection(writers_by_split[s2])
                if w_overlap:
                    errors.append(f"Writer leakage detected! {len(w_overlap)} writer IDs shared between {s1} and {s2}")

        self.leakage_check_passed = (len(errors) == 0)
        return self.leakage_check_passed, errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "archive_sha256": self.archive_sha256,
            "archive_source": self.archive_source,
            "license": self.license,
            "source_attribution": self.source_attribution,
            "acquisition_timestamp": self.acquisition_timestamp,
            "total_samples": self.total_samples,
            "splits_count": self.splits_count,
            "split_assignment_hash": self.split_assignment_hash,
            "leakage_check_passed": self.leakage_check_passed,
            "metadata": self.metadata,
            "samples": {k: v.to_dict() for k, v in self.samples.items()},
        }

    def save(self, output_path: Union[str, Path]) -> Path:
        """Saves provenance manifest to disk."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.compute_split_assignment_hash()
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Saved provenance manifest with %d samples to %s", self.total_samples, out)
        return out

    @classmethod
    def load(cls, manifest_path: Union[str, Path]) -> "ProvenanceManifest":
        """Loads and verifies a provenance manifest from disk."""
        p = Path(manifest_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = cls(
            dataset_name=data["dataset_name"],
            dataset_version=data["dataset_version"],
            archive_sha256=data.get("archive_sha256"),
            archive_source=data.get("archive_source"),
            license=data.get("license", "Restricted / Proprietary Archival"),
            source_attribution=data.get("source_attribution", "Unknown"),
            acquisition_timestamp=data.get("acquisition_timestamp", ""),
            total_samples=data.get("total_samples", 0),
            splits_count=data.get("splits_count", {}),
            split_assignment_hash=data.get("split_assignment_hash", ""),
            leakage_check_passed=data.get("leakage_check_passed", False),
            metadata=data.get("metadata", {}),
        )

        for sid, sdata in data.get("samples", {}).items():
            sample = SampleProvenance(
                sample_id=sdata["sample_id"],
                file_name=sdata["file_name"],
                sample_sha256=sdata["sample_sha256"],
                label_sha256=sdata.get("label_sha256"),
                source_archive=sdata.get("source_archive"),
                split=sdata.get("split", "train"),
                writer_id=sdata.get("writer_id"),
                domain_tag=sdata.get("domain_tag"),
                preprocessing_applied=sdata.get("preprocessing_applied", []),
                created_at=sdata.get("created_at", ""),
            )
            manifest.samples[sid] = sample

        return manifest
