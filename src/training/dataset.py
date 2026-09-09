"""Multilingual Handwritten Dataset Loader and Manifest Parser.

Supports manifest-based JSONL datasets for multilingual handwriting recognition,
with strict/lenient image verification, Unicode script normalization, and
seamless integration with PyTorch DataLoaders and Hugging Face VisionEncoderDecoder processors.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    HAS_TORCH = True
except ImportError:
    TorchDataset = object
    torch = None
    HAS_TORCH = False


@dataclass
class MultilingualHandwritingSample:
    """Represents a single handwriting training/evaluation instance."""
    image_path: Path
    text: str
    language: str = "kannada"
    script: str = "Kannada"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes sample into manifest dictionary."""
        return {
            "image": str(self.image_path).replace("\\", "/"),
            "text": self.text,
            "language": self.language.lower().strip(),
            "script": self.script,
            "metadata": self.metadata,
        }


def parse_manifest_record(
    record: Union[str, Dict[str, Any]],
    root_dir: Optional[Union[str, Path]] = None,
    line_number: Optional[int] = None,
) -> MultilingualHandwritingSample:
    """Parses a single JSONL record into a MultilingualHandwritingSample.

    Args:
        record: A JSON string or pre-parsed dictionary.
        root_dir: Optional base directory to resolve relative image paths.
        line_number: Optional line number for error reporting.

    Returns:
        MultilingualHandwritingSample: Normalized sample object.

    Raises:
        ValueError: If required fields ('image'/'image_path' and 'text'/'ground_truth') are missing.
    """
    if isinstance(record, str):
        record_str = record.strip()
        if not record_str:
            raise ValueError(f"Empty manifest line{' at line ' + str(line_number) if line_number else ''}")
        try:
            data = json.loads(record_str)
        except json.JSONDecodeError as exc:
            line_info = f" at line {line_number}" if line_number else ""
            raise ValueError(f"Invalid JSON in manifest{line_info}: {str(exc)}") from exc
    elif isinstance(record, dict):
        data = record
    else:
        raise TypeError(f"Expected str or dict for manifest record, got {type(record).__name__}")

    # Resolve image path with alias support
    image_val = data.get("image") or data.get("image_path")
    if not image_val or not isinstance(image_val, str):
        line_info = f" at line {line_number}" if line_number else ""
        raise ValueError(f"Manifest record{line_info} missing required 'image' or 'image_path' string field.")

    # Resolve text with alias support
    text_val = data.get("text")
    if text_val is None:
        text_val = data.get("ground_truth")
    if text_val is None:
        line_info = f" at line {line_number}" if line_number else ""
        raise ValueError(f"Manifest record{line_info} missing required 'text' or 'ground_truth' field.")
    text_val = str(text_val)

    # Resolve image path against root_dir
    img_path = Path(image_val)
    if root_dir is not None and not img_path.is_absolute():
        img_path = Path(root_dir) / img_path

    # Language and script defaults
    language_val = str(data.get("language", "kannada")).lower().strip()
    script_val = str(data.get("script", language_val.capitalize()))
    metadata_val = data.get("metadata", {})
    if not isinstance(metadata_val, dict):
        metadata_val = {"raw_metadata": metadata_val}

    return MultilingualHandwritingSample(
        image_path=img_path,
        text=text_val,
        language=language_val,
        script=script_val,
        metadata=metadata_val,
    )


class HandwritingDataset(TorchDataset):
    """Multilingual Dataset for handwriting OCR training and evaluation.

    Inherits from torch.utils.data.Dataset when PyTorch is available,
    while remaining fully functional as an iterable container otherwise.
    """

    def __init__(
        self,
        manifest_path: Optional[Union[str, Path]] = None,
        samples: Optional[Sequence[MultilingualHandwritingSample]] = None,
        root_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        processor: Optional[Any] = None,
        max_target_length: int = 128,
        max_samples: Optional[int] = None,
        validate_images: bool = True,
        ignore_missing: bool = False,
    ):
        """Initializes the dataset from a JSONL manifest or sample list.

        Args:
            manifest_path: Path to JSONL manifest file.
            samples: Direct list of MultilingualHandwritingSample objects.
            root_dir: Base directory for relative image paths.
            transform: Optional image augmentation callable (e.g. HandwritingAugmentor).
            processor: Optional Hugging Face processor (e.g. TrOCRProcessor).
            max_target_length: Max sequence token length for processor labels.
            max_samples: Optional maximum number of samples to load (useful for fast smoke tests).
            validate_images: If True, checks that image files exist on disk.
            ignore_missing: If True and validate_images is True, skips missing images instead of raising.
        """
        self.root_dir = Path(root_dir) if root_dir else None
        self.transform = transform
        self.processor = processor
        self.max_target_length = max_target_length
        self.max_samples = max_samples
        self.samples: List[MultilingualHandwritingSample] = []
        self.missing_samples: List[MultilingualHandwritingSample] = []

        if samples is not None:
            raw_samples = list(samples)
        elif manifest_path is not None:
            raw_samples = self._load_manifest(Path(manifest_path))
        else:
            raw_samples = []

        if max_samples is not None and max_samples > 0:
            raw_samples = raw_samples[:max_samples]

        # Validate existence if requested
        for s in raw_samples:
            if validate_images:
                if not s.image_path.exists():
                    self.missing_samples.append(s)
                    if not ignore_missing:
                        raise FileNotFoundError(
                            f"Dataset image file not found: '{s.image_path}' for sample text: '{s.text}'"
                        )
                    continue
            self.samples.append(s)

    def _load_manifest(self, manifest_file: Path) -> List[MultilingualHandwritingSample]:
        """Loads and parses a JSONL manifest file."""
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest file not found: '{manifest_file}'")

        loaded: List[MultilingualHandwritingSample] = []
        base_dir = self.root_dir or manifest_file.parent

        with open(manifest_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue
                sample = parse_manifest_record(clean_line, root_dir=base_dir, line_number=line_idx)
                loaded.append(sample)

        return loaded

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Returns a single processed training item."""
        sample = self.samples[idx]

        # Load image safely
        try:
            image = Image.open(sample.image_path).convert("RGB")
        except Exception as exc:
            raise IOError(f"Failed to open image at '{sample.image_path}': {str(exc)}") from exc

        # Apply augmentation if specified
        if self.transform is not None:
            image = self.transform(image)

        item: Dict[str, Any] = {
            "image": image,
            "text": sample.text,
            "language": sample.language,
            "script": sample.script,
            "image_path": str(sample.image_path),
            "metadata": sample.metadata,
        }

        # If Hugging Face processor is attached, format tensor features
        if self.processor is not None:
            try:
                # Processor extracts image pixel_values
                pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
                item["pixel_values"] = pixel_values.squeeze(0) if HAS_TORCH else pixel_values

                # Tokenizer encodes ground truth text into labels
                labels = self.processor.tokenizer(
                    sample.text,
                    padding="max_length",
                    max_length=self.max_target_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids

                if HAS_TORCH:
                    labels = labels.squeeze(0)
                    pad_token_id = self.processor.tokenizer.pad_token_id or 1
                    # Replace padding token id with -100 so PyTorch CrossEntropyLoss ignores it
                    labels[labels == pad_token_id] = -100
                item["labels"] = labels
            except Exception as proc_exc:
                item["processor_error"] = str(proc_exc)

        return item

    def filter_by_language(self, language: str) -> "HandwritingDataset":
        """Returns a subset dataset containing only samples of the specified language."""
        target_lang = language.lower().strip()
        matched = [s for s in self.samples if s.language == target_lang]
        return HandwritingDataset(
            samples=matched,
            root_dir=self.root_dir,
            transform=self.transform,
            processor=self.processor,
            max_target_length=self.max_target_length,
            validate_images=False,
        )

    def get_languages(self) -> List[str]:
        """Returns list of distinct languages present in the dataset."""
        return sorted(list({s.language for s in self.samples}))


def create_split_datasets(
    train_manifest: Union[str, Path],
    val_manifest: Optional[Union[str, Path]] = None,
    test_manifest: Optional[Union[str, Path]] = None,
    root_dir: Optional[Union[str, Path]] = None,
    train_transform: Optional[Callable[[Image.Image], Image.Image]] = None,
    processor: Optional[Any] = None,
    max_target_length: int = 128,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    validate_images: bool = True,
    ignore_missing: bool = False,
) -> Dict[str, HandwritingDataset]:
    """Factory creating train/val/test HandwritingDataset instances from manifest paths."""
    datasets: Dict[str, HandwritingDataset] = {}

    datasets["train"] = HandwritingDataset(
        manifest_path=train_manifest,
        root_dir=root_dir,
        transform=train_transform,
        processor=processor,
        max_target_length=max_target_length,
        max_samples=max_train_samples,
        validate_images=validate_images,
        ignore_missing=ignore_missing,
    )

    if val_manifest:
        datasets["val"] = HandwritingDataset(
            manifest_path=val_manifest,
            root_dir=root_dir,
            transform=None,  # No augmentation during validation
            processor=processor,
            max_target_length=max_target_length,
            max_samples=max_val_samples,
            validate_images=validate_images,
            ignore_missing=ignore_missing,
        )

    if test_manifest:
        datasets["test"] = HandwritingDataset(
            manifest_path=test_manifest,
            root_dir=root_dir,
            transform=None,  # No augmentation during testing
            processor=processor,
            max_target_length=max_target_length,
            max_samples=max_test_samples,
            validate_images=validate_images,
            ignore_missing=ignore_missing,
        )

    return datasets
