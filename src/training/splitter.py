"""Dataset Splitter for Multilingual Handwriting OCR.

Provides deterministic, stratified splitting of handwriting dataset samples
into train, validation, and test splits while preserving language distribution.
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from src.training.data_preparation import validate_manifest
from src.training.dataset import MultilingualHandwritingSample


@dataclass
class DatasetSplitResult:
    """Represents the outcome of a dataset partitioning operation."""
    train_samples: List[MultilingualHandwritingSample] = field(default_factory=list)
    val_samples: List[MultilingualHandwritingSample] = field(default_factory=list)
    test_samples: List[MultilingualHandwritingSample] = field(default_factory=list)
    language_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    seed: int = 42
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)

    @property
    def total_samples(self) -> int:
        return len(self.train_samples) + len(self.val_samples) + len(self.test_samples)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "train_count": len(self.train_samples),
            "val_count": len(self.val_samples),
            "test_count": len(self.test_samples),
            "ratios": list(self.ratios),
            "seed": self.seed,
            "language_distribution": self.language_distribution,
        }


def _split_group(
    items: List[MultilingualHandwritingSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[MultilingualHandwritingSample], List[MultilingualHandwritingSample], List[MultilingualHandwritingSample]]:
    """Splits a single list of samples deterministically into 3 partitions."""
    shuffled = list(items)
    rng.shuffle(shuffled)

    n = len(shuffled)
    if n == 0:
        return [], [], []

    if n == 1:
        # Single sample defaults to train
        return shuffled, [], []

    if n == 2:
        # Two samples split into train and val
        return [shuffled[0]], [shuffled[1]], []

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    # Ensure valid non-empty train split
    n_train = max(1, min(n - 1, n_train))
    remaining = n - n_train

    if val_ratio > 0.0 and remaining > 1 and test_ratio > 0.0:
        n_val = max(1, min(remaining - 1, n_val))
    elif val_ratio > 0.0:
        n_val = remaining
    else:
        n_val = 0

    n_test = n - n_train - n_val

    train_part = shuffled[:n_train]
    val_part = shuffled[n_train : n_train + n_val]
    test_part = shuffled[n_train + n_val :]

    return train_part, val_part, test_part


def split_dataset_samples(
    samples: Sequence[MultilingualHandwritingSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_by_language: bool = True,
) -> DatasetSplitResult:
    """Splits samples into train, validation, and test subsets.

    Args:
        samples: Sequence of input MultilingualHandwritingSample instances.
        train_ratio: Proportion for training set (e.g. 0.8).
        val_ratio: Proportion for validation set (e.g. 0.1).
        test_ratio: Proportion for test set (e.g. 0.1).
        seed: Random seed for deterministic reproducibility.
        stratify_by_language: If True, balances splits per language.

    Returns:
        DatasetSplitResult: Subsets and language distribution breakdown.

    Raises:
        ValueError: If ratios do not sum to approximately 1.0 or are negative.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        raise ValueError(
            f"Split ratios must sum to 1.0, got train={train_ratio}, val={val_ratio}, test={test_ratio} (sum={total_ratio})"
        )
    if train_ratio < 0.0 or val_ratio < 0.0 or test_ratio < 0.0:
        raise ValueError("Split ratios cannot be negative.")

    rng = random.Random(seed)
    train_out: List[MultilingualHandwritingSample] = []
    val_out: List[MultilingualHandwritingSample] = []
    test_out: List[MultilingualHandwritingSample] = []

    if stratify_by_language:
        # Group by language
        lang_groups: Dict[str, List[MultilingualHandwritingSample]] = defaultdict(list)
        for s in samples:
            lang_groups[s.language].append(s)

        for lang, group_items in sorted(lang_groups.items()):
            tr, va, te = _split_group(group_items, train_ratio, val_ratio, test_ratio, rng)
            train_out.extend(tr)
            val_out.extend(va)
            test_out.extend(te)
    else:
        train_out, val_out, test_out = _split_group(list(samples), train_ratio, val_ratio, test_ratio, rng)

    # Compute language distribution breakdown
    lang_dist: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})
    for s in train_out:
        lang_dist[s.language]["train"] += 1
        lang_dist[s.language]["total"] += 1
    for s in val_out:
        lang_dist[s.language]["val"] += 1
        lang_dist[s.language]["total"] += 1
    for s in test_out:
        lang_dist[s.language]["test"] += 1
        lang_dist[s.language]["total"] += 1

    return DatasetSplitResult(
        train_samples=train_out,
        val_samples=val_out,
        test_samples=test_out,
        language_distribution=dict(lang_dist),
        seed=seed,
        ratios=(train_ratio, val_ratio, test_ratio),
    )


def split_and_save_manifests(
    input_manifest: Union[str, Path],
    output_dir: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    prefix: str = "",
    root_dir: Optional[Union[str, Path]] = None,
    verify_images: bool = True,
    stratify_by_language: bool = True,
) -> Tuple[DatasetSplitResult, Dict[str, Path]]:
    """Validates an input manifest, splits it into train/val/test, and writes JSONL files.

    Args:
        input_manifest: Source JSONL manifest path.
        output_dir: Target directory for manifest files.
        train_ratio: Training proportion.
        val_ratio: Validation proportion.
        test_ratio: Test proportion.
        seed: Random seed.
        prefix: Optional prefix for filenames (e.g. 'kannada_').
        root_dir: Base directory for relative paths.
        verify_images: Whether to verify images during validation.
        stratify_by_language: Stratify across languages.

    Returns:
        Tuple[DatasetSplitResult, Dict[str, Path]]: Partition results and output file paths.
    """
    valid_samples, val_report = validate_manifest(
        manifest_path=input_manifest,
        root_dir=root_dir,
        verify_images=verify_images,
    )

    if not valid_samples:
        raise ValueError(
            f"Cannot split manifest '{input_manifest}': 0 valid samples found ({val_report.error_count} errors)."
        )

    split_result = split_dataset_samples(
        samples=valid_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify_by_language=stratify_by_language,
    )

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    # Write train manifest
    p_train = out_base / f"{prefix}train.jsonl"
    with open(p_train, "w", encoding="utf-8") as f:
        for s in split_result.train_samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    paths["train"] = p_train

    # Write val manifest if non-empty
    if split_result.val_samples:
        p_val = out_base / f"{prefix}val.jsonl"
        with open(p_val, "w", encoding="utf-8") as f:
            for s in split_result.val_samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        paths["val"] = p_val

    # Write test manifest if non-empty
    if split_result.test_samples:
        p_test = out_base / f"{prefix}test.jsonl"
        with open(p_test, "w", encoding="utf-8") as f:
            for s in split_result.test_samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        paths["test"] = p_test

    return split_result, paths
