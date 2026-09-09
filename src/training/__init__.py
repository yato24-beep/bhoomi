"""Multilingual Handwriting Recognition Training and Dataset Management Package.

Exports:
- Dataset & Manifest Loading: HandwritingDataset, MultilingualHandwritingSample, create_split_datasets, parse_manifest_record
- Dataset Bootstrapping: bootstrap_multilingual_datasets, scan_language_directory, generate_annotation_template, ensure_dataset_directories, LanguageScanResult, BootstrapSummary, LANGUAGE_SCRIPT_MAP
- Data Preparation & Validation: validate_manifest, validate_image_file, clean_and_save_manifest, ValidationReport, ValidationIssue, normalize_unicode_text
- Dataset Splitting: split_dataset_samples, split_and_save_manifests, DatasetSplitResult
- Dataset Readiness Audit: run_dataset_audit, DatasetAuditReport, DatasetSufficiency, classify_sample_text_type
- Configuration: TrainingConfig, ModelConfig, AugmentationConfig, DatasetConfig
- Augmentation: HandwritingAugmentor, small_rotation, adjust_contrast, adjust_brightness, add_mild_gaussian_noise
- Training: HandwritingTrainer, default_collate_fn
- Evaluation: compute_cer, compute_wer, evaluate_predictions, EvaluationReport, LanguageMetrics
- Checkpointing: CheckpointMetadata, save_training_checkpoint, load_training_checkpoint
"""

from src.training.augmentation import (
    HandwritingAugmentor,
    add_mild_gaussian_noise,
    adjust_brightness,
    adjust_contrast,
    small_rotation,
)
from src.training.bootstrap import (
    LANGUAGE_SCRIPT_MAP,
    BootstrapSummary,
    LanguageScanResult,
    bootstrap_multilingual_datasets,
    ensure_dataset_directories,
    generate_annotation_template,
    scan_language_directory,
)
from src.training.checkpointing import (
    CheckpointMetadata,
    load_training_checkpoint,
    save_training_checkpoint,
)
from src.training.config import (
    AugmentationConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from src.training.data_preparation import (
    ValidationIssue,
    ValidationReport,
    clean_and_save_manifest,
    normalize_unicode_text,
    validate_image_file,
    validate_manifest,
)
from src.training.dataset import (
    HandwritingDataset,
    MultilingualHandwritingSample,
    create_split_datasets,
    parse_manifest_record,
)
from src.training.dataset_audit import (
    DatasetAuditReport,
    DatasetSufficiency,
    classify_sample_text_type,
    run_dataset_audit,
)
from src.training.evaluate import (
    EvaluationReport,
    LanguageMetrics,
    compute_cer,
    compute_wer,
    evaluate_predictions,
    levenshtein_distance,
)
from src.training.splitter import (
    DatasetSplitResult,
    split_and_save_manifests,
    split_dataset_samples,
)
from src.training.importers import (
    IIITIndicHWImporter,
    ImportIssue,
    ImportReport,
)
from src.training.trainer import (
    HandwritingTrainer,
    default_collate_fn,
)

__all__ = [
    # Dataset and samples
    "MultilingualHandwritingSample",
    "HandwritingDataset",
    "parse_manifest_record",
    "create_split_datasets",
    # Importers
    "IIITIndicHWImporter",
    "ImportReport",
    "ImportIssue",

    # Dataset bootstrapping
    "LANGUAGE_SCRIPT_MAP",
    "ensure_dataset_directories",
    "scan_language_directory",
    "generate_annotation_template",
    "bootstrap_multilingual_datasets",
    "LanguageScanResult",
    "BootstrapSummary",
    # Data preparation & validation
    "ValidationIssue",
    "ValidationReport",
    "validate_image_file",
    "normalize_unicode_text",
    "validate_manifest",
    "clean_and_save_manifest",
    # Splitting
    "DatasetSplitResult",
    "split_dataset_samples",
    "split_and_save_manifests",
    # Dataset readiness audit
    "DatasetAuditReport",
    "DatasetSufficiency",
    "classify_sample_text_type",
    "run_dataset_audit",
    # Configuration
    "ModelConfig",
    "AugmentationConfig",
    "DatasetConfig",
    "TrainingConfig",
    # Augmentation
    "HandwritingAugmentor",
    "small_rotation",
    "adjust_contrast",
    "adjust_brightness",
    "add_mild_gaussian_noise",
    # Trainer
    "HandwritingTrainer",
    "default_collate_fn",
    # Evaluation
    "levenshtein_distance",
    "compute_cer",
    "compute_wer",
    "evaluate_predictions",
    "EvaluationReport",
    "LanguageMetrics",
    # Checkpointing
    "CheckpointMetadata",
    "save_training_checkpoint",
    "load_training_checkpoint",
]
