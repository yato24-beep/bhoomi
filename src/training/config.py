"""Typed Training and Model Configurations for Handwriting OCR Pipeline.

Defines validated configuration dataclasses for datasets, model architectures,
augmentation parameters, and training hyperparameters with YAML serialization.
"""

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class ModelConfig:
    """Configuration for handwriting recognition model architecture."""
    model_name_or_path: str = "microsoft/trocr-small-handwritten"
    tokenizer_name_or_path: Optional[str] = None
    model_version: str = "v1"
    max_sequence_length: int = 64
    image_size: Tuple[int, int] = (384, 384)
    freeze_encoder: bool = False
    use_auth_token: Optional[str] = None


@dataclass
class AugmentationConfig:
    """Configuration for conservative handwritten document augmentation."""
    enabled: bool = True
    rotation_range_deg: float = 3.0
    contrast_range: Tuple[float, float] = (0.85, 1.15)
    brightness_range: Tuple[float, float] = (0.90, 1.10)
    noise_std: float = 0.02
    blur_kernel_prob: float = 0.10


@dataclass
class DatasetConfig:
    """Dataset manifest and split paths."""
    train_manifest: str = "training/datasets/train_kannada.jsonl"
    val_manifest: Optional[str] = "training/datasets/val_kannada.jsonl"
    test_manifest: Optional[str] = "training/datasets/test_kannada.jsonl"
    root_dir: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    validate_images: bool = True
    ignore_missing: bool = False


@dataclass
class TrainingConfig:
    """Comprehensive configuration for the handwriting OCR training pipeline."""
    experiment_name: str = "kannada_handwriting_baseline"
    languages: List[str] = field(default_factory=lambda: ["kannada"])
    output_dir: str = "models/trocr/checkpoints"
    batch_size: int = 4
    eval_batch_size: int = 4
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 50
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3
    mixed_precision: str = "no"  # "no", "fp16", "bf16"
    device: str = "auto"         # "auto", "cpu", "cuda"
    seed: int = 42

    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def validate(self) -> None:
        """Validates configuration parameters, raising ValueError on invalid values."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.eval_batch_size <= 0:
            raise ValueError(f"eval_batch_size must be positive, got {self.eval_batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not self.languages:
            raise ValueError("languages list cannot be empty.")
        if self.model.max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be positive, got {self.model.max_sequence_length}")
        if self.augmentation.rotation_range_deg < 0.0 or self.augmentation.rotation_range_deg > 45.0:
            raise ValueError(
                f"rotation_range_deg should be conservative (0-45 deg), got {self.augmentation.rotation_range_deg}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration dataclass tree to a pure dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Creates a TrainingConfig from a dictionary with nested sub-configs."""
        cfg_dict = dict(data)

        # Parse nested model config
        model_data = cfg_dict.pop("model", {})
        if isinstance(model_data, dict):
            # Convert image_size if loaded as list from yaml
            if "image_size" in model_data and isinstance(model_data["image_size"], list):
                model_data["image_size"] = tuple(model_data["image_size"])
            model_cfg = ModelConfig(**model_data)
        elif isinstance(model_data, ModelConfig):
            model_cfg = model_data
        else:
            model_cfg = ModelConfig()

        # Parse nested augmentation config
        aug_data = cfg_dict.pop("augmentation", {})
        if isinstance(aug_data, dict):
            if "contrast_range" in aug_data and isinstance(aug_data["contrast_range"], list):
                aug_data["contrast_range"] = tuple(aug_data["contrast_range"])
            if "brightness_range" in aug_data and isinstance(aug_data["brightness_range"], list):
                aug_data["brightness_range"] = tuple(aug_data["brightness_range"])
            aug_cfg = AugmentationConfig(**aug_data)
        elif isinstance(aug_data, AugmentationConfig):
            aug_cfg = aug_data
        else:
            aug_cfg = AugmentationConfig()

        # Parse nested dataset config
        ds_data = cfg_dict.pop("dataset", {})
        if isinstance(ds_data, dict):
            ds_cfg = DatasetConfig(**ds_data)
        elif isinstance(ds_data, DatasetConfig):
            ds_cfg = ds_data
        else:
            ds_cfg = DatasetConfig()

        config = cls(
            model=model_cfg,
            augmentation=aug_cfg,
            dataset=ds_cfg,
            **cfg_dict,
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "TrainingConfig":
        """Loads configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: '{path}'")

        if not HAS_YAML:
            raise ImportError("PyYAML is required for loading YAML configs. Install with: pip install pyyaml")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    def save_yaml(self, yaml_path: Union[str, Path]) -> Path:
        """Saves configuration to a YAML file."""
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not HAS_YAML:
            raise ImportError("PyYAML is required for saving YAML configs. Install with: pip install pyyaml")

        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return path
