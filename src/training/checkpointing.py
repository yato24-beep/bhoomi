"""Model Checkpointing and Training Provenance Tracking.

Provides standardized saving and loading of model weights, processors,
and comprehensive audit metadata (epoch, step, metrics, hyperparameters, device, timestamps).
"""

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class CheckpointMetadata:
    """Metadata recorded with every saved model checkpoint."""
    checkpoint_id: str
    model_name: str
    model_version: str
    languages: List[str]
    epoch: int
    step: int
    training_config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    device: str = "cpu"
    git_commit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        """Instantiates metadata from dictionary."""
        return cls(**data)


def save_training_checkpoint(
    output_dir: Union[str, Path],
    model: Any,
    processor: Optional[Any],
    metadata: CheckpointMetadata,
    is_best: bool = False,
) -> Path:
    """Saves model weights, processor configs, and training metadata to disk.

    Args:
        output_dir: Base directory for checkpoints.
        model: Model object (Hugging Face model or PyTorch nn.Module or mock).
        processor: Optional tokenizer / image processor.
        metadata: CheckpointMetadata instance.
        is_best: If True, also copies/saves into 'best_checkpoint/' directory.

    Returns:
        Path: Directory path of the saved checkpoint.
    """
    base_path = Path(output_dir)
    ckpt_dir = base_path / metadata.checkpoint_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Hugging Face model if supported
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(ckpt_dir))
    elif hasattr(model, "state_dict"):
        try:
            import torch
            torch.save(model.state_dict(), str(ckpt_dir / "pytorch_model.bin"))
        except Exception as save_err:
            logger.error("Failed saving PyTorch state dict: %s", save_err)
            raise

    # 2. Save processor / tokenizer if supported
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(ckpt_dir))

    # 3. Save training metadata JSON
    meta_path = ckpt_dir / "training_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

    # 4. Handle best checkpoint mirror
    if is_best:
        best_dir = base_path / "best_checkpoint"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(ckpt_dir, best_dir)

    return ckpt_dir


def load_training_checkpoint(checkpoint_dir: Union[str, Path]) -> Tuple[Dict[str, Any], Path]:
    """Loads metadata from a saved checkpoint directory.

    Args:
        checkpoint_dir: Path to the saved checkpoint directory.

    Returns:
        Tuple[Dict[str, Any], Path]: Metadata dictionary and verified path.

    Raises:
        FileNotFoundError: If checkpoint directory or metadata file is missing.
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: '{ckpt_path}'")

    meta_file = ckpt_path / "training_metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing 'training_metadata.json' in checkpoint: '{ckpt_path}'")

    with open(meta_file, "r", encoding="utf-8") as f:
        metadata_dict = json.load(f)

    return metadata_dict, ckpt_path
