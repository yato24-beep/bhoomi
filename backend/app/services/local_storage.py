import os
import re
from pathlib import Path
from typing import BinaryIO
import shutil

from app.config import settings


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and invalid path characters."""
    # Keep only base filename
    clean_name = os.path.basename(filename)
    # Replace unsafe characters with underscores
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_name)
    return clean_name or "document.bin"


def save_file_locally(file_stream: BinaryIO, filename: str, file_hash: str) -> str:
    """Save a binary file stream to the local uploads directory.
    
    Args:
        file_stream: Open binary stream of the uploaded file.
        filename: Original user filename.
        file_hash: SHA-256 hash string used as part of the unique storage name.
        
    Returns:
        The relative storage path of the saved file.
    """
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    clean_filename = sanitize_filename(filename)
    stored_filename = f"{file_hash[:16]}_{clean_filename}"
    target_path = upload_dir / stored_filename

    # Ensure reading from beginning
    file_stream.seek(0)

    # Write stream to disk in 64KB chunks
    with open(target_path, "wb") as dest:
        shutil.copyfileobj(file_stream, dest, length=65536)

    # Reset stream pointer
    file_stream.seek(0)

    # Return standard forward-slash relative path
    return target_path.as_posix()


def get_file_path(storage_path: str) -> Path:
    """Resolve and check existence of a local storage file path."""
    path = Path(storage_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found at {storage_path}")
    return path
