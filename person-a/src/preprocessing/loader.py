"""person-a/src/preprocessing/loader.py
Multi-modal document ingestion and rasterization.
Handles PDFs, PNGs, JPGs, TIFFs, in-memory bytes, streams, and numpy arrays.
"""

import io
from pathlib import Path
from typing import BinaryIO, List, Tuple, Union
import cv2
import numpy as np
from PIL import Image

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


def load_document_images(
    source: Union[str, Path, bytes, BinaryIO, np.ndarray, Image.Image],
    target_dpi: int = 300,
) -> List[np.ndarray]:
    """Ingest document from various input representations and return list of BGR page images."""
    if isinstance(source, np.ndarray):
        img = source.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return [img]

    if isinstance(source, Image.Image):
        rgb_img = np.array(source.convert("RGB"))
        return [cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)]

    raw_bytes: bytes
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.is_file():
            raise FileNotFoundError(f"Input file does not exist: {p}")
        raw_bytes = p.read_bytes()
    elif isinstance(source, bytes):
        raw_bytes = source
    elif hasattr(source, "read"):
        raw_bytes = source.read()
    else:
        raise TypeError(f"Unsupported document source type: {type(source)}")

    if not raw_bytes:
        return []

    # 1. Check if PDF
    if raw_bytes.startswith(b"%PDF") and fitz is not None:
        pages: List[np.ndarray] = []
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            zoom = target_dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for page in doc:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif pix.n == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                pages.append(img)
            doc.close()
            return pages
        except Exception:
            pass

    # 2. Decode raster image via OpenCV / PIL
    nparr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return [img]

    # Fallback to PIL
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pages = []
        try:
            for frame_idx in range(getattr(pil_img, "n_frames", 1)):
                pil_img.seek(frame_idx)
                rgb = np.array(pil_img.convert("RGB"))
                pages.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except EOFError:
            pass
        return pages
    except Exception:
        pass

    return []
