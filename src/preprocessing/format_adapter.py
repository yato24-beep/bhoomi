"""Unified Document Ingestion & Format Adapter for Land Record Digitization.

Normalizes input files across all supported document and image formats:
- PDF (via pypdfium2 high-resolution rendering at 300 DPI equivalent)
- TIFF / Multi-page TIFF (frame extraction)
- PNG, JPG, JPEG, BMP (native PIL decoding with EXIF transpose)
- HEIF / HEIC (via pillow_heif registered opener)

Guarantees a unified return type of List[PIL.Image.Image] (RGB, upright).
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger("format_adapter")

# Register HEIF opener if available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HAS_HEIF = True
except Exception as _heif_err:
    logger.debug("pillow_heif not available: %s", _heif_err)
    HAS_HEIF = False

# Soft import pypdfium2
try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except ImportError:
    pdfium = None
    HAS_PDFIUM = False


class DocumentIngestionError(Exception):
    """Structured exception raised when document ingestion or page rendering fails."""
    def __init__(self, message: str, error_code: str = "CORRUPT_OR_UNSUPPORTED_FILE", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class DocumentFormatAdapter:
    """Unified ingest adapter that converts any document/image source to clean PIL Images."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".heif", ".heic"
    }

    @classmethod
    def is_pdf(cls, file_bytes: bytes, filename: Optional[str] = None) -> bool:
        """Determines if the raw payload or filename represents a PDF."""
        if filename and filename.lower().endswith(".pdf"):
            return True
        return b"%PDF" in file_bytes[:1024]

    @classmethod
    def is_tiff(cls, file_bytes: bytes, filename: Optional[str] = None) -> bool:
        """Determines if the raw payload or filename represents a TIFF."""
        if filename and any(filename.lower().endswith(ext) for ext in (".tiff", ".tif")):
            return True
        return file_bytes.startswith(b"II*\x00") or file_bytes.startswith(b"MM\x00*")

    @classmethod
    def load_pages(
        cls,
        source: Union[bytes, io.BytesIO, str, Path, Image.Image, np.ndarray],
        filename: Optional[str] = None,
        dpi_scale: float = 2.0,  # 2.0 = ~144-300 DPI rendering for PDFs
    ) -> List[Image.Image]:
        """Ingests arbitrary document input and returns a list of normalized RGB PIL Image pages.

        Args:
            source: Raw file bytes, BytesIO, filepath, existing PIL Image, or NumPy array.
            filename: Optional filename hint to assist format detection.
            dpi_scale: Scaling factor for PDF rasterization (2.0 gives ~300 DPI).

        Returns:
            List of PIL.Image.Image instances in RGB mode with EXIF orientation corrected.
        """
        # Case 1: Already a PIL Image
        if isinstance(source, Image.Image):
            return [cls._normalize_single_image(source)]

        # Case 2: NumPy array
        if isinstance(source, np.ndarray):
            pil_img = Image.fromarray(source)
            return [cls._normalize_single_image(pil_img)]

        # Case 3: Read bytes if source is path or file-like
        raw_bytes: bytes = b""
        resolved_name: Optional[str] = filename

        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"Input document file not found at: {source}")
            resolved_name = resolved_name or p.name
            with open(p, "rb") as f:
                raw_bytes = f.read()
        elif isinstance(source, io.BytesIO):
            raw_bytes = source.getvalue()
        elif isinstance(source, bytes):
            raw_bytes = source
        else:
            # Handle stream-like objects
            if hasattr(source, "read"):
                source.seek(0)
                raw_bytes = source.read()
                source.seek(0)
            else:
                raise ValueError(f"Unsupported document source type: {type(source)}")

        if not raw_bytes:
            raise DocumentIngestionError("Input document payload is empty (0 bytes).", error_code="EMPTY_PAYLOAD")

        # Case 4: PDF Document
        if cls.is_pdf(raw_bytes, resolved_name):
            try:
                return cls._render_pdf_pages(raw_bytes, dpi_scale=dpi_scale)
            except Exception as pdf_err:
                raise DocumentIngestionError(f"PDF rendering failed: {pdf_err}", error_code="PDF_RENDER_FAILED") from pdf_err

        # Case 5: TIFF (potentially multi-page)
        if cls.is_tiff(raw_bytes, resolved_name):
            try:
                return cls._load_tiff_pages(raw_bytes)
            except Exception as tiff_err:
                raise DocumentIngestionError(f"TIFF extraction failed: {tiff_err}", error_code="TIFF_EXTRACTION_FAILED") from tiff_err

        # Case 6: Standard raster image (PNG, JPG, BMP, WEBP, HEIF)
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            pil_img.load()
            return [cls._normalize_single_image(pil_img)]
        except Exception as img_exc:
            if raw_bytes.startswith(b"%PDF") and HAS_PDFIUM:
                try:
                    return cls._render_pdf_pages(raw_bytes, dpi_scale=dpi_scale)
                except Exception:
                    pass
            raise DocumentIngestionError(
                f"Failed to decode document image format: {img_exc}",
                error_code="CORRUPT_OR_UNSUPPORTED_FILE",
                details={"filename": resolved_name, "raw_byte_len": len(raw_bytes)}
            ) from img_exc

    @classmethod
    def _render_pdf_pages(cls, pdf_bytes: bytes, dpi_scale: float = 2.0) -> List[Image.Image]:
        """Renders all pages of a PDF document into high-resolution RGB PIL Images."""
        if not HAS_PDFIUM or pdfium is None:
            raise RuntimeError("pypdfium2 is required for PDF document ingestion but is not installed.")

        pages: List[Image.Image] = []
        doc = None
        try:
            doc = pdfium.PdfDocument(pdf_bytes)
            page_count = len(doc)
            if page_count == 0:
                raise ValueError("PDF document contains 0 pages.")
            for page_idx in range(page_count):
                page = doc[page_idx]
                # Render page to bitmap at designated resolution scale
                bitmap = page.render(scale=dpi_scale)
                pil_img = bitmap.to_pil()
                pages.append(cls._normalize_single_image(pil_img))
            logger.info("Rendered %d PDF page(s) at scale=%.1f", len(pages), dpi_scale)
            return pages
        except Exception as pdf_exc:
            raise ValueError(f"Failed to render PDF document: {pdf_exc}") from pdf_exc
        finally:
            if doc is not None and hasattr(doc, "close"):
                try:
                    doc.close()
                except Exception:
                    pass

    @classmethod
    def _load_tiff_pages(cls, tiff_bytes: bytes) -> List[Image.Image]:
        """Extracts all frames/pages from a multi-page TIFF file."""
        pages: List[Image.Image] = []
        try:
            tiff_img = Image.open(io.BytesIO(tiff_bytes))
            frame = 0
            while True:
                try:
                    tiff_img.seek(frame)
                    # Force copy to detach from the open stream
                    frame_copy = tiff_img.copy()
                    pages.append(cls._normalize_single_image(frame_copy))
                    frame += 1
                except EOFError:
                    break
            logger.info("Extracted %d frame(s) from TIFF document", len(pages))
            return pages
        except Exception as tiff_exc:
            raise ValueError(f"Failed to extract TIFF frames: {tiff_exc}") from tiff_exc

    @classmethod
    def _normalize_single_image(cls, img: Image.Image) -> Image.Image:
        """Applies EXIF transpose normalization and ensures standard 24-bit RGB format."""
        try:
            # Transpose according to EXIF orientation tag (handles mobile phone camera orientation)
            img = ImageOps.exif_transpose(img)
        except Exception as exif_err:
            logger.debug("EXIF transpose notice: %s", exif_err)

        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
