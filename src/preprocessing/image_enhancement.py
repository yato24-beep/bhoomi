"""Image preprocessing and enhancement for scanned and handwritten archival documents.

Includes grayscale conversion, contrast enhancement, light noise reduction,
deskewing, and optional adaptive thresholding designed to preserve faint handwriting strokes.
"""

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Attempt OpenCV import; fallback to pure PIL/NumPy when not installed
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False


@dataclass
class PreprocessingResult:
    """Structured result of image preprocessing containing processed image and audit trail."""
    image: Image.Image
    audit_metadata: Dict[str, Any] = field(default_factory=dict)
    original_size: Tuple[int, int] = (0, 0)
    processed_size: Tuple[int, int] = (0, 0)


def load_image_as_pil(image_input: Union[str, Path, Image.Image, np.ndarray, bytes]) -> Image.Image:
    """Standardizes various image input types into a PIL Image.

    Args:
        image_input: File path (str/Path), PIL Image, NumPy array, or byte buffer.

    Returns:
        PIL Image instance.

    Raises:
        ValueError: If the input cannot be decoded into an image.
    """
    if isinstance(image_input, Image.Image):
        return image_input.copy()

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        return Image.open(path).copy()

    if isinstance(image_input, np.ndarray):
        # Handle grayscale vs BGR/RGB
        if image_input.ndim == 2:
            return Image.fromarray(image_input, mode="L")
        elif image_input.ndim == 3:
            if image_input.shape[2] == 3:
                # Assume RGB
                return Image.fromarray(image_input, mode="RGB")
            elif image_input.shape[2] == 4:
                return Image.fromarray(image_input, mode="RGBA")
        raise ValueError(f"Unsupported NumPy image shape: {image_input.shape}")

    if isinstance(image_input, bytes):
        import io
        return Image.open(io.BytesIO(image_input)).copy()

    raise TypeError(f"Unsupported image input type: {type(image_input).__name__}")


def to_grayscale(image: Image.Image) -> Image.Image:
    """Converts image to 8-bit grayscale mode 'L' handling alpha channels cleanly."""
    if image.mode == "L":
        return image.copy()
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        # Flatten alpha against a clean white background
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        converted = image.convert("RGBA")
        composite = Image.alpha_composite(bg, converted)
        return composite.convert("L")
    return image.convert("L")


def enhance_contrast(
    image: Image.Image,
    contrast_factor: float = 1.3,
    auto_contrast_cutoff: float = 0.5,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Enhances image contrast for archival documents without blowing out faint text.

    Applies mild percentile-based auto-contrast followed by controlled factor scaling.

    Args:
        image: Grayscale PIL image.
        contrast_factor: Multiplier for ImageEnhance (1.0 = no change, >1.0 = higher contrast).
        auto_contrast_cutoff: Percentile cutoff for histogram stretching.

    Returns:
        Tuple of (enhanced_image, audit_metadata).
    """
    gray = to_grayscale(image)
    stretched = ImageOps.autocontrast(gray, cutoff=auto_contrast_cutoff)

    if contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(stretched)
        enhanced = enhancer.enhance(contrast_factor)
    else:
        enhanced = stretched

    metadata = {
        "contrast_factor": contrast_factor,
        "auto_contrast_cutoff": auto_contrast_cutoff,
    }
    return enhanced, metadata


def light_denoise(
    image: Image.Image,
    filter_type: str = "median",
    kernel_size: int = 3,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Applies gentle noise reduction that preserves thin pen/pencil strokes.

    Args:
        image: Grayscale PIL image.
        filter_type: 'median' (recommended for salt-and-pepper scan noise) or 'gaussian'.
        kernel_size: Filter diameter / radius size (3 is optimal for preserving handwriting).

    Returns:
        Tuple of (denoised_image, audit_metadata).
    """
    gray = to_grayscale(image)

    if filter_type == "median":
        # PIL median filter size must be 3 or 5
        size = 3 if kernel_size <= 3 else 5
        denoised = gray.filter(ImageFilter.MedianFilter(size=size))
    elif filter_type == "gaussian":
        radius = max(0.5, (kernel_size - 1) / 4.0)
        denoised = gray.filter(ImageFilter.GaussianBlur(radius=radius))
    else:
        denoised = gray
        filter_type = "none"

    metadata = {
        "denoise_filter": filter_type,
        "kernel_size": kernel_size,
    }
    return denoised, metadata


def estimate_skew_angle(image: Image.Image, max_angle: float = 15.0) -> float:
    """Estimates document skew angle in degrees using projection profile variance.

    Works robustly on NumPy arrays without requiring external heavy dependencies.

    Args:
        image: Grayscale PIL image.
        max_angle: Maximum allowable rotation angle in degrees (+/-).

    Returns:
        Estimated skew angle in degrees (positive = clockwise skew).
    """
    gray = to_grayscale(image)
    # Downscale for fast angle estimation if image is very large
    w, h = gray.size
    if max(w, h) > 800:
        scale = 800.0 / max(w, h)
        thumb = gray.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    else:
        thumb = gray

    arr = np.array(thumb, dtype=np.float32)
    # Binarize approximately for projection analysis (dark ink has lower value)
    inv_arr = 255.0 - arr

    best_score = -1.0
    best_angle = 0.0

    # Search in 0.5 degree steps
    angles = np.arange(-max_angle, max_angle + 0.5, 0.5)

    for angle in angles:
        if abs(angle) < 0.1:
            rot_arr = inv_arr
        else:
            rot_img = Image.fromarray(inv_arr).rotate(
                float(angle),
                resample=Image.Resampling.BILINEAR,
                expand=False,
                fillcolor=0,
            )
            rot_arr = np.array(rot_img)

        # Horizontal projection profile: sum along width
        proj = np.sum(rot_arr, axis=1)
        # Variance of projection profile peaks when lines are horizontally aligned
        score = float(np.var(proj))

        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return round(best_angle, 2)


def deskew_image(
    image: Image.Image,
    max_angle: float = 15.0,
    min_angle_threshold: float = 0.5,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Detects and corrects document skew while auditing the transformation.

    Args:
        image: PIL image.
        max_angle: Maximum allowable angle to correct.
        min_angle_threshold: Skew below this threshold is considered negligible.

    Returns:
        Tuple of (deskewed_image, audit_metadata).
    """
    angle = estimate_skew_angle(image, max_angle=max_angle)

    if abs(angle) >= min_angle_threshold:
        # Rotate image to cancel the skew; fill new border with white (255)
        deskewed = image.rotate(
            -angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=255 if image.mode == "L" else (255, 255, 255),
        )
        corrected = True
    else:
        deskewed = image.copy()
        corrected = False

    metadata = {
        "detected_skew_angle": angle,
        "deskew_applied": corrected,
        "rotation_applied_deg": -angle if corrected else 0.0,
    }
    return deskewed, metadata


def adaptive_soft_threshold(
    image: Image.Image,
    block_size: int = 21,
    constant: int = 10,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Applies adaptive local thresholding for binarization when explicitly needed.

    Args:
        image: Grayscale PIL image.
        block_size: Neighborhood block size.
        constant: Subtraction constant.

    Returns:
        Tuple of (binarized_image, audit_metadata).
    """
    gray = to_grayscale(image)

    if HAS_CV2 and cv2 is not None:
        arr = np.array(gray)
        b_size = block_size if block_size % 2 == 1 else block_size + 1
        bin_arr = cv2.adaptiveThreshold(
            arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b_size, constant
        )
        out_img = Image.fromarray(bin_arr, mode="L")
        method = "cv2_adaptive_gaussian"
    else:
        # NumPy-based local mean approximation using BoxBlur
        blurred = gray.filter(ImageFilter.BoxBlur(radius=block_size // 2))
        arr_gray = np.array(gray, dtype=np.int16)
        arr_blur = np.array(blurred, dtype=np.int16)
        bin_arr = np.where(arr_gray < (arr_blur - constant), 0, 255).astype(np.uint8)
        out_img = Image.fromarray(bin_arr, mode="L")
        method = "numpy_boxblur_adaptive"

    metadata = {
        "threshold_method": method,
        "block_size": block_size,
        "constant": constant,
    }
    return out_img, metadata


def preprocess_document_image(
    image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
    apply_deskew: bool = True,
    apply_denoise: bool = True,
    apply_contrast: bool = True,
    apply_thresholding: bool = False,
    contrast_factor: float = 1.25,
    denoise_kernel_size: int = 3,
) -> PreprocessingResult:
    """Full preprocessing pipeline for land record documents and handwriting crops.

    By default, delivers high-contrast, clean, deskewed grayscale images suitable
    for neural sequence recognition (TrOCR / CRNN / PaddleOCR) without harsh binarization
    that could break faint handwritten strokes.

    Args:
        image_input: Path, PIL Image, or NumPy array.
        apply_deskew: Whether to estimate and correct orientation skew.
        apply_denoise: Whether to apply light median filtering.
        apply_contrast: Whether to normalize and enhance contrast.
        apply_thresholding: Whether to apply adaptive binarization (False by default for handwriting).
        contrast_factor: Contrast boost level (1.0 - 1.5 recommended).
        denoise_kernel_size: Kernel size for noise reduction.

    Returns:
        PreprocessingResult containing the processed PIL Image and comprehensive audit metadata.
    """
    raw_pil = load_image_as_pil(image_input)
    orig_size = raw_pil.size
    audit: Dict[str, Any] = {
        "original_mode": raw_pil.mode,
        "original_size": {"width": orig_size[0], "height": orig_size[1]},
        "pipeline_steps": [],
    }

    # Step 1: Grayscale conversion
    current_img = to_grayscale(raw_pil)
    audit["pipeline_steps"].append("grayscale_conversion")

    # Step 2: Deskew
    if apply_deskew:
        current_img, deskew_meta = deskew_image(current_img)
        audit["deskew"] = deskew_meta
        audit["pipeline_steps"].append("deskew")

    # Step 3: Contrast Enhancement
    if apply_contrast:
        current_img, contrast_meta = enhance_contrast(current_img, contrast_factor=contrast_factor)
        audit["contrast"] = contrast_meta
        audit["pipeline_steps"].append("contrast_enhancement")

    # Step 4: Light Denoise
    if apply_denoise:
        current_img, denoise_meta = light_denoise(current_img, kernel_size=denoise_kernel_size)
        audit["denoise"] = denoise_meta
        audit["pipeline_steps"].append("light_denoise")

    # Step 5: Optional Adaptive Thresholding
    if apply_thresholding:
        current_img, thresh_meta = adaptive_soft_threshold(current_img)
        audit["thresholding"] = thresh_meta
        audit["pipeline_steps"].append("adaptive_thresholding")
    else:
        audit["thresholding"] = {"applied": False, "reason": "faint_handwriting_preservation"}

    audit["final_size"] = {"width": current_img.size[0], "height": current_img.size[1]}

    return PreprocessingResult(
        image=current_img,
        audit_metadata=audit,
        original_size=orig_size,
        processed_size=current_img.size,
    )
