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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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
        PIL Image instance with EXIF orientation metadata transposed.

    Raises:
        ValueError: If the input cannot be decoded into an image.
    """
    if isinstance(image_input, Image.Image):
        return ImageOps.exif_transpose(image_input).copy()

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        im = Image.open(path)
        return ImageOps.exif_transpose(im).copy()

    if isinstance(image_input, np.ndarray):
        # Handle grayscale vs BGR/RGB
        if image_input.ndim == 2:
            im = Image.fromarray(image_input, mode="L")
        elif image_input.ndim == 3:
            if image_input.shape[2] == 3:
                im = Image.fromarray(image_input, mode="RGB")
            elif image_input.shape[2] == 4:
                im = Image.fromarray(image_input, mode="RGBA")
            else:
                raise ValueError(f"Unsupported NumPy image shape: {image_input.shape}")
        else:
            raise ValueError(f"Unsupported NumPy image shape: {image_input.shape}")
        return ImageOps.exif_transpose(im).copy()

    if isinstance(image_input, bytes):
        import io
        im = Image.open(io.BytesIO(image_input))
        return ImageOps.exif_transpose(im).copy()

    raise TypeError(f"Unsupported image input type: {type(image_input).__name__}")


_COARSE_DETECTOR_CACHE: Optional[Any] = None


def get_coarse_orientation_detector() -> Optional[Any]:
    """Retrieves or creates cached Paddle text detector for orientation estimation."""
    global _COARSE_DETECTOR_CACHE
    if _COARSE_DETECTOR_CACHE is None:
        try:
            from paddleocr import PaddleOCR
            _COARSE_DETECTOR_CACHE = PaddleOCR(use_angle_cls=False, lang="en", enable_mkldnn=False)
        except Exception:
            _COARSE_DETECTOR_CACHE = False
    return _COARSE_DETECTOR_CACHE if _COARSE_DETECTOR_CACHE is not False else None


def detect_and_correct_coarse_orientation(
    image: Image.Image,
    detector: Optional[Any] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Detects if document text lines run vertically (sideways photo) and rotates 90 degrees.

    In standard horizontal text, line bounding boxes are wider than tall (width >= height).
    When a document is photographed in portrait orientation with text lines running sideways,
    the detected line boxes are predominantly vertical (height > width).
    If vertical boxes >= 1.5 * horizontal boxes and vertical count >= 8, the document is
    rotated 90 degrees counter-clockwise (.rotate(90, expand=True)) so text lines align horizontally.
    Paddle's use_angle_cls=True then handles 0 vs 180 degree line orientation.
    """
    transposed = ImageOps.exif_transpose(image)
    det = detector if detector is not None else get_coarse_orientation_detector()
    if det is None:
        return transposed, {"coarse_rotation_applied": False, "angle_degrees": 0, "reason": "detector_unavailable"}

    try:
        arr = np.array(transposed.convert("RGB"))
        res = det.ocr(arr, cls=False, rec=False)
        boxes = res[0] if res and res[0] else []
        h_count, v_count = 0, 0
        for b in boxes:
            xs = [pt[0] for pt in b]
            ys = [pt[1] for pt in b]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w >= h:
                h_count += 1
            else:
                v_count += 1

        if v_count >= 1.5 * h_count and v_count >= 8:
            rotated = transposed.rotate(90, expand=True)
            meta = {
                "coarse_rotation_applied": True,
                "angle_degrees": 90,
                "horizontal_boxes": h_count,
                "vertical_boxes": v_count,
                "reason": "sideways_document_lines_detected",
            }
            return rotated, meta

        # Fallback: if document was photographed in portrait (height significantly > width),
        # but registers are landscape format with vertical stroke energy
        w_img, h_img = transposed.size
        if h_img > w_img * 1.35 and (v_count > h_count or v_count >= 5):
            rotated = transposed.rotate(90, expand=True)
            return rotated, {
                "coarse_rotation_applied": True,
                "angle_degrees": 90,
                "horizontal_boxes": h_count,
                "vertical_boxes": v_count,
                "reason": "aspect_ratio_and_vertical_strokes",
            }

        return transposed, {
            "coarse_rotation_applied": False,
            "angle_degrees": 0,
            "horizontal_boxes": h_count,
            "vertical_boxes": v_count,
            "reason": "upright_orientation_confirmed",
        }
    except Exception as exc:
        w_img, h_img = transposed.size
        if h_img > w_img * 1.35:
            rotated = transposed.rotate(90, expand=True)
            return rotated, {"coarse_rotation_applied": True, "angle_degrees": 90, "reason": f"aspect_ratio_fallback ({exc})"}
        return transposed, {"coarse_rotation_applied": False, "angle_degrees": 0, "reason": f"error: {str(exc)}"}


def crop_document_background(
    image: Image.Image,
    max_margin_ratio: float = 0.08,
    brightness_ratio: float = 0.40,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Detects and crops dark non-paper margins (e.g. laptop keyboard, desk surface) without cutting document text.

    Compares row and column luminance against the central paper median. If outer edges are significantly darker,
    they are cropped away up to max_margin_ratio (default 8% conservative limit).
    """
    gray = np.array(image.convert("L"))
    h, w = gray.shape

    # Sample central 50% region to estimate true paper luminance
    y1, y2 = int(h * 0.25), int(h * 0.75)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    paper_median = float(np.median(gray[y1:y2, x1:x2]))
    thresh = paper_median * brightness_ratio

    row_means = np.mean(gray, axis=1)
    col_means = np.mean(gray, axis=0)

    top = 0
    while top < h * max_margin_ratio and row_means[top] < thresh:
        top += 1

    bottom = h - 1
    while bottom > h * (1.0 - max_margin_ratio) and row_means[bottom] < thresh:
        bottom -= 1

    left = 0
    while left < w * max_margin_ratio and col_means[left] < thresh:
        left += 1

    right = w - 1
    while right > w * (1.0 - max_margin_ratio) and col_means[right] < thresh:
        right -= 1

    cropped_applied = bool(top > 0 or bottom < h - 1 or left > 0 or right < w - 1)
    if cropped_applied and (right > left + 50) and (bottom > top + 50):
        cropped = image.crop((left, top, right + 1, bottom + 1))
    else:
        cropped = image
        left, top, right, bottom = 0, 0, w - 1, h - 1

    metadata = {
        "background_crop_applied": cropped_applied,
        "crop_box": {"left": int(left), "top": int(top), "right": int(right), "bottom": int(bottom)},
        "paper_median_brightness": round(paper_median, 1),
    }
    return cropped, metadata


def upscale_document_image(
    image: Image.Image,
    target_min_dim: int = 1400,
    max_scale: float = 2.0,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Upscales low-resolution document images using Lanczos resampling to improve Indic ligature OCR clarity.

    If the maximum dimension is below target_min_dim, upscales the document proportionally and applies subtle sharpening.
    """
    w, h = image.size
    max_dim = max(w, h)
    if max_dim < target_min_dim:
        scale = min(max_scale, float(target_min_dim) / max_dim)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        upscaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Apply subtle edge enhancement for Indic vowel strokes
        enhancer = ImageEnhance.Sharpness(upscaled)
        sharpened = enhancer.enhance(1.2)
        return sharpened, {"upscaled": True, "scale_factor": round(scale, 2), "original_size": (w, h), "new_size": (new_w, new_h)}
    return image, {"upscaled": False, "scale_factor": 1.0, "original_size": (w, h), "new_size": (w, h)}


def enhance_contrast_clahe(
    image: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> Tuple[Image.Image, Dict[str, Any]]:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to handle uneven shadows across photographs."""
    gray = to_grayscale(image)
    if HAS_CV2 and cv2 is not None:
        arr = np.array(gray)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_arr = clahe.apply(arr)
        out_img = Image.fromarray(enhanced_arr, mode="L")
        return out_img, {"clahe_applied": True, "clip_limit": clip_limit}
    return enhance_contrast(gray)


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
    apply_denoise: bool = False,
    apply_contrast: bool = True,
    apply_thresholding: bool = False,
    contrast_factor: float = 1.15,
    denoise_kernel_size: int = 3,
    apply_coarse_orientation: bool = True,
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
        apply_coarse_orientation: Whether to detect and correct sideways 90-degree photographed orientation.

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

    # Step 0: Coarse Orientation Detection & Correction (Sideways 90-degree check)
    if apply_coarse_orientation:
        raw_pil, coarse_meta = detect_and_correct_coarse_orientation(raw_pil)
        audit["coarse_orientation"] = coarse_meta
        if coarse_meta.get("coarse_rotation_applied"):
            audit["pipeline_steps"].append("coarse_orientation_correction")

    # Step 0b: Dark Background / Keyboard Margins Crop
    raw_pil, bg_crop_meta = crop_document_background(raw_pil)
    audit["background_crop"] = bg_crop_meta
    if bg_crop_meta.get("background_crop_applied"):
        audit["pipeline_steps"].append("background_crop")

    # Step 0c: High-resolution Upscale for Small Text / Archival Legibility
    raw_pil, upscale_meta = upscale_document_image(raw_pil, target_min_dim=1400, max_scale=2.0)
    audit["upscale"] = upscale_meta
    if upscale_meta.get("upscaled"):
        audit["pipeline_steps"].append("upscale_lanczos")

    # Step 1: Grayscale conversion
    current_img = to_grayscale(raw_pil)
    audit["pipeline_steps"].append("grayscale_conversion")

    # Step 2: Deskew
    if apply_deskew:
        current_img, deskew_meta = deskew_image(current_img)
        audit["deskew"] = deskew_meta
        audit["pipeline_steps"].append("deskew")

    # Step 3: Contrast Enhancement (using CLAHE for non-destructive local contrast)
    if apply_contrast:
        current_img, contrast_meta = enhance_contrast_clahe(current_img, clip_limit=2.0)
        audit["contrast"] = contrast_meta
        audit["pipeline_steps"].append("contrast_enhancement_clahe")

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

    # Save intermediate debug images for inspection
    try:
        debug_dir = Path(PROJECT_ROOT if "PROJECT_ROOT" in globals() else "c:/Land Record") / "storage" / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        raw_pil.save(debug_dir / "prep_step0_upright.png")
        current_img.save(debug_dir / "prep_step4_enhanced.png")
        audit["debug_saved"] = True
    except Exception:
        audit["debug_saved"] = False

    return PreprocessingResult(
        image=current_img,
        audit_metadata=audit,
        original_size=orig_size,
        processed_size=current_img.size,
    )
