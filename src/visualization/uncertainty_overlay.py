"""Visual Uncertainty and Low-Confidence Debugging Overlay Tool.

Provides visual diagnostics for OCR confidence, routing, and human review needs:
- Color-coded bounding box overlays on document pages:
  - Green: High confidence (>0.85)
  - Yellow: Medium confidence (0.60 - 0.85)
  - Red: Low confidence / needs review (<0.60)
  - Blue: Handwritten (routed to IITB TrOCR)
  - Purple: Printed (routed to EasyOCR)
- Side-by-side comparison generator: original crop vs verbatim OCR text vs confidence badge
- Standalone CLI utility + importable library function
"""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Curated palette
COLOR_HIGH_CONF = (40, 167, 69)      # Green
COLOR_MED_CONF = (255, 193, 7)       # Amber/Yellow
COLOR_LOW_CONF = (220, 53, 69)       # Red
COLOR_HANDWRITTEN = (13, 110, 253)   # Blue
COLOR_PRINTED = (111, 66, 193)       # Purple
COLOR_NEUTRAL = (108, 117, 125)      # Grey


def get_color_for_region(region: Any, color_by: str = "confidence") -> Tuple[int, int, int]:
    """Determines outline/badge color based on confidence tier or routing engine."""
    # Extract attributes safely
    conf = getattr(region, "confidence", None)
    if conf is None:
        conf = getattr(region, "recognizer_confidence_raw", None)
    if conf is None and isinstance(region, dict):
        conf = region.get("confidence", region.get("recognizer_confidence_raw"))

    is_hw = getattr(region, "is_handwritten", None)
    if is_hw is None and isinstance(region, dict):
        is_hw = region.get("is_handwritten", region.get("handwriting", False))

    recognizer = str(getattr(region, "model_name", None) or getattr(region, "recognizer", "")).lower()
    if not recognizer and isinstance(region, dict):
        recognizer = str(region.get("model_name", region.get("recognizer", ""))).lower()

    needs_review = getattr(region, "requires_human_review", None)
    if needs_review is None and isinstance(region, dict):
        needs_review = region.get("requires_human_review", region.get("needs_review", False))

    if color_by == "engine":
        if is_hw or "trocr" in recognizer or "iitb" in recognizer:
            return COLOR_HANDWRITTEN
        return COLOR_PRINTED
    else:  # Default: color by confidence tier
        if needs_review or (conf is not None and conf < 0.60):
            return COLOR_LOW_CONF
        elif conf is not None and conf >= 0.85:
            return COLOR_HIGH_CONF
        elif conf is not None and conf >= 0.60:
            return COLOR_MED_CONF
        return COLOR_LOW_CONF


def extract_box_coords(bbox: Any, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Extracts absolute pixel (x_min, y_min, x_max, y_max) from various bbox formats."""
    if hasattr(bbox, "x_min") and hasattr(bbox, "y_min"):
        x1, y1 = bbox.x_min, bbox.y_min
        x2, y2 = bbox.x_max, bbox.y_max
    elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
    elif isinstance(bbox, dict):
        x1 = bbox.get("x_min", bbox.get("x", 0))
        y1 = bbox.get("y_min", bbox.get("y", 0))
        x2 = bbox.get("x_max", x1 + bbox.get("width", 10))
        y2 = bbox.get("y_max", y1 + bbox.get("height", 10))
    else:
        return 0, 0, img_w, img_h

    # Check if normalized (<= 1.0)
    if max(x1, x2) <= 1.0 and max(y1, y2) <= 1.0 and img_w > 1 and img_h > 1:
        x1 = int(x1 * img_w)
        x2 = int(x2 * img_w)
        y1 = int(y1 * img_h)
        y2 = int(y2 * img_h)
    else:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(x1 + 1, min(img_w, x2))
    y2 = max(y1 + 1, min(img_h, y2))
    return x1, y1, x2, y2


def render_uncertainty_overlay(
    image: Union[str, Path, Image.Image],
    regions: Sequence[Any],
    output_path: Optional[Union[str, Path]] = None,
    color_by: str = "confidence",  # "confidence" or "engine"
    box_width: int = 3,
) -> Image.Image:
    """Draws color-coded bounding boxes and badges on document image."""
    if isinstance(image, (str, Path)):
        base_img = Image.open(image).convert("RGBA")
    else:
        base_img = image.convert("RGBA")

    w, h = base_img.size
    overlay = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    # Detect whether calibrated confidence exists across regions
    has_calibrated = any(
        (getattr(r, "calibrated_confidence", None) is not None) or
        (isinstance(r, dict) and r.get("calibrated_confidence") is not None)
        for r in regions
    )

    # Top banner: Explicit tier statement
    tier_label = (
        "CALIBRATED CONFIDENCE TIER | Green: Cal >= 0.85 | Yellow: 0.60-0.85 | Red: Review < 0.60"
        if has_calibrated
        else "OPERATIONAL REVIEW TIER (RAW SCORES - UNCALIBRATED) | Green: Raw >= 0.85 | Yellow: 0.60-0.85 | Red: Needs Review"
    )
    banner_h = 24
    draw.rectangle([0, 0, w, banner_h], fill=(33, 37, 41, 230))
    draw.text((10, 6), tier_label, fill=(255, 255, 255, 255), font=font)

    for r in regions:
        bbox = getattr(r, "bbox", None) if hasattr(r, "bbox") else r.get("bbox") if isinstance(r, dict) else None
        if bbox is None:
            continue

        x1, y1, x2, y2 = extract_box_coords(bbox, w, h)
        color = get_color_for_region(r, color_by=color_by)

        # Semi-transparent fill + solid border
        fill_color = color + (35,)
        outline_color = color + (255,)

        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=box_width)

        # Label badge
        cal_conf = getattr(r, "calibrated_confidence", None) if hasattr(r, "calibrated_confidence") else r.get("calibrated_confidence") if isinstance(r, dict) else None
        raw_conf = getattr(r, "confidence", None) if hasattr(r, "confidence") else r.get("confidence") if isinstance(r, dict) else None
        if raw_conf is None:
            raw_conf = getattr(r, "recognizer_confidence_raw", None) if hasattr(r, "recognizer_confidence_raw") else r.get("recognizer_confidence_raw") if isinstance(r, dict) else None

        recognizer = str(getattr(r, "model_name", None) or getattr(r, "recognizer", "")).lower()
        if not recognizer and isinstance(r, dict):
            recognizer = str(r.get("model_name", r.get("recognizer", ""))).lower()
        engine_short = "TrOCR" if ("trocr" in recognizer or "iitb" in recognizer) else "EasyOCR"

        if cal_conf is not None:
            badge_text = f"Cal: {cal_conf:.2f}"
        elif raw_conf is not None:
            badge_text = f"Raw: {raw_conf:.2f} (UNCALIBRATED)"
        else:
            badge_text = "UNCALIBRATED"

        label_str = f" {engine_short} | {badge_text} "

        # Compute badge background
        badge_y1 = max(banner_h + 2, y1 - 16)
        badge_y2 = badge_y1 + 14
        badge_x2 = min(w, x1 + len(label_str) * 6 + 6)
        draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=outline_color)
        draw.text((x1 + 2, badge_y1 + 1), label_str, fill=(255, 255, 255, 255), font=font)

    # Composite overlay over base image
    combined = Image.alpha_composite(base_img, overlay).convert("RGB")

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        combined.save(out_p)
        logger.info("Saved uncertainty overlay to %s", out_p)

    return combined


def render_side_by_side_comparison(
    image: Union[str, Path, Image.Image],
    regions: Sequence[Any],
    output_path: Optional[Union[str, Path]] = None,
    max_regions: int = 25,
) -> Image.Image:
    """Generates side-by-side comparison: Region Crop vs Verbatim OCR Text vs Confidence Badge."""
    if isinstance(image, (str, Path)):
        base_img = Image.open(image).convert("RGB")
    else:
        base_img = image.convert("RGB")

    img_w, img_h = base_img.size
    font = ImageFont.load_default()

    sample_regions = list(regions)[:max_regions]
    row_height = 65
    canvas_w = 900
    canvas_h = max(100, 70 + len(sample_regions) * row_height)

    comp_img = Image.new("RGB", (canvas_w, canvas_h), (248, 249, 250))
    draw = ImageDraw.Draw(comp_img)

    # Header
    draw.rectangle([0, 0, canvas_w, 45], fill=(33, 37, 41))
    draw.text((15, 14), "Crop", fill=(255, 255, 255), font=font)
    draw.text((220, 14), "Verbatim Recognized Text", fill=(255, 255, 255), font=font)
    draw.text((560, 14), "Engine / Mode", fill=(255, 255, 255), font=font)
    draw.text((715, 14), "Raw Score Tier", fill=(255, 255, 255), font=font)
    draw.text((820, 14), "Review?", fill=(255, 255, 255), font=font)

    y_offset = 55
    for idx, r in enumerate(sample_regions):
        bbox = getattr(r, "bbox", None) if hasattr(r, "bbox") else r.get("bbox") if isinstance(r, dict) else None
        if bbox is None:
            continue

        x1, y1, x2, y2 = extract_box_coords(bbox, img_w, img_h)
        crop = base_img.crop((x1, y1, x2, y2))
        crop_w, crop_h = crop.size

        # Resize crop preserving aspect ratio to fit (180, 50)
        scale = min(180 / max(1, crop_w), 50 / max(1, crop_h))
        new_w, new_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))
        crop_thumb = crop.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Paste crop
        comp_img.paste(crop_thumb, (15, y_offset + (55 - new_h) // 2))

        # Extract text
        text = str(getattr(r, "normalized_text", None) or getattr(r, "raw_text", "") or "")
        if not text and isinstance(r, dict):
            text = str(r.get("normalized_text") or r.get("raw_text") or r.get("text") or "")
        display_text = text[:45] + "..." if len(text) > 45 else (text or "[EMPTY]")

        # Engine
        recognizer = str(getattr(r, "model_name", None) or getattr(r, "recognizer", "")).lower()
        if not recognizer and isinstance(r, dict):
            recognizer = str(r.get("model_name", r.get("recognizer", ""))).lower()
        engine_str = "IITB TrOCR (HW)" if ("trocr" in recognizer or "iitb" in recognizer) else "EasyOCR (Printed)"

        # Confidence & Needs Review
        cal_conf = getattr(r, "calibrated_confidence", None) if hasattr(r, "calibrated_confidence") else r.get("calibrated_confidence") if isinstance(r, dict) else None
        conf = getattr(r, "confidence", None)
        if conf is None:
            conf = getattr(r, "recognizer_confidence_raw", None)
        if conf is None and isinstance(r, dict):
            conf = r.get("confidence", r.get("recognizer_confidence_raw"))

        if cal_conf is not None:
            conf_str = f"Cal: {cal_conf:.2f}"
        elif conf is not None:
            conf_str = f"Raw: {conf:.2f}"
        else:
            conf_str = "UNCAL"

        needs_review = getattr(r, "requires_human_review", None)
        if needs_review is None and isinstance(r, dict):
            needs_review = r.get("requires_human_review", r.get("needs_review", False))

        # Draw row text
        draw.text((220, y_offset + 18), display_text, fill=(30, 30, 30), font=font)
        draw.text((560, y_offset + 18), engine_str, fill=(70, 70, 70), font=font)

        # Color-coded confidence badge
        color = get_color_for_region(r, color_by="confidence")
        draw.rectangle([715, y_offset + 12, 805, y_offset + 36], fill=color)
        draw.text((725, y_offset + 18), conf_str, fill=(255, 255, 255), font=font)

        # Review flag
        rev_color = COLOR_LOW_CONF if needs_review else COLOR_HIGH_CONF
        rev_text = "YES" if needs_review else "NO"
        draw.rectangle([820, y_offset + 12, 875, y_offset + 36], fill=rev_color)
        draw.text((838, y_offset + 18), rev_text, fill=(255, 255, 255), font=font)

        # Separator line
        draw.line([(10, y_offset + row_height - 1), (canvas_w - 10, y_offset + row_height - 1)], fill=(220, 224, 230))
        y_offset += row_height

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        comp_img.save(out_p)
        logger.info("Saved side-by-side comparison to %s", out_p)

    return comp_img


def generate_uncertainty_report(
    image: Union[str, Path, Image.Image],
    regions: Sequence[Any],
    output_dir: Union[str, Path],
    base_name: str = "document",
) -> Dict[str, str]:
    """Generates both overlay and side-by-side comparison images, returning their paths."""
    out_d = Path(output_dir)
    out_d.mkdir(parents=True, exist_ok=True)

    overlay_conf = out_d / f"{base_name}_uncertainty_overlay.png"
    overlay_eng = out_d / f"{base_name}_engine_overlay.png"
    side_by_side = out_d / f"{base_name}_side_by_side.png"

    render_uncertainty_overlay(image, regions, output_path=overlay_conf, color_by="confidence")
    render_uncertainty_overlay(image, regions, output_path=overlay_eng, color_by="engine")
    render_side_by_side_comparison(image, regions, output_path=side_by_side)

    return {
        "confidence_overlay": str(overlay_conf),
        "engine_overlay": str(overlay_eng),
        "side_by_side": str(side_by_side),
    }


def main():
    parser = argparse.ArgumentParser(description="Render OCR uncertainty and confidence overlay on document image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input document image")
    parser.add_argument("--output-dir", type=str, default="output/visualizations", help="Output directory for visualizations")
    args = parser.parse_args()

    # Standalone demo
    from src.integration.document_pipeline import process_document
    resp = process_document(image=args.image)
    paths = generate_uncertainty_report(args.image, resp.ordered_regions, args.output_dir)
    print(f"Visualizations successfully generated:\n{paths}")


if __name__ == "__main__":
    main()
