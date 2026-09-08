"""Person A Layout Analysis Integration Adapter.

Converts varied layout detection and segmentation formats into standard,
strongly-typed Person B input structures (RegionRequest & DocumentProcessingRequest)
without tightly coupling to Person A's internal architecture.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from PIL import Image

from schemas import BoundingBox
from src.integration.schemas import (
    DocumentProcessingRequest,
    RegionRequest,
    RegionType,
)


def _infer_region_type(label_str: Optional[str]) -> RegionType:
    """Infers semantic RegionType from upstream layout classification label."""
    if not label_str:
        return RegionType.TEXT
    cleaned = str(label_str).strip().lower()
    if any(k in cleaned for k in ("table", "cell", "grid")):
        return RegionType.TABLE_CELL
    if any(k in cleaned for k in ("header", "title", "heading")):
        return RegionType.HEADER
    if any(k in cleaned for k in ("sign", "signature", "initial")):
        return RegionType.SIGNATURE
    if any(k in cleaned for k in ("stamp", "seal")):
        return RegionType.STAMP
    if any(k in cleaned for k in ("thumb", "fingerprint")):
        return RegionType.THUMBPRINT
    if any(k in cleaned for k in ("map", "diagram", "sketch")):
        return RegionType.MAP_OR_DIAGRAM
    return RegionType.TEXT


def _infer_is_handwritten(
    explicit_val: Optional[bool],
    label_str: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    """Infers whether region is handwritten from explicit flags or label text."""
    if explicit_val is not None:
        return bool(explicit_val)

    # Check label string
    combined_signals = []
    if label_str:
        combined_signals.append(str(label_str).lower())
    if metadata:
        for k in ("label", "class_name", "type", "tag", "classification", "text_type"):
            if k in metadata and metadata[k]:
                combined_signals.append(str(metadata[k]).lower())

    for sig in combined_signals:
        if any(h in sig for h in ("handwritten", "handwriting", "hw", "manuscript", "signature", "thumb")):
            return True
        if any(p in sig for p in ("printed", "print", "machine", "typed", "typeset", "header")):
            return False

    return None


class PersonAAdapter:
    """Flexible adapter that ingests layout analysis outputs and produces valid Person B requests."""

    @staticmethod
    def validate_and_clamp_bbox(
        raw_box: Any,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> BoundingBox:
        """Validates bounding box coordinates and clamps them within valid image dimensions.

        Supports:
        - BoundingBox object
        - Dict with 'x_min', 'y_min', 'x_max', 'y_max' or 'x1', 'y1', 'x2', 'y2' or 'left', 'top', 'right', 'bottom'
        - Dict with 'bbox' / 'box' / 'coordinates'
        - Tuple or list of 4 integers/floats (x_min, y_min, x_max, y_max)
        - YOLO normalized format [cx, cy, w, h] if floats in [0, 1] and image_size provided
        """
        img_w = image_size[0] if image_size else 100000
        img_h = image_size[1] if image_size else 100000

        x_min, y_min, x_max, y_max = 0, 0, img_w, img_h

        if isinstance(raw_box, BoundingBox):
            x_min, y_min, x_max, y_max = raw_box.x_min, raw_box.y_min, raw_box.x_max, raw_box.y_max

        elif isinstance(raw_box, dict):
            if "bbox" in raw_box:
                return PersonAAdapter.validate_and_clamp_bbox(raw_box["bbox"], image_size)
            if "box" in raw_box:
                return PersonAAdapter.validate_and_clamp_bbox(raw_box["box"], image_size)
            if "coordinates" in raw_box:
                return PersonAAdapter.validate_and_clamp_bbox(raw_box["coordinates"], image_size)

            x_min = raw_box.get("x_min", raw_box.get("x1", raw_box.get("left", 0)))
            y_min = raw_box.get("y_min", raw_box.get("y1", raw_box.get("top", 0)))
            x_max = raw_box.get("x_max", raw_box.get("x2", raw_box.get("right", img_w)))
            y_max = raw_box.get("y_max", raw_box.get("y2", raw_box.get("bottom", img_h)))

        elif isinstance(raw_box, (tuple, list)) and len(raw_box) == 4:
            vals = [float(v) for v in raw_box]
            # Check if YOLO normalized [0, 1] format was passed
            if image_size and all(0.0 <= v <= 1.0 for v in vals) and vals[2] <= 1.0 and vals[3] <= 1.0:
                cx, cy, bw, bh = vals
                x_min = int((cx - bw / 2.0) * img_w)
                y_min = int((cy - bh / 2.0) * img_h)
                x_max = int((cx + bw / 2.0) * img_w)
                y_max = int((cy + bh / 2.0) * img_h)
            else:
                x_min, y_min, x_max, y_max = int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])

        else:
            raise TypeError(f"Cannot parse bounding box from type '{type(raw_box).__name__}': {raw_box}")

        # Ensure min < max
        real_x_min = min(int(x_min), int(x_max))
        real_x_max = max(int(x_min), int(x_max))
        real_y_min = min(int(y_min), int(y_max))
        real_y_max = max(int(y_min), int(y_max))

        # Clamp inside boundaries
        clamped_x_min = max(0, min(real_x_min, img_w - 1 if img_w > 1 else 0))
        clamped_y_min = max(0, min(real_y_min, img_h - 1 if img_h > 1 else 0))
        clamped_x_max = max(clamped_x_min + 1, min(real_x_max, img_w))
        clamped_y_max = max(clamped_y_min + 1, min(real_y_max, img_h))

        return BoundingBox(
            x_min=clamped_x_min,
            y_min=clamped_y_min,
            x_max=clamped_x_max,
            y_max=clamped_y_max,
        )

    @classmethod
    def adapt_layout_regions(
        cls,
        regions: Sequence[Any],
        image_size: Optional[Tuple[int, int]] = None,
        default_language: str = "kannada",
        default_script: str = "Kannada",
        default_is_handwritten: Optional[bool] = None,
    ) -> List[RegionRequest]:
        """Converts raw list of layout regions into validated RegionRequest models."""
        adapted: List[RegionRequest] = []

        for idx, item in enumerate(regions):
            reg_id = f"region_{idx + 1:03d}"
            reg_lang = default_language
            reg_script = default_script
            reg_type = RegionType.TEXT
            layout_conf: Optional[float] = None
            meta: Dict[str, Any] = {}
            reg_is_hw = default_is_handwritten

            if isinstance(item, RegionRequest):
                adapted.append(item)
                continue

            if isinstance(item, BoundingBox):
                bbox = cls.validate_and_clamp_bbox(item, image_size)
            elif isinstance(item, (tuple, list)) and len(item) == 4:
                bbox = cls.validate_and_clamp_bbox(item, image_size)
            elif isinstance(item, dict):
                # Extract explicit identifiers
                reg_id = str(item.get("region_id") or item.get("id") or item.get("box_id") or reg_id)
                reg_lang = item.get("language") or item.get("lang") or default_language
                reg_script = item.get("script") or default_script
                layout_conf = item.get("confidence") or item.get("score") or item.get("layout_confidence")

                label_val = item.get("label") or item.get("class_name") or item.get("type") or item.get("category")
                reg_type = _infer_region_type(label_val)
                reg_is_hw = _infer_is_handwritten(
                    explicit_val=item.get("is_handwritten"),
                    label_str=label_val,
                    metadata=item,
                )
                bbox = cls.validate_and_clamp_bbox(item, image_size)
                meta = {
                    k: v for k, v in item.items()
                    if k not in ("bbox", "box", "coordinates", "x_min", "y_min", "x_max", "y_max", "region_id", "id", "language", "script", "is_handwritten", "confidence", "score")
                }
            else:
                # Fallback for arbitrary object with attributes
                bbox_val = getattr(item, "bbox", getattr(item, "box", item))
                bbox = cls.validate_and_clamp_bbox(bbox_val, image_size)
                reg_id = str(getattr(item, "region_id", getattr(item, "id", reg_id)))
                reg_lang = getattr(item, "language", default_language)
                reg_script = getattr(item, "script", default_script)
                layout_conf = getattr(item, "confidence", None)

            req = RegionRequest(
                region_id=reg_id,
                bbox=bbox,
                language=reg_lang,
                script=reg_script,
                is_handwritten=reg_is_hw,
                region_type=reg_type,
                layout_confidence=float(layout_conf) if layout_conf is not None else None,
                metadata=meta,
            )
            adapted.append(req)

        # Sort adapted regions in reading order
        return cls.sort_regions_reading_order(adapted)

    @staticmethod
    def sort_regions_reading_order(regions: List[RegionRequest]) -> List[RegionRequest]:
        """Sorts regions top-to-bottom, left-to-right with line quantization tolerance."""
        def reading_order_key(r: RegionRequest) -> Tuple[int, int]:
            # Quantize vertical position to 24px line bins
            y_bin = (r.bbox.y_min // 24) * 24
            return (y_bin, r.bbox.x_min)

        return sorted(regions, key=reading_order_key)

    @classmethod
    def create_document_request(
        cls,
        image: Any,
        regions: Optional[Sequence[Any]] = None,
        document_id: Optional[str] = None,
        page_number: int = 1,
        language: str = "kannada",
        script: str = "Kannada",
        is_handwritten: Optional[bool] = None,
        document_metadata: Optional[Dict[str, Any]] = None,
        apply_preprocessing: bool = True,
        apply_normalization: bool = True,
    ) -> DocumentProcessingRequest:
        """Constructs a fully normalized DocumentProcessingRequest from Person A inputs."""
        # Determine image size if PIL Image
        img_size = None
        if isinstance(image, Image.Image):
            img_size = image.size

        parsed_regions = None
        if regions is not None:
            parsed_regions = cls.adapt_layout_regions(
                regions=regions,
                image_size=img_size,
                default_language=language,
                default_script=script,
                default_is_handwritten=is_handwritten,
            )

        return DocumentProcessingRequest(
            image=image,
            document_id=document_id,
            page_number=page_number,
            regions=parsed_regions,
            language=language,
            script=script,
            is_handwritten=is_handwritten,
            apply_preprocessing=apply_preprocessing,
            apply_normalization=apply_normalization,
            document_metadata=document_metadata or {},
        )
