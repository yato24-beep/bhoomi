"""Table Reconstructor for Land Record Structured Sections.

Clusters OCR tokens/regions into tabular geometry:
- Table -> Rows -> Columns -> Cells -> Source OCR Regions
- Preserves 2D row/column bounding box spatial layout
- Decouples OCR text recognition from table structural semantics
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from schemas import BoundingBox
from src.semantic.schema import SemanticTable, SemanticTableCell

logger = logging.getLogger(__name__)


class TableReconstructor:
    """Reconstructs tabular grids from discrete OCR text regions."""

    def __init__(self, row_y_tolerance: float = 14.0, col_x_tolerance: float = 24.0):
        self.row_y_tolerance = row_y_tolerance
        self.col_x_tolerance = col_x_tolerance

    def reconstruct_table(
        self,
        regions: List[Any],
        table_bbox: Optional[BoundingBox] = None,
        table_id: str = "table_0",
        page_number: int = 1,
    ) -> SemanticTable:
        """Reconstructs table grid from regions located inside the table boundary.

        Args:
            regions: List of OCR regions with bounding boxes.
            table_bbox: Optional bounding box encompassing the table.
            table_id: Unique table identifier.
            page_number: 1-indexed document page number.

        Returns:
            SemanticTable containing headers, rows, and structured cells linked to source region IDs.
        """
        # Filter regions belonging to this table area if table_bbox is provided
        if table_bbox:
            table_regions = [
                r for r in regions
                if getattr(r, "bbox", None) and table_bbox.overlaps_with(r.bbox, iou_threshold=0.01)
            ]
        else:
            table_regions = [r for r in regions if getattr(r, "bbox", None)]

        if not table_regions:
            return SemanticTable(
                table_id=table_id,
                page_number=page_number,
                bbox=table_bbox,
                headers=[],
                rows=[],
                cells=[],
            )

        # 1. Cluster regions into rows by vertical center position (y_mid)
        regions_with_geom = []
        for r in table_regions:
            box = r.bbox
            y_mid = (box.y_min + box.y_max) / 2.0
            x_mid = (box.x_min + box.x_max) / 2.0
            regions_with_geom.append((y_mid, x_mid, box, r))

        # Sort all items vertically top-to-bottom
        regions_with_geom.sort(key=lambda item: item[0])

        rows_clustered: List[List[Tuple[float, float, BoundingBox, Any]]] = []
        for item in regions_with_geom:
            y_mid = item[0]
            if not rows_clustered:
                rows_clustered.append([item])
                continue

            last_row = rows_clustered[-1]
            last_row_y_avg = sum(elem[0] for elem in last_row) / len(last_row)

            if abs(y_mid - last_row_y_avg) <= self.row_y_tolerance:
                last_row.append(item)
            else:
                rows_clustered.append([item])

        # 2. Sort items within each row strictly left-to-right (x_min)
        for row in rows_clustered:
            row.sort(key=lambda item: item[2].x_min)

        # 3. Build structured cells, string rows, and header detection
        cells: List[SemanticTableCell] = []
        string_rows: List[List[str]] = []

        for r_idx, row in enumerate(rows_clustered):
            row_texts: List[str] = []
            for c_idx, elem in enumerate(row):
                r_obj = elem[3]
                text = (getattr(r_obj, "normalized_text", None) or getattr(r_obj, "text", "")).strip()
                r_id = getattr(r_obj, "region_id", f"cell_{r_idx}_{c_idx}")
                conf = getattr(r_obj, "confidence", None)

                cell = SemanticTableCell(
                    row_index=r_idx,
                    col_index=c_idx,
                    text=text,
                    bbox=elem[2],
                    source_region_id=r_id,
                    confidence=conf,
                )
                cells.append(cell)
                row_texts.append(text)
            string_rows.append(row_texts)

        # Header detection: first row is treated as header if multiple rows exist
        headers = string_rows[0] if len(string_rows) > 1 else []
        data_rows = string_rows[1:] if len(string_rows) > 1 else string_rows

        # Calculate bounding box of table from constituent cells if not provided
        if not table_bbox and cells:
            x1 = min(c.bbox.x_min for c in cells if c.bbox)
            y1 = min(c.bbox.y_min for c in cells if c.bbox)
            x2 = max(c.bbox.x_max for c in cells if c.bbox)
            y2 = max(c.bbox.y_max for c in cells if c.bbox)
            table_bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

        return SemanticTable(
            table_id=table_id,
            page_number=page_number,
            bbox=table_bbox,
            headers=headers,
            rows=data_rows,
            cells=cells,
        )

    def reconstruct_tables(
        self,
        regions: List[Any],
        table_bbox: Optional[BoundingBox] = None,
        table_id: str = "tbl_01",
        page_number: int = 1,
    ) -> List[SemanticTable]:
        """Reconstructs all tabular grids found in the regions."""
        t = self.reconstruct_table(regions, table_bbox=table_bbox, table_id=table_id, page_number=page_number)
        return [t] if t else []
