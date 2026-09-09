"""person-a/src/layout/detector.py
Semantic Layout and Region Understanding Engine.
Classifies document visual zones (Headers, Footers, Tables, Paragraphs, Handwritten/Signature areas).
Integrates TableDetector and provides bounding-box region routing for Person A & Person B.
"""

import json
from typing import List, Optional, Tuple
import cv2
import numpy as np

from ..schemas import BlockType, BoundingBox, OCRBlock, TableStructure
from ..tables.detector import TableDetector


class LayoutDetector:
    """Performs spatial and morphological layout analysis across document pages."""

    def __init__(self, table_detector: Optional[TableDetector] = None):
        self.table_detector = table_detector or TableDetector()

    def analyze_layout(
        self,
        image: np.ndarray,
        ocr_blocks: List[OCRBlock],
        page_num: int = 1,
    ) -> Tuple[List[OCRBlock], List[TableStructure]]:
        """Detect tables, map intersecting OCR lines into table cells, and refine layout block types."""
        if image is None or image.size == 0:
            return ocr_blocks, []

        h, w = image.shape[:2]

        # 1. Detect Tables using morphological line analysis
        tables = self.table_detector.detect_tables(image, page_num=page_num)

        # 2. Map OCR lines into table cells
        if tables and ocr_blocks:
            for table in tables:
                self._map_ocr_to_table(ocr_blocks, table)

        # 3. Refine OCR blocks: label table blocks vs headers vs paragraphs
        refined_blocks: List[OCRBlock] = []
        for block in ocr_blocks:
            # Check if block falls inside any table bbox
            in_table = False
            for t in tables:
                if self._box_inside(block.bbox, t.bbox):
                    block.block_type = BlockType.TABLE
                    in_table = True
                    break
            refined_blocks.append(block)

        return refined_blocks, tables

    def _map_ocr_to_table(self, blocks: List[OCRBlock], table: TableStructure) -> None:
        for block in blocks:
            for line in block.lines:
                for cell in table.cells:
                    if cell.bbox and self._box_inside(line.bbox, cell.bbox, threshold=0.4):
                        if cell.text:
                            cell.text += " " + line.text
                        else:
                            cell.text = line.text

        # Re-sync table rows data with cell text
        for cell in table.cells:
            if cell.row_index < len(table.rows_data) and cell.col_index < len(table.rows_data[cell.row_index]):
                table.rows_data[cell.row_index][cell.col_index] = cell.text.strip()

        # Regenerate headers, markdown preview, and dataframe JSON
        headers = table.rows_data[0] if table.rows_data else []
        table.headers = headers
        table.markdown = self.table_detector._generate_markdown(
            headers, table.rows_data[1:] if len(table.rows_data) > 1 else []
        )
        table.dataframe_json = json.dumps(table.rows_data)

    def _box_inside(self, inner: BoundingBox, outer: BoundingBox, threshold: float = 0.5) -> bool:
        ix1 = max(inner.x_min, outer.x_min)
        iy1 = max(inner.y_min, outer.y_min)
        ix2 = min(inner.x_max, outer.x_max)
        iy2 = min(inner.y_max, outer.y_max)
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        inner_area = inner.area
        if inner_area <= 0:
            return False
        return (inter_area / inner_area) >= threshold
