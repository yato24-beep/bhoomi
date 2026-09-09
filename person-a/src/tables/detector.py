"""person-a/src/tables/detector.py
OpenCV Morphological Table & Grid Extraction Fallback.
Detects horizontal and vertical grid lines, extracts intersecting cells, row/column matrices,
and generates structured Markdown & JSON dataframe representations for downstream LLMs/Person C.
"""

import json
from typing import List, Optional, Tuple
import cv2
import numpy as np

from ..schemas import BoundingBox, TableCell, TableStructure


class TableDetector:
    """Detects tabular grids and extracts cells from document images."""

    def __init__(self, min_table_area: int = 10000, line_scale: int = 25):
        self.min_table_area = min_table_area
        self.line_scale = line_scale

    def detect_tables(
        self,
        image: np.ndarray,
        page_num: int = 1,
    ) -> List[TableStructure]:
        """Detect all table boundaries and internal grid structures."""
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 1. Binarize with Otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Extract Horizontal Lines
        h_kernel_len = max(w // self.line_scale, 10)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

        # 3. Extract Vertical Lines
        v_kernel_len = max(h // self.line_scale, 10)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

        # 4. Table Grid Structure Mask
        table_grid = cv2.add(h_lines, v_lines)

        # 5. Find Outer Table Contours
        contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tables: List[TableStructure] = []
        table_count = 0

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area = bw * bh
            if area < self.min_table_area:
                continue

            table_count += 1
            table_id = f"page_{page_num}_table_{table_count}"
            table_bbox = BoundingBox(x_min=float(x), y_min=float(y), x_max=float(x + bw), y_max=float(y + bh))

            # Extract cell contours inside the table crop
            grid_crop = table_grid[y:y + bh, x:x + bw]
            cells = self._extract_cells(grid_crop, offset_x=x, offset_y=y)

            rows_count, cols_count, rows_data = self._arrange_cells_into_grid(cells)
            headers = rows_data[0] if rows_data else []
            markdown = self._generate_markdown(headers, rows_data[1:] if len(rows_data) > 1 else [])
            df_json = json.dumps(rows_data)

            tables.append(
                TableStructure(
                    table_id=table_id,
                    page_number=page_num,
                    bbox=table_bbox,
                    rows_count=max(rows_count, 1),
                    cols_count=max(cols_count, 1),
                    cells=cells,
                    headers=headers,
                    rows_data=rows_data,
                    markdown=markdown,
                    dataframe_json=df_json,
                    confidence=0.90,
                )
            )

        return tables

    def _extract_cells(self, grid_crop: np.ndarray, offset_x: int, offset_y: int) -> List[TableCell]:
        cell_contours, _ = cv2.findContours(grid_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cells: List[TableCell] = []

        for c_cnt in cell_contours:
            cx, cy, cw, ch = cv2.boundingRect(c_cnt)
            # Filter out tiny specks or full-bounding contour
            if cw < 15 or ch < 10 or (cw == grid_crop.shape[1] and ch == grid_crop.shape[0]):
                continue

            cell_bbox = BoundingBox(
                x_min=float(offset_x + cx),
                y_min=float(offset_y + cy),
                x_max=float(offset_x + cx + cw),
                y_max=float(offset_y + cy + ch),
            )
            cells.append(
                TableCell(
                    row_index=0,
                    col_index=0,
                    text="",
                    bbox=cell_bbox,
                    confidence=1.0,
                )
            )

        # Sort cells top-to-bottom, left-to-right
        cells.sort(key=lambda c: (c.bbox.y_min if c.bbox else 0, c.bbox.x_min if c.bbox else 0))
        return cells

    def _arrange_cells_into_grid(self, cells: List[TableCell]) -> Tuple[int, int, List[List[str]]]:
        if not cells:
            return 0, 0, []

        valid_cells = [c for c in cells if c.bbox]
        if not valid_cells:
            return 1, 1, [[""]]

        # 1. Cluster cells into rows based on vertical overlap / center distance
        valid_cells.sort(key=lambda c: (c.bbox.y_min + c.bbox.y_max) / 2)
        rows: List[List[TableCell]] = []

        for cell in valid_cells:
            c_yc = (cell.bbox.y_min + cell.bbox.y_max) / 2
            c_h = max(cell.bbox.height, 1.0)
            placed = False
            for r in rows:
                r_yc = sum((x.bbox.y_min + x.bbox.y_max) / 2 for x in r) / len(r)
                r_h = max(sum(x.bbox.height for x in r) / len(r), 1.0)
                if abs(c_yc - r_yc) < min(c_h, r_h) * 0.5:
                    r.append(cell)
                    placed = True
                    break
            if not placed:
                rows.append([cell])

        # Sort rows top-to-bottom
        rows.sort(key=lambda r: sum((x.bbox.y_min + x.bbox.y_max) / 2 for x in r) / len(r))

        # 2. Cluster columns across all rows
        col_clusters: List[float] = []
        for cell in valid_cells:
            c_xc = (cell.bbox.x_min + cell.bbox.x_max) / 2
            c_w = max(cell.bbox.width, 1.0)
            col_placed = False
            for idx, col_c in enumerate(col_clusters):
                if abs(c_xc - col_c) < c_w * 0.4:
                    # Refine cluster center
                    col_clusters[idx] = (col_clusters[idx] + c_xc) / 2.0
                    col_placed = True
                    break
            if not col_placed:
                col_clusters.append(c_xc)

        col_clusters.sort()
        num_rows = len(rows)
        num_cols = max(len(col_clusters), max((len(r) for r in rows), default=1))

        # 3. Assign row_index and col_index to each cell
        rows_data: List[List[str]] = [["" for _ in range(num_cols)] for _ in range(num_rows)]

        for r_idx, r in enumerate(rows):
            r.sort(key=lambda c: c.bbox.x_min)
            for default_c_idx, cell in enumerate(r):
                c_xc = (cell.bbox.x_min + cell.bbox.x_max) / 2
                # Match closest column cluster if available
                if col_clusters:
                    closest_col = min(range(len(col_clusters)), key=lambda ci: abs(c_xc - col_clusters[ci]))
                else:
                    closest_col = default_c_idx

                cell.row_index = r_idx
                cell.col_index = min(closest_col, num_cols - 1)
                rows_data[cell.row_index][cell.col_index] = cell.text

        return num_rows, num_cols, rows_data

    def _generate_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        if not headers and not rows:
            return ""

        cols = max(len(headers), max((len(r) for r in rows), default=0))
        if cols == 0:
            return ""

        h_padded = headers + [""] * (cols - len(headers))
        md_lines = ["| " + " | ".join(h_padded) + " |", "| " + " | ".join(["---"] * cols) + " |"]

        for row in rows:
            r_padded = row + [""] * (cols - len(row))
            md_lines.append("| " + " | ".join(r_padded) + " |")

        return "\n".join(md_lines)
