"""Document export services for generating downloadable PDF and DOCX reports.

Generates judge-ready, beautifully styled documents including:
- Official disclaimer: 'Digitally reconstructed from uploaded land record. Requires official verification before legal use.'
- Property details table
- Recognized Kannada text
- English translation
- Verification & review status
"""

import io
from typing import Any, Dict, List, Optional

# Optional ReportLab and Docx imports with fallback
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


def _generate_minimal_pdf(
    title: str,
    disclaimer: str,
    metadata: List[tuple],
    fields: List[tuple],
    english_text: str,
    kannada_text: str,
) -> bytes:
    """Generates standard standalone valid PDF 1.4 byte stream without third-party dependencies."""
    content_lines = [
        "BT",
        "/F1 16 Tf",
        "50 750 Td",
        f"({title}) Tj",
        "/F1 9 Tf",
        "0 -20 Td",
        f"({disclaimer}) Tj",
        "/F1 11 Tf",
        "0 -30 Td",
        "(DOCUMENT SUMMARY:) Tj",
        "/F1 9 Tf",
    ]
    for k, v in metadata:
        safe_v = str(v).replace("(", "[").replace(")", "]")
        content_lines.extend(["0 -15 Td", f"({k}: {safe_v}) Tj"])

    content_lines.extend([
        "/F1 11 Tf",
        "0 -25 Td",
        "(STRUCTURED PROPERTY DETAILS:) Tj",
        "/F1 9 Tf",
    ])
    for label, val in fields:
        safe_val = str(val).replace("(", "[").replace(")", "]")
        content_lines.extend(["0 -15 Td", f"({label}: {safe_val}) Tj"])

    content_lines.extend([
        "/F1 11 Tf",
        "0 -25 Td",
        "(ENGLISH TRANSLATION:) Tj",
        "/F1 9 Tf",
    ])
    for eline in (english_text or "No translation available").splitlines()[:6]:
        if eline.strip():
            safe_el = eline.strip().replace("(", "[").replace(")", "]")
            content_lines.extend(["0 -14 Td", f"({safe_el}) Tj"])

    content_lines.append("ET")
    stream_content = "\n".join(content_lines).encode("latin-1", errors="replace")
    stream_len = len(stream_content)

    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"5 0 obj\n<< /Length " + str(stream_len).encode("ascii") + b" >>\nstream\n"
        + stream_content +
        b"\nendstream\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n0000000226 00000 n \n0000000297 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + str(350 + stream_len).encode("ascii") + b"\n%%EOF\n"
    )
    return pdf


def build_pdf_export(
    document_id: str,
    filename: str,
    overall_confidence: float,
    status: str,
    fields: Dict[str, str],
    kannada_text: str,
    english_translation: str,
    requires_review: bool,
    review_reasons: Optional[List[str]] = None,
) -> bytes:
    """Generates a professional PDF report using ReportLab or built-in generator."""
    field_labels = [
        ("Survey Number", fields.get("khasra_number") or fields.get("survey_number", "Not confidently detected")),
        ("Document Date", fields.get("document_date") or fields.get("date", "Not confidently detected")),
        ("Extent / Area", fields.get("land_area") or fields.get("site_area", "Not confidently detected")),
        ("Document Type", fields.get("document_title") or fields.get("document_type", "Not confidently detected")),
        ("Owner Name", fields.get("owner_name", "Not confidently detected")),
        ("Locality / Village", fields.get("locality") or fields.get("village", "Not confidently detected")),
        ("Taluk", fields.get("tehsil") or fields.get("taluk", "Not confidently detected")),
        ("District", fields.get("district", "Not confidently detected")),
    ]
    conf_pct = f"{int(round(overall_confidence * 100))}%"
    status_label = "Digitized Successfully" if not requires_review else "Digitized (Requires Review)"

    if not HAS_REPORTLAB:
        return _generate_minimal_pdf(
            title="LAND RECORD DIGITIZATION REPORT",
            disclaimer="Notice: Digitally reconstructed from uploaded land record. Requires official verification.",
            metadata=[
                ("Document ID", str(document_id)),
                ("Status", status_label),
                ("Filename", filename),
                ("Confidence", conf_pct),
            ],
            fields=field_labels,
            english_text=english_translation,
            kannada_text=kannada_text,
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        alignment=1,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        alignment=1,
    )
    badge_style = ParagraphStyle(
        "BadgeText",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#047857") if not requires_review else colors.HexColor("#b45309"),
    )

    elements = []
    elements.append(Paragraph("LAND RECORD DIGITIZATION REPORT", title_style))
    elements.append(Paragraph("Automated Multimodal Land Record Reconstruction System", subtitle_style))
    elements.append(Spacer(1, 10))

    disclaimer_text = (
        "<b>Notice:</b> Digitally reconstructed from uploaded land record. "
        "Requires official verification before legal use."
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    meta_data = [
        [
            Paragraph(f"<b>Document ID:</b> {document_id}", body_style),
            Paragraph(f"<b>Status:</b> {status_label}", badge_style),
        ],
        [
            Paragraph(f"<b>Source File:</b> {filename}", body_style),
            Paragraph(f"<b>Confidence:</b> {conf_pct}", body_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[3.5 * inch, 3.5 * inch])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("PROPERTY DETAILS", section_style))
    prop_rows = [["Property Field", "Extracted Value"]]
    for label, val in field_labels:
        val_display = val if val else "Not confidently detected"
        prop_rows.append([Paragraph(f"<b>{label}</b>", body_style), Paragraph(str(val_display), body_style)])

    prop_table = Table(prop_rows, colWidths=[2.5 * inch, 4.5 * inch])
    prop_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
        ])
    )
    elements.append(prop_table)
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("ENGLISH TRANSLATION", section_style))
    clean_en = (english_translation or "").strip() or "No English translation available."
    for p in clean_en.split("\n")[:8]:
        if p.strip():
            elements.append(Paragraph(p.strip(), body_style))
    elements.append(Spacer(1, 14))

    elements.append(Paragraph("ORIGINAL RECOGNIZED SCRIPT (KANNADA)", section_style))
    clean_kn = (kannada_text or "").strip() or "No Kannada script extracted."
    for p in clean_kn.split("\n")[:15]:
        if p.strip():
            elements.append(Paragraph(p.strip(), body_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def build_docx_export(
    document_id: str,
    filename: str,
    overall_confidence: float,
    status: str,
    fields: Dict[str, str],
    kannada_text: str,
    english_translation: str,
    requires_review: bool,
    review_reasons: Optional[List[str]] = None,
) -> bytes:
    """Generates a professional Word document or formatted rich text."""
    conf_pct = f"{int(round(overall_confidence * 100))}%"
    status_label = "Digitized Successfully" if not requires_review else "Digitized (Requires Review)"
    field_labels = [
        ("Survey Number", fields.get("khasra_number") or fields.get("survey_number", "Not confidently detected")),
        ("Document Date", fields.get("document_date") or fields.get("date", "Not confidently detected")),
        ("Extent / Area", fields.get("land_area") or fields.get("site_area", "Not confidently detected")),
        ("Document Type", fields.get("document_title") or fields.get("document_type", "Not confidently detected")),
        ("Owner Name", fields.get("owner_name", "Not confidently detected")),
        ("Locality / Village", fields.get("locality") or fields.get("village", "Not confidently detected")),
        ("Taluk", fields.get("tehsil") or fields.get("taluk", "Not confidently detected")),
        ("District", fields.get("district", "Not confidently detected")),
    ]

    if not HAS_DOCX:
        # Generate formatted plain text / RTF stream
        doc_lines = [
            "LAND RECORD DIGITIZATION REPORT",
            "=" * 50,
            "Notice: Digitally reconstructed from uploaded land record. Requires official verification.",
            f"Document ID: {document_id} | Status: {status_label} | Confidence: {conf_pct}",
            f"Filename: {filename}",
            "-" * 50,
            "PROPERTY DETAILS:",
        ]
        for lbl, val in field_labels:
            doc_lines.append(f"  • {lbl}: {val}")
        doc_lines.extend([
            "-" * 50,
            "ENGLISH TRANSLATION:",
            english_translation or "No translation available",
            "-" * 50,
            "ORIGINAL RECOGNIZED SCRIPT (KANNADA):",
            kannada_text or "No Kannada script extracted",
        ])
        return "\n".join(doc_lines).encode("utf-8")

    doc = Document()
    title = doc.add_heading("LAND RECORD DIGITIZATION REPORT", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph("Automated Multimodal Land Record Reconstruction System")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    disc = doc.add_paragraph("Digitally reconstructed from uploaded land record. Requires official verification before legal use.")
    disc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    disc.italic = True

    doc.add_paragraph()

    summary_table = doc.add_table(rows=2, cols=2)
    summary_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    summary_table.rows[0].cells[0].text = f"Document ID: {document_id}"
    summary_table.rows[0].cells[1].text = f"Status: {status_label}"
    summary_table.rows[1].cells[0].text = f"Source File: {filename}"
    summary_table.rows[1].cells[1].text = f"Overall Confidence: {conf_pct}"

    doc.add_paragraph()

    doc.add_heading("PROPERTY DETAILS", level=2)
    prop_table = doc.add_table(rows=len(field_labels) + 1, cols=2)
    prop_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    prop_table.rows[0].cells[0].text = "Field"
    prop_table.rows[0].cells[1].text = "Extracted Value"
    for r_idx, (lbl, val) in enumerate(field_labels, start=1):
        prop_table.rows[r_idx].cells[0].text = lbl
        prop_table.rows[r_idx].cells[1].text = str(val) if val else "Not confidently detected"

    doc.add_paragraph()
    doc.add_heading("ENGLISH TRANSLATION", level=2)
    doc.add_paragraph((english_translation or "No English translation available.").strip())

    doc.add_paragraph()
    doc.add_heading("ORIGINAL RECOGNIZED SCRIPT (KANNADA)", level=2)
    doc.add_paragraph((kannada_text or "No Kannada script extracted.").strip())

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
