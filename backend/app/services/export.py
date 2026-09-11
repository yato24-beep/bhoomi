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
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


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
    """Generates a professional PDF report using ReportLab."""
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
        alignment=1,  # Center
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

    # Title & Header
    elements.append(Paragraph("LAND RECORD DIGITIZATION REPORT", title_style))
    elements.append(Paragraph("Automated Multimodal Land Record Reconstruction System", subtitle_style))
    elements.append(Spacer(1, 10))

    # Official Disclaimer
    disclaimer_text = (
        "<b>Notice:</b> Digitally reconstructed from uploaded land record. "
        "Requires official verification before legal use."
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    # Meta Status Table
    conf_pct = f"{int(round(overall_confidence * 100))}%"
    status_label = "Digitized Successfully" if not requires_review else "Digitized (Requires Review)"
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

    # Property Details Section
    elements.append(Paragraph("PROPERTY DETAILS", section_style))
    prop_rows = [["Property Field", "Extracted Value"]]
    field_labels = [
        ("Survey Number", fields.get("khasra_number", "Not confidently detected")),
        ("Document Date", fields.get("document_date", "Not confidently detected")),
        ("Extent / Area", fields.get("land_area", "Not confidently detected")),
        ("Document Type", fields.get("document_title", "Not confidently detected")),
        ("Owner Name", fields.get("owner_name", "Not confidently detected")),
        ("Father / Husband", fields.get("father_or_husband_name", "Not confidently detected")),
        ("Village", fields.get("village", "Not confidently detected")),
        ("Taluk", fields.get("tehsil", "Not confidently detected")),
        ("District", fields.get("district", "Not confidently detected")),
    ]
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

    # English Translation Section
    elements.append(Paragraph("ENGLISH TRANSLATION", section_style))
    clean_en = (english_translation or "").strip()
    if not clean_en:
        clean_en = "No English translation available."
    en_paragraphs = [Paragraph(p.strip(), body_style) for p in clean_en.split("\n") if p.strip()]
    if en_paragraphs:
        elements.extend(en_paragraphs[:8])
    else:
        elements.append(Paragraph("No English translation available.", body_style))
    elements.append(Spacer(1, 14))

    # Original Kannada Script Section
    elements.append(Paragraph("ORIGINAL RECOGNIZED SCRIPT (KANNADA)", section_style))
    clean_kn = (kannada_text or "").strip()
    if not clean_kn:
        clean_kn = "No Kannada script extracted."
    kn_paragraphs = [Paragraph(p.strip(), body_style) for p in clean_kn.split("\n") if p.strip()]
    if kn_paragraphs:
        elements.extend(kn_paragraphs[:20])
    else:
        elements.append(Paragraph("No Kannada script extracted.", body_style))

    # Verification / Review Reasons (if any)
    if requires_review and review_reasons:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("OCR REVIEW NOTICES", section_style))
        for r in review_reasons[:3]:
            elements.append(Paragraph(f"• {r}", body_style))

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
    """Generates a professional Word (.docx) document using python-docx."""
    doc = Document()

    # Title
    title = doc.add_heading("LAND RECORD DIGITIZATION REPORT", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph("Automated Multimodal Land Record Reconstruction System")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Disclaimer
    disc = doc.add_paragraph("Digitally reconstructed from uploaded land record. Requires official verification before legal use.")
    disc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    disc.italic = True

    doc.add_paragraph()  # Spacer

    # Summary Info Table
    summary_table = doc.add_table(rows=2, cols=2)
    summary_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    summary_table.rows[0].cells[0].text = f"Document ID: {document_id}"
    status_label = "Digitized Successfully" if not requires_review else "Digitized (Requires Review)"
    summary_table.rows[0].cells[1].text = f"Status: {status_label}"
    summary_table.rows[1].cells[0].text = f"Source File: {filename}"
    conf_pct = f"{int(round(overall_confidence * 100))}%"
    summary_table.rows[1].cells[1].text = f"Overall Confidence: {conf_pct}"

    doc.add_paragraph()

    # Property Details
    doc.add_heading("PROPERTY DETAILS", level=2)
    field_labels = [
        ("Survey Number", fields.get("khasra_number", "Not confidently detected")),
        ("Document Date", fields.get("document_date", "Not confidently detected")),
        ("Extent / Area", fields.get("land_area", "Not confidently detected")),
        ("Document Type", fields.get("document_title", "Not confidently detected")),
        ("Owner Name", fields.get("owner_name", "Not confidently detected")),
        ("Father / Husband", fields.get("father_or_husband_name", "Not confidently detected")),
        ("Village", fields.get("village", "Not confidently detected")),
        ("Taluk", fields.get("tehsil", "Not confidently detected")),
        ("District", fields.get("district", "Not confidently detected")),
    ]

    prop_table = doc.add_table(rows=len(field_labels) + 1, cols=2)
    prop_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    prop_table.rows[0].cells[0].text = "Field"
    prop_table.rows[0].cells[1].text = "Extracted Value"
    for r_idx, (lbl, val) in enumerate(field_labels, start=1):
        prop_table.rows[r_idx].cells[0].text = lbl
        prop_table.rows[r_idx].cells[1].text = val if val else "Not confidently detected"

    doc.add_paragraph()

    # English Translation
    doc.add_heading("ENGLISH TRANSLATION", level=2)
    doc.add_paragraph((english_translation or "No English translation available.").strip())

    doc.add_paragraph()

    # Original Kannada Script
    doc.add_heading("ORIGINAL RECOGNIZED SCRIPT (KANNADA)", level=2)
    doc.add_paragraph((kannada_text or "No Kannada script extracted.").strip())

    if requires_review and review_reasons:
        doc.add_paragraph()
        doc.add_heading("OCR REVIEW NOTICES", level=2)
        for r in review_reasons:
            doc.add_paragraph(f"• {r}")

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
