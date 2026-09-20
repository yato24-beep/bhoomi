"""Synthetic Karnataka Bhoomi RTC (Pahani) Document Generator.

Generates:
- demo_artifacts/synthetic_karnataka_rtc.png
- demo_artifacts/synthetic_karnataka_rtc.pdf

Contains realistic government headers, cadastral administrative details,
ownership tables, printed Kannada text, and short handwritten entries
with controlled review triggers.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = Path("demo_artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUTPUT_DIR / "synthetic_karnataka_rtc.png"
PDF_PATH = OUTPUT_DIR / "synthetic_karnataka_rtc.pdf"

FONT_PATH = r"C:\Windows\Fonts\Nirmala.ttc"
FALLBACK_FONT = r"C:\Windows\Fonts\arial.ttf"


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        # Index 0 or 1 in TTC is usually regular/bold
        return ImageFont.truetype(FONT_PATH, size=size, index=1 if bold else 0)
    except Exception:
        try:
            return ImageFont.truetype(FONT_PATH, size=size)
        except Exception:
            return ImageFont.truetype(FALLBACK_FONT, size=size)


def generate_rtc_image() -> Image.Image:
    # A4 standard at 200 DPI: ~1654 x 2338
    width = 1654
    height = 2338
    img = Image.new("RGB", (width, height), color=(254, 254, 252))
    draw = ImageDraw.Draw(img)

    # Outer Document Border
    draw.rectangle([(40, 40), (width - 40, height - 40)], outline=(40, 40, 40), width=3)
    draw.rectangle([(48, 48), (width - 48, height - 48)], outline=(120, 120, 120), width=1)

    # Fonts
    f_title = get_font(38, bold=True)
    f_subtitle = get_font(28, bold=True)
    f_sec_hdr = get_font(22, bold=True)
    f_tbl_hdr = get_font(18, bold=True)
    f_body = get_font(18, bold=False)
    f_body_bold = get_font(18, bold=True)
    f_small = get_font(14, bold=False)

    # 1. Header Section
    y = 70
    draw.text((width // 2, y), "ಕರ್ನಾಟಕ ಸರ್ಕಾರ", font=f_title, fill=(20, 20, 20), anchor="mt")
    y += 55
    draw.text((width // 2, y), "ಕಂದಾಯ ಇಲಾಖೆ - ಭೂಮಿ ಯೋಜನೆ (RTC / ಪಹಣಿ)", font=f_subtitle, fill=(30, 30, 30), anchor="mt")
    y += 45
    draw.text((width // 2, y), "ಹಕ್ಕು ದಾಖಲೆ, ಬೆಳೆ ಮತ್ತು ಬಾಡಿಗೆ ಪರಿಶೀಲನೆ (ನಮೂನೆ 16)", font=f_sec_hdr, fill=(50, 50, 50), anchor="mt")
    y += 40
    draw.line([(70, y), (width - 70, y)], fill=(60, 60, 60), width=2)
    y += 20

    # 2. Administrative Details Table (Location Hierarchy)
    admin_data = [
        [("ಜಿಲ್ಲೆ (District):", "ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ"), ("ತಾಲೂಕು (Taluk):", "ದೊಡ್ಡಬಳ್ಳಾಪುರ")],
        [("ಹೋಬಳಿ (Hobli):", "ಕಸಬಾ"), ("ಗ್ರಾಮ (Village):", "ನಾಗದೇವನಹಳ್ಳಿ")],
        [("ವರ್ಷ (Year):", "2024-2025"), ("ದಿನಾಂಕ (Date):", "15/03/2024")],
    ]

    table_left = 70
    table_right = width - 70
    table_width = table_right - table_left
    col_w = table_width // 2

    for row in admin_data:
        row_h = 45
        draw.rectangle([(table_left, y), (table_right, y + row_h)], outline=(160, 160, 160), width=1)
        draw.line([(table_left + col_w, y), (table_left + col_w, y + row_h)], fill=(160, 160, 160), width=1)

        # Col 1
        label1, val1 = row[0]
        draw.text((table_left + 15, y + 12), label1, font=f_tbl_hdr, fill=(40, 40, 40))
        draw.text((table_left + 220, y + 12), val1, font=f_body_bold, fill=(10, 10, 10))

        # Col 2
        label2, val2 = row[1]
        draw.text((table_left + col_w + 15, y + 12), label2, font=f_tbl_hdr, fill=(40, 40, 40))
        draw.text((table_left + col_w + 220, y + 12), val2, font=f_body_bold, fill=(10, 10, 10))

        y += row_h

    y += 25

    # 3. Land Identification & Area Details Section
    draw.text((table_left, y), "೧. ಜಮೀನಿನ ವಿವರಗಳು (Land Survey Details)", font=f_sec_hdr, fill=(30, 30, 30))
    y += 35

    survey_cols = [
        ("ಸರ್ವೆ ನಂಬರ್\n(Survey No)", 220),
        ("ಹಿಸ್ಸಾ ನಂಬರ್\n(Hissa No)", 200),
        ("ಖಾತಾ ನಂಬರ್\n(Khata No)", 200),
        ("ಒಟ್ಟು ವಿಸ್ತೀರ್ಣ\n(Total Extent)", 250),
        ("ಖರಾಬು\n(Kharab)", 180),
        ("ಬಾಕಿ ವಿಸ್ತೀರ್ಣ\n(Net Extent)", 250),
        ("ಭೂಮಿ ಕಂದಾಯ\n(Assessment)", table_width - (220 + 200 + 200 + 250 + 180 + 250)),
    ]

    # Header Row
    hdr_h = 55
    draw.rectangle([(table_left, y), (table_right, y + hdr_h)], fill=(240, 243, 246), outline=(120, 120, 120), width=1)
    cur_x = table_left
    for col_name, w in survey_cols:
        draw.text((cur_x + 10, y + 8), col_name, font=f_small, fill=(30, 30, 30))
        cur_x += w
        if cur_x < table_right:
            draw.line([(cur_x, y), (cur_x, y + hdr_h)], fill=(160, 160, 160), width=1)

    y += hdr_h

    # Value Row
    val_h = 50
    draw.rectangle([(table_left, y), (table_right, y + val_h)], outline=(120, 120, 120), width=1)
    cur_x = table_left
    survey_values = ["42", "1", "108", "2-15", "0-02", "2-13", "14.50 ರೂ."]
    for idx, (val, (_, w)) in enumerate(zip(survey_values, survey_cols)):
        draw.text((cur_x + 20, y + 14), val, font=f_body_bold, fill=(10, 10, 10))
        cur_x += w
        if cur_x < table_right:
            draw.line([(cur_x, y), (cur_x, y + val_h)], fill=(160, 160, 160), width=1)

    y += val_h + 30

    # 4. Ownership Details Section (Form 9 / Column 9 & 10)
    draw.text((table_left, y), "೨. ಮಾಲೀಕರ ವಿವರಗಳು (Ownership & Occupancy Details)", font=f_sec_hdr, fill=(30, 30, 30))
    y += 35

    owner_cols = [
        ("ಕ್ರಮ ಸಂಖ್ಯೆ\n(Sl. No)", 120),
        ("ಖಾತೆದಾರರ ಹೆಸರು / ಮಾಲೀಕರು\n(Owner / Khatedar Name)", 420),
        ("ವಿಸ್ತೀರ್ಣ\n(Extent)", 220),
        ("ಸ್ವಾಧೀನದ ವಿಧ / ಹಕ್ಕು\n(Nature of Rights)", 320),
        ("ಖಾತೆ ಸಂಖ್ಯೆ\n(Khata No)", 200),
        ("ಷರಾ / ಮ್ಯುಟೇಶನ್ ನಂ\n(Mutation No)", table_width - (120 + 420 + 220 + 320 + 200)),
    ]

    hdr_h = 55
    draw.rectangle([(table_left, y), (table_right, y + hdr_h)], fill=(240, 243, 246), outline=(120, 120, 120), width=1)
    cur_x = table_left
    for col_name, w in owner_cols:
        draw.text((cur_x + 10, y + 8), col_name, font=f_small, fill=(30, 30, 30))
        cur_x += w
        if cur_x < table_right:
            draw.line([(cur_x, y), (cur_x, y + hdr_h)], fill=(160, 160, 160), width=1)

    y += hdr_h

    # Row 1: Primary Owner (ರಾಮಪ್ಪ)
    val_h = 60
    draw.rectangle([(table_left, y), (table_right, y + val_h)], outline=(120, 120, 120), width=1)
    cur_x = table_left
    owner_row_vals = ["1", "ರಾಮಪ್ಪ ಬಿನ್ ತಿಮ್ಮಪ್ಪ", "2-13", "ಖುದ್ದ / ಪಿತ್ರಾರ್ಜಿತ", "108", "MR T-42/2018"]
    for idx, (val, (_, w)) in enumerate(zip(owner_row_vals, owner_cols)):
        draw.text((cur_x + 15, y + 18), val, font=f_body_bold, fill=(10, 10, 10))
        cur_x += w
        if cur_x < table_right:
            draw.line([(cur_x, y), (cur_x, y + val_h)], fill=(160, 160, 160), width=1)

    y += val_h + 30

    # 5. Cultivation Details (Pahani / Col 11 - Cultivator Information)
    draw.text((table_left, y), "೩. ಸಾಗುವಳಿ ವಿವರಗಳು (Cultivation & Tenancy Details)", font=f_sec_hdr, fill=(30, 30, 30))
    y += 35

    cult_cols = [
        ("ಹಂಗಾಮು\n(Season)", 200),
        ("ಬೆಳೆ ಹೆಸರು\n(Crop Name)", 280),
        ("ವಿಸ್ತೀರ್ಣ\n(Extent)", 220),
        ("ನೀರಿನ ಮೂಲ\n(Water Source)", 250),
        ("ಸಾಗುವಳಿದಾರರ ಹೆಸರು\n(Cultivator Name)", table_width - (200 + 280 + 220 + 250)),
    ]

    hdr_h = 55
    draw.rectangle([(table_left, y), (table_right, y + hdr_h)], fill=(240, 243, 246), outline=(120, 120, 120), width=1)
    cur_x = table_left
    for col_name, w in cult_cols:
        draw.text((cur_x + 10, y + 8), col_name, font=f_small, fill=(30, 30, 30))
        cur_x += w
        if cur_x < table_right:
            draw.line([(cur_x, y), (cur_x, y + hdr_h)], fill=(160, 160, 160), width=1)

    y += hdr_h

    # Row: Cultivation record with printed crop + a controlled handwriting/ambiguous entry for cultivator
    val_h = 60
    draw.rectangle([(table_left, y), (table_right, y + val_h)], outline=(120, 120, 120), width=1)
    cur_x = table_left
    cult_vals = ["ಖಾರೀಫ್ 2024", "ರಾಗಿ (ಕೃಷಿ)", "2-13", "ಮಳೆ ಆಶ್ರಯ (Rainfed)"]
    for idx, (val, (_, w)) in enumerate(zip(cult_vals, cult_cols[:4])):
        draw.text((cur_x + 15, y + 18), val, font=f_body, fill=(20, 20, 20))
        cur_x += w
        draw.line([(cur_x, y), (cur_x, y + val_h)], fill=(160, 160, 160), width=1)

    # Controlled handwritten cultivator entry:
    # "ಭೀಮಯ್ಯ" rendered in a cursive style with slight blur/fade
    # to naturally yield low confidence (< 0.70) in recognizer and trigger honest REVIEW_REQUIRED
    cult_box_x = cur_x + 15
    cult_box_y = y + 12

    hw_img = Image.new("RGBA", (280, 45), (255, 255, 255, 0))
    hw_draw = ImageDraw.Draw(hw_img)
    hw_font = get_font(19, bold=False)
    hw_draw.text((10, 6), "ಭೀಮಯ್ಯ", font=hw_font, fill=(90, 90, 110, 210))
    hw_blurred = hw_img.filter(ImageFilter.GaussianBlur(radius=0.7))
    img.paste(hw_blurred, (cult_box_x, cult_box_y), hw_blurred)

    y += val_h + 50

    # 6. Footer / Official Seal & Disclaimer
    draw.line([(table_left, y), (table_right, y)], fill=(180, 180, 180), width=1)
    y += 20
    draw.text((table_left, y), "ಗಮನಿಸಿ: ಇದು ಗಣಕೀಕೃತ ಭೂಮಿ ಕಂದಾಯ ದಾಖಲೆಯಾಗಿದೆ (Bhoomi Digital Land Record System).", font=f_small, fill=(70, 70, 70))
    y += 24
    draw.text((table_left, y), "ಅಧಿಕೃತ ಸಹಿ / ಕಂದಾಯ ಪರಿವೀಕ್ಷಕರು (Revenue Inspector, Doddaballapura Taluk).", font=f_small, fill=(70, 70, 70))

    # Official Seal / Stamp graphic
    seal_x = table_right - 220
    seal_y = y - 40
    draw.ellipse([(seal_x, seal_y), (seal_x + 140, seal_y + 70)], outline=(40, 60, 140), width=2)
    draw.text((seal_x + 25, seal_y + 15), "ಕಂದಾಯ ಇಲಾಖೆ", font=get_font(12, bold=True), fill=(40, 60, 140))
    draw.text((seal_x + 35, seal_y + 35), "ಅಧಿಕೃತ ಮುದ್ರೆ", font=get_font(12, bold=False), fill=(40, 60, 140))

    return img


def main():
    print("Generating synthetic Karnataka RTC document...")
    img = generate_rtc_image()

    # Save PNG
    img.save(PNG_PATH, format="PNG", dpi=(200, 200))
    print(f"Saved PNG to: {PNG_PATH}")

    # Save PDF
    img.save(PDF_PATH, format="PDF", resolution=200.0)
    print(f"Saved PDF to: {PDF_PATH}")


if __name__ == "__main__":
    main()
