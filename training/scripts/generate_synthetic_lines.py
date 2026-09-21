"""Synthetic Line-Level Kannada Handwriting Image & Manifest Generator.

Generates realistic multi-word text-line crops with real inter-word spacing,
known land-record vocabulary, rich Indic conjunct consonants (ottaksharas),
and scan-realistic image augmentations (blur, noise, skew, ink bleed)
for TrOCR line-level fine-tuning.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Rich Kannada land record vocabulary templates focusing heavily on conjuncts (ottaksharas)
NAMES_WITH_CONJUNCTS = [
    "ಕೃಷ್ಣಮೂರ್ತಿ", "ಲಕ್ಷ್ಮಣ ರಾವ್", "ಶ್ರೀನಿವಾಸಮೂರ್ತಿ", "ವಿಶ್ವನಾಥ್", "ಭಾಗ್ಯಮ್ಮ",
    "ಸಿದ್ಧರಾಮಯ್ಯ", "ದೊಡ್ಡನಂಜಪ್ಪ", "ಚಿಕ್ಕಮುನಿಯಪ್ಪ", "ಮಲ್ಲಪ್ಪ", "ಬಸವರಾಜು",
    "ಯಲ್ಲಮ್ಮ", "ವೆಂಕಟೇಶ್ವರ", "ಧನಂಜಯ", "ಪ್ರದೀಪ್ ಕುಮಾರ್", "ಸುಬ್ರಹ್ಮಣ್ಯ",
    "ಮಂಜುನಾಥ ಸ್ವಾಮಿ", "ರಾಮಕೃಷ್ಣಪ್ಪ", "ರುಕ್ಮಿಣಿಯಮ್ಮ", "ಅಶ್ವತ್ಥನಾರಾಯಣ",
    "ಸತ್ಯನಾರಾಯಣ", "ಚಂದ್ರಶೇಖರ್", "ಪರಮೇಶ್ವರಪ್ಪ", "ರುದ್ರಪ್ಪ", "ದ್ಯಾಮವ್ವ",
]

VILLAGES_AND_TALUKS = [
    ("ದೊಡ್ಡಬಳ್ಳಾಪುರ", "ಕಸಬಾ", "ದೊಡ್ಡಬಳ್ಳಾಪುರ", "ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ"),
    ("ಯಲಹಂಕ", "ಅತ್ತೂರು", "ಯಲಹಂಕ", "ಬೆಂಗಳೂರು ಉತ್ತರ"),
    ("ಕೆಂಗೇರಿ", "ತಾವರೆಕೆರೆ", "ಬೆಂಗಳೂರು ದಕ್ಷಿಣ", "ಬೆಂಗಳೂರು ನಗರ"),
    ("ಹೊಸಕೋಟೆ", "ಜಾಡಿಗೇನಹಳ್ಳಿ", "ಹೊಸಕೋಟೆ", "ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ"),
    ("ನೆಲಮಂಗಲ", "ತ್ಯಾಮಗೊಂಡ್ಲು", "ನೆಲಮಂಗಲ", "ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ"),
    ("ದೇವನಹಳ್ಳಿ", "ಕುಂದಾಣ", "ದೇವನಹಳ್ಳಿ", "ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ"),
    ("ಚನ್ನಪಟ್ಟಣ", "ವಿರೂಪಾಕ್ಷಿಪುರ", "ಚನ್ನಪಟ್ಟಣ", "ರಾಮನಗರ"),
    ("ಮಾಗಡಿ", "ಕುದೂರು", "ಮಾಗಡಿ", "ರಾಮನಗರ"),
]

CONJUNCT_LAND_TEMPLATES = [
    "ಖಾತೆದಾರರ ಹೆಸರು: {name} ಬಿನ್ {father_name}",
    "ಸರ್ವೆ ನಂಬರ್ {s_num}/{h_num} ಹಿಸ್ಸಾ {h_num} ರ ವಿಸ್ತೀರ್ಣ {acre} ಎಕರೆ {gunte} ಗುಂಟೆ",
    "ಗ್ರಾಮ: {village} ಹೋಬಳಿ: {hobli} ತಾಲೂಕು: {taluk} ಜಿಲ್ಲೆ: {district}",
    "ಸ್ಥಳೀಯ ಆದಿನಾರಾಯಣ ದೇವಸ್ಥಾನದ ಕ್ರಯಪತ್ರ ಮತ್ತು ಹಕ್ಕು ಬದಲಾವಣೆ ನಮೂನೆ ೧೨",
    "ನಿರ್ದಿಷ್ಟ ಜಮೀನಿನ ಚಕ್ಕುಬಂದಿ: ಪೂರ್ವಕ್ಕೆ {name}, ಪಶ್ಚಿಮಕ್ಕೆ ರಸ್ತೆ, ಉತ್ತರಕ್ಕೆ {name2}, ದಕ್ಷಿಣಕ್ಕೆ ಹಳ್ಳ",
    "ಹೊಯ್ಸಳ ಕಾಲದ ಬಲ್ಲಾಪುರ ತಾಂಡಾ ಜಂಟಿ ಖಾತೆದಾರರ ಸ್ವಾಧೀನಾನುಭವ ದೃಢೀಕರಣ ಪತ್ರ",
    "ಭ್ರಷ್ಟಗೊಂಡಿರದ ಸ್ಪಷ್ಟ ದಾಖಲೆಯ ಪ್ರಕಾರ ಪಹಣಿ ಮತ್ತು ಆರ್ ಟಿ ಸಿ ನಕಲು ಪ್ರತಿ",
    "ದೊಡ್ಡಬಳ್ಳಾಪುರ ಉಪನೋಂದಣಾಧಿಕಾರಿ ಕಚೇರಿಯಲ್ಲಿ ನೋಂದಾಯಿತ ಕ್ರಯಪತ್ರ ಸಂಖ್ಯೆ {reg_num}",
    "ಕಂದಾಯ ನಿಗದಿ: {tax_amount} ರೂಪಾಯಿ ವಸೂಲಾತಿ ತಹಶೀಲ್ದಾರ್ ಆದೇಶ ಸಂಖ್ಯೆ {order_num}",
    "ಖರಾಬು ಜಮೀನು {kharabu} ಗುಂಟೆ ಹೊರತುಪಡಿಸಿ ಉಳಿದ ಸ್ವಾಧೀನ ವಿಸ್ತೀರ್ಣ",
    "ದಿನಾಂಕ: {date} ರಂದು ತಹಶೀಲ್ದಾರ್ ನ್ಯಾಯಾಲಯದ ಮುಕ್ತ ನ್ಯಾಯಾಂಗ ತೀರ್ಪಿನನ್ವಯ",
    "ಅರ್ಜಿದಾರರ ಸಮ್ಮುಖದಲ್ಲಿ ಸ್ಥಳ ಪರಿಶೀಲನೆ ನಡೆಸಿ ಮಹಜರು ಜಪ್ತಿ ವರದಿ ಸಿದ್ಧಪಡಿಸಲಾಗಿದೆ",
    "ಪ್ರಸ್ತುತ ಸ್ವಾಧೀನದಾರರು: {name} ಮತ್ತು ಸಹೋದರರು ಜಂಟಿ ಹಿಸ್ಸಾದಾರರು",
    "ಸ್ವತ್ತಿನ ವಿವರ: ಕೃಷಿ ಜಮೀನು ನೀರಾವರಿ ಆಧಾರಿತ ಹಿಡುವಳಿ ಭಾಗಪತ್ರ",
    "ದಾನಪತ್ರ ನೋಂದಣಿ ದಿನಾಂಕ: {date} ರಜಿಸ್ಟ್ರಾರ್ ಕಚೇರಿ ಬೆಂಗಳೂರು ಗ್ರಾಮಾಂತರ",
    "ವಿಸ್ತೀರ್ಣ: {acre} ಎಕರೆ {gunte} ಗುಂಟೆ ಪೈಕಿ ದಕ್ಷಿಣ ಭಾಗದ ಸ್ವತ್ತು",
    "ಗ್ರಾಮದ ಲೆಕ್ಕಾಧಿಕಾರಿ ನಮೂನೆ ೧೨ ಪ್ರಕಾರ ಹಕ್ಕು ಬದಲಾವಣೆ ಮಂಜೂರಾಗಿದೆ",
    "ಪಟ್ಟಣದ ಅಡಿಪಾಯಕ್ಕೆ ಕಾರಣವಾಯಿತು ಎಂದು ಪ್ರಾಚೀನ ಇತಿಹಾಸ ದಾಖಲೆಗಳಲ್ಲಿ ಉಲ್ಲೇಖಿಸಲಾಗಿದೆ",
]

def generate_random_kannada_line() -> str:
    """Generates a random, conjunct-rich Kannada land-record line."""
    tmpl = random.choice(CONJUNCT_LAND_TEMPLATES)
    name = random.choice(NAMES_WITH_CONJUNCTS)
    father_name = random.choice(NAMES_WITH_CONJUNCTS)
    name2 = random.choice(NAMES_WITH_CONJUNCTS)
    village, hobli, taluk, district = random.choice(VILLAGES_AND_TALUKS)
    s_num = random.randint(12, 480)
    h_num = random.randint(1, 15)
    acre = random.randint(0, 8)
    gunte = random.randint(1, 39)
    kharabu = random.randint(0, 10)
    tax_amount = random.randint(25, 650)
    order_num = f"RD/{random.randint(100, 999)}/{random.randint(2018, 2024)}"
    reg_num = f"{random.randint(1000, 9999)}/{random.randint(2015, 2024)}"
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(1995, 2024)
    date_str = f"{day:02d}/{month:02d}/{year}"

    text = tmpl.format(
        name=name,
        father_name=father_name,
        name2=name2,
        village=village,
        hobli=hobli,
        taluk=taluk,
        district=district,
        s_num=s_num,
        h_num=h_num,
        acre=acre,
        gunte=gunte,
        kharabu=kharabu,
        tax_amount=tax_amount,
        order_num=order_num,
        reg_num=reg_num,
        date=date_str,
    )
    return text


# Handwriting and informal Kannada fonts
HANDWRITING_FONTS = [
    ("Navilu", Path("training/fonts/Navilu.ttf"), 0.55),
    ("Hubballi", Path("training/fonts/Hubballi-Regular.ttf"), 0.30),
    ("BalooTamma2", Path("training/fonts/BalooTamma2-Regular.ttf"), 0.15),
]


def pick_handwriting_font() -> Tuple[str, str]:
    """Selects a genuine handwriting or informal Kannada font with weighted probability."""
    # Filter available fonts
    available = []
    weights = []
    for name, p, w in HANDWRITING_FONTS:
        # Check relative to cwd or project root
        if p.exists():
            available.append((name, str(p.resolve())))
            weights.append(w)
        else:
            cand = Path(__file__).resolve().parent.parent / "fonts" / p.name
            if cand.exists():
                available.append((name, str(cand.resolve())))
                weights.append(w)

    if available:
        total_w = sum(weights)
        norm_w = [w / total_w for w in weights]
        chosen_idx = np.random.choice(len(available), p=norm_w)
        return available[chosen_idx]

    # Fallback to system fonts if none in training/fonts
    for sys_p in [
        r"C:\Windows\Fonts\Nirmala.ttc",
        "/usr/share/fonts/truetype/noto/NotoSerifKannada-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
    ]:
        if os.path.exists(sys_p):
            return "SystemFallback", sys_p

    return "Default", "arial.ttf"


def render_shaped_handwriting_line(
    text: str,
    font_path: str,
    font_size: int = 34,
) -> Image.Image:
    """Renders Kannada text with full OpenType conjunct ligature shaping and handwriting dynamics."""
    try:
        import freetype
        import uharfbuzz as hb
        has_hb = True
    except ImportError:
        has_hb = False

    if not has_hb or not os.path.exists(font_path):
        # Fallback to PIL basic rendering
        try:
            pil_font = ImageFont.truetype(font_path, font_size)
        except Exception:
            pil_font = ImageFont.load_default()
        dummy = Image.new("RGB", (100, 100), (255, 255, 255))
        d = ImageDraw.Draw(dummy)
        bbox = d.textbbox((0, 0), text, font=pil_font)
        w = max(120, bbox[2] - bbox[0] + 50)
        h = max(45, bbox[3] - bbox[1] + 30)
        canvas = Image.new("RGB", (w, h), (250, 250, 248))
        cd = ImageDraw.Draw(canvas)
        cd.text((25, 12), text, font=pil_font, fill=(25, 25, 30))
        return canvas

    # HarfBuzz OpenType Indic Shaping
    face = freetype.Face(font_path)
    face.set_char_size(font_size * 64)

    with open(font_path, "rb") as f:
        font_data = f.read()
    hb_face = hb.Face(font_data)
    hb_font = hb.Font(hb_face)
    hb_font.scale = (font_size * 64, font_size * 64)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    buf.script = "Knda"
    buf.language = "kan"
    hb.shape(hb_font, buf)

    infos = buf.glyph_infos
    positions = buf.glyph_positions

    total_w = sum(pos.x_advance for pos in positions) // 64 + 80
    h = int(font_size * 3.2)

    # Baseline wave parameters (human hand drift on unlined paper)
    wave_period = random.uniform(250, 480)
    wave_amp = random.uniform(1.2, 3.2)
    wave_phase = random.uniform(0, 2 * math.pi)

    canvas_gray = Image.new("L", (total_w, h), color=255)

    x = 40 * 64
    base_y = int(font_size * 1.8 * 64)

    for info, pos in zip(infos, positions):
        face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        bitmap = face.glyph.bitmap

        curr_x_px = x // 64
        wave_dy = int(math.sin(curr_x_px / wave_period * 2 * math.pi + wave_phase) * wave_amp)

        bx = (x + pos.x_offset) // 64 + face.glyph.bitmap_left
        by = (base_y - pos.y_offset) // 64 - face.glyph.bitmap_top + wave_dy

        if bitmap.width > 0 and bitmap.rows > 0:
            glyph_arr = np.array(bitmap.buffer, dtype=np.uint8).reshape((bitmap.rows, bitmap.width))
            for r in range(bitmap.rows):
                for c in range(bitmap.width):
                    cy = by + r
                    cx = bx + c
                    if 0 <= cy < h and 0 <= cx < total_w:
                        curr = canvas_gray.getpixel((cx, cy))
                        alpha = glyph_arr[r, c] / 255.0
                        new_val = int(curr * (1.0 - alpha))
                        canvas_gray.putpixel((cx, cy), new_val)

        # Subtle natural inter-character advance jitter
        advance_jitter = int(random.uniform(-1, 2) * 64)
        x += pos.x_advance + advance_jitter

    # Handwriting shear / slant angle (radians)
    shear_angle = random.uniform(-0.10, 0.12)
    m = [1, shear_angle, -shear_angle * (h / 2), 0, 1, 0]
    canvas_gray = canvas_gray.transform(canvas_gray.size, Image.AFFINE, m, Image.BILINEAR, fillcolor=255)

    # Convert to RGB paper surface with authentic ink simulation
    arr = np.array(canvas_gray, dtype=np.float32)

    bg_val = random.randint(244, 252)
    bg_tint = random.randint(2, 6)
    rgb = np.zeros((h, total_w, 3), dtype=np.uint8)

    # Realistic ink shades: blue, black, blue-black
    ink_choices = [
        (random.randint(15, 35), random.randint(20, 45), random.randint(35, 75)),  # Blue-black
        (random.randint(15, 30), random.randint(15, 30), random.randint(18, 35)),  # Black
        (random.randint(10, 30), random.randint(30, 60), random.randint(80, 130)), # Blue ink
    ]
    ink_rgb = random.choice(ink_choices)

    alpha = (255.0 - arr) / 255.0
    for ch, c_val in enumerate(ink_rgb):
        bg_ch = bg_val - (bg_tint if ch == 2 else 0)
        ch_arr = bg_ch * (1.0 - alpha) + c_val * alpha
        ch_arr += np.random.normal(0, 1.8, ch_arr.shape)
        rgb[:, :, ch] = np.clip(ch_arr, 0, 255).astype(np.uint8)

    out = Image.fromarray(rgb)
    if random.random() < 0.40:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.5)))
    return out


def generate_dataset(
    output_dir: str = "training/datasets/synthetic_lines",
    num_samples: int = 500,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> Tuple[str, str, str]:
    """Generates synthetic handwriting line images and manifests with explicit provenance."""
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"Generating {num_samples} synthetic handwriting line samples...")
    for i in range(num_samples):
        text = generate_random_kannada_line()
        font_name, font_path = pick_handwriting_font()
        font_size = random.randint(30, 36)

        img = render_shaped_handwriting_line(text, font_path=font_path, font_size=font_size)
        img_name = f"line_{i + 1:05d}.png"
        img_file = images_dir / img_name
        img.save(img_file)

        has_conjunct = "್" in text
        records.append({
            "image": str((images_dir / img_name).as_posix()),
            "text": text,
            "language": "kannada",
            "script": "Kannada",
            "metadata": {
                "source": "synthetic_handwriting_simulation",
                "font": font_name,
                "dataset_stage": "smoke_test_synthetic_handwriting_v1",
                "is_real_handwriting": False,
                "purpose": "anti_repetition_and_conjunct_smoke_test",
                "has_conjunct": has_conjunct,
                "contains_spaces": True,
            },
        })

    # Shuffle and split
    random.seed(42)
    random.shuffle(records)
    n_train = int(len(records) * train_ratio)
    n_val = int(len(records) * val_ratio)

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

    train_file = out_path / "train.jsonl"
    val_file = out_path / "val.jsonl"
    test_file = out_path / "test.jsonl"

    for path, data in [(train_file, train_records), (val_file, val_records), (test_file, test_records)]:
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Generated {num_samples} samples: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")
    return str(train_file), str(val_file), str(test_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic line-level handwriting dataset")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="training/datasets/synthetic_lines")
    args = parser.parse_args()
    generate_dataset(output_dir=args.output_dir, num_samples=args.num_samples)

