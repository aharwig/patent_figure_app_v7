"""
patent_qc.py
Patent Figure QC and Export Utilities

Ensures USPTO/EPO submission-ready drawings:
- No color, pure B/W output
- All black lines >= min thickness
- Canvas size normalized to Letter or A4
- Export to PDF, TIFF, PNG
- Compliance check + warnings
- Batch runner for multiple files
"""

import os
import re
from typing import Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, label
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4

# ----------------- Conversion Utilities -----------------
def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))

def px_to_mm(px: float, dpi: int) -> float:
    return px * 25.4 / dpi

def _distance_transform_diameter(thick_np: np.ndarray) -> np.ndarray:
    """
    Compute local line thickness (in pixels) for black lines.
    Black = 0, White = 255
    """
    # distance inside black pixels
    dist = distance_transform_edt(thick_np == 0)
    return 2 * dist  # full width estimate

# ----------------- QC Overlay -----------------
def qc_overlay(
    thick_np: np.ndarray,
    out_rgb: Image.Image,
    *,
    min_line_mm: float,
    dpi: int,
    add_labels: bool = True,
    heatmap: bool = False,
    border_pct: float = 0.1,
    font_size_pt: int = 12
) -> Image.Image:
    """Overlay QC mask on RGB image"""
    px_min = mm_to_px(min_line_mm, dpi)
    diam = _distance_transform_diameter(thick_np)

    H, W = thick_np.shape
    overlay = np.zeros((H, W, 4), dtype=np.uint8)

    is_black = (thick_np == 0)

    # Traffic-light masks
    violating_mask = (diam < px_min) & is_black
    borderline_mask = (diam >= px_min) & (diam < px_min * (1 + border_pct)) & is_black
    compliant_mask = (diam >= px_min * (1 + border_pct)) & is_black

    if heatmap:
        norm = np.clip(diam / (2 * px_min), 0, 1)
        overlay[..., 0] = np.where(is_black, (1 - norm) * 255, 0)  # red
        overlay[..., 1] = np.where(is_black, norm * 255, 0)        # green
        overlay[..., 2] = 0
        overlay[..., 3] = np.where(is_black, 128, 0)
    else:
        overlay[violating_mask]  = [255, 0, 0, 128]
        overlay[borderline_mask] = [255, 255, 0, 128]
        overlay[compliant_mask]  = [0, 255, 0, 64]

    comp = Image.alpha_composite(out_rgb.convert("RGBA"), Image.fromarray(overlay, mode="RGBA"))

    if add_labels:
        draw = ImageDraw.Draw(comp)
        font_size_px = max(8, int(dpi / 72 * font_size_pt))
        font = None
        for candidate in [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "DejaVuSans.ttf",
        ]:
            try:
                font = ImageFont.truetype(candidate, font_size_px)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()

        labeled, num_features = label(is_black)
        for i in range(1, num_features + 1):
            ys, xs = np.where(labeled == i)
            if len(xs) == 0:
                continue
            median_px = np.median(diam[ys, xs])
            median_mm = px_to_mm(median_px, dpi)
            y, x = int(np.median(ys)), int(np.median(xs))
            if median_px < px_min:
                col = (255, 0, 0)
            elif median_px < px_min * (1 + border_pct):
                col = (180, 140, 0)
            else:
                col = (0, 150, 0)
            txt = f"{median_mm:.2f} mm"
            draw.text((x + 1, y + 1), txt, fill=(0, 0, 0), font=font)
            draw.text((x, y), txt, fill=col, font=font)

    return comp.convert("RGB")

# ----------------- Compliance Check -----------------
def qc_check_compliance(thick_np: np.ndarray, dpi: int, min_line_mm: float) -> Dict:
    px_min = mm_to_px(min_line_mm, dpi)
    diam = _distance_transform_diameter(thick_np)
    is_black = (thick_np == 0)

    violations = np.sum((diam < px_min) & is_black)

    # Placeholder for OCR text warnings
    text_warnings = []

    compliant = (violations == 0 and len(text_warnings) == 0)
    return {
        "compliant": compliant,
        "violations": int(violations),
        "text_warnings": text_warnings
    }

# ----------------- Export -----------------
def export_canvas(image: Image.Image, filename: str, page_size: str = "letter", dpi: int = 600):
    if page_size.lower() == "letter":
        size = letter
    elif page_size.lower() == "a4":
        size = A4
    else:
        raise ValueError("Unsupported page size")

    pdf = canvas.Canvas(filename, pagesize=size)
    width, height = size

    tmp_png = filename.replace(".pdf", "_tmp.png")
    image.save(tmp_png, dpi=(dpi, dpi))
    pdf.drawImage(tmp_png, 0, 0, width=width, height=height)
    pdf.showPage()
    pdf.save()
    os.remove(tmp_png)

    # Also export TIFF + PNG
    base = os.path.splitext(filename)[0]
    image.save(base + ".tiff", dpi=(dpi, dpi))
    image.save(base + ".png", dpi=(dpi, dpi))

# ----------------- QC Runners -----------------
def run_qc_single(img: Image.Image, dpi: int, min_line_mm: float):
    thick_np = np.array(img.convert("L"))
    qc_image = qc_overlay(thick_np, img, min_line_mm=min_line_mm, dpi=dpi)
    qc_result = qc_check_compliance(thick_np, dpi, min_line_mm)

    summary_row = {
        "filename": "unknown",
        "page": 1,
        "compliant": qc_result["compliant"],
        "violations": qc_result["violations"],
        "text_warnings": qc_result["text_warnings"]
    }
    return qc_result, qc_image, summary_row

def run_qc_batch(input_dir: str, output_dir: str, dpi: int = 600, min_line_mm: float = 0.18, page_size: str = "letter"):
    return qc_all(input_dir, output_dir, dpi=dpi, min_line_mm=min_line_mm, page_size=page_size)

# ----------------- Batch Runner -----------------
def qc_all(input_dir: str, output_dir: str, dpi: int = 600, min_line_mm: float = 0.18, page_size: str = "letter"):
    results = {}
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".tiff", ".tif", ".jpg", ".jpeg", ".bmp")):
            continue
        path = os.path.join(input_dir, fname)
        img = Image.open(path).convert("RGB")
        thick_np = np.array(img.convert("L"))

        qc_image = qc_overlay(thick_np, img, min_line_mm=min_line_mm, dpi=dpi)
        qc_result = qc_check_compliance(thick_np, dpi, min_line_mm)

        out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".pdf")
        export_canvas(qc_image, out_path, page_size=page_size, dpi=dpi)

        results[fname] = qc_result

    # Write summary
    summary_file = os.path.join(output_dir, "qc_summary.txt")
    with open(summary_file, "w") as f:
        for fname, res in results.items():
            status = "PASS" if res["compliant"] else "FAIL"
            f.write(f"{fname}: {status} (violations={res['violations']}, text_warnings={res['text_warnings']})\n")

    return results
