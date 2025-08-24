# qc_tools.py
import io, json, csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract

# --- robust distance transform: SciPy (preferred) -> OpenCV fallback
try:
    from scipy.ndimage import distance_transform_edt as _edt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    import cv2

def _distance_transform_diameter(thick_np: np.ndarray) -> np.ndarray:
    """
    Input: thick_np with 0=black (line), 255=white (background)
    Output: approximate local line *diameter* (pixels) at each pixel location.
    """
    if _HAS_SCIPY:
        # distance to nearest background from each black pixel; multiply by 2 -> diameter
        return 2.0 * _edt(thick_np == 0)
    else:
        # OpenCV distanceTransform requires non-zero=foreground, zeros=background
        # So invert: foreground=lines=1, background=0
        fg = (thick_np == 0).astype(np.uint8) * 255
        dist = cv2.distanceTransform(fg, distanceType=cv2.DIST_L2, maskSize=5)
        return 2.0 * dist

def mm_to_px(mm: float, dpi: int) -> int:
    return max(1, int(round((mm / 25.4) * dpi)))

def px_to_mm(px: float, dpi: int) -> float:
    return (px / dpi) * 25.4

from skimage.morphology import skeletonize
from scipy.ndimage import label as ndi_label, find_objects

def qc_overlay(
    thick_np: np.ndarray,
    out_rgb: Image.Image,
    *,
    min_line_mm: float,
    dpi: int,
    add_labels: bool = True,
    heatmap: bool = False,
    border_pct: float = 0.10,
    font_size_pt: int = 12,
    legend_pos: str = "top-left",  # options: "top-left", "top-right", "bottom-left", "bottom-right"
) -> Image.Image:
    """
    Build a QC overlay (traffic-light or heatmap) on top of the processed RGB image.
    Legend is semi-transparent and placed at legend_pos.
    """
    from scipy.ndimage import label

    px_min = mm_to_px(min_line_mm, dpi)
    diam = _distance_transform_diameter(thick_np)  # pixels
    H, W = thick_np.shape
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    is_black = (thick_np == 0)

    # Traffic-light masks
    violating_mask = (diam < px_min) & is_black
    borderline_mask = (diam >= px_min) & (diam < px_min * (1 + border_pct)) & is_black
    compliant_mask = (diam >= px_min * (1 + border_pct)) & is_black

    if heatmap:
        norm = np.clip(diam / (2 * px_min), 0, 1)
        overlay[..., 0] = np.where(is_black, (1 - norm) * 255, 0)
        overlay[..., 1] = np.where(is_black, norm * 255, 0)
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
            mm = px_to_mm(median_px, dpi)
            for y, x in zip(ys[::max(1, len(ys)//3)], xs[::max(1, len(xs)//3)]):
                if heatmap:
                    col = (0, 0, 0)
                else:
                    if median_px < px_min:
                        col = (255, 0, 0)
                    elif median_px < px_min * (1 + border_pct):
                        col = (180, 140, 0)
                    else:
                        col = (0, 150, 0)
                txt = f"{mm:.2f} mm"
                draw.text((x + 1, y + 1), txt, fill=(0, 0, 0), font=font)
                draw.text((x, y), txt, fill=col, font=font)

        # --- Semi-transparent legend ---
        legend_padding = int(font_size_px * 0.5)
        box_w, box_h = font_size_px * 6, font_size_px + 4
        if legend_pos == "top-left":
            lx, ly = legend_padding, legend_padding
        elif legend_pos == "top-right":
            lx, ly = W - box_w - legend_padding, legend_padding
        elif legend_pos == "bottom-left":
            lx, ly = legend_padding, H - 4*box_h - 4*legend_padding
        elif legend_pos == "bottom-right":
            lx, ly = W - box_w - legend_padding, H - 4*box_h - 4*legend_padding
        else:
            lx, ly = legend_padding, legend_padding  # default top-left

        # Transparent background
        legend_bg = Image.new("RGBA", (box_w, box_h*4), (255,255,255,150))
        comp.paste(legend_bg, (lx, ly), legend_bg)
        legend_draw = ImageDraw.Draw(comp)

        if heatmap:
            for i in range(6):
                fraction = i / 5
                color = (
                    int((1 - fraction) * 255),
                    int(fraction * 255),
                    0
                )
                legend_draw.rectangle(
                    [lx + i*(box_w//6), ly, lx + (i+1)*(box_w//6), ly + box_h],
                    fill=color
                )
            legend_draw.text((lx, ly + box_h + 2), "Heatmap: thin→thick", fill=(0,0,0), font=font)
        else:
            legend_draw.rectangle([lx, ly, lx+box_h, ly+box_h], fill=(255, 0, 0))
            legend_draw.text((lx + box_h + 2, ly), "Fail", fill=(0,0,0), font=font)
            legend_draw.rectangle([lx, ly + box_h + 2, lx+box_h, ly + 2*box_h + 2], fill=(255, 255, 0))
            legend_draw.text((lx + box_h + 2, ly + box_h + 2), "Borderline", fill=(0,0,0), font=font)
            legend_draw.rectangle([lx, ly + 2*(box_h+2), lx+box_h, ly + 3*box_h+4], fill=(0, 255, 0))
            legend_draw.text((lx + box_h + 2, ly + 2*(box_h+2)), "Pass", fill=(0,0,0), font=font)

    return comp.convert("RGB")


def qc_report(thick_np: np.ndarray, *, min_line_mm: float, dpi: int, border_pct: float = 0.10) -> dict:
    """
    Compute QC statistics (pass/borderline/fail; distribution in mm).
    """
    px_min = mm_to_px(min_line_mm, dpi)
    diam = _distance_transform_diameter(thick_np)
    is_black = (thick_np == 0)
    black_vals = diam[is_black]
    total = int(black_vals.size)

    if total == 0:
        return {
            "total_black_pixels": 0,
            "pass": 0, "borderline": 0, "fail": 0,
            "pass_%": 0.0, "borderline_%": 0.0, "fail_%": 0.0,
            "thickness_mm_stats": {}
        }

    fail = int(np.sum(black_vals < px_min))
    borderline = int(np.sum((black_vals >= px_min) & (black_vals < px_min * (1 + border_pct))))
    passed = total - fail - borderline

    # thickness stats in mm
    mm_vals = px_to_mm(black_vals, dpi)
    stats = {
        "min_mm": float(np.min(mm_vals)),
        "p05_mm": float(np.percentile(mm_vals, 5)),
        "median_mm": float(np.median(mm_vals)),
        "p95_mm": float(np.percentile(mm_vals, 95)),
        "max_mm": float(np.max(mm_vals)),
    }

    return {
        "total_black_pixels": total,
        "pass": passed, "borderline": borderline, "fail": fail,
        "pass_%": round(passed / total * 100, 2),
        "borderline_%": round(borderline / total * 100, 2),
        "fail_%": round(fail / total * 100, 2),
        "thickness_mm_stats": stats
    }

def qc_text_check(pil_img: Image.Image, *, min_text_mm: float, dpi: int) -> list:
    """
    OCR-based text size check. Returns list of dicts per detected word.
    Size is estimated from OCR bbox height -> px -> mm.
    """
    try:
        data = pytesseract.image_to_data(pil_img, output_type="dict")
    except Exception:
        return []

    rows = []
    n = len(data.get("text", []))
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        h_px = int(data["height"][i] or 0)
        mm = px_to_mm(h_px, dpi)
        compliant = mm >= min_text_mm
        rows.append({"text": txt, "height_mm": round(mm, 2), "compliant": bool(compliant)})
    return rows

def qc_fill_check(thick_np: np.ndarray, *, large_blob_px: int = 0) -> dict:
    """
    Heuristic: report overall black fill ratio. Optionally mark warning if > 20%.
    large_blob_px kept for future extension (connected-components).
    """
    is_black = (thick_np == 0)
    ratio = float(np.mean(is_black))  # fraction of black pixels
    return {
        "black_fill_ratio": round(ratio, 4),
        "warning_solid_fill": bool(ratio > 0.20)
    }

def build_combined_report(
    office: str, dpi: int, page_label: str,
    line_report: dict, text_report: list, fill_report: dict,
    min_line_mm: float, min_text_mm: float
) -> dict:
    """
    Bundle everything for a single page into a flat dict for CSV/JSON.
    """
    return {
        "page": page_label,
        "office": office,
        "dpi": dpi,
        "min_line_mm": min_line_mm,
        "min_text_mm": min_text_mm,
        # line stats:
        "black_pixels": line_report.get("total_black_pixels", 0),
        "pass": line_report.get("pass", 0),
        "borderline": line_report.get("borderline", 0),
        "fail": line_report.get("fail", 0),
        "pass_%": line_report.get("pass_%", 0.0),
        "borderline_%": line_report.get("borderline_%", 0.0),
        "fail_%": line_report.get("fail_%", 0.0),
        "thickness_min_mm": line_report.get("thickness_mm_stats", {}).get("min_mm", 0.0),
        "thickness_p05_mm": line_report.get("thickness_mm_stats", {}).get("p05_mm", 0.0),
        "thickness_median_mm": line_report.get("thickness_mm_stats", {}).get("median_mm", 0.0),
        "thickness_p95_mm": line_report.get("thickness_mm_stats", {}).get("p95_mm", 0.0),
        "thickness_max_mm": line_report.get("thickness_mm_stats", {}).get("max_mm", 0.0),
        # simple aggregations for text:
        "text_items": len(text_report),
        "text_noncompliant": int(sum(1 for r in text_report if not r.get("compliant", True))),
        # fill:
        "black_fill_ratio": fill_report.get("black_fill_ratio", 0.0),
        "fill_warning": fill_report.get("warning_solid_fill", False),
    }

def dumps_json(report_rows: list) -> bytes:
    return json.dumps(report_rows, indent=2).encode("utf-8")

def dumps_csv(report_rows: list) -> bytes:
    if not report_rows:
        return b""
    fields = list(report_rows[0].keys())
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=fields)
    w.writeheader()
    for r in report_rows:
        w.writerow(r)
    return sio.getvalue().encode("utf-8")
