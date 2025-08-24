import streamlit as st
from io import BytesIO
from PIL import Image, ImageOps
from pdf2image import convert_from_bytes
import numpy as np
import pytesseract
import re
import cv2
from scipy.ndimage import distance_transform_edt
from patent_qc import run_qc_single, run_qc_batch

# import QC tools (from qc_tools.py in the same folder)
from qc_tools import (
    qc_overlay,
    qc_report,
    qc_text_check,
    qc_fill_check,
    build_combined_report,
    dumps_csv,
    dumps_json,
    px_to_mm,
    mm_to_px,
)

# ------------------ Streamlit page config ------------------
st.set_page_config(page_title="Patent Figure Converter v5", layout="wide")
st.title("Patent Figure Converter (v5)")

# ------------------ Office presets ------------------
OFFICE_PRESETS = {
    "USPTO": {
        "page_sizes": {"Letter": (216, 279), "A4": (210, 297)},
        "default_size": "Letter",
        "margins_mm": {"top": 25, "left": 25, "right": 15, "bottom": 15},
        "allow_color": False,
        "min_line_mm": 0.2,
    },
    "WIPO": {
        "page_sizes": {"A4": (210, 297)},
        "default_size": "A4",
        "margins_mm": {"top": 20, "left": 25, "right": 20, "bottom": 20},
        "allow_color": True,
        "min_line_mm": 0.2,
    },
    "EPO": {
        "page_sizes": {"A4": (210, 297)},
        "default_size": "A4",
        "margins_mm": {"top": 20, "left": 25, "right": 15, "bottom": 15},
        "allow_color": True,
        "min_line_mm": 0.2,
    },
}

# ------------------ Controls ------------------
office = st.selectbox("Select patent office:", list(OFFICE_PRESETS.keys()))
preset = OFFICE_PRESETS[office]

if len(preset["page_sizes"]) > 1:
    page_size_name = st.selectbox(
        "Target page size",
        list(preset["page_sizes"].keys()),
        index=list(preset["page_sizes"].keys()).index(preset["default_size"])
    )
else:
    page_size_name = preset["default_size"]

dpi = st.slider("Rasterization DPI (higher = better OCR & edges, slower)", 200, 600, 300, step=50)

if office == "USPTO":
    bw_mode = True
    st.checkbox("Convert to Black & White (USPTO requires B&W)", value=True, disabled=True)
else:
    bw_mode = st.checkbox("Convert to Black & White", value=False)

force_bw_we = st.checkbox("Force B&W for WIPO/EPO (optional)", value=False)
if office in ("WIPO", "EPO") and force_bw_we:
    bw_mode = True

debug_line_overlay = st.checkbox("Show line-width verification overlay", value=False)
qc_enable = st.checkbox("Enable QC (v6): line-width & text checks", value=False)
qc_heatmap = st.checkbox("Show QC as heatmap (v6)", value=False, disabled=not qc_enable)
qc_labels  = st.checkbox("Show line thickness labels (v6)", value=True, disabled=not qc_enable)
# Optional: minimum text height rule (mm). Many offices are ~3.2 mm for legibility.
min_text_mm = st.number_input("Minimum text height (mm) for QC", min_value=1.0, max_value=6.0, value=3.2, step=0.1)
qc_font_pt = st.slider(
    "QC label font size (pt)", 6, 24, 12, 
    help="Font size for measurement labels in the QC overlay"
)

run_cross_check = st.checkbox("Run cross-figure reference consistency check (OCR)", value=True)

uploaded_files = st.file_uploader(
    "Upload figures (PDF, PNG, JPG, JPEG supported)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True
)

# ------------------ Helpers ------------------
def mm_to_px(mm: float, dpi: int) -> int:
    return int(round((mm / 25.4) * dpi))

def target_canvas(office: str, page_size_name: str, dpi: int):
    w_mm, h_mm = OFFICE_PRESETS[office]["page_sizes"][page_size_name]
    margins = OFFICE_PRESETS[office]["margins_mm"]
    W = mm_to_px(w_mm, dpi)
    H = mm_to_px(h_mm, dpi)
    box = (
        mm_to_px(margins["left"], dpi),
        mm_to_px(margins["top"], dpi),
        W - mm_to_px(margins["right"], dpi),
        H - mm_to_px(margins["bottom"], dpi),
    )
    return (W, H), box

def fit_into_box(img: Image.Image, box):
    left, top, right, bottom = box
    box_w = right - left
    box_h = bottom - top
    im_w, im_h = img.size
    scale = min(box_w / im_w, box_h / im_h)
    new_w, new_h = max(1, int(im_w * scale)), max(1, int(im_h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)

def normalize_canvas(img: Image.Image, office: str, page_size_name: str, dpi: int) -> Image.Image:
    (W, H), inner_box = target_canvas(office, page_size_name, dpi)
    canvas = Image.new("RGB", (W, H), color="white")
    if img.mode not in ("RGB", "L", "1"):
        img = img.convert("RGB")
    fitted = fit_into_box(img, inner_box)
    left, top, right, bottom = inner_box
    off_x = left + ((right - left) - fitted.size[0]) // 2
    off_y = top + ((bottom - top) - fitted.size[1]) // 2
    canvas.paste(fitted, (off_x, off_y))
    return canvas

def binarize_true_bw(img: Image.Image) -> Image.Image:
    if img.mode != "L":
        img = ImageOps.grayscale(img)
    np_img = np.array(img)
    _, bw_np = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(bw_np)

def thicken_lines(img_bw: Image.Image, office: str, dpi: int) -> Image.Image:
    bw_np = np.array(img_bw)
    inv = (bw_np == 0).astype(np.uint8)  # black=1, white=0

    min_mm = OFFICE_PRESETS[office]["min_line_mm"]
    px_min = max(1, int(round((min_mm / 25.4) * dpi)))

    # --- Step 1: Measure local thickness with distance transform ---
    dist = distance_transform_edt(inv)        # distance inside black areas
    local_width = dist * 2                    # full thickness estimate

    # --- Step 2: Mask of too-thin regions ---
    violating_mask = local_width < px_min

    # --- Step 3: Dilate only thin regions ---
    kernel_size = max(3, px_min if px_min % 2 == 1 else px_min + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(inv, kernel, iterations=1)

    # Keep dilated pixels only where violating, else keep original
    corrected = np.where(violating_mask, dilated, inv)

    # --- Step 4: Convert back to white=255/black=0 ---
    thick_img = (255 - corrected * 255).astype(np.uint8)
    return Image.fromarray(thick_img)

REF_NUM_RE = re.compile(r"\b[0-9]{1,4}\b")

def extract_reference_numbers(img: Image.Image) -> set:
    g = ImageOps.autocontrast(ImageOps.grayscale(img))
    try:
        text = pytesseract.image_to_string(g, config="--psm 6")
    except Exception:
        text = ""
    return set(REF_NUM_RE.findall(text))

def images_from_upload(uploaded_file, dpi=300):
    name = uploaded_file.name
    ext = name.lower().split(".")[-1]
    originals = []
    if ext == "pdf":
        pages = convert_from_bytes(uploaded_file.read(), dpi=dpi)
        originals.extend(pages)
    elif ext in ("png", "jpg", "jpeg"):
        originals.append(Image.open(uploaded_file).convert("RGB"))
    else:
        st.warning(f"Unsupported file type: {name}")
    return originals

def process_one(img: Image.Image, office: str, page_size_name: str, dpi: int,
                bw_mode: bool, debug: bool = False) -> Image.Image:
    # 1) normalize to target page & margins
    base = normalize_canvas(img, office, page_size_name, dpi)

    # 2) If color not allowed (USPTO) or user requested B&W => binarize & thicken
    if (not OFFICE_PRESETS[office]["allow_color"]) or bw_mode:
        bw = binarize_true_bw(base)
        thick = thicken_lines(bw, office, dpi)

        # Prepare RGB output (white background, black lines)
        out_rgb = Image.new("RGB", thick.size, "white")
        thick_np = np.array(thick)
        mask = thick_np == 0  # black pixels
        out_np = np.array(out_rgb)
        out_np[mask] = [0, 0, 0]
        out_rgb = Image.fromarray(out_np)

        # 3) Optional debug overlay: highlight lines below minimum width
        if debug:
            min_mm = OFFICE_PRESETS[office]["min_line_mm"]
            px_min = max(1, int(round((min_mm / 25.4) * dpi)))
            dist = distance_transform_edt(thick_np == 0)  # black = line
            overlay = np.zeros((*thick_np.shape, 4), dtype=np.uint8)  # RGBA
            red_mask = dist < px_min / 2
            overlay[red_mask] = [255, 0, 0, 128]  # semi-transparent red
            overlay_img = Image.fromarray(overlay, mode="RGBA")
            out_rgb = out_rgb.convert("RGBA")
            out_rgb = Image.alpha_composite(out_rgb, overlay_img)
            out_rgb = out_rgb.convert("RGB")  # back to RGB for Streamlit

        return out_rgb

    else:
        # Color allowed: just return normalized base canvas
        return base

def save_png_bytes(img: Image.Image, name_hint: str) -> BytesIO:
    bio = BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio

# ------------------ Main flow ------------------
if uploaded_files:
    all_originals = []
    all_processed = []
    file_pages = {}
    for uf in uploaded_files:
        pages = images_from_upload(uf, dpi=dpi)
        file_pages[uf.name] = pages
        for i, p in enumerate(pages):
            all_originals.append((uf.name, i+1, p))

    progress = st.progress(0, text="Processing figures...")
    total = sum(len(pgs) for pgs in file_pages.values())
    done = 0
    qc_rows_all_pages = []

    for fname, pages in file_pages.items():
        st.subheader(fname)
        for i, orig in enumerate(pages, start=1):
            processed = process_one(orig, office, page_size_name, dpi, bw_mode, debug_line_overlay)
            all_processed.append((fname, i, processed))

            # Display
            c1, c2 = st.columns(2)
            with c1:
                st.caption(f"Original — {fname} (page {i})")
                st.image(orig, use_container_width=True)
            with c2:
                st.caption(f"Processed — {fname} (page {i})")
                st.image(processed, use_container_width=True)

            # Download
            dl = save_png_bytes(processed, f"{fname}_p{i}.png")
            st.download_button(
                f"Download processed page {i} as PNG",
                data=dl,
                file_name=f"processed_{fname.replace(' ', '_')}_p{i}.png",
                mime="image/png",
                use_container_width=True,
            )

            # --- QC per page ---
            if qc_enable:
                qc_result, qc_image, summary_row = run_qc_single(
                    processed,
                    dpi=dpi,
                    min_line_mm=OFFICE_PRESETS[office]["min_line_mm"]
                )
                qc_rows_all_pages.append(summary_row)
                st.image(qc_image, caption=f"QC Overlay — page {i}", use_container_width=True)
                if qc_result["compliant"]:
                    st.success("✅ Figure passed all compliance checks!")
                else:
                    st.error("❌ Figure failed compliance checks.")
                    st.write("Details:", {
                    "violations": qc_result["violations"],
                    "text_warnings": qc_result["text_warnings"]
                    })

            done += 1
            progress.progress(done / total, text=f"Processed {done}/{total}")

if qc_enable and qc_rows_all_pages:
    st.subheader("QC Report (all pages) – v6")
    csv_bytes = dumps_csv(qc_rows_all_pages)
    json_bytes = dumps_json(qc_rows_all_pages)

    st.download_button(
        "Download QC Report (CSV)",
        data=csv_bytes,
        file_name="qc_report_all_pages_v6.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.download_button(
        "Download QC Report (JSON)",
        data=json_bytes,
        file_name="qc_report_all_pages_v6.json",
        mime="application/json",
        use_container_width=True
    )

    # ------------------ OCR cross-check ------------------
    if run_cross_check and all_originals:
        st.subheader("Reference consistency check (OCR across all uploaded figures)")
        refs_per_page = []
        all_refs = set()
        for fname, pidx, img in all_originals:
            refs = extract_reference_numbers(img)
            refs_per_page.append(((fname, pidx), refs))
            all_refs |= refs
        missing_summary = {}
        pages_list = [(f, i) for (f, i, _) in all_originals]
        for ref in sorted(all_refs, key=lambda x: (len(x), x)):
            missing = []
            for (fname, pidx) in pages_list:
                page_refs = next((r for ((fn, pi), r) in refs_per_page if fn==fname and pi==pidx), set())
                if ref not in page_refs:
                    missing.append((fname, pidx))
            if 0 < len(missing) < len(pages_list):
                missing_summary[ref] = missing
        if missing_summary:
            st.error("Reference signs appear inconsistent across pages (some present on certain pages but missing on others).")
            for ref, miss in missing_summary.items():
                miss_str = ", ".join([f"{fn} p.{pi}" for (fn, pi) in miss])
                st.markdown(f"- **Ref {ref}** missing on: {miss_str}")
        else:
            st.success("No cross-page inconsistencies detected in numeric references (OCR-based).")

    if office == "USPTO" and not bw_mode:
        st.error("USPTO drawings must be in **black and white**. The app forces B&W for USPTO.")
    if bw_mode or office=="USPTO":
        st.info("Monochrome binarization enforced (true B/W). Anti-aliased grey pixels are removed.")

else:
    st.info("Upload PDF or image files to begin.")
