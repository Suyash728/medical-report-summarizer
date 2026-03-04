"""
PDF Text Extractor - Streamlit App
Supports both regular PDFs (text-based) and scanned PDFs (via OCR).

Dependencies:
    pip install streamlit pymupdf pytesseract pillow pdf2image

System requirements (for OCR):
    - Tesseract OCR: https://github.com/tesseract-ocr/tesseract
      Linux:   sudo apt install tesseract-ocr
      macOS:   brew install tesseract
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
    - Poppler (for pdf2image):
      Linux:   sudo apt install poppler-utils
      macOS:   brew install poppler
      Windows: https://github.com/oschwartz10612/poppler-windows
"""

import io
import zipfile

import fitz  # PyMuPDF — fast native PDF text extraction
import pytesseract
import streamlit as st
from pdf2image import convert_from_bytes  # converts PDF pages → PIL Images for OCR
from PIL import Image

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def extract_text_native(pdf_bytes: bytes) -> dict[int, str]:
    """
    Extract selectable text from a PDF using PyMuPDF (no OCR).
    Returns a dict mapping {page_number (1-indexed): text}.
    Fast and accurate for text-based PDFs.
    """
    results: dict[int, str] = {}
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            results[page_num] = page.get_text("text")  # plain-text extraction
    return results


def extract_text_ocr(
    pdf_bytes: bytes,
    lang: str = "eng",
    dpi: int = 300,
) -> dict[int, str]:
    """
    Convert each PDF page to a high-resolution image, then run Tesseract OCR.
    Returns a dict mapping {page_number (1-indexed): text}.
    Required for scanned / image-only PDFs.

    Args:
        pdf_bytes: Raw PDF file content.
        lang:      Tesseract language code(s), e.g. 'eng', 'eng+fra'.
        dpi:       Render resolution; 300 dpi is a good balance of speed & quality.
    """
    results: dict[int, str] = {}
    # Render all pages to PIL Images
    images: list[Image.Image] = convert_from_bytes(pdf_bytes, dpi=dpi)
    for page_num, img in enumerate(images, start=1):
        # pytesseract returns a UTF-8 string
        results[page_num] = pytesseract.image_to_string(img, lang=lang)
    return results


def is_scanned_pdf(pdf_bytes: bytes, text_threshold: int = 50) -> bool:
    """
    Heuristic: if fewer than `text_threshold` characters are extracted
    natively across the whole document, treat it as a scanned PDF.
    """
    pages = extract_text_native(pdf_bytes)
    total_chars = sum(len(t.strip()) for t in pages.values())
    return total_chars < text_threshold


def build_zip(pages: dict[int, str]) -> bytes:
    """
    Pack every page's text into a ZIP archive (one .txt file per page).
    Returns raw ZIP bytes ready for st.download_button.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for page_num, text in pages.items():
            zf.writestr(f"page_{page_num:04d}.txt", text)
    return buf.getvalue()


# ──────────────────────────────────────────────
# Page config & styling
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="PDF Text Extractor",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF Text Extractor")
st.caption("Extracts text from regular **and** scanned (image-only) PDFs via OCR.")

# ──────────────────────────────────────────────
# Sidebar — settings
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    mode = st.radio(
        "Extraction mode",
        options=["Auto-detect", "Native (text PDF)", "OCR (scanned PDF)"],
        help=(
            "Auto-detect checks whether the PDF has selectable text and "
            "falls back to OCR if it does not."
        ),
    )

    ocr_lang = st.text_input(
        "Tesseract language code",
        value="eng",
        help="Use '+' to combine languages, e.g. 'eng+fra'. "
             "Run `tesseract --list-langs` to see installed languages.",
    )

    ocr_dpi = st.slider(
        "OCR render DPI",
        min_value=100,
        max_value=600,
        value=300,
        step=50,
        help="Higher DPI = better OCR accuracy but slower processing.",
    )

    st.divider()
    st.markdown(
        "**System requirements for OCR**\n"
        "- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)\n"
        "- [Poppler](https://poppler.freedesktop.org/) (`pdf2image` dependency)"
    )

# ──────────────────────────────────────────────
# Main — file upload
# ──────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a PDF file",
    type=["pdf"],
    help="Drag & drop or browse. Max 200 MB (Streamlit default).",
)

if uploaded_file is None:
    st.info("👆 Upload a PDF to get started.")
    st.stop()  # halt execution until a file is provided

# Read bytes once; reuse everywhere
pdf_bytes: bytes = uploaded_file.read()

st.success(f"Loaded **{uploaded_file.name}** ({len(pdf_bytes) / 1_024:.1f} KB)")

# ──────────────────────────────────────────────
# Extraction
# ──────────────────────────────────────────────

if st.button("🔍 Extract Text", type="primary"):

    with st.spinner("Analysing PDF…"):

        # Determine which method to use
        if mode == "Native (text PDF)":
            use_ocr = False
            method_label = "Native (PyMuPDF)"
        elif mode == "OCR (scanned PDF)":
            use_ocr = True
            method_label = f"OCR — Tesseract ({ocr_lang}, {ocr_dpi} dpi)"
        else:  # Auto-detect
            scanned = is_scanned_pdf(pdf_bytes)
            use_ocr = scanned
            method_label = (
                f"Auto → OCR — Tesseract ({ocr_lang}, {ocr_dpi} dpi)"
                if scanned
                else "Auto → Native (PyMuPDF)"
            )

    st.info(f"**Method:** {method_label}")

    # Run extraction with a progress bar
    progress = st.progress(0, text="Starting…")

    try:
        if use_ocr:
            with st.spinner("Running OCR (this may take a moment for large files)…"):
                pages = extract_text_ocr(pdf_bytes, lang=ocr_lang, dpi=ocr_dpi)
        else:
            pages = extract_text_native(pdf_bytes)
    except Exception as exc:
        st.error(f"Extraction failed: {exc}")
        st.stop()

    progress.progress(100, text="Done!")

    # ──────────────────────────────────────────
    # Results
    # ──────────────────────────────────────────

    total_pages = len(pages)
    total_chars = sum(len(t) for t in pages.values())
    total_words = sum(len(t.split()) for t in pages.values())

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Pages", total_pages)
    col2.metric("Words", f"{total_words:,}")
    col3.metric("Characters", f"{total_chars:,}")

    st.divider()

    # Per-page display in expandable sections
    st.subheader("📑 Extracted Text")

    for page_num, text in pages.items():
        with st.expander(
            f"Page {page_num} — {len(text.split()):,} words",
            expanded=(page_num == 1),  # open first page by default
        ):
            if text.strip():
                st.text_area(
                    label="",           # hide label; expander title is enough
                    value=text,
                    height=300,
                    key=f"page_{page_num}",
                )
            else:
                st.warning("No text found on this page.")

    # ──────────────────────────────────────────
    # Download options
    # ──────────────────────────────────────────

    st.divider()
    st.subheader("⬇️ Download")

    dl_col1, dl_col2 = st.columns(2)

    # Single combined .txt file
    combined_text = "\n\n".join(
        f"--- Page {p} ---\n{t}" for p, t in pages.items()
    )
    dl_col1.download_button(
        label="Download all pages as .txt",
        data=combined_text.encode("utf-8"),
        file_name=f"{uploaded_file.name.removesuffix('.pdf')}_extracted.txt",
        mime="text/plain",
    )

    # ZIP of individual page files
    dl_col2.download_button(
        label="Download pages as .zip (one file per page)",
        data=build_zip(pages),
        file_name=f"{uploaded_file.name.removesuffix('.pdf')}_pages.zip",
        mime="application/zip",
    )