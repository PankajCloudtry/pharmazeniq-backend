import os, io, re, base64, json
from typing import Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from rapidfuzz import process, fuzz
from pdf2image import convert_from_bytes, exceptions as pdf2image_exc
import fitz  # PyMuPDF
from google.oauth2 import service_account
from google.cloud import vision_v1

# ─── 0️⃣ CREDENTIALS ─────────────────────────────────────────────────────────
raw = st.secrets.get("GOOGLE_CREDENTIALS") or st.secrets.get("service_account_json")
if raw is None:
    st.error("❌ Google Vision credentials missing in Secrets.")
    st.stop()
info = json.loads(raw) if isinstance(raw, str) else dict(raw)
creds = service_account.Credentials.from_service_account_info(info)
client = vision_v1.ImageAnnotatorClient(credentials=creds)

# ─── 1️⃣ PAGE CONFIG & THEME ─────────────────────────────────────────────────
st.set_page_config(page_title="Pharmazeniq", page_icon="💊", layout="wide")

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
      html, body, div[class^="st"]  { font-family: 'Inter', sans-serif; }
      .block-container { padding: 0rem 1rem; }

      .card{border:1px solid #eee;border-radius:.8rem;padding:.8rem;
            box-shadow:0 1px 4px rgba(0,0,0,.06);transition:.15s}
      .card:hover{transform:translateY(-4px);box-shadow:0 6px 16px rgba(0,0,0,.12)}
      .card img{width:100%;border-radius:.6rem}
      .card-title{font-size:.95rem;font-weight:600;margin:.4rem 0 .2rem}
      .card-price{font-size:.9rem;color:#e91e63;font-weight:600}
      .stock-badge{font-size:.75rem;color:#555;background:#f5f5f5;
                   padding:0 .4rem;border-radius:.4rem}

      .grid{display:grid;gap:1rem;padding:0 .5rem;overflow-x:hidden;}
      @media(max-width:600px){.grid{grid-template-columns:repeat(2,1fr)}}
      @media(min-width:601px) and (max-width:992px){.grid{grid-template-columns:repeat(3,1fr)}}
      @media(min-width:993px){.grid{grid-template-columns:repeat(3,1fr)}}
    </style>
    """,
    unsafe_allow_html=True,
)

def product_card(img_url: str, name: str, total: float, stock_note: str) -> str:
    badge = f'<span class="stock-badge">{stock_note}</span>' if stock_note else ""
    return f"""
    <div class="card">
      <img src="{img_url}" loading="lazy">
      <div class="card-title">{name}</div>
      <div class="card-price">₹{total:.2f}</div>
      {badge}
    </div>
    """

if os.path.exists("assets/header_banner.png"):
    st.image("assets/header_banner.png", use_container_width=True)

# ─── 2️⃣ SIDEBAR ─────────────────────────────────────────────────────────────
with st.sidebar:
    if os.path.exists("animation.gif"):
        b64 = base64.b64encode(open("animation.gif", "rb").read()).decode()
        st.markdown(
            f'<img src="data:image/gif;base64,{b64}" style="width:100%;margin-bottom:1rem;">',
            unsafe_allow_html=True,
        )
    st.markdown("## Filters")
    sort_by = st.radio("Sort by", ["Price (Low→High)", "Fastest ETA"])
    st.markdown("---")
    st.markdown("Need help? 📧 support@pharmazeniq.com")

# ─── 3️⃣ DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv("data/medicines.csv"), pd.read_csv("data/vendor_prices.csv")

meds_df, vendor_df = load_data()
name_to_id = dict(zip(meds_df.name, meds_df.id))

# ─── 4️⃣ OCR ─────────────────────────────────────────────────────────────────
def deskew_and_encode(img: Image.Image) -> bytes:
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(th > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle += 90
    h, w = th.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    deskew = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    _, buf = cv2.imencode(".jpg", deskew)
    return buf.tobytes()

def ocr_bytes(b: bytes) -> str:
    return client.document_text_detection(image=vision_v1.Image(content=b)).full_text_annotation.text

def pdf_to_images(data: bytes, dpi=300) -> List[Image.Image]:
    return [Image.open(io.BytesIO(p.get_pixmap(dpi=dpi).tobytes())) for p in fitz.open(stream=data, filetype="pdf")]

def extract_text(uploaded) -> str:
    raw = uploaded.read()
    pages: List[Image.Image] = []
    if uploaded.type == "application/pdf":
        try:
            pages = convert_from_bytes(raw, dpi=300)
        except Exception:
            pages = pdf_to_images(raw)
    if not pages:
        try: pages = [Image.open(io.BytesIO(raw))]
        except Exception:
            st.error("❌ Not a valid image/PDF.")
            return ""
    return "\n".join(ocr_bytes(deskew_and_encode(p)) for p in pages)

# ─── 5️⃣ MATCHING ────────────────────────────────────────────────────────────
def normalize(line: str) -> str:
    x = re.sub(r"^[\s\-\•\d\.]+", "", line)
    x = re.sub(r"\b\d+(\.\d+)?\s?(mg|g|ml)\b", "", x, flags=re.IGNORECASE)
    x = re.sub(r"\b(tab|tablet|cap|capsule|caps|syp|syr|syrup|inj)\b", "", x, flags=re.IGNORECASE)
    x = re.sub(r"[^A-Za-z0-9 ]+", "", x)
    return x.lower().strip()

def fuzzy_opts(key: str) -> list[str]:
    matches = process.extract(key, meds_df.name.tolist(), limit=5, scorer=fuzz.token_set_ratio)
    return [n for n, score, _ in matches if score >= 60]

# ─── 6️⃣ UI ──────────────────────────────────────────────────────────────────
tabs = st.tabs(["1. Upload Rx", "2. Confirm", "3. Quotes"])

# Upload
with tabs[0]:
    ico, col = st.columns([4, 6])
    if os.path.exists("assets/RX Upload.svg"): ico.image("assets/RX Upload.svg", width=500, clamp=True)
    col.header("1️⃣ Upload Prescription")
    uploaded = col.file_uploader("", type=["jpg", "jpeg", "png", "pdf"])
    if uploaded:
        with st.spinner("🔍 Running OCR…"):
            st.session_state.raw = extract_text(uploaded)
        st.success("✅ OCR complete")
    if "raw" in st.session_state:
        st.text_area("Extracted Text", st.session_state.raw, height=200)

# Confirm
with tabs[1]:
    ico, col = st.columns([4, 6])
    if os.path.exists("assets/Confirm Medicine.svg"):
        ico.image("assets/Confirm Medicine.svg", width=500, clamp=True)
    col.header("2️⃣ Confirm Medicines")

    if "raw" not in st.session_state:
        col.info("🔎 Complete Step 1 first.")
    else:
        TOKENS = ("tab", "tablet", "cap", "capsule", "syp", "syr", "syrup", "inj", "mg", "ml")
        lines = [l for l in st.session_state.raw.split("\n") if any(t in l.lower() for t in TOKENS)]
        if not lines:
            lines = [l for l in st.session_state.raw.split("\n") if l.strip()]

        confirmed = []
        for i, line in enumerate(lines, 1):
            key = normalize(line)
            if len(key) < 4: continue
            opts = fuzzy_opts(key)
            if not opts: continue
            c1, c2 = st.columns([3, 1])
            med = c1.selectbox(f"{i}. {line}", opts, key=f"med_{i}")
            qty = c2.text_input("Qty", key=f"qty_{i}")
            confirmed.append((med, qty))

        # optional manual entry
        if not confirmed:
            new_med = col.text_input("Manual medicine name")
            new_qty = col.text_input("Qty", key="manual_qty")
            if new_med: confirmed.append((new_med, new_qty))

        if confirmed: st.session_state.confirmed = confirmed

# Quotes
with tabs[2]:
    ico, col = st.columns([4, 6])
    if os.path.exists("assets/Price Comparison.svg"):
        ico.image("assets/Price Comparison.svg", width=500, clamp=True)
    col.header("3️⃣ Compare Prices & ETA")

    if "confirmed" not in st.session_state:
        col.info("📝 Complete Step 2 first.")
    else:
        for med, qty in st.session_state.confirmed:
            try:
                qty_int = max(1, int(qty))
            except Exception:
                qty_int = 1

            col.subheader(f"{med} × {qty_int}")
            mid = name_to_id.get(med)
            df = vendor_df[vendor_df.medicine_id == mid].copy()
            if df.empty:
                col.warning("No vendor data.")
                continue

            df["price_per_tab"] = df.price
            df["total"] = df.price_per_tab * qty_int
            df["stock_note"] = np.where(df.stock >= qty_int, "", "only " + df.stock.astype(str) + " left")
            df["instock_flag"] = (df.stock >= qty_int).astype(int)

            metric = "total" if sort_by.startswith("Price") else "eta_minutes"
            df = df.sort_values(["instock_flag", metric], ascending=[False, True])

            best = df.iloc[0]
            if sort_by.startswith("Price"):
                col.metric("Best Price", f"₹{best.total:.2f}", f"{best.eta_minutes} min ETA")
            else:
                col.metric("Fastest ETA", f"{best.eta_minutes} min", f"₹{best.total:.2f}")

            cards_html = "".join(
                product_card(
                    img_url=r.get("image_url") or f"https://dummyimage.com/300x200/ffffff/000000&text={r.vendor_name.split()[0]}",
                    name=r.vendor_name,
                    total=r.total,
                    stock_note=r.stock_note,
                )
                for _, r in df.iterrows()
            )
            col.markdown(f'<div class="grid">{cards_html}</div>', unsafe_allow_html=True)
