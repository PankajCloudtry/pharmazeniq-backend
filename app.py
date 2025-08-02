import os, io, re, base64, json
from typing import Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from rapidfuzz import process, fuzz
from pdf2image import convert_from_bytes, exceptions as pdf2image_exc
import fitz
from google.oauth2 import service_account
from google.cloud import vision_v1

# ── 0. CREDENTIALS ──────────────────────────────────────────────────────────
raw = st.secrets.get("GOOGLE_CREDENTIALS") or st.secrets.get("service_account_json")
if raw is None:
    st.error("❌ Add Vision creds in Secrets then reload.")
    st.stop()
creds = service_account.Credentials.from_service_account_info(
    json.loads(raw) if isinstance(raw, str) else dict(raw)
)
client = vision_v1.ImageAnnotatorClient(credentials=creds)

# ── 1. PAGE CONFIG & GLOBAL CSS ─────────────────────────────────────────────
st.set_page_config("Pharmazeniq", "💊", layout="wide")

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
html,body,div[class^="st"]{font-family:'Inter',sans-serif}
.block-container{padding:0 1rem;min-height:100vh;overflow-y:auto}
.card{border:1px solid #eee;border-radius:.8rem;padding:.8rem;
      box-shadow:0 1px 4px rgba(0,0,0,.06);transition:.15s}
.card:hover{transform:translateY(-4px);
            box-shadow:0 6px 16px rgba(0,0,0,.12)}
.card img{width:100%;border-radius:.6rem}
.card-title{font-size:.95rem;font-weight:600;margin:.4rem 0 .2rem}
.card-price{font-size:.9rem;color:#e91e63;font-weight:600}
.stock-badge{font-size:.75rem;color:#555;background:#f5f5f5;
             padding:0 .4rem;border-radius:.4rem}
.grid{display:grid;gap:1rem;padding:0 .5rem;overflow-x:hidden}
@media(max-width:600px){.grid{grid-template-columns:repeat(2,1fr)}}
@media(min-width:601px) and (max-width:992px){.grid{grid-template-columns:repeat(3,1fr)}}
@media(min-width:993px){.grid{grid-template-columns:repeat(3,1fr)}}
footer{visibility:hidden}
</style>
""",
    unsafe_allow_html=True,
)

def card(img: str, vendor: str, total: float, note: str):
    badge = f'<span class="stock-badge">{note}</span>' if note else ""
    return (
        f'<div class="card"><img src="{img}" loading="lazy">'
        f'<div class="card-title">{vendor}</div>'
        f'<div class="card-price">₹{total:.2f}</div>{badge}</div>'
    )

# ── 2. DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv("data/medicines.csv"), pd.read_csv("data/vendor_prices.csv")

meds_df, vendor_df = load_data()
name_to_id = dict(zip(meds_df.name, meds_df.id))

# ── 3. OCR HELPERS ──────────────────────────────────────────────────────────
def deskew(img: Image.Image) -> bytes:
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(th > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = angle + 90 if angle < -45 else angle
    M = cv2.getRotationMatrix2D((th.shape[1] / 2, th.shape[0] / 2), angle, 1)
    warped = cv2.warpAffine(th, M, th.shape[::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    _, buf = cv2.imencode(".jpg", warped)
    return buf.tobytes()

def ocr(image_bytes: bytes) -> str:
    return client.document_text_detection(image=vision_v1.Image(content=image_bytes)).full_text_annotation.text

def pdf_to_images(data: bytes, dpi=300):
    return [Image.open(io.BytesIO(p.get_pixmap(dpi=dpi).tobytes())) for p in fitz.open(stream=data, filetype="pdf")]

def extract_text(uploaded) -> str:
    raw = uploaded.read()
    pages = []
    if uploaded.type == "application/pdf":
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(raw, dpi=300)
        except Exception:
            pages = pdf_to_images(raw)
    if not pages:
        try:
            pages = [Image.open(io.BytesIO(raw))]
        except Exception:
            st.error("❌ Unsupported file.")
            return ""
    return "\n".join(ocr(deskew(p)) for p in pages)

# ── 4. FUZZY MATCH ───────────────────────────────────────────────────────────
TOKENS = ("tab", "tablet", "cap", "capsule", "syr", "syp", "inj", "mg", "ml")
def norm(txt: str):
    txt = re.sub(r"^[\s\-\•\d\.]+", "", txt)
    txt = re.sub(r"[^A-Za-z0-9 ]+", "", txt)
    return txt.lower().strip()

def match_options(key: str):
    return [n for n, s, _ in process.extract(key, meds_df.name, limit=5) if s >= 60]

# ── 5. UI ────────────────────────────────────────────────────────────────────
tabs = st.tabs(["Upload Rx", "Confirm", "Quotes"])

# Upload tab
with tabs[0]:
    st.header("1️⃣ Upload Prescription")
    file = st.file_uploader("", type=["jpg", "jpeg", "png", "pdf"])
    if file:
        with st.spinner("OCR in progress…"):
            st.session_state.raw = extract_text(file)
        st.success("Done ✅")
    if "raw" in st.session_state:
        st.text_area("Extracted Text", st.session_state.raw, height=200)

# Confirm tab
with tabs[1]:
    st.header("2️⃣ Confirm Medicines")
    if "raw" not in st.session_state:
        st.info("Upload a prescription first.")
    else:
        lines = [
            l for l in st.session_state.raw.split("\n")
            if any(t in l.lower() for t in TOKENS) and len(norm(l)) >= 4
        ] or [l for l in st.session_state.raw.split("\n") if l.strip()]
        confirmed = []
        for i, line in enumerate(lines, 1):
            opts = match_options(norm(line))
            if not opts: continue
            c1, c2 = st.columns([3, 1])
            med = c1.selectbox(f"{i}. {line}", opts, key=f"med_{i}")
            qty = c2.text_input("Qty", key=f"qty_{i}")
            confirmed.append((med, qty))
        if not confirmed:
            st.info("No matches—enter manually below.")
            med = st.text_input("Medicine name")
            qty = st.text_input("Qty")
            if med: confirmed.append((med, qty))
        if confirmed:
            st.session_state.confirmed = confirmed

# Quotes tab
with tabs[2]:
    st.header("3️⃣ Compare Prices & ETA")
    if "confirmed" not in st.session_state:
        st.info("Confirm medicines first.")
    else:
        for med, qty in st.session_state.confirmed:
            try: qty_int = max(1, int(qty))
            except: qty_int = 1
            st.subheader(f"{med} × {qty_int}")
            mid = name_to_id.get(med)
            df = vendor_df[vendor_df.medicine_id == mid].copy()
            if df.empty:
                st.warning("No vendors found.")
                continue
            df["total"] = df.price * qty_int
            df["note"] = np.where(df.stock >= qty_int, "", "only " + df.stock.astype(str) + " left")
            df["flag"] = (df.stock >= qty_int).astype(int)
            metric = "total" if sort_by.startswith("Price") else "eta_minutes"
            df = df.sort_values(["flag", metric], ascending=[False, True])
            best = df.iloc[0]
            st.metric(
                "Best Price" if metric == "total" else "Fastest ETA",
                f"₹{best.total:.2f}" if metric == "total" else f"{best.eta_minutes} min",
                f"{best.eta_minutes} min" if metric == "total" else f"₹{best.total:.2f}",
            )
            cards = "".join(
                card(
                    img=r.get("image_url") or "https://dummyimage.com/300x200/ffffff/000000&text=%20",
                    vendor=r.vendor_name,
                    total=r.total,
                    stock_note=r.note,
                )
                for _, r in df.iterrows()
            )
            st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)
