# ===============================  app.py  ====================================
import os, io, re, json, base64, unicodedata, pathlib
from typing import Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from rapidfuzz import process
from pdf2image import convert_from_bytes
import fitz                                   # PyMuPDF
from google.oauth2 import service_account
from google.cloud import vision_v1


# ───────────────── 0. GOOGLE VISION CREDS ───────────────────────────────────
raw = st.secrets.get("GOOGLE_CREDENTIALS") or st.secrets.get("service_account_json")
info = json.loads(raw) if isinstance(raw, str) else dict(raw)
creds = service_account.Credentials.from_service_account_info(info)
client = vision_v1.ImageAnnotatorClient(credentials=creds)


# ───────────────── 1. PAGE & CSS ────────────────────────────────────────────
st.set_page_config("Pharmazeniq", "💊", layout="wide")

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
html,body{height:100%;margin:0;font-family:'Inter',sans-serif}
.block-container{padding:0 1rem 4rem;height:100%;overflow-y:auto}
.card{border:1px solid #eee;border-radius:.8rem;padding:.8rem;box-shadow:0 1px 4px rgba(0,0,0,.06);transition:.15s}
.card:hover{transform:translateY(-4px);box-shadow:0 6px 16px rgba(0,0,0,.12)}
.card img{width:100%;border-radius:.6rem}
.card-title{font-size:.95rem;font-weight:600;margin:.4rem 0 .2rem}
.grid{display:grid;gap:1rem;padding:0 .5rem}
@media(max-width:600px){.grid{grid-template-columns:repeat(2,1fr)}}
@media(min-width:601px) and (max-width:992px){.grid{grid-template-columns:repeat(3,1fr)}}
@media(min-width:993px){.grid{grid-template-columns:repeat(3,1fr)}}
footer{visibility:hidden}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────── 1-b. helper for vendor logos and cards ────────────────────
def slug(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").lower()

def logo_for(vendor: str) -> str:
    p = pathlib.Path("assets/vendor_logos") / f"{slug(vendor)}.png"
    return str(p) if p.exists() else "https://dummyimage.com/300x200/ffffff/000000&text=%20"

def make_card(vendor: str, subtitle_html: str) -> str:
    img = logo_for(vendor)
    return f'<div class="card"><img src="{img}" loading="lazy"><div class="card-title">{vendor}<br><small>{subtitle_html}</small></div></div>'


# ───────────────── 2. HEADER & SIDEBAR ──────────────────────────────────────
if os.path.exists("assets/header_banner.png"):
    st.image("assets/header_banner.png", use_container_width=True)

with st.sidebar:
    if os.path.exists("animation.gif"):
        b64 = base64.b64encode(open("animation.gif", "rb").read()).decode()
        st.markdown(f'<img src="data:image/gif;base64,{b64}" style="width:100%;margin-bottom:1rem;">', unsafe_allow_html=True)
    st.markdown("## Filters")
    sort_by = st.radio("Sort by", ["Price (Low→High)", "Fastest ETA"])
    st.markdown("---")
    st.markdown("Need help? 📧 support@pharmazeniq.com")


# ───────────────── 3. DATA LOAD ─────────────────────────────────────────────
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv("data/medicines.csv"), pd.read_csv("data/vendor_prices.csv")

meds_df, vendor_df = load_data()
name_to_id = dict(zip(meds_df.name, meds_df.id))
id_to_name = dict(zip(meds_df.id, meds_df.name))


# ───────────────── 4. OCR UTILITIES ─────────────────────────────────────────
def deskew(img: Image.Image) -> bytes:
    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(th > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle += 90 if angle < -45 else 0
    M = cv2.getRotationMatrix2D((th.shape[1] / 2, th.shape[0] / 2), angle, 1)
    warped = cv2.warpAffine(th, M, th.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
    _, buf = cv2.imencode(".jpg", warped)
    return buf.tobytes()

def ocr(b: bytes) -> str:
    return client.document_text_detection(image=vision_v1.Image(content=b)).full_text_annotation.text or ""

def pdf_to_images(data: bytes, dpi=300):
    return [Image.open(io.BytesIO(p.get_pixmap(dpi=dpi).tobytes())) for p in fitz.open(stream=data, filetype="pdf")]

def extract_text(upload) -> str:
    raw = upload.read()
    pages: List[Image.Image] = []
    if upload.type == "application/pdf":
        try:
            pages = convert_from_bytes(raw, dpi=300)
        except Exception:
            pages = pdf_to_images(raw)
    if not pages:
        try:
            pages = [Image.open(io.BytesIO(raw))]
        except Exception:
            st.error("❌ Unsupported file."); return ""
    return "\n".join(ocr(deskew(p)) for p in pages)


# ───────────────── 5. MATCH HELPERS ─────────────────────────────────────────
TOKENS = ("tab", "tablet", "cap", "capsule", "syr", "syp", "inj", "mg", "ml")
def norm(t: str): return re.sub(r"[^A-Za-z0-9 ]+", "", t).lower().strip()
def opts(key: str): return [n for n, s, _ in process.extract(key, meds_df.name, limit=5) if s >= 50]


# ───────────────── 6. UI TABS ───────────────────────────────────────────────
tabs = st.tabs(["Upload Rx", "Confirm", "Quotes"])


# Upload
with tabs[0]:
    st.header("1️⃣ Upload Prescription")
    up = st.file_uploader("", ["jpg", "jpeg", "png", "pdf"])
    if up:
        with st.spinner("Running OCR…"): st.session_state.raw = extract_text(up)
        st.success("Done ✅")
    if "raw" in st.session_state:
        st.text_area("Extracted Text", st.session_state.raw, height=200)


# Confirm
with tabs[1]:
    st.header("2️⃣ Confirm Medicines")
    if "raw" not in st.session_state:
        st.info("Upload prescription first.")
    else:
        lines = [l.strip() for l in st.session_state.raw.split("\n") if l.strip()]
        merged = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if any(t in line.lower() for t in TOKENS) and i + 1 < len(lines):
                nxt = lines[i + 1]
                if not any(t in nxt.lower() for t in TOKENS):
                    line = f"{line} {nxt}"; i += 1
            merged.append(line); i += 1

        confirmed = []
        for idx, line in enumerate(merged, 1):
            key = norm(line)
            if len(key) < 4: continue
            cand = opts(key)
            if not cand: continue
            c1, c2 = st.columns([3, 1])
            med = c1.selectbox(f"{idx}. {line}", cand, key=f"med_{idx}")
            qty = c2.text_input("Qty", key=f"qty_{idx}")
            confirmed.append((med, qty))

        if not confirmed:
            st.info("No matches—enter manually.")
            med = st.text_input("Medicine"); qty = st.text_input("Qty")
            if med: confirmed.append((med, qty))

        if confirmed: st.session_state.confirmed = confirmed


# Quotes – vendor comparison with full per-medicine table
with tabs[2]:
    st.header("3️⃣ Compare Prices & ETA")

    if "confirmed" not in st.session_state:
        st.info("Confirm medicines first."); st.stop()

    # build order
    order = []
    for m, q in st.session_state.confirmed:
        try: q = max(1, int(q))
        except: q = 1
        order.append({"mid": name_to_id.get(m), "name": m, "qty": q})

    wanted = {o["mid"] for o in order}
    df = vendor_df[vendor_df.medicine_id.isin(wanted)].copy()
    qty_map = {o["mid"]: o["qty"] for o in order}

    df["ordered_qty"] = df.medicine_id.map(qty_map)
    df["supplied"] = df[["stock", "ordered_qty"]].min(axis=1)
    df["line_total"] = df.price * df.supplied
    df["short"] = df.ordered_qty - df.supplied

    roll = (df.groupby("vendor_name")
              .agg(total_price=("line_total", "sum"),
                   eta=("eta_minutes", "max"),
                   missing=("short", lambda x: (x > 0).sum()))
              .reset_index())
    roll = roll[roll.total_price > 0]
    if roll.empty:
        st.warning("No vendor can supply any part of this order."); st.stop()

    key = "total_price" if sort_by.startswith("Price") else "eta"
    roll = roll.sort_values(["missing", key]).reset_index(drop=True)

    col1, col2 = st.columns(2)
    col1.metric("Average price", f"₹{roll.total_price.mean():.0f}")
    col2.metric("Average ETA", f"{roll.eta.mean():.0f} min")

    # cards
    cards = ""
    for _, r in roll.iterrows():
        p = f"₹{r.total_price:.0f}"; e = f"{r.eta} min"
        sub = f"{p}<br>{e}" if key == "total_price" else f"{e}<br>{p}"
        cards += make_card(r.vendor_name, sub)
    st.markdown(f'<div class="grid">{cards}</div>', unsafe_allow_html=True)

    # breakdown tables
    st.write("---"); st.subheader("Vendor breakdown")
    for _, r in roll.iterrows():
        head = f"**{r.vendor_name}** — ₹{r.total_price:.0f} • {r.eta} min"
        if r.missing: head += f" _(short {r.missing} item{'s' if r.missing>1 else ''})_"
        with st.expander(head):
            base = pd.DataFrame(order)                    # mid,name,qty
            vd = df[df.vendor_name == r.vendor_name][["medicine_id","price","supplied","line_total"]]\
                    .rename(columns={"medicine_id":"mid","price":"Price/Tab","supplied":"Supplied","line_total":"Line Total"})
            full = base.merge(vd, on="mid", how="left")
            full["Medicine"] = full["name"]
            full["Ordered"] = full["qty"]
            full["Price/Tab"] = full["Price/Tab"].fillna("—")
            full["Supplied"] = full["Supplied"].fillna(0).astype(int)
            full["Line Total"] = full["Line Total"].fillna(0).astype(int)
            show = full[["Medicine","Price/Tab","Ordered","Supplied","Line Total"]]
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.markdown(f"**Vendor total:** ₹{r.total_price:.0f} &nbsp;|&nbsp; **ETA:** {r.eta} min")
