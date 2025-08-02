# ============================  app.py  ======================================
import os, io, re, base64, json
from typing import Tuple, List

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from rapidfuzz import process, fuzz
from pdf2image import convert_from_bytes
import fitz                                  # PyMuPDF
from google.oauth2 import service_account
from google.cloud import vision_v1


# ────────────────── 0. VISION CREDS ─────────────────────────────────────────
raw = st.secrets.get("GOOGLE_CREDENTIALS") or st.secrets.get("service_account_json")
info = json.loads(raw) if isinstance(raw, str) else dict(raw)
creds = service_account.Credentials.from_service_account_info(info)
client = vision_v1.ImageAnnotatorClient(credentials=creds)


# ────────────────── 1. PAGE + CSS ───────────────────────────────────────────
st.set_page_config(page_title="Pharmazeniq", page_icon="💊", layout="wide")

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
html,body{height:100%;margin:0;font-family:'Inter',sans-serif}
.block-container{padding:0 1rem 4rem;height:100%;overflow-y:auto}

.card{border:1px solid #eee;border-radius:.8rem;padding:.8rem;
      box-shadow:0 1px 4px rgba(0,0,0,.06);transition:.15s}
.card:hover{transform:translateY(-4px);box-shadow:0 6px 16px rgba(0,0,0,.12)}
.card img{width:100%;border-radius:.6rem}
.card-title{font-size:.95rem;font-weight:600;margin:.4rem 0 .2rem}
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

def make_card(img: str, vendor_html: str, note: str = "") -> str:
    badge = f'<span class="stock-badge">{note}</span>' if note else ""
    return (
        f'<div class="card"><img src="{img}" loading="lazy">'
        f'<div class="card-title">{vendor_html}</div>{badge}</div>'
    )


# ────────────────── 2. HEADER & SIDEBAR ─────────────────────────────────────
if os.path.exists("assets/header_banner.png"):
    st.image("assets/header_banner.png", use_container_width=True)

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


# ────────────────── 3. DATA ────────────────────────────────────────────────
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv("data/medicines.csv"), pd.read_csv("data/vendor_prices.csv")

meds_df, vendor_df = load_data()
name_to_id = dict(zip(meds_df.name, meds_df.id))
id_to_name = dict(zip(meds_df.id, meds_df.name))


# ────────────────── 4. OCR ─────────────────────────────────────────────────
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

def ocr(img_bytes: bytes) -> str:
    return client.document_text_detection(
        image=vision_v1.Image(content=img_bytes)
    ).full_text_annotation.text or ""

def pdf_to_images(data: bytes, dpi=300):
    return [
        Image.open(io.BytesIO(p.get_pixmap(dpi=dpi).tobytes()))
        for p in fitz.open(stream=data, filetype="pdf")
    ]

def extract_text(upload) -> str:
    raw = upload.read()
    pages: List[Image.Image] = []
    if upload.type == "application/pdf":
        try: pages = convert_from_bytes(raw, dpi=300)
        except Exception: pages = pdf_to_images(raw)
    if not pages:
        try: pages = [Image.open(io.BytesIO(raw))]
        except Exception:
            st.error("❌ Unsupported file."); return ""
    return "\n".join(ocr(deskew(p)) for p in pages)


# ────────────────── 5. MATCH HELPERS ───────────────────────────────────────
TOKENS = ("tab","tablet","cap","capsule","syr","syp","inj","mg","ml")
def norm(t:str): return re.sub(r"[^A-Za-z0-9 ]+","",t).lower().strip()
def opts(key:str): return [n for n,s,_ in process.extract(key, meds_df.name, limit=5) if s>=50]


# ────────────────── 6. UI TABS ─────────────────────────────────────────────
tabs = st.tabs(["Upload Rx","Confirm","Quotes"])


# Upload
with tabs[0]:
    st.header("1️⃣ Upload Prescription")
    up = st.file_uploader("", ["jpg","jpeg","png","pdf"])
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
        raw_lines=[l.strip() for l in st.session_state.raw.split("\n") if l.strip()]
        merged=[]
        i=0
        while i<len(raw_lines):
            line=raw_lines[i]
            if any(t in line.lower() for t in TOKENS) and i+1<len(raw_lines):
                nxt=raw_lines[i+1]
                if not any(t in nxt.lower() for t in TOKENS):
                    line=f"{line} {nxt}"; i+=1
            merged.append(line); i+=1

        confirmed=[]
        for idx,line in enumerate(merged,1):
            key=norm(line)
            if len(key)<4: continue
            o=opts(key)
            if not o: continue
            c1,c2=st.columns([3,1])
            med=c1.selectbox(f"{idx}. {line}",o,key=f"med_{idx}")
            qty=c2.text_input("Qty",key=f"qty_{idx}")
            confirmed.append((med,qty))

        if not confirmed:
            st.info("No matches—enter manually.")
            med=st.text_input("Medicine"); qty=st.text_input("Qty")
            if med: confirmed.append((med,qty))

        if confirmed: st.session_state.confirmed=confirmed


# Quotes  – vendor-level comparison with shortfall info
with tabs[2]:
    st.header("3️⃣ Compare Prices & ETA")

    if "confirmed" not in st.session_state:
        st.info("Confirm medicines first.")
        st.stop()

    # Build order list
    order=[]
    for med_name, qty in st.session_state.confirmed:
        try: qty=max(1,int(qty))
        except: qty=1
        order.append({"mid": name_to_id.get(med_name), "qty": qty, "name": med_name})

    wanted_ids={o["mid"] for o in order}
    df=vendor_df[vendor_df.medicine_id.isin(wanted_ids)].copy()
    qty_map={o["mid"]:o["qty"] for o in order}
    df["ordered_qty"]=df.medicine_id.map(qty_map)
    df["filled_qty"]=df[["stock","ordered_qty"]].min(axis=1)
    df["total_line"]=df.price*df.filled_qty
    df["shortfall"]=df.ordered_qty-df.filled_qty

    vendor_roll=(df.groupby("vendor_name")
                   .agg(total_price=("total_line","sum"),
                        eta=("eta_minutes","max"),
                        missing_items=("shortfall",lambda x:(x>0).sum()))
                   .reset_index())
    vendor_roll=vendor_roll[vendor_roll.total_price>0]
    if vendor_roll.empty:
        st.warning("No vendor has stock for any item.")
        st.stop()

    sort_key="total_price" if sort_by.startswith("Price") else "eta"
    vendor_roll=vendor_roll.sort_values(["missing_items",sort_key]).reset_index(drop=True)

    st.metric("Average price",f"₹{vendor_roll.total_price.mean():.0f}")
    st.metric("Average ETA",f"{vendor_roll.eta.mean():.0f} min")

    cards=""
    for _,v in vendor_roll.iterrows():
        subtitle=(f"{v.eta} min<br>₹{v.total_price:.0f}"
                  if sort_key=="eta"
                  else f"₹{v.total_price:.0f}<br>{v.eta} min")
        # build shortfall note
        rows=df[df.vendor_name==v.vendor_name]
        notes=[f"{r.filled_qty}/{r.ordered_qty} {id_to_name[r.medicine_id]}"
               for _,r in rows.iterrows() if r.shortfall>0]
        note_text=", ".join(notes)
        cards+=make_card(
            img="https://dummyimage.com/300x200/ffffff/000000&text=%20",
            vendor_html=f"{v.vendor_name}<br><small>{subtitle}</small>",
            note=note_text
        )
    st.markdown(f'<div class="grid">{cards}</div>',unsafe_allow_html=True)
