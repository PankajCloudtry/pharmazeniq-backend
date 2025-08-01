import streamlit as st
import os, io, re, base64
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from rapidfuzz import process, fuzz
from pdf2image import convert_from_bytes
from google.cloud import vision

# 1️⃣ Page config & header
st.set_page_config("Pharmazeniq", "💊", layout="wide")
if os.path.exists("assets/header_banner.png"):
    st.image("assets/header_banner.png", use_container_width=True)
else:
    st.warning("⚠️ header_banner.png not found in assets/")

# 1b️⃣ Enormous tabs CSS
st.markdown("""
    <style>
      /* Enlarge tab labels drastically */
      [role="tablist"] [role="tab"] {
        font-size:64px !important;
        padding: 1rem 2rem !important;
      }
    </style>
""", unsafe_allow_html=True)

# 2️⃣ Sidebar: big animation.gif, sort, support
with st.sidebar:
    if os.path.exists("animation.gif"):
        gif_bytes = open("animation.gif", "rb").read()
        b64       = base64.b64encode(gif_bytes).decode()
        st.markdown(
            f'<img src="data:image/gif;base64,{b64}" '
            'style="width:100%;margin-bottom:1rem;">',
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ animation.gif not found in project root")
    st.markdown("## Filters")
    sort_by = st.radio("Sort by", ["Price (Low→High)", "ETA"])
    st.markdown("---")
    st.markdown("Need help? 📧 support@pharmazeniq.com")

# 3️⃣ Google Vision client
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"
client = vision.ImageAnnotatorClient()

# 4️⃣ Load data
@st.cache_data
def load_data():
    meds_df   = pd.read_csv("data/medicines.csv")
    vendor_df = pd.read_csv("data/vendor_prices.csv")
    return meds_df, vendor_df

meds_df, vendor_df = load_data()
name_to_id = dict(zip(meds_df.name, meds_df.id))

# 5️⃣ OCR helpers
def deskew_and_encode(img: Image.Image) -> bytes:
    arr  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(th>0))
    angle  = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle += 90
    h,w = th.shape
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    deskew = cv2.warpAffine(th, M, (w,h),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
    _, buf = cv2.imencode(".jpg", deskew)
    return buf.tobytes()

def ocr_bytes(b: bytes) -> str:
    resp = client.document_text_detection(image=vision.Image(content=b))
    return resp.full_text_annotation.text or ""

def extract_text(uploaded) -> str:
    raw   = uploaded.read()
    pages = []
    if uploaded.type=="application/pdf":
        try:
            pages = convert_from_bytes(raw, dpi=300)
        except:
            pages = []
    if not pages:
        pages = [Image.open(io.BytesIO(raw))]
    texts = []
    for pg in pages:
        texts.append( ocr_bytes(deskew_and_encode(pg)) )
    return "\n".join(texts)

# 6️⃣ Normalize & fuzzy-match
def normalize(line: str) -> str:
    x = re.sub(r"^[\s\-\•\d\.]+","", line)
    x = re.sub(r"\b\d+(\.\d+)?\s?(mg|g|ml)\b","", x, flags=re.IGNORECASE)
    x = re.sub(r"\b(tab|tabs|cap|caps)\b","", x, flags=re.IGNORECASE)
    x = re.sub(r"[^A-Za-z0-9 ]+","", x)
    return x.lower().strip()

def fuzzy_opts(key: str) -> list[str]:
    names   = meds_df.name.tolist()
    matches = process.extract(key, names, limit=5, scorer=fuzz.token_set_ratio)
    opts    = [n for n,score,_ in matches if score>=50]
    if not opts:
        opts = [n for n in names if key in normalize(n)]
    return opts

# 7️⃣ Build 3-step UI
tabs = st.tabs(["1. OCR", "2. Confirm", "3. Quotes"])

# Tab 1: Upload Prescription
with tabs[0]:
    ico, col = st.columns([4,6])  # icon gets 40% width
    if os.path.exists("assets/RX Upload.svg"):
        ico.image("assets/RX Upload.svg", width=600)
    col.header("1️⃣ Upload Prescription")
    uploaded = col.file_uploader("", type=["jpg","jpeg","png","pdf"])
    if uploaded:
        with st.spinner("Running OCR…"):
            st.session_state.raw = extract_text(uploaded)
        st.success("✅ OCR complete")
    if "raw" in st.session_state:
        st.text_area("Extracted Text", st.session_state.raw, height=200)

# Tab 2: Confirm Medicines
with tabs[1]:
    ico, col = st.columns([4,6])
    if os.path.exists("assets/Confirm Medicine.svg"):
        ico.image("assets/Confirm Medicine.svg", width=600)
    col.header("2️⃣ Confirm Medicines")
    if "raw" not in st.session_state:
        col.info("🔍 Please complete Step 1 first.")
    else:
        lines = [l for l in st.session_state.raw.split("\n")
                 if any(tok in l.lower() for tok in ("tab","mg","cap"))]
        if not lines:
            col.info("⚠️ No lines with “tab”/“mg”/“cap” found.")
        else:
            confirmed = []
            for idx,line in enumerate(lines,1):
                opts = fuzzy_opts(normalize(line))
                if not opts: continue
                c1,c2 = st.columns([3,1])
                med = c1.selectbox(f"{idx}. {line}", opts, key=f"med_{idx}")
                qty = c2.text_input("Qty", key=f"qty_{idx}")
                confirmed.append((med,qty))
            if confirmed:
                st.session_state.confirmed = confirmed

# Tab 3: Quotes & ETA
with tabs[2]:
    ico, col = st.columns([4,6])
    if os.path.exists("assets/Price Comparison.svg"):
        ico.image("assets/Price Comparison.svg", width=600)
    col.header("3️⃣ Compare Prices & ETA")
    if "confirmed" not in st.session_state:
        col.info("📝 Please complete Step 2 first.")
    else:
        for med,qty in st.session_state.confirmed:
            col.subheader(f"{med} × {qty or '–'}")
            mid = name_to_id.get(med)
            df  = vendor_df[vendor_df.medicine_id==mid].copy()
            if df.empty:
                col.warning("No quotes available.")
                continue
            df["total"] = df.price * (int(qty) if qty.isdigit() else 1)
            sort_col    = "total" if sort_by.startswith("Price") else "eta_minutes"
            df = df.sort_values(sort_col)
            best = df.iloc[0]
            if sort_by.startswith("Price"):
                col.metric("Best Price", f"₹{best.total:.2f}",
                           f"{best.eta_minutes} min ETA")
            else:
                col.metric("Fastest ETA", f"{best.eta_minutes} min",
                           f"₹{best.total:.2f}")
            col.dataframe(
                df[["vendor_name","price","stock","eta_minutes","total"]],
                use_container_width=True
            )
