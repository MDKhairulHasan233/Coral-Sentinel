"""
প্রবাল প্রহরী (Coral Sentinel) – Coral Early-Warning System

Designed so that a general viewer understands the project in ~10 seconds:
- Bangla-first, strong plain words (প্রবাল ক্ষয়, পর্যটক কমান, ভিড় কমান)
- 5-colour traffic-light alert (green / yellow / orange / red / purple)
- Same alert colours used everywhere (banner, legend, chart zones, metrics)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_ROOT))

from config.settings import (
    TRAINING_SITE,
    SITE_FEATURE_FILENAMES,
    get_data_dir,
    get_models_dir,
    get_docs_dir,
    resolve_site_csv,
    is_transfer_site,
    site_model_status,
)
from features.columns import FEATURE_COLUMNS
from features.engine import build_feature_matrix
from models.loader import load_bundle, list_available_horizons
from utils.alerts import map_alert, alert_badge_html
from utils.reliability import (
    get_horizon_metrics,
    reliability_caption,
    reliability_indicator_label,
    DEFAULT_WARNING_THR,
    DEFAULT_ALERT_THR,
)
from data.openmeteo import SITE_COORDS, SITE_MMM, build_forecast_frame
from data.oni_updater import latest_oni_lags, oni_status

BASE_COLS = [
    "date", "SST", "SSTA", "HotSpot", "DHW",
    "tmax", "tmean", "precipitation", "relative_humidity",
    "wind_speed", "solar_radiation", "cloud_cover",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
_BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def bn(value) -> str:
    """Convert any number / text to Bangla digits (72.5% -> ৭২.৫%)."""
    return str(value).translate(_BN_DIGITS)


def html(markup: str) -> None:
    """Render HTML safely: strip indentation and blank lines so Markdown
    never mistakes indented HTML for a code block."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


import inspect
import base64


def _stretch(fn) -> dict:
    """Full-width kwargs that work on both old and new Streamlit versions."""
    try:
        if "width" in inspect.signature(fn).parameters:
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


@st.cache_resource(show_spinner=False)
def get_bundle(h: int, site_name: str):
    """Load a model bundle once and reuse it on every rerun."""
    return load_bundle(h, site=site_name)


def local_image_uri(*names):
    """Return a data: URI for assets/<name> if the file exists, else None."""
    for n in names:
        fp = APP_ROOT / "assets" / n
        if fp.exists():
            mime = "image/png" if fp.suffix.lower() == ".png" else "image/jpeg"
            return f"data:{mime};base64," + base64.b64encode(fp.read_bytes()).decode()
    return None


# Shared alert palette (matches utils/alerts.py)
C_GREEN, C_YELLOW, C_ORANGE, C_RED, C_PURPLE = "#1B9E5A", "#F5B700", "#F57C00", "#D32F2F", "#7B1FA2"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="প্রবাল প্রহরী | Coral Sentinel",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS – deep-ocean + coral palette, traffic-light alerts
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Bengali:wght@400;600;700;800&family=Poppins:wght@400;600;700;800&display=swap');

:root{
  --navy:#062B4A; --ocean:#0A6EA8; --aqua:#14B8D4; --coral:#FF5A4F; --sun:#FFB020;
  --ink:#0B2A3C; --muted:#4B6577;
  --green:#1B9E5A; --yellow:#F5B700; --orange:#F57C00; --red:#D32F2F; --purple:#7B1FA2;
}
html, body, [class*="css"], .stMarkdown, button, input, label {
  font-family:'Noto Sans Bengali','Poppins',sans-serif;
}
.stApp{
  background:
    radial-gradient(1200px 500px at 90% -5%, rgba(255,90,79,.10), transparent 60%),
    radial-gradient(1000px 600px at -10% 10%, rgba(20,184,212,.18), transparent 60%),
    linear-gradient(180deg,#EAF8FC 0%, #F6FBFD 45%, #FFF6EA 100%);
  color:var(--ink);
}
header[data-testid="stHeader"]{background:transparent;}
.block-container, [data-testid="stMainBlockContainer"]{
  padding-top:1.6rem !important; padding-left:3rem !important; padding-right:3rem !important;
  max-width:100% !important; width:100% !important;}
@media (max-width:900px){
  .block-container, [data-testid="stMainBlockContainer"]{padding-left:1rem !important; padding-right:1rem !important;}
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#062B4A 0%,#0A4F7A 60%,#0A6EA8 100%)!important;
}
section[data-testid="stSidebar"] > div{background:transparent!important;}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown{color:#fff!important;}
section[data-testid="stSidebar"] h2{border-bottom:2px solid var(--sun); padding-bottom:.3rem;}
section[data-testid="stSidebar"] [data-baseweb="select"] > div{background:#E8F7FC!important;}
section[data-testid="stSidebar"] [data-baseweb="select"] *{color:#062B4A!important;}
section[data-testid="stSidebar"] .stRadio > div{background:rgba(255,255,255,.10);border-radius:12px;padding:6px 10px;}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small{color:#BFE9F5!important;}
section[data-testid="stSidebar"] div.stButton > button{
  background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.5); color:#fff!important; font-size:.95rem;
}
section[data-testid="stSidebar"] div[data-testid="stAlert"] *{color:#062B4A!important;}

/* ===== Hero ===== */
.hero{
  position:relative; overflow:hidden; border-radius:28px; padding:2.6rem 2.4rem 2.2rem;
  background:linear-gradient(120deg,#062B4A 0%,#0A5E93 55%,#0FA3C4 100%);
  box-shadow:0 18px 44px rgba(6,43,74,.35); text-align:center; margin-bottom:1.2rem;
}
.hero::before{content:""; position:absolute; right:-80px; top:-80px; width:300px; height:300px; border-radius:50%;
  background:radial-gradient(circle,rgba(255,90,79,.55),transparent 65%);}
.hero::after{content:""; position:absolute; left:-90px; bottom:-110px; width:340px; height:340px; border-radius:50%;
  background:radial-gradient(circle,rgba(255,176,32,.35),transparent 65%);}
.hero > *{position:relative; z-index:1;}
.hero-badge{display:inline-block; background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.4);
  color:#fff; padding:.3rem 1rem; border-radius:999px; font-weight:600; font-size:1rem; margin-bottom:.9rem;}
.hero-title{font-size:2.9rem; font-weight:800; line-height:1.35; color:#fff; margin:0 0 .6rem;}
.hero-title .hl{color:#FFD166;}
.hero-title .hl2{color:#FF8A80;}
.hero-sub{font-size:1.35rem; color:#DDF4FB; line-height:1.7; margin:0 auto; max-width:900px;}
.hero-sub b{color:#FFD166;}

/* ===== Section header ===== */
.section-header{
  font-size:1.8rem; font-weight:800; color:var(--navy); margin:2.2rem 0 1rem; padding:.2rem 0 .5rem .9rem;
  border-left:8px solid var(--coral); border-bottom:2px solid rgba(10,110,168,.2);
}

/* ===== Problem flow ===== */
.flow{display:flex; gap:.6rem; align-items:stretch; flex-wrap:wrap; margin-bottom:.6rem;}
.flow-card{flex:1 1 250px; border-radius:22px; padding:1.5rem 1.3rem; color:#fff; text-align:center;
  box-shadow:0 10px 26px rgba(0,0,0,.16);}
.flow-card .ico{font-size:3rem; line-height:1.1;}
.flow-card .ttl{font-size:1.45rem; font-weight:800; margin:.4rem 0 .35rem;}
.flow-card .txt{font-size:1.08rem; line-height:1.65; opacity:.96;}
.flow-1{background:linear-gradient(135deg,#F57C00,#E65100);}
.flow-2{background:linear-gradient(135deg,#E53935,#B71C1C);}
.flow-3{background:linear-gradient(135deg,#6A1B9A,#4A148C);}
.flow-arrow{align-self:center; font-size:2.2rem; color:var(--coral); font-weight:800;}

/* ===== Cards ===== */
.card{background:#fff; border-radius:22px; padding:1.6rem 1.8rem; box-shadow:0 10px 28px rgba(6,43,74,.10);
  margin-bottom:1.1rem; line-height:1.8; font-size:1.12rem;}
.card h3{margin:.1rem 0 .6rem; font-size:1.45rem; font-weight:800; color:var(--navy);}
.card ul{margin:.3rem 0 0 1.1rem; padding:0;}
.card li{margin-bottom:.4rem;}
.card-blue{border-top:7px solid var(--ocean);}
.card-coral{border-top:7px solid var(--coral);}
.card-green{border-top:7px solid var(--green);}
.card-sun{border-top:7px solid var(--sun);}

.solution{background:linear-gradient(135deg,#E3F6FB,#FFFFFF); border:2px solid var(--aqua);
  border-radius:24px; padding:1.6rem 1.9rem; box-shadow:0 10px 26px rgba(20,184,212,.18); font-size:1.15rem; line-height:1.9;}
.solution h3{color:var(--ocean); font-size:1.55rem; font-weight:800; margin:0 0 .5rem;}
.tick{color:var(--green); font-weight:800;}

/* ===== Big action banner ===== */
.action-banner{
  background:linear-gradient(100deg,#D32F2F,#FF5A4F 60%,#FF8A3D); color:#fff; border-radius:24px;
  padding:1.6rem 1.8rem; text-align:center; box-shadow:0 14px 34px rgba(211,47,47,.40); margin:1.2rem 0;}
.action-banner .big{font-size:2.3rem; font-weight:800; line-height:1.4;}
.action-banner .small{font-size:1.2rem; margin-top:.4rem; opacity:.97;}

/* ===== Signal legend ===== */
.legend{display:flex; gap:.7rem; flex-wrap:wrap;}
.sig{flex:1 1 180px; border-radius:20px; padding:1.1rem 1rem; color:#fff; text-align:center; box-shadow:0 8px 20px rgba(0,0,0,.15);}
.sig .dot{font-size:2rem;}
.sig .nm{font-size:1.3rem; font-weight:800; margin:.15rem 0;}
.sig .ac{font-size:1.02rem; line-height:1.5;}
.sig-0{background:linear-gradient(135deg,#22B36A,#1B9E5A);}
.sig-1{background:linear-gradient(135deg,#FFC929,#E0A800); color:#2B2100;}
.sig-2{background:linear-gradient(135deg,#FF9800,#E65100);}
.sig-3{background:linear-gradient(135deg,#EF5350,#B71C1C);}
.sig-4{background:linear-gradient(135deg,#9C27B0,#4A148C);}

/* ===== Alert banners (dashboard) ===== */
.alert-box{border-radius:26px; padding:1.8rem 1.8rem; color:#fff; text-align:center;}
.alert-box .a-icon{font-size:3.2rem; line-height:1;}
.alert-box .a-title{font-size:2.2rem; font-weight:800; line-height:1.35; margin:.3rem 0;}
.alert-box .a-sub{font-size:1.3rem; font-weight:600; opacity:.97;}
.alert-box .a-en{font-size:.98rem; opacity:.88; margin-top:.6rem;}
.alert-0{background:linear-gradient(135deg,#22B36A,#137A44); box-shadow:0 14px 34px rgba(27,158,90,.40);}
.alert-1{background:linear-gradient(135deg,#FFC929,#D99A00); color:#2B2100; box-shadow:0 14px 34px rgba(245,183,0,.45);}
.alert-2{background:linear-gradient(135deg,#FF9800,#E65100); box-shadow:0 14px 34px rgba(245,124,0,.45);}
.alert-3{background:linear-gradient(135deg,#EF5350,#B71C1C); box-shadow:0 14px 34px rgba(211,47,47,.45);}
.alert-4{background:linear-gradient(135deg,#9C27B0,#4A148C); box-shadow:0 14px 34px rgba(123,31,162,.50);}

.todo{border-radius:22px; background:#fff; padding:1.4rem 1.7rem; margin-top:1rem; font-size:1.2rem; line-height:1.9;
  box-shadow:0 8px 24px rgba(6,43,74,.10);}
.todo .hd{font-size:1.4rem; font-weight:800; margin-bottom:.3rem;}

/* ===== Risk bar ===== */
.riskwrap{background:#fff; border-radius:22px; padding:1.3rem 1.6rem; box-shadow:0 8px 24px rgba(6,43,74,.10); margin-top:1rem;}
.riskhead{display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap;}
.riskhead .lbl{font-size:1.25rem; font-weight:700; color:var(--navy);}
.riskhead .val{font-size:2.6rem; font-weight:800;}
.riskbar{height:26px; border-radius:999px; overflow:hidden; position:relative;
  background:linear-gradient(90deg,#1B9E5A 0%,#F5B700 30%,#F57C00 50%,#D32F2F 75%,#7B1FA2 100%);}
.riskmark{position:absolute; top:-4px; width:6px; height:34px; background:#0B2A3C; border-radius:3px; box-shadow:0 0 0 3px #fff;}
.riskscale{display:flex; justify-content:space-between; font-size:.95rem; color:var(--muted); margin-top:.3rem;}

/* ===== Metric cards ===== */
.metric-card{background:#fff; border-radius:20px; padding:1.2rem .8rem; text-align:center;
  box-shadow:0 8px 22px rgba(6,43,74,.10); border-top:6px solid var(--ocean); height:100%;}
.metric-value{font-size:2.1rem; font-weight:800; color:var(--ocean);}
.metric-label{font-size:1.02rem; color:var(--muted); font-weight:600; line-height:1.5;}
.m-coral{border-top-color:var(--coral);} .m-coral .metric-value{color:var(--coral);}
.m-sun{border-top-color:var(--sun);}   .m-sun .metric-value{color:#C98600;}
.m-green{border-top-color:var(--green);} .m-green .metric-value{color:var(--green);}

/* ===== Images ===== */
.img-card{background:#fff; border-radius:20px; padding:8px; box-shadow:0 8px 22px rgba(6,43,74,.12); margin-bottom:.5rem;}
.coral-img{border-radius:14px; width:100% !important; max-width:100% !important; height:300px !important;
  object-fit:cover !important; display:block; margin:0 !important;}
.img-card{padding:8px; overflow:hidden;}
/* Realistic "bleached" look: pale bone-white + cream haze + dull algae patches (illustration only) */
.bleach{position:relative; border-radius:14px; overflow:hidden; line-height:0;}
.bleach img{filter:saturate(.10) brightness(1.42) contrast(.62) sepia(.30) hue-rotate(-10deg);}
.bleach::before{content:""; position:absolute; inset:0; z-index:2; pointer-events:none; mix-blend-mode:multiply;
  background:
   radial-gradient(circle at 18% 88%, rgba(120,105,55,.55), transparent 38%),
   radial-gradient(circle at 78% 92%, rgba(85,110,65,.45), transparent 36%),
   radial-gradient(circle at 55% 55%, rgba(150,140,110,.25), transparent 50%);}
.bleach::after{content:""; position:absolute; inset:0; z-index:3; pointer-events:none; mix-blend-mode:screen;
  background:linear-gradient(180deg, rgba(255,250,238,.50), rgba(236,229,205,.38));}
.img-caption{font-size:1.1rem; font-weight:700; color:var(--navy); text-align:center; margin:.5rem 0 .2rem;}
.img-caption.bad{color:var(--red);} .img-caption.good{color:var(--green);}

/* ===== Buttons ===== */
div.stButton > button{
  background:linear-gradient(90deg,#0A6EA8,#14B8D4); color:#fff!important; border:none; border-radius:14px;
  font-weight:700; font-size:1.1rem; padding:.7rem 1.4rem; transition:all .25s ease;}
div.stButton > button:hover{transform:translateY(-2px); box-shadow:0 8px 20px rgba(10,110,168,.45);}
div.stButton > button[kind="primary"]{
  background:linear-gradient(90deg,#E8453C,#FF8A3D); font-size:1.35rem; padding:.95rem 1.5rem;
  box-shadow:0 10px 26px rgba(232,69,60,.40);}
div.stButton > button[kind="primary"]:hover{box-shadow:0 14px 30px rgba(232,69,60,.55);}

/* Expanders & misc */
div[data-testid="stExpander"]{background:#fff; border-radius:16px; border:1px solid rgba(10,110,168,.2);}
.footer{text-align:center; color:var(--muted); font-size:.98rem; margin-top:2.2rem; padding:1.2rem;
  border-top:2px dashed rgba(10,110,168,.25);}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state for navigation
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "welcome"


def go_to_app():
    st.session_state.page = "app"


def go_to_welcome():
    st.session_state.page = "welcome"


# Reliable, reusable image URLs (same Unsplash photos as before)
IMG_HEALTHY = "https://images.unsplash.com/photo-1546026423-cc4642628d2b?w=700&q=80&auto=format&fit=crop"
IMG_DIVER = "https://images.unsplash.com/photo-1682687981907-170c006e3744?w=700&q=80&auto=format&fit=crop"
IMG_REEF = "https://images.unsplash.com/photo-1708649290066-5f617003b93f?w=700&q=80&auto=format&fit=crop"
IMG_CLOSE = "https://images.unsplash.com/photo-1583212292454-1fe6229603b7?w=700&q=80&auto=format&fit=crop"
IMG_FISH = "https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=700&q=80&auto=format&fit=crop"
IMG_WIDE = "https://images.unsplash.com/photo-1682687220742-aba13b6e50ba?w=1200&q=80&auto=format&fit=crop"

# ---------------------------------------------------------------------------
# WELCOME PAGE  (10-second understanding)
# ---------------------------------------------------------------------------
if st.session_state.page == "welcome":

    html("""
    <div class="hero">
      <div class="hero-badge">🪸 প্রবাল প্রহরী &nbsp;|&nbsp; Coral Sentinel</div>
      <div class="hero-title">
        সমুদ্র গরম হচ্ছে — <span class="hl2">প্রবাল ক্ষয় হচ্ছে</span><br>
        আগেই জানুন, <span class="hl">প্রবাল বাঁচান</span>
      </div>
      <div class="hero-sub">
        এআই (AI) প্রতিদিন সাগরের তাপ দেখে <b>আগেভাগেই</b> বলে দেয়,
        কবে প্রবাল ক্ষয়ের ঝুঁকি বাড়ছে — আর তখন <b>পর্যটক কমানো</b> ও <b>ভিড় কমানো</b> কতটা জরুরি।
      </div>
    </div>
    """)

    cta_l, cta_m, cta_r = st.columns([1, 2, 1])
    with cta_m:
        st.button("🚀 আজকের প্রবাল ঝুঁকি দেখুন (ড্যাশবোর্ড)", on_click=go_to_app,
                  **_stretch(st.button), type="primary", key="cta_top")

    # ---- 1. Problem in 3 steps ----
    html('<div class="section-header">😟 সমস্যাটা কী? — ৩ ধাপে বুঝুন</div>')
    html("""
    <div class="flow">
      <div class="flow-card flow-1">
        <div class="ico">🌡️</div>
        <div class="ttl">১. সাগর খুব গরম</div>
        <div class="txt">অনেক দিন ধরে পানি বেশি গরম থাকলে প্রবালের ভেতরের রঙিন শৈবাল চলে যায়।</div>
      </div>
      <div class="flow-arrow">➜</div>
      <div class="flow-card flow-2">
        <div class="ico">🪸</div>
        <div class="ttl">২. প্রবাল ক্ষয়</div>
        <div class="txt">প্রবাল রং হারিয়ে সাদা হয়ে যায় (ব্লিচিং)। গরম না কমলে প্রবাল মরে যায়।</div>
      </div>
      <div class="flow-arrow">➜</div>
      <div class="flow-card flow-3">
        <div class="ico">🐟</div>
        <div class="ttl">৩. সবার ক্ষতি</div>
        <div class="txt">মাছ কমে যায়, তীর সুরক্ষা কমে, পর্যটন ও জেলেদের আয় কমে।</div>
      </div>
    </div>
    """)

    # ---- 2. Before / after ----
    html('<div class="section-header">👀 নিজের চোখে দেখুন — সুস্থ প্রবাল বনাম ক্ষয়ে যাওয়া প্রবাল</div>')
    b1, b2 = st.columns(2, gap="large")
    real_bleached = local_image_uri("bleached_coral.jpg", "bleached_coral.jpeg", "bleached_coral.png")
    with b1:
        html(f"""
        <div class="img-card"><img class="coral-img" src="{IMG_HEALTHY}" alt="সুস্থ প্রবাল">
        <div class="img-caption good">✅ আগে: সুস্থ, রঙিন প্রবাল</div></div>
        """)
    with b2:
        if real_bleached:
            html(f"""
            <div class="img-card"><img class="coral-img" src="{real_bleached}" alt="ক্ষয়ে যাওয়া প্রবাল">
            <div class="img-caption bad">❌ পরে: সাদা হয়ে ক্ষয়ে যাওয়া প্রবাল</div></div>
            """)
        else:
            html(f"""
            <div class="img-card"><div class="bleach"><img class="coral-img" src="{IMG_HEALTHY}" alt="ক্ষয়ে যাওয়া প্রবাল (উদাহরণ)"></div>
            <div class="img-caption bad">❌ পরে: সাদা হয়ে ক্ষয়ে যাওয়া প্রবাল (উদাহরণ চিত্র)</div></div>
            """)
    if not real_bleached:
        st.caption("ডানের ছবিটি বোঝানোর জন্য একই ছবি ফ্যাকাশে করে তৈরি — এটি প্রকৃত ক্ষয়ের ছবি নয়। (প্রকৃত ছবি দিতে: assets/bleached_coral.jpg রাখুন)")

    # ---- 3. Our solution ----
    html('<div class="section-header">💡 আমাদের সমাধান — প্রবাল প্রহরী</div>')
    html("""
    <div class="solution">
      <h3>ক্ষতি হওয়ার আগেই সতর্কবার্তা</h3>
      <div><span class="tick">✔</span> প্রতিদিন সাগরের তাপ ও জলবায়ুর তথ্য দেখে</div>
      <div><span class="tick">✔</span> আগামী <b>১ থেকে ২৮ দিনে</b> প্রবাল ক্ষয়ের ঝুঁকি কতটা — শতকরা হারে বলে</div>
      <div><span class="tick">✔</span> <b>৫টি রঙে</b> বুঝিয়ে দেয়: 🟢 নিরাপদ → 🟣 চরম বিপদ</div>
      <div><span class="tick">✔</span> কী করতে হবে — সরাসরি বলে দেয়: <b>“পর্যটক কমান”</b>, <b>“ভিড় কমান”</b></div>
      <div><span class="tick">✔</span> সেন্ট মার্টিনসহ ৩টি প্রবাল এলাকার জন্য তৈরি</div>
    </div>
    """)

    # ---- 4. Signals ----
    html('<div class="section-header">🚦 রং দেখুন, কাজ বুঝুন</div>')
    html("""
    <div class="legend">
      <div class="sig sig-0"><div class="dot">🟢</div><div class="nm">নিরাপদ</div><div class="ac">প্রবাল ভালো আছে।<br>প্রবাল ছোঁবেন না।</div></div>
      <div class="sig sig-1"><div class="dot">🟡</div><div class="nm">সাবধান</div><div class="ac">সাগর গরম হচ্ছে।<br>নতুন ভিড় বাড়াবেন না।</div></div>
      <div class="sig sig-2"><div class="dot">🟠</div><div class="nm">বিপদ</div><div class="ac">প্রবাল ক্ষয় শুরু হতে পারে।<br><b>পর্যটক কমান!</b></div></div>
      <div class="sig sig-3"><div class="dot">🔴</div><div class="nm">মহাবিপদ</div><div class="ac">প্রবাল মরার ঝুঁকি অনেক বেশি।<br><b>পর্যটন সীমিত করুন!</b></div></div>
      <div class="sig sig-4"><div class="dot">🟣</div><div class="nm">চরম জরুরি</div><div class="ac">ব্যাপক প্রবাল মৃত্যুর আশঙ্কা।<br><b>পর্যটন বন্ধ করুন!</b></div></div>
    </div>
    """)

    # ---- 5. Big action banner ----
    html("""
    <div class="action-banner">
      <div class="big">🚫 কমলা বা লাল সংকেত হলে — পর্যটক কমান, ভিড় কমান!</div>
      <div class="small">কম চাপ = সুস্থ প্রবাল = বেশি মাছ = ভবিষ্যতেও পর্যটন ও জীবিকা</div>
    </div>
    """)

    # ---- 6. Who should do what ----
    html('<div class="section-header">🤝 কে কী করবেন?</div>')
    w1, w2, w3 = st.columns(3)
    with w1:
        html("""
        <div class="card card-green"><h3>🧳 পর্যটক</h3><ul>
          <li><b>প্রবালে দাঁড়াবেন না, ছোঁবেন না</b></li>
          <li>প্রবাল, ঝিনুক, শামুক <b>তুলে আনবেন না</b></li>
          <li>রিফ-সেফ সানস্ক্রিন বা র‍্যাশ গার্ড পরুন</li>
          <li>লাল সংকেতে ভ্রমণ <b>পিছিয়ে দিন</b></li>
        </ul></div>
        """)
    with w2:
        html("""
        <div class="card card-sun"><h3>⛵ নৌকা ও রিসোর্ট মালিক</h3><ul>
          <li><b>প্রবালের উপর নোঙর ফেলবেন না</b></li>
          <li>একসাথে বেশি নৌকা নেবেন না</li>
          <li>পর্যটকদের <b>ভাগ করে</b> পাঠান, ভিড় কমান</li>
          <li>সংকেতের রং দেখে সিদ্ধান্ত নিন</li>
        </ul></div>
        """)
    with w3:
        html("""
        <div class="card card-coral"><h3>🏛️ প্রশাসন ও পরিবেশকর্মী</h3><ul>
          <li>কমলা/লালে <b>পর্যটকসংখ্যা সীমিত</b> করুন</li>
          <li>ডুব ও স্নরকেলিং <b>সাময়িক বন্ধ</b> রাখুন</li>
          <li>নজরদারি ও জরিপ বাড়ান</li>
          <li>স্থানীয় মানুষকে আগে জানান</li>
        </ul></div>
        """)

    # ---- 7. Gallery ----
    html('<div class="section-header">🌊 যে সৌন্দর্য আমরা হারাতে চাই না</div>')
    g1, g2, g3 = st.columns(3)
    for col, img, cap in ((g1, IMG_CLOSE, "🪸 জীবন্ত প্রবাল"), (g2, IMG_FISH, "🐟 প্রবালের মাঝে মাছ"), (g3, IMG_REEF, "🐠 রঙিন সমুদ্রজীবন")):
        with col:
            html(f'<div class="img-card"><img class="coral-img" src="{img}" alt="coral"><div class="img-caption">{cap}</div></div>')

    # ---- 8. Features ----
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        html('<div class="metric-card"><div style="font-size:2.3rem;">🎯</div><div class="metric-label">শতকরা হারে<br>নির্ভরযোগ্য ঝুঁকি</div></div>')
    with f2:
        html('<div class="metric-card m-coral"><div style="font-size:2.3rem;">🏝️</div><div class="metric-label">৩টি এলাকা<br>সেন্ট মার্টিন • হ্যাভলক • নর্থ আন্দামান</div></div>')
    with f3:
        html('<div class="metric-card m-sun"><div style="font-size:2.3rem;">📅</div><div class="metric-label">১–২৮ দিন আগে<br>থেকে সতর্কতা</div></div>')
    with f4:
        html('<div class="metric-card m-green"><div style="font-size:2.3rem;">🚦</div><div class="metric-label">৫ রঙের<br>সহজ সংকেত</div></div>')

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.button("🚀 ড্যাশবোর্ডে যান — আজকের ঝুঁকি দেখুন", on_click=go_to_app,
                  **_stretch(st.button), type="primary", key="cta_bottom")

    with st.expander("🇬🇧 English summary (for reviewers)"):
        st.markdown(
            "**Coral Sentinel** is an AI early-warning system that estimates the probability of dangerous "
            "marine heat stress (DHW ≥ 4 °C-weeks) 1–28 days ahead for coral reefs at Saint Martin, "
            "Havelock and North Andaman. Results are shown as a 5-level traffic-light alert with plain "
            "action guidance (e.g. *reduce tourist numbers* at orange/red). "
            "“প্রবাল ক্ষয়” is used in the UI as the plain-language term for coral bleaching / heat-induced coral decline."
        )

    html("""
    <div class="footer">
      গবেষণা প্রোটোটাইপ — NOAA Coral Reef Watch-এর পরিপূরক, বিকল্প নয়।<br>
      ❤️ সমুদ্র ও সমুদ্রনির্ভর মানুষের জন্য তৈরি &nbsp;|&nbsp; ছবি: Unsplash
    </div>
    """)

    st.stop()


# ---------------------------------------------------------------------------
# MAIN APP PAGE
# ---------------------------------------------------------------------------
SITE_BN = {"Saint Martin": "সেন্ট মার্টিন", "Havelock": "হ্যাভলক", "North Andaman": "নর্থ আন্দামান"}
MODE_BN = {
    "Latest from research dataset": "📁 গবেষণা তথ্যভান্ডারের সর্বশেষ তথ্য",
    "Upload recent daily CSV": "⬆️ নিজের CSV ফাইল আপলোড",
    "Forecast from today (Open-Meteo)": "🌦️ আজ থেকে পূর্বাভাস (Open-Meteo)",
}

nav1, nav2, nav3 = st.columns([2, 5, 2])
with nav1:
    st.button("← শুরুর পাতা", on_click=go_to_welcome)
with nav2:
    html("""
    <div style="text-align:center; font-size:1.9rem; font-weight:800; color:#062B4A;">
    🪸 প্রবাল প্রহরী — <span style="color:#E8453C;">ড্যাশবোর্ড</span></div>
    """)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ সেটিংস")

site = st.sidebar.selectbox(
    "📍 প্রবাল এলাকা বেছে নিন",
    list(SITE_FEATURE_FILENAMES.keys()),
    format_func=lambda s: f"{SITE_BN.get(s, s)} ({s})",
)

if is_transfer_site(site):
    st.sidebar.error(
        f"**পরীক্ষামূলক এলাকা**\n\n"
        f"মডেলটি মূলত **{SITE_BN.get(TRAINING_SITE, TRAINING_SITE)}**-এর তথ্যে তৈরি। "
        f"**{SITE_BN.get(site, site)}**-এর ফলাফল কম নির্ভরযোগ্য হতে পারে।"
    )
else:
    st.sidebar.success(f"✅ {SITE_BN.get(site, site)}: এই এলাকার জন্য মডেল প্রস্তুত")

available = list_available_horizons(site=site)
if not available:
    available = list_available_horizons()
if not available:
    st.sidebar.error("কোনো মডেল পাওয়া যায়নি। আগে training script চালান।")
    st.sidebar.caption(f"Models dir: {get_models_dir()}")
    st.stop()

default_h = 14 if 14 in available else available[len(available) // 2]
horizon = st.sidebar.slider(
    "📅 কত দিন আগের পূর্বাভাস? (দিন)",
    min_value=min(available),
    max_value=max(available),
    value=default_h,
    step=1,
    help="আগামী কত দিনের মধ্যে বিপজ্জনক তাপ চাপ (DHW ≥ ৪) আসার সম্ভাবনা দেখতে চান।",
)
if horizon not in available:
    horizon = min(available, key=lambda h: abs(h - horizon))
    st.sidebar.warning(f"নিকটতম প্রশিক্ষিত সময় ব্যবহার হচ্ছে: {horizon} দিন")

st.sidebar.caption("মডেল: Logistic (isotonic-calibrated)")

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "📡 তথ্যের উৎস",
    list(MODE_BN.keys()),
    format_func=lambda m: MODE_BN[m],
)

# ONI status
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 জলবায়ু (এল নিনো / ONI)")
if st.sidebar.button("NOAA থেকে ONI হালনাগাদ করুন"):
    try:
        with st.spinner("ONI ডাউনলোড হচ্ছে …"):
            lags = latest_oni_lags(prefer_cache=False)
            st.session_state["oni_lags"] = lags
            st.session_state["oni_status"] = oni_status()
            st.sidebar.success("NOAA CPC থেকে ONI হালনাগাদ হয়েছে।")
    except Exception as e:
        st.sidebar.warning("লাইভ ডাউনলোড হয়নি — সংরক্ষিত তথ্য ব্যবহার হচ্ছে।")
        st.sidebar.caption(str(e)[:180])

if "oni_lags" not in st.session_state:
    try:
        st.session_state["oni_lags"] = latest_oni_lags(prefer_cache=True)
        st.session_state["oni_status"] = oni_status()
    except Exception:
        st.session_state["oni_lags"] = None
        st.session_state["oni_status"] = "ONI not loaded – click Refresh or use dataset values."

st.sidebar.caption(st.session_state.get("oni_status", ""))
if st.session_state.get("oni_lags"):
    lags = st.session_state["oni_lags"]
    st.sidebar.write(
        f"lag0={lags.get('ONI_lag0', 0):+.2f}, "
        f"lag1={lags.get('ONI_lag1', 0):+.2f}, "
        f"lag2={lags.get('ONI_lag2', 0):+.2f}, "
        f"lag3={lags.get('ONI_lag3', 0):+.2f}"
    )

mmm_val = SITE_MMM.get(site)
if mmm_val is not None:
    st.sidebar.caption(f"এলাকার সর্বোচ্চ গড় মাসিক তাপমাত্রা (MMM): **{bn(f'{mmm_val:.2f}')} °C**")

with st.sidebar.expander("Paths / deployment", expanded=False):
    st.caption(f"Data dir: `{get_data_dir()}`")
    st.caption(f"Models dir: `{get_models_dir()}`")
    st.caption(f"Docs dir: `{get_docs_dir()}`")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_site_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_oni_from_row(row: pd.Series) -> dict:
    return {
        "ONI_lag0": float(row.get("ONI_lag0", 0.0)),
        "ONI_lag1": float(row.get("ONI_lag1", 0.0)),
        "ONI_lag2": float(row.get("ONI_lag2", 0.0)),
        "ONI_lag3": float(row.get("ONI_lag3", 0.0)),
    }


def extract_oni_from_df(df):
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    if all(c in df.columns for c in ["ONI_lag0", "ONI_lag1", "ONI_lag2", "ONI_lag3"]):
        return get_oni_from_row(last)
    return None


def resolve_oni(history_df=None) -> dict:
    if st.session_state.get("oni_lags"):
        return st.session_state["oni_lags"]
    from_hist = extract_oni_from_df(history_df) if history_df is not None else None
    if from_hist:
        return from_hist
    return {"ONI_lag0": 0.0, "ONI_lag1": 0.0, "ONI_lag2": 0.0, "ONI_lag3": 0.0}



def plot_dhw_chart(history_df, combined_df=None, horizon=14, site_name=""):
    """DHW trajectory with coloured risk zones (same colours as the alerts)."""
    import altair as alt

    frames = []
    end_note = ""

    if history_df is not None and not history_df.empty:
        hist = history_df.copy()
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date").tail(120)
        if "DHW" in hist.columns:
            hist_plot = hist[["date", "DHW"]].copy()
            hist_plot["series"] = "Observed (research)"
            frames.append(hist_plot)
            end_note = f"{hist['date'].max().date()}"

    if combined_df is not None and not combined_df.empty:
        comb = combined_df.copy()
        comb["date"] = pd.to_datetime(comb["date"])
        if "DHW" in comb.columns:
            if "series_kind" in comb.columns:
                for kind in ["Observed (NOAA CRW)", "Recent (Open-Meteo)", "Forecast (approx.)"]:
                    sub = comb[comb["series_kind"] == kind]
                    if not sub.empty:
                        p = sub[["date", "DHW"]].copy()
                        p["series"] = kind
                        frames.append(p)
            elif "is_forecast" in comb.columns:
                fut = comb[comb["is_forecast"]].copy()
                if not fut.empty:
                    p = fut[["date", "DHW"]].copy()
                    p["series"] = "Forecast (approx.)"
                    frames.append(p)
                bridge = comb[~comb["is_forecast"]].copy()
                if not bridge.empty and frames:
                    last_obs = frames[0]["date"].max()
                    bridge = bridge[bridge["date"] > last_obs]
                    if not bridge.empty:
                        p = bridge[["date", "DHW"]].copy()
                        p["series"] = "Recent (Open-Meteo)"
                        frames.append(p)

    if not frames:
        st.caption("চার্টের জন্য DHW তথ্য পাওয়া যায়নি।")
        return

    plot_df = pd.concat(frames, ignore_index=True).dropna(subset=["DHW"])
    dmin = plot_df["date"].min().date()
    dmax = plot_df["date"].max().date()

    # Bangla legend names
    name_bn = {
        "Observed (research)": "পর্যবেক্ষিত (গবেষণা তথ্য)",
        "Observed (NOAA CRW)": "পর্যবেক্ষিত (NOAA)",
        "Recent (Open-Meteo)": "সাম্প্রতিক (Open-Meteo)",
        "Forecast (approx.)": "পূর্বাভাস (আনুমানিক)",
    }
    plot_df["series"] = plot_df["series"].map(lambda s: name_bn.get(s, s))
    domain = list(name_bn.values())
    range_ = ["#0A6EA8", "#1B9E5A", "#14B8D4", "#E8453C"]

    ymax = max(14.0, float(plot_df["DHW"].max()) * 1.1)
    y_scale = alt.Scale(domain=[0, ymax], nice=False)

    zones = pd.DataFrame({
        "y0": [0, 2, 4, 8, 12],
        "y1": [2, 4, 8, 12, ymax],
        "zone": ["নিরাপদ", "সাবধান", "বিপদ", "মহাবিপদ", "চরম"],
    })
    zone_names = ["নিরাপদ", "সাবধান", "বিপদ", "মহাবিপদ", "চরম"]
    zone_cols = [C_GREEN, C_YELLOW, C_ORANGE, C_RED, C_PURPLE]

    bands = (
        alt.Chart(zones)
        .mark_rect(opacity=0.20)
        .encode(
            y=alt.Y("y0:Q", scale=y_scale, title="তাপ চাপ (DHW, °C-সপ্তাহ)"),
            y2="y1:Q",
            color=alt.Color("zone:N", scale=alt.Scale(domain=zone_names, range=zone_cols),
                            legend=alt.Legend(title="ঝুঁকির স্তর", orient="right")),
        )
    )
    line = (
        alt.Chart(plot_df)
        .mark_line(strokeWidth=3.2)
        .encode(
            x=alt.X("date:T", title="তারিখ"),
            y=alt.Y("DHW:Q", scale=y_scale, title="তাপ চাপ (DHW, °C-সপ্তাহ)"),
            color=alt.Color("series:N", scale=alt.Scale(domain=domain, range=range_),
                            legend=alt.Legend(title="তথ্য", orient="bottom")),
        )
    )
    rules = (
        alt.Chart(pd.DataFrame({"y": [4, 8, 12]}))
        .mark_rule(strokeDash=[5, 5], opacity=0.55, color="#062B4A")
        .encode(y=alt.Y("y:Q", scale=y_scale, title="তাপ চাপ (DHW, °C-সপ্তাহ)"))
    )
    chart = (
        (bands + line + rules)
        .resolve_scale(color="independent")
        .properties(height=520, title=alt.TitleParams(f"তাপ চাপের গতিপথ – {SITE_BN.get(site_name, site_name)} ({dmin} → {dmax})", fontSize=20))
        .configure_axis(labelFontSize=14, titleFontSize=15)
        .configure_legend(labelFontSize=14, titleFontSize=14, symbolSize=200)
        .interactive()
    )
    try:
        st.altair_chart(chart, **_stretch(st.altair_chart))
    except Exception:
        # Safe fallback: simple line chart, so the page never breaks
        simple = plot_df.pivot_table(index="date", columns="series", values="DHW")
        st.line_chart(simple)
    st.caption(
        "রেখা যত উপরে ওঠে, প্রবাল ক্ষয়ের ঝুঁকি তত বাড়ে। "
        "🟢 ০–২ নিরাপদ • 🟡 ২–৪ সাবধান • 🟠 ৪–৮ বিপদ • 🔴 ৮–১২ মহাবিপদ • 🟣 ১২+ চরম"
        + (f"  |  গবেষণা তথ্য: {end_note} পর্যন্ত" if end_note else "")
    )


# Plain-language guidance for each alert level (Bangla first, strong wording)
ALERT_TEXT = {
    0: dict(
        icon="🟢", title="নিরাপদ — প্রবাল ভালো আছে",
        sub="এখন প্রবাল ক্ষয়ের ঝুঁকি কম",
        todo_title="✅ এখন কী করবেন?",
        todo=["স্বাভাবিক পর্যটন চলতে পারে",
              "তবুও প্রবাল ছোঁবেন না, প্রবালের উপর দাঁড়াবেন না",
              "রিফ-সেফ সানস্ক্রিন বা র‍্যাশ গার্ড ব্যবহার করুন",
              "নিয়মিত নজরদারি চালিয়ে যান"],
        accent=C_GREEN,
    ),
    1: dict(
        icon="🟡", title="সাবধান — সাগর গরম হচ্ছে",
        sub="প্রবাল ক্ষয়ের ঝুঁকি বাড়ছে",
        todo_title="⚠️ এখন কী করবেন?",
        todo=["নজরদারি বাড়ান",
              "<b>নতুন করে পর্যটক ও নৌকার ভিড় বাড়াবেন না</b>",
              "গাইড, নৌকা ও রিসোর্ট মালিকদের আগেই জানিয়ে রাখুন",
              "প্রবালে নোঙর ফেলা ও প্রবাল স্পর্শ করা একদম নয়"],
        accent="#C98600",
    ),
    2: dict(
        icon="🟠", title="বিপদ — প্রবাল ক্ষয় শুরু হতে পারে",
        sub="👉 পর্যটক কমান, ভিড় কমান!",
        todo_title="🚫 এখনই যা করতে হবে",
        todo=["<b>পর্যটক কমান</b> — একই জায়গায় ভিড় করবেন না",
              "ডুব ও সার্ফিং  সংখ্যা কমান",
              "প্রবালে নোঙর ফেলা বন্ধ রাখুন",
              "স্থানীয় প্রশাসন, রিসোর্ট ও নৌকা মালিকদের <b>এখনই</b> জানান",
              "জরিপ ও নজরদারি বাড়ান"],
        accent=C_ORANGE,
    ),
    3: dict(
        icon="🔴", title="মহাবিপদ — প্রবাল মরার ঝুঁকি অনেক বেশি",
        sub="👉 পর্যটন সীমিত করুন!",
        todo_title="🛑 জরুরি পদক্ষেপ",
        todo=["<b>পর্যটন সীমিত করুন</b> — প্রয়োজনে কিছু এলাকায় ডুব/স্নরকেলিং বন্ধ",
              "জরুরি সাড়া পরিকল্পনা চালু করুন",
              "প্রবালের অবস্থা প্রতিদিন পর্যবেক্ষণ করুন",
              "প্রশাসন ও পরিবেশকর্মীদের সঙ্গে <b>এখনই</b> যোগাযোগ করুন"],
        accent=C_RED,
    ),
    4: dict(
        icon="🟣", title="চরম জরুরি — ব্যাপক প্রবাল মৃত্যুর আশঙ্কা",
        sub="👉 পর্যটন বন্ধ করুন!",
        todo_title="🆘 সর্বোচ্চ জরুরি অবস্থা",
        todo=["<b>প্রবাল এলাকায় পর্যটন বন্ধ করুন</b>",
              "জরুরি পদক্ষেপ নিন, প্রশাসনকে এখনই জানান",
              "ক্ষতির ছবি ও তথ্য সংরক্ষণ করুন",
              "স্থানীয় মানুষ ও ব্যবসায়ীদের সঙ্গে একসাথে সিদ্ধান্ত নিন"],
        accent=C_PURPLE,
    ),
}


def render_alert_visual(alert):
    """Big colour banner + plain Bangla actions. Level 0–4 → green…purple."""
    t = ALERT_TEXT.get(alert.level, ALERT_TEXT[4])
    html(f"""
    <div class="alert-box alert-{alert.level}">
      <div class="a-icon">{t['icon']}</div>
      <div class="a-title">{t['title']}</div>
      <div class="a-sub">{t['sub']}</div>
      <div class="a-en">{alert.name} — {alert.description}</div>
    </div>
    """)
    items = "".join(f"<li>{x}</li>" for x in t["todo"])
    html(f"""
    <div class="todo" style="border-left:10px solid {t['accent']};">
      <div class="hd" style="color:{t['accent']};">{t['todo_title']}</div>
      <ul style="margin:.2rem 0 0 1.2rem; padding:0;">{items}</ul>
    </div>
    """)


# ---------------------------------------------------------------------------
# Data loading / feature construction
# ---------------------------------------------------------------------------
X_pred = None
current_dhw = None
as_of_date = None
info_msg = ""
forecast_meta = None
history_for_chart = None
combined_for_chart = None
dhw_is_proxy = False

if mode == "Latest from research dataset":
    csv_path = resolve_site_csv(site)
    if csv_path is None or not csv_path.exists():
        st.error(
            f"Feature file not found for **{site}**.\n\n"
            f"Expected one of:\n"
            f"- `{get_data_dir() / SITE_FEATURE_FILENAMES[site]}`\n"
            f"- `{APP_ROOT.parent / SITE_FEATURE_FILENAMES[site]}`\n\n"
            f"Set `CBEW_DATA_DIR` to the folder that contains the CSVs."
        )
        st.stop()

    df = load_site_features(str(csv_path))
    feat_present = [c for c in FEATURE_COLUMNS if c in df.columns]
    recent = df.dropna(subset=feat_present).iloc[[-1]].copy()

    as_of_date = recent["date"].iloc[0]
    current_dhw = float(recent["DHW"].iloc[0])
    X_pred = recent.reindex(columns=FEATURE_COLUMNS).fillna(0.0)
    history_for_chart = df
    info_msg = (
        f"Using the latest complete record from the research dataset "
        f"({site}, {as_of_date.date()})."
    )

elif mode == "Upload recent daily CSV":
    uploaded = st.file_uploader(
        "Upload recent daily observations (CSV)",
        type=["csv"],
        help=(
            "Required columns: date, SST, SSTA, HotSpot, DHW, tmax, tmean, "
            "precipitation, relative_humidity, wind_speed, solar_radiation, cloud_cover."
        ),
    )
    if uploaded is None:
        st.info("পূর্বাভাসের জন্য সাম্প্রতিক দৈনিক তথ্যের একটি CSV ফাইল আপলোড করুন।")
        st.stop()

    raw = pd.read_csv(uploaded)
    raw.columns = [c.strip() for c in raw.columns]
    if "date" not in raw.columns:
        st.error("CSV must contain a 'date' column.")
        st.stop()

    missing_base = [c for c in BASE_COLS if c != "date" and c not in raw.columns]
    if missing_base:
        st.error(f"Missing required columns: {missing_base}")
        st.stop()

    oni_vals = resolve_oni(raw)

    with st.spinner("Building feature vector …"):
        try:
            feat_row = build_feature_matrix(raw, oni_values=oni_vals)
            X_pred = feat_row
            current_dhw = float(raw.sort_values("date").iloc[-1]["DHW"])
            as_of_date = pd.to_datetime(raw["date"]).max()
            history_for_chart = raw.copy()
            history_for_chart["date"] = pd.to_datetime(history_for_chart["date"])
            info_msg = f"Features built from uploaded file ({len(raw)} days, last date {as_of_date.date()})."
        except Exception as e:
            st.error(f"Feature building failed: {e}")
            st.stop()

elif mode == "Forecast from today (Open-Meteo)":
    csv_path = resolve_site_csv(site)
    if csv_path is None or not csv_path.exists():
        st.error(f"Historical feature file not found for **{site}**.")
        st.stop()

    hist_full = load_site_features(str(csv_path))
    keep = [c for c in BASE_COLS + ["ONI_lag0", "ONI_lag1", "ONI_lag2", "ONI_lag3"] if c in hist_full.columns]
    history = hist_full[keep].tail(120).copy()
    oni_vals = resolve_oni(history)

    fkey = f"forecast::{site}::{horizon}::{pd.Timestamp.today().date()}"
    run_forecast = st.button("🌊 সর্বশেষ আবহাওয়া-পূর্বাভাস আনুন ও ঝুঁকি দেখুন", type="primary", key="run_fc")

    if run_forecast:
        with st.spinner("🌊 আবহাওয়া ও সাগরের তথ্য আনা হচ্ছে… (ইন্টারনেট ধীর হলে ১ মিনিট পর্যন্ত লাগতে পারে)"):
            try:
                st.session_state[fkey] = build_forecast_frame(
                    site=site, days=horizon, history_df=history, mmm=SITE_MMM.get(site),
                )
            except Exception as e:
                st.error(f"পূর্বাভাসের তথ্য আনা যায়নি: {e}")
                st.stop()

    if fkey not in st.session_state:
        st.info(
            f"👆 উপরের বোতামে ক্লিক করুন — **{SITE_BN.get(site, site)}**-এর জন্য আগামী "
            f"{bn(horizon)} দিনের ঝুঁকি দেখানো হবে।"
        )
        lat_, lon_ = SITE_COORDS[site]
        mmm_ = SITE_MMM.get(site)
        mmm_txt = bn(f"{mmm_:.2f}") if mmm_ is not None else "—"
        st.caption(
            f"স্থানাঙ্ক: {bn(f'{lat_:.4f}')}°N, {bn(f'{lon_:.4f}')}°E  |  MMM: {mmm_txt} °C"
        )
        plot_dhw_chart(hist_full, None, horizon, site_name=site)
        st.stop()

    combined, forecast_meta = st.session_state[fkey]
    try:
        today = pd.Timestamp.today().normalize()
        comb = combined.copy()
        comb["date"] = pd.to_datetime(comb["date"])
        asof_mask = comb["date"] <= today
        asof_df = comb.loc[asof_mask] if asof_mask.any() else comb
        feat_row = build_feature_matrix(asof_df, oni_values=oni_vals)
        X_pred = feat_row
        if "DHW" in asof_df.columns and asof_df["DHW"].notna().any():
            current_dhw = float(asof_df.dropna(subset=["DHW"]).iloc[-1]["DHW"])
        else:
            current_dhw = float(history.iloc[-1]["DHW"])
        as_of_date = today
        history_for_chart = hist_full
        combined_for_chart = combined
        dhw_is_proxy = True
        re_end = forecast_meta.get("research_end", "?")
        info_msg = (
            f"{site}: research archive to {re_end}; "
            f"Open-Meteo fill to {today.date()}; forecast next {horizon} day(s)."
        )
    except Exception as e:
        st.error(f"পূর্বাভাস তৈরি করা যায়নি: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if X_pred is None:
    st.warning("No prediction data available yet.")
    st.stop()

try:
    bundle = get_bundle(horizon, site)
except Exception as e:
    st.error(f"Could not load model for horizon {horizon} d: {e}")
    st.stop()

prob = float(bundle.predict_proba(X_pred, calibrated=True)[0])
alert = map_alert(
    probability=prob,
    current_dhw=current_dhw,
    prob_threshold_warning=DEFAULT_WARNING_THR,
    prob_threshold_alert=DEFAULT_ALERT_THR,
    dhw_is_proxy=dhw_is_proxy,
)

# ---------------------------------------------------------------------------
# Main results display (Bangla first, strong plain words)
# ---------------------------------------------------------------------------
html(f'<div class="section-header">📊 আজকের অবস্থা — {SITE_BN.get(site, site)}</div>')

if info_msg:
    st.info(info_msg)

# 1. Big colour alert + what to do
render_alert_visual(alert)

# 2. Risk bar
ALERT_COLOURS = {0: C_GREEN, 1: "#C98600", 2: C_ORANGE, 3: C_RED, 4: C_PURPLE}
risk_colour = ALERT_COLOURS.get(alert.level, C_PURPLE)
marker_pos = min(max(prob * 100, 0.5), 99.5)
html(f"""
<div class="riskwrap">
  <div class="riskhead">
    <div class="lbl">🪸 আগামী {bn(horizon)} দিনে প্রবাল ক্ষয়ের ঝুঁকি</div>
    <div class="val" style="color:{risk_colour};">{bn(f'{prob:.0%}')}</div>
  </div>
  <div class="riskbar"><div class="riskmark" style="left:calc({marker_pos}% - 3px);"></div></div>
  <div class="riskscale"><span>০% কম ঝুঁকি</span><span>৫০%</span><span>১০০% অতি উচ্চ ঝুঁকি</span></div>
  <div style="font-size:.98rem; color:#4B6577; margin-top:.5rem;">
    ℹ️ এটি পূর্বাভাস — প্রবাল ক্ষয় হবেই এমন নিশ্চয়তা নয়। স্থানীয় পর্যবেক্ষণের সঙ্গে মিলিয়ে সিদ্ধান্ত নিন।
  </div>
</div>
""")

st.markdown("<br>", unsafe_allow_html=True)

# 3. Metric cards
m1, m2, m3, m4 = st.columns(4)
with m1:
    html(f'<div class="metric-card m-coral"><div class="metric-value">{bn(horizon)} দিন</div>'
         f'<div class="metric-label">কত দিন আগের পূর্বাভাস</div></div>')
with m2:
    html(f'<div class="metric-card m-sun"><div class="metric-value">{bn(f"{current_dhw:.2f}")}</div>'
         f'<div class="metric-label">বর্তমান তাপ চাপ (DHW)<br>৪ বা বেশি = ঝুঁকিপূর্ণ</div></div>')
with m3:
    d_txt = bn(as_of_date.date()) if as_of_date is not None else "—"
    html(f'<div class="metric-card"><div class="metric-value" style="font-size:1.55rem;">{d_txt}</div>'
         f'<div class="metric-label">তথ্যের তারিখ</div></div>')
with m4:
    status = site_model_status(site)
    ok = status == "in-domain"
    html(f'<div class="metric-card {"m-green" if ok else "m-sun"}">'
         f'<div class="metric-value" style="font-size:1.3rem;">{"✅ প্রস্তুত" if ok else "⚠️ পরীক্ষামূলক"}</div>'
         f'<div class="metric-label">{"এই এলাকার জন্য মডেল প্রশিক্ষিত" if ok else "অন্য এলাকার মডেল ব্যবহৃত"}</div></div>')

# 4. How to read
html("""
<div class="card card-blue" style="margin-top:1.4rem;">
  <h3>🔎 ফলাফল কীভাবে পড়বেন?</h3>
  <div>মডেল আজকের সাগরের তাপমাত্রা, জমে থাকা তাপ (DHW) ও জলবায়ুর ইঙ্গিত দেখে বলে —
  আগামী কয়েক দিনে বিপজ্জনক তাপ চাপ (DHW ≥ ৪) আসার সম্ভাবনা কতটা।</div>
  <ul>
    <li><b style="color:#1B9E5A;">🟢 সবুজ</b> → ভালো আছে, স্বাভাবিক নজরদারি</li>
    <li><b style="color:#C98600;">🟡 হলুদ</b> → সাবধান, ভিড় বাড়াবেন না</li>
    <li><b style="color:#F57C00;">🟠 কমলা</b> → বিপদ, <b>পর্যটক কমান</b></li>
    <li><b style="color:#D32F2F;">🔴 লাল</b> → মহাবিপদ, <b>পর্যটন সীমিত করুন</b></li>
    <li><b style="color:#7B1FA2;">🟣 বেগুনি</b> → চরম জরুরি, <b>পর্যটন বন্ধ করুন</b></li>
  </ul>
</div>
""")

# 5. Image strip
html('<div class="section-header">🌊 যা বাঁচাতে চাই</div>')
img_a, img_b, img_c = st.columns(3)
for col, img, cap in ((img_a, IMG_HEALTHY, "🪸 সুস্থ প্রবাল বাগান"), (img_b, IMG_DIVER, "🤿 ডুবুরি ও রঙিন মাছ"), (img_c, IMG_REEF, "🐠 রঙিন সমুদ্রজীবন")):
    with col:
        html(f'<div class="img-card"><img class="coral-img" src="{img}" alt="coral"><div class="img-caption">{cap}</div></div>')

# 6. Chart
html('<div class="section-header">📈 তাপ চাপের গতিপথ — ঝুঁকি কোন দিকে যাচ্ছে?</div>')
plot_dhw_chart(history_for_chart, combined_for_chart, horizon, site_name=site)

# 7. Reliability
html('<div class="section-header">🔍 মডেল কতটা নির্ভরযোগ্য?</div>')
metrics = get_horizon_metrics(horizon, model="logistic", site=site)
if metrics and metrics.get("available"):
    label_en = reliability_indicator_label(metrics.get("cal_brier"))
    label_bn = {"good": "ভালো ✅", "moderate": "মাঝারি 🟡", "limited": "সীমিত ⚠️", "unknown": "অজানা"}.get(label_en, label_en)
    html(f"""
    <div class="card card-green" style="padding:1rem 1.4rem;">
      <b>নির্ভরযোগ্যতা: {label_bn}</b> — অতীত পরীক্ষার তথ্যের ভিত্তিতে।
    </div>
    """)
    with st.expander("🔬 কারিগরি বিবরণ (বিজ্ঞানীদের জন্য)"):
        st.markdown(reliability_caption(
            horizon=horizon, model="logistic", probability=prob, site=site,
            in_domain=(not is_transfer_site(site)),
        ))
else:
    st.caption("এই সময়ের জন্য বিস্তারিত নির্ভরযোগ্যতার তথ্য নেই।")

# 8. Supporting detail
with st.expander("Model & data details", expanded=False):
    st.write(
        f"- Horizon: **{horizon} days**  \n"
        f"- Model: **logistic** (isotonic calibrated)  \n"
        f"- Training site: **{TRAINING_SITE}**  \n"
        f"- Selected site status: **{site_model_status(site)}**  \n"
        f"- Site MMM: **{SITE_MMM.get(site, 'n/a')} °C**  \n"
        f"- Alert thresholds: Warning ≥ {DEFAULT_WARNING_THR:.0%} probability or DHW ≥ 2; "
        f"Alert Level 1 ≥ {DEFAULT_ALERT_THR:.0%} probability or DHW ≥ 4; "
        f"Level 2 DHW ≥ 8; Extreme DHW ≥ 12 (observed DHW only)."
    )
    if is_transfer_site(site):
        st.warning(
            f"This prediction is a **transfer / experimental** application of the "
            f"{TRAINING_SITE} models. Domain shift is expected."
        )

if forecast_meta and forecast_meta.get("warnings"):
    with st.expander("🛰️ তথ্য সংগ্রহের বিস্তারিত (কোন উৎস থেকে কী এসেছে)", expanded=False):
        for w in forecast_meta["warnings"]:
            st.caption("• " + str(w))

with st.expander("Feature vector used for this prediction", expanded=False):
    st.dataframe(X_pred.T.rename(columns={X_pred.index[0]: "value"}), height=300)

with st.expander("Reproducibility artefacts", expanded=False):
    st.markdown(
        """
- **Model card**: `docs/MODEL_CARD.md`
- **Data card**: `docs/DATA_CARD.md`
- **Calibration notes**: `docs/calibration_notes.md`
- **Reliability table**: `docs/reliability_summary.csv`
- **Unit tests**: `tests/`
- **Pinned dependencies**: `requirements.txt`
        """
    )

html("""
<div class="footer">
  গবেষণা প্রোটোটাইপ — NOAA Coral Reef Watch-এর পরিপূরক, বিকল্প নয়।<br>
  ঝুঁকির শতকরা হার একটি পূর্বাভাস; সবসময় স্থানীয় পর্যবেক্ষণের সঙ্গে মিলিয়ে দেখুন।
</div>
""")
