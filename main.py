import streamlit as st
import json
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

try:
    import google.generativeai as genai
    AI_AVAILABLE = True
    if "GENAI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GENAI_API_KEY"])
    else:
        AI_AVAILABLE = False
        st.error("🔑 API Key Missing: Please add 'GENAI_API_KEY' to your Streamlit Cloud Secrets.")
except ImportError:
    AI_AVAILABLE = False

st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon="🧬")

# ==========================================
# REFERENCE DATABASE FOR ENVIRONMENTAL BENCHMARKS
# ==========================================
REFERENCE_BENCHMARKS = {
    "clinical": {
        "Aminoglycoside": 0.72, "Beta-lactam": 0.85, "Fluoroquinolone": 0.68,
        "Carbapenem": 0.45, "Tetracycline": 0.55, "Macrolide": 0.60,
        "Sulfonamide": 0.50, "Glycopeptide": 0.30, "Colistin": 0.20,
        "Trimethoprim": 0.58, "Chloramphenicol": 0.42, "Rifampicin": 0.35,
        "avg_mri": 0.42, "avg_ari": 0.18, "avg_genes": 85
    },
    "agricultural": {
        "Aminoglycoside": 0.65, "Beta-lactam": 0.55, "Fluoroquinolone": 0.40,
        "Carbapenem": 0.15, "Tetracycline": 0.80, "Macrolide": 0.70,
        "Sulfonamide": 0.75, "Glycopeptide": 0.10, "Colistin": 0.55,
        "Trimethoprim": 0.45, "Chloramphenicol": 0.60, "Rifampicin": 0.20,
        "avg_mri": 0.32, "avg_ari": 0.12, "avg_genes": 52
    },
    "environmental": {
        "Aminoglycoside": 0.40, "Beta-lactam": 0.35, "Fluoroquinolone": 0.28,
        "Carbapenem": 0.10, "Tetracycline": 0.45, "Macrolide": 0.38,
        "Sulfonamide": 0.42, "Glycopeptide": 0.08, "Colistin": 0.12,
        "Trimethoprim": 0.30, "Chloramphenicol": 0.35, "Rifampicin": 0.15,
        "avg_mri": 0.18, "avg_ari": 0.06, "avg_genes": 28
    },
    "wastewater": {
        "Aminoglycoside": 0.58, "Beta-lactam": 0.62, "Fluoroquinolone": 0.50,
        "Carbapenem": 0.28, "Tetracycline": 0.70, "Macrolide": 0.65,
        "Sulfonamide": 0.68, "Glycopeptide": 0.18, "Colistin": 0.35,
        "Trimethoprim": 0.55, "Chloramphenicol": 0.50, "Rifampicin": 0.25,
        "avg_mri": 0.35, "avg_ari": 0.14, "avg_genes": 65
    },
    "food_production": {
        "Aminoglycoside": 0.50, "Beta-lactam": 0.48, "Fluoroquinolone": 0.35,
        "Carbapenem": 0.12, "Tetracycline": 0.75, "Macrolide": 0.60,
        "Sulfonamide": 0.65, "Glycopeptide": 0.08, "Colistin": 0.45,
        "Trimethoprim": 0.40, "Chloramphenicol": 0.55, "Rifampicin": 0.18,
        "avg_mri": 0.28, "avg_ari": 0.10, "avg_genes": 42
    }
}

ORIGIN_LABELS = {
    "🏥 Clinical / Hospital": "clinical",
    "🌾 Agricultural / Livestock": "agricultural",
    "🌿 Environmental / Soil": "environmental",
    "💧 Wastewater / Sewage": "wastewater",
    "🍖 Food Production": "food_production"
}

DRUG_CLASS_KEYWORDS = {
    "Aminoglycoside": ["aminoglycoside", "amikacin", "gentamicin", "tobramycin", "streptomycin"],
    "Beta-lactam": ["beta-lactam", "penicillin", "cephalosporin", "ampicillin", "methicillin"],
    "Fluoroquinolone": ["fluoroquinolone", "quinolone", "ciprofloxacin", "levofloxacin"],
    "Carbapenem": ["carbapenem", "imipenem", "meropenem", "ertapenem"],
    "Tetracycline": ["tetracycline", "doxycycline", "minocycline"],
    "Macrolide": ["macrolide", "erythromycin", "azithromycin", "clarithromycin"],
    "Sulfonamide": ["sulfonamide", "sulfamethoxazole", "trimethoprim-sulfamethoxazole"],
    "Glycopeptide": ["glycopeptide", "vancomycin", "teicoplanin"],
    "Colistin": ["colistin", "polymyxin"],
    "Trimethoprim": ["trimethoprim", "diaminopyrimidine"],
    "Chloramphenicol": ["chloramphenicol", "phenicol"],
    "Rifampicin": ["rifampicin", "rifamycin", "rifampin"]
}

def map_drug_to_standard_class(drug_name):
    dl = drug_name.lower()
    for std_class, keywords in DRUG_CLASS_KEYWORDS.items():
        if any(kw in dl for kw in keywords):
            return std_class
    return None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');

* { box-sizing: border-box; }

.main { background-color: #0e1117; }
h1, h2, h3 { color: #ffffff; }

/* ── MAIN HEADER ── */
.main-header {
  background: linear-gradient(120deg, #020b18 0%, #0c1f3f 30%, #06304a 60%, #0f2a1a 100%);
  border: 1px solid rgba(0,212,255,0.2);
  padding: 2rem 1.5rem;
  border-radius: 15px;
  color: white;
  text-align: center;
  margin-bottom: 1.5rem;
  box-shadow: 0 10px 40px rgba(0,212,255,0.1);
  position: relative;
  overflow: hidden;
}
.main-header::before {
  content:'';
  position:absolute; top:0;left:0;right:0; height:3px;
  background: linear-gradient(90deg, #00d4ff, #7c3aed, #10b981, #00d4ff);
  background-size:300% auto;
  animation: headerBar 4s linear infinite;
}
@keyframes headerBar { 0%{background-position:0% center} 100%{background-position:300% center} }
.main-header h1 { margin:0; font-size: clamp(1.4rem, 5vw, 2.8rem); }
.main-header p { font-size: clamp(0.85rem, 3vw, 1.3rem); opacity:0.9; margin-top:10px; }

/* ── SECTION EXPLANATION BOX ── */
.section-explainer {
  background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(124,58,237,0.05));
  border: 1px solid rgba(0,212,255,0.15);
  border-left: 4px solid #00d4ff;
  padding: clamp(12px,3vw,18px) clamp(14px,3vw,20px);
  border-radius: 10px;
  margin-bottom: 18px;
  font-size: clamp(0.82rem,2.5vw,0.92rem);
  color: #94a3b8;
  line-height: 1.7;
}
.section-explainer .ex-title {
  color: #00d4ff;
  font-size: clamp(0.72rem,2vw,0.78rem);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}

/* ── WELCOME HERO ── */
.welcome-hero {
  background: linear-gradient(135deg, rgba(30,58,138,0.8) 0%, rgba(15,23,42,0.95) 100%);
  border-radius: 15px;
  padding: clamp(20px, 5vw, 45px);
  border: 1px solid rgba(59,130,246,0.4);
  margin-bottom: 30px;
  text-align: center;
  box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}
.welcome-hero h2 { color: #60a5fa !important; font-size: clamp(1.4rem, 5vw, 2.8rem) !important; font-weight: 800 !important; margin-bottom: 12px !important; }
.welcome-hero p { color: #e2e8f0 !important; font-size: clamp(0.9rem, 3vw, 1.25rem) !important; }

/* ── ALERT BANNER ── */
.alert-banner {
  background: rgba(239,68,68,0.2);
  border: 1px solid #ef4444;
  padding: 12px 15px;
  border-radius: 8px;
  color: #f87171;
  font-weight: bold;
  text-align: center;
  margin-bottom: 20px;
  animation: pulse 2s infinite;
  font-size: clamp(0.8rem, 3vw, 1rem);
  word-break: break-word;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }

/* ── METRIC CARDS ── */
.metric-card {
  background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(6,30,50,0.9));
  border: 1px solid rgba(0,212,255,0.15);
  padding: clamp(12px, 3vw, 1.5rem);
  border-radius: 12px;
  text-align: center;
  transition: all 0.3s ease;
}
.metric-card:hover { border-color: rgba(0,212,255,0.4); box-shadow: 0 8px 25px rgba(0,212,255,0.1); }
.metric-label { color: #00d4ff; font-size: clamp(0.65rem, 2vw, 0.75rem); font-weight: bold; text-transform: uppercase; letter-spacing: 0.1em; }
.metric-value { color: #ffffff; font-size: clamp(1.1rem, 4vw, 1.8rem); font-weight: 700; margin-top: 0.3rem; }

/* ── REPORT CARD ── */
.report-card {
  background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(6,30,50,0.9));
  border-left: 5px solid #00d4ff;
  padding: clamp(14px, 3vw, 20px);
  border-radius: 10px;
  margin-bottom: 25px;
}
.report-header { color: #00d4ff; font-weight: bold; text-transform: uppercase; font-size: 0.9rem; margin-bottom: 15px; border-bottom: 1px solid #334155; padding-bottom: 5px; }
.report-row { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px; padding: 8px 0; border-bottom: 1px solid #2d3748; }
.report-label { color: #94a3b8; font-weight: 500; font-size: clamp(0.8rem, 2.5vw, 1rem); }
.report-value { color: #ffffff; font-weight: 600; font-size: clamp(0.8rem, 2.5vw, 1rem); text-align: right; }

/* ── AI BADGE ── */
.ai-badge { background: rgba(0,212,255,0.1); color: #00d4ff; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid rgba(0,212,255,0.3); }

/* ── MATH CARD ── */
.math-card {
  background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(6,30,50,0.9));
  border: 1px solid rgba(0,212,255,0.15);
  padding: clamp(14px, 3vw, 25px);
  border-radius: 12px;
  margin-bottom: 20px;
}
.math-card-header { color: #00d4ff; font-size: 0.85rem; font-weight: bold; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 15px; }
.annotation-box { background: rgba(15,23,42,0.6); border: 1px dashed #334155; border-radius: 8px; padding: 12px; margin-top: 10px; font-size: clamp(0.78rem, 2.5vw, 0.85rem); color: #94a3b8; }
.annotation-item { margin-bottom: 4px; }
.annotation-key { color: #60a5fa; font-weight: bold; font-family: monospace; }

/* ── SUSCEPTIBILITY CARD ── */
.susceptibility-card { background: rgba(16,185,129,0.1); border: 1px solid #10b981; border-radius: 10px; padding: 15px; color: #10b981; font-weight: 600; word-break: break-word; }

/* ── REASONING BOX ── */
.reasoning-box {
  background: rgba(6,30,50,0.85);
  border-radius: 12px;
  border-left: 5px solid #00d4ff;
  padding: clamp(14px, 3vw, 20px);
  color: #ffffff;
  line-height: 1.7;
  font-size: clamp(0.9rem, 2.5vw, 1.05rem);
}

/* ── ORIGIN BADGE ── */
.origin-badge {
  display: inline-block;
  padding: 5px 14px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.origin-clinical    { background: rgba(239,68,68,0.15);   border: 1px solid #ef4444;   color: #f87171; }
.origin-agricultural{ background: rgba(16,185,129,0.15);  border: 1px solid #10b981;   color: #34d399; }
.origin-environmental{background: rgba(59,130,246,0.15);  border: 1px solid #3b82f6;   color: #60a5fa; }
.origin-wastewater  { background: rgba(245,158,11,0.15);  border: 1px solid #f59e0b;   color: #fbbf24; }
.origin-food_production{background:rgba(167,139,250,0.15);border: 1px solid #a78bfa;   color: #c4b5fd; }

/* ── HEATMAP INSIGHT CARD ── */
.insight-card {
  background: rgba(15,23,42,0.9);
  border: 1px solid rgba(0,212,255,0.12);
  border-radius: 12px;
  padding: clamp(14px,3vw,20px);
  margin-bottom: 14px;
}
.insight-card .ins-title { color: #00d4ff; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }
.insight-card .ins-body  { color: #94a3b8; font-size: 0.87rem; line-height: 1.65; }

/* ── DATA FRAME ── */
[data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 10px; overflow: hidden; }

/* ── ANIMATIONS ── */
@keyframes fadeInDown { from{opacity:0;transform:translateY(-40px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeInUp   { from{opacity:0;transform:translateY(40px)}  to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn     { from{opacity:0} to{opacity:1} }
@keyframes shimmer    { 0%{background-position:-200% center} 100%{background-position:200% center} }
@keyframes float      { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }

/* ═══════════════════════════════════════
   SPLASH SCREEN
═══════════════════════════════════════ */
.splash-root {
  min-height: 100vh;
  background: #020b18;
  background-image:
    radial-gradient(ellipse 80% 60% at 10% 20%, rgba(0,212,255,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 60% 80% at 90% 80%, rgba(124,58,237,0.1) 0%, transparent 60%);
  font-family: 'Rajdhani', sans-serif;
  overflow-x: hidden;
}
.splash-root::before {
  content:'';
  position:fixed; inset:0;
  background-image: linear-gradient(rgba(0,212,255,0.03) 1px,transparent 1px), linear-gradient(90deg,rgba(0,212,255,0.03) 1px,transparent 1px);
  background-size:60px 60px;
  animation: gridDrift 20s linear infinite;
  pointer-events:none; z-index:0;
}
@keyframes gridDrift { 0%{transform:translate(0,0)} 100%{transform:translate(60px,60px)} }
.splash-content { position:relative; z-index:2; }

.sp-hero { padding: clamp(40px,8vw,80px) clamp(16px,5vw,60px) clamp(30px,5vw,60px); text-align:center; }
.sp-dna-ring { display:inline-block; font-size:clamp(3rem,10vw,5.5rem); animation:dnaFloat 4s ease-in-out infinite,dnaGlow 3s ease-in-out infinite; margin-bottom:20px; }
@keyframes dnaFloat { 0%,100%{transform:translateY(0) scale(1)} 50%{transform:translateY(-15px) scale(1.05)} }
@keyframes dnaGlow  { 0%,100%{filter:drop-shadow(0 0 20px rgba(0,212,255,0.5))} 50%{filter:drop-shadow(0 0 50px rgba(124,58,237,0.8)) drop-shadow(0 0 20px rgba(0,212,255,0.6))} }

.sp-title {
  font-family:'Orbitron',monospace !important;
  font-size: clamp(1.8rem,8vw,4.5rem) !important;
  font-weight:900 !important; letter-spacing:clamp(1px,2vw,4px);
  background: linear-gradient(135deg,#00d4ff 0%,#7c3aed 40%,#10b981 70%,#00d4ff 100%);
  background-size:300% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  animation: titleShimmer 5s linear infinite, titleReveal 1.2s ease forwards;
  margin-bottom:8px !important; opacity:0;
}
@keyframes titleShimmer { 0%{background-position:0% center} 100%{background-position:300% center} }
@keyframes titleReveal  { 0%{opacity:0;transform:translateY(-30px)} 100%{opacity:1;transform:translateY(0)} }

.sp-tagline { font-family:'Orbitron',monospace; font-size:clamp(0.6rem,2.5vw,0.85rem) !important; letter-spacing:clamp(2px,3vw,6px); color:#00d4ff !important; text-transform:uppercase; margin-bottom:20px !important; animation:fadeInUp 1.5s ease forwards; opacity:0; animation-delay:0.6s; }
.sp-desc { color:#94a3b8 !important; font-size:clamp(0.9rem,3vw,1.15rem) !important; line-height:1.8; max-width:700px; margin:0 auto 35px auto !important; animation:fadeInUp 1.8s ease forwards; opacity:0; animation-delay:0.9s; }

.sp-stats { display:flex; justify-content:center; gap:clamp(8px,2vw,20px); flex-wrap:wrap; margin-bottom:50px; animation:fadeInUp 2s ease forwards; opacity:0; animation-delay:1.2s; }
.sp-stat { background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.2); border-radius:16px; padding:clamp(14px,3vw,22px) clamp(16px,4vw,32px); min-width:clamp(70px,20vw,120px); transition:all 0.3s ease; }
.sp-stat:hover { transform:translateY(-5px); border-color:rgba(0,212,255,0.5); }
.sp-stat-num { font-family:'Orbitron',monospace; font-size:clamp(1.4rem,5vw,2.2rem); font-weight:900; color:#00d4ff; display:block; }
.sp-stat-lbl { font-size:clamp(0.6rem,2vw,0.7rem); color:#64748b; letter-spacing:2px; text-transform:uppercase; }

.sp-section-title { font-family:'Orbitron',monospace !important; font-size:clamp(1.2rem,4vw,1.8rem) !important; font-weight:700 !important; color:#f1f5f9 !important; text-align:center; margin-bottom:10px !important; letter-spacing:2px; }
.sp-divider { width:80px; height:3px; background:linear-gradient(90deg,#00d4ff,#7c3aed); margin:0 auto 30px auto; border-radius:3px; }

.sp-feat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(260px,100%),1fr)); gap:16px; padding:0 clamp(16px,4vw,60px) 50px; }
.sp-feat-card { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:clamp(18px,3vw,28px) clamp(14px,3vw,22px); transition:all 0.35s ease; }
.sp-feat-card:hover { transform:translateY(-6px); border-color:rgba(0,212,255,0.25); }
.sp-feat-icon { font-size:clamp(1.6rem,4vw,2.2rem); display:block; margin-bottom:12px; }
.sp-feat-title { color:#00d4ff !important; font-size:clamp(0.75rem,2.5vw,0.85rem); font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
.sp-feat-desc { color:#64748b; font-size:clamp(0.82rem,2.5vw,0.88rem); line-height:1.65; }

.sp-unique-section { padding:0 clamp(16px,4vw,60px) 60px; }
.sp-compare { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr)); gap:20px; margin-bottom:30px; }
.sp-comp-card { border-radius:18px; padding:clamp(18px,3vw,30px); }
.sp-comp-old { background:rgba(239,68,68,0.04); border:1px solid rgba(239,68,68,0.18); }
.sp-comp-new { background:rgba(0,212,255,0.04); border:1px solid rgba(0,212,255,0.22); position:relative; }
.sp-comp-new::before { content:'★ EXCLUSIVE'; position:absolute; top:-13px; right:20px; background:linear-gradient(90deg,#00d4ff,#10b981); color:#020b18; font-size:0.65rem; font-weight:800; padding:4px 14px; border-radius:20px; letter-spacing:1.5px; font-family:'Orbitron',monospace; }
.sp-comp-title { font-size:clamp(0.9rem,3vw,1rem); font-weight:700; margin-bottom:16px; padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.06); }
.sp-comp-old .sp-comp-title { color:#f87171; }
.sp-comp-new .sp-comp-title { color:#00d4ff; }
.sp-comp-item { display:flex; gap:10px; margin-bottom:12px; color:#94a3b8; font-size:clamp(0.82rem,2.5vw,0.88rem); line-height:1.5; }
.sp-chk { flex-shrink:0; }

.sp-pillars { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(200px,100%),1fr)); gap:16px; }
.sp-pillar { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:clamp(18px,3vw,26px) clamp(14px,3vw,20px); text-align:center; transition:all 0.3s ease; position:relative; overflow:hidden; }
.sp-pillar::before { content:''; position:absolute; top:0;left:0;right:0; height:3px; }
.sp-pillar.c1::before{background:linear-gradient(90deg,#00d4ff,#10b981)} .sp-pillar.c2::before{background:linear-gradient(90deg,#7c3aed,#a78bfa)} .sp-pillar.c3::before{background:linear-gradient(90deg,#f59e0b,#fbbf24)} .sp-pillar.c4::before{background:linear-gradient(90deg,#f43f5e,#fb7185)} .sp-pillar.c5::before{background:linear-gradient(90deg,#10b981,#34d399)} .sp-pillar.c6::before{background:linear-gradient(90deg,#06b6d4,#67e8f9)} .sp-pillar.c7::before{background:linear-gradient(90deg,#f59e0b,#10b981)}
.sp-pillar:hover{transform:translateY(-6px);border-color:rgba(0,212,255,0.3)}
.sp-pillar-icon{font-size:clamp(1.6rem,4vw,2.3rem);display:block;margin-bottom:10px}
.sp-pillar-title{color:#f1f5f9 !important;font-size:clamp(0.75rem,2.5vw,0.82rem);font-weight:700;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px}
.sp-pillar-desc{color:#475569;font-size:clamp(0.78rem,2.5vw,0.82rem);line-height:1.6}

.sp-team-section{padding:0 clamp(16px,4vw,60px) 60px}
.sp-team-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:16px;margin-bottom:30px}
.sp-team-card{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:18px;padding:clamp(20px,3vw,32px) clamp(14px,3vw,22px);text-align:center;transition:all 0.35s ease;position:relative;overflow:hidden}
.sp-team-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#00d4ff,#7c3aed,#10b981);background-size:200% auto;animation:titleShimmer 4s linear infinite}
.sp-team-card:hover{transform:translateY(-8px);border-color:rgba(0,212,255,0.3)}
.sp-avatar{width:clamp(56px,12vw,80px);height:clamp(56px,12vw,80px);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:clamp(1.4rem,5vw,2.2rem);margin:0 auto 14px auto;border:2px solid rgba(0,212,255,0.3)}
.sp-tname{color:#f1f5f9 !important;font-size:clamp(0.95rem,3vw,1.1rem);font-weight:700;margin-bottom:4px !important}
.sp-trole{color:#00d4ff !important;font-size:clamp(0.62rem,2vw,0.72rem);text-transform:uppercase;letter-spacing:2px;margin-bottom:10px !important;font-family:'Orbitron',monospace}
.sp-tdesc{color:#475569;font-size:clamp(0.78rem,2.5vw,0.82rem);line-height:1.6;margin-bottom:12px}
.sp-ttags{display:flex;flex-wrap:wrap;gap:5px;justify-content:center}
.sp-ttag{background:rgba(0,212,255,0.07);border:1px solid rgba(0,212,255,0.2);color:#67e8f9;padding:3px 9px;border-radius:20px;font-size:clamp(0.62rem,2vw,0.68rem)}

.sp-mission-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:16px;margin-top:24px}
.sp-mission-card{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:clamp(18px,3vw,28px) clamp(14px,3vw,22px);text-align:center}
.sp-mission-title{color:#00d4ff !important;font-size:clamp(0.72rem,2.5vw,0.8rem);font-weight:700;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px !important;font-family:'Orbitron',monospace}
.sp-mission-text{color:#64748b;font-size:clamp(0.82rem,2.5vw,0.85rem);line-height:1.7}

.sp-cta-section{padding:clamp(30px,5vw,60px) clamp(16px,5vw,60px);text-align:center;background:linear-gradient(180deg,transparent,rgba(0,212,255,0.04),transparent);border-top:1px solid rgba(0,212,255,0.08)}
.sp-cta-label{font-family:'Orbitron',monospace;font-size:clamp(0.65rem,2vw,0.8rem);letter-spacing:5px;text-transform:uppercase;color:#475569 !important;margin-bottom:16px !important}
.sp-cta-headline{font-size:clamp(1.4rem,5vw,2.4rem) !important;font-weight:700 !important;color:#f1f5f9 !important;margin-bottom:10px !important;line-height:1.3}
.sp-cta-sub{color:#64748b !important;font-size:clamp(0.85rem,3vw,1rem) !important;margin-bottom:30px !important}

@media (max-width: 640px) {
  .report-row { flex-direction: column; gap: 2px; }
  .report-value { text-align: left; }
  .sp-comp-new::before { font-size: 0.55rem; padding: 3px 10px; right: 10px; }
  .sp-stats { justify-content: center; }
  .sp-stat { flex: 1 1 calc(50% - 8px); min-width: 0; }
}
@media (max-width: 480px) {
  .sp-title { letter-spacing: 1px !important; }
  .sp-tagline { letter-spacing: 2px !important; }
}
@media (max-width: 768px) {
  [data-baseweb="tab"] { padding: 8px 6px !important; }
  [data-baseweb="tab"] p { font-size: 0.75rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER: Section Explainer Component
# ==========================================
def section_explainer(icon, title, body, position="above"):
    html = f"""
    <div class="section-explainer">
      <div class="ex-title">{icon} {title}</div>
      {body}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ==========================================
# 2. CORE BACKEND FUNCTIONS
# ==========================================
@st.cache_data
def get_bacteria_info(file_name):
    name = file_name.lower()
    info = {"gram": "Unknown", "disease": "Various opportunistic infections"}
    if any(k in name for k in ["ecoli","escherichia","shigella"]):
        info = {"gram": "Negative (-)", "disease": "Gastroenteritis, UTI, Sepsis"}
    elif "salmonella" in name:
        info = {"gram": "Negative (-)", "disease": "Salmonellosis, Typhoid Fever"}
    elif any(k in name for k in ["klebsiella","enterobacter","citrobacter","serratia"]):
        info = {"gram": "Negative (-)", "disease": "Pneumonia, UTI, Bloodstream infections"}
    elif "pseudomonas" in name:
        info = {"gram": "Negative (-)", "disease": "Cystic fibrosis lung infections, Burn wound infections"}
    elif "acinetobacter" in name:
        info = {"gram": "Negative (-)", "disease": "Nosocomial pneumonia, Bacteremia"}
    elif "vibrio" in name:
        info = {"gram": "Negative (-)", "disease": "Cholera, Vibriosis"}
    elif any(k in name for k in ["proteus","morganella","providencia"]):
        info = {"gram": "Negative (-)", "disease": "Complicated UTI, Kidney stones"}
    elif "campylobacter" in name or "helicobacter" in name:
        info = {"gram": "Negative (-)", "disease": "Peptic ulcers, Gastroenteritis"}
    elif "neisseria" in name:
        info = {"gram": "Negative (-)", "disease": "Gonorrhea, Meningitis"}
    elif "haemophilus" in name:
        info = {"gram": "Negative (-)", "disease": "Respiratory infections, Meningitis"}
    elif "staphylococcus" in name:
        info = {"gram": "Positive (+)", "disease": "Skin infections, MRSA, Endocarditis"}
    elif "streptococcus" in name:
        info = {"gram": "Positive (+)", "disease": "Strep throat, Pneumonia, Necrotizing fasciitis"}
    elif "enterococcus" in name:
        info = {"gram": "Positive (+)", "disease": "UTI, Endocarditis, VRE infections"}
    elif "bacillus" in name:
        info = {"gram": "Positive (+)", "disease": "Anthrax, Food poisoning"}
    elif "clostridium" in name:
        info = {"gram": "Positive (+)", "disease": "Tetanus, Botulism, C. diff diarrhea"}
    elif "listeria" in name:
        info = {"gram": "Positive (+)", "disease": "Listeriosis, Foodborne illness"}
    elif "corynebacterium" in name:
        info = {"gram": "Positive (+)", "disease": "Diphtheria"}
    elif "mycobacterium" in name:
        info = {"gram": "Acid-Fast (Gram Variable)", "disease": "Tuberculosis, Leprosy"}
    return info

@st.cache_data
def get_habitat(file_name):
    name = file_name.lower()
    if any(k in name for k in ["ecoli","escherichia","staphylococcus","salmonella","klebsiella","streptococcus","enterococcus"]):
        return "Clinical"
    elif any(k in name for k in ["pseudomonas","acinetobacter"]):
        return "Environmental"
    elif any(k in name for k in ["bacillus","clostridium","mycobacterium"]):
        return "Soil"
    elif "vibrio" in name:
        return "Marine"
    return "General"

@st.cache_data
def extract_data(file_name):
    with open(file_name,'r',encoding='utf-8') as f:
        data = json.load(f)
    drug, mech, records = [], [], []
    habitat = get_habitat(file_name)
    for k in data:
        try:
            inner = list(data[k].values())[0]
            gene_name = inner.get("ARO_name",k).upper()
            d, m = [], []
            for c in inner.get("ARO_category",{}).values():
                cname = c.get("category_aro_class_name","").lower()
                val = c.get("category_aro_name","").title()
                if "drug" in cname: d.append(val)
                elif "mechanism" in cname: m.append(val)
            drug.extend(d); mech.extend(m)
            records.append((gene_name,", ".join(set(d)),", ".join(set(m)),habitat.title()))
        except: continue
    genes = len(data)
    mri = (len(set(drug))+len(set(mech)))/(len(drug)+len(mech)+1)
    ari = len(set(mech))/(genes+1)
    return genes, drug, mech, mri, ari, records

def get_level(mri):
    if mri < 0.15: return "LOW","🟢"
    elif mri < 0.35: return "MODERATE","🟡"
    else: return "HIGH","🔴"

def get_risk_reason(level, u_drugs, u_mechs):
    if "HIGH" in level:
        return f"This pathogen's elevated MRI score reflects a sophisticated and redundant resistance architecture. Deploying {u_mechs} distinct biological mechanisms against {u_drugs} drug classes, the organism exhibits the capacity to dynamically bypass standard frontline therapeutics. When one resistance pathway is circumvented, alternative mechanisms compensate — a hallmark of high-priority clinical threats."
    elif "MODERATE" in level:
        return f"This strain demonstrates a clinically significant, though not maximal, resistance burden. Resistance determinants spanning {u_drugs} drug classes have been identified, necessitating careful antibiogram-guided therapy selection. Secondary and combination treatment regimens should be considered to achieve effective clinical outcomes."
    else:
        return f"The calculated MRI indicates a restricted resistance profile, suggestive of ecological specialization rather than broad clinical adaptation. With resistance distributed across {u_drugs} drug classes via {u_mechs} mechanisms, standard empirical treatment protocols remain viable. Continued genomic surveillance is recommended to monitor for resistance acquisition."

@st.cache_resource
def train_rf_model():
    X, y = [], []
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                genes, drug, mech, mri, _, _ = extract_data(f)
                features = [genes, len(set(drug)), len(set(mech))]
                label = "LOW" if mri < 0.15 else "MODERATE" if mri < 0.35 else "HIGH"
                X.append(features); y.append(label)
            except: continue
    if len(X) < 2: return None
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================
def plot_full_dashboard(drug, mech, mri, genes, records, name):
    level, icon = get_level(mri)
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0e1117')
    color = "red" if level=="HIGH" else "orange" if level=="MODERATE" else "green"
    for a in ax.flat:
        a.set_facecolor('#0e1117'); a.tick_params(colors='white'); a.title.set_color('white')
    drug_c = Counter(drug).most_common(8)
    if drug_c:
        ax[0,0].pie([v for k,v in drug_c],labels=[k[:15]+".." for k,v in drug_c],autopct='%1.1f%%',textprops={'color':"w"})
    ax[0,0].set_title("Drug Classes Resisted")
    mech_c = Counter(mech)
    if mech_c:
        ax[0,1].bar([k[:15]+".." for k in mech_c.keys()],mech_c.values(),color=color)
        ax[0,1].tick_params(axis='x',rotation=35)
    ax[0,1].set_title("Mechanisms Deployed")
    theta = np.linspace(0,np.pi,100)
    ax[0,2].plot(np.cos(theta),np.sin(theta),color='gray')
    ang = mri*np.pi
    ax[0,2].plot([0,np.cos(ang)],[0,np.sin(ang)],color=color,linewidth=5)
    ax[0,2].axis('off'); ax[0,2].set_title(f"MRI Indicator: {round(mri,3)} ({level})")
    gene_list = [r[0] for r in records]
    gene_c = Counter(gene_list).most_common(5)
    if gene_c:
        ax[1,0].bar([k[:15]+".." for k,v in gene_c],[v for k,v in gene_c],color='#87CEEB')
        ax[1,0].tick_params(axis='x',rotation=35)
    ax[1,0].set_title("Top Gene Frequency")
    ax[1,1].bar(["Total Genes"],[genes],color='#2E86C1'); ax[1,1].set_title("Overall Gene Count")
    ax[1,2].bar(["Unique Drugs","Unique Mechs"],[len(set(drug)),len(set(mech))],color=["#9B59B6","#E67E22"]); ax[1,2].set_title("Diversity Comparison")
    fig.tight_layout()
    return fig

def generate_network_html(records, organism_name, color):
    net = Network(height='500px',width='100%',bgcolor='#222222',font_color='white',cdn_resources="in_line",select_menu=True,filter_menu=True)
    net.add_node("HUB",label=organism_name,color=color,size=30)
    for g,d,m,h in records:
        net.add_node(g,label=g[:10],color="#87CEEB",size=15); net.add_edge("HUB",g,color="#ffffff")
        if m:
            for mech in set(m.split(", ")):
                if not mech: continue
                net.add_node(mech,label=mech[:10],color="#FFA500",size=10,shape="box"); net.add_edge(g,mech,color="#aaaaaa")
    net.barnes_hut(gravity=-5000)
    html_path = "temp_network.html"
    with open(html_path,"w",encoding="utf-8") as f:
        f.write(net.generate_html())
    return html_path

def plot_3d_pca_plotly(current_file):
    X, files, risk_levels = [], [], []
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                g,d,m,mr,ar,_ = extract_data(f)
                X.append([mr,len(set(m)),len(set(d))]); files.append(f)
                lv,_ = get_level(mr)
                risk_levels.append("TARGET 🎯" if f==current_file else lv)
            except: continue
    if len(X)<3:
        st.warning("Insufficient data points for 3D PCA rendering.")
        return
    pca = PCA(n_components=3).fit_transform(X)
    df_pca = pd.DataFrame(pca,columns=['Overall Resistance (PC1)','Mechanism Diversity (PC2)','Genetic Density (PC3)'])
    df_pca['Genome']=files; df_pca['Risk Category']=risk_levels
    color_map={"HIGH":"red","MODERATE":"orange","LOW":"green","TARGET 🎯":"gold"}
    fig=px.scatter_3d(df_pca,x='Overall Resistance (PC1)',y='Mechanism Diversity (PC2)',z='Genetic Density (PC3)',color='Risk Category',hover_name='Genome',color_discrete_map=color_map,opacity=0.8,size_max=10)
    fig.update_traces(marker=dict(size=5,line=dict(width=2,color='DarkSlateGrey')),selector=dict(name="TARGET 🎯"))
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0),paper_bgcolor='#0e1117',font_color='white',scene=dict(xaxis=dict(backgroundcolor="#0e1117",gridcolor="gray"),yaxis=dict(backgroundcolor="#0e1117",gridcolor="gray"),zaxis=dict(backgroundcolor="#0e1117",gridcolor="gray")))
    st.plotly_chart(fig,use_container_width=True)

# ==========================================
# NEW: ORIGIN LABELING & COMPARATIVE HEATMAP FUNCTIONS
# ==========================================
def build_drug_class_profile(drug_list):
    """Map detected drugs to standardized class counts."""
    profile = {cls: 0 for cls in DRUG_CLASS_KEYWORDS}
    for d in drug_list:
        mapped = map_drug_to_standard_class(d)
        if mapped:
            profile[mapped] += 1
    return profile

def normalize_profile(profile, total):
    """Normalize gene counts to 0-1 resistance intensity score."""
    if total == 0:
        return {k: 0.0 for k in profile}
    return {k: min(v / max(total * 0.1, 1), 1.0) for k, v in profile.items()}

def plot_origin_comparison_heatmap(sample_profile_norm, selected_origin_key, organism_name):
    """Full comparative heatmap: sample vs all reference origins."""
    drug_classes = list(DRUG_CLASS_KEYWORDS.keys())
    origins = list(REFERENCE_BENCHMARKS.keys())
    display_names = {
        "clinical": "🏥 Clinical",
        "agricultural": "🌾 Agricultural",
        "environmental": "🌿 Environmental",
        "wastewater": "💧 Wastewater",
        "food_production": "🍖 Food Prod."
    }

    # Build matrix: rows = origins + sample, cols = drug classes
    matrix_data = []
    row_labels = []
    for orig in origins:
        bench = REFERENCE_BENCHMARKS[orig]
        row = [bench.get(cls, 0) for cls in drug_classes]
        matrix_data.append(row)
        row_labels.append(display_names.get(orig, orig))

    # Add sample row
    sample_row = [sample_profile_norm.get(cls, 0) for cls in drug_classes]
    matrix_data.append(sample_row)
    row_labels.append(f"🎯 {organism_name[:18]}")

    matrix = np.array(matrix_data)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor('#0e1117')

    # --- LEFT: Full heatmap ---
    ax = axes[0]
    ax.set_facecolor('#0e1117')
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(drug_classes)))
    ax.set_xticklabels(drug_classes, rotation=40, ha='right', color='white', fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, color='white', fontsize=10)
    ax.set_title(f"Comparative Resistance Heatmap — {organism_name}", color='white', fontsize=13, pad=12)

    # Highlight sample row
    for j in range(len(drug_classes)):
        ax.add_patch(plt.Rectangle((j-0.5, len(origins)-0.5), 1, 1,
                                   fill=False, edgecolor='#00d4ff', linewidth=2))

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(drug_classes)):
            val = matrix[i, j]
            txt_color = 'black' if 0.3 < val < 0.8 else 'white'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    color=txt_color, fontsize=7.5, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Resistance Intensity (0=None, 1=Max)', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # --- RIGHT: Delta bar chart (sample vs selected origin) ---
    ax2 = axes[1]
    ax2.set_facecolor('#0e1117')
    selected_bench = REFERENCE_BENCHMARKS[selected_origin_key]
    deltas = []
    for cls in drug_classes:
        delta = sample_profile_norm.get(cls, 0) - selected_bench.get(cls, 0)
        deltas.append(delta)

    colors_bar = ['#ef4444' if d > 0 else '#10b981' for d in deltas]
    bars = ax2.barh(range(len(drug_classes)), deltas, color=colors_bar, alpha=0.85)
    ax2.set_yticks(range(len(drug_classes)))
    ax2.set_yticklabels(drug_classes, color='white', fontsize=9)
    ax2.axvline(0, color='#94a3b8', linewidth=1.2, linestyle='--')
    ax2.set_title(f"Δ vs {display_names.get(selected_origin_key,'')}\nReference", color='white', fontsize=11, pad=10)
    ax2.tick_params(colors='white')
    ax2.set_xlabel("Sample − Reference", color='#94a3b8', fontsize=9)
    ax2.set_facecolor('#0e1117')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#334155')

    plt.tight_layout()
    return fig

def plot_radar_origin_comparison(sample_profile_norm, organism_name):
    """Plotly radar chart comparing sample against all origins."""
    drug_classes = list(DRUG_CLASS_KEYWORDS.keys())
    display_names = {
        "clinical": "Clinical",
        "agricultural": "Agricultural",
        "environmental": "Environmental",
        "wastewater": "Wastewater",
        "food_production": "Food Prod."
    }
    palette = {
        "clinical": "#ef4444", "agricultural": "#10b981",
        "environmental": "#3b82f6", "wastewater": "#f59e0b",
        "food_production": "#a78bfa"
    }

    fig = go.Figure()

    for orig, bench in REFERENCE_BENCHMARKS.items():
        vals = [bench.get(cls, 0) for cls in drug_classes]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=drug_classes + [drug_classes[0]],
            mode='lines', name=display_names[orig],
            line=dict(color=palette[orig], width=1.5, dash='dot'),
            opacity=0.6
        ))

    sample_vals = [sample_profile_norm.get(cls, 0) for cls in drug_classes]
    sample_vals += [sample_vals[0]]
    fig.add_trace(go.Scatterpolar(
        r=sample_vals, theta=drug_classes + [drug_classes[0]],
        mode='lines+markers', name=f"🎯 {organism_name[:20]}",
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=6, color='#00d4ff'),
        fill='toself', fillcolor='rgba(0,212,255,0.08)'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='#0e1117',
            angularaxis=dict(tickcolor='white', color='white', linecolor='#334155'),
            radialaxis=dict(visible=True, range=[0, 1], tickcolor='white', color='#94a3b8',
                            gridcolor='#1e293b', linecolor='#334155')
        ),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font_color='white',
        legend=dict(bgcolor='rgba(15,23,42,0.8)', bordercolor='#334155', borderwidth=1),
        margin=dict(l=60, r=60, t=40, b=40),
        height=480
    )
    return fig

def compute_origin_affinity_scores(sample_profile_norm, genes, mri):
    """Score similarity of sample to each reference origin (0-100)."""
    drug_classes = list(DRUG_CLASS_KEYWORDS.keys())
    scores = {}
    for orig, bench in REFERENCE_BENCHMARKS.items():
        # Euclidean distance in drug-class space
        vec_sample = np.array([sample_profile_norm.get(cls, 0) for cls in drug_classes])
        vec_bench  = np.array([bench.get(cls, 0) for cls in drug_classes])
        dist = np.linalg.norm(vec_sample - vec_bench)
        # MRI proximity
        mri_diff = abs(mri - bench.get("avg_mri", 0.3))
        # Gene count proximity
        gene_diff = abs(genes - bench.get("avg_genes", 50)) / max(bench.get("avg_genes", 50), 1)
        # Combined score (lower = more similar)
        raw = dist * 0.5 + mri_diff * 0.3 + gene_diff * 0.2
        scores[orig] = raw
    # Invert and normalize to 0-100
    max_s = max(scores.values()) or 1
    affinity = {k: round((1 - v / max_s) * 100, 1) for k, v in scores.items()}
    return affinity

def generate_origin_insights(sample_profile_norm, selected_origin_key, affinity_scores, organism_name, mri, genes):
    """Generate automated source-specific resistance pattern insights."""
    bench = REFERENCE_BENCHMARKS[selected_origin_key]
    drug_classes = list(DRUG_CLASS_KEYWORDS.keys())
    origin_display = {
        "clinical": "Clinical/Hospital",
        "agricultural": "Agricultural/Livestock",
        "environmental": "Environmental/Soil",
        "wastewater": "Wastewater/Sewage",
        "food_production": "Food Production"
    }

    high_excess = []
    high_deficit = []
    for cls in drug_classes:
        delta = sample_profile_norm.get(cls, 0) - bench.get(cls, 0)
        if delta > 0.2: high_excess.append((cls, delta))
        elif delta < -0.2: high_deficit.append((cls, abs(delta)))

    high_excess.sort(key=lambda x: -x[1])
    high_deficit.sort(key=lambda x: -x[1])

    best_origin = max(affinity_scores, key=affinity_scores.get)
    best_score = affinity_scores[best_origin]

    insights = {
        "best_match": (origin_display.get(best_origin, best_origin), best_score),
        "selected_match": (origin_display.get(selected_origin_key, selected_origin_key), affinity_scores[selected_origin_key]),
        "excess_classes": high_excess[:3],
        "deficit_classes": high_deficit[:3],
        "mri_vs_bench": mri - bench.get("avg_mri", 0.3),
        "gene_vs_bench": genes - bench.get("avg_genes", 50)
    }
    return insights

# ==========================================
# 4. PDF GENERATOR
# ==========================================
def create_advanced_pdf_report(bac_name,genes,drug,mech,mri,ari,level,icon,records,dashboard_fig,bac_info,habitat,ai_pred_text,ai_conf_text, origin_label="Unknown"):
    pdf_file=f"{bac_name.replace('.json','')}_Detailed_Report.pdf"
    doc=SimpleDocTemplate(pdf_file,pagesize=letter,rightMargin=30,leftMargin=30,topMargin=30,bottomMargin=30)
    styles=getSampleStyleSheet()
    title_style=ParagraphStyle(name='TitleStyle',parent=styles['Heading1'],fontSize=18,spaceAfter=15,textColor=colors.HexColor('#1E3A8A'))
    h2_style=ParagraphStyle(name='H2',parent=styles['Heading2'],fontSize=14,spaceBefore=15,spaceAfter=8,textColor=colors.HexColor('#2E86C1'))
    normal_style=styles['Normal']
    u_drugs,u_mechs=len(set(drug)),len(set(mech))
    t_drugs,t_mechs=len(drug),len(mech)
    elements=[]
    elements.append(Paragraph(f"AI-MRI Report: {bac_name.replace('.json','')}",title_style))
    elements.append(Paragraph("Executive Summary & AI Analysis",h2_style))
    summary_text=f"<b>Pathogen File:</b> {bac_name}<br/><b>Gram Stain:</b> {bac_info['gram']}<br/><b>Associated Pathology:</b> {bac_info['disease']}<br/><b>Ecological Habitat:</b> {habitat}<br/><b>Genomic Origin Label:</b> {origin_label}<br/><b>Total Resistance Genes:</b> {genes}<br/><b>Drug Classes Resisted:</b> {u_drugs}<br/><b>Resistance Mechanisms:</b> {u_mechs}<br/><b>MRI Score:</b> {round(mri,3)} ({level})<br/><b>ARI Score:</b> {round(ari,3)}<br/><br/><b>Machine Learning Prediction:</b> {ai_pred_text} Risk<br/><b>Confidence Distribution:</b> {ai_conf_text}"
    elements.append(Paragraph(summary_text,normal_style))
    elements.append(Spacer(1,15))
    elements.append(Paragraph("Metric Definitions & Clinical Significance",h2_style))
    elements.append(Paragraph("<b>Pathogen Profile:</b> Establishes the essential biological and ecological context.<br/><br/><b>Total Resistance Genes:</b> Absolute count of ARGs identified within the sequenced genomic data.<br/><br/><b>Drug Classes Resisted & Mechanisms Deployed:</b> Enumerates the distinct pharmaceutical classes and molecular strategies.<br/><br/><b>AI Prediction & Confidence Distribution:</b> Random Forest probabilistic classification trained on reference genomic profiles.<br/><br/><b>Origin Label & Comparative Heatmap:</b> Assigns the isolate to one of five environmental/clinical origin categories, then benchmarks its drug-class resistance intensity against curated reference databases for that category. Elevated delta scores (red cells) indicate resistance patterns exceeding the expected baseline for the labelled source.",normal_style))
    elements.append(Paragraph("Clinical Rationale for MRI and ARI Frameworks",h2_style))
    elements.append(Paragraph("The <b>ARI</b> measures resistance density — how efficiently the organism converts its gene count into functional resistance. The <b>MRI</b> consolidates breadth and depth into a single normalized score enabling rapid inter-species triage.",normal_style))
    elements.append(PageBreak())
    elements.append(Paragraph("Mathematical Derivations",h2_style))
    elements.append(Paragraph(f"<b>MRI:</b> ({u_drugs} + {u_mechs}) / ({t_drugs} + {t_mechs} + 1) = <b>{round(mri,3)}</b><br/><b>ARI:</b> {u_mechs} / ({genes} + 1) = <b>{round(ari,3)}</b>",normal_style))
    elements.append(Paragraph("Risk Assessment Interpretation",h2_style))
    elements.append(Paragraph(get_risk_reason(level,u_drugs,u_mechs),normal_style))
    elements.append(Paragraph("Systems Analysis Dashboard",h2_style))
    buf=io.BytesIO()
    dashboard_fig.patch.set_facecolor('white')
    for ax in dashboard_fig.axes:
        ax.set_facecolor('white'); ax.tick_params(colors='black'); ax.title.set_color('black')
        for text in ax.texts: text.set_color('black')
    dashboard_fig.savefig(buf,format='png',bbox_inches='tight',dpi=150)
    buf.seek(0)
    elements.append(RLImage(buf,width=7.5*inch,height=5*inch))
    elements.append(PageBreak())
    elements.append(Paragraph("Complete Gene Resistance Ledger",h2_style))
    table_data=[["Gene Name","Drug Classes Resisted","Mechanisms Deployed","Habitat"]]
    for g,d,m,h in records:
        table_data.append([Paragraph(g,normal_style),Paragraph(d,normal_style),Paragraph(m,normal_style),Paragraph(h,normal_style)])
    t=Table(table_data,colWidths=[1.2*inch,2.2*inch,2.2*inch,0.9*inch])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2E86C1')),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),('ALIGN',(0,0),(-1,-1),'LEFT'),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('BOTTOMPADDING',(0,0),(-1,0),12),('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#F8F9F9')),('GRID',(0,0),(-1,-1),1,colors.black),('VALIGN',(0,0),(-1,-1),'TOP')]))
    elements.append(t)
    doc.build(elements)
    return pdf_file

# ==========================================
# 5. SPLASH SCREEN
# ==========================================
if 'show_splash' not in st.session_state:
    st.session_state.show_splash = True

if st.session_state.show_splash:
    st.markdown("""
    <style>
    #MainMenu{visibility:hidden} footer{visibility:hidden}
    [data-testid="stSidebar"]{display:none !important}
    .block-container{padding:0 !important;max-width:100% !important}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="splash-root">
    <div class="splash-content">
    <div class="sp-hero">
      <span class="sp-dna-ring">🧬</span>
      <h1 class="sp-title">AI-MRI HUB</h1>
      <p class="sp-tagline">Antimicrobial Resistance Intelligence Platform</p>
      <p class="sp-desc">The world's most advanced quantitative framework for decoding antibiotic resistance genes. We transform raw genomic data into actionable clinical intelligence — instantly, accurately, and at scale.</p>
      <div class="sp-stats">
        <div class="sp-stat"><span class="sp-stat-num">2</span><span class="sp-stat-lbl">Novel Indices</span></div>
        <div class="sp-stat"><span class="sp-stat-num">9</span><span class="sp-stat-lbl">Modules</span></div>
        <div class="sp-stat"><span class="sp-stat-num">100+</span><span class="sp-stat-lbl">Genomes</span></div>
        <div class="sp-stat"><span class="sp-stat-num">7</span><span class="sp-stat-lbl">Researchers</span></div>
      </div>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0 clamp(16px,4vw,60px);margin-bottom:10px;background:#020b18;">
      <div class="sp-section-title">Platform Features</div>
      <div class="sp-divider"></div>
    </div>
    <div class="sp-feat-grid" style="background:#020b18;">
      <div class="sp-feat-card"><span class="sp-feat-icon">🦠</span><div class="sp-feat-title">Genomic ARG Profiling</div><div class="sp-feat-desc">Deep extraction and classification of all Antibiotic Resistance Genes from CARD-format data. Every gene, every drug class, every mechanism — catalogued precisely.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">📐</span><div class="sp-feat-title">MRI &amp; ARI Calculation</div><div class="sp-feat-desc">Proprietary indices transform complex gene counts into a single, instantly interpretable risk score using Laplace-smoothed mathematics.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">🤖</span><div class="sp-feat-title">Random Forest AI Prediction</div><div class="sp-feat-desc">Machine learning classifies any pathogen as LOW, MODERATE, or HIGH risk with full probability confidence scores — even for novel strains.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">🕸️</span><div class="sp-feat-title">Interactive Gene Network</div><div class="sp-feat-desc">Visualize the full resistance topology as a live, draggable network graph with filter and selection menus.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">🌌</span><div class="sp-feat-title">3D Landscape PCA</div><div class="sp-feat-desc">Rotate a 3-dimensional scatter map comparing your target genome against every pathogen in the database across resistance axes.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">🩺</span><div class="sp-feat-title">Clinical Susceptibility Zones</div><div class="sp-feat-desc">Automatically identifies drug classes with zero resistance markers — providing an instant safe-zone shortlist for treatment consideration.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">💬</span><div class="sp-feat-title">J.A.R.V.I.S. Bio-AI Chat</div><div class="sp-feat-desc">Ask questions about any genome in plain English. Gemini-powered AI with full genomic context automatically loaded.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">🗺️</span><div class="sp-feat-title">Origin Labeling &amp; Heatmaps</div><div class="sp-feat-desc">Label isolates by source — clinical, agricultural, environmental, wastewater, or food — and instantly compare resistance profiles against curated environmental benchmarks via interactive heatmaps and radar charts.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">📄</span><div class="sp-feat-title">Master PDF Export</div><div class="sp-feat-desc">Generate a comprehensive, publication-ready PDF report with executive summaries, math proofs, dashboards, and gene ledgers in one click.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sp-unique-section" style="background:#020b18;">
      <div class="sp-section-title">Why We Stand Apart</div>
      <div class="sp-divider"></div>
      <div class="sp-compare">
        <div class="sp-comp-card sp-comp-old">
          <div class="sp-comp-title">❌ Traditional AMR Tools</div>
          <div class="sp-comp-item"><span class="sp-chk">✗</span> Output raw gene lists — clinicians must manually interpret hundreds of genes.</div>
          <div class="sp-comp-item"><span class="sp-chk">✗</span> No unified index to compare pathogen severity across species.</div>
          <div class="sp-comp-item"><span class="sp-chk">✗</span> Require bioinformatics expertise — inaccessible to clinical staff.</div>
          <div class="sp-comp-item"><span class="sp-chk">✗</span> Static reports with no interactivity or network exploration.</div>
          <div class="sp-comp-item"><span class="sp-chk">✗</span> No AI chatbot for natural language genomic queries.</div>
          <div class="sp-comp-item"><span class="sp-chk">✗</span> No ML prediction for unknown or novel strains.</div>
          <div class="sp-comp-item"><span class="sp-chk">✗</span> Cannot identify safe drug zones automatically.</div>
          <div class="sp-comp-item"><span class="sp-chk">✗</span> No origin labeling or cross-environment comparative analysis.</div>
        </div>
        <div class="sp-comp-card sp-comp-new">
          <div class="sp-comp-title">✅ AI-MRI Hub</div>
          <div class="sp-comp-item"><span class="sp-chk">✓</span> Proprietary MRI and ARI compress all genomic data into a single, instantly readable risk number.</div>
          <div class="sp-comp-item"><span class="sp-chk">✓</span> Cross-species, cross-habitat standardized scoring enables true pathogen comparison.</div>
          <div class="sp-comp-item"><span class="sp-chk">✓</span> Zero bioinformatics expertise required — upload JSON, get full analysis in seconds.</div>
          <div class="sp-comp-item"><span class="sp-chk">✓</span> Live interactive networks and 3D PCA landscape for spatial resistance topology.</div>
          <div class="sp-comp-item"><span class="sp-chk">✓</span> J.A.R.V.I.S. AI chatbot with genome-aware context injected automatically.</div>
          <div class="sp-comp-item"><span class="sp-chk">✓</span> Random Forest model predicts risk of completely unknown pathogens.</div>
          <div class="sp-comp-item"><span class="sp-chk">✓</span> Automatic clinical susceptibility zone detection identifies zero-resistance drugs instantly.</div>
          <div class="sp-comp-item"><span class="sp-chk">✓</span> Origin labeling + comparative heatmaps reveal source-specific resistance signatures vs 5 curated environmental benchmarks.</div>
        </div>
      </div>
      <div class="sp-pillars">
        <div class="sp-pillar c1"><span class="sp-pillar-icon">📐</span><div class="sp-pillar-title">Dual-Index Scoring</div><div class="sp-pillar-desc">MRI and ARI are original mathematical frameworks. No published tool uses both simultaneously.</div></div>
        <div class="sp-pillar c2"><span class="sp-pillar-icon">🌌</span><div class="sp-pillar-title">3D Resistance Landscape</div><div class="sp-pillar-desc">Plotly-powered PCA maps the entire database in 3 dimensions — a spatial view no standard AMR tool offers.</div></div>
        <div class="sp-pillar c3"><span class="sp-pillar-icon">🤖</span><div class="sp-pillar-title">Context-Aware AI Chat</div><div class="sp-pillar-desc">J.A.R.V.I.S. auto-injects MRI scores and gene counts into every query — real data, not generic biology.</div></div>
        <div class="sp-pillar c4"><span class="sp-pillar-icon">🕸️</span><div class="sp-pillar-title">Live Mechanism Networks</div><div class="sp-pillar-desc">PyVis-powered interactive graphs render gene-to-mechanism relationships as a live filterable topology.</div></div>
        <div class="sp-pillar c5"><span class="sp-pillar-icon">🩺</span><div class="sp-pillar-title">Safe-Zone Clinical Logic</div><div class="sp-pillar-desc">Genomic exclusion logic cross-references resisted classes against a clinical universe for treatment guidance.</div></div>
        <div class="sp-pillar c6"><span class="sp-pillar-icon">📄</span><div class="sp-pillar-title">One-Click Master Reports</div><div class="sp-pillar-desc">ReportLab PDF compiles math, dashboards, and ledgers into a professional document with one click.</div></div>
        <div class="sp-pillar c7"><span class="sp-pillar-icon">🗺️</span><div class="sp-pillar-title">Origin-Comparative Heatmaps</div><div class="sp-pillar-desc">Assign isolate origin and instantly benchmark it against 5 curated environmental reference databases via heatmaps, radar charts, and affinity scoring.</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sp-team-section" style="background:#020b18;">
      <div class="sp-section-title">Meet the Team</div>
      <div class="sp-divider"></div>
      <div class="sp-team-grid">
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(0,212,255,0.1);">👨‍💻</div><div class="sp-tname">Hardik Agrawal</div><div class="sp-trole">Lead Developer</div><div class="sp-tdesc">Full-stack Streamlit architecture, MRI/ARI mathematical framework, and AI prediction pipeline.</div><div class="sp-ttags"><span class="sp-ttag">Python</span><span class="sp-ttag">ML</span><span class="sp-ttag">Streamlit</span></div></div>
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(236,72,153,0.1);">👩‍🔬</div><div class="sp-tname">Poorva Dongarkar</div><div class="sp-trole">Genomics Analyst</div><div class="sp-tdesc">CARD database integration, ARG extraction logic, and bacterial classification system.</div><div class="sp-ttags"><span class="sp-ttag">Genomics</span><span class="sp-ttag">AMR</span><span class="sp-ttag">Data</span></div></div>
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(16,185,129,0.1);">👨‍🔬</div><div class="sp-tname">Yashraj Patil</div><div class="sp-trole">Visualization Engineer</div><div class="sp-tdesc">PyVis networks, Plotly 3D PCA landscape, and Matplotlib 6-panel dashboard.</div><div class="sp-ttags"><span class="sp-ttag">PyVis</span><span class="sp-ttag">Plotly</span><span class="sp-ttag">Matplotlib</span></div></div>
        <div class="sp-team-card"><div class="sp="sp-avatar" style="background:rgba(245,158,11,0.1);">👩‍💻</div><div class="sp-tname">Avani Laswante</div><div class="sp-trole">UI/UX Designer</div><div class="sp-tdesc">Complete CSS design system, dark-mode aesthetic, and animated visual identity.</div><div class="sp-ttags"><span class="sp-ttag">CSS</span><span class="sp-ttag">UI/UX</span><span class="sp-ttag">Design</span></div></div>
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(139,92,246,0.1);">👩‍🔬</div><div class="sp-tname">Zeel Bhanushali</div><div class="sp-trole">Clinical Research</div><div class="sp-tdesc">Clinical susceptibility logic, drug-class universe mapping, and MRI clinical validation.</div><div class="sp-ttags"><span class="sp-ttag">Microbiology</span><span class="sp-ttag">Pharmacology</span></div></div>
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(6,182,212,0.1);">👩‍💻</div><div class="sp-tname">Aayushi Wasnik</div><div class="sp-trole">AI Integration</div><div class="sp-tdesc">J.A.R.V.I.S. chatbot, Gemini API, context injection, and multi-session state management.</div><div class="sp-ttags"><span class="sp-ttag">Gemini API</span><span class="sp-ttag">LLM</span><span class="sp-ttag">Prompt Eng.</span></div></div>
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(244,63,94,0.1);">👨‍🔬</div><div class="sp-tname">Indranil Patil</div><div class="sp-trole">PDF &amp; Documentation</div><div class="sp-tdesc">ReportLab PDF pipeline, mathematical documentation, and academic framework write-up.</div><div class="sp-ttags"><span class="sp-ttag">ReportLab</span><span class="sp-ttag">LaTeX</span><span class="sp-ttag">Writing</span></div></div>
      </div>
      <div class="sp-mission-row">
        <div class="sp-mission-card"><div class="sp-mission-title">🎯 Mission</div><div class="sp-mission-text">Democratize antibiotic resistance genomics through open, intelligent, and beautifully designed scientific tools any researcher or clinician can use.</div></div>
        <div class="sp-mission-card"><div class="sp-mission-title">🔬 Methods</div><div class="sp-mission-text">Rigorous mathematical indexing (MRI, ARI), validated machine learning classification, evidence-based clinical susceptibility mapping, and iterative peer review.</div></div>
        <div class="sp-mission-card"><div class="sp-mission-title">🌍 Vision</div><div class="sp-mission-text">Every hospital and research lab equipped with instant AI-driven resistance profiling — making the next superbug detectable before it becomes untreatable.</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sp-cta-section" style="background:#020b18;">
      <p class="sp-cta-label">Ready to Analyze</p>
      <h2 class="sp-cta-headline">Begin Your Genomic Analysis</h2>
      <p class="sp-cta-sub">Click below to enter the AI-MRI Hub platform.</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([2, 1, 2])
    with col_center:
        if st.button("🚀  ENTER AI-MRI HUB", type="primary", use_container_width=True):
            st.session_state.show_splash = False
            st.rerun()
    st.stop()

# ==========================================
# 6. MAIN APPLICATION
# ==========================================
st.markdown("""
<div class="main-header">
<h1>🧬 AI-Driven Multidimensional Resistance Index</h1>
<p>Quantitative Bio-Analysis of Antibiotic Resistance Genes</p>
<hr style='border:0.5px solid rgba(255,255,255,0.2);margin:16px auto;width:80%;'>
<p style='font-size:clamp(0.75rem,2.5vw,0.95rem);font-weight:300;'>
<b>Developed by:</b> Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik &amp; Indranil Patil
</p>
</div>
""", unsafe_allow_html=True)

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
    st.session_state.current_session = "Chat 1"
    st.session_state.chat_counter = 1

# Session state for origin labels across files
if 'origin_labels' not in st.session_state:
    st.session_state.origin_labels = {}

with st.sidebar:
    st.header("🗄️ Database Sync")
    if st.button("🔄 Refresh Database"):
        st.rerun()
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    analysis_mode = st.radio("Mode:", ["Select Known Bacteria", "AI Predict Unknown"])
    if analysis_mode == "Select Known Bacteria":
        selected_file = st.selectbox("Select a Genome:", json_files)
    st.markdown("---")
    st.success("✅ AI Brain Connected")

if analysis_mode == "Select Known Bacteria" and json_files:
    genes, drug, mech, mri, ari, records = extract_data(selected_file)
    u_drugs, u_mechs = len(set(drug)), len(set(mech))
    level, icon = get_level(mri)
    habitat = get_habitat(selected_file)
    bac_info = get_bacteria_info(selected_file)

    # Compute drug class profile for this genome
    drug_profile = build_drug_class_profile(drug)
    drug_profile_norm = normalize_profile(drug_profile, len(drug))

    if mri > 0.6:
        st.markdown(f'<div class="alert-banner">⚠️ CRITICAL ALERT: {selected_file} identified as High-Priority Superbug — MRI Score: {round(mri,3)}</div>', unsafe_allow_html=True)

    model = train_rf_model()
    ai_pred_text = "N/A"
    ai_conf_text = "N/A"
    if model:
        pred = model.predict([[genes, u_drugs, u_mechs]])[0]
        probs = model.predict_proba([[genes, u_drugs, u_mechs]])[0]
        classes = model.classes_
        conf_dict = {str(c): round(float(p), 3) for c, p in zip(classes, probs)}
        ai_pred_text = str(pred)
        ai_conf_text = str(conf_dict).replace("'", "")

    # ── METRIC CARDS
    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value">{level} {icon}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">MRI Score</div><div class="metric-value">{round(mri,3)}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Genes</div><div class="metric-value">{genes}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Habitat</div><div class="metric-value">{habitat}</div></div>', unsafe_allow_html=True)

    st.write(" ")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "ℹ️ Summary",
        "📊 Dashboard",
        "🧮 Math",
        "🕸️ Network",
        "🤖 AI Chat",
        "🩺 Clinical",
        "🗺️ Origin",
        "📄 PDF",
        "🌌 3D Map"
    ])

    with tab1:
        section_explainer("ℹ️", "About This Tab",
            "This section provides a high-level overview of the platform's capabilities and a detailed pathogen identification card for the selected genome. It summarises all key metrics — gram stain, disease associations, MRI/ARI scores, and the AI risk prediction — in a single structured ledger for rapid clinical or research reference.")

        st.markdown("""
        <div class="welcome-hero">
            <h2>Genomic Resistance Intelligence Dashboard</h2>
            <p>Comprehensive AMR Analysis Platform — Powered by AI</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🧬 Platform Capabilities")
        st.info("""
**The AI-MRI Hub provides a state-of-the-art multidimensional genomic analysis suite:**

- **Genomic ARG Profiling:** Structured identification and classification of Antibiotic Resistance Genes from CARD-format sequence data.
- **Multidimensional Resistance Index (MRI):** A validated clinical risk score consolidating drug evasion breadth and mechanism diversity into a single normalized value.
- **Random Forest Risk Classification:** Machine learning-driven threat stratification benchmarked against a population of reference genomic profiles.
- **Clinical Susceptibility Zone Analysis:** Genomic exclusion logic identifying drug classes with zero resistance markers for treatment guidance.
- **Interactive Resistance Topology:** 3D PCA landscape and live network graphs for spatial visualization of mechanism diversity and genetic density.
- **Origin Labeling & Comparative Heatmaps:** Label genomic data by source (clinical, agricultural, environmental, wastewater, food production) and compare resistance profiles against curated environmental benchmarks.
        """)
        st.markdown("### 🦠 Pathogen Identification Profile")
        # Get stored origin label for display
        stored_origin = st.session_state.origin_labels.get(selected_file, "Not Set")
        st.markdown(f"""
<div class="report-card">
<div class="report-header">Bacterial Resistance Intelligence Ledger</div>
<div class="report-row"><span class="report-label">Target Genome File</span><span class="report-value" style="color:#60a5fa;">{selected_file}</span></div>
<div class="report-row"><span class="report-label">Gram Classification</span><span class="report-value">{bac_info['gram']}</span></div>
<div class="report-row"><span class="report-label">Associated Pathology</span><span class="report-value">{bac_info['disease']}</span></div>
<div class="report-row"><span class="report-label">Ecological Habitat</span><span class="report-value">{habitat}</span></div>
<div class="report-row"><span class="report-label">Genomic Origin Label</span><span class="report-value">{stored_origin}</span></div>
<div class="report-row"><span class="report-label">Total Resistance Genes</span><span class="report-value">{genes}</span></div>
<div class="report-row"><span class="report-label">Drug Classes Resisted</span><span class="report-value">{u_drugs} Classes</span></div>
<div class="report-row"><span class="report-label">Mechanisms Deployed</span><span class="report-value">{u_mechs} Strategies</span></div>
<div class="report-row"><span class="report-label">MRI / ARI Score</span><span class="report-value">{round(mri,3)} ({level}) / {round(ari,3)}</span></div>
<div style="margin-top:16px;padding-top:10px;">
<span class="report-label">AI Risk Classification:</span>&nbsp;
<span class="ai-badge">{ai_pred_text} Risk</span>
<br><br>
<small style="color:#64748b;font-size:0.8rem;">Probability Distribution: {ai_conf_text}</small>
</div>
</div>
""", unsafe_allow_html=True)
        st.markdown("### 🎯 Index Definitions & Clinical Rationale")
        st.info("""
**Why MRI and ARI are clinically necessary:** Conventional genomic analysis outputs raw gene inventories, which fail to provide standardized, comparable risk quantification. The **ARI** measures resistance *density*. The **MRI** consolidates resistance breadth and depth into a single normalized score enabling rapid inter-species triage without manual gene ledger interpretation.
        """)

    with tab2:
        section_explainer("📊", "About This Dashboard",
            "This six-panel systems analysis dashboard provides a simultaneous visual summary of all core resistance dimensions: drug class distribution (pie), mechanism frequency (bar), the MRI semicircle gauge, top gene frequency, total gene count, and a diversity comparison between unique drug classes and mechanisms. Together, these panels reveal the <em>shape</em> of resistance architecture at a glance — helping identify whether a pathogen is broad-spectrum resistant or mechanism-specialized.")

        st.markdown(f"### Systems Analysis Dashboard — `{selected_file}`")
        fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)
        st.pyplot(fig)

        if genes > 50:
            section_explainer("📌", "Interpretation Note",
                f"With <strong>{genes} resistance genes</strong> detected, this isolate has a large genomic resistance burden. The dashboard above reveals its diversity across {u_drugs} drug classes and {u_mechs} mechanisms — use the Origin tab to benchmark this against environmental reference populations.")
        else:
            section_explainer("📌", "Interpretation Note",
                f"This isolate carries <strong>{genes} resistance genes</strong>, a relatively compact profile. The dashboard panels above characterise its resistance specificity — pay attention to which drug classes dominate the pie chart to identify its primary evasion strategy.")

    with tab3:
        section_explainer("🧮", "About This Section",
            "This section presents the full mathematical derivation of the MRI (Multidimensional Resistance Index) and ARI (Antibiotic Resistance Index) frameworks with computed values for the selected genome. Laplace smoothing (+1) prevents division-by-zero for sparse genomes. The risk assessment interpretation box below translates the numerical outputs into plain-language clinical reasoning.")

        st.markdown("### 🧮 Mathematical Framework & Validation")
        col_m1, col_m2 = st.columns([1,1])
        with col_m1:
            st.markdown('<div class="math-card">', unsafe_allow_html=True)
            st.markdown('<div class="math-card-header">Multidimensional Resistance Index (MRI)</div>', unsafe_allow_html=True)
            st.latex(r"MRI = \frac{U_{drugs} + U_{mechs}}{T_{drugs} + T_{mechs} + 1}")
            st.markdown(f'<div class="annotation-box"><div class="annotation-item"><span class="annotation-key">U_drugs</span> — Unique drug classes resisted</div><div class="annotation-item"><span class="annotation-key">U_mechs</span> — Unique resistance mechanisms</div><div class="annotation-item"><span class="annotation-key">+ 1</span> — Laplace smoothing constant</div></div>', unsafe_allow_html=True)
            st.markdown("**Computed Value:**")
            st.latex(rf"\frac{{{u_drugs} + {u_mechs}}}{{{len(drug)} + {len(mech)} + 1}} = {round(mri,3)}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col_m2:
            st.markdown('<div class="math-card">', unsafe_allow_html=True)
            st.markdown('<div class="math-card-header">Antibiotic Resistance Index (ARI)</div>', unsafe_allow_html=True)
            st.latex(r"ARI = \frac{U_{mechs}}{G_{total} + 1}")
            st.markdown(f'<div class="annotation-box"><div class="annotation-item"><span class="annotation-key">U_mechs</span> — Unique resistance mechanisms</div><div class="annotation-item"><span class="annotation-key">G_total</span> — Total genomic resistance genes</div><div class="annotation-item"><span class="annotation-key">+ 1</span> — Laplace smoothing constant</div></div>', unsafe_allow_html=True)
            st.markdown("**Computed Value:**")
            st.latex(rf"\frac{{{u_mechs}}}{{{genes} + 1}} = {round(ari,3)}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("### 🎯 Risk Assessment Interpretation")
        st.markdown(f'<div class="reasoning-box">{get_risk_reason(level,u_drugs,u_mechs)}</div>', unsafe_allow_html=True)
        st.write("")
        section_explainer("📜", "Complete Resistance Gene Ledger",
            "The table below lists every antibiotic resistance gene extracted from the CARD-format JSON, alongside the drug classes it confers resistance to, the molecular mechanism employed, and the ecological habitat annotation. This ledger can be exported as part of the PDF report.")
        df = pd.DataFrame(records, columns=["Gene Name","Drug Classes Resisted","Mechanisms Deployed","Habitat"])
        st.dataframe(df, use_container_width=True)

    with tab4:
        section_explainer("🕸️", "About the Resistance Network",
            "This interactive network graph maps the resistance topology of the selected genome. The central hub node represents the organism. Blue nodes are individual resistance genes; orange box nodes are the molecular mechanisms they deploy. Edges connect genes to their mechanisms, revealing which biological strategies are most heavily convergent. Use the built-in filter and selection menus to isolate specific genes or mechanism clusters.")

        st.markdown("### 🕸️ Interactive Resistance Mechanism Network")
        html_path = generate_network_html(records, selected_file, "red" if level=="HIGH" else "orange" if level=="MODERATE" else "green")
        with open(html_path,'r',encoding='utf-8') as f:
            components.html(f.read(), height=550)

        if u_mechs > 5:
            section_explainer("📌", "Network Complexity Note",
                f"This genome deploys <strong>{u_mechs} distinct resistance mechanisms</strong> — a high-complexity network. Highly connected mechanism nodes (orange boxes with many edges) represent bottleneck strategies that, if pharmacologically bypassed, could simultaneously disable multiple resistance genes.")
        else:
            section_explainer("📌", "Network Complexity Note",
                f"This genome's resistance network is relatively streamlined with <strong>{u_mechs} mechanisms</strong>. Fewer mechanism nodes means the organism's resistance relies on a narrower but potentially highly efficient set of strategies.")

    with tab5:
        section_explainer("🤖", "About J.A.R.V.I.S. Bio-AI",
            "J.A.R.V.I.S. (Justified Analytical Resistance & Virulence Intelligence System) is a Gemini-powered genomic AI assistant. Unlike generic chatbots, J.A.R.V.I.S. automatically receives the full context of the selected genome — MRI score, gene count, drug classes, and mechanism diversity — before answering. Ask it clinical questions, mechanism explanations, treatment strategy suggestions, or comparative queries across the database.")

        st.session_state.current_session = st.selectbox(
            "Active Chat Session:",
            list(st.session_state.chat_sessions.keys())
        )
        col_new, col_clear = st.columns(2)
        with col_new:
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.chat_counter += 1
                new_chat_name = f"Chat {st.session_state.chat_counter}"
                st.session_state.chat_sessions[new_chat_name] = []
                st.session_state.current_session = new_chat_name
                st.rerun()
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_sessions[st.session_state.current_session] = []
                st.rerun()

        for msg in st.session_state.chat_sessions[st.session_state.current_session]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_msg = st.chat_input(f"Query J.A.R.V.I.S. about {selected_file}...")
        if user_msg:
            st.chat_message("user").markdown(user_msg)
            st.session_state.chat_sessions[st.session_state.current_session].append({"role":"user","content":user_msg})
            if not AI_AVAILABLE:
                st.error("⚠️ AI library unavailable.")
            else:
                try:
                    origin_ctx = st.session_state.origin_labels.get(selected_file, "Unknown")
                    context = f"""You are J.A.R.V.I.S., an expert Bioinformatics AI specializing in antimicrobial resistance genomics.
The analyst is examining genome: '{selected_file}'.
Genomic Data: Total ARGs: {genes}, Drug Classes: {u_drugs}, Mechanisms: {u_mechs}, MRI: {round(mri,3)} ({level}), ARI: {round(ari,3)}.
Genomic Origin: {origin_ctx}.
Provide precise, evidence-based responses. Analyst Query: {user_msg}"""
                    with st.spinner("Processing genomic data..."):
                        available_models=[m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        if not available_models:
                            st.error("No models available.")
                        else:
                            target_model=next((m for m in available_models if 'flash' in m),next((m for m in available_models if 'pro' in m),available_models[0]))
                            model_ai=genai.GenerativeModel(target_model)
                            response=model_ai.generate_content(context)
                            st.chat_message("assistant").markdown(response.text)
                            st.session_state.chat_sessions[st.session_state.current_session].append({"role":"assistant","content":response.text})
                except Exception as e:
                    error_msg=str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        st.error("⚠️ API Quota Exceeded. Please wait 60 seconds before re-submitting.")
                    else:
                        st.error(f"AI Connection Error: {e}")

    with tab6:
        section_explainer("🩺", "About Clinical Susceptibility Analysis",
            "This section identifies drug classes that carry <strong>zero detected resistance genes</strong> in the selected isolate — these are potential candidate therapeutic classes based purely on genomic evidence. The population benchmark chart below compares this genome's MRI against the database average, contextualising its severity relative to all other sequenced isolates. Always confirm susceptibility via standard MIC (Minimum Inhibitory Concentration) antibiogram testing before clinical application.")

        st.markdown("### 🩺 Clinical Susceptibility Zone Analysis")
        DRUG_UNIVERSE=["Penicillin","Cephalosporin","Carbapenem","Macrolide","Aminoglycoside","Fluoroquinolone","Tetracycline","Sulfonamide","Glycopeptide"]
        resisted_norm=set([d.lower() for d in drug])
        safe_zones=[d for d in DRUG_UNIVERSE if d.lower() not in resisted_norm]
        st.write("Drug classes with **zero resistance markers** in this isolate — potential therapeutic candidates:")
        st.markdown(f'<div class="susceptibility-card">🛡️ Candidate Therapeutic Classes:<br>{", ".join(safe_zones) if safe_zones else "⚠️ No unresisted classes found in the standard panel."}</div>', unsafe_allow_html=True)
        st.caption("⚠️ Clinical confirmation via standard antibiogram (MIC testing) is required before therapeutic application.")
        st.write("---")
        st.markdown("### 📈 Population Benchmark Comparison")
        section_explainer("📊", "What This Chart Shows",
            "The bar chart compares this genome's MRI score against the mean MRI of all other genomes in the local database. A score significantly above the database average signals an outlier-level resistance burden, potentially indicating a novel superbug candidate or a highly adapted clinical strain.")
        all_mris=[]
        for f in json_files:
            try:
                _,_,_,f_mri,_,_=extract_data(f); all_mris.append(f_mri)
            except: continue
        if all_mris:
            avg_mri=sum(all_mris)/len(all_mris)
            comparison_df=pd.DataFrame({"MRI Score":[mri,avg_mri]},index=["Target Genome","Database Average"])
            st.bar_chart(comparison_df)

    # ─────────────────────────────────────────────────────────
    # TAB 7: ORIGIN LABELING & COMPARATIVE HEATMAPS (NEW)
    # ─────────────────────────────────────────────────────────
    with tab7:
        section_explainer("🗺️", "About Origin Labeling & Comparative Heatmaps",
            "This module allows you to <strong>label each genome by its biological source</strong> (Clinical, Agricultural, Environmental, Wastewater, or Food Production) and then automatically compare its drug-class resistance profile against a curated reference database of environmental benchmarks. "
            "The <strong>comparative heatmap</strong> shows resistance intensity across 12 standardised drug classes for both the sample and all 5 reference origins simultaneously — red cells indicate elevated resistance, green cells indicate lower resistance. "
            "The <strong>delta bar chart</strong> shows exactly how much higher or lower the sample's resistance is vs. your chosen reference. "
            "The <strong>radar chart</strong> overlays all origins in a single polar view for a holistic topological comparison. "
            "The <strong>origin affinity scores</strong> quantify which reference population the sample most closely resembles — a key tool for source attribution in One Health and AMR epidemiology research.")

        st.markdown("### 🗺️ Genomic Origin Labeling & Source-Comparative Analysis")

        # ── ORIGIN LABELING UI ──
        col_label, col_display = st.columns([2, 1])
        with col_label:
            origin_choice = st.selectbox(
                "📌 Assign Origin Label to this Genome:",
                list(ORIGIN_LABELS.keys()),
                index=list(ORIGIN_LABELS.values()).index(
                    st.session_state.origin_labels.get(selected_file, "clinical")
                ) if selected_file in st.session_state.origin_labels else 0
            )
            if st.button("✅ Confirm Origin Label", type="primary"):
                st.session_state.origin_labels[selected_file] = ORIGIN_LABELS[origin_choice]
                st.success(f"Origin label saved: **{origin_choice}** for `{selected_file}`")

        selected_origin_key = st.session_state.origin_labels.get(selected_file, "clinical")

        with col_display:
            origin_css = {
                "clinical": "origin-clinical", "agricultural": "origin-agricultural",
                "environmental": "origin-environmental", "wastewater": "origin-wastewater",
                "food_production": "origin-food_production"
            }
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="text-align:center;">
                <div style="color:#94a3b8;font-size:0.8rem;margin-bottom:8px;">CURRENT LABEL</div>
                <span class="origin-badge {origin_css.get(selected_origin_key,'origin-clinical')}">
                    {origin_choice}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── REFERENCE BENCHMARK SELECTOR ──
        st.markdown("#### 🔬 Select Reference Origin for Delta Comparison")
        ref_labels = list(ORIGIN_LABELS.keys())
        ref_choice = st.selectbox(
            "Compare against this reference population:",
            ref_labels,
            index=ref_labels.index(origin_choice) if origin_choice in ref_labels else 0
        )
        ref_key = ORIGIN_LABELS[ref_choice]

        st.markdown("---")

        # ── AFFINITY SCORES ──
        st.markdown("#### 🎯 Origin Affinity Scores — Which Source Does This Genome Most Resemble?")
        section_explainer("📐", "How Affinity is Computed",
            "Affinity scores are computed as a weighted composite of (1) Euclidean distance in 12-dimensional drug-class resistance space [50%], (2) MRI proximity to the reference population mean [30%], and (3) gene count proximity to the reference mean [20%]. Scores are normalized to 0–100, where 100 = perfect match.")

        affinity_scores = compute_origin_affinity_scores(drug_profile_norm, genes, mri)
        aff_display = {
            "🏥 Clinical": affinity_scores["clinical"],
            "🌾 Agricultural": affinity_scores["agricultural"],
            "🌿 Environmental": affinity_scores["environmental"],
            "💧 Wastewater": affinity_scores["wastewater"],
            "🍖 Food Prod.": affinity_scores["food_production"]
        }
        best_match_key = max(affinity_scores, key=affinity_scores.get)
        best_match_label = [k for k,v in ORIGIN_LABELS.items() if v==best_match_key][0]

        aff_cols = st.columns(5)
        aff_keys = list(aff_display.keys())
        aff_vals = list(aff_display.values())
        for i, col in enumerate(aff_cols):
            val = aff_vals[i]
            highlight = "border:2px solid #00d4ff;" if aff_vals[i]==max(aff_vals) else ""
            col.markdown(f"""
            <div class="metric-card" style="{highlight}">
              <div class="metric-label">{aff_keys[i]}</div>
              <div class="metric-value" style="font-size:1.4rem;">{val}</div>
              <div style="color:#94a3b8;font-size:0.7rem;">/ 100</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(0,212,255,0.07);border:1px solid rgba(0,212,255,0.25);border-radius:10px;padding:12px 16px;margin-top:14px;">
        <span style="color:#00d4ff;font-weight:700;">🏆 Best Match:</span>
        <span style="color:#ffffff;margin-left:8px;font-size:1.05rem;">{best_match_label}</span>
        <span style="color:#94a3b8;margin-left:8px;font-size:0.85rem;">(Affinity Score: {affinity_scores[best_match_key]}/100)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── AUTOMATED INSIGHTS ──
        insights = generate_origin_insights(drug_profile_norm, ref_key, affinity_scores,
                                             selected_file.replace('.json',''), mri, genes)

        st.markdown("#### 🔍 Automated Source-Specific Resistance Pattern Insights")

        ins_cols = st.columns(2)
        with ins_cols[0]:
            if insights["excess_classes"]:
                excess_str = ", ".join([f"<strong>{cls}</strong> (+{delta:.2f})" for cls, delta in insights["excess_classes"]])
                st.markdown(f"""
                <div class="insight-card">
                  <div class="ins-title">⬆️ Resistance Exceeds Reference Baseline</div>
                  <div class="ins-body">This isolate shows significantly elevated resistance vs. <em>{ref_choice}</em> benchmarks in: {excess_str}. This may indicate selective pressure from therapeutic or agricultural antibiotic use not typical of this source.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-card">
                  <div class="ins-title">⬆️ Resistance vs Reference</div>
                  <div class="ins-body">No drug class shows resistance intensity more than 0.2 above the reference baseline — the isolate's profile is broadly consistent with the selected origin.</div>
                </div>
                """, unsafe_allow_html=True)

        with ins_cols[1]:
            if insights["deficit_classes"]:
                deficit_str = ", ".join([f"<strong>{cls}</strong> (−{delta:.2f})" for cls, delta in insights["deficit_classes"]])
                st.markdown(f"""
                <div class="insight-card">
                  <div class="ins-title">⬇️ Resistance Below Reference Baseline</div>
                  <div class="ins-body">This isolate shows notably lower resistance vs. <em>{ref_choice}</em> in: {deficit_str}. These classes may represent viable therapeutic options or indicate this isolate has not been historically exposed to these agents.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="insight-card">
                  <div class="ins-title">⬇️ Resistance vs Reference</div>
                  <div class="ins-body">No drug class shows resistance more than 0.2 below the reference baseline — resistance is broadly distributed in line with expected patterns for this source.</div>
                </div>
                """, unsafe_allow_html=True)

        mri_delta = insights["mri_vs_bench"]
        gene_delta = insights["gene_vs_bench"]
        st.markdown(f"""
        <div class="insight-card" style="margin-top:0;">
          <div class="ins-title">📊 Index Benchmarking vs {ref_choice}</div>
          <div class="ins-body">
            <strong>MRI Delta:</strong> {'+' if mri_delta>=0 else ''}{mri_delta:.3f} — This isolate's MRI is {'<span style="color:#ef4444">higher</span>' if mri_delta>0 else '<span style="color:#10b981">lower</span>'} than the {ref_choice} reference mean ({REFERENCE_BENCHMARKS[ref_key]['avg_mri']}).<br>
            <strong>Gene Count Delta:</strong> {'+' if gene_delta>=0 else ''}{gene_delta} genes vs reference mean ({REFERENCE_BENCHMARKS[ref_key]['avg_genes']}).
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── HEATMAP ──
        st.markdown(f"#### 🔥 Comparative Resistance Heatmap — Sample vs All Origins")
        section_explainer("🎨", "Reading the Heatmap",
            "Each row represents a reference origin population or the target genome (highlighted with a cyan border). Each column represents a standardised drug class. Cell colour intensity indicates resistance level: <strong style='color:#ef4444'>red = high resistance</strong>, <strong style='color:#10b981'>green = low resistance</strong>. The delta bar chart on the right quantifies the gap between your sample and the selected reference — red bars indicate the sample exceeds the reference, green bars indicate it is below.")

        heatmap_fig = plot_origin_comparison_heatmap(drug_profile_norm, ref_key,
                                                      selected_file.replace('.json',''))
        st.pyplot(heatmap_fig, use_container_width=True)

        st.markdown("---")

        # ── RADAR CHART ──
        st.markdown("#### 📡 Resistance Topology Radar — Sample vs All Reference Origins")
        section_explainer("📡", "Reading the Radar Chart",
            "Each axis represents a drug class. The solid cyan polygon is your target genome. Dotted coloured lines are each reference origin's expected resistance profile. Axes where your sample polygon extends significantly beyond all reference lines indicate resistance patterns that are elevated above all known environmental benchmarks — a potential indicator of anthropogenic selection pressure or multi-source contamination.")

        radar_fig = plot_radar_origin_comparison(drug_profile_norm,
                                                  selected_file.replace('.json',''))
        st.plotly_chart(radar_fig, use_container_width=True)

        st.markdown("---")

        # ── ALL LABELLED GENOMES TABLE ──
        st.markdown("#### 🗃️ All Labelled Genomes in Database")
        section_explainer("📋", "What This Table Shows",
            "This table summarises all genomes in your local database that have been assigned an origin label. It enables cross-genome comparison by source — for example, clustering all agricultural isolates to detect shared resistance signatures, or comparing clinical vs environmental isolates for One Health epidemiology.")

        if st.session_state.origin_labels:
            label_rows = []
            for fname, orig_key in st.session_state.origin_labels.items():
                try:
                    g, dr, me, mr, ar, _ = extract_data(fname)
                    lv, ic = get_level(mr)
                    display_orig = [k for k,v in ORIGIN_LABELS.items() if v==orig_key]
                    label_rows.append({
                        "Genome": fname,
                        "Origin": display_orig[0] if display_orig else orig_key,
                        "Genes": g,
                        "MRI": round(mr, 3),
                        "ARI": round(ar, 3),
                        "Risk": f"{lv} {ic}"
                    })
                except:
                    continue
            if label_rows:
                label_df = pd.DataFrame(label_rows)
                st.dataframe(label_df, use_container_width=True)
        else:
            st.info("No genomes have been labelled yet. Use the selector above to assign an origin to the current genome.")

    with tab8:
        section_explainer("📄", "About the PDF Report",
            "This module generates a comprehensive, publication-ready analytical report in PDF format. The document includes: executive summary with all key metrics and origin label, metric definitions and clinical significance, MRI and ARI mathematical derivations, risk interpretation, the 6-panel systems dashboard rendered in print colours, and the complete gene resistance ledger table. The origin label assigned in the Origin tab is automatically incorporated into the report.")

        st.markdown("### 📥 Generate Master Analytical Report")
        st.write("Compile a comprehensive, publication-ready PDF containing executive summary, mathematical derivations, systems dashboard, and complete gene resistance ledger.")
        current_origin_label = [k for k,v in ORIGIN_LABELS.items() if v==selected_origin_key]
        origin_label_str = current_origin_label[0] if current_origin_label else "Not Set"
        st.info(f"📌 Origin label that will be included in the PDF: **{origin_label_str}**")
        if st.button("Generate Master PDF Report", type="primary"):
            with st.spinner("Compiling analytical components into PDF..."):
                pdf_path=create_advanced_pdf_report(selected_file,genes,drug,mech,mri,ari,level,icon,records,fig,bac_info,habitat,ai_pred_text,ai_conf_text, origin_label=origin_label_str)
                with open(pdf_path,"rb") as file:
                    st.download_button(label="⬇️ Download Analytical Report (PDF)",data=file,file_name=pdf_path,mime="application/pdf")

    with tab9:
        section_explainer("🌌", "About the 3D Resistance Landscape",
            "This three-dimensional PCA (Principal Component Analysis) map projects every genome in the database into a spatial coordinate system derived from three composite resistance dimensions: Overall Resistance (PC1), Mechanism Diversity (PC2), and Genetic Density (PC3). Your target genome appears as a gold star. Spatial proximity to other genomes indicates similar resistance architecture. Use the toolbar to rotate, zoom, and export the plot. Genomes clustering in the upper-right-forward region represent the highest combined resistance threat.")

        st.markdown("### 🌌 Interactive Global Resistance Landscape (3D PCA)")
        plot_3d_pca_plotly(selected_file)

elif analysis_mode == "AI Predict Unknown":
    section_explainer("🤖", "About AI Prediction Mode",
        "This mode allows you to input resistance parameters for an <strong>uncharacterised isolate</strong> — one for which you have no JSON genome file — and receive a Random Forest AI risk classification. Enter the total resistance gene count, the number of unique drug classes resisted, and the number of distinct mechanisms detected. The model was trained on all genomes present in the local database and outputs a LOW / MODERATE / HIGH risk prediction with full probability confidence scores.")

    st.header("🤖 Machine Learning Risk Classification — Unknown Pathogen")
    st.write("Input resistance parameters for an uncharacterized isolate to obtain an AI-driven risk stratification.")
    in_genes=st.number_input("Total Resistance Genes Identified",min_value=1,value=15)
    in_drugs=st.number_input("Unique Drug Classes Resisted",min_value=1,value=5)
    in_mechs=st.number_input("Unique Resistance Mechanisms",min_value=1,value=2)
    model=train_rf_model()
    if model and st.button("Run Risk Classification",type="primary"):
        prediction=model.predict([[in_genes,in_drugs,in_mechs]])[0]
        probs=model.predict_proba([[in_genes,in_drugs,in_mechs]])[0]
        classes=model.classes_
        prob_str=" | ".join([f"{c}: {p:.3f}" for c,p in zip(classes,probs)])
        st.success(f"### AI Risk Classification: **{prediction}**")
        st.info(f"**Probability Distribution:** {prob_str}")
        section_explainer("📊", "Interpreting the Prediction",
            f"The model classified this isolate as <strong>{prediction}</strong> risk based on {in_genes} total genes, {in_drugs} drug classes, and {in_mechs} mechanisms. The probability distribution above reflects the model's confidence across all three risk tiers. For isolates near classification boundaries, consider running a full genome CARD analysis and uploading the JSON for comprehensive MRI/ARI scoring.")
