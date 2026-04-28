import streamlit as st
import json
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.express as px

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

/* ── DATA FRAME ── */
[data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 10px; overflow: hidden; }

/* ── ANIMATIONS ── */
@keyframes fadeInDown { from{opacity:0;transform:translateY(-40px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeInUp   { from{opacity:0;transform:translateY(40px)}  to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn     { from{opacity:0} to{opacity:1} }
@keyframes shimmer    { 0%{background-position:-200% center} 100%{background-position:200% center} }
@keyframes float      { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
@keyframes borderPulse{ 0%,100%{border-color:rgba(59,130,246,0.4)} 50%{border-color:rgba(96,165,250,0.9)} }

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

/* HERO */
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

/* STATS */
.sp-stats { display:flex; justify-content:center; gap:clamp(8px,2vw,20px); flex-wrap:wrap; margin-bottom:50px; animation:fadeInUp 2s ease forwards; opacity:0; animation-delay:1.2s; }
.sp-stat { background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.2); border-radius:16px; padding:clamp(14px,3vw,22px) clamp(16px,4vw,32px); min-width:clamp(70px,20vw,120px); transition:all 0.3s ease; }
.sp-stat:hover { transform:translateY(-5px); border-color:rgba(0,212,255,0.5); }
.sp-stat-num { font-family:'Orbitron',monospace; font-size:clamp(1.4rem,5vw,2.2rem); font-weight:900; color:#00d4ff; display:block; }
.sp-stat-lbl { font-size:clamp(0.6rem,2vw,0.7rem); color:#64748b; letter-spacing:2px; text-transform:uppercase; }

/* SECTION TITLES */
.sp-section-title { font-family:'Orbitron',monospace !important; font-size:clamp(1.2rem,4vw,1.8rem) !important; font-weight:700 !important; color:#f1f5f9 !important; text-align:center; margin-bottom:10px !important; letter-spacing:2px; }
.sp-divider { width:80px; height:3px; background:linear-gradient(90deg,#00d4ff,#7c3aed); margin:0 auto 30px auto; border-radius:3px; }

/* FEATURE GRID */
.sp-feat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(260px,100%),1fr)); gap:16px; padding:0 clamp(16px,4vw,60px) 50px; }
.sp-feat-card { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:clamp(18px,3vw,28px) clamp(14px,3vw,22px); transition:all 0.35s ease; }
.sp-feat-card:hover { transform:translateY(-6px); border-color:rgba(0,212,255,0.25); }
.sp-feat-icon { font-size:clamp(1.6rem,4vw,2.2rem); display:block; margin-bottom:12px; }
.sp-feat-title { color:#00d4ff !important; font-size:clamp(0.75rem,2.5vw,0.85rem); font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
.sp-feat-desc { color:#64748b; font-size:clamp(0.82rem,2.5vw,0.88rem); line-height:1.65; }

/* COMPARE */
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

/* PILLARS */
.sp-pillars { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(200px,100%),1fr)); gap:16px; }
.sp-pillar { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:clamp(18px,3vw,26px) clamp(14px,3vw,20px); text-align:center; transition:all 0.3s ease; position:relative; overflow:hidden; }
.sp-pillar::before { content:''; position:absolute; top:0;left:0;right:0; height:3px; }
.sp-pillar.c1::before{background:linear-gradient(90deg,#00d4ff,#10b981)} .sp-pillar.c2::before{background:linear-gradient(90deg,#7c3aed,#a78bfa)} .sp-pillar.c3::before{background:linear-gradient(90deg,#f59e0b,#fbbf24)} .sp-pillar.c4::before{background:linear-gradient(90deg,#f43f5e,#fb7185)} .sp-pillar.c5::before{background:linear-gradient(90deg,#10b981,#34d399)} .sp-pillar.c6::before{background:linear-gradient(90deg,#06b6d4,#67e8f9)}
.sp-pillar:hover{transform:translateY(-6px);border-color:rgba(0,212,255,0.3)}
.sp-pillar-icon{font-size:clamp(1.6rem,4vw,2.3rem);display:block;margin-bottom:10px}
.sp-pillar-title{color:#f1f5f9 !important;font-size:clamp(0.75rem,2.5vw,0.82rem);font-weight:700;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px}
.sp-pillar-desc{color:#475569;font-size:clamp(0.78rem,2.5vw,0.82rem);line-height:1.6}

/* TEAM */
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

/* MISSION */
.sp-mission-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:16px;margin-top:24px}
.sp-mission-card{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:clamp(18px,3vw,28px) clamp(14px,3vw,22px);text-align:center}
.sp-mission-title{color:#00d4ff !important;font-size:clamp(0.72rem,2.5vw,0.8rem);font-weight:700;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px !important;font-family:'Orbitron',monospace}
.sp-mission-text{color:#64748b;font-size:clamp(0.82rem,2.5vw,0.85rem);line-height:1.7}

/* CTA */
.sp-cta-section{padding:clamp(30px,5vw,60px) clamp(16px,5vw,60px);text-align:center;background:linear-gradient(180deg,transparent,rgba(0,212,255,0.04),transparent);border-top:1px solid rgba(0,212,255,0.08)}
.sp-cta-label{font-family:'Orbitron',monospace;font-size:clamp(0.65rem,2vw,0.8rem);letter-spacing:5px;text-transform:uppercase;color:#475569 !important;margin-bottom:16px !important}
.sp-cta-headline{font-size:clamp(1.4rem,5vw,2.4rem) !important;font-weight:700 !important;color:#f1f5f9 !important;margin-bottom:10px !important;line-height:1.3}
.sp-cta-sub{color:#64748b !important;font-size:clamp(0.85rem,3vw,1rem) !important;margin-bottom:30px !important}

/* ═══ RESPONSIVE OVERRIDES ═══ */
@media (max-width: 640px) {
  .report-row { flex-direction: column; gap: 2px; }
  .report-value { text-align: left; }
  .sp-comp-new::before { font-size: 0.55rem; padding: 3px 10px; right: 10px; }
  /* Keep stats 2-per-row on very small screens */
  .sp-stats { justify-content: center; }
  .sp-stat { flex: 1 1 calc(50% - 8px); min-width: 0; }
}

@media (max-width: 480px) {
  .sp-title { letter-spacing: 1px !important; }
  .sp-tagline { letter-spacing: 2px !important; }
}

/* Tab label fix for mobile */
@media (max-width: 768px) {
  [data-baseweb="tab"] { padding: 8px 6px !important; }
  [data-baseweb="tab"] p { font-size: 0.75rem !important; }
}
</style>
""", unsafe_allow_html=True)

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
# 4. PDF GENERATOR
# ==========================================
def create_advanced_pdf_report(bac_name,genes,drug,mech,mri,ari,level,icon,records,dashboard_fig,bac_info,habitat,ai_pred_text,ai_conf_text):
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
    summary_text=f"<b>Pathogen File:</b> {bac_name}<br/><b>Gram Stain:</b> {bac_info['gram']}<br/><b>Associated Pathology:</b> {bac_info['disease']}<br/><b>Ecological Habitat:</b> {habitat}<br/><b>Total Resistance Genes:</b> {genes}<br/><b>Drug Classes Resisted:</b> {u_drugs}<br/><b>Resistance Mechanisms:</b> {u_mechs}<br/><b>MRI Score:</b> {round(mri,3)} ({level})<br/><b>ARI Score:</b> {round(ari,3)}<br/><br/><b>Machine Learning Prediction:</b> {ai_pred_text} Risk<br/><b>Confidence Distribution:</b> {ai_conf_text}"
    elements.append(Paragraph(summary_text,normal_style))
    elements.append(Spacer(1,15))
    elements.append(Paragraph("Metric Definitions & Clinical Significance",h2_style))
    elements.append(Paragraph("<b>Pathogen Profile:</b> Establishes the essential biological and ecological context.<br/><br/><b>Total Resistance Genes:</b> Absolute count of ARGs identified within the sequenced genomic data.<br/><br/><b>Drug Classes Resisted & Mechanisms Deployed:</b> Enumerates the distinct pharmaceutical classes and molecular strategies.<br/><br/><b>AI Prediction & Confidence Distribution:</b> Random Forest probabilistic classification trained on reference genomic profiles.",normal_style))
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
        <div class="sp-stat"><span class="sp-stat-num">8</span><span class="sp-stat-lbl">Modules</span></div>
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
        </div>
      </div>
      <div class="sp-pillars">
        <div class="sp-pillar c1"><span class="sp-pillar-icon">📐</span><div class="sp-pillar-title">Dual-Index Scoring</div><div class="sp-pillar-desc">MRI and ARI are original mathematical frameworks. No published tool uses both simultaneously.</div></div>
        <div class="sp-pillar c2"><span class="sp-pillar-icon">🌌</span><div class="sp-pillar-title">3D Resistance Landscape</div><div class="sp-pillar-desc">Plotly-powered PCA maps the entire database in 3 dimensions — a spatial view no standard AMR tool offers.</div></div>
        <div class="sp-pillar c3"><span class="sp-pillar-icon">🤖</span><div class="sp-pillar-title">Context-Aware AI Chat</div><div class="sp-pillar-desc">J.A.R.V.I.S. auto-injects MRI scores and gene counts into every query — real data, not generic biology.</div></div>
        <div class="sp-pillar c4"><span class="sp-pillar-icon">🕸️</span><div class="sp-pillar-title">Live Mechanism Networks</div><div class="sp-pillar-desc">PyVis-powered interactive graphs render gene-to-mechanism relationships as a live filterable topology.</div></div>
        <div class="sp-pillar c5"><span class="sp-pillar-icon">🩺</span><div class="sp-pillar-title">Safe-Zone Clinical Logic</div><div class="sp-pillar-desc">Genomic exclusion logic cross-references resisted classes against a clinical universe for treatment guidance.</div></div>
        <div class="sp-pillar c6"><span class="sp-pillar-icon">📄</span><div class="sp-pillar-title">One-Click Master Reports</div><div class="sp-pillar-desc">ReportLab PDF compiles math, dashboards, and ledgers into a professional document with one click.</div></div>
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
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(245,158,11,0.1);">👩‍💻</div><div class="sp-tname">Avani Laswante</div><div class="sp-trole">UI/UX Designer</div><div class="sp-tdesc">Complete CSS design system, dark-mode aesthetic, and animated visual identity.</div><div class="sp-ttags"><span class="sp-ttag">CSS</span><span class="sp-ttag">UI/UX</span><span class="sp-ttag">Design</span></div></div>
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

    # ── METRIC CARDS: 2x2 grid for mobile ──
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

    # ── TABS: shortened labels for mobile ──
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "ℹ️ Summary",
        "📊 Dashboard",
        "🧮 Math",
        "🕸️ Network",
        "🤖 AI Chat",
        "🩺 Clinical",
        "📄 PDF",
        "🌌 3D Map"
    ])

    with tab1:
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
        """)
        st.markdown("### 🦠 Pathogen Identification Profile")
        st.markdown(f"""
<div class="report-card">
<div class="report-header">Bacterial Resistance Intelligence Ledger</div>
<div class="report-row"><span class="report-label">Target Genome File</span><span class="report-value" style="color:#60a5fa;">{selected_file}</span></div>
<div class="report-row"><span class="report-label">Gram Classification</span><span class="report-value">{bac_info['gram']}</span></div>
<div class="report-row"><span class="report-label">Associated Pathology</span><span class="report-value">{bac_info['disease']}</span></div>
<div class="report-row"><span class="report-label">Ecological Habitat</span><span class="report-value">{habitat}</span></div>
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
        st.markdown(f"### Systems Analysis Dashboard — `{selected_file}`")
        fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)
        st.pyplot(fig)

    with tab3:
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
        st.markdown("### 📜 Complete Resistance Gene Ledger")
        df = pd.DataFrame(records, columns=["Gene Name","Drug Classes Resisted","Mechanisms Deployed","Habitat"])
        st.dataframe(df, use_container_width=True)

    with tab4:
        st.markdown("### 🕸️ Interactive Resistance Mechanism Network")
        st.write("Use the filter and selection menus within the graph to isolate specific gene or mechanism nodes.")
        html_path = generate_network_html(records, selected_file, "red" if level=="HIGH" else "orange" if level=="MODERATE" else "green")
        with open(html_path,'r',encoding='utf-8') as f:
            components.html(f.read(), height=550)

    with tab5:
        # Mobile-friendly chat controls: selectbox + 2 buttons stacked
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
                    context = f"""You are J.A.R.V.I.S., an expert Bioinformatics AI specializing in antimicrobial resistance genomics.
The analyst is examining genome: '{selected_file}'.
Genomic Data: Total ARGs: {genes}, Drug Classes: {u_drugs}, Mechanisms: {u_mechs}, MRI: {round(mri,3)} ({level}), ARI: {round(ari,3)}.
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
        st.markdown("### 🩺 Clinical Susceptibility Zone Analysis")
        DRUG_UNIVERSE=["Penicillin","Cephalosporin","Carbapenem","Macrolide","Aminoglycoside","Fluoroquinolone","Tetracycline","Sulfonamide","Glycopeptide"]
        resisted_norm=set([d.lower() for d in drug])
        safe_zones=[d for d in DRUG_UNIVERSE if d.lower() not in resisted_norm]
        st.write("Drug classes with **zero resistance markers** in this isolate — potential therapeutic candidates:")
        st.markdown(f'<div class="susceptibility-card">🛡️ Candidate Therapeutic Classes:<br>{", ".join(safe_zones)}</div>', unsafe_allow_html=True)
        st.caption("⚠️ Clinical confirmation via standard antibiogram (MIC testing) is required before therapeutic application.")
        st.write("---")
        st.markdown("### 📈 Population Benchmark Comparison")
        all_mris=[]
        for f in json_files:
            try:
                _,_,_,f_mri,_,_=extract_data(f); all_mris.append(f_mri)
            except: continue
        if all_mris:
            avg_mri=sum(all_mris)/len(all_mris)
            comparison_df=pd.DataFrame({"MRI Score":[mri,avg_mri]},index=["Target Genome","Database Average"])
            st.bar_chart(comparison_df)

    with tab7:
        st.markdown("### 📥 Generate Master Analytical Report")
        st.write("Compile a comprehensive, publication-ready PDF containing executive summary, mathematical derivations, systems dashboard, and complete gene resistance ledger.")
        if st.button("Generate Master PDF Report", type="primary"):
            with st.spinner("Compiling analytical components into PDF..."):
                pdf_path=create_advanced_pdf_report(selected_file,genes,drug,mech,mri,ari,level,icon,records,fig,bac_info,habitat,ai_pred_text,ai_conf_text)
                with open(pdf_path,"rb") as file:
                    st.download_button(label="⬇️ Download Analytical Report (PDF)",data=file,file_name=pdf_path,mime="application/pdf")

    with tab8:
        st.markdown("### 🌌 Interactive Global Resistance Landscape (3D PCA)")
        st.write("Rotate, zoom, and export this three-dimensional comparative map using the toolbar in the top-right corner.")
        plot_3d_pca_plotly(selected_file)

elif analysis_mode == "AI Predict Unknown":
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
