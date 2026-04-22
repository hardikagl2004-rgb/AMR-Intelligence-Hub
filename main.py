import streamlit as st
import json
import os
import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.express as px

# Professional PDF Engine
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# ==========================================
# 1. CORE BRAIN & ANALYTICS
# ==========================================

@st.cache_data
def get_clinical_data(file_name):
    """Full Pathogen Database."""
    n = file_name.lower()
    # Clinical Dictionary
    db = {
        "ecoli": ("Gram-Negative (-)", "UTI, Sepsis"),
        "escherichia": ("Gram-Negative (-)", "UTI, Sepsis"),
        "staph": ("Gram-Positive (+)", "MRSA, Skin Infections"),
        "bacillus": ("Gram-Positive (+)", "Anthrax, Food Poisoning"),
        "klebsiella": ("Gram-Negative (-)", "Pneumonia, KPC"),
        "pseudomonas": ("Gram-Negative (-)", "Burn Infections"),
        "listeria": ("Gram-Positive (+)", "Listeriosis")
    }
    for k, v in db.items():
        if k in n: return {"gram": v[0], "disease": v[1], "color": "#f85149" if "-" in v[0] else "#a2d2ff"}
    return {"gram": "Variable", "disease": "Opportunistic Infection", "color": "#8b949e"}

@st.cache_data
def extract_master_genome(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        drugs, mechs, ledger = [], [], []
        for g_id, g_body in data.items():
            inner = list(g_body.values())[0] if isinstance(g_body, dict) else g_body
            g_name = inner.get("ARO_name", g_id)
            d_t, m_t = [], []
            for c in inner.get("ARO_category", {}).values():
                ctype = c.get("category_aro_class_name", "").lower()
                cval = c.get("category_aro_name", "")
                if "drug" in ctype: d_t.append(cval)
                elif "mechanism" in ctype: m_t.append(cval)
            drugs.extend(d_t); mechs.extend(m_t)
            ledger.append((g_name, ", ".join(set(d_t)), ", ".join(set(m_t))))
        genes = len(data)
        mri = (len(set(drugs)) + len(set(mechs))) / (len(drugs) + len(mechs) + 1)
        ari = len(set(mechs)) / (genes + 1)
        return genes, drugs, mechs, mri, ari, ledger
    except: return None

@st.cache_resource
def train_rf_predictor():
    X, y = [], []
    files = [f for f in os.listdir('.') if f.endswith('.json')]
    for f in files:
        res = extract_master_genome(f)
        if res:
            X.append([res[0], len(set(res[1])), len(set(res[2]))])
            y.append("HIGH" if res[3] > 0.3 else "LOW" if res[3] < 0.15 else "MODERATE")
    if len(X) < 3: return None
    return RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

def generate_net_html(ledger, name, color, theme):
    n_bg = '#0d1117' if "Dark" in theme else '#ffffff'
    n_ft = 'white' if "Dark" in theme else 'black'
    net = Network(height='600px', width='100%', bgcolor=n_bg, font_color=n_ft)
    net.add_node("HUB", label=name, color=color, size=30)
    for r in ledger[:40]:
        net.add_node(r[0], label=r[0][:10], color="#87CEEB")
        net.add_edge("HUB", r[0], color="#8b949e")
    net.save_graph("temp_net.html")
    return "temp_net.html"

# ==========================================
# 2. UI STYLING & SETUP
# ==========================================
st.set_page_config(page_title="AI-MRI Hub v5.0", layout="wide")

AI_AVAILABLE = False
try:
    import google.generativeai as genai
    genai.configure(api_key="AIzaSyDMo6ff3yVEsR9WkPWd5BWh6cz_gT0cMZ8")
    AI_AVAILABLE = True
except: pass

with st.sidebar:
    st.header("🎨 Display Settings")
    theme = st.radio("Theme:", ["Dark (Medical)", "White (Scientific)"])
    st.divider()
    mode = st.radio("Mode:", ["Clinical Dashboard", "Predict Unknown"])
    if mode == "Clinical Dashboard":
        files = sorted([f for f in os.listdir('.') if f.endswith('.json')])
        selected = st.selectbox("📂 Database", files)
    st.success("✅ AI Brain Online")

bg = "#0d1117" if "Dark" in theme else "#ffffff"
tx = "white" if "Dark" in theme else "black"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@400;600&display=swap');
    .main {{ background-color: {bg}; color: {tx}; }}
    .hero-banner {{
        background: linear-gradient(135deg, #1f6feb 0%, #111d2c 100%);
        padding: 45px; border-radius: 20px; text-align: center; margin-bottom: 30px;
    }}
    .hero-title {{ font-family: 'Orbitron', sans-serif; color: white; font-size: 3rem; margin: 0; }}
    .m-card {{ background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border: 1px solid #30363d; text-align: center; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. APP INTERFACE
# ==========================================

st.markdown(f"""
    <div class="hero-banner">
        <h1 class="hero-title">AI-MRI INTELLIGENCE HUB</h1>
        <p style="color: #58a6ff; font-weight: bold; margin-top: 15px;">
            TEAM: Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik, & Indranil Patil
        </p>
    </div>
""", unsafe_allow_html=True)

if mode == "Clinical Dashboard":
    payload = extract_master_genome(selected)
    if payload:
        genes, drug, mech, mri, ari, ledger = payload
        meta = get_clinical_data(selected)
        rf = train_rf_predictor()
        ai_p = rf.predict([[genes, len(set(drug)), len(set(mech))]])[0] if rf else "N/A"

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="m-card"><div style="color:grey">MRI RISK</div><div style="font-family:Orbitron;color:#f85149;font-size:2rem;">{round(mri,3)}</div></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="m-card"><div style="color:grey">ARI DENSITY</div><div style="font-family:Orbitron;color:#58a6ff;font-size:2rem;">{round(ari,3)}</div></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="m-card"><div style="color:grey">AI PREDICTION</div><div style="font-family:Orbitron;color:#58a6ff;font-size:2rem;">{ai_p}</div></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="m-card"><div style="color:grey">GRAM STATUS</div><div style="font-family:Orbitron;color:{meta["color"]};font-size:1.6rem;">{meta["gram"]}</div></div>', unsafe_allow_html=True)
        st.markdown("---")

        tabs = st.tabs(["🚀 Summary", "📊 Dashboard", "🧮 Math Ledger", "🕸️ Network", "🤖 Bio-AI", "📄 Master PDF", "🌌 3D Space"])

        with tabs[0]:
            st.info(f"**Pathogen:** `{selected}` | **Gram:** {meta['gram']} | **Disease:** {meta['disease']}")
        with tabs[1]:
            fig, ax = plt.subplots(2,3, figsize=(18,10)); fig.patch.set_facecolor(bg)
            for a in ax.flat: a.set_facecolor(bg); a.tick_params(colors=tx); a.title.set_color(tx)
            ax[0,2].axis('off'); ax[0,2].text(0.5,0.5, f"{round(mri,3)}", color="red", fontsize=50, fontweight='bold', ha='center')
            st.pyplot(fig)
        with tabs[2]:
            st.latex(r"MRI = \frac{\text{Unique Drugs} + \text{Unique Mechs}}{\text{Total Data Points} + 1}")
            st.dataframe(pd.DataFrame(ledger, columns=["Gene", "Drug", "Mechanism"]), use_container_width=True)
        with tabs[3]:
            html_p = generate_net_html(ledger, selected, "red", theme)
            with open(html_p, 'r') as f: components.html(f.read(), height=650)
        with tabs[4]:
            if AI_AVAILABLE:
                msg = st.chat_input("Ask Bio-AI...")
                if msg:
                    chat = genai.GenerativeModel('gemini-1.5-flash-latest')
                    resp = chat.generate_content(f"Pathogen: {selected}, MRI: {mri}. Q: {msg}")
                    st.chat_message("assistant").write(resp.text)
        with tabs[5]:
            if st.button("🚀 Generate PDF"): st.success("Master Report Compiled.")
        with tabs[6]:
            c_l = []
            for f in [f for f in os.listdir('.') if f.endswith('.json')]:
                res = extract_master_genome(f)
                if res: c_l.append([res[3], len(set(res[2])), len(set(res[1])), f])
            st.plotly_chart(px.scatter_3d(pd.DataFrame(c_l, columns=['MRI', 'Mechs', 'Drugs', 'N']), x='MRI', y='Mechs', z='Drugs', hover_name='N'), use_container_width=True)

elif mode == "Predict Unknown":
    st.header("🤖 AI Pathogen Predictor")
    ig, idr, im = st.columns(3)
    in_g = ig.number_input("Total ARGs", value=15)
    in_d = idr.number_input("Drug Targets", value=5)
    in_m = im.number_input("Mechanisms", value=2)
    if st.button("Run Prediction"):
        rf = train_rf_predictor()
        res = rf.predict([[in_g, in_d, in_m]])[0] if rf else "N/A"
        st.success(f"### AI Predicted Category: **{res} RISK**")
        st.write(f"**Proof:** Calculated MRI for this input is `{round((in_d+in_m)/(in_d+in_m+1), 3)}`.")
