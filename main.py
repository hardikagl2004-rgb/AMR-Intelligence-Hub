import streamlit as st
import json
import os
import io
import time
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.express as px

# PDF Engine Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# ==========================================
# 1. CORE BRAIN & UTILITIES (Top-Level)
# ==========================================

def get_level(mri):
    """Core mathematical risk logic."""
    if mri < 0.15: return "LOW", "🟢"
    if mri < 0.35: return "MODERATE", "🟡"
    return "HIGH", "🔴"

@st.cache_data
def get_extended_metadata(file_name):
    """Clinical metadata for known pathogens."""
    name = file_name.lower()
    meta = {"gram": "Variable", "disease": "Opportunistic Infection", "habitat": "General"}
    if any(k in name for k in ["ecoli", "escherichia", "shigella"]):
        meta = {"gram": "Negative (-)", "disease": "Gastroenteritis, Sepsis", "habitat": "Clinical/Gut"}
    elif "staphylococcus" in name:
        meta = {"gram": "Positive (+)", "disease": "MRSA, Skin Infections", "habitat": "Clinical/Skin"}
    elif "campylobacter" in name:
        meta = {"gram": "Negative (-)", "disease": "Severe Enteritis", "habitat": "Zoonotic"}
    return meta

@st.cache_data
def parse_genomic_json(file_path):
    """Robust genomic parser for AMR JSON formats."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        drugs, mechs, records = [], [], []
        for gene_key, gene_val in data.items():
            try:
                node = list(gene_val.values())[0] if isinstance(gene_val, dict) else gene_val
                g_name = node.get("ARO_name", gene_key)
                
                c_drugs, c_mechs = [], []
                cats = node.get("ARO_category", {})
                if isinstance(cats, dict):
                    for c_data in cats.values():
                        c_type = c_data.get("category_aro_class_name", "").lower()
                        c_val = c_data.get("category_aro_name", "")
                        if "drug" in c_type: c_drugs.append(c_val)
                        elif "mechanism" in c_type: c_mechs.append(c_val)
                
                drugs.extend(c_drugs)
                mechs.extend(c_mechs)
                records.append((g_name, ", ".join(set(c_drugs)), ", ".join(set(c_mechs))))
            except: continue
            
        genes_count = len(data)
        # MRI Score: Multidimensional Resistance Index
        mri_score = (len(set(drugs)) + len(set(mechs))) / (len(drugs) + len(mechs) + 1)
        # ARI Score: Antibiotic Resistance Index
        ari_score = len(set(mechs)) / (genes_count + 1)
        
        return genes_count, drugs, mechs, mri_score, ari_score, records
    except Exception: return None

# ==========================================
# 2. AI PREDICTOR ENGINE (Random Forest)
# ==========================================

@st.cache_resource
def initialize_ai_predictor():
    """Trains a Random Forest model on the entire local database."""
    X, y = [], []
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    for f in json_files:
        res = parse_genomic_json(f)
        if res:
            # Features: [Total Genes, Unique Drugs, Unique Mechanisms]
            X.append([res[0], len(set(res[1])), len(set(res[2]))])
            y.append(get_level(res[3])[0])
            
    if len(X) < 4: # Minimum data threshold for training
        return None
        
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ==========================================
# 3. ADVANCED FRONTEND CSS (PROFESSIONAL WOW)
# ==========================================
st.set_page_config(page_title="AI-MRI Hub v2.0", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .main { background-color: #0d1117; color: #c9d1d9; }
    
    /* Responsive Metric Cards with proper wrapping */
    .metric-container {
        display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 30px;
    }
    .wow-card {
        background: linear-gradient(145deg, #161b22, #0d1117);
        border: 1px solid #30363d; border-radius: 15px;
        padding: 20px; flex: 1; min-width: 200px;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    .wow-label { color: #8b949e; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; margin-bottom: 10px; }
    .wow-value { font-family: 'Orbitron', sans-serif; color: #58a6ff; font-size: 2.2rem; font-weight: 700; white-space: normal; word-break: break-all; }
    .wow-delta { font-size: 0.9rem; margin-top: 5px; font-weight: 700; }

    /* Custom Header */
    .hero-banner {
        background: linear-gradient(90deg, #1f6feb 0%, #111d2c 100%);
        padding: 45px; border-radius: 20px; border: 1px solid #30363d;
        margin-bottom: 35px; text-align: center;
    }
    .hero-title { font-family: 'Orbitron', sans-serif; font-size: 3rem; color: white; margin: 0; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. MASTER PDF ENGINE (WITH GRAPH IMAGES)
# ==========================================

def create_advanced_master_pdf(file_name, stats, records, dash_img, meta, ai_pred):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Clinical Styles
    title_style = ParagraphStyle(name='T', fontSize=22, textColor=colors.HexColor('#1f6feb'), alignment=1, spaceAfter=20)
    h2_style = ParagraphStyle(name='H2', fontSize=14, textColor=colors.HexColor('#238636'), spaceBefore=15, spaceAfter=10)
    
    elements = []
    
    # Header
    elements.append(Paragraph("BIO-AI CLINICAL MASTER REPORT", title_style))
    elements.append(Paragraph(f"<b>Pathogen ID:</b> {file_name}", styles['Normal']))
    elements.append(Paragraph(f"<b>AI Predicted Risk:</b> {ai_pred}", styles['Normal']))
    elements.append(Spacer(1, 15))
    
    # Metrics Table
    m_data = [
        ["MRI SCORE", "ARI DENSITY", "TOTAL GENES", "GRAM STATUS"],
        [f"{round(stats['mri'], 3)}", f"{round(stats['ari'], 3)}", f"{stats['genes']}", meta['gram']]
    ]
    t1 = Table(m_data, colWidths=[1.5*inch]*4)
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    elements.append(t1)
    
    # Dashboard Image
    elements.append(Paragraph("I. Multi-Panel Analytical Dashboard", h2_style))
    img = RLImage(dash_img, width=6*inch, height=4*inch)
    elements.append(img)
    
    # Ledger Table
    elements.append(PageBreak())
    elements.append(Paragraph("II. Comprehensive ARG Ledger", h2_style))
    l_data = [["Gene Name", "Resisted Targets", "Mechanism"]]
    for r in records[:60]:
        l_data.append([Paragraph(r[0], styles['Normal']), Paragraph(r[1], styles['Normal']), Paragraph(r[2], styles['Normal'])])
    
    t2 = Table(l_data, colWidths=[1.2*inch, 2.4*inch, 2.4*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTSIZE', (0,0), (-1,-1), 8)
    ]))
    elements.append(t2)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==========================================
# 5. DATA VISUALIZATION FUNCTIONS
# ==========================================

def plot_professional_dashboard(drugs, mechs, mri, genes, records):
    """Creates a high-readability 6-panel clinical view."""
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0d1117')
    color = "red" if mri > 0.3 else "orange" if mri > 0.15 else "green"
    
    for a in ax.flat:
        a.set_facecolor('#161b22')
        a.tick_params(colors='white', labelsize=10)
        a.title.set_color('white')

    # Drugs Pie
    d_c = Counter(drugs).most_common(6)
    if d_c: ax[0,0].pie([v for k,v in d_c], labels=[k[:15] for k,v in d_c], autopct='%1.1f%%', textprops={'color':"w"})
    
    # Mechs Bar
    m_c = Counter(mechs)
    if m_c: 
        ax[0,1].bar([k[:12] for k in m_c.keys()], m_c.values(), color=color)
        plt.setp(ax[0,1].get_xticklabels(), rotation=30, ha='right')

    # Indicator
    ax[0,2].axis('off')
    ax[0,2].text(0.5, 0.5, f"{round(mri, 3)}", color=color, fontsize=55, ha='center', fontweight='bold')
    ax[0,2].text(0.5, 0.2, "MRI SCORE", color='white', fontsize=15, ha='center')

    # Top Genes Barh
    g_c = Counter([r[0] for r in records]).most_common(5)
    if g_c: ax[1,0].barh([k[:15] for k,v in g_c], [v for k,v in g_c], color='#58a6ff')
    
    ax[1,1].bar(["Total Genes"], [genes], color='#1f6feb')
    ax[1,2].bar(["Unique Drugs", "Unique Mechs"], [len(set(drugs)), len(set(mechs))], color=['#bc8cff', '#ffa657'])
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return fig, buf

# ==========================================
# 6. APP EXECUTION
# ==========================================

# Hero Banner
st.markdown(f"""
    <div class="hero-banner">
        <h1 class="hero-title">AI-MRI INTELLIGENCE HUB</h1>
        <p style="color: #8b949e; margin-top:10px;">Professional Genomic Quantification & Risk Analysis</p>
        <p style="color: #58a6ff; font-size: 0.8rem; margin-top: 15px;">
            <b>AUTHORS:</b> Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik, and Indranil Patil
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Logic
with st.sidebar:
    st.header("⚙️ Core Controls")
    json_files = sorted([f for f in os.listdir('.') if f.endswith('.json')])
    if not json_files:
        st.error("No JSON data found.")
        st.stop()
    selected = st.selectbox("📂 Access Genome Database", json_files)
    st.divider()
    bg_mode = st.radio("Background Context:", ["Dark (Medical)", "White (Scientific)"])
    st.success("AI Brain Connected")

# Load and Process Data
data = parse_genomic_json(selected)
if data:
    genes, drugs, mechs, mri, ari, records = data
    level, icon = get_level(mri)
    meta = get_extended_metadata(selected)
    
    # --- AI PREDICTOR INTEGRATION ---
    ai_model = initialize_ai_predictor()
    if ai_model:
        # Features for current strain
        features = [[genes, len(set(drugs)), len(set(mechs))]]
        ai_prediction = ai_model.predict(features)[0]
    else:
        ai_prediction = "Insuff. Data"

    # --- WOW METRIC CARDS (WRAPPED) ---
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="wow-card"><div class="wow-label">MRI RISK SCORE</div><div class="wow-value">{round(mri,3)}</div><div class="wow-delta" style="color:red;">{level} {icon}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="wow-card"><div class="wow-label">ARI DENSITY</div><div class="wow-value">{round(ari,3)}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="wow-card"><div class="wow-label">AI PREDICTED RISK</div><div class="wow-value">{ai_prediction}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="wow-card"><div class="wow-label">TOTAL GENE LOAD</div><div class="wow-value">{genes}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Tabs
    tabs = st.tabs(["🚀 Summary", "📊 Dashboard", "🧮 Math Ledger", "🕸️ Network", "🤖 Bio-AI Chat", "🌌 3D Landscape", "📄 Master PDF", "💡 Simulation"])

    with tabs[0]:
        st.markdown("### 🧬 Clinical Pathogen Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Strain:** `{selected}`\n\n**Gram:** {meta['gram']}\n\n**Common Pathology:** {meta['disease']}")
        with col2:
            st.success(f"**AI Prediction:** {ai_prediction} Category\n\n**Mechanism Strategies:** {len(set(mechs))}\n\n**Habitat:** {meta['habitat']}")
        st.divider()
        st.markdown("**Bio-AI Insight:** This pathogen displays redundant genetic strategies, making clinical neutralization difficult.")

    with tabs[1]:
        st.subheader("📊 Systems Analytical Dashboard")
        fig, dash_buf = plot_professional_dashboard(drugs, mechs, mri, genes, records)
        st.pyplot(fig)

    with tabs[3]:
        st.subheader("🕸️ Resistance Node Network")
        net_bg = '#0d1117' if bg_mode == "Dark (Medical)" else '#ffffff'
        net_font = 'white' if bg_mode == "Dark (Medical)" else 'black'
        net = Network(height='550px', width='100%', bgcolor=net_bg, font_color=net_font)
        net.add_node("HUB", label=selected, color="red", size=25)
        for r in records[:40]:
            net.add_node(r[0], label=r[0][:12], color="#87CEEB")
            net.add_edge("HUB", r[0])
        components.html(net.generate_html(), height=600)

    with tabs[4]:
        st.subheader("🤖 Bio-AI Chat")
        if AI_AVAILABLE:
            user_msg = st.chat_input("Query Bio-AI...")
            if user_msg:
                try:
                    model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    resp = model.generate_content(f"Pathogen: {selected}, MRI: {mri}. Q: {user_msg}")
                    st.chat_message("assistant").write(resp.text)
                except Exception as e:
                    if "429" in str(e): st.error("🚀 **BIO-AI IS RECHARGING.** Quota reached. Wait 60s.")
                    else: st.error(f"Error: {e}")

    with tabs[6]:
        st.subheader("📄 Master Clinical Export")
        if st.button("🚀 GENERATE MASTER PDF", type="primary"):
            with st.spinner("Compiling Evidence..."):
                pdf_blob = create_advanced_master_pdf(selected, {"mri":mri, "ari":ari, "genes":genes}, records, dash_buf, meta, ai_prediction)
                st.download_button("📥 DOWNLOAD CLINICAL REPORT", pdf_blob, file_name=f"Report_{selected}.pdf")

    with tabs[7]:
        st.subheader("💡 Treatment Efficacy Simulator")
        st.write("Simulate how neutralizing specific mechanisms impacts the strain's mathematical risk.")
        neu_slider = st.slider("Mechanisms to neutralize via new treatment:", 0, len(set(mechs)), 0)
        sim_mri = (len(set(drugs)) + (len(set(mechs)) - neu_slider)) / (len(drugs) + len(mechs) + 1)
        sim_lvl = get_level(sim_mri)[0]
        
        st.markdown(f"""
        <div style="background-color: #161b22; padding: 20px; border-radius: 15px; border: 1px solid #30363d;">
            <h4>Simulation Outcomes</h4>
            <p>New Predicted MRI: <b style="color:#58a6ff;">{round(sim_mri, 3)}</b></p>
            <p>New Risk Category: <b style="color:{"red" if sim_lvl=="HIGH" else "green"};">{sim_lvl}</b></p>
            <p>Efficacy Gain: {round(((mri-sim_mri)/mri)*100, 1)}% improvement</p>
        </div>
        """, unsafe_allow_html=True)
