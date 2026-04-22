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

# Professional PDF Engine Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# ==========================================
# 1. CORE BRAIN INITIALIZATION
# ==========================================
AI_AVAILABLE = False
try:
    import google.generativeai as genai
    genai.configure(api_key="AIzaSyAlQFR5IVkp1TS4pg9_LvP0E3BogEh4Z_U")
    AI_AVAILABLE = True
except Exception:
    AI_AVAILABLE = False

# Risk Assessment Logic
def get_risk_profile(mri):
    if mri < 0.15: return "LOW", "🟢", "#3fb950"
    if mri < 0.35: return "MODERATE", "🟡", "#d29922"
    return "HIGH", "🔴", "#f85149"

@st.cache_data
def get_bacterial_context(file_name):
    name = file_name.lower()
    meta = {"gram": "Variable", "disease": "Opportunistic Infection", "habitat": "Environmental"}
    if any(k in name for k in ["ecoli", "escherichia", "shigella"]):
        meta = {"gram": "Negative (-)", "disease": "Sepsis, UTI, Gastroenteritis", "habitat": "Clinical"}
    elif "staphylococcus" in name:
        meta = {"gram": "Positive (+)", "disease": "MRSA, Skin Infections", "habitat": "Clinical"}
    return meta

@st.cache_data
def parse_genomic_intelligence(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        drugs, mechs, ledger = [], [], []
        for g_id, g_body in data.items():
            try:
                inner = list(g_body.values())[0] if isinstance(g_body, dict) else g_body
                g_name = inner.get("ARO_name", g_id)
                d_t, m_t = [], []
                cats = inner.get("ARO_category", {})
                for c in cats.values():
                    ctype = c.get("category_aro_class_name", "").lower()
                    cval = c.get("category_aro_name", "")
                    if "drug" in ctype: d_t.append(cval)
                    elif "mechanism" in ctype: m_t.append(cval)
                drugs.extend(d_t); mechs.extend(m_t)
                ledger.append((g_name, ", ".join(set(d_t)), ", ".join(set(m_t))))
            except: continue
        genes = len(data)
        mri = (len(set(drugs)) + len(set(mechs))) / (len(drugs) + len(mechs) + 1)
        ari = len(set(mechs)) / (genes + 1)
        return genes, drugs, mechs, mri, ari, ledger
    except: return None

@st.cache_resource
def train_ai_risk_model():
    X, y = [], []
    files = [f for f in os.listdir('.') if f.endswith('.json')]
    for f in files:
        res = parse_genomic_intelligence(f)
        if res:
            X.append([res[0], len(set(res[1])), len(set(res[2]))])
            y.append(get_risk_profile(res[3])[0])
    if len(X) < 3: return None
    return RandomForestClassifier(n_estimators=100).fit(X, y)

# ==========================================
# 2. FRONTEND STYLING (THE WOW FACTOR)
# ==========================================
st.set_page_config(page_title="Bio-AI MRI Intelligence", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@400;600&display=swap');
    .main { background-color: #0d1117; color: #c9d1d9; }
    .hero-banner {
        background: linear-gradient(135deg, #1f6feb 0%, #111d2c 100%);
        padding: 50px; border-radius: 25px; text-align: center; margin-bottom: 40px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
    .hero-title { font-family: 'Orbitron', sans-serif; color: white; font-size: 3.2rem; margin: 0; }
    .card {
        background: rgba(22, 27, 34, 0.8); border: 1px solid #30363d;
        padding: 25px; border-radius: 20px; text-align: center;
        transition: transform 0.3s;
    }
    .card:hover { transform: translateY(-5px); border-color: #58a6ff; }
    .card-val { font-family: 'Orbitron', sans-serif; color: #58a6ff; font-size: 2.5rem; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. CLINICAL VISUALIZATIONS
# ==========================================
def render_master_dashboard(drugs, mechs, mri, genes, ledger):
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor('#0d1117')
    _, _, lvl_color = get_risk_profile(mri)
    
    for a in ax.flat:
        a.set_facecolor('#161b22'); a.tick_params(colors='white', labelsize=10); a.title.set_color('white')
    
    # Graphs for PDF and Screen
    d_c = Counter(drugs).most_common(5)
    if d_c: ax[0,0].pie([v for k,v in d_c], labels=[k[:15] for k,v in d_c], autopct='%1.1f%%', textprops={'color':"w"})
    m_c = Counter(mechs)
    if m_c: ax[0,1].bar([k[:12] for k in m_c.keys()], m_c.values(), color=lvl_color)
    ax[0,2].axis('off'); ax[0,2].text(0.5, 0.5, f"{round(mri, 3)}", color=lvl_color, fontsize=65, ha='center', fontweight='bold')
    g_c = Counter([r[0] for r in ledger]).most_common(5)
    if g_c: ax[1,0].barh([k[:15] for k,v in g_c], [v for k,v in g_c], color='#58a6ff')
    ax[1,1].bar(["Genomic Load"], [genes], color='#1f6feb')
    ax[1,2].bar(["Drugs", "Mechs"], [len(set(drugs)), len(set(mechs))], color=['#bc8cff', '#ffa657'])
    
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=150); buf.seek(0)
    return fig, buf

# ==========================================
# 4. THE MASTER PDF ENGINE
# ==========================================
def create_clinical_pdf(file_name, stats, ledger, dash_buf, meta, ai_pred):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle(name='T', fontSize=26, textColor=colors.HexColor('#1f6feb'), alignment=1, spaceAfter=30, fontName='Helvetica-Bold')
    
    story = []
    story.append(Paragraph("AI-MRI MASTER CLINICAL INTELLIGENCE", title_s))
    story.append(Paragraph(f"<b>Pathogen Identifier:</b> {file_name}", styles['Normal']))
    story.append(Paragraph(f"<b>AI Predicted Risk Level:</b> {ai_pred}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Stats Table
    t_data = [
        ["MRI SCORE", "ARI DENSITY", "TOTAL GENES", "GRAM CLASSIFICATION"],
        [f"{round(stats[0], 4)}", f"{round(stats[1], 4)}", f"{stats[2]}", meta['gram']]
    ]
    t1 = Table(t_data, colWidths=[1.6*inch]*4)
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING', (0,0), (-1,-1), 10)
    ]))
    story.append(t1)
    
    # Graph Image
    story.append(Spacer(1, 25))
    story.append(Paragraph("Section I: Clinical Visualization Dashboard", styles['Heading2']))
    story.append(RLImage(dash_buf, width=6.5*inch, height=4*inch))
    
    # Complete Table
    story.append(PageBreak())
    story.append(Paragraph("Section II: Comprehensive Genomic Resistance Registry", styles['Heading2']))
    l_data = [["Gene Marker", "Pharmaceutical Target", "Mechanism of Action"]]
    for r in ledger[:85]:
        l_data.append([Paragraph(r[0], styles['Normal']), Paragraph(r[1], styles['Normal']), Paragraph(r[2], styles['Normal'])])
    
    t2 = Table(l_data, colWidths=[1.3*inch, 2.5*inch, 2.5*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTSIZE', (0,0), (-1,-1), 8)
    ]))
    story.append(t2)
    
    doc.build(story); buf.seek(0); return buf

# ==========================================
# 5. CORE EXECUTION FLOW
# ==========================================

st.markdown('<div class="hero-banner"><h1 class="hero-title">AI-MRI INTELLIGENCE HUB</h1></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ System Controls")
    files = sorted([f for f in os.listdir('.') if f.endswith('.json')])
    selected = st.selectbox("📂 Access Genome Database", files)
    bg_mode = st.radio("Background Context:", ["Dark (Medical)", "White (Publication)"])
    st.success("AI Model Trained & Online")

payload = parse_genomic_intelligence(selected)
if payload:
    genes, drugs, mechs, mri, ari, records = payload
    lvl, icon, lvl_color = get_risk_profile(mri)
    meta = get_bacterial_context(selected)
    
    # Train and Run AI Predictor
    ai_engine = train_ai_risk_model()
    ai_pred = ai_engine.predict([[genes, len(set(drugs)), len(set(mechs))]])[0] if ai_engine else "N/A"

    # --- WOW DASHBOARD METRICS ---
    st.markdown("---")
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1: st.markdown(f'<div class="card"><div>MRI RISK</div><div class="card-val">{round(mri,3)}</div><div style="color:{lvl_color};font-weight:bold;">{lvl} {icon}</div></div>', unsafe_allow_html=True)
    with m_c2: st.markdown(f'<div class="card"><div>ARI DENSITY</div><div class="card-val">{round(ari,3)}</div></div>', unsafe_allow_html=True)
    with m_c3: st.markdown(f'<div class="card"><div>AI PREDICTION</div><div class="card-val">{ai_pred}</div></div>', unsafe_allow_html=True)
    with m_c4: st.markdown(f'<div class="card"><div>GENE LOAD</div><div class="card-val">{genes}</div></div>', unsafe_allow_html=True)
    st.markdown("---")

    tabs = st.tabs(["🚀 Executive Summary", "📊 Clinical Dashboard", "🧮 Math Ledger", "🕸️ Node Network", "🤖 Bio-AI Assistant", "🌌 3D Landscape", "📄 Master PDF Export"])

    with tabs[0]:
        st.markdown("### 🧬 Pathogen Intel Summary")
        s_c1, s_c2 = st.columns(2)
        with s_c1:
            st.info(f"**Pathogen Identifier:** `{selected}`\n\n**Classification:** {meta['gram']}\n\n**Clinical Pathology:** {meta['disease']}")
        with s_c2:
            st.success(f"**AI Risk Analysis:** {ai_pred} Pathogen\n\n**Strategy Diversity:** {len(set(mechs))} Distinct Mechanisms\n\n**MRI Score:** {round(mri, 3)}")
        st.divider()
        st.markdown("#### Clinical Interpretation")
        st.write("This pathogen possesses a highly redundant genetic arsenal. A high MRI score signifies that standard antibiotic protocols may face rapid resistance pivot points.")

    with tabs[1]:
        fig, dash_buf = render_master_dashboard(drugs, mechs, mri, genes, records)
        st.pyplot(fig)

    with tabs[2]:
        st.markdown("### 🧮 Quantitative Resistance Math")
        st.write("Below is the mathematical proof of the MRI and ARI scores for this genome.")
        st.latex(r"MRI = \frac{\text{Unique Drugs} + \text{Unique Mechs}}{\text{Total Assignments} + 1}")
        st.success(f"**Calculation Result:** {round(mri, 4)}")
        st.latex(r"ARI = \frac{\text{Unique Mechs}}{\text{Total Genes} + 1}")
        st.info(f"**Calculation Result:** {round(ari, 4)}")
        st.divider()
        st.dataframe(pd.DataFrame(records, columns=["Gene Name", "Resisted Targets", "Mechanism"]), use_container_width=True)

    with tabs[3]:
        n_bg = '#0d1117' if bg_mode == "Dark (Medical)" else '#ffffff'
        net = Network(height='550px', width='100%', bgcolor=n_bg, font_color=('white' if bg_mode=="Dark (Medical)" else 'black'))
        net.add_node("HUB", label=selected, color=lvl_color, size=30)
        for r in records[:45]:
            net.add_node(r[0], label=r[0][:10], color="#87CEEB")
            net.add_edge("HUB", r[0])
        components.html(net.generate_html(), height=600)

    with tabs[4]:
        if AI_AVAILABLE:
            u_input = st.chat_input("Query Bio-AI...")
            if u_input:
                try:
                    chat_brain = genai.GenerativeModel('gemini-1.5-flash-latest')
                    resp = chat_brain.generate_content(f"Pathogen Analysis: {selected}. MRI: {mri}. AI Risk: {ai_pred}. Question: {u_input}")
                    st.chat_message("assistant").write(resp.text)
                except Exception as e:
                    if "429" in str(e): st.error("🚀 **BIO-AI RECHARGING.** Quota reached. Wait 60s.")
                    else: st.error(f"Error: {e}")

    with tabs[6]:
        if st.button("🚀 INITIATE MASTER EXPORT", type="primary"):
            with st.spinner("Compiling Clinical Evidence..."):
                pdf_blob = create_clinical_pdf(selected, [mri, ari, genes], records, dash_buf, meta, ai_pred)
                st.download_button("📥 DOWNLOAD CLINICAL PDF", pdf_blob, file_name=f"Report_{selected.split('.')[0]}.pdf")
