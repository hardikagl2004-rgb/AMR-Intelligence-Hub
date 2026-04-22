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

# PDF Engine Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# ==========================================
# 1. CORE BRAIN INITIALIZATION (MUST BE FIRST)
# ==========================================

# Variables must be defined at the absolute top to avoid NameErrors
AI_AVAILABLE = False
try:
    import google.generativeai as genai
    genai.configure(api_key="AIzaSyAlQFR5IVkp1TS4pg9_LvP0E3BogEh4Z_U")
    AI_AVAILABLE = True
except Exception:
    AI_AVAILABLE = False

# ==========================================
# 2. ANALYTICAL FUNCTIONS
# ==========================================

def get_level(mri):
    if mri < 0.15: return "LOW", "🟢"
    if mri < 0.35: return "MODERATE", "🟡"
    return "HIGH", "🔴"

@st.cache_data
def get_bac_meta(file_name):
    name = file_name.lower()
    meta = {"gram": "Variable", "disease": "Opportunistic Infection", "habitat": "General"}
    if any(k in name for k in ["ecoli", "escherichia", "shigella"]):
        meta = {"gram": "Negative (-)", "disease": "Gastroenteritis, Sepsis", "habitat": "Clinical"}
    elif "staphylococcus" in name:
        meta = {"gram": "Positive (+)", "disease": "MRSA, Skin Infection", "habitat": "Clinical"}
    return meta

@st.cache_data
def parse_genome(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        drugs, mechs, records = [], [], []
        for g_k, g_v in data.items():
            try:
                node = list(g_v.values())[0] if isinstance(g_v, dict) else g_v
                name = node.get("ARO_name", g_k)
                d_l, m_l = [], []
                cats = node.get("ARO_category", {})
                for c in cats.values():
                    ctype = c.get("category_aro_class_name", "").lower()
                    cval = c.get("category_aro_name", "")
                    if "drug" in ctype: d_l.append(cval)
                    elif "mechanism" in ctype: m_l.append(cval)
                drugs.extend(d_l); mechs.extend(m_l)
                records.append((name, ", ".join(set(d_l)), ", ".join(set(m_l))))
            except: continue
        genes = len(data)
        mri = (len(set(drugs)) + len(set(mechs))) / (len(drugs) + len(mechs) + 1)
        ari = len(set(mechs)) / (genes + 1)
        return genes, drugs, mechs, mri, ari, records
    except: return None

@st.cache_resource
def train_rf_predictor():
    """AI Predictor: Trains Random Forest on local dataset."""
    X, y = [], []
    files = [f for f in os.listdir('.') if f.endswith('.json')]
    for f in files:
        res = parse_genome(f)
        if res:
            X.append([res[0], len(set(res[1])), len(set(res[2]))])
            y.append(get_level(res[3])[0])
    if len(X) < 3: return None
    return RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

# ==========================================
# 3. PROFESSIONAL VISUALIZATION
# ==========================================

def plot_dashboard(drugs, mechs, mri, genes, records):
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0d1117')
    color = "red" if mri > 0.3 else "orange" if mri > 0.15 else "green"
    for a in ax.flat:
        a.set_facecolor('#161b22'); a.tick_params(colors='white', labelsize=10); a.title.set_color('white')
    
    # 1. Drugs Pie
    d_c = Counter(drugs).most_common(5)
    if d_c: ax[0,0].pie([v for k,v in d_c], labels=[k[:15] for k,v in d_c], autopct='%1.1f%%', textprops={'color':"w"})
    # 2. Mechanisms Bar
    m_c = Counter(mechs)
    if m_c: 
        ax[0,1].bar([k[:12] for k in m_c.keys()], m_c.values(), color=color)
        plt.setp(ax[0,1].get_xticklabels(), rotation=30, ha='right')
    # 3. MRI Big Indicator
    ax[0,2].axis('off')
    ax[0,2].text(0.5, 0.5, f"{round(mri, 3)}", color=color, fontsize=60, ha='center', fontweight='bold')
    # 4. Top Genes
    g_c = Counter([r[0] for r in records]).most_common(5)
    if g_c: ax[1,0].barh([k[:15] for k,v in g_c], [v for k,v in g_c], color='#58a6ff')
    # 5. Genetic Density
    ax[1,1].bar(["Total ARGs"], [genes], color='#1f6feb')
    # 6. Unique Strategy Count
    ax[1,2].bar(["Unique Drugs", "Unique Mechs"], [len(set(drugs)), len(set(mechs))], color=['#bc8cff', '#ffa657'])
    
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=150); buf.seek(0)
    return fig, buf

# ==========================================
# 4. FRONTEND STYLING
# ==========================================
st.set_page_config(page_title="AI-MRI Hub v2.0", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@400;600&display=swap');
    .main { background-color: #0d1117; color: #c9d1d9; }
    .hero-banner {
        background: linear-gradient(90deg, #1f6feb 0%, #111d2c 100%);
        padding: 45px; border-radius: 20px; text-align: center; margin-bottom: 30px;
    }
    .wow-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 15px;
        padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    .wow-value { font-family: 'Orbitron', sans-serif; color: #58a6ff; font-size: 2.3rem; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 5. MASTER PDF ENGINE
# ==========================================

def build_pdf(file_name, stats, records, dash_img, meta, ai_pred):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    title_s = ParagraphStyle(name='T', fontSize=22, textColor=colors.HexColor('#1f6feb'), alignment=1, spaceAfter=20)
    
    els = []
    els.append(Paragraph("AI-MRI MASTER CLINICAL REPORT", title_s))
    els.append(Paragraph(f"<b>Pathogen ID:</b> {file_name}", styles['Normal']))
    els.append(Paragraph(f"<b>AI Predicted Risk:</b> {ai_pred}", styles['Normal']))
    
    # Summary Table
    t_data = [["MRI Score", "ARI Density", "Total Genes", "Gram Status"], [f"{round(stats[0], 3)}", f"{round(stats[1], 3)}", f"{stats[2]}", meta['gram']]]
    t1 = Table(t_data, colWidths=[1.5*inch]*4)
    t1.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f6feb')), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
    els.append(Spacer(1, 15)); els.append(t1)
    
    # Image
    els.append(Spacer(1, 20)); els.append(Paragraph("I. Clinical Analytics Dashboard", styles['Heading2']))
    els.append(RLImage(dash_img, width=6*inch, height=4*inch))
    
    # Ledger
    els.append(PageBreak()); els.append(Paragraph("II. Full Genetic Ledger", styles['Heading2']))
    l_data = [["Gene Name", "Resisted Targets", "Mechanism"]]
    for r in records[:60]:
        l_data.append([Paragraph(r[0], styles['Normal']), Paragraph(r[1], styles['Normal']), Paragraph(r[2], styles['Normal'])])
    t2 = Table(l_data, colWidths=[1.2*inch, 2.4*inch, 2.4*inch])
    t2.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.black), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('FONTSIZE', (0,0), (-1,-1), 8)]))
    els.append(t2)
    
    doc.build(els); buf.seek(0); return buf

# ==========================================
# 6. APPLICATION FLOW
# ==========================================

st.markdown('<div class="hero-banner"><h1 class="hero-title">AI-MRI INTELLIGENCE HUB</h1></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Controls")
    files = sorted([f for f in os.listdir('.') if f.endswith('.json')])
    selected = st.selectbox("📂 Genome Archive", files)
    bg_mode = st.radio("Background Context:", ["Dark (Medical)", "White (Scientific)"])
    st.success("AI Brain Initialized")

data = parse_genome(selected)
if data:
    genes, drugs, mechs, mri, ari, records = data
    level, icon = get_level(mri)
    meta = get_bac_meta(selected)
    
    # AI Predictor
    ai_model = train_rf_predictor()
    ai_pred = ai_model.predict([[genes, len(set(drugs)), len(set(mechs))]])[0] if ai_model else "Insuff. Data"

    # WOW Metrics
    st.markdown("---")
    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
    with m_c1: st.markdown(f'<div class="wow-card"><div>MRI RISK</div><div class="wow-value">{round(mri,3)}</div><div style="color:red">{level}</div></div>', unsafe_allow_html=True)
    with m_c2: st.markdown(f'<div class="wow-card"><div>ARI DENSITY</div><div class="wow-value">{round(ari,3)}</div></div>', unsafe_allow_html=True)
    with m_c3: st.markdown(f'<div class="wow-card"><div>AI PREDICTION</div><div class="wow-value">{ai_pred}</div></div>', unsafe_allow_html=True)
    with m_c4: st.markdown(f'<div class="wow-card"><div>GENE LOAD</div><div class="wow-value">{genes}</div></div>', unsafe_allow_html=True)
    st.markdown("---")

    tabs = st.tabs(["🚀 Summary", "📊 Dashboard", "🧮 Math Ledger", "🕸️ Network", "🤖 Bio-AI Chat", "🌌 3D Landscape", "📄 Master PDF", "💡 Simulation"])

    with tabs[0]:
        st.info(f"**Pathogen:** `{selected}` | **Gram:** {meta['gram']} | **Pathology:** {meta['disease']}")
    
    with tabs[1]:
        fig, dash_buf = plot_dashboard(drugs, mechs, mri, genes, records)
        st.pyplot(fig)

    with tabs[2]:
        st.success(f"**Calculation:** ({len(set(drugs))} + {len(set(mechs))}) / ({len(drugs)} + {len(mechs)} + 1) = {round(mri, 3)}")
        st.dataframe(pd.DataFrame(records, columns=["Gene", "Target", "Mech"]), use_container_width=True)

    with tabs[3]:
        n_bg = '#0d1117' if bg_mode == "Dark (Medical)" else '#ffffff'
        n_ft = 'white' if bg_mode == "Dark (Medical)" else 'black'
        net = Network(height='500px', width='100%', bgcolor=n_bg, font_color=n_ft)
        net.add_node("HUB", label=selected, color="red", size=25)
        for r in records[:40]:
            net.add_node(r[0], label=r[0][:12], color="#87CEEB")
            net.add_edge("HUB", r[0])
        components.html(net.generate_html(), height=550)

    with tabs[4]:
        if AI_AVAILABLE:
            u_q = st.chat_input("Query Bio-AI...")
            if u_q:
                try:
                    chat = genai.GenerativeModel('gemini-1.5-flash-latest')
                    resp = chat.generate_content(f"Pathogen: {selected}, MRI: {mri}. Q: {u_q}")
                    st.chat_message("assistant").write(resp.text)
                except Exception as e:
                    if "429" in str(e): st.error("🚀 **AI RECHARGING.** Wait 60s.")
                    else: st.error(f"Error: {e}")

    with tabs[5]:
        c_list = []
        for f in files:
            res = parse_genome(f)
            if res: c_list.append([res[3], len(set(res[2])), len(set(res[1])), f])
        df_3d = pd.DataFrame(c_list, columns=['X', 'Y', 'Z', 'N'])
        st.plotly_chart(px.scatter_3d(df_3d, x='X', y='Y', z='Z', hover_name='N'), use_container_width=True)

    with tabs[6]:
        if st.button("🚀 GENERATE MASTER PDF"):
            with st.spinner("Compiling Evidence..."):
                pdf_blob = build_pdf(selected, [mri, ari, genes], records, dash_buf, meta, ai_pred)
                st.download_button("📥 DOWNLOAD CLINICAL REPORT", pdf_blob, file_name=f"Report_{selected}.pdf")

    with tabs[7]:
        st.subheader("💡 Treatment Impact Simulator")
        neu = st.slider("Neutralize Mechanisms:", 0, len(set(mechs)), 0)
        s_mri = (len(set(drugs)) + (len(set(mechs)) - neu)) / (len(drugs) + len(mechs) + 1)
        st.write(f"Baseline MRI: {round(mri, 3)} → **New MRI: {round(s_mri, 3)}**")
