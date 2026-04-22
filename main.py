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
from sklearn.decomposition import PCA
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
# 1. CORE SYSTEM INITIALIZATION
# ==========================================

# Secure AI Brain Setup
try:
    import google.generativeai as genai
    AI_AVAILABLE = True
    # Using your validated high-stability key
    genai.configure(api_key="AIzaSyAlQFR5IVkp1TS4pg9_LvP0E3BogEh4Z_U")
except Exception:
    AI_AVAILABLE = False

# Professional Page Config
st.set_page_config(
    page_title="Bio-AI MRI Intelligence Hub",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. ADVANCED FRONTEND CSS (THE "WOW" FACTOR)
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    /* Global Background */
    .main { background-color: #0d1117; color: #c9d1d9; }
    
    /* Professional Title Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1f6feb 0%, #111d2c 100%);
        padding: 45px;
        border-radius: 20px;
        border: 1px solid #30363d;
        margin-bottom: 35px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.6);
        text-align: center;
    }
    
    .hero-title {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
        font-size: 3.2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 20px rgba(88, 166, 255, 0.4);
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        color: #8b949e;
        font-size: 1.3rem;
        margin-top: 15px;
        font-weight: 300;
    }

    /* Metric Card Styling */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        color: #58a6ff !important;
        font-size: 2.4rem !important;
    }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
        transform: translateY(-2px);
    }
    
    /* Footer Styling */
    .footer-text {
        text-align: center;
        color: #484f58;
        font-size: 0.8rem;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. CLINICAL DATA ENGINE
# ==========================================

@st.cache_data
def get_extended_metadata(file_name):
    """Provides deep biological context for known pathogens."""
    name = file_name.lower()
    meta = {
        "gram": "Gram Variable",
        "disease": "Opportunistic Infection",
        "habitat": "General Environment",
        "severity": "Moderate"
    }
    if any(x in name for x in ["ecoli", "escherichia", "shigella"]):
        meta.update({"gram": "Negative (-)", "disease": "Gastroenteritis, UTI, Sepsis", "habitat": "Clinical/Human Gut", "severity": "High"})
    elif "staphylococcus" in name:
        meta.update({"gram": "Positive (+)", "disease": "Skin Infection, MRSA, Endocarditis", "habitat": "Clinical/Skin", "severity": "High"})
    elif "campylobacter" in name:
        meta.update({"gram": "Negative (-)", "disease": "Campylobacteriosis (Severe Enteritis)", "habitat": "Zoonotic/Clinical", "severity": "High"})
    elif "pseudomonas" in name:
        meta.update({"gram": "Negative (-)", "disease": "Nosocomial Pneumonia", "habitat": "Environmental/Clinical", "severity": "Critical"})
    return meta

@st.cache_data
def process_genomic_json(file_path):
    """Robust parser for complex AMR JSON structures."""
    try:
        if not os.path.exists(file_path): return None
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        drug_classes, mechanisms, master_records = [], [], []
        
        for gene_key, gene_val in data.items():
            try:
                # Normalize nested dictionaries
                node = list(gene_val.values())[0] if isinstance(gene_val, dict) else gene_val
                name = node.get("ARO_name", gene_key)
                
                cur_drugs, cur_mechs = [], []
                categories = node.get("ARO_category", {})
                
                if isinstance(categories, dict):
                    for cid, cval in categories.items():
                        c_type = cval.get("category_aro_class_name", "").lower()
                        c_name = cval.get("category_aro_name", "")
                        if "drug" in c_type: cur_drugs.append(c_name)
                        elif "mechanism" in c_type: cur_mechs.append(c_name)
                
                drug_classes.extend(cur_drugs)
                mechanisms.extend(cur_mechs)
                master_records.append((name, ", ".join(set(cur_drugs)), ", ".join(set(cur_mechs))))
            except: continue
            
        total_genes = len(data)
        # MRI: Normalized diversity score
        mri = (len(set(drug_classes)) + len(set(mechanisms))) / (len(drug_classes) + len(mechanisms) + 1)
        # ARI: Mechanism density score
        ari = len(set(mechanisms)) / (total_genes + 1)
        
        return total_genes, drug_classes, mechanisms, mri, ari, master_records
    except Exception: return None

# ==========================================
# 4. ADVANCED VISUALIZATION SUITE
# ==========================================

def render_6_panel_dashboard(drugs, mechs, mri, genes, records):
    """Generates a high-fidelity systems overview."""
    fig, ax = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor('#0d1117')
    
    # Theme settings
    level = "HIGH" if mri > 0.3 else "MODERATE" if mri > 0.15 else "LOW"
    accent = "#f85149" if level == "HIGH" else "#d29922" if level == "MODERATE" else "#3fb950"
    
    for a in ax.flat:
        a.set_facecolor('#161b22')
        a.tick_params(colors='#8b949e', labelsize=8)
        a.title.set_color('#ffffff')

    # Panel 1: Drug Diversity (Pie)
    d_counts = Counter(drugs).most_common(6)
    if d_counts:
        ax[0,0].pie([v for k,v in d_counts], labels=[k[:15] for k,v in d_counts], autopct='%1.1f%%', textprops={'color':"w", 'size':7})
    ax[0,0].set_title("Resisted Drug Classes")

    # Panel 2: Mechanism Profile (Bar)
    m_counts = Counter(mechs)
    if m_counts:
        ax[0,1].bar([k[:12] for k in m_counts.keys()], m_counts.values(), color=accent)
        plt.setp(ax[0,1].get_xticklabels(), rotation=30)
    ax[0,1].set_title("Deployment Strategies")

    # Panel 3: Risk Indicator (Text/Gauge)
    ax[0,2].axis('off')
    ax[0,2].text(0.5, 0.6, f"{round(mri, 3)}", color=accent, fontsize=45, ha='center', fontweight='bold', fontname='Orbitron')
    ax[0,2].text(0.5, 0.3, f"MRI RISK: {level}", color='white', fontsize=15, ha='center')

    # Panel 4: Top Genes
    g_counts = Counter([r[0] for r in records]).most_common(5)
    if g_counts:
        ax[1,0].barh([k[:15] for k,v in g_counts], [v for k,v in g_counts], color='#58a6ff')
    ax[1,0].set_title("Primary ARG Markers")

    # Panel 5: Volume Metrics
    ax[1,1].bar(["Total Genes"], [genes], color='#1f6feb')
    ax[1,1].set_title("Genomic Payload")

    # Panel 6: Diversity Ratio
    ax[1,2].bar(["Unique Drugs", "Unique Mechs"], [len(set(drugs)), len(set(mechs))], color=['#bc8cff', '#ffa657'])
    ax[1,2].set_title("Defense Redundancy")

    plt.tight_layout()
    return fig

# ==========================================
# 5. CLINICAL PDF EXPORT ENGINE
# ==========================================

def build_master_pdf(file_name, genes, mri, ari, records, metadata):
    """Compiles a professional, multi-page clinical report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Brand Styles
    title_s = ParagraphStyle(name='T', fontSize=24, textColor=colors.HexColor('#1f6feb'), alignment=1, spaceAfter=25, fontName='Helvetica-Bold')
    header_s = ParagraphStyle(name='H', fontSize=15, textColor=colors.HexColor('#238636'), spaceBefore=20, spaceAfter=10, fontName='Helvetica-Bold')
    body_s = styles['Normal']
    
    elements = []
    
    # Page 1: Executive Overview
    elements.append(Paragraph("BIO-AI MRI PATHOGEN REPORT", title_s))
    elements.append(Paragraph(f"<b>Pathogen Identifier:</b> {file_name}", body_s))
    elements.append(Paragraph(f"<b>Timestamp:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}", body_s))
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("I. Clinical Risk Summary", header_s))
    risk_level = "HIGH" if mri > 0.3 else "MODERATE" if mri > 0.15 else "LOW"
    
    summary_data = [
        ["Clinical Attribute", "Assessment Data"],
        ["MRI Score (Risk)", f"{round(mri, 4)}"],
        ["ARI Score (Density)", f"{round(ari, 4)}"],
        ["Risk Category", risk_level],
        ["Gram Classification", metadata['gram']],
        ["Total ARG Burden", f"{genes} Identified Genes"],
        ["Primary Disease Context", metadata['disease']]
    ]
    
    t_summary = Table(summary_data, colWidths=[2.2*inch, 3.8*inch])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(t_summary)
    
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Report Disclaimer:", ParagraphStyle(name='D', fontSize=8, textColor=colors.grey)))
    elements.append(Paragraph("This report is generated via the AI-MRI framework and is intended for clinical research purposes only.", ParagraphStyle(name='D', fontSize=8, textColor=colors.grey)))
    
    elements.append(PageBreak())
    
    # Page 2: Genetic Ledger
    elements.append(Paragraph("II. Comprehensive Genetic Ledger", header_s))
    ledger_data = [["Gene Marker", "Pharmaceutical Target(s)", "Mechanism of Action"]]
    
    for r in records:
        ledger_data.append([
            Paragraph(r[0], body_s),
            Paragraph(r[1], body_s),
            Paragraph(r[2], body_s)
        ])
    
    t_ledger = Table(ledger_data, colWidths=[1.3*inch, 2.35*inch, 2.35*inch])
    t_ledger.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#21262d')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    elements.append(t_ledger)
    
    doc.build(elements)
    buf.seek(0)
    return buf

# ==========================================
# 6. APP MAIN EXECUTION FLOW
# ==========================================

# High-Impact Hero Banner
st.markdown(f"""
    <div class="hero-banner">
        <h1 class="hero-title">AI-MRI INTELLIGENCE HUB</h1>
        <p class="hero-subtitle">Advanced Clinical Quantification of Antibiotic Resistance Genes</p>
        <p style="color: #58a6ff; font-size: 0.85rem; margin-top: 20px;">
            <b>DEVELOPED BY:</b> Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik, & Indranil Patil
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.header("🧬 CORE CONTROLS")
    st.info("AI Brain Online & Connected")
    
    all_json = sorted([f for f in os.listdir('.') if f.endswith('.json')])
    if not all_json:
        st.error("No genome data detected in root directory.")
        st.stop()
        
    selected_genome = st.selectbox("📂 SELECT GENOME ARCHIVE", all_json)
    st.divider()
    if st.button("🔄 FORCE SYNC DATABASE"):
        st.cache_data.clear()
        st.rerun()

# Data Processing Pipeline
raw_data = process_genomic_json(selected_genome)
meta_data = get_extended_metadata(selected_genome)

if raw_data:
    n_genes, d_classes, m_strategies, score_mri, score_ari, records_list = raw_data
    risk_lvl, risk_icon = ("HIGH", "🔴") if score_mri > 0.3 else ("MODERATE", "🟡") if score_mri > 0.15 else ("LOW", "🟢")
    
    # 1. PRIMARY METRIC DASHBOARD
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MRI RISK", round(score_mri, 3), delta=risk_lvl)
    col2.metric("ARI DENSITY", round(score_ari, 3))
    col3.metric("ARG PAYLOAD", n_genes)
    col4.metric("GRAM CLASS", meta_data['gram'])

    # 2. MULTI-TAB INTERFACE
    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "🚀 Summary", "📊 6-Panel View", "🧮 Math Ledger", 
        "🕸️ Network", "🤖 Bio-AI", "🌌 3D Landscape", "📄 Export Report"
    ])

    # --- TAB 1: EXECUTIVE SUMMARY ---
    with t1:
        st.markdown("### 🧬 Pathogen Executive Analysis")
        sum_c1, sum_c2 = st.columns(2)
        with sum_c1:
            st.info(f"**Identified Strain:** `{selected_genome}`")
            st.write(f"**Gram Classification:** {meta_data['gram']}")
            st.write(f"**Primary Pathology:** {meta_data['disease']}")
        with sum_c2:
            st.success(f"**MRI Security Status:** {risk_lvl} RISK {risk_icon}")
            st.write(f"**Unique Mechanisms:** {len(set(m_strategies))} Strategies")
            st.write(f"**Clinical Habitat:** {meta_data['habitat']}")
        
        st.divider()
        st.markdown("#### **CLINICAL INTERPRETATION**")
        st.write("""
            The **Multidimensional Resistance Index (MRI)** quantifies the clinical danger of a pathogen. 
            High scores indicate a pathogen with redundant mechanisms, allowing it to easily pivot between 
            defense strategies when exposed to different antibiotic classes.
        """)

    # --- TAB 2: 6-PANEL DASHBOARD ---
    with t2:
        st.markdown(f"### Systems Overview: {selected_genome}")
        with st.spinner("Rendering High-Fidelity Dashboards..."):
            dash_fig = render_6_panel_dashboard(d_classes, m_strategies, score_mri, n_genes, records_list)
            st.pyplot(dash_fig)

    # --- TAB 3: MATH LEDGER ---
    with t3:
        st.markdown("### 🧮 Quantitative Calculation Ledger")
        math_c1, math_c2 = st.columns(2)
        with math_c1:
            st.success(f"**MRI Calc:** `({len(set(d_classes))} + {len(set(m_strategies))}) / ({len(d_classes)+len(m_strategies)} + 1)` = **{round(score_mri, 3)}**")
        with math_c2:
            st.info(f"**ARI Calc:** `{len(set(m_strategies))} / ({n_genes} + 1)` = **{round(score_ari, 3)}**")
        
        st.divider()
        st.markdown("#### Full Resistance Registry")
        df_final = pd.DataFrame(records_list, columns=["Gene Marker", "Targets", "Mechanism"])
        st.dataframe(df_final, use_container_width=True)

    # --- TAB 4: INTERACTIVE NETWORK ---
    with t4:
        st.markdown("### 🕸️ Genomic Interaction Network")
        with st.spinner("Building Node Topography..."):
            net = Network(height='600px', width='100%', bgcolor='#0d1117', font_color='white')
            net.add_node("HUB", label=selected_genome, color="#1f6feb", size=35)
            # Displaying top 40 nodes for performance
            for r in records_list[:40]:
                net.add_node(r[0], label=r[0][:12], color="#238636", size=18)
                net.add_edge("HUB", r[0], color="#30363d")
            components.html(net.generate_html(), height=650)

    # --- TAB 5: BIO-AI CHAT ---
    with t5:
        st.markdown("### 🤖 Bio-AI Assistant")
        if AI_AVAILABLE:
            st.write("Ready for biological interrogation. Ask about specific gene markers or clinical risks.")
            chat_input = st.chat_input("Query the Bio-AI Brain...")
            if chat_input:
                try:
                    ai_brain = genai.GenerativeModel('gemini-1.5-flash-latest')
                    context = f"Genome: {selected_genome}, MRI: {score_mri}, Genes: {n_genes}. Question: {chat_input}"
                    response = ai_brain.generate_content(context)
                    st.chat_message("user").write(chat_input)
                    st.chat_message("assistant").write(response.text)
                except Exception as e:
                    if "429" in str(e):
                        st.error("🚀 **BIO-AI IS COOLING DOWN.** Quota limit reached. Please wait 60 seconds.")
                    else:
                        st.error(f"AI System Error: {e}")
        else:
            st.warning("AI Module currently offline. Check API configuration.")

    # --- TAB 6: 3D LANDSCAPE ---
    with t6:
        st.markdown("### 🌌 Global Comparison Landscape")
        with st.spinner("Calculating Multi-Variant PCA..."):
            pca_data = []
            for f in all_json:
                res = process_genomic_json(f)
                if res:
                    pca_data.append([res[3], len(set(res[2])), len(set(res[1])), f])
            
            if len(pca_data) >= 3:
                df_3d = pd.DataFrame(pca_data, columns=['Intensity', 'Diversity', 'Density', 'Genome'])
                df_3d['Category'] = ["TARGET 🎯" if x == selected_genome else "REFERENCE" for x in df_3d['Genome']]
                
                fig_3d = px.scatter_3d(
                    df_3d, x='Intensity', y='Diversity', z='Density', 
                    hover_name='Genome', color='Category',
                    color_discrete_map={"TARGET 🎯": "#f85149", "REFERENCE": "#58a6ff"}
                )
                fig_3d.update_layout(paper_bgcolor='#0d1117', scene=dict(bgcolor='#0d1117'))
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.info("Additional genome data is required to render 3D space.")

    # --- TAB 7: MASTER PDF EXPORT ---
    with t7:
        st.markdown("### 📄 Master Clinical PDF Export")
        st.write("Generate a formal clinical document containing all quantified data and resistance markers.")
        if st.button("🚀 INITIATE MASTER EXPORT", type="primary"):
            with st.spinner("Compiling Clinical Evidence..."):
                try:
                    pdf_blob = build_master_pdf(selected_genome, n_genes, score_mri, score_ari, records_list, meta_data)
                    st.success("Master Report Compiled Successfully!")
                    st.download_button(
                        label="📥 DOWNLOAD CLINICAL PDF",
                        data=pdf_blob,
                        file_name=f"Clinical_Report_{selected_genome.split('.')[0]}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Export Failure: {e}")

st.markdown("<div class='footer-text'>AI-MRI HUB | PROTECTING GLOBAL HEALTH THROUGH GENOMIC INTELLIGENCE</div>", unsafe_allow_html=True)
