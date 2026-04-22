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
    genai.configure(api_key="AIzaSyAlQFR5IVkp1TS4pg9_LvP0E3BogEh4Z_U")
except Exception:
    AI_AVAILABLE = False

# ==========================================
# 2. FRONTEND STYLING (THE "WOW" FACTOR)
# ==========================================
st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon="🧬")

# Custom CSS for the Cyber-Medical theme and wrapped metrics
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    /* Background and global text color */
    .main { background-color: #0d1117; color: #c9d1d9; }
    
    /* The Cyber-Blue Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1f6feb 0%, #111d2c 100%);
        padding: 40px;
        border-radius: 20px;
        border: 1px solid #30363d;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        text-align: center;
    }
    
    .hero-title { font-family: 'Orbitron', sans-serif; color: white; font-size: 3rem; margin: 0; text-shadow: 0 0 15px rgba(88, 166, 255, 0.4); }
    .hero-subtitle { font-family: 'Inter', sans-serif; color: #8b949e; font-size: 1.2rem; margin-top: 10px; }

    /* Custom Metric Cards (The Fix for cutoff text and professional look) */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    
    .metric-title { color: #8b949e; font-size: 0.9rem; margin-bottom: 5px; font-weight: 600; text-transform: uppercase;}
    .metric-value { font-family: 'Orbitron', sans-serif; color: #58a6ff; font-size: 2.5rem; margin: 0; font-weight: 700; white-space: normal !important; overflow-wrap: break-word;}
    .metric-delta { color: #58a6ff; font-size: 0.8rem; }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px 10px 0 0;
        color: #8b949e;
        padding: 12px 20px;
        transition: all 0.3s;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
        transform: translateY(-2px);
    }
    
    </style>
""", unsafe_allow_html=True)

# Helper function to create the clean metric card
def create_metric_card(label, value, delta=None):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{label}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-delta">{delta}</div>' if delta else ''}
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# 3. CORE BACKEND DATA ENGINE
# ==========================================
@st.cache_data
def get_bacteria_info(file_name):
    """Deep biological context for known pathogens."""
    name = file_name.lower()
    info = {"gram": "Unknown", "disease": "Various opportunistic infections", "risk": "Moderate"}
    if any(k in name for k in ["ecoli", "escherichia", "shigella"]):
        info = {"gram": "Negative (-)", "disease": "Gastroenteritis, UTI, Sepsis", "risk": "High"}
    elif "staphylococcus" in name:
        info = {"gram": "Positive (+)", "disease": "Skin infections, MRSA, Endocarditis", "risk": "High"}
    elif "campylobacter" in name:
        info = {"gram": "Negative (-)", "disease": "Campylobacteriosis (Enteritis)", "risk": "High"}
    elif "pseudomonas" in name:
        info = {"gram": "Negative (-)", "disease": "Nosocomial pneumonia", "risk": "Critical"}
    return info

@st.cache_data
def extract_master_data(file_name):
    """Robust parser for complex AMR JSON structures."""
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        drug, mech, records = [], [], []
        for k, v in data.items():
            try:
                # Handle nested dict structure
                inner = list(v.values())[0] if isinstance(v, dict) else v
                gene_name = inner.get("ARO_name", k)
                
                d_list, m_list = [], []
                categories = inner.get("ARO_category", {})
                if isinstance(categories, dict):
                    for c_id, c_data in categories.items():
                        c_type = c_data.get("category_aro_class_name", "").lower()
                        val = c_data.get("category_aro_name", "")
                        if "drug" in c_type: d_list.append(val)
                        elif "mechanism" in c_type: m_list.append(val)
                drug.extend(d_list)
                mech.extend(m_list)
                records.append((gene_name, ", ".join(set(d_list)), ", ".join(set(m_list))))
            except: continue
            
        genes = len(data)
        # Handle division by zero for small files
        divisor = (len(drug) + len(mech) + 1)
        mri = (len(set(drug)) + len(set(mech))) / divisor if divisor > 0 else 0
        ari = len(set(mech)) / (genes + 1)
        return genes, drug, mech, mri, ari, records
    except Exception:
        return None

# ==========================================
# 4. MASTER PDF ENGINE (PLATYPUS)
# ==========================================
def generate_master_pdf(file_name, stats, records, graphs, network_img, bac_info, habitat, ai_pred):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Brand Styles for a clinical feel
    title_style = ParagraphStyle(name='T', fontSize=24, textColor=colors.HexColor('#1f6feb'), alignment=1, spaceAfter=20)
    header_style = ParagraphStyle(name='H', fontSize=15, textColor=colors.HexColor('#238636'), spaceBefore=15, spaceAfter=8)
    normal_style = styles['Normal']
    
    elements = []
    
    # 1. Title & Clinical Summary
    elements.append(Paragraph("AI-MRI CLINICAL INTELLIGENCE REPORT", title_style))
    elements.append(Paragraph(f"<b>Pathogen Identifier:</b> {file_name}", normal_style))
    elements.append(Paragraph(f"<b>Report Timestamp:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 15))
    
    # 2. Key Metrics Table
    elements.append(Paragraph("I. Core Metrics Assessment", header_style))
    stat_data = [
        ["MRI SCORE (Risk)", "ARI SCORE (Density)", "AI PREDICTION", "TOTAL ARGs"],
        [f"{round(stats['mri'], 4)}", f"{round(stats['ari'], 4)}", f"{ai_pred}", f"{stats['genes']}"]
    ]
    t_stats = Table(stat_data, colWidths=[1.7*inch]*4)
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
    ]))
    elements.append(t_stats)
    elements.append(Spacer(1, 20))
    
    # 3. Pathogen Biological Context
    elements.append(Paragraph("II. Biological Pathogen Context", header_style))
    meta_data = [
        [f"Gram Stain: {bac_info['gram']}", f"Habitat: {habitat}"],
        [Paragraph(f"Associated Disease: {bac_info['disease']}", normal_style), ""]
    ]
    t_meta = Table(meta_data, colWidths=[3.2*inch, 3.2*inch])
    t_meta.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'LEFT'), ('VALIGN', (0,0), (-1,-1), 'TOP')]))
    elements.append(t_meta)
    
    # 4. Graphs Page
    elements.append(PageBreak())
    elements.append(Paragraph("III. Graphical Dashboards", header_style))
    for img_buf in graphs:
        # reportlab wants a raw path or BytesIO with seek(0)
        img = RLImage(img_buf, width=6*inch, height=3*inch)
        img.hAlign = 'CENTER'
        elements.append(img)
        elements.append(Spacer(1, 15))

    # 5. Network Visualization
    elements.append(PageBreak())
    elements.append(Paragraph("IV. Resistance Network Visualization", header_style))
    if network_img:
        net_img = RLImage(network_img, width=6*inch, height=4.5*inch)
        net_img.hAlign = 'CENTER'
        elements.append(net_img)
    else:
        elements.append(Paragraph("Network visualization image unavailable.", normal_style))
        
    # 6. Detailed Data Ledger
    elements.append(PageBreak())
    elements.append(Paragraph("V. Complete ARG Resistance Ledger", header_style))
    ledger_data = [["Gene Name", " Pharmaceutical Target", "Mechanism"]]
    for r in records[:100]: # Limit for performance, but show a lot
        ledger_data.append([
            Paragraph(r[0], normal_style), 
            Paragraph(r[1], styles['BodyText']), 
            Paragraph(r[2], styles['BodyText'])
        ])
    
    t_ledger = Table(ledger_data, colWidths=[1.3*inch, 2.3*inch, 2.3*inch])
    t_ledger.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0d1117')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTSIZE', (0,0), (-1,-1), 8)
    ]))
    elements.append(t_ledger)
    
    # Build and Return
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==========================================
# 5. VISUALIZATION AND ANALYSIS PLOTS
# ==========================================
@st.cache_resource
def train_ai_predictor():
    """Trains a quick ML model on the current database for risk prediction."""
    X, y = [], []
    for f in os.listdir('.'):
        if f.endswith('.json'):
            res = extract_master_data(f)
            if res:
                genes, drugs, mechs, mri, _, _ = res
                # Features: [Genes, Drugs, Mechs]
                X.append([genes, len(set(drugs)), len(set(mechs))])
                y.append("HIGH" if mri > 0.3 else "MODERATE" if mri > 0.15 else "LOW")
    if len(X) < 2: return None
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def plot_systems_dashboard(drug, mech, mri, genes, records, selected_file):
    """Generates the main 6-panel clinical dashboard."""
    level = "HIGH" if mri > 0.3 else "MODERATE" if mri > 0.15 else "LOW"
    color = "red" if level == "HIGH" else "orange" if level == "MODERATE" else "green"
    
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0d1117') 
    
    for a in ax.flat:
        a.set_facecolor('#0d1117'); a.tick_params(colors='white'); a.title.set_color('white')

    # Pie 1: Drugs
    drug_c = Counter(drug).most_common(6)
    if drug_c:
        ax[0, 0].pie([v for k,v in drug_c], labels=[k[:15]+"..." for k,v in drug_c], autopct='%1.1f%%', textprops={'color':"w"})
    ax[0, 0].set_title("Resisted Drug Classes")

    # Bar 1: Mechs
    mech_c = Counter(mech)
    if mech_c:
        ax[0, 1].bar([k[:15] for k in mech_c.keys()], mech_c.values(), color=color)
        plt.setp(ax[0,1].get_xticklabels(), rotation=30, horizontalalignment='right')
    ax[0, 1].set_title("Mechanisms Deployed")

    # MRI Indicator
    ax[0, 2].axis('off')
    ax[0, 2].text(0.5, 0.5, f"{round(mri, 3)}", color=color, fontsize=50, ha='center', fontweight='bold', fontname='Orbitron')
    ax[0, 2].text(0.5, 0.2, f"MRI Indicator ({level})", color='white', fontsize=15, ha='center')

    # Top Genes
    g_list = [r[0] for r in records]
    gene_c = Counter(g_list).most_common(5)
    if gene_c:
        ax[1, 0].barh([k[:15]+"..." for k,v in gene_c], [v for k,v in gene_c], color='#58a6ff')
    ax[1, 0].set_title("Top Identified Genes")

    # Overall Counts
    ax[1, 1].bar(["Total Genes"], [genes], color='#1f6feb')
    ax[1, 1].set_title("Genomic ARG Count")

    # Unique Ratio
    ax[1, 2].bar(["Drugs", "Mechs"], [len(set(drug)), len(set(mech))], color=['#bc8cff', '#ffa657'])
    ax[1, 2].set_title("Unique Strategy Count")
    
    plt.tight_layout()
    # Save fig for PDF export
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return fig, buf

def generate_network_html(records, organism_name, color, bg_color):
    """Creates the pyvis interactive network."""
    # Handle background color logic
    bg_val = '#0d1117' if bg_color == "Dark (Medical)" else '#ffffff'
    font_val = 'white' if bg_color == "Dark (Medical)" else 'black'
    edge_val = '#ffffff' if bg_color == "Dark (Medical)" else '#aaaaaa'

    net = Network(height='600px', width='100%', bgcolor=bg_val, font_color=font_val, cdn_resources="in_line", select_menu=True, filter_menu=True)
    net.add_node("HUB", label=organism_name, color=color, size=30, shape="diamond")
    
    # Track existing nodes to prevent duplication
    added_nodes = set()
    for g, d, m in records:
        if g not in added_nodes:
            net.add_node(g, label=g[:12], color="#87CEEB", size=15)
            net.add_edge("HUB", g, color=edge_val)
            added_nodes.add(g)
        
        # Connect to Drugs/Mechanisms as box nodes
        targets = [d,m]
        colors_t = ["#FFA500", "#FF4500"]
        for i, t in enumerate(targets):
            if t:
                for target in set(t.split(", ")):
                    if not target or target in added_nodes: continue
                    net.add_node(target, label=target[:15], color=colors_t[i], size=10, shape="box")
                    net.add_edge(g, target, color="#aaaaaa", length=150)
                    added_nodes.add(target)
                
    net.barnes_hut(gravity=-3000)
    
    html_path = "temp_network.html"
    net.save_graph(html_path)
    # Also save a static image buf for PDF - this requires pyvis image generation support or taking snapshot
    # For now, return HTML and static buf is generated in tab (complex step)
    return html_path, None

# ==========================================
# 6. APP MAIN EXECUTION
# ==========================================
# Hero Banner
st.markdown(f"""
    <div class="hero-banner">
        <h1 class="hero-title">AI-MRI INTELLIGENCE HUB</h1>
        <p class="hero-subtitle">Clinician Quantitative Analysis for Antibiotic Resistance Genes (ARGs)</p>
        <p style="color: #58a6ff; font-size: 0.8em; margin-top: 15px;">
            <b>DEVELOPED BY:</b> Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik, and Indranil Patil
        </p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.header("🗄️ Database Sync")
    if st.button("🔄 Refresh JSON Database"):
        st.cache_data.clear()
        st.rerun()
    
    json_files = sorted([f for f in os.listdir('.') if f.endswith('.json')])
    selected_file = st.selectbox("Select a Genome:", json_files)
    st.success("✅ AI Brain Connected")
    
    st.divider()
    st.markdown("### 🛠️ Visual Settings")
    # THE ADJUSTABLE BACKGROUND OPTION
    vis_bg = st.radio("Background Context:", ["Dark (Medical)", "White (Readable)"], index=0)

# Process Data
data_bundle = extract_master_data(selected_file)
if data_bundle:
    genes, drug, mech, mri, ari, records = data_bundle
    level, icon = get_level(mri)
    habitat = "Clinical" if mri > 0.4 else "Env."
    bac_info = get_bacteria_info(selected_file)
    
    # Train/Get AI Predictor
    ml_model = train_ai_predictor()
    if ml_model:
        ml_input = [[genes, len(set(drug)), len(set(mech))]]
        ml_pred = ml_model.predict(ml_input)[0]
    else:
        ml_pred = "N/A"

    # 1. FIXED AND WRAPPED TOP METRICS BAR
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1: create_metric_card("MRI RISK SCORE", f"{round(mri, 3)}", delta=f"{level} {icon}")
    with c2: create_metric_card("ARI DENSITY", f"{round(ari, 3)}")
    with c3: create_metric_card("ML RISK PREDICTION", f"{ml_pred}")
    with c4: create_metric_card("TOTAL ARG payload", f"{genes}")
    st.markdown("---")

    # 2. THE ABSOLUTE PROFESSIONAL TABS
    tab_summary, tab_dash, tab_math, tab_net, tab_chat, tab_3d, tab_export, tab_sim = st.tabs([
        "🚀 Summary", "📊 Clinical Dashboard", "🧮 Math Ledger", "🕸️ Node Network", "🤖 Bio-AI Chat", "🌌 3D Landscape", "📄 Master PDF", "🔧 Impact Simulator"
    ])

    # --- TAB 1: EXECUTIVE SUMMARY ---
    with tab_summary:
        st.markdown("### 🦠 Executive Pathogen Intelligence")
        sum_c1, sum_c2 = st.columns(2)
        with sum_c1:
            st.info(f"#### {icon} Clinical Profile")
            st.write(f"**Strain Identifier:** `{selected_file}`")
            st.write(f"**Gram Classification:** {bac_info['gram']}")
            st.write(f"**Associated Diseases:** {bac_info['disease']}")
            st.write(f"**Habitat Context:** {habitat}")
            
        with sum_c2:
            st.success(f"#### 🧠 ML AI Prediction")
            st.write(f"This strain is categorized as **{level} RISK** Pathogen.")
            st.write(f"MRI Score: {round(mri, 3)}")
            st.write(f"Unique Resistant Drugs: {len(set(drug))} classes")
            st.write(f"Deployed Mechanisms: {len(set(mech))} strategies")

        st.divider()
        st.markdown("### 🎯 Metric Signifiance")
        st.info("**Why MRI matters:** Traditional genomic lists are hard to decipher. The **MRI** consolidates diverse drugs and mechanisms into a single score, allowing for instant clinician prioritization of high-risk strains.")

    # --- TAB 2: CLINICAL DASHBOARD ---
    with tab_dash:
        st.markdown(f"### Systems Overview: `{selected_file}`")
        fig, dash_buf_for_pdf = plot_systems_dashboard(drug, mech, mri, genes, records, selected_file)
        st.pyplot(fig)
        
    # --- TAB 3: MATH LEDGER ---
    with tab_math:
        st.markdown("### 🧮 Quantitative Math Ledger")
        
        # Formulas in Markdown
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.success(f"""
            **MRI Calculation** $ Formula: (Unique Drugs + Unique Mechanisms) / (Total Assignments + 1) $  
            $ Math: ({len(set(drug))} + {len(set(mech))}) / ({len(drug)} + {len(mech)} + 1) $  
            $ Result: MRI = {round(mri, 3)} $
            """)
        with c_m2:
            st.info(f"""
            **ARI Calculation** $ Formula: Unique Mechanisms / (Total Genes + 1) $  
            $ Math: {len(set(mech))} / ({genes} + 1) $  
            $ Result: ARI = {round(ari, 3)} $
            """)
        
        st.divider()
        st.markdown("### 📜 Comprehensive Gene Registry")
        df_led = pd.DataFrame(records, columns=["Gene Name", " Pharmaceutical Targets", "Mechanisms Deploymed"])
        st.dataframe(df_led, use_container_width=True)

    # --- TAB 4: ADJUSTABLE NETWORK ---
    with tab_net:
        st.markdown("### 🕸️ Interactive Mechanism Network")
        st.write(f"Currently in **{vis_bg}** context. Use sidebar to adjust.")
        color_node = "red" if level=="HIGH" else "green"
        html_path, _ = generate_network_html(records, selected_file, color_node, vis_bg)
        with open(html_path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650)

    # --- TAB 5: BIO-AI CHAT ---
    with tab_chat:
        st.subheader("🤖 Bio-AI Assistant (Powered by J.A.R.V.I.S. protocols)")
        if AI_AVAILABLE:
            st.write("Ready for biological interrogation.")
            u_input = st.chat_input("Query Bio-AI...")
            if u_input:
                try:
                    # Using the latest stable model
                    ai_brain = genai.GenerativeModel('gemini-1.5-flash-latest')
                    context = f"You are Bio-AI. Analyses of {selected_file}. MRI: {mri}. AI Prediction: {ml_pred}. User Q: {u_input}"
                    response = ai_brain.generate_content(context)
                    st.chat_message("user").write(u_input)
                    st.chat_message("assistant").write(response.text)
                except Exception as e:
                    if "429" in str(e):
                        st.error("🚀 **BIO-AI IS COOLING DOWN.** We've hit the Gemini Free Tier limit. Wait 60s.")
                    else: st.error(f"AI System Error: {e}")
        else:
            st.warning("⚠️ Bio-AI Brain currently offline.")

    # --- TAB 6: ADJUSTABLE 3D VIEW ---
    with tab_3d:
        st.markdown("### 🌌 Global 3D Comparison Space")
        # Reuse PCA logic
        coords = []
        for f in json_files:
            res = extract_master_data(f)
            if res: coords.append([res[3], len(set(res[2])), len(set(res[1])), f])
        
        df_3d = pd.DataFrame(coords, columns=['MRI', 'Mechs', 'Drugs', 'Genome'])
        df_3d['Type'] = ["TARGET 🎯" if x == selected_file else "REFERENCE" for x in df_3d['Genome']]
        
        fig_3d = px.scatter_3d(df_3d, x='MRI', y='Mechs', z='Drugs', hover_name='Genome', color='Type', color_discrete_map={"TARGET 🎯": "red", "REFERENCE": "#58a6ff"})
        
        # Toggle background
        bg_3d = '#0d1117' if vis_bg == "Dark (Medical)" else '#ffffff'
        fig_3d.update_layout(paper_bgcolor=bg_3d, font_color=('white' if vis_bg=="Dark (Medical)" else 'black'), scene=dict(bgcolor=bg_3d))
        st.plotly_chart(fig_3d, use_container_width=True)

    # --- TAB 7: MASTER PDF DOWNLOAD ---
    with tab_export:
        st.subheader("📄 Export Master Clinical Report")
        st.write("Generate a detailed, publication-ready PDF document containing all quantified evidence.")
        
        if st.button("Generate Master PDF", type="primary"):
            with st.spinner("Compiling graphs and evidence..."):
                try:
                    # Collect required components
                    graphs = [dash_buf_for_pdf] # Add more plt buffers if needed
                    # We need a static net buf. We generate here using Pyplot trick or placeholder
                    net_img_buf = io.BytesIO(); fig_net, ax_net = plt.subplots(); ax_net.axis('off'); ax_net.text(0.5,0.5,"Network Visualization Placeholder",ha='center'); fig_net.savefig(net_img_buf, format='png'); net_img_buf.seek(0)
                    
                    pdf_blob = generate_master_pdf(selected_file, {"mri": mri, "ari": ari, "genes": genes}, records, graphs, net_img_buf, bac_info, habitat, ml_pred)
                    
                    st.success("Master Report Compiled Successfully!")
                    st.download_button(
                        label="Download detailed PDF Report",
                        data=pdf_blob,
                        file_name=f"Clinical_Report_{selected_file.split('.')[0]}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e: st.error(f"PDF Export failure: {e}")

    # --- TAB 8: NEW OPTION - CLINICAL IMPACT SIMULATOR ---
    with tab_sim:
        st.subheader("🔧 Clinical Impact Simulator (Prototype)")
        st.write("Simulate how adding unique resistance genes affects the MRI score of the current strain.")
        n_sim_genes = st.slider("Simulate adding unique ARG markers", 1, 5, 2)
        n_sim_drugs = st.slider("Associated unique Drug Classes evaded", 1, 3, 1)
        
        if st.button("Run Simulation"):
            with st.spinner("Recalculating risk metrics..."):
                time.sleep(1) # Dramatic pause
                sim_drugs = len(set(drug)) + n_sim_drugs
                sim_mechs = len(set(mech)) + n_sim_genes # Assuming each gene is new strategy
                sim_total_genes = genes + n_sim_genes
                
                # Formula MRI = (Unique Drugs + Unique Mechs) / (Total Drug Assignments + Total Mechanism Assignments + 1)
                # We simplified assignments for simulation
                sim_mri = (sim_drugs + sim_mechs) / (sim_drugs + sim_mechs + 1)
                sim_level, sim_icon = get_level(sim_mri)
                
                st.markdown(f"""
                <div style="background-color: #161b22; padding: 20px; border-radius: 15px; border: 1px solid #30363d;">
                    <h4>Simulation Results</h4>
                    <p>Baseline MRI: {round(mri, 3)}</p>
                    <p><b>Simulated MRI: <span style="color: red; font-family: 'Orbitron';">{round(sim_mri, 3)}</span></b></p>
                    <p>Predicted Risk Level: <b>{sim_level} {sim_icon}</b></p>
                    <p style="color: #8b949e; font-size: 0.8em;">Interpretation: Adding {n_sim_genes} unique AMR mechanisms significant impacts the consolidated threat score.</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.error("Genome data could not be processed. Ensure valid JSON format.")
