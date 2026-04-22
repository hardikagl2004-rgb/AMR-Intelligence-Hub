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

# PDF Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# --- AI BRAIN INITIALIZATION ---
try:
    import google.generativeai as genai
    AI_AVAILABLE = True
    # Your validated API Key
    genai.configure(api_key="AIzaSyDMo6ff3yVEsR9WkPWd5BWh6cz_gT0cMZ8")
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="AI-MRI Hub v4.0", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Inter:wght@400;600&display=swap');
    .main {background-color: #0e1117;}
    .hero-banner {
        background: linear-gradient(135deg, #1f6feb 0%, #111d2c 100%);
        padding: 40px; border-radius: 20px; text-align: center; margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .hero-title { font-family: 'Orbitron', sans-serif; color: white; font-size: 2.8rem; margin: 0; }
    .stMetric { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE BACKEND FUNCTIONS
# ==========================================
@st.cache_data
def get_bacteria_info(file_name):
    name = file_name.lower()
    info = {"gram": "Unknown", "disease": "Various opportunistic infections"}
    if any(k in name for k in ["ecoli", "escherichia", "shigella"]):
        info = {"gram": "Negative (-)", "disease": "Gastroenteritis, UTI, Sepsis"}
    elif "staphylococcus" in name:
        info = {"gram": "Positive (+)", "disease": "Skin infections, MRSA, Endocarditis"}
    elif "bacillus" in name:
        info = {"gram": "Positive (+)", "disease": "Anthrax, Food poisoning"}
    elif "listeria" in name:
        info = {"gram": "Positive (+)", "disease": "Listeriosis, Foodborne illness"}
    return info

@st.cache_data
def get_habitat(file_name):
    name = file_name.lower()
    if any(k in name for k in ["ecoli","staph","salmonella","streptococcus"]): return "Clinical"
    if any(k in name for k in ["pseudomonas","acinetobacter"]): return "Environmental"
    return "General"

@st.cache_data
def extract_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    drug, mech, records = [], [], []
    habitat = get_habitat(file_name)
    for k in data:
        try:
            inner = list(data[k].values())[0] if isinstance(data[k], dict) else data[k]
            gene_name = inner.get("ARO_name", k)
            d, m = [], []
            for c in inner.get("ARO_category", {}).values():
                cname = c.get("category_aro_class_name","").lower()
                val = c.get("category_aro_name","")
                if "drug" in cname: d.append(val)
                elif "mechanism" in cname: m.append(val)
            drug.extend(d); mech.extend(m)
            records.append((gene_name, ", ".join(set(d)), ", ".join(set(m)), habitat))
        except: continue
    genes = len(data)
    mri = (len(set(drug)) + len(set(mech))) / (len(drug) + len(mech) + 1)
    ari = len(set(mech)) / (genes + 1)
    return genes, drug, mech, mri, ari, records

def get_level(mri):
    if mri < 0.15: return "LOW", "🟢"
    elif mri < 0.35: return "MODERATE", "🟡"
    else: return "HIGH", "🔴"

def get_risk_reason(level, u_drugs, u_mechs):
    if "HIGH" in level:
        return f"This strain utilizes multiple redundant strategies ({u_mechs} mechanisms) to block diverse threats ({u_drugs} drugs). High MRI indicates standard clinical treatments are likely to face resistance pivot points."
    return f"This strain shows localized adaptation against {u_drugs} drug classes."

@st.cache_resource
def train_rf_model():
    X, y = [], []
    for f in [f for f in os.listdir('.') if f.endswith('.json')]:
        try:
            g, d, m, mr, _, _ = extract_data(f)
            X.append([g, len(set(d)), len(set(m))])
            y.append(get_level(mr)[0])
        except: continue
    if len(X) < 2: return None
    return RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================
def plot_full_dashboard(drug, mech, mri, genes, records, name):
    level, icon = get_level(mri)
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0e1117') 
    color = "#f85149" if level == "HIGH" else "#d29922" if level == "MODERATE" else "#3fb950"
    for a in ax.flat:
        a.set_facecolor('#161b22'); a.tick_params(colors='white'); a.title.set_color('white')
    drug_c = Counter(drug).most_common(6)
    if drug_c: ax[0, 0].pie([v for k,v in drug_c], labels=[k[:15] for k,v in drug_c], autopct='%1.1f%%', textprops={'color':"w"})
    mech_c = Counter(mech)
    if mech_c: ax[0, 1].bar([k[:12] for k in mech_c.keys()], mech_c.values(), color=color)
    ax[0, 2].axis('off')
    ax[0, 2].text(0.5, 0.5, f"{round(mri, 3)}", color=color, fontsize=50, ha='center', fontweight='bold')
    ax[1, 1].bar(["Total Genes"], [genes], color='#2E86C1')
    ax[1, 2].bar(["Unique Drugs", "Unique Mechs"], [len(set(drug)), len(set(mech))], color=["#9B59B6", "#E67E22"])
    fig.tight_layout()
    return fig

# ==========================================
# 4. MASTER PDF GENERATOR
# ==========================================
def create_advanced_pdf_report(bac_name, genes, drug, mech, mri, ari, level, icon, records, dashboard_fig, bac_info, habitat, ai_pred_text, ai_conf_text):
    pdf_file = f"{bac_name.replace('.json', '')}_Detailed_Report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='T', fontSize=18, textColor=colors.HexColor('#1E3A8A'))
    elements = []
    elements.append(Paragraph(f"AI-MRI Report: {bac_name}", title_style))
    sum_t = f"<b>Risk:</b> {level} | <b>AI Prediction:</b> {ai_pred_text}<br/><b>Gram:</b> {bac_info['gram']}<br/><b>MRI:</b> {round(mri,3)}"
    elements.append(Paragraph(sum_t, styles['Normal']))
    buf = io.BytesIO()
    dashboard_fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    elements.append(RLImage(buf, width=7*inch, height=4.5*inch))
    doc.build(elements)
    return pdf_file

# ==========================================
# 5. FRONTEND: THE WEBSITE LAYOUT
# ==========================================
st.markdown('<div class="hero-banner"><h1 class="hero-title">AI-MRI INTELLIGENCE HUB</h1></div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Developed by: Hardik Agrawal et al.</p>", unsafe_allow_html=True)

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}; st.session_state.current_session = "Chat 1"; st.session_state.chat_counter = 1

with st.sidebar:
    st.header("🗄️ Database Sync")
    json_files = sorted([f for f in os.listdir('.') if f.endswith('.json')])
    analysis_mode = st.radio("Mode:", ["Select Known Bacteria", "AI Predict Unknown"])
    if analysis_mode == "Select Known Bacteria":
        selected_file = st.selectbox("Select a Genome:", json_files)
    st.success("✅ AI Brain Connected")

if analysis_mode == "Select Known Bacteria" and json_files:
    genes, drug, mech, mri, ari, records = extract_data(selected_file)
    u_drugs, u_mechs = len(set(drug)), len(set(mech))
    level, icon = get_level(mri)
    habitat = get_habitat(selected_file)
    bac_info = get_bacteria_info(selected_file)

    model = train_rf_model()
    ai_pred_text = "N/A"
    if model:
        pred = model.predict([[genes, u_drugs, u_mechs]])[0]
        ai_pred_text = str(pred)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🚀 Summary", "📊 Dashboard", "🧮 Math Ledger", "🕸️ Network", "🤖 Bio-AI", "📄 Master PDF", "🌌 3D Landscape"
    ])

    with tab1:
        st.markdown("<h2 style='text-align: center;'>🧬 Executive Pathogen Intelligence</h2>", unsafe_allow_html=True)
        risk_color = "#f85149" if level == "HIGH" else "#d29922" if level == "MODERATE" else "#3fb950"
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1: st.metric("MRI Risk Score", f"{round(mri, 3)}", delta=level)
        with col_m2: st.metric("ARI Density", f"{round(ari, 3)}")
        with col_m3: st.metric("Total ARG Payload", genes)
        with col_m4: st.metric("Gram Stain", bac_info.get('gram', 'N/A'))

        st.divider()
        cl1, cl2 = st.columns(2)
        with cl1:
            st.markdown("#### 🦠 Biological Identification")
            st.write(f"**Strain:** `{selected_file}`")
            st.write(f"**Associated Disease:** {bac_info.get('disease', 'N/A')}")
        with cl2:
            st.markdown("#### 🏥 Clinical Evidence")
            st.write(f"**Drug Classes Evaded:** {u_drugs}")
            st.markdown(f"**AI Prediction:** <span style='background-color:{risk_color}; color:white; padding:3px 8px; border-radius:5px;'>{ai_pred_text} RISK</span>", unsafe_allow_html=True)
        st.info(f"**Clinician Interpretation:** This strain is categorized as **{level} RISK**. Redundant strategies complicate standard clinical therapy.")

    with tab2:
        fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)
        st.pyplot(fig)
        
    with tab3:
        st.markdown("### 🧮 Quantitative Analytical Ledger")
        cm1, cm2 = st.columns(2)
        with cm1:
            st.markdown("#### MRI Calculation")
            st.latex(r"MRI = \frac{Unique Drugs + Unique Mechs}{Total Assignments + 1}")
            st.success(f"Result: **{round(mri, 3)}**")
        with cm2:
            st.markdown("#### ARI Calculation")
            st.latex(r"ARI = \frac{Unique Mechanisms}{Total Genes + 1}")
            st.info(f"Result: **{round(ari, 3)}**")
        st.divider()
        st.dataframe(pd.DataFrame(records, columns=["Gene", "Drugs", "Mechanism", "Habitat"]), use_container_width=True)

    with tab4:
        html_path = generate_network_html(records, selected_file, "red" if level=="HIGH" else "green")
        with open(html_path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650)

    with tab5:
        if AI_AVAILABLE:
            u_input = st.chat_input("Ask Bio-AI...")
            if u_input:
                model_ai = genai.GenerativeModel('gemini-1.5-flash-latest')
                resp = model_ai.generate_content(f"Pathogen: {selected_file}, MRI: {mri}. Q: {u_input}")
                st.chat_message("assistant").write(resp.text)

    with tab6:
        if st.button("Generate Master PDF", type="primary"):
            pdf_path = create_advanced_pdf_report(selected_file, genes, drug, mech, mri, ari, level, icon, records, fig, bac_info, habitat, ai_pred_text, "N/A")
            with open(pdf_path, "rb") as f:
                st.download_button("Download Report", f, file_name=pdf_path)

    with tab7:
        plot_3d_pca_plotly(selected_file)

elif analysis_mode == "AI Predict Unknown":
    st.header("🤖 ML Risk Prediction")
    ig, idr, im = st.columns(3)
    in_g = ig.number_input("Total Genes", value=15)
    in_d = idr.number_input("Unique Drugs", value=5)
    in_m = im.number_input("Unique Mechanisms", value=2)
    model = train_rf_model()
    if model and st.button("Predict Risk"):
        st.success(f"### AI Prediction: **{model.predict([[in_g, in_d, in_m]])[0]}**")
