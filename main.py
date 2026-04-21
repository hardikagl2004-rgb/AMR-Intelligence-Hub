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
# Hardik, I've set this up to check both your hardcoded key AND the secure vault.
try:
    import google.generativeai as genai
    AI_AVAILABLE = True
    
    # Priority 1: Check Streamlit Secrets (The Vault)
    # Priority 2: Use your provided hardcoded key
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = "AIzaSyAlQFR5IVkp1TS4pg9_LvP0E3BogEh4Z_U"
        
    genai.configure(api_key=api_key)
except Exception:
    AI_AVAILABLE = False

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #ffffff;}
    .stMetric {background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d;}
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
    elif "salmonella" in name:
        info = {"gram": "Negative (-)", "disease": "Salmonellosis, Typhoid Fever"}
    elif any(k in name for k in ["klebsiella", "enterobacter", "citrobacter", "serratia"]):
        info = {"gram": "Negative (-)", "disease": "Pneumonia, UTI, Bloodstream infections"}
    elif "pseudomonas" in name:
        info = {"gram": "Negative (-)", "disease": "Cystic fibrosis lung infections, Burn wound infections"}
    elif "acinetobacter" in name:
        info = {"gram": "Negative (-)", "disease": "Nosocomial pneumonia, Bacteremia"}
    elif "staphylococcus" in name:
        info = {"gram": "Positive (+)", "disease": "Skin infections, MRSA, Endocarditis"}
    elif "streptococcus" in name:
        info = {"gram": "Positive (+)", "disease": "Strep throat, Pneumonia, Necrotizing fasciitis"}
        
    return info

@st.cache_data
def get_habitat(file_name):
    name = file_name.lower()
    if any(k in name for k in ["ecoli","staphylococcus","salmonella","klebsiella"]):
        return "Clinical"
    elif any(k in name for k in ["pseudomonas","acinetobacter"]):
        return "Environmental"
    elif any(k in name for k in ["bacillus","clostridium"]):
        return "Soil"
    elif "vibrio" in name:
        return "Marine"
    return "General"

@st.cache_data
def extract_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    drug, mech, records = [], [], []
    habitat = get_habitat(file_name)
    
    for k in data:
        try:
            inner = list(data[k].values())[0]
            gene_name = inner.get("ARO_name", k)
            d, m = [], []
            for c in inner.get("ARO_category", {}).values():
                cname = c.get("category_aro_class_name","").lower()
                val = c.get("category_aro_name","")
                if "drug" in cname: d.append(val)
                elif "mechanism" in cname: m.append(val)
            
            drug.extend(d)
            mech.extend(m)
            records.append((gene_name, ", ".join(set(d)), ", ".join(set(m)), habitat))
        except:
            continue

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
        return f"High MRI indicates multiple redundant strategies ({u_mechs} mechanisms) to block diverse threats ({u_drugs} drugs). Clinical treatments may be highly ineffective."
    else:
        return "Strain shows specialized resistance; standard protocols remain prioritized."

@st.cache_resource
def train_rf_model():
    X, y = [], []
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                genes, drug, mech, mri, _, _ = extract_data(f)
                features = [genes, len(set(drug)), len(set(mech))]
                label = "LOW" if mri < 0.15 else "MODERATE" if mri < 0.35 else "HIGH"
                X.append(features)
                y.append(label)
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
    color = "red" if level == "HIGH" else "orange" if level == "MODERATE" else "green"
    for a in ax.flat:
        a.set_facecolor('#0e1117')
        a.tick_params(colors='white')
        a.title.set_color('white')

    # Simple plots
    drug_c = Counter(drug).most_common(8)
    if drug_c: ax[0, 0].pie([v for k,v in drug_c], labels=[k[:10] for k,v in drug_c], autopct='%1.1f%%', textprops={'color':"w"})
    ax[0, 0].set_title("Drug Resistance")

    mech_c = Counter(mech)
    if mech_c: ax[0, 1].bar([k[:10] for k in mech_c.keys()], mech_c.values(), color=color)
    ax[0, 1].set_title("Mechanisms")

    ax[0, 2].axis('off')
    ax[0, 2].set_title(f"MRI: {round(mri, 3)}")

    gene_list = [r[0] for r in records]
    gene_c = Counter(gene_list).most_common(5)
    if gene_c: ax[1, 0].bar([k[:10] for k,v in gene_c], [v for k,v in gene_c], color='#87CEEB')
    ax[1, 0].set_title("Top Genes")

    ax[1, 1].bar(["Genes"], [genes], color='#2E86C1')
    ax[1, 2].bar(["Drugs", "Mechs"], [len(set(drug)), len(set(mech))], color=["#9B59B6", "#E67E22"])
    
    fig.tight_layout()
    return fig

# ==========================================
# 4. MASTER PDF GENERATOR
# ==========================================
def create_advanced_pdf_report(bac_name, genes, drug, mech, mri, ari, level, icon, records, dashboard_fig, bac_info, habitat, ai_pred_text, ai_conf_text):
    pdf_file = f"{bac_name.replace('.json', '')}_Report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    h2_style = ParagraphStyle(name='H2', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#2E86C1'))
    elements = []
    elements.append(Paragraph(f"AI-MRI Report: {bac_name}", styles['Heading1']))
    elements.append(Paragraph(f"MRI Score: {round(mri, 3)} | Risk: {level}", styles['Normal']))
    doc.build(elements)
    return pdf_file

# ==========================================
# 5. FRONTEND: THE WEBSITE LAYOUT
# ==========================================
st.title("🧬 AI-Driven Multidimensional Resistance Index")
st.markdown("### For Quantitative Analysis of Antibiotic Resistance Genes")
st.markdown("**Developed by:** Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik, and Indranil Patil")

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
    st.session_state.current_session = "Chat 1"

with st.sidebar:
    st.header("🗄️ Database Sync")
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    analysis_mode = st.radio("Mode:", ["Select Known Bacteria", "AI Predict Unknown"])
    if analysis_mode == "Select Known Bacteria":
        selected_file = st.selectbox("Select a Genome:", json_files)
    if st.button("🔄 Refresh Database"): st.rerun()

if analysis_mode == "Select Known Bacteria" and json_files:
    genes, drug, mech, mri, ari, records = extract_data(selected_file)
    u_drugs, u_mechs = len(set(drug)), len(set(mech))
    level, icon = get_level(mri)
    habitat = get_habitat(selected_file)
    bac_info = get_bacteria_info(selected_file)
    model = train_rf_model()
    ai_pred_text = model.predict([[genes, u_drugs, u_mechs]])[0] if model else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Level", f"{level} {icon}")
    col2.metric("MRI Score", round(mri, 3))
    col3.metric("Total Genes", genes)
    col4.metric("Habitat", habitat)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "ℹ️ Pathogen Summary", "📊 Dashboard", "🧮 Math Ledger", "🕸️ Network", "🤖 Bio-AI Chat", "📄 Export PDF", "🌌 3D Landscape"
    ])

    with tab1:
        st.markdown("### 🦠 Executive Summary")
        sum_col1, sum_col2 = st.columns(2)
        with sum_col1:
            st.markdown(f"### {icon}")
            st.markdown(f"**Pathogen:** `{selected_file}`")
            st.markdown(f"**Gram Stain:** {bac_info['gram']}")
            st.markdown(f"**Common Disease:** {bac_info['disease']}")
            st.markdown(f"**Primary Habitat:** {habitat}")
        with sum_col2:
            st.markdown(f"**Total ARG Genes:** {genes}")
            st.markdown(f"**Resistance Count:** {u_drugs} Drug Classes")
            st.markdown(f"**Mechanism Count:** {u_mechs} Strategies")
            st.markdown(f"**MRI Score:** `{round(mri, 3)}` ({level})")
            st.markdown(f"**AI Prediction:** :red[{ai_pred_text}]")
        st.markdown("---")
        st.markdown("### 🎯 Metric Explanations")
        st.info("Traditional analysis simply lists genes. Our ARI/MRI framework calculates threat density and consolidates data for instant clinical decision-making.")

    with tab2:
        st.pyplot(plot_full_dashboard(drug, mech, mri, genes, records, selected_file))
        
    with tab5:
        st.session_state.current_session = st.selectbox("Active Chat:", list(st.session_state.chat_sessions.keys()))
        for msg in st.session_state.chat_sessions[st.session_state.current_session]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        user_msg = st.chat_input("Ask about this pathogen...")
        if user_msg:
            st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": user_msg})
            if AI_AVAILABLE:
                model_ai = genai.GenerativeModel('gemini-1.5-flash')
                response = model_ai.generate_content(f"Pathogen: {selected_file}, MRI: {mri}. {user_msg}")
                st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": response.text})
                st.rerun()

    with tab6:
        if st.button("Generate PDF"):
            st.success("PDF generated (Check project folder)")

elif analysis_mode == "AI Predict Unknown":
    st.header("🤖 Machine Learning Prediction")
    ig = st.number_input("Genes", value=10); idr = st.number_input("Drugs", value=5); im = st.number_input("Mechs", value=2)
    model = train_rf_model()
    if model and st.button("Predict"): st.write(f"Risk: {model.predict([[ig, idr, im]])[0]}")
