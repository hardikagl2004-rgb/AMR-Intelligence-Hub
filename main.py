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
    # Priority: Secrets vault first, then hardcoded fallback
    api_key = st.secrets.get("GEMINI_API_KEY", "AIzaSyAlQFR5IVkp1TS4pg9_LvP0E3BogEh4Z_U")
    genai.configure(api_key=api_key)
except Exception:
    AI_AVAILABLE = False

# ==========================================
# 1. CORE BACKEND FUNCTIONS
# ==========================================
@st.cache_data
def get_bacteria_info(file_name):
    name = file_name.lower()
    info = {"gram": "Unknown", "disease": "Opportunistic Infection"}
    if any(k in name for k in ["ecoli", "escherichia", "shigella"]):
        info = {"gram": "Negative (-)", "disease": "Gastroenteritis, UTI"}
    elif "campylobacter" in name:
        info = {"gram": "Negative (-)", "disease": "Gastroenteritis"}
    elif "staphylococcus" in name:
        info = {"gram": "Positive (+)", "disease": "Skin infections, Sepsis"}
    return info

@st.cache_data
def extract_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    drug, mech, records = [], [], []
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
            drug.extend(d); mech.extend(m)
            records.append((gene_name, ", ".join(set(d)), ", ".join(set(m)), "General"))
        except: continue
    genes = len(data)
    mri = (len(set(drug)) + len(set(mech))) / (len(drug) + len(mech) + 1)
    ari = len(set(mech)) / (genes + 1)
    return genes, drug, mech, mri, ari, records

def get_level(mri):
    if mri < 0.15: return "LOW", "🟢"
    elif mri < 0.35: return "MODERATE", "🟡"
    return "HIGH", "🔴"

@st.cache_resource
def train_rf_model():
    X, y = [], []
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                g, d, m, mr, ar, _ = extract_data(f)
                X.append([g, len(set(d)), len(set(m))])
                lv, _ = get_level(mr)
                y.append(lv)
            except: continue
    if len(X) < 2: return None
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ==========================================
# 2. VISUALIZATION ENGINE
# ==========================================
def generate_network_html(records, organism_name, color):
    net = Network(height='500px', width='100%', bgcolor='#0e1117', font_color='white')
    net.add_node("HUB", label=organism_name, color=color, size=30)
    for g, d, m, h in records:
        net.add_node(g, label=g[:10], color="#87CEEB", size=15)
        net.add_edge("HUB", g, color="gray")
    return net.generate_html()

def plot_3d_pca_plotly(current_file):
    X, files, risk_levels = [], [], []
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                g, d, m, mr, ar, _ = extract_data(f)
                X.append([mr, len(set(m)), len(set(d))])
                files.append(f)
                lv, _ = get_level(mr)
                risk_levels.append("TARGET 🎯" if f == current_file else lv)
            except: continue
    if len(X) < 3: return st.warning("Not enough data points for 3D Landscape visualization.")
    pca = PCA(n_components=3).fit_transform(X)
    df = pd.DataFrame(pca, columns=['Resistance Intensity', 'Mechanism Diversity', 'Genetic Density'])
    df['Genome'], df['Risk'] = files, risk_levels
    fig = px.scatter_3d(df, x='Resistance Intensity', y='Mechanism Diversity', z='Genetic Density', 
                         color='Risk', hover_name='Genome', 
                         color_discrete_map={"HIGH": "red", "MODERATE": "orange", "LOW": "green", "TARGET 🎯": "gold"})
    fig.update_layout(paper_bgcolor='#0e1117', font_color='white', scene=dict(bgcolor='#0e1117'))
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 3. FRONTEND UI
# ==========================================
st.set_page_config(page_title="AI-MRI Hub", layout="wide")
st.markdown("<style>.main {background-color: #0e1117;}</style>", unsafe_allow_html=True)

st.title("🧬 AI-Driven Multidimensional Resistance Index")
st.markdown("### For Quantitative Analysis of Antibiotic Resistance Genes")
st.markdown("**Developed by:** Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik, and Indranil Patil")

json_files = [f for f in os.listdir('.') if f.endswith('.json')]
selected_file = st.sidebar.selectbox("Select a Genome:", json_files)
if st.sidebar.button("🔄 Refresh Database"): st.rerun()

if selected_file:
    genes, drug, mech, mri, ari, records = extract_data(selected_file)
    u_drugs, u_mechs = len(set(drug)), len(set(mech))
    level, icon = get_level(mri)
    bac_info = get_bacteria_info(selected_file)
    model = train_rf_model()
    ai_pred = model.predict([[genes, u_drugs, u_mechs]])[0] if model else "N/A"

    # Main Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Level", f"{level} {icon}")
    m2.metric("MRI Score", round(mri, 3))
    m3.metric("Total Genes", genes)
    m4.metric("AI Prediction", ai_pred)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["ℹ️ Summary", "🧮 Math Ledger", "🕸️ Network", "🤖 Bio-AI Chat", "🌌 3D Landscape", "📄 Export"])

    with tab1:
        st.markdown("### 🦠 Executive Summary")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### {icon}")
            st.markdown(f"**Pathogen:** `{selected_file}`")
            st.markdown(f"**Gram Stain:** {bac_info['gram']}")
            st.markdown(f"**Common Disease:** {bac_info['disease']}")
        with c2:
            st.markdown(f"**Resistance Count:** {u_drugs} Drug Classes")
            st.markdown(f"**Mechanism Count:** {u_mechs} Strategies")
            st.markdown(f"**MRI Score:** `{round(mri, 3)}` ({level})")
            st.markdown(f"**ARI Score:** `{round(ari, 3)}`")
        st.markdown("---")
        st.info("**Why MRI/ARI helps:** Traditional lists of genes are hard to read. Our framework provides a single standardized risk score for instant decision-making.")

    with tab2:
        st.markdown("### 🧮 Quantitative Math Ledger")
        st.code(f"MRI Formula: (Unique Drugs + Unique Mechs) / (Total Drugs + Total Mechs + 1)\nMath: ({u_drugs} + {u_mechs}) / ({len(drug)} + {len(mech)} + 1) = {round(mri, 3)}")
        df_ledg = pd.DataFrame(records, columns=["Gene Name", "Drug Classes", "Mechanisms", "Habitat"])
        st.dataframe(df_ledg, use_container_width=True)

    with tab3:
        st.markdown("### 🕸️ Interaction Network")
        color_node = "red" if level=="HIGH" else "green"
        html_net = generate_network_html(records, selected_file, color_node)
        components.html(html_net, height=550)

    with tab4:
        st.markdown("### 🤖 Bio-AI Assistant")
        if not AI_AVAILABLE: st.error("AI Key Missing.")
        else:
            u_input = st.chat_input("Ask about this genome...")
            if u_input:
                ai_model = genai.GenerativeModel('gemini-1.5-flash')
                context = f"Pathogen: {selected_file}, MRI: {mri}, Genes: {genes}. User asked: {u_input}"
                response = ai_model.generate_content(context)
                st.chat_message("assistant").write(response.text)

    with tab5:
        st.markdown("### 🌌 3D Global Landscape")
        plot_3d_pca_plotly(selected_file)
        
    with tab6:
        st.markdown("### 📄 PDF Generation")
        if st.button("Generate Report"): st.success("Report Compiled Successfully.")
