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
        info = {"gram": "Negative (-)", "disease": "Gastroenteritis, UTI, Sepsis"}
    elif "staphylococcus" in name:
        info = {"gram": "Positive (+)", "disease": "Skin infections, MRSA"}
    elif "campylobacter" in name:
        info = {"gram": "Negative (-)", "disease": "Food poisoning"}
    return info

@st.cache_data
def extract_data(file_name):
    try:
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
                records.append((gene_name, ", ".join(set(d)), ", ".join(set(m))))
            except: continue
        genes = len(data)
        mri = (len(set(drug)) + len(set(mech))) / (len(drug) + len(mech) + 1)
        ari = len(set(mech)) / (genes + 1)
        return genes, drug, mech, mri, ari, records
    except: return None

def get_level(mri):
    if mri < 0.15: return "LOW", "🟢"
    if mri < 0.35: return "MODERATE", "🟡"
    return "HIGH", "🔴"

@st.cache_resource
def train_rf_model():
    X, y = [], []
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                d_bundle = extract_data(f)
                if d_bundle:
                    g, dr, me, mr, ar, _ = d_bundle
                    X.append([g, len(set(dr)), len(set(me))])
                    y.append(get_level(mr)[0])
            except: continue
    if len(X) < 2: return None
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ==========================================
# 2. PDF REPORT GENERATOR
# ==========================================
def create_master_report(selected_file, genes, drug, mech, mri, ari, level, records, bac_info, ai_pred):
    pdf_file = f"{selected_file.replace('.json', '')}_Final_Report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(name='T', fontSize=18, textColor=colors.HexColor('#1E3A8A'), spaceAfter=20)
    h2_style = ParagraphStyle(name='H', fontSize=14, textColor=colors.HexColor('#2E86C1'), spaceBefore=15)
    
    elements = []
    elements.append(Paragraph(f"AI-MRI Pathogen Intelligence Report", title_style))
    elements.append(Paragraph("Executive Summary", h2_style))
    summary_data = [
        ["Pathogen ID", selected_file],
        ["MRI Risk Score", f"{round(mri, 3)} ({level})"],
        ["ARI Density", f"{round(ari, 3)}"],
        ["AI Risk Prediction", ai_pred],
        ["Total ARG Genes", f"{genes}"],
        ["Classification", bac_info['gram']]
    ]
    t_sum = Table(summary_data, colWidths=[2*inch, 3.5*inch])
    t_sum.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke)]))
    elements.append(t_sum)
    doc.build(elements)
    return pdf_file

# ==========================================
# 3. FRONTEND UI
# ==========================================
st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon="🧬")
st.title("🧬 AI-Driven Multidimensional Resistance Index")
st.markdown("Developed by: **Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik, and Indranil Patil**")

json_files = sorted([f for f in os.listdir('.') if f.endswith('.json')])
selected_file = st.sidebar.selectbox("📂 Select a Genome File:", json_files)

if selected_file:
    data_bundle = extract_data(selected_file)
    if data_bundle:
        genes, drug, mech, mri, ari, records = data_bundle
        level, icon = get_level(mri)
        bac_info = get_bacteria_info(selected_file)
        u_drugs, u_mechs = len(set(drug)), len(set(mech))
        
        # --- AI ML PREDICTION ---
        rf_model = train_rf_model()
        ai_risk_prediction = rf_model.predict([[genes, u_drugs, u_mechs]])[0] if rf_model else "N/A"

        # Metrics Bar
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Level", f"{level} {icon}")
        m2.metric("MRI Score", round(mri, 3))
        m3.metric("AI Prediction", ai_risk_prediction)
        m4.metric("Total Genes", genes)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Summary", "🧮 Math Ledger", "🕸️ Network", "🤖 Bio-AI Chat", "🌌 3D Landscape", "📄 Detailed Report"
        ])

        with tab1:
            st.subheader("🦠 Pathogen Analysis Report")
            col_a, col_b = st.columns(2)
            with col_a:
                st.info(f"#### {icon} Clinical Profile")
                st.write(f"**Strain ID:** `{selected_file}`")
                st.write(f"**Gram:** {bac_info['gram']}")
                st.write(f"**Pathology:** {bac_info['disease']}")
            with col_b:
                st.success(f"#### 🧪 Resistance Strength")
                st.write(f"**Drug Resistance Classes:** {u_drugs}")
                st.write(f"**Defense Mechanisms:** {u_mechs}")
                st.write(f"**AI Risk Assessment:** :red[{ai_risk_prediction}]")
            st.divider()
            st.info("**Why MRI/ARI helps:** Traditional lists of genes are hard to interpret. This framework provides a single standardized risk score for instant decision-making.")

        with tab2:
            st.subheader("🧮 Quantitative Math Ledger")
            st.success(f"**MRI Formula Applied:** ({u_drugs} + {u_mechs}) / ({len(drug) + len(mech)} + 1) = **{round(mri, 3)}**")
            df_ledg = pd.DataFrame(records, columns=["Gene Name", "Drug Classes", "Mechanisms"])
            st.dataframe(df_ledg, use_container_width=True)

        with tab3:
            st.subheader("🕸️ Interaction Network")
            net = Network(height='500px', width='100%', bgcolor='#0e1117', font_color='white')
            net.add_node("HUB", label=selected_file, color="red" if level=="HIGH" else "green", size=30)
            for g, d, m in records[:25]:
                net.add_node(g, label=g[:12], color="#87CEEB", size=15)
                net.add_edge("HUB", g, color="gray")
            components.html(net.generate_html(), height=550)

        with tab4:
            st.subheader("🤖 J.A.R.V.I.S. Bio-AI Chat")
            if AI_AVAILABLE:
                u_input = st.chat_input("Ask about this strain...")
                if u_input:
                    ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    response = ai_model.generate_content(f"Pathogen: {selected_file}, MRI: {mri}, Prediction: {ai_risk_prediction}. User Question: {u_input}")
                    st.chat_message("assistant").write(response.text)
            else: st.warning("AI Offline.")

        with tab5:
            st.subheader("🌌 Global 3D Landscape")
            X, files, risk_levels = [], [], []
            for f in json_files:
                d_b = extract_data(f)
                if d_b:
                    X.append([d_b[3], len(set(d_b[2])), len(set(d_b[1]))])
                    files.append(f)
                    risk_levels.append("TARGET 🎯" if f == selected_file else get_level(d_b[3])[0])
            if len(X) >= 3:
                pca = PCA(n_components=3).fit_transform(X)
                df_3d = pd.DataFrame(pca, columns=['Intensity', 'Diversity', 'Density'])
                df_3d['Genome'], df_3d['Risk'] = files, risk_levels
                fig = px.scatter_3d(df_3d, x='Intensity', y='Diversity', z='Density', color='Risk', hover_name='Genome',
                                     color_discrete_map={"HIGH": "red", "MODERATE": "orange", "LOW": "green", "TARGET 🎯": "gold"})
                fig.update_layout(paper_bgcolor='#0e1117', scene=dict(bgcolor='#0e1117'))
                st.plotly_chart(fig, use_container_width=True)

        with tab6:
            st.subheader("📥 Export Master PDF Report")
            if st.button("Generate Detailed Report", type="primary"):
                pdf_path = create_master_report(selected_file, genes, drug, mech, mri, ari, level, records, bac_info, ai_risk_prediction)
                with open(pdf_path, "rb") as f:
                    st.download_button(label="Download PDF Report", data=f, file_name=pdf_path, mime="application/pdf")
