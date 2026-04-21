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

# --- AI BRAIN INITIALIZATION ---
try:
    import google.generativeai as genai
    AI_AVAILABLE = True
    # Using the specific API key you provided
    genai.configure(api_key="AIzaSyAlQFR5IVkp1TS4pg9_LvP0E3BogEh4Z_U")
except Exception:
    AI_AVAILABLE = False

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3, h4 {color: #ffffff !important;}
    .stMetric {background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d;}
    .stAlert {border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE BACKEND FUNCTIONS
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
    except Exception as e:
        return None

def get_level(mri):
    if mri < 0.15: return "LOW", "🟢"
    if mri < 0.35: return "MODERATE", "🟡"
    return "HIGH", "🔴"

# ==========================================
# 3. FRONTEND UI
# ==========================================
st.title("🧬 AI-Driven Multidimensional Resistance Index")
st.markdown("Developed by: **Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik, and Indranil Patil**")

json_files = sorted([f for f in os.listdir('.') if f.endswith('.json')])
selected_file = st.sidebar.selectbox("📂 Select a Genome File:", json_files)
st.sidebar.markdown("---")

if selected_file:
    # DATA EXTRACTION
    data_bundle = extract_data(selected_file)
    
    if data_bundle:
        genes, drug, mech, mri, ari, records = data_bundle
        u_drugs, u_mechs = len(set(drug)), len(set(mech))
        level, icon = get_level(mri)
        bac_info = get_bacteria_info(selected_file)

        # MAIN METRICS BAR
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Level", f"{level} {icon}")
        m2.metric("MRI Score", round(mri, 3))
        m3.metric("ARI Density", round(ari, 3))
        m4.metric("Gene Count", genes)

        # NAVIGATION TABS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive Summary", "🧮 Math Ledger", "🕸️ Interaction Network", "🤖 Bio-AI Chat", "🌌 3D Landscape"
        ])

        with tab1:
            st.subheader("🦠 Pathogen Analysis Report")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"#### {icon} Clinical Profile")
                st.write(f"**Strain Identifier:** `{selected_file}`")
                st.write(f"**Gram Classification:** {bac_info['gram']}")
                st.write(f"**Associated Diseases:** {bac_info['disease']}")
            with col_b:
                st.markdown("#### 🧪 Genomic Strength")
                st.write(f"**Drug Resistance Classes:** {u_drugs}")
                st.write(f"**Defense Mechanisms:** {u_mechs}")
                st.write(f"**Status:** {level} Risk Pathogen")
            
            st.divider()
            st.markdown("### 🎯 Why MRI & ARI Matter")
            st.info("""
            **The Clinical Problem:** Simply listing genes doesn't tell a doctor how dangerous a bacteria is.  
            **Our Solution:** The **ARI** measures how 'packed' the bacteria is with resistance, while the **MRI** gives a single, standardized score (0 to 1) so researchers can instantly prioritize high-risk outbreaks.
            """)

        with tab2:
            st.subheader("🧮 Quantitative Calculation Ledger")
            
            st.success(f"""
            **MRI Formula:** (Unique Drugs + Unique Mechanisms) / (Total Assignments + 1)  
            **Current Calculation:** ({u_drugs} + {u_mechs}) / ({len(drug) + len(mech)} + 1) = **{round(mri, 3)}**
            """)
            
            st.markdown("#### 📜 Complete Gene Database")
            df_ledg = pd.DataFrame(records, columns=["Gene Name", "Drug Classes", "Mechanisms"])
            st.dataframe(df_ledg, use_container_width=True)

        with tab3:
            st.subheader("🕸️ Genomic Interaction Network")
            st.write("Mapping the connections between identified genes and their resistance pathways.")
            net = Network(height='500px', width='100%', bgcolor='#0e1117', font_color='white')
            net.add_node("HUB", label=selected_file, color="red" if level=="HIGH" else "green", size=30)
            for g, d, m in records[:25]: # Optimization for speed
                net.add_node(g, label=g[:12], color="#87CEEB", size=15)
                net.add_edge("HUB", g, color="gray")
            components.html(net.generate_html(), height=550)

        with tab4:
            st.subheader("🤖 J.A.R.V.I.S. Bio-AI Chat")
            if AI_AVAILABLE:
                u_input = st.chat_input("Ask me about this strain's resistance mechanism...")
                if u_input:
                    try:
                        ai_model = genai.GenerativeModel('gemini-1.5-flash-latest')
                        context = f"Pathogen: {selected_file}, MRI: {mri}, Genes: {genes}. Question: {u_input}"
                        response = ai_model.generate_content(context)
                        st.chat_message("user").write(u_input)
                        st.chat_message("assistant").write(response.text)
                    except Exception as e:
                        st.error(f"AI Quota Reached. Please wait 60s. Error: {e}")
            else:
                st.warning("AI System Offline. Check API Key.")

        with tab5:
            st.subheader("🌌 Global 3D Landscape")
            st.write("Visualizing this strain's position relative to the global database.")
            X, files, risk_levels = [], [], []
            for f in json_files:
                d_bundle = extract_data(f)
                if d_bundle:
                    g, dr, me, mr, ar, _ = d_bundle
                    X.append([mr, len(set(me)), len(set(dr))])
                    files.append(f)
                    risk_levels.append("TARGET 🎯" if f == selected_file else get_level(mr)[0])
            
            if len(X) >= 3:
                pca = PCA(n_components=3).fit_transform(X)
                df_3d = pd.DataFrame(pca, columns=['Intensity', 'Diversity', 'Density'])
                df_3d['Genome'], df_3d['Risk'] = files, risk_levels
                fig = px.scatter_3d(df_3d, x='Intensity', y='Diversity', z='Density', color='Risk', hover_name='Genome',
                                     color_discrete_map={"HIGH": "red", "MODERATE": "orange", "LOW": "green", "TARGET 🎯": "gold"})
                fig.update_layout(paper_bgcolor='#0e1117', scene=dict(bgcolor='#0e1117'))
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Error loading genome data. Please check JSON file structure.")
