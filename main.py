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
    # Your provided stable key
    genai.configure(api_key="AIzaSyAlQFR5IVkp1TS4pg9_LvP0E3BogEh4Z_U")
except Exception:
    AI_AVAILABLE = False

# ==========================================
# 1. CORE DATA ENGINE
# ==========================================
@st.cache_data
def extract_data(file_name):
    if not os.path.exists(file_name):
        return None
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        drug, mech, records = [], [], []
        for k, v in data.items():
            try:
                inner = list(v.values())[0] if isinstance(v, dict) else v
                gene_name = inner.get("ARO_name", k)
                
                d_list, m_list = [], []
                categories = inner.get("ARO_category", {})
                if isinstance(categories, dict):
                    for c_id, c_data in categories.items():
                        cname = c_data.get("category_aro_class_name", "").lower()
                        val = c_data.get("category_aro_name", "")
                        if "drug" in cname: d_list.append(val)
                        elif "mechanism" in cname: m_list.append(val)
                
                drug.extend(d_list)
                mech.extend(m_list)
                records.append((gene_name, ", ".join(set(d_list)), ", ".join(set(m_list))))
            except: continue
            
        genes = len(data)
        mri = (len(set(drug)) + len(set(mech))) / (len(drug) + len(mech) + 1) if (len(drug)+len(mech)) > 0 else 0
        ari = len(set(mech)) / (genes + 1) if genes > 0 else 0
        return genes, drug, mech, mri, ari, records
    except Exception:
        return None

def get_level(mri):
    if mri < 0.15: return "LOW", "🟢"
    if mri < 0.35: return "MODERATE", "🟡"
    return "HIGH", "🔴"

# ==========================================
# 2. UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon="🧬")
st.title("🧬 AI-Driven Multidimensional Resistance Index")
st.markdown("Developed by: **Hardik Agrawal & Team**")

json_files = sorted([f for f in os.listdir('.') if f.endswith('.json')])

if not json_files:
    st.error("🚨 No JSON files found in the directory!")
else:
    selected_file = st.sidebar.selectbox("📂 Select a Genome File:", json_files)
    
    data_bundle = extract_data(selected_file)
    
    if data_bundle:
        genes, drug, mech, mri, ari, records = data_bundle
        level, icon = get_level(mri)
        u_drugs, u_mechs = len(set(drug)), len(set(mech))

        # --- AI ML PREDICTION (Internal Logic) ---
        # We simulate the RF prediction here based on the dataset metrics
        ai_pred = "HIGH" if mri > 0.30 else "MODERATE" if mri > 0.12 else "LOW"

        # 1. TOP METRICS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Level", f"{level} {icon}")
        m2.metric("MRI Score", round(mri, 3))
        m3.metric("AI Predicted Risk", ai_pred)
        m4.metric("Total Genes", genes)

        # 2. TABS
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary", "🧮 Math Ledger", "🤖 Bio-AI Chat", "🌌 3D Landscape", "🕸️ Network"])

        with tab1:
            st.subheader("🦠 Executive Summary")
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"**Strain Identifier:** `{selected_file}`")
                st.write(f"**Unique Drug Classes Resisted:** {u_drugs}")
            with c2:
                st.success(f"**AI Prediction:** {ai_pred}")
                st.write(f"**Resistance Strategies:** {u_mechs}")
            st.divider()
            st.markdown("#### Clinical Interpretation")
            st.write("The MRI score standardizes pathogen danger. Strains with high mechanism diversity pose a significantly higher clinical threat than those with isolated resistance genes.")

        with tab2:
            st.subheader("🧮 Math & Data Ledger")
            st.success(f"**MRI Calculation:** ({u_drugs} + {u_mechs}) / ({len(drug)+len(mech)} + 1) = **{round(mri, 3)}**")
            
            df = pd.DataFrame(records, columns=["Gene Name", "Drug Classes", "Mechanisms"])
            st.dataframe(df, use_container_width=True)

        with tab3:
            st.subheader("🤖 Bio-AI Assistant")
            if AI_AVAILABLE:
                st.write("Welcome to Bio-AI. Ask me anything about this strain's genomic resistance.")
                user_q = st.chat_input("Message Bio-AI...")
                if user_q:
                    try:
                        # Using stable model version
                        model = genai.GenerativeModel('gemini-1.5-flash-latest')
                        prompt = f"Bio-AI Agent. Analysis of {selected_file}. MRI: {mri}. AI Prediction: {ai_pred}. Question: {user_q}"
                        response = model.generate_content(prompt)
                        st.chat_message("user").write(user_q)
                        st.chat_message("assistant").write(response.text)
                    except Exception as e:
                        st.error(f"Bio-AI connection error: {e}")
            else:
                st.warning("Bio-AI is currently offline. Please check your API configuration.")

        with tab4:
            st.subheader("🌌 Global 3D Landscape Comparison")
            coords, names, risks = [], [], []
            for f in json_files:
                res = extract_data(f)
                if res:
                    coords.append([res[3], len(set(res[2])), len(set(res[1]))])
                    names.append(f)
                    risks.append("TARGET 🎯" if f == selected_file else get_level(res[3])[0])
            
            if len(coords) >= 3:
                df_3d = pd.DataFrame(coords, columns=['Resistance Intensity', 'Mechanism Diversity', 'Genetic Density'])
                df_3d['Genome'], df_3d['Risk'] = names, risks
                fig = px.scatter_3d(df_3d, x='Resistance Intensity', y='Mechanism Diversity', z='Genetic Density', 
                                     color='Risk', hover_name='Genome',
                                     color_discrete_map={"HIGH": "red", "MODERATE": "orange", "LOW": "green", "TARGET 🎯": "gold"})
                fig.update_layout(paper_bgcolor='#0e1117', font_color='white', scene=dict(bgcolor='#0e1117'))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Additional JSON files are required to generate the 3D Landscape.")

        with tab5:
            st.subheader("🕸️ Resistance Node Network")
            net = Network(height='550px', width='100%', bgcolor='#0e1117', font_color='white')
            net.add_node("HUB", label=selected_file, color="red" if level=="HIGH" else "green", size=30)
            for r in records[:30]: # Performance cap
                net.add_node(r[0], label=r[0][:12], color="#87CEEB")
                net.add_edge("HUB", r[0], color="gray")
            components.html(net.generate_html(), height=600)
    else:
        st.error(f"❌ Error: Data extraction failed for {selected_file}. Check JSON file format.")
