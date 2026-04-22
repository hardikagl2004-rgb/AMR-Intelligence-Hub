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
    genai.configure(api_key="AIzaSyDMo6ff3yVEsR9WkPWd5BWh6cz_gT0cMZ8")
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon="🧬")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #ffffff;}
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
    elif "vibrio" in name:
        info = {"gram": "Negative (-)", "disease": "Cholera, Vibriosis"}
    elif any(k in name for k in ["proteus", "morganella", "providencia"]):
        info = {"gram": "Negative (-)", "disease": "Complicated UTI, Kidney stones"}
    elif "campylobacter" in name or "helicobacter" in name:
        info = {"gram": "Negative (-)", "disease": "Peptic ulcers, Gastroenteritis"}
    elif "neisseria" in name:
        info = {"gram": "Negative (-)", "disease": "Gonorrhea, Meningitis"}
    elif "haemophilus" in name:
        info = {"gram": "Negative (-)", "disease": "Respiratory infections, Meningitis"}
    elif "staphylococcus" in name:
        info = {"gram": "Positive (+)", "disease": "Skin infections, MRSA, Endocarditis"}
    elif "streptococcus" in name:
        info = {"gram": "Positive (+)", "disease": "Strep throat, Pneumonia, Necrotizing fasciitis"}
    elif "enterococcus" in name:
        info = {"gram": "Positive (+)", "disease": "UTI, Endocarditis, VRE infections"}
    elif "bacillus" in name:
        info = {"gram": "Positive (+)", "disease": "Anthrax, Food poisoning"}
    elif "clostridium" in name:
        info = {"gram": "Positive (+)", "disease": "Tetanus, Botulism, C. diff diarrhea"}
    elif "listeria" in name:
        info = {"gram": "Positive (+)", "disease": "Listeriosis, Foodborne illness"}
    elif "corynebacterium" in name:
        info = {"gram": "Positive (+)", "disease": "Diphtheria"}
    elif "mycobacterium" in name:
        info = {"gram": "Acid-Fast (Gram Variable)", "disease": "Tuberculosis, Leprosy"}
        
    return info

@st.cache_data
def get_habitat(file_name):
    name = file_name.lower()
    if any(k in name for k in ["ecoli","escherichia","staphylococcus","salmonella","klebsiella", "streptococcus", "enterococcus"]):
        return "Clinical"
    elif any(k in name for k in ["pseudomonas","acinetobacter"]):
        return "Environmental"
    elif any(k in name for k in ["bacillus","clostridium", "mycobacterium"]):
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
        return f"Because its MRI is high, it utilizes multiple redundant strategies ({u_mechs} mechanisms) to block diverse threats ({u_drugs} drugs). If one antibiotic pathway is bypassed, the bacteria actively pivots to another, making standard frontline clinical treatments highly ineffective."
    elif "MODERATE" in level:
        return f"With a moderate MRI, this strain shows significant adaptation. It has built defenses against standard antibiotics ({u_drugs} drugs), forcing doctors to rely on secondary treatments."
    else:
        return f"This strain has a low MRI, indicating a narrow resistance profile. It likely specializes against specific antibiotics found in its direct natural habitat rather than hoarding a massive clinical arsenal."

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

    drug_c = Counter(drug).most_common(8)
    if drug_c:
        ax[0, 0].pie([v for k,v in drug_c], labels=[k[:15]+".." for k,v in drug_c], autopct='%1.1f%%', textprops={'color':"w"})
    ax[0, 0].set_title("Drug Classes Resisted")

    mech_c = Counter(mech)
    if mech_c:
        ax[0, 1].bar([k[:15]+".." for k in mech_c.keys()], mech_c.values(), color=color)
        ax[0, 1].tick_params(axis='x', rotation=35)
    ax[0, 1].set_title("Mechanisms Deployed")

    theta = np.linspace(0, np.pi, 100)
    ax[0, 2].plot(np.cos(theta), np.sin(theta), color='gray')
    ang = mri * np.pi
    ax[0, 2].plot([0, np.cos(ang)], [0, np.sin(ang)], color=color, linewidth=5)
    ax[0, 2].axis('off')
    ax[0, 2].set_title(f"MRI Indicator: {round(mri, 3)} ({level})")

    gene_list = [r[0] for r in records]
    gene_c = Counter(gene_list).most_common(5)
    if gene_c:
        ax[1, 0].bar([k[:15]+".." for k,v in gene_c], [v for k,v in gene_c], color='#87CEEB')
        ax[1, 0].tick_params(axis='x', rotation=35)
    ax[1, 0].set_title("Top Gene Frequency")

    ax[1, 1].bar(["Total Genes"], [genes], color='#2E86C1')
    ax[1, 1].set_title("Overall Gene Count")

    ax[1, 2].bar(["Unique Drugs", "Unique Mechs"], [len(set(drug)), len(set(mech))], color=["#9B59B6", "#E67E22"])
    ax[1, 2].set_title("Diversity Comparison")

    fig.tight_layout()
    return fig

def generate_network_html(records, organism_name, color):
    net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', cdn_resources="in_line", select_menu=True, filter_menu=True)
    net.add_node("HUB", label=organism_name, color=color, size=30)
    
    for g, d, m, h in records:
        net.add_node(g, label=g[:10], color="#87CEEB", size=15)
        net.add_edge("HUB", g, color="#ffffff")
        if m:
            for mech in set(m.split(", ")):
                if not mech: continue
                net.add_node(mech, label=mech[:10], color="#FFA500", size=10, shape="box")
                net.add_edge(g, mech, color="#aaaaaa")
                
    net.barnes_hut(gravity=-5000)
    html_path = "temp_network.html"
    html_content = net.generate_html()
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path

def plot_3d_pca_plotly(current_file):
    X, files, risk_levels = [], [], []
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                g, d, m, mr, ar, _ = extract_data(f)
                X.append([mr, len(set(m)), len(set(d))])
                files.append(f)
                lv, _ = get_level(mr)
                
                if f == current_file:
                    risk_levels.append("TARGET 🎯")
                else:
                    risk_levels.append(lv)
            except: continue
            
    if len(X) < 3:
        st.warning("Not enough data for 3D PCA.")
        return
        
    pca = PCA(n_components=3).fit_transform(X)
    df_pca = pd.DataFrame(pca, columns=['Overall Resistance (PC1)', 'Mechanism Diversity (PC2)', 'Genetic Density (PC3)'])
    df_pca['Genome'] = files
    df_pca['Risk Category'] = risk_levels
    
    color_discrete_map = {
        "HIGH": "red",
        "MODERATE": "orange",
        "LOW": "green",
        "TARGET 🎯": "gold"
    }

    fig = px.scatter_3d(df_pca, x='Overall Resistance (PC1)', y='Mechanism Diversity (PC2)', z='Genetic Density (PC3)',
              color='Risk Category', hover_name='Genome', color_discrete_map=color_discrete_map,
              opacity=0.8, size_max=10)
              
    fig.update_traces(marker=dict(size=5, line=dict(width=2, color='DarkSlateGrey')), selector=dict(name="TARGET 🎯"))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='#0e1117', font_color='white', scene=dict(
        xaxis=dict(backgroundcolor="#0e1117", gridcolor="gray"),
        yaxis=dict(backgroundcolor="#0e1117", gridcolor="gray"),
        zaxis=dict(backgroundcolor="#0e1117", gridcolor="gray")
    ))
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. MASTER PLATYPUS PDF GENERATOR
# ==========================================
def create_advanced_pdf_report(bac_name, genes, drug, mech, mri, ari, level, icon, records, dashboard_fig, bac_info, habitat, ai_pred_text, ai_conf_text):
    pdf_file = f"{bac_name.replace('.json', '')}_Detailed_Report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(name='TitleStyle', parent=styles['Heading1'], fontSize=18, spaceAfter=15, textColor=colors.HexColor('#1E3A8A'))
    h2_style = ParagraphStyle(name='H2', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#2E86C1'))
    normal_style = styles['Normal']
    
    u_drugs, u_mechs = len(set(drug)), len(set(mech))
    t_drugs, t_mechs = len(drug), len(mech)
    
    elements = []
    
    elements.append(Paragraph(f"AI-MRI Report: {bac_name.replace('.json', '')}", title_style))
    
    elements.append(Paragraph("Executive Summary & AI Analysis", h2_style))
    summary_text = f"""
    <b>Pathogen File:</b> {bac_name}<br/>
    <b>Gram Stain:</b> {bac_info['gram']}<br/>
    <b>Common Disease:</b> {bac_info['disease']}<br/>
    <b>Habitat:</b> {habitat}<br/>
    <b>Total Genes:</b> {genes}<br/>
    <b>Resistance (Drugs):</b> {u_drugs}<br/>
    <b>Mechanisms:</b> {u_mechs}<br/>
    <b>MRI Score:</b> {round(mri, 3)} ({level})<br/>
    <b>ARI Score:</b> {round(ari, 3)}<br/><br/>
    <b>Machine Learning AI Prediction:</b> {ai_pred_text}<br/>
    <b>Confidence Matrix:</b> {ai_conf_text}
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("Metric Explanations & Significance", h2_style))
    explanation_text = """
    <b>Pathogen Profile (Gram, Disease, Habitat):</b> Provides the essential biological and ecological context of the strain.<br/><br/>
    <b>Total Genes:</b> The absolute count of Antibiotic Resistance Genes (ARGs) identified in the sequence data.<br/><br/>
    <b>Resistance & Mechanisms:</b> Quantifies the distinct pharmaceutical drug classes the pathogen can evade and the specific biological strategies it deploys to do so.<br/><br/>
    <b>AI Prediction & Confidence:</b> A Random Forest model's probabilistic assessment of the pathogen's overall threat level based on its learned resistance profile.
    """
    elements.append(Paragraph(explanation_text, normal_style))
    
    elements.append(Paragraph("The Clinical Necessity of MRI and ARI", h2_style))
    mri_ari_text = """
    Traditional genomic analysis often simply lists detected genes, which fails to quantify the actual danger a pathogen poses. Our framework utilizes two calculated metrics to solve this:<br/><br/>
    <b>Why ARI is Required:</b> The Antibiotic Resistance Index (ARI) calculates the <i>density</i> of the threat by normalizing the unique mechanisms against the total gene count. It reveals how efficiently the bacteria utilizes its genetic payload to resist treatments.<br/><br/>
    <b>Why MRI Helps:</b> The Multidimensional Resistance Index (MRI) mathematically consolidates the diversity of resisted drugs and deployed mechanisms into a single, standardized risk score. This allows clinicians and researchers to instantly gauge and compare the severity of different strains, prioritizing high-risk pathogens for immediate intervention without needing to manually decipher complex gene ledgers.
    """
    elements.append(Paragraph(mri_ari_text, normal_style))
    
    elements.append(PageBreak())
    
    elements.append(Paragraph("Exact Mathematical Calculations", h2_style))
    calc_text = f"""
    <b>MRI Calculation:</b> ({u_drugs} + {u_mechs}) / ({t_drugs} + {t_mechs} + 1) = <b>{round(mri, 3)}</b><br/>
    <b>ARI Calculation:</b> {u_mechs} / ({genes} + 1) = <b>{round(ari, 3)}</b>
    """
    elements.append(Paragraph(calc_text, normal_style))
    
    elements.append(Paragraph("Risk Assessment Reasoning", h2_style))
    reason = get_risk_reason(level, u_drugs, u_mechs)
    elements.append(Paragraph(reason, normal_style))
    
    elements.append(Paragraph("Graphical Systems Dashboard", h2_style))
    buf = io.BytesIO()
    dashboard_fig.patch.set_facecolor('white')
    for ax in dashboard_fig.axes:
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        ax.title.set_color('black')
        for text in ax.texts:
            text.set_color('black')
            
    dashboard_fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = RLImage(buf, width=7.5*inch, height=5*inch)
    elements.append(img)
    
    elements.append(PageBreak())
    
    elements.append(Paragraph("Complete Gene Ledger", h2_style))
    table_data = [["Gene Name", "Drugs Resisted", "Mechanisms Used", "Habitat"]]
    for g, d, m, h in records:
        table_data.append([Paragraph(g, normal_style), Paragraph(d, normal_style), Paragraph(m, normal_style), Paragraph(h, normal_style)])
    
    t = Table(table_data, colWidths=[1.2*inch, 2.2*inch, 2.2*inch, 0.9*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E86C1')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8F9F9')),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    
    elements.append(t)
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
    st.session_state.chat_counter = 1

with st.sidebar:
    st.header("🗄️ Database Sync")
    if st.button("🔄 Refresh Database"):
        st.rerun()
        
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    analysis_mode = st.radio("Mode:", ["Select Known Bacteria", "AI Predict Unknown"])
    
    if analysis_mode == "Select Known Bacteria":
        selected_file = st.selectbox("Select a Genome:", json_files)
    
    st.markdown("---")
    st.success("✅ AI Brain Connected")

if analysis_mode == "Select Known Bacteria" and json_files:
    genes, drug, mech, mri, ari, records = extract_data(selected_file)
    u_drugs, u_mechs = len(set(drug)), len(set(mech))
    level, icon = get_level(mri)
    habitat = get_habitat(selected_file)
    bac_info = get_bacteria_info(selected_file)

    model = train_rf_model()
    ai_pred_text = "N/A"
    ai_conf_text = "N/A"
    if model:
        pred = model.predict([[genes, u_drugs, u_mechs]])[0]
        probs = model.predict_proba([[genes, u_drugs, u_mechs]])[0]
        classes = model.classes_
        conf_dict = {str(c): round(float(p), 3) for c, p in zip(classes, probs)}
        ai_pred_text = str(pred)
        ai_conf_text = str(conf_dict).replace("'", "")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Level", f"{level} {icon}")
    col2.metric("MRI Score", round(mri, 3))
    col3.metric("Total Genes", genes)
    col4.metric("Habitat", habitat)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "ℹ️ Pathogen Summary", 
        "📊 6-Panel Dashboard", 
        "🧮 Math & Data Ledger", 
        "🕸️ Network", 
        "🤖 Bio-AI Chat", 
        "📄 Export Master PDF", 
        "🌌 3D Landscape"
    ])
   with tab1:
    # 1. Professional Header with Icon
    st.markdown("""
        <div style="text-align: center; margin-bottom: 25px;">
            <h1 style="font-family: sans-serif; font-size: 2.8rem; margin: 0;">🧬 Executive Pathogen Intelligence</h1>
            <p style="color: grey; font-size: 1.1rem; margin-top: 5px;">Quantitative Analysis of Antibiogram Genomic Data</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Define Dynamic colors based on Risk Level (Assumes you have mri variable)
    level = "HIGH" if mri > 0.35 else "MODERATE" if mri > 0.15 else "LOW"
    color = "#f85149" if level == "HIGH" else "#d29922" if level == "MODERATE" else "#3fb950"
    
    # 2. Executive Metrics Bar (Proper Wrapping and Styling)
    # Define custom CSS for the metrics
    st.markdown(f"""
        <style>
        .summary-metric-value {{
            font-family: 'Orbitron', sans-serif;
            color: #58a6ff !important;
            font-size: 2.4rem !important;
        }}
        </style>
    """, unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1: st.metric("MRI Risk Score", f"{round(mri, 3)}", delta=level)
    with col_m2: st.metric("ARI Density", f"{round(ari, 3)}")
    with col_m3: st.metric("Total ARG Payload", genes)
    with col_m4: st.metric("Gram Classification", bac_info.get('gram', 'N/A'))

    st.divider()

    # 3. Pathogen Intel Grid (The Summary Text Fix)
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"#### 🦠 Bacterial Identification")
        st.write(f"**Identified Strain:** `{selected_file}`")
        st.write(f"**Associated Disease:** {bac_info.get('disease', 'N/A')}")
        st.write(f"**Clinical pathology:** gastroenteritis / septicemia")
        
    with col_right:
        st.markdown(f"#### 🏥 Clinical Evidence")
        st.write(f"**Multidrug Targets:** {len(set(drug))} drug classes evaded")
        st.write(f"**Mechanism Strategies:** {len(set(mech))} unique defenses deployed")
        # ML Prediction with colored badge (Assumes you have ai_pred variable)
        st.markdown(f"**AI Predicton:** <span style='background-color:{color}; color:white; padding:3px 8px; border-radius:5px;'>{level} RISK</span>", unsafe_allow_html=True)

    st.divider()
    
    # 4. Clinician Interpretation Box
    st.info(f"**Clinician Interpretation:** This strain is categorized as **{level} RISK**. The higher consolidated MRI score relative to the raw gene count suggests the pathogen utilizes multiple redundant strategies to bypass clinical treatments.")
   
    with tab2:
        st.markdown(f"### Systems Overview: `{selected_file}`")
        fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)
        st.pyplot(fig)
        
    with tab3:
        st.markdown("### Exact Calculations")
        st.code(f"""
-- MRI Calculation --
Formula: (Unique Drugs + Unique Mechanisms) / (Total Drugs + Total Mechanisms + 1)
Math: ({u_drugs} + {u_mechs}) / ({len(drug)} + {len(mech)} + 1)
Result: MRI = {round(mri, 3)} ({level})

-- ARI Calculation --
Formula: Unique Mechanisms / (Total Genes + 1)
Math: {u_mechs} / ({genes} + 1)
Result: ARI = {round(ari, 3)}
        """)
        
        st.markdown("### Risk Reasoning")
        st.info(get_risk_reason(level, u_drugs, u_mechs))
        
        st.markdown("### Full Gene Ledger (All Genes)")
        df = pd.DataFrame(records, columns=["Gene Name", "Drug Class", "Mechanism", "Habitat"])
        st.dataframe(df, use_container_width=True)

    with tab4:
        st.markdown("### Interactive Mechanism Network")
        st.write("Use the filter menu generated within the interactive map to isolate specific nodes.")
        html_path = generate_network_html(records, selected_file, "red" if level=="HIGH" else "orange" if level=="MODERATE" else "green")
        with open(html_path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650)

    with tab5:
        colA, colB, colC = st.columns([0.6, 0.2, 0.2])
        with colA:
            st.session_state.current_session = st.selectbox("Active Chat Session:", list(st.session_state.chat_sessions.keys()))
        with colB:
            st.write("")
            st.write("")
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.chat_counter += 1
                new_chat_name = f"Chat {st.session_state.chat_counter}"
                st.session_state.chat_sessions[new_chat_name] = []
                st.session_state.current_session = new_chat_name
                st.rerun()
        with colC:
            st.write("")
            st.write("")
            if st.button("🗑️ Clear This Chat", use_container_width=True):
                st.session_state.chat_sessions[st.session_state.current_session] = []
                st.rerun()
        
        for msg in st.session_state.chat_sessions[st.session_state.current_session]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        user_msg = st.chat_input(f"Ask me about {selected_file}...")
        if user_msg:
            st.chat_message("user").markdown(user_msg)
            st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": user_msg})
            
            if not AI_AVAILABLE:
                st.error("⚠️ AI Library missing. Check your terminal installation.")
            else:
                try:
                    context = f"""
                    You are J.A.R.V.I.S., an expert Bioinformatics AI. 
                    The user is analyzing the genome file: '{selected_file}'.
                    Data Profile:
                    - Total Genes: {genes} | Drugs Resisted: {u_drugs} | Mechanisms: {u_mechs} 
                    - MRI Score: {round(mri, 3)} ({level} Risk) | ARI Score: {round(ari, 3)}
                    
                    Answer the user's question intelligently based on this data.
                    User Question: {user_msg}
                    """
                    with st.spinner("Processing genome logic..."):
                        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        if not available_models:
                            st.error("Your API key does not have access to any text generation models.")
                        else:
                            target_model = next((m for m in available_models if 'flash' in m), next((m for m in available_models if 'pro' in m), available_models[0]))
                            model_ai = genai.GenerativeModel(target_model)
                            response = model_ai.generate_content(context)

                            st.chat_message("assistant").markdown(response.text)
                            st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": response.text})
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        st.error("⚠️ **Quota Exceeded (HTTP 429 Error).** The Gemini Free Tier allows a limited number of requests per minute. Please wait 60 seconds and try your question again.")
                    else:
                        st.error(f"AI Connection Error: {e}")

    with tab6:
        st.markdown("### 📥 Generate Complete Master Report")
        
        if st.button("Generate Master PDF", type="primary"):
            with st.spinner("Compiling graphs, explanations, and data into PDF..."):
                pdf_path = create_advanced_pdf_report(selected_file, genes, drug, mech, mri, ari, level, icon, records, fig, bac_info, habitat, ai_pred_text, ai_conf_text)
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="Download Detailed PDF Report",
                        data=file,
                        file_name=pdf_path,
                        mime="application/pdf"
                    )

    with tab7:
        st.markdown("### 🌌 Interactive Global Landscape Comparison (Plotly 3D)")
        st.write("You can rotate, zoom, and download this 3D map using the camera icon in the top right corner of the plot.")
        plot_3d_pca_plotly(selected_file)

elif analysis_mode == "AI Predict Unknown":
    st.header("🤖 Machine Learning Risk Prediction")
    in_genes = st.number_input("Total Genes Found", min_value=1, value=15)
    in_drugs = st.number_input("Unique Drugs Resisted", min_value=1, value=5)
    in_mechs = st.number_input("Unique Mechanisms Found", min_value=1, value=2)
    
    model = train_rf_model()
    if model and st.button("Predict Risk Level", type="primary"):
        prediction = model.predict([[in_genes, in_drugs, in_mechs]])[0]
        probs = model.predict_proba([[in_genes, in_drugs, in_mechs]])[0]
        classes = model.classes_
        prob_str = " ".join([f"{c[0]}:{p:.2f}" for c, p in zip(classes, probs)])
        
        st.success(f"### AI Prediction: **{prediction}** ({prob_str})")
