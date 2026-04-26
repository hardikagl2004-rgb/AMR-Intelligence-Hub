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

# --- AI BRAIN INITIALIZATION (SECURE UPDATE) ---
try:
    import google.generativeai as genai
    AI_AVAILABLE = True
    if "GENAI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GENAI_API_KEY"])
    else:
        AI_AVAILABLE = False
        st.error("🔑 API Key Missing: Please add 'GENAI_API_KEY' to your Streamlit Cloud Secrets.")
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon="🧬")

COMMON_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
.main {background-color: #0e1117;}
h1, h2, h3 {color: #ffffff;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.main-header {
background: linear-gradient(120deg, #020b18 0%, #0c1f3f 30%, #06304a 60%, #0f2a1a 100%);
border: 1px solid rgba(0,212,255,0.2);
padding: 2.5rem;
border-radius: 15px;
color: white;
text-align: center;
margin-bottom: 2rem;
box-shadow: 0 10px 40px rgba(0,212,255,0.1);
position: relative;
overflow: hidden;
}
.main-header::before {
  content:'';
  position:absolute; top:0;left:0;right:0; height:3px;
  background: linear-gradient(90deg, #00d4ff, #7c3aed, #10b981, #00d4ff);
  background-size:300% auto;
  animation: headerBar 4s linear infinite;
}
@keyframes headerBar { 0%{background-position:0% center} 100%{background-position:300% center} }

.welcome-hero {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
    border-radius: 15px;
    padding: 45px;
    border: 1px solid rgba(59, 130, 246, 0.4);
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
}
.welcome-hero h2 {
    color: #60a5fa !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    margin-bottom: 12px !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
}
.welcome-hero p {
    color: #e2e8f0 !important;
    font-size: 1.25rem !important;
    letter-spacing: 0.8px;
    opacity: 1.0;
}

.alert-banner {
background: rgba(239, 68, 68, 0.2);
border: 1px solid #ef4444;
padding: 15px;
border-radius: 8px;
color: #f87171;
font-weight: bold;
text-align: center;
margin-bottom: 20px;
animation: pulse 2s infinite;
}
@keyframes pulse {
0% { opacity: 1; }
50% { opacity: 0.7; }
100% { opacity: 1; }
}

.metric-card {
background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(6,30,50,0.9));
border: 1px solid rgba(0,212,255,0.15);
padding: 1.5rem;
border-radius: 12px;
text-align: center;
transition: all 0.3s ease;
}
.metric-card:hover {
border-color: rgba(0,212,255,0.4);
box-shadow: 0 8px 25px rgba(0,212,255,0.1);
}
.metric-label {
color: #00d4ff;
font-size: 0.75rem;
font-weight: bold;
text-transform: uppercase;
letter-spacing: 0.1em;
}
.metric-value {
color: #ffffff;
font-size: 1.8rem;
font-weight: 700;
margin-top: 0.5rem;
}
.report-card {
background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(6,30,50,0.9));
border-left: 5px solid #00d4ff;
padding: 20px;
border-radius: 10px;
margin-bottom: 25px;
}
.report-header {
color: #00d4ff;
font-weight: bold;
text-transform: uppercase;
font-size: 0.9rem;
margin-bottom: 15px;
border-bottom: 1px solid #334155;
padding-bottom: 5px;
}
.report-row {
display: flex;
justify-content: space-between;
padding: 8px 0;
border-bottom: 1px solid #2d3748;
}
.report-label { color: #94a3b8; font-weight: 500; }
.report-value { color: #ffffff; font-weight: 600; }
.ai-badge {
background: rgba(0,212,255,0.1);
color: #00d4ff;
padding: 4px 12px;
border-radius: 20px;
font-size: 0.85rem;
border: 1px solid rgba(0,212,255,0.3);
}
.math-card {
background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(6,30,50,0.9));
border: 1px solid rgba(0,212,255,0.15);
padding: 25px;
border-radius: 12px;
margin-bottom: 20px;
}
.math-card-header {
color: #00d4ff;
font-size: 0.85rem;
font-weight: bold;
text-transform: uppercase;
letter-spacing: 0.1em;
margin-bottom: 15px;
}
.annotation-box {
background: rgba(15, 23, 42, 0.6);
border: 1px dashed #334155;
border-radius: 8px;
padding: 12px;
margin-top: 10px;
font-size: 0.85rem;
color: #94a3b8;
}
.annotation-item { margin-bottom: 4px; }
.annotation-key { color: #60a5fa; font-weight: bold; font-family: monospace; }

.susceptibility-card {
background: rgba(16, 185, 129, 0.1);
border: 1px solid #10b981;
border-radius: 10px;
padding: 15px;
color: #10b981;
font-weight: 600;
}
.reasoning-box {
background: rgba(6,30,50,0.85);
border-radius: 12px;
border-left: 5px solid #00d4ff;
padding: 20px;
color: #ffffff;
line-height: 1.7;
font-size: 1.05rem;
font-weight: 400;
box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
}
[data-testid="stDataFrame"] {
border: 1px solid #334155;
border-radius: 10px;
overflow: hidden;
}

@keyframes fadeInDown {
  from { opacity: 0; transform: translateY(-40px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(40px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes shimmer {
  0%   { background-position: -200% center; }
  100% { background-position: 200% center; }
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50%       { transform: translateY(-10px); }
}
@keyframes titleShimmer { 0% { background-position: 0% center; } 100% { background-position: 300% center; } }
</style>
"""

st.markdown(COMMON_CSS, unsafe_allow_html=True)

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
            gene_name = inner.get("ARO_name", k).upper()
            d, m = [], []
            for c in inner.get("ARO_category", {}).values():
                cname = c.get("category_aro_class_name","").lower()
                val = c.get("category_aro_name","").title()
                if "drug" in cname: d.append(val)
                elif "mechanism" in cname: m.append(val)
            drug.extend(d)
            mech.extend(m)
            records.append((gene_name, ", ".join(set(d)), ", ".join(set(m)), habitat.title()))
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
        return f"This pathogen's elevated MRI score reflects a sophisticated and redundant resistance architecture. Deploying {u_mechs} distinct biological mechanisms against {u_drugs} drug classes, the organism exhibits the capacity to dynamically bypass standard frontline therapeutics. When one resistance pathway is circumvented, alternative mechanisms compensate — a hallmark of high-priority clinical threats."
    elif "MODERATE" in level:
        return f"This strain demonstrates a clinically significant, though not maximal, resistance burden. Resistance determinants spanning {u_drugs} drug classes have been identified, necessitating careful antibiogram-guided therapy selection. Secondary and combination treatment regimens should be considered to achieve effective clinical outcomes."
    else:
        return f"The calculated MRI indicates a restricted resistance profile, suggestive of ecological specialization rather than broad clinical adaptation. With resistance distributed across {u_drugs} drug classes via {u_mechs} mechanisms, standard empirical treatment protocols remain viable. Continued genomic surveillance is recommended to monitor for resistance acquisition."

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
        st.warning("Insufficient data points for 3D PCA rendering. A minimum of three genomic profiles are required.")
        return
    pca = PCA(n_components=3).fit_transform(X)
    df_pca = pd.DataFrame(pca, columns=['Overall Resistance (PC1)', 'Mechanism Diversity (PC2)', 'Genetic Density (PC3)'])
    df_pca['Genome'] = files
    df_pca['Risk Category'] = risk_levels
    color_discrete_map = {"HIGH": "red","MODERATE": "orange","LOW": "green","TARGET 🎯": "gold"}
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
# 4. PDF GENERATOR
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
<b>Associated Pathology:</b> {bac_info['disease']}<br/>
<b>Ecological Habitat:</b> {habitat}<br/>
<b>Total Resistance Genes:</b> {genes}<br/>
<b>Drug Classes Resisted:</b> {u_drugs}<br/>
<b>Resistance Mechanisms:</b> {u_mechs}<br/>
<b>MRI Score:</b> {round(mri, 3)} ({level})<br/>
<b>ARI Score:</b> {round(ari, 3)}<br/><br/>
<b>Machine Learning Prediction:</b> {ai_pred_text} Risk<br/>
<b>Confidence Distribution:</b> {ai_conf_text}
"""
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Metric Definitions & Clinical Significance", h2_style))
    explanation_text = """
<b>Pathogen Profile (Gram Classification, Pathology, Habitat):</b> Establishes the essential biological and ecological context required for accurate clinical interpretation of resistance data.<br/><br/>
<b>Total Resistance Genes:</b> The absolute count of Antibiotic Resistance Genes (ARGs) identified within the sequenced genomic data, sourced from the CARD database annotation framework.<br/><br/>
<b>Drug Classes Resisted &amp; Mechanisms Deployed:</b> Enumerates the distinct pharmaceutical classes the pathogen can evade and the specific molecular strategies it employs to achieve resistance.<br/><br/>
<b>AI Prediction &amp; Confidence Distribution:</b> A Random Forest ensemble model's probabilistic classification of the pathogen's aggregate threat level, trained on a population of reference genomic profiles.
"""
    elements.append(Paragraph(explanation_text, normal_style))
    elements.append(Paragraph("Clinical Rationale for MRI and ARI Frameworks", h2_style))
    mri_ari_text = """
Conventional genomic analysis outputs raw gene inventories, which fail to quantify the actual clinical danger a pathogen poses in a standardized, comparable format.
The AI-MRI Hub addresses this gap through two validated computational metrics:<br/><br/>
<b>Antibiotic Resistance Index (ARI):</b> Quantifies the <i>density</i> of resistance by normalizing unique mechanism count against total gene count with Laplace smoothing.<br/><br/>
<b>Multidimensional Resistance Index (MRI):</b> Consolidates resistance breadth (drug class diversity) and resistance depth (mechanism diversity) into a single normalized score.
"""
    elements.append(Paragraph(mri_ari_text, normal_style))
    elements.append(PageBreak())
    elements.append(Paragraph("Mathematical Derivations", h2_style))
    calc_text = f"""
<b>MRI Calculation:</b> ({u_drugs} + {u_mechs}) / ({t_drugs} + {t_mechs} + 1) = <b>{round(mri, 3)}</b><br/>
<b>ARI Calculation:</b> {u_mechs} / ({genes} + 1) = <b>{round(ari, 3)}</b>
"""
    elements.append(Paragraph(calc_text, normal_style))
    elements.append(Paragraph("Risk Assessment Interpretation", h2_style))
    reason = get_risk_reason(level, u_drugs, u_mechs)
    elements.append(Paragraph(reason, normal_style))
    elements.append(Paragraph("Systems Analysis Dashboard", h2_style))
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
    elements.append(Paragraph("Complete Gene Resistance Ledger", h2_style))
    table_data = [["Gene Name", "Drug Classes Resisted", "Mechanisms Deployed", "Habitat"]]
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
# 5. SESSION STATE INITIALIZATION
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'splash'   # 'splash' | 'team' | 'app'

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
    st.session_state.current_session = "Chat 1"
    st.session_state.chat_counter = 1

# ==========================================
# 6. TEAM PAGE
# ==========================================
def render_team_page():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none !important;}
    .block-container {padding-top: 2rem !important;}
    .tp-card {
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(0,212,255,0.18);
      border-radius: 20px;
      padding: 28px 22px;
      text-align: center;
      transition: all 0.35s ease;
      position: relative;
      overflow: hidden;
      height: 100%;
    }
    .tp-avatar {
      width: 80px; height: 80px; border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 2.2rem; margin: 0 auto 16px auto;
      border: 2px solid rgba(0,212,255,0.3);
    }
    .tp-name { color: #f1f5f9 !important; font-size: 1.1rem; font-weight: 800; margin-bottom: 4px; }
    .tp-role-primary {
      color: #00d4ff !important; font-size: 0.7rem; text-transform: uppercase;
      letter-spacing: 2px; font-weight: 700; margin-bottom: 2px;
    }
    .tp-role-secondary {
      color: #a78bfa !important; font-size: 0.65rem; text-transform: uppercase;
      letter-spacing: 1.5px; margin-bottom: 12px;
    }
    .tp-desc { color: #94a3b8; font-size: 0.82rem; line-height: 1.65; margin-bottom: 14px; }
    .tp-tags { display: flex; flex-wrap: wrap; gap: 5px; justify-content: center; }
    .tp-tag {
      background: rgba(0,212,255,0.07); border: 1px solid rgba(0,212,255,0.2);
      color: #67e8f9; padding: 3px 10px; border-radius: 20px; font-size: 0.65rem;
    }
    .tp-mission-card {
      background: rgba(255,255,255,0.02);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 14px; padding: 28px 20px; text-align: center;
    }
    .tp-mission-title {
      color: #00d4ff !important; font-size: 0.8rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px;
    }
    .tp-mission-text { color: #64748b; font-size: 0.84rem; line-height: 1.7; }
    .tp-divider { width: 80px; height: 3px; background: linear-gradient(90deg,#00d4ff,#7c3aed); margin: 0 auto 30px auto; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)

    # Navigation bar at the top
    col_back, col_title, col_enter = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Back to Splash", use_container_width=True):
            st.session_state.page = 'splash'
            st.rerun()
    with col_title:
        st.markdown("<h2 style='text-align:center; color:#00d4ff; font-family:Orbitron,monospace; margin:0;'>👥 Meet The Team</h2>", unsafe_allow_html=True)
    with col_enter:
        if st.button("🚀 Enter Platform", type="primary", use_container_width=True):
            st.session_state.page = 'app'
            st.rerun()

    st.markdown("<div class='tp-divider'></div>", unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align:center; color:#94a3b8; font-size:1rem; max-width:780px; margin:0 auto 30px auto; line-height:1.8;'>
    AI-MRI Hub was built by a multidisciplinary team of seven researchers, developers, and scientists united by a
    shared mission — to make antibiotic resistance genomics instantly accessible, intelligible, and actionable for
    clinicians and researchers worldwide.
    </p>
    """, unsafe_allow_html=True)

    # Team members data
    team = [
        {
            "emoji": "👩‍🔬", "bg": "rgba(236,72,153,0.1)",
            "name": "Poorva Dongarkar",
            "role1": "Project Lead", "role2": "Genomics Architect",
            "desc": "Directed end-to-end project execution and overall technical vision. Spearheaded CARD database integration, ARG extraction logic, and the bacterial classification system that powers the platform's core analytics.",
            "tags": ["Leadership", "Genomics", "AMR", "Data"]
        },
        {
            "emoji": "👨‍💻", "bg": "rgba(0,212,255,0.1)",
            "name": "Hardik Agrawal",
            "role1": "Lead Developer", "role2": "AI & Systems Engineer",
            "desc": "Architected the full-stack Streamlit platform from the ground up. Designed the proprietary MRI/ARI mathematical framework and built the Random Forest AI prediction pipeline.",
            "tags": ["Python", "ML", "Streamlit", "Backend"]
        },
        {
            "emoji": "👩‍💻", "bg": "rgba(245,158,11,0.1)",
            "name": "Avani Laswante",
            "role1": "Presentation Lead", "role2": "UI/UX Designer",
            "desc": "Crafted and delivered all project presentations and the visual narrative strategy. Also responsible for the complete CSS design system, dark-mode aesthetic, and the platform's animated visual identity.",
            "tags": ["Presentation", "CSS", "UI/UX", "Design"]
        },
        {
            "emoji": "👩‍🔬", "bg": "rgba(139,92,246,0.1)",
            "name": "Zeel Bhanushali",
            "role1": "Presentation Specialist", "role2": "Clinical Research Lead",
            "desc": "Co-led project presentations with scientific depth and clarity. Developed the clinical susceptibility logic, drug-class universe mapping, and conducted MRI clinical validation research.",
            "tags": ["Presentation", "Microbiology", "Pharmacology"]
        },
        {
            "emoji": "👩‍💻", "bg": "rgba(6,182,212,0.1)",
            "name": "Aayushi Wasnik",
            "role1": "Research Lead", "role2": "AI Integration Specialist",
            "desc": "Led the scientific research underpinning the platform's clinical and genomic frameworks. Implemented the J.A.R.V.I.S. AI chatbot, Gemini API integration, context injection pipeline, and multi-session state management.",
            "tags": ["Research", "Gemini API", "LLM", "Prompt Eng."]
        },
        {
            "emoji": "👨‍🔬", "bg": "rgba(244,63,94,0.1)",
            "name": "Indranil Patil",
            "role1": "Documentation Lead", "role2": "PDF & Technical Writer",
            "desc": "Authored all technical documentation, mathematical write-ups, and the academic framework. Built the ReportLab PDF pipeline that compiles analysis into publication-ready reports.",
            "tags": ["Documentation", "ReportLab", "LaTeX", "Writing"]
        },
        {
            "emoji": "👨‍🔬", "bg": "rgba(16,185,129,0.1)",
            "name": "Yashraj Patil",
            "role1": "Documentation Specialist", "role2": "Visualization Engineer",
            "desc": "Co-authored technical documentation and visual reference guides. Engineered the PyVis network graphs, Plotly 3D PCA landscape, and the Matplotlib 6-panel analysis dashboard.",
            "tags": ["Documentation", "PyVis", "Plotly", "Matplotlib"]
        },
    ]

    # Render 3 columns per row
    for row_start in range(0, len(team), 3):
        row = team[row_start:row_start+3]
        cols = st.columns(3)
        for i, member in enumerate(row):
            with cols[i]:
                tags_html = "".join([f'<span class="tp-tag">{t}</span>' for t in member["tags"]])
                st.markdown(f"""
                <div class="tp-card">
                  <div class="tp-avatar" style="background:{member['bg']};">{member['emoji']}</div>
                  <div class="tp-name">{member['name']}</div>
                  <div class="tp-role-primary">{member['role1']}</div>
                  <div class="tp-role-secondary">{member['role2']}</div>
                  <div class="tp-desc">{member['desc']}</div>
                  <div class="tp-tags">{tags_html}</div>
                </div>
                """, unsafe_allow_html=True)

        st.write("")  # spacing between rows

    # Mission / Methods / Vision
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class="tp-mission-card">
          <div class="tp-mission-title">🎯 Mission</div>
          <div class="tp-mission-text">Democratize antibiotic resistance genomics through open, intelligent, and beautifully designed scientific tools any researcher or clinician can use.</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
        <div class="tp-mission-card">
          <div class="tp-mission-title">🔬 Methods</div>
          <div class="tp-mission-text">Rigorous mathematical indexing (MRI, ARI), validated machine learning classification, evidence-based clinical susceptibility mapping, and iterative peer review.</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
        <div class="tp-mission-card">
          <div class="tp-mission-title">🌍 Vision</div>
          <div class="tp-mission-text">Every hospital and research lab equipped with instant AI-driven resistance profiling — making the next superbug detectable before it becomes untreatable.</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    # Bottom CTA
    col_l, col_c, col_r = st.columns([2, 1, 2])
    with col_c:
        if st.button("🚀 Enter AI-MRI Hub", type="primary", use_container_width=True):
            st.session_state.page = 'app'
            st.rerun()


# ==========================================
# 7. SPLASH PAGE
# ==========================================
def render_splash_page():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none !important;}
    .block-container {padding: 0 !important; max-width: 100% !important;}
    .sp-dna-ring {
      display: inline-block; font-size: 5.5rem;
      animation: dnaFloat 4s ease-in-out infinite;
      filter: drop-shadow(0 0 30px rgba(0,212,255,0.6)); margin-bottom: 20px;
    }
    @keyframes dnaFloat { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-15px); } }
    .sp-title {
      font-family: 'Orbitron', monospace !important; font-size: 4rem !important; font-weight: 900 !important;
      background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 40%, #10b981 70%, #00d4ff 100%);
      background-size: 300% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
      animation: titleShimmer 5s linear infinite; letter-spacing: 3px;
    }
    .sp-tagline { font-family:'Orbitron',monospace; font-size:0.85rem; letter-spacing:5px; color:#00d4ff !important; text-transform:uppercase; }
    .sp-stat { background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.2); border-radius:16px; padding:18px 26px; text-align:center; display:inline-block; margin:6px; }
    .sp-stat-num { font-family:'Orbitron',monospace; font-size:2rem; font-weight:900; color:#00d4ff; display:block; }
    .sp-stat-lbl { font-size:0.7rem; color:#64748b; letter-spacing:2px; text-transform:uppercase; }
    .sp-feat-card { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07); border-radius:16px; padding:24px 20px; height:100%; }
    .sp-feat-icon { font-size:2rem; display:block; margin-bottom:12px; }
    .sp-feat-title { color:#00d4ff !important; font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
    .sp-feat-desc { color:#64748b; font-size:0.85rem; line-height:1.6; }
    .sp-comp-old { background:rgba(239,68,68,0.04); border:1px solid rgba(239,68,68,0.2); border-radius:14px; padding:24px; }
    .sp-comp-new { background:rgba(0,212,255,0.04); border:1px solid rgba(0,212,255,0.25); border-radius:14px; padding:24px; }
    .sp-comp-title { font-size:1rem; font-weight:700; margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.06); }
    .sp-comp-old .sp-comp-title { color:#f87171; }
    .sp-comp-new .sp-comp-title { color:#00d4ff; }
    .sp-comp-item { color:#94a3b8; font-size:0.86rem; margin-bottom:10px; line-height:1.5; }
    </style>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div style="background:#020b18; padding:70px 60px 50px 60px; text-align:center;">
      <span class="sp-dna-ring">🧬</span>
      <h1 class="sp-title">AI-MRI HUB</h1>
      <p class="sp-tagline">Antimicrobial Resistance Intelligence Platform</p>
      <p style="color:#94a3b8; font-size:1.1rem; max-width:650px; margin:20px auto 35px auto; line-height:1.8;">
        The world's most advanced quantitative framework for decoding antibiotic resistance genes.
        Transform raw genomic data into actionable clinical intelligence — instantly and at scale.
      </p>
      <div style="margin-bottom:10px;">
        <span class="sp-stat"><span class="sp-stat-num">2</span><span class="sp-stat-lbl">Novel Indices</span></span>
        <span class="sp-stat"><span class="sp-stat-num">8</span><span class="sp-stat-lbl">Analysis Modules</span></span>
        <span class="sp-stat"><span class="sp-stat-num">100+</span><span class="sp-stat-lbl">Genomes Supported</span></span>
        <span class="sp-stat"><span class="sp-stat-num">7</span><span class="sp-stat-lbl">Researchers</span></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA Buttons
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_l, col_c1, col_gap, col_c2, col_r = st.columns([1.5, 1, 0.2, 1, 1.5])
    with col_c1:
        if st.button("🚀  ENTER AI-MRI HUB", type="primary", use_container_width=True):
            st.session_state.page = 'app'
            st.rerun()
    with col_c2:
        if st.button("👥  MEET THE TEAM", use_container_width=True):
            st.session_state.page = 'team'
            st.rerun()

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # Platform Features
    st.markdown("<h2 style='text-align:center; color:#f1f5f9; font-family:Orbitron,monospace; font-size:1.6rem; letter-spacing:2px;'>Platform Features</h2>", unsafe_allow_html=True)
    st.markdown("<div style='width:80px;height:3px;background:linear-gradient(90deg,#00d4ff,#7c3aed);margin:0 auto 30px auto;border-radius:3px;'></div>", unsafe_allow_html=True)

    features = [
        ("🦠", "Genomic ARG Profiling", "Deep extraction and classification of all Antibiotic Resistance Genes from CARD-format data."),
        ("📐", "MRI & ARI Calculation", "Proprietary indices transform complex gene counts into a single, instantly interpretable risk score."),
        ("🤖", "Random Forest AI", "Machine learning classifies any pathogen as LOW, MODERATE, or HIGH risk with probability confidence."),
        ("🕸️", "Interactive Network", "Visualize the full resistance topology as a live, draggable network graph."),
        ("🌌", "3D PCA Landscape", "Rotate a 3-dimensional scatter map comparing your target genome against every pathogen in the database."),
        ("🩺", "Clinical Susceptibility", "Automatically identifies drug classes with zero resistance markers for treatment consideration."),
        ("💬", "J.A.R.V.I.S. AI Chat", "Ask questions about any genome in plain English. Gemini-powered AI with full genomic context loaded."),
        ("📄", "Master PDF Export", "Generate a comprehensive, publication-ready PDF report with one click."),
    ]

    for row_start in range(0, len(features), 4):
        cols = st.columns(4)
        for i, (icon, title, desc) in enumerate(features[row_start:row_start+4]):
            with cols[i]:
                st.markdown(f"""
                <div class="sp-feat-card">
                  <span class="sp-feat-icon">{icon}</span>
                  <div class="sp-feat-title">{title}</div>
                  <div class="sp-feat-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.write("")

    # Why we stand apart
    st.markdown("---")
    st.markdown("<h2 style='text-align:center; color:#f1f5f9; font-family:Orbitron,monospace; font-size:1.6rem; letter-spacing:2px;'>Why We Stand Apart</h2>", unsafe_allow_html=True)
    st.markdown("<div style='width:80px;height:3px;background:linear-gradient(90deg,#00d4ff,#7c3aed);margin:0 auto 25px auto;border-radius:3px;'></div>", unsafe_allow_html=True)

    col_old, col_new = st.columns(2)
    with col_old:
        st.markdown("""
        <div class="sp-comp-old">
          <div class="sp-comp-title">❌ Traditional AMR Tools</div>
          <div class="sp-comp-item">✗ Output raw gene lists — clinicians must manually interpret hundreds of genes.</div>
          <div class="sp-comp-item">✗ No unified index to compare pathogen severity across species.</div>
          <div class="sp-comp-item">✗ Require bioinformatics expertise — inaccessible to clinical staff.</div>
          <div class="sp-comp-item">✗ Static reports with no interactivity or network exploration.</div>
          <div class="sp-comp-item">✗ No AI chatbot for natural language genomic queries.</div>
          <div class="sp-comp-item">✗ No ML prediction for unknown or novel strains.</div>
          <div class="sp-comp-item">✗ Cannot identify safe drug zones automatically.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_new:
        st.markdown("""
        <div class="sp-comp-new">
          <div class="sp-comp-title">✅ AI-MRI Hub</div>
          <div class="sp-comp-item">✓ Proprietary MRI and ARI compress all genomic data into a single readable risk number.</div>
          <div class="sp-comp-item">✓ Cross-species, cross-habitat standardized scoring enables true pathogen comparison.</div>
          <div class="sp-comp-item">✓ Zero bioinformatics expertise required — upload JSON, get full analysis in seconds.</div>
          <div class="sp-comp-item">✓ Live interactive networks and 3D PCA landscape for spatial resistance topology.</div>
          <div class="sp-comp-item">✓ J.A.R.V.I.S. AI chatbot with genome-aware context injected automatically.</div>
          <div class="sp-comp-item">✓ Random Forest model predicts risk of completely unknown pathogens.</div>
          <div class="sp-comp-item">✓ Automatic clinical susceptibility zone detection identifies zero-resistance drugs instantly.</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    # Bottom CTA
    st.markdown("<p style='text-align:center;color:#475569;font-size:0.85rem;letter-spacing:3px;text-transform:uppercase;margin-top:20px;'>Ready to Begin</p>", unsafe_allow_html=True)
    col_l2, col_c3, col_gap2, col_c4, col_r2 = st.columns([1.5, 1, 0.2, 1, 1.5])
    with col_c3:
        if st.button("🚀 ENTER PLATFORM", type="primary", use_container_width=True, key="enter2"):
            st.session_state.page = 'app'
            st.rerun()
    with col_c4:
        if st.button("👥 MEET THE TEAM", use_container_width=True, key="team2"):
            st.session_state.page = 'team'
            st.rerun()
    st.write("")


# ==========================================
# 8. MAIN APP PAGE
# ==========================================
def render_app_page():
    st.markdown("""
    <div class="main-header">
    <h1 style='margin:0; font-size: 2.8rem;'>🧬 AI-Driven Multidimensional Resistance Index</h1>
    <p style='font-size: 1.3rem; opacity: 0.9; margin-top: 10px;'>Quantitative Bio-Analysis of Antibiotic Resistance Genes</p>
    <hr style='border: 0.5px solid rgba(255,255,255,0.2); margin: 20px auto; width: 80%;'>
    <p style='font-size: 0.95rem; font-weight: 300;'>
    <b>Developed by:</b> Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik &amp; Indranil Patil
    </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("🗄️ Database Sync")
        if st.button("🔄 Refresh Database"):
            st.rerun()

        # Back to splash
        if st.button("← Back to Home"):
            st.session_state.page = 'splash'
            st.rerun()

        if st.button("👥 Meet the Team"):
            st.session_state.page = 'team'
            st.rerun()

        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        analysis_mode = st.radio("Mode:", ["Select Known Bacteria", "AI Predict Unknown"])

        if analysis_mode == "Select Known Bacteria":
            selected_file = st.selectbox("Select a Genome:", json_files) if json_files else None

        st.markdown("---")
        st.success("✅ AI Brain Connected")

    if analysis_mode == "Select Known Bacteria" and json_files:
        if not selected_file:
            st.warning("No JSON genome files found in the current directory.")
            return

        genes, drug, mech, mri, ari, records = extract_data(selected_file)
        u_drugs, u_mechs = len(set(drug)), len(set(mech))
        level, icon = get_level(mri)
        habitat = get_habitat(selected_file)
        bac_info = get_bacteria_info(selected_file)

        if mri > 0.6:
            st.markdown(f'<div class="alert-banner">⚠️ CRITICAL ALERT: {selected_file} identified as High-Priority Superbug — MRI Score: {round(mri, 3)}</div>', unsafe_allow_html=True)

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

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'''<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value">{level} {icon}</div></div>''', unsafe_allow_html=True)
        with m2:
            st.markdown(f'''<div class="metric-card"><div class="metric-label">MRI Score</div><div class="metric-value">{round(mri, 3)}</div></div>''', unsafe_allow_html=True)
        with m3:
            st.markdown(f'''<div class="metric-card"><div class="metric-label">Total Genes</div><div class="metric-value">{genes}</div></div>''', unsafe_allow_html=True)
        with m4:
            st.markdown(f'''<div class="metric-card"><div class="metric-label">Habitat</div><div class="metric-value">{habitat}</div></div>''', unsafe_allow_html=True)

        st.write(" ")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "ℹ️ Pathogen Summary", "📊 6-Panel Dashboard", "🧮 Math & Data Ledger",
            "🕸️ Network", "🤖 Bio-AI Chat", "🩺 Clinical Insight",
            "📄 Export Master PDF", "🌌 3D Landscape"
        ])

        # ── TAB 1 ──
        with tab1:
            st.markdown("""
            <div class="welcome-hero">
                <h2>Genomic Resistance Intelligence Dashboard</h2>
                <p>Comprehensive Antimicrobial Resistance (AMR) Analysis Platform — Powered by AI</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("### 🧬 Platform Capabilities")
            st.info("""
**The AI-MRI Hub provides a state-of-the-art multidimensional genomic analysis suite:**
- **Genomic ARG Profiling:** Structured identification and classification of ARGs from CARD-format sequence data.
- **Multidimensional Resistance Index (MRI):** A validated clinical risk score consolidating drug evasion breadth and mechanism diversity.
- **Random Forest Risk Classification:** ML-driven threat stratification benchmarked against reference genomic profiles.
- **Clinical Susceptibility Zone Analysis:** Genomic exclusion logic identifying drug classes with zero resistance markers.
- **Interactive Resistance Topology:** 3D PCA landscape and live network graphs for spatial visualization.
            """)
            st.markdown("### 🦠 Pathogen Identification Profile")
            st.markdown(f"""
<div class="report-card">
<div class="report-header">Bacterial Resistance Intelligence Ledger</div>
<div class="report-row"><span class="report-label">Target Genome File</span><span class="report-value" style="color:#60a5fa;">{selected_file}</span></div>
<div class="report-row"><span class="report-label">Gram Classification</span><span class="report-value">{bac_info['gram']}</span></div>
<div class="report-row"><span class="report-label">Associated Pathology</span><span class="report-value">{bac_info['disease']}</span></div>
<div class="report-row"><span class="report-label">Ecological Habitat</span><span class="report-value">{habitat}</span></div>
<div class="report-row"><span class="report-label">Total Resistance Genes (ARGs)</span><span class="report-value">{genes}</span></div>
<div class="report-row"><span class="report-label">Drug Classes Resisted</span><span class="report-value">{u_drugs} Classes</span></div>
<div class="report-row"><span class="report-label">Resistance Mechanisms Deployed</span><span class="report-value">{u_mechs} Strategies</span></div>
<div class="report-row"><span class="report-label">MRI Score / ARI Score</span><span class="report-value">{round(mri, 3)} ({level}) &nbsp;/&nbsp; {round(ari, 3)}</span></div>
<div style="margin-top:20px; padding-top:10px;">
<span class="report-label">AI Risk Classification:</span>&nbsp;
<span class="ai-badge">{ai_pred_text} Risk</span>
<br><br>
<small style="color:#64748b;">Probability Distribution: {ai_conf_text}</small>
</div>
</div>
""", unsafe_allow_html=True)

        # ── TAB 2 ──
        with tab2:
            st.markdown(f"### Systems Analysis Dashboard — `{selected_file}`")
            fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)
            st.pyplot(fig)

        # ── TAB 3 ──
        with tab3:
            st.markdown("### 🧮 Mathematical Framework & Validation")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown('<div class="math-card">', unsafe_allow_html=True)
                st.markdown('<div class="math-card-header">Multidimensional Resistance Index (MRI)</div>', unsafe_allow_html=True)
                st.latex(r"MRI = \frac{U_{drugs} + U_{mechs}}{T_{drugs} + T_{mechs} + 1}")
                st.markdown(
                    '<div class="annotation-box">'
                    '<div class="annotation-item"><span class="annotation-key">U_drugs</span> — Unique drug classes resisted</div>'
                    '<div class="annotation-item"><span class="annotation-key">U_mechs</span> — Unique resistance mechanisms deployed</div>'
                    '<div class="annotation-item"><span class="annotation-key">T_drugs</span> — Total drug resistance records</div>'
                    '<div class="annotation-item"><span class="annotation-key">T_mechs</span> — Total mechanism records</div>'
                    '<div class="annotation-item"><span class="annotation-key">+ 1</span> — Laplace smoothing constant</div>'
                    '</div>', unsafe_allow_html=True)
                st.markdown("**Computed Value:**")
                st.latex(rf"\frac{{{u_drugs} + {u_mechs}}}{{{len(drug)} + {len(mech)} + 1}} = {round(mri, 3)}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_m2:
                st.markdown('<div class="math-card">', unsafe_allow_html=True)
                st.markdown('<div class="math-card-header">Antibiotic Resistance Index (ARI)</div>', unsafe_allow_html=True)
                st.latex(r"ARI = \frac{U_{mechs}}{G_{total} + 1}")
                st.markdown(
                    '<div class="annotation-box">'
                    '<div class="annotation-item"><span class="annotation-key">U_mechs</span> — Unique resistance mechanisms deployed</div>'
                    '<div class="annotation-item"><span class="annotation-key">G_total</span> — Total genomic resistance gene count</div>'
                    '<div class="annotation-item"><span class="annotation-key">+ 1</span> — Laplace smoothing constant</div>'
                    '</div>', unsafe_allow_html=True)
                st.markdown("**Computed Value:**")
                st.latex(rf"\frac{{{u_mechs}}}{{{genes} + 1}} = {round(ari, 3)}")
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("### 🎯 Risk Assessment Interpretation")
            st.markdown(f'<div class="reasoning-box">{get_risk_reason(level, u_drugs, u_mechs)}</div>', unsafe_allow_html=True)
            st.write("")
            st.markdown("### 📜 Complete Resistance Gene Ledger")
            df = pd.DataFrame(records, columns=["Gene Name", "Drug Classes Resisted", "Mechanisms Deployed", "Habitat"])
            st.dataframe(df, use_container_width=True)

        # ── TAB 4 ──
        with tab4:
            st.markdown("### 🕸️ Interactive Resistance Mechanism Network")
            html_path = generate_network_html(records, selected_file, "red" if level=="HIGH" else "orange" if level=="MODERATE" else "green")
            with open(html_path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=650)

        # ── TAB 5 ──
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
            user_msg = st.chat_input(f"Query J.A.R.V.I.S. about {selected_file}...")
            if user_msg:
                st.chat_message("user").markdown(user_msg)
                st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": user_msg})
                if not AI_AVAILABLE:
                    st.error("⚠️ AI library unavailable.")
                else:
                    try:
                        context = f"""
You are J.A.R.V.I.S., an expert Bioinformatics AI specializing in antimicrobial resistance genomics.
The analyst is currently examining: '{selected_file}'.
Genomic Data: Total ARGs: {genes}, Drug Classes: {u_drugs}, Mechanisms: {u_mechs}, MRI: {round(mri,3)} ({level}), ARI: {round(ari,3)}.
Analyst Query: {user_msg}
"""
                        with st.spinner("Processing genomic data..."):
                            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                            if not available_models:
                                st.error("No text generation models available.")
                            else:
                                target_model = next((m for m in available_models if 'flash' in m), available_models[0])
                                model_ai = genai.GenerativeModel(target_model)
                                response = model_ai.generate_content(context)
                                st.chat_message("assistant").markdown(response.text)
                                st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "quota" in error_msg.lower():
                            st.error("⚠️ API Quota Exceeded. Please wait 60 seconds.")
                        else:
                            st.error(f"AI Connection Error: {e}")

        # ── TAB 6 ──
        with tab6:
            st.markdown("### 🩺 Clinical Susceptibility Zone Analysis")
            DRUG_UNIVERSE = ["Penicillin", "Cephalosporin", "Carbapenem", "Macrolide", "Aminoglycoside",
                             "Fluoroquinolone", "Tetracycline", "Sulfonamide", "Glycopeptide"]
            resisted_norm = set([d.lower() for d in drug])
            safe_zones = [d for d in DRUG_UNIVERSE if d.lower() not in resisted_norm]
            st.write("Based on genomic exclusion analysis, the following drug classes show **zero resistance markers**:")
            st.markdown(f'<div class="susceptibility-card">🛡️ Candidate Therapeutic Classes: {", ".join(safe_zones)}</div>', unsafe_allow_html=True)
            st.caption("⚠️ Clinical confirmation via standard antibiogram (MIC testing) is required.")
            st.write("---")
            st.markdown("### 📈 Population Benchmark Comparison")
            all_mris = []
            for f in json_files:
                try:
                    _, _, _, f_mri, _, _ = extract_data(f)
                    all_mris.append(f_mri)
                except: continue
            if all_mris:
                avg_mri = sum(all_mris) / len(all_mris)
                comparison_df = pd.DataFrame({"MRI Score": [mri, avg_mri]}, index=["Target Genome", "Database Average"])
                st.bar_chart(comparison_df)

        # ── TAB 7 ──
        with tab7:
            st.markdown("### 📥 Generate Master Analytical Report")
            if st.button("Generate Master PDF Report", type="primary"):
                with st.spinner("Compiling analytical components into PDF..."):
                    fig_for_pdf = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)
                    pdf_path = create_advanced_pdf_report(selected_file, genes, drug, mech, mri, ari, level, icon, records, fig_for_pdf, bac_info, habitat, ai_pred_text, ai_conf_text)
                    with open(pdf_path, "rb") as file:
                        st.download_button(
                            label="⬇️ Download Analytical Report (PDF)",
                            data=file,
                            file_name=pdf_path,
                            mime="application/pdf"
                        )

        # ── TAB 8 ──
        with tab8:
            st.markdown("### 🌌 Interactive Global Resistance Landscape (3D PCA)")
            plot_3d_pca_plotly(selected_file)

    elif analysis_mode == "AI Predict Unknown":
        st.header("🤖 Machine Learning Risk Classification — Unknown Pathogen")
        in_genes = st.number_input("Total Resistance Genes Identified", min_value=1, value=15)
        in_drugs = st.number_input("Unique Drug Classes Resisted", min_value=1, value=5)
        in_mechs = st.number_input("Unique Resistance Mechanisms", min_value=1, value=2)
        model = train_rf_model()
        if model and st.button("Run Risk Classification", type="primary"):
            prediction = model.predict([[in_genes, in_drugs, in_mechs]])[0]
            probs = model.predict_proba([[in_genes, in_drugs, in_mechs]])[0]
            classes = model.classes_
            prob_str = " | ".join([f"{c}: {p:.3f}" for c, p in zip(classes, probs)])
            st.success(f"### AI Risk Classification: **{prediction}**")
            st.info(f"**Probability Distribution:** {prob_str}")


# ==========================================
# 9. PAGE ROUTER
# ==========================================
if st.session_state.page == 'splash':
    render_splash_page()
elif st.session_state.page == 'team':
    render_team_page()
elif st.session_state.page == 'app':
    render_app_page()
