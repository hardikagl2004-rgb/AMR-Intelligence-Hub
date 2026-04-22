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
    genai.configure(api_key="AIzaSyBMDZ-1JJlk08cAlvQk6XIAD4vkeWAadQ4")
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
        ('VALIGN', (0,0), (-
