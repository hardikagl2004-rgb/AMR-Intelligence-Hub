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

# ==========================================
# 0. ENVIRONMENT CONFIGURATION
# ==========================================
# Disable matplotlib font warnings
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="AI-MRI Hub | Quantitative Resistance Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #0a0c10;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid #2a2c30;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #1e2025;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        color: #9ca3af;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2d2f36;
        color: #ffffff;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
    }
    
    /* Code block styling */
    .stCodeBlock {
        border-radius: 8px;
    }
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin: 8px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE BACKEND FUNCTIONS
# ==========================================
@st.cache_data
def get_bacteria_info(file_name):
    """Extract bacterial characteristics from filename"""
    name = file_name.lower()
    info = {"gram": "Unknown", "disease": "Various opportunistic infections"}
    
    # Gram-negative bacteria
    gram_negative = {
        "ecoli", "escherichia", "shigella", "salmonella", "klebsiella", 
        "enterobacter", "citrobacter", "serratia", "pseudomonas", 
        "acinetobacter", "vibrio", "proteus", "morganella", "providencia", 
        "campylobacter", "helicobacter", "neisseria", "haemophilus"
    }
    
    # Gram-positive bacteria
    gram_positive = {
        "staphylococcus", "streptococcus", "enterococcus", "bacillus", 
        "clostridium", "listeria", "corynebacterium"
    }
    
    # Specific assignments
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
    """Determine bacterial habitat from filename"""
    name = file_name.lower()
    if any(k in name for k in ["ecoli", "escherichia", "staphylococcus", "salmonella", 
                                "klebsiella", "streptococcus", "enterococcus", "shigella",
                                "proteus", "citrobacter", "enterobacter", "serratia"]):
        return "Clinical"
    elif any(k in name for k in ["pseudomonas", "acinetobacter", "legionella"]):
        return "Environmental"
    elif any(k in name for k in ["bacillus", "clostridium", "mycobacterium", "nocardia"]):
        return "Soil"
    elif "vibrio" in name:
        return "Marine"
    return "General"

@st.cache_data
def extract_data(file_name):
    """Extract resistance data from JSON file"""
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    drug, mech, records = [], [], []
    habitat = get_habitat(file_name)
    
    for key in data:
        try:
            inner = list(data[key].values())[0]
            gene_name = inner.get("ARO_name", key)
            drug_list, mech_list = [], []
            
            for category in inner.get("ARO_category", {}).values():
                category_name = category.get("category_aro_class_name", "").lower()
                category_value = category.get("category_aro_name", "")
                if "drug" in category_name:
                    drug_list.append(category_value)
                elif "mechanism" in category_name:
                    mech_list.append(category_value)
            
            drug.extend(drug_list)
            mech.extend(mech_list)
            records.append((
                gene_name, 
                ", ".join(set(drug_list)), 
                ", ".join(set(mech_list)), 
                habitat
            ))
        except Exception:
            continue

    genes = len(data)
    unique_drugs = len(set(drug))
    unique_mechs = len(set(mech))
    total_occurrences = len(drug) + len(mech)
    
    # Calculate MRI (Multidimensional Resistance Index)
    mri = (unique_drugs + unique_mechs) / (total_occurrences + 1) if total_occurrences > 0 else 0
    
    # Calculate ARI (Antibiotic Resistance Index)
    ari = unique_mechs / (genes + 1) if genes > 0 else 0

    return genes, drug, mech, mri, ari, records

def get_level(mri):
    """Determine risk level based on MRI score"""
    if mri < 0.15:
        return "LOW", "🟢"
    elif mri < 0.35:
        return "MODERATE", "🟡"
    else:
        return "HIGH", "🔴"

def get_risk_reason(level, unique_drugs, unique_mechs):
    """Generate clinical reasoning for risk assessment"""
    if "HIGH" in level:
        return (f"⚠️ **Critical Risk Assessment:** This pathogen exhibits a HIGH Multidimensional Resistance Index. "
                f"It deploys {unique_mechs} distinct resistance mechanisms against {unique_drugs} antibiotic classes. "
                f"This indicates sophisticated genetic adaptation where multiple redundant pathways neutralize standard treatments. "
                f"Clinical intervention requires advanced therapeutic strategies and susceptibility testing.")
    elif "MODERATE" in level:
        return (f"📊 **Elevated Risk Assessment:** This pathogen shows MODERATE resistance complexity with "
                f"{unique_mechs} mechanisms targeting {unique_drugs} drug classes. "
                f"While frontline antibiotics may be compromised, alternative therapeutic options likely remain effective. "
                f"Continued surveillance and combination therapy considerations are recommended.")
    else:
        return (f"✅ **Controlled Risk Assessment:** This pathogen demonstrates LOW resistance complexity. "
                f"The narrow resistance profile ({unique_mechs} mechanisms, {unique_drugs} drug classes) suggests "
                f"specialized environmental adaptation rather than broad clinical resistance. "
                f"Standard antibiotic protocols are likely to remain effective.")

@st.cache_resource
def train_rf_model():
    """Train Random Forest classifier for risk prediction"""
    features_list, labels = [], []
    
    for file in os.listdir('.'):
        if file.endswith('.json'):
            try:
                genes, drug, mech, mri, _, _ = extract_data(file)
                feature_vector = [genes, len(set(drug)), len(set(mech))]
                risk_label = "LOW" if mri < 0.15 else "MODERATE" if mri < 0.35 else "HIGH"
                features_list.append(feature_vector)
                labels.append(risk_label)
            except Exception:
                continue
    
    if len(features_list) < 2:
        return None
        
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(features_list, labels)
    return model

# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================
def plot_full_dashboard(drug, mech, mri, genes, records, name):
    """Generate comprehensive 6-panel visualization dashboard"""
    level, icon = get_level(mri)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0e1117')
    
    risk_color = "#ef4444" if level == "HIGH" else "#f59e0b" if level == "MODERATE" else "#10b981"
    
    # Style all axes
    for ax in axes.flat:
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='#9ca3af')
        ax.title.set_color('#ffffff')
        ax.spines['bottom'].set_color('#2a2c30')
        ax.spines['top'].set_color('#2a2c30')
        ax.spines['left'].set_color('#2a2c30')
        ax.spines['right'].set_color('#2a2c30')

    # Panel 1: Drug Classes (Pie Chart)
    drug_counts = Counter(drug).most_common(8)
    if drug_counts:
        labels = [f"{d[:20]}..." if len(d) > 20 else d for d, _ in drug_counts]
        sizes = [v for _, v in drug_counts]
        axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', 
                       textprops={'color': 'white', 'fontsize': 9},
                       colors=plt.cm.Set3(range(len(drug_counts))))
    axes[0, 0].set_title("Drug Classes Resisted", fontsize=12, fontweight='bold')

    # Panel 2: Mechanisms (Bar Chart)
    mech_counts = Counter(mech)
    if mech_counts:
        mech_labels = [f"{m[:20]}..." if len(m) > 20 else m for m in mech_counts.keys()]
        axes[0, 1].bar(mech_labels, mech_counts.values(), color=risk_color, alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=8)
    axes[0, 1].set_title("Resistance Mechanisms Deployed", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("Frequency", color='#9ca3af')

    # Panel 3: MRI Gauge
    theta = np.linspace(0, np.pi, 100)
    axes[0, 2].plot(np.cos(theta), np.sin(theta), color='#2a2c30', linewidth=2)
    angle = mri * np.pi
    axes[0, 2].plot([0, np.cos(angle)], [0, np.sin(angle)], color=risk_color, linewidth=6)
    axes[0, 2].plot(0, 0, 'o', color=risk_color, markersize=8)
    axes[0, 2].axis('off')
    axes[0, 2].set_title(f"MRI Score: {round(mri, 3)} ({level})", fontsize=12, fontweight='bold', color=risk_color)

    # Panel 4: Top Genes
    gene_list = [record[0] for record in records]
    gene_counts = Counter(gene_list).most_common(5)
    if gene_counts:
        gene_labels = [f"{g[:20]}..." if len(g) > 20 else g for g, _ in gene_counts]
        axes[1, 0].bar(gene_labels, [v for _, v in gene_counts], color='#3b82f6', alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[1, 0].tick_params(axis='x', rotation=45, labelsize=8)
    axes[1, 0].set_title("Most Prevalent Resistance Genes", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Frequency", color='#9ca3af')

    # Panel 5: Gene Count
    axes[1, 1].bar(["Total ARGs"], [genes], color='#8b5cf6', alpha=0.8, edgecolor='white', linewidth=0.5)
    axes[1, 1].set_title(f"Total Resistance Genes: {genes}", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Count", color='#9ca3af')

    # Panel 6: Diversity Comparison
    axes[1, 2].bar(["Drug Classes", "Mechanisms"], [len(set(drug)), len(set(mech))], 
                   color=['#ef4444', '#f59e0b'], alpha=0.8, edgecolor='white', linewidth=0.5)
    axes[1, 2].set_title("Resistance Diversity Profile", fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel("Unique Categories", color='#9ca3af')

    fig.tight_layout()
    return fig

def generate_network_html(records, organism_name, risk_color):
    """Generate interactive network visualization"""
    net = Network(
        height='650px', 
        width='100%', 
        bgcolor='#111318', 
        font_color='#ffffff',
        cdn_resources="in_line",
        select_menu=True,
        filter_menu=True
    )
    
    # Color mapping
    color_map = {"HIGH": "#ef4444", "MODERATE": "#f59e0b", "LOW": "#10b981"}
    hub_color = color_map.get(risk_color, "#6b7280")
    
    # Add central hub
    net.add_node("HUB", label=organism_name, color=hub_color, size=35, title="Primary Organism")
    
    # Add gene and mechanism nodes
    for gene_name, drugs, mechanisms, habitat in records:
        net.add_node(gene_name, label=gene_name[:25], color="#3b82f6", size=20, title=f"Gene: {gene_name}\nHabitat: {habitat}")
        net.add_edge("HUB", gene_name, color="#6b7280", width=1)
        
        if mechanisms:
            for mech in set(mechanisms.split(", ")):
                if mech:
                    net.add_node(mech, label=mech[:25], color="#f59e0b", size=12, shape="box", title=f"Mechanism: {mech}")
                    net.add_edge(gene_name, mech, color="#9ca3af", width=0.8)
    
    # Configure physics
    net.barnes_hut(gravity=-4000, central_gravity=0.3, spring_length=95, spring_strength=0.1, damping=0.09)
    net.set_options("""
    var options = {
        nodes: {
            font: { size: 12 },
            borderWidth: 1,
            shadow: { enabled: true, size: 5 }
        },
        edges: {
            smooth: { type: 'continuous' },
            shadow: { enabled: false }
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            navigationButtons: true
        }
    }
    """)
    
    html_path = "temp_network.html"
    net.save_graph(html_path)
    return html_path

def plot_3d_pca_plotly(current_file):
    """Generate interactive 3D PCA visualization"""
    feature_matrix, file_names, risk_labels = [], [], []
    
    for file in os.listdir('.'):
        if file.endswith('.json'):
            try:
                genes, drug, mech, mri, _, _ = extract_data(file)
                feature_matrix.append([mri, len(set(mech)), len(set(drug))])
                file_names.append(file)
                level, _ = get_level(mri)
                risk_labels.append("TARGET" if file == current_file else level)
            except Exception:
                continue
    
    if len(feature_matrix) < 3:
        st.warning("⚠️ Insufficient data for 3D visualization (minimum 3 samples required).")
        return
        
    # PCA Transformation
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(feature_matrix)
    
    # Create DataFrame
    df_pca = pd.DataFrame(
        pca_result, 
        columns=['Resistance PC1', 'Mechanism PC2', 'Diversity PC3']
    )
    df_pca['Genome'] = file_names
    df_pca['Risk Category'] = risk_labels
    
    # Color mapping
    color_discrete_map = {
        "HIGH": "#ef4444",
        "MODERATE": "#f59e0b", 
        "LOW": "#10b981",
        "TARGET": "#fbbf24"
    }
    
    # Create plot
    fig = px.scatter_3d(
        df_pca, 
        x='Resistance PC1', 
        y='Mechanism PC2', 
        z='Diversity PC3',
        color='Risk Category', 
        hover_name='Genome', 
        color_discrete_map=color_discrete_map,
        opacity=0.85,
        size_max=12,
        title="Global Resistance Landscape (PCA Projection)"
    )
    
    # Highlight target
    fig.update_traces(
        marker=dict(size=6, line=dict(width=2, color='#ffffff')),
        selector=dict(name="TARGET")
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='#0a0c10',
        font_color='#ffffff',
        scene=dict(
            xaxis=dict(backgroundcolor="#0a0c10", gridcolor="#2a2c30", title_font=dict(color="#ffffff")),
            yaxis=dict(backgroundcolor="#0a0c10", gridcolor="#2a2c30", title_font=dict(color="#ffffff")),
            zaxis=dict(backgroundcolor="#0a0c10", gridcolor="#2a2c30", title_font=dict(color="#ffffff")),
            bgcolor='#0a0c10'
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. PDF REPORT GENERATOR
# ==========================================
def create_advanced_pdf_report(bac_name, genes, drug, mech, mri, ari, level, icon, records, dashboard_fig, bac_info, habitat, ai_pred_text, ai_conf_text):
    """Generate professional PDF report"""
    pdf_file = f"{bac_name.replace('.json', '')}_Resistance_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_file, 
        pagesize=letter, 
        rightMargin=40, 
        leftMargin=40, 
        topMargin=40, 
        bottomMargin=40,
        title=f"Resistance Report - {bac_name}"
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=20,
        textColor=colors.HexColor('#1e3a8a'),
        alignment=1  # Center alignment
    )
    
    section_style = ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10,
        textColor=colors.HexColor('#2563eb'),
        borderWidth=0,
        borderPadding=5
    )
    
    normal_style = styles['Normal']
    
    unique_drugs = len(set(drug))
    unique_mechs = len(set(mech))
    total_occurrences = len(drug) + len(mech)
    
    elements = []
    
    # Title
    elements.append(Paragraph(f"🧬 AI-MRI Resistance Analysis Report", title_style))
    elements.append(Paragraph(f"<b>Organism:</b> {bac_name.replace('.json', '')}", normal_style))
    elements.append(Spacer(1, 10))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", section_style))
    summary_text = f"""
    <table width="100%" cellpadding="5">
    <tr><td width="40%"><b>Gram Stain:</b></td><td>{bac_info['gram']}</td></tr>
    <tr><td><b>Common Disease:</b></td><td>{bac_info['disease']}</td></tr>
    <tr><td><b>Primary Habitat:</b></td><td>{habitat}</td></tr>
    <tr><td><b>Total ARGs Detected:</b></td><td>{genes}</td></tr>
    <tr><td><b>Unique Drug Classes:</b></td><td>{unique_drugs}</td></tr>
    <tr><td><b>Unique Mechanisms:</b></td><td>{unique_mechs}</td></tr>
    <tr><td><b>MRI Score:</b></td><td>{round(mri, 3)} ({level})</td></tr>
    <tr><td><b>ARI Score:</b></td><td>{round(ari, 3)}</td></tr>
    <tr><td><b>AI Risk Prediction:</b></td><td>{ai_pred_text}</td></tr>
    </table>
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 15))
    
    # Risk Assessment
    elements.append(Paragraph("Clinical Risk Assessment", section_style))
    risk_reason = get_risk_reason(level, unique_drugs, unique_mechs)
    elements.append(Paragraph(risk_reason, normal_style))
    elements.append(Spacer(1, 15))
    
    # Methodology
    elements.append(Paragraph("Metric Calculations", section_style))
    calc_text = f"""
    <b>Multidimensional Resistance Index (MRI):</b><br/>
    Formula: (Unique Drugs + Unique Mechanisms) / (Total Occurrences + 1)<br/>
    Calculation: ({unique_drugs} + {unique_mechs}) / ({total_occurrences} + 1) = <b>{round(mri, 3)}</b><br/><br/>
    
    <b>Antibiotic Resistance Index (ARI):</b><br/>
    Formula: Unique Mechanisms / (Total Genes + 1)<br/>
    Calculation: {unique_mechs} / ({genes} + 1) = <b>{round(ari, 3)}</b>
    """
    elements.append(Paragraph(calc_text, normal_style))
    elements.append(Spacer(1, 15))
    
    # Dashboard Visualization
    elements.append(PageBreak())
    elements.append(Paragraph("Resistance Profile Dashboard", section_style))
    
    # Convert matplotlib figure for PDF
    buf = io.BytesIO()
    dashboard_fig.patch.set_facecolor('white')
    for ax in dashboard_fig.axes:
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        ax.title.set_color('black')
        for text in ax.texts:
            text.set_color('black')
        for spine in ax.spines.values():
            spine.set_color('black')
    
    dashboard_fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
    buf.seek(0)
    dashboard_img = RLImage(buf, width=7.2*inch, height=5*inch)
    elements.append(dashboard_img)
    elements.append(Spacer(1, 10))
    
    # Gene Ledger
    elements.append(PageBreak())
    elements.append(Paragraph("Complete Resistance Gene Inventory", section_style))
    
    table_data = [["Gene", "Drug Classes", "Mechanisms", "Habitat"]]
    for gene, drugs, mechanisms, hab in records[:50]:  # Limit to 50 for PDF size
        table_data.append([
            Paragraph(gene[:40], normal_style), 
            Paragraph(drugs[:50], normal_style), 
            Paragraph(mechanisms[:50], normal_style), 
            Paragraph(hab, normal_style)
        ])
    
    if len(records) > 50:
        table_data.append(["...", f"and {len(records) - 50} more genes", "...", "..."])
    
    gene_table = Table(table_data, colWidths=[1.5*inch, 2*inch, 2*inch, 1*inch])
    gene_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 10),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8fafc')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
    ]))
    
    elements.append(gene_table)
    
    # Footer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("<i>Report generated by AI-MRI Hub - Quantitative Resistance Analysis Platform</i>", 
                              ParagraphStyle(name='Footer', parent=normal_style, alignment=1, fontSize=8, textColor=colors.grey)))
    
    doc.build(elements)
    return pdf_file

# ==========================================
# 5. AI CHATBACKEND
# ==========================================
try:
    import google.generativeai as genai
    AI_AVAILABLE = True
    # Configure API key (consider using environment variables in production)
    genai.configure(api_key="AIzaSyBMDZ-1JJlk08cAlvQk6XIAD4vkeWAadQ4")
except ImportError:
    AI_AVAILABLE = False
    st.warning("⚠️ Google Generative AI library not available. AI chat features will be disabled.")

def get_ai_response(user_message, selected_file, genes, unique_drugs, unique_mechs, mri, level):
    """Generate AI response using Gemini API"""
    if not AI_AVAILABLE:
        return "AI services are currently unavailable. Please check your installation of google-generativeai."
    
    try:
        context = f"""
        You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), an expert Bioinformatics AI specializing in antimicrobial resistance analysis.
        
        CURRENT ANALYSIS CONTEXT:
        - Organism File: {selected_file}
        - Total Resistance Genes: {genes}
        - Unique Drug Classes Resisted: {unique_drugs}
        - Unique Resistance Mechanisms: {unique_mechs}
        - MRI Score: {round(mri, 3)} ({level} Risk)
        
        USER QUERY: {user_message}
        
        Please provide a professional, informative response based on the genomic data above. 
        Focus on clinical relevance, resistance mechanisms, and practical implications.
        Keep responses concise but thorough. Use bullet points where appropriate.
        """
        
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not available_models:
            return "No AI models available. Please check your API configuration."
        
        target_model = next((m for m in available_models if 'flash' in m), 
                           next((m for m in available_models if 'pro' in m), available_models[0]))
        model = genai.GenerativeModel(target_model)
        response = model.generate_content(context)
        return response.text
        
    except Exception as e:
        error_str = str(e)
        if "429" in error_str:
            return "⚠️ **Rate Limit Exceeded** - The API has reached its request limit. Please wait a moment before sending another message."
        elif "API key" in error_str:
            return "⚠️ **API Configuration Error** - Please check your API key configuration."
        else:
            return f"⚠️ **AI Error** - {error_str[:200]}"

# ==========================================
# 6. MAIN APPLICATION
# ==========================================
def main():
    """Main application entry point"""
    
    # Header
    st.title("🧬 AI-MRI Hub")
    st.markdown("### Quantitative Analysis of Antimicrobial Resistance Genes")
    st.caption("Developed by: Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik, Indranil Patil")
    st.divider()
    
    # Initialize session state
    if 'chat_sessions' not in st.session_state:
        st.session_state.chat_sessions = {"Default Session": []}
        st.session_state.current_session = "Default Session"
        st.session_state.chat_counter = 1
    
    # Sidebar
    with st.sidebar:
        st.header("🗄️ Data Management")
        
        # File refresh
        if st.button("🔄 Refresh Database", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        
        # File selection
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        
        if not json_files:
            st.error("⚠️ No JSON files found in current directory.")
            st.info("Please ensure your resistance gene data files (.json) are in the same directory as this application.")
            return
        
        analysis_mode = st.radio(
            "Analysis Mode",
            ["🔬 Select Known Bacteria", "🤖 Predict Unknown Sample"],
            help="Choose between analyzing existing genomic data or predicting risk for new samples"
        )
        
        if analysis_mode == "🔬 Select Known Bacteria":
            selected_file = st.selectbox("Select Genome File:", json_files)
        else:
            selected_file = None
            
        st.divider()
        st.success("✅ AI Engine Ready")
        st.caption("Powered by Google Gemini AI")
    
    # Main content based on mode
    if analysis_mode == "🔬 Select Known Bacteria" and selected_file:
        # Extract data
        genes, drug, mech, mri, ari, records = extract_data(selected_file)
        unique_drugs = len(set(drug))
        unique_mechs = len(set(mech))
        level, icon = get_level(mri)
        habitat = get_habitat(selected_file)
        bac_info = get_bacteria_info(selected_file)
        
        # AI Prediction
        model = train_rf_model()
        ai_pred_text = "N/A"
        ai_conf_text = "N/A"
        
        if model:
            prediction = model.predict([[genes, unique_drugs, unique_mechs]])[0]
            probabilities = model.predict_proba([[genes, unique_drugs, unique_mechs]])[0]
            confidence_dict = {str(cls): round(prob, 3) for cls, prob in zip(model.classes_, probabilities)}
            ai_pred_text = prediction
