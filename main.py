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
except ImportError:
    AI_AVAILABLE = False

# ==========================================
# 1. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon="🧬")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

.main {background-color: #060d1a;}
h1, h2, h3 {color: #ffffff;}

/* ===== ANIMATED WELCOME PAGE ===== */
.welcome-screen {
    min-height: 100vh;
    background: #060d1a;
    position: relative;
    overflow: hidden;
}

/* Starfield canvas */
.star-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 0;
    pointer-events: none;
}

.welcome-content {
    position: relative;
    z-index: 10;
    padding: 40px 20px;
}

/* DNA helix animation container */
.dna-container {
    display: flex;
    justify-content: center;
    margin: 30px auto;
}

.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(2rem, 5vw, 4rem);
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #00f5ff 0%, #0080ff 50%, #8000ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: titleGlow 3s ease-in-out infinite alternate;
    margin-bottom: 10px;
    letter-spacing: 3px;
}

@keyframes titleGlow {
    from { filter: drop-shadow(0 0 20px rgba(0,245,255,0.5)); }
    to { filter: drop-shadow(0 0 40px rgba(128,0,255,0.8)); }
}

.hero-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3rem;
    text-align: center;
    color: #64b5f6;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 50px;
    animation: fadeInUp 1s ease 0.5s both;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Feature Cards */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    max-width: 1200px;
    margin: 40px auto;
    padding: 0 20px;
}

.feature-card {
    background: rgba(0, 20, 60, 0.6);
    border: 1px solid rgba(0, 245, 255, 0.2);
    border-radius: 16px;
    padding: 30px 20px;
    text-align: center;
    transition: all 0.3s ease;
    animation: cardEntrance 0.8s ease both;
    backdrop-filter: blur(10px);
}

.feature-card:nth-child(1) { animation-delay: 0.2s; }
.feature-card:nth-child(2) { animation-delay: 0.4s; }
.feature-card:nth-child(3) { animation-delay: 0.6s; }
.feature-card:nth-child(4) { animation-delay: 0.8s; }
.feature-card:nth-child(5) { animation-delay: 1.0s; }
.feature-card:nth-child(6) { animation-delay: 1.2s; }

@keyframes cardEntrance {
    from { opacity: 0; transform: translateY(50px) scale(0.9); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

.feature-card:hover {
    border-color: rgba(0, 245, 255, 0.6);
    background: rgba(0, 40, 100, 0.8);
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0, 245, 255, 0.15);
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 15px;
    display: block;
}

.feature-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.9rem;
    color: #00f5ff;
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 2px;
}

.feature-desc {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
    color: #8eb4d4;
    line-height: 1.6;
}

/* Team Section */
.team-section {
    max-width: 1200px;
    margin: 60px auto;
    padding: 0 20px;
    text-align: center;
}

.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.5rem;
    color: #00f5ff;
    letter-spacing: 4px;
    margin-bottom: 10px;
    text-transform: uppercase;
}

.section-line {
    width: 100px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00f5ff, transparent);
    margin: 0 auto 40px;
}

.team-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 20px;
    margin-top: 30px;
}

.team-card {
    background: rgba(0, 15, 45, 0.8);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 25px 15px;
    transition: all 0.3s ease;
    animation: cardEntrance 1s ease both;
}

.team-card:nth-child(1) { animation-delay: 0.1s; }
.team-card:nth-child(2) { animation-delay: 0.2s; }
.team-card:nth-child(3) { animation-delay: 0.3s; }
.team-card:nth-child(4) { animation-delay: 0.4s; }
.team-card:nth-child(5) { animation-delay: 0.5s; }
.team-card:nth-child(6) { animation-delay: 0.6s; }
.team-card:nth-child(7) { animation-delay: 0.7s; }

.team-card:hover {
    border-color: rgba(59, 130, 246, 0.7);
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
}

.team-avatar {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    margin: 0 auto 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    color: #ffffff;
    border: 2px solid rgba(59, 130, 246, 0.5);
    position: relative;
    overflow: hidden;
}

.team-avatar::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent 40%, rgba(255,255,255,0.1) 50%, transparent 60%);
    animation: avatarShine 3s ease infinite;
}

@keyframes avatarShine {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.team-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
    color: #e2e8f0;
    font-weight: 600;
    letter-spacing: 1px;
}

.team-role {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #4a9eff;
    margin-top: 4px;
    letter-spacing: 1px;
}

/* Stats bar */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 40px;
    flex-wrap: wrap;
    padding: 40px 20px;
    background: rgba(0, 20, 60, 0.4);
    border-top: 1px solid rgba(0, 245, 255, 0.1);
    border-bottom: 1px solid rgba(0, 245, 255, 0.1);
    margin: 40px 0;
}

.stat-item {
    text-align: center;
    animation: countUp 1s ease both;
}

.stat-number {
    font-family: 'Orbitron', monospace;
    font-size: 2.5rem;
    font-weight: 900;
    color: #00f5ff;
    display: block;
    text-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
}

.stat-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    color: #64b5f6;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Tech info section */
.tech-section {
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
}

.tech-card {
    background: rgba(0, 15, 45, 0.7);
    border: 1px solid rgba(0, 245, 255, 0.15);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

.tech-card-title {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    color: #00f5ff;
    margin-bottom: 15px;
    letter-spacing: 2px;
}

.tech-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.tech-list li {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    color: #8eb4d4;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    gap: 10px;
}

.tech-list li::before {
    content: '▶';
    color: #00f5ff;
    font-size: 0.6rem;
}

/* Enter button */
.enter-btn-wrapper {
    text-align: center;
    margin: 50px 0;
}

/* ===== TAB SELECTOR STYLES ===== */
.tab-selector-container {
    background: rgba(0, 15, 45, 0.8);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 25px;
}

.tab-selector-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    color: #4a9eff;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

/* Professional Header */
.main-header {
    background: linear-gradient(90deg, #060d1a 0%, #0a1628 50%, #060d1a 100%);
    border-bottom: 1px solid rgba(0, 245, 255, 0.2);
    padding: 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 300%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00f5ff, #0080ff, #8000ff, transparent);
    animation: headerScan 3s linear infinite;
}

@keyframes headerScan {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Alert Banner */
.alert-banner {
    background: rgba(239, 68, 68, 0.15);
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

/* Metric Cards */
.metric-card {
    background: rgba(0, 15, 45, 0.8);
    border: 1px solid rgba(59, 130, 246, 0.3);
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(0, 245, 255, 0.5);
    transform: translateY(-2px);
}
.metric-label { color: #64b5f6; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; letter-spacing: 0.05em; font-family: 'Rajdhani', sans-serif; }
.metric-value { color: #ffffff; font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem; font-family: 'Orbitron', monospace; }

/* Report Cards */
.report-card { background: rgba(0, 15, 45, 0.7); border-left: 4px solid #00f5ff; padding: 20px; border-radius: 10px; margin-bottom: 25px; }
.report-header { color: #00f5ff; font-weight: bold; text-transform: uppercase; font-size: 0.85rem; margin-bottom: 15px; border-bottom: 1px solid rgba(0,245,255,0.2); padding-bottom: 5px; font-family: 'Orbitron', monospace; letter-spacing: 2px; }
.report-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
.report-label { color: #64b5f6; font-weight: 500; }
.report-value { color: #ffffff; font-weight: 600; }
.ai-badge { background: rgba(0, 245, 255, 0.1); color: #00f5ff; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid rgba(0, 245, 255, 0.3); }

/* Math Cards */
.math-card { background: rgba(0, 15, 45, 0.7); border: 1px solid rgba(59, 130, 246, 0.25); padding: 25px; border-radius: 12px; margin-bottom: 20px; }
.math-card-header { color: #00f5ff; font-size: 0.85rem; font-weight: bold; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 15px; font-family: 'Orbitron', monospace; }
.annotation-box { background: rgba(0, 0, 0, 0.3); border: 1px dashed rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 12px; margin-top: 10px; font-size: 0.85rem; color: #8eb4d4; }
.annotation-item { margin-bottom: 4px; }
.annotation-key { color: #00f5ff; font-weight: bold; font-family: monospace; }

.susceptibility-card { background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 10px; padding: 15px; color: #10b981; font-weight: 600; }
.reasoning-box { background: rgba(0, 15, 45, 0.8); border-radius: 12px; border-left: 5px solid #00f5ff; padding: 20px; color: #e2e8f0; line-height: 1.7; font-size: 1.05rem; }

/* Chatbot Styles */
.chat-container {
    background: rgba(0, 10, 30, 0.9);
    border: 1px solid rgba(0, 245, 255, 0.2);
    border-radius: 16px;
    padding: 20px;
    min-height: 400px;
    margin-bottom: 20px;
}

.chat-header {
    font-family: 'Orbitron', monospace;
    font-size: 1rem;
    color: #00f5ff;
    letter-spacing: 3px;
    margin-bottom: 20px;
    text-align: center;
    border-bottom: 1px solid rgba(0, 245, 255, 0.15);
    padding-bottom: 15px;
}

.chat-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 20px;
    padding: 5px 15px;
    font-size: 0.8rem;
    color: #10b981;
    margin-bottom: 15px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    animation: blink 1.5s infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Bacteria Selector Panel */
.bacteria-panel {
    background: rgba(0, 10, 30, 0.9);
    border: 1px solid rgba(0, 245, 255, 0.25);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
}

.panel-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    color: #00f5ff;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 15px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
::-webkit-scrollbar-thumb { background: rgba(0, 245, 255, 0.3); border-radius: 3px; }

[data-testid="stDataFrame"] { border: 1px solid rgba(0, 245, 255, 0.2); border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE BACKEND FUNCTIONS
# ==========================================
TEAM_MEMBERS = [
    {"name": "Hardik Agrawal", "initials": "HA", "role": "Lead Developer"},
    {"name": "Poorva Dongarkar", "initials": "PD", "role": "Bio-Analyst"},
    {"name": "Yashraj Patil", "initials": "YP", "role": "ML Engineer"},
    {"name": "Avani Laswante", "initials": "AL", "role": "Data Scientist"},
    {"name": "Zeel Bhanushali", "initials": "ZB", "role": "UI Designer"},
    {"name": "Aayushi Wasnik", "initials": "AW", "role": "Researcher"},
    {"name": "Indranil Patil", "initials": "IP", "role": "Systems Architect"},
]

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
    color_discrete_map = {"HIGH": "red", "MODERATE": "orange", "LOW": "green", "TARGET 🎯": "gold"}
    fig = px.scatter_3d(df_pca, x='Overall Resistance (PC1)', y='Mechanism Diversity (PC2)', z='Genetic Density (PC3)',
                      color='Risk Category', hover_name='Genome', color_discrete_map=color_discrete_map, opacity=0.8, size_max=10)
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
Traditional genomic analysis often simply lists detected genes, which fails to quantify the actual danger a pathogen poses.
Our framework utilizes two calculated metrics to solve this:<br/><br/>
<b>Why ARI is Required:</b> The Antibiotic Resistance Index (ARI) calculates the <i>density</i> of the threat by normalizing the unique mechanisms against the total gene count.<br/><br/>
<b>Why MRI Helps:</b> The Multidimensional Resistance Index (MRI) mathematically consolidates the diversity of resisted drugs and deployed mechanisms into a single, standardized risk score.
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
# 5. SESSION STATE INIT
# ==========================================
if 'show_welcome' not in st.session_state:
    st.session_state.show_welcome = True
if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
    st.session_state.current_session = "Chat 1"
    st.session_state.chat_counter = 1
if 'selected_panel' not in st.session_state:
    st.session_state.selected_panel = "ℹ️ Pathogen Summary"
# Per-bacteria chat sessions
if 'bacteria_chat' not in st.session_state:
    st.session_state.bacteria_chat = {}

# ==========================================
# 6. ANIMATED WELCOME PAGE
# ==========================================
def render_welcome_page():
    team_cards_html = ""
    colors_list = ["#1e3a8a", "#1e4d8a", "#1e6a8a", "#1e8a7a", "#1e8a5a", "#3a6a1e", "#6a3a1e"]
    for i, member in enumerate(TEAM_MEMBERS):
        team_cards_html += f"""
        <div class="team-card" style="animation-delay: {0.1*(i+1)}s;">
            <div class="team-avatar" style="background: linear-gradient(135deg, {colors_list[i % len(colors_list)]}, #3b82f6);">{member['initials']}</div>
            <div class="team-name">{member['name']}</div>
            <div class="team-role">{member['role']}</div>
        </div>
        """

    welcome_html = f"""
    <div style="background: #060d1a; min-height: 100vh; padding: 20px 0; position: relative; overflow: hidden;">
        
        <!-- Animated background particles -->
        <canvas id="particleCanvas" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none;"></canvas>
        
        <!-- DNA Helix Animation -->
        <div style="position: relative; z-index: 10;">
            <div style="text-align: center; padding: 40px 20px 20px;">
                <div style="margin: 0 auto 30px; width: 120px; height: 120px; position: relative;">
                    <svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" width="120" height="120">
                        <defs>
                            <radialGradient id="glowGrad" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" style="stop-color:#00f5ff;stop-opacity:0.3"/>
                                <stop offset="100%" style="stop-color:#00f5ff;stop-opacity:0"/>
                            </radialGradient>
                        </defs>
                        <circle cx="60" cy="60" r="55" fill="url(#glowGrad)"/>
                        <circle cx="60" cy="60" r="50" fill="none" stroke="rgba(0,245,255,0.1)" stroke-width="1"/>
                        <circle cx="60" cy="60" r="35" fill="none" stroke="rgba(0,245,255,0.15)" stroke-width="1"/>
                        <g style="animation: spin 8s linear infinite; transform-origin: 60px 60px;">
                            <circle cx="60" cy="25" r="8" fill="#00f5ff" opacity="0.9"/>
                            <circle cx="60" cy="95" r="8" fill="#8000ff" opacity="0.9"/>
                            <line x1="60" y1="33" x2="60" y2="87" stroke="rgba(0,245,255,0.4)" stroke-width="1"/>
                        </g>
                        <g style="animation: spin 8s linear infinite reverse; transform-origin: 60px 60px;">
                            <circle cx="25" cy="60" r="6" fill="#0080ff" opacity="0.8"/>
                            <circle cx="95" cy="60" r="6" fill="#00f5ff" opacity="0.8"/>
                        </g>
                        <text x="60" y="65" text-anchor="middle" fill="#00f5ff" font-size="14" font-family="monospace" font-weight="bold">🧬</text>
                    </svg>
                </div>
                
                <h1 class="hero-title">AI-MRI HUB</h1>
                <p class="hero-subtitle">Multidimensional Resistance Intelligence Platform</p>
            </div>
            
            <!-- Stats Bar -->
            <div class="stats-bar">
                <div class="stat-item">
                    <span class="stat-number">8+</span>
                    <span class="stat-label">Analysis Modules</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">3</span>
                    <span class="stat-label">AI Risk Levels</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">100+</span>
                    <span class="stat-label">Gene Patterns</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">7</span>
                    <span class="stat-label">Team Members</span>
                </div>
            </div>
            
            <!-- Feature Cards -->
            <div class="features-grid">
                <div class="feature-card">
                    <span class="feature-icon">🧬</span>
                    <div class="feature-title">Genomic ARG Profiling</div>
                    <div class="feature-desc">Absolute identification and analysis of Antibiotic Resistance Genes from raw sequence data.</div>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">📊</span>
                    <div class="feature-title">MRI Scoring Engine</div>
                    <div class="feature-desc">Unified clinical risk score consolidating drug evasion and mechanism diversity into one metric.</div>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">🤖</span>
                    <div class="feature-title">Random Forest AI</div>
                    <div class="feature-desc">Machine learning threat classification benchmarked against global genomic populations.</div>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">🕸️</span>
                    <div class="feature-title">Network Mapping</div>
                    <div class="feature-desc">Interactive resistance gene network visualization with real-time filtering and exploration.</div>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">🌌</span>
                    <div class="feature-title">3D PCA Landscape</div>
                    <div class="feature-desc">Spatial visualization of mechanism diversity and genetic density across all loaded genomes.</div>
                </div>
                <div class="feature-card">
                    <span class="feature-icon">💬</span>
                    <div class="feature-title">J.A.R.V.I.S. Bio-AI</div>
                    <div class="feature-desc">Contextual AI chatbot that answers any bioinformatics question about your selected pathogen.</div>
                </div>
            </div>
            
            <!-- How It Works -->
            <div class="tech-section">
                <div class="section-title">HOW IT WORKS</div>
                <div class="section-line"></div>
                <div class="tech-card">
                    <div class="tech-card-title">🔬 ANALYSIS PIPELINE</div>
                    <ul class="tech-list">
                        <li>Upload CARD/RGI JSON resistance gene annotation files</li>
                        <li>System parses ARG categories: drug classes and resistance mechanisms</li>
                        <li>MRI and ARI indices calculated using Laplace-smoothed normalization</li>
                        <li>Random Forest model classifies risk: LOW / MODERATE / HIGH</li>
                        <li>6-panel dashboard, network map, and 3D PCA landscape generated</li>
                        <li>J.A.R.V.I.S. AI provides contextual clinical insights on demand</li>
                    </ul>
                </div>
                <div class="tech-card">
                    <div class="tech-card-title">📐 CORE FORMULAS</div>
                    <ul class="tech-list">
                        <li>MRI = (Unique Drugs + Unique Mechanisms) / (Total Drugs + Total Mechanisms + 1)</li>
                        <li>ARI = Unique Mechanisms / (Total Genes + 1)</li>
                        <li>Risk: LOW if MRI &lt; 0.15 | MODERATE if MRI &lt; 0.35 | HIGH if MRI ≥ 0.35</li>
                    </ul>
                </div>
            </div>
            
            <!-- Team Section -->
            <div class="team-section">
                <div class="section-title">THE TEAM</div>
                <div class="section-line"></div>
                <p style="font-family: 'Rajdhani', sans-serif; color: #8eb4d4; font-size: 1rem; margin-bottom: 30px;">
                    A multidisciplinary group of researchers, developers, and data scientists dedicated to advancing antimicrobial resistance intelligence.
                </p>
                <div class="team-grid">
                    {team_cards_html}
                </div>
            </div>
        </div>
        
        <style>
        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        </style>
        
        <script>
        const canvas = document.getElementById('particleCanvas');
        if (canvas) {{
            const ctx = canvas.getContext('2d');
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            
            const particles = [];
            for (let i = 0; i < 80; i++) {{
                particles.push({{
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    vx: (Math.random() - 0.5) * 0.5,
                    vy: (Math.random() - 0.5) * 0.5,
                    r: Math.random() * 2 + 0.5,
                    alpha: Math.random() * 0.5 + 0.2,
                    color: Math.random() > 0.5 ? '#00f5ff' : '#0080ff'
                }});
            }}
            
            function animate() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                particles.forEach(p => {{
                    p.x += p.vx;
                    p.y += p.vy;
                    if (p.x < 0) p.x = canvas.width;
                    if (p.x > canvas.width) p.x = 0;
                    if (p.y < 0) p.y = canvas.height;
                    if (p.y > canvas.height) p.y = 0;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                    ctx.fillStyle = p.color;
                    ctx.globalAlpha = p.alpha;
                    ctx.fill();
                }});
                ctx.globalAlpha = 1;
                requestAnimationFrame(animate);
            }}
            animate();
        }}
        </script>
    </div>
    """
    
    components.html(welcome_html, height=2200, scrolling=True)

# ==========================================
# 7. MAIN APP LAYOUT
# ==========================================
json_files = [f for f in os.listdir('.') if f.endswith('.json')]

# Navigation
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 20px;">
        <div style="font-family: 'Orbitron', monospace; font-size: 1.1rem; color: #00f5ff; letter-spacing: 3px;">AI-MRI HUB</div>
        <div style="font-size: 0.7rem; color: #64b5f6; letter-spacing: 2px; margin-top: 4px;">NAVIGATION</div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("", ["🏠 Welcome", "🔬 Analysis"], label_visibility="collapsed")
    
    if page == "🔬 Analysis":
        st.markdown("---")
        st.markdown('<div style="font-size: 0.8rem; color: #64b5f6; letter-spacing: 2px; margin-bottom: 8px;">GENOME DATABASE</div>', unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Database"):
            st.rerun()
        
        analysis_mode = st.radio("Analysis Mode:", ["Select Known Bacteria", "AI Predict Unknown"])
        
        if analysis_mode == "Select Known Bacteria" and json_files:
            selected_file = st.selectbox("Select Genome File:", json_files)
            
            st.markdown("---")
            st.markdown('<div style="font-size: 0.8rem; color: #64b5f6; letter-spacing: 2px; margin-bottom: 8px;">VIEW PANEL</div>', unsafe_allow_html=True)
            
            panel_options = [
                "ℹ️ Pathogen Summary",
                "📊 6-Panel Dashboard",
                "🧮 Math & Data Ledger",
                "🕸️ Network Map",
                "💬 J.A.R.V.I.S. Chat",
                "🩺 Clinical Insight",
                "📄 Export PDF",
                "🌌 3D Landscape"
            ]
            
            selected_panel = st.radio("Select Panel to View:", panel_options, key="panel_radio")
            st.session_state.selected_panel = selected_panel
        
        st.markdown("---")
        st.markdown("""
        <div style="background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-radius: 8px; padding: 10px; text-align: center;">
            <div style="color: #10b981; font-size: 0.8rem; font-weight: bold;">✅ SYSTEM ONLINE</div>
        </div>
        """, unsafe_allow_html=True)

# ===== WELCOME PAGE =====
if page == "🏠 Welcome":
    render_welcome_page()

# ===== ANALYSIS PAGE =====
elif page == "🔬 Analysis":
    
    if not json_files:
        st.error("No JSON genome files found in the current directory. Please add CARD/RGI JSON files.")
        st.stop()
    
    if analysis_mode == "Select Known Bacteria":
        
        # Header
        st.markdown(f"""
        <div class="main-header">
            <h1 style='margin:0; font-size: 2rem; font-family: Orbitron, monospace; color: #00f5ff;'>🧬 AI-MRI ANALYSIS HUB</h1>
            <p style='font-size: 1rem; color: #64b5f6; margin-top: 8px; letter-spacing: 3px; text-transform: uppercase;'>Quantitative Bio-Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load data
        genes, drug, mech, mri, ari, records = extract_data(selected_file)
        u_drugs, u_mechs = len(set(drug)), len(set(mech))
        level, icon = get_level(mri)
        habitat = get_habitat(selected_file)
        bac_info = get_bacteria_info(selected_file)
        
        # Alert
        if mri > 0.6:
            st.markdown(f'<div class="alert-banner">⚠️ CRITICAL ALERT: {selected_file} identified as High-Priority Superbug (MRI: {round(mri, 3)})</div>', unsafe_allow_html=True)
        
        # AI Model
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
        
        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value">{level} {icon}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">MRI Score</div><div class="metric-value">{round(mri, 3)}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Genes</div><div class="metric-value">{genes}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Habitat</div><div class="metric-value">{habitat}</div></div>', unsafe_allow_html=True)
        
        st.write("")
        
        # Panel indicator
        st.markdown(f"""
        <div class="bacteria-panel">
            <div class="panel-title">CURRENT VIEW: {st.session_state.selected_panel}</div>
            <div style="font-family: 'Rajdhani', sans-serif; color: #8eb4d4; font-size: 0.9rem;">
                Genome: <span style="color: #00f5ff;">{selected_file}</span> &nbsp;|&nbsp; 
                Risk: <span style="color: {'#ef4444' if level == 'HIGH' else '#f59e0b' if level == 'MODERATE' else '#10b981'};">{level}</span> &nbsp;|&nbsp;
                AI Verdict: <span style="color: #a78bfa;">{ai_pred_text}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ===== PANEL RENDERING =====
        panel = st.session_state.selected_panel
        
        # ------ PATHOGEN SUMMARY ------
        if panel == "ℹ️ Pathogen Summary":
            st.markdown("### 🦠 Detailed Pathogen Profile")
            st.markdown(f"""
<div class="report-card">
<div class="report-header">Bacterial Identification Ledger</div>
<div class="report-row"><span class="report-label">Target Genome</span><span class="report-value" style="color:#00f5ff;">{selected_file}</span></div>
<div class="report-row"><span class="report-label">Gram Classification</span><span class="report-value">{bac_info['gram']}</span></div>
<div class="report-row"><span class="report-label">Associated Pathology</span><span class="report-value">{bac_info['disease']}</span></div>
<div class="report-row"><span class="report-label">Ecological Habitat</span><span class="report-value">{habitat}</span></div>
<div class="report-row"><span class="report-label">Genomic ARG Count</span><span class="report-value">{genes} Genes</span></div>
<div class="report-row"><span class="report-label">Resistance Breadth</span><span class="report-value">{u_drugs} Drug Classes</span></div>
<div class="report-row"><span class="report-label">Deployed Mechanisms</span><span class="report-value">{u_mechs} Strategies</span></div>
<div class="report-row"><span class="report-label">Calculated MRI / ARI</span><span class="report-value">{round(mri, 3)} ({level}) / {round(ari, 3)}</span></div>
<div style="margin-top:20px; padding-top:10px;">
<span class="report-label">AI Predictive Verdict:</span>
<span class="ai-badge">{ai_pred_text} Risk</span>
<br><br>
<small style="color:#64748b;">Confidence Matrix: {ai_conf_text}</small>
</div>
</div>
""", unsafe_allow_html=True)
            
            st.markdown("### 🎯 About the Metrics")
            st.info("""
**Pathogen Profile:** Provides biological and ecological context (Gram, Disease, Habitat) of the strain.

**Total Genes:** The absolute count of Antibiotic Resistance Genes (ARGs) identified.

**Resistance & Mechanisms:** The distinct drug classes evaded and biological strategies deployed.

**AI Prediction:** A machine learning probability assessment of the overall threat level.

**The Clinical Necessity of MRI and ARI:** Traditional analysis simply lists detected genes.
The **ARI** calculates the *density* and efficiency of the threat relative to the gene count.
The **MRI** mathematically consolidates the diversity of resisted drugs and mechanisms into a single standardized risk score.
""")
        
        # ------ 6-PANEL DASHBOARD ------
        elif panel == "📊 6-Panel Dashboard":
            st.markdown(f"### 📊 Systems Overview: `{selected_file}`")
            fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)
            st.pyplot(fig)
        
        # ------ MATH & DATA LEDGER ------
        elif panel == "🧮 Math & Data Ledger":
            st.markdown("### 🧮 Mathematical Validation")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown('<div class="math-card">', unsafe_allow_html=True)
                st.markdown('<div class="math-card-header">Multidimensional Resistance Index (MRI)</div>', unsafe_allow_html=True)
                st.latex(r"MRI = \frac{U_{drugs} + U_{mechs}}{T_{drugs} + T_{mechs} + 1}")
                st.markdown('<div class="annotation-box">'
                    '<div class="annotation-item"><span class="annotation-key">U_drugs</span>: Unique Drug Classes Resisted</div>'
                    '<div class="annotation-item"><span class="annotation-key">U_mechs</span>: Unique Mechanisms Deployed</div>'
                    '<div class="annotation-item"><span class="annotation-key">T_drugs</span>: Total Drug Records in sequence</div>'
                    '<div class="annotation-item"><span class="annotation-key">T_mechs</span>: Total Mechanism Records in sequence</div>'
                    '<div class="annotation-item"><span class="annotation-key">+ 1</span>: Laplace smoothing constant</div>'
                    '</div>', unsafe_allow_html=True)
                st.markdown(f"**Current Calculation:**")
                st.latex(rf"\frac{{{u_drugs} + {u_mechs}}}{{{len(drug)} + {len(mech)} + 1}} = {round(mri, 3)}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col_m2:
                st.markdown('<div class="math-card">', unsafe_allow_html=True)
                st.markdown('<div class="math-card-header">Antibiotic Resistance Index (ARI)</div>', unsafe_allow_html=True)
                st.latex(r"ARI = \frac{U_{mechs}}{G_{total} + 1}")
                st.markdown('<div class="annotation-box">'
                    '<div class="annotation-item"><span class="annotation-key">U_mechs</span>: Unique Mechanisms Deployed</div>'
                    '<div class="annotation-item"><span class="annotation-key">G_total</span>: Total Genomic Gene count</div>'
                    '<div class="annotation-item"><span class="annotation-key">+ 1</span>: Laplace smoothing constant</div>'
                    '</div>', unsafe_allow_html=True)
                st.markdown(f"**Current Calculation:**")
                st.latex(rf"\frac{{{u_mechs}}}{{{genes} + 1}} = {round(ari, 3)}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### 🎯 Risk Assessment Reasoning")
            st.markdown(f'<div class="reasoning-box">{get_risk_reason(level, u_drugs, u_mechs)}</div>', unsafe_allow_html=True)
            st.write("")
            st.markdown("### 📜 Comprehensive Gene Ledger")
            df = pd.DataFrame(records, columns=["Gene Name", "Drug Class", "Mechanism", "Habitat"])
            st.dataframe(df, use_container_width=True)
        
        # ------ NETWORK MAP ------
        elif panel == "🕸️ Network Map":
            st.markdown("### 🕸️ Interactive Mechanism Network")
            st.write("Use the filter menu within the map to isolate specific nodes.")
            html_path = generate_network_html(records, selected_file, "red" if level=="HIGH" else "orange" if level=="MODERATE" else "green")
            with open(html_path, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=650)
        
        # ------ JARVIS CHATBOT ------
        elif panel == "💬 J.A.R.V.I.S. Chat":
            st.markdown("""
            <div class="chat-container">
                <div class="chat-header">J.A.R.V.I.S. — BIOINFORMATICS INTELLIGENCE CORE</div>
                <div class="chat-status"><div class="status-dot"></div> ONLINE — Ready to answer any bioinformatics question</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Per-bacteria chat init
            if selected_file not in st.session_state.bacteria_chat:
                st.session_state.bacteria_chat[selected_file] = {"Chat 1": []}
                st.session_state.bacteria_chat[f"{selected_file}_current"] = "Chat 1"
                st.session_state.bacteria_chat[f"{selected_file}_counter"] = 1
            
            bac_sessions = st.session_state.bacteria_chat[selected_file]
            current_key = f"{selected_file}_current"
            counter_key = f"{selected_file}_counter"
            
            colA, colB, colC = st.columns([0.6, 0.2, 0.2])
            with colA:
                session_name = st.selectbox("Session:", list(bac_sessions.keys()), key=f"sess_{selected_file}")
                st.session_state.bacteria_chat[current_key] = session_name
            with colB:
                st.write("")
                if st.button("➕ New Chat", use_container_width=True, key=f"new_{selected_file}"):
                    st.session_state.bacteria_chat[counter_key] += 1
                    new_name = f"Chat {st.session_state.bacteria_chat[counter_key]}"
                    bac_sessions[new_name] = []
                    st.session_state.bacteria_chat[current_key] = new_name
                    st.rerun()
            with colC:
                st.write("")
                if st.button("🗑️ Clear", use_container_width=True, key=f"clear_{selected_file}"):
                    bac_sessions[st.session_state.bacteria_chat[current_key]] = []
                    st.rerun()
            
            active_session = st.session_state.bacteria_chat[current_key]
            
            for msg in bac_sessions[active_session]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            user_msg = st.chat_input(f"Ask me anything about {selected_file}, AMR, or bioinformatics...")
            if user_msg:
                st.chat_message("user").markdown(user_msg)
                bac_sessions[active_session].append({"role": "user", "content": user_msg})
                
                if not AI_AVAILABLE:
                    st.error("⚠️ AI Library missing. Check your terminal installation.")
                else:
                    try:
                        context = f"""
You are J.A.R.V.I.S., an expert Bioinformatics AI assistant embedded in the AI-MRI Hub platform.
You are currently analyzing the genome file: '{selected_file}'.

CURRENT GENOME DATA:
- Total Genes: {genes}
- Unique Drugs Resisted: {u_drugs} drug classes
- Unique Mechanisms: {u_mechs} strategies  
- MRI Score: {round(mri, 3)} ({level} Risk)
- ARI Score: {round(ari, 3)}
- Gram Classification: {bac_info['gram']}
- Associated Diseases: {bac_info['disease']}
- Ecological Habitat: {habitat}
- AI Prediction: {ai_pred_text} Risk | Confidence: {ai_conf_text}

Top Drug Classes: {', '.join(list(set(drug))[:8])}
Mechanisms: {', '.join(list(set(mech))[:6])}

INSTRUCTIONS:
- You are an expert in antimicrobial resistance (AMR), genomics, microbiology, and clinical medicine.
- Answer ANY question the user asks — whether it's about this specific bacteria, general AMR science, the formulas (MRI/ARI), clinical treatment options, genomics, or any other topic.
- For questions about this genome, use the data above to give specific, quantitative answers.
- For general questions not related to this genome, draw on your full knowledge of bioinformatics, microbiology, and medicine.
- Be precise, scientific, and helpful. Use concrete numbers when available.
- Format responses clearly with relevant sections when appropriate.

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
                                bac_sessions[active_session].append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg or "quota" in error_msg.lower():
                            st.error("⚠️ **Quota Exceeded (HTTP 429).** Please wait 60 seconds and retry.")
                        else:
                            st.error(f"AI Connection Error: {e}")
        
        # ------ CLINICAL INSIGHT ------
        elif panel == "🩺 Clinical Insight":
            st.markdown("### 🩺 Clinical Actionability (Susceptibility Zone)")
            DRUG_UNIVERSE = ["Penicillin", "Cephalosporin", "Carbapenem", "Macrolide", "Aminoglycoside", 
                             "Fluoroquinolone", "Tetracycline", "Sulfonamide", "Glycopeptide"]
            resisted_norm = set([d.lower() for d in drug])
            safe_zones = [d for d in DRUG_UNIVERSE if d.lower() not in resisted_norm]
            st.write("Based on genomic exclusion, the following drug classes show **Zero Resistance Markers** in this sample:")
            st.markdown(f'<div class="susceptibility-card">🛡️ Recommended Target Classes: {", ".join(safe_zones) if safe_zones else "All major classes show resistance markers"}</div>', unsafe_allow_html=True)
            st.write("---")
            st.markdown("### 📈 Population Benchmark")
            all_mris = []
            for f in json_files:
                try:
                    _, _, _, f_mri, _, _ = extract_data(f)
                    all_mris.append(f_mri)
                except: continue
            if all_mris:
                avg_mri = sum(all_mris) / len(all_mris)
                comparison_df = pd.DataFrame({"MRI Score": [mri, avg_mri]}, index=["Target Genome", "Global Average"])
                st.bar_chart(comparison_df)
        
        # ------ EXPORT PDF ------
        elif panel == "📄 Export PDF":
            st.markdown("### 📥 Generate Complete Master Report")
            st.write(f"Generating full PDF report for: **{selected_file}**")
            if st.button("Generate Master PDF", type="primary"):
                with st.spinner("Compiling graphs, explanations, and data into PDF..."):
                    fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)
                    pdf_path = create_advanced_pdf_report(selected_file, genes, drug, mech, mri, ari, level, icon, records, fig, bac_info, habitat, ai_pred_text, ai_conf_text)
                    with open(pdf_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Detailed PDF Report",
                            data=file,
                            file_name=pdf_path,
                            mime="application/pdf"
                        )
        
        # ------ 3D LANDSCAPE ------
        elif panel == "🌌 3D Landscape":
            st.markdown("### 🌌 Interactive Global Landscape Comparison (Plotly 3D)")
            st.write("Rotate, zoom, and download using the camera icon in the top-right corner of the plot.")
            plot_3d_pca_plotly(selected_file)
    
    # ===== AI PREDICT UNKNOWN =====
    elif analysis_mode == "AI Predict Unknown":
        st.markdown("""
        <div class="main-header">
            <h1 style='margin:0; font-size: 2rem; font-family: Orbitron, monospace; color: #00f5ff;'>🤖 ML RISK PREDICTOR</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Enter Genome Parameters")
        in_genes = st.number_input("Total Genes Found", min_value=1, value=15)
        in_drugs = st.number_input("Unique Drugs Resisted", min_value=1, value=5)
        in_mechs = st.number_input("Unique Mechanisms Found", min_value=1, value=2)
        
        model = train_rf_model()
        if model and st.button("🔮 Predict Risk Level", type="primary"):
            prediction = model.predict([[in_genes, in_drugs, in_mechs]])[0]
            probs = model.predict_proba([[in_genes, in_drugs, in_mechs]])[0]
            classes = model.classes_
            prob_str = " | ".join([f"{c}: {p:.2f}" for c, p in zip(classes, probs)])
            color = "red" if prediction == "HIGH" else "orange" if prediction == "MODERATE" else "green"
            st.markdown(f"""
            <div style="background: rgba(0,15,45,0.8); border: 2px solid {color}; border-radius: 16px; padding: 30px; text-align: center; margin-top: 20px;">
                <div style="font-family: Orbitron, monospace; font-size: 0.8rem; color: #64b5f6; letter-spacing: 3px; margin-bottom: 10px;">AI PREDICTION</div>
                <div style="font-family: Orbitron, monospace; font-size: 2.5rem; color: {color}; font-weight: 900;">{prediction}</div>
                <div style="font-family: Rajdhani, sans-serif; color: #8eb4d4; margin-top: 10px;">Confidence: {prob_str}</div>
            </div>
            """, unsafe_allow_html=True)
