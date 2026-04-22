import streamlit as st
import json
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ─── Base ─── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 15px;
    color: #e1e4e8;
    background: #161b22;
}
.main { background: #161b22; }
.block-container { padding: 2rem 2.8rem !important; max-width: 1400px; }

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c9d1d9 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stSidebar"] label {
    color: #8b949e !important;
    font-size: 0.82rem;
    font-family: 'Inter', sans-serif !important;
}

/* ─── Headings ─── */
h1 {
    font-family: 'Inter', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #f0f6fc !important;
    letter-spacing: -0.03em;
}
h2 {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #c9d1d9 !important;
}
h3 {
    font-family: 'Inter', sans-serif !important;
    color: #8b949e !important;
    font-size: 0.78rem !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

/* ─── Metric Cards ─── */
[data-testid="metric-container"] {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
}
[data-testid="metric-container"] > div > div:first-child {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500;
    color: #8b949e !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #f0f6fc !important;
}

/* ─── Tabs ─── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem;
    font-weight: 500;
    color: #8b949e;
    padding: 0.6rem 1.2rem;
    border-bottom: 2px solid transparent;
    letter-spacing: 0;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff;
    background: transparent;
    font-weight: 600;
}

/* ─── Buttons ─── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    background: #21262d;
    border: 1px solid #30363d;
    color: #c9d1d9;
    font-size: 0.9rem;
    font-weight: 500;
    border-radius: 6px;
    padding: 0.45rem 1.1rem;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #30363d;
    border-color: #58a6ff;
    color: #f0f6fc;
}
.stButton > button[kind="primary"] {
    background: #1f6feb;
    border: 1px solid #1f6feb;
    color: #ffffff;
    font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
    background: #388bfd;
    border-color: #388bfd;
}

/* ─── Code / mono ─── */
.stCodeBlock { background: #0d1117 !important; border: 1px solid #21262d; border-radius: 8px; }
code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem;
    color: #79c0ff !important;
    background: #161b22;
    padding: 1px 5px;
    border-radius: 4px;
}

/* ─── Selectbox / Radio ─── */
[data-testid="stSelectbox"] > div > div,
.stRadio label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    color: #c9d1d9 !important;
}

/* ─── Info / Success ─── */
.stInfo {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    color: #c9d1d9;
    font-size: 0.93rem;
    border-radius: 6px;
    font-family: 'Inter', sans-serif;
}
.stSuccess {
    background: #0f2b1d;
    border: 1px solid #1a5c34;
    border-left: 3px solid #3fb950;
    color: #aff5b4;
    font-size: 0.93rem;
    border-radius: 6px;
}

/* ─── DataFrames ─── */
[data-testid="stDataFrame"] { border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }

/* ─── Divider ─── */
hr { border-color: #21262d; margin: 1.5rem 0; }

/* ─── Chat ─── */
[data-testid="stChatMessage"] {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 8px;
    margin-bottom: 0.6rem;
    font-family: 'Inter', sans-serif;
}

/* ─── Title strip ─── */
.title-strip {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.8rem;
}
.title-strip h1 { margin: 0 0 0.3rem 0 !important; }
.title-strip .subtitle {
    color: #58a6ff;
    font-size: 0.88rem;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    letter-spacing: 0.03em;
}
.title-strip .authors {
    color: #484f58;
    font-size: 0.84rem;
    margin-top: 0.6rem;
    font-family: 'Inter', sans-serif;
}

/* ─── Risk badges ─── */
.badge {
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    letter-spacing: 0.04em;
}
.badge-high     { background: #3d1f1f; color: #f78166; border: 1px solid #6e2020; }
.badge-moderate { background: #2d2008; color: #e3b341; border: 1px solid #5a3e10; }
.badge-low      { background: #0f2b1d; color: #3fb950; border: 1px solid #1a5c34; }

/* ─── Section label ─── */
.section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 600;
    color: #484f58;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

/* ─── Inputs ─── */
[data-testid="stNumberInput"] input {
    font-family: 'Inter', sans-serif !important;
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #e1e4e8 !important;
    font-size: 0.95rem !important;
    border-radius: 6px;
}
[data-testid="stSpinner"] { color: #58a6ff; }

/* ─── Paragraph / body ─── */
p, li, .stMarkdown p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.96rem;
    line-height: 1.75;
    color: #c9d1d9;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE BACKEND FUNCTIONS (unchanged logic)
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
    if any(k in name for k in ["ecoli","escherichia","staphylococcus","salmonella","klebsiella","streptococcus","enterococcus"]):
        return "Clinical"
    elif any(k in name for k in ["pseudomonas","acinetobacter"]):
        return "Environmental"
    elif any(k in name for k in ["bacillus","clostridium","mycobacterium"]):
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
        return f"Because its MRI is high, this strain utilizes multiple redundant strategies ({u_mechs} mechanisms) to block diverse threats ({u_drugs} drugs). If one antibiotic pathway is bypassed, the bacteria actively pivots to another, making standard frontline clinical treatments highly ineffective."
    elif "MODERATE" in level:
        return f"With a moderate MRI, this strain shows significant adaptation. It has built defenses against standard antibiotics ({u_drugs} drugs), forcing clinicians to rely on secondary treatments."
    else:
        return f"This strain has a low MRI, indicating a narrow resistance profile. It likely specializes against specific antibiotics found in its direct natural habitat rather than amassing a large clinical arsenal."

@st.cache_resource
def train_rf_model():
    X, y = [], []
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                genes, drug, mech, mri, _, _ = extract_data(f)
                features = [genes, len(set(drug)), len(set(mech))]
                label = "LOW" if mri < 0.15 else "MODERATE" if mri < 0.35 else "HIGH"
                X.append(features); y.append(label)
            except: continue
    if len(X) < 2: return None
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# ==========================================
# 3. VISUALIZATION FUNCTIONS — REDESIGNED
# ==========================================
DARK_BG   = "#161b22"
PANEL_BG  = "#21262d"
BORDER    = "#30363d"
ACCENT    = "#58a6ff"
TEXT_PRI  = "#f0f6fc"
TEXT_SEC  = "#8b949e"
BLUE      = "#79c0ff"
CYAN      = "#39d0d8"
GREEN     = "#3fb950"
AMBER     = "#e3b341"
RED       = "#f78166"
PURPLE    = "#d2a8ff"


def risk_color(level):
    return {"HIGH": RED, "MODERATE": AMBER, "LOW": GREEN}.get(level, ACCENT)


def plot_full_dashboard(drug, mech, mri, genes, records, name):
    level, icon = get_level(mri)
    rc = risk_color(level)

    fig = plt.figure(figsize=(18, 11), facecolor=DARK_BG)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35,
                          left=0.06, right=0.97, top=0.91, bottom=0.08)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER)
        ax.tick_params(colors=TEXT_SEC, labelsize=8)
        ax.set_title(title, color=TEXT_SEC, fontsize=8.5,
                     fontfamily="monospace", loc="left", pad=8,
                     fontweight="normal")

    # ── Drug Classes Pie ──
    ax0 = fig.add_subplot(gs[0, 0])
    style_ax(ax0, "DRUG CLASSES RESISTED")
    drug_c = Counter(drug).most_common(8)
    if drug_c:
        palette = [BLUE, CYAN, PURPLE, ACCENT, "#818cf8", "#67e8f9", "#a5b4fc", "#7dd3fc"]
        wedges, texts, autotexts = ax0.pie(
            [v for _, v in drug_c],
            labels=[k[:18] for k, _ in drug_c],
            autopct='%1.0f%%',
            colors=palette[:len(drug_c)],
            startangle=90,
            wedgeprops=dict(linewidth=0.8, edgecolor=DARK_BG),
            textprops={'color': TEXT_SEC, 'fontsize': 7}
        )
        for at in autotexts: at.set_color(TEXT_PRI); at.set_fontsize(7)

    # ── Mechanism Bar ──
    ax1 = fig.add_subplot(gs[0, 1])
    style_ax(ax1, "RESISTANCE MECHANISMS")
    mech_c = Counter(mech)
    if mech_c:
        keys = [k[:20] for k in mech_c.keys()]
        vals = list(mech_c.values())
        bars = ax1.barh(keys, vals, color=PANEL_BG, edgecolor=rc, linewidth=1.2, height=0.55)
        for bar, val in zip(bars, vals):
            bar.set_facecolor(rc + "22")
        ax1.set_xlabel("count", color=TEXT_SEC, fontsize=7)
        ax1.tick_params(axis='y', labelsize=7.5, labelcolor=TEXT_SEC)
        ax1.invert_yaxis()
        ax1.xaxis.set_tick_params(labelsize=7)

    # ── MRI Gauge (arc) ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(PANEL_BG)
    for sp in ax2.spines.values(): sp.set_visible(False)
    ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title("MRI GAUGE", color=TEXT_SEC, fontsize=8.5,
                  fontfamily="monospace", loc="left", pad=8, fontweight="normal")
    # track
    theta = np.linspace(np.pi, 0, 200)
    ax2.plot(np.cos(theta), np.sin(theta), color=BORDER, linewidth=8, solid_capstyle='round')
    # fill
    angle = np.pi - mri * np.pi
    theta_fill = np.linspace(np.pi, angle, 200)
    ax2.plot(np.cos(theta_fill), np.sin(theta_fill), color=rc, linewidth=8, solid_capstyle='round')
    # needle
    ax2.plot([0, 0.72 * np.cos(angle)], [0, 0.72 * np.sin(angle)],
             color=TEXT_PRI, linewidth=2, solid_capstyle='round')
    ax2.add_patch(plt.Circle((0, 0), 0.06, color=ACCENT, zorder=5))
    ax2.text(0, -0.28, f"{mri:.3f}", ha='center', va='center',
             fontsize=18, color=rc, fontfamily="monospace", fontweight="bold")
    ax2.text(0, -0.48, level, ha='center', va='center',
             fontsize=9, color=rc, fontfamily="monospace")
    ax2.set_xlim(-1.25, 1.25); ax2.set_ylim(-0.65, 1.15)
    # zone labels
    for ang, lbl, col in [(np.pi * 0.9, "LOW", GREEN),
                          (np.pi * 0.6, "MOD", AMBER),
                          (np.pi * 0.15, "HIGH", RED)]:
        ax2.text(1.18 * np.cos(ang), 1.18 * np.sin(ang), lbl,
                 ha='center', va='center', fontsize=6.5, color=col, fontfamily="monospace")

    # ── Top Genes Bar ──
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "TOP GENE FREQUENCY")
    gene_c = Counter([r[0] for r in records]).most_common(6)
    if gene_c:
        gkeys = [k[:22] for k, _ in gene_c]
        gvals = [v for _, v in gene_c]
        bars = ax3.bar(range(len(gkeys)), gvals,
                       color=[CYAN + "33"] * len(gkeys), edgecolor=CYAN, linewidth=1, width=0.55)
        ax3.set_xticks(range(len(gkeys)))
        ax3.set_xticklabels(gkeys, rotation=35, ha='right', fontsize=7, color=TEXT_SEC)
        ax3.set_ylabel("freq", color=TEXT_SEC, fontsize=7)
        for bar, val in zip(bars, gvals):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     str(val), ha='center', fontsize=7, color=CYAN)

    # ── Gene / Score Summary ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL_BG)
    for sp in ax4.spines.values(): sp.set_edgecolor(BORDER)
    ax4.axis('off')
    ax4.set_title("GENOME SUMMARY", color=TEXT_SEC, fontsize=8.5,
                  fontfamily="monospace", loc="left", pad=8)
    summary_rows = [
        ("Total Genes", str(genes), ACCENT),
        ("Drugs Resisted", str(len(set(drug))), BLUE),
        ("Mechanisms", str(len(set(mech))), PURPLE),
        ("MRI Score", f"{mri:.4f}", rc),
        ("ARI Score", f"{len(set(mech))/(genes+1):.4f}", CYAN),
    ]
    for i, (label, value, col) in enumerate(summary_rows):
        y_pos = 0.82 - i * 0.17
        ax4.text(0.05, y_pos, label, transform=ax4.transAxes,
                 color=TEXT_SEC, fontsize=8.5, va='center')
        ax4.add_patch(mpatches.FancyBboxPatch(
            (0.55, y_pos - 0.06), 0.38, 0.12,
            boxstyle="round,pad=0.02", linewidth=0.8,
            facecolor=col + "20", edgecolor=col + "60",
            transform=ax4.transAxes))
        ax4.text(0.74, y_pos, value, transform=ax4.transAxes,
                 color=col, fontsize=9, va='center', ha='center',
                 fontfamily='monospace', fontweight='bold')

    # ── Diversity Comparison Bars ──
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, "RESISTANCE DIVERSITY")
    categories = ["Drug Classes", "Mechanisms"]
    values = [len(set(drug)), len(set(mech))]
    clrs = [BLUE, PURPLE]
    bars = ax5.bar(categories, values, color=[c + "33" for c in clrs],
                   edgecolor=clrs, linewidth=1.2, width=0.45)
    ax5.set_ylabel("unique count", color=TEXT_SEC, fontsize=7)
    ax5.tick_params(labelcolor=TEXT_SEC, labelsize=8)
    for bar, val in zip(bars, values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 str(val), ha='center', fontsize=10, color=TEXT_PRI,
                 fontfamily="monospace", fontweight="bold")

    # ── Title ──
    clean = name.replace('.json', '')
    fig.text(0.5, 0.965, f"Resistance Profile — {clean}",
             ha='center', color=TEXT_PRI, fontsize=12,
             fontfamily="monospace", fontweight="bold")
    fig.text(0.5, 0.948, "Multidimensional Resistance Index Dashboard",
             ha='center', color=TEXT_SEC, fontsize=8.5)

    return fig


def generate_network_html(records, organism_name, color):
    net = Network(height='600px', width='100%', bgcolor='#060a12',
                  font_color='#94a3b8', cdn_resources="in_line",
                  select_menu=True, filter_menu=True)
    net.add_node("HUB", label=organism_name, color=color, size=30,
                 title=f"Organism: {organism_name}", font={"size": 14, "color": "#e2e8f0"})
    for g, d, m, h in records:
        net.add_node(g, label=g[:14], color="#0ea5e9", size=14,
                     title=f"Gene: {g}", font={"size": 10})
        net.add_edge("HUB", g, color="#1e3a5f", width=1)
        if m:
            for mech in set(m.split(", ")):
                if not mech: continue
                net.add_node(mech, label=mech[:14], color="#a78bfa", size=9,
                             shape="box", title=f"Mechanism: {mech}", font={"size": 9})
                net.add_edge(g, mech, color="#1e2d4a", width=0.8)
    net.barnes_hut(gravity=-5000)
    html_path = "temp_network.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(net.generate_html())
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
                risk_levels.append("TARGET" if f == current_file else lv)
            except: continue
    if len(X) < 3:
        st.warning("Not enough genome files for 3D PCA — requires at least 3.")
        return
    pca = PCA(n_components=3).fit_transform(X)
    df_pca = pd.DataFrame(pca, columns=['PC1 · Overall Resistance', 'PC2 · Mechanism Diversity', 'PC3 · Genetic Density'])
    df_pca['Genome'] = files
    df_pca['Risk Category'] = risk_levels
    color_map = {"HIGH": RED, "MODERATE": AMBER, "LOW": GREEN, "TARGET": "#facc15"}
    fig = px.scatter_3d(df_pca,
        x='PC1 · Overall Resistance', y='PC2 · Mechanism Diversity', z='PC3 · Genetic Density',
        color='Risk Category', hover_name='Genome',
        color_discrete_map=color_map, opacity=0.85)
    fig.update_traces(marker=dict(size=5, line=dict(width=1.5, color='#0d1b2e')))
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor=DARK_BG,
        font=dict(family="IBM Plex Mono, monospace", color=TEXT_SEC, size=10),
        scene=dict(
            xaxis=dict(backgroundcolor=PANEL_BG, gridcolor=BORDER, title_font_color=TEXT_SEC, tickfont_color=TEXT_SEC, showbackground=True),
            yaxis=dict(backgroundcolor=PANEL_BG, gridcolor=BORDER, title_font_color=TEXT_SEC, tickfont_color=TEXT_SEC, showbackground=True),
            zaxis=dict(backgroundcolor=PANEL_BG, gridcolor=BORDER, title_font_color=TEXT_SEC, tickfont_color=TEXT_SEC, showbackground=True),
        ),
        legend=dict(bgcolor=PANEL_BG, bordercolor=BORDER, borderwidth=1, font_color=TEXT_SEC)
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. PDF GENERATOR (logic unchanged, minor style update)
# ==========================================
def create_advanced_pdf_report(bac_name, genes, drug, mech, mri, ari, level, icon, records, dashboard_fig, bac_info, habitat, ai_pred_text, ai_conf_text):
    pdf_file = f"{bac_name.replace('.json', '')}_Detailed_Report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, spaceAfter=15, textColor=colors.HexColor('#1E3A8A'))
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=14, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor('#2E86C1'))
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
    <b>ML AI Prediction:</b> {ai_pred_text}<br/>
    <b>Confidence Matrix:</b> {ai_conf_text}
    """
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("Metric Explanations & Significance", h2_style))
    elements.append(Paragraph("""
    <b>Pathogen Profile (Gram, Disease, Habitat):</b> Provides the essential biological and ecological context of the strain.<br/><br/>
    <b>Total Genes:</b> The absolute count of Antibiotic Resistance Genes (ARGs) identified in the sequence data.<br/><br/>
    <b>Resistance & Mechanisms:</b> Quantifies the distinct pharmaceutical drug classes the pathogen can evade and the specific biological strategies it deploys.<br/><br/>
    <b>AI Prediction & Confidence:</b> A Random Forest model's probabilistic assessment of the pathogen's overall threat level based on its learned resistance profile.
    """, normal_style))
    elements.append(Paragraph("The Clinical Necessity of MRI and ARI", h2_style))
    elements.append(Paragraph("""
    Traditional genomic analysis often simply lists detected genes, which fails to quantify the actual danger a pathogen poses.<br/><br/>
    <b>Why ARI is Required:</b> The Antibiotic Resistance Index calculates the <i>density</i> of the threat by normalizing unique mechanisms against total gene count.<br/><br/>
    <b>Why MRI Helps:</b> The Multidimensional Resistance Index mathematically consolidates the diversity of resisted drugs and deployed mechanisms into a single, standardized risk score, allowing clinicians to instantly gauge severity and prioritize high-risk pathogens for immediate intervention.
    """, normal_style))
    elements.append(PageBreak())
    elements.append(Paragraph("Exact Mathematical Calculations", h2_style))
    elements.append(Paragraph(f"""
    <b>MRI Calculation:</b> ({u_drugs} + {u_mechs}) / ({t_drugs} + {t_mechs} + 1) = <b>{round(mri, 3)}</b><br/>
    <b>ARI Calculation:</b> {u_mechs} / ({genes} + 1) = <b>{round(ari, 3)}</b>
    """, normal_style))
    elements.append(Paragraph("Risk Assessment Reasoning", h2_style))
    elements.append(Paragraph(get_risk_reason(level, u_drugs, u_mechs), normal_style))
    elements.append(Paragraph("Graphical Systems Dashboard", h2_style))
    buf = io.BytesIO()
    dashboard_fig.patch.set_facecolor('white')
    for ax in dashboard_fig.axes:
        ax.set_facecolor('white')
        ax.tick_params(colors='black')
        ax.title.set_color('black')
        for text in ax.texts: text.set_color('black')
    dashboard_fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    elements.append(RLImage(buf, width=7.5*inch, height=5*inch))
    elements.append(PageBreak())
    elements.append(Paragraph("Complete Gene Ledger", h2_style))
    table_data = [["Gene Name", "Drugs Resisted", "Mechanisms Used", "Habitat"]]
    for g, d, m, h in records:
        table_data.append([Paragraph(g, normal_style), Paragraph(d, normal_style), Paragraph(m, normal_style), Paragraph(h, normal_style)])
    t = Table(table_data, colWidths=[1.2*inch, 2.2*inch, 2.2*inch, 0.9*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0C4A6E')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8F9FA')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    elements.append(t)
    doc.build(elements)
    return pdf_file

# ==========================================
# 5. FRONTEND LAYOUT
# ==========================================

# ── Page Header ──
st.markdown("""
<div class="title-strip">
    <h1>AI-MRI Hub</h1>
    <div class="subtitle">ANTIBIOTIC RESISTANCE GENOMIC ANALYSIS PLATFORM</div>
    <div class="authors">Hardik Agrawal · Poorva Dongarkar · Yashraj Patil · Avani Laswante · Zeel Bhanushali · Aayushi Wasnik · Indranil Patil</div>
</div>
""", unsafe_allow_html=True)

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {"Session 1": []}
    st.session_state.current_session = "Session 1"
    st.session_state.chat_counter = 1

# ── Sidebar ──
with st.sidebar:
    st.markdown("### DATABASE")
    st.markdown('<div class="section-label">Genome Files</div>', unsafe_allow_html=True)
    if st.button("⟳  Refresh", use_container_width=True):
        st.rerun()

    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    st.markdown(f"<span style='font-family:Inter,sans-serif;font-size:0.84rem;color:#484f58;'>{len(json_files)} genome(s) indexed</span>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### ANALYSIS MODE")
    analysis_mode = st.radio("", ["Known Genome", "AI Predict Unknown"], label_visibility="collapsed")
    if analysis_mode == "Known Genome":
        selected_file = st.selectbox("Select genome file:", json_files)
    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Inter',sans-serif;font-size:0.86rem;font-weight:500;color:#3fb950;padding:0.6rem 0.9rem;background:#0f2b1d;border:1px solid #1a5c34;border-radius:6px;">
    ● &nbsp;AI Engine Online
    </div>""", unsafe_allow_html=True)


# ==========================================
# ── KNOWN GENOME ANALYSIS ──
# ==========================================
if analysis_mode == "Known Genome" and json_files:
    genes, drug, mech, mri, ari, records = extract_data(selected_file)
    u_drugs, u_mechs = len(set(drug)), len(set(mech))
    level, icon = get_level(mri)
    habitat = get_habitat(selected_file)
    bac_info = get_bacteria_info(selected_file)
    rc = risk_color(level)

    model = train_rf_model()
    ai_pred_text, ai_conf_text = "N/A", "N/A"
    if model:
        pred = model.predict([[genes, u_drugs, u_mechs]])[0]
        probs = model.predict_proba([[genes, u_drugs, u_mechs]])[0]
        classes = model.classes_
        conf_dict = {str(c): round(float(p), 3) for c, p in zip(classes, probs)}
        ai_pred_text = str(pred)
        ai_conf_text = str(conf_dict).replace("'", "")

    badge_cls = {"HIGH": "badge-high", "MODERATE": "badge-moderate", "LOW": "badge-low"}.get(level, "badge-low")

    # ── KPI Row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Risk Level", f"{level} {icon}")
    c2.metric("MRI Score", f"{mri:.4f}")
    c3.metric("ARI Score", f"{ari:.4f}")
    c4.metric("Total Genes", genes)
    c5.metric("Habitat", habitat)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "SUMMARY", "DASHBOARD", "MATH & LEDGER", "NETWORK", "BIO-AI", "EXPORT PDF", "3D LANDSCAPE"
    ])

    # ── TAB 1: SUMMARY ──
    with tab1:
        st.markdown('<div class="section-label">Pathogen Executive Profile</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1.6])

        with col_a:
            st.markdown(f"""
            <div style="background:{PANEL_BG};border:1px solid {BORDER};border-radius:10px;padding:1.4rem 1.6rem;font-family:'Inter',sans-serif;font-size:0.95rem;line-height:1;">
                <div style="color:{TEXT_SEC};margin-bottom:1rem;font-size:0.72rem;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;">Organism Profile</div>

                <div style="margin-bottom:0.7rem;">
                  <div style="color:{TEXT_SEC};font-size:0.76rem;font-weight:500;margin-bottom:2px;">File</div>
                  <div style="color:{TEXT_PRI};font-size:0.88rem;word-break:break-all;">{selected_file}</div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;margin-bottom:0.7rem;">
                  <div><div style="color:{TEXT_SEC};font-size:0.76rem;font-weight:500;margin-bottom:2px;">Gram Stain</div><div style="color:{ACCENT};font-weight:600;">{bac_info['gram']}</div></div>
                  <div><div style="color:{TEXT_SEC};font-size:0.76rem;font-weight:500;margin-bottom:2px;">Habitat</div><div style="color:{CYAN};font-weight:600;">{habitat}</div></div>
                </div>
                <div style="margin-bottom:1rem;">
                  <div style="color:{TEXT_SEC};font-size:0.76rem;font-weight:500;margin-bottom:2px;">Disease</div>
                  <div style="color:{TEXT_PRI};font-size:0.9rem;">{bac_info['disease']}</div>
                </div>

                <div style="border-top:1px solid {BORDER};padding-top:1rem;margin-bottom:1rem;">
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.6rem;">
                    <div style="background:#0d1117;border:1px solid {BORDER};border-radius:8px;padding:0.6rem;text-align:center;">
                      <div style="color:{TEXT_SEC};font-size:0.7rem;font-weight:500;margin-bottom:4px;">Genes</div>
                      <div style="color:{ACCENT};font-size:1.3rem;font-weight:700;">{genes}</div>
                    </div>
                    <div style="background:#0d1117;border:1px solid {BORDER};border-radius:8px;padding:0.6rem;text-align:center;">
                      <div style="color:{TEXT_SEC};font-size:0.7rem;font-weight:500;margin-bottom:4px;">Drug Classes</div>
                      <div style="color:{BLUE};font-size:1.3rem;font-weight:700;">{u_drugs}</div>
                    </div>
                    <div style="background:#0d1117;border:1px solid {BORDER};border-radius:8px;padding:0.6rem;text-align:center;">
                      <div style="color:{TEXT_SEC};font-size:0.7rem;font-weight:500;margin-bottom:4px;">Mechanisms</div>
                      <div style="color:{PURPLE};font-size:1.3rem;font-weight:700;">{u_mechs}</div>
                    </div>
                  </div>
                </div>

                <div style="border-top:1px solid {BORDER};padding-top:1rem;">
                  <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-bottom:0.6rem;">
                    <div style="background:#0d1117;border:1px solid {BORDER};border-radius:8px;padding:0.6rem;">
                      <div style="color:{TEXT_SEC};font-size:0.72rem;font-weight:500;margin-bottom:2px;">MRI Score</div>
                      <div style="color:{rc};font-size:1.1rem;font-weight:700;">{mri:.4f}</div>
                    </div>
                    <div style="background:#0d1117;border:1px solid {BORDER};border-radius:8px;padding:0.6rem;">
                      <div style="color:{TEXT_SEC};font-size:0.72rem;font-weight:500;margin-bottom:2px;">ARI Score</div>
                      <div style="color:{CYAN};font-size:1.1rem;font-weight:700;">{ari:.4f}</div>
                    </div>
                  </div>
                  <div style="background:#0d1117;border:1px solid {BORDER};border-radius:8px;padding:0.7rem;">
                    <div style="color:{TEXT_SEC};font-size:0.72rem;font-weight:500;margin-bottom:4px;">ML Prediction</div>
                    <div style="color:{AMBER};font-size:1rem;font-weight:600;">{ai_pred_text}</div>
                    <div style="color:{TEXT_SEC};font-size:0.78rem;margin-top:4px;">{ai_conf_text}</div>
                  </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div style="background:{PANEL_BG};border:1px solid {BORDER};border-radius:10px;padding:1.4rem 1.6rem;">
                <div style="color:{TEXT_SEC};font-family:'Inter',sans-serif;font-size:0.72rem;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;margin-bottom:1rem;">Why These Metrics Matter</div>
                <div style="font-size:0.95rem;line-height:1.8;color:#c9d1d9;">
                <p><strong style="color:{ACCENT};">Total Genes</strong> — The absolute count of Antibiotic Resistance Genes (ARGs) identified in the sequence. A higher count does not automatically mean higher danger.</p>
                <p><strong style="color:{BLUE};">ARI (Antibiotic Resistance Index)</strong> — Calculates the <em>density</em> of the threat by normalizing unique mechanisms against total gene count. Reveals how efficiently the bacteria uses its genomic payload.</p>
                <p><strong style="color:{rc};">MRI (Multidimensional Resistance Index)</strong> — Consolidates drug-class diversity and mechanism diversity into a single standardized risk score. Enables rapid, objective triage without deciphering raw gene ledgers.</p>
                <p><strong style="color:{AMBER};">ML Prediction</strong> — A Random Forest classifier trained on all indexed genomes produces a probabilistic risk tier assessment, cross-validating the calculated MRI.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2: DASHBOARD ──
    with tab2:
        st.markdown(f'<div class="section-label">Six-Panel Systems Dashboard — {selected_file.replace(".json","")}</div>', unsafe_allow_html=True)
        fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)
        st.pyplot(fig, use_container_width=True)

    # ── TAB 3: MATH & LEDGER ──
    with tab3:
        st.markdown('<div class="section-label">Calculations</div>', unsafe_allow_html=True)

        col_m, col_r = st.columns(2)
        with col_m:
            st.markdown(f"""
            <div style="background:{PANEL_BG};border:1px solid {BORDER};border-radius:10px;padding:1.3rem 1.5rem;font-family:'Inter',sans-serif;">
                <div style="color:{TEXT_SEC};font-size:0.72rem;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;margin-bottom:1rem;">MRI Formula</div>
                <div style="background:#0d1117;border:1px solid {BORDER};border-radius:8px;padding:1rem;margin-bottom:0.8rem;font-family:'JetBrains Mono',monospace;font-size:0.9rem;line-height:1.9;color:{TEXT_SEC};">
                  <div>(Unique Drugs + Unique Mechs)</div>
                  <div style="border-top:1px solid {BORDER};border-bottom:1px solid {BORDER};padding:2px 0;margin:4px 0;">──────────────────────────</div>
                  <div>(Total Drugs + Total Mechs + 1)</div>
                  <div style="margin-top:0.5rem;color:{TEXT_PRI};font-size:0.88rem;">= ({u_drugs} + {u_mechs}) / ({len(drug)} + {len(mech)} + 1)</div>
                  <div style="color:{rc};font-size:1.1rem;font-weight:700;margin-top:0.3rem;">= {mri:.4f} &nbsp;<span style="font-size:0.8rem;background:{rc}22;color:{rc};padding:2px 8px;border-radius:4px;border:1px solid {rc}44;">{level}</span></div>
                </div>
                <div style="color:{TEXT_SEC};font-size:0.72rem;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;margin-bottom:0.6rem;margin-top:0.8rem;">ARI Formula</div>
                <div style="background:#0d1117;border:1px solid {BORDER};border-radius:8px;padding:1rem;font-family:'JetBrains Mono',monospace;font-size:0.9rem;line-height:1.9;color:{TEXT_SEC};">
                  <div>Unique Mechs / (Total Genes + 1)</div>
                  <div style="color:{TEXT_PRI};margin-top:0.3rem;font-size:0.88rem;">= {u_mechs} / ({genes} + 1)</div>
                  <div style="color:{CYAN};font-size:1.1rem;font-weight:700;margin-top:0.3rem;">= {ari:.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_r:
            st.markdown(f"""
            <div style="background:{PANEL_BG};border:1px solid {BORDER};border-left:3px solid {rc};border-radius:10px;padding:1.3rem 1.5rem;font-family:'Inter',sans-serif;">
                <div style="color:{TEXT_SEC};font-size:0.72rem;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;margin-bottom:0.8rem;">Risk Assessment</div>
                <div style="font-size:0.96rem;line-height:1.85;color:#c9d1d9;">{get_risk_reason(level, u_drugs, u_mechs)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Complete Gene Ledger</div>', unsafe_allow_html=True)
        df = pd.DataFrame(records, columns=["Gene Name", "Drug Class", "Mechanism", "Habitat"])
        st.dataframe(df, use_container_width=True, height=420)

    # ── TAB 4: NETWORK ──
    with tab4:
        st.markdown('<div class="section-label">Gene-Mechanism Interaction Network</div>', unsafe_allow_html=True)
        st.markdown(f'<span style="font-family:Inter,sans-serif;font-size:0.92rem;color:{TEXT_SEC};">Use the filter panel to isolate nodes. Hub = organism, blue = genes, purple = mechanisms.</span>', unsafe_allow_html=True)
        html_path = generate_network_html(records, selected_file, rc)
        with open(html_path, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=650)

    # ── TAB 5: BIO-AI CHAT ──
    with tab5:
        st.markdown('<div class="section-label">Bio-AI Assistant</div>', unsafe_allow_html=True)
        colA, colB, colC = st.columns([0.6, 0.2, 0.2])
        with colA:
            st.session_state.current_session = st.selectbox(
                "Active session:", list(st.session_state.chat_sessions.keys()), label_visibility="collapsed")
        with colB:
            if st.button("+ New Session", use_container_width=True):
                st.session_state.chat_counter += 1
                new_name = f"Session {st.session_state.chat_counter}"
                st.session_state.chat_sessions[new_name] = []
                st.session_state.current_session = new_name
                st.rerun()
        with colC:
            if st.button("Clear", use_container_width=True):
                st.session_state.chat_sessions[st.session_state.current_session] = []
                st.rerun()

        for msg in st.session_state.chat_sessions[st.session_state.current_session]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_msg = st.chat_input(f"Ask about {selected_file} — mechanisms, risk, treatment options...")
        if user_msg:
            st.chat_message("user").markdown(user_msg)
            st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": user_msg})
            if not AI_AVAILABLE:
                st.error("AI library unavailable. Check your environment installation.")
            else:
                try:
                    context = f"""You are J.A.R.V.I.S., an expert Bioinformatics AI assistant.
The user is analyzing genome file: '{selected_file}'.
Profile:
- Total Genes: {genes} | Unique Drug Classes: {u_drugs} | Unique Mechanisms: {u_mechs}
- MRI Score: {round(mri, 4)} ({level} Risk) | ARI Score: {round(ari, 4)}
- Gram Stain: {bac_info['gram']} | Habitat: {habitat} | Disease: {bac_info['disease']}

Answer the user's question with precision and clarity based on this data.
User Question: {user_msg}"""
                    with st.spinner("Processing..."):
                        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        if not available_models:
                            st.error("API key has no access to text generation models.")
                        else:
                            target_model = next((m for m in available_models if 'flash' in m),
                                                next((m for m in available_models if 'pro' in m), available_models[0]))
                            model_ai = genai.GenerativeModel(target_model)
                            response = model_ai.generate_content(context)
                            st.chat_message("assistant").markdown(response.text)
                            st.session_state.chat_sessions[st.session_state.current_session].append(
                                {"role": "assistant", "content": response.text})
                except Exception as e:
                    err = str(e)
                    if "429" in err or "quota" in err.lower():
                        st.error("Rate limit reached (HTTP 429). Please wait ~60 seconds before retrying.")
                    else:
                        st.error(f"AI Error: {e}")

    # ── TAB 6: PDF EXPORT ──
    with tab6:
        st.markdown('<div class="section-label">Export Master Report</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{PANEL_BG};border:1px solid {BORDER};border-radius:10px;padding:1.3rem 1.6rem;margin-bottom:1.2rem;font-family:'Inter',sans-serif;">
            <div style="font-size:0.72rem;font-weight:600;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.8rem;">Report Will Include</div>
            <div style="font-size:0.95rem;color:#c9d1d9;line-height:2.1;">
            ✦ &nbsp;Executive pathogen profile and AI prediction<br>
            ✦ &nbsp;MRI &amp; ARI formula derivations with exact values<br>
            ✦ &nbsp;Clinical risk reasoning narrative<br>
            ✦ &nbsp;Six-panel graphical dashboard (high-res)<br>
            ✦ &nbsp;Complete gene ledger with mechanism table
            </div>
        </div>
        """, unsafe_allow_html=True)

        if 'fig' not in dir():
            fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)

        if st.button("Generate PDF Report", type="primary", use_container_width=False):
            with st.spinner("Compiling report..."):
                pdf_path = create_advanced_pdf_report(
                    selected_file, genes, drug, mech, mri, ari,
                    level, icon, records, fig, bac_info, habitat,
                    ai_pred_text, ai_conf_text)
                with open(pdf_path, "rb") as file:
                    st.download_button(
                        label="Download PDF",
                        data=file,
                        file_name=pdf_path,
                        mime="application/pdf"
                    )

    # ── TAB 7: 3D LANDSCAPE ──
    with tab7:
        st.markdown('<div class="section-label">3D PCA Resistance Landscape</div>', unsafe_allow_html=True)
        st.markdown(f'<span style="font-family:Inter,sans-serif;font-size:0.92rem;color:{TEXT_SEC};">Rotate · Zoom · Hover for genome details. Your target genome is highlighted in gold.</span>', unsafe_allow_html=True)
        plot_3d_pca_plotly(selected_file)


# ==========================================
# ── AI PREDICT UNKNOWN MODE ──
# ==========================================
elif analysis_mode == "AI Predict Unknown":
    st.markdown('<div class="section-label">Machine Learning — Unknown Genome Prediction</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{PANEL_BG};border:1px solid {BORDER};border-radius:10px;padding:1.3rem 1.6rem;margin-bottom:1.5rem;font-family:'Inter',sans-serif;font-size:0.95rem;color:#c9d1d9;line-height:1.8;">
    Enter the three key genomic features below. The Random Forest model will predict the risk tier and output a probability breakdown across all risk classes.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        in_genes = st.number_input("Total Genes Found", min_value=1, value=15)
    with col2:
        in_drugs = st.number_input("Unique Drug Classes Resisted", min_value=1, value=5)
    with col3:
        in_mechs = st.number_input("Unique Mechanisms Found", min_value=1, value=2)

    model = train_rf_model()
    if model and st.button("Run Prediction", type="primary"):
        prediction = model.predict([[in_genes, in_drugs, in_mechs]])[0]
        probs = model.predict_proba([[in_genes, in_drugs, in_mechs]])[0]
        classes = model.classes_
        rc = risk_color(prediction)
        badge_cls = {"HIGH": "badge-high", "MODERATE": "badge-moderate", "LOW": "badge-low"}.get(prediction, "badge-low")

        st.markdown(f"""
        <div style="background:{PANEL_BG};border:1px solid {rc}55;border-left:4px solid {rc};border-radius:10px;padding:1.4rem 1.6rem;margin-top:1rem;font-family:'Inter',sans-serif;">
            <div style="font-size:0.72rem;font-weight:600;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.6rem;">Prediction Result</div>
            <div style="font-size:2.2rem;font-weight:700;color:{rc};">{prediction}</div>
        </div>
        """, unsafe_allow_html=True)

        prob_str = " &nbsp;·&nbsp; ".join(
            [f'<span style="color:{risk_color(c)};font-family:Inter,sans-serif;font-size:0.95rem;font-weight:600;">{c}: {p:.3f}</span>'
             for c, p in zip(classes, probs)]
        )
        st.markdown(f"""
        <div style="background:{PANEL_BG};border:1px solid {BORDER};border-radius:10px;padding:1rem 1.4rem;margin-top:0.8rem;font-family:'Inter',sans-serif;">
            <span style="color:{TEXT_SEC};font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;">Class Probabilities &nbsp;→&nbsp; </span>
            {prob_str}
        </div>
        """, unsafe_allow_html=True)
