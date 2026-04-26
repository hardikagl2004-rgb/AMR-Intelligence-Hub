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

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

try:
    import google.generativeai as genai
    AI_AVAILABLE = True
    if "GENAI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GENAI_API_KEY"])
    else:
        AI_AVAILABLE = False
except ImportError:
    AI_AVAILABLE = False

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon="🧬")

# ─────────────────────────────────────────
# GLOBAL CSS  — forces dark bg everywhere
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset / global dark ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main, .block-container,
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
section.main > div { background-color: #050d1a !important; color: #e2e8f0 !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] > div:first-child { background: #080f1e !important; border-right: 1px solid #1e3a5f; }

/* Streamlit default widgets dark override */
.stButton > button {
  background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
  color: #fff !important; border: none !important;
  border-radius: 8px !important; font-weight: 600 !important;
  padding: 0.5rem 1.2rem !important; transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button[kind="secondary"] {
  background: #1e3a5f !important; color: #7dd3fc !important;
  border: 1px solid #2563eb !important;
}
.stSelectbox > div, .stRadio > div { color: #e2e8f0 !important; }
[data-baseweb="select"] > div { background: #0f1f38 !important; border-color: #1e40af !important; color: #e2e8f0 !important; }
.stTabs [data-baseweb="tab-list"] { background: #080f1e !important; border-bottom: 2px solid #1e3a5f; }
.stTabs [data-baseweb="tab"] { color: #94a3b8 !important; background: transparent !important; }
.stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }
[data-testid="stDataFrame"] { background: #0a1628 !important; border: 1px solid #1e3a5f; border-radius: 10px; }
.stAlert > div { background: #0f1f38 !important; border-color: #2563eb !important; color: #bfdbfe !important; }
[data-testid="stMarkdown"] p { color: #cbd5e1 !important; }

/* ── Shared tokens ── */
:root {
  --cyan: #22d3ee;
  --indigo: #818cf8;
  --emerald: #34d399;
  --amber: #fbbf24;
  --rose: #fb7185;
  --surface: #0d1b2e;
  --surface2: #0f2340;
  --border: rgba(34,211,238,0.18);
  --text: #e2e8f0;
  --muted: #64748b;
}

/* ── Splash – hero ── */
.sp-hero {
  background: linear-gradient(160deg, #020c1b 0%, #051a35 40%, #071e3d 100%);
  padding: 72px 48px 56px; text-align: center;
  border-bottom: 1px solid rgba(34,211,238,0.12);
}
.sp-dna { font-size: 5rem; display: inline-block; animation: dnaFloat 3.5s ease-in-out infinite;
  filter: drop-shadow(0 0 28px rgba(34,211,238,0.55)); margin-bottom: 18px; }
@keyframes dnaFloat { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-14px)} }

.sp-title {
  font-family: 'Orbitron', monospace; font-size: 3.8rem; font-weight: 900; letter-spacing: 3px;
  background: linear-gradient(90deg, #22d3ee 0%, #818cf8 40%, #34d399 75%, #22d3ee 100%);
  background-size: 300% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; animation: shine 6s linear infinite; margin: 0 0 10px 0;
}
@keyframes shine { 0%{background-position:0% center} 100%{background-position:300% center} }
.sp-sub { font-family:'Orbitron',monospace; font-size:0.78rem; letter-spacing:5px; color:var(--cyan); text-transform:uppercase; margin-bottom:18px; }
.sp-desc { color: #94a3b8; font-size: 1.05rem; max-width: 620px; margin: 0 auto 36px; line-height: 1.85; }

.sp-stats-row { display:flex; justify-content:center; gap:14px; flex-wrap:wrap; margin-bottom:6px; }
.sp-stat {
  background: rgba(34,211,238,0.07); border: 1px solid rgba(34,211,238,0.22);
  border-radius: 14px; padding: 16px 28px; text-align: center; min-width: 110px;
}
.sp-stat-num { font-family:'Orbitron',monospace; font-size:2rem; font-weight:900; color:var(--cyan); display:block; }
.sp-stat-lbl { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 2px; }

/* ── Section heading ── */
.sec-title {
  font-family:'Orbitron',monospace; font-size:1.5rem; font-weight:700; color:#f1f5f9;
  text-align:center; margin-bottom:6px;
}
.sec-bar { width:72px; height:3px; background:linear-gradient(90deg,var(--cyan),var(--indigo)); margin:0 auto 28px; border-radius:3px; }
.sec-wrap { background:#050d1a; padding:40px 0; }

/* ── Feature cards ── */
.feat-card {
  background: linear-gradient(145deg, #0d1b2e, #0f2340);
  border: 1px solid rgba(34,211,238,0.15);
  border-radius: 14px; padding: 24px 20px; height: 100%;
  transition: border-color 0.25s, box-shadow 0.25s;
}
.feat-card:hover { border-color: rgba(34,211,238,0.45); box-shadow: 0 8px 30px rgba(34,211,238,0.08); }
.feat-icon { font-size: 2rem; margin-bottom: 12px; display: block; }
.feat-title { color: var(--cyan) !important; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.feat-desc { color: #6b7f9e; font-size: 0.84rem; line-height: 1.65; }

/* ── Compare cards ── */
.comp-bad {
  background: linear-gradient(145deg, #1a0a0a, #200d0d);
  border: 1px solid rgba(251,113,133,0.3); border-radius: 14px; padding: 24px;
}
.comp-good {
  background: linear-gradient(145deg, #071a2e, #0c2240);
  border: 1px solid rgba(34,211,238,0.35); border-radius: 14px; padding: 24px;
}
.comp-title { font-size:1rem; font-weight:700; margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.07); }
.comp-bad .comp-title  { color: #fb7185; }
.comp-good .comp-title { color: var(--cyan); }
.comp-item { color: #8da0bc; font-size: 0.85rem; margin-bottom: 9px; line-height: 1.55; }

/* ── Team cards ── */
.tp-card {
  background: linear-gradient(145deg, #0d1b2e, #0f2340);
  border: 1px solid rgba(34,211,238,0.18);
  border-radius: 18px; padding: 26px 20px 22px; text-align: center;
  position: relative; overflow: hidden;
}
.tp-card::before {
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background: linear-gradient(90deg, var(--cyan), var(--indigo), var(--emerald));
  background-size: 200% auto; animation: shine 4s linear infinite;
}
.tp-avatar {
  width:78px; height:78px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; font-size:2.1rem;
  margin: 0 auto 14px; border: 2px solid rgba(34,211,238,0.3);
}
.tp-name { color: #f1f5f9 !important; font-size: 1.05rem; font-weight: 800; margin-bottom: 3px; }
.tp-role1 { color: var(--cyan) !important; font-size: 0.65rem; text-transform:uppercase; letter-spacing:2px; font-weight:700; margin-bottom:2px; }
.tp-role2 { color: var(--indigo) !important; font-size: 0.62rem; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:12px; }
.tp-desc { color: #6b7f9e; font-size: 0.8rem; line-height: 1.65; margin-bottom: 14px; }
.tp-tags { display:flex; flex-wrap:wrap; gap:5px; justify-content:center; }
.tp-tag { background:rgba(34,211,238,0.08); border:1px solid rgba(34,211,238,0.22); color:#67e8f9; padding:3px 10px; border-radius:20px; font-size:0.62rem; }

/* ── Mission cards ── */
.miss-card {
  background: linear-gradient(145deg, #0d1b2e, #0f2340);
  border: 1px solid rgba(34,211,238,0.14); border-radius: 14px; padding: 26px 20px; text-align:center;
}
.miss-title { color: var(--cyan) !important; font-size:0.78rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:10px; }
.miss-text { color: #6b7f9e; font-size: 0.83rem; line-height: 1.7; }

/* ── App page – metric cards ── */
.m-card {
  background: linear-gradient(145deg, #0d1b2e, #0f2340);
  border: 1px solid rgba(34,211,238,0.2); border-radius: 12px;
  padding: 20px 16px; text-align:center;
}
.m-label { color: var(--cyan); font-size: 0.7rem; font-weight:700; text-transform:uppercase; letter-spacing: 0.1em; }
.m-value { color: #f1f5f9; font-size: 1.7rem; font-weight: 700; margin-top: 6px; }

/* ── App page – report card ── */
.rep-card {
  background: linear-gradient(145deg, #0d1b2e, #0f2340);
  border-left: 4px solid var(--cyan); border-radius: 10px; padding: 20px 22px; margin-bottom: 20px;
}
.rep-hdr { color: var(--cyan); font-weight:700; text-transform:uppercase; font-size:0.82rem; margin-bottom:12px; padding-bottom:6px; border-bottom:1px solid #1e3a5f; }
.rep-row { display:flex; justify-content:space-between; padding:7px 0; border-bottom:1px solid #111e30; }
.rep-lbl { color: #64748b; font-weight:500; font-size:0.88rem; }
.rep-val { color: #e2e8f0; font-weight:600; font-size:0.88rem; }
.ai-badge { background:rgba(34,211,238,0.12); color:var(--cyan); padding:3px 12px; border-radius:20px; font-size:0.82rem; border:1px solid rgba(34,211,238,0.3); }

/* ── Math cards ── */
.math-card {
  background: linear-gradient(145deg, #0d1b2e, #0f2340);
  border: 1px solid rgba(34,211,238,0.15); border-radius: 12px; padding: 22px; margin-bottom: 18px;
}
.math-hdr { color: var(--cyan); font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:14px; }
.ann-box { background:rgba(5,13,26,0.8); border:1px dashed #1e3a5f; border-radius:8px; padding:12px; margin-top:10px; font-size:0.82rem; color:#64748b; }
.ann-item { margin-bottom:4px; }
.ann-key { color:#60a5fa; font-weight:bold; font-family:monospace; }

/* ── Reasoning box ── */
.reason-box {
  background: #071428; border-left:4px solid var(--cyan); border-radius:10px;
  padding:18px 22px; color:#cbd5e1; line-height:1.75; font-size:1rem;
}

/* ── Alert banner ── */
.alert-ban {
  background:rgba(251,113,133,0.15); border:1px solid #f43f5e;
  padding:14px; border-radius:8px; color:#fb7185; font-weight:700;
  text-align:center; margin-bottom:18px; animation:pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.65} }

/* ── App header banner ── */
.app-header {
  background: linear-gradient(120deg, #020c1b 0%, #0a1e3d 50%, #071a35 100%);
  border: 1px solid rgba(34,211,238,0.2); border-radius: 14px;
  padding: 2.2rem 2.5rem; text-align:center; margin-bottom:1.8rem;
  position:relative; overflow:hidden;
  box-shadow: 0 10px 40px rgba(34,211,238,0.07);
}
.app-header::before {
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background:linear-gradient(90deg,var(--cyan),var(--indigo),var(--emerald),var(--cyan));
  background-size:300% auto; animation:shine 4s linear infinite;
}
.app-header h1 { color:#f1f5f9 !important; font-size:2.4rem !important; margin:0 !important; }
.app-header p { color:#94a3b8 !important; margin:8px 0 0 !important; }

/* ── Susceptibility card ── */
.susc-card { background:rgba(52,211,153,0.09); border:1px solid #34d399; border-radius:10px; padding:14px 18px; color:#34d399; font-weight:600; }

/* ── Welcome hero (app tab 1) ── */
.w-hero { background:linear-gradient(135deg,rgba(14,50,100,0.8),rgba(7,20,40,0.95)); border:1px solid rgba(56,189,248,0.35); border-radius:14px; padding:40px; text-align:center; margin-bottom:24px; }
.w-hero h2 { color:#60a5fa !important; font-size:2.2rem !important; font-weight:800 !important; }
.w-hero p  { color:#cbd5e1 !important; font-size:1.1rem !important; }

/* Divider utility */
.divider-bar { width:72px; height:3px; background:linear-gradient(90deg,var(--cyan),var(--indigo)); margin:0 auto 26px; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# BACKEND FUNCTIONS
# ─────────────────────────────────────────
@st.cache_data
def get_bacteria_info(file_name):
    name = file_name.lower()
    info = {"gram": "Unknown", "disease": "Various opportunistic infections"}
    if any(k in name for k in ["ecoli","escherichia","shigella"]):
        info = {"gram":"Negative (-)","disease":"Gastroenteritis, UTI, Sepsis"}
    elif "salmonella" in name:
        info = {"gram":"Negative (-)","disease":"Salmonellosis, Typhoid Fever"}
    elif any(k in name for k in ["klebsiella","enterobacter","citrobacter","serratia"]):
        info = {"gram":"Negative (-)","disease":"Pneumonia, UTI, Bloodstream infections"}
    elif "pseudomonas" in name:
        info = {"gram":"Negative (-)","disease":"Cystic fibrosis lung infections, Burn wound infections"}
    elif "acinetobacter" in name:
        info = {"gram":"Negative (-)","disease":"Nosocomial pneumonia, Bacteremia"}
    elif "vibrio" in name:
        info = {"gram":"Negative (-)","disease":"Cholera, Vibriosis"}
    elif any(k in name for k in ["proteus","morganella","providencia"]):
        info = {"gram":"Negative (-)","disease":"Complicated UTI, Kidney stones"}
    elif "campylobacter" in name or "helicobacter" in name:
        info = {"gram":"Negative (-)","disease":"Peptic ulcers, Gastroenteritis"}
    elif "neisseria" in name:
        info = {"gram":"Negative (-)","disease":"Gonorrhea, Meningitis"}
    elif "haemophilus" in name:
        info = {"gram":"Negative (-)","disease":"Respiratory infections, Meningitis"}
    elif "staphylococcus" in name:
        info = {"gram":"Positive (+)","disease":"Skin infections, MRSA, Endocarditis"}
    elif "streptococcus" in name:
        info = {"gram":"Positive (+)","disease":"Strep throat, Pneumonia, Necrotizing fasciitis"}
    elif "enterococcus" in name:
        info = {"gram":"Positive (+)","disease":"UTI, Endocarditis, VRE infections"}
    elif "bacillus" in name:
        info = {"gram":"Positive (+)","disease":"Anthrax, Food poisoning"}
    elif "clostridium" in name:
        info = {"gram":"Positive (+)","disease":"Tetanus, Botulism, C. diff diarrhea"}
    elif "listeria" in name:
        info = {"gram":"Positive (+)","disease":"Listeriosis, Foodborne illness"}
    elif "corynebacterium" in name:
        info = {"gram":"Positive (+)","disease":"Diphtheria"}
    elif "mycobacterium" in name:
        info = {"gram":"Acid-Fast (Gram Variable)","disease":"Tuberculosis, Leprosy"}
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
    with open(file_name,'r',encoding='utf-8') as f:
        data = json.load(f)
    drug,mech,records = [],[],[]
    habitat = get_habitat(file_name)
    for k in data:
        try:
            inner = list(data[k].values())[0]
            gene_name = inner.get("ARO_name",k).upper()
            d,m = [],[]
            for c in inner.get("ARO_category",{}).values():
                cname = c.get("category_aro_class_name","").lower()
                val   = c.get("category_aro_name","").title()
                if "drug" in cname: d.append(val)
                elif "mechanism" in cname: m.append(val)
            drug.extend(d); mech.extend(m)
            records.append((gene_name,", ".join(set(d)),", ".join(set(m)),habitat.title()))
        except: continue
    genes = len(data)
    mri = (len(set(drug))+len(set(mech)))/(len(drug)+len(mech)+1)
    ari = len(set(mech))/(genes+1)
    return genes,drug,mech,mri,ari,records

def get_level(mri):
    if mri < 0.15: return "LOW","🟢"
    elif mri < 0.35: return "MODERATE","🟡"
    else: return "HIGH","🔴"

def get_risk_reason(level, u_drugs, u_mechs):
    if "HIGH" in level:
        return f"This pathogen's elevated MRI score reflects a sophisticated and redundant resistance architecture. Deploying {u_mechs} distinct biological mechanisms against {u_drugs} drug classes, the organism exhibits the capacity to dynamically bypass standard frontline therapeutics. When one resistance pathway is circumvented, alternative mechanisms compensate — a hallmark of high-priority clinical threats."
    elif "MODERATE" in level:
        return f"This strain demonstrates a clinically significant, though not maximal, resistance burden. Resistance determinants spanning {u_drugs} drug classes have been identified, necessitating careful antibiogram-guided therapy selection. Secondary and combination treatment regimens should be considered to achieve effective clinical outcomes."
    else:
        return f"The calculated MRI indicates a restricted resistance profile, suggestive of ecological specialization rather than broad clinical adaptation. With resistance distributed across {u_drugs} drug classes via {u_mechs} mechanisms, standard empirical treatment protocols remain viable. Continued genomic surveillance is recommended."

@st.cache_resource
def train_rf_model():
    X,y = [],[]
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                genes,drug,mech,mri,_,_ = extract_data(f)
                X.append([genes,len(set(drug)),len(set(mech))])
                y.append("LOW" if mri<0.15 else "MODERATE" if mri<0.35 else "HIGH")
            except: continue
    if len(X)<2: return None
    mdl = RandomForestClassifier(n_estimators=100,random_state=42)
    mdl.fit(X,y); return mdl

# ─────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────
def plot_full_dashboard(drug,mech,mri,genes,records,name):
    level,_ = get_level(mri)
    fig,ax  = plt.subplots(2,3,figsize=(18,12))
    fig.patch.set_facecolor('#050d1a')
    color = "#fb7185" if level=="HIGH" else "#fbbf24" if level=="MODERATE" else "#34d399"
    for a in ax.flat:
        a.set_facecolor('#0d1b2e'); a.tick_params(colors='#94a3b8'); a.title.set_color('#e2e8f0')
        for sp in a.spines.values(): sp.set_color('#1e3a5f')
    drug_c = Counter(drug).most_common(8)
    if drug_c:
        ax[0,0].pie([v for k,v in drug_c],labels=[k[:14]+".." for k,v in drug_c],autopct='%1.1f%%',textprops={'color':'#e2e8f0','fontsize':8})
    ax[0,0].set_title("Drug Classes Resisted",fontweight='bold')
    mech_c = Counter(mech)
    if mech_c:
        ax[0,1].bar([k[:14]+".." for k in mech_c],mech_c.values(),color=color)
        ax[0,1].tick_params(axis='x',rotation=35)
    ax[0,1].set_title("Mechanisms Deployed",fontweight='bold')
    theta = np.linspace(0,np.pi,100)
    ax[0,2].plot(np.cos(theta),np.sin(theta),color='#1e3a5f',linewidth=2)
    ang = mri*np.pi
    ax[0,2].plot([0,np.cos(ang)],[0,np.sin(ang)],color=color,linewidth=6)
    ax[0,2].axis('off'); ax[0,2].set_title(f"MRI: {round(mri,3)} ({level})",fontweight='bold')
    gene_list = [r[0] for r in records]
    gene_c = Counter(gene_list).most_common(5)
    if gene_c:
        ax[1,0].bar([k[:14]+".." for k,v in gene_c],[v for k,v in gene_c],color='#22d3ee')
        ax[1,0].tick_params(axis='x',rotation=35)
    ax[1,0].set_title("Top Gene Frequency",fontweight='bold')
    ax[1,1].bar(["Total Genes"],[genes],color='#6366f1')
    ax[1,1].set_title("Overall Gene Count",fontweight='bold')
    ax[1,2].bar(["Unique Drugs","Unique Mechs"],[len(set(drug)),len(set(mech))],color=["#818cf8","#f59e0b"])
    ax[1,2].set_title("Diversity Comparison",fontweight='bold')
    fig.tight_layout(pad=2)
    return fig

def generate_network_html(records,organism_name,color):
    net = Network(height='600px',width='100%',bgcolor='#050d1a',font_color='#e2e8f0',cdn_resources="in_line",select_menu=True,filter_menu=True)
    net.add_node("HUB",label=organism_name,color=color,size=30)
    for g,d,m,h in records:
        net.add_node(g,label=g[:10],color="#22d3ee",size=15)
        net.add_edge("HUB",g,color="#1e3a5f")
        if m:
            for mech in set(m.split(", ")):
                if not mech: continue
                net.add_node(mech,label=mech[:10],color="#f59e0b",size=10,shape="box")
                net.add_edge(g,mech,color="#334155")
    net.barnes_hut(gravity=-5000)
    html_path = "temp_network.html"
    with open(html_path,"w",encoding="utf-8") as f:
        f.write(net.generate_html())
    return html_path

def plot_3d_pca_plotly(current_file):
    X,files,risk_levels = [],[],[]
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                g,d,m,mr,ar,_ = extract_data(f)
                X.append([mr,len(set(m)),len(set(d))])
                files.append(f)
                lv,_ = get_level(mr)
                risk_levels.append("TARGET 🎯" if f==current_file else lv)
            except: continue
    if len(X)<3:
        st.warning("Insufficient data points for 3D PCA (need ≥3 JSON files)."); return
    pca = PCA(n_components=3).fit_transform(X)
    df_pca = pd.DataFrame(pca,columns=['Resistance (PC1)','Mechanism Diversity (PC2)','Gene Density (PC3)'])
    df_pca['Genome'] = files; df_pca['Risk'] = risk_levels
    cmap = {"HIGH":"#fb7185","MODERATE":"#fbbf24","LOW":"#34d399","TARGET 🎯":"#fde68a"}
    fig = px.scatter_3d(df_pca,x='Resistance (PC1)',y='Mechanism Diversity (PC2)',z='Gene Density (PC3)',
                        color='Risk',hover_name='Genome',color_discrete_map=cmap,opacity=0.85)
    fig.update_layout(paper_bgcolor='#050d1a',font_color='#e2e8f0',margin=dict(l=0,r=0,b=0,t=0),
                      scene=dict(xaxis=dict(backgroundcolor="#0d1b2e",gridcolor="#1e3a5f"),
                                 yaxis=dict(backgroundcolor="#0d1b2e",gridcolor="#1e3a5f"),
                                 zaxis=dict(backgroundcolor="#0d1b2e",gridcolor="#1e3a5f")))
    st.plotly_chart(fig,use_container_width=True)

# ─────────────────────────────────────────
# PDF GENERATOR
# ─────────────────────────────────────────
def create_pdf_report(bac_name,genes,drug,mech,mri,ari,level,icon,records,dashboard_fig,bac_info,habitat,ai_pred,ai_conf):
    pdf_file = f"{bac_name.replace('.json','')}_Report.pdf"
    doc = SimpleDocTemplate(pdf_file,pagesize=letter,rightMargin=30,leftMargin=30,topMargin=30,bottomMargin=30)
    styles = getSampleStyleSheet()
    ts = ParagraphStyle('T',parent=styles['Heading1'],fontSize=18,spaceAfter=14,textColor=colors.HexColor('#1E3A8A'))
    h2 = ParagraphStyle('H2',parent=styles['Heading2'],fontSize=14,spaceBefore=14,spaceAfter=8,textColor=colors.HexColor('#2E86C1'))
    ns = styles['Normal']
    ud,um = len(set(drug)),len(set(mech))
    el = []
    el.append(Paragraph(f"AI-MRI Report: {bac_name.replace('.json','')}",ts))
    el.append(Paragraph("Executive Summary",h2))
    el.append(Paragraph(f"<b>File:</b> {bac_name}<br/><b>Gram:</b> {bac_info['gram']}<br/><b>Pathology:</b> {bac_info['disease']}<br/><b>Habitat:</b> {habitat}<br/><b>Genes:</b> {genes}<br/><b>Drug Classes:</b> {ud}<br/><b>Mechanisms:</b> {um}<br/><b>MRI:</b> {round(mri,3)} ({level})<br/><b>ARI:</b> {round(ari,3)}<br/><b>AI Prediction:</b> {ai_pred}<br/><b>Confidence:</b> {ai_conf}",ns))
    el.append(Spacer(1,14))
    el.append(Paragraph("Mathematical Derivations",h2))
    el.append(Paragraph(f"<b>MRI:</b> ({ud}+{um}) / ({len(drug)}+{len(mech)}+1) = {round(mri,3)}<br/><b>ARI:</b> {um} / ({genes}+1) = {round(ari,3)}",ns))
    el.append(Paragraph("Risk Interpretation",h2))
    el.append(Paragraph(get_risk_reason(level,ud,um),ns))
    el.append(Paragraph("Systems Dashboard",h2))
    buf = io.BytesIO()
    dashboard_fig.patch.set_facecolor('white')
    for a in dashboard_fig.axes:
        a.set_facecolor('white'); a.tick_params(colors='black'); a.title.set_color('black')
        for t in a.texts: t.set_color('black')
    dashboard_fig.savefig(buf,format='png',bbox_inches='tight',dpi=150)
    buf.seek(0)
    el.append(RLImage(buf,width=7.5*inch,height=5*inch))
    el.append(PageBreak())
    el.append(Paragraph("Gene Resistance Ledger",h2))
    td = [["Gene Name","Drug Classes","Mechanisms","Habitat"]]
    for g,d,m,h in records:
        td.append([Paragraph(g,ns),Paragraph(d,ns),Paragraph(m,ns),Paragraph(h,ns)])
    t = Table(td,colWidths=[1.2*inch,2.2*inch,2.2*inch,0.9*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2E86C1')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#F8F9F9')),
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
    ]))
    el.append(t)
    doc.build(el)
    return pdf_file

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'splash'
if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1":[]}
    st.session_state.current_session = "Chat 1"
    st.session_state.chat_counter = 1

# ─────────────────────────────────────────
# ── PAGE: TEAM ──────────────────────────
# ─────────────────────────────────────────
def render_team_page():
    st.markdown("<style>[data-testid='stSidebar']{display:none!important;}</style>", unsafe_allow_html=True)

    # ── Nav bar ──
    col_back, col_title, col_enter = st.columns([1, 3, 1])
    with col_back:
        if st.button("← Splash", use_container_width=True):
            st.session_state.page = 'splash'; st.rerun()
    with col_title:
        st.markdown("<h2 style='text-align:center;color:#22d3ee;font-family:Orbitron,monospace;margin:0;letter-spacing:2px;'>👥 Meet The Team</h2>", unsafe_allow_html=True)
    with col_enter:
        if st.button("🚀 Enter Platform", type="primary", use_container_width=True):
            st.session_state.page = 'app'; st.rerun()

    st.markdown("<div class='divider-bar' style='margin-top:18px;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#64748b;font-size:0.95rem;max-width:720px;margin:0 auto 28px;line-height:1.85;'>AI-MRI Hub was built by a multidisciplinary team of seven researchers, developers, and scientists united by a shared mission — to make antibiotic resistance genomics instantly accessible, intelligible, and actionable for clinicians and researchers worldwide.</p>", unsafe_allow_html=True)

    team = [
        {"emoji":"👩‍🔬","bg":"rgba(236,72,153,0.12)","name":"Poorva Dongarkar","role1":"Project Lead","role2":"Genomics Architect","desc":"Directed end-to-end project execution and overall technical vision. Spearheaded CARD database integration, ARG extraction logic, and the bacterial classification system that powers the platform's core analytics.","tags":["Leadership","Genomics","AMR","Data"]},
        {"emoji":"👨‍💻","bg":"rgba(34,211,238,0.12)","name":"Hardik Agrawal","role1":"Lead Developer","role2":"AI & Systems Engineer","desc":"Architected the full-stack Streamlit platform from the ground up. Designed the proprietary MRI/ARI mathematical framework and built the Random Forest AI prediction pipeline.","tags":["Python","ML","Streamlit","Backend"]},
        {"emoji":"👩‍💻","bg":"rgba(251,191,36,0.12)","name":"Avani Laswante","role1":"Presentation Lead","role2":"UI/UX Designer","desc":"Crafted and delivered all project presentations and the visual narrative strategy. Also responsible for the complete CSS design system, dark-mode aesthetic, and animated visual identity.","tags":["Presentation","CSS","UI/UX","Design"]},
        {"emoji":"👩‍🔬","bg":"rgba(167,139,250,0.12)","name":"Zeel Bhanushali","role1":"Presentation Specialist","role2":"Clinical Research Lead","desc":"Co-led project presentations with scientific depth. Developed the clinical susceptibility logic, drug-class universe mapping, and conducted MRI clinical validation research.","tags":["Presentation","Microbiology","Pharmacology"]},
        {"emoji":"👩‍💻","bg":"rgba(52,211,153,0.12)","name":"Aayushi Wasnik","role1":"Research Lead","role2":"AI Integration Specialist","desc":"Led the scientific research underpinning the platform. Implemented J.A.R.V.I.S. chatbot, Gemini API integration, context injection pipeline, and multi-session state management.","tags":["Research","Gemini API","LLM","Prompt Eng."]},
        {"emoji":"👨‍🔬","bg":"rgba(251,113,133,0.12)","name":"Indranil Patil","role1":"Documentation Lead","role2":"PDF & Technical Writer","desc":"Authored all technical documentation, mathematical write-ups, and the academic framework. Built the ReportLab PDF pipeline that compiles analysis into publication-ready reports.","tags":["Documentation","ReportLab","LaTeX","Writing"]},
        {"emoji":"👨‍🔬","bg":"rgba(34,211,238,0.09)","name":"Yashraj Patil","role1":"Documentation Specialist","role2":"Visualization Engineer","desc":"Co-authored technical documentation and visual reference guides. Engineered PyVis network graphs, Plotly 3D PCA landscape, and the Matplotlib 6-panel analysis dashboard.","tags":["Documentation","PyVis","Plotly","Matplotlib"]},
    ]

    for row_start in range(0, len(team), 3):
        row = team[row_start:row_start+3]
        # Pad to 3 for alignment
        while len(row) < 3: row.append(None)
        cols = st.columns(3)
        for i, member in enumerate(row):
            with cols[i]:
                if member is None: continue
                tags_html = "".join([f'<span class="tp-tag">{t}</span>' for t in member["tags"]])
                st.markdown(f"""
                <div class="tp-card">
                  <div class="tp-avatar" style="background:{member['bg']};">{member['emoji']}</div>
                  <div class="tp-name">{member['name']}</div>
                  <div class="tp-role1">{member['role1']}</div>
                  <div class="tp-role2">{member['role2']}</div>
                  <div class="tp-desc">{member['desc']}</div>
                  <div class="tp-tags">{tags_html}</div>
                </div>
                """, unsafe_allow_html=True)
        st.write("")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("---")

    m1, m2, m3 = st.columns(3)
    for col, icon_e, title, text in [
        (m1, "🎯", "Mission", "Democratize antibiotic resistance genomics through open, intelligent, and beautifully designed scientific tools any researcher or clinician can use."),
        (m2, "🔬", "Methods", "Rigorous mathematical indexing (MRI, ARI), validated machine learning classification, evidence-based clinical susceptibility mapping, and iterative peer review."),
        (m3, "🌍", "Vision",  "Every hospital and research lab equipped with instant AI-driven resistance profiling — making the next superbug detectable before it becomes untreatable."),
    ]:
        with col:
            st.markdown(f'<div class="miss-card"><div class="miss-title">{icon_e} {title}</div><div class="miss-text">{text}</div></div>', unsafe_allow_html=True)

    st.write("")
    col_l, col_c, col_r = st.columns([2,1,2])
    with col_c:
        if st.button("🚀 Enter AI-MRI Hub", type="primary", use_container_width=True, key="team_enter_bottom"):
            st.session_state.page = 'app'; st.rerun()

# ─────────────────────────────────────────
# ── PAGE: SPLASH ─────────────────────────
# ─────────────────────────────────────────
def render_splash_page():
    st.markdown("<style>[data-testid='stSidebar']{display:none!important;} .block-container{padding:0!important;max-width:100%!important;}</style>", unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="sp-hero">
      <div class="sp-dna">🧬</div>
      <h1 class="sp-title">AI-MRI HUB</h1>
      <p class="sp-sub">Antimicrobial Resistance Intelligence Platform</p>
      <p class="sp-desc">The world's most advanced quantitative framework for decoding antibiotic resistance genes. Transform raw genomic data into actionable clinical intelligence — instantly and at scale.</p>
      <div class="sp-stats-row">
        <div class="sp-stat"><span class="sp-stat-num">2</span><span class="sp-stat-lbl">Novel Indices</span></div>
        <div class="sp-stat"><span class="sp-stat-num">8</span><span class="sp-stat-lbl">Analysis Modules</span></div>
        <div class="sp-stat"><span class="sp-stat-num">100+</span><span class="sp-stat-lbl">Genomes Supported</span></div>
        <div class="sp-stat"><span class="sp-stat-num">7</span><span class="sp-stat-lbl">Researchers</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA row
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1.5, 1, 0.25, 1, 1.5])
    with c2:
        if st.button("🚀  ENTER AI-MRI HUB", type="primary", use_container_width=True):
            st.session_state.page = 'app'; st.rerun()
    with c4:
        if st.button("👥  MEET THE TEAM", use_container_width=True):
            st.session_state.page = 'team'; st.rerun()
    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    # ── Features ──
    st.markdown("""
    <div style='text-align:center; padding:10px 0 4px;'>
      <div class='sec-title'>Platform Features</div>
      <div class='sec-bar'></div>
    </div>
    """, unsafe_allow_html=True)

    features = [
        ("🦠","Genomic ARG Profiling","Deep extraction and classification of all Antibiotic Resistance Genes from CARD-format data. Every gene, every drug class — catalogued precisely."),
        ("📐","MRI & ARI Calculation","Proprietary indices transform complex gene counts into a single, instantly interpretable risk score using Laplace-smoothed mathematics."),
        ("🤖","Random Forest AI","Machine learning classifies any pathogen as LOW, MODERATE, or HIGH risk with full probability confidence scores — even for novel strains."),
        ("🕸️","Interactive Network","Visualize the full resistance topology as a live, draggable network graph with filter and selection menus."),
        ("🌌","3D PCA Landscape","Rotate a 3-dimensional scatter map comparing your target genome against every pathogen in the database across resistance axes."),
        ("🩺","Clinical Susceptibility","Automatically identifies drug classes with zero resistance markers — providing instant safe-zone shortlist for treatment."),
        ("💬","J.A.R.V.I.S. AI Chat","Ask questions about any genome in plain English. Gemini-powered AI with full genomic context automatically loaded."),
        ("📄","Master PDF Export","Generate a comprehensive, publication-ready PDF with executive summaries, math proofs, dashboards, and gene ledgers in one click."),
    ]

    for row_start in range(0, len(features), 4):
        cols = st.columns(4)
        for i, (icon_e, title, desc) in enumerate(features[row_start:row_start+4]):
            with cols[i]:
                st.markdown(f"""
                <div class="feat-card">
                  <span class="feat-icon">{icon_e}</span>
                  <div class="feat-title">{title}</div>
                  <div class="feat-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── Why stand apart ──
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding:10px 0 4px;'>
      <div class='sec-title'>Why We Stand Apart</div>
      <div class='sec-bar'></div>
    </div>
    """, unsafe_allow_html=True)

    col_bad, col_good = st.columns(2)
    with col_bad:
        st.markdown("""
        <div class="comp-bad">
          <div class="comp-title">❌ Traditional AMR Tools</div>
          <div class="comp-item">✗ Output raw gene lists — clinicians must manually interpret hundreds of genes.</div>
          <div class="comp-item">✗ No unified index to compare pathogen severity across species.</div>
          <div class="comp-item">✗ Require bioinformatics expertise — inaccessible to clinical staff.</div>
          <div class="comp-item">✗ Static reports with no interactivity or network exploration.</div>
          <div class="comp-item">✗ No AI chatbot for natural language genomic queries.</div>
          <div class="comp-item">✗ No ML prediction for unknown or novel strains.</div>
          <div class="comp-item">✗ Cannot identify safe drug zones automatically.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_good:
        st.markdown("""
        <div class="comp-good">
          <div class="comp-title">✅ AI-MRI Hub</div>
          <div class="comp-item">✓ Proprietary MRI and ARI compress all genomic data into a single readable risk number.</div>
          <div class="comp-item">✓ Cross-species, cross-habitat standardized scoring enables true pathogen comparison.</div>
          <div class="comp-item">✓ Zero bioinformatics expertise required — upload JSON, get full analysis in seconds.</div>
          <div class="comp-item">✓ Live interactive networks and 3D PCA landscape for spatial resistance topology.</div>
          <div class="comp-item">✓ J.A.R.V.I.S. AI chatbot with genome-aware context injected automatically.</div>
          <div class="comp-item">✓ Random Forest model predicts risk of completely unknown pathogens.</div>
          <div class="comp-item">✓ Automatic clinical susceptibility zone detection identifies zero-resistance drugs instantly.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#475569;font-size:0.78rem;letter-spacing:4px;text-transform:uppercase;'>Ready to Begin</p>", unsafe_allow_html=True)
    d1, d2, d3, d4, d5 = st.columns([1.5, 1, 0.25, 1, 1.5])
    with d2:
        if st.button("🚀 ENTER PLATFORM", type="primary", use_container_width=True, key="sp_enter2"):
            st.session_state.page = 'app'; st.rerun()
    with d4:
        if st.button("👥 MEET THE TEAM", use_container_width=True, key="sp_team2"):
            st.session_state.page = 'team'; st.rerun()
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# ── PAGE: APP ────────────────────────────
# ─────────────────────────────────────────
def render_app_page():
    st.markdown("""
    <div class="app-header">
      <h1>🧬 AI-Driven Multidimensional Resistance Index</h1>
      <p style="font-size:1.2rem;">Quantitative Bio-Analysis of Antibiotic Resistance Genes</p>
      <hr style="border:0.5px solid rgba(255,255,255,0.1);margin:16px auto;width:80%;">
      <p style="font-size:0.88rem;font-weight:300;">
        <b>Developed by:</b> Hardik Agrawal · Poorva Dongarkar · Yashraj Patil · Avani Laswante · Zeel Bhanushali · Aayushi Wasnik · Indranil Patil
      </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h3 style='color:#22d3ee;'>🗄️ Navigation</h3>", unsafe_allow_html=True)
        if st.button("🏠 Home / Splash"):
            st.session_state.page = 'splash'; st.rerun()
        if st.button("👥 Meet the Team"):
            st.session_state.page = 'team'; st.rerun()
        st.markdown("---")
        if st.button("🔄 Refresh Database"):
            st.rerun()
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        analysis_mode = st.radio("Analysis Mode:", ["Select Known Bacteria","AI Predict Unknown"])
        selected_file = None
        if analysis_mode == "Select Known Bacteria" and json_files:
            selected_file = st.selectbox("Select a Genome:", json_files)
        st.markdown("---")
        if AI_AVAILABLE:
            st.success("✅ AI Brain Connected")
        else:
            st.warning("⚠️ AI Unavailable")

    json_files = [f for f in os.listdir('.') if f.endswith('.json')]

    if analysis_mode == "Select Known Bacteria" and json_files and selected_file:
        genes,drug,mech,mri,ari,records = extract_data(selected_file)
        u_drugs,u_mechs = len(set(drug)),len(set(mech))
        level,icon = get_level(mri)
        habitat  = get_habitat(selected_file)
        bac_info = get_bacteria_info(selected_file)

        if mri > 0.6:
            st.markdown(f'<div class="alert-ban">⚠️ CRITICAL ALERT: {selected_file} — High-Priority Superbug — MRI: {round(mri,3)}</div>', unsafe_allow_html=True)

        rf_model = train_rf_model()
        ai_pred = "N/A"; ai_conf = "N/A"
        if rf_model:
            pred  = rf_model.predict([[genes,u_drugs,u_mechs]])[0]
            probs = rf_model.predict_proba([[genes,u_drugs,u_mechs]])[0]
            conf_dict = {str(c):round(float(p),3) for c,p in zip(rf_model.classes_,probs)}
            ai_pred = str(pred); ai_conf = str(conf_dict).replace("'","")

        # Metric row
        mc1,mc2,mc3,mc4 = st.columns(4)
        for col, lbl, val in [
            (mc1,"Risk Level",f"{level} {icon}"),
            (mc2,"MRI Score",str(round(mri,3))),
            (mc3,"Total Genes",str(genes)),
            (mc4,"Habitat",habitat),
        ]:
            with col:
                st.markdown(f'<div class="m-card"><div class="m-label">{lbl}</div><div class="m-value">{val}</div></div>', unsafe_allow_html=True)

        st.write("")

        tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
            "ℹ️ Summary","📊 Dashboard","🧮 Math & Ledger",
            "🕸️ Network","🤖 AI Chat","🩺 Clinical","📄 PDF Export","🌌 3D Landscape"
        ])

        # ── Tab 1 ──
        with tab1:
            st.markdown('<div class="w-hero"><h2>Genomic Resistance Intelligence Dashboard</h2><p>Comprehensive AMR Analysis Platform — Powered by AI</p></div>', unsafe_allow_html=True)
            st.info("""**Platform Capabilities:**\n- **Genomic ARG Profiling** — Structured identification of ARGs from CARD-format data\n- **MRI & ARI Indices** — Validated clinical risk scores\n- **Random Forest Classification** — ML-driven threat stratification\n- **Clinical Susceptibility Zones** — Zero-resistance drug class detection\n- **Interactive Topology** — 3D PCA & live network graphs""")
            st.markdown(f"""
<div class="rep-card">
  <div class="rep-hdr">🦠 Bacterial Resistance Intelligence Ledger</div>
  <div class="rep-row"><span class="rep-lbl">Target Genome</span><span class="rep-val" style="color:#60a5fa;">{selected_file}</span></div>
  <div class="rep-row"><span class="rep-lbl">Gram Classification</span><span class="rep-val">{bac_info['gram']}</span></div>
  <div class="rep-row"><span class="rep-lbl">Associated Pathology</span><span class="rep-val">{bac_info['disease']}</span></div>
  <div class="rep-row"><span class="rep-lbl">Ecological Habitat</span><span class="rep-val">{habitat}</span></div>
  <div class="rep-row"><span class="rep-lbl">Total Resistance Genes</span><span class="rep-val">{genes}</span></div>
  <div class="rep-row"><span class="rep-lbl">Drug Classes Resisted</span><span class="rep-val">{u_drugs} Classes</span></div>
  <div class="rep-row"><span class="rep-lbl">Resistance Mechanisms</span><span class="rep-val">{u_mechs} Strategies</span></div>
  <div class="rep-row"><span class="rep-lbl">MRI / ARI Score</span><span class="rep-val">{round(mri,3)} ({level}) &nbsp;/&nbsp; {round(ari,3)}</span></div>
  <div style="margin-top:16px;">
    <span class="rep-lbl">AI Risk Classification:</span>&nbsp;<span class="ai-badge">{ai_pred} Risk</span>
    <br><small style="color:#475569;margin-top:6px;display:block;">Probability: {ai_conf}</small>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Tab 2 ──
        with tab2:
            st.markdown(f"### 📊 Systems Dashboard — `{selected_file}`")
            fig = plot_full_dashboard(drug,mech,mri,genes,records,selected_file)
            st.pyplot(fig)

        # ── Tab 3 ──
        with tab3:
            st.markdown("### 🧮 Mathematical Framework")
            cm1,cm2 = st.columns(2)
            with cm1:
                st.markdown('<div class="math-card"><div class="math-hdr">Multidimensional Resistance Index (MRI)</div>', unsafe_allow_html=True)
                st.latex(r"MRI = \frac{U_{drugs} + U_{mechs}}{T_{drugs} + T_{mechs} + 1}")
                st.markdown(f'<div class="ann-box"><div class="ann-item"><span class="ann-key">U_drugs</span> = {u_drugs} unique drug classes</div><div class="ann-item"><span class="ann-key">U_mechs</span> = {u_mechs} unique mechanisms</div><div class="ann-item"><span class="ann-key">T_drugs</span> = {len(drug)} total drug records</div><div class="ann-item"><span class="ann-key">T_mechs</span> = {len(mech)} total mechanism records</div></div>', unsafe_allow_html=True)
                st.markdown("**Result:**")
                st.latex(rf"\frac{{{u_drugs}+{u_mechs}}}{{{len(drug)}+{len(mech)}+1}} = {round(mri,3)}")
                st.markdown('</div>', unsafe_allow_html=True)
            with cm2:
                st.markdown('<div class="math-card"><div class="math-hdr">Antibiotic Resistance Index (ARI)</div>', unsafe_allow_html=True)
                st.latex(r"ARI = \frac{U_{mechs}}{G_{total} + 1}")
                st.markdown(f'<div class="ann-box"><div class="ann-item"><span class="ann-key">U_mechs</span> = {u_mechs} unique mechanisms</div><div class="ann-item"><span class="ann-key">G_total</span> = {genes} total resistance genes</div></div>', unsafe_allow_html=True)
                st.markdown("**Result:**")
                st.latex(rf"\frac{{{u_mechs}}}{{{genes}+1}} = {round(ari,3)}")
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("### 🎯 Risk Interpretation")
            st.markdown(f'<div class="reason-box">{get_risk_reason(level,u_drugs,u_mechs)}</div>', unsafe_allow_html=True)
            st.write("")
            st.markdown("### 📜 Gene Resistance Ledger")
            df = pd.DataFrame(records,columns=["Gene","Drug Classes","Mechanisms","Habitat"])
            st.dataframe(df,use_container_width=True)

        # ── Tab 4 ──
        with tab4:
            st.markdown("### 🕸️ Resistance Mechanism Network")
            st.caption("Use the filter / selection menus inside the graph to isolate specific nodes.")
            hl_path = generate_network_html(records,selected_file,"#fb7185" if level=="HIGH" else "#fbbf24" if level=="MODERATE" else "#34d399")
            with open(hl_path,'r',encoding='utf-8') as f:
                components.html(f.read(),height=650)

        # ── Tab 5 ──
        with tab5:
            colA,colB,colC = st.columns([0.6,0.2,0.2])
            with colA:
                st.session_state.current_session = st.selectbox("Active Session:", list(st.session_state.chat_sessions.keys()))
            with colB:
                st.write("")
                if st.button("➕ New Chat",use_container_width=True):
                    st.session_state.chat_counter += 1
                    nn = f"Chat {st.session_state.chat_counter}"
                    st.session_state.chat_sessions[nn] = []
                    st.session_state.current_session = nn; st.rerun()
            with colC:
                st.write("")
                if st.button("🗑️ Clear",use_container_width=True):
                    st.session_state.chat_sessions[st.session_state.current_session] = []; st.rerun()
            for msg in st.session_state.chat_sessions[st.session_state.current_session]:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            user_msg = st.chat_input(f"Query J.A.R.V.I.S. about {selected_file}…")
            if user_msg:
                st.chat_message("user").markdown(user_msg)
                st.session_state.chat_sessions[st.session_state.current_session].append({"role":"user","content":user_msg})
                if not AI_AVAILABLE:
                    st.error("⚠️ AI library unavailable.")
                else:
                    try:
                        ctx = f"You are J.A.R.V.I.S., expert Bioinformatics AI. File: '{selected_file}'. ARGs: {genes}, Drug Classes: {u_drugs}, Mechanisms: {u_mechs}, MRI: {round(mri,3)} ({level}), ARI: {round(ari,3)}. Query: {user_msg}"
                        with st.spinner("Processing…"):
                            avail = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                            tgt = next((m for m in avail if 'flash' in m), avail[0] if avail else None)
                            if tgt:
                                resp = genai.GenerativeModel(tgt).generate_content(ctx)
                                st.chat_message("assistant").markdown(resp.text)
                                st.session_state.chat_sessions[st.session_state.current_session].append({"role":"assistant","content":resp.text})
                    except Exception as e:
                        em = str(e)
                        if "429" in em or "quota" in em.lower():
                            st.error("⚠️ API Quota Exceeded. Please wait 60 seconds.")
                        else:
                            st.error(f"AI Error: {e}")

        # ── Tab 6 ──
        with tab6:
            st.markdown("### 🩺 Clinical Susceptibility Zone Analysis")
            DRUG_UNIVERSE = ["Penicillin","Cephalosporin","Carbapenem","Macrolide","Aminoglycoside","Fluoroquinolone","Tetracycline","Sulfonamide","Glycopeptide"]
            resisted_norm = {d.lower() for d in drug}
            safe = [d for d in DRUG_UNIVERSE if d.lower() not in resisted_norm]
            st.write("Drug classes with **zero resistance markers** identified in this isolate:")
            st.markdown(f'<div class="susc-card">🛡️ Candidate Classes: {", ".join(safe) if safe else "None identified"}</div>', unsafe_allow_html=True)
            st.caption("⚠️ Confirmation via standard antibiogram (MIC testing) required before clinical use.")
            st.write("---")
            st.markdown("### 📈 Population Benchmark")
            all_mris = []
            for f in json_files:
                try:
                    _,_,_,fm,_,_ = extract_data(f); all_mris.append(fm)
                except: pass
            if all_mris:
                avg_mri = sum(all_mris)/len(all_mris)
                st.bar_chart(pd.DataFrame({"MRI Score":[mri,avg_mri]},index=["Target","DB Average"]))

        # ── Tab 7 ──
        with tab7:
            st.markdown("### 📥 Generate Master PDF Report")
            st.write("Compiles executive summary, math derivations, dashboard charts, and the full gene ledger into a single PDF.")
            if st.button("⚙️ Generate PDF Report", type="primary"):
                with st.spinner("Compiling PDF…"):
                    fig_pdf = plot_full_dashboard(drug,mech,mri,genes,records,selected_file)
                    pdf_path = create_pdf_report(selected_file,genes,drug,mech,mri,ari,level,icon,records,fig_pdf,bac_info,habitat,ai_pred,ai_conf)
                    with open(pdf_path,"rb") as pf:
                        st.download_button("⬇️ Download PDF",data=pf,file_name=pdf_path,mime="application/pdf")

        # ── Tab 8 ──
        with tab8:
            st.markdown("### 🌌 3D Resistance Landscape (PCA)")
            st.caption("Rotate / zoom using the toolbar. Your target genome is marked in gold.")
            plot_3d_pca_plotly(selected_file)

    elif analysis_mode == "AI Predict Unknown":
        st.header("🤖 ML Risk Classification — Unknown Pathogen")
        st.write("Input resistance parameters for an uncharacterised isolate to obtain an AI-driven risk stratification.")
        ig = st.number_input("Total Resistance Genes",min_value=1,value=15)
        id_ = st.number_input("Unique Drug Classes Resisted",min_value=1,value=5)
        im = st.number_input("Unique Resistance Mechanisms",min_value=1,value=2)
        rf_model = train_rf_model()
        if rf_model and st.button("Run Risk Classification",type="primary"):
            pred  = rf_model.predict([[ig,id_,im]])[0]
            probs = rf_model.predict_proba([[ig,id_,im]])[0]
            prob_str = " | ".join([f"{c}: {p:.3f}" for c,p in zip(rf_model.classes_,probs)])
            st.success(f"### AI Classification: **{pred}**")
            st.info(f"**Probability Distribution:** {prob_str}")
    else:
        if not json_files:
            st.warning("⚠️ No JSON genome files found in the working directory. Please upload CARD-format JSON files.")

# ─────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────
if st.session_state.page == 'splash':
    render_splash_page()
elif st.session_state.page == 'team':
    render_team_page()
elif st.session_state.page == 'app':
    render_app_page()
