import streamlit as st
import json
import os
import io
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

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
        st.error("🔑 API Key Missing: Please add 'GENAI_API_KEY' to your Streamlit Cloud Secrets.")
except ImportError:
    AI_AVAILABLE = False

st.set_page_config(page_title="AI-MRI Hub", layout="wide", page_icon="🧬")

# ==========================================
# REFERENCE DATABASE FOR ENVIRONMENTAL BENCHMARKS
# ==========================================
REFERENCE_BENCHMARKS = {
    "clinical": {
        "Aminoglycoside": 0.72, "Beta-lactam": 0.85, "Fluoroquinolone": 0.68,
        "Carbapenem": 0.45, "Tetracycline": 0.55, "Macrolide": 0.60,
        "Sulfonamide": 0.50, "Glycopeptide": 0.30, "Colistin": 0.20,
        "Trimethoprim": 0.58, "Chloramphenicol": 0.42, "Rifampicin": 0.35,
        "avg_mri": 0.42, "avg_ari": 0.18, "avg_genes": 85
    },
    "agricultural": {
        "Aminoglycoside": 0.65, "Beta-lactam": 0.55, "Fluoroquinolone": 0.40,
        "Carbapenem": 0.15, "Tetracycline": 0.80, "Macrolide": 0.70,
        "Sulfonamide": 0.75, "Glycopeptide": 0.10, "Colistin": 0.55,
        "Trimethoprim": 0.45, "Chloramphenicol": 0.60, "Rifampicin": 0.20,
        "avg_mri": 0.32, "avg_ari": 0.12, "avg_genes": 52
    },
    "environmental": {
        "Aminoglycoside": 0.40, "Beta-lactam": 0.35, "Fluoroquinolone": 0.28,
        "Carbapenem": 0.10, "Tetracycline": 0.45, "Macrolide": 0.38,
        "Sulfonamide": 0.42, "Glycopeptide": 0.08, "Colistin": 0.12,
        "Trimethoprim": 0.30, "Chloramphenicol": 0.35, "Rifampicin": 0.15,
        "avg_mri": 0.18, "avg_ari": 0.06, "avg_genes": 28
    },
    "wastewater": {
        "Aminoglycoside": 0.58, "Beta-lactam": 0.62, "Fluoroquinolone": 0.50,
        "Carbapenem": 0.28, "Tetracycline": 0.70, "Macrolide": 0.65,
        "Sulfonamide": 0.68, "Glycopeptide": 0.18, "Colistin": 0.35,
        "Trimethoprim": 0.55, "Chloramphenicol": 0.50, "Rifampicin": 0.25,
        "avg_mri": 0.35, "avg_ari": 0.14, "avg_genes": 65
    },
    "food_production": {
        "Aminoglycoside": 0.50, "Beta-lactam": 0.48, "Fluoroquinolone": 0.35,
        "Carbapenem": 0.12, "Tetracycline": 0.75, "Macrolide": 0.60,
        "Sulfonamide": 0.65, "Glycopeptide": 0.08, "Colistin": 0.45,
        "Trimethoprim": 0.40, "Chloramphenicol": 0.55, "Rifampicin": 0.18,
        "avg_mri": 0.28, "avg_ari": 0.10, "avg_genes": 42
    }
}

ORIGIN_LABELS = {
    "🏥 Clinical / Hospital": "clinical",
    "🌾 Agricultural / Livestock": "agricultural",
    "🌿 Environmental / Soil": "environmental",
    "💧 Wastewater / Sewage": "wastewater",
    "🍖 Food Production": "food_production"
}

DRUG_CLASS_KEYWORDS = {
    "Aminoglycoside": ["aminoglycoside", "amikacin", "gentamicin", "tobramycin", "streptomycin"],
    "Beta-lactam": ["beta-lactam", "penicillin", "cephalosporin", "ampicillin", "methicillin"],
    "Fluoroquinolone": ["fluoroquinolone", "quinolone", "ciprofloxacin", "levofloxacin"],
    "Carbapenem": ["carbapenem", "imipenem", "meropenem", "ertapenem"],
    "Tetracycline": ["tetracycline", "doxycycline", "minocycline"],
    "Macrolide": ["macrolide", "erythromycin", "azithromycin", "clarithromycin"],
    "Sulfonamide": ["sulfonamide", "sulfamethoxazole", "trimethoprim-sulfamethoxazole"],
    "Glycopeptide": ["glycopeptide", "vancomycin", "teicoplanin"],
    "Colistin": ["colistin", "polymyxin"],
    "Trimethoprim": ["trimethoprim", "diaminopyrimidine"],
    "Chloramphenicol": ["chloramphenicol", "phenicol"],
    "Rifampicin": ["rifampicin", "rifamycin", "rifampin"]
}

def map_drug_to_standard_class(drug_name):
    dl = drug_name.lower()
    for std_class, keywords in DRUG_CLASS_KEYWORDS.items():
        if any(kw in dl for kw in keywords):
            return std_class
    return None

# ==========================================
# ══════════════════════════════════════════
#  VIRTUAL LAB MODULE — inlined from lab_simulation.py
# ══════════════════════════════════════════
# ==========================================

LAB_BACTERIA_DB = {
    "Escherichia coli": {
        "gram": "Negative", "shape": "Rod", "arrangement": "Single/Pairs",
        "motility": True, "spore": False, "capsule": False,
        "colony_color": "#c8e6c9", "colony_size": "medium",
        "disease": "Gastroenteritis, UTI, Sepsis, Neonatal meningitis",
        "optimal_temp": 37, "optimal_ph": 7.2,
        "typical_mri": 0.31, "typical_ari": 0.11,
        "base_resistance": {"Beta-lactam":0.45,"Fluoroquinolone":0.38,"Aminoglycoside":0.25,
                            "Tetracycline":0.50,"Sulfonamide":0.55,"Trimethoprim":0.48,
                            "Carbapenem":0.08,"Macrolide":0.20,"Glycopeptide":0.02,
                            "Colistin":0.05,"Chloramphenicol":0.30,"Rifampicin":0.10},
        "notable_genes": ["blaTEM","blaCTX-M","aadA","sul1","tetA","qnrB","mcr-1"],
        "gram_color": "#ef9f27", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "MacConkey Agar", "ferments_lactose": True,
    },
    "Klebsiella pneumoniae": {
        "gram": "Negative", "shape": "Rod", "arrangement": "Single/Pairs",
        "motility": False, "spore": False, "capsule": True,
        "colony_color": "#ffe082", "colony_size": "large",
        "disease": "Pneumonia, UTI, Bloodstream infections, Liver abscess",
        "optimal_temp": 37, "optimal_ph": 7.0,
        "typical_mri": 0.38, "typical_ari": 0.14,
        "base_resistance": {"Beta-lactam":0.55,"Fluoroquinolone":0.42,"Aminoglycoside":0.30,
                            "Tetracycline":0.45,"Sulfonamide":0.50,"Trimethoprim":0.42,
                            "Carbapenem":0.20,"Macrolide":0.22,"Glycopeptide":0.03,
                            "Colistin":0.10,"Chloramphenicol":0.28,"Rifampicin":0.12},
        "notable_genes": ["blaNDM","blaKPC","blaOXA-48","rmtB","oqxAB","mcr-1","tet(A)"],
        "gram_color": "#ef9f27", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "MacConkey Agar", "ferments_lactose": True,
    },
    "Pseudomonas aeruginosa": {
        "gram": "Negative", "shape": "Rod", "arrangement": "Single",
        "motility": True, "spore": False, "capsule": False,
        "colony_color": "#b3e5fc", "colony_size": "medium",
        "disease": "Cystic fibrosis lung infections, Burn wound infections, HAP",
        "optimal_temp": 37, "optimal_ph": 7.0,
        "typical_mri": 0.40, "typical_ari": 0.15,
        "base_resistance": {"Beta-lactam":0.50,"Fluoroquinolone":0.45,"Aminoglycoside":0.38,
                            "Tetracycline":0.30,"Sulfonamide":0.25,"Trimethoprim":0.20,
                            "Carbapenem":0.30,"Macrolide":0.15,"Glycopeptide":0.02,
                            "Colistin":0.08,"Chloramphenicol":0.22,"Rifampicin":0.18},
        "notable_genes": ["mexAB-oprM","mexXY","oprD","blaVIM","blaIMP","aac(6')-Ib","fosA"],
        "gram_color": "#ef9f27", "oxidase": "Positive", "catalase": "Positive",
        "selective_media": "Cetrimide Agar", "ferments_lactose": False,
    },
    "Acinetobacter baumannii": {
        "gram": "Negative", "shape": "Coccobacillus", "arrangement": "Pairs/Clusters",
        "motility": False, "spore": False, "capsule": False,
        "colony_color": "#ffccbc", "colony_size": "small",
        "disease": "Nosocomial pneumonia, Bacteremia, Wound infections",
        "optimal_temp": 37, "optimal_ph": 7.0,
        "typical_mri": 0.44, "typical_ari": 0.17,
        "base_resistance": {"Beta-lactam":0.60,"Fluoroquinolone":0.55,"Aminoglycoside":0.45,
                            "Tetracycline":0.42,"Sulfonamide":0.38,"Trimethoprim":0.35,
                            "Carbapenem":0.35,"Macrolide":0.20,"Glycopeptide":0.05,
                            "Colistin":0.12,"Chloramphenicol":0.32,"Rifampicin":0.28},
        "notable_genes": ["blaOXA-23","blaOXA-51","blaADC","armA","abaR","adeABC"],
        "gram_color": "#ef9f27", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "Leeds Acinetobacter Medium", "ferments_lactose": False,
    },
    "Salmonella enterica": {
        "gram": "Negative", "shape": "Rod", "arrangement": "Single",
        "motility": True, "spore": False, "capsule": False,
        "colony_color": "#e8f5e9", "colony_size": "medium",
        "disease": "Salmonellosis, Typhoid Fever, Bacteremia",
        "optimal_temp": 37, "optimal_ph": 7.2,
        "typical_mri": 0.28, "typical_ari": 0.09,
        "base_resistance": {"Beta-lactam":0.35,"Fluoroquinolone":0.32,"Aminoglycoside":0.22,
                            "Tetracycline":0.48,"Sulfonamide":0.55,"Trimethoprim":0.40,
                            "Carbapenem":0.05,"Macrolide":0.15,"Glycopeptide":0.01,
                            "Colistin":0.08,"Chloramphenicol":0.42,"Rifampicin":0.08},
        "notable_genes": ["blaTEM","aadA","cmlA","sul1","tet(G)","invA","spvC"],
        "gram_color": "#ef9f27", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "Salmonella-Shigella Agar", "ferments_lactose": False,
    },
    "Shigella sonnei": {
        "gram": "Negative", "shape": "Rod", "arrangement": "Single",
        "motility": False, "spore": False, "capsule": False,
        "colony_color": "#fce4ec", "colony_size": "small",
        "disease": "Dysentery, Bloody diarrhoea, HUS",
        "optimal_temp": 37, "optimal_ph": 7.2,
        "typical_mri": 0.26, "typical_ari": 0.09,
        "base_resistance": {"Beta-lactam":0.38,"Fluoroquinolone":0.30,"Aminoglycoside":0.20,
                            "Tetracycline":0.55,"Sulfonamide":0.60,"Trimethoprim":0.50,
                            "Carbapenem":0.04,"Macrolide":0.12,"Glycopeptide":0.01,
                            "Colistin":0.04,"Chloramphenicol":0.50,"Rifampicin":0.06},
        "notable_genes": ["blaTEM","sul1","dfrA","tetB","icsA","set1A"],
        "gram_color": "#ef9f27", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "Xylose Lysine Deoxycholate Agar", "ferments_lactose": False,
    },
    "Vibrio cholerae": {
        "gram": "Negative", "shape": "Curved rod", "arrangement": "Single",
        "motility": True, "spore": False, "capsule": False,
        "colony_color": "#e0f2f1", "colony_size": "medium",
        "disease": "Cholera, Vibriosis, Diarrhoea",
        "optimal_temp": 37, "optimal_ph": 8.5,
        "typical_mri": 0.22, "typical_ari": 0.07,
        "base_resistance": {"Beta-lactam":0.20,"Fluoroquinolone":0.25,"Aminoglycoside":0.15,
                            "Tetracycline":0.35,"Sulfonamide":0.40,"Trimethoprim":0.30,
                            "Carbapenem":0.03,"Macrolide":0.10,"Glycopeptide":0.01,
                            "Colistin":0.05,"Chloramphenicol":0.28,"Rifampicin":0.05},
        "notable_genes": ["ctxA","ctxB","tcpA","VPI-1","SXT","VC0395"],
        "gram_color": "#ef9f27", "oxidase": "Positive", "catalase": "Positive",
        "selective_media": "TCBS Agar", "ferments_lactose": False,
    },
    "Enterobacter cloacae": {
        "gram": "Negative", "shape": "Rod", "arrangement": "Single",
        "motility": True, "spore": False, "capsule": False,
        "colony_color": "#f0f4c3", "colony_size": "medium",
        "disease": "Hospital-acquired pneumonia, UTI, Wound infections",
        "optimal_temp": 37, "optimal_ph": 7.2,
        "typical_mri": 0.35, "typical_ari": 0.13,
        "base_resistance": {"Beta-lactam":0.48,"Fluoroquinolone":0.40,"Aminoglycoside":0.28,
                            "Tetracycline":0.42,"Sulfonamide":0.45,"Trimethoprim":0.38,
                            "Carbapenem":0.15,"Macrolide":0.18,"Glycopeptide":0.02,
                            "Colistin":0.08,"Chloramphenicol":0.25,"Rifampicin":0.10},
        "notable_genes": ["AmpC","blaNDM","ompC","OXA-1","qnrS","aac(6')-Ib"],
        "gram_color": "#ef9f27", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "MacConkey Agar", "ferments_lactose": True,
    },
    "Proteus mirabilis": {
        "gram": "Negative", "shape": "Rod", "arrangement": "Single",
        "motility": True, "spore": False, "capsule": False,
        "colony_color": "#ede7f6", "colony_size": "swarming",
        "disease": "Complicated UTI, Kidney stones, Wound infections",
        "optimal_temp": 37, "optimal_ph": 7.0,
        "typical_mri": 0.29, "typical_ari": 0.10,
        "base_resistance": {"Beta-lactam":0.40,"Fluoroquinolone":0.35,"Aminoglycoside":0.20,
                            "Tetracycline":0.45,"Sulfonamide":0.42,"Trimethoprim":0.38,
                            "Carbapenem":0.06,"Macrolide":0.55,"Glycopeptide":0.02,
                            "Colistin":0.80,"Chloramphenicol":0.20,"Rifampicin":0.08},
        "notable_genes": ["blaTEM","aac(3)-Ia","tetM","ZapB","fliA","mrpA"],
        "gram_color": "#ef9f27", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "CLED Agar", "ferments_lactose": False,
    },
    "Campylobacter jejuni": {
        "gram": "Negative", "shape": "Curved rod", "arrangement": "Single/Pairs",
        "motility": True, "spore": False, "capsule": False,
        "colony_color": "#f8bbd0", "colony_size": "small",
        "disease": "Gastroenteritis, Guillain-Barré syndrome",
        "optimal_temp": 42, "optimal_ph": 7.0,
        "typical_mri": 0.25, "typical_ari": 0.08,
        "base_resistance": {"Beta-lactam":0.30,"Fluoroquinolone":0.55,"Aminoglycoside":0.18,
                            "Tetracycline":0.60,"Sulfonamide":0.35,"Trimethoprim":0.28,
                            "Carbapenem":0.05,"Macrolide":0.18,"Glycopeptide":0.01,
                            "Colistin":0.04,"Chloramphenicol":0.22,"Rifampicin":0.08},
        "notable_genes": ["gyrA(T86I)","tet(O)","aph(3')-Ia","cmeABC","cadF","flaA"],
        "gram_color": "#ef9f27", "oxidase": "Positive", "catalase": "Positive",
        "selective_media": "mCCDA Agar", "ferments_lactose": False,
    },
    "Neisseria gonorrhoeae": {
        "gram": "Negative", "shape": "Coccus", "arrangement": "Diplococci",
        "motility": False, "spore": False, "capsule": False,
        "colony_color": "#fce4ec", "colony_size": "small",
        "disease": "Gonorrhea, Pelvic inflammatory disease, Neonatal ophthalmia",
        "optimal_temp": 37, "optimal_ph": 7.2,
        "typical_mri": 0.33, "typical_ari": 0.11,
        "base_resistance": {"Beta-lactam":0.50,"Fluoroquinolone":0.60,"Aminoglycoside":0.10,
                            "Tetracycline":0.55,"Sulfonamide":0.50,"Trimethoprim":0.40,
                            "Carbapenem":0.04,"Macrolide":0.15,"Glycopeptide":0.02,
                            "Colistin":0.06,"Chloramphenicol":0.18,"Rifampicin":0.12},
        "notable_genes": ["penA","mtrR","porB","ponA","tetM","blaTEM"],
        "gram_color": "#ef9f27", "oxidase": "Positive", "catalase": "Positive",
        "selective_media": "Thayer-Martin Agar", "ferments_lactose": False,
    },
    "Haemophilus influenzae": {
        "gram": "Negative", "shape": "Coccobacillus", "arrangement": "Single",
        "motility": False, "spore": False, "capsule": True,
        "colony_color": "#e8eaf6", "colony_size": "small",
        "disease": "Respiratory infections, Meningitis, Epiglottitis",
        "optimal_temp": 37, "optimal_ph": 7.4,
        "typical_mri": 0.24, "typical_ari": 0.08,
        "base_resistance": {"Beta-lactam":0.45,"Fluoroquinolone":0.20,"Aminoglycoside":0.12,
                            "Tetracycline":0.38,"Sulfonamide":0.45,"Trimethoprim":0.40,
                            "Carbapenem":0.02,"Macrolide":0.12,"Glycopeptide":0.01,
                            "Colistin":0.04,"Chloramphenicol":0.28,"Rifampicin":0.05},
        "notable_genes": ["blaTEM","ROB-1","ftsi","acrAB","mef(A)","cat"],
        "gram_color": "#ef9f27", "oxidase": "Positive", "catalase": "Positive",
        "selective_media": "Chocolate Agar", "ferments_lactose": False,
    },
    "Staphylococcus aureus": {
        "gram": "Positive", "shape": "Coccus", "arrangement": "Clusters (grapes)",
        "motility": False, "spore": False, "capsule": True,
        "colony_color": "#fff9c4", "colony_size": "medium",
        "disease": "Skin infections, MRSA, Endocarditis, Toxic shock syndrome",
        "optimal_temp": 37, "optimal_ph": 7.4,
        "typical_mri": 0.35, "typical_ari": 0.13,
        "base_resistance": {"Beta-lactam":0.60,"Fluoroquinolone":0.40,"Aminoglycoside":0.28,
                            "Tetracycline":0.38,"Sulfonamide":0.30,"Trimethoprim":0.25,
                            "Carbapenem":0.10,"Macrolide":0.42,"Glycopeptide":0.05,
                            "Colistin":0.50,"Chloramphenicol":0.15,"Rifampicin":0.10},
        "notable_genes": ["mecA","pvl","blaZ","aacA-aphD","tetM","msrA","vanA"],
        "gram_color": "#d85a30", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "Mannitol Salt Agar", "ferments_lactose": True,
    },
    "Streptococcus pneumoniae": {
        "gram": "Positive", "shape": "Coccus", "arrangement": "Diplococci/Chains",
        "motility": False, "spore": False, "capsule": True,
        "colony_color": "#e8f5e9", "colony_size": "small",
        "disease": "Pneumonia, Meningitis, Otitis media, Bacteremia",
        "optimal_temp": 37, "optimal_ph": 7.4,
        "typical_mri": 0.30, "typical_ari": 0.10,
        "base_resistance": {"Beta-lactam":0.40,"Fluoroquinolone":0.18,"Aminoglycoside":0.08,
                            "Tetracycline":0.35,"Sulfonamide":0.40,"Trimethoprim":0.38,
                            "Carbapenem":0.08,"Macrolide":0.38,"Glycopeptide":0.01,
                            "Colistin":0.30,"Chloramphenicol":0.12,"Rifampicin":0.08},
        "notable_genes": ["pbp1a","pbp2b","pbp2x","mefA","erm(B)","tet(M)","catpC194"],
        "gram_color": "#d85a30", "oxidase": "Negative", "catalase": "Negative",
        "selective_media": "Blood Agar", "ferments_lactose": True,
    },
    "Enterococcus faecium": {
        "gram": "Positive", "shape": "Coccus", "arrangement": "Pairs/Short chains",
        "motility": False, "spore": False, "capsule": False,
        "colony_color": "#f3e5f5", "colony_size": "small",
        "disease": "UTI, Endocarditis, VRE infections, Bacteremia",
        "optimal_temp": 37, "optimal_ph": 7.2,
        "typical_mri": 0.33, "typical_ari": 0.12,
        "base_resistance": {"Beta-lactam":0.55,"Fluoroquinolone":0.45,"Aminoglycoside":0.35,
                            "Tetracycline":0.45,"Sulfonamide":0.35,"Trimethoprim":0.40,
                            "Carbapenem":0.25,"Macrolide":0.50,"Glycopeptide":0.20,
                            "Colistin":0.40,"Chloramphenicol":0.18,"Rifampicin":0.15},
        "notable_genes": ["vanA","vanB","pbp5","aph(3')-IIIa","erm(B)","tetM"],
        "gram_color": "#d85a30", "oxidase": "Negative", "catalase": "Negative",
        "selective_media": "Bile Aesculin Agar", "ferments_lactose": True,
    },
    "Bacillus anthracis": {
        "gram": "Positive", "shape": "Rod", "arrangement": "Chains",
        "motility": False, "spore": True, "capsule": True,
        "colony_color": "#f5f5f5", "colony_size": "large",
        "disease": "Anthrax (cutaneous, inhalation, gastrointestinal)",
        "optimal_temp": 37, "optimal_ph": 7.2,
        "typical_mri": 0.18, "typical_ari": 0.06,
        "base_resistance": {"Beta-lactam":0.15,"Fluoroquinolone":0.05,"Aminoglycoside":0.08,
                            "Tetracycline":0.05,"Sulfonamide":0.10,"Trimethoprim":0.08,
                            "Carbapenem":0.02,"Macrolide":0.10,"Glycopeptide":0.01,
                            "Colistin":0.02,"Chloramphenicol":0.05,"Rifampicin":0.02},
        "notable_genes": ["pXO1","pXO2","pagA","lef","cya","capB"],
        "gram_color": "#d85a30", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "Blood Agar (BSL-3)", "ferments_lactose": False,
    },
    "Clostridium difficile": {
        "gram": "Positive", "shape": "Rod", "arrangement": "Single",
        "motility": True, "spore": True, "capsule": False,
        "colony_color": "#fff8e1", "colony_size": "medium",
        "disease": "C. diff diarrhoea, Pseudomembranous colitis",
        "optimal_temp": 37, "optimal_ph": 7.0,
        "typical_mri": 0.22, "typical_ari": 0.08,
        "base_resistance": {"Beta-lactam":0.55,"Fluoroquinolone":0.68,"Aminoglycoside":0.20,
                            "Tetracycline":0.25,"Sulfonamide":0.30,"Trimethoprim":0.28,
                            "Carbapenem":0.05,"Macrolide":0.30,"Glycopeptide":0.01,
                            "Colistin":0.15,"Chloramphenicol":0.10,"Rifampicin":0.08},
        "notable_genes": ["tcdA","tcdB","cdtA","cdtB","erm(B)","gyrA(T82I)"],
        "gram_color": "#d85a30", "oxidase": "Negative", "catalase": "Negative",
        "selective_media": "CCFA Agar (anaerobic)", "ferments_lactose": False,
    },
    "Listeria monocytogenes": {
        "gram": "Positive", "shape": "Rod", "arrangement": "Single/Pairs",
        "motility": True, "spore": False, "capsule": False,
        "colony_color": "#e8f5e9", "colony_size": "small",
        "disease": "Listeriosis, Foodborne illness, Neonatal meningitis",
        "optimal_temp": 37, "optimal_ph": 7.2,
        "typical_mri": 0.20, "typical_ari": 0.07,
        "base_resistance": {"Beta-lactam":0.10,"Fluoroquinolone":0.15,"Aminoglycoside":0.12,
                            "Tetracycline":0.20,"Sulfonamide":0.18,"Trimethoprim":0.15,
                            "Carbapenem":0.02,"Macrolide":0.08,"Glycopeptide":0.01,
                            "Colistin":0.05,"Chloramphenicol":0.08,"Rifampicin":0.03},
        "notable_genes": ["hlyA","actA","inlA","inlB","prfA","tet(M)","cat"],
        "gram_color": "#d85a30", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "Oxford Agar", "ferments_lactose": True,
    },
    "Mycobacterium tuberculosis": {
        "gram": "Acid-Fast", "shape": "Rod", "arrangement": "Single/Cords",
        "motility": False, "spore": False, "capsule": True,
        "colony_color": "#fff9c4", "colony_size": "very slow (weeks)",
        "disease": "Tuberculosis (pulmonary, extrapulmonary), Miliary TB",
        "optimal_temp": 37, "optimal_ph": 7.0,
        "typical_mri": 0.28, "typical_ari": 0.09,
        "base_resistance": {"Beta-lactam":0.70,"Fluoroquinolone":0.20,"Aminoglycoside":0.15,
                            "Tetracycline":0.30,"Sulfonamide":0.25,"Trimethoprim":0.20,
                            "Carbapenem":0.10,"Macrolide":0.12,"Glycopeptide":0.02,
                            "Colistin":0.05,"Chloramphenicol":0.08,"Rifampicin":0.15},
        "notable_genes": ["katG(S315T)","rpoB","embB","gyrA","rpsL","rrs","inhA"],
        "gram_color": "#9e9e9e", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "Löwenstein-Jensen Medium", "ferments_lactose": False,
    },
    "Corynebacterium diphtheriae": {
        "gram": "Positive", "shape": "Rod (club-shaped)", "arrangement": "V/L shapes",
        "motility": False, "spore": False, "capsule": False,
        "colony_color": "#ffe0b2", "colony_size": "small",
        "disease": "Diphtheria, Pharyngitis, Cutaneous infections",
        "optimal_temp": 37, "optimal_ph": 7.2,
        "typical_mri": 0.18, "typical_ari": 0.06,
        "base_resistance": {"Beta-lactam":0.15,"Fluoroquinolone":0.10,"Aminoglycoside":0.10,
                            "Tetracycline":0.20,"Sulfonamide":0.18,"Trimethoprim":0.15,
                            "Carbapenem":0.02,"Macrolide":0.10,"Glycopeptide":0.01,
                            "Colistin":0.04,"Chloramphenicol":0.05,"Rifampicin":0.03},
        "notable_genes": ["dtxR","tox","diphtheria_tox","erm(X)","aad9"],
        "gram_color": "#d85a30", "oxidase": "Negative", "catalase": "Positive",
        "selective_media": "Tellurite Blood Agar", "ferments_lactose": False,
    },
}

LAB_ANTIBIOTICS = [
    {"name": "Ciprofloxacin",    "class": "Fluoroquinolone", "S_break": 1,   "R_break": 4,   "unit": "µg/mL"},
    {"name": "Ampicillin",       "class": "Beta-lactam",     "S_break": 8,   "R_break": 32,  "unit": "µg/mL"},
    {"name": "Meropenem",        "class": "Carbapenem",      "S_break": 1,   "R_break": 8,   "unit": "µg/mL"},
    {"name": "Gentamicin",       "class": "Aminoglycoside",  "S_break": 4,   "R_break": 16,  "unit": "µg/mL"},
    {"name": "Tetracycline",     "class": "Tetracycline",    "S_break": 4,   "R_break": 16,  "unit": "µg/mL"},
    {"name": "Trimethoprim",     "class": "Trimethoprim",    "S_break": 8,   "R_break": 16,  "unit": "µg/mL"},
    {"name": "Chloramphenicol",  "class": "Chloramphenicol", "S_break": 8,   "R_break": 32,  "unit": "µg/mL"},
    {"name": "Erythromycin",     "class": "Macrolide",       "S_break": 1,   "R_break": 4,   "unit": "µg/mL"},
    {"name": "Vancomycin",       "class": "Glycopeptide",    "S_break": 4,   "R_break": 16,  "unit": "µg/mL"},
    {"name": "Colistin",         "class": "Colistin",        "S_break": 2,   "R_break": 4,   "unit": "µg/mL"},
    {"name": "Rifampicin",       "class": "Rifampicin",      "S_break": 1,   "R_break": 4,   "unit": "µg/mL"},
    {"name": "Sulfamethoxazole", "class": "Sulfonamide",     "S_break": 8,   "R_break": 256, "unit": "µg/mL"},
]

LAB_DILUTIONS = [0.06, 0.12, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]


def _lab_mic_for_antibiotic(ab, resistance_factor):
    base = ab["S_break"]
    if resistance_factor > 0.65:
        mic_raw = base * (2 ** (3 + resistance_factor * 5))
    elif resistance_factor > 0.35:
        mic_raw = base * (2 ** (1 + resistance_factor * 3))
    else:
        mic_raw = base * (2 ** (-1 + resistance_factor * 2))
    closest = min(LAB_DILUTIONS, key=lambda d: abs(d - mic_raw))
    if closest <= ab["S_break"]:    interp = "S"
    elif closest <= ab["R_break"]:  interp = "I"
    else:                           interp = "R"
    return closest, interp


def _lab_get_risk(mri):
    if mri < 0.15:  return "LOW",      "#3B6D11", "🟢"
    if mri < 0.35:  return "MODERATE", "#BA7517", "🟡"
    return "HIGH", "#A32D2D", "🔴"


def _lab_scan_json_files():
    mapping = {}
    for f in os.listdir('.'):
        if not f.endswith('.json'):
            continue
        fl = f.lower()
        for bname in LAB_BACTERIA_DB:
            key = bname.split()[0].lower()
            if key in fl:
                mapping[f] = bname
                break
    return mapping


def _lab_simulate(bacteria_name, json_file, resistance_override,
                  temperature, ph, incubation_h, cfu_exp, medium):
    db   = LAB_BACTERIA_DB[bacteria_name]
    seed = hash(bacteria_name + json_file) % 99999
    rng  = random.Random(seed + int(resistance_override * 1000))

    real_genes = real_u_drugs = real_u_mechs = 0
    drug_set, mech_set = set(), set()
    try:
        with open(json_file, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        real_genes = len(raw)
        for k in raw:
            try:
                inner = list(raw[k].values())[0]
                for c in inner.get("ARO_category", {}).values():
                    cname = c.get("category_aro_class_name", "").lower()
                    val   = c.get("category_aro_name", "")
                    if "drug" in cname:        drug_set.add(val)
                    elif "mechanism" in cname: mech_set.add(val)
            except:
                pass
        real_u_drugs = len(drug_set)
        real_u_mechs = len(mech_set)
    except:
        real_genes   = max(5, int(db["typical_mri"] * 80))
        real_u_drugs = max(2, int(db["typical_mri"] * 10))
        real_u_mechs = max(1, int(db["typical_mri"] * 7))

    t_drugs   = max(real_u_drugs * 3, real_genes // 2)
    t_mechs   = max(real_u_mechs * 3, real_genes // 3)
    lab_mri   = (real_u_drugs + real_u_mechs) / (t_drugs + t_mechs + 1)
    lab_mri   = min(0.99, lab_mri * (0.85 + resistance_override * 0.30))
    lab_ari   = real_u_mechs / (real_genes + 1)
    lab_ari   = min(0.99, lab_ari * (0.9 + resistance_override * 0.20))

    ai_mri = round(min(0.99, lab_mri * (0.93 + rng.uniform(0, 0.14))), 3)
    ai_ari = round(min(0.99, lab_ari * (0.90 + rng.uniform(0, 0.20))), 3)

    temp_pen    = max(0, abs(temperature - db["optimal_temp"]) * 0.06)
    ph_pen      = max(0, abs(ph - db["optimal_ph"]) * 0.15)
    growth_rate = max(0.1, 1.0 - temp_pen - ph_pen + resistance_override * 0.05)

    time_pts  = list(range(0, incubation_h + 1, max(1, incubation_h // 20)))
    LAG       = max(1, int(2 / growth_rate))
    LOG_DUR   = max(4, int(8 * growth_rate))
    growth_cfu = []
    for t in time_pts:
        base_cfu = 10 ** cfu_exp
        if t < LAG:
            growth_cfu.append(round(base_cfu * 0.9))
        elif t < LAG + LOG_DUR:
            growth_cfu.append(round(base_cfu * (2 ** ((t - LAG) * growth_rate * 0.8))))
        else:
            plateau = base_cfu * (2 ** (LOG_DUR * growth_rate * 0.8))
            growth_cfu.append(round(plateau * max(0.5, 1.0 - (t - LAG - LOG_DUR) * 0.01)))

    mic_results = []
    for ab in LAB_ANTIBIOTICS:
        base_res   = db["base_resistance"].get(ab["class"], 0.2)
        res_factor = min(0.98, base_res * (1.0 + resistance_override * 0.8) + rng.uniform(-0.05, 0.05))
        mic_val, mic_interp = _lab_mic_for_antibiotic(ab, res_factor)
        ai_res  = min(0.98, res_factor * (0.88 + rng.uniform(0, 0.24)))
        ai_mic, ai_interp = _lab_mic_for_antibiotic(ab, ai_res)
        mic_results.append({
            "name": ab["name"], "class": ab["class"],
            "lab_mic": mic_val, "lab_interp": mic_interp,
            "ai_mic":  ai_mic,  "ai_interp":  ai_interp,
            "S_break": ab["S_break"], "R_break": ab["R_break"],
        })

    colony_count = max(5, int(30 * growth_rate + rng.uniform(-5, 10)))

    return {
        "bacteria": bacteria_name, "json_file": json_file, "db": db,
        "real_genes": real_genes, "real_u_drugs": real_u_drugs, "real_u_mechs": real_u_mechs,
        "lab_mri": round(lab_mri, 3), "lab_ari": round(lab_ari, 3),
        "ai_mri": ai_mri, "ai_ari": ai_ari,
        "lab_risk": _lab_get_risk(lab_mri), "ai_risk": _lab_get_risk(ai_mri),
        "growth_rate": round(growth_rate, 3),
        "growth_times": [str(t) + "h" for t in time_pts],
        "growth_cfu": growth_cfu, "colony_count": colony_count,
        "mic_results": mic_results,
        "temperature": temperature, "ph": ph,
        "incubation_h": incubation_h, "cfu_exp": cfu_exp,
        "medium": medium, "resistance_override": resistance_override,
    }


def _lab_build_html(sim):
    db        = sim["db"]
    gram_col  = db["gram_color"]
    colony_c  = db["colony_color"]
    risk_lab  = sim["lab_risk"]
    risk_ai   = sim["ai_risk"]

    growth_labels_js = json.dumps(sim["growth_times"])
    growth_data_js   = json.dumps([round(v / 1e6, 2) for v in sim["growth_cfu"]])

    ab_names  = [r["name"] for r in sim["mic_results"]]
    lab_mics  = [math.log2(max(r["lab_mic"], 0.03)) for r in sim["mic_results"]]
    ai_mics   = [math.log2(max(r["ai_mic"],  0.03)) for r in sim["mic_results"]]

    classes          = list(dict.fromkeys([r["class"] for r in sim["mic_results"]]))
    lab_profile_vals = []
    ai_profile_vals  = []
    for cl in classes:
        lv = [math.log2(max(r["lab_mic"],0.03)) for r in sim["mic_results"] if r["class"]==cl]
        av = [math.log2(max(r["ai_mic"], 0.03)) for r in sim["mic_results"] if r["class"]==cl]
        lab_profile_vals.append(round(sum(lv)/len(lv), 2) if lv else 0)
        ai_profile_vals.append(round(sum(av)/len(av),  2) if av else 0)

    mri_delta  = round(abs(sim["lab_mri"] - sim["ai_mri"]), 3)
    ari_delta  = round(abs(sim["lab_ari"] - sim["ai_ari"]), 3)
    mri_match  = "IDENTICAL" if mri_delta < 0.03 else "NEAR MATCH" if mri_delta < 0.08 else "DIVERGED"
    risk_agree = sim["lab_risk"][0] == sim["ai_risk"][0]

    matched_ab = sum(1 for r in sim["mic_results"] if r["lab_interp"] == r["ai_interp"])
    total_ab   = len(sim["mic_results"])
    match_pct  = round(matched_ab / total_ab * 100)

    mic_rows = ""
    for r in sim["mic_results"]:
        lab_cls   = {"S":"#3B6D11","I":"#BA7517","R":"#A32D2D"}[r["lab_interp"]]
        ai_cls    = {"S":"#3B6D11","I":"#BA7517","R":"#A32D2D"}[r["ai_interp"]]
        match     = r["lab_interp"] == r["ai_interp"]
        match_sym = "✓" if match else "≈" if abs(r["lab_mic"]-r["ai_mic"]) < r["S_break"]*2 else "✗"
        match_col = "#3B6D11" if match else "#BA7517" if match_sym=="≈" else "#A32D2D"
        mic_rows += f"""<tr>
          <td style="padding:8px 10px;font-size:13px;">{r['name']}</td>
          <td style="padding:8px 10px;font-size:11px;color:#888;">{r['class']}</td>
          <td style="padding:8px 10px;font-size:13px;font-weight:500;">{r['lab_mic']} µg/mL</td>
          <td style="padding:8px 10px;"><span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;background:{lab_cls}22;color:{lab_cls};">{r['lab_interp']}</span></td>
          <td style="padding:8px 10px;font-size:13px;font-weight:500;">{r['ai_mic']} µg/mL</td>
          <td style="padding:8px 10px;"><span style="display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;background:{ai_cls}22;color:{ai_cls};">{r['ai_interp']}</span></td>
          <td style="padding:8px 10px;font-size:18px;color:{match_col};text-align:center;">{match_sym}</td>
        </tr>"""

    rng2       = random.Random(hash(sim["bacteria"]) % 9999)
    colony_svg = ""
    for _ in range(sim["colony_count"]):
        cx = rng2.randint(15, 185); cy = rng2.randint(15, 185); r2 = rng2.randint(3, 9)
        colony_svg += f'<circle cx="{cx}" cy="{cy}" r="{r2}" fill="{colony_c}" stroke="#aaa" stroke-width="0.5" opacity="0.85"/>'

    gram_fill      = gram_col
    gram_shape_svg = ""
    shape          = db["shape"].lower()
    gram_rng       = random.Random(42)
    if "coccus" in shape or "cocci" in shape:
        for _ in range(18):
            gx = gram_rng.randint(10, 110); gy = gram_rng.randint(10, 70)
            gram_shape_svg += f'<ellipse cx="{gx}" cy="{gy}" rx="5" ry="5" fill="{gram_fill}" opacity="0.8"/>'
            if db["arrangement"] in ["Diplococci", "Pairs/Chains", "Diplococci/Chains"]:
                gram_shape_svg += f'<ellipse cx="{gx+11}" cy="{gy}" rx="5" ry="5" fill="{gram_fill}" opacity="0.8"/>'
    else:
        for _ in range(14):
            gx = gram_rng.randint(5, 90); gy = gram_rng.randint(5, 65); ang = gram_rng.randint(0, 180)
            gram_shape_svg += f'<rect x="{gx}" y="{gy}" width="18" height="7" rx="3" fill="{gram_fill}" opacity="0.8" transform="rotate({ang},{gx+9},{gy+3})"/>'

    def risk_badge(rt):
        label, col, icon = rt
        return f'<span style="display:inline-block;padding:3px 10px;border-radius:4px;background:{col}22;color:{col};font-size:12px;font-weight:700;">{icon} {label}</span>'

    res_bars = ""
    for cl in ["Beta-lactam","Fluoroquinolone","Aminoglycoside","Carbapenem",
               "Tetracycline","Macrolide","Glycopeptide","Colistin"]:
        v   = db["base_resistance"].get(cl, 0)
        pct = round(v * 100)
        bar_col = '#ef4444' if v > 0.5 else '#f59e0b' if v > 0.25 else '#22c55e'
        res_bars += f'''<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
          <span style="font-size:12px;min-width:120px;color:#e2e8f0;">{cl}</span>
          <div style="flex:1;height:6px;background:#1e293b;border-radius:3px;overflow:hidden;">
            <div style="height:100%;width:{pct}%;background:{bar_col};border-radius:3px;"></div></div>
          <span style="font-size:11px;min-width:36px;text-align:right;color:#64748b;">{pct}%</span>
        </div>'''

    prof_bars = ""
    for i, (ab, r) in enumerate(zip(LAB_ANTIBIOTICS, sim["mic_results"])):
        w   = round(min(r["lab_mic"] / r["R_break"], 1) * 100)
        bc  = '#ef4444' if r["lab_interp"]=="R" else '#f59e0b' if r["lab_interp"]=="I" else '#22c55e'
        lbl = ab["class"] if i == 0 or ab["class"] not in [LAB_ANTIBIOTICS[j]["class"] for j in range(i)] else ""
        prof_bars += f'''<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
          <span style="font-size:11px;min-width:120px;color:#e2e8f0;">{lbl}</span>
          <div style="flex:1;height:6px;background:#1e293b;border-radius:3px;overflow:hidden;">
            <div style="height:100%;width:{w}%;background:{bc};border-radius:3px;"></div></div>
          <span style="font-size:11px;min-width:120px;text-align:right;color:{bc};">{r["name"]}: {r["lab_mic"]} µg/mL</span>
        </div>'''

    gene_tags = "".join(f'<span style="display:inline-block;padding:3px 9px;border-radius:20px;font-size:11px;font-family:monospace;background:rgba(0,201,255,0.08);border:1px solid rgba(0,201,255,0.2);color:#67e8f9;margin:2px;">{g}</span>' for g in db['notable_genes'])

    alert_html = (
        '<div style="padding:10px 16px;border-radius:8px;font-size:13px;font-weight:600;margin-bottom:14px;background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);color:#f87171;">⚠️ HIGH PRIORITY PATHOGEN — MRI Score exceeds 0.35 threshold</div>'
        if sim["lab_risk"][0] == "HIGH" else
        '<div style="padding:10px 16px;border-radius:8px;font-size:13px;font-weight:600;margin-bottom:14px;background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);color:#fbbf24;">⚡ MODERATE RISK — Antibiogram-guided therapy recommended</div>'
        if sim["lab_risk"][0] == "MODERATE" else
        '<div style="padding:10px 16px;border-radius:8px;font-size:13px;font-weight:600;margin-bottom:14px;background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);color:#4ade80;">✓ LOW RISK — Standard empirical therapy protocols viable</div>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Virtual Lab — {sim['bacteria']}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Sora',sans-serif;background:#0a0f1a;color:#e2e8f0;min-height:100vh}}
:root{{--accent:#00c9ff;--accent2:#7c3aed;--card:#111827;--border:rgba(255,255,255,0.07);--muted:#64748b}}
.lab-shell{{max-width:1200px;margin:0 auto;padding:24px 20px}}
.lab-header{{background:linear-gradient(135deg,#0c1f3f,#111827,#0f1a2e);border:1px solid var(--border);border-radius:16px;padding:24px 28px;margin-bottom:20px;position:relative;overflow:hidden}}
.lab-header::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00c9ff,#7c3aed,#10b981,#00c9ff);background-size:300% auto;animation:scanBar 4s linear infinite}}
@keyframes scanBar{{0%{{background-position:0% center}}100%{{background-position:300% center}}}}
.bac-name{{font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:600;color:#00c9ff;margin-bottom:4px}}
.tab-nav{{display:flex;gap:2px;border-bottom:1px solid var(--border);margin-bottom:20px;overflow-x:auto}}
.tab-btn{{padding:10px 18px;font-size:13px;font-weight:600;color:var(--muted);background:none;border:none;border-bottom:2px solid transparent;cursor:pointer;white-space:nowrap;transition:all 0.2s;font-family:'Sora',sans-serif}}
.tab-btn:hover{{color:#e2e8f0}}.tab-btn.active{{color:#00c9ff;border-bottom-color:#00c9ff}}
.tab-panel{{display:none}}.tab-panel.active{{display:block}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px 22px;margin-bottom:16px}}
.card-title{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#00c9ff;margin-bottom:14px}}
.grid-2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}}
.grid-3{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}}
.grid-4{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.grid-5{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}}
.metric{{background:#0d1420;border:1px solid var(--border);border-radius:10px;padding:14px 16px;text-align:center}}
.metric-lbl{{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin-bottom:6px}}
.metric-val{{font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace}}
.metric-sub{{font-size:10px;color:var(--muted);margin-top:3px}}
.match-card{{border-radius:10px;padding:12px 16px;margin-bottom:10px;display:flex;align-items:center;gap:14px}}
.match-icon{{font-size:20px;flex-shrink:0}}
.match-label{{font-size:13px;font-weight:600}}
.match-detail{{font-size:11px;color:var(--muted);margin-top:2px}}
.match-ok{{background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2)}}
.match-near{{background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2)}}
.match-diff{{background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2)}}
.mic-table{{width:100%;border-collapse:collapse}}
.mic-table th{{padding:8px 10px;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);border-bottom:1px solid var(--border);text-align:left;font-weight:600}}
.mic-table td{{border-bottom:1px solid rgba(255,255,255,0.03)}}
.mic-table tr:hover td{{background:rgba(255,255,255,0.02)}}
.chart-wrap{{position:relative;width:100%;height:220px}}
.chart-wrap-xl{{position:relative;width:100%;height:320px}}
.scope-viewport{{width:200px;height:200px;border-radius:50%;border:4px solid #334155;overflow:hidden;background:#f5f0e8;margin:0 auto}}
.plate-viewport{{width:200px;height:200px;border-radius:50%;border:4px solid #334155;background:#1a2e1a;margin:0 auto;position:relative;overflow:hidden}}
.plate-label{{text-align:center;font-size:12px;color:var(--muted);margin-top:8px}}
.badge{{display:inline-block;padding:3px 10px;border-radius:4px;font-size:11px;font-weight:700}}
.info-row{{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:13px}}
.info-row:last-child{{border-bottom:none}}
.info-key{{color:var(--muted)}}
.info-val{{font-weight:600;text-align:right}}
.log-box{{background:#060d1a;border:1px solid var(--border);border-radius:8px;padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:11px;max-height:220px;overflow-y:auto;line-height:1.8}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:translateY(0)}}}}
.animate{{animation:fadeUp .35s ease both}}
hr{{border:none;border-top:1px solid var(--border);margin:16px 0}}
</style>
</head>
<body>
<div class="lab-shell">

<div class="lab-header animate">
  <div style="display:grid;grid-template-columns:1fr auto;gap:20px;align-items:start">
    <div>
      <div class="bac-name">🔬 {sim['bacteria']}</div>
      <div style="font-size:13px;color:var(--muted);margin-top:4px">{db['disease']}</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:10px;">
        <span class="badge" style="background:rgba(0,201,255,0.1);color:#00c9ff;border:1px solid rgba(0,201,255,0.2)">Gram {db['gram']}</span>
        <span class="badge" style="background:rgba(124,58,237,0.1);color:#a78bfa;border:1px solid rgba(124,58,237,0.2)">{db['shape']}</span>
        <span class="badge" style="background:rgba(16,185,129,0.1);color:#34d399;border:1px solid rgba(16,185,129,0.2)">{'Motile' if db['motility'] else 'Non-motile'}</span>
        <span class="badge" style="background:rgba(245,158,11,0.1);color:#fbbf24;border:1px solid rgba(245,158,11,0.2)">{'Spore-forming' if db['spore'] else 'Non-spore'}</span>
        <span class="badge" style="background:rgba(239,68,68,0.1);color:#f87171;border:1px solid rgba(239,68,68,0.2)">{'Encapsulated' if db['capsule'] else 'No capsule'}</span>
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-size:11px;color:var(--muted);margin-bottom:4px">FILE</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#67e8f9">{sim['json_file']}</div>
      <div style="font-size:11px;color:var(--muted);margin-top:8px">LAB CONDITIONS</div>
      <div style="font-size:12px;color:#e2e8f0">{sim['temperature']}°C · pH {sim['ph']} · {sim['incubation_h']}h · 10<sup>{sim['cfu_exp']}</sup> CFU/mL</div>
    </div>
  </div>
</div>

<div class="tab-nav">
  <button class="tab-btn active" onclick="switchTab('overview',this)">Overview</button>
  <button class="tab-btn" onclick="switchTab('culture',this)">Culture</button>
  <button class="tab-btn" onclick="switchTab('mic',this)">MIC Plate</button>
  <button class="tab-btn" onclick="switchTab('compare',this)">Lab vs AI</button>
  <button class="tab-btn" onclick="switchTab('profile',this)">Drug Profile</button>
  <button class="tab-btn" onclick="switchTab('genes',this)">Genes & Identity</button>
</div>

<!-- TAB 1: OVERVIEW -->
<div id="tab-overview" class="tab-panel active animate">
  {alert_html}
  <div class="grid-5" style="margin-bottom:14px">
    <div class="metric"><div class="metric-lbl">Lab MRI</div><div class="metric-val" style="color:{risk_lab[1]}">{sim['lab_mri']}</div><div class="metric-sub">{risk_lab[0]}</div></div>
    <div class="metric"><div class="metric-lbl">AI MRI</div><div class="metric-val" style="color:{risk_ai[1]}">{sim['ai_mri']}</div><div class="metric-sub">{risk_ai[0]}</div></div>
    <div class="metric"><div class="metric-lbl">Lab ARI</div><div class="metric-val" style="color:#00c9ff">{sim['lab_ari']}</div><div class="metric-sub">Density index</div></div>
    <div class="metric"><div class="metric-lbl">Total Genes</div><div class="metric-val">{sim['real_genes']}</div><div class="metric-sub">ARGs detected</div></div>
    <div class="metric"><div class="metric-lbl">Drug Classes</div><div class="metric-val">{sim['real_u_drugs']}</div><div class="metric-sub">Classes resisted</div></div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Organism identity</div>
      <div class="info-row"><span class="info-key">Gram stain</span><span class="info-val" style="color:{db['gram_color']}">{db['gram']}</span></div>
      <div class="info-row"><span class="info-key">Morphology</span><span class="info-val">{db['shape']}</span></div>
      <div class="info-row"><span class="info-key">Arrangement</span><span class="info-val">{db['arrangement']}</span></div>
      <div class="info-row"><span class="info-key">Optimal temp</span><span class="info-val">{db['optimal_temp']}°C</span></div>
      <div class="info-row"><span class="info-key">Optimal pH</span><span class="info-val">{db['optimal_ph']}</span></div>
      <div class="info-row"><span class="info-key">Oxidase</span><span class="info-val">{db['oxidase']}</span></div>
      <div class="info-row"><span class="info-key">Catalase</span><span class="info-val">{db['catalase']}</span></div>
      <div class="info-row"><span class="info-key">Lactose fermentation</span><span class="info-val">{'Positive' if db['ferments_lactose'] else 'Negative'}</span></div>
      <div class="info-row"><span class="info-key">Selective medium</span><span class="info-val" style="max-width:180px;text-align:right">{db['selective_media']}</span></div>
    </div>
    <div class="card">
      <div class="card-title">Resistance profile summary</div>
      {res_bars}
      <div style="font-size:10px;color:var(--muted);margin-top:10px">* Population-level reference benchmarks</div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">AI vs Lab match summary</div>
    <div class="grid-3">
      <div class="match-card {'match-ok' if mri_delta<0.03 else 'match-near' if mri_delta<0.08 else 'match-diff'}">
        <span class="match-icon">{'✓' if mri_delta<0.03 else '~' if mri_delta<0.08 else '✗'}</span>
        <div><div class="match-label">MRI reading</div><div class="match-detail">Lab: {sim['lab_mri']} | AI: {sim['ai_mri']} | Δ={mri_delta}</div></div>
      </div>
      <div class="match-card {'match-ok' if ari_delta<0.02 else 'match-near' if ari_delta<0.06 else 'match-diff'}">
        <span class="match-icon">{'✓' if ari_delta<0.02 else '~' if ari_delta<0.06 else '✗'}</span>
        <div><div class="match-label">ARI reading</div><div class="match-detail">Lab: {sim['lab_ari']} | AI: {sim['ai_ari']} | Δ={ari_delta}</div></div>
      </div>
      <div class="match-card {'match-ok' if risk_agree else 'match-diff'}">
        <span class="match-icon">{'✓' if risk_agree else '✗'}</span>
        <div><div class="match-label">Risk classification</div><div class="match-detail">Lab: {risk_lab[0]} | AI: {risk_ai[0]}</div></div>
      </div>
    </div>
    <div style="margin-top:10px;padding:10px 14px;background:rgba(0,201,255,0.05);border-radius:8px;font-size:12px;color:#94a3b8">
      Antibiotic concordance: <strong style="color:{'#22c55e' if match_pct>=80 else '#f59e0b' if match_pct>=60 else '#ef4444'}">{matched_ab}/{total_ab} agents ({match_pct}%)</strong> agree between lab and AI.
      {'<span style="color:#22c55e"> High concordance — AI well-calibrated.</span>' if match_pct>=80 else '<span style="color:#f59e0b"> Moderate concordance — review individual MICs.</span>' if match_pct>=60 else '<span style="color:#ef4444"> Low concordance — phenotypic confirmation recommended.</span>'}
    </div>
  </div>
</div>

<!-- TAB 2: CULTURE -->
<div id="tab-culture" class="tab-panel animate">
  <div class="grid-4" style="margin-bottom:14px">
    <div class="metric"><div class="metric-lbl">Growth rate</div><div class="metric-val" style="color:#00c9ff">{sim['growth_rate']}</div><div class="metric-sub">factor</div></div>
    <div class="metric"><div class="metric-lbl">Final density</div><div class="metric-val">{round(sim['growth_cfu'][-1]/1e6,1)}</div><div class="metric-sub">×10⁶ CFU/mL</div></div>
    <div class="metric"><div class="metric-lbl">Colonies</div><div class="metric-val">{sim['colony_count']}</div><div class="metric-sub">on plate</div></div>
    <div class="metric"><div class="metric-lbl">Gram stain</div><div class="metric-val" style="font-size:14px;color:{db['gram_color']}">{db['gram']}</div><div class="metric-sub">{db['shape']}</div></div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title">Growth curve (CFU/mL × 10⁶)</div>
      <div class="chart-wrap"><canvas id="growthChart"></canvas></div>
      <div style="font-size:11px;color:var(--muted);margin-top:8px">Medium: {sim['medium']} · {sim['incubation_h']}h</div>
    </div>
    <div class="card">
      <div class="card-title">Colony morphology plate</div>
      <div class="plate-viewport"><svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">{colony_svg}</svg></div>
      <div class="plate-label">{db['colony_size'].capitalize()} colonies · {db['selective_media']}</div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Gram stain microscopy (×1000)</div>
    <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap">
      <div>
        <div class="scope-viewport">
          <svg width="120" height="80" xmlns="http://www.w3.org/2000/svg" style="background:#f5f0e8">{gram_shape_svg}</svg>
        </div>
        <div class="plate-label" style="margin-top:8px">Gram {db['gram']} · {db['shape']}</div>
      </div>
      <div style="flex:1;min-width:200px">
        <div class="info-row"><span class="info-key">Gram stain result</span><span class="info-val" style="color:{db['gram_color']}">{'Purple (crystal violet)' if db['gram']=='Positive' else 'Pink/red (safranin)' if db['gram']=='Negative' else 'Red (acid-fast)'}</span></div>
        <div class="info-row"><span class="info-key">Cell morphology</span><span class="info-val">{db['shape']}</span></div>
        <div class="info-row"><span class="info-key">Arrangement</span><span class="info-val">{db['arrangement']}</span></div>
        <div class="info-row"><span class="info-key">Spore formation</span><span class="info-val">{'Endospores visible' if db['spore'] else 'None'}</span></div>
        <div class="info-row"><span class="info-key">Capsule</span><span class="info-val">{'Present (India ink halo)' if db['capsule'] else 'Absent'}</span></div>
        <div class="info-row"><span class="info-key">Motility test</span><span class="info-val">{'Positive (turbid)' if db['motility'] else 'Negative (stab line)'}</span></div>
        <div class="info-row"><span class="info-key">Oxidase</span><span class="info-val">{db['oxidase']}</span></div>
        <div class="info-row"><span class="info-key">Catalase</span><span class="info-val">{db['catalase']}</span></div>
      </div>
    </div>
  </div>
</div>

<!-- TAB 3: MIC PLATE -->
<div id="tab-mic" class="tab-panel animate">
  <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-bottom:14px">
    <span style="font-size:12px;color:var(--muted)">Select antibiotic:</span>
    <div id="ab-selector" style="display:flex;gap:6px;flex-wrap:wrap"></div>
  </div>
  <div class="card">
    <div class="card-title" id="mic-plate-title">MIC Plate</div>
    <div style="font-size:11px;color:var(--muted);margin-bottom:12px">96-well format · Rows = replicates · Columns = doubling dilutions</div>
    <div id="mic-plate-grid" style="display:grid;grid-template-columns:repeat(12,1fr);gap:4px;margin-bottom:8px"></div>
    <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--muted);padding:0 2px">
      <span>0.06</span><span>0.12</span><span>0.25</span><span>0.5</span><span>1</span><span>2</span><span>4</span><span>8</span><span>16</span><span>32</span><span>64</span><span>128 µg/mL</span>
    </div>
    <div style="margin-top:14px;padding:10px 16px;background:#0d1420;border-radius:8px;font-size:13px">
      MIC = <strong id="mic-result-val" style="color:#00c9ff;font-family:'JetBrains Mono',monospace">—</strong> µg/mL &nbsp;|&nbsp;
      Interpretation: <strong id="mic-result-interp">—</strong> &nbsp;|&nbsp;
      S ≤ <span id="mic-s-break">—</span> µg/mL &nbsp;|&nbsp; R > <span id="mic-r-break">—</span> µg/mL
    </div>
  </div>
  <div class="card">
    <div class="card-title">Complete MIC summary</div>
    <div style="overflow-x:auto">
      <table class="mic-table">
        <thead><tr>
          <th>Antibiotic</th><th>Class</th><th>Lab MIC</th><th>Lab</th><th>AI MIC</th><th>AI</th><th style="text-align:center">Match</th>
        </tr></thead>
        <tbody>{mic_rows}</tbody>
      </table>
    </div>
    <div style="display:flex;gap:16px;margin-top:12px;font-size:12px;flex-wrap:wrap">
      <span style="color:#22c55e">S=Susceptible</span>
      <span style="color:#f59e0b">I=Intermediate</span>
      <span style="color:#ef4444">R=Resistant</span>
      <span style="color:var(--muted)">✓=Match ≈=Near ✗=Diverged</span>
    </div>
  </div>
</div>

<!-- TAB 4: LAB vs AI -->
<div id="tab-compare" class="tab-panel animate">
  <div class="grid-4" style="margin-bottom:14px">
    <div class="metric"><div class="metric-lbl">Lab MRI</div><div class="metric-val" style="color:{risk_lab[1]}">{sim['lab_mri']}</div><div class="metric-sub">{risk_badge(risk_lab)}</div></div>
    <div class="metric"><div class="metric-lbl">AI MRI</div><div class="metric-val" style="color:{risk_ai[1]}">{sim['ai_mri']}</div><div class="metric-sub">{risk_badge(risk_ai)}</div></div>
    <div class="metric"><div class="metric-lbl">MRI Δ</div><div class="metric-val" style="color:{'#22c55e' if mri_delta<0.03 else '#f59e0b' if mri_delta<0.08 else '#ef4444'}">{mri_delta}</div><div class="metric-sub">{mri_match}</div></div>
    <div class="metric"><div class="metric-lbl">AB Concordance</div><div class="metric-val" style="color:{'#22c55e' if match_pct>=80 else '#f59e0b' if match_pct>=60 else '#ef4444'}">{match_pct}%</div><div class="metric-sub">{matched_ab}/{total_ab} agents</div></div>
  </div>
  <div class="grid-2">
    <div class="card">
      <div class="card-title">MRI / ARI bar comparison</div>
      <div class="chart-wrap"><canvas id="mriChart"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Risk confidence</div>
      <div id="risk-conf" style="padding:8px 0"></div>
      <hr>
      <div class="card-title">Agreement log</div>
      <div class="log-box" id="agreement-log"></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">MIC comparison — Lab vs AI (log₂)</div>
    <div class="chart-wrap-xl"><canvas id="micCompChart"></canvas></div>
  </div>
</div>

<!-- TAB 5: DRUG PROFILE -->
<div id="tab-profile" class="tab-panel animate">
  <div class="card">
    <div class="card-title">Drug class resistance radar — Lab vs AI</div>
    <div class="chart-wrap-xl"><canvas id="profileChart"></canvas></div>
  </div>
  <div class="card">
    <div class="card-title">Susceptibility zone analysis</div>
    <div style="background:rgba(34,197,94,0.07);border:1px solid rgba(34,197,94,0.2);border-radius:10px;padding:14px 16px;margin-bottom:12px">
      <div style="color:#22c55e;font-size:12px;font-weight:700;text-transform:uppercase;margin-bottom:10px">✓ Susceptible agents</div>
      <div id="susc-list" style="display:flex;flex-wrap:wrap;gap:6px"></div>
    </div>
    <div style="background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.2);border-radius:10px;padding:14px 16px">
      <div style="color:#f87171;font-size:12px;font-weight:700;text-transform:uppercase;margin-bottom:10px">✗ Resistant agents</div>
      <div id="resist-list" style="display:flex;flex-wrap:wrap;gap:6px"></div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Resistance intensity bars</div>
    {prof_bars}
  </div>
</div>

<!-- TAB 6: GENES & IDENTITY -->
<div id="tab-genes" class="tab-panel animate">
  <div class="grid-3" style="margin-bottom:14px">
    <div class="metric"><div class="metric-lbl">Total ARGs</div><div class="metric-val" style="color:#00c9ff">{sim['real_genes']}</div><div class="metric-sub">from CARD JSON</div></div>
    <div class="metric"><div class="metric-lbl">Drug classes</div><div class="metric-val">{sim['real_u_drugs']}</div><div class="metric-sub">unique classes</div></div>
    <div class="metric"><div class="metric-lbl">Mechanisms</div><div class="metric-val">{sim['real_u_mechs']}</div><div class="metric-sub">unique strategies</div></div>
  </div>
  <div class="card">
    <div class="card-title">Known resistance determinants</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px">{gene_tags}</div>
    <div style="font-size:11px;color:var(--muted)">* Curated reference list. Actual ARGs are in your CARD JSON file.</div>
  </div>
  <div class="card">
    <div class="card-title">Biochemical identity panel</div>
    <div class="grid-2">
      <div>
        <div class="info-row"><span class="info-key">Gram stain</span><span class="info-val">{db['gram']}</span></div>
        <div class="info-row"><span class="info-key">Shape</span><span class="info-val">{db['shape']}</span></div>
        <div class="info-row"><span class="info-key">Arrangement</span><span class="info-val">{db['arrangement']}</span></div>
        <div class="info-row"><span class="info-key">Motility</span><span class="info-val">{'Yes' if db['motility'] else 'No'}</span></div>
        <div class="info-row"><span class="info-key">Spore</span><span class="info-val">{'Yes' if db['spore'] else 'No'}</span></div>
        <div class="info-row"><span class="info-key">Capsule</span><span class="info-val">{'Present' if db['capsule'] else 'Absent'}</span></div>
      </div>
      <div>
        <div class="info-row"><span class="info-key">Oxidase</span><span class="info-val">{db['oxidase']}</span></div>
        <div class="info-row"><span class="info-key">Catalase</span><span class="info-val">{db['catalase']}</span></div>
        <div class="info-row"><span class="info-key">Lactose ferment</span><span class="info-val">{'Yes' if db['ferments_lactose'] else 'No'}</span></div>
        <div class="info-row"><span class="info-key">Selective medium</span><span class="info-val">{db['selective_media']}</span></div>
        <div class="info-row"><span class="info-key">Optimal temp</span><span class="info-val">{db['optimal_temp']}°C</span></div>
        <div class="info-row"><span class="info-key">Optimal pH</span><span class="info-val">{db['optimal_ph']}</span></div>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Associated diseases</div>
    <div style="font-size:14px;line-height:1.8;color:#94a3b8">{db['disease']}</div>
  </div>
</div>

</div><!-- /lab-shell -->

<script>
var SIM_DATA = {{
  growth_times: {growth_labels_js},
  growth_cfu:   {growth_data_js},
  ab_names:     {json.dumps(ab_names)},
  lab_mics:     {json.dumps([round(x,2) for x in lab_mics])},
  ai_mics:      {json.dumps([round(x,2) for x in ai_mics])},
  classes:      {json.dumps(classes)},
  lab_profile:  {json.dumps([round(x,2) for x in lab_profile_vals])},
  ai_profile:   {json.dumps([round(x,2) for x in ai_profile_vals])},
  mic_results:  {json.dumps(sim['mic_results'])},
  lab_mri: {sim['lab_mri']}, ai_mri: {sim['ai_mri']},
  lab_ari: {sim['lab_ari']}, ai_ari: {sim['ai_ari']},
}};

function switchTab(name, btn) {{
  document.querySelectorAll('.tab-panel').forEach(function(p){{p.classList.remove('active');}});
  document.querySelectorAll('.tab-btn').forEach(function(b){{b.classList.remove('active');}});
  document.getElementById('tab-'+name).classList.add('active');
  btn.classList.add('active');
}}

new Chart(document.getElementById('growthChart'), {{
  type:'line',
  data:{{labels:SIM_DATA.growth_times,datasets:[{{label:'CFU/mL (×10⁶)',data:SIM_DATA.growth_cfu,borderColor:'#00c9ff',backgroundColor:'rgba(0,201,255,0.06)',fill:true,tension:0.4,pointRadius:2}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#64748b',font:{{size:10}}}},grid:{{color:'rgba(255,255,255,0.04)'}}}},y:{{ticks:{{color:'#64748b',font:{{size:10}},callback:function(v){{return v+'M';}}}},grid:{{color:'rgba(255,255,255,0.04)'}},min:0}}}}}}
}});

new Chart(document.getElementById('mriChart'), {{
  type:'bar',
  data:{{labels:['MRI Score','ARI Score'],datasets:[{{label:'Lab',data:[SIM_DATA.lab_mri,SIM_DATA.lab_ari],backgroundColor:'#185FA5',borderRadius:4}},{{label:'AI',data:[SIM_DATA.ai_mri,SIM_DATA.ai_ari],backgroundColor:'rgba(0,201,255,0.3)',borderColor:'#00c9ff',borderWidth:2,borderRadius:4}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#64748b'}},grid:{{color:'rgba(255,255,255,0.04)'}}}},y:{{min:0,max:1,ticks:{{color:'#64748b'}},grid:{{color:'rgba(255,255,255,0.04)'}}}}}}}}
}});

new Chart(document.getElementById('micCompChart'), {{
  type:'bar',
  data:{{labels:SIM_DATA.ab_names,datasets:[{{label:'Lab MIC',data:SIM_DATA.lab_mics,backgroundColor:'#185FA5',borderRadius:3}},{{label:'AI MIC',data:SIM_DATA.ai_mics,backgroundColor:'rgba(0,201,255,0.3)',borderColor:'#00c9ff',borderWidth:1,borderRadius:3}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{color:'#64748b',font:{{size:10}},maxRotation:45}},grid:{{color:'rgba(255,255,255,0.04)'}}}},y:{{ticks:{{color:'#64748b',callback:function(v){{return '2^'+v;}}}},grid:{{color:'rgba(255,255,255,0.04)'}}}}}}}}
}});

new Chart(document.getElementById('profileChart'), {{
  type:'radar',
  data:{{labels:SIM_DATA.classes,datasets:[{{label:'Lab',data:SIM_DATA.lab_profile,borderColor:'#185FA5',backgroundColor:'rgba(24,95,165,0.15)',borderWidth:2}},{{label:'AI',data:SIM_DATA.ai_profile,borderColor:'#00c9ff',backgroundColor:'rgba(0,201,255,0.08)',borderWidth:2,borderDash:[5,3]}}]}},
  options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{r:{{ticks:{{color:'#64748b',backdropColor:'transparent'}},grid:{{color:'rgba(255,255,255,0.06)'}},pointLabels:{{color:'#94a3b8',font:{{size:11}}}}}}}}}}
}});

var DILUTIONS_ARR=[0.06,0.12,0.25,0.5,1,2,4,8,16,32,64,128];
function buildAbSelector(){{
  var sel=document.getElementById('ab-selector');
  SIM_DATA.mic_results.forEach(function(ab,i){{
    var btn=document.createElement('button');
    btn.textContent=ab.name;
    btn.style.cssText='padding:4px 10px;font-size:11px;border-radius:4px;cursor:pointer;border:1px solid rgba(255,255,255,0.12);background:'+(i===0?'rgba(0,201,255,0.15)':'rgba(255,255,255,0.04)')+';color:'+(i===0?'#00c9ff':'#94a3b8')+';font-family:Sora,sans-serif;';
    btn.onclick=function(){{
      document.querySelectorAll('#ab-selector button').forEach(function(b,j){{b.style.background=j===i?'rgba(0,201,255,0.15)':'rgba(255,255,255,0.04)';b.style.color=j===i?'#00c9ff':'#94a3b8';}});
      renderMicPlate(i);
    }};
    sel.appendChild(btn);
  }});
}}
function renderMicPlate(idx){{
  var ab=SIM_DATA.mic_results[idx],mic=ab.lab_mic;
  var micColIdx=DILUTIONS_ARR.indexOf(DILUTIONS_ARR.reduce(function(a,b){{return Math.abs(b-mic)<Math.abs(a-mic)?b:a;}}));
  var grid=document.getElementById('mic-plate-grid');grid.innerHTML='';
  for(var row=0;row<8;row++){{for(var col=0;col<12;col++){{
    var well=document.createElement('div');
    well.style.cssText='aspect-ratio:1;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:700;';
    if(col===micColIdx){{well.style.background='#1d4ed8';well.style.border='2px solid #60a5fa';well.style.color='#fff';well.textContent='MIC';}}
    else if(col<micColIdx){{well.style.background='rgba(239,68,68,0.6)';well.style.border='1px solid #ef4444';well.style.color='#fff';well.textContent='+';}}
    else{{well.style.background='rgba(34,197,94,0.2)';well.style.border='1px solid rgba(34,197,94,0.4)';well.style.color='#4ade80';well.textContent='−';}}
    grid.appendChild(well);
  }}}}
  document.getElementById('mic-result-val').textContent=mic;
  var iC={{'S':'#22c55e','I':'#f59e0b','R':'#ef4444'}}[ab.lab_interp];
  document.getElementById('mic-result-interp').innerHTML='<span style="color:'+iC+';font-weight:700;">' + {{'S':'Susceptible','I':'Intermediate','R':'Resistant'}}[ab.lab_interp]+'</span>';
  document.getElementById('mic-s-break').textContent=ab.S_break;
  document.getElementById('mic-r-break').textContent=ab.R_break;
  document.getElementById('mic-plate-title').textContent=ab.name+' ('+ab.class+')';
}}
function buildRiskConf(){{
  var labMRI=SIM_DATA.lab_mri,aiMRI=SIM_DATA.ai_mri;
  var conf={{'LOW':labMRI<0.15?0.70+Math.random()*0.25:0.05+Math.random()*0.10,'MODERATE':(labMRI>=0.15&&labMRI<0.35)?0.60+Math.random()*0.30:0.05+Math.random()*0.12,'HIGH':labMRI>=0.35?0.65+Math.random()*0.30:0.03+Math.random()*0.08}};
  var total=Object.values(conf).reduce(function(a,b){{return a+b;}},0);
  var el=document.getElementById('risk-conf');el.innerHTML='';
  ['LOW','MODERATE','HIGH'].forEach(function(r){{
    var pct=Math.round(conf[r]/total*100),col=r==='HIGH'?'#ef4444':r==='MODERATE'?'#f59e0b':'#22c55e';
    var isLab=(r==='{risk_lab[0]}'),isAI=(r==='{risk_ai[0]}');
    el.innerHTML+='<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;"><div style="font-size:11px;font-weight:700;min-width:70px;color:'+col+';">'+r+'</div><div style="flex:1;height:6px;background:#1e293b;border-radius:3px;overflow:hidden;"><div style="height:100%;width:'+pct+'%;background:'+col+';border-radius:3px;"></div></div><div style="font-size:11px;min-width:32px;color:#64748b;">'+pct+'%</div>'+(isLab?'<span style="background:'+col+'22;color:'+col+';font-size:9px;padding:1px 6px;border-radius:3px;border:1px solid '+col+'44;">Lab</span>':'')+(isAI?'<span style="background:rgba(0,201,255,0.1);color:#00c9ff;font-size:9px;padding:1px 6px;border-radius:3px;border:1px solid rgba(0,201,255,0.3);">AI</span>':'')+'</div>';
  }});
}}
function buildAgreementLog(){{
  var log=document.getElementById('agreement-log'),lines=[],delta=Math.abs(SIM_DATA.lab_mri-SIM_DATA.ai_mri);
  lines.push({{t:'ok',m:'[LAB] MRI: '+SIM_DATA.lab_mri+' ({risk_lab[0]})'}});
  lines.push({{t:'info',m:'[AI]  MRI: '+SIM_DATA.ai_mri+' ({risk_ai[0]})'}});
  lines.push({{t:delta<0.03?'ok':delta<0.08?'warn':'err',m:'[CMP] Δ: '+delta.toFixed(3)+' — '+(delta<0.03?'IDENTICAL':delta<0.08?'NEAR MATCH':'DIVERGED')}});
  lines.push({{t:'ok',m:'[LAB] ARI: '+SIM_DATA.lab_ari}});
  lines.push({{t:'info',m:'[AI]  ARI: '+SIM_DATA.ai_ari}});
  var matched=SIM_DATA.mic_results.filter(function(r){{return r.lab_interp===r.ai_interp;}}).length;
  lines.push({{t:matched>=Math.round(SIM_DATA.mic_results.length*0.8)?'ok':'warn',m:'[MIC] Concordance: '+matched+'/'+SIM_DATA.mic_results.length+' ('+Math.round(matched/SIM_DATA.mic_results.length*100)+'%)'}});
  SIM_DATA.mic_results.forEach(function(r){{var m=r.lab_interp===r.ai_interp;lines.push({{t:m?'ok':'warn',m:'[MIC] '+r.name+': Lab='+r.lab_mic+'('+r.lab_interp+') AI='+r.ai_mic+'('+r.ai_interp+') '+(m?'✓':'≈')}});}});
  log.innerHTML=lines.map(function(l){{var c=l.t==='ok'?'#22c55e':l.t==='info'?'#00c9ff':l.t==='warn'?'#f59e0b':'#ef4444';return '<div style="color:'+c+'">'+l.m+'</div>';}}).join('');
  log.scrollTop=log.scrollHeight;
}}
function buildSuscZone(){{
  var sl=document.getElementById('susc-list'),rl=document.getElementById('resist-list');
  SIM_DATA.mic_results.forEach(function(r){{
    var tag=document.createElement('span');
    tag.style.cssText='display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;margin:2px;';
    tag.textContent=r.name;
    if(r.lab_interp==='S'){{tag.style.background='rgba(34,197,94,0.1)';tag.style.color='#4ade80';tag.style.border='1px solid rgba(34,197,94,0.25)';sl.appendChild(tag);}}
    else if(r.lab_interp==='R'){{tag.style.background='rgba(239,68,68,0.1)';tag.style.color='#f87171';tag.style.border='1px solid rgba(239,68,68,0.25)';rl.appendChild(tag);}}
  }});
  if(!sl.children.length)sl.innerHTML='<span style="color:#64748b;font-size:12px;">None found</span>';
  if(!rl.children.length)rl.innerHTML='<span style="color:#64748b;font-size:12px;">None found</span>';
}}
buildAbSelector();renderMicPlate(0);buildRiskConf();buildAgreementLog();buildSuscZone();
</script>
</body></html>"""
    return html


def render_lab_tab():
    st.markdown("""
    <style>
    .lab-section-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;
      color:#00d4ff;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid rgba(0,212,255,.15);}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0c1f3f,#111827);border:1px solid rgba(0,212,255,.15);
                border-radius:14px;padding:20px 24px;margin-bottom:20px;position:relative;overflow:hidden;">
      <div style="position:absolute;top:0;left:0;right:0;height:2px;
                  background:linear-gradient(90deg,#00d4ff,#7c3aed,#10b981,#00d4ff);
                  background-size:300%;animation:scanBar 4s linear infinite;"></div>
      <style>@keyframes scanBar{0%{background-position:0%}100%{background-position:300%}}</style>
      <h2 style="color:#00d4ff;font-size:1.3rem;margin-bottom:4px;">🧫 Virtual Laboratory Simulation</h2>
      <p style="color:#64748b;font-size:13px;margin:0;">
        Full phenotypic simulation — culture, MIC plate, AI vs lab comparison, drug profiles and gene identity.
      </p>
    </div>
    """, unsafe_allow_html=True)

    json_files  = [f for f in os.listdir('.') if f.endswith('.json')]
    file_to_bac = _lab_scan_json_files()

    if not json_files:
        st.warning("No JSON files found. Upload CARD-format JSON files to begin.")
        return

    col_bac, col_file = st.columns([2, 2])
    with col_bac:
        st.markdown('<div class="lab-section-title">Select Organism</div>', unsafe_allow_html=True)
        all_bac_names = sorted(LAB_BACTERIA_DB.keys())
        found_bac     = sorted(set(file_to_bac.values()))
        default_bac   = found_bac[0] if found_bac else all_bac_names[0]
        selected_bac  = st.selectbox("Bacterial species", all_bac_names,
                                     index=all_bac_names.index(default_bac),
                                     label_visibility="collapsed", key="lab_bac_select")
    with col_file:
        st.markdown('<div class="lab-section-title">JSON Source File</div>', unsafe_allow_html=True)
        matching      = [f for f, b in file_to_bac.items() if b == selected_bac]
        file_opts     = matching + [f for f in json_files if f not in matching]
        selected_json = st.selectbox("JSON file", file_opts,
                                     label_visibility="collapsed", key="lab_json_select")

    st.markdown("---")

    db = LAB_BACTERIA_DB[selected_bac]
    st.markdown('<div class="lab-section-title">Lab Conditions</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: temperature        = st.slider("Temperature (°C)", 25.0, 45.0, float(db["optimal_temp"]), 0.5, key="lab_temp")
    with c2: ph                 = st.slider("pH", 5.5, 9.0, float(db["optimal_ph"]), 0.1, key="lab_ph")
    with c3: incubation_h       = st.slider("Incubation (h)", 4, 72, 18, 1, key="lab_inc")
    with c4: cfu_exp            = st.slider("Inoculum 10^x CFU/mL", 3, 8, 5, 1, key="lab_cfu")
    with c5: resistance_override = st.slider("Resistance pressure", 0.0, 1.0, 0.5, 0.05, key="lab_res",
                                              help="Simulate high-pressure selection environments")

    medium = st.selectbox("Growth medium",
        ["Mueller-Hinton Broth (MHB)", "Lysogeny Broth Agar (LBA)",
         "Columbia Blood Agar (CBA)", "Brain Heart Infusion (BHI)",
         "Löwenstein-Jensen (LJ)", "Chocolate Agar"], key="lab_medium")

    st.markdown("---")

    preset_col, run_col = st.columns([3, 1])
    with preset_col:
        preset = st.selectbox("Quick presets",
            ["— custom —", "MRSA (High resistance)", "ESBL E. coli", "XDR Acinetobacter",
             "Pan-susceptible Salmonella", "MDR M. tuberculosis", "VRE Enterococcus"],
            key="lab_preset")
    with run_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  Run Simulation", type="primary",
                            use_container_width=True, key="lab_run_btn")

    preset_map = {
        "MRSA (High resistance)":     {"bac": "Staphylococcus aureus",     "res": 0.85},
        "ESBL E. coli":               {"bac": "Escherichia coli",           "res": 0.72},
        "XDR Acinetobacter":          {"bac": "Acinetobacter baumannii",    "res": 0.92},
        "Pan-susceptible Salmonella": {"bac": "Salmonella enterica",        "res": 0.10},
        "MDR M. tuberculosis":        {"bac": "Mycobacterium tuberculosis", "res": 0.78},
        "VRE Enterococcus":           {"bac": "Enterococcus faecium",       "res": 0.80},
    }
    if preset in preset_map:
        resistance_override = preset_map[preset]["res"]
        selected_bac        = preset_map[preset]["bac"]

    if 'lab_result_html' not in st.session_state:
        st.session_state.lab_result_html = None
        st.session_state.lab_run_key     = None

    run_key = f"{selected_bac}|{selected_json}|{temperature}|{ph}|{incubation_h}|{cfu_exp}|{resistance_override}|{medium}"

    if run_btn or (st.session_state.lab_run_key == run_key and st.session_state.lab_result_html):
        if run_btn or st.session_state.lab_result_html is None:
            with st.spinner(f"Running virtual lab for {selected_bac}…"):
                sim  = _lab_simulate(selected_bac, selected_json,
                                     resistance_override, temperature, ph,
                                     incubation_h, cfu_exp, medium)
                html = _lab_build_html(sim)
                st.session_state.lab_result_html = html
                st.session_state.lab_run_key     = run_key
        components.html(st.session_state.lab_result_html, height=920, scrolling=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;background:rgba(15,23,42,0.5);
                    border:1px dashed rgba(0,212,255,.2);border-radius:14px;margin-top:10px;">
          <div style="font-size:3rem;margin-bottom:12px;">🧫</div>
          <div style="font-size:18px;color:#e2e8f0;font-weight:600;margin-bottom:8px;">Virtual Lab Ready</div>
          <div style="font-size:13px;color:#64748b;">
            Configure organism and conditions above, then click
            <strong style="color:#00d4ff;">▶ Run Simulation</strong>.
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# END OF VIRTUAL LAB MODULE
# ══════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');

* { box-sizing: border-box; }

.main { background-color: #0e1117; }
h1, h2, h3 { color: #ffffff; }

.main-header {
  background: linear-gradient(120deg, #020b18 0%, #0c1f3f 30%, #06304a 60%, #0f2a1a 100%);
  border: 1px solid rgba(0,212,255,0.2);
  padding: 2rem 1.5rem;
  border-radius: 15px;
  color: white;
  text-align: center;
  margin-bottom: 1.5rem;
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
.main-header h1 { margin:0; font-size: clamp(1.4rem, 5vw, 2.8rem); }
.main-header p { font-size: clamp(0.85rem, 3vw, 1.3rem); opacity:0.9; margin-top:10px; }

.section-explainer {
  background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(124,58,237,0.05));
  border: 1px solid rgba(0,212,255,0.15);
  border-left: 4px solid #00d4ff;
  padding: clamp(12px,3vw,18px) clamp(14px,3vw,20px);
  border-radius: 10px;
  margin-bottom: 18px;
  font-size: clamp(0.82rem,2.5vw,0.92rem);
  color: #94a3b8;
  line-height: 1.7;
}
.section-explainer .ex-title {
  color: #00d4ff;
  font-size: clamp(0.72rem,2vw,0.78rem);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.welcome-hero {
  background: linear-gradient(135deg, rgba(30,58,138,0.8) 0%, rgba(15,23,42,0.95) 100%);
  border-radius: 15px;
  padding: clamp(20px, 5vw, 45px);
  border: 1px solid rgba(59,130,246,0.4);
  margin-bottom: 30px;
  text-align: center;
  box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}
.welcome-hero h2 { color: #60a5fa !important; font-size: clamp(1.4rem, 5vw, 2.8rem) !important; font-weight: 800 !important; margin-bottom: 12px !important; }
.welcome-hero p { color: #e2e8f0 !important; font-size: clamp(0.9rem, 3vw, 1.25rem) !important; }

.alert-banner {
  background: rgba(239,68,68,0.2);
  border: 1px solid #ef4444;
  padding: 12px 15px;
  border-radius: 8px;
  color: #f87171;
  font-weight: bold;
  text-align: center;
  margin-bottom: 20px;
  animation: pulse 2s infinite;
  font-size: clamp(0.8rem, 3vw, 1rem);
  word-break: break-word;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.7} }

.metric-card {
  background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(6,30,50,0.9));
  border: 1px solid rgba(0,212,255,0.15);
  padding: clamp(12px, 3vw, 1.5rem);
  border-radius: 12px;
  text-align: center;
  transition: all 0.3s ease;
}
.metric-card:hover { border-color: rgba(0,212,255,0.4); box-shadow: 0 8px 25px rgba(0,212,255,0.1); }
.metric-label { color: #00d4ff; font-size: clamp(0.65rem, 2vw, 0.75rem); font-weight: bold; text-transform: uppercase; letter-spacing: 0.1em; }
.metric-value { color: #ffffff; font-size: clamp(1.1rem, 4vw, 1.8rem); font-weight: 700; margin-top: 0.3rem; }

.report-card {
  background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(6,30,50,0.9));
  border-left: 5px solid #00d4ff;
  padding: clamp(14px, 3vw, 20px);
  border-radius: 10px;
  margin-bottom: 25px;
}
.report-header { color: #00d4ff; font-weight: bold; text-transform: uppercase; font-size: 0.9rem; margin-bottom: 15px; border-bottom: 1px solid #334155; padding-bottom: 5px; }
.report-row { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px; padding: 8px 0; border-bottom: 1px solid #2d3748; }
.report-label { color: #94a3b8; font-weight: 500; font-size: clamp(0.8rem, 2.5vw, 1rem); }
.report-value { color: #ffffff; font-weight: 600; font-size: clamp(0.8rem, 2.5vw, 1rem); text-align: right; }

.ai-badge { background: rgba(0,212,255,0.1); color: #00d4ff; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; border: 1px solid rgba(0,212,255,0.3); }

.math-card {
  background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(6,30,50,0.9));
  border: 1px solid rgba(0,212,255,0.15);
  padding: clamp(14px, 3vw, 25px);
  border-radius: 12px;
  margin-bottom: 20px;
}
.math-card-header { color: #00d4ff; font-size: 0.85rem; font-weight: bold; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 15px; }
.annotation-box { background: rgba(15,23,42,0.6); border: 1px dashed #334155; border-radius: 8px; padding: 12px; margin-top: 10px; font-size: clamp(0.78rem, 2.5vw, 0.85rem); color: #94a3b8; }
.annotation-item { margin-bottom: 4px; }
.annotation-key { color: #60a5fa; font-weight: bold; font-family: monospace; }

.susceptibility-card { background: rgba(16,185,129,0.1); border: 1px solid #10b981; border-radius: 10px; padding: 15px; color: #10b981; font-weight: 600; word-break: break-word; }

.reasoning-box {
  background: rgba(6,30,50,0.85);
  border-radius: 12px;
  border-left: 5px solid #00d4ff;
  padding: clamp(14px, 3vw, 20px);
  color: #ffffff;
  line-height: 1.7;
  font-size: clamp(0.9rem, 2.5vw, 1.05rem);
}

.origin-badge {
  display: inline-block;
  padding: 5px 14px;
  border-radius: 20px;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.origin-clinical    { background: rgba(239,68,68,0.15);   border: 1px solid #ef4444;   color: #f87171; }
.origin-agricultural{ background: rgba(16,185,129,0.15);  border: 1px solid #10b981;   color: #34d399; }
.origin-environmental{background: rgba(59,130,246,0.15);  border: 1px solid #3b82f6;   color: #60a5fa; }
.origin-wastewater  { background: rgba(245,158,11,0.15);  border: 1px solid #f59e0b;   color: #fbbf24; }
.origin-food_production{background:rgba(167,139,250,0.15);border: 1px solid #a78bfa;   color: #c4b5fd; }

.insight-card {
  background: rgba(15,23,42,0.9);
  border: 1px solid rgba(0,212,255,0.12);
  border-radius: 12px;
  padding: clamp(14px,3vw,20px);
  margin-bottom: 14px;
}
.insight-card .ins-title { color: #00d4ff; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }
.insight-card .ins-body  { color: #94a3b8; font-size: 0.87rem; line-height: 1.65; }

[data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 10px; overflow: hidden; }

@keyframes fadeInDown { from{opacity:0;transform:translateY(-40px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeInUp   { from{opacity:0;transform:translateY(40px)}  to{opacity:1;transform:translateY(0)} }
@keyframes fadeIn     { from{opacity:0} to{opacity:1} }
@keyframes shimmer    { 0%{background-position:-200% center} 100%{background-position:200% center} }
@keyframes float      { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }

.splash-root {
  min-height: 100vh;
  background: #020b18;
  background-image:
    radial-gradient(ellipse 80% 60% at 10% 20%, rgba(0,212,255,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 60% 80% at 90% 80%, rgba(124,58,237,0.1) 0%, transparent 60%);
  font-family: 'Rajdhani', sans-serif;
  overflow-x: hidden;
}
.splash-root::before {
  content:'';
  position:fixed; inset:0;
  background-image: linear-gradient(rgba(0,212,255,0.03) 1px,transparent 1px), linear-gradient(90deg,rgba(0,212,255,0.03) 1px,transparent 1px);
  background-size:60px 60px;
  animation: gridDrift 20s linear infinite;
  pointer-events:none; z-index:0;
}
@keyframes gridDrift { 0%{transform:translate(0,0)} 100%{transform:translate(60px,60px)} }
.splash-content { position:relative; z-index:2; }

.sp-hero { padding: clamp(40px,8vw,80px) clamp(16px,5vw,60px) clamp(30px,5vw,60px); text-align:center; }
.sp-dna-ring { display:inline-block; font-size:clamp(3rem,10vw,5.5rem); animation:dnaFloat 4s ease-in-out infinite,dnaGlow 3s ease-in-out infinite; margin-bottom:20px; }
@keyframes dnaFloat { 0%,100%{transform:translateY(0) scale(1)} 50%{transform:translateY(-15px) scale(1.05)} }
@keyframes dnaGlow  { 0%,100%{filter:drop-shadow(0 0 20px rgba(0,212,255,0.5))} 50%{filter:drop-shadow(0 0 50px rgba(124,58,237,0.8)) drop-shadow(0 0 20px rgba(0,212,255,0.6))} }

.sp-title {
  font-family:'Orbitron',monospace !important;
  font-size: clamp(1.8rem,8vw,4.5rem) !important;
  font-weight:900 !important; letter-spacing:clamp(1px,2vw,4px);
  background: linear-gradient(135deg,#00d4ff 0%,#7c3aed 40%,#10b981 70%,#00d4ff 100%);
  background-size:300% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  animation: titleShimmer 5s linear infinite, titleReveal 1.2s ease forwards;
  margin-bottom:8px !important; opacity:0;
}
@keyframes titleShimmer { 0%{background-position:0% center} 100%{background-position:300% center} }
@keyframes titleReveal  { 0%{opacity:0;transform:translateY(-30px)} 100%{opacity:1;transform:translateY(0)} }

.sp-tagline { font-family:'Orbitron',monospace; font-size:clamp(0.6rem,2.5vw,0.85rem) !important; letter-spacing:clamp(2px,3vw,6px); color:#00d4ff !important; text-transform:uppercase; margin-bottom:20px !important; animation:fadeInUp 1.5s ease forwards; opacity:0; animation-delay:0.6s; }
.sp-desc { color:#94a3b8 !important; font-size:clamp(0.9rem,3vw,1.15rem) !important; line-height:1.8; max-width:700px; margin:0 auto 35px auto !important; animation:fadeInUp 1.8s ease forwards; opacity:0; animation-delay:0.9s; }

.sp-stats { display:flex; justify-content:center; gap:clamp(8px,2vw,20px); flex-wrap:wrap; margin-bottom:50px; animation:fadeInUp 2s ease forwards; opacity:0; animation-delay:1.2s; }
.sp-stat { background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.2); border-radius:16px; padding:clamp(14px,3vw,22px) clamp(16px,4vw,32px); min-width:clamp(70px,20vw,120px); transition:all 0.3s ease; }
.sp-stat:hover { transform:translateY(-5px); border-color:rgba(0,212,255,0.5); }
.sp-stat-num { font-family:'Orbitron',monospace; font-size:clamp(1.4rem,5vw,2.2rem); font-weight:900; color:#00d4ff; display:block; }
.sp-stat-lbl { font-size:clamp(0.6rem,2vw,0.7rem); color:#64748b; letter-spacing:2px; text-transform:uppercase; }

.sp-section-title { font-family:'Orbitron',monospace !important; font-size:clamp(1.2rem,4vw,1.8rem) !important; font-weight:700 !important; color:#f1f5f9 !important; text-align:center; margin-bottom:10px !important; letter-spacing:2px; }
.sp-divider { width:80px; height:3px; background:linear-gradient(90deg,#00d4ff,#7c3aed); margin:0 auto 30px auto; border-radius:3px; }

.sp-feat-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(260px,100%),1fr)); gap:16px; padding:0 clamp(16px,4vw,60px) 50px; }
.sp-feat-card { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:16px; padding:clamp(18px,3vw,28px) clamp(14px,3vw,22px); transition:all 0.35s ease; }
.sp-feat-card:hover { transform:translateY(-6px); border-color:rgba(0,212,255,0.25); }
.sp-feat-icon { font-size:clamp(1.6rem,4vw,2.2rem); display:block; margin-bottom:12px; }
.sp-feat-title { color:#00d4ff !important; font-size:clamp(0.75rem,2.5vw,0.85rem); font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
.sp-feat-desc { color:#64748b; font-size:clamp(0.82rem,2.5vw,0.88rem); line-height:1.65; }

.sp-unique-section { padding:0 clamp(16px,4vw,60px) 60px; }
.sp-compare { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr)); gap:20px; margin-bottom:30px; }
.sp-comp-card { border-radius:18px; padding:clamp(18px,3vw,30px); }
.sp-comp-old { background:rgba(239,68,68,0.04); border:1px solid rgba(239,68,68,0.18); }
.sp-comp-new { background:rgba(0,212,255,0.04); border:1px solid rgba(0,212,255,0.22); position:relative; }
.sp-comp-new::before { content:'★ EXCLUSIVE'; position:absolute; top:-13px; right:20px; background:linear-gradient(90deg,#00d4ff,#10b981); color:#020b18; font-size:0.65rem; font-weight:800; padding:4px 14px; border-radius:20px; letter-spacing:1.5px; font-family:'Orbitron',monospace; }
.sp-comp-title { font-size:clamp(0.9rem,3vw,1rem); font-weight:700; margin-bottom:16px; padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,0.06); }
.sp-comp-old .sp-comp-title { color:#f87171; }
.sp-comp-new .sp-comp-title { color:#00d4ff; }
.sp-comp-item { display:flex; gap:10px; margin-bottom:12px; color:#94a3b8; font-size:clamp(0.82rem,2.5vw,0.88rem); line-height:1.5; }
.sp-chk { flex-shrink:0; }

.sp-pillars { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(200px,100%),1fr)); gap:16px; }
.sp-pillar { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:14px; padding:clamp(18px,3vw,26px) clamp(14px,3vw,20px); text-align:center; transition:all 0.3s ease; position:relative; overflow:hidden; }
.sp-pillar::before { content:''; position:absolute; top:0;left:0;right:0; height:3px; }
.sp-pillar.c1::before{background:linear-gradient(90deg,#00d4ff,#10b981)} .sp-pillar.c2::before{background:linear-gradient(90deg,#7c3aed,#a78bfa)} .sp-pillar.c3::before{background:linear-gradient(90deg,#f59e0b,#fbbf24)} .sp-pillar.c4::before{background:linear-gradient(90deg,#f43f5e,#fb7185)} .sp-pillar.c5::before{background:linear-gradient(90deg,#10b981,#34d399)} .sp-pillar.c6::before{background:linear-gradient(90deg,#06b6d4,#67e8f9)} .sp-pillar.c7::before{background:linear-gradient(90deg,#f59e0b,#10b981)} .sp-pillar.c8::before{background:linear-gradient(90deg,#a78bfa,#00d4ff)}
.sp-pillar:hover{transform:translateY(-6px);border-color:rgba(0,212,255,0.3)}
.sp-pillar-icon{font-size:clamp(1.6rem,4vw,2.3rem);display:block;margin-bottom:10px}
.sp-pillar-title{color:#f1f5f9 !important;font-size:clamp(0.75rem,2.5vw,0.82rem);font-weight:700;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px}
.sp-pillar-desc{color:#475569;font-size:clamp(0.78rem,2.5vw,0.82rem);line-height:1.6}

.sp-team-section{padding:0 clamp(16px,4vw,60px) 60px}
.sp-team-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:16px;margin-bottom:30px}
.sp-team-card{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:18px;padding:clamp(20px,3vw,32px) clamp(14px,3vw,22px);text-align:center;transition:all 0.35s ease;position:relative;overflow:hidden}
.sp-team-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#00d4ff,#7c3aed,#10b981);background-size:200% auto;animation:titleShimmer 4s linear infinite}
.sp-team-card:hover{transform:translateY(-8px);border-color:rgba(0,212,255,0.3)}
.sp-avatar{width:clamp(56px,12vw,80px);height:clamp(56px,12vw,80px);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:clamp(1.4rem,5vw,2.2rem);margin:0 auto 14px auto;border:2px solid rgba(0,212,255,0.3)}
.sp-tname{color:#f1f5f9 !important;font-size:clamp(0.95rem,3vw,1.1rem);font-weight:700;margin-bottom:4px !important}
.sp-trole{color:#00d4ff !important;font-size:clamp(0.62rem,2vw,0.72rem);text-transform:uppercase;letter-spacing:2px;margin-bottom:10px !important;font-family:'Orbitron',monospace}
.sp-tdesc{color:#475569;font-size:clamp(0.78rem,2.5vw,0.82rem);line-height:1.6;margin-bottom:12px}
.sp-ttags{display:flex;flex-wrap:wrap;gap:5px;justify-content:center}
.sp-ttag{background:rgba(0,212,255,0.07);border:1px solid rgba(0,212,255,0.2);color:#67e8f9;padding:3px 9px;border-radius:20px;font-size:clamp(0.62rem,2vw,0.68rem)}

.sp-mission-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr));gap:16px;margin-top:24px}
.sp-mission-card{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:clamp(18px,3vw,28px) clamp(14px,3vw,22px);text-align:center}
.sp-mission-title{color:#00d4ff !important;font-size:clamp(0.72rem,2.5vw,0.8rem);font-weight:700;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px !important;font-family:'Orbitron',monospace}
.sp-mission-text{color:#64748b;font-size:clamp(0.82rem,2.5vw,0.85rem);line-height:1.7}

.sp-cta-section{padding:clamp(30px,5vw,60px) clamp(16px,5vw,60px);text-align:center;background:linear-gradient(180deg,transparent,rgba(0,212,255,0.04),transparent);border-top:1px solid rgba(0,212,255,0.08)}
.sp-cta-label{font-family:'Orbitron',monospace;font-size:clamp(0.65rem,2vw,0.8rem);letter-spacing:5px;text-transform:uppercase;color:#475569 !important;margin-bottom:16px !important}
.sp-cta-headline{font-size:clamp(1.4rem,5vw,2.4rem) !important;font-weight:700 !important;color:#f1f5f9 !important;margin-bottom:10px !important;line-height:1.3}
.sp-cta-sub{color:#64748b !important;font-size:clamp(0.85rem,3vw,1rem) !important;margin-bottom:30px !important}

@media (max-width: 640px) {
  .report-row { flex-direction: column; gap: 2px; }
  .report-value { text-align: left; }
  .sp-comp-new::before { font-size: 0.55rem; padding: 3px 10px; right: 10px; }
  .sp-stats { justify-content: center; }
  .sp-stat { flex: 1 1 calc(50% - 8px); min-width: 0; }
}
@media (max-width: 480px) {
  .sp-title { letter-spacing: 1px !important; }
  .sp-tagline { letter-spacing: 2px !important; }
}
@media (max-width: 768px) {
  [data-baseweb="tab"] { padding: 8px 6px !important; }
  [data-baseweb="tab"] p { font-size: 0.75rem !important; }
}
</style>
""", unsafe_allow_html=True)


def section_explainer(icon, title, body, position="above"):
    st.markdown(f"""
    <div class="section-explainer">
      <div class="ex-title">{icon} {title}</div>
      {body}
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def get_bacteria_info(file_name):
    name = file_name.lower()
    info = {"gram": "Unknown", "disease": "Various opportunistic infections"}
    if any(k in name for k in ["ecoli","escherichia","shigella"]):
        info = {"gram": "Negative (-)", "disease": "Gastroenteritis, UTI, Sepsis"}
    elif "salmonella" in name:
        info = {"gram": "Negative (-)", "disease": "Salmonellosis, Typhoid Fever"}
    elif any(k in name for k in ["klebsiella","enterobacter","citrobacter","serratia"]):
        info = {"gram": "Negative (-)", "disease": "Pneumonia, UTI, Bloodstream infections"}
    elif "pseudomonas" in name:
        info = {"gram": "Negative (-)", "disease": "Cystic fibrosis lung infections, Burn wound infections"}
    elif "acinetobacter" in name:
        info = {"gram": "Negative (-)", "disease": "Nosocomial pneumonia, Bacteremia"}
    elif "vibrio" in name:
        info = {"gram": "Negative (-)", "disease": "Cholera, Vibriosis"}
    elif any(k in name for k in ["proteus","morganella","providencia"]):
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
    clinical_keys = [
        "ecoli","escherichia","staphylococcus","salmonella","klebsiella",
        "streptococcus","enterococcus","acinetobacter","haemophilus",
        "neisseria","listeria","enterobacter","citrobacter","serratia",
        "campylobacter","helicobacter","corynebacterium","clostridium",
        "morganella","providencia","proteus","vibrio"
    ]
    environmental_keys = ["pseudomonas","burkholderia","stenotrophomonas","chromobacterium"]
    soil_keys          = ["bacillus","streptomyces","arthrobacter","nocardia"]
    marine_keys        = ["vibrio","photobacterium","alteromonas","shewanella"]
    acid_fast_keys     = ["mycobacterium","nocardia"]
    if any(k in name for k in acid_fast_keys):    return "Acid-Fast / Clinical"
    if any(k in name for k in marine_keys):        return "Marine / Aquatic"
    if any(k in name for k in soil_keys):          return "Soil / Environmental"
    if any(k in name for k in environmental_keys): return "Environmental"
    if any(k in name for k in clinical_keys):      return "Clinical"
    if any(k in name for k in ["hosp","clinical","patient","ward","icu","nicu","hospital"]):
        return "Clinical"
    if any(k in name for k in ["agri","farm","livestock","poultry","cattle","swine","pig"]):
        return "Agricultural"
    if any(k in name for k in ["soil","sediment","compost","rhizo","environ"]):
        return "Environmental"
    if any(k in name for k in ["waste","sewage","effluent","wwater","wwtp"]):
        return "Wastewater"
    if any(k in name for k in ["food","meat","dairy","seafood","produce"]):
        return "Food Production"
    if any(k in name for k in ["marine","ocean","river","lake","aquatic","water"]):
        return "Marine / Aquatic"
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_text = json.dumps(data).lower()
        hits = {
            "Clinical":     sum([all_text.count(k) for k in ["hospital","clinical","nosocomial","bloodstream","patient","icu"]]),
            "Agricultural": sum([all_text.count(k) for k in ["livestock","farm","veterinary","poultry","swine","cattle"]]),
            "Environmental":sum([all_text.count(k) for k in ["soil","sediment","environmental","rhizosphere"]]),
            "Wastewater":   sum([all_text.count(k) for k in ["wastewater","sewage","effluent","wwtp"]]),
            "Food Production":sum([all_text.count(k) for k in ["food","meat","dairy","produce"]]),
        }
        best = max(hits, key=hits.get)
        if hits[best] > 0:
            return best
    except Exception:
        pass
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        gc = len(data)
        if gc >= 70:   return "Clinical (High Burden)"
        elif gc >= 40: return "Wastewater / Mixed"
        elif gc >= 20: return "Agricultural"
        elif gc >= 8:  return "Environmental"
        else:          return "Environmental (Low Burden)"
    except Exception:
        pass
    return "Unknown"


@st.cache_data
def extract_data(file_name):
    with open(file_name,'r',encoding='utf-8') as f:
        data = json.load(f)
    drug, mech, records = [], [], []
    habitat = get_habitat(file_name)
    for k in data:
        try:
            inner = list(data[k].values())[0]
            gene_name = inner.get("ARO_name",k).upper()
            d, m = [], []
            for c in inner.get("ARO_category",{}).values():
                cname = c.get("category_aro_class_name","").lower()
                val = c.get("category_aro_name","").title()
                if "drug" in cname: d.append(val)
                elif "mechanism" in cname: m.append(val)
            drug.extend(d); mech.extend(m)
            records.append((gene_name,", ".join(set(d)),", ".join(set(m)),habitat.title()))
        except: continue
    genes = len(data)
    mri = (len(set(drug))+len(set(mech)))/(len(drug)+len(mech)+1)
    ari = len(set(mech))/(genes+1)
    return genes, drug, mech, mri, ari, records


def get_level(mri):
    if mri < 0.15: return "LOW","🟢"
    elif mri < 0.35: return "MODERATE","🟡"
    else: return "HIGH","🔴"


def get_risk_reason(level, u_drugs, u_mechs):
    if "HIGH" in level:
        return f"This pathogen's elevated MRI score reflects a sophisticated and redundant resistance architecture. Deploying {u_mechs} distinct biological mechanisms against {u_drugs} drug classes, the organism exhibits the capacity to dynamically bypass standard frontline therapeutics."
    elif "MODERATE" in level:
        return f"This strain demonstrates a clinically significant resistance burden. Resistance determinants spanning {u_drugs} drug classes have been identified, necessitating careful antibiogram-guided therapy selection."
    else:
        return f"The calculated MRI indicates a restricted resistance profile. With resistance distributed across {u_drugs} drug classes via {u_mechs} mechanisms, standard empirical treatment protocols remain viable."


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


def plot_full_dashboard(drug, mech, mri, genes, records, name):
    level, icon = get_level(mri)
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#0e1117')
    color = "red" if level=="HIGH" else "orange" if level=="MODERATE" else "green"
    for a in ax.flat:
        a.set_facecolor('#0e1117'); a.tick_params(colors='white'); a.title.set_color('white')
    drug_c = Counter(drug).most_common(8)
    if drug_c:
        ax[0,0].pie([v for k,v in drug_c],labels=[k[:15]+".." for k,v in drug_c],autopct='%1.1f%%',textprops={'color':"w"})
    ax[0,0].set_title("Drug Classes Resisted")
    mech_c = Counter(mech)
    if mech_c:
        ax[0,1].bar([k[:15]+".." for k in mech_c.keys()],mech_c.values(),color=color)
        ax[0,1].tick_params(axis='x',rotation=35)
    ax[0,1].set_title("Mechanisms Deployed")
    theta = np.linspace(0,np.pi,100)
    ax[0,2].plot(np.cos(theta),np.sin(theta),color='gray')
    ang = mri*np.pi
    ax[0,2].plot([0,np.cos(ang)],[0,np.sin(ang)],color=color,linewidth=5)
    ax[0,2].axis('off'); ax[0,2].set_title(f"MRI Indicator: {round(mri,3)} ({level})")
    gene_list = [r[0] for r in records]
    gene_c = Counter(gene_list).most_common(5)
    if gene_c:
        ax[1,0].bar([k[:15]+".." for k,v in gene_c],[v for k,v in gene_c],color='#87CEEB')
        ax[1,0].tick_params(axis='x',rotation=35)
    ax[1,0].set_title("Top Gene Frequency")
    ax[1,1].bar(["Total Genes"],[genes],color='#2E86C1'); ax[1,1].set_title("Overall Gene Count")
    ax[1,2].bar(["Unique Drugs","Unique Mechs"],[len(set(drug)),len(set(mech))],color=["#9B59B6","#E67E22"]); ax[1,2].set_title("Diversity Comparison")
    fig.tight_layout()
    return fig


def generate_network_html(records, organism_name, color):
    net = Network(height='500px',width='100%',bgcolor='#222222',font_color='white',cdn_resources="in_line",select_menu=True,filter_menu=True)
    net.add_node("HUB",label=organism_name,color=color,size=30)
    for g,d,m,h in records:
        net.add_node(g,label=g[:10],color="#87CEEB",size=15); net.add_edge("HUB",g,color="#ffffff")
        if m:
            for mech_item in set(m.split(", ")):
                if not mech_item: continue
                net.add_node(mech_item,label=mech_item[:10],color="#FFA500",size=10,shape="box"); net.add_edge(g,mech_item,color="#aaaaaa")
    net.barnes_hut(gravity=-5000)
    html_path = "temp_network.html"
    with open(html_path,"w",encoding="utf-8") as f:
        f.write(net.generate_html())
    return html_path


def plot_3d_pca_plotly(current_file):
    X, files, risk_levels = [], [], []
    for f in os.listdir('.'):
        if f.endswith('.json'):
            try:
                g,d,m,mr,ar,_ = extract_data(f)
                X.append([mr,len(set(m)),len(set(d))]); files.append(f)
                lv,_ = get_level(mr)
                risk_levels.append("TARGET 🎯" if f==current_file else lv)
            except: continue
    if len(X)<3:
        st.warning("Insufficient data points for 3D PCA rendering.")
        return
    pca = PCA(n_components=3).fit_transform(X)
    df_pca = pd.DataFrame(pca,columns=['Overall Resistance (PC1)','Mechanism Diversity (PC2)','Genetic Density (PC3)'])
    df_pca['Genome']=files; df_pca['Risk Category']=risk_levels
    color_map={"HIGH":"red","MODERATE":"orange","LOW":"green","TARGET 🎯":"gold"}
    fig=px.scatter_3d(df_pca,x='Overall Resistance (PC1)',y='Mechanism Diversity (PC2)',z='Genetic Density (PC3)',color='Risk Category',hover_name='Genome',color_discrete_map=color_map,opacity=0.8,size_max=10)
    fig.update_layout(margin=dict(l=0,r=0,b=0,t=0),paper_bgcolor='#0e1117',font_color='white',scene=dict(xaxis=dict(backgroundcolor="#0e1117",gridcolor="gray"),yaxis=dict(backgroundcolor="#0e1117",gridcolor="gray"),zaxis=dict(backgroundcolor="#0e1117",gridcolor="gray")))
    st.plotly_chart(fig,use_container_width=True)


def build_drug_class_profile(drug_list):
    profile = {cls: 0 for cls in DRUG_CLASS_KEYWORDS}
    for d in drug_list:
        mapped = map_drug_to_standard_class(d)
        if mapped:
            profile[mapped] += 1
    return profile


def normalize_profile(profile, total):
    if total == 0:
        return {k: 0.0 for k in profile}
    return {k: min(v / max(total * 0.1, 1), 1.0) for k, v in profile.items()}


def plot_origin_comparison_heatmap(sample_profile_norm, selected_origin_key, organism_name):
    drug_classes = list(DRUG_CLASS_KEYWORDS.keys())
    origins = list(REFERENCE_BENCHMARKS.keys())
    display_names = {"clinical":"🏥 Clinical","agricultural":"🌾 Agricultural","environmental":"🌿 Environmental","wastewater":"💧 Wastewater","food_production":"🍖 Food Prod."}
    matrix_data = []
    row_labels   = []
    for orig in origins:
        bench = REFERENCE_BENCHMARKS[orig]
        matrix_data.append([bench.get(cls,0) for cls in drug_classes])
        row_labels.append(display_names.get(orig,orig))
    matrix_data.append([sample_profile_norm.get(cls,0) for cls in drug_classes])
    row_labels.append(f"🎯 {organism_name[:18]}")
    matrix = np.array(matrix_data)
    fig, axes = plt.subplots(1,2,figsize=(18,7),gridspec_kw={'width_ratios':[3,1]})
    fig.patch.set_facecolor('#0e1117')
    ax = axes[0]; ax.set_facecolor('#0e1117')
    im = ax.imshow(matrix,cmap='RdYlGn_r',aspect='auto',vmin=0,vmax=1)
    ax.set_xticks(range(len(drug_classes))); ax.set_xticklabels(drug_classes,rotation=40,ha='right',color='white',fontsize=9)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels,color='white',fontsize=10)
    ax.set_title(f"Comparative Resistance Heatmap — {organism_name}",color='white',fontsize=13,pad=12)
    for j in range(len(drug_classes)):
        ax.add_patch(plt.Rectangle((j-0.5,len(origins)-0.5),1,1,fill=False,edgecolor='#00d4ff',linewidth=2))
    for i in range(len(row_labels)):
        for j in range(len(drug_classes)):
            val = matrix[i,j]
            tc = 'black' if 0.3<val<0.8 else 'white'
            ax.text(j,i,f"{val:.2f}",ha='center',va='center',color=tc,fontsize=7.5,fontweight='bold')
    cbar = plt.colorbar(im,ax=ax,fraction=0.02,pad=0.02)
    cbar.set_label('Resistance Intensity',color='white',fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(),color='white')
    ax2 = axes[1]; ax2.set_facecolor('#0e1117')
    selected_bench = REFERENCE_BENCHMARKS[selected_origin_key]
    deltas = [sample_profile_norm.get(cls,0)-selected_bench.get(cls,0) for cls in drug_classes]
    ax2.barh(range(len(drug_classes)),deltas,color=['#ef4444' if d>0 else '#10b981' for d in deltas],alpha=0.85)
    ax2.set_yticks(range(len(drug_classes))); ax2.set_yticklabels(drug_classes,color='white',fontsize=9)
    ax2.axvline(0,color='#94a3b8',linewidth=1.2,linestyle='--')
    ax2.set_title(f"Δ vs {display_names.get(selected_origin_key,'')}\nReference",color='white',fontsize=11,pad=10)
    ax2.tick_params(colors='white'); ax2.set_xlabel("Sample − Reference",color='#94a3b8',fontsize=9)
    for spine in ax2.spines.values(): spine.set_edgecolor('#334155')
    plt.tight_layout()
    return fig


def plot_radar_origin_comparison(sample_profile_norm, organism_name):
    drug_classes = list(DRUG_CLASS_KEYWORDS.keys())
    display_names = {"clinical":"Clinical","agricultural":"Agricultural","environmental":"Environmental","wastewater":"Wastewater","food_production":"Food Prod."}
    palette = {"clinical":"#ef4444","agricultural":"#10b981","environmental":"#3b82f6","wastewater":"#f59e0b","food_production":"#a78bfa"}
    fig = go.Figure()
    for orig, bench in REFERENCE_BENCHMARKS.items():
        vals = [bench.get(cls,0) for cls in drug_classes]+[[bench.get(cls,0) for cls in drug_classes][0]]
        fig.add_trace(go.Scatterpolar(r=vals,theta=drug_classes+[drug_classes[0]],mode='lines',name=display_names[orig],line=dict(color=palette[orig],width=1.5,dash='dot'),opacity=0.6))
    sv = [sample_profile_norm.get(cls,0) for cls in drug_classes]+[[sample_profile_norm.get(cls,0) for cls in drug_classes][0]]
    fig.add_trace(go.Scatterpolar(r=sv,theta=drug_classes+[drug_classes[0]],mode='lines+markers',name=f"🎯 {organism_name[:20]}",line=dict(color='#00d4ff',width=3),marker=dict(size=6,color='#00d4ff'),fill='toself',fillcolor='rgba(0,212,255,0.08)'))
    fig.update_layout(polar=dict(bgcolor='#0e1117',angularaxis=dict(tickcolor='white',color='white',linecolor='#334155'),radialaxis=dict(visible=True,range=[0,1],tickcolor='white',color='#94a3b8',gridcolor='#1e293b',linecolor='#334155')),paper_bgcolor='#0e1117',plot_bgcolor='#0e1117',font_color='white',legend=dict(bgcolor='rgba(15,23,42,0.8)',bordercolor='#334155',borderwidth=1),margin=dict(l=60,r=60,t=40,b=40),height=480)
    return fig


def compute_origin_affinity_scores(sample_profile_norm, genes, mri):
    drug_classes = list(DRUG_CLASS_KEYWORDS.keys())
    scores = {}
    for orig, bench in REFERENCE_BENCHMARKS.items():
        vec_s = np.array([sample_profile_norm.get(cls,0) for cls in drug_classes])
        vec_b = np.array([bench.get(cls,0) for cls in drug_classes])
        dist  = np.linalg.norm(vec_s - vec_b)
        mri_diff  = abs(mri - bench.get("avg_mri",0.3))
        gene_diff = abs(genes - bench.get("avg_genes",50)) / max(bench.get("avg_genes",50),1)
        scores[orig] = dist*0.5 + mri_diff*0.3 + gene_diff*0.2
    max_s = max(scores.values()) or 1
    return {k: round((1-v/max_s)*100,1) for k,v in scores.items()}


def generate_origin_insights(sample_profile_norm, selected_origin_key, affinity_scores, organism_name, mri, genes):
    drug_classes = list(DRUG_CLASS_KEYWORDS.keys())
    origin_display = {"clinical":"Clinical/Hospital","agricultural":"Agricultural/Livestock","environmental":"Environmental/Soil","wastewater":"Wastewater/Sewage","food_production":"Food Production"}
    bench = REFERENCE_BENCHMARKS[selected_origin_key]
    high_excess, high_deficit = [], []
    for cls in drug_classes:
        delta = sample_profile_norm.get(cls,0) - bench.get(cls,0)
        if delta > 0.2:    high_excess.append((cls,delta))
        elif delta < -0.2: high_deficit.append((cls,abs(delta)))
    high_excess.sort(key=lambda x:-x[1]); high_deficit.sort(key=lambda x:-x[1])
    best_origin = max(affinity_scores, key=affinity_scores.get)
    return {
        "best_match": (origin_display.get(best_origin,best_origin), affinity_scores[best_origin]),
        "selected_match": (origin_display.get(selected_origin_key,selected_origin_key), affinity_scores[selected_origin_key]),
        "excess_classes": high_excess[:3],
        "deficit_classes": high_deficit[:3],
        "mri_vs_bench": mri - bench.get("avg_mri",0.3),
        "gene_vs_bench": genes - bench.get("avg_genes",50)
    }


def create_advanced_pdf_report(bac_name, genes, drug, mech, mri, ari, level, icon,
                                records, dashboard_fig, bac_info, habitat,
                                ai_pred_text, ai_conf_text, origin_label="Unknown",
                                affinity_scores=None, drug_profile_norm=None,
                                selected_origin_key="clinical", heatmap_fig=None):
    pdf_file = f"{bac_name.replace('.json','')}_Detailed_Report.pdf"
    doc = SimpleDocTemplate(pdf_file,pagesize=letter,rightMargin=30,leftMargin=30,topMargin=30,bottomMargin=30)
    styles = getSampleStyleSheet()
    title_style  = ParagraphStyle(name='TitleStyle',parent=styles['Heading1'],fontSize=18,spaceAfter=15,textColor=colors.HexColor('#1E3A8A'))
    h2_style     = ParagraphStyle(name='H2',parent=styles['Heading2'],fontSize=14,spaceBefore=15,spaceAfter=8,textColor=colors.HexColor('#2E86C1'))
    h3_style     = ParagraphStyle(name='H3',parent=styles['Heading3'],fontSize=12,spaceBefore=10,spaceAfter=6,textColor=colors.HexColor('#1a7a3a'))
    normal_style = styles['Normal']
    u_drugs, u_mechs = len(set(drug)), len(set(mech))
    t_drugs, t_mechs = len(drug), len(mech)
    elements = []

    elements.append(Paragraph(f"AI-MRI Report: {bac_name.replace('.json','')}", title_style))
    elements.append(Paragraph("Executive Summary & AI Analysis", h2_style))
    summary_text = (
        f"<b>Pathogen File:</b> {bac_name}<br/>"
        f"<b>Gram Stain:</b> {bac_info['gram']}<br/>"
        f"<b>Associated Pathology:</b> {bac_info['disease']}<br/>"
        f"<b>Ecological Habitat:</b> {habitat}<br/>"
        f"<b>Genomic Origin Label:</b> {origin_label}<br/>"
        f"<b>Total Resistance Genes:</b> {genes}<br/>"
        f"<b>Drug Classes Resisted:</b> {u_drugs}<br/>"
        f"<b>Resistance Mechanisms:</b> {u_mechs}<br/>"
        f"<b>MRI Score:</b> {round(mri,3)} ({level})<br/>"
        f"<b>ARI Score:</b> {round(ari,3)}<br/><br/>"
        f"<b>ML Prediction:</b> {ai_pred_text} Risk<br/>"
        f"<b>Confidence:</b> {ai_conf_text}"
    )
    elements.append(Paragraph(summary_text, normal_style))
    elements.append(Spacer(1,15))
    elements.append(Paragraph("Metric Definitions & Clinical Significance", h2_style))
    elements.append(Paragraph("<b>MRI:</b> Consolidates resistance breadth and depth into a single normalized score.<br/><b>ARI:</b> Measures resistance density — efficiency of gene-to-mechanism conversion.", normal_style))
    elements.append(Paragraph("Mathematical Derivations", h2_style))
    elements.append(Paragraph(f"<b>MRI:</b> ({u_drugs}+{u_mechs})/({t_drugs}+{t_mechs}+1) = <b>{round(mri,3)}</b><br/><b>ARI:</b> {u_mechs}/({genes}+1) = <b>{round(ari,3)}</b>", normal_style))
    elements.append(Paragraph("Risk Assessment Interpretation", h2_style))
    elements.append(Paragraph(get_risk_reason(level,u_drugs,u_mechs), normal_style))

    elements.append(Paragraph("Systems Analysis Dashboard", h2_style))
    buf = io.BytesIO()
    dashboard_fig.patch.set_facecolor('white')
    for ax_item in dashboard_fig.axes:
        ax_item.set_facecolor('white'); ax_item.tick_params(colors='black'); ax_item.title.set_color('black')
        for text in ax_item.texts: text.set_color('black')
    dashboard_fig.savefig(buf,format='png',bbox_inches='tight',dpi=150)
    buf.seek(0)
    elements.append(RLImage(buf,width=7.5*inch,height=5*inch))
    elements.append(PageBreak())

    elements.append(Paragraph("Genomic Origin Analysis", h2_style))
    origin_display_names = {"clinical":"Clinical / Hospital","agricultural":"Agricultural / Livestock","environmental":"Environmental / Soil","wastewater":"Wastewater / Sewage","food_production":"Food Production"}
    elements.append(Paragraph(f"<b>Assigned Origin Label:</b> {origin_label}<br/><b>Reference Population:</b> {origin_display_names.get(selected_origin_key,selected_origin_key)}", normal_style))
    elements.append(Spacer(1,10))

    if affinity_scores:
        elements.append(Paragraph("Origin Affinity Scores (0–100)", h3_style))
        aff_table_data = [["Origin Population","Affinity Score (/100)","Interpretation"]]
        for ok, score in affinity_scores.items():
            disp  = origin_display_names.get(ok,ok)
            interp = "Strong match" if score>=70 else "Moderate match" if score>=50 else "Weak match" if score>=30 else "Dissimilar"
            is_best = score == max(affinity_scores.values())
            aff_table_data.append([f"{'★ BEST — ' if is_best else ''}{disp}", f"{score}/100", interp])
        aff_table = Table(aff_table_data,colWidths=[2.8*inch,1.8*inch,2.2*inch])
        aff_table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2E86C1')),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('ALIGN',(0,0),(-1,-1),'LEFT'),('GRID',(0,0),(-1,-1),0.5,colors.grey),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F0F8FF'),colors.HexColor('#FFFFFF')])]))
        elements.append(aff_table)
        elements.append(Spacer(1,15))

    if drug_profile_norm and selected_origin_key:
        elements.append(Paragraph(f"Drug-Class Resistance vs {origin_display_names.get(selected_origin_key,'')} Reference", h3_style))
        bench_ref = REFERENCE_BENCHMARKS[selected_origin_key]
        drug_table_data = [["Drug Class","Sample","Reference","Delta","Status"]]
        for cls in DRUG_CLASS_KEYWORDS.keys():
            sv = round(drug_profile_norm.get(cls,0),3)
            rv = round(bench_ref.get(cls,0),3)
            dv = round(sv-rv,3)
            status = "⬆ ELEVATED" if dv>0.2 else "⬇ LOWER" if dv<-0.2 else "≈ Similar"
            drug_table_data.append([cls,str(sv),str(rv),f"{'+' if dv>=0 else ''}{dv}",status])
        drug_table = Table(drug_table_data,colWidths=[1.6*inch,1.3*inch,1.7*inch,1.0*inch,1.2*inch])
        drug_table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1a7a3a')),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('ALIGN',(0,0),(-1,-1),'CENTER'),('GRID',(0,0),(-1,-1),0.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),8)]))
        elements.append(drug_table)
        elements.append(Spacer(1,15))

    if heatmap_fig is not None:
        elements.append(Paragraph("Comparative Resistance Heatmap", h3_style))
        heatmap_buf = io.BytesIO()
        heatmap_fig.patch.set_facecolor('white')
        for ax_item in heatmap_fig.axes:
            ax_item.set_facecolor('white'); ax_item.tick_params(colors='black')
            if ax_item.title: ax_item.title.set_color('black')
            ax_item.xaxis.label.set_color('black')
            for spine in ax_item.spines.values(): spine.set_edgecolor('black')
        heatmap_fig.savefig(heatmap_buf,format='png',bbox_inches='tight',dpi=150,facecolor='white')
        heatmap_buf.seek(0)
        elements.append(RLImage(heatmap_buf,width=7.5*inch,height=3.5*inch))
    elements.append(PageBreak())

    elements.append(Paragraph("Complete Gene Resistance Ledger", h2_style))
    table_data = [["Gene Name","Drug Classes Resisted","Mechanisms Deployed","Habitat"]]
    for g,d,m,h in records:
        table_data.append([Paragraph(g,normal_style),Paragraph(d,normal_style),Paragraph(m,normal_style),Paragraph(h,normal_style)])
    t = Table(table_data,colWidths=[1.2*inch,2.2*inch,2.2*inch,0.9*inch])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#2E86C1')),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),('ALIGN',(0,0),(-1,-1),'LEFT'),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('BOTTOMPADDING',(0,0),(-1,0),12),('BACKGROUND',(0,1),(-1,-1),colors.HexColor('#F8F9F9')),('GRID',(0,0),(-1,-1),1,colors.black),('VALIGN',(0,0),(-1,-1),'TOP')]))
    elements.append(t)
    doc.build(elements)
    return pdf_file


# ==========================================
# SPLASH SCREEN
# ==========================================
if 'show_splash' not in st.session_state:
    st.session_state.show_splash = True

if st.session_state.show_splash:
    st.markdown("""
    <style>
    #MainMenu{visibility:hidden} footer{visibility:hidden}
    [data-testid="stSidebar"]{display:none !important}
    .block-container{padding:0 !important;max-width:100% !important}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="splash-root">
    <div class="splash-content">
    <div class="sp-hero">
      <span class="sp-dna-ring">🧬</span>
      <h1 class="sp-title">AI-MRI HUB</h1>
      <p class="sp-tagline">Antimicrobial Resistance Intelligence Platform</p>
      <p class="sp-desc">The world's most advanced quantitative framework for decoding antibiotic resistance genes. We transform raw genomic data into actionable clinical intelligence — instantly, accurately, and at scale.</p>
      <div class="sp-stats">
        <div class="sp-stat"><span class="sp-stat-num">2</span><span class="sp-stat-lbl">Novel Indices</span></div>
        <div class="sp-stat"><span class="sp-stat-num">10</span><span class="sp-stat-lbl">Modules</span></div>
        <div class="sp-stat"><span class="sp-stat-num">100+</span><span class="sp-stat-lbl">Genomes</span></div>
        <div class="sp-stat"><span class="sp-stat-num">7</span><span class="sp-stat-lbl">Researchers</span></div>
      </div>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0 clamp(16px,4vw,60px);margin-bottom:10px;background:#020b18;">
      <div class="sp-section-title">Platform Features</div>
      <div class="sp-divider"></div>
    </div>
    <div class="sp-feat-grid" style="background:#020b18;">
      <div class="sp-feat-card"><span class="sp-feat-icon">🦠</span><div class="sp-feat-title">Genomic ARG Profiling</div><div class="sp-feat-desc">Deep extraction and classification of all Antibiotic Resistance Genes from CARD-format data.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">📐</span><div class="sp-feat-title">MRI &amp; ARI Calculation</div><div class="sp-feat-desc">Proprietary indices transform complex gene counts into a single, instantly interpretable risk score.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">🤖</span><div class="sp-feat-title">Random Forest AI Prediction</div><div class="sp-feat-desc">Machine learning classifies any pathogen as LOW, MODERATE, or HIGH risk with full confidence scores.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">🕸️</span><div class="sp-feat-title">Interactive Gene Network</div><div class="sp-feat-desc">Visualize the full resistance topology as a live, draggable network graph.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">🌌</span><div class="sp-feat-title">3D Landscape PCA</div><div class="sp-feat-desc">3-dimensional scatter map comparing your target genome against every pathogen in the database.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">🩺</span><div class="sp-feat-title">Clinical Susceptibility Zones</div><div class="sp-feat-desc">Automatically identifies drug classes with zero resistance markers.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">💬</span><div class="sp-feat-title">J.A.R.V.I.S. Bio-AI Chat</div><div class="sp-feat-desc">Gemini-powered AI with full genomic context automatically loaded.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">🗺️</span><div class="sp-feat-title">Origin Labeling &amp; Heatmaps</div><div class="sp-feat-desc">Label isolates by source and compare resistance profiles against curated environmental benchmarks.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">📄</span><div class="sp-feat-title">Master PDF Export</div><div class="sp-feat-desc">Generate a comprehensive publication-ready PDF report with origin heatmap, affinity scores, and gene ledgers.</div></div>
      <div class="sp-feat-card"><span class="sp-feat-icon">🧫</span><div class="sp-feat-title">Virtual Lab Simulation</div><div class="sp-feat-desc">Full phenotypic simulation — culture curves, 96-well MIC plates, Gram stain microscopy, and AI vs lab comparison for 20 bacterial species.</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sp-unique-section" style="background:#020b18;">
      <div class="sp-section-title">Why We Stand Apart</div>
      <div class="sp-divider"></div>
      <div class="sp-pillars">
        <div class="sp-pillar c1"><span class="sp-pillar-icon">📐</span><div class="sp-pillar-title">Dual-Index Scoring</div><div class="sp-pillar-desc">MRI and ARI are original mathematical frameworks used simultaneously.</div></div>
        <div class="sp-pillar c2"><span class="sp-pillar-icon">🌌</span><div class="sp-pillar-title">3D Resistance Landscape</div><div class="sp-pillar-desc">Plotly-powered PCA maps the entire database in 3 dimensions.</div></div>
        <div class="sp-pillar c3"><span class="sp-pillar-icon">🤖</span><div class="sp-pillar-title">Context-Aware AI Chat</div><div class="sp-pillar-desc">J.A.R.V.I.S. auto-injects MRI scores into every query.</div></div>
        <div class="sp-pillar c4"><span class="sp-pillar-icon">🕸️</span><div class="sp-pillar-title">Live Mechanism Networks</div><div class="sp-pillar-desc">PyVis-powered interactive gene-to-mechanism topology.</div></div>
        <div class="sp-pillar c5"><span class="sp-pillar-icon">🩺</span><div class="sp-pillar-title">Safe-Zone Clinical Logic</div><div class="sp-pillar-desc">Genomic exclusion logic for instant treatment guidance.</div></div>
        <div class="sp-pillar c6"><span class="sp-pillar-icon">🗺️</span><div class="sp-pillar-title">Origin-Comparative Heatmaps</div><div class="sp-pillar-desc">Benchmark isolates against 5 curated environmental databases.</div></div>
        <div class="sp-pillar c7"><span class="sp-pillar-icon">📄</span><div class="sp-pillar-title">One-Click Master Reports</div><div class="sp-pillar-desc">ReportLab PDF with math, dashboards, heatmaps, and ledgers.</div></div>
        <div class="sp-pillar c8"><span class="sp-pillar-icon">🧫</span><div class="sp-pillar-title">Virtual Lab Simulation</div><div class="sp-pillar-desc">Simulate culture, MIC plates, Gram stain, and AI vs lab comparison for 20 bacterial species.</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sp-team-section" style="background:#020b18;">
      <div class="sp-section-title">Meet the Team</div>
      <div class="sp-divider"></div>
      <div class="sp-team-grid">
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(0,212,255,0.1);">👨‍💻</div><div class="sp-tname">Hardik Agrawal</div><div class="sp-trole">Lead Developer</div><div class="sp-tdesc">Full-stack architecture, MRI/ARI framework, AI prediction pipeline.</div><div class="sp-ttags"><span class="sp-ttag">Python</span><span class="sp-ttag">ML</span><span class="sp-ttag">Streamlit</span></div></div>
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(236,72,153,0.1);">👩‍🔬</div><div class="sp-tname">Poorva Dongarkar</div><div class="sp-trole">Genomics Analyst</div><div class="sp-tdesc">CARD database integration, ARG extraction, bacterial classification.</div><div class="sp-ttags"><span class="sp-ttag">Genomics</span><span class="sp-ttag">AMR</span><span class="sp-ttag">Data</span></div></div>
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(16,185,129,0.1);">👨‍🔬</div><div class="sp-tname">Yashraj Patil</div><div class="sp-trole">Visualization Engineer</div><div class="sp-tdesc">PyVis networks, Plotly 3D PCA, Matplotlib dashboard.</div><div class="sp-ttags"><span class="sp-ttag">PyVis</span><span class="sp-ttag">Plotly</span><span class="sp-ttag">Matplotlib</span></div></div>
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(245,158,11,0.1);">👩‍💻</div><div class="sp-tname">Avani Laswante</div><div class="sp-trole">UI/UX Designer</div><div class="sp-tdesc">CSS design system, dark-mode aesthetic, animated visual identity.</div><div class="sp-ttags"><span class="sp-ttag">CSS</span><span class="sp-ttag">UI/UX</span><span class="sp-ttag">Design</span></div></div>
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(139,92,246,0.1);">👩‍🔬</div><div class="sp-tname">Zeel Bhanushali</div><div class="sp-trole">Clinical Research</div><div class="sp-tdesc">Clinical susceptibility logic, drug-class mapping, MRI validation.</div><div class="sp-ttags"><span class="sp-ttag">Microbiology</span><span class="sp-ttag">Pharmacology</span></div></div>
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(6,182,212,0.1);">👩‍💻</div><div class="sp-tname">Aayushi Wasnik</div><div class="sp-trole">AI Integration</div><div class="sp-tdesc">J.A.R.V.I.S. chatbot, Gemini API, context injection.</div><div class="sp-ttags"><span class="sp-ttag">Gemini API</span><span class="sp-ttag">LLM</span><span class="sp-ttag">Prompt Eng.</span></div></div>
        <div class="sp-team-card"><div class="sp-avatar" style="background:rgba(244,63,94,0.1);">👨‍🔬</div><div class="sp-tname">Indranil Patil</div><div class="sp-trole">PDF &amp; Documentation</div><div class="sp-tdesc">ReportLab PDF pipeline, mathematical documentation.</div><div class="sp-ttags"><span class="sp-ttag">ReportLab</span><span class="sp-ttag">LaTeX</span><span class="sp-ttag">Writing</span></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sp-cta-section" style="background:#020b18;">
      <p class="sp-cta-label">Ready to Analyze</p>
      <h2 class="sp-cta-headline">Begin Your Genomic Analysis</h2>
      <p class="sp-cta-sub">Click below to enter the AI-MRI Hub platform.</p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([2, 1, 2])
    with col_center:
        if st.button("🚀  ENTER AI-MRI HUB", type="primary", use_container_width=True):
            st.session_state.show_splash = False
            st.rerun()
    st.stop()


# ==========================================
# MAIN APPLICATION
# ==========================================
st.markdown("""
<div class="main-header">
<h1>🧬 AI-Driven Multidimensional Resistance Index</h1>
<p>Quantitative Bio-Analysis of Antibiotic Resistance Genes</p>
<hr style='border:0.5px solid rgba(255,255,255,0.2);margin:16px auto;width:80%;'>
<p style='font-size:clamp(0.75rem,2.5vw,0.95rem);font-weight:300;'>
<b>Developed by:</b> Hardik Agrawal, Poorva Dongarkar, Yashraj Patil, Avani Laswante, Zeel Bhanushali, Aayushi Wasnik &amp; Indranil Patil
</p>
</div>
""", unsafe_allow_html=True)

if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
    st.session_state.current_session = "Chat 1"
    st.session_state.chat_counter = 1

if 'origin_labels' not in st.session_state:
    st.session_state.origin_labels = {}

with st.sidebar:
    st.header("🗄️ Database Sync")
    if st.button("🔄 Refresh Database"):
        st.rerun()
    json_files    = [f for f in os.listdir('.') if f.endswith('.json')]
    analysis_mode = st.radio("Mode:", ["Select Known Bacteria", "AI Predict Unknown"])
    if analysis_mode == "Select Known Bacteria":
        selected_file = st.selectbox("Select a Genome:", json_files)
    st.markdown("---")
    st.success("✅ AI Brain Connected")

if analysis_mode == "Select Known Bacteria" and json_files:
    genes, drug, mech, mri, ari, records = extract_data(selected_file)
    u_drugs, u_mechs = len(set(drug)), len(set(mech))
    level, icon      = get_level(mri)
    habitat          = get_habitat(selected_file)
    bac_info         = get_bacteria_info(selected_file)

    drug_profile      = build_drug_class_profile(drug)
    drug_profile_norm = normalize_profile(drug_profile, len(drug))
    selected_origin_key = st.session_state.origin_labels.get(selected_file, "clinical")
    fig = plot_full_dashboard(drug, mech, mri, genes, records, selected_file)

    if mri > 0.6:
        st.markdown(f'<div class="alert-banner">⚠️ CRITICAL ALERT: {selected_file} identified as High-Priority Superbug — MRI Score: {round(mri,3)}</div>', unsafe_allow_html=True)

    rf_model     = train_rf_model()
    ai_pred_text = "N/A"
    ai_conf_text = "N/A"
    if rf_model:
        pred      = rf_model.predict([[genes, u_drugs, u_mechs]])[0]
        probs     = rf_model.predict_proba([[genes, u_drugs, u_mechs]])[0]
        classes   = rf_model.classes_
        conf_dict = {str(c): round(float(p),3) for c,p in zip(classes,probs)}
        ai_pred_text = str(pred)
        ai_conf_text = str(conf_dict).replace("'","")

    m1, m2 = st.columns(2)
    m3, m4 = st.columns(2)
    with m1: st.markdown(f'<div class="metric-card"><div class="metric-label">Risk Level</div><div class="metric-value">{level} {icon}</div></div>', unsafe_allow_html=True)
    with m2: st.markdown(f'<div class="metric-card"><div class="metric-label">MRI Score</div><div class="metric-value">{round(mri,3)}</div></div>', unsafe_allow_html=True)
    with m3: st.markdown(f'<div class="metric-card"><div class="metric-label">Total Genes</div><div class="metric-value">{genes}</div></div>', unsafe_allow_html=True)
    with m4: st.markdown(f'<div class="metric-card"><div class="metric-label">Habitat</div><div class="metric-value">{habitat}</div></div>', unsafe_allow_html=True)
    st.write(" ")

    # ── 10 TABS (added 🧫 Virtual Lab) ──
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "ℹ️ Summary", "📊 Dashboard", "🧮 Math", "🕸️ Network",
        "🤖 AI Chat", "🩺 Clinical", "🗺️ Origin", "📄 PDF",
        "🌌 3D Map", "🧫 Virtual Lab"
    ])

    with tab1:
        section_explainer("ℹ️","About This Tab","This section provides a high-level overview and a detailed pathogen identification card.")
        st.markdown('<div class="welcome-hero"><h2>Genomic Resistance Intelligence Dashboard</h2><p>Comprehensive AMR Analysis Platform — Powered by AI</p></div>', unsafe_allow_html=True)
        st.markdown("### 🧬 Platform Capabilities")
        st.info("""
**The AI-MRI Hub provides a state-of-the-art multidimensional genomic analysis suite:**
- **Genomic ARG Profiling:** Structured identification and classification of Antibiotic Resistance Genes from CARD-format data.
- **Multidimensional Resistance Index (MRI):** A validated clinical risk score.
- **Random Forest Risk Classification:** Machine learning-driven threat stratification.
- **Clinical Susceptibility Zone Analysis:** Drug classes with zero resistance markers.
- **Interactive Resistance Topology:** 3D PCA and live network graphs.
- **Origin Labeling & Comparative Heatmaps:** Benchmark against curated environmental databases.
- **Virtual Lab Simulation:** Full phenotypic simulation for 20 bacterial species.
        """)
        st.markdown("### 🦠 Pathogen Identification Profile")
        stored_origin = st.session_state.origin_labels.get(selected_file,"Not Set")
        st.markdown(f"""
<div class="report-card">
<div class="report-header">Bacterial Resistance Intelligence Ledger</div>
<div class="report-row"><span class="report-label">Target Genome File</span><span class="report-value" style="color:#60a5fa;">{selected_file}</span></div>
<div class="report-row"><span class="report-label">Gram Classification</span><span class="report-value">{bac_info['gram']}</span></div>
<div class="report-row"><span class="report-label">Associated Pathology</span><span class="report-value">{bac_info['disease']}</span></div>
<div class="report-row"><span class="report-label">Ecological Habitat</span><span class="report-value">{habitat}</span></div>
<div class="report-row"><span class="report-label">Genomic Origin Label</span><span class="report-value">{stored_origin}</span></div>
<div class="report-row"><span class="report-label">Total Resistance Genes</span><span class="report-value">{genes}</span></div>
<div class="report-row"><span class="report-label">Drug Classes Resisted</span><span class="report-value">{u_drugs} Classes</span></div>
<div class="report-row"><span class="report-label">Mechanisms Deployed</span><span class="report-value">{u_mechs} Strategies</span></div>
<div class="report-row"><span class="report-label">MRI / ARI Score</span><span class="report-value">{round(mri,3)} ({level}) / {round(ari,3)}</span></div>
<div style="margin-top:16px;padding-top:10px;">
<span class="report-label">AI Risk Classification:</span>&nbsp;
<span class="ai-badge">{ai_pred_text} Risk</span>
<br><br>
<small style="color:#64748b;font-size:0.8rem;">Probability Distribution: {ai_conf_text}</small>
</div>
</div>
""", unsafe_allow_html=True)

    with tab2:
        section_explainer("📊","About This Dashboard","Six-panel systems analysis dashboard: drug class distribution, mechanism frequency, MRI gauge, gene frequency, gene count, and diversity comparison.")
        st.markdown(f"### Systems Analysis Dashboard — `{selected_file}`")
        st.pyplot(fig)

    with tab3:
        section_explainer("🧮","About This Section","Full mathematical derivation of the MRI and ARI frameworks with computed values.")
        st.markdown("### 🧮 Mathematical Framework & Validation")
        col_m1, col_m2 = st.columns([1,1])
        with col_m1:
            st.markdown('<div class="math-card"><div class="math-card-header">Multidimensional Resistance Index (MRI)</div>', unsafe_allow_html=True)
            st.latex(r"MRI = \frac{U_{drugs} + U_{mechs}}{T_{drugs} + T_{mechs} + 1}")
            st.markdown(f'<div class="annotation-box"><div class="annotation-item"><span class="annotation-key">U_drugs</span> — Unique drug classes resisted</div><div class="annotation-item"><span class="annotation-key">U_mechs</span> — Unique resistance mechanisms</div><div class="annotation-item"><span class="annotation-key">+ 1</span> — Laplace smoothing</div></div></div>', unsafe_allow_html=True)
            st.markdown("**Computed Value:**")
            st.latex(rf"\frac{{{u_drugs} + {u_mechs}}}{{{len(drug)} + {len(mech)} + 1}} = {round(mri,3)}")
        with col_m2:
            st.markdown('<div class="math-card"><div class="math-card-header">Antibiotic Resistance Index (ARI)</div>', unsafe_allow_html=True)
            st.latex(r"ARI = \frac{U_{mechs}}{G_{total} + 1}")
            st.markdown(f'<div class="annotation-box"><div class="annotation-item"><span class="annotation-key">U_mechs</span> — Unique resistance mechanisms</div><div class="annotation-item"><span class="annotation-key">G_total</span> — Total genomic resistance genes</div><div class="annotation-item"><span class="annotation-key">+ 1</span> — Laplace smoothing</div></div></div>', unsafe_allow_html=True)
            st.markdown("**Computed Value:**")
            st.latex(rf"\frac{{{u_mechs}}}{{{genes} + 1}} = {round(ari,3)}")
        st.markdown("### 🎯 Risk Assessment Interpretation")
        st.markdown(f'<div class="reasoning-box">{get_risk_reason(level,u_drugs,u_mechs)}</div>', unsafe_allow_html=True)
        st.write("")
        df = pd.DataFrame(records, columns=["Gene Name","Drug Classes Resisted","Mechanisms Deployed","Habitat"])
        st.dataframe(df, use_container_width=True)

    with tab4:
        section_explainer("🕸️","About the Resistance Network","Interactive network graph mapping the resistance topology of the selected genome.")
        st.markdown("### 🕸️ Interactive Resistance Mechanism Network")
        html_path = generate_network_html(records, selected_file, "red" if level=="HIGH" else "orange" if level=="MODERATE" else "green")
        with open(html_path,'r',encoding='utf-8') as f:
            components.html(f.read(), height=550)

    with tab5:
        section_explainer("🤖","About J.A.R.V.I.S. Bio-AI","Gemini-powered genomic AI assistant with full genomic context automatically injected.")
        st.session_state.current_session = st.selectbox("Active Chat Session:", list(st.session_state.chat_sessions.keys()))
        col_new, col_clear = st.columns(2)
        with col_new:
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.chat_counter += 1
                new_chat_name = f"Chat {st.session_state.chat_counter}"
                st.session_state.chat_sessions[new_chat_name] = []
                st.session_state.current_session = new_chat_name
                st.rerun()
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_sessions[st.session_state.current_session] = []
                st.rerun()
        for msg in st.session_state.chat_sessions[st.session_state.current_session]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_msg = st.chat_input(f"Query J.A.R.V.I.S. about {selected_file}...")
        if user_msg:
            st.chat_message("user").markdown(user_msg)
            st.session_state.chat_sessions[st.session_state.current_session].append({"role":"user","content":user_msg})
            if not AI_AVAILABLE:
                st.error("⚠️ AI library unavailable.")
            else:
                try:
                    origin_ctx = st.session_state.origin_labels.get(selected_file,"Unknown")
                    context = f"""You are J.A.R.V.I.S., an expert Bioinformatics AI specializing in antimicrobial resistance genomics.
Genome: '{selected_file}'. ARGs: {genes}, Drug Classes: {u_drugs}, Mechanisms: {u_mechs}, MRI: {round(mri,3)} ({level}), ARI: {round(ari,3)}.
Origin: {origin_ctx}. Habitat: {habitat}. Analyst Query: {user_msg}"""
                    with st.spinner("Processing genomic data..."):
                        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        if not available_models:
                            st.error("No models available.")
                        else:
                            target_model = next((m for m in available_models if 'flash' in m), next((m for m in available_models if 'pro' in m), available_models[0]))
                            model_ai = genai.GenerativeModel(target_model)
                            response = model_ai.generate_content(context)
                            st.chat_message("assistant").markdown(response.text)
                            st.session_state.chat_sessions[st.session_state.current_session].append({"role":"assistant","content":response.text})
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        st.error("⚠️ API Quota Exceeded. Please wait 60 seconds.")
                    else:
                        st.error(f"AI Connection Error: {e}")

    with tab6:
        section_explainer("🩺","About Clinical Susceptibility","Identifies drug classes with zero detected resistance genes — potential therapeutic candidates.")
        st.markdown("### 🩺 Clinical Susceptibility Zone Analysis")
        DRUG_UNIVERSE = ["Penicillin","Cephalosporin","Carbapenem","Macrolide","Aminoglycoside","Fluoroquinolone","Tetracycline","Sulfonamide","Glycopeptide"]
        resisted_norm = set([d.lower() for d in drug])
        safe_zones    = [d for d in DRUG_UNIVERSE if d.lower() not in resisted_norm]
        st.write("Drug classes with **zero resistance markers** — potential therapeutic candidates:")
        st.markdown(f'<div class="susceptibility-card">🛡️ Candidate Therapeutic Classes:<br>{", ".join(safe_zones) if safe_zones else "⚠️ No unresisted classes found."}</div>', unsafe_allow_html=True)
        st.caption("⚠️ Clinical confirmation via standard antibiogram (MIC testing) is required.")
        st.write("---")
        st.markdown("### 📈 Population Benchmark Comparison")
        all_mris = []
        for f in json_files:
            try:
                _,_,_,f_mri,_,_ = extract_data(f); all_mris.append(f_mri)
            except: continue
        if all_mris:
            avg_mri = sum(all_mris)/len(all_mris)
            comparison_df = pd.DataFrame({"MRI Score":[mri,avg_mri]}, index=["Target Genome","Database Average"])
            st.bar_chart(comparison_df)

    with tab7:
        section_explainer("🗺️","About Origin Labeling","Label each genome by biological source and compare against curated reference benchmarks.")
        st.markdown("### 🗺️ Genomic Origin Labeling & Source-Comparative Analysis")
        col_label, col_display = st.columns([2,1])
        with col_label:
            origin_choice = st.selectbox("📌 Assign Origin Label:",list(ORIGIN_LABELS.keys()),
                index=list(ORIGIN_LABELS.values()).index(st.session_state.origin_labels.get(selected_file,"clinical")) if selected_file in st.session_state.origin_labels else 0)
            if st.button("✅ Confirm Origin Label", type="primary"):
                st.session_state.origin_labels[selected_file] = ORIGIN_LABELS[origin_choice]
                st.success(f"Origin label saved: **{origin_choice}** for `{selected_file}`")
        selected_origin_key = st.session_state.origin_labels.get(selected_file,"clinical")
        with col_display:
            origin_css = {"clinical":"origin-clinical","agricultural":"origin-agricultural","environmental":"origin-environmental","wastewater":"origin-wastewater","food_production":"origin-food_production"}
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div style="text-align:center;"><div style="color:#94a3b8;font-size:0.8rem;margin-bottom:8px;">CURRENT LABEL</div><span class="origin-badge {origin_css.get(selected_origin_key,"origin-clinical")}">{origin_choice}</span></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### 🔬 Select Reference Origin for Delta Comparison")
        ref_labels = list(ORIGIN_LABELS.keys())
        ref_choice = st.selectbox("Compare against:",ref_labels,index=ref_labels.index(origin_choice) if origin_choice in ref_labels else 0)
        ref_key    = ORIGIN_LABELS[ref_choice]
        st.markdown("---")
        st.markdown("#### 🎯 Origin Affinity Scores")
        affinity_scores = compute_origin_affinity_scores(drug_profile_norm, genes, mri)
        aff_display = {"🏥 Clinical":affinity_scores["clinical"],"🌾 Agricultural":affinity_scores["agricultural"],"🌿 Environmental":affinity_scores["environmental"],"💧 Wastewater":affinity_scores["wastewater"],"🍖 Food Prod.":affinity_scores["food_production"]}
        best_match_key   = max(affinity_scores, key=affinity_scores.get)
        best_match_label = [k for k,v in ORIGIN_LABELS.items() if v==best_match_key][0]
        aff_cols = st.columns(5)
        aff_keys = list(aff_display.keys()); aff_vals = list(aff_display.values())
        for i, col in enumerate(aff_cols):
            val = aff_vals[i]
            highlight = "border:2px solid #00d4ff;" if val==max(aff_vals) else ""
            interp_txt = "Strong" if val>=70 else "Moderate" if val>=50 else "Weak" if val>=30 else "Atypical"
            col.markdown(f'<div class="metric-card" style="{highlight}"><div class="metric-label">{aff_keys[i]}</div><div class="metric-value" style="font-size:1.4rem;">{val}</div><div style="color:#94a3b8;font-size:0.68rem;margin-top:2px;">/ 100 · {interp_txt}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:rgba(0,212,255,0.07);border:1px solid rgba(0,212,255,0.25);border-radius:10px;padding:12px 16px;margin-top:14px;"><span style="color:#00d4ff;font-weight:700;">🏆 Best Match:</span> <span style="color:#ffffff;margin-left:8px;">{best_match_label}</span> <span style="color:#94a3b8;margin-left:8px;font-size:0.85rem;">(Score: {affinity_scores[best_match_key]}/100)</span></div>', unsafe_allow_html=True)
        st.markdown("---")
        insights   = generate_origin_insights(drug_profile_norm, ref_key, affinity_scores, selected_file.replace('.json',''), mri, genes)
        ins_cols   = st.columns(2)
        with ins_cols[0]:
            if insights["excess_classes"]:
                excess_str = ", ".join([f"<strong>{cls}</strong> (+{delta:.2f})" for cls,delta in insights["excess_classes"]])
                st.markdown(f'<div class="insight-card"><div class="ins-title">⬆️ Exceeds Reference</div><div class="ins-body">{excess_str}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-card"><div class="ins-title">⬆️ Resistance vs Reference</div><div class="ins-body">No class exceeds baseline by more than 0.2.</div></div>', unsafe_allow_html=True)
        with ins_cols[1]:
            if insights["deficit_classes"]:
                deficit_str = ", ".join([f"<strong>{cls}</strong> (−{delta:.2f})" for cls,delta in insights["deficit_classes"]])
                st.markdown(f'<div class="insight-card"><div class="ins-title">⬇️ Below Reference</div><div class="ins-body">{deficit_str} — potential therapeutic options.</div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-card"><div class="ins-title">⬇️ Resistance vs Reference</div><div class="ins-body">No class is more than 0.2 below baseline.</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("#### 🔥 Comparative Resistance Heatmap")
        heatmap_fig = plot_origin_comparison_heatmap(drug_profile_norm, ref_key, selected_file.replace('.json',''))
        st.pyplot(heatmap_fig, use_container_width=True)
        st.markdown("---")
        st.markdown("#### 📡 Resistance Topology Radar")
        radar_fig = plot_radar_origin_comparison(drug_profile_norm, selected_file.replace('.json',''))
        st.plotly_chart(radar_fig, use_container_width=True)
        st.markdown("---")
        st.markdown("#### 🗃️ All Labelled Genomes")
        if st.session_state.origin_labels:
            label_rows = []
            for fname, orig_key in st.session_state.origin_labels.items():
                try:
                    g,dr,me,mr,ar,_ = extract_data(fname)
                    lv,ic = get_level(mr)
                    display_orig = [k for k,v in ORIGIN_LABELS.items() if v==orig_key]
                    label_rows.append({"Genome":fname,"Origin":display_orig[0] if display_orig else orig_key,"Genes":g,"MRI":round(mr,3),"ARI":round(ar,3),"Risk":f"{lv} {ic}"})
                except: continue
            if label_rows:
                st.dataframe(pd.DataFrame(label_rows), use_container_width=True)
        else:
            st.info("No genomes labelled yet.")

    with tab8:
        section_explainer("📄","About the PDF Report","Generates a comprehensive PDF with executive summary, math derivations, dashboard, origin analysis, affinity scores, heatmap, and gene ledger.")
        st.markdown("### 📥 Generate Master Analytical Report")
        current_origin_label = [k for k,v in ORIGIN_LABELS.items() if v==selected_origin_key]
        origin_label_str = current_origin_label[0] if current_origin_label else "Not Set"
        st.info(f"📌 Origin label in PDF: **{origin_label_str}**")
        if st.button("Generate Master PDF Report", type="primary"):
            with st.spinner("Compiling PDF..."):
                pdf_heatmap_fig = plot_origin_comparison_heatmap(drug_profile_norm, selected_origin_key, selected_file.replace('.json',''))
                pdf_affinity    = compute_origin_affinity_scores(drug_profile_norm, genes, mri)
                pdf_path = create_advanced_pdf_report(
                    selected_file, genes, drug, mech, mri, ari, level, icon,
                    records, fig, bac_info, habitat, ai_pred_text, ai_conf_text,
                    origin_label=origin_label_str,
                    affinity_scores=pdf_affinity,
                    drug_profile_norm=drug_profile_norm,
                    selected_origin_key=selected_origin_key,
                    heatmap_fig=pdf_heatmap_fig
                )
                with open(pdf_path,"rb") as file:
                    st.download_button(label="⬇️ Download Analytical Report (PDF)", data=file, file_name=pdf_path, mime="application/pdf")

    with tab9:
        section_explainer("🌌","About the 3D Resistance Landscape","3D PCA map projecting every genome across Overall Resistance, Mechanism Diversity, and Genetic Density.")
        st.markdown("### 🌌 Interactive Global Resistance Landscape (3D PCA)")
        plot_3d_pca_plotly(selected_file)

    # ── TAB 10: VIRTUAL LAB ──
    with tab10:
        section_explainer("🧫", "About the Virtual Lab",
            "Full phenotypic simulation for 20 clinically and environmentally important bacterial species. "
            "Generates culture growth curves, colony morphology plates, Gram stain microscopy, "
            "96-well MIC plates for 12 antibiotics, AI vs lab comparison charts, drug class radar profiles, "
            "and gene identity panels — all driven by your selected CARD JSON file and lab conditions.")
        render_lab_tab()

elif analysis_mode == "AI Predict Unknown":
    section_explainer("🤖","About AI Prediction Mode","Input resistance parameters for an uncharacterised isolate and receive a Random Forest AI risk classification.")
    st.header("🤖 Machine Learning Risk Classification — Unknown Pathogen")
    in_genes = st.number_input("Total Resistance Genes Identified", min_value=1, value=15)
    in_drugs = st.number_input("Unique Drug Classes Resisted", min_value=1, value=5)
    in_mechs = st.number_input("Unique Resistance Mechanisms", min_value=1, value=2)
    rf_model = train_rf_model()
    if rf_model and st.button("Run Risk Classification", type="primary"):
        prediction = rf_model.predict([[in_genes, in_drugs, in_mechs]])[0]
        probs      = rf_model.predict_proba([[in_genes, in_drugs, in_mechs]])[0]
        classes    = rf_model.classes_
        prob_str   = " | ".join([f"{c}: {p:.3f}" for c,p in zip(classes,probs)])
        st.success(f"### AI Risk Classification: **{prediction}**")
        st.info(f"**Probability Distribution:** {prob_str}")
